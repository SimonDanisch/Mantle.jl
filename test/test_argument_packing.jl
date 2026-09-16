# How a kernel argument reaches the kernel is core's answer, and one answer.
#
# Both recording backends ask the same three questions of every argument of a
# dispatch, twice each — once when the dispatch compiles (which slot, how many
# bytes) and once when it is written (what to put there). Each used to answer
# them itself, in three places between them, with two rules that are NOT
# equivalent:
#
#   Vulkan  `slotpositions`   skipped `sizeof(Ti) == 0 || Ti <: Type`
#   Metal   `recorded_args`   skipped `isghosttype(T) || isconstType(T)`
#   Metal   `pack_recorded!`  repeated Metal's test to re-derive the slot index
#                             `recorded_args` had already assigned
#
# `Core.Compiler.isconstType` answers `true` only for a SIGNATURE type such as
# `Type{Float32}`. Both walks classify from `typeof(a)`, and `typeof(Float32)`
# is `DataType`, so Metal's test never fired for a type-valued argument: its
# bytes branch then copied `sizeof(DataType)` bytes out of a `Ref` holding a
# pointer, into a slot the compiled kernel does not have. Only one of the two
# backends is ever compiled into a given build, so no test could see the two
# rules disagree — which is why the first testset below compares core's rule
# against GPUCompiler's own on a table of types rather than against a backend.
#
# `graph/packing.jl` is the one rule and the two walks; a backend now declares
# only which of ITS OWN handle types bind an allocation instead of bytes.

using Test, Mantle
using KernelAbstractions
import KernelInterface as KI
import GPUCompiler

const M = Mantle
const BE = Main.MANTLE_TEST_BACKEND

# A kernel with two arguments the compiled kernel has no parameter for, and one
# of each on either side of them, so an off-by-one in the slot count writes the
# wrong value rather than the right one by luck.
#
# `::Type{T}` is the type-valued case: `Core.Typeof(Float32)` is
# `Type{Float32}`, which GPUCompiler bakes in and emits no parameter for, while
# the packer sees only `DataType`. `f` is the zero-size case.
# Shaped like a device array descriptor on either backend: the device address
# first, the length after it. `devicepointeroffsets` walks a type, so this is the
# whole of what it needs and the offsets do not depend on which backend is built.
struct Descriptor{P}
    ptr::P
    len::Int
end

# A type you declared, holding a descriptor a field deep: the case the depth
# budget exists for, and the thing a tuple of operands must NOT be confused with.
struct Nested{A}
    a::A
    n::Int
end

function scaletyped!(out, n::Int, ::Type{T}, f, x::Float32) where {T}
    i = KI.get_global_id().x
    i <= n || return
    @inbounds out[i] = T(f(x, Float32(i)))
    return
end

# Operands as ONE tuple argument, which is what lets a single kernel cover every
# arity — `elementwise!` in DNNKernels is this shape, and it was three kernels
# (`ew1!`, `ew2!`, `ew3!`) because of it. The tuple is packed as an inline
# aggregate, so every operand's device address sits one level further down than
# it would as a separate argument.
@inline gatherat(::Tuple{}, i) = ()
@inline gatherat(ops::Tuple, i) =
    (@inbounds(first(ops)[i]), gatherat(Base.tail(ops), i)...)

function combine!(out, n::Int, ops::Tuple, f)
    i = KI.get_global_id().x
    i <= n || return
    @inbounds out[i] = f(gatherat(ops, i)...)
    return
end

@testset "argument packing — $(nameof(typeof(BE)))" begin
    # ── the rule ─────────────────────────────────────────────────────────────
    #
    # Ground truth is GPUCompiler's own predicate applied to the signature type
    # the kernel is specialised on, which is `Core.Typeof` of the value.
    # Core's rule is stated on `typeof` instead, because that is what a packer
    # holding a tuple of values actually has, and it has to agree.
    gpudrops(T) = GPUCompiler.isghosttype(T) || Core.Compiler.isconstType(T)

    for v in Any[1.0f0, 1, Float32, Int64, nothing, Val(3), +, (), (1, 2),
                 UInt64(7), Ptr{Float32}(0), (a = 1, b = 2.0f0), missing]
        @test M.notpassed(typeof(v)) == gpudrops(Core.Typeof(v))
    end

    # The disagreement, stated so this file records it on a build of either
    # backend: asked about the type of the VALUE `Float32`, GPUCompiler's
    # signature-level rule says "passed" and is wrong, because the signature it
    # would have been given is `Type{Float32}` and not `DataType`.
    @test !gpudrops(DataType)
    @test M.notpassed(DataType)
    @test M.passedas(DataType) === M.NotPassed()

    # Totality matters as much as agreement: the old Vulkan rule opened with
    # `sizeof(Ti) == 0`, and `sizeof` throws for a type with no fixed size, so
    # a `String` argument crashed the packer instead of being refused by the
    # compiler.
    @test M.passedas(String) === M.InlineBytes()
    @test M.passedas(Float32) === M.InlineBytes()

    # ── the two walks agree ──────────────────────────────────────────────────
    #
    # Vulkan's packer specialises on the argument tuple type and uses
    # `passedslots` so the walk constant-folds; Metal's runs at record time over
    # the values and uses `eachpassedarg`. They have to number slots the same
    # way, and nothing in either backend can check that, because only one of
    # them is in any given build.
    args = (+, Float32, 1.0f0, nothing, 7, Val(2), UInt64(3))
    seen = Tuple{Int,Any}[]
    nslots = M.eachpassedarg(args) do slot, a, passing
        push!(seen, (slot, a))
        @test !(passing isa M.NotPassed)
        return nothing
    end
    @test nslots == 3
    @test first.(seen) == 1:3
    @test M.passedslots(map(typeof, args)) == [3, 5, 7]
    @test last.(seen) == Any[args[i] for i in M.passedslots(map(typeof, args))]

    # `passedas` is the only piece a backend extends, and it extends it for its
    # own handle types only. Core's default answers one of the other two for
    # everything, never `DeviceAllocation`: core does not know a backend's
    # handles, and guessing one would bind a slot to a boxed object's host
    # address.
    for T in (Float32, Int, UInt64, Ptr{Float32}, NTuple{3,Int},
              @NamedTuple{a::Int, b::Float32}, Nothing, Val{4}, typeof(*), DataType)
        @test !(M.passedas(T) isa M.DeviceAllocation)
        @test M.passedas(T) isa M.ArgPassing
    end

    # ── where the device pointers inside a packed argument landed ────────────
    #
    # The table `notepacked!` builds is what makes a `resize!` under a recorded
    # plan a few stores instead of a re-record, and it is keyed off
    # `devicepointeroffsets`, which walks the argument's TYPE. A tuple used to
    # count as a level of nesting, which put every device pointer in a tuple
    # argument one past the depth budget: `devicepointeroffsets` answered `()`,
    # nothing was noted, and a move of an operand left the recorded plan reading
    # the old storage with no error anywhere. That is the whole reason
    # `elementwise!` could not take its operands as one tuple.
    # Asked of a TYPE, so both backends' descriptors can be checked on either
    # build. `Ptr` is Vulkan's device pointer and `Core.LLVMPtr` is Metal's, and
    # `notify_move!` re-keys on the whole 64-bit address in both cases.
    for DA in (Descriptor{Ptr{Float32}}, Descriptor{Core.LLVMPtr{Float32,1}})
        @test M.devicepointeroffsets(DA) == (0,)
        n1 = sizeof(DA)
        # A tuple of operands: each one found, exactly as if it had been a
        # separate argument. These are the assertions that fail without
        # `nestinglevels`.
        @test M.devicepointeroffsets(Tuple{DA,DA}) == (0, n1)
        @test M.devicepointeroffsets(Tuple{DA,DA,DA}) == (0, n1, 2n1)
        @test M.devicepointeroffsets(@NamedTuple{a::DA, b::DA}) == (0, n1)
        # Nesting the grouping changes nothing, because a grouping is not a level.
        @test M.devicepointeroffsets(Tuple{Tuple{DA},Tuple{DA}}) == (0, n1)
        # And a type you declared still is one: `Nested` holds a descriptor a
        # field deep, and a move under it invalidates the plan rather than
        # patching a half-updated copy of it. `Tuple{Nested}` has to behave the
        # same way, which is what makes the rule statable.
        @test M.devicepointeroffsets(Nested{DA}) == ()
        @test M.devicepointeroffsets(Tuple{Nested{DA}}) == ()
        @test M.nestinglevels(Tuple{DA}) == 0
        @test M.nestinglevels(Nested{DA}) == 1
    end

    # ── and it runs ──────────────────────────────────────────────────────────
    dev = M.Device(BE)
    n = 32
    want = [Float32(0.25f0 * i) for i in 1:n]
    if !M.kisupported(dev, scaletyped!)
        # The host has no `KI.kernel_function` at all; stated rather than
        # skipped, as in `test_declared_kernel.jl`.
        @test !M.kisupported(dev, scaletyped!)
        return
    end

    g = M.Graph(dev)
    out = M.Transient.Buffer(g, Float32, n)
    M.compute!(g, "scaletyped") do p
        M.dispatch!(p, scaletyped!,
                    (M.use(p, out; write = true), n, Float32, *, 0.25f0), n)
    end
    pl = M.Plan(g)
    M.record!(pl)
    M.run!(pl)
    M.waitidle(dev)
    @test Array(M.storage(out)) == want

    # Recorded, so the pack ran once and the slots it wrote are replayed. On
    # Metal this is the case that wrote 40 bytes of a `Ref`'s pointer into the
    # argument block; here it must be a second identical run.
    fill!(M.storage(out), 0.0f0)
    M.run!(pl)
    M.waitidle(dev)
    @test Array(M.storage(out)) == want
    M.free!(pl)

    # ── a tuple of operands, packed and run ──────────────────────────────────
    #
    # One kernel, two arities, and the device addresses two levels inside the
    # argument. `resolve(::Tuple)` is what gets the operands from graph handles
    # to device arrays; the generic `pack_arg!` branch inlines the tuple and
    # hands its TYPE to `recpatchfields!`, which is where the depth budget above
    # decides whether the addresses are noted at all.
    xs = M.Buffer(dev, collect(Float32, 1:n))
    ys = M.Buffer(dev, fill(2.0f0, n))
    zs = M.Buffer(dev, fill(10.0f0, n))
    for (ops, want2) in ((( xs, ys), [Float32(i) * 2 for i in 1:n]),
                         ((xs, ys, zs), [Float32(i) * 2 + 10 for i in 1:n]))
        g3 = M.Graph(dev)
        out3 = M.Transient.Buffer(g3, Float32, n)
        M.compute!(g3, "combine") do p
            M.dispatch!(p, combine!,
                        (M.use(p, out3; write = true), n,
                         map(o -> M.use(p, o; read = true), ops),
                         length(ops) == 2 ? (*) : ((a, b, c) -> a * b + c)), n)
        end
        pl3 = M.Plan(g3)
        M.record!(pl3)
        M.run!(pl3)
        M.waitidle(dev)
        @test Array(M.storage(out3)) == want2

        # Every device address the plan holds, by address. The output is one and
        # each operand is another, so a plan that noted only what it could reach
        # at the top level of an argument would have exactly one entry — which is
        # what a tuple counting as a level of nesting produced.
        if M.recordsplans(dev)
            @test length(pl3.patchtab) == 1 + length(ops)
        end
        M.free!(pl3)
    end
end
