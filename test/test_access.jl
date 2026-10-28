# What the access walk decides from a TYPE, with no device in the room.
#
# The walk itself needs the interpreter a backend compiles through — a device
# array's `getindex` lives on that method table and nowhere else — so the
# kernels, and what right looks like for them, are in `vulkan/test_access.jl`.
# What is here is the part that has no backend in it: whether a type can reach
# memory at all, and what a `Touch` means as a usage.
using Test, Mantle
const M = Mantle

@testset "reflection failure is not an inferred no-access proof" begin
    rows=Base.code_ircode_by_type(Tuple{typeof(identity),Int})
    @test length(rows)==1 && first(only(rows)) isa Core.Compiler.IRCode
    ir=first(only(rows))
    @test M.inferred_access_ir(rows)===ir
    @test M.inferred_access_ir([which(identity,(Int,))=>Any])===nothing
    @test M.inferred_access_ir([nothing=>Any])===nothing
    @test M.inferred_access_ir([ir=>Union{}])===nothing
    @test M.inferred_access_ir([])===nothing
    @test M.inferred_access_ir([ir=>Int,ir=>Int])===nothing
end

@testset "inferred access: the unknown is refused, not guessed at" begin
    # `accessof` on a signature nothing can be inferred for must not answer
    # read+write for every argument: safe for the barrier phase, and
    # indistinguishable from a proof: a kernel that stopped being analysable kept
    # compiling and paid a barrier per pass for ever with nothing to say so.
    err = try
        M.accessof(nothing, sin, (Any,))
        nothing
    catch e
        e
    end
    @test err isa M.UnanalysableAccess
    msg = sprint(showerror, err)
    @test occursin("cannot say what", msg)
    @test occursin("will not guess", msg)
    # `OPAQUE` is still the usage a read+write argument has; what changed is that
    # nothing produces it by giving up.
    @test M.usagetype(M.OPAQUE, M.BufferKind()) === Storage{BufferKind, ReadWrite}

    # And the pieces the walk decides with.
    @test M.carries(Float32) === false
    @test M.carries(Tuple{Int,Int}) === false
    @test M.carries(Ptr{Float32}) === true
    @test M.carries(Any) === true

    # A `Touch` with an ordinary write is ordered; one whose writes are all
    # atomic is not.
    @test M.usagetype(M.WRITE, M.BufferKind()) === Storage{BufferKind, WriteOnly}
    @test M.usagetype(M.ATOMIC, M.BufferKind()) ===
          Unordered{Storage{BufferKind, ReadWrite}}
    @test M.usagetype(M.READ, M.BufferKind()) === Storage{BufferKind, ReadOnly}
end

# A dynamic call names its callee. A generated kernel called `Atomix.atomic_max`,
# which Atomix does not define; the refusal said "a call to `%12`", and that was
# the whole of the runtime feedback a repair attempt was given. The callee is a
# `getglobal(Atomix, :atomic_max)` statement, so the module and name are known.
module NoSuchMember end
function callsmissing(out::Ptr{Float32})
    NoSuchMember.atomic_max(out, 1, 2f0, :monotonic)
    return
end
callsdefined(out::Ptr{Float32}, f) = (f(out); return)

@testset "a refused dynamic call names its callee" begin
    err = try
        M.accessof(nothing, callsmissing, (Ptr{Float32},))
        nothing
    catch e
        e
    end
    @test err isa M.UnanalysableAccess
    msg = sprint(showerror, err)
    @test occursin("NoSuchMember.atomic_max", msg)
    @test occursin("does not define", msg)
    @test !occursin("call to `%", msg)
    # A callee that exists but stays dynamic is still named, and not called missing.
    err = try
        M.accessof(nothing, callsdefined, (Ptr{Float32}, Any))
        nothing
    catch e
        e
    end
    @test err isa M.UnanalysableAccess
    @test !occursin("does not define", sprint(showerror, err))
end

# A device address laundered through an integer is still an address.
#
# `carries` asks what a TYPE can hold and answers nothing for `UInt64`, which is
# right for a length and wrong for an address. The walk had one hop of an
# exception for it -- taint survived a statement with a `Ptr`-typed operand -- and
# Lava's cooperative-matrix intrinsics need two, because they take the address as
# a `UInt64` with the element offset already added in:
#
#     %18 = getfield(C, :ptr)::Ptr{Float16}   # tainted by C
#     %19 = bitcast(UInt64, %18)              # kept: an operand is a `Ptr`
#     %31 = add_int(%19, %28)                 # DROPPED: both operands are UInt64
#           llvmcall(@_lava_coopmat_store_..., %31, ...)
#
# `llvmcalltaint!` then found no tainted operand and recorded nothing, so every
# argument of `coopmat_gemm_kernel_4!` came back untouched -- INCLUDING its
# destination -- and `Place` saw its output transient with no interval at all.
# `gemm_cm2!` reads correctly because its intrinsics take a `Ptr` directly, which
# is why the failure was confined to the block kernels.
#
# Stated here on `unsafe_load`/`unsafe_store!` rather than on a kernel, because
# the laundering is the bug and neither the intrinsics nor a device are part of
# it: written this way it reproduces on the host walk, with no backend loaded.
function storelaundered(out::Ptr{Float32}, src::Ptr{Float32}, n::Int)
    a = Base.bitcast(UInt64, out)
    b = Base.bitcast(UInt64, src)
    for i in 1:n
        v = unsafe_load(Base.bitcast(Ptr{Float32}, b + UInt64(4 * (i - 1))))
        unsafe_store!(Base.bitcast(Ptr{Float32}, a + UInt64(4 * (i - 1))), v)
    end
    return
end

@testset "an address laundered through an integer" begin
    t = M.accessof(nothing, storelaundered, (Ptr{Float32}, Ptr{Float32}, Int))
    @test t[2] === M.WRITE
    @test t[3] === M.READ
    # And the reason the type test exists in the first place: `n` reaches the
    # same arithmetic and is NOT an address, so following every integer is not
    # the fix. Before `carries` was consulted at all, an argument whose length a
    # kernel read came back read+write.
    @test t[4] === M.NOTOUCH
end

# A summary holds while the code it was read from does. The device's cache was
# emptied whenever the world age moved, which in Julia 1.12 is any method defined
# anywhere and even a new global at the REPL: the next scene re-inferred all 43
# kernels of a Hikari render, two seconds of its first frame, for code that had
# not changed. A redefinition reaches a summary through the `CodeInstance` it was
# read from, whose world range Julia ends through the backedges of every caller.
module AccessHoldsCode
    load_or_store(p::Ptr{Float32}) = unsafe_load(p)
    kernel(p::Ptr{Float32}) = load_or_store(p)
end

@testset "an access summary holds until its own code changes" begin
    cache = M.AccessCache()
    sig = Tuple{typeof(AccessHoldsCode.kernel), Ptr{Float32}}
    touches() = M.accessof(nothing, AccessHoldsCode.kernel, (Ptr{Float32},); cache)[2]
    @test touches() == M.READ
    held = cache.summaries[sig]
    # Defining something the kernel does not call keeps the summary.
    Core.eval(AccessHoldsCode, :(unrelated() = 1))
    @test touches() == M.READ
    @test cache.summaries[sig] === held
    # Redefining what the kernel calls renews it.
    Core.eval(AccessHoldsCode, :(load_or_store(p::Ptr{Float32}) = (unsafe_store!(p, 1f0); 0f0)))
    @test touches() == M.WRITE
    @test cache.summaries[sig] !== held
end
