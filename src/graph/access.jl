# What a kernel does to its arguments, read off the kernel.
#
# This is what replaced `use(p, x; read, write)`. A dispatch already names the
# resources it takes; whether each one is read or written is a property of the
# BODY, and a call site restating it is a second source for one fact — the kind
# that is right until someone edits the kernel and not the call.
#
# The analysis is a taint walk over the same typed IR the backend is about to
# compile, so it sees what the device will do and not what the host methods
# would: a device array's `getindex` exists only on the backend's method table,
# and inference run without it reaches `error_if_canonical_getindex` and proves
# nothing. Core owns the walk; a backend supplies the interpreter that makes it
# the right IR — see `kerneltouches`.
#
# EVERY unhandled construct widens to read+write on whatever tainted values it
# consumed. That is the only safe direction: claiming an access that does not
# happen costs a barrier, and missing one costs a race. The tests below the fold
# in `test_access.jl` pin both halves — that known kernels come out exact, and
# that an opaque call comes out conservative.

"""
What one dispatch does to one argument.

`atomic` is not a third kind of access, it is a property of the writes: a
resource whose every write is an atomic read-modify-write can be shared by
passes that would otherwise need a barrier between them, because their order
does not change the result. That is what `Unordered` means to the barrier phase,
and inferring it is the difference between a wavefront tracer's per-pixel
radiance costing one barrier per stage and costing none.
"""
struct Touch
    read::Bool
    write::Bool
    atomic::Bool     # every write so far was an atomic read-modify-write
end

const NOTOUCH  = Touch(false, false, true)
"""
An ordinary load, and `atomic` is false for it too.

`atomic` reads as "nothing about this access can collide", not "the writes are
atomic" — a pass that APPENDS to a queue at claimed slots and a pass that READS
that queue are a hazard, and a resource whose atomic writes made it `Unordered`
while an ordinary read of it was ignored would lose exactly that barrier. It
only ever decides anything where there is a write (see `usagetype`), so a
read-only argument is unaffected by it.
"""
const READ     = Touch(true, false, false)
"""A plain store, which is what `atomic` is false for: two of these racing is a
race, and that is exactly what the barrier between their passes is for."""
const WRITE    = Touch(false, true, false)
const ATOMIC   = Touch(true, true, true)
"""What an argument gets when the walk cannot see what happens to it."""
const OPAQUE   = Touch(true, true, false)

Base.:|(a::Touch, b::Touch) = Touch(a.read | b.read, a.write | b.write, a.atomic & b.atomic)
touched(t::Touch) = t.read | t.write

"""
    usagetype(t::Touch, kind) -> Type

The usage a `Touch` is, in the vocabulary the barrier phase reads. `Unordered`
only where there is a write to be unordered about: an argument that is merely
read is ordered by its writers and nothing about it commutes.
"""
function usagetype(t::Touch, kind::ResourceKind)
    S = Storage{typeof(kind), Access{t.read, t.write}}
    return (t.write && t.atomic) ? Unordered{S} : S
end

# ── Declaring an access, where there is no body to read it off ───────────────

"""
    argument_usage(f, args)          -> NTuple{N,Touch} or nothing
    argument_usage(F::Type, A::Type) -> NTuple{N,Touch} or nothing

What `f` does to each of `args`, where that is DECLARED rather than inferred.

`nothing` means undeclared, which is not permission to guess: the caller either
infers or refuses.

Two places ask, and that is the point of it being one function. `dispatch!` asks
about the callable it was handed, for a library call with no body on this side of
the boundary — `mul!` disappears into rocBLAS or into a cooperative-matrix
kernel. The taint walk asks about every call it is about to descend into, which
is how a device intrinsic states what it does instead of having the LLVM its
generator emitted sniffed for the word `load`.

Stated on TYPES, because the walk has only types; the value form forwards through
`Core.Typeof`, which is the type a kernel is specialised at.

No `device` parameter, deliberately. `mul!(C, A, B)` writing `C`, and
`coopmat_load` reading through its pointer, are not facts about a driver, and a
backend able to specialise this would be deciding what a function MEANS.

A declared callable must not close over device memory. The walk credits the
function's own entry with `NOTOUCH`, because there is no body being read to see
what it captured, and `dispatch!` already holds compute kernels to that rule.
"""
argument_usage(@nospecialize(F::Type), @nospecialize(A::Type)) = nothing
argument_usage(@nospecialize(f), args::Tuple) =
    argument_usage(Core.Typeof(f), Base.to_tuple_type(map(Core.Typeof, args)))

"""
The declared answer for a whole signature, shaped the way [`signaturetouches`](@ref)
returns one: the function's own entry first, then one per argument.
"""
function declaredsignature(@nospecialize(sig))
    sig isa DataType || return nothing
    ps = sig.parameters
    isempty(ps) && return nothing
    u = argument_usage(ps[1], Tuple{ps[2:end]...})
    u === nothing && return nothing
    length(u) == length(ps) - 1 || throw(ArgumentError(
        "argument_usage($(ps[1])) answered $(length(u)) touches for " *
        "$(length(ps) - 1) arguments"))
    return Touch[NOTOUCH, u...]
end

# ── The single entry point: declared, inferred, or refused ───────────────────

"""
    argument_usage(device, f, args, ndrange, group) -> Vector{Touch}

What `f` does to each of `args`, when dispatched on `device`. The one answer,
and the one place that produces it.

Two ways to get one and no third:

  * a DECLARATION, `argument_usage(f, args)` above, for a callable with no body
    on this side of the boundary. `mul!` disappears into rocBLAS or into a
    cooperative-matrix kernel, and there is nothing to read.
  * INFERENCE, [`kerneltouches`](@ref), the taint walk over the same typed IR
    the backend is about to compile.

Anything else raises. It does not widen to read+write and call that an answer:
that is safe for the barrier phase and wrong for everything else, because it is
indistinguishable from a proof. A kernel that stops being analysable would keep
compiling, pay a barrier per pass for ever, and nothing would say so.

`Read`/`Write`/`ReadWrite` wrappers were here, one per argument at the call
site. They went for the reason `use` went: N call sites restating one fact, and
the N+1st getting it wrong. `mul!(C, A, B, α, β)` is the case that makes it
concrete — `C = A*B*α + C*β`, so `C` is READ as well as written, and a call site
that copied `Write(C)` from the three-argument form is a missing barrier rather
than a compile error. The direction belongs to the function.
"""
function argument_usage(dev, @nospecialize(f), args::Tuple, ndrange, group)
    declared = argument_usage(f, args)
    declared === nothing || return Touch[declared...]
    ndrange === nothing && throw(UndeclaredCall(f, args))
    return kerneltouches(dev, f, args, ndrange, group)
end

"""
A call with no declaration and no body to read: `dispatch!` was handed something
that submits its own work, and nothing says which of its arguments it writes.
"""
struct UndeclaredCall <: Exception
    f::Any
    args::Tuple
end

function Base.showerror(io::IO, e::UndeclaredCall)
    ts = join(map(a -> string(Core.Typeof(a)), e.args), ", ")
    # One entry per argument, so the suggestion can be pasted: `WRITE` for the
    # first because a destination-first call is the overwhelming case, and the
    # sentence after says what the other two words are for.
    us = join(["Mantle." * (i == 1 ? "WRITE" : "READ") for i in eachindex(e.args)],
              ", ")
    print(io, """
        dispatch!: `$(e.f)` was declared as a call (no ndrange) and no
        `argument_usage` says what it does to its $(length(e.args)) arguments.

        A call hands work to something that submits its own, so there is no body
        on this side of the boundary to read the direction off. Declare it once,
        next to the function, rather than at each call site:

            Mantle.argument_usage(::Type{typeof($(e.f))}, ::Type{<:Tuple{$ts}}) =
                ($us)

        One `Touch` per argument, in order, and the guess above is only a
        shape. `Mantle.NOTOUCH` for a scalar, and `Mantle.Touch(true, true,
        false)` for an argument that is read as well as written -- which is
        `mul!`'s `C` in the five-argument form, where `beta` makes it an
        accumulate.""")
end

"""
`mul!` writes its first argument and reads the rest, on every backend that has
one: rocBLAS, a cooperative-matrix kernel, Metal Performance Shaders. That is
not a fact about a driver, which is why it is declared here and not in a backend.

The five-argument form is the reason this dispatches on the argument types and
not on `typeof(mul!)` alone. `mul!(C, A, B, α, β)` is `C = A*B*α + C*β`, so `C`
is read as well as written whenever `β` is nonzero, and nothing here knows `β`.
`α` and `β` are scalars, which is no access at all — the first arguments anything
declares as `NOTOUCH`.
"""
argument_usage(::Type{typeof(LinearAlgebra.mul!)}, ::Type{<:Tuple{Any,Any,Any}}) =
    (WRITE, READ, READ)
argument_usage(::Type{typeof(LinearAlgebra.mul!)},
               ::Type{<:Tuple{Any,Any,Any,Number,Number}}) =
    (Touch(true, true, false), READ, READ, NOTOUCH, NOTOUCH)

"""
Cooperative-matrix memory operations have one backend-independent meaning.

Their implementations necessarily disappear behind backend intrinsics: Lava
emits named SPIR-V stubs while AMDGPU's native WMMA adapter expands to
`Core.LLVMPtr` loads/stores, represented by unnamed `llvmcall`s in inferred IR.
The access belongs to the portable operation, not to either representation, so
declare it here once. Matrix arithmetic has no pointer argument and remains
ordinary inferred code.
"""
argument_usage(::Type{typeof(coopmat_load)},
               ::Type{<:Tuple{Any,Any,Any,Any}}) =
    (NOTOUCH, READ, NOTOUCH, NOTOUCH)
argument_usage(::Type{typeof(coopmat_load)},
               ::Type{<:Tuple{Any,Any,Any,Any,Any}}) =
    (NOTOUCH, READ, NOTOUCH, NOTOUCH, NOTOUCH)
argument_usage(::Type{typeof(coopmat_store)},
               ::Type{<:Tuple{Any,Any,Any,Any}}) =
    (WRITE, NOTOUCH, NOTOUCH, NOTOUCH)
argument_usage(::Type{typeof(coopmat_store)},
               ::Type{<:Tuple{Any,Any,Any,Any,Any}}) =
    (WRITE, NOTOUCH, NOTOUCH, NOTOUCH, NOTOUCH)

# ── Which values can reach memory ────────────────────────────────────────────

"""
Whether a value of type `T` can lead to a memory access at all.

An index computed from a buffer's length is derived from the buffer and touches
nothing; taint that follows it would mark every argument whose SIZE a kernel
reads. So taint follows types that can REACH memory — a pointer, an array, or
anything not fully described by its bits — and stops at the ones that cannot.

Abstract types, `Union`s and `Any` answer `true`, because "cannot reach memory"
is a claim and an unknown type does not support it.

`Core.LLVMPtr` counts for exactly the same reason `Ptr` does, and has to be
named separately because it is NOT a `Ptr`: it is its own primitive type, with
no fields and `isbitstype`, so the field walk below reaches it and concludes a
device array touches nothing. That is what a GPU device array IS on every
backend that goes through GPUCompiler — Metal's `MtlDeviceArray` is an
`LLVMPtr` and a `Dims` — so without this a kernel's every argument comes back
`NOTOUCH`, the placer finds nothing uses its transients, and the graph refuses
to build.
"""
carries(@nospecialize(T)) = carries(T, 0)
rawpointer(@nospecialize(T)) = T <: Ptr || T <: Core.LLVMPtr
function carries(@nospecialize(T), depth::Int)
    depth > 8 && return true                    # deep enough to be unknowable
    T === Union{} && return false
    rawpointer(T) && return true
    isconcretetype(T) || return true
    isbitstype(T) || return true                # arrays, memory, mutables
    for F in fieldtypes(T)
        F === T && continue                     # a self-referential bits field cannot exist
        carries(F, depth + 1) && return true
    end
    return false
end

# ── The statements that touch memory ─────────────────────────────────────────
#
# By name rather than by `===` against the function object: the same operation
# appears as `Core.Intrinsics.pointerref`, as `Base.pointerref` and as a
# `GlobalRef` into whichever module inlined it, and a name is what all three
# agree on. Anything NOT here is opaque, which is the safe default — a new Base
# memory intrinsic is conservative rather than invisible.

const READ_OPS  = (:pointerref, :atomic_pointerref, :memoryrefget, :unsafe_load,
                   :arrayref, :getindex_unchecked)
const WRITE_OPS = (:pointerset, :atomic_pointerset, :memoryrefset!, :unsafe_store!,
                   :arrayset)
const RMW_OPS   = (:atomic_pointerswap, :atomic_pointerreplace, :atomic_pointermodify,
                   :memoryrefswap!, :memoryrefreplace!, :memoryrefmodify!)

"""Builtins and intrinsics that move a value around without touching memory."""
const PURE_OPS = (:getfield, :setfield!, :tuple, :bitcast, :add_ptr, :sub_ptr,
                  :memoryrefnew, :memoryref, :pointer, :unsafe_convert, :cconvert,
                  :typeassert, :ifelse, :sext_int, :zext_int, :trunc_int,
                  :ptrtoint, :inttoptr, :getproperty, :nfields, :sizeof,
                  :arraysize, :memorynew, :memoryrefoffset)

opname(@nospecialize(f)) = nothing
opname(f::GlobalRef) = f.name
opname(f::Core.IntrinsicFunction) = Symbol(string(f))
opname(f::Function) = nameof(f)

"""
Whether a call target is an intrinsic, RESOLVING a `GlobalRef` first.

`Base.add_int` reaches the walk as a `GlobalRef` and not as the intrinsic
object, so an `isa` test on the operand answers no for the arithmetic every
index is made of. That mattered twice: it made the whole of integer arithmetic
an unknown call whose operands widen to read+write, and it dropped the claim
that says a store lands where nothing else can.
"""
function iscoreintrinsic(@nospecialize(f))
    f isa Core.IntrinsicFunction && return true
    f isa GlobalRef || return false
    isdefined(f.mod, f.name) || return false
    return getglobal(f.mod, f.name) isa Core.IntrinsicFunction
end

# ── The walk ─────────────────────────────────────────────────────────────────

"""
Where a value may have come from: an argument of the signature being walked, and
— for the one argument that is a collapsed vararg tail — which element of it.

A `Set{Int}` of packed `(argument, element)` pairs. The sets are one or two
entries wide almost everywhere, and packing keeps a union a set union of
integers rather than of tuples.

**Why an element index at all.** A kernel written `f(queue, args...)` is ONE
argument as far as the inferred IR is concerned — the tail is collapsed into a
tuple — and a wavefront tracer's stages are all of that shape: `f` is the real
kernel and the other twenty arguments are the queues and the scene. Without the
element, every one of those twenty comes back with whatever any of them got, and
the stage that pushes one ray queue is recorded as writing all of them.

One level is all there is to track. The splat is fully expanded by the time the
walk sees it (`Core.getfield(_5, 3)`), and past that first `getfield` the value
is a resource whose leaves a declaration names whole.
"""
const Taint = Set{Int}

# 10 bits of element index, which is 1023 vararg arguments before the packing
# has to give up. It does not: an index past the range packs as 0, which means
# "the whole tail" and is the conservative answer.
const ELEMBITS = 10
const MAXELEM = (1 << ELEMBITS) - 1
pack(arg::Int, elem::Int) = (arg << ELEMBITS) | (0 <= elem <= MAXELEM ? elem : 0)
unpack(r::Int) = (r >> ELEMBITS, r & MAXELEM)

"""
The state one `accessof` shares across the method instances it walks into.

`summaries` is what makes an interprocedural walk affordable: a callee is
analysed once for its own arguments, and every call site maps that result onto
its own taint. `active` is the cycle guard — a recursive kernel gets `OPAQUE`
for the arguments it passes to itself rather than a stack overflow.
"""
struct Walk{I}
    interp::I
    summaries::IdDict{Any,Vector{Touch}}
    active::Base.IdSet{Any}
    maxdepth::Int
    # The signature `accessof` was asked about, so a refusal deep in an inlined
    # callee can name the kernel the caller dispatched rather than the frame it
    # happened in.
    top::Any
end

"""
The walk reached something it cannot see through, on a value that leads to one of
the arguments.

Thrown rather than widened to read+write. Widening is safe for the barrier phase
and indistinguishable from a proof, so a kernel that stopped being analysable
would keep compiling and pay a barrier per pass for ever with nothing to say so.
Four of the five constructs that would be widened cannot reach a compiled
kernel at all — a `ccall` is C, SPIR-V forbids recursion, and GPUCompiler rejects
a dynamic call — so this mostly moves an error that was going to happen anyway
from `Pipelines` to `dispatch!`, where the message can name the argument.
"""
struct UnanalysableAccess <: Exception
    top::Any
    positions::Vector{Int}
    reason::String
    stmt::Any
end

function Base.showerror(io::IO, e::UnanalysableAccess)
    # Position 1 of a signature is the function itself, so the caller's argument
    # numbering is one lower: `argument_usage` hands back the tail. An empty
    # `positions` is the whole-signature case, where naming them says less than
    # saying that none of it was read.
    args = isempty(e.positions) ? "any of its arguments" :
           "argument(s) " * join(string.(max.(e.positions .- 1, 0)), ", ")
    print(io, """
        argument_usage: cannot say what `$(e.top)` does to $args.

        The walk reached $(e.reason), and those arguments lead into it:

            $(e.stmt)

        It will not guess. If this is a construct the device cannot run either --
        a `ccall`, recursion, a call the optimiser left dynamic -- the backend
        was going to refuse it too, and the kernel is what needs changing. If it
        is a device intrinsic, declare what it does with `Mantle.intrinsic_usage`;
        if it is a callable with no body on this side, declare it with
        `Mantle.argument_usage`.""")
end


"""
How the arguments of an inferred `IRCode` line up with the signature it was
inferred at.

They differ exactly when the method takes a vararg: the signature is concrete
and twenty-six long, the IR has five arguments and the last one is a tuple of
the other twenty-two. `tail` is that argument's index, or 0 when there is none.
"""
struct ArgMap
    nsig::Int      # arguments of the signature, the function at 1
    tail::Int      # IR index of the collapsed vararg tuple, 0 if not collapsed
end

function argmap(ir, @nospecialize(sig))
    nsig = length(sig.parameters)
    nir = length(ir.argtypes)
    return ArgMap(nsig, nir < nsig ? nir : 0)
end

"""Signature positions a taint reaches, so a refusal can name them."""
function taintedpositions(am::ArgMap, taint::Taint)
    ps = Int[]
    for r in taint
        a, e = unpack(r)
        if a == am.tail && am.tail != 0
            if e == 0
                append!(ps, am.tail:am.nsig)
            else
                i = am.tail + e - 1
                i <= am.nsig && push!(ps, i)
            end
        elseif a <= am.nsig
            push!(ps, a)
        end
    end
    return sort!(unique!(ps))
end

"""
Refuse, unless the taint reaches no argument at all.

The guard is the whole precision of this: a `ccall` over values none of the
arguments lead to is invisible to the declaration, and erroring on it would
refuse kernels for a construct that cannot affect the answer.
"""
function refuse!(w::Walk, am::ArgMap, taint::Taint, reason::AbstractString,
                 @nospecialize(stmt))
    ps = taintedpositions(am, taint)
    isempty(ps) && return Taint()
    throw(UnanalysableAccess(w.top, ps, reason, stmt))
end

"""
The summaries one device has already worked out, reused across every graph it
builds.

A kernel's access is a property of `(function, argument types)` and of the method
table it will be compiled through, so two dispatches of one kernel — two rounds
of a bounce loop, two draws of one shader — ask the same question. Answering it
costs an inference pass over the whole kernel, which for a wavefront stage is ten
thousand statements, so asking twice is the difference between a graph that
builds in a second and one that takes a minute.

Keyed by signature, held by the DEVICE: the answers come from that device's
interpreter. Dropped whole when the world age moves, because a redefined method
is a different answer and nothing here could tell which.
"""
mutable struct AccessCache
    world::UInt64
    summaries::IdDict{Any,Vector{Touch}}
end
AccessCache() = AccessCache(Base.get_world_counter(), IdDict{Any,Vector{Touch}}())

function summaries!(c::AccessCache)
    w = Base.get_world_counter()
    if c.world != w
        empty!(c.summaries)
        c.world = w
    end
    return c.summaries
end
summaries!(::Nothing) = IdDict{Any,Vector{Touch}}()

"""
    accesscache(device) -> AccessCache

Where this device keeps the summaries above. A backend with nowhere to put them
answers `nothing` and pays the analysis every time.
"""
accesscache(::Device) = nothing

"""
    accessof(interp, f, argtypes) -> Vector{Touch}

What `f(argtypes...)` does to each of its arguments, in order. `interp` is the
`AbstractInterpreter` whose method table the kernel will be compiled against;
`nothing` means Julia's own, which is right for a backend that runs kernels as
ordinary Julia code.

The returned vector covers the whole signature: index 1 is `f` ITSELF, and
`argtypes` start at 2. `f`'s own entry is not a curiosity — a ray-tracing shader
is a closure over the material it shades, and what it does to what it captured is
exactly that entry.
"""
function accessof(interp, @nospecialize(f), @nospecialize(argtypes);
                  maxdepth::Int = 24, cache = nothing)
    sig = Tuple{typeof(f), argtypes...}
    w = Walk(interp, summaries!(cache), Base.IdSet{Any}(), maxdepth, sig)
    touches = signaturetouches(w, sig, 0)
    touches === nothing && throw(UnanalysableAccess(sig, Int[],
        "a signature it cannot infer: either no single method applies, or every " *
        "path through it throws", sig))
    return touches
end

"""
IR for one signature, through the backend's interpreter or Julia's own.

`nothing` for anything the walk must not read as an answer:

  * more than one method, or none — a dynamic call site, not ours to guess;
  * a signature that infers to `Union{}`, which means every path through it
    throws. The IR is then all unreachable and the walk finds no store — and
    "no store" is the one answer that must never be a guess, because it is the
    one that removes a barrier. A kernel that does not infer does not compile
    either, so the plan is going to fail; it should fail saying THAT.
"""
function irfor(w::Walk, @nospecialize(sig))
    irs = w.interp === nothing ? Base.code_ircode_by_type(sig) :
                                 Base.code_ircode_by_type(sig; interp = w.interp)
    length(irs) == 1 || return nothing
    ir, rt = irs[1]
    rt === Union{} && return nothing
    return ir
end

"""Per-argument `Touch` for a whole signature, or `nothing` if it cannot be seen."""
function signaturetouches(w::Walk, @nospecialize(sig), depth::Int)
    haskey(w.summaries, sig) && return w.summaries[sig]
    # A DECLARATION outranks the walk, and is checked before the depth limit and
    # before the recursion guard: the whole reason to declare a callable is that
    # reading it is either impossible or the wrong thing to read. A device
    # intrinsic is the case — `coopmat_load`'s body is an `llvmcall` whose LLVM
    # says `call @_lava_coopmat_load_f16_16x16_a`, and the only thing that knows
    # a load is a read is whoever emitted that name. Cached like an inferred
    # answer, since the lookup is a method dispatch per signature.
    declared = declaredsignature(sig)
    if declared !== nothing
        w.summaries[sig] = declared
        return declared
    end
    depth > w.maxdepth && return nothing
    sig in w.active && return nothing           # recursion: the caller widens
    any(p -> p isa Core.TypeofVararg, sig.parameters) && return nothing
    ir = irfor(w, sig)
    ir === nothing && return nothing
    push!(w.active, sig)
    try
        touches = walkir(w, ir, argmap(ir, sig), depth)
        w.summaries[sig] = touches
        return touches
    finally
        delete!(w.active, sig)
    end
end

"""
Whether a write at an index derived from `claim` is disjoint from every other
invocation's, given the write itself lands in `target`.

An atomic read-modify-write hands back a value no other invocation gets. A store
at that index is therefore to a slot nothing else writes — which is what
`Unordered` means for a work queue: a dozen stages append to one ray queue, each
claiming its slot from that queue's own counter, and ordering them against each
other buys nothing because they cannot collide.

The counter has to belong to the SAME argument the store lands in. Claiming a
slot from one queue and storing into another is not a disjointness argument, and
"they were both atomic somewhere" is exactly the reasoning that would make it
look like one.
"""
function disjoint(claim::Taint, target::Taint)
    isempty(claim) && return false
    for c in claim, t in target
        (c >> ELEMBITS) == (t >> ELEMBITS) && return true
    end
    return false
end

"""
The taint walk over one `IRCode`.

Two passes are not needed and one is not enough for loops: a value defined later
in the statement list can be used earlier through a back edge, so the fixed
point is reached by repeating until nothing changes. Kernels are small and the
lattice is a handful of bits, so this settles in two or three rounds.
"""
function walkir(w::Walk, ir, am::ArgMap, depth::Int)
    touches = fill(NOTOUCH, am.nsig)
    nstmt = length(ir.stmts)
    st = State([Taint() for _ in 1:nstmt], [Taint() for _ in 1:nstmt], BitSet())

    changed = true
    rounds = 0
    while changed && rounds < 16
        changed = false
        rounds += 1
        for i in 1:nstmt
            stmt = ir.stmts[i][:stmt]
            T = ir.stmts[i][:type]
            n, m, a = length(st.taints[i]), length(st.claims[i]),
                      length(st.addresses)
            newtaint = stmttaint!(w, ir, touches, st, am, i, stmt, T, depth)
            union!(st.taints[i], newtaint)
            (length(st.taints[i]) == n && length(st.claims[i]) == m &&
             length(st.addresses) == a) || (changed = true)
        end
    end
    return touches
end

"""
Per-statement state: where a value came from, and whether it is an index some
invocation claimed for itself.

Two channels rather than one lattice, because they answer different questions
and only meet at a store — `taints` says which argument's bytes are being
touched, `claims` says whether the touch can collide with another invocation's.
"""
struct State
    taints::Vector{Taint}
    claims::Vector{Taint}
    # SSA values that ARE a device address, whatever type they are inferred at.
    # See `pointerlike`.
    addresses::BitSet
end

"""
Taint of an operand, as the set of signature arguments it may come from.

An argument taints only if its type can reach memory. Otherwise a `Float32`
scale factor handed to a call the walk cannot see through would come back
read+write — an access on a value that has no bytes on the device to access.
"""
function operandtaint(ir, st::State, @nospecialize(x))
    x isa Core.Argument &&
        return carries(widen(ir.argtypes[x.n])) ? Taint((pack(x.n, 0),)) : Taint()
    x isa Core.SSAValue && return st.taints[x.id]
    return Taint()
end

"""The claim an operand carries: empty unless its value came from an atomic
read-modify-write, in which case it is the taint of what that atomic was on."""
function operandclaim(st::State, @nospecialize(x))
    x isa Core.SSAValue && return st.claims[x.id]
    return Taint()
end

"""
Whether a pure computation is working on a device ADDRESS rather than on a value.

`carries` asks what a TYPE can hold, and the answer for `UInt64` is nothing --
which is right for a length and wrong for an address, because a pointer laundered
through an integer is still the thing a store lands on. Provenance is the only
thing that tells the two apart, so it is tracked: a result whose operands include
a `Ptr`, or an integer already known to be an address, is an address too, and
`st.addresses` carries that across the arithmetic in between.

One hop, as a local `anyptr` test at the `getfield` and `PURE_OPS` branches, is
not enough. Lava's cooperative-matrix intrinsics take their operand as
`UInt64`:

    %1018 = getfield(C, :ptr)::Ptr{Float16}   # taint from C
    %1019 = bitcast(UInt64, %1018)            # kept: an operand is a `Ptr`
    %1031 = add_int(%1019, %1028)             # DROPPED: both operands are UInt64
            llvmcall(@_lava_coopmat_store_..., %1031, ...)

so `llvmcalltaint!` found no tainted operand and recorded nothing. Every
argument of `coopmat_gemm_kernel_4!` came back `NOTOUCH`, including its
destination, and `Place` then saw its output transient with no interval at all.
`gemm_cm2!` reads correctly because its intrinsics take a `Ptr` directly, which
is why only the block kernels were affected.
"""
pointerlike(ir, st::State, rest) =
    any(a -> rawpointer(operandtype(ir, a)) || isaddress(st, a), rest)

isaddress(st::State, @nospecialize(x)) = x isa Core.SSAValue && x.id in st.addresses

"""
The type an operand was inferred at, for `carries` and for `pointerlike`.

`widen` on BOTH sides. This read `ir.argtypes[x.n]` raw, and an argument can be
inferred as a `Core.Const` — so `pointerlike`'s `<: Ptr` was a `TypeError: in <:,
expected Type, got a value of type Core.Const` on seven of MatAnyone's graphs.
`operandtaint` widens the same field two functions above, which is why this
survived: one of the two readers of `argtypes` did it and the other did not.
"""
function operandtype(ir, @nospecialize(x))
    x isa Core.Argument && return widen(ir.argtypes[x.n])
    x isa Core.SSAValue && return widen(ir.stmts[x.id][:type])
    return typeof(x)
end

widen(@nospecialize(T)) = T isa Core.Const ? typeof(T.val) :
                          T isa Type ? T : Any

"""
Record `t` against every signature argument the tainted operands came from.

A root on the collapsed vararg tail lands on ONE signature argument when the
element is known, and on all of them when it is not — which is what an opaque
call handed the whole tuple means.
"""
function record!(touches::Vector{Touch}, am::ArgMap, taint::Taint, t::Touch)
    for r in taint
        a, e = unpack(r)
        if a == am.tail && am.tail != 0
            if e == 0
                for i in am.tail:am.nsig
                    touches[i] = touches[i] | t
                end
            else
                i = am.tail + e - 1
                i <= am.nsig && (touches[i] = touches[i] | t)
            end
        elseif a <= length(touches)
            touches[a] = touches[a] | t
        end
    end
end

"""
An element of the collapsed vararg tail, when a `getfield` names one.

Only that tail: for every other argument the graph declares a resource's leaves
whole, so a field index would be carried and then thrown away.
"""
function fieldtaint(am::ArgMap, taint::Taint, @nospecialize(idx))
    (am.tail == 0 || !(idx isa Integer)) && return taint
    out = Taint()
    for r in taint
        a, e = unpack(r)
        push!(out, (a == am.tail && e == 0) ? pack(a, Int(idx)) : r)
    end
    return out
end

"""
One statement: what it does to memory, and what its result is tainted by.

The result taint is propagated when the statement's TYPE can reach memory
(`carries`) or its PROVENANCE says it is an address ([`pointerlike`](@ref)).
Type alone is not enough and was the first version's bug in both directions:
following every integer marked every argument whose length the kernel read, and
following none of them lost a store through an address laundered into a `UInt64`.
"""
function stmttaint!(w::Walk, ir, touches, st::State, am::ArgMap, i::Int,
                    @nospecialize(stmt), @nospecialize(T), depth::Int)
    out = Taint()
    tt = widen(T)

    if stmt isa Core.PhiNode || stmt isa Core.PhiCNode
        # Taint passes a phi unconditionally -- a merge is not a computation --
        # but the address flag has to cross it too, or an address that arrives
        # through a loop-carried value stops being one.
        vals = Any[stmt.values[j] for j in eachindex(stmt.values)
                   if isassigned(stmt.values, j)]   # unassigned: undefined on that edge
        for v in vals
            union!(out, operandtaint(ir, st, v))
        end
        pointerlike(ir, st, vals) && push!(st.addresses, i)
        return out
    elseif stmt isa Core.PiNode
        pointerlike(ir, st, (stmt.val,)) && push!(st.addresses, i)
        return operandtaint(ir, st, stmt.val)
    elseif stmt isa Core.UpsilonNode
        isdefined(stmt, :val) || return out
        pointerlike(ir, st, (stmt.val,)) && push!(st.addresses, i)
        return operandtaint(ir, st, stmt.val)
    elseif stmt isa Core.ReturnNode || stmt isa Core.GotoNode ||
           stmt isa Core.GotoIfNot || stmt isa Nothing || stmt isa Core.SSAValue ||
           stmt isa Core.Argument
        return stmt isa Core.SSAValue || stmt isa Core.Argument ?
               operandtaint(ir, st, stmt) : out
    elseif !(stmt isa Expr)
        return out
    end

    head = stmt.head
    # A CALL that infers to `Union{}` does not return: it throws, and the
    # statement after it is `unreachable`. Nothing it did is observable, so it
    # orders nothing and touches nothing.
    #
    # Without this, every bounds check poisons the array it guards.
    # `throw_boundserror(::MtlDeviceVector, ::Tuple{Int})` is an `:invoke` with
    # the array as an argument and no IR the walk can enter — `irfor` refuses a
    # `Union{}` signature by design — so the fallback marked it read+write, and a
    # vertex buffer a shader only READS came back written. The graph then put a
    # host-update pass in front of a render pass that needed none.
    #
    # It is not the same question `irfor` answers. There, `Union{}` is the whole
    # signature and "this kernel does not infer" must fail loudly rather than
    # silently report no store. Here the caller infers fine and one callee is a
    # throw. A function that stores and THEN throws would be missed, and that is
    # accepted: the dispatch is a crash, and passes are not ordered around one.
    if tt === Union{} && (head === :invoke || head === :call)
        return Taint()
    end
    if head === :invoke
        return invoketaint!(w, ir, touches, st, am, i, stmt, tt, depth)
    elseif head === :call
        return calltaint!(w, ir, touches, st, am, i, stmt.args, tt, depth)
    elseif head === :new || head === :splatnew
        fields = stmt.args[2:end]
        for a in fields
            union!(out, operandtaint(ir, st, a))
        end
        return keptaddress!(st, i, ir, fields, tt) ? out : Taint()
    elseif head === :foreigncall
        # Linked device functions use `ccall("extern name", llvmcall, ...)`,
        # which inference retains as a foreigncall even though the GPU compiler
        # links it into the device pipeline.  Its globally unique symbol can be
        # declared by the backend just like an inline llvmcall intrinsic.
        rawname = stmt.args[1]
        name = rawname isa QuoteNode ? rawname.value : rawname
        namestr = name isa Symbol ? String(name) :
                  name isa String ? name : string(name)
        m = match(r"([A-Za-z_][A-Za-z0-9_.$]*)$", namestr)
        if m !== nothing
            usage = intrinsic_usage(Symbol(m.captures[1]))
            if usage !== nothing
                for a in stmt.args
                    record!(touches, am, operandtaint(ir, st, a), usage)
                end
                return Taint()
            end
        end
        # An ordinary `ccall` is host C, and nothing here reads C. It is not
        # compilable for a device, so refusing costs nothing a kernel could use.
        for a in stmt.args
            refuse!(w, am, operandtaint(ir, st, a), "a `ccall`, which is C", stmt)
        end
        return Taint()
    elseif head === :boundscheck || head === :aliasscope || head === :popaliasscope ||
           head === :leave || head === :enter || head === :meta || head === :gc_preserve_begin ||
           head === :gc_preserve_end || head === :throw_undef_if_not || head === :loopinfo
        return Taint()
    else
        for a in stmt.args
            a isa Core.SSAValue || a isa Core.Argument || continue
            refuse!(w, am, operandtaint(ir, st, a),
                    "an expression head it has no case for, `:$head`", stmt)
        end
        return Taint()
    end
end

"""A call whose callee is resolved: walk into it and map its summary back."""
function invoketaint!(w::Walk, ir, touches, st::State, am::ArgMap, i::Int,
                      stmt::Expr, @nospecialize(tt), depth::Int)
    ci = stmt.args[1]
    mi = ci isa Core.CodeInstance ? ci.def : ci
    sig = mi isa Core.MethodInstance ? mi.specTypes : nothing
    callargs = stmt.args[2:end]
    # A call that CANNOT RETURN is an abort path, and contributes nothing.
    # `throw_boundserror`, `throw_methoderror` and every other thrower infers to
    # `Union{}`, and they are reached from the bounds check and the `setindex!`
    # fallback of ordinary kernels -- so this is not an edge case, it is most
    # kernels. Refusing here refused them all, and walking in refuses too: a
    # thrower boxes its arguments into an error and raises through a
    # `:foreigncall`, which is the construct above.
    #
    # Sound, not a convenience. A store that happens only on the way to a throw
    # cannot be observed by a later pass, because the invocation does not
    # complete and the graph's subsequent passes are reading a crashed kernel's
    # output either way. A barrier for it buys nothing.
    tt === Union{} && return Taint()
    inner = sig === nothing ? nothing : signaturetouches(w, sig, depth + 1)
    out = Taint()
    for (j, a) in enumerate(callargs)
        at = operandtaint(ir, st, a)
        isempty(at) && continue
        if inner === nothing
            # No single method, a signature that infers to `Union{}`, recursion,
            # or past the depth limit -- `signaturetouches` says which by
            # returning nothing, and none of them is an answer.
            refuse!(w, am, at,
                    "a call to `$(sig === nothing ? "an unresolved callee" : sig)` " *
                    "it could not summarise: no single method applies, every path " *
                    "through it throws, it recurses, or it nests deeper than the " *
                    "walk goes", stmt)
        elseif j > length(inner)
            refuse!(w, am, at,
                    "a call whose summary is shorter than its argument list, " *
                    "which means the signature and the IR disagree about arity",
                    stmt)
        else
            record!(touches, am, at, inner[j])
        end
        union!(out, at)
    end
    # Same question as at the `PURE_OPS` branch: the callee's own accesses are
    # recorded already, so what is left is whether its RESULT is an address.
    # `pointer(A)` answers `Ptr` and `carries` covers it; a callee that hands the
    # same address back as a `UInt64` needs the provenance.
    return keptaddress!(st, i, ir, callargs, tt) ? out : Taint()
end

"""
Whether the result of a pure computation keeps its operands' taint, noting it as
an address when that is why.

`carries(tt)` is the type-level question and [`pointerlike`](@ref) the
provenance one; either is enough, and the second is the one that has to be
remembered for the statements downstream.
"""
function keptaddress!(st::State, i::Int, ir, rest, @nospecialize(tt))
    if pointerlike(ir, st, rest)
        push!(st.addresses, i)
        return true
    end
    return carries(tt)
end

"""
A builtin, an intrinsic, or a call the optimiser left dynamic.

The memory operations are named in `READ_OPS` / `WRITE_OPS` / `RMW_OPS`; a
handful more move values without touching memory. Everything else is opaque,
including `llvmcall` — Lava's atomics are llvmcall, and an atomic is a read and
a write whatever else it is.
"""
function calltaint!(w::Walk, ir, touches, st::State, am::ArgMap, i::Int,
                    args::Vector{Any}, @nospecialize(tt), depth::Int)
    f = args[1]
    name = opname(f)
    rest = @view args[2:end]
    out = Taint()
    for a in rest
        union!(out, operandtaint(ir, st, a))
    end

    if name in READ_OPS
        record!(touches, am, operandtaint(ir, st, rest[1]), READ)
        return Taint()
    elseif name in WRITE_OPS
        target = operandtaint(ir, st, rest[1])
        record!(touches, am, target, storetouch(ir, st, rest, target))
        return Taint()
    elseif name in RMW_OPS
        t = operandtaint(ir, st, rest[1])
        record!(touches, am, t, ATOMIC)
        # What this atomic handed back is a slot no other invocation gets.
        union!(st.claims[i], t)
        return Taint()
    elseif name === :llvmcall
        return llvmcalltaint!(w, ir, touches, st, am, i, rest, args)
    elseif name === :getfield && length(rest) >= 2
        # The one place an element index is worth keeping: `Core.getfield(_5, 3)`
        # is how the expanded splat reaches the third vararg argument.
        t = fieldtaint(am, operandtaint(ir, st, rest[1]), rest[2])
        return keptaddress!(st, i, ir, rest, tt) ? t : Taint()
    elseif name in PURE_OPS || iscoreintrinsic(f)
        # A claimed index is still a claimed index after the arithmetic that
        # turns it into an offset, so the claim rides along with the value.
        for a in rest
            union!(st.claims[i], operandclaim(st, a))
        end
    end

    if name in PURE_OPS
        return keptaddress!(st, i, ir, rest, tt) ? out : Taint()
    elseif name === :apply_type || name === :isa || name === :(===) ||
           name === :typeof || name === :not_int
        return Taint()
    end

    # A call the optimiser could not resolve, or a Base function we have no
    # summary for: whatever it was given, assume the worst about.
    if iscoreintrinsic(f)
        # An intrinsic that is not one of the memory operations above computes on
        # values -- but the values an address is computed FROM are addresses, so
        # this is the same question as at the `PURE_OPS` branch above.
        return keptaddress!(st, i, ir, rest, tt) ? out : Taint()
    end
    # Same as in `invoketaint!`: a call that cannot return is an abort path.
    # `Core.throw_methoderror` arrives here rather than there, because the
    # optimiser leaves it unresolved.
    tt === Union{} && return Taint()
    for a in rest
        refuse!(w, am, operandtaint(ir, st, a),
                "a call to `$f` it has no summary for and could not resolve, " *
                "which a device compiler rejects as a dynamic invocation",
                Expr(:call, args...))
    end
    return carries(tt) ? out : Taint()
end

"""
A store: ordered, unless the slot it lands in was claimed.

`pointerset(ptr, value, index, align)` — the VALUE is deliberately not consulted.
Storing a claimed index somewhere says nothing about where the store landed;
only the pointer and the index do.
"""
function storetouch(ir, st::State, rest, target::Taint)
    claim = operandclaim(st, rest[1])
    length(rest) >= 3 && union!(claim, operandclaim(st, rest[3]))
    return disjoint(claim, target) ? ATOMIC : WRITE
end

"""
    intrinsic_usage(name::Symbol) -> Touch or nothing

What the external symbol an `llvmcall` calls does to the pointer it is handed.

A backend spells a device intrinsic as a `declare` plus a `call`, so by the time
the walk sees it there is no Julia callee left to ask — the `@generated` wrapper
was inlined — and the module contains no memory instruction either. The NAME is
the only thing left, and the only thing that knows what it means is whoever
emitted it. So this is a declaration, like [`argument_usage`](@ref), and for the
same reason: there is nothing to read.

`nothing` means undeclared, which is not permission to guess.

No device parameter: an intrinsic symbol is globally unique, and `_lava_*` is
Lava's whatever asks. The default is untyped rather than `::Symbol` so that a
backend's method ADDS to it instead of overwriting it, which is not permitted
during precompilation and cost a silent fallback to running Mantle from source.
"""
intrinsic_usage(@nospecialize(name)) = nothing

"""
    llvmcallee(src) -> Symbol or nothing

The one external symbol an `llvmcall`'s module declares, or `nothing` if it
declares none or several.
"""
function llvmcallee(src::AbstractString)
    ms = collect(eachmatch(r"declare[^@\n]*@([A-Za-z0-9_.$]+)\s*\(", src))
    length(ms) == 1 || return nothing
    return Symbol(ms[1].captures[1])
end

"""
What an `llvmcall` carrying a tainted operand does to it.

The INSTRUCTIONS in its module are authoritative and are read first: an
`atomicrmw` or a `cmpxchg` is a commutative read-modify-write, which is the case
`Unordered` exists for, and a `load` or `store` instruction is what it says.

Matched as instructions, anchored to the start of a line, and not as substrings
anywhere in the text. `declare i32 @_lava_coopmat_load_f16_16x16_a(i64, i32)`
contains the word `load` and is not one. `occursin("load ", src)` misses that
name by a space, so every cooperative-matrix load falls through to read+write
and `gemm_cm2!` declares its two `@Const` operands as WRITTEN; without the space
it comes out READ for the same
bad reason.

A module with no memory instruction at all is a `declare` plus a `call`, which is
how every backend spells an intrinsic; what that symbol does is
[`intrinsic_usage`](@ref)'s to say.
"""
const RMW_INSTR   = r"(?m)^\s*(%[^=\n]*=\s*)?(atomicrmw|cmpxchg)\s"
const STORE_INSTR = r"(?m)^\s*store\s"
const LOAD_INSTR  = r"(?m)^\s*%[^=\n]*=\s*load\s"
const ADDRESS_INSTR = r"(?m)^\s*%[^=\n]*=\s*(addrspacecast|bitcast|getelementptr)\b"

function llvmcallusage(src)
    src === nothing && return nothing
    occursin(RMW_INSTR, src) && return ATOMIC
    occursin(STORE_INSTR, src) && return WRITE
    occursin(LOAD_INSTR, src) && return READ
    callee = llvmcallee(src)
    callee === nothing && return nothing
    return intrinsic_usage(callee)
end

"""
`llvmcall` is where a backend's device intrinsics live, and the IR it carries is
the only thing that says what they do. Two are worth reading rather than
widening: an `atomicrmw` is a commutative read-modify-write — the case
`Unordered` exists for — and an intrinsic that takes no tainted pointer at all
(a workgroup id, a subgroup reduce) touches nothing.
"""
function llvmcalltaint!(w::Walk, ir, touches, st::State, am::ArgMap, i::Int,
                        rest, args::Vector{Any})
    src = llvmsource(rest[1], ir)
    tainted = [a for a in rest if !isempty(operandtaint(ir, st, a))]
    isempty(tainted) && return Taint()
    # LLVM.jl implements `LLVMPtr` casts and arithmetic as tiny `llvmcall`
    # modules containing `addrspacecast`, `bitcast`, or `getelementptr`. They
    # neither read nor write; their result is another spelling of the same
    # address. Preserve provenance so the later load/store is attributed to the
    # original argument. This is an LLVM fact, not an AMDGPU declaration, and
    # belongs in the common walk. A module containing any memory instruction
    # still takes the authoritative paths below.
    if src !== nothing && occursin(ADDRESS_INSTR, src) &&
       !occursin(RMW_INSTR, src) && !occursin(STORE_INSTR, src) &&
       !occursin(LOAD_INSTR, src)
        push!(st.addresses, i)
        out = Taint()
        foreach(a -> union!(out, operandtaint(ir, st, a)), tainted)
        return out
    end
    t = llvmcallusage(src)
    if t === nothing
        callee = src === nothing ? "an `llvmcall` whose module could not be read" :
                 "the intrinsic `$(something(llvmcallee(src), "(unnamed)"))`"
        for a in tainted
            refuse!(w, am, operandtaint(ir, st, a),
                    "$callee, which nothing declares -- it carries no load, " *
                    "store or atomic instruction, so its NAME is the only thing " *
                    "that says what it does", Expr(:call, args...))
        end
        # `refuse!` is silent when the taint reaches no argument of the
        # signature, and then there is nothing to record either.
        return Taint()
    end
    for a in tainted
        record!(touches, am, operandtaint(ir, st, a), t)
        t === ATOMIC && union!(st.claims[i], operandtaint(ir, st, a))
    end
    return Taint()
end

"""
The LLVM source of an `llvmcall`: written inline, built into a tuple by the IR,
or — the usual case for a backend's intrinsic library — named by a `GlobalRef` to
a const holding it. Lava's atomics are the third: `Lava.ir_fadd_f32` IS the
module, and reading it is what tells an atomic add apart from an opaque call.
"""
llvmsource(@nospecialize(x), ir) = nothing
llvmsource(x::String, ir) = x
llvmsource(x::Tuple{String,String}, ir) = x[1]
function llvmsource(x::GlobalRef, ir)
    isdefined(x.mod, x.name) || return nothing
    return llvmsource(getglobal(x.mod, x.name), ir)
end
function llvmsource(x::Core.SSAValue, ir)
    stmt = ir.stmts[x.id][:stmt]
    stmt isa Expr && stmt.head === :call && length(stmt.args) >= 2 || return nothing
    a = stmt.args[2]
    return a isa String ? a : nothing
end

# ── What a backend answers ───────────────────────────────────────────────────

"""
    kerneltouches(device, kernel, args, ndrange) -> Vector{Touch}

What `kernel` does to each element of `args`, when dispatched on `device`.

The backend's job is the SIGNATURE — which interpreter the kernel will be
compiled through, what the device-side type of each argument is, and what the
kernel's own leading arguments are (an iteration context, a kernel state). The
walk itself is `accessof` and is the same for every backend, because "does this
kernel store through this pointer" is not a property of a driver.
"""
function kerneltouches(dev, kernel, args::Tuple, ndrange, group)
    argT = map(a -> devicetype(dev, a), args)
    if buildskernel(kernel, backend(dev))
        interp, body, tt = kakernelaccesssignature(dev, kernel, argT, ndrange, group)
        return accessof(interp, body, tt; cache = accesscache(dev))[3:end]
    end
    # Refuse an unsupported macro-free kernel at the same boundary as `bake`.
    # Walking its host fallback first reports whichever device intrinsic throws
    # during inference, hiding the actionable fact that this device has no KI
    # compiler at all.
    kisupported(dev, kernel) || kikernel(kernel, dev, map(a -> resolve(dev, a), args))
    interp = kernelinterpreter(dev, kernel, Tuple{argT...})
    return accessof(interp, kernel, argT; cache = accesscache(dev))[2:end]
end

"""The interpreter a backend compiles a macro-free kernel through."""
function kernelinterpreter end

"""
    kakernelaccesssignature(device, kernel, argtypes, ndrange, group)

Interpreter, body and signature of a retained KernelAbstractions kernel.

The ordinary KA launch shape is backend-independent: construct the retained
kernel, ask KA for its iteration space, make its context, then ask the device
for the compiler interpreter. Backends override this only when the signature
they actually compile differs from KA's ordinary launch (Lava retains an
`IterPlan`; the host constructs one block's CPU context).
"""
function kakernelaccesssignature(dev, kernel, argT::Tuple, ndrange, group)
    obj = kernelfor(kernel, group, backend(dev))
    nd = dispatchrange(ndrange)
    cg = callgroup(obj, group)
    ndr, _, iterspace, _ = KA.launch_config(obj, nd, cg)
    ctx = KA.mkcontext(obj, ndr, iterspace)
    tt = (typeof(ctx), argT...)
    return kernelinterpreter(dev, obj.f, Tuple{tt...}), obj.f, tt
end

"""
    devicetype(device, x) -> Type

The type a kernel's argument `x` arrives as.

`resolve` already answers what a launch passes, and on a backend that adapts its
arguments the adapted form of that is what a shader receives — so the default is
derived from the same rules a dispatch is packed with rather than from a second
table that could disagree with them.

A transient is the one case that cannot go through `resolve`: it has no storage
until `Place` has run, and this is asked while the graph is still being built.
Its device type is a function of its element type alone, and a backend states
that as [`devicebuffertype`](@ref).
"""
devicetype(dev::Device, x) = argtype(dev, resolve(dev, x))
# A transient's RANK is its own, not 1. `Place` gives it an array of the dims it
# was declared with, so answering rank 1 made the walk infer the kernel against a
# 1-D `getindex` where the dispatch is packed with a 2-D one -- which is the
# regression `devicebuffertype`'s docstring claims not to have. Measured: every
# GEMM whose operand was a transient declined the cooperative-matrix path,
# because that path asks for a rank-2 operand and got told rank 1.
devicetype(dev::Device, ::TransientBuffer{T,N}) where {T,N} =
    devicebuffertype(dev, T, N)
# A VIEW of a transient cannot go through `resolve` either, and for one step
# further along the same reason: `storage(::ResourceView)` is
# `deriveview(T, storage(parent), …)`, so it asks the parent for bytes that
# `Place` has not handed out yet. `GPUArrays.derive` answers an array of the
# view's own rank, which is why this asks for `N` and not `1`.
#
# Uniform over views rather than only transient-rooted ones: a view of a placed
# `Buffer` derives to the same type, so distinguishing by root would be two
# answers to one question.
devicetype(dev::Device, ::ResourceView{T,N}) where {T,N} = devicebuffertype(dev, T, N)

# A RANGE is the other half of a view, and its answer is the PARENT's: a range
# declares what a pass touches and the kernel still receives the whole buffer
# (see `BufferRange`'s docstring). `storage(::BufferRange)` already says that on
# the value side and this is its companion, so the two cannot disagree.
#
# Found by `refusenothing`: without it a range fell through to
# `argtype(dev, resolve(dev, x))`, which answers the HANDLE type, so the walk
# inferred the kernel against a `Mantle.BufferRange` that has no device
# `setindex!` and saw no store at all. `test_window.jl`'s "disjoint slices of one
# buffer are not ordered against each other" was passing for that reason and not
# for the one it names -- an all-`NOTOUCH` declaration orders nothing.
devicetype(dev::Device, r::BufferRange) = devicetype(dev, r.parent)

# A CONTAINER of them, recursing through `devicetype` and not through `resolve`.
# `resolve(::Tuple)` maps `resolve` over the elements, which is right for a
# launch and wrong here for the same reason the two methods above exist: an
# element that is a transient, or a view of one, has no storage yet. The operand
# tuple `ew!` takes is exactly this shape -- one kernel over any number of
# operands -- so it is not a corner.
devicetype(dev::Device, x::Tuple) = Tuple{map(a -> devicetype(dev, a), x)...}
devicetype(dev::Device, x::NamedTuple{K}) where {K} =
    NamedTuple{K, Tuple{map(a -> devicetype(dev, a), values(x))...}}

"""
    argtype(device, y) -> Type

What a launch on this device turns an already-`resolve`d value into. The default
is nothing at all — the host backend hands a kernel the array it resolved — and a
backend that adapts its arguments on the way to the shader answers with the
adapted type, through the same `Adapt` rules the dispatch is packed with.

One-argument dispatch on purpose: `devicetype` above has the two cases, so a
backend never has to repeat the transient one and cannot be ambiguous with it.

`Core.Typeof` and not `typeof`, here and in every backend's method. What the
walk needs is the type the kernel is SPECIALISED at, and for a type-valued
argument the two differ: `typeof(Float32)` is `DataType`, which is not a
dispatch-tuple element, so `methodinstance` asserts
`Base.isdispatchtuple(sig)` before anything is inferred. `Core.Typeof(Float32)`
is `Type{Float32}`, which is what the kernel is actually compiled at and what
GPUCompiler then drops as a compile-time constant — the same fact
`graph/packing.jl` states from the other side, where writing a slot for it is a
segfault. `kikernel` in `graph/kalaunch.jl` already built its `tt` this way.
"""
argtype(::Device, @nospecialize(y)) = Core.Typeof(y)

"""
    devicebuffertype(device, T, N = 1) -> Type

The device-side type of a one-dimensional buffer of `T` — what a kernel sees
when it is handed a transient. `Place` produces exactly this type, and the
regression for it is that a materialised transient's `devicetype` and the type
its dispatch is actually packed with are the same.
"""
function devicebuffertype end
# A transient is one-dimensional; a view of one carries its own rank. One
# argument fewer at the common call site, and a backend answers the general form.
devicebuffertype(dev, ::Type{T}) where {T} = devicebuffertype(dev, T, 1)
