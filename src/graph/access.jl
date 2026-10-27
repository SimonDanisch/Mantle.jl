# What a kernel does to its arguments, read off the kernel by a taint walk over the
# typed IR the backend is about to compile, through the backend's interpreter: a
# device array's `getindex` exists only on its method table (see `kerneltouches`).
# A construct the walk cannot see through, on a value that reaches an argument,
# throws `UnanalysableAccess` rather than widening to read+write.

"""
What one dispatch does to one argument. `atomic` is a property of the writes: a
resource whose every write is an atomic read-modify-write is `Unordered`, so passes
sharing it need no barrier between them.
"""
struct Touch
    read::Bool
    write::Bool
    atomic::Bool     # every write so far was an atomic read-modify-write
end

const NOTOUCH  = Touch(false, false, true)
"""
An ordinary load. `atomic` is false so an ordinary read keeps an atomically written
resource from being `Unordered`: a pass reading a queue others append to is a hazard.
"""
const READ     = Touch(true, false, false)
"""A plain store: two racing stores are a race, so `atomic` is false."""
const WRITE    = Touch(false, true, false)
const ATOMIC   = Touch(true, true, true)
"""Read and written, not atomic. An access the walk cannot see throws instead."""
const OPAQUE   = Touch(true, true, false)

Base.:|(a::Touch, b::Touch) = Touch(a.read | b.read, a.write | b.write, a.atomic & b.atomic)
touched(t::Touch) = t.read | t.write

"""
    usagetype(t::Touch, kind) -> Type

The barrier-phase usage for `t`: `Unordered` only when there is a write and every
write is atomic; a merely read argument is ordered by its writers.
"""
function usagetype(t::Touch, kind::ResourceKind)
    S = Storage{typeof(kind), Access{t.read, t.write}}
    return (t.write && t.atomic) ? Unordered{S} : S
end

"""
    argument_usage(f, args)          -> NTuple{N,Touch} or nothing
    argument_usage(F::Type, A::Type) -> NTuple{N,Touch} or nothing

What `f` does to each of `args` where that is declared rather than inferred;
`nothing` means undeclared, never permission to guess. `dispatch!` asks for library
calls with no body to read (`mul!` into rocBLAS), and the taint walk asks before
descending into any call, so a device intrinsic states its access. Declared on
types, which is all the walk has; the value form forwards through `Core.Typeof`. No
device parameter: what a function does to its arguments is not a driver fact. A
declared callable must not close over device memory; its own entry gets `NOTOUCH`.
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

"""
    argument_usage(device, f, args, ndrange, group) -> Vector{Touch}

What `f` does to each of `args` when dispatched on `device`: the declaration
`argument_usage(f, args)` if there is one, else [`kerneltouches`](@ref). A call
(no ndrange) without a declaration throws `UndeclaredCall`. Nothing widens to
read+write: that answer is indistinguishable from a proof and would silently cost
a barrier per pass.
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
    # One entry per argument so the suggestion can be pasted; `WRITE` first since
    # destination-first calls dominate.
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
`mul!` writes its first argument and reads the rest on every backend, so it is
declared in core. The five-argument form `C = A*B*α + C*β` also reads `C` (`β` is
unknown here); `α` and `β` are scalars and touch nothing.
"""
argument_usage(::Type{typeof(LinearAlgebra.mul!)}, ::Type{<:Tuple{Any,Any,Any}}) =
    (WRITE, READ, READ)
argument_usage(::Type{typeof(LinearAlgebra.mul!)},
               ::Type{<:Tuple{Any,Any,Any,Number,Number}}) =
    (Touch(true, true, false), READ, READ, NOTOUCH, NOTOUCH)

"""
Cooperative-matrix loads and stores have one backend-independent access, declared
here because each backend hides them behind intrinsics (named SPIR-V stubs on Lava,
unnamed `llvmcall`s on AMDGPU). Matrix arithmetic has no pointer and is inferred.
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

"""
Whether a value of type `T` can lead to a memory access. Taint follows only such
types, so an index computed from a buffer's length does not mark the buffer.
Abstract types, `Union`s and `Any` answer `true`. `Core.LLVMPtr` counts like `Ptr`
but must be named: it is an `isbitstype` primitive, and a GPUCompiler device array
(e.g. `MtlDeviceArray`) is an `LLVMPtr` plus dims, so without it every argument
reads as `NOTOUCH`.
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

# ── The statements that touch memory, by name: one operation appears as an
# intrinsic, a Base function and a `GlobalRef`, and the name is what they share.
# Anything not listed is opaque.

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
Whether a call target is an intrinsic, resolving a `GlobalRef` first: `Base.add_int`
reaches the walk as a `GlobalRef`, and missing it makes all index arithmetic opaque.
"""
function iscoreintrinsic(@nospecialize(f))
    f isa Core.IntrinsicFunction && return true
    f isa GlobalRef || return false
    isdefined(f.mod, f.name) || return false
    return getglobal(f.mod, f.name) isa Core.IntrinsicFunction
end

"""
Where a value may have come from: a set of packed `(argument, element)` pairs. The
element names one entry of a collapsed vararg tail (`f(queue, args...)` is one IR
argument), so a stage touching one queue is not recorded as touching all of them.
One level suffices: the splat is expanded at its first `getfield`.
"""
const Taint = Set{Int}

# 10 bits of element index; an index past 1023 packs as 0, "the whole tail", which
# is conservative.
const ELEMBITS = 10
const MAXELEM = (1 << ELEMBITS) - 1
pack(arg::Int, elem::Int) = (arg << ELEMBITS) | (0 <= elem <= MAXELEM ? elem : 0)
unpack(r::Int) = (r >> ELEMBITS, r & MAXELEM)

"""
What a signature does to its arguments, and how long that holds: while `ci`, the
`CodeInstance` its IR was inferred as, is valid. Julia ends a `CodeInstance`'s
world range when a method its code depended on is redefined, through backedges
and so transitively for every caller, which makes a summary stale exactly when
the code it was read from changes. A declared signature (`declaredsignature`), or
one with no `CodeInstance`, holds in the world it was worked out in.
"""
struct Summary
    touches::Vector{Touch}
    ci::Union{Nothing,Core.CodeInstance}
    world::UInt
end

holds(s::Summary, world::UInt) =
    s.ci === nothing ? s.world == world : (@atomic :monotonic s.ci.max_world) >= world

"""
State one `accessof` shares across the method instances it walks. `summaries`
caches each callee's result per signature; `active` detects recursion, which is
refused.
"""
struct Walk{I}
    interp::I
    summaries::IdDict{Any,Summary}
    active::Base.IdSet{Any}
    maxdepth::Int
    # The signature `accessof` was asked about, so a refusal in a callee names the
    # dispatched kernel.
    top::Any
end

"""
The walk reached a construct it cannot see through, on a value that leads to an
argument. Thrown rather than widened, which would be indistinguishable from a
proof. Most such constructs (`ccall`, recursion, dynamic calls) cannot compile for a
device anyway; this reports them at `dispatch!`, naming the argument.
"""
struct UnanalysableAccess <: Exception
    top::Any
    positions::Vector{Int}
    reason::String
    stmt::Any
end

function Base.showerror(io::IO, e::UnanalysableAccess)
    # Signature position 1 is the function, so argument numbers are one lower.
    # Empty `positions`: none of the signature could be read.
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
How an inferred `IRCode`'s arguments line up with its signature. They differ for a
vararg method, whose tail the IR collapses into one tuple; `tail` is that argument's
index, 0 if none.
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
Throw `UnanalysableAccess` unless `taint` reaches no argument: a construct over
values no argument leads to cannot affect the answer.
"""
function refuse!(w::Walk, am::ArgMap, taint::Taint, reason::AbstractString,
                 @nospecialize(stmt))
    ps = taintedpositions(am, taint)
    isempty(ps) && return Taint()
    throw(UnanalysableAccess(w.top, ps, reason, stmt))
end

"""
Summaries one device has worked out, reused across every graph it builds; a
wavefront stage is ten thousand statements to infer. Keyed by signature and held
per device, since answers come from its interpreter. Each holds while the code it
was read from does ([`Summary`](@ref)), so defining an unrelated method costs
nothing.
"""
struct AccessCache
    summaries::IdDict{Any,Summary}
end
AccessCache() = AccessCache(IdDict{Any,Summary}())

summaries!(c::AccessCache) = c.summaries
summaries!(::Nothing) = IdDict{Any,Summary}()

"""
    accesscache(device) -> AccessCache

Where `device` keeps its summaries; `nothing` re-runs the analysis every time.
"""
accesscache(::Device) = nothing

"""
    accessof(interp, f, argtypes) -> Vector{Touch}

What `f(argtypes...)` does to each argument, inferred through `interp`'s method
table (`nothing`: Julia's own). Index 1 is `f` itself, which matters for a closure
such as a ray-tracing shader over its material; `argtypes` start at 2. Throws
`UnanalysableAccess` when the signature cannot be inferred.
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
IR for one signature through the walk's interpreter, or `nothing` unless exactly
one method applies and the return type is not `Union{}`. An always-throwing
signature has unreachable IR with no store, and "no store" must never be a guess
because it removes a barrier.
"""
function irfor(w::Walk, @nospecialize(sig))
    irs = w.interp === nothing ? Base.code_ircode_by_type(sig) :
                                 Base.code_ircode_by_type(sig; interp = w.interp)
    length(irs) == 1 || return nothing
    ir, rt = irs[1]
    rt === Union{} && return nothing
    return ir
end

"""
The `CodeInstance` `sig` has in the walk's interpreter's cache, inferred into it
if it has none, or `nothing` when no single method applies. `code_ircode_by_type`
does not cache the signature it is asked about, only what that calls.
"""
function codeinstance(w::Walk, @nospecialize(sig))
    interp = w.interp === nothing ? Base.Compiler.NativeInterpreter() : w.interp
    match = Base._which(sig; method_table = Base.Compiler.method_table(interp),
                        world = Base.get_world_counter(), raise = false)
    match === nothing && return nothing
    ci = Base.Compiler.typeinf_ext(interp, Base.Compiler.specialize_method(match),
                                   Base.Compiler.SOURCE_MODE_NOT_REQUIRED)
    return ci isa Core.CodeInstance ? ci : nothing
end

"""Per-argument `Touch` for a whole signature, or `nothing` if it cannot be seen."""
function signaturetouches(w::Walk, @nospecialize(sig), depth::Int)
    world = Base.get_world_counter()
    known = get(w.summaries, sig, nothing)
    known !== nothing && holds(known, world) && return known.touches
    # A declaration outranks the walk and is checked before the depth limit and the
    # recursion guard: a device intrinsic's body is an `llvmcall` naming a symbol
    # only its emitter understands. Cached like an inferred answer.
    declared = declaredsignature(sig)
    if declared !== nothing
        w.summaries[sig] = Summary(declared, nothing, world)
        return declared
    end
    depth > w.maxdepth && return nothing
    sig in w.active && return nothing           # recursion: the caller refuses
    any(p -> p isa Core.TypeofVararg, sig.parameters) && return nothing
    ir = irfor(w, sig)
    ir === nothing && return nothing
    push!(w.active, sig)
    try
        touches = walkir(w, ir, argmap(ir, sig), depth)
        w.summaries[sig] = Summary(touches, codeinstance(w, sig), world)
        return touches
    finally
        delete!(w.active, sig)
    end
end

"""
Whether a write indexed by `claim` is disjoint from every other invocation's, given
it lands in `target`. An atomic read-modify-write returns a value no other
invocation gets, so a store at that index hits a slot nobody else writes (stages
appending to one ray queue). The counter must belong to the store's own argument;
a slot claimed from one queue says nothing about another.
"""
function disjoint(claim::Taint, target::Taint)
    isempty(claim) && return false
    for c in claim, t in target
        (c >> ELEMBITS) == (t >> ELEMBITS) && return true
    end
    return false
end

"""
The taint walk over one `IRCode`, repeated to a fixed point (at most 16 rounds)
because a back edge lets a later definition reach an earlier use.
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
Per-statement state. `taints` says which arguments' bytes a value reaches, `claims`
whether it is an index some invocation claimed; the two meet only at a store.
"""
struct State
    taints::Vector{Taint}
    claims::Vector{Taint}
    # SSA values that are device addresses, whatever their type (see `pointerlike`).
    addresses::BitSet
end

"""
Taint of an operand: the signature arguments it may come from. An argument taints
only if its type can reach memory ([`carries`](@ref)).
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
Whether a pure computation works on a device address. `carries(UInt64)` is false,
right for a length and wrong for a pointer laundered through an integer, so
provenance is tracked: a result with a `Ptr` operand, or one already in
`st.addresses`, is an address. Lava's cooperative-matrix intrinsics take their
pointer as `UInt64` after `bitcast` and `add_int`; without this, block GEMM kernels
reported their destination as untouched.
"""
pointerlike(ir, st::State, rest) =
    any(a -> rawpointer(operandtype(ir, a)) || isaddress(st, a), rest)

isaddress(st::State, @nospecialize(x)) = x isa Core.SSAValue && x.id in st.addresses

"""
The widened type an operand was inferred at, for `carries` and `pointerlike`. An
argument can be inferred as a `Core.Const`, which `<: Ptr` cannot take.
"""
function operandtype(ir, @nospecialize(x))
    x isa Core.Argument && return widen(ir.argtypes[x.n])
    x isa Core.SSAValue && return widen(ir.stmts[x.id][:type])
    return typeof(x)
end

widen(@nospecialize(T)) = T isa Core.Const ? typeof(T.val) :
                          T isa Type ? T : Any

"""
Record `t` against every signature argument `taint` comes from. A root on the
collapsed vararg tail lands on one argument when the element is known, else on all.
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
An element of the collapsed vararg tail, when a `getfield` names one. Other
arguments are declared whole, so their field index is not tracked.
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
One statement: what it does to memory, and its result's taint. The result keeps
taint when its type can reach memory ([`carries`](@ref)) or its provenance makes it
an address ([`pointerlike`](@ref)); type alone either marks every length read or
loses stores through an address laundered into a `UInt64`.
"""
function stmttaint!(w::Walk, ir, touches, st::State, am::ArgMap, i::Int,
                    @nospecialize(stmt), @nospecialize(T), depth::Int)
    out = Taint()
    tt = widen(T)

    if stmt isa Core.PhiNode || stmt isa Core.PhiCNode
        # Taint and the address flag both cross a phi, or a loop-carried address
        # stops being one.
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
    # A call inferring to `Union{}` throws, so it touches and orders nothing.
    # Otherwise every bounds check (`throw_boundserror(array, …)`, whose IR `irfor`
    # refuses) would mark the array it guards as written. A store before a throw is
    # missed, which is accepted: the dispatch crashes either way.
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
        # Linked device functions arrive as `ccall("extern name", llvmcall, ...)`;
        # their unique symbol is declared through `intrinsic_usage`.
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
                    # An atomic returns a slot no other invocation gets; Metal's
                    # atomics arrive here.
                    usage === ATOMIC && union!(st.claims[i], operandtaint(ir, st, a))
                end
                return Taint()
            end
        end
        # An ordinary `ccall` is host C and cannot compile for a device.
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
    # A call that cannot return (`throw_boundserror` and every other thrower) is an
    # abort path, reached from most kernels' bounds checks. A store before the throw
    # cannot be observed by a later pass, so it needs no barrier.
    tt === Union{} && return Taint()
    inner = sig === nothing ? nothing : signaturetouches(w, sig, depth + 1)
    out = Taint()
    for (j, a) in enumerate(callargs)
        at = operandtaint(ir, st, a)
        isempty(at) && continue
        if inner === nothing
            # `signaturetouches` found no answer: no single method, `Union{}`,
            # recursion, or past the depth limit.
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
    # The callee's accesses are recorded; what is left is whether its result is an
    # address, which a `UInt64` handed back needs provenance for.
    return keptaddress!(st, i, ir, callargs, tt) ? out : Taint()
end

"""
Whether a pure computation's result keeps its operands' taint: by type
([`carries`](@ref)) or by provenance ([`pointerlike`](@ref)), the latter recorded in
`st.addresses` for later statements.
"""
function keptaddress!(st::State, i::Int, ir, rest, @nospecialize(tt))
    if pointerlike(ir, st, rest)
        push!(st.addresses, i)
        return true
    end
    return carries(tt)
end

"""
A builtin, an intrinsic, or a call the optimiser left dynamic. `READ_OPS`,
`WRITE_OPS` and `RMW_OPS` touch memory, `PURE_OPS` move values, `llvmcall` goes to
`llvmcalltaint!`, and an unresolved call on a tainted value is refused.
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
        # A claimed index stays claimed through the arithmetic that makes it an offset.
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

    # Left: other intrinsics, which compute on values, and unresolved calls.
    if iscoreintrinsic(f)
        # Values an address is computed from are addresses, as at `PURE_OPS`.
        return keptaddress!(st, i, ir, rest, tt) ? out : Taint()
    end
    # An abort path, as in `invoketaint!`; `Core.throw_methoderror` arrives here
    # unresolved.
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
A store `pointerset(ptr, value, index, align)`: ordered unless the slot was claimed.
Only the pointer and index decide; the stored value says nothing about where it lands.
"""
function storetouch(ir, st::State, rest, target::Taint)
    claim = operandclaim(st, rest[1])
    length(rest) >= 3 && union!(claim, operandclaim(st, rest[3]))
    return disjoint(claim, target) ? ATOMIC : WRITE
end

"""
    intrinsic_usage(name::Symbol) -> Touch or nothing

What the external symbol an `llvmcall` calls does to the pointer it is handed. The
wrapper is inlined and the module has no memory instruction, so the name is all
that is left; `nothing` means undeclared. Symbols are globally unique, so each
compiled backend is asked through `intrinsic_usage(api, name)`. A backend must not
define `intrinsic_usage(::Symbol)`: with two backends compiled (a Mac) that is one
method defined twice, which precompilation refuses.
"""
function intrinsic_usage(name::Symbol)
    u = intrinsic_usage(VulkanAPI(), name)
    return u === nothing ? intrinsic_usage(MetalAPI(), name) : u
end
intrinsic_usage(@nospecialize(name)) = nothing
intrinsic_usage(::Backend, name::Symbol) = nothing

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

const RMW_INSTR   = r"(?m)^\s*(%[^=\n]*=\s*)?(atomicrmw|cmpxchg)\s"
const STORE_INSTR = r"(?m)^\s*store\s"
const LOAD_INSTR  = r"(?m)^\s*%[^=\n]*=\s*load\s"
const ADDRESS_INSTR = r"(?m)^\s*%[^=\n]*=\s*(addrspacecast|bitcast|getelementptr)\b"

"""
    llvmcallusage(src) -> Touch or nothing

What an `llvmcall` module does to its tainted operand. Instructions decide first,
matched at line start because `declare ... @_lava_coopmat_load_...` contains `load`
and is none: `atomicrmw`/`cmpxchg` is `ATOMIC`, `store` `WRITE`, `load` `READ`. A
module with none is a `declare` plus a `call`, answered by [`intrinsic_usage`](@ref).
"""
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
An `llvmcall` with a tainted operand, read from its IR: address arithmetic keeps
provenance, a memory instruction or declared intrinsic name gives the access, and
anything else is refused. One without tainted operands touches nothing.
"""
function llvmcalltaint!(w::Walk, ir, touches, st::State, am::ArgMap, i::Int,
                        rest, args::Vector{Any})
    src = llvmsource(rest[1], ir)
    tainted = [a for a in rest if !isempty(operandtaint(ir, st, a))]
    isempty(tainted) && return Taint()
    # LLVM.jl spells `LLVMPtr` casts and arithmetic as tiny `llvmcall` modules
    # (`addrspacecast`, `bitcast`, `getelementptr`) that touch no memory; keep the
    # provenance so the later load or store lands on the original argument.
    if src !== nothing && occursin(ADDRESS_INSTR, src) &&
       !occursin(RMW_INSTR, src) && !occursin(STORE_INSTR, src) &&
       !occursin(LOAD_INSTR, src)
        push!(st.addresses, i)
        out = Taint()
        foreach(a -> union!(out, operandtaint(ir, st, a)), tainted)
        # An address computed from a claimed index is a claimed slot.
        foreach(a -> union!(st.claims[i], operandclaim(st, a)), rest)
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
        # `refuse!` returns silently when no argument is reached; nothing to record.
        return Taint()
    end
    for a in tainted
        target = operandtaint(ir, st, a)
        record!(touches, am, target,
                t === WRITE ? llvmstoretouch(st, src, rest, target) : t)
        t === ATOMIC && union!(st.claims[i], target)
    end
    return Taint()
end

const STORED_VALUE = r"(?m)^\s*store\s+(?:atomic\s+)?(?:volatile\s+)?(?:<[^>]*>|\S+)\s+([^,\s]+)\s*,"

"""
A store spelled as an `llvmcall` (LLVM.jl's `LLVMPtr` store, how a Metal array
writes): ordered unless the slot was claimed, as in [`storetouch`](@ref). The claim
comes from every operand except the stored value; a module whose store cannot be
matched to one parameter is ordered.
"""
function llvmstoretouch(st::State, src::String, rest, target::Taint)
    stores = collect(eachmatch(STORED_VALUE, src))
    length(stores) == 1 || return WRITE
    value = only(stores).captures[1]
    # `rest` is the `llvmcall`'s: the module, the return type, the argument types,
    # then the operands, whose parameters are `%0`, `%1`, ...
    ops = @view rest[4:end]
    stored = 0
    if startswith(value, '%')
        p = tryparse(Int, value[2:end])
        (p === nothing || p >= length(ops)) && return WRITE
        stored = p + 1
    end
    claim = Taint()
    for k in eachindex(ops)
        k == stored || union!(claim, operandclaim(st, ops[k]))
    end
    return disjoint(claim, target) ? ATOMIC : WRITE
end

"""
The LLVM source of an `llvmcall`: inline, built into a tuple by the IR, or named by
a `GlobalRef` to a const (Lava's atomics, e.g. `Lava.ir_fadd_f32`).
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

"""
    kerneltouches(device, kernel, args, ndrange, group) -> Vector{Touch}

What `kernel` does to each of `args` when dispatched on `device`. The backend
supplies the signature (interpreter, device-side argument types, leading kernel
arguments); the walk, `accessof`, is shared.
"""
function kerneltouches(dev, kernel, args::Tuple, ndrange, group)
    argT = map(a -> devicetype(dev, a), args)
    if buildskernel(kernel, dev)
        interp, body, tt = kakernelaccesssignature(dev, kernel, argT, ndrange, group)
        return accessof(interp, body, tt; cache = accesscache(dev))[3:end]
    end
    # Refuse a kernel this device cannot compile at the same boundary as `bake`,
    # before inference reports some device intrinsic instead.
    kisupported(dev, kernel) || kikernel(kernel, dev, argmap(a -> resolve(dev, a), args))
    interp = kernelinterpreter(dev, kernel, Tuple{argT...})
    return accessof(interp, kernel, argT; cache = accesscache(dev))[2:end]
end

"""The interpreter a backend compiles a macro-free kernel through."""
function kernelinterpreter end

"""
    kakernelaccesssignature(device, kernel, argtypes, ndrange, group)

Interpreter, body and signature of a retained KernelAbstractions kernel, built the
way a KA launch builds its context. Backends override it only when their launch
signature differs (Lava retains an `IterPlan`).
"""
function kakernelaccesssignature(dev, kernel, argT::Tuple, ndrange, group)
    obj = kernelfor(kernel, group, dev)
    nd = dispatchrange(ndrange)
    cg = callgroup(obj, group)
    ndr, _, iterspace, _ = KA.launch_config(obj, nd, cg)
    ctx = KA.mkcontext(obj, ndr, iterspace)
    tt = (argtype(dev, ctx), argT...)
    return kernelinterpreter(dev, obj.f, Tuple{tt...}), obj.f, tt
end

"""
    devicetype(device, x) -> Type

The type a kernel's argument `x` arrives as, derived through `resolve` and
`argtype`, the rules a dispatch is packed with. Transients and views have no
storage while the graph is built and answer through [`devicebuffertype`](@ref).
"""
devicetype(dev::Device, x) = argtype(dev, resolve(dev, x))
# A transient keeps its declared rank: `Place` gives it an array of its dims, and
# rank 1 would infer against the wrong `getindex` and decline rank-2 GEMM paths.
devicetype(dev::Device, ::TransientBuffer{T,N}) where {T,N} =
    devicebuffertype(dev, T, N)
# A view's storage derives from its parent's, which `Place` has not handed out yet;
# `GPUArrays.derive` answers the view's own rank. Uniform over views, since a view
# of a placed `Buffer` derives to the same type.
devicetype(dev::Device, ::ResourceView{T,N}) where {T,N} = devicebuffertype(dev, T, N)

# A range answers its parent's type: the kernel receives the whole buffer (see
# `BufferRange`). Through `resolve` it would answer the handle type, which has no
# device `setindex!`, so no store would be seen.
devicetype(dev::Device, r::BufferRange) = devicetype(dev, r.parent)

# Containers recurse through `devicetype`, not `resolve`: an element may be a
# transient or a view without storage (e.g. `ew!`'s operand tuple).
devicetype(dev::Device, x::Tuple) = Tuple{map(a -> devicetype(dev, a), x)...}
devicetype(dev::Device, x::NamedTuple{K}) where {K} =
    NamedTuple{K, Tuple{map(a -> devicetype(dev, a), values(x))...}}

"""
    argtype(device, y) -> Type

The type a launch turns an already-`resolve`d value into: by default its own, or the
`Adapt`ed type on a backend that adapts arguments. Use `Core.Typeof`, not `typeof`,
here and in every backend method: a type argument is specialised at
`Type{Float32}`, and `DataType` is not a dispatch-tuple element.
"""
argtype(::Device, @nospecialize(y)) = Core.Typeof(y)

"""
    devicebuffertype(device, T, N = 1) -> Type

The device-side type of an `N`-dimensional buffer of `T`, which a kernel sees for a
transient or view of rank `N`. Must equal the type `Place` materialises.
"""
function devicebuffertype end
# Rank 1 when omitted; a backend answers the three-argument form.
devicebuffertype(dev, ::Type{T}) where {T} = devicebuffertype(dev, T, 1)
