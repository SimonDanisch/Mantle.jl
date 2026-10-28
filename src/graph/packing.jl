# Shared compiled-argument classification, slot order and pointer-patch registration.

"""
    ArgPassing

How a compiled argument is passed: `NotPassed`, `InlineBytes` or `DeviceAllocation`.
"""
abstract type ArgPassing end

"""
    NotPassed()

No compiled parameter and no slot: zero-size isbits values and type-valued
constants. Skipping the bytes but not the slot would misalign every later argument.
"""
struct NotPassed <: ArgPassing end

"""
    InlineBytes()

Copy the argument's bytes (including isbits device-array descriptors) into its slot.
"""
struct InlineBytes <: ArgPassing end

"""
    DeviceAllocation()

Bind a native allocation instead of copying bytes (Vulkan writes its address,
Metal binds the buffer). Backends specialise `passedas` for their handles.
"""
struct DeviceAllocation <: ArgPassing end

"""
    passedas(T) -> ArgPassing

How an argument of type `T` is passed: `passedas(Float32)` is a `Float32` scalar,
`passedas(typeof(Float32))` the value `Float32`.
"""
passedas(::Type{T}) where {T} = notpassed(T) ? NotPassed() : InlineBytes()

"""
    notpassed(T) -> Bool

Whether GPUCompiler emits no parameter for `T`: `Type` values and zero-size
isbits types, decided without LLVM so generated functions can call it.
Non-isbits types are not ghosts; kernel validation rejects them.
"""
notpassed(::Type{T}) where {T} = T <: Type || (isbitstype(T) && sizeof(T) == 0)

"""
    passedslots(types) -> Vector{Int}

Indices of the arguments that occupy slots, in layout order, for generated packers.
"""
function passedslots(types)
    slots = Int[]
    for (i, Ti) in enumerate(types)
        notpassed(Ti) || push!(slots, i)
    end
    return slots
end

"""
    storebits!(p, x) -> nothing

Store isbits `x` at `p`, field by field where it has padding, so the padding keeps
the bytes `p` held. `unsafe_store!` of a padded struct leaves them undefined, in
practice whatever the stack held: packing the same values twice wrote different
bytes, and a re-recorded plan's arguments changed though nothing in them had.
"""
@generated function storebits!(p::Ptr{UInt8}, x::T) where {T}
    sizeof(T) == 0 && return :(nothing)
    (isprimitivetype(T) || !Base.datatype_haspadding(T)) &&
        return :(unsafe_store!(Ptr{$T}(p), x); nothing)
    stores = [:(storebits!(p + $(Int(fieldoffset(T, i))), getfield(x, $i)))
              for i in 1:fieldcount(T)]
    return quote
        $(stores...)
        nothing
    end
end

"""
    eachpassedarg(f, args) -> Int

Call `f(slot, arg, passing)` for each passed argument and return the slot count.
`NotPassed` arguments take no slot; byte offsets and alignment are the backend's.
"""
function eachpassedarg(f, args::Tuple)
    slot = 1
    for a in args
        p = passedas(typeof(a))
        p isa NotPassed && continue
        f(slot, a, p)
        slot += 1
    end
    return slot - 1
end

# Record-time pointer locations; `notify_move!` applies the patches.

"""
    nestinglevels(T) -> Int

Nesting depth an aggregate adds when searching for patchable device pointers:
zero for `Tuple`/`NamedTuple`, one for a struct.
"""
nestinglevels(::Type{<:Tuple}) = 0
nestinglevels(::Type{<:NamedTuple}) = 0
nestinglevels(::Type) = 1

"""Append pointer-field offsets of `S` at most one struct level deep; deeper pointers
are not patched, so their recording is invalidated on a move."""
function devptroffsets!(offs::Vector{Int}, S::Type, base::Int, depth::Int)
    depth > 1 && return offs
    for i in 1:fieldcount(S)
        F = fieldtype(S, i)
        off = base + Int(fieldoffset(S, i))
        if F <: Core.LLVMPtr || F <: Ptr
            push!(offs, off)
        elseif isstructtype(F) && isbitstype(F)
            devptroffsets!(offs, F, off, depth + nestinglevels(F))
        end
    end
    return offs
end

"""
    devicepointeroffsets(T) -> NTuple{N,Int}

Byte offsets of the device pointers (`Core.LLVMPtr` or `Ptr`) inside isbits `T`,
resolved at specialisation; empty for non-isbits types.
"""
@generated function devicepointeroffsets(::Type{T}) where {T}
    offs = Int[]
    isbitstype(T) && isstructtype(T) &&
        devptroffsets!(offs, T, 0, nestinglevels(T))
    return :($(Tuple(offs)))
end

"""
    PatchSite(plan, target, base)

Where a recording packs bytes: into `target` (memory the backend's `emitinline!`
writes), whose first byte is at host address `base`. A backend's packer hands each
value it writes to [`notepacked!`](@ref) with the site; a pack that is not recorded
passes `nothing`, since nothing replays it.
"""
struct PatchSite{T}
    plan::Plan
    target::T
    base::Ptr{UInt8}
end

"""
    notepacked!(site, T, at)

Register in the plan's `patchtab` each nonzero device pointer of a `T` packed at
host address `at` inside the site, for `notify_move!`. A no-op for types without
pointers and for `site === nothing`.
"""
@inline function notepacked!(s::PatchSite, ::Type{T}, at::Ptr{UInt8}) where {T}
    fields = devicepointeroffsets(T)
    isempty(fields) && return nothing
    for f in fields
        notepointer!(s, at + f)
    end
    return nothing
end
notepacked!(::Nothing, ::Type, ::Ptr{UInt8}) = nothing

"""
    notepointer!(site, at)

Register the device address stored at host address `at`, when nonzero: a packed
slot that holds an allocation's address rather than a value's bytes.
"""
function notepointer!(s::PatchSite, at::Ptr{UInt8})
    addr = unsafe_load(Ptr{UInt64}(at))
    addr == 0 && return nothing
    push!(get!(Vector{Tuple{Any,Int}}, s.plan.patchtab, addr), (s.target, Int(at - s.base)))
    return nothing
end
notepointer!(::Nothing, ::Ptr{UInt8}) = nothing
