# Shared compiled-argument classification, slot order and pointer-patch registration.

"""
    ArgPassing

How a compiled argument is passed: NotPassed, InlineBytes or DeviceAllocation.
"""
abstract type ArgPassing end

"""
    NotPassed()

No compiled parameter or slot. Covers zero-size isbits values and type-valued constants;
skipping bytes without skipping the slot would misalign every following argument.
"""
struct NotPassed <: ArgPassing end

"""
    InlineBytes()

Copy an ordinary argument's bytes, including isbits device-array descriptors, into its slot.
"""
struct InlineBytes <: ArgPassing end

"""
    DeviceAllocation()

Bind a native allocation rather than copying its bytes. Vulkan writes its device address;
Metal binds the resident buffer. Backends specialize passedas for allocation handles.
"""
struct DeviceAllocation <: ArgPassing end

"""
    passedas(T) -> ArgPassing

Choose passing from the argument's type. Ask passedas(typeof(Float32)) about the value
Float32; passedas(Float32) describes a Float32 scalar. Only native allocation classification
is backend-specific; the NotPassed rule is shared.
"""
passedas(::Type{T}) where {T} = notpassed(T) ? NotPassed() : InlineBytes()

"""
    notpassed(T) -> Bool

Whether GPUCompiler emits no parameter. T <: Type covers both Type{T} signature types
and typeof(T)==DataType. Zero-size isbits types cover ghost values without requiring LLVM
inside generated functions. Boxed non-isbits types are not ghosts; kernel validation rejects them.
See test_argument_packing.jl for comparison with the compiler rule.
"""
notpassed(::Type{T}) where {T} = T <: Type || (isbitstype(T) && sizeof(T) == 0)

"""
    passedslots(types) -> Vector{Int}

Return argument indices that occupy slots, in compiled-layout order. Used by generated
packers so the walk is resolved when specializing.
"""
function passedslots(types)
    slots = Int[]
    for (i, Ti) in enumerate(types)
        notpassed(Ti) || push!(slots, i)
    end
    return slots
end

"""
    eachpassedarg(f, args) -> Int

Call f(slot,arg,passing) for each passed argument and return the slot count. Slots skip
NotPassed arguments. Backends choose their own byte offsets/alignment.
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

# Record-time pointer locations; notify_move! owns allocator-side patch application.

"""
    nestinglevels(T) -> Int

Depth consumed by an aggregate when finding patchable device pointers: Tuple/NamedTuple
are transparent argument groups (zero); declared structs consume one level. Deeper scene
structures invalidate their recording on movement rather than receiving partial patches.
"""
nestinglevels(::Type{<:Tuple}) = 0
nestinglevels(::Type{<:NamedTuple}) = 0
nestinglevels(::Type) = 1

"""Find pointer fields within the aggregate depth budget. Tuples/NamedTuples are transparent;
structures deeper than one level are invalidated on movement, not partially patched."""
function devptroffsets!(offs::Vector{Int}, S::Type, base::Int, depth::Int)
    # Deeper structs require re-recording; partial pointer patches would leave stale fields.
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

Return device-pointer byte offsets resolved from T at specialization. Core.LLVMPtr and
Ptr both carry complete addresses. Non-isbits types have no inline pointer fields to patch.
"""
@generated function devicepointeroffsets(::Type{T}) where {T}
    offs = Int[]
    isbitstype(T) && isstructtype(T) &&
        devptroffsets!(offs, T, 0, nestinglevels(T))
    return :($(Tuple(offs)))
end

"""
    notepacked!(plan, T, at, target, off)

Register device-pointer addresses read from packed bytes at host address at, byte off of
target. notify_move! consumes the table. Types without pointers compile to a no-op.
"""
@inline function notepacked!(pl, ::Type{T}, at::Ptr{UInt8}, target, off::Int) where {T}
    fields = devicepointeroffsets(T)
    isempty(fields) && return nothing
    for f in fields
        addr = unsafe_load(Ptr{UInt64}(at + f))
        addr == 0 && continue
        push!(get!(Vector{Tuple{Any,Int}}, pl.patchtab, addr), (target, off + f))
    end
    return nothing
end
