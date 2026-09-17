# ── How a kernel argument reaches the kernel ─────────────────────────────────
#
# Every backend that records commands has to answer the same three questions
# about each argument of a dispatch, in the same order, twice: once when the
# dispatch compiles (which slot, how many bytes) and once when it is written
# (what to put there). Before this file each backend answered them itself:
#
#   * Vulkan: `slotpositions` skipped `sizeof(Ti) == 0 || Ti <: Type`.
#   * Metal:  `recorded_args` skipped `isghosttype(T) || isconstType(T)`, and
#             `pack_recorded!` repeated that same test to re-derive the slot
#             index it had assigned in the first walk.
#
# Three copies of one rule, and the two backends' versions are not equivalent.
# `Core.Compiler.isconstType` only answers `true` for a signature type
# `Type{Float32}`; both walks classify from `typeof(a)`, which for the value
# `Float32` is `DataType`, so Metal's test never fires for a type-valued
# argument and its bytes branch then copies `sizeof(DataType)` bytes out of a
# `Ref` holding a pointer — into a slot the compiled kernel does not have. No
# build compiles both backends, so no test could see the two rules disagree.
#
# The rule is one function here, the walk that applies it is one function here,
# and what is left for a backend is what genuinely differs: where the bytes go
# and what binding an allocation means on its own command buffer.

"""
    ArgPassing

How one argument of a kernel launch reaches the kernel: [`NotPassed`](@ref),
[`InlineBytes`](@ref) or [`DeviceAllocation`](@ref). Answered by
[`passedas`](@ref) from the argument's type alone, so a compile resolves it
once and a run never asks.
"""
abstract type ArgPassing end

"""
    NotPassed()

The compiled kernel has no parameter for this argument, so there is no slot and
nothing to write.

Two kinds of argument land here. A zero-size value — a singleton like
`typeof(+)`, a `Val{N}`, an empty struct — has no bytes to pass. A type-valued
argument — `T = Float32` captured by a kernel — is a compile-time constant that
the specialised kernel bakes in; `typeof(Float32) === DataType` has nonzero
`sizeof`, but GPUCompiler emits no parameter for it.

Both must be skipped and not merely written as zero bytes: writing anything for
an argument with no slot either shifts every later argument down by one slot or
runs off the end of the argument block.
"""
struct NotPassed <: ArgPassing end

"""
    InlineBytes()

The argument's own bytes are copied into the plan's argument memory, and the
slot addresses those bytes. Every isbits value that is not a device allocation:
primitives, and aggregates like a device array descriptor or a `NamedTuple` of
sizes.
"""
struct InlineBytes <: ArgPassing end

"""
    DeviceAllocation()

The argument IS a device allocation, and the slot binds that allocation rather
than a copy of its bytes.

The one class where the backends genuinely differ, which is why it is named
rather than folded into [`InlineBytes`](@ref). Vulkan holds the allocation for
the submission's lifetime and writes its 64-bit device address into the slot;
Metal makes it resident and binds the `MTLBuffer` itself, because an indirect
command has no counterpart to `setBytes:`.

A backend declares its own allocation handles — see `passedas` methods under
`src/vulkan/` and `src/metal/`. Core's default is [`InlineBytes`](@ref), since
core does not know a backend's handle types.
"""
struct DeviceAllocation <: ArgPassing end

"""
    passedas(T) -> ArgPassing

How an argument of type `T` reaches the kernel.

A property of the type, so it costs nothing per run: in Vulkan's generated
packer the whole walk is constant-folded, and Metal resolves it once per
dispatch at compile.

The default answers [`NotPassed`](@ref) for what GPUCompiler emits no parameter
for and [`InlineBytes`](@ref) for everything else. A backend adds a method for
its own allocation handles answering [`DeviceAllocation`](@ref), and nothing
else: a backend never restates the `NotPassed` rule, which is the one that used
to differ between them.

Note the argument is a TYPE, as for `sizeof`. To ask about a type-valued
argument, pass `typeof` of it: `passedas(Float32)` is how a `Float32` value is
passed (inline bytes), `passedas(typeof(Float32))` is how the value `Float32`
is passed (not passed at all).
"""
passedas(::Type{T}) where {T} = notpassed(T) ? NotPassed() : InlineBytes()

"""
    notpassed(T) -> Bool

Whether the compiled kernel has no parameter for an argument of type `T`.

GPUCompiler's own rule is `isghosttype(T) || Core.Compiler.isconstType(T)`,
applied to the SIGNATURE type. This is that rule restated so it holds for the
type of a value as well, which is what both packers actually have:

  * `T <: Type` is `isconstType` made to work on `typeof(a)`. The signature
    type of the argument `Float32` is `Type{Float32}` and `isconstType` answers
    `true`, but `typeof(Float32)` is `DataType` and it answers `false`. Both are
    `<: Type`. Nothing is lost by being this broad: a `DataType` argument that
    GPUCompiler did NOT drop is a boxed, non-isbits kernel argument, which its
    own validation rejects, so there is no `T <: Type` that both reaches a
    compiled kernel and needs a slot.

  * `isbitstype(T) && sizeof(T) == 0` is `isghosttype`, which converts `T` to an
    LLVM type and asks whether it is empty. Restated with `sizeof` because this
    runs inside a generated function, where creating an LLVM context to answer a
    question about a Julia type is not something to do; the `isbitstype` guard is
    what makes it total, since `sizeof` throws on types with no fixed size. A
    non-isbits type is boxed, so its LLVM type is a pointer and never empty.

`test/test_argument_packing.jl` checks the two agree on a table of types,
against LLVM.jl's `isghosttype` and `Core.Compiler.isconstType` directly.
"""
notpassed(::Type{T}) where {T} = T <: Type || (isbitstype(T) && sizeof(T) == 0)

"""
    passedslots(types) -> Vector{Int}

Positions in an argument tuple that take a slot in the compiled kernel, in
order, so `passedslots(types)[slot]` is the argument index that slot binds.

The type-level half of [`eachpassedarg`](@ref), for a packer that specialises on
the argument tuple type and wants the walk constant-folded away. Vulkan's
`pack_args_direct!` is generated from this: the returned position is the layout
index its `offsets` and `byval_sizes` vectors are indexed by, which come from the
compiled SPIR-V entry wrapper and are therefore already in slot order.
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

Call `f(slot, arg, passing)` for each of `args` that the compiled kernel takes,
skipping the [`NotPassed`](@ref) ones, and return how many slots that was.

`slot` counts only passed arguments, which is what makes this the whole of the
agreement between a backend's two walks: the slot a compile assigns to an
argument and the slot a record writes it into come from the same count, so they
cannot drift. Two copies of the skip rule with an index advanced by hand
between them agree only by coincidence.

Where the bytes of an [`InlineBytes`](@ref) argument go is deliberately not
decided here. Vulkan's offsets are the SPIR-V entry wrapper's and are handed to
it by the pipeline; Metal chooses its own, 256-byte aligned. Both accumulate
their own cursor in the closure.
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

# ── Where the device pointers inside a packed argument landed ────────────────
#
# Moved here from `memory/pool.jl`, which is the allocator: these three are what
# a PACKER calls, once per argument at record time, and the table they build is
# what makes a `resize!` under a recorded plan eight bytes instead of a
# re-record. `notify_move!` is the allocator's half and stayed there.

"""
    nestinglevels(T) -> Int

How many levels of aggregate an argument of type `T` is, for the depth budget in
[`devptroffsets!`](@ref).

`Tuple` and `NamedTuple` are **zero**: they are the language's own spelling of
"a group of arguments", not structures with an identity of their own.
`ew!(out, dims, operands, strides, f)` takes its operands as one tuple so that
one kernel covers every arity, and those operands are top-level arguments that
happen to travel together. Counting the tuple as a level puts every device
pointer inside it one past the budget, so
`devicepointeroffsets(Tuple{LavaDeviceArray,LavaDeviceArray})` answers `()` and
a resize of an operand leaves the recorded plan reading freed storage, with
nothing to see at the call site.

A type you declared is **one**, which is what the budget was written against: a
device array descriptor carries its address as a direct field, and a scene
structure holding one several fields deep is invalidated on a move rather than
patched. That line is what separates the two — a grouping is transparent, a
named datum is not — so `Tuple{Scene}` behaves exactly as `Scene` does.
"""
nestinglevels(::Type{<:Tuple}) = 0
nestinglevels(::Type{<:NamedTuple}) = 0
nestinglevels(::Type) = 1

"""
Recurse a packed type for device-pointer fields. Depth-limited: a scene
structure several aggregates deep is invalidated on a move, not patched — which
is the rule Vulkan's packer already stated for the same reason. Tuples are free,
see [`nestinglevels`](@ref).
"""
function devptroffsets!(offs::Vector{Int}, S::Type, base::Int, depth::Int)
    # Going deeper than one aggregate would silently start patching scene
    # structures — which Vulkan's packer deliberately does not: a move under one
    # of those invalidates the plan and it is re-recorded, rather than having a
    # half-updated copy of itself written into it.
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

Byte offsets of every device pointer inside a packed argument of type `T`.

A property of the TYPE, so it is resolved once when the packer specialises and
costs nothing per dispatch — the tuple is a literal by the time it runs, and a
`T` holding none constant-folds the loop over it away entirely.

This is what lets the patch table be core's. Asking a backend's packer to
report each pointer it writes lets a backend that reports none have no table and
no patching, silently. Nothing is asked here: the pointers are found from the
type, so a backend can neither answer wrongly nor fail to answer.

`Core.LLVMPtr` is Metal's device pointer and `Ptr` is Vulkan's; both are the
whole 64-bit address, which is what `notify_move!` re-keys on.
"""
@generated function devicepointeroffsets(::Type{T}) where {T}
    offs = Int[]
    isbitstype(T) && isstructtype(T) &&
        devptroffsets!(offs, T, 0, nestinglevels(T))
    return :($(Tuple(offs)))
end

"""
    notepacked!(plan, T, at, target, off)

Record where the device pointers of a just-packed argument of type `T` landed:
its bytes are at host address `at`, which is byte `off` of `target`.

Reads the addresses back out of the bytes the packer has already written rather
than re-boxing the value — so this allocates nothing beyond the table entries
themselves, and for a `T` with no device pointers the whole call compiles to
nothing.

Consumed only by [`notify_move!`](@ref), and only if something moves. A plan
whose buffers never resize pays this once per argument at RECORD time and
nothing per run.
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
