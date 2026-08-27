# Kernel-arg pinning walker for Lava.jl
#
# Lava batches dispatches into command buffers that are submitted later, so
# every `LavaArray` passed to a kernel must be pinned into `batch.pinned`
# BEFORE submit. Without the pin, the LavaArray can be GC'd between record
# and submit, its finalizer runs, the backing `VkManagedBuffer` is freed,
# and the GPU reads from released memory.
#
# The old design had `adapt_storage(::LavaAdaptor, ::LavaArray)` do both the
# strip (pure) and the pin (side-effect) — convenient but entangled two
# concerns and prevented ever caching the adapt result.  Now adapt is pure,
# and this file's @generated `pin_leaves!` is the side-effect step.
#
# `pin_leaves!(batch, x)` recurses into `x` via the struct-field layout
# known at compile time and calls `pin!(batch, buf)` on every `LavaArray`
# leaf it finds.  The walk is specialized per concrete type, so it compiles
# to straight-line code — zero runtime dispatch, zero allocation, no
# anonymous closures.

"""
    pin_leaves!(batch::CommandBatch, x) -> nothing

Recursively pin every `LavaArray` leaf inside `x` into `batch.pinned`.
Specializes per type via `@generated` — no runtime walker overhead, no
allocation, safe to call from the hot dispatch path.

Must be called before the batch is submitted if any `LavaArray` inside `x`
was stripped to `LavaDeviceArray` (via `Adapt.adapt`) for the kernel.
"""
function pin_leaves! end

# Leaf: the one case that has side effects.  `pin!` is idempotent (IdSet
# push) so repeated calls on the same buffer are O(1).
@inline pin_leaves!(batch::CommandBatch, a::LavaArray) =
    (pin!(batch, a); nothing)

# Tuple / NamedTuple — fixed-arity, unrolled via @generated.
@generated function pin_leaves!(batch::CommandBatch, x::Tuple)
    exprs = Expr[:(pin_leaves!(batch, x[$i])) for i in 1:fieldcount(x)]
    Expr(:block, exprs..., :(nothing))
end

@generated function pin_leaves!(batch::CommandBatch, x::NamedTuple)
    exprs = Expr[:(pin_leaves!(batch, x[$i])) for i in 1:fieldcount(x)]
    Expr(:block, exprs..., :(nothing))
end

# ── Address ranges, for barrier elision ──
#
# Same walk, different leaf action: collect the device address range each
# `LavaArray` occupies. Two dispatches whose ranges are pairwise disjoint cannot
# alias, so no memory barrier is needed between them — regardless of which side
# reads and which writes, which is what makes this sound without any read/write
# annotation on kernel arguments.
#
# Walking the *pre-adapt* arguments is what makes it complete: that is the same
# tree `pin_leaves!` walks, so every buffer the kernel can reach through a BDA is
# accounted for, including `LavaArray`s nested inside wrapper structs.

"""
    range_leaves!(dst::Vector{UInt64}, x) -> nothing

Append `lo, hi` device-address pairs for every `LavaArray` leaf inside `x`.
"""
function range_leaves! end

@inline function range_leaves!(dst::Vector{UInt64}, a::LavaArray)
    lo = bda_address(a)
    push!(dst, lo)
    push!(dst, lo + UInt64(sizeof(eltype(a)) * length(a)))
    nothing
end

@generated function range_leaves!(dst::Vector{UInt64}, x::Tuple)
    exprs = Expr[:(range_leaves!(dst, x[$i])) for i in 1:fieldcount(x)]
    Expr(:block, exprs..., :(nothing))
end

@generated function range_leaves!(dst::Vector{UInt64}, x::NamedTuple)
    exprs = Expr[:(range_leaves!(dst, x[$i])) for i in 1:fieldcount(x)]
    Expr(:block, exprs..., :(nothing))
end

@generated function range_leaves!(dst::Vector{UInt64}, x::T) where T
    isbitstype(T) && return :(nothing)
    T <: Ptr               && return :(nothing)
    T <: Type              && return :(nothing)
    T === Nothing          && return :(nothing)
    T <: Symbol            && return :(nothing)
    T <: AbstractChar      && return :(nothing)
    T <: AbstractString    && return :(nothing)
    T <: Module            && return :(nothing)
    n = fieldcount(T)
    n == 0 && return :(nothing)
    exprs = Expr[:(range_leaves!(dst, getfield(x, $i))) for i in 1:n]
    Expr(:block, exprs..., :(nothing))
end

# Generic struct walker. One @generated method covers every `x::T` that isn't
# already caught by a more-specific signature above (LavaArray / Tuple /
# NamedTuple).  Handles the short-circuit cases (isbits, Ptr, Type, …) at
# codegen time so the fast path is a literal `:(nothing)` with no runtime
# branching.  This is also the sole fallback — no `::Any` method exists, so
# there is nothing for the @generated to collide with during precompile.
@generated function pin_leaves!(batch::CommandBatch, x::T) where T
    # isbits: LavaArray is mutable, so an isbits type cannot transitively
    # contain one.  Emit a no-op.
    isbitstype(T) && return :(nothing)
    # Hard-coded short-circuits for reference types that clearly can't hold
    # a LavaArray.  Extend as new leaf types show up in kernel arg trees.
    T <: Ptr               && return :(nothing)
    T <: Type              && return :(nothing)
    T === Nothing          && return :(nothing)
    T <: Symbol            && return :(nothing)
    T <: AbstractChar      && return :(nothing)
    T <: AbstractString    && return :(nothing)
    T <: Module            && return :(nothing)
    n = fieldcount(T)
    n == 0 && return :(nothing)
    # Walk fields; each sub-call re-dispatches to the most-specific
    # `pin_leaves!` method for that field's type.
    exprs = Expr[:(pin_leaves!(batch, getfield(x, $i))) for i in 1:n]
    Expr(:block, exprs..., :(nothing))
end
