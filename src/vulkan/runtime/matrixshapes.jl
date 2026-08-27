# The driver's cooperative-matrix table, and the two mappings that read it.
#
# Split out of `device/acceleratedmatrix.jl`, which holds `CoopMatrix` — the TYPE
# a kernel is compiled against. These are device QUERIES: they read
# `ctx.coopmat_shapes`, which is a Vulkan property array fetched at device
# creation, so they belong with the runtime and not with the type.
#
# `caps(ctx)` calls `matrixshapes` to fill `DeviceCaps.shapes`, and from there
# `KernelInterface`'s `supports` / `bestshape` answer for the device without
# anything else asking the driver again.

"""
    coopmat_shape(ctx, T, M, N, K) -> Bool

Whether the running device implements this tile. The hardware supports a fixed
set of `(M, N, K, dtype)` combinations, so a kernel picks one of those or uses
`MMatrix` instead.
"""
# VkComponentTypeKHR. Only the types a cooperative-matrix operand can currently
# have in Lava; an unmapped type reports "no such shape" rather than matching one
# by accident.
#
# The table is written once, in one direction, and the inverse is derived from it
# — two hand-written tables are two chances for one entry to disagree, and the
# symptom would be a shape silently reported as a type it is not.
const VK_COMPONENT_TYPES = (Float16, Float32, Float64, Int8, Int16, Int32, Int64,
                            UInt8, UInt16, UInt32, UInt64)

vkcomponenttype(::Type{T}) where {T} =
    (i = findfirst(==(T), VK_COMPONENT_TYPES); i === nothing ? nothing : UInt32(i - 1))

"The Julia type a `VkComponentTypeKHR` code names, or `nothing` if Lava has none."
juliacomponenttype(code::Integer) =
    (1 <= code + 1 <= length(VK_COMPONENT_TYPES)) ? VK_COMPONENT_TYPES[code+1] : nothing

const VK_SCOPE_SUBGROUP = UInt32(3)
const VK_SCOPE_WORKGROUP = UInt32(2)

"""
    matrixshapes(ctx) -> Vector{MatrixShape}

The driver's cooperative-matrix table as [`MatrixShape`](@ref)s.

Lava has always held this — `ctx.coopmat_shapes`, queried at device creation —
and then thrown all but a boolean away: `caps` reported a single hardcoded tile.
This is the same data in the vocabulary Mantle can also name, so a kernel picks
its tile from what the device said rather than from a constant that happened to
be right on the card it was written on.

Entries whose component type Lava does not map are dropped rather than guessed.
"""
function matrixshapes(ctx::VkContext)
    out = MatrixShape[]
    ctx.coopmat_available || return out
    for s in ctx.coopmat_shapes
        ab = juliacomponenttype(s.ab_type)
        acc = juliacomponenttype(s.c_type)
        (ab === nothing || acc === nothing) && continue
        scope = s.scope == VK_SCOPE_SUBGROUP  ? SubgroupScope()  :
                s.scope == VK_SCOPE_WORKGROUP ? WorkgroupScope() : nothing
        scope === nothing && continue
        push!(out, MatrixShape(ab, acc, s.M, s.N, s.K, scope))
    end
    return out
end

# `T` is the A/B operand type, and it used to be accepted and then ignored: the
# match was on M, N and K alone. A device can report the same extents for
# completely different component types — this one lists 16x16x16 for
# (Float16 -> Float32), (Float16 -> Float16), (UInt8 -> Int32) and
# (Int8 -> Int32) — so an extent-only match says "yes" for Float16 on hardware
# that only does the integer forms, and the kernel then emits cooperative-matrix
# instructions the device cannot execute.
#
# The accumulator type is not checked because the signature does not carry one;
# callers that care (`coopmat_gemm_available`) rely on the operand type plus the
# extents, which is what distinguishes the shapes in practice.
function coopmat_shape(ctx::VkContext, ::Type{T}, M::Integer, N::Integer,
                       K::Integer) where {T}
    ctx.coopmat_available || return false
    want = vkcomponenttype(T)
    want === nothing && return false
    any(s -> s.M == M && s.N == N && s.K == K &&
             s.ab_type == want && s.scope == VK_SCOPE_SUBGROUP,
        ctx.coopmat_shapes)
end
