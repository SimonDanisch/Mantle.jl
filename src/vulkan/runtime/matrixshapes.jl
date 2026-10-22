# Queries of the driver's cooperative-matrix table, `ctx.coopmat_shapes`, read
# at device creation. `caps(ctx)` fills `DeviceCaps.shapes` from `matrixshapes`.

# `VkComponentTypeKHR` codes in order (code = index - 1), only the types Lava
# maps; an unmapped type matches no shape. The inverse is derived from this one
# table so the two cannot disagree.
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

The driver's cooperative-matrix table as [`MatrixShape`](@ref)s, so kernels
pick tiles from what the device reports. Entries with an unmapped component type
or scope are dropped.
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

"""
    coopmat_shape(ctx, T, M, N, K) -> Bool

Whether the device implements a subgroup-scope `M×N×K` cooperative-matrix tile
with A/B operand type `T`. `T` is matched, not only the extents: devices list
16×16×16 for both Float16 and Int8 operands, and an extent-only match emits
instructions the device cannot run. The accumulator type is not checked.
"""
function coopmat_shape(ctx::VkContext, ::Type{T}, M::Integer, N::Integer,
                       K::Integer) where {T}
    ctx.coopmat_available || return false
    want = vkcomponenttype(T)
    want === nothing && return false
    any(s -> s.M == M && s.N == N && s.K == K &&
             s.ab_type == want && s.scope == VK_SCOPE_SUBGROUP,
        ctx.coopmat_shapes)
end
