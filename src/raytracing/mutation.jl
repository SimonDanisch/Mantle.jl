# One registration/mutation algorithm for incremental hardware TLASes. Native
# backends implement buildgeometry!, instancearray and updatebatch!, and may
# select a table index instead of a native object with geometryref.

function buildgeometry! end
function instancearray end
function updatebatch! end

function geometryindex!(t::BatchedTLAS, blas)
    i = findfirst(b -> b === blas, t.blas_list)
    i === nothing || return i
    push!(t.blas_list, blas)
    return length(t.blas_list)
end

batchgeometry!(t::BatchedTLAS, blas, triangles) =
    (geometryindex!(t, blas), triangles)

function batchgeometry!(t::BatchedTLAS{Tri}, mesh::GeometryBasics.Mesh, _) where {Tri}
    triangles = meshtriangles(Tri, mesh)
    blas = buildgeometry!(t, trianglevertices(triangles))
    i = geometryindex!(t, blas)
    for tri in triangles, v in tri.vertices
        t.root_aabb = Raycore.union(t.root_aabb, Raycore.Bounds3(Point3f(v)))
    end
    return i, triangles
end

geometryref(t::BatchedTLAS, i::Int) = t.blas_list[i]

function addbatch!(t::BatchedTLAS, i::Int, records, n::Int, triangles, sbt::UInt32)
    B = eltype(t.instances)
    handle = register!(t.instances, h ->
        B(geometryref(t, i), records, n, h, triangles, sbt, false))
    t.dirty = true
    return handle
end

"""
    push!(t::BatchedTLAS, geometry, records; n=length(records), triangles, sbt_offset)

Register one batch of device-resident instance records, retaining their ids and
masks. Only `records[1:n]` are instances. A mesh builds its native BLAS; a
pre-built BLAS is registered once by identity. Invalid counts are refused before
building geometry. Native arrays are retained, while host records are uploaded.
"""
function Base.push!(t::BatchedTLAS{Tri}, geometry,
                    records::AbstractVector{Raycore.InstanceRecord};
                    n::Integer = length(records), triangles::Vector{Tri} = Tri[],
                    sbt_offset::UInt32 = UInt32(0)) where {Tri}
    count = Int(n)
    0 <= count <= length(records) || throw(ArgumentError(
        "push!: n=$count must be between 0 and $(length(records)) records"))
    i, metadata = batchgeometry!(t, geometry, triangles)
    return addbatch!(t, i, instancearray(t, records), count, metadata, sbt_offset)
end

function Base.push!(t::BatchedTLAS, geometry, transform::Mat4f = Mat4f(I);
                    instance_id::UInt32 = UInt32(0), instance_mask::UInt8 = UInt8(0xff),
                    sbt_offset::UInt32 = UInt32(0))
    return push!(t, geometry, instancerecords([transform], [instance_id], instance_mask);
                 sbt_offset)
end

function Base.push!(t::BatchedTLAS, geometry, transforms::AbstractVector{Mat4f};
                    instance_ids::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                    instance_mask::UInt8 = UInt8(0xff), sbt_offset::UInt32 = UInt32(0))
    records = instancerecords(transforms, instance_ids, instance_mask)
    return push!(t, geometry, records; sbt_offset)
end

function requirebatch(t::BatchedTLAS, handle::Raycore.TLASHandle, operation)
    batch = batchof(t.instances, handle)
    batch === nothing && throw(ArgumentError("$operation: unknown handle $handle"))
    return batch
end

# Native Mat3x4 buffers pass through instancearray without a host transfer.
# Other arrays follow the existing host boundary, including device Mat4 arrays.
transformbuffer(t::BatchedTLAS, transforms::AbstractVector) =
    instancearray(t, Mat3x4f[m isa Mat3x4f ? m : mat4_to_vk_transform(m)
                            for m in Array(transforms)])
transformbuffer(t::BatchedTLAS, transforms::AbstractVector{Mat3x4f}) =
    instancearray(t, transforms)

function Raycore.update_transforms!(t::BatchedTLAS, handle::Raycore.TLASHandle,
                                    transforms::AbstractVector)
    b = requirebatch(t, handle, "update_transforms!")
    length(transforms) == b.n || throw(ArgumentError(
        "update_transforms!: $(length(transforms)) transforms for a batch of $(b.n) instances"))
    result = updatebatch!(t, b, transformbuffer(t, transforms))
    t.transforms_dirty = true
    return result
end

function Raycore.update_transform!(t::BatchedTLAS, handle::Raycore.TLASHandle, transform)
    b = requirebatch(t, handle, "update_transform!")
    canonical = transform isa Mat3x4f ? transform : mat4_to_vk_transform(transform)
    return Raycore.update_transforms!(t, handle, fill(canonical, b.n))
end

"""Map a flat instance range onto exactly one complete batch, before mutation."""
function Raycore.update_instance_transforms!(t::BatchedTLAS, transforms::AbstractVector,
                                             n_to_update::Integer, first_idx::Integer = 1)
    n, i0 = Int(n_to_update), Int(first_idx)
    n > 0 || return nothing
    base = 0
    for b in t.instances
        if i0 <= base + b.n
            (i0 == base + 1 && n == b.n) || error(
                "update_instance_transforms!: instances $(i0):$(i0 + n - 1) do not " *
                "cover exactly one batch (batch at $(base + 1):$(base + b.n)). " *
                "A $(nameof(typeof(t))) updates whole batches; address it by handle instead.")
            return Raycore.update_transforms!(t, b.handle, transforms)
        end
        base += b.n
    end
    error("update_instance_transforms!: instance index $(i0) past the end " *
          "($(base) instances in $(length(t.instances)) batches).")
end

function Base.delete!(t::BatchedTLAS, handle::Raycore.TLASHandle)
    delete!(t.instances, handle) || return false
    t.dirty = true
    return true
end

function Raycore.set_visible!(t::BatchedTLAS, handle::Raycore.TLASHandle, visible::Bool)
    changed = sethidden!(t.instances, handle, !visible)
    changed === nothing && return false
    changed && (t.dirty = true)
    return true
end

Raycore.world_bound(t::BatchedTLAS) = t.root_aabb
Raycore.n_geometries(t::BatchedTLAS) = length(t.blas_list)
Raycore.n_instances(t::BatchedTLAS) = ninstances(t.instances)
Raycore.instance_buffer(t::BatchedTLAS, handle::Raycore.TLASHandle) =
    recordsof(t.instances, handle)
Raycore.refit!(t::BatchedTLAS) = (t.transforms_dirty = true; Raycore.sync!(t))
