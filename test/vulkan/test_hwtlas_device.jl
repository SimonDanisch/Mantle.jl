"""
An HWTLAS built on a non-default device keeps its buffers on that device.

The mesh-push path (`push!(::VulkanTLAS, ::Mesh)` → `hwtlas_add_geometry!` →
`_register_batch!` → `sync!`) allocates several `LavaArray`s: the per-batch
instance buffer, the concatenated instance buffer, and the triangle/offset GPU
arrays. Each used the bare `LavaArray` constructor, whose queue defaults to the
PROCESS-GLOBAL context — so a TLAS built for a second device put its instance
records on the first, and the first cross-context use faulted (`submit!:
buffer was last written on a VulkanBatchQueue from a DIFFERENT VkContext`, or a
segfault). This was the leak that made a full RayMakie render on a non-default
GPU fail while the isolated primitives all passed.

Needs a second driver; every machine here has lavapipe beside the real GPU.
"""

using Test, Mantle, Lava
using GeometryBasics: Point3f, GLTriangleFace

# A minimal two-triangle mesh, welded, so `hwtlas_add_geometry!` has real
# geometry to build a BLAS from.
function tri_mesh()
    pts = Point3f[(0,0,0), (1,0,0), (0,1,0), (1,1,0)]
    faces = GLTriangleFace[(1,2,3), (2,4,3)]
    return GeometryBasics.Mesh(pts, faces)
end

@testset "an HWTLAS built on a second device stays on it" begin
    infos = Mantle.devices(Mantle.VulkanAPI())
    any(i -> i.kind == :cpu, infos) || (@info "no software rasterizer; skipping"; return)

    default = Mantle.Device(Mantle.VulkanAPI())          # the process default
    other   = Mantle.Device(Mantle.VulkanAPI(); select = "llvmpipe")
    @test default.ctx !== other.ctx
    try
        tlas = Mantle.VulkanTLAS(Mantle.backend(other))
        @test Mantle.ctxof(tlas.bq)::Mantle.VkContext === other.ctx
        push!(tlas, tri_mesh())
        Raycore.sync!(tlas)

        # THE assertions: every GPU array the push/sync allocated is on `other`,
        # not on the default. Before the fix these were on `default.ctx`.
        batch = tlas.instances[1]
        @test (batch.instance_buf.buf[].ctx)::Mantle.VkContext === other.ctx
        @test tlas.combined_instance_buf !== nothing
        @test (tlas.combined_instance_buf.buf[].ctx)::Mantle.VkContext === other.ctx
        tlas.tri_gpu === nothing || @test (tlas.tri_gpu.buf[].ctx)::Mantle.VkContext === other.ctx
        tlas.off_gpu === nothing || @test (tlas.off_gpu.buf[].ctx)::Mantle.VkContext === other.ctx

        # And the default context was never written to by this build.
        @test Mantle.vk_context() === default.ctx
    finally
        Mantle.mark_device_lost!(other.ctx)
    end
end
