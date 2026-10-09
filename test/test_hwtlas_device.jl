"""
An HWTLAS built on a non-default device keeps its buffers on that device.

The mesh-push path (`push!(::VulkanTLAS, ::Mesh)` → `hwtlas_add_geometry!` →
`_register_batch!` → `sync!`) allocates several device arrays: the per-batch
instance buffer, the concatenated instance buffer, and the triangle/offset GPU
arrays. Each used the bare `LavaArray` constructor, whose queue defaults to the
PROCESS-GLOBAL context — so a TLAS built for a second device put its instance
records on the first, and the first cross-context use faulted (`submit!:
buffer was last written on a VulkanBatchQueue from a DIFFERENT VkContext`, or a
segfault). That is the leak that makes a full RayMakie render on a non-default
GPU fail while the isolated primitives all pass.

Two things a caller would see, and both are asserted: tracing the structure on
the device it was built for works, and the device under test lent it no memory.
The second device is a device of its own (`Device(select)`: a new context, never
installed as the default) on the best GPU here, so this needs no second driver.
"""

using Test, Mantle, Raycore, Adapt
using GeometryBasics
using GeometryBasics: Point3f, Vec3f, GLTriangleFace
include(joinpath(@__DIR__, "testbackend.jl"))
isdefined(@__MODULE__, :tlasprobe!) || include(joinpath(@__DIR__, "hwtlas_helpers.jl"))

# A minimal two-triangle mesh, welded, so `push!` has real geometry to build a
# BLAS from.
function tri_mesh()
    pts = Point3f[(0,0,0), (1,0,0), (0,1,0), (1,1,0)]
    faces = GLTriangleFace[(1,2,3), (2,4,3)]
    return GeometryBasics.Mesh(pts, faces)
end

"""Regions the device's pool has on loan, across every block."""
onloan(dev) = sum(b -> length(b.live), Iterators.flatten(values(Mantle.pool(dev).blocks)); init = 0)

@testset "an HWTLAS built on a second device stays on it" begin
    default = Mantle.Device()
    mine = Mantle.Device(TESTBACKEND)
    other = Mantle.Device(i -> true)          # the best GPU, in a context of its own
    @test other !== mine
    if !Mantle.supports_hwtlas(other)
        @info "no hardware acceleration structures on this device; skipping"
    else
        GC.gc(true)
        Mantle.reclaim!(Mantle.pool(mine), mine; wait = true)
        before = onloan(mine)

        be = Mantle.backend(other)
        tlas = Mantle.HWTLAS{Raycore.Triangle{UInt32}}(be)
        push!(tlas, tri_mesh())
        Raycore.sync!(tlas)

        # Traced ON `other`: before the fix this was the cross-context fault.
        r = tlastrace(be, tlas, [Point3f(0.25f0, 0.25f0, 1f0), Point3f(5f0, 5f0, 1f0)],
                      fill(Vec3f(0, 0, -1), 2))
        @test r.hit == [true, false]
        @test isapprox(r.t[1], 1f0; atol = 1f-3)

        # THE assertion: nothing the push, the sync or the trace allocated is on
        # the device under test. `tlas` is alive, so a buffer of it that had gone
        # there would still be on loan.
        @test onloan(mine) == before
        # And the default device was not swapped for this build.
        @test Mantle.Device() === default
    end
end
