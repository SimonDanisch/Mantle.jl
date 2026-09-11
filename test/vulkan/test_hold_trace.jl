"""
One `hold!` on a trace's acceleration structure covers both levels.

`pintrace!` used to be core's statement that a trace holds the top level and
every bottom level it instances, and it existed because the four Vulkan sites
that trace — the recorded walk, `trace_rays!`, `trace_rays_indirect!` and
`bindtlas!` — each wrote the walk out and could drift: a BLAS missing from one
of them is a use-after-free that only shows when `Raycore.sync!` swaps a BLAS
under a trace that is still walking it.

There is no walk now. A `LavaTLAS` REFERENCES every `LavaBLAS` it instances, so
holding it holds them; what the ordering needs is the storage of each level, and
`syncbuf!(::Closed, ::LavaTLAS)` reports both. This pins that one call at an
emit site gives a trace everything it reads.
"""

using Test, GeometryBasics, StaticArrays, LinearAlgebra
using Raycore, Lava, Mantle

holders(x) = @atomic Mantle.stampof(x).holders

@testset "hold!(e, tlas) covers the top level and every bottom level" begin
    backend = Mantle.LavaBackend()
    bq = backend.dispatch_bq
    hwtlas = Mantle.VulkanTLAS(backend)
    push!(hwtlas, GeometryBasics.normal_mesh(Sphere(Point3f(0, 0, 0), 1f0)),
          SMatrix{4,4,Float32}(I); instance_id = UInt32(1))
    push!(hwtlas, GeometryBasics.normal_mesh(Rect3f(Vec3f(2, 0, 0), Vec3f(1))),
          SMatrix{4,4,Float32}(I); instance_id = UInt32(2))
    Raycore.sync!(hwtlas)
    tlas = hwtlas.hw_tlas::Mantle.LavaTLAS
    @test length(tlas.blases) == 2

    # Through the emitter, as the recorded walk and `bindtlas!` call it.
    o = Mantle.oneshot(bq) do e
        Mantle.hold!(e, tlas)
    end
    # Held: the structure itself, which is what keeps every level reachable.
    @test holders(tlas) == 1
    # Ordered against: the storage of the top level AND of each bottom level,
    # because that is the memory a ray query reads.
    @test tlas.storage.buf[] in o.sync
    for blas in tlas.blases
        @test blas.storage.buf[] in o.sync
    end

    Mantle.handover!(bq, Mantle.submit!(bq, o), o)
    Mantle.vk_flush!(bq)
    @test holders(tlas) == 0

    # The statement is one call, and what a level is stays the backend's.
    @test Mantle.hold! === Mantle.hold!
    @test !isdefined(Mantle, :pintrace!)
end
