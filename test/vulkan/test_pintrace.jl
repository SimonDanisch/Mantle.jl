"""
The acceleration structures a trace reads are pinned by ONE statement, core's.

`pintrace!` holds the top level and every bottom level it instances for as long
as the owner can run. It was written out four times in the Vulkan backend — the
recorded walk, `trace_rays!`, `trace_rays_indirect!` and `bindtlas!` — and the
four could drift: a BLAS pin missing from one of them is a use-after-free that
only shows when `Raycore.sync!` swaps a BLAS under a trace that is still
walking it. There is one now, over two backend answers, and this pins both
that it holds everything and that it is core's.
"""

using Test, GeometryBasics, StaticArrays, LinearAlgebra
using Raycore, Lava, Mantle

@testset "pintrace! holds the top level and every bottom level" begin
    backend = MVE.LavaBackend()
    bq = backend.dispatch_bq
    hwtlas = MVE.VulkanTLAS(backend)
    push!(hwtlas, GeometryBasics.normal_mesh(Sphere(Point3f(0, 0, 0), 1f0)),
          SMatrix{4,4,Float32}(I); instance_id = UInt32(1))
    push!(hwtlas, GeometryBasics.normal_mesh(Rect3f(Vec3f(2, 0, 0), Vec3f(1))),
          SMatrix{4,4,Float32}(I); instance_id = UInt32(2))
    Raycore.sync!(hwtlas)
    tlas = hwtlas.hw_tlas::MVE.LavaTLAS
    @test length(Mantle.blases(tlas)) == 2

    # Through the emitter, as the recorded walk and `bindtlas!` call it.
    batch = MVE.oneshot(bq) do e
        Mantle.pintrace!(e, tlas)
    end
    @test tlas.accel in batch.pinned
    @test tlas.storage in batch.pinned
    for blas in tlas.blases
        @test blas.accel in batch.pinned
        @test blas.storage in batch.pinned
    end
    MVE.recycle!(bq, batch)

    # And on the closed buffer itself, as the unmodelled paths call it.
    batch = MVE.oneshot(bq) do e
        Mantle.pintrace!(e.owner, tlas)
    end
    @test all(b -> b.accel in batch.pinned && b.storage in batch.pinned, tlas.blases)
    MVE.recycle!(bq, batch)

    # The statement is core's, and the backend only answers what a level is.
    @test only(methods(Mantle.pintrace!)).module === Mantle
    @test MVE.pin! === Mantle.pin!
    @test MVE.blases === Mantle.blases
end
