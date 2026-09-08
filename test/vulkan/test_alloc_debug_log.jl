# `ctx.diag.alloc_debug` logs every allocation into `ctx.diag.alloc_log`. The
# acceleration-structure sites still pushed to `ALLOC_DEBUG_LOG`, the
# module-level vector the diag refactor deleted, so switching the flag on and
# building a BLAS threw `UndefVarError` — inside RayMakie's plot resolution,
# where it surfaced as "failed to resolve trace_renderobject" and a scene with no
# geometry. Pins: with the flag on, a BLAS build logs a `:blas_as` entry with
# the same fields the direct path logs, and throws nothing.
using Test, Mantle, Lava, Raycore, GeometryBasics
using StaticArrays: SMatrix

@testset "alloc_debug logs acceleration structures" begin
    ctx = MVE.vk_context()
    was = ctx.diag.alloc_debug
    ctx.diag.alloc_debug = true
    n0 = length(ctx.diag.alloc_log)
    try
        mesh = GeometryBasics.normal_mesh(GeometryBasics.Mesh(
            [Point3f(0, 0, 0), Point3f(1, 0, 0), Point3f(0, 1, 0)], [GLTriangleFace(1, 2, 3)]))
        hwtlas = MVE.VulkanTLAS(LavaBackend())
        push!(hwtlas, mesh, SMatrix{4,4,Float32,16}(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
              instance_id = UInt32(1))
        Raycore.sync!(hwtlas)
    finally
        ctx.diag.alloc_debug = was
    end
    added = ctx.diag.alloc_log[n0+1:end]
    @test !isempty(added)
    as = filter(e -> e.kind in (:blas_as, :blas_as_pool, :blas_as_pool2, :blas_aabb_as), added)
    @test !isempty(as)
    # Every entry, from every site, has the shape the direct path writes.
    @test all(e -> keys(e) == (:kind, :addr, :size, :pool, :mtype, :unified, :usage), added)
end
