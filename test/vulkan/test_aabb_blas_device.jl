# Procedural boxes made and moved on the device: a kernel writes them into
# `Mantle.aabbinput`, the bottom-level structure is built from that buffer, a kernel
# moves them, and `Raycore.refit!` refits the structure from the same buffer. Nothing
# goes through the host, which is what a particle simulation rebuilding its neighbour
# structure every step needs.
#
# Each query point walks the boxes its inline ray query is offered and counts those
# that contain it, so "did the structure move" is a count, not a judgement: a refit
# that left the structure where it was would still offer the old boxes at the old
# points. The walk uses Vulkan's ray-query intrinsics, so the test runs on Vulkan.

using Test, Mantle, Raycore, Adapt
import KernelAbstractions as KA
using KernelAbstractions: @kernel, @index, @Const
using GeometryBasics: Point3f
include(joinpath(@__DIR__, "..", "testbackend.jl"))

if TESTBACKEND isa Mantle.LavaDevice && Mantle.supports_hwtlas(TESTBACKEND) &&
   Mantle.supports_procedural_traversal(TESTBACKEND)

using Mantle.Lava: lava_ray_query_init, lava_ray_query_proceed, lava_ray_query_get_type,
                   lava_ray_query_get_primitive_index

@kernel function centred_boxes!(boxes, @Const(c), r::Float32)
    i = @index(Global)
    @inbounds p = c[i]
    @inbounds boxes[i] = Mantle.AABB(Point3f(p[1] - r, p[2] - r, p[3] - r), Point3f(p[1] + r, p[2] + r, p[3] + r))
end

@kernel function shift_x!(c, d::Float32)
    i = @index(Global)
    @inbounds p = c[i]
    @inbounds c[i] = Point3f(p[1] + d, p[2], p[3])
end

# How many boxes round the centres `c` contain each query point, through the structure.
@kernel function count_containing!(n, @Const(q), @Const(c), r::Float32, accel)
    i = @index(Global)
    @inbounds x = q[i]
    lava_ray_query_init(UInt32(256), UInt32(0xff), x[1], x[2], x[3], 0f0, 1f0, 0f0, 0f0, 1f-4)
    k = Int32(0)
    while lava_ray_query_proceed()
        if lava_ray_query_get_type(false) == UInt32(1)
            j = lava_ray_query_get_primitive_index(false) + UInt32(1)
            @inbounds p = c[j]
            if abs(x[1] - p[1]) <= r && abs(x[2] - p[2]) <= r && abs(x[3] - p[3]) <= r
                k += Int32(1)
            end
        end
    end
    @inbounds n[i] = k
end

function counts(points, centres, r, accel)
    q = Mantle.devicearray(TESTBACKEND, points)
    n = KA.zeros(TESTBACKEND, Int32, length(points))
    count_containing!(TESTBACKEND, 64)(n, q, centres, r, accel; ndrange = length(points))
    KA.synchronize(TESTBACKEND)
    return Array(n)
end

@testset "procedural boxes built and refit on the device" begin
    start = [Point3f(0, 0, 0), Point3f(5, 0, 0), Point3f(10, 0, 0)]
    r = 1f0
    centres = Mantle.devicearray(TESTBACKEND, start)
    boxes = Mantle.aabbinput(TESTBACKEND, length(start))
    centred_boxes!(TESTBACKEND, 64)(boxes, centres, r; ndrange = length(start))
    blas = Mantle.build_accel!(ctx -> Mantle.build_blas_aabb(ctx, boxes; allow_update = true), Mantle.batchqueue(TESTBACKEND))
    tlas = Mantle.HWTLAS{Raycore.Triangle{UInt32}}(TESTBACKEND)
    push!(tlas, blas; instance_id = UInt32(0))
    Raycore.sync!(tlas)
    accel = Adapt.adapt(TESTBACKEND, tlas)

    moved = [p + Point3f(2.5, 0, 0) for p in start]
    @test counts(start, centres, r, accel) == Int32[1, 1, 1]
    @test counts(moved, centres, r, accel) == Int32[0, 0, 0]

    # Moved by a kernel, boxes written again by a kernel, refit from that buffer.
    shift_x!(TESTBACKEND, 64)(centres, 2.5f0; ndrange = length(start))
    centred_boxes!(TESTBACKEND, 64)(boxes, centres, r; ndrange = length(start))
    Raycore.refit!(blas, boxes)
    Raycore.refit!(tlas)
    accel = Adapt.adapt(TESTBACKEND, tlas)
    @test counts(moved, centres, r, accel) == Int32[1, 1, 1]
    @test counts(start, centres, r, accel) == Int32[0, 0, 0]

    # A refit cannot change the number of boxes.
    @test_throws ArgumentError Raycore.refit!(blas, Mantle.aabbinput(TESTBACKEND, 2))
end

end
