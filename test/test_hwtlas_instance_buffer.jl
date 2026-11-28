# `Raycore.instance_buffer(tlas, handle)`: the device array of a batch's
# `Raycore.InstanceRecord`s, which a kernel may write and `Raycore.refit!` commits.
# It must be THE array — the one pushed, not a copy — or what a kernel writes
# into it never reaches the structure; and a push that made its own array must
# hand back what it wrote.

using Test, Mantle, Raycore
using GeometryBasics: Point3f, Mat4f, GLTriangleFace
import GeometryBasics
import LinearAlgebra
include(joinpath(@__DIR__, "testbackend.jl"))

const Tri = Raycore.Triangle{UInt32}

if Mantle.supports_hwtlas(TESTBACKEND)

@testset "instance_buffer returns the array a batch was pushed with" begin
    blas = Mantle.build_accel!(Mantle.batchqueue(Mantle.Device(TESTBACKEND))) do ctx
        Mantle.build_blas(ctx, [(0f0, 0f0, 0f0), (1f0, 0f0, 0f0), (0f0, 1f0, 0f0)], UInt32[0, 1, 2])
    end
    n = 8
    records = Mantle.devicearray(TESTBACKEND,
        [Raycore.InstanceRecord(Mat4f(LinearAlgebra.I), UInt32(i), UInt32(0x04)) for i in 1:n])
    tlas = Mantle.HWTLAS{Tri}(TESTBACKEND)
    handle = push!(tlas, blas, records; n = n)
    # The SAME array (identity, not a copy).
    @test Raycore.instance_buffer(tlas, handle) === records
    Raycore.sync!(tlas)
    @test Raycore.instance_buffer(tlas, handle) === records
end

@testset "instance_buffer of a pushed mesh holds what push! wrote" begin
    mesh = GeometryBasics.normal_mesh(GeometryBasics.Mesh(
        [Point3f(0, 0, 0), Point3f(1, 0, 0), Point3f(0, 1, 0)], [GLTriangleFace(1, 2, 3)]))
    moved = Mat4f(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 3, 0, 0, 1)
    tlas = Mantle.HWTLAS{Tri}(TESTBACKEND)
    handle = push!(tlas, mesh, [Mat4f(LinearAlgebra.I), moved];
                   instance_ids = UInt32[7, 9], instance_mask = UInt8(0x06))
    got = Array(Raycore.instance_buffer(tlas, handle))[1:2]
    @test got == [Raycore.InstanceRecord(Mat4f(LinearAlgebra.I), UInt32(7), UInt32(0x06)),
                  Raycore.InstanceRecord(moved, UInt32(9), UInt32(0x06))]
end

@testset "instance_buffer refuses a handle that names no batch" begin
    tlas = Mantle.HWTLAS{Tri}(TESTBACKEND)
    @test_throws ArgumentError Raycore.instance_buffer(tlas, Raycore.TLASHandle(UInt32(99)))
end

else
    @info "no hardware acceleration structures on this backend; skipping"
end
