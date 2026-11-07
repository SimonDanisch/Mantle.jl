# A dispatch argument holding an acceleration structure reaches its device
# through `HWTLAS.backend`. The device's queue holds retired storage until the
# GPU is done with it, so a walk into the device named buffers no dispatch reads,
# including ones already freed. Hikari's shade plans then failed to compile when
# a collection finalized an old scene's BLAS during plan building.
using Test, Mantle, Raycore, GeometryBasics, LinearAlgebra
include(joinpath(@__DIR__, "..", "testbackend.jl"))

struct ArgumentWithDevice{A, D}
    array::A
    device::D
end

function walked(g, x)
    return Mantle.resourceleaves!(Any[], g, x)
end

@testset "a device is not one of a dispatch argument's resources" begin
    dev = Mantle.Device(TESTBACKEND)
    x = Mantle.devicearray(dev, zeros(Float32, 16))
    g = Mantle.Graph(dev)
    leaves = walked(g, ArgumentWithDevice(x, dev))
    # One leaf, the array: the bytes of its owner it covers.
    @test length(leaves) == 1 && only(leaves) === Mantle.nativeslice(g, x)
    if Mantle.supports_hwtlas(TESTBACKEND)
        mesh = GeometryBasics.Mesh(Point3f[(0, 0, 0), (1, 0, 0), (0, 1, 0)], GLTriangleFace[(1, 2, 3)])
        records = Mantle.instancerecords([Mat4f(I)], UInt32[1], UInt8(0xff))
        live = Mantle.HWTLAS{Raycore.Triangle{UInt32}}(TESTBACKEND)
        push!(live, mesh, records; n = 1)
        Raycore.sync!(live)
        old = Mantle.HWTLAS{Raycore.Triangle{UInt32}}(TESTBACKEND)
        push!(old, mesh, records; n = 1)
        Raycore.sync!(old)
        retired = walked(g, only(old.blas_list).storage)
        Mantle.unsafe_free!(only(old.blas_list))
        # The retirement is pending on the device the live structure names.
        @test any(l -> any(r -> r === l, retired), walked(g, dev.default_bq))
        @test !any(l -> any(r -> r === l, retired), walked(g, live))
    end
end
