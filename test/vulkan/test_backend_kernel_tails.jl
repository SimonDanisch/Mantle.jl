# This backend's own plain kernels, held to the rule `test_kernel_tails.jl`
# pins for core's: a launch is whole workgroups, so a kernel guards its own
# tail, and a lane past the ndrange writes nothing. Three of this backend's lost
# the guard with `@kernel`: the split-K reduction, and both instance-record
# writers, which a TLAS of more than 1024 instances reaches.
#
# Same shape as there: a window onto the front of a larger buffer, an ndrange
# the group does not divide, and a canary in the rest.

using Test, Mantle
using Mantle: viewof, VulkanInstanceRecord, Mat3x4f
using GeometryBasics: Point3f, Vec4f

@testset "this backend's plain kernels write nothing past their ndrange" begin
    dev = Mantle.todevice(Mantle.defaultbackend())
    tailed(n, pad, v) = (p = Mantle.Buffer(dev, fill(v, n + pad)); (p, viewof(p, n)))
    tail(p, n) = Array(Mantle.storage(p))[(n + 1):end]
    function runtails!(f, args, n)
        g = Mantle.Graph(dev)
        Mantle.dispatch!(g, f, args, n; group = 64)   # divides none of the ndranges below
        Mantle.runonce!(g)
        return nothing
    end
    canary = VulkanInstanceRecord(Mat3x4f(ntuple(Float32, 12)), 0xdeadbeef, 0xdeadbeef,
                                  0xdeadbeefdeadbeef)

    @testset "splitk_reduce_kernel!" begin
        n, S = 99, 3
        Cp = Mantle.Buffer(dev, ones(Float32, S * n))
        p, C = tailed(n, 64, -1f0)
        runtails!(Mantle.splitk_reduce_kernel!, (C, Cp, Val(S), n), n)
        got = Array(Mantle.storage(p))
        @test all(==(Float32(S)), got[1:n])
        @test all(==(-1f0), got[(n + 1):end])
        foreach(Mantle.free!, (p, Cp))
    end

    # Past its last grain a lane would read the next position in the parent and
    # write two records into the canary.
    @testset "write_grain_instances_kernel" begin
        n = 5
        pp, positions = tailed(n, 64, Point3f(1, 2, 3))
        quats = Mantle.Buffer(dev, fill(Vec4f(0, 0, 0, 1), n + 64))
        ip, instances = tailed(2n, 128, canary)
        runtails!(Mantle.write_grain_instances_kernel,
                  (positions, quats, 1f0, UInt64(1), UInt64(2), instances), n)
        got = Array(Mantle.storage(ip))
        @test all(r -> r.blas_address in (UInt64(1), UInt64(2)), got[1:2n])
        @test all(==(canary), got[(2n + 1):end])
        foreach(Mantle.free!, (pp, quats, ip))
    end

    @testset "update_instance_records_kernel!" begin
        n = 5
        rp, records = tailed(n, 64, canary)
        transforms = Mantle.Buffer(dev, fill(Mat3x4f(ntuple(_ -> 0.5f0, 12)), n + 64))
        runtails!(Mantle.update_instance_records_kernel!, (records, transforms, UInt64(7)), n)
        got = Array(Mantle.storage(rp))
        @test all(r -> r.blas_address == UInt64(7), got[1:n])
        @test all(==(canary), got[(n + 1):end])
        foreach(Mantle.free!, (rp, transforms))
    end
end
