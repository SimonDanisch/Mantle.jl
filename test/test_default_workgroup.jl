"""
A kernel whose workgroup is `DynamicSize`, launched without one, gets core's
`defaultgroup` on every backend: 64 work-items, filling the dimensions in order.

Before, each backend had its own: Metal's recorder filled the first dimension up to
the kernel's maximum (two 1024-wide groups per row of a 1080-wide frame, so a
1080x1920 editor render took 2.03 ms against 1.27 ms for 1920x1080), and Vulkan's
put `min(prod(ndrange), 64)` along the first dimension (a `(3, n)` range ran 3 of
every 64 lanes). The shapes are pinned, and so is the coverage of ranges whose
first extent is below 64, which now get two-dimensional workgroups.

The last dimension is not clamped to its extent: Lava compiles the workgroup into
the kernel, and a range of 10 got a group of 10, so every small launch size was a
kernel of its own (a scene's TLAS kernels compiled again for each instance count).
A small range now runs one full group, and the lanes past it must write nothing.
"""

using Test, Mantle, KernelAbstractions
include(joinpath(@__DIR__, "testbackend.jl"))
const KA = KernelAbstractions

@kernel function dg_coords!(out)
    i, j = @index(Global, NTuple)
    @inbounds out[i, j] = Int32(i + 10000 * j)
end

@testset "the default workgroup: 64 work-items, the dimensions in order" begin
    @test Mantle.defaultgroup((1080, 1920), 1024) == (64, 1)
    @test Mantle.defaultgroup((3, 1000), 1024) == (3, 21)
    @test Mantle.defaultgroup((1, 300), 1024) == (1, 64)
    @test Mantle.defaultgroup((5000,), 32) == (32,)
    # Independent of how small the range is, so one kernel serves every size.
    @test Mantle.defaultgroup((10,), 1024) == Mantle.defaultgroup((3,), 1024) == (64,)
    @test Mantle.defaultgroup((3, 2), 1024) == Mantle.defaultgroup((3, 7), 1024) == (3, 21)
end

@kernel function dg_index!(out)
    i = @index(Global)
    @inbounds out[i] = Int32(i)
end

@testset "a range smaller than one group writes only its own elements" begin
    dev = Mantle.Device(TESTBACKEND)
    for n in (1, 3, 10, 63, 65)
        # One element past the range, which no lane may write.
        out = Mantle.Buffer(dev, zeros(Int32, n + 1))
        g = Mantle.Graph(dev)
        Mantle.dispatch!(g, dg_index!, (out,), (n,); name = "index")
        pl = Mantle.record!(Base.invokelatest(Mantle.Plan, g))
        Mantle.run!(pl); Mantle.waitfor!(pl)
        @test Array(Mantle.storage(out)) == [Int32.(1:n); Int32(0)]
        Mantle.free!(pl); Mantle.free!(out)
    end
end

@testset "a dynamic workgroup covers every element, whatever the first extent" begin
    dev = Mantle.Device(TESTBACKEND)
    for (w, h) in ((3, 1000), (1080, 7), (65, 3), (1, 300))
        out = Mantle.Buffer(dev, zeros(Int32, w, h))
        g = Mantle.Graph(dev)
        Mantle.dispatch!(g, dg_coords!, (out,), (w, h); name = "coords")
        pl = Mantle.record!(Base.invokelatest(Mantle.Plan, g))
        Mantle.run!(pl); Mantle.waitfor!(pl)
        @test Array(Mantle.storage(out)) == [Int32(i + 10000j) for i in 1:w, j in 1:h]
        Mantle.free!(pl); Mantle.free!(out)
    end
end
