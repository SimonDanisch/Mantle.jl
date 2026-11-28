# `DeviceCaps` — what a kernel asks the device before it picks a tiling.
#
# Written against whatever the running device reports rather than a vendor's
# numbers, so it means the same thing everywhere.
#
# The Vulkan version of this file also cross-checked each field against a second,
# raw driver query, because the failure that produced it was silent:
# `vkGetPhysicalDeviceProperties2` fills its `pNext` chain only if the
# application opted into it (core in Vulkan 1.1, or
# `VK_KHR_get_physical_device_properties2` before that), and without it the
# driver drops the chain and returns void. Measured on an RTX 4000 Ada: a 1.0
# instance read `shader_sm_count = 0` where a 1.1+ instance read 48, and every
# launch heuristic downstream quietly fell back to its default. Those
# cross-checks are one driver API's and are not repeated here. What stays is what
# a kernel can rely on from any device: the limits are real, a launch at the
# reported workgroup limit runs, the shape table and `tile` agree, and the struct
# is one cached value a copy cannot reach.

using Test, Mantle
import KernelAbstractions as KA
using KernelAbstractions: @kernel, @index
include(joinpath(@__DIR__, "testbackend.jl"))

@kernel function caps_fill_index!(a)
    i = @index(Global)
    @inbounds a[i] = Int32(i)
end

@testset "DeviceCaps" begin
    c = Mantle.caps(TESTBACKEND)

    @testset "shared memory limit is real" begin
        m = c.sharedbudget
        @test m isa Int
        @test m > 0
        # A workgroup cannot be given more than this; kernels that size
        # `@localmem` against a budget must stay at or under it. 16 KiB is the
        # least any device Mantle runs on guarantees (Vulkan's mandated minimum).
        @test m >= 16 * 1024
    end

    @testset "workgroup limit is queried, not assumed" begin
        # This is the regression the field exists for: it was `Ref(1024)` with a
        # docstring claiming to be the device's query. A device whose real limit
        # is not 1024 would have been told it was.
        @test c.workgrouplimit >= 128   # the Vulkan-mandated minimum
        # The limit is one the device actually launches at: one workgroup of
        # exactly `workgrouplimit` invocations runs and writes every element.
        n = c.workgrouplimit
        a = KA.allocate(TESTBACKEND, Int32, n)
        fill!(a, Int32(0))
        caps_fill_index!(TESTBACKEND, n)(a; ndrange = n)
        KA.synchronize(TESTBACKEND)
        @test Array(a) == Int32.(1:n)
    end

    @testset "core count: zero, or a positive number" begin
        # `0` is DeviceCaps' "the device will not say", which Metal answers.
        @test c.cores isa Int && c.cores >= 0
        @test c.warps isa Int && c.warps >= 0
    end

    @testset "the coopmat width fits a workgroup" begin
        # The distinction the field exists to make: on RDNA 3.5 the device default
        # subgroup is 64 while every cooperative-matrix module is pinned to 32, so
        # a workgroup sized in units of `subgroup` asks for twice the threads the
        # kernel indexes and writes half its tile.
        @test c.subgroup > 0
        @test c.coopmatsubgroup > 0
        # A coopmat kernel's workgroup is a multiple of `coopmatsubgroup`; that
        # multiple has to fit in the device's limit for any of this to launch.
        @test c.coopmatsubgroup <= c.workgrouplimit
    end

    @testset "queried once, cached on the device" begin
        # Not a performance claim — an identity one. Two reads must be the same
        # object, because a kernel that reads `caps` twice while deciding a
        # tiling must not see two different devices. A device and the backend
        # under it are two handles on one answer.
        @test Mantle.caps(TESTBACKEND) === c
        @test Mantle.caps(Mantle.Device(TESTBACKEND)) === c
    end

    @testset "a modified copy leaves the device's own answer alone" begin
        # How a test asks what a kernel would decide on a device it does not
        # have. The copy is a value; nothing about it can reach the device.
        w64 = Mantle.DeviceCaps(c; subgroup = 64, coopmat = false)
        @test w64.subgroup == 64
        @test w64.coopmat == false
        @test w64.tile == c.tile
        @test w64.workgrouplimit == c.workgrouplimit
        @test w64.coopmatsubgroup == c.coopmatsubgroup   # not named, so unchanged
        @test Mantle.caps(TESTBACKEND) === c              # the device is untouched
        @test Mantle.DeviceCaps(c) == c                   # no keywords = no change
    end

    @testset "the shape table is the driver's, and tile is one entry of it" begin
        # `tile` as a module constant described as "the cooperative-matrix tile
        # this device implements" was a device fact that no device had answered
        # for. It is read from the table the device reports now: 16 on the Vulkan
        # GPUs here, 8 for Metal's simdgroup matrix.
        @test all(s -> s.scope isa Mantle.SubgroupScope, c.shapes)
        if c.coopmat
            @test !isempty(c.shapes)
            sq = Mantle.bestshape(c, Float16, Float32)
            @test sq !== nothing
            @test sq.M == sq.N == sq.K          # `bestshape` prefers square
            @test c.tile == sq.M                # …and `tile` IS that entry
            @test Mantle.supports(c, sq)
        else
            # No matrix hardware: nothing is offered, whatever the table holds.
            @test Mantle.bestshape(c, Float16, Float32) === nothing
        end
    end

    @testset "a device with no matrix hardware reports no shapes" begin
        # The gate is in the accessor, NOT in the copy constructor: a copy has to
        # change exactly the field it names, so `shapes` and `tile` stay put and
        # it is `bestshape`/`supports` that answer as the device claims to be.
        off = Mantle.DeviceCaps(c; coopmat = false)
        @test off.shapes == c.shapes                     # the copy contract holds
        @test off.tile == c.tile
        @test Mantle.bestshape(off, Float16, Float32) === nothing
        if c.coopmat
            @test !Mantle.supports(off, Mantle.bestshape(c, Float16, Float32))
        end
    end

    @testset "a shape the device does not have is refused" begin
        # The point of carrying types rather than extents alone: a device lists
        # its tile extent for several type pairs, so an extent-only match would
        # say yes for a pair it cannot execute. No device here has a Float64
        # cooperative matrix.
        @test Mantle.bestshape(c, Float64, Float64) === nothing
        @test !Mantle.supports(c, Mantle.MatrixShape(Float64, Float64, c.tile, c.tile,
                                                     c.tile, Mantle.SubgroupScope()))
    end
end
