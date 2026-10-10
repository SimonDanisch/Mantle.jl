# A compiled kernel survives a device reset: the device a reset builds launches it
# without compiling it again.
#
# What the Vulkan original asked was narrower and could not be asked of Metal: that
# the driver's own pipeline cache (`VkPipelineCache`, saved to disk at the reset)
# spared the DRIVER's shader compiler, observed through
# `VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED`. It also recorded that the
# instrument never fires on RADV, which builds the pipeline anyway, so on AMD it
# asserted nothing. What both backends answer is `kernelcompiles(dev)`: whether
# the compiler ran — Julia to SPIR-V on Vulkan, Julia to AIR on Metal — and that
# is what a reset could throw away and must not.
#
# A second launch compiling nothing, and an edited kernel compiling once, are
# `test_caching_and_allocations.jl` and `test_kernel_cache.jl`. The negative control
# here is a kernel no cache can hold — a literal drawn per run — which must count
# as a compile, or the zero after the reset means nothing.
#
# In a process of its own: a reset replaces the default device, which the suite's
# other files are using.

using Test, Mantle, KernelAbstractions
include(joinpath(@__DIR__, "testbackend.jl"))
include(joinpath(@__DIR__, "freshprocess.jl"))

@testset "a device reset compiles nothing again" begin
    status, out = infreshprocess(raw"""
        @kernel function kcr_warm!(d)
            i = @index(Global)
            @inbounds d[i] = d[i] + 1.0f0
        end
        # Novel on every run: `K` is a literal in the compiled code, drawn per run.
        @kernel function kcr_novel!(d, ::Val{K}) where {K}
            i = @index(Global)
            @inbounds d[i] = d[i] * 3.0f0 - Float32(K)
        end
        function bump(be, n)
            a = Mantle.devicearray(be, zeros(Float32, n))
            kcr_warm!(be, 64)(a; ndrange = n)
            KA.synchronize(be)
            return Array(a)
        end
        dev = Mantle.Device(BACKEND)
        @testset "compiled once, before and after a reset" begin
            @test bump(BACKEND, 64) == ones(Float32, 64)
            # The instrument fires: a kernel nothing has seen is a compile.
            Mantle.resetkernelcompiles!(dev)
            K = rand(10_000:99_999)
            b = Mantle.devicearray(BACKEND, ones(Float32, 64))
            kcr_novel!(BACKEND, 64)(b, Val(K); ndrange = 64)
            KA.synchronize(BACKEND)
            @test Mantle.kernelcompiles(dev).misses >= 1
            @test Array(b) == fill(3.0f0 - Float32(K), 64)

            new = Mantle.reset_device!(dev)
            Mantle.resetkernelcompiles!(new)
            @test bump(Mantle.backend(new), 64) == ones(Float32, 64)
            @test Mantle.kernelcompiles(new).misses == 0
        end
        """)
    reportfresh(status, out)
    @test status == 0
end
