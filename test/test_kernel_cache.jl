# The kernel cache, end to end: the compiled kernel is kept with each kernel's
# `CodeInstance` (SPIR-V on Vulkan, through `Lava.compile_or_lookup`; Metal.jl's
# own cache on Metal), the pipeline with the device.
#
# The two things a cache keyed on anything else got wrong:
#   * an edit to a function a kernel inlines (Revise, or `@eval` as here) has to
#     reach the next launch. The frozen cache keyed on the kernel's signature and
#     its package's build id, and drew the old body for the rest of the session;
#   * an unrelated method definition moves the world age and must not compile
#     anything, or every REPL line costs a recompile of every kernel.
#
# Counted with `kernelcompiles` rather than timed: a compile and a lookup give the
# same answer, so the answer alone cannot tell them apart.

using Test, Mantle, KernelAbstractions
include(joinpath(@__DIR__, "testbackend.jl"))
const KA = KernelAbstractions

kc_value(i) = Float32(i) * 2f0

@kernel function kc_fill!(out)
    i = @index(Global)
    @inbounds out[i] = kc_value(i)
end

function kc_run(backend)
    out = KA.zeros(backend, Float32, 64)
    kc_fill!(backend)(out; ndrange = 64)
    KA.synchronize(backend)
    return Array(out)[3]
end

@testset "kernel cache" begin
    backend = TESTBACKEND
    dev = Mantle.Device(TESTBACKEND)
    @test kc_run(backend) == 6f0

    @testset "an unrelated definition compiles nothing" begin
        Mantle.resetkernelcompiles!(dev)
        @eval kc_unrelated() = 1
        @test Base.invokelatest(kc_run, backend) == 6f0
        @test Mantle.kernelcompiles(dev).misses == 0
    end

    @testset "an edited callee is launched with its new code" begin
        Mantle.resetkernelcompiles!(dev)
        @eval kc_value(i) = Float32(i) * 3f0
        @test Base.invokelatest(kc_run, backend) == 9f0
        @test Mantle.kernelcompiles(dev).misses == 1
    end
end
