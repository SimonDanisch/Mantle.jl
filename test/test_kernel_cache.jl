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
# And kernels are kept by their code, not by the session: an edit undone makes a
# new `CodeInstance` of code compiled before, which compiles nothing. Across two
# processes, `test_kernel_persistence.jl`.
#
# Counted with `kernelcompiles` rather than timed: a compile and a lookup give the
# same answer, so the answer alone cannot tell them apart.

using Test, Mantle, KernelAbstractions
import KernelInterface as KI
include(joinpath(@__DIR__, "testbackend.jl"))
const KA = KernelAbstractions

kc_value(i) = Float32(i) * 2f0
# Drawn per run: kernels outlive the process, so an edit an earlier run made is
# found, not compiled.
const KC_FACTOR = Float32(rand(10_000:99_999))

@kernel function kc_fill!(out)
    i = @index(Global)
    @inbounds out[i] = kc_value(i)
end

# Both entry functions use a helper that reads a builtin itself. Linked functions must
# share the builtin variable, and a cached helper must still follow method invalidation.
@noinline kc_shared(x::Float32) = x*x + Float32(KI.get_global_id().x)
@kernel function kc_first!(out)
    i = @index(Global)
    @inbounds out[i] = kc_shared(Float32(i))
end
@kernel function kc_second!(out)
    i = @index(Global)
    @inbounds out[i] = kc_shared(Float32(i)) + 2f0
end

function kc_shared_images(backend)
    a, b = KA.zeros(backend, Float32, 64), KA.zeros(backend, Float32, 64)
    kc_first!(backend)(a; ndrange = 64)
    kc_second!(backend)(b; ndrange = 64)
    Mantle.waitidle(Mantle.Device(backend))
    return Array(a), Array(b)
end

@testset "linked helpers execute and observe a callee edit" begin
    expected = Float32.(1:64).^2 + Float32.(1:64)
    a, b = kc_shared_images(TESTBACKEND)
    @test a == expected
    @test b == expected .+ 2f0
    @eval @noinline kc_shared(x::Float32) = x*x + Float32(KI.get_global_id().x) + 3f0
    a, b = Base.invokelatest(kc_shared_images, TESTBACKEND)
    @test a == expected .+ 3f0
    @test b == expected .+ 5f0
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
        @eval kc_value(i) = Float32(i) * $KC_FACTOR
        @test Base.invokelatest(kc_run, backend) == 3 * KC_FACTOR
        @test Mantle.kernelcompiles(dev).misses == 1
    end

    @testset "an edit undone compiles nothing" begin
        Mantle.resetkernelcompiles!(dev)
        @eval kc_value(i) = Float32(i) * 2f0
        @test Base.invokelatest(kc_run, backend) == 6f0
        @test Mantle.kernelcompiles(dev).misses == 0
    end
end
