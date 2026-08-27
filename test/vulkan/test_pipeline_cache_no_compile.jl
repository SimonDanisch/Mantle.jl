# The VkPipelineCache must eliminate DRIVER compilation, not just look fast.
#
# Lava has two caches and they answer different questions:
#
#   frozen / disk kernel cache  Julia  -> SPIR-V   (test_frozen_cache.jl, test_disk_cache.jl)
#   VkPipelineCache             SPIR-V -> binary   (here)
#
# Only the second one avoids the driver's shader compiler, and it was the one
# with no coverage: `pipeline_cache.jl` had no registered test at all, just an
# assertion-free MWE.
#
# Timing cannot verify this. A fast second run is equally explained by Lava's
# in-memory `PIPELINE_CACHE` dict, which never touches disk. So this uses
# `pipelineCreationCacheControl` (enabled in vk_device!'s Vulkan 1.3 feature set):
# `no_pipeline_compilation` sets FAIL_ON_PIPELINE_COMPILE_REQUIRED, and the driver
# then REFUSES to build a binary, returning VK_PIPELINE_COMPILE_REQUIRED, instead
# of quietly compiling one. A run that completes did zero driver compilation
# because the driver said so.
#
# The negative control is not optional. A green run against an instrument that
# never fires proves nothing — the same trap `verify_gpu_av` exists to avoid, and
# GPU-AV really has been observed attaching without ever firing. So a kernel the
# driver has never seen must be REFUSED here, or this test is vacuous.

using Test, Lava, KernelAbstractions
const KA = KernelAbstractions

@kernel function pcnc_warm!(d)
    i = @index(Global)
    @inbounds d[i] = d[i] + 1.0f0
end

# Novel on EVERY RUN, not merely unlike the rest of the suite, and that
# distinction is the whole reason the negative control works.
#
# `Val{K}` with a per-run random `K` puts a different literal in the emitted
# SPIR-V each time, so no cache anywhere can hold this module. A fixed body is
# novel exactly ONCE PER MACHINE and silently stops firing after that — which is
# what it had been doing on both vendors:
#
#   * NVIDIA RTX 4000 Ada: fired once, on the first run after an emitter change
#     made its SPIR-V new again, and never afterwards.
#   * AMD RADV: never observed firing at all.
#
# Deleting Lava's own `VkPipelineCache` blob does not restore it on either, and
# neither does `MESA_SHADER_CACHE_DISABLE` on RADV, because the vendor keeps its
# own shader cache outside anything Lava controls. So "delete the blob" is not a
# workaround, and a control that fires once per machine is not a control.
@kernel function pcnc_novel!(d, ::Val{K}) where {K}
    i = @index(Global)
    @inbounds d[i] = d[i] * 3.0f0 - Float32(K) + sqrt(abs(d[i]) + 1.5f0)
end

@testset "VkPipelineCache avoids driver compilation" begin
    backend = LavaBackend()

    # ── Cold: compile normally, which populates the VkPipelineCache ──
    a = Mantle.LavaArray(zeros(Float32, 64))
    pcnc_warm!(backend, 64)(a; ndrange = 64)
    KA.synchronize(backend)
    @test Array(a) == ones(Float32, 64)

    # ── The instrument must FIRE. A kernel the driver has never compiled has to
    #    be refused; without this the rest of the testset is vacuous. ──
    #
    # `no_pipeline_compilation` does NOT throw, by design — `create_pipeline`
    # records the miss and retries without the flag, so a workload finishes and
    # reports EVERY miss instead of dying at the first. The instrument firing is
    # therefore a recorded miss, not an exception, and this used to assert the
    # exception. That assertion could only ever fail:
    #
    #   * on Linux before 2026-08-02 it failed for a second reason too — the
    #     refusal was discarded entirely (`VK_PIPELINE_COMPILE_REQUIRED` is a
    #     SUCCESS-class code, and the check lived inside the `Sys.iswindows()`
    #     branch), so the counter was stuck at 0 and a NULL pipeline was cached
    #     and later bound. Found on RDNA 3.5, fixed for both.
    #   * with that fixed, the counter is right and only the `refused` assertion
    #     is left failing, which is this test disagreeing with the API rather
    #     than the API misbehaving.
    #
    # Note for whoever runs this on AMD: RADV creates the "novel" pipeline rather
    # than refusing, even with `pipelineCreationCacheControl` enabled and both the
    # VkPipelineCache blob and MESA_SHADER_CACHE_DISABLE ruled out. So the
    # instrument does not fire there at all, and everything below it is vacuous on
    # that device — which is exactly what the paragraph above warns about.
    Mantle.PIPELINE_COMPILES_REFUSED[] = 0
    K = rand(10_000:99_999)                           # never compiled before
    b = Mantle.LavaArray(ones(Float32, 64))
    Mantle.no_pipeline_compilation() do
        pcnc_novel!(backend, 64)(b, Val(K); ndrange = 64)
        KA.synchronize(backend)
    end
    @test Mantle.PIPELINE_COMPILES_REFUSED[] == 1       # instrument fires…
    @test length(Mantle.PIPELINE_COMPILE_MISSES) == 1   # …and names what it caught
    # The retry has to have produced a working pipeline, or "a miss is RECORDED,
    # not fatal" is a lie. `no_pipeline_compilation` does not throw by design, so
    # this is what "the instrument fired" has to mean.
    @test Array(b) ≈ fill(3.0f0 - Float32(K) + sqrt(2.5f0), 64)

    # ── Warm, same session: the cached pipeline needs no compilation ──
    Mantle.PIPELINE_COMPILES_REFUSED[] = 0
    a2 = Mantle.LavaArray(zeros(Float32, 64))
    Mantle.no_pipeline_compilation() do               # throws if it would compile
        pcnc_warm!(backend, 64)(a2; ndrange = 64)
        KA.synchronize(backend)
    end
    @test Array(a2) == ones(Float32, 64)
    @test Mantle.PIPELINE_COMPILES_REFUSED[] == 0

    # ── Across a device reset: the ONLY thing that can satisfy creation now is
    #    the on-disk blob, since the VkPipelineCache is rebuilt from scratch. ──
    ctx = Mantle.vk_context()
    path = Mantle.lava_pipeline_cache_path(ctx.device_name, string(ctx.driver_version))
    Mantle.save_pipeline_cache!(ctx)
    @test isfile(path)
    @test filesize(path) > Mantle.PIPELINE_CACHE_HEADER_BYTES

    Mantle.vk_reset_device!()

    Mantle.PIPELINE_COMPILES_REFUSED[] = 0
    a3 = Mantle.LavaArray(zeros(Float32, 64))
    Mantle.no_pipeline_compilation() do
        pcnc_warm!(LavaBackend(), 64)(a3; ndrange = 64)
        KA.synchronize(LavaBackend())
    end
    @test Array(a3) == ones(Float32, 64)
    @test Mantle.PIPELINE_COMPILES_REFUSED[] == 0     # zero driver compilation
end
