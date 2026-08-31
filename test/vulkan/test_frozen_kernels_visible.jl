# A frozen kernel is visible to the profiler, and distinguishable from the rest.
#
# `get_compiled_kernel_and_pipeline` consults the frozen cache FIRST and returns
# on a hit, so a kernel that came off disk never reaches
# `GPUCompiler.cached_compilation` and never enters `ctx.caches.linked`. Every
# shipped runner calls `use_frozen_kernels` in its `__init__`, which makes
# `caches.frozen_mem` the only cache in play for the configuration that ships —
# and `list_compiled_kernels` walked `caches.linked` alone.
#
# Measured on a Depth Anything forward before this: **0 kernels reported against
# 45 live dispatch names**, taking `kernel_stats`, `pipeline_exec_stats` and
# every register/scratch number with it. A profiler that is blind precisely where
# it is needed reports an absence that reads as a fact about the hardware — which
# is how "AMD reports no pipeline statistics" got written down when RADV was in
# fact returning 26 statistics per pipeline.
#
# The second assertion is about NAMES. `KernelStats.name` is the SPIR-V entry
# point, and Lava calls every compute entry `main`, so a list of stats was a list
# of indistinguishable `"main"`s with nothing to attribute or join a timing table
# on. `source` carries the Julia kernel: recovered from the mangled symbol for a
# compiled kernel, and from the cache KEY for a frozen one, since `frozen_store`
# deliberately writes `ir = ""` ("session-specific and large").
#
# NOT tested here: that `frozen_mem` is per device. That is `ctx.caches.frozen_mem`,
# a field on the context, and `twodevice_probe.jl` owns it — a field cannot be
# shared between two contexts by construction, so there is nothing left to assert.

using Test, Lava, KernelAbstractions
const KA = KernelAbstractions

@kernel function fcpd_scale!(out, @Const(x), s)
    i = @index(Global)
    @inbounds out[i] = x[i] * s
end

@testset "frozen kernels are visible to the profiler" begin
    ctx = MVE.vk_context()
    saved_version = Lava.FROZEN_VERSION[]
    saved_recording = Lava.FROZEN_RECORDING[]
    try
        # A version of our own, so this cannot read or write the real cache.
        Lava.FROZEN_VERSION[] = "testvis_" * string(ctx.id; base = 16)
        Lava.FROZEN_RECORDING[] = true
        empty!(ctx.caches.frozen_mem)

        back = LavaBackend()
        x = MVE.LavaArray(collect(Float32, 1:256))
        out = MVE.LavaArray(zeros(Float32, 256))
        fcpd_scale!(back, 64)(out, x, 2.0f0; ndrange = 256)
        KA.synchronize(back)
        @test Array(out) == Float32.(2 .* (1:256))

        # The first launch COMPILES (a frozen miss) and `frozen_store` writes the
        # SPIR-V; only a later `frozen_load` disk HIT populates the in-memory
        # memo. So the second launch is what puts anything in `frozen_mem`, and
        # that asymmetry is worth pinning rather than asserting away.
        fcpd_scale!(back, 64)(out, x, 3.0f0; ndrange = 256)
        KA.synchronize(back)
        @test Array(out) == Float32.(3 .* (1:256))

        ks = MVE.list_compiled_kernels(ctx)
        @test !isempty(ks)
        @test all(k -> k isa MVE.KernelStats, ks)
        @test all(k -> k.spirv.bytes > 0 && k.spirv.n_instructions > 0, ks)

        # `.name` cannot tell two kernels apart; `.source` must.
        @test all(k -> k.name == "main", ks)
        @test any(k -> occursin("fcpd_scale", k.source), ks)

        # And every kernel in the frozen memo is reported, which is the half that
        # was missing: with `caches.linked` empty this returned nothing at all.
        for (_, linked) in ctx.caches.frozen_mem
            linked isa MVE.LavaLinkedKernel || continue
            @test any(k -> k.spirv.bytes == length(linked.compiled.spirv_bytes), ks)
        end
    finally
        Lava.FROZEN_RECORDING[] = saved_recording
        MVE.frozen_clear!()
        Lava.FROZEN_VERSION[] = saved_version
        empty!(ctx.caches.frozen_mem)
    end
end
