# Cooperative-matrix pipelines and the 32-lane pin: pinned where possible, and
# warned about where the workgroup spans more than one subgroup.
#
# `get_compute_pipeline` pins a module declaring `CooperativeMatrixKHR` to
# `COOPMAT_SUBGROUP` whenever the device default is something else, and errors
# where the device will not accept the pin. Both halves need a test, and only one
# of them is reachable by running the kernel:
#
#   * that the guard does NOT fire here is asserted directly, because on RDNA 3.5
#     `minSubgroupSize == 32` and every coopmat kernel in the suite depends on
#     that being true. If this half ever fails, the whole cooperative-matrix path
#     on this device is running at the wrong width.
#
#   * that the guard DOES fire is otherwise unreachable on any hardware here:
#     it needs a device with cooperative matrices AND a subgroup that cannot be
#     pinned to 32, which neither the discrete GPU nor lavapipe is (lavapipe has
#     8-lane subgroups but no coopmat). Without a positive control the guard is a
#     branch nobody has executed, which is how it would rot. So the capability
#     cache is driven to describe a wave64-only device and the pipeline is asked
#     for again.
#
# Driving the CACHE rather than the kernel is deliberate: the guard reads
# `subgroup_size_control(ctx)` and `device_subgroup_size(ctx)`, both of which are
# per-device `Dict`s keyed by `ctx.id`, so a test can state the device's answer
# without a second device and without touching the emitter.

using Test, Lava, KernelAbstractions
const KA = KernelAbstractions

# Exists only to make the module declare CooperativeMatrixKHR, which is the
# condition `get_compute_pipeline`'s pin (and its refusal) keys on. The value is
# written out so nothing optimises the matrix away.
@kernel cpu=false unsafe_indices=true function cmr_probe!(out)
    i = Lava.lava_local_invocation_index() + UInt32(1)
    @inbounds begin
        m = Lava.AcceleratedMatrix{Float32,Mantle.GEMM_TILE,Mantle.GEMM_TILE,Lava.Accumulator}(
                pointer(out), 1, Mantle.GEMM_TILE)
        out[i] = Lava.coopmat_getcomp(m, Int32(0))
    end
end

@testset "coopmat pipelines and the 32-lane pin" begin
    ctx = Mantle.vk_context()

    if !Mantle.coopmat_gemm_available(ctx)
        @info "no cooperative matrices on this device; the guard is unreachable"
        @test_skip Mantle.coopmat_gemm_available(ctx)
    else
        # ── the half that must hold on this machine ──────────────────────────
        @test Mantle.can_require_subgroup_size(ctx, Mantle.COOPMAT_SUBGROUP)

        # A real coopmat kernel builds and computes correctly at the pinned width.
        # `mul!` on fp16 operands with an fp32 destination is the shipped route
        # onto `coopmat_gemm!`.
        M = N = K = 128
        A = Mantle.LavaArray(Float16.(randn(Float32, M, K) .* 0.1f0))
        B = Mantle.LavaArray(Float16.(randn(Float32, K, N) .* 0.1f0))
        C = Mantle.LavaArray(zeros(Float32, M, N))
        ref = Float32.(Array(A)) * Float32.(Array(B))
        Mantle.LinearAlgebra.mul!(C, A, B)
        KA.synchronize(LavaBackend())
        got = Array(C)
        @test maximum(abs, got) > 1e-3                       # it wrote something
        @test maximum(abs, got .- ref) / maximum(abs, ref) < 5e-3

        # ── the positive control ─────────────────────────────────────────────
        # Describe a device whose subgroups are 64 and cannot be narrowed, which
        # is what an RDNA part without `VK_EXT_subgroup_size_control` looks like.
        #
        # This asserts a WARNING, not a refusal, and the difference was earned by
        # 328827f: refusing outright rejected a VALID 8-lane cooperative-matrix
        # kernel on lavapipe, which turns out to be a second, independent coopmat
        # consumer. A workgroup holding ONE subgroup is correct at any width —
        # every lane computes `tid ÷ COOPMAT_SUBGROUP == 0` — so only a workgroup
        # spanning more than one is dangerous. The gate that protects correctness
        # is the capability query (`coopmat_gemm_available`, `flashcmfits`); this
        # is the backstop for a hand-written kernel that bypasses them.
        saved_ctl  = ctx.caches.subgroup_control
        saved_size = Mantle.device_subgroup_size(ctx)
        saved_warn = ctx.caches.coopmat_warned
        try
            ctx.caches.subgroup_control = Mantle.SubgroupSizeControl(64, 64, true)
            ctx.caches.subgroup_size    = 64
            ctx.caches.coopmat_warned   = false     # one warning per device
            @test !Mantle.can_require_subgroup_size(ctx, Mantle.COOPMAT_SUBGROUP)

            let c = ctx.caches
                empty!(c.pipelines); empty!(c.pipeline_order)
                empty!(c.launchplans); empty!(c.linked)
                # `frozen_mem` TOO: `get_compiled_kernel_and_pipeline` consults it
                # first and RETURNS on a hit, so a frozen kernel never reaches
                # `get_compute_pipeline` and the guard cannot fire at all.
                empty!(c.frozen_mem); empty!(c.iterplans)
            end

            # 128 threads against a faked 64-lane subgroup: two subgroups, which
            # is the shape the warning exists for.
            out2 = KA.zeros(LavaBackend(), Float32, 128)
            @test_logs (:warn, r"cannot pin them to"i) match_mode = :any begin
                cmr_probe!(LavaBackend(), 128)(out2; ndrange = 128)
                KA.synchronize(LavaBackend())
            end
            @test ctx.caches.coopmat_warned          # and it latched, once
        finally
            ctx.caches.subgroup_control = saved_ctl
            ctx.caches.subgroup_size    = saved_size
            ctx.caches.coopmat_warned   = saved_warn
            let c = ctx.caches      # fields on the context since 28bf2de
                # `frozen_mem` and `iterplans` TOO. `get_compiled_kernel_and_pipeline`
                # consults the frozen memo first and RETURNS on a hit, so a cached
                # frozen kernel never reaches `get_compute_pipeline` — and the
                # refusal under test lives there. Leaving it populated made this
                # positive control pass vacuously: no error, because no pipeline
                # was ever built.
                empty!(c.pipelines); empty!(c.pipeline_order)
                empty!(c.launchplans); empty!(c.linked)
                empty!(c.frozen_mem); empty!(c.iterplans)
            end
        end
    end
end
