# A pinned subgroup width is honoured, and the kernel observes the width asked for.
#
# This is the precondition for making plan objects carry a subgroup width. On the
# desktop it is untestable: an RTX 4000 Ada reports 32 and can only be 32, so
# "the width the kernel sees" and "the device default" are the same number and
# nothing distinguishes a correct pin from no pin at all. RDNA 3.5 reports a
# DEFAULT of 64 with size control min=32 max=64, so the same kernel can run at
# either width and the two answers are distinguishable.
#
# What is asserted:
#   1. the width the kernel OBSERVES equals the width that was requested;
#   2. a width-INDEPENDENT result is identical at both widths (the pin does not
#      change what the kernel computes);
#   3. a width-DEPENDENT result matches the observed width exactly — a subgroup
#      reduction over 64 lanes sums 64 values at width 64 and 32 at width 32, so
#      this fails if the pin is silently ignored while `subgroup_size()` reports
#      the requested number.
#
# (3) is the one that matters. Without it a driver could report the requested
# width and still schedule the other one, which is precisely the failure a plan
# object storing an unpinned width would produce.
#
# Lava pins to COOPMAT_SUBGROUP (32) exactly when the module declares
# CooperativeMatrixKHR and `device_subgroup_size(ctx) != 32`
# (`pipeline.jl:363`). The kernel below touches a cooperative matrix so the
# capability is declared, and the test drives that gate through the
# `ctx.caches.subgroup_size` field rather than editing the emitter.

using Test, Lava, KernelAbstractions
const KA = KernelAbstractions
const AMsp = Lava.AcceleratedMatrix
const TILEsp = Mantle.GEMM_TILE

const WG_SP = 64          # one wave64, or two wave32 subgroups

# `cm` exists only to make the module declare CooperativeMatrixKHR, which is what
# Lava's auto-pin keys on. Its value is written out so it cannot be optimised away.
@kernel cpu=false unsafe_indices=true function sgp_probe!(sz, lane, red, indep, cm, @Const(x))
    i = Lava.lava_local_invocation_index() + UInt32(1)
    @inbounds begin
        sz[i]   = Lava.subgroup_size()
        lane[i] = Lava.subgroup_lane()
        red[i]  = Lava.subgroup_add(x[i])        # width-DEPENDENT
        indep[i] = x[i] * 2.0f0                  # width-INDEPENDENT
        m = AMsp{Float32,TILEsp,TILEsp,Lava.Accumulator}(pointer(cm), 1, TILEsp)
        cm[i] = Lava.coopmat_getcomp(m, Int32(0))
    end
end

"Drop every cache that can hand back a pipeline built for a different width."
function sgp_clear_caches!()
    # Fields on the context since 28bf2de, not globals keyed by `ctx.id`.
    c = Mantle.vk_context().caches
    empty!(c.pipelines); empty!(c.pipeline_order)
    empty!(c.launchplans); empty!(c.linked)
end

"Run the probe with Lava's auto-pin gate driven to produce `want` lanes."
function sgp_run(be, want::Int)
    ctx = Mantle.vk_context()
    # The gate is `device_subgroup_size(ctx) != COOPMAT_SUBGROUP`. Telling Lava the
    # device is already 32 suppresses the pin, so the module runs at the hardware
    # default; leaving the true value (64) makes the pin fire and request 32.
    # A FIELD on the context since 28bf2de — it was a Ref, then a Dict keyed by
    # ctx.id, and is now `ctx.caches.subgroup_size`. Query through the accessor
    # first so the lazy query has run before the value is overridden.
    # `ctx.caches.subgroup_size` since 28bf2de: the per-device caches are fields
    # on the context now, not globals keyed by `ctx.id`.
    saved = Mantle.device_subgroup_size(ctx)
    ctx.caches.subgroup_size = want == Mantle.COOPMAT_SUBGROUP ? saved : Mantle.COOPMAT_SUBGROUP
    # ALL THREE caches, not just PIPELINE_CACHE. The required subgroup size is part
    # of the pipeline's create-info but NOT part of `get_compute_pipeline`'s cache
    # key, and the KA launch path caches a LaunchPlan that owns a pipeline on top
    # of that. Clearing only PIPELINE_CACHE lets the second width silently reuse
    # the first width's pipeline — which is exactly how this test first "passed"
    # at 32 twice while reporting two different requested widths.
    sgp_clear_caches!()
    try
        x = KA.allocate(be, Float32, WG_SP); copyto!(x, Float32.(1:WG_SP))
        sz    = KA.zeros(be, UInt32, WG_SP)
        lane  = KA.zeros(be, UInt32, WG_SP)
        red   = KA.zeros(be, Float32, WG_SP)
        indep = KA.zeros(be, Float32, WG_SP)
        cm    = KA.zeros(be, Float32, WG_SP)
        sgp_probe!(be, WG_SP)(sz, lane, red, indep, cm, x; ndrange = WG_SP)
        KA.synchronize(be)
        (sz = Int.(Array(sz)), lane = Int.(Array(lane)),
         red = Array(red), indep = Array(indep))
    finally
        ctx.caches.subgroup_size = saved
        sgp_clear_caches!()
    end
end

@testset "pinned subgroup width is honoured" begin
    ctx = Mantle.vk_context()
    c = Mantle.subgroup_size_control(ctx)
    @info "subgroup size control" min=c.min max=c.max compute=c.compute default=Mantle.device_subgroup_size(ctx)

    if !c.compute
        @info "compute pipelines cannot pin a subgroup size here; skipping"
        @test_skip c.compute
    else
        widths = filter(w -> Mantle.can_require_subgroup_size(ctx, w), (32, 64))
        @test length(widths) >= 1
        results = Dict{Int,Any}()
        for w in widths
            r = sgp_run(LavaBackend(), w)
            results[w] = r

            @testset "width $w" begin
                # 1. the kernel observes the width that was requested, on every lane
                @test all(==(w), r.sz)
                # lane ids restart at every subgroup boundary
                @test r.lane == [mod(i - 1, w) for i in 1:WG_SP]
                # 3. the width-dependent reduction really used that many lanes
                want = [sum(Float32, (fld(i - 1, w) * w + 1):(fld(i - 1, w) * w + w))
                        for i in 1:WG_SP]
                @test r.red == want
            end
        end

        # 2. the pin changes the width, not the arithmetic
        if length(widths) == 2
            @test results[32].indep == results[64].indep
            @test results[32].sz != results[64].sz        # the two runs really differed
            # And the reductions must NOT agree, which is what proves the
            # width-dependent assertion above had teeth.
            @test results[32].red != results[64].red
        end
    end
end
