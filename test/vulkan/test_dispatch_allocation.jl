using Test, Lava, KernelAbstractions
const KA = KernelAbstractions

# Launching a fully compiled kernel on a warm cache constructs nothing: two cache
# hits, a bump allocation, a memcpy and one Vulkan call. It allocated **781 bytes
# per dispatch**, all of it inference falling over, and nothing failed — every
# suite stayed green while every model paid it.
#
# What the 781 was, and what each fix was worth (measured with
# `--track-allocation=user`, compilation cleared, 1000 dispatches):
#
#     baseline                                          781
#     `Vector{Any}` for the launch-plan cache           739   immutable-with-refs
#                                                             boxes on inline read
#     `bq.ctx::VkContext` in `launch_plan`              275   untyped ctx made the
#                                                             lookup AND its loop
#                                                             dynamic: -464 alone
#     function barrier on the abstract `IterPlan`       115   dynamic getfields +
#                                                             uninferrable tuple
#     `pin!` by `===` instead of `in`                    67   generic `==` over a
#                                                             Vector{Any}
#     gating the submit-time debug `String`              ~59
#
# This test exists because none of that was visible. A number is the only thing
# that catches it: every one of those regressions would have passed a correctness
# suite unchanged.
@testset "dispatch allocation" begin
    @kernel function allocprobe!(a)
        i = @index(Global)
        @inbounds a[i] = a[i] + 1f0
    end

    backend = LavaBackend()
    a = KA.allocate(backend, Float32, 1024)
    fill!(a, 0f0)

    run(n) = for _ in 1:n
        allocprobe!(backend)(a; ndrange = 1024)
    end

    run(200); KA.synchronize(backend)          # warm: compile, plan, pipeline, slab
    run(200); KA.synchronize(backend)          # settle the arg-slab pool

    n = 500
    bytes = @allocated run(n)
    KA.synchronize(backend)
    per = bytes / n

    # The ceiling is deliberately loose (measured ~59 B/dispatch here, and the
    # remaining bytes are amortised submit-time work that scales with batch size,
    # not per dispatch). It is a CLIFF detector: 781 -> 115 was four separate
    # inference failures, and any one of them coming back blows through 250.
    @test per < 250

    # A `compiled` kernel resolves its plan and queue into concrete fields, so the
    # launch is a static call. It must not be WORSE than the generic path — an
    # earlier version of it was 3x worse, because a concrete `isbits` plan handed
    # to a still-dynamic call gets boxed whole.
    held = MVE.compiled(allocprobe!(backend), 1024)
    heldrun(n) = for _ in 1:n; held(a); end
    heldrun(200); KA.synchronize(backend)
    heldbytes = @allocated heldrun(n)
    KA.synchronize(backend)
    @test heldbytes / n <= per
end
