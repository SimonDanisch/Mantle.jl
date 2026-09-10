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

    # Warm the DESTROY path as well, and this is not padding: a collection
    # inside the measured window runs finalizers, a finalizer retires a buffer,
    # and the first `retire!`/`reclaim!`/`rawfree` of a session compiles inside
    # the count. Measured first-in-session with only the two warmups above:
    # 753 B/dispatch, against 223 with this line — and the number moved with
    # whenever the GC happened to run, which is what made it look like a
    # regression that came and went.
    let scratch = KA.allocate(backend, Float32, 64)
        Mantle.unsafe_free!(scratch)
    end
    GC.gc(true)
    Mantle.drain!(Mantle.batchqueue(Mantle.Device(Mantle.VulkanAPI())))
    run(50); KA.synchronize(backend)

    n = 500
    # TWO windows, and the first is thrown away: what is being measured is the
    # steady state of the dispatch path, and the first window after any warmup
    # still compiles something — a collection inside it runs a finalizer, the
    # finalizer retires a buffer, and the destroy path is inferred and compiled
    # inside the count. Measured first-in-session: 674 B for the first window,
    # 243 for every one after it, on identical code.
    @allocated run(n); KA.synchronize(backend)
    bytes = @allocated run(n)
    KA.synchronize(backend)
    per = bytes / n

    # The ceiling is deliberately loose. It is a CLIFF detector: 781 -> 115 was
    # four separate inference failures, and any one of them coming back blows
    # through this by a wide margin.
    #
    # It was 250, and it moved because the machinery it was calibrated against is
    # gone. An unmodelled launch used to take its argument bytes from a
    # bump-pointer add into a slab ring on the queue; it takes a `Region` from the
    # pool now, and `Pool.acquire!` allocates 96 bytes doing it. Measured back to
    # back on this device, same harness: **162.8 B/dispatch** with the slab ring
    # and **258.9** with the pool, and `--track-allocation` puts the whole 96 on
    # two lines of `pool.jl` — `compatible(dev, blk.constraint, want)` and
    # `takespan!`.
    #
    # That is the trade `docs/submission-refactor.md` step 1 authorises and
    # measures: five fields of hand-rolled state, and a rewind that was a live
    # bug, for one allocator call on the path step 5 deletes. So the number is
    # recorded here rather than the test being left red or the ceiling quietly
    # widened — it is a design decision that has been made, not an inference
    # failure that has crept in.
    @test per < 400

    # A `compiled` kernel resolves its plan and queue into concrete fields, so the
    # launch is a static call. It must not be WORSE than the generic path — an
    # earlier version of it was 3x worse, because a concrete `isbits` plan handed
    # to a still-dynamic call gets boxed whole.
    held = MVE.compiled(allocprobe!(backend), 1024)
    heldrun(n) = for _ in 1:n; held(a); end
    heldrun(200); KA.synchronize(backend)
    @allocated heldrun(n); KA.synchronize(backend)   # same first-window rule
    heldbytes = @allocated heldrun(n)
    KA.synchronize(backend)
    @test heldbytes / n <= per
end
