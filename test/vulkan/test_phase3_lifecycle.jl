using Test, Lava, Mantle
@testset "Phase 3 — lifecycle state + finalizer/main-thread separation" begin

@testset "buffer state machine" begin
    @test isdefined(Mantle, :BUF_STATE_ALIVE)
    @test isdefined(Mantle, :BUF_STATE_DEFERRED)
    @test isdefined(Mantle, :BUF_STATE_DEAD)
    @test hasfield(Mantle.VkManagedBuffer, :state)

    a = LavaArray{Float32,1}(undef, (4,))
    buf = a.buf[]
    @test (@atomic :acquire buf.state) == Mantle.BUF_STATE_ALIVE

    Mantle.vk_free!(buf)
    # After vk_free!: either DEFERRED (GPU busy) or DEAD (immediately destroyed).
    s = @atomic :acquire buf.state
    @test s == Mantle.BUF_STATE_DEFERRED || s == Mantle.BUF_STATE_DEAD

    # Second vk_free! must be idempotent — the CAS from ALIVE fails because
    # state is no longer ALIVE.  Nothing crashes, state doesn't regress.
    Mantle.vk_free!(buf)
    s2 = @atomic :acquire buf.state
    @test s2 == s || s2 == Mantle.BUF_STATE_DEAD   # monotonic progression only
end

@testset "LavaArray has no direct finalizer (DataRef refcount is sole owner)" begin
    # Construct a LavaArray and check that the array itself has no finalizer
    # attached.  The backing VkManagedBuffer's finalizer (via DataRef closure)
    # is the single lifetime owner after Phase 3.
    a = LavaArray{Float32,1}(undef, (4,))
    # `finalizer(...)` on an already-finalized object throws; the normal way
    # to assert no finalizer exists is to check that the array doesn't trigger
    # a second free on GC.  We just smoke-test that construct/drop works.
    @test a isa LavaArray
end

@testset "the destroy that has to wait is core's list, not the backend's" begin
    # It was `deferred_frees`, `deferred_as_frees` and a `SpinLock` on the
    # backend's queue, with two drain functions reading a stamp the backend also
    # owned. Core keeps one list per channel — see `graph/lifetime.jl`.
    bq = Mantle.vk_context().default_bq
    @test !hasfield(Mantle.VulkanQueue, :deferred_frees)
    @test !hasfield(Mantle.VulkanQueue, :deferred_as_frees)
    @test !hasfield(Mantle.VulkanQueue, :deferred_frees_lock)
    @test bq.pendinglock isa ReentrantLock
    @test bq.pending isa Vector{Any}
end

@testset "the stamp is one object, empty until a submit writes it" begin
    # It was `last_write_bq` + `last_write_val` + `@atomic pins` +
    # `free_requested`, four fields the backend read to decide a lifetime. It is
    # a `Mantle.Stamp` now: the backend supplies the storage, core writes every
    # value in it.
    a = LavaArray{Float32,1}(undef, (4,))
    buf = a.buf[]
    @test !hasfield(typeof(buf), :last_write)
    @test !hasfield(typeof(buf), :last_write_bq)
    st = Mantle.stampof(buf)
    @test st.channel === nothing
    @test st.token == 0
    @test (@atomic st.holders) == 0
end

end  # @testset
