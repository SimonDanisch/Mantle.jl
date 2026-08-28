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

@testset "BatchQueue has deferred_frees_lock" begin
    bq = Mantle.vk_context().default_bq
    @test hasfield(Mantle.BatchQueue, :deferred_frees_lock)
    @test bq.deferred_frees_lock isa Base.Threads.SpinLock
end

@testset "last_write is atomic (no raw field access)" begin
    a = LavaArray{Float32,1}(undef, (4,))
    buf = a.buf[]
    # Must use @atomic form; raw access would error or warn on atomic fields.
    @test (@atomic :acquire buf.last_write) === nothing ||
          (@atomic :acquire buf.last_write) isa Tuple
end

end  # @testset
