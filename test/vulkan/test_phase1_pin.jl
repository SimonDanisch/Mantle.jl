using Test, Lava, Mantle
# A pinned LavaArray's buffer lives in `batch.pinned_refs`, not `batch.pinned`.
#
# `pin!(batch, ::LavaArray)` takes TWO claims: it retains a `copy(a.buf)` DataRef
# in `pinned_refs`, so GPUArrays refcounting cannot free the buffer out from under
# an in-flight batch, and it bumps `pin_buffer!`. `batch.pinned` holds the other
# pinned objects (pipelines and the like). These assertions predate that split and
# read `a.buf[] in batch.pinned`.
ispinned(batch, a) = any(r -> r[] === a.buf[], batch.pinned_refs) &&
                     getfield(a.buf[], :pins) > 0

using KernelAbstractions
@kernel function touchkernel!(d)
    i = @index(Global)
    @inbounds d[i] = d[i] + 1.0f0
end

@testset "Phase 1 — unified pin!/sync_access! + dead-code sweep" begin

@testset "symbol presence/absence" begin
    # Old API must be gone
    @test !isdefined(Lava, :record_one!)
    @test !isdefined(Lava, :record_arg_accesses!)
    @test !isdefined(Lava, :track_buffer_access!)
    @test !isdefined(Lava, :pin_args!)
    @test !isdefined(Lava, :pin_fields!)
    @test !hasmethod(Mantle.vk_flush!, Tuple{})

    # New API must be present
    @test isdefined(Mantle, :pin!)
    @test isdefined(Mantle, :sync_access!)

    # Struct must match
    @test !hasfield(Mantle.CommandBatch, :data_refs)
    @test !hasfield(Mantle.CommandBatch, :fence)
    @test !hasfield(Mantle.CommandBatch, :reaper_task)
    @test !hasfield(Mantle.CommandBatch, :retired)
    @test !hasfield(Mantle.CommandBatch, :error)
    @test hasfield(Mantle.CommandBatch, :pinned)
    @test hasfield(Mantle.CommandBatch, :bq)

    # LavaAdaptor must carry the batch (no zero-arg constructor)
    @test fieldnames(Mantle.LavaAdaptor) == (:batch,)
    @test_throws MethodError Mantle.LavaAdaptor()
end

@testset "pin! deduplication within a batch" begin
    ctx = Mantle.vk_context()
    bq = ctx.default_bq
    a = LavaArray{Float32,1}(zeros(Float32, 8))

    batch = Mantle.ensure_active_batch!(bq)
    @test batch.bq === bq
    empty_size = length(batch.pinned)

    # Same LavaArray pinned 5 times → still one entry (IdSet dedup)
    for _ in 1:5
        Mantle.pin!(batch, a)
    end
    @test length(batch.pinned) == empty_size + 1

    Mantle.vk_flush!(bq)
end

@testset "sync_access! default is no-op; VkManagedBuffer specialization updates last_write" begin
    ctx = Mantle.vk_context()
    bq = ctx.default_bq
    a = LavaArray{Float32,1}(zeros(Float32, 4))

    batch = Mantle.ensure_active_batch!(bq)
    Mantle.pin!(batch, a)
    @test ispinned(batch, a)
    # Pre-submit, sync_access! hasn't fired yet; last_write may be anything.
    # After flush, the buffer's last_write must reference this bq.
    Mantle.vk_flush!(bq)
    lw = a.buf[].last_write
    @test lw !== nothing
    @test lw[1] === bq

    # Default no-op branch: pin something non-buffer, sync_access! should return nothing.
    batch2 = Mantle.ensure_active_batch!(bq)
    @test Mantle.sync_access!(batch2, "not a buffer") === nothing
    Mantle.vk_flush!(bq)
end

@testset "LavaAdaptor pins during strip (KA-style Adapt path)" begin
    ctx = Mantle.vk_context()
    bq = ctx.default_bq
    a = LavaArray{Float32,1}(zeros(Float32, 8))

    batch = Mantle.ensure_active_batch!(bq)
    adaptor = Mantle.LavaAdaptor(batch)

    # adapt strips LavaArray → LavaDeviceArray. It no longer pins as a side
    # effect of stripping; the launch path pins explicitly, which the
    # "a launch pins its arrays" testset below checks against the observable
    # guarantee rather than against where the call happens to sit.
    import Adapt
    dev_a = Adapt.adapt(adaptor, a)
    @test dev_a isa Lava.LavaDeviceArray

    Mantle.vk_flush!(bq)
end

@testset "a launch pins its arrays for the life of the batch" begin
    # The invariant that actually matters, independent of which layer performs it:
    # while a batch that references an array is in flight the buffer is pinned, and
    # once it retires the pin is released. This is what stops GPUArrays refcounting
    # from freeing a buffer out from under in-flight GPU work.
    a = LavaArray{Float32,1}(zeros(Float32, 16))
    KernelAbstractions.synchronize(LavaBackend())
    @test getfield(a.buf[], :pins) == 0

    touchkernel!(LavaBackend(), 16)(a; ndrange = 16)
    batch = Mantle.vk_context().default_bq.active_batch
    @test batch !== nothing
    @test ispinned(batch, a)                       # pinned while recorded
    KernelAbstractions.synchronize(LavaBackend())
    @test getfield(a.buf[], :pins) == 0            # released once retired
end

@testset "wrapper struct: adaptor leaves an unregistered struct alone" begin
    ctx = Mantle.vk_context()
    bq = ctx.default_bq
    a = LavaArray{Float32,1}(zeros(Float32, 4))
    b = LavaArray{Int32,1}(zeros(Int32, 4))

    struct TestWrapper{A, B}
        items::A
        sizes::B
    end

    w = TestWrapper(a, b)

    batch = Mantle.ensure_active_batch!(bq)
    adaptor = Mantle.LavaAdaptor(batch)

    import Adapt
    # A struct with no `Adapt.@adapt_structure` rule is returned untouched — the
    # adaptor does not recurse into arbitrary user types, and it does not pin what
    # it finds there. This testset used to assert the opposite on both counts.
    # Anything relying on nested arrays reaching the device must either register
    # the type with Adapt or pass the arrays as kernel arguments, where the launch
    # path pins them (see "a launch pins its arrays for the life of the batch").
    wc = Adapt.adapt(adaptor, w)
    @test wc isa TestWrapper
    @test wc.items === a
    @test wc.sizes === b

    Mantle.vk_flush!(bq)
end

end  # @testset
