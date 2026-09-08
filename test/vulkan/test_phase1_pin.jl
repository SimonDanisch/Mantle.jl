using Test, Lava, Mantle
# A pinned LavaArray's buffer lives in `owner.pinned_refs`, not `owner.pinned`.
#
# `pin!(owner, ::LavaArray)` takes TWO claims: it retains a `copy(a.buf)` DataRef
# in `pinned_refs`, so GPUArrays refcounting cannot free the buffer out from under
# an in-flight one-shot or recording, and it bumps `pin_buffer!`. `owner.pinned`
# holds the other pinned objects (pipelines and the like). These assertions predate
# that split and read `a.buf[] in owner.pinned`.
ispinned(owner, a) = any(r -> r[] === a.buf[], owner.pinned_refs) &&
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
    @test !hasmethod(MVE.vk_flush!, Tuple{})

    # New API must be present
    @test isdefined(MVE, :pin!)
    @test isdefined(MVE, :sync_access!)

    # Struct must match
    @test !hasfield(MVE.OneShot, :data_refs)
    @test !hasfield(MVE.OneShot, :fence)
    @test !hasfield(MVE.OneShot, :reaper_task)
    @test !hasfield(MVE.OneShot, :retired)
    @test !hasfield(MVE.OneShot, :error)
    @test hasfield(MVE.OneShot, :pinned)
    @test hasfield(MVE.OneShot, :bq)
    @test hasfield(MVE.Recording, :bq)

    # LavaAdaptor must carry the owner (no zero-arg constructor, but a nothing
    # owner is allowed for a pure strip)
    @test fieldnames(MVE.LavaAdaptor) == (:batch,)
    @test_throws MethodError MVE.LavaAdaptor()
end

@testset "pin! deduplication within a one-shot" begin
    ctx = MVE.vk_context()
    bq = ctx.default_bq
    a = LavaArray{Float32,1}(zeros(Float32, 8))

    MVE.oneshot!(bq) do e
        @test e.owner.bq === bq
        empty_size = length(e.owner.pinned)

        # Same LavaArray pinned 5 times → still one entry (IdSet dedup)
        for _ in 1:5
            MVE.pin!(e.owner, a)
        end
        @test length(e.owner.pinned) == empty_size + 1
    end

    MVE.vk_flush!(bq)
end

@testset "sync_access! default is no-op; VkManagedBuffer specialization updates last_write" begin
    ctx = MVE.vk_context()
    bq = ctx.default_bq
    a = LavaArray{Float32,1}(zeros(Float32, 4))

    MVE.oneshot!(bq) do e
        MVE.pin!(e.owner, a)
        @test ispinned(e.owner, a)
    end
    # Pre-submit, sync_access! hasn't fired yet; last_write may be anything.
    # After flush, the buffer's last_write must reference this bq.
    MVE.vk_flush!(bq)
    buf = a.buf[]
    @test buf.last_write_bq === bq
    @test buf.last_write_val == bq.next_timeline

    # Default no-op branch: pin something non-buffer, sync_access! should return nothing.
    @test MVE.sync_access!(MVE.Submission(bq), "not a buffer") === nothing
    MVE.vk_flush!(bq)
end

@testset "LavaAdaptor pins during strip (KA-style Adapt path)" begin
    ctx = MVE.vk_context()
    bq = ctx.default_bq
    a = LavaArray{Float32,1}(zeros(Float32, 8))

    import Adapt
    MVE.oneshot!(bq) do e
        adaptor = MVE.LavaAdaptor(e.owner)

        # adapt strips LavaArray → LavaDeviceArray. It no longer pins as a side
        # effect of stripping; the launch path pins explicitly, which the
        # "a launch pins its arrays" testset below checks against the observable
        # guarantee rather than against where the call happens to sit.
        dev_a = Adapt.adapt(adaptor, a)
        @test dev_a isa Lava.LavaDeviceArray
    end

    MVE.vk_flush!(bq)
end

@testset "a launch pins its arrays for the life of the submission" begin
    # The invariant that actually matters, independent of which layer performs it:
    # while a submission that references an array is in flight the buffer is
    # pinned, and once it retires the pin is released. This is what stops
    # GPUArrays refcounting from freeing a buffer out from under in-flight GPU work.
    a = LavaArray{Float32,1}(zeros(Float32, 16))
    KernelAbstractions.synchronize(LavaBackend())
    @test getfield(a.buf[], :pins) == 0

    touchkernel!(LavaBackend(), 16)(a; ndrange = 16)
    sub = last(MVE.vk_context().default_bq.outstanding).payload
    @test ispinned(sub.oneshots[1], a)              # pinned while in flight
    KernelAbstractions.synchronize(LavaBackend())
    @test getfield(a.buf[], :pins) == 0            # released once retired
end

@testset "wrapper struct: adaptor leaves an unregistered struct alone" begin
    ctx = MVE.vk_context()
    bq = ctx.default_bq
    a = LavaArray{Float32,1}(zeros(Float32, 4))
    b = LavaArray{Int32,1}(zeros(Int32, 4))

    struct TestWrapper{A, B}
        items::A
        sizes::B
    end

    w = TestWrapper(a, b)

    import Adapt
    MVE.oneshot!(bq) do e
        adaptor = MVE.LavaAdaptor(e.owner)

        # A struct with no `Adapt.@adapt_structure` rule is returned untouched — the
        # adaptor does not recurse into arbitrary user types, and it does not pin what
        # it finds there. This testset used to assert the opposite on both counts.
        # Anything relying on nested arrays reaching the device must either register
        # the type with Adapt or pass the arrays as kernel arguments, where the launch
        # path pins them (see "a launch pins its arrays for the life of the submission").
        wc = Adapt.adapt(adaptor, w)
        @test wc isa TestWrapper
        @test wc.items === a
        @test wc.sizes === b
    end

    MVE.vk_flush!(bq)
end

end  # @testset
