# What a submission holds, and when it lets go — phase 2.2 of
# `docs/mantle-owns-it.md`, from the outside.
#
# This was `test_phase1_pin.jl`, which asserted the mechanism: an `IdSet` of
# pinned objects on the closed command buffer, a `pinned_refs` list of retained
# `DataRef`s beside it, and `sync_access!` stamping `last_write_bq` at submit.
# All three were the backend deciding a lifetime, and all three are gone. The
# PROPERTY they existed for is the same and is what this file pins:
#
#   * while a submission that names an array is in flight, the buffer is held
#     and cannot be destroyed;
#   * once the device has passed it, the hold goes;
#   * the submission that ran is recorded on the buffer, which is what a
#     cross-channel wait and a deferred destroy read.

using Test, Lava, Mantle
using KernelAbstractions
import Adapt

# Top level, because a `struct` cannot be defined inside a `@testset` body.
struct TestWrapper{A, B}
    items::A
    sizes::B
end

@kernel function touchkernel!(d)
    i = @index(Global)
    @inbounds d[i] = d[i] + 1.0f0
end

# How many recordings that can still submit hold this array's buffer.
holders(a) = @atomic Mantle.stampof(a).holders

@testset "hold! — what a submission holds, and when it lets go" begin

@testset "the deleted mechanism is gone" begin
    # Not "renamed": these named a decision the backend was making.
    @test !isdefined(Mantle, :pin!)
    @test !isdefined(Mantle, :sync_access!)
    @test !isdefined(Mantle, :pin_leaves!)
    @test !isdefined(Mantle, :collectsync!)
    @test !hasfield(Mantle.OneShot, :pinned)
    @test !hasfield(Mantle.OneShot, :pinned_refs)
    @test !hasfield(Mantle.VkManagedBuffer, :last_write_bq)
    @test !hasfield(Mantle.VkManagedBuffer, :pins)

    # What replaced it: core keeps the lists, the backend keeps the facts.
    @test hasfield(Mantle.OneShot, :sync)          # the buffers these commands name
    @test hasfield(Mantle.Recording, :holds)       # a recording outlives its submissions
    @test hasfield(Mantle.VkManagedBuffer, :stamp)
    @test fieldnames(Mantle.Stamp) == (:channel, :token, :holders)
    for f in (:outstanding, :free, :holds, :spare, :pending, :retiring)
        @test hasfield(Mantle.SubmitChannel, f)
    end

    # LavaAdaptor still carries the owner (no zero-arg constructor, but a
    # `nothing` owner is allowed for a pure strip).
    @test fieldnames(Mantle.LavaAdaptor) == (:batch,)
    @test_throws MethodError Mantle.LavaAdaptor()
end

@testset "a buffer is recorded once however often it is held" begin
    bq = Mantle.batchqueue(Mantle.Device())
    a = LavaArray{Float32,1}(zeros(Float32, 8))

    Mantle.oneshot!(bq) do e
        @test e.owner.bq === bq
        for _ in 1:5
            Mantle.hold!(e, a)
        end
        # The hold list takes each one — they are references, and balanced by
        # `unhold!` — but the ordering list is deduplicated by identity, because
        # every entry in it costs a `crosswaits!` and a `stamp!` per run.
        @test length(e.owner.sync) == 1
        @test e.owner.sync[1] === a.buf[]
        @test holders(a) == 5
    end
    Mantle.flush!(bq)
    @test holders(a) == 0
end

@testset "the submission that ran is stamped on the buffer" begin
    bq = Mantle.batchqueue(Mantle.Device())
    a = LavaArray{Float32,1}(zeros(Float32, 4))

    Mantle.oneshot!(bq) do e
        Mantle.hold!(e, a)
    end
    Mantle.flush!(bq)
    st = Mantle.stampof(a)
    @test st.channel === bq
    @test st.token == Mantle.driver(bq).next_timeline

    # Nothing without device-visible bytes of its own is tracked, and asking is
    # not an error: that is what makes `hold!` usable for a pipeline.
    @test Mantle.stampof("not a buffer") === nothing
end

@testset "a launch holds its arrays for the life of the submission" begin
    # The invariant that matters, independent of which layer performs it: while
    # a submission that names an array is in flight the buffer is held, and once
    # it has passed the hold is released. This is what stops GPUArrays
    # refcounting from freeing a buffer out from under in-flight GPU work.
    be = LavaBackend()
    a = LavaArray{Float32,1}(zeros(Float32, 16))
    KernelAbstractions.synchronize(be)
    @test holders(a) == 0

    bq = Mantle.batchqueue(Mantle.Device())
    touchkernel!(be, 16)(a; ndrange = 16)
    @test holders(a) > 0                        # held while in flight
    # And the one-shot that carried it names the buffer, which is what the
    # ordering between channels is derived from.
    @test a.buf[] in last(bq.outstanding).payload.recording.sync
    KernelAbstractions.synchronize(be)
    @test holders(a) == 0                       # released once it has passed
end

@testset "the adaptor strips and holds nothing" begin
    bq = Mantle.batchqueue(Mantle.Device())
    a = LavaArray{Float32,1}(zeros(Float32, 8))

    Mantle.oneshot!(bq) do e
        adaptor = Mantle.LavaAdaptor(e.owner)
        # `adapt` strips LavaArray → LavaDeviceArray and takes no claim: the
        # launch path holds explicitly, which the testset above checks against
        # the observable guarantee rather than against where the call sits.
        dev_a = Adapt.adapt(adaptor, a)
        @test dev_a isa Lava.LavaDeviceArray
        @test holders(a) == 0
    end
    Mantle.flush!(bq)
end

@testset "wrapper struct: the adaptor leaves an unregistered struct alone" begin
    bq = Mantle.batchqueue(Mantle.Device())
    a = LavaArray{Float32,1}(zeros(Float32, 4))
    b = LavaArray{Int32,1}(zeros(Int32, 4))
    w = TestWrapper(a, b)

    Mantle.oneshot!(bq) do e
        adaptor = Mantle.LavaAdaptor(e.owner)
        # A struct with no `Adapt.@adapt_structure` rule is returned untouched —
        # the adaptor does not recurse into arbitrary user types. Anything
        # relying on nested arrays reaching the device must either register the
        # type with Adapt or pass the arrays as kernel arguments, where the
        # launch path holds them.
        wc = Adapt.adapt(adaptor, w)
        @test wc isa TestWrapper
        @test wc.items === a
        @test wc.sizes === b
    end
    Mantle.flush!(bq)
end

end  # @testset
