# A tolerated allocation failure must not leave validation messages behind.
#
# `try_vk_alloc` exists to attempt an allocation that may legitimately fail, and
# it returns an `AllocFailure` rather than throwing when the device says no. With
# `LAVA_VALIDATION=1` that refusal also produces validation-layer messages
# (`VkBufferCreateInfo-size-06409`, `vkAllocateMemory-pAllocateInfo-01713`), and
# those are the caller's to absorb — the failure was expected and handled.
#
# It absorbed them incorrectly. The callback writes into a lock-free ring
# (`ctx.validation`, per device since 49f3f17); its `.messages` list is only
# populated when `drain_validation_messages!` moves entries out. `try_vk_alloc`
# emptied the drained list WITHOUT draining first, so its own messages stayed in
# the ring and the next `check_validation_errors!` — belonging to entirely
# unrelated code — drained them and threw.
#
# Observed exactly that way: `test_source_mapping.jl:699` deliberately asks for
# 40 GB, and the failure surfaced 40 lines later at `:739`, in
# "Source mapping doesn't break kernel execution", as a `LavaError during
# vk_flush!` on a four-element upload. The test that reported it was innocent,
# which is the part worth pinning: a misattributed validation error sends
# whoever reads the log to the wrong file.
#
# This test is a no-op unless validation is on — without the layer there are no
# messages to leak, and asserting on their absence would pass for the wrong
# reason. That is stated rather than silently skipped.

using Test, Lava, KernelAbstractions
const KA = KernelAbstractions

@testset "a tolerated alloc failure leaves no validation messages" begin
    ctx = MVE.vk_context()

    # `LAVA_VALIDATION` is read once, when the Vulkan instance is created, so this
    # is the same condition that decided whether a layer is attached to `ctx`.
    validating = get(ENV, "LAVA_VALIDATION", "0") != "0"
    if !validating
        @info "validation layer off; nothing can leak and this test cannot assert"
        @test_skip validating
    else
        # Start from a known-clean queue, so what we observe is what we caused.
        MVE.drain_validation_messages!(ctx)
        empty!(ctx.validation.messages)

        # Larger than `maxBufferSize` AND larger than any heap, so it is refused
        # rather than merely unlucky.
        huge = 40_000_000_000
        res = MVE.try_vk_alloc(ctx.default_bq, huge)
        @test res isa MVE.AllocFailure          # refused, and refused gracefully

        # The ring must be empty too, not just the drained list. Draining is what
        # the buggy version skipped, so this is the assertion with teeth: it fails
        # if the messages are still in the ring waiting to ambush someone else.
        MVE.drain_validation_messages!(ctx)
        @test isempty(ctx.validation.messages)

        # And the next unrelated operation must survive — this is the shape of the
        # original symptom, a small upload that has nothing to do with the 40 GB.
        a = MVE.LavaArray(Float32[1, 2, 3, 4])
        b = a .+ 10.0f0
        MVE.vk_flush!(ctx)
        @test Array(b) == Float32[11, 12, 13, 14]
    end
end
