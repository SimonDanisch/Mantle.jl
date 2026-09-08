# Debug / validation: the PROOF step.
#
# There is exactly one way to switch validation on, and it is not in this file:
#
#     reset_device!(debug = DebugConfig(gpu_av = true, pool_disabled = true))
#
# See `DebugConfig` and `reset_device!`. This file used to hold five preset
# functions — `enable_gpu_av`, `disable_gpu_av`, `enable_debug_printf!`,
# `disable_debug_printf!`, `activate_all_debugging` — over seven `LAVA_*`
# environment variables. All twelve are deleted. Each preset encoded a slightly
# different combination (`enable_gpu_av` defaulted to `pool_disabled = false`,
# which its own docstring then explained was blind to the bugs you turn GPU-AV on
# to find), so "which one do I call" was itself a way to end up instrumented for
# something other than what you were hunting.
#
# What remains here is the one thing a config object cannot express, because it
# is a measurement rather than a setting: **"GPU-AV is enabled" and "GPU-AV is
# catching errors" are not the same thing on every driver.** `verify_gpu_av` runs
# a known out-of-bounds write and asserts the layer reports it, so a passing call
# is proof end-to-end and a clean run afterwards means something.

"""
    verify_gpu_av(; timeout=10.0) -> Bool

Run a known out-of-bounds device write and assert that the GPU-AV layer
reports it via its debug-utils callback. Returns `true` on success; throws
`LavaError` if the layer did not catch the OOB within `timeout` seconds.

**Call this every time you switch GPU-AV on.** It is the only step that can tell
you the instrument is live, and skipping it is how a clean run gets mistaken for
"no fault found":

    reset_device!(debug = DebugConfig(gpu_av = true, pool_disabled = true))
    verify_gpu_av()

**It can take the process with it, and that is also an answer.** GPU-AV can
attach (`gpu_assisted = true`) and still silently fail to instrument shaders on
some driver/loader combinations — observed on RADV / Mesa 26 here: the layer
attaches, never fires, and this probe's deliberate out-of-bounds write
**segfaults the process** rather than being reported. So a segfault or a thrown
`LavaError` here is definitive: GPU-AV is non-functional on this driver, and
anything you were about to conclude from a clean run under it means nothing. Do
GPU-AV bug-hunting on a driver where this returns cleanly.

The non-instrumenting layers need no probe — nothing about
`DebugConfig(sync_val = true, best_practices = true)` can silently not-instrument
— so that is the configuration to reach for when you want to keep the session
alive.

Uses a write at index `100_000_000_000` (~400 GB past the array start), so
the destination address is past every possible underlying `VkBuffer` —
this proves the layer is alive whether or not `pool_disabled` is set.

Implementation: the flush after a GPU-AV-caught dispatch can hang in cleanup
on some drivers (observed on AMDVLK Windows), so the flush is spawned on a
worker task and the main task polls `ctx.validation.messages` until either the
expected message appears or the timeout expires. On success, the function
calls `reset_device!` to clear the half-flushed batch state.
"""
function verify_gpu_av(; timeout::Float64=30.0, ctx::VkContext = vk_context())
    if !ctx.gpu_assisted
        throw(LavaError("verify_gpu_av",
            "GPU-AV is not enabled (gpu_assisted=false)",
            "Build the device with it on first: " *
            "`reset_device!(debug = DebugConfig(gpu_av = true, pool_disabled = true))`"))
    end
    bq  = ctx.default_bq
    dev = ctx.device
    clear_validation_messages!()
    arr = LavaArray{Int32,1}(undef, (16,); bq)
    @kernel inbounds = true function _lava_gpuav_check_kernel!(out, bad_idx::Int)
        i = @index(Global)
        if i == 1
            out[bad_idx] = Int32(0xBAD)
        end
    end
    # Write FAR past the (pool-disabled) dedicated 64-byte buffer — 400 KB out,
    # so the faulting address lies outside EVERY tracked VkBuffer range.  GPU-AV's
    # BufferDeviceAddressPass flags a store only when its address is within *no*
    # known buffer; a small offset (a few KB) can land inside a neighbouring
    # tracked buffer and is, correctly, not flagged.  (An earlier 1000-index /
    # 4 KB probe gave a false negative on the RTX 4000 Ada for exactly this
    # reason.)
    bad_idx = 100_000
    k = _lava_gpuav_check_kernel!(LavaBackend(), 1)
    Base.invokelatest(k, arr, bad_idx; ndrange=4)
    # GPU-AV surfaces the violation only after the host waits for the dispatch
    # to complete (the layer reads back the instrumented error log in its
    # post-wait hook, then invokes our debug callback).  Drive that readback
    # ourselves with a SHORT, FINITE semaphore wait in a poll loop — never a
    # blocking `KA.synchronize`, which (a) trips the single-writer assertion if
    # spawned off-thread and (b) blocks forever if a faulting submit never
    # signals.  The async callback only copies bytes into a ring, so neither the
    # wait nor the drain can hang; we drain + scan after each short wait.
    submit!(bq)
    target = something(newest(bq), UInt64(0))
    matches(s) = occursin("Out of bounds access", s) || occursin("device address", s)
    caught_msg = ""
    deadline = time() + timeout
    while time() < deadline && isempty(caught_msg)
        VK.wait_semaphores(dev,
            VK.SemaphoreWaitInfo([bq.timeline_sem], [target]),
            UInt64(200_000_000))   # 200 ms — finite, so a non-signalling fault can't hang us
        drain_validation_messages!(ctx)
        for m in ctx.validation.messages
            matches(m) && (caught_msg = m; break)
        end
    end
    if isempty(caught_msg)
        n = length(ctx.validation.messages)
        reset_device!()   # clear the half-flushed faulting batch
        throw(LavaError("verify_gpu_av",
            "GPU-AV did not report a known out-of-bounds write within $(timeout)s " *
            "($(n) validation messages captured, none matching 'Out of bounds access').",
            "GPU-AV is not instrumenting BDA stores on this driver/layer — check that " *
            "gpuav_buffer_address_oob is enabled and shaderInt64 is supported. Note the " *
            "device must have been BUILT with `DebugConfig(gpu_av = true)`; there is no " *
            "way to switch it on afterwards."))
    end
    @info "Lava: GPU-AV verified working" sample=first(caught_msg, 240)
    # The OOB left the batch queue in an errored state; reset so subsequent
    # code starts clean.
    reset_device!()
    return true
end
