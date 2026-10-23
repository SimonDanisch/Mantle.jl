# Validation is switched on only by building the device with
# `reset_device!(debug = DebugConfig(...))`. `verify_gpu_av` checks that GPU-AV,
# once enabled, actually reports errors on this driver.

# The probe kernel. Kernels are compiled by function, so this cannot be a closure
# inside `verify_gpu_av`.
function gpuav_check_kernel!(out, bad_idx::Int)
    i = KI.get_global_id().x
    if i == 1
        @inbounds out[bad_idx] = Int32(0xBAD)
    end
    return nothing
end

"""
    verify_gpu_av(; timeout = 30.0, ctx = vk_context()) -> Bool

Run a known out-of-bounds device write and return `true` once GPU-AV reports it
through the debug-utils callback; throw `LavaError` if no matching message
arrives within `timeout` seconds. Requires a device built with
`DebugConfig(gpu_av = true)`; call it after every such build, since a clean run
under a GPU-AV that does not instrument proves nothing. On RADV / Mesa 26 the
layer attaches but does not instrument, and the probe write segfaults the
process. Success and the throw both reset `ctx`'s device. `sync_val` and
`best_practices` need no probe.
"""
function verify_gpu_av(; timeout::Float64=30.0, ctx::LavaDevice = vk_context())
    if !ctx.gpu_assisted
        throw(LavaError("verify_gpu_av",
            "GPU-AV is not enabled (gpu_assisted=false)",
            "Build the device with it on first: " *
            "`reset_device!(debug = DebugConfig(gpu_av = true, pool_disabled = true))`"))
    end
    bq  = ctx.default_bq
    dev = ctx.device
    clear_validation_messages!(ctx)
    arr = LavaArray{Int32,1}(undef, (16,); bq)
    # 400 KB past the 64-byte array. GPU-AV flags a device-address store only
    # when no tracked VkBuffer contains it; an offset of a few KB can land in a
    # neighbouring buffer and go unreported.
    bad_idx = 100_000
    Base.invokelatest(KI.Kernel(LavaBackend(bq), gpuav_check_kernel!),
                      arr, bad_idx; ndrange=4, workgroupsize=1)
    # GPU-AV reports only after a host wait on the dispatch. Poll with short
    # semaphore waits: `KA.synchronize` blocks forever if a faulting submission
    # never signals.
    submit!(bq)
    target = something(newest(bq), UInt64(0))
    matches(s) = occursin("Out of bounds access", s) || occursin("device address", s)
    caught_msg = ""
    deadline = time() + timeout
    while time() < deadline && isempty(caught_msg)
        VK.wait_semaphores(dev,
            VK.SemaphoreWaitInfo([timelineof(bq)], [target]),
            UInt64(200_000_000))   # 200 ms
        drain_validation_messages!(ctx)
        for m in ctx.validation.messages
            matches(m) && (caught_msg = m; break)
        end
    end
    if isempty(caught_msg)
        n = length(ctx.validation.messages)
        reset_device!(ctx)
        throw(LavaError("verify_gpu_av",
            "GPU-AV did not report a known out-of-bounds write within $(timeout)s " *
            "($(n) validation messages captured, none matching 'Out of bounds access').",
            "GPU-AV is not instrumenting BDA stores on this driver/layer — check that " *
            "gpuav_buffer_address_oob is enabled and shaderInt64 is supported. Note the " *
            "device must have been BUILT with `DebugConfig(gpu_av = true)`; there is no " *
            "way to switch it on afterwards."))
    end
    @info "Lava: GPU-AV verified working" sample=first(caught_msg, 240)
    # The fault leaves the batch queue errored. Reset the same physical device;
    # an unselected reset may pick another GPU.
    reset_device!(ctx)
    return true
end
