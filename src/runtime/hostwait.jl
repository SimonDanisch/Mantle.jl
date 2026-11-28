# Waiting for the device on the host.
#
# A wait that blocks in a foreign call holds its thread, and every task sharing that thread
# stops with it. On the main thread that is the whole interface: a window's input loop runs
# beside its render loop as a task, and with the render loop parked for a frame's worth of
# GPU work, key presses waited a frame and a short tap was lost (MakieOrg/Makie.jl#5856).
# Metal.jl already waits cooperatively; Mantle's Vulkan waits did not, and neither did its
# Metal 4 queue or either backend's wait for an image. Nor did the present, which is where
# NVIDIA's X11 driver waits out a frame's rendering, whatever the present mode.

import GPUToolbox

"""
    hostwait(block, isdone, obj) -> Union{Nothing,Some}

Wait for `obj` without holding the thread. `isdone(obj)` answers without blocking and is
polled first, briefly; when it stays `false`, `block(obj)` runs on a worker thread while
this task yields, and its value comes back as `Some`. `nothing` means the poll saw `obj`
done and `block` never ran. `isdone = nothing` is a wait nothing can be asked about (an
image acquire), which always goes to a worker.

Every wait in Mantle for submitted work, for a presentable image and for a present goes
through here. A `block` that uses a queue runs under that queue's lock, held by the waiting
task, so no other task enters the queue meanwhile. The exceptions are `vkDeviceWaitIdle`
and `vkQueueWaitIdle`, which block on the calling thread under the queue's lock: they run
only to rebuild or tear down (a resize, a close, a reset, a stalled flush's probe).

`block` runs on another thread, so it must
- block GC-safely (`GPUToolbox.@gcsafe_ccall`, or ObjectiveC's `@objc`, which is), or a
  collection anywhere waits for the device;
- touch nothing another task may change meanwhile: what it needs is computed on the
  calling thread and carried in `obj`, and errors come back as its value for the caller
  to raise with context.

The waiter is GPUToolbox's `cooperative_wait`, the one CUDA and OpenCL use. Its polling
fast path allocates nothing. An allocation-free worker path additionally requires pooled,
typed wait requests in GPUToolbox; bits inputs and results alone do not guarantee it.
`isdone` stays with the waiting task, so it may close over what `obj` must not hold.
"""
@inline hostwait(block::B, isdone::D, obj) where {B,D} =
    GPUToolbox.cooperative_wait(block, obj; isdone)
