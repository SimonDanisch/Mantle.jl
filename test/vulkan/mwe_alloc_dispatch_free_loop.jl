# MWE-1: pure-compute alloc/dispatch/free loop.
#
# Question: does the cascade fault reproduce WITHOUT HW RT, with just
# a tight per-iter LavaArray alloc + compute dispatch + drop + GC?
#
# If YES — the bug is in Lava's compute path (alloc + finalize race), not
# RT-specific.  Likely H4 (pool memory aliasing).
#
# If NO — the bug needs RT.  H3 (vkDestroyBuffer races vkCmdTraceRays
# indirect-buffer reading) or some RT-specific BDA capture.

using Lava, Mantle
using KernelAbstractions
const KA = KernelAbstractions

@kernel function _compute_fill!(out, val)
    i = @index(Global)
    out[i] = val
end

backend = LavaBackend()
ctx = Mantle.vk_context()

println("=== MWE-1: pure-compute alloc/dispatch/free loop ===")
const N_ITERS = 30
const N_BUFS_PER_ITER = 12  # ~match VolPathState's count

# DON'T disable GC — let it race like Hikari does.
crashed_at = 0
for iter in 1:N_ITERS
    bufs = Mantle.LavaArray{Float32, 1}[]
    # Allocate N_BUFS_PER_ITER LavaArrays.
    for k in 1:N_BUFS_PER_ITER
        push!(bufs, Mantle.LavaArray(zeros(Float32, 1024)))
    end
    # Dispatch a kernel against each (forces last_write to be set).
    for arr in bufs
        _compute_fill!(backend)(arr, Float32(iter); ndrange=length(arr))
    end
    KA.synchronize(backend)
    # Drop all refs — finalizer thread will free buffers eventually.
    bufs = nothing
    if Mantle.device_lost(ctx)
        global crashed_at = iter
        break
    end
end
if crashed_at == 0
    println("MWE-1: $N_ITERS iters clean — pure compute alloc/free loop is OK.")
else
    println("MWE-1: !!! crashed at iter $crashed_at — bug is NOT RT-specific.")
end

using Test
@test crashed_at == 0
@test !Mantle.device_lost(ctx)
