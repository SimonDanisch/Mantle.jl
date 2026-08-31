# MWE-5: SoA work queue (StructArray of LavaArrays) per iter.
#
# Hikari's WorkQueue<VPRayWorkItem> uses StructArray-based SoA layout
# (per `should_use_soa(::Type{VPRayWorkItem}) = true` in workqueue.jl).
# `pin_leaves!` walks struct fields recursively, which SHOULD pin every
# inner LavaArray.  Test: does this pattern repro the cascade?

using Lava, StructArrays
using KernelAbstractions
const KA = KernelAbstractions

backend = LavaBackend()
ctx = MVE.vk_context()

# Mirror VPRayWorkItem-ish layout (Bool + struct fields that go through SoA).
struct WorkItem
    a::Float32
    b::Float32
    c::Float32
    flag::Bool
end

# Build a SoA StructArray of LavaArrays — same shape as Hikari's WorkQueue.items.
function alloc_soa_workqueue(n::Int)
    a    = MVE.LavaArray(zeros(Float32, n))
    b    = MVE.LavaArray(zeros(Float32, n))
    c    = MVE.LavaArray(zeros(Float32, n))
    flag = MVE.LavaArray(zeros(Bool,    n))
    return StructArray{WorkItem}((a=a, b=b, c=c, flag=flag))
end

@kernel function _soa_kernel!(items, val)
    i = @index(Global)
    items[i] = WorkItem(val + 1f0, val + 2f0, val + 3f0, val > 0f0)
end

println("=== MWE-5: SoA work queue + per-iter alloc/free ===")
const N_ITERS = 25
crashed_at = 0
for iter in 1:N_ITERS
    queue = alloc_soa_workqueue(1024)
    aux1 = alloc_soa_workqueue(1024)
    aux2 = alloc_soa_workqueue(1024)
    for sample in 1:4
        _soa_kernel!(backend)(queue, Float32(iter*sample); ndrange=1024)
        _soa_kernel!(backend)(aux1, Float32(iter*sample+1); ndrange=1024)
        _soa_kernel!(backend)(aux2, Float32(iter*sample+2); ndrange=1024)
        KA.synchronize(backend)
    end
    queue = nothing; aux1 = nothing; aux2 = nothing
    if MVE.device_lost(ctx)
        global crashed_at = iter
        break
    end
    print("$iter ")
end
println()
println(crashed_at == 0 ? "MWE-5: $N_ITERS iters clean." : "MWE-5: !!! crashed iter $crashed_at — SoA workqueue repros it!")

using Test
@test crashed_at == 0
@test !MVE.device_lost(ctx)
