"""
Two devices in one process compute correctly, each on its own: the acceptance
probe for `GUARDRAILS.md` §8. It was `twodevice_probe.jl`, a Vulkan script.

## Why this is possible

`Device(api; select)` builds a device of the caller's own **without** installing
it as the default. The Vulkan loader enumerates the real GPU and lavapipe from
one instance, so every Linux machine here has a two-device pair with no second
card. A Mac has one Metal device, and there this skips.

## What it is for

Four caches hold device-owned handles at module scope. Keying them by
`VkContext.id` (never reused) is necessary and — as this probe demonstrated on
its first run — **not sufficient**. Module-scope device state is a larger
category than the cache list, and the only way to enumerate it honestly is to
run two devices and see what breaks.

## What it found, in the order it found them

Each was fixed and the probe re-run, which is why the list is ordered: nothing
below was visible until everything above it was fixed.

1. **`CMD_PIPELINE_BARRIER_FPTR`** — FIXED, now `VkContext.cmd_pipeline_barrier_fptr`.
   A device function pointer is per device: `vkGetDeviceProcAddr` returns one
   valid only for the device asked. As a global, creating the second context
   overwrote it, and the FIRST device's command buffers were then recorded
   through the SECOND device's driver. The crash was inside `libvulkan_lvp.so`
   while dispatching on the NVIDIA context, which is as confusing as it sounds.

   This is the most dangerous class of the lot and `GUARDRAILS.md` §8 does not
   name it: §8 lists four caches holding *handles*, and says nothing about the
   function table. A stale handle is undefined behaviour; a foreign function
   pointer is an immediate jump into another driver.

2. **`PREPARE_INDIRECT_*`** — FIXED. Four `Ref`s holding one compiled pipeline
   and its layout, now one `PrepareIndirect` per device.

3. **`DEVICE_SUBGROUP_SIZE` and `SUBGROUP_SIZE_CONTROL`** — FIXED, now per-device
   dicts. These differ from the handle caches in a way worth keeping in mind: a
   stale pipeline is undefined behaviour and usually crashes, while a stale
   device *property* just returns the wrong number. 32 here, 64 on RDNA 3.5 —
   every tiling decision keyed on it would be made for the other device, and
   nothing would crash.

4. **THE ALLOCATOR** — the one that actually mattered. FIXED.

   `POOL_BLOCKS` and `POOL_FREE_LISTS` are module-level, and `PoolBlock` carries
   no device:

       mutable struct PoolBlock
           buffer, memory, base_address, capacity, bump, live_count
       end

   Measured: allocate on the GPU (one 64 MiB block is created), then allocate on
   lavapipe — `length(POOL_BLOCKS)` is **still 1**. The lavapipe array was carved
   out of the NVIDIA device's block. `buf.ctx` correctly says `cpu`; the memory
   underneath belongs to the other device.

   That is both observed symptoms at once. `fill!` on the second context then
   reads back **0.0** — it wrote into memory that device does not own — and in a
   different call order it segfaults instead.

   **A bigger class than anything in `GUARDRAILS.md` §8, which lists four caches
   holding pipeline handles.** The allocator hands out *memory*, so the failure
   is silent data corruption rather than a bad handle, and no amount of cache
   keying reaches it.

   Now `MemoryPolicy` per device, with each `PoolBlock` carrying a back-reference
   to its pool so `return_to_pool!` — which runs from a **finalizer**, where a
   lookup must not allocate and must not be able to miss — is a field hop.

5. **`TESTBACKEND` built inside the library, six places.** An unpinned backend
   resolves its queue through `vk_context()`, so `fill!`, `mul!`,
   `coopmat_gemm!`, broadcast `_copyto!` and the identity-matrix constructor all
   dispatched on whichever context was global rather than on the array's own.
   The array read back as zeros and Lava's `sync_access!` guard caught it much
   later as *"buffer was last written on a VulkanBatchQueue from a DIFFERENT
   VkContext"* — a good error a long way from its cause. All six now derive the
   backend from the data with `KA.get_backend`.

   That guard existing and firing is worth noting: the library already knew this
   was possible and said so precisely.

6. **`GEMM_SPLIT_SCRATCH`** — FIXED (now `ctx.caches.gemm_split_scratch`), and
   exercised above. It began as a single `Ref`
   holding device memory: the memory pool's defect in miniature, and it would
   only have fired on a split-K GEMM, which is why the first version of this
   probe missed it. The probe now runs a GEMM and a reduction on each device
   rather than a bare dispatch, because a global that only some kernels touch is
   invisible to a probe that only runs one kernel.

7. **`_REDUCE_SCRATCH`** — FIXED (now `ctx.caches.reduce_scratch`). As an
   `IdDict` keyed by context it still allocated its buffer on `vk_context()`:
   keyed right, allocated wrong, which reads as correct on any machine with one
   device and is the subtlest shape in this whole list.

8. **`WORKGROUP_LIMIT`** — FIXED (now `caps(ctx).workgrouplimit`). Listed here as
   "a policy limit, not a queried one", which was the whole defect: it was a
   module-level `Ref(1024)` whose docstring said it was the device's
   `maxComputeWorkGroupInvocations`. A device whose real limit is lower would
   have been handed the other one's, and the launch fails validation rather than
   returning a wrong answer — so this is the loud member of the list.

Still unaudited, because nothing on this path reaches them: `TIMESTAMP_POOL` (a
`Vulkan.QueryPool`), and `BLIT_PIPELINE` / `GFX_SHADER_CACHE` (graphics). Extend
the probe before trusting a second device for graphics or dispatch profiling.

The list above was produced by RUNNING two devices. Reading produced a list of
four caches, and **none of the four was what actually broke it.**

## What is asserted

That a dispatch with the `fill!` feeding it, a reduction and a GEMM big enough to
split K each give the right answer on BOTH devices, which is what every item
above broke. The probe also checked the Vulkan contexts' internals directly —
distinct ids, distinct `vkCmdPipelineBarrier` pointers, distinct validation
rings and cache objects, a non-empty pipeline cache on each — and its own note
says those checks only approximate "each device got its own": the results are
the observable. It retired the second context with `mark_device_lost!`; no
portable verb retires a device, and the shutdown hook covers every device
(`vulkan/test_twodevice_shutdown.jl`).
"""

using Test, Mantle, KernelAbstractions, LinearAlgebra
include(joinpath(@__DIR__, "testbackend.jl"))
const KA = KernelAbstractions

@kernel function twodev!(d, s)
    i = @index(Global)
    @inbounds d[i] = d[i] * 2.0f0 + s
end

"""
Another device on the API `dev` is on, built now and not installed as the
default, or `nothing` when this machine has no second one. A CPU device first
where there is one: lavapipe costs no GPU memory and is on every Linux machine
here.
"""
function otherdevice(dev)
    api = Mantle.syncbackend(dev)
    others = filter(i -> i.name != Mantle.devicename(dev), Mantle.devices(api))
    isempty(others) && return nothing
    pick = something(findfirst(i -> i.kind == :cpu, others), 1)
    return Mantle.Device(api; select = others[pick].index)
end

@testset "two devices in one process" begin
    dev = Mantle.Device(TESTBACKEND)
    other = otherdevice(dev)
    if other === nothing
        @info "only one device on this API here; the two-device probe needs a second" device = Mantle.devicename(dev)
    else
        @test other !== dev
        for d in (dev, other)
            @testset "$(Mantle.devicename(d))" begin
                b = Mantle.backend(d)

                # ── a dispatch, and the fill! that feeds it
                a = KA.allocate(b, Float32, 64)
                fill!(a, 1.0f0)
                twodev!(b, 64)(a, 3.0f0; ndrange = 64)
                KA.synchronize(b)
                @test all(Array(a) .== 5.0f0)

                # ── a reduction: the scratch was keyed by device but ALLOCATED on
                #    the default one, so the second device's entry held the first
                #    device's buffer. Keyed right, allocated wrong.
                @test sum(a) ≈ 64 * 5.0f0

                # ── a GEMM big enough to split K: the split-K scratch was one
                #    `Ref` holding device memory, the memory pool's defect in
                #    miniature.
                m = 64
                A = KA.allocate(b, Float16, m, m); fill!(A, Float16(1))
                B = KA.allocate(b, Float16, m, m); fill!(B, Float16(2))
                C = KA.allocate(b, Float32, m, m); fill!(C, 0.0f0)
                LinearAlgebra.mul!(C, A, B)
                KA.synchronize(b)
                @test all(Array(C) .== Float32(2 * m))
            end
        end
    end
end
