"""
Two devices in one process: the acceptance probe for `GUARDRAILS.md` §8.

    julia --project=<env> dev/Lava/test/twodevice_probe.jl

**It passes**, and is in `runtests.jl`. It was held out while it segfaulted,
because a segfault takes the whole suite with it.

## Why this is possible now, and was not before

`VkContext(; select)` lets a caller choose the physical device and returns a
context **without** installing it as the global. The Vulkan loader enumerates the
real GPU and lavapipe from one instance, so every machine here has a two-device
pair with no second card:

    gpu = vk_context()
    cpu = VkContext(select = devs -> only(filter(islavapipe, devs)))

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

5. **`LavaBackend()` built inside the library, six places.** An unpinned backend
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

7. **`_REDUCE_SCRATCH`** — FIXED (now `ctx.caches.reduce_scratch`). It was an
   `IdDict` keyed by context,
   and still allocated its buffer on `vk_context()`. Keyed right, allocated
   wrong — which reads as correct on any machine with one device, and is the
   subtlest shape in this whole list.

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
"""

using Lava, KernelAbstractions, LinearAlgebra
const KA = KernelAbstractions

@kernel function twodev!(d, s)
    i = @index(Global)
    @inbounds d[i] = d[i] * 2.0f0 + s
end

function probe()
    gpu = MVE.vk_context()
    cpu = MVE.VkContext(select = devs -> only(filter(MVE.islavapipe, devs)))

    println("gpu id=$(gpu.id)  $(gpu.device_name)")
    println("cpu id=$(cpu.id)  $(cpu.device_name)")
    gpu.id == cpu.id && error("two contexts share an id — every per-device key is void")

    # Two devices must resolve their own `vkCmdPipelineBarrier` — unless a layer
    # is loaded. `vkGetDeviceProcAddr` then returns the LAYER's dispatch
    # trampoline, which is one piece of code for every device by construction, so
    # the pointers are legitimately equal and this says nothing about Lava.
    #
    # The check fired on exactly that: with `debug = DebugConfig(validation = true)` the probe aborted
    # here, before reaching anything it exists to test — which is the one
    # configuration you would want to run it in.
    if gpu.debug_messenger === nothing && cpu.debug_messenger === nothing
        gpu.cmd_pipeline_barrier_fptr == cpu.cmd_pipeline_barrier_fptr &&
            error("both devices report the same vkCmdPipelineBarrier — the global is back")
    else
        println("  (validation layers active: skipping the fptr check — the layer's " *
                "dispatch trampoline is shared by design)")
    end

    for (name, ctx) in (("gpu", gpu), ("cpu", cpu))
        b = LavaBackend(ctx)

        # ── a dispatch, and the fill! that feeds it
        a = KA.allocate(b, Float32, 64)
        fill!(a, 1.0f0)
        twodev!(b, 64)(a, 3.0f0; ndrange = 64)
        KA.synchronize(b)
        got = Array(a)
        ok = all(got .== 5.0f0)

        # ── a reduction: the scratch was keyed by context but ALLOCATED on the
        #    global one, so the second device's entry held the first device's
        #    buffer. Keyed right, allocated wrong. Now `ctx.caches.reduce_scratch`.
        r = MVE.vk_reduce_sum(a)
        okr = r ≈ 64 * 5.0f0

        # ── a GEMM big enough to split K: the split-K scratch was one `Ref`
        #    holding device memory, i.e. the memory pool's defect in miniature.
        #    Now `ctx.caches.gemm_split_scratch`.
        m = 64
        A = KA.allocate(b, Float16, m, m); fill!(A, Float16(1))
        B = KA.allocate(b, Float16, m, m); fill!(B, Float16(2))
        C = KA.allocate(b, Float32, m, m); fill!(C, 0.0f0)
        LinearAlgebra.mul!(C, A, B)
        KA.synchronize(b)
        okg = all(Array(C) .== Float32(2 * m))

        println("  $name: dispatch ", ok  ? "ok" : "WRONG ($(got[1]))",
                "   reduce ", okr ? "ok" : "WRONG ($r)",
                "   gemm ",   okg ? "ok" : "WRONG ($(Array(C)[1]))")
        (ok && okr && okg) || error("$name produced a wrong result")
        A = B = C = a = nothing
    end

    # ── the validation ring, which was EIGHT module-level globals ────────────
    #
    # `create_vulkan_context` builds a fresh `Vulkan.Instance` and
    # `DebugUtilsMessengerEXT` per context, so two contexts meant two messengers
    # writing into one ring: device A's errors surfaced in device B's
    # `get_validation_messages()`, and whichever drained first consumed the
    # other's messages. The messenger now carries the ring's address as
    # `pUserData`, which is what the callback reads.
    gpu.validation === cpu.validation &&
        error("both contexts share one ValidationRing — the global is back")
    MVE.ring_user_data(gpu.validation) == MVE.ring_user_data(cpu.validation) &&
        error("both messengers were handed the same pUserData")
    for (name, ctx) in (("gpu", gpu), ("cpu", cpu))
        MVE.drain_validation_messages!(ctx)
        println("  $name: ring wrote $(ctx.validation.write[1]), " *
                "drained $(length(ctx.validation.messages)) message(s)")
    end

    # The assertion this probe was built around — "one kernel on two devices must
    # compile TWICE, because a single new entry means the second device was handed
    # the first's pipeline" — counted entries in a global dict. There is no global
    # dict now: each context owns its caches, so a shared pipeline is not a thing
    # that can happen, and counting cannot express it.
    #
    # What replaces it is weaker as a test and stronger as a guarantee. Assert
    # that the caches are DISTINCT OBJECTS and that each device populated its
    # own, which is what "keyed correctly" was always trying to approximate.
    println()
    for (name, ctx) in (("gpu", gpu), ("cpu", cpu))
        println("  $name: pipelines=$(length(ctx.caches.pipelines)) ",
                "linked=$(length(ctx.caches.linked)) ",
                "launchplans=$(length(ctx.caches.launchplans)) ",
                "pool blocks=$(length(ctx.caches.pool.blocks))")
    end
    gpu.caches === cpu.caches && error("both contexts share one DeviceCaches")
    gpu.caches.pipelines === cpu.caches.pipelines && error("both contexts share one pipeline cache")
    gpu.caches.pool === cpu.caches.pool && error("both contexts share one memory pool")
    isempty(gpu.caches.pipelines) && error("the gpu compiled nothing")
    isempty(cpu.caches.pipelines) && error("lavapipe compiled nothing")

    # Retire the context this probe built. Nothing else can: `VkContext(;
    # select)` deliberately does NOT install it as the global, so it is the
    # caller's, and `reset_device!` — which retires the context it replaces —
    # never sees it.
    #
    # Without this the arrays above outlive the probe, and their finalizers run
    # at whatever GC comes next — in practice `ijl_atexit_hook`, against a
    # lavapipe device Vulkan.jl has already torn down. The suite printed its
    # summary and then the process died with SIGSEGV in `libvulkan_lvp.so`,
    # which reads as "the tests crashed" and is a long way from this line.
    MVE.mark_device_lost!(cpu)
    println("\nPASS")
end

if abspath(PROGRAM_FILE) == @__FILE__
    probe()
end
