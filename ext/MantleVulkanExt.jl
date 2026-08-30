"""
Mantle's Vulkan backend.

The 27,000 lines under `src/vulkan/` are included from here rather than from
`Mantle.jl`, which is what makes `using Mantle` work on a machine with no Vulkan
loader — a Mac, a CI runner, anything that only wants the allocator and the
graph. `VulkanCore`'s `__init__` calls `error()` when it cannot `dlopen` the
loader, so merely *depending* on Vulkan.jl was fatal, not lazy.

Triggered on **both** Lava and Vulkan, because the backend needs both: Vulkan is
the driver and Lava is the Julia→SPIR-V compiler that feeds it. Neither is
Mantle's business — a Metal backend needs neither, which is the entire reason
this boundary exists.

## The import lists below are load-bearing

`src/vulkan/` was written inside `module Mantle`, where every one of Mantle's
names resolved for free. Here it is a separate module, and the difference is
silent in the worst way: a method on an un-imported `release!` does not extend
`Mantle.release!`, it defines `MantleVulkanExt.release!`, and nothing fails
until whatever calls `release!` gets the wrong one — or nothing at all.

So the lists are exact and mechanically derived, not guessed:

  * `import Mantle: …` — everything this backend adds METHODS to, or names at
    type-definition time. `import` and not `using`, because only `import` lets a
    method extend the original rather than shadow it. It is a long list because
    the graph is Mantle's: `Graph`, `Plan`, `Pass`, `Compile`, `PassPlan` and
    the ~40 functions over them used to be defined here, and are now consumed.
  * `using Mantle: …` — the names it merely READS that Mantle does not export.
    Everything Mantle does export arrives via the bare `using Mantle`.

`test/vulkan/test_lava_import_completeness.jl` checks both against the loaded
modules, which is the only way to check them: names generated in `@eval` loops
are invisible to a scan of the source.
"""
module MantleVulkanExt

using Mantle

# Extended here — see the note above on why `import`.
import Mantle: Attribute, Device, Graph, Plan, Surface, Transient, Update,
    Window, access, alignment, arena, backend, bake!, baked, bufferusage,
    capacity, compatible, constraintof, copy!, count, describe, devicecopy!,
    deviceview, dispatches, download, draw!, fence, free!, handle,
    indirectcount!, layout, materialize!, maxalloc, mergeconstraints, nbytes,
    needs_transition, newpass, npipelines, overlapping, passed, passes, pool,
    rawalloc, rawfree, rebind!, rebindable, release!, remap!, remappable,
    render!, retire!, run!, screenshot, stages, storage, stride, timings,
    touch!, upload!, usages, use, vkformat, waitfor

# The API verbs. Every one of these is declared in `graphics/commands.jl` or
# `raytracing/api.jl` and implemented below — `import`, so the methods land on
# Mantle's function and a caller who wrote `using Mantle` reaches them. With
# `using` they would be new functions in this module and silently unreachable,
# which is the same trap the lists above exist for.
# The queue lifecycle verbs and the device-idle wait. Declared in
# `src/graph/queue.jl` beside the shared `BatchQueue` struct; the methods below
# are Vulkan's. `import`, so `Mantle.flush!` resolves for a caller like RayMakie
# rather than being a separate `MantleVulkanExt.flush!` nothing can reach.
import Mantle: allocate_batch_queue!, release_batch_queue!, ensure_active_batch!,
    flush!, waitidle, supports_graphics, use_bindings!, devicearray, supports_rt_pipeline

import Mantle: begin_pass!, end_pass!, draw_in_pass!, draw_indexed_in_pass!,
    draw_indirect_in_pass!, set_viewport!, reset_device!, blit!, present_frame!,
    acquire_next_image!, transition_image!, readback_framebuffer,
    readback_window, bind_textures
import Mantle: build_accel!, refit_tlas!, set_anyhit_pipeline!, trace_rays!,
    trace_rays_indirect!, trace_closest_hits!, trace_closest_hits_indirect!,
    trace_closest_hits_anyhit!, trace_closest_hits_anyhit_indirect!

# The graph's backend interface — `src/graph/backend.jl` is the whole of what
# Mantle asks of a backend, and these are this backend's answers. `isdepth` is
# in the list because a `TransientImage` wraps its element type, which the core
# fallback cannot see through; `aspect` is NOT, because it is Vulkan's own
# spelling and nothing outside this module says it.
import Mantle: isdepth, target_extent, checkextents, refit!, record!,
    record_pass!, rename!, inplace!, nextslot!, collect!

# The abstract resource types whose concretes are defined below. Imported rather
# than reached through `using Mantle` because `VulkanTexture2D <: Texture2D`
# needs the name at type-definition time.
import Mantle: Texture, Texture1D, Texture2D, Sampler, TextureBindings,
    Framebuffer, RenderTarget, HWTLAS, AccelBuildContext, BatchQueue,
    ExternalImage, CompiledGraphicsPipeline, Window


# The graph. `Graph`, `Plan`, `Pass`, `Compile` and the rest are Mantle's now —
# this backend used to define them and no longer does. It records and submits
# what the graph produced, so it needs the names, and it adds methods to a good
# many of them: `import`, for the reason at the top of this file.
import Mantle: ArgMemory, Attr, BufferBlock, BufferRange, Buffers, Commands,
    Compile, CompiledDispatch, CompiledDraw, DrawCall, Images, Pass,
    PassHandle, PassPlan, Profiler, Recycler, TransientBuffer,
    TransientImage, TransientResource, WindowSurface, argalign, barrierbuffer,
    barrierspan,
    clearvalue, depthclear, devargs, dispatchrange, drawover, elapsed,
    emit_draw!, extrausage, first_target, imageusage, initial_state,
    initial_usage, kernelfor, lastuses, lp_of, makeimage, packdispatch!,
    rawargs, remakeimage!,
    record_draw!, record_updates!, recordlaunch!, recycle!, renameable,
    resourcekind, rootresource, sample!, slice, sliceindex, slotbase,
    takehost!, target_format, target_image, target_view, updates_pass!,
    usage_of, write_update!, writes_it

# Read but not exported by Mantle, so `using Mantle` alone would not bind them.
# `RayTracingPipeline` is a concrete Mantle type the backend consumes rather
# than subtypes, and `_normalise_chit` is its unexported helper.
using Mantle: _normalise_chit
# `mat4_to_vk_transform` moved to `geometry/transform.jl` — it is arithmetic on
# `Mat3x4f`, which both backends' instance descriptors start with.
using Mantle: mat4_to_vk_transform

using Mantle: Analysis, Compilation, DiscardOp, KeepOp, alias, allocate,
    analysis, arenaof, coopmatgemm, device, humanbytes, memoryof, offset,
    ordered, pinned, policy, region, transients

# The device vocabulary. `caps` in particular is one function with a `Device`
# method in Mantle and a `KI.Backend` method here.
using KernelInterface: MatrixUse, MatrixA, MatrixB, Accumulator, MatrixScope,
    SubgroupScope, WorkgroupScope, MatrixShape, DeviceCaps
import KernelInterface: supports, bestshape, caps, matrix_shapes, wggranularity

# The packages the moved runtime calls unqualified, copied from `Mantle.jl`'s
# preamble verbatim. `using Vulkan` binds the ~60 bare names it uses —
# `unwrap`, `cmd_dispatch`, `DeviceMemory`, the `FORMAT_*` and `ERROR_*`
# constants — and `as VK` gives the qualified spelling the same code uses for
# everything else.
using Vulkan
import Vulkan as VK
import Serialization
import PrecompileTools
using GPUCompiler
using LLVM
using LLVM: API
using GPUArrays
using GPUArraysCore
using KernelAbstractions
using Adapt
using Atomix
using UnsafeAtomics
using AcceleratedKernels
using SPIRV_Tools_jll
using LinearAlgebra
using StaticArrays
using GeometryBasics
using Raycore: Ray
import GLFW

# The backend itself. Left under `src/vulkan/` rather than moved into `ext/`:
# `src/metal/` goes beside it, the two are read against each other constantly,
# and `test_lava_import_completeness.jl` walks `src/` to check exactly the
# import lists above.
include("../src/vulkan/vulkan.jl")

"""
Everything that needs a live device, at load.

This was `Mantle.__init__`, and it moved here with the runtime it serves: both
halves reach into the Vulkan context, so in `Mantle` they would run on a machine
that has no driver at all.

Leaving the `atexit` hook behind was not theoretical when the runtime first
moved out of Lava: without it, a `LavaArray` finalizer running during Julia's
shutdown sweep calls `query_timeline` on a semaphore whose device is already
gone, and the process takes a SIGSEGV inside the driver.
"""
function __init__()
    # Announce Vulkan as a source of a default device. Highest priority: it is
    # the backend with the graphics pipeline, so a caller who did not choose gets
    # the one that can do everything. `vk_context()` builds the context on first
    # call and throws when no loader/ICD is present, which is a question this
    # probe has to answer with `nothing` — hence the explicit availability check
    # rather than a call wrapped in a swallowing `try`.
    Mantle.register_backend!(; name = :vulkan, priority = 100) do
        vulkan_available() ? LavaBackend() : nothing
    end

    # The pipeline builder thread. Pipelines are created off the main thread so a
    # first launch does not block on the driver's compiler.
    init_pipeline_thread!()

    # Mark the device lost during shutdown so GC finalizers do not call into the
    # Vulkan driver after it has been torn down. `atexit` runs BEFORE Julia's
    # global finalizer sweep, which is the whole point — a finalizer that reaches
    # a dead device segfaults inside the driver, where no Julia `try` can catch
    # it and no stack trace names the buffer that did it.
    atexit() do
        ctx = VK_CONTEXT_REF[]
        ctx === nothing || mark_device_lost!(ctx)
        bind_context!(nothing)
    end
end

end
