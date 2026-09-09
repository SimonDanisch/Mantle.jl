"""
Mantle's Metal backend.

`src/metal/` is loaded from here when Metal.jl is, the same way `src/vulkan/` is
loaded by `MantleVulkanExt` when Vulkan and Lava are. Triggered on Metal alone:
unlike the Vulkan backend this one needs no separate compiler, because Metal.jl
brings its own.

The import list below is the same contract the Vulkan extension's is, and
load-bearing for the same reason: a method on an un-imported `rawalloc` does not
extend `Mantle.rawalloc`, it defines `MantleMetalExt.rawalloc`, and nothing
fails until the pool calls the one it can see — or finds none at all.

It is much shorter than the Vulkan one, and that is the measurement this whole
refactor was for. The graph, the allocator, the plan and the array algorithms
are Mantle's; a backend answers questions about its device.
"""
module MantleMetalExt

using Mantle

# Extended here — the eleven pool primitives, the four transfer verbs, and the
# device itself.
import Mantle: Device, backend, pool, devices, defaultdevice!,
    rawalloc, rawfree, constraintof, compatible, mergeconstraints,
    fence, passed, waitfor, maxalloc, capacity, blocksize,
    upload!, download, devicecopy!, deviceview, bufferusage

import Mantle: caps
# The graph's execution path is core's (`graph/kalaunch.jl`); this backend
# supplies `resolve` and says that barriers are a no-op.
import Mantle: resolve, syncbackend, needs_transition, materialize!, alignment
# Hardware ray tracing — the verbs Mantle declares in `raytracing/api.jl`.
import Mantle: build_accel!, refit_tlas!, trace_closest_hits!
# Device-wide sync, the graphics capability answer, and the queue verb that
# refuses. Declared in `src/graph/queue.jl` and `src/graphics/commands.jl`.
import Mantle: waitidle, supports_graphics, allocate_batch_queue!, supports_rt_pipeline
# The rasterisation half (`src/metal/graphics.jl`).
import Mantle: Framebuffer, Window, draw!, readback_framebuffer
# The graph's render-pass verbs, declared in `src/graphics/commands.jl`.
import Mantle: compile_draw, begin_render_pass!, record_draw!, end_render_pass!
# The incremental acceleration structure (`src/metal/hwtlas.jl`) implements
# Mantle's `HWTLAS` and Raycore's traversal interface.
import Mantle: HWTLAS, AdaptedAccel, Mat4f, Mat3x4f, mat4_to_vk_transform
import Raycore: closest_hit, any_hit, sync!, world_bound, n_geometries,
    n_instances, update_transforms!, update_transform!, wait_for_gpu!
import GeometryBasics
# `Vec4f` because a clip position IS one: the stage output struct declares the
# vector, not the tuple it wraps, which is what the Vulkan side already did.
using GeometryBasics: decompose, Point3f, Point2f, Vec2f, Vec3f, Vec4f
using Metal: MtlArray
using StaticArrays: SVector
using Base: @propagate_inbounds
import LinearAlgebra
import Adapt
# `MetalTLAS{Tri}` names its triangle type, and that type is Raycore's —
# `HWTLAS{Tri} <: Raycore.AbstractAccel` is where the parameter comes from.
# Missing here, `build_accel!` threw `UndefVarError: Raycore` at the one line
# that constructs the TLAS, so the structure built and then failed to be
# wrapped. The Vulkan extension imports Raycore for the same reason.
import Raycore
using Metal.ObjectiveC: NSArray
using Mantle: TransientBuffer
using Mantle: DeviceInfo, selectdevice
# Read, not extended.
using Mantle: Pool, DeviceArray, Persistent, Buffers, Images, region, memoryof, offset

# `KernelInterface` is what a backend implements; `caps` in particular is one
# function with a `Device` method in Mantle and a `KI.Backend` method here.
using KernelInterface: DeviceCaps, MatrixShape, MatrixScope, SubgroupScope
import KernelInterface
# `N0f8` for `mtlformat`'s table: ColorTypes re-exports FixedPointNumbers'
# normalised types, and Mantle re-exports `RGBA`/`BGRA`.
# `N0f8` through ColorTypes, which re-exports FixedPointNumbers' normalised
# types — FixedPointNumbers is not a direct dependency of Mantle and naming
# four types is not a reason to make it one.
using ColorTypes: RGBA, BGRA
using ColorTypes.FixedPointNumbers: N0f8, FixedPoint
using ColorTypes: FixedPointNumbers
const KI = KernelInterface

using GPUArrays
using KernelAbstractions
const KA = KernelAbstractions

include("../src/metal/metal.jl")

"""
Announce Metal as a source of a default device.

The probe answers `nothing` rather than throwing when the machine has no usable
Metal GPU, because `Mantle.defaultbackend` asks every registered backend in turn
and one that raises would hide the rest. Priority below Vulkan: where both are
present the discrete-GPU path is the one a caller who did not choose most likely
wants, and on a Mac Vulkan is MoltenVK over this same device anyway.
"""
function __init__()
    Mantle.register_backend!(; name = :metal, priority = 90) do
        Metal.functional() ? Metal.MetalBackend() : nothing
    end
end

end
