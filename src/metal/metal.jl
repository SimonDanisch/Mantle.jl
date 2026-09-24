# Mantle's Metal backend.
#
# Beside `src/vulkan/`, implementing the same interface. What that interface IS
# was established by making Mantle portable first: the graph, the allocator, the
# plan, the pass and the array algorithms are all core, and a backend answers a
# short list of questions about its device. This file is the Metal answers.
#
# It needs neither Lava nor SPIR-V. Vulkan reaches the GPU through a Julia→SPIR-V
# compiler; Metal has its own, in Metal.jl, and the whole reason Lava became a
# weak dependency is that a second backend must not import the first one's.
#
# ── What Metal does not have, and what that simplifies ────────────────────────
#
# **No buffer usage flags.** A `VkBuffer` is created with every use it will be
# put to, which is why `constraintof`/`compatible`/`mergeconstraints` traffic in
# bitmasks on that backend. A `MTLBuffer` is bytes; use is decided at bind time.
# The only property that constrains sharing is the storage mode.
#
# **Unified memory.** On Apple silicon the CPU and GPU address the same bytes, so
# a `Shared` buffer needs no staging, no transfer queue and no barrier between a
# host write and a device read. `upload!` is a `copyto!`.
#
# **One timeline instead of fences plus semaphores.** A single `MTLSharedEvent`
# counts submissions, which is the shape `fence`/`passed`/`waitfor` already
# wanted — Vulkan's timeline semaphore is the same object under another name.

using Metal
using Metal: MtlArray
# `@device_override` is Metal.jl's own (not GPUCompiler's, which does not export
# one). It is how a backend replaces a host stub with a device instruction —
# the same mechanism Lava wraps as `@lava_device_override`, and the reason the
# device half must use it rather than a plain method: `KI.barrier` has a HOST
# method that errors, and a plain definition would shadow it so a host-side call
# reached a GPU instruction on the CPU.
using Metal: @device_override
# The presentation surface needs both: GLFW to own the window, and ObjectiveC's
# `@objc`/`id` to put a `CAMetalLayer` on its view. GLFW is already a Mantle
# dependency (the Vulkan backend's windows use it), and `ObjectiveC` comes in
# through Metal.jl rather than as a dependency of its own.
using GLFW
using Metal.ObjectiveC: @objc, id
const OCObject = Metal.ObjectiveC.Object
const MTL = Metal.MTL

include("device.jl")
include("memory.jl")
include("caps.jl")
include("kernelinterface.jl")
include("device_intrinsics.jl")
include("ka.jl")
include("raytracing.jl")
# Hardware traversal. After raytracing.jl: needs `MetalTLAS`.
include("trace.jl")
# The incremental acceleration structure + device-side traversal.
include("hwtlas.jl")
# Tracing AABB geometry: the query loop is MSL, the candidate is Julia.
include("procedural.jl")

include("graphics.jl")

# After `graphics.jl`: the mesh vocabulary names `Metal.MeshPtr` and the AIR
# intrinsics, and is a peer of the graphics hooks rather than part of them.
include("mesh.jl")
# Transient render targets. After graphics.jl: needs `mtlformat`.
include("images.jl")
# The presentation surface. After graphics.jl: needs `mtlformat` and the
# submission helpers (`framebuffer!`, `commit!`).
include("window.jl")

# Recording a plan into an indirect command buffer, which is what baking is here.
# Last, because it is the only file that needs every other one: the device, the
# pool's storage, the KA glue and the window are all in what a frame replays.
include("record.jl")

# ── What this backend does once, at load ──────────────────────────────────────
#
# Declared by core in `graph/backend.jl`. No pipeline thread and no `atexit`
# hook: neither half of Vulkan's has anything to reach for here.
Mantle.initbackend!() = register_backend!(; name = :metal, priority = 90) do
    Metal.functional() ? Metal.MetalBackend() : nothing
end

# ── What this backend does not have ───────────────────────────────────────────
#
# Declared by core in `runtime/backendhooks.jl`, and answered here rather than
# left undefined: a caller asking either of these is asking whether the feature
# EXISTS, and a name that is missing cannot say no. See that file.
#
# Metal.jl compiles its kernels through its own pipeline and keeps no cache of
# Mantle's, so there are no frozen entries to read.
use_frozen_kernels(version) = nothing

# `simdgroup_matrix` is 8x8 and the staged GEMM is emitted at 16 — its `@nexprs`
# unroll counts are literals derived from that extent, so it is not a parameter
# this backend could pass a different value for. `nothing` and not `8`: the
# question is which tile MANTLE has kernels at, and here the answer is none.
staged_gemm_tile() = nothing

# The compile cache is Metal.jl's, which is where it belongs; this is the runtime
# forwarding the question so a caller never has to name Metal — the same two lines
# the Vulkan backend writes over `Lava.frozen_stats`.
#
# Answered rather than left to core's `(; hits = 0, misses = 0)`. That default is
# indistinguishable from "nothing was compiled", and zero is a measurement, not an
# absence: a caller reading it off this backend would have concluded the cache was
# never touched. `compile_or_lookup` knows which of the two happened, so this
# backend can say.
kernelcompiles(::MetalDevice) = Metal.compile_stats()
resetkernelcompiles!(::MetalDevice) = (Metal.reset_compile_stats!(); nothing)
