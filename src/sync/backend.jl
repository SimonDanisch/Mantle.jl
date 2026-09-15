"""
Backends are types, so a capability a backend lacks is a missing method rather
than a runtime check. Compute-only backends get no `render!`, WebGPU gets no
barriers, and each failure names the thing that was not available.

The lowering functions are declared here without methods. Extensions add them:
`stages`, `access` and `layout` for Vulkan, an encoder kind and a residency hint
for Metal, a creation flag for WebGPU.
"""
abstract type Backend end

# `…API`, not the bare API name, and the suffix is doing real work: Mantle now
# DEPENDS on Vulkan.jl, and a Metal backend will depend on Metal.jl. A marker
# called `Vulkan` shadows the package inside this module — `using Vulkan` then
# either fails to bind or, worse, `vkformat(::VulkanAPI, …)` silently takes a
# module. It cost an afternoon once; naming it out is cheaper than remembering.
struct VulkanAPI <: Backend end
struct MetalAPI <: Backend end
struct WebGPUAPI <: Backend end

"""
Execution on HIP, through AMDGPU.jl — the backend in `ext/MantleROCmExt.jl`.

A marker here rather than in the extension, beside `WebGPUAPI`, which has no
backend in tree either: what a marker is for is the `needs_transition` answer
below and the `Device(api)` spelling, and both are core vocabulary. It also
keeps the two ROCm facts that are not AMDGPU.jl's — that this is one in-order
stream, and what that means for barriers — in the file that states the same
thing about the host and about WebGPU.
"""
struct ROCmAPI <: Backend end

"""
Execution on the host, through KernelAbstractions' CPU backend.

Compute only: no `render!`, no `Window`, no `Surface` — missing methods, which is
what this file's opening paragraph says a backend should do with a capability it
lacks.

Named `HostAPI` and not `CPU` because `KernelAbstractions.CPU` already exists and
anything doing `using Mantle, KernelAbstractions` would then have to qualify one
of them. `copy!`, `update!` and `overlaps` are all kept out of the export list
here for the same reason; a fourth clash would be careless rather than unlucky.
The `API` suffix matches its three siblings above.
"""
struct HostAPI <: Backend end

function stages end
function access end
function layout end

"""WebGPU has no barriers. The method returning false is the honest statement of
that, not an omission."""
needs_transition(::WebGPUAPI, ::ResourceKind, before::Type, after::Type) = false

"""
Neither has the host: a kernel launch has completed by the time it returns, so
the scheduled order IS the synchronisation and there is nothing to emit between
two passes.

This is not the same claim as "ordering does not matter here". `Barriers` still
runs on the host, asking this; answering `false` derives nothing, which is what
a backend whose `passbarriers` would lower every transition to nothing should
say. If host passes are ever run concurrently, this is where the join goes.
"""
needs_transition(::HostAPI, ::ResourceKind, before::Type, after::Type) = false

"""
Nor has a single HIP stream. Work submitted to one stream runs in issue order
and a kernel's writes are visible to the next kernel on that stream, so the
scheduled order IS the synchronisation — the host backend's argument, with a
queue instead of a calling thread.

It is a claim about ONE stream, which is what `ROCmDevice` holds and what
AMDGPU.jl launches on. A backend that spread a plan's passes over several
streams to overlap them would have to answer `true` here and emit events
between them, and that is the change this method would have to be revisited by.
"""
needs_transition(::ROCmAPI, ::ResourceKind, before::Type, after::Type) = false
