"""
Backends are types, so a capability a backend lacks is a missing method that names
what was unavailable, not a runtime check. The lowering functions `stages`,
`access` and `layout` are declared here without methods; backends add them.
"""
abstract type Backend end

# The `API` suffix keeps the markers from shadowing the Vulkan.jl and Metal.jl
# package names inside this module.
struct VulkanAPI <: Backend end
struct MetalAPI <: Backend end
struct WebGPUAPI <: Backend end

"""
Execution on HIP through AMDGPU.jl (`ext/MantleROCmExt.jl`). Declared in core
because its `needs_transition` answer and the `Device(api)` spelling are core vocabulary.
"""
struct ROCmAPI <: Backend end

"""Whether an emitted buffer dependency covers every buffer with the same
usage scopes, rather than only its named resource. Barrier elimination is core's
decision; these methods describe the scope of the native barrier verb."""
global_buffer_barriers(::Backend) = false
global_buffer_barriers(::VulkanAPI) = true
global_buffer_barriers(::MetalAPI) = true

function stages end
function access end
function layout end

"""WebGPU has no barriers."""
needs_transition(::WebGPUAPI, ::ResourceKind, before::Type, after::Type) = false

"""
A single HIP stream needs no barriers: work runs in issue order and a kernel's
writes are visible to the next kernel on the stream. Valid only while
`ROCmDevice` launches on one stream; several streams would need `true` and events.
"""
needs_transition(::ROCmAPI, ::ResourceKind, before::Type, after::Type) = false
