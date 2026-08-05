"""
Backends are types, so a capability a backend lacks is a missing method rather
than a runtime check. Compute-only backends get no `render!`, WebGPU gets no
barriers, and each failure names the thing that was not available.

The lowering functions are declared here without methods. Extensions add them:
`stages`, `access` and `layout` for Vulkan, an encoder kind and a residency hint
for Metal, a creation flag for WebGPU.
"""
abstract type Backend end

struct Vulkan <: Backend end
struct Metal <: Backend end
struct WebGPU <: Backend end

function stages end
function access end
function layout end

"""WebGPU has no barriers. The method returning false is the honest statement of
that, not an omission."""
needs_transition(::WebGPU, ::ResourceKind, before::Type, after::Type) = false
