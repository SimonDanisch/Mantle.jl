# The ray-tracing pipeline: which Julia functions serve as raygen,
# closest-hit, miss and any-hit, and what payload they exchange.
#
# ONE struct, shared by every backend — not an abstract type with a copy per
# backend. Nothing in it is a driver object: the fields are functions, a tuple,
# a symbol and a flag, which is exactly what `GraphicsPipeline` next door is.
#
# It did not look that way at first, because it carried a `PIPELINE_CACHE` field
# holding compiled Vulkan pipelines. That field was the anomaly, not a reason to
# split the type: the graphics side has always cached in `DeviceCaches`, and a
# compiled pipeline is a device object — it must not outlive a `reset_device!`,
# and two identical pipelines should not compile twice. The cache moved to
# `DeviceCaches.rt_pipelines` and the description became portable by itself.
#
# So a caller describes a pipeline and the runtime owns compiling it. There is
# nothing here for a user to manage.

mutable struct RayTracingPipeline
    # User-provided Julia functions
    raygen_func::Any
    # Tuple of one-or-more closest-hit functions.  When the tuple has length
    # N > 1 the SBT is built with N hit groups; each HWTLAS instance picks a
    # group via `instanceShaderBindingTableRecordOffset`.  Per-material chit
    # shaders + SER live on this path.
    closesthit_funcs::Tuple
    miss_func::Any
    anyhit_func::Any          # nothing = no any-hit shader
    payload_type::Symbol
    # When true, the closest-hit and miss shaders are compiled with the same
    # BDA argument signature as the raygen — they receive the raygen's args
    # as function parameters and can read every buffer/queue the raygen sees.
    # Required for the pbrt-v4 OptiX pattern (shading happens in closesthit,
    # raygen just calls traceRay).  Default `false` keeps the legacy contract
    # where chit/miss take no args (the `hw_closesthit` / `hw_miss` pattern
    # used by `trace_closest_hits!`).
    chit_miss_take_args::Bool
end

# Normalise `closest_hit` to a Tuple: accept a single function or a
# tuple/vector of functions. Stored as a Tuple so each chit's identity
# participates in the dispatch caches and pin walks.
_normalise_chit(f) = (f,)
_normalise_chit(t::Tuple) = t
_normalise_chit(v::AbstractVector) = tuple(v...)

function RayTracingPipeline(; raygen, closest_hit, miss, any_hit=nothing,
                              payload_type::Symbol=:f32,
                              chit_miss_take_args::Bool=false)
    chits = _normalise_chit(closest_hit)
    isempty(chits) && throw(ArgumentError("RayTracingPipeline: at least one closest_hit shader is required"))
    RayTracingPipeline(raygen, chits, miss, any_hit, payload_type,
                       chit_miss_take_args)
end
