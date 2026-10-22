# The portable ray-tracing pipeline description: the Julia functions serving as
# raygen, closest-hit, miss and any-hit, and the payload they exchange. Compiled
# pipelines are device objects, cached in `DeviceCaches.rt_pipelines`.

mutable struct RayTracingPipeline
    raygen_func::Any
    # One or more closest-hit functions. N > 1 builds N hit groups; an instance
    # picks one with its SBT record offset.
    closesthit_funcs::Tuple
    miss_func::Any
    anyhit_func::Any          # nothing = no any-hit shader
    payload_type::Symbol
    # true: closest-hit and miss are compiled with the raygen's argument signature
    # and see every buffer it sees (shading in closest-hit). false: they take no
    # arguments, as `hw_closesthit` / `hw_miss` do.
    chit_miss_take_args::Bool
end

# A function, tuple or vector of closest-hits, stored as a Tuple so each one's
# identity is part of the cache key.
normalise_chit(f) = (f,)
normalise_chit(t::Tuple) = t
normalise_chit(v::AbstractVector) = tuple(v...)

function RayTracingPipeline(; raygen, closest_hit, miss, any_hit=nothing,
                              payload_type::Symbol=:f32,
                              chit_miss_take_args::Bool=false)
    chits = normalise_chit(closest_hit)
    isempty(chits) && throw(ArgumentError("RayTracingPipeline: at least one closest_hit shader is required"))
    RayTracingPipeline(raygen, chits, miss, any_hit, payload_type,
                       chit_miss_take_args)
end

# Order: raygen, closest-hits, miss, any-hit; `shadertouches` answers in the same
# order. This is not the SBT group order (raygen, miss, hit groups).
shaders(p::RayTracingPipeline) =
    (p.raygen_func, p.closesthit_funcs..., p.miss_func,
     (p.anyhit_func === nothing ? () : (p.anyhit_func,))...)
