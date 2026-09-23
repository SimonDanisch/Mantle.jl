# Hardware ray tracing.
#
# Declared here, implemented per backend. Vulkan reaches this through
# `VK_KHR_ray_tracing_pipeline` and `VK_KHR_ray_query`, Metal through
# `MTLAccelerationStructure` and intersection functions, and the shapes of the
# two are close enough that the verbs below are the same on both: build an
# acceleration structure, refit it when geometry moves, trace against it.
#
# What is NOT here is the instance record. Its bytes are an ABI —
# `VkAccelerationStructureInstanceKHR` is laid out differently from
# `MTLAccelerationStructureInstanceDescriptor` — so each backend keeps its own,
# and what they share is [`Mat3x4f`](@ref), the transform inside it.

"""
    build_accel!(ctx, geometry) -> HWTLAS

Build an acceleration structure over `geometry`.

`ctx` is an [`AccelBuildContext`](@ref), which carries the scratch memory so a
sequence of builds does not allocate one each.
"""
function build_accel! end

"""
    refit_tlas!(tlas)

Update `tlas` in place for instance transforms that have changed.

A refit is much cheaper than a rebuild and is the reason a HWTLAS is worth
holding across frames, but it does not re-cluster: geometry that moves far
enough degrades trace performance until it is rebuilt.
"""
function refit_tlas! end

"""
    trace_rays!(pipeline, tlas, rays)

Trace `rays` against `tlas` through a [`RayTracingPipeline`](@ref).
"""
function trace_rays! end

"""
    trace_rays_indirect!(pipeline, tlas, buffer)

Trace with the ray count read from the device, as for
[`draw_indirect_in_pass!`](@ref).
"""
function trace_rays_indirect! end

"""
    trace!(pass, pipeline, accel, args, ndrange)

Trace as part of a pass, so what it reads and writes is what orders it against
everything else.

The ray-tracing counterpart of [`dispatch!`](@ref), and deliberately the same
shape: `pipeline` stands where a kernel stands, `ndrange` is a ray count and may
be a [`DeviceRange`](@ref) for one that only exists on the device, and `args` are
laid out in the plan's argument memory at compile — which is what makes a traced
plan bakeable. See [`Trace`](@ref) for why that last part is the point.

    Mantle.trace!(g, rt_pipeline, accel, (queue_in, queue_out, film),
                  Mantle.DeviceRange(n_rays))

Nothing is declared by hand here either. A trace has more shaders than a dispatch
has kernels — a raygen, a closest-hit per material, a miss, sometimes an any-hit
— so what the pass touches is the union over all of them, and what each one does
to what it CAPTURED counts too: a per-material closest-hit closes over the
arrays of the material it shades, and those are read by the trace as surely as
its arguments are.
"""
function trace!(g::Graph, pipeline, accel, args::Tuple, ndrange; name = nothing)
    refuserefs(args)
    p = newpass(g, name === nothing ? "trace" : String(name), :compute)
    push!(passes(g), p)
    h = handle(g, p)
    declare!(h, args, kerneltouches(g.dev, pipeline, args, ndrange, nothing))
    declare!(h, shaders(pipeline), shadertouches(g.dev, pipeline, args))
    n = countresource(ndrange)
    n === nothing || indirectcount!(h, n)
    push!(dispatches(p), Trace(pipeline, accel, args, ndrange))
    # Shader accesses from a ray-tracing pipeline are a stage of their own — see
    # `Traced`. After the declarations, for the reason `compute!` said it: every
    # usage this pass has belongs to the trace.
    for (i, (id, U)) in enumerate(p.usages)
        p.usages[i] = id => traced(U)
    end
    return p
end

"""
    shaders(pipeline) -> Tuple

Every shader a trace runs, as the closures they are. What they capture is
declared from this — see [`shadertouches`](@ref) — and it is also the list the
backend pins.
"""
function shaders end

"""
    shadertouches(device, pipeline, args) -> Vector{Touch}

What each of `shaders(pipeline)` does to ITS OWN captured state, in the same
order. A shader that captures nothing answers `NOTOUCH` and declares nothing.
"""
function shadertouches end

"""
    trace_closest_hits!(pipeline, tlas, rays, hits)

Trace `rays` and write the nearest intersection of each into `hits`.

The common case, and separate from [`trace_rays!`](@ref) because it needs no
closest-hit shader: the backend can use a ray query rather than a full pipeline.
"""
function trace_closest_hits! end

"""
    trace_closest_hits_indirect!(pipeline, tlas, buffer, hits)

[`trace_closest_hits!`](@ref) with a device-supplied ray count.
"""
function trace_closest_hits_indirect! end

"""
    trace_closest_hits_anyhit!(pipeline, tlas, rays, hits)

[`trace_closest_hits!`](@ref) with the any-hit shader run, for alpha-tested
geometry where the nearest *opaque* hit is not the nearest hit.
"""
function trace_closest_hits_anyhit! end

"""
    trace_closest_hits_anyhit_indirect!(pipeline, tlas, buffer, hits)

[`trace_closest_hits_anyhit!`](@ref) with a device-supplied ray count.
"""
function trace_closest_hits_anyhit_indirect! end

"""
    set_anyhit_pipeline!(tlas, pipeline)

Set the any-hit pipeline used when tracing against `tlas`.
"""
function set_anyhit_pipeline! end

"""
    supports_rt_pipeline(x) -> Bool

Whether tracing can go through a ray-tracing PIPELINE — a shader binding table
plus an indirect trace-rays command — as opposed to an inline ray query in an
ordinary compute kernel.

`false` by default, so a backend opts in. Both forms are hardware traversal;
what differs is who runs the hit shaders. Vulkan can hand a whole SBT to the
driver and have it invoke per-material closest-hit shaders. Metal traverses
through a linked MSL function called from the shading kernel, so the kernel
keeps control and there is no SBT — the same hardware work, dispatched the
other way round.

Accepts a backend, an `HWTLAS`, or an [`AdaptedAccel`](@ref), because the
callers that need to ask have one of those to hand and not always the same one.
"""
supports_rt_pipeline(::Any) = false
# PROCEDURAL geometry forces the inline path, whatever the backend says.
#
# The two ray paths answer a box by different mechanisms: an inline ray query
# does it in the traversal loop with `OpRayQueryGenerateIntersectionKHR`, while
# an RT pipeline needs an INTERSECTION SHADER in a `PROCEDURAL_HIT_GROUP` —
# a different instruction in a different stage, bound through the shader binding
# table. Only the first is wired here, so an accel carrying a procedural payload
# says no and is traced inline. A triangles-only accel is unaffected.
supports_rt_pipeline(a::AdaptedAccel) =
    a.procedural === nothing && supports_rt_pipeline(a.hwtlas)

#
# There is no pinning verb here. "Hold this Julia object until a submission
# completes" is `Outstanding(token, payload, tag)` in `graph/submission.jl`,
# whose payload holds the closed command buffers the submission carried, with
# their pins and scratch. One register of that fact, not one per backend.
#
# `pintrace!` was the one that reached core, and only because an acceleration
# structure travels as a kernel argument and the graph reads no BLAS edge out of
# one, so `Raycore.sync!` can swap a BLAS out
# from under a running trace. Phase 2.4 declares them instead, and then there is
# nothing left for a pin to guard.
