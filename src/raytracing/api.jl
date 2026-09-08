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

    Mantle.trace!(p, rt_pipeline, accel, (queue_in, queue_out, film),
                  Mantle.DeviceRange(n_rays))

Declare the resources with `use` as for any other pass. Nothing here inspects
`args` for them: an argument is how the work reaches its data, a declaration is
what the graph orders on, and a trace needs both said.
"""
function trace!(p, pipeline, accel, args, ndrange)
    refuserefs(args)
    n = countresource(ndrange)
    n === nothing || indirectcount!(p, n)
    push!(dispatches(passof(p)), Trace(pipeline, accel, args, ndrange))
end

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
supports_rt_pipeline(a::AdaptedAccel) = supports_rt_pipeline(a.hwtlas)

"""
    pin!(owner, x)

Hold `x` for as long as `owner` — a closed command buffer, a recording, or the
emitter writing into one — can still run. A backend implements it for what its
commands can name: arrays, acceleration structures, descriptor pools.
"""
function pin! end

"""
    blases(tlas)

The bottom-level acceleration structures `tlas` instances. A backend answers
for its top-level type.
"""
function blases end

"""
    pintrace!(owner, tlas)

The acceleration structures a trace reads, held for as long as it can run: the
top level, and every bottom level it instances. A trace walks from the one into
the others, so a BLAS swapped out underneath a running trace (`Raycore.sync!`
does that) must stay alive until the trace has passed, and pinning the top
level alone does not say so.

Core over two backend answers, `pin!` and `blases`. It used to be written out
four times in the Vulkan backend — once in the recorded walk and once in each
unmodelled path — and a Metal backend would have written a fifth.
"""
function pintrace!(owner, tlas)
    pin!(owner, tlas)
    for blas in blases(tlas)
        pin!(owner, blas)
    end
    return nothing
end
