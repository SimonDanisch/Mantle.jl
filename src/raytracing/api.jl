# Hardware ray tracing verbs, implemented per backend. The driver's instance record
# is a per-backend ABI; callers and kernels write `Raycore.InstanceRecord`s, and each
# backend converts them at build and refit (`raytracing/instances.jl`).

"""
    build_accel!(f, queue) -> f(ctx)

Run `f(ctx)` with the backend's build context ([`AccelBuildContext`](@ref) on
Vulkan) and return its result. The builds recorded through `ctx` have completed and
their scratch is released when this returns.
"""
function build_accel! end

"""
    build_blas(ctx, vertices, indices; opaque = true, allow_update = false) -> BLAS

A bottom-level structure over triangles: `vertices` a `Vector{NTuple{3,Float32}}`,
`indices` a `Vector{UInt32}` of zero-based vertex indices, three per triangle.

`allow_update = true` makes it refittable through `Raycore.refit!(blas, vertices)`.
Off by default because a driver may build a slower tree for it. Register it with
`push!(tlas, blas, transform)`; its hits carry no triangle record (`closest_hit`
returns the empty triangle), since only positions went in.
"""
function build_blas end

"""
    build_blas_aabb(ctx, aabbs; opaque = true) -> BLAS

A bottom-level structure over procedural geometry: [`AABB`](@ref)s standing in for
primitives the traversal cannot intersect itself. A ray entering a box is answered
by the payload through `procedural_candidate` / `procedural_commit`
(`raytracing/accel.jl`). Declared in core so a caller spells it the same on every
backend. `opaque` is a hint a backend may ignore.
"""
function build_blas_aabb end

"""
    refit_tlas!(ctx, tlas, ...)

Update `tlas` in place from rewritten instance records, same instance count;
`ctx` is the backend's build context or device. It does not re-cluster: instances
that move far degrade traversal until the structure is rebuilt.
"""
function refit_tlas! end

"""
    trace_rays!(queue, pipeline, tlas, args...; width, height, depth = 1)

Trace through a [`RayTracingPipeline`](@ref), passing `args` to the raygen shader.
"""
function trace_rays! end

"""
    trace_rays_indirect!(queue, pipeline, tlas, args...; n_rays)

[`trace_rays!`](@ref) with the ray count read from device memory.
"""
function trace_rays_indirect! end

"""
    trace!(g, pipeline, accel, args, ndrange; name = nothing) -> pass

Trace as a graph pass, ordered by what it reads and writes; the ray-tracing
counterpart of [`dispatch!`](@ref). `ndrange` is a ray count or a
[`DeviceRange`](@ref), and `args` live in the plan's argument memory, which makes a
traced plan bakeable (see [`Trace`](@ref)). Accesses are inferred from `args` and
from what every shader of the pipeline captures.

    Mantle.trace!(g, rt_pipeline, accel, (queue_in, queue_out, film),
                  Mantle.DeviceRange(n_rays))
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
    # Every usage of this pass is made at the ray-tracing stage (`Traced`), so
    # retag them once all are declared.
    for (i, (id, U)) in enumerate(p.usages)
        p.usages[i] = id => traced(U)
    end
    return p
end

"""
    shaders(pipeline) -> Tuple

Every shader a trace runs, as closures; [`trace!`](@ref) declares what they
capture through [`shadertouches`](@ref).
"""
function shaders end

"""
    shadertouches(device, pipeline, args) -> Vector{Touch}

What each of `shaders(pipeline)` does to its own captured state, in the same
order. A shader that captures nothing answers `NOTOUCH`.
"""
function shadertouches end

"""
    trace_closest_hits!(results, rays, accel, n_rays; cull_mask = 0xff)

Trace `rays` and write the nearest intersection of each into `results`. Needs no
user closest-hit shader.
"""
function trace_closest_hits! end

"""
    trace_closest_hits_indirect!(results, rays, accel, n_rays)

[`trace_closest_hits!`](@ref) with a device-supplied ray count.
"""
function trace_closest_hits_indirect! end

"""
    trace_closest_hits_anyhit!(results, rays, accel, n_rays, extra_args...)

[`trace_closest_hits!`](@ref) with the any-hit shader run, for alpha-tested
geometry where the nearest *opaque* hit is not the nearest hit.
"""
function trace_closest_hits_anyhit! end

"""
    trace_closest_hits_anyhit_indirect!(results, rays, accel, n_rays, extra_args...)

[`trace_closest_hits_anyhit!`](@ref) with a device-supplied ray count.
"""
function trace_closest_hits_anyhit_indirect! end

"""
    set_anyhit_pipeline!(accel, anyhit_func, raygen_func)

Set the any-hit pipeline used when tracing against `accel`.
"""
function set_anyhit_pipeline! end

"""
    supports_rt_pipeline(x) -> Bool

Whether tracing can use a ray-tracing pipeline (shader binding table plus
trace-rays command) rather than an inline ray query in a compute kernel. Both
traverse in hardware; Metal has no SBT and calls its intersector from the shading
kernel. `false` unless a backend opts in. Accepts a device, a backend, an `HWTLAS`
or an [`AdaptedAccel`](@ref).
"""
supports_rt_pipeline(::Any) = false
# A device answers what its backend answers, as `supports_graphics` does.
supports_rt_pipeline(dev::Device) = supports_rt_pipeline(backend(dev))
# Procedural geometry forces the inline path: a pipeline would need an
# intersection shader in a `PROCEDURAL_HIT_GROUP`, which is not wired.
supports_rt_pipeline(a::AdaptedAccel) =
    a.procedural === nothing && supports_rt_pipeline(a.hwtlas)

"""
    supports_procedural_traversal(x) -> Bool

Whether this backend can trace procedural (AABB) geometry, which is more than
building a BLAS over the boxes. A backend opts in by implementing
[`candidate_primitive_index`](@ref), [`candidate_object_ray`](@ref) and
[`commit_intersection!`](@ref). Asked up front so an unsupported scene is refused
by name, not as a `MethodError` inside a shader compile.
"""
supports_procedural_traversal(::Any) = false
supports_procedural_traversal(dev::Device) = supports_procedural_traversal(backend(dev))

# Callers usually hold the adapted accel; the `::Any` default would answer a wrong `false`.
supports_procedural_traversal(a::AdaptedAccel) = supports_procedural_traversal(a.hwtlas)

"""
    supports_hwtlas(x) -> Bool

Whether this device builds an [`HWTLAS`](@ref) and traverses it in hardware from an
ordinary kernel (Vulkan ray query, Metal intersector). Ask before
`HWTLAS{Tri}(backend)`. Accepts a device or its KernelAbstractions backend; `false`
unless a backend opts in.
"""
supports_hwtlas(::Any) = false

# No pinning verb: a submission's payload holds what its command buffers pinned
# (`Outstanding` in `graph/submission.jl`).
