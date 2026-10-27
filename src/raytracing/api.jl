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
    build_blas(ctx, vertices, indices; allow_update = false) -> BLAS

A bottom-level structure over triangles: `vertices` a `Vector{NTuple{3,Float32}}`,
`indices` a `Vector{UInt32}` of zero-based vertex indices, three per triangle.

`allow_update = true` makes it refittable through `Raycore.refit!(blas, vertices)`.
Off by default because a driver may build a slower tree for it. Register it with
`push!(tlas, blas, transform)`; its hits carry no triangle record (`closest_hit`
returns the empty triangle), since only positions went in. Whether its hits are
opaque is up to each instance (`opaque` at `push!`, `Raycore.set_opaque!`).
"""
function build_blas end

"""
    build_blas_aabb(ctx, aabbs) -> BLAS

A bottom-level structure over procedural geometry: [`AABB`](@ref)s standing in for
primitives the traversal cannot intersect itself. A ray entering a box is answered
by the payload through `procedural_candidate` / `procedural_commit`
(`raytracing/accel.jl`). Declared in core so a caller spells it the same on every
backend.
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
    supports_procedural_traversal(x) -> Bool

Whether this backend can trace procedural (AABB) geometry, which is more than
building a BLAS over the boxes. A backend opts in by implementing
[`candidate_primitive_index`](@ref), [`candidate_object_ray`](@ref) and
[`commit_intersection!`](@ref). Asked up front so an unsupported scene is refused
by name, not as a `MethodError` inside a shader compile.
"""
supports_procedural_traversal(::Any) = false

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
