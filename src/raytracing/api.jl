# Hardware ray tracing verbs, implemented per backend. The driver's instance record
# is a per-backend ABI; callers and kernels write `Raycore.InstanceRecord`s, and each
# backend converts them at build and refit (`raytracing/instances.jl`).

"""
    build_accel!(f, queue) -> f(ctx)

Run `f(ctx)` with an [`AccelBuildContext`](@ref) over `queue` and return its result.
The builds recorded through `ctx` run after the work submitted to `queue` before, and
have completed with their scratch released when this returns.
"""
function build_accel! end

"""
    AccelStructure

A backend's built acceleration structure, bottom level (`build_blas`,
`build_blas_aabb`) or top level (`build_tlas`).
"""
abstract type AccelStructure end

# What a build reads, as device arrays ([`inputarray`](@ref)) in core's types: the
# validation and the uploads before a backend's `buildnative!` are written once.
struct TriangleInput{V,I}
    vertices::V   # `NTuple{3,Float32}` positions
    indices::I    # zero-based `UInt32` vertex indices, three per triangle
end
struct BoxInput{A}
    boxes::A      # `AABB`s
end
struct InstanceInput{R,B}
    records::R    # driver instance records (`nativerecords`)
    n::Int
    blases::B     # what the records name
end

"""
    inputarray(device, T, n) -> device array

`n` uninitialised elements of `T` a build reads in place: vertices, indices or boxes a
kernel writes before `build_blas`/`build_blas_aabb`, or before `Raycore.refit!`.
"""
inputarray(device, ::Type{T}, n::Integer) where {T} = devicearray(device, T, Int(n))

"""The device array a build reads `data` from: `data` itself when it is one
([`inputarray`](@ref)), else an upload."""
buildinput(device, data::AbstractVector) =
    isdevicearray(data) ? data : copyto!(inputarray(device, eltype(data), length(data)), data)

"""
    buildnative!(ctx, input, allow_update) -> AccelStructure

Build the backend's structure over `input` (a `TriangleInput`, `BoxInput` or
`InstanceInput` of device arrays), refittable with `allow_update`. Its inputs are
validated; the structure keeps the arrays it reads.
"""
function buildnative! end

"""
    refitnative!(ctx, as)

Refit `as`, built refittable, in place from its inputs, which the caller rewrote.
"""
function refitnative! end

"""
    refittable(as) -> Bool

Whether `as` was built with `allow_update = true`.
"""
function refittable end

"""
    refitinput(as) -> device array

The input a refit of `as` reads: a bottom-level structure's vertices or boxes, a
top-level structure's instance records.
"""
function refitinput end

deviceof(ctx::AccelBuildContext) = deviceof(ctx.bq)

"""
    build_blas(ctx, vertices, indices; allow_update = false) -> AccelStructure

A bottom-level structure over triangles: `vertices` `NTuple{3,Float32}` positions,
`indices` zero-based `UInt32` vertex indices, three per triangle. A device array
([`inputarray`](@ref)) is read in place, a host vector uploaded.

`allow_update = true` makes it refittable through `Raycore.refit!(blas, vertices)`.
Off by default because a driver may build a slower tree for it. Register it with
`push!(tlas, blas, transform)`; its hits carry no triangle record (`closest_hit`
returns the empty triangle), since only positions went in. Whether its hits are
opaque is up to each instance (`opaque` at `push!`, `Raycore.set_opaque!`).
"""
function build_blas(ctx::AccelBuildContext, vertices::AbstractVector{NTuple{3,Float32}},
                    indices::AbstractVector{UInt32}; allow_update::Bool = false)
    isempty(vertices) && throw(ArgumentError("build_blas: no vertices to build over"))
    !isempty(indices) && length(indices) % 3 == 0 || throw(ArgumentError(
        "build_blas: $(length(indices)) indices are not a whole, nonzero number of triangles"))
    d = deviceof(ctx)
    return buildnative!(ctx, TriangleInput(buildinput(d, vertices), buildinput(d, indices)), allow_update)
end

"""
    build_blas_aabb(ctx, aabbs; allow_update = false) -> AccelStructure

A bottom-level structure over procedural geometry: [`AABB`](@ref)s standing in for
primitives the traversal cannot intersect itself. A ray entering a box is answered
by the payload through `procedural_candidate` / `procedural_commit`
(`raytracing/accel.jl`). A device array ([`inputarray`](@ref)) is read in place,
and refit from with `Raycore.refit!(blas, boxes)` after `allow_update = true`.
"""
function build_blas_aabb(ctx::AccelBuildContext, aabbs::AbstractVector{AABB}; allow_update::Bool = false)
    d = deviceof(ctx)
    supports_procedural_traversal(d) || throw(ArgumentError(
        "build_blas_aabb: $(devicename(d)) cannot trace procedural geometry, so a box " *
        "structure would never be intersected (`supports_procedural_traversal`)"))
    isempty(aabbs) && throw(ArgumentError("build_blas_aabb: no boxes to build over"))
    return buildnative!(ctx, BoxInput(buildinput(d, aabbs)), allow_update)
end

"""
    build_tlas(ctx, records, n; allow_update = false, blases) -> AccelStructure

A top-level structure over the first `n` driver instance records of `records`
([`nativerecords`](@ref)), which name the bottom-level structures `blases` by
[`geometryvalue`](@ref). The structure owns `records` from then on and holds
`blases` for every trace of it. `allow_update = true` makes it refittable.
"""
function build_tlas(ctx::AccelBuildContext, records, n::Integer; allow_update::Bool = false,
                    blases = blastype(deviceof(ctx))[])
    0 <= n <= length(records) || throw(ArgumentError(
        "build_tlas: $n instances, but $(length(records)) records"))
    return buildnative!(ctx, InstanceInput(records, Int(n), blases), allow_update)
end

function refusefixed(as, operation)
    refittable(as) || throw(ArgumentError(
        "$operation: the structure was built with allow_update = false; rebuild it, " *
        "or build it with allow_update = true"))
    return nothing
end

"""
    refit_tlas!(ctx, tlas)

Update `tlas`, built with `allow_update = true`, in place from the records it was
built from, which the caller rewrote. It does not re-cluster: instances that move
far degrade traversal until the structure is rebuilt.
"""
function refit_tlas!(ctx::AccelBuildContext, tlas::AccelStructure)
    refusefixed(tlas, "refit_tlas!")
    return refitnative!(ctx, tlas)
end

# What a refit writes into a structure's input: positions as `NTuple{3,Float32}`, boxes as they are.
refitdata(::Type{NTuple{3,Float32}}, v::AbstractVector{NTuple{3,Float32}}) = v
refitdata(::Type{NTuple{3,Float32}}, v::AbstractVector{<:StaticVector{3}}) =
    NTuple{3,Float32}[Tuple(Float32.(p)) for p in v]
refitdata(::Type{AABB}, boxes::AbstractVector{AABB}) = boxes
refitdata(::Type{T}, x::AbstractVector) where {T} = throw(ArgumentError(
    "refit!: the structure was built over $T, not $(eltype(x))"))

"""
    Raycore.refit!(blas::AccelStructure, input) -> blas

Move a bottom-level structure built with `allow_update = true` in place: `input` holds
the new vertices (positions) or boxes, as many as it was built over, on the host or
the device. A refit keeps the topology and does not re-cluster; rebuild for a
different mesh or after large motion. Refit every TLAS naming it afterwards.
"""
function Raycore.refit!(blas::AccelStructure, input::AbstractVector)
    refusefixed(blas, "refit!")
    target = refitinput(blas)
    data = refitdata(eltype(target), input)
    length(data) == length(target) || throw(ArgumentError(
        "refit!: $(length(data)) elements for a structure built over $(length(target)); " *
        "a refit keeps the topology, rebuild for a different one"))
    data === target || copyto!(target, data)
    build_accel!(batchqueue(deviceof(blas))) do ctx
        refitnative!(ctx, blas)
    end
    return blas
end

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
