# Ray queries: Metal's `intersection_query<instancing, triangle_data>` and
# `intersector<instancing, triangle_data>`.
#
# MSL's `intersection_query<>` and `intersector<>` are C++ templates in the
# `metal_raytracing` header, and the header is a thin layer over AIR intrinsics:
# compiled, every member function is one `air.*_intersection_query.instancing.
# triangle_data` call on an opaque thread-private query object. Called here directly,
# the traversal loop is ordinary Julia, as Lava's ray query is on Vulkan.
#
# The names and signatures are read off AIR the Metal compiler emits for MSL (macOS 26,
# AIR 2.9); Apple documents MSL, not AIR. Each intrinsic is called through
# `Base.llvmcall` of a module declaring it, with the structure passed as its
# `gpuResourceID` reinterpreted as a device pointer.

const RAY_QUERY = "intersection_query.instancing.triangle_data"

"""
    IntersectionQuery

A ray query in flight: the thread-private object [`intersection_query`](@ref)
allocates. Start it with [`query_reset!`](@ref), step it with [`query_next!`](@ref),
read the candidate and the committed intersection, and end it with
[`query_release!`](@ref).
"""
struct IntersectionQuery
    ptr::Core.LLVMPtr{UInt8,0}
end

# What `candidate_type` and `committed_type` answer (`intersection_type`).
const INTERSECTION_NONE         = UInt32(0)
const INTERSECTION_TRIANGLE     = UInt32(1)
const INTERSECTION_BOUNDING_BOX = UInt32(2)

# `forced_opacity`: leave the geometry's own flag, or force every triangle opaque or
# non-opaque for this query.
const FORCED_OPACITY_NONE       = UInt32(0)
const FORCED_OPACITY_OPAQUE     = UInt32(1)
const FORCED_OPACITY_NON_OPAQUE = UInt32(2)

"""
    rayquerymodule(name, ret, params, body) -> String

An LLVM module declaring `air.<name>` and defining an always-inlined `entry` with
`params` and `body`, for `Base.llvmcall((module, "entry"), ...)`.
"""
function rayquerymodule(name, ret, params, body)
    decl = join(params, ", ")
    args = join(("$t %a$(i - 1)" for (i, t) in enumerate(params)), ", ")
    return """
        declare $ret @air.$(name)($decl)
        define $ret @entry($args) #0 {
        $body
        }
        attributes #0 = { alwaysinline }
        """
end

"""
    intersection_query() -> IntersectionQuery

Allocate a query object in thread memory.
"""
function intersection_query end

let name = "allocate_$RAY_QUERY"
    ir = rayquerymodule(name, "ptr", String[], "  %q = call ptr @air.$name()\n  ret ptr %q")
    @eval @inline intersection_query() =
        IntersectionQuery(Base.llvmcall(($ir, "entry"), Core.LLVMPtr{UInt8,0}, Tuple{}))
end

"""
    query_reset!(q, origin, direction, tmin, tmax, accel, mask, forced_opacity, accept_any)

Start `q` on a ray against the instance acceleration structure whose `gpuResourceID`
is `accel`. `mask` is ANDed with each instance's mask; an instance sharing no bit is
not traversed. `accept_any` ends the traversal at the first committed intersection
rather than the nearest.
"""
function query_reset! end

# The ten integers after the structure, in order: instance mask, front-facing winding,
# triangle cull mode, geometry cull mode, opacity cull mode, forced opacity, the
# geometry types present (1 triangles, 2 boxes), and three values every MSL query
# passes as -1, -1, 0. The first flag is false in every MSL query; the second is
# `accept_any_intersection`.
let name = "reset_$RAY_QUERY"
    ir = """
        declare void @air.$name(ptr, <3 x float>, <3 x float>, float, float, ptr addrspace(1), i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, i1)
        define void @entry(ptr %q, <3 x float> %o, <3 x float> %d, float %tmin, float %tmax, ptr addrspace(1) %accel, i32 %mask, i32 %forced, i8 %anyb) #0 {
          %any = trunc i8 %anyb to i1
          call void @air.$name(ptr %q, <3 x float> %o, <3 x float> %d, float %tmin, float %tmax, ptr addrspace(1) %accel, i32 %mask, i32 0, i32 0, i32 0, i32 0, i32 %forced, i32 3, i32 -1, i32 -1, i32 0, i1 false, i1 %any)
          ret void
        }
        attributes #0 = { alwaysinline }
        """
    V3 = NTuple{3,VecElement{Float32}}
    @eval @inline function query_reset!(q::IntersectionQuery, o::NTuple{3,Float32}, d::NTuple{3,Float32},
                                        tmin::Float32, tmax::Float32, accel::UInt64, mask::UInt32,
                                        forced_opacity::UInt32, accept_any::Bool)
        Base.llvmcall(($ir, "entry"), Cvoid,
                      Tuple{Core.LLVMPtr{UInt8,0}, $V3, $V3, Float32, Float32, Core.LLVMPtr{UInt8,1},
                            UInt32, UInt32, Bool},
                      q.ptr, map(VecElement, o), map(VecElement, d), tmin, tmax,
                      reinterpret(Core.LLVMPtr{UInt8,1}, accel), mask, forced_opacity, accept_any)
        return nothing
    end
end

"""
    query_next!(q) -> Bool

Advance `q` to the next candidate it cannot decide alone (a non-opaque triangle or a
bounding box); `false` once the traversal is done.
"""
function query_next! end

let name = "next_$RAY_QUERY"
    ir = """
        declare i1 @air.$name(ptr)
        define i8 @entry(ptr %a0) #0 {
          %r = call i1 @air.$name(ptr %a0)
          %z = zext i1 %r to i8
          ret i8 %z
        }
        attributes #0 = { alwaysinline }
        """
    @eval @inline query_next!(q::IntersectionQuery) =
        Base.llvmcall(($ir, "entry"), Bool, Tuple{Core.LLVMPtr{UInt8,0}}, q.ptr)
end

# Calls that take the query and return nothing: `query_abort!`, `query_release!` (end
# the query and free its object) and `commit_triangle!` (accept the current triangle
# candidate).
for (jl, air) in ((:query_abort!, "abort"), (:query_release!, "deallocate"),
                  (:commit_triangle!, "commit_triangle_intersection"))
    name = "$(air)_$RAY_QUERY"
    ir = rayquerymodule(name, "void", ["ptr"], "  call void @air.$name(ptr %a0)\n  ret void")
    @eval @inline function $jl(q::IntersectionQuery)
        Base.llvmcall(($ir, "entry"), Cvoid, Tuple{Core.LLVMPtr{UInt8,0}}, q.ptr)
        return nothing
    end
end

"""
    commit_bounding_box!(q, t)

Commit a generated intersection with the current bounding-box candidate at distance `t`.
"""
function commit_bounding_box! end

let name = "commit_bounding_box_intersection_$RAY_QUERY"
    ir = rayquerymodule(name, "void", ["ptr", "float"], "  call void @air.$name(ptr %a0, float %a1)\n  ret void")
    @eval @inline function commit_bounding_box!(q::IntersectionQuery, t::Float32)
        Base.llvmcall(($ir, "entry"), Cvoid, Tuple{Core.LLVMPtr{UInt8,0}, Float32}, q.ptr, t)
        return nothing
    end
end

# Scalar getters of the candidate and the committed intersection.
for (jl, air, T, LT) in ((:candidate_type, "get_candidate_intersection_type", UInt32, "i32"),
                         (:candidate_triangle_distance, "get_candidate_triangle_distance", Float32, "float"),
                         (:candidate_primitive_id, "get_candidate_primitive_id", UInt32, "i32"),
                         (:candidate_instance_id, "get_candidate_instance_id", UInt32, "i32"),
                         (:candidate_user_instance_id, "get_candidate_user_instance_id", UInt32, "i32"),
                         (:committed_type, "get_committed_intersection_type", UInt32, "i32"),
                         (:committed_distance, "get_committed_distance", Float32, "float"),
                         (:committed_primitive_id, "get_committed_primitive_id", UInt32, "i32"),
                         (:committed_instance_id, "get_committed_instance_id", UInt32, "i32"),
                         (:committed_user_instance_id, "get_committed_user_instance_id", UInt32, "i32"))
    name = "$(air)_$RAY_QUERY"
    ir = rayquerymodule(name, LT, ["ptr"], "  %r = call $LT @air.$name(ptr %a0)\n  ret $LT %r")
    @eval @inline $jl(q::IntersectionQuery) =
        Base.llvmcall(($ir, "entry"), $T, Tuple{Core.LLVMPtr{UInt8,0}}, q.ptr)
end

# Vector getters, returned as tuples.
for (jl, air, N) in ((:candidate_triangle_barycentric, "get_candidate_triangle_barycentric_coord", 2),
                     (:committed_triangle_barycentric, "get_committed_triangle_barycentric_coord", 2),
                     (:candidate_ray_origin, "get_candidate_ray_origin", 3),
                     (:candidate_ray_direction, "get_candidate_ray_direction", 3))
    name = "$(air)_$RAY_QUERY"
    LT = "<$N x float>"
    ir = rayquerymodule(name, LT, ["ptr"], "  %r = call $LT @air.$name(ptr %a0)\n  ret $LT %r")
    @eval @inline function $jl(q::IntersectionQuery)
        v = Base.llvmcall(($ir, "entry"), NTuple{$N,VecElement{Float32}}, Tuple{Core.LLVMPtr{UInt8,0}}, q.ptr)
        return map(x -> x.value, v)
    end
end

"""
    IntersectionResult

What [`intersection`](@ref) answers: Metal's `intersection_result<instancing,
triangle_data>`.
"""
struct IntersectionResult
    type::UInt32
    distance::Float32
    primitive_id::UInt32
    instance_id::UInt32
    user_instance_id::UInt32
    barycentric::NTuple{2,Float32}
end

"""
    intersection(origin, direction, tmin, tmax, accel, mask, forced_opacity, triangles_only, accept_any) -> IntersectionResult

Trace a ray in one call: `intersector<instancing, triangle_data>::intersect`, with no
intersection functions, so a non-opaque triangle or a bounding box is never offered
back. For a traversal whose candidates need an answer, use an
[`IntersectionQuery`](@ref). Arguments as [`query_reset!`](@ref); `triangles_only`
tells traversal the structure holds no boxes.
"""
function intersection end

# The result struct's slots, by what MSL reads from them: type, distance,
# primitive_id, geometry_id, (a pointer no field reads), instance_id,
# user_instance_id, barycentrics, front facing. After the structure: the mask, the
# function table (the null one), the payload and its size, then the integers `reset`
# takes after the mask, and the two flags.
let name = "intersect.instancing.triangle_data"
    RT = "{ i32, float, i32, i32, ptr addrspace(1), i32, i32, <2 x float>, i1 }"
    ST = "{ i32, float, i32, i32, i32, float, float }"
    ir = """
        declare $RT @air.$name(<3 x float>, <3 x float>, float, float, ptr addrspace(1), i32, ptr addrspace(1), ptr, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, i1)
        declare ptr addrspace(1) @air.get_null_intersection_function_table()
        define $ST @entry(<3 x float> %o, <3 x float> %d, float %tmin, float %tmax, ptr addrspace(1) %accel, i32 %mask, i32 %forced, i32 %geometry, i8 %anyb) #0 {
          %any = trunc i8 %anyb to i1
          %table = call ptr addrspace(1) @air.get_null_intersection_function_table()
          %r = call $RT @air.$name(<3 x float> %o, <3 x float> %d, float %tmin, float %tmax, ptr addrspace(1) %accel, i32 %mask, ptr addrspace(1) %table, ptr null, i64 0, i32 0, i32 0, i32 0, i32 0, i32 %forced, i32 %geometry, i32 -1, i32 -1, i32 0, i1 false, i1 %any)
          %type = extractvalue $RT %r, 0
          %t = extractvalue $RT %r, 1
          %prim = extractvalue $RT %r, 2
          %inst = extractvalue $RT %r, 5
          %user = extractvalue $RT %r, 6
          %bary = extractvalue $RT %r, 7
          %bu = extractelement <2 x float> %bary, i64 0
          %bv = extractelement <2 x float> %bary, i64 1
          %s0 = insertvalue $ST undef, i32 %type, 0
          %s1 = insertvalue $ST %s0, float %t, 1
          %s2 = insertvalue $ST %s1, i32 %prim, 2
          %s3 = insertvalue $ST %s2, i32 %inst, 3
          %s4 = insertvalue $ST %s3, i32 %user, 4
          %s5 = insertvalue $ST %s4, float %bu, 5
          %s6 = insertvalue $ST %s5, float %bv, 6
          ret $ST %s6
        }
        attributes #0 = { alwaysinline }
        """
    V3 = NTuple{3,VecElement{Float32}}
    Raw = Tuple{UInt32,Float32,UInt32,UInt32,UInt32,Float32,Float32}
    @eval @inline function intersection(o::NTuple{3,Float32}, d::NTuple{3,Float32}, tmin::Float32,
                                        tmax::Float32, accel::UInt64, mask::UInt32,
                                        forced_opacity::UInt32, triangles_only::Bool, accept_any::Bool)
        r = Base.llvmcall(($ir, "entry"), $Raw,
                          Tuple{$V3, $V3, Float32, Float32, Core.LLVMPtr{UInt8,1}, UInt32, UInt32, UInt32, Bool},
                          map(VecElement, o), map(VecElement, d), tmin, tmax,
                          reinterpret(Core.LLVMPtr{UInt8,1}, accel), mask, forced_opacity,
                          triangles_only ? UInt32(1) : UInt32(3), accept_any)
        return IntersectionResult(r[1], r[2], r[3], r[4], r[5], (r[6], r[7]))
    end
end
