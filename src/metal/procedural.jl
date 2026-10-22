# Tracing procedural (AABB) geometry on Metal. The traversal loop is an MSL
# `intersection_query<>` (a C++ template, so MSL only); at each box candidate it
# calls `procedural_candidate`, compiled from Julia as a `[[visible]]` function
# in a visible function table. Only scalars and one payload pointer cross; the
# candidate's ray and primitive arrive as parameters `stage_builtins!` appends.

# The running best hit crosses back as four floats, packed per field because
# `reinterpret` refuses non-primitive structs on the GPU.

@inline as_f32(x::Float32) = x
@inline as_f32(x::UInt32)  = reinterpret(Float32, x)
@inline as_f32(x::Int32)   = reinterpret(Float32, x)
@inline from_f32(::Type{Float32}, x::Float32) = x
@inline from_f32(::Type{UInt32},  x::Float32) = reinterpret(UInt32, x)
@inline from_f32(::Type{Int32},   x::Float32) = reinterpret(Int32, x)

"""
    packbest(b) -> NTuple{4,Float32}

A payload's running best hit, as the four floats a visible function returns.
"""
@generated function packbest(b::B) where {B}
    n = fieldcount(B)
    n <= 4 || error("a procedural payload's best hit must have at most 4 fields " *
                    "to cross a visible function's return; `$B` has $n")
    vals = [i <= n ? :(as_f32(getfield(b, $i))) : :(0f0) for i in 1:4]
    return Expr(:block, Expr(:meta, :inline), Expr(:tuple, vals...))
end

"""
    unpackbest(B, v) -> B

The inverse of [`packbest`](@ref).
"""
@generated function unpackbest(::Type{B}, v::NTuple{4,Float32}) where {B}
    args = [:(from_f32($(fieldtype(B, i)), v[$i])) for i in 1:fieldcount(B)]
    return Expr(:block, Expr(:meta, :inline), Expr(:call, B, args...))
end

"""The four floats a procedural visible function writes through its output pointer."""
struct ProceduralOut
    v::NTuple{4,Float32}
end

"""
    ProceduralVisible{P,B}()

The callable compiled as an AIR visible function for device-side payload type `P`
(plain bits) with best-hit type `B`. A struct, not a closure, so it compiles by
type. The payload arrives as one pointer, so the MSL traversal is independent of
its layout.
"""
struct ProceduralVisible{P,B} end

function (::ProceduralVisible{P,B})(b1::Float32, b2::Float32, b3::Float32, b4::Float32,
                                    payload::Core.LLVMPtr{UInt8, Metal.AS.Device},
                                    out::StageOut{ProceduralOut}) where {P,B}
    p = unsafe_load(reinterpret(Core.LLVMPtr{P, Metal.AS.Device}, payload))
    best = unpackbest(B, (b1, b2, b3, b4))
    nb = procedural_candidate(p, best)::B
    unsafe_store!(out, ProceduralOut(packbest(nb)))
    return nothing
end

"""The argument tuple `ProceduralVisible` is compiled against."""
procedural_visible_tt() = Tuple{Float32, Float32, Float32, Float32,
                                Core.LLVMPtr{UInt8, Metal.AS.Device},
                                StageOut{ProceduralOut}}

# The table (`scene->cand`) and the payload bytes share the acceleration
# structure's argument buffer, so the Julia `ccall` keeps a one-pointer signature.
# The builtins are passed as `(direction, origin, prim)` because `stage_builtins!`
# appends them sorted by global name; another order reports every ray a miss.
const PROCEDURAL_MSL = """
#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace raytracing;

// `device uchar*`, matching what the Julia side's `Core.LLVMPtr{UInt8,1}` is
// named in its AIR metadata (`uchar*`). A table's signature is how Metal
// resolves the call; spelled `void*` here it simply does not happen, and
// nothing reports it.
using cand_fn = float4(float, float, float, float, device uchar *,
                       float, float, float, float, float, float, uint);

struct SceneProcAB {
    instance_acceleration_structure accel;
    visible_function_table<cand_fn> cand;
    device uchar *payload;
};
// The same record as `HWTLAS_MSL`'s, `user` included: both linked functions
// return `MetalHit`, so the two declarations are one ABI.
struct HitOut { float t; uint hit; uint prim; uint inst; float bu; float bv; uint user; };

static inline HitOut trace_proc(device const SceneProcAB *scene,
        float ox, float oy, float oz, float dx, float dy, float dz,
        float tmin, float tmax, uint cull, bool anyhit)
{
    ray r; r.origin = float3(ox,oy,oz); r.direction = float3(dx,dy,dz);
    r.min_distance = tmin; r.max_distance = tmax;

    intersection_query<instancing> q;
    // Two arguments, not three. The mask overload compiles and then offers NO
    // candidates at all -- `intersection_query::reset`'s third parameter is not
    // the instance mask a `intersector::intersect` call takes there.
    q.reset(r, scene->accel);

    // The payload's own `procedural_miss` is what seeds this on the Julia side;
    // `1e30` in slot 0 is the `t` every hit record puts first and the only field
    // this loop reads.
    float4 best = float4(INFINITY, 0.0f, 0.0f, 0.0f);
    uint prim = 0u, inst = 0u;
    while (q.next()) {
        if (q.get_candidate_intersection_type() == intersection_type::bounding_box) {
            float3 o = q.get_candidate_ray_origin();
            float3 d = q.get_candidate_ray_direction();
            uint   p = q.get_candidate_primitive_id();
            // dx, dy, dz, ox, oy, oz, prim — the order `stage_builtins!`
            // appends them in, which is ALPHABETICAL by global name.
            float4 nb = scene->cand[0](best.x, best.y, best.z, best.w, scene->payload,
                                       d.x, d.y, d.z, o.x, o.y, o.z, p);
            // "the returned best improved" IS "commit_intersection! was called".
            if (nb.x < best.x) {
                best = nb; prim = p; inst = q.get_candidate_instance_id();
                q.commit_bounding_box_intersection(nb.x);
                if (anyhit) break;
            }
        }
    }

    HitOut h;
    if (q.get_committed_intersection_type() == intersection_type::bounding_box) {
        h.t = best.x; h.hit = 1u; h.prim = prim;
        // All FOUR slots of the payload's best hit travel back, because the
        // payload is what rebuilds it: slots 1 and 2 as the barycentric pair,
        // and slot 3 in `inst`, whose own value a procedural hit has no use for
        // (it has no triangle table to index). Dropping slot 3 and
        // reconstructing it from the primitive id instead is off by one the
        // moment a payload numbers its own primitives — which `Hikari.FEMHit`
        // does: its `cell` is 1-based.
        h.bu = best.y; h.bv = best.z; h.inst = as_type<uint>(best.w);
        // The custom index DOES travel, as on the triangle path.
        h.user = q.get_committed_user_instance_id();
    } else {
        h.t = tmax; h.hit = 0u; h.prim = 0u; h.inst = 0u;
        h.bu = 0.0f; h.bv = 0.0f; h.user = 0u;
    }
    return h;
}

// A pipeline has to be created from a KERNEL, and a visible function is not
// one ("Empty request" from the compiler, naming nothing). This exists only to
// own the pipeline the candidate's function handle comes from.
kernel void __metal_proc_table_host(device uint *sink [[buffer(0)]],
                                    uint tid [[thread_position_in_grid]])
{ sink[tid] = tid; }

[[visible]] HitOut __metal_linked_closest_proc(device const SceneProcAB *scene,
        float ox, float oy, float oz, float dx, float dy, float dz,
        float tmin, float tmax, uint cull)
{ return trace_proc(scene, ox,oy,oz, dx,dy,dz, tmin, tmax, cull, false); }

[[visible]] HitOut __metal_linked_any_proc(device const SceneProcAB *scene,
        float ox, float oy, float oz, float dx, float dy, float dz,
        float tmin, float tmax, uint cull)
{ return trace_proc(scene, ox,oy,oz, dx,dy,dz, tmin, tmax, cull, true); }
"""

procedurallib(d::MetalDevice) = linkmsl!(d, :procedural, PROCEDURAL_MSL,
                                         ("__metal_linked_closest_proc", "__metal_linked_any_proc"))

"""
    procedural_payload_bytes(payload) -> MtlVector{UInt8}

The bytes of `Metal.mtlconvert(payload)`, the struct a kernel launch would pass.
Throws if that is not plain bits.
"""
function procedural_payload_bytes(payload)
    dev_side = Metal.mtlconvert(payload)
    isbitstype(typeof(dev_side)) ||
        error("a procedural payload must adapt to plain bits to cross into a " *
              "visible function; `$(typeof(dev_side))` does not")
    # Through a pointer: `reinterpret` refuses structs with padding.
    n = sizeof(dev_side)
    bytes = Vector{UInt8}(undef, n)
    ref = Ref(dev_side)
    GC.@preserve ref bytes begin
        src = Base.unsafe_convert(Ptr{typeof(dev_side)}, ref)
        unsafe_copyto!(pointer(bytes), Ptr{UInt8}(src), n)
    end
    return MtlArray(bytes)
end

"""
    procedural_resident!(d, payload)

Make the buffers of `payload`'s top-level `MtlArray` fields resident. Metal cannot
infer residency for buffers reached by stored address; the shader then reads
zeros and the candidate rejects every box.
"""
function procedural_resident!(d::MetalDevice, payload)
    for i in 1:fieldcount(typeof(payload))
        f = getfield(payload, i)
        f isa MtlArray || continue
        make_resident!(d, f.data[].buffer)
    end
    return nothing
end

"""
    procedural_table(d, payload, best_type) -> (table, pipeline)

Compile this payload's `procedural_candidate` as a visible function and put it
in a one-entry table. A function handle only exists for a pipeline that links the
function, so the table is built against a pipeline created for that purpose.
"""
function procedural_table(d::MetalDevice, payload, ::Type{B}) where {B}
    fn = compile_procedural_candidate(d, typeof(Metal.mtlconvert(payload)), B)
    lib = procedurallib(d)
    host = MTL.MTLFunction(lib, "__metal_proc_table_host")
    desc = MTL.MTLComputePipelineDescriptor()
    desc.computeFunction = host
    lf = MTL.MTLLinkedFunctions()
    lf.functions = NSArray([fn])
    desc.linkedFunctions = lf
    desc.maxCallStackDepth = 4
    pipe = MTL.MTLComputePipelineState(d.dev, desc)
    handle = MTL.function_handle(pipe, fn)
    handle === nothing &&
        error("the procedural candidate did not link into its pipeline; " *
              "an unset table entry is a GPU fault, not an error")
    table = MTL.MTLVisibleFunctionTable(pipe, MTL.MTLVisibleFunctionTableDescriptor(1))
    MTL.set_function!(table, handle, 0)
    # Leaves the residency set when collected, as in `newstructure`.
    finalizer(Metal.forget_resident!, table)
    return (table, pipe)
end

"""The start of every procedural candidate's function name; the rest names its types."""
const PROCEDURAL_CANDIDATE_PREFIX = "__metal_linked_procedural_candidate_"

"""Compile `ProceduralVisible{P,B}` and register it for every pipeline on `d`."""
function compile_procedural_candidate(d::MetalDevice, ::Type{P}, ::Type{B}) where {P,B}
    # One name per payload type: pipelines link registered functions by name, so
    # a shared name calls whichever payload's candidate registered first.
    name = string(PROCEDURAL_CANDIDATE_PREFIX, string(hash((P, B)); base = 16))
    # `always_inline` as in `compile_stage_function`.
    cfg = Metal.compiler_config(d.dev; stage = :candidate, name, always_inline = true)
    job = Metal.GPUCompiler.CompilerJob(
        Metal.methodinstance(ProceduralVisible{P,B}, procedural_visible_tt()), cfg)
    lib = MTL.MTLLibraryFromData(d.dev, Metal.compile_to_metallib(job).metallib)
    fn = MTL.MTLFunction(lib, name)
    # Every kernel pipeline on this device must link it, or the shading
    # kernel's table slot resolves to nothing and every ray misses. A table
    # function, not a linked one: a `privateFunctions` entry may be inlined and
    # then has no address, so the call returns undef and nothing commits.
    Metal.register_table_function!(d.dev, fn)
    return fn
end

