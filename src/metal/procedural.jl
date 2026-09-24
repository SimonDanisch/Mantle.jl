# Tracing PROCEDURAL (AABB) geometry on Metal.
#
# ── Why this is not just another linked function ─────────────────────────────
#
# Triangle traversal is one MSL call: `intersector<instancing, triangle_data>`
# answers a ray with a hit and nothing in between. A BOX cannot be answered that
# way — what is inside it is the shader's to say, which is what
# `procedural_candidate` says — so the traversal has to STOP at every candidate
# and ask.
#
# Metal's inline ray query (`intersection_query<>`) does exactly that, and like
# `intersector<>` it is a C++ class template the Metal frontend instantiates and
# inlines, so the LOOP can only be MSL. The BODY is Julia. They meet through a
# `[[visible]]` function in a visible function table:
#
#     MSL  intersection_query        -> the loop, and only MSL can hold it
#     AIR  visible function          -> `procedural_candidate`, compiled from Julia
#     back the new best, four floats -> the loop commits if it improved
#
# Only scalars and one pointer cross that boundary. No query object does, which
# is what made this look impossible from the Julia side: there is no query in
# scope to ask, so the candidate's primitive and ray arrive as PARAMETERS that
# `stage_builtins!` appends — the same mechanism `vertex_index()` uses.

# ── The running best, as four floats ─────────────────────────────────────────
#
# A visible function returns a value and has no other way out, so the payload's
# best hit has to fit in one. Four 32-bit fields is what a hit record is in
# practice -- `Hikari.FEMHit` is `(t, ξ₁, ξ₂, cell)` -- and packing by FIELD
# rather than by `reinterpret` keeps it honest for a struct that is not a
# primitive type, which `reinterpret` refuses on a GPU.

@inline _as_f32(x::Float32) = x
@inline _as_f32(x::UInt32)  = reinterpret(Float32, x)
@inline _as_f32(x::Int32)   = reinterpret(Float32, x)
@inline _from_f32(::Type{Float32}, x::Float32) = x
@inline _from_f32(::Type{UInt32},  x::Float32) = reinterpret(UInt32, x)
@inline _from_f32(::Type{Int32},   x::Float32) = reinterpret(Int32, x)

"""
    packbest(b) -> NTuple{4,Float32}

A payload's running best hit, as the four floats a visible function returns.
"""
@generated function packbest(b::B) where {B}
    n = fieldcount(B)
    n <= 4 || error("a procedural payload's best hit must have at most 4 fields " *
                    "to cross a visible function's return; `$B` has $n")
    vals = [i <= n ? :(_as_f32(getfield(b, $i))) : :(0f0) for i in 1:4]
    return Expr(:block, Expr(:meta, :inline), Expr(:tuple, vals...))
end

"""    unpackbest(B, v) — the inverse of [`packbest`](@ref)."""
@generated function unpackbest(::Type{B}, v::NTuple{4,Float32}) where {B}
    args = [:(_from_f32($(fieldtype(B, i)), v[$i])) for i in 1:fieldcount(B)]
    return Expr(:block, Expr(:meta, :inline), Expr(:call, B, args...))
end

"""The four floats a procedural visible function writes through its output pointer."""
struct ProceduralOut
    v::NTuple{4,Float32}
end

# ── The visible function ─────────────────────────────────────────────────────

"""
    ProceduralVisible{P,B}()

The callable compiled as an AIR visible function for a payload of type `P` whose
best hit is `B`.

A callable STRUCT and not a closure, because it has to be compiled by type:
`Metal.methodinstance(ProceduralVisible{P,B}, tt)` is what names it, exactly as
`MetalFragmentStage` names a fragment shader.

`P` is the DEVICE side of the payload — what `Adapt.adapt` made of it — so its
array fields are `MtlDeviceArray`s and the whole struct is plain bits. It
arrives as ONE pointer at the buffer holding those bits, which is what keeps the
MSL below free of the payload's shape: one source, compiled once per device,
whatever a caller is tracing.
"""
struct ProceduralVisible{P,B} end

function (::ProceduralVisible{P,B})(b1::Float32, b2::Float32, b3::Float32, b4::Float32,
                                    payload::Core.LLVMPtr{UInt8, Metal.AS.Device},
                                    out::Core.LLVMPtr{ProceduralOut,1}) where {P,B}
    p = unsafe_load(reinterpret(Core.LLVMPtr{P, Metal.AS.Device}, payload))
    best = unpackbest(B, (b1, b2, b3, b4))
    nb = procedural_candidate(p, best)::B
    unsafe_store!(out, ProceduralOut(packbest(nb)))
    return nothing
end

"""The argument tuple `ProceduralVisible` is compiled against."""
procedural_visible_tt() = Tuple{Float32, Float32, Float32, Float32,
                                Core.LLVMPtr{UInt8, Metal.AS.Device},
                                Core.LLVMPtr{ProceduralOut,1}}

# ── The traversal ────────────────────────────────────────────────────────────
#
# `scene->cand` is the visible function table, and `scene->payload` the bytes of
# the adapted payload. Both live in the SAME argument buffer as the acceleration
# structure, which is why the Julia side's `ccall` keeps its one-pointer
# signature and nothing about it is payload-shaped.
#
# The builtin order is `(direction, origin, prim)` and NOT declaration order:
# `stage_builtins!` appends the globals it turns into parameters sorted BY NAME,
# and `__air_candidate_direction` < `__air_candidate_origin` <
# `__air_candidate_prim`. Getting it wrong compiles, links, runs, and reports
# every ray a miss.
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

# ── The host side ────────────────────────────────────────────────────────────

"""One device's procedural traversal: the MSL pair, compiled once."""
const PROC_FUNCS = Dict{Any,Nothing}()
const PROC_FUNCS_LOCK = ReentrantLock()

function ensure_procedural_linked!(d::MetalDevice)
    key = pointer(d.dev)
    Base.@lock PROC_FUNCS_LOCK begin
        haskey(PROC_FUNCS, key) && return nothing
        lib = MTL.MTLLibrary(d.dev, PROCEDURAL_MSL)
        for name in ("__metal_linked_closest_proc", "__metal_linked_any_proc")
            Metal.register_linked_function!(d.dev, MTL.MTLFunction(lib, name))
        end
        PROC_FUNCS[key] = nothing
    end
    return nothing
end

"""
    procedural_payload_bytes(payload) -> MtlVector{UInt8}

The DEVICE-side payload, as bytes a visible function can load in one go.

`Metal.mtlconvert` is what a kernel launch would do to it, so what lands here is
exactly the struct the shader sees — `MtlDeviceArray`s holding device addresses,
scalars inline. Nothing about the payload's SHAPE reaches the MSL that way,
which is why there is one traversal source and not one per payload type.
"""
function procedural_payload_bytes(payload)
    dev_side = Metal.mtlconvert(payload)
    isbitstype(typeof(dev_side)) ||
        error("a procedural payload must adapt to plain bits to cross into a " *
              "visible function; `$(typeof(dev_side))` does not")
    # Copied through a pointer, not `reinterpret(UInt8, [x])`: a payload with
    # PADDING — `Hikari.FEMElements` has some — makes `reinterpret` refuse with
    # "Padding of type UInt8 is not compatible". The padding bytes travel too and
    # are never read; what matters is that the fields land at the offsets the
    # device-side struct declares.
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

Make every buffer the payload REACHES resident.

The payload's own bytes being resident is not enough: they hold device
ADDRESSES, and a buffer reached by a stored address is exactly the case Metal
cannot infer — the shader reads zeros and the candidate rejects every box. The
symptom is a traversal that offers candidates and never commits one, which looks
like a wrong solve and is not.
"""
function procedural_resident!(d::MetalDevice, payload)
    for i in 1:fieldcount(typeof(payload))
        f = getfield(payload, i)
        f isa MtlArray || continue
        make_resident!(d, f.data[])
    end
    return nothing
end

"""The device-side type of `payload`, which is what `ProceduralVisible` is compiled for."""
procedural_device_type(payload) = typeof(Metal.mtlconvert(payload))

"""
    procedural_table(d, payload, best_type) -> (table, pipeline)

Compile this payload's `procedural_candidate` as a visible function and put it
in a one-entry table.

The table is built against a PIPELINE because a function handle is only
meaningful for the pipeline that linked the function — so the pipeline here is
the traversal's own, and the table it yields is what every shading kernel binds.
"""
function procedural_table(d::MetalDevice, payload, ::Type{B}) where {B}
    P = procedural_device_type(payload)
    fn = compile_procedural_candidate(d, P, B)
    # A pipeline whose only job is to own the handle. `__metal_linked_closest_proc`
    # is the natural carrier: it is the function that will call it.
    ensure_procedural_linked!(d)
    lib = MTL.MTLLibrary(d.dev, PROCEDURAL_MSL)
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
    return (table, pipe)
end

"""Compile `ProceduralVisible{P,B}` and keep its library alive with the function."""
function compile_procedural_candidate(d::MetalDevice, ::Type{P}, ::Type{B}) where {P,B}
    name = "__metal_linked_procedural_candidate"
    cfg = Metal.compiler_config(d.dev; stage = :candidate, name)
    job = Metal.GPUCompiler.CompilerJob(
        Metal.methodinstance(ProceduralVisible{P,B}, procedural_visible_tt()), cfg)
    lib = MTL.MTLLibraryFromData(d.dev, Metal.compile_to_metallib(job).metallib)
    fn = MTL.MTLFunction(lib, name)
    # Registered like the MSL traversal is, so EVERY kernel pipeline compiled
    # for this device links it. A visible function table entry names a function
    # handle, and a handle is only meaningful for a pipeline that has the
    # function — the shading kernel is a different pipeline from the one the
    # table was built against, and without this its traversal calls a table slot
    # that resolves to nothing and reports every ray a miss.
    # `register_table_function!`, not `register_linked_function!`: this one is
    # reached through a VISIBLE FUNCTION TABLE, so it has to land in the
    # pipeline's `functions` and not its `privateFunctions`. A private function
    # may be inlined, and an inlined function has no address for a table entry
    # to point at — the call is made, returns undef, and the traversal commits
    # nothing. That is not hypothetical; it is what this did.
    Metal.register_table_function!(d.dev, fn)
    return fn
end


