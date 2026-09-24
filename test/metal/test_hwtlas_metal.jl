"""
The Metal hardware acceleration structure and the traversal a Julia kernel does
against it.

This is the path `hw_accel=true` takes. It exists because a Julia kernel CAN
call hardware traversal on Metal, contrary to the obvious reading: the
`intersector<>` is a C++ template with no AIR symbol, but a `[[visible]]` MSL
function wrapping it can be LINKED into the kernel's pipeline
(`MTLLinkedFunctions`), and the acceleration structure reaches it through an
argument buffer because a Julia kernel cannot declare an
`instance_acceleration_structure` parameter.

Three things in that chain fail silently rather than loudly, and each has an
assertion below:

  * residency — a TLAS bound through an argument buffer is not automatically
    resident, and a non-resident structure reports every ray as a miss;
  * the 28-byte hit record is an ABI shared with hand-written MSL that no
    compiler checks;
  * the fifth return value is the instance CUSTOM index, not Metal's
    `instance_id`. Returning the latter type-checks, runs, and quietly gives
    every instance past the first the wrong medium interface.
"""

using Test, Mantle, Metal, Raycore, GeometryBasics, KernelAbstractions
using Adapt, StaticArrays
# QUALIFIED below, not `using LinearAlgebra`: these files are included into
# `Main` one after another, and once some earlier one has touched an undefined
# `Main.normalize` the binding is resolved and a later `using` cannot introduce
# it — an `UndefVarError` whose cause is the ORDER of the includes.
import LinearAlgebra
const KA = KernelAbstractions

@kernel function _hw_probe!(hits, ts, insts, dirs, accel, O)
    i = @index(Global)
    hit, tri, t, bary, inst = Raycore.closest_hit(accel, Raycore.Ray(o = O, d = dirs[i], t_max = 1f30))
    hits[i]  = hit ? Int32(1) : Int32(0)
    ts[i]    = hit ? t : -1f0
    insts[i] = inst
end

# Deterministic directions from `O` towards the unit cube around the origin.
function _probe_dirs(n, O)
    s = Ref(987)
    nr() = (s[] = (1103515245 * s[] + 12345) % 2147483648; Float32(s[]) / 2147483648f0)
    [Vec3f(LinearAlgebra.normalize(Point3f(2*(nr()-0.5f0), 2*(nr()-0.5f0), 0) - O)) for _ in 1:n]
end

@testset "Metal: the hit-record ABI matches the MSL it is shared with" begin
    @test sizeof(Mantle.MetalHit) == 28
    @test isbitstype(Mantle.MetalHit)
    @test fieldnames(Mantle.MetalHit) == (:t, :hit, :prim, :inst, :bu, :bv, :user)
    # Plain scalars, no padding — the MSL struct is declared the same way.
    @test all(fieldoffset(Mantle.MetalHit, i) == 4(i - 1) for i in 1:7)
end

@testset "Metal: HWTLAS is reachable through the portable constructor" begin
    be = Metal.MetalBackend()
    # `HWTLAS{Tri}(backend)` is what Hikari's `default_accel` calls; a caller
    # must never name the backend's own type.
    t = Mantle.HWTLAS{Raycore.Triangle{UInt32}}(be)
    @test t isa Mantle.HWTLAS{Raycore.Triangle{UInt32}}
    @test Raycore.n_geometries(t) == 0
    @test Raycore.n_instances(t) == 0

    mesh = GeometryBasics.normal_mesh(Sphere(Point3f(0), 1.0f0))
    h = push!(t, mesh)
    @test h isa Raycore.TLASHandle
    @test Raycore.n_geometries(t) == 1

    Raycore.sync!(t)
    @test Raycore.n_instances(t) == 1
    b = Raycore.world_bound(t)
    @test all(b.p_min .< -0.9f0) && all(b.p_max .> 0.9f0)

    # `sync!` is the sole owner of the adapted form, and a second one with
    # nothing dirty must not rebuild.
    a1 = Adapt.adapt(be, t)
    a2 = Adapt.adapt(be, t)
    @test a1 === a2
end

@testset "Metal: traversal agrees with the software BVH" begin
    be = Metal.MetalBackend()
    mesh = GeometryBasics.normal_mesh(Sphere(Point3f(0), 1.0f0))
    O = Point3f(0, 0, 4)
    n = 2048
    dirs = _probe_dirs(n, O)

    hw = Mantle.HWTLAS{Raycore.Triangle{UInt32}}(be)
    push!(hw, mesh); Raycore.sync!(hw)
    accel_hw = Adapt.adapt(be, hw)

    d_d = KA.allocate(be, Vec3f, n); copyto!(d_d, dirs)
    h_d = KA.zeros(be, Int32, n); t_d = KA.zeros(be, Float32, n); i_d = KA.zeros(be, UInt32, n)
    _hw_probe!(be)(h_d, t_d, i_d, d_d, accel_hw, O; ndrange = n)
    KA.synchronize(be)
    hh, ht, hi = Array(h_d), Array(t_d), Array(i_d)

    sw = Raycore.TLAS(KA.CPU())
    push!(sw, mesh); Raycore.sync!(sw)
    accel_sw = Adapt.adapt(KA.CPU(), sw)
    sh = zeros(Int32, n); st = zeros(Float32, n); si = zeros(UInt32, n)
    _hw_probe!(KA.CPU())(sh, st, si, copy(dirs), accel_sw, O; ndrange = n)
    KA.synchronize(KA.CPU())

    # Something must actually be hit, or every assertion below is vacuous.
    @test sum(sh) > n ÷ 4
    # Residency failure shows up exactly here: all-miss on the hardware side
    # while the software side hits.
    @test sum(hh) == sum(sh)
    @test hh == sh
    @test maximum(abs.(ht[hh .== 1] .- st[sh .== 1])) < 1e-3

    # The instance CUSTOM index, which no `push!` in this test sets, so 0 —
    # NOT Metal's instance_id.
    @test all(==(UInt32(0)), hi[hh .== 1])
end

@testset "Metal: several instances, and the custom index stays 0" begin
    # Two instances of one mesh, offset along x. A traversal that returned
    # `instance_id` here would give 1 for the second instance, and Hikari would
    # look up a medium interface that does not exist.
    be = Metal.MetalBackend()
    mesh = GeometryBasics.normal_mesh(Sphere(Point3f(0), 0.5f0))
    t = Mantle.HWTLAS{Raycore.Triangle{UInt32}}(be)
    push!(t, mesh, Mat4f(LinearAlgebra.I))
    shifted = Mat4f(1,0,0,0, 0,1,0,0, 0,0,1,0, 1.5f0,0,0,1)   # column-major translate
    push!(t, mesh, shifted)
    Raycore.sync!(t)
    @test Raycore.n_geometries(t) == 2
    @test Raycore.n_instances(t) == 2
    accel = Adapt.adapt(be, t)

    # One ray at each sphere.
    O = Point3f(0, 0, 4)
    dirs = [Vec3f(0, 0, -1), Vec3f(LinearAlgebra.normalize(Point3f(1.5f0, 0, 0) - O))]
    n = length(dirs)
    d_d = KA.allocate(be, Vec3f, n); copyto!(d_d, dirs)
    h_d = KA.zeros(be, Int32, n); t_d = KA.zeros(be, Float32, n); i_d = KA.zeros(be, UInt32, n)
    _hw_probe!(be)(h_d, t_d, i_d, d_d, accel, O; ndrange = n)
    KA.synchronize(be)
    @test Array(h_d) == Int32[1, 1]              # both instances are hit
    @test all(==(UInt32(0)), Array(i_d))         # …and neither reports a custom index
end

@testset "Metal: a custom index set by push! comes back from the hit" begin
    # What `meshscatter!` depends on. Hikari bakes every face of the marker
    # mesh with medium interface 0 ("inherit") and hands each instance's
    # material over as `instance_ids`; `resolve_mi_idx` reads it back as the
    # fifth value of `closest_hit`. Until the user-ID instance descriptor was
    # wired up, `push!` dropped the ids and the traversal returned 0, so every
    # sphere inherited interface 0 and rendered BLACK — on the default
    # `hw_accel = true` path, while the software TLAS drew them correctly.
    #
    # Both spellings: one instance with `instance_id`, a batch with
    # `instance_ids`, and both before and after a refit, which rewrites every
    # descriptor in place and must carry the id across.
    be = Metal.MetalBackend()
    mesh = GeometryBasics.normal_mesh(Sphere(Point3f(0), 0.5f0))
    tr(x) = Mat4f(1,0,0,0, 0,1,0,0, 0,0,1,0, x,0,0,1)   # column-major translate
    t = Mantle.HWTLAS{Raycore.Triangle{UInt32}}(be)
    push!(t, mesh, tr(0f0); instance_id = UInt32(7))
    hb = push!(t, mesh, [tr(1.5f0), tr(3f0)]; instance_ids = UInt32[11, 42])
    Raycore.sync!(t)
    @test Raycore.n_instances(t) == 3

    O = Point3f(0, 0, 4)
    at(x) = Vec3f(LinearAlgebra.normalize(Point3f(x, 0, 0) - O))
    probe(xs) = begin
        n = length(xs)
        d_d = KA.allocate(be, Vec3f, n); copyto!(d_d, [at(x) for x in xs])
        h_d = KA.zeros(be, Int32, n); t_d = KA.zeros(be, Float32, n); i_d = KA.zeros(be, UInt32, n)
        _hw_probe!(be)(h_d, t_d, i_d, d_d, Adapt.adapt(be, t), O; ndrange = n)
        KA.synchronize(be)
        (Array(h_d), Array(i_d))
    end

    hits, ids = probe([0f0, 1.5f0, 3f0])
    @test hits == Int32[1, 1, 1]
    @test ids == UInt32[7, 11, 42]

    # A transform-only change is a REFIT (same instance count), which rewrites
    # the instance buffer: the ids have to survive it.
    Raycore.update_transforms!(t, hb, [tr(1.5f0), tr(4.5f0)])
    Raycore.sync!(t)
    hits2, ids2 = probe([0f0, 1.5f0, 4.5f0])
    @test hits2 == Int32[1, 1, 1]
    @test ids2 == UInt32[7, 11, 42]

    # A length mismatch is refused before anything is added.
    ng = Raycore.n_geometries(t)
    @test_throws ArgumentError push!(t, mesh, [tr(9f0)]; instance_ids = UInt32[1, 2])
    @test Raycore.n_geometries(t) == ng
end

@testset "Metal: an empty structure syncs and misses" begin
    be = Metal.MetalBackend()
    t = Mantle.HWTLAS{Raycore.Triangle{UInt32}}(be)
    Raycore.sync!(t)
    @test Raycore.n_instances(t) == 0
    @test Adapt.adapt(be, t) isa Mantle.AdaptedAccel
end

@testset "Metal: no ray-tracing pipeline, and Hikari must be told so" begin
    # Metal traverses from inside the shading kernel, so there is no shader
    # binding table. Hikari picks its trace path off exactly this.
    be = Metal.MetalBackend()
    t = Mantle.HWTLAS{Raycore.Triangle{UInt32}}(be)
    @test Mantle.supports_rt_pipeline(be) == false
    @test Mantle.supports_rt_pipeline(t) == false
    push!(t, GeometryBasics.normal_mesh(Sphere(Point3f(0), 1f0)))
    Raycore.sync!(t)
    @test Mantle.supports_rt_pipeline(Adapt.adapt(be, t)) == false
end

@testset "Metal: a procedural AABB BLAS builds and registers as an instance" begin
    # Hikari's `FEMMaterial` is PROCEDURAL geometry: boxes plus an intersection
    # routine, not triangles. `push!(scene, ::FEMMaterial)` calls
    # `build_blas_aabb`, which existed only inside `src/vulkan/` — so a scene
    # holding one was an `UndefVarError` at CONSTRUCTION on Metal, before any
    # question of traversal. That is the half this pins; traversal against a
    # procedural BLAS needs an intersection function table and is not here yet.
    be = Metal.MetalBackend()
    dev = Mantle.Device()
    aabbs = [Mantle.AABB(Point3f(-1, -1, -1), Point3f(1, 1, 1)),
             Mantle.AABB(Point3f( 2,  0,  0), Point3f(3, 1, 1))]

    blas = Mantle.build_accel!(Mantle.batchqueue(dev)) do ctx
        # Six `Float32` per box, stride 24 — the layout Metal's descriptor wants.
        Mantle.build_blas_aabb(ctx, aabbs)
    end
    @test blas isa Mantle.MetalBLAS

    # Registering it takes the pre-built-BLAS `push!`, with NO triangles: an
    # empty `Tri[]` keeps the per-BLAS arrays aligned, and nothing reads
    # triangle metadata for a procedural instance. Same shape as Vulkan's.
    t = Mantle.HWTLAS{Raycore.Triangle{UInt32}}(be)
    h = push!(t, blas, Mat4f(LinearAlgebra.I); instance_id = UInt32(7))
    @test h isa Raycore.TLASHandle
    @test length(t.blas_list) == 1
    @test isempty(t.blas_triangles[1])
    @test Raycore.n_instances(t) == 1

    # …and the structure still syncs and adapts with a triangle-less BLAS in it,
    # which is what a mixed scene (a mesh and an FEM surface) needs.
    Raycore.sync!(t)
    @test Adapt.adapt(be, t) isa Mantle.AdaptedAccel

    # Building and tracing are separate questions and the capability answers the
    # second. Metal answers YES: the two testsets below trace boxes through an
    # `intersection_query` whose candidates a Julia function answers.
    @test Mantle.supports_procedural_traversal(be) == true
    @test Mantle.supports_procedural_traversal(t) == true
    # The default is still `false`, so a backend opts in — nothing traces boxes
    # by accident.
    @test Mantle.supports_procedural_traversal(nothing) == false
    # …and an `AdaptedAccel` answers too, rather than falling to that default.
    # A caller that has to ask usually holds one of these; falling through would
    # give a WRONG answer on a backend that does trace boxes, not a missing one.
    @test Mantle.supports_procedural_traversal(Adapt.adapt(be, t)) ==
          Mantle.supports_procedural_traversal(be)
end

# ── Procedural traversal: the loop is MSL, the body is Julia ─────────────────

struct ProcHit
    v::NTuple{4,Float32}
end

"""
A sphere of radius 0.75 inscribed in box `prim`, whose centre is `(2*prim, 0, 0)`.

Compiled as an AIR VISIBLE function and called from inside Metal's traversal
loop. Object space, as the query hands it over: asking for the candidate's ray
rather than transforming the world one is what keeps an instance transform from
being applied twice.
"""
function proc_sphere_candidate(prim::UInt32, ox::Float32, oy::Float32, oz::Float32,
                               dx::Float32, dy::Float32, dz::Float32, best::Float32,
                               out::Core.LLVMPtr{ProcHit,1})
    ex = ox - 2f0 * Float32(prim); ey = oy; ez = oz
    b = ex*dx + ey*dy + ez*dz
    c = ex*ex + ey*ey + ez*ez - 0.75f0 * 0.75f0
    disc = b*b - c
    hit = 0f0; t = 0f0
    if disc >= 0f0
        tt = -b - sqrt(disc)
        # `tt < best` is what makes the shader's running best the committed one.
        if tt > 1f-4 && tt < best
            hit = 1f0; t = tt
        end
    end
    Base.unsafe_store!(out, ProcHit((hit, t, 0f0, 0f0)))
    return nothing
end

const PROC_MSL = """
#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace raytracing;
kernel void trace_proc(
    instance_acceleration_structure accel [[buffer(0)]],
    device float4 *out [[buffer(1)]],
    visible_function_table<float4(uint, float, float, float, float, float, float, float)> tbl [[buffer(2)]],
    device const float4 *orig [[buffer(3)]],
    device const float4 *dir  [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    ray r;
    r.origin = orig[tid].xyz; r.direction = dir[tid].xyz;
    r.min_distance = 0.0f;    r.max_distance = 1e30f;

    intersection_query<instancing> q;
    q.reset(r, accel);

    float bestT = 1e30f;
    while (q.next()) {
        if (q.get_candidate_intersection_type() == intersection_type::bounding_box) {
            uint   p = q.get_candidate_primitive_id();
            float3 o = q.get_candidate_ray_origin();
            float3 d = q.get_candidate_ray_direction();
            float4 res = tbl[0](p, o.x, o.y, o.z, d.x, d.y, d.z, bestT);
            if (res.x > 0.5f && res.y < bestT) {
                bestT = res.y;
                q.commit_bounding_box_intersection(res.y);
            }
        }
    }
    bool hit = q.get_committed_intersection_type() == intersection_type::bounding_box;
    out[tid] = float4(hit ? 1.0f : 0.0f, hit ? bestT : 0.0f, 0.0f, 0.0f);
}
"""

@testset "Metal: procedural geometry is TRACED, solve in Julia" begin
    # This is the mechanism the FEM/isubd traced half needs, at its smallest.
    #
    # `intersector<>` and `intersection_query<>` are C++ class templates the
    # Metal frontend instantiates and inlines, so the traversal LOOP can only be
    # MSL — while the body that answers a box is Julia. They meet through a
    # visible function table: MSL runs the query, Julia runs the solve, and only
    # scalars cross. No intersection function table, and no MSL rewrite of the
    # solve.
    be  = Metal.MetalBackend()
    dev = Metal.device()
    nb  = 4

    aabbs = [Mantle.AABB(Point3f(2f0*i - 1, -1, -1), Point3f(2f0*i + 1, 1, 1))
             for i in 0:(nb - 1)]
    blas = Mantle.build_accel!(Mantle.batchqueue(Mantle.Device())) do ctx
        Mantle.build_blas_aabb(ctx, aabbs)
    end
    hw = Mantle.HWTLAS{Raycore.Triangle{UInt32}}(be)
    push!(hw, blas, Mat4f(LinearAlgebra.I); instance_id = UInt32(0))
    Raycore.sync!(hw)

    stt = Tuple{UInt32, Float32, Float32, Float32, Float32, Float32, Float32,
                Float32, Core.LLVMPtr{ProcHit,1}}
    cfg = Metal.compiler_config(dev; stage = :visible, name = "proc_sphere_candidate")
    job = Metal.GPUCompiler.CompilerJob(
        Metal.methodinstance(typeof(proc_sphere_candidate), stt), cfg)
    sfn = Metal.MTL.MTLFunction(
        Metal.MTL.MTLLibraryFromData(dev, Metal.compile_to_metallib(job).metallib),
        "proc_sphere_candidate")
    @test sfn.functionType == Metal.MTL.MTLFunctionTypeVisible

    pfn  = Metal.MTL.MTLFunction(Metal.MTL.MTLLibrary(dev, PROC_MSL), "trace_proc")
    desc = Metal.MTLComputePipelineDescriptor()
    desc.computeFunction = pfn
    lf = Metal.MTL.MTLLinkedFunctions()
    lf.functions = Metal.NSArray([sfn])
    desc.linkedFunctions = lf
    desc.maxCallStackDepth = 4
    pipe = Metal.MTLComputePipelineState(dev, desc)
    handle = Metal.MTL.function_handle(pipe, sfn)
    @test handle !== nothing
    tbl = Metal.MTL.MTLVisibleFunctionTable(
        pipe, Metal.MTL.MTLVisibleFunctionTableDescriptor(1))
    Metal.MTL.set_function!(tbl, handle, 0)

    # Rays down +x from x = -5. `|y| < 0.75` meets sphere 0; `y = 1` is inside
    # the BOX and outside the sphere, which is the case that matters — hardware
    # traversal offers the candidate and the Julia solve has to reject it.
    ys  = Float32[0.0, 0.3, 0.7, 0.75, 1.0, 2.0]
    nr  = length(ys)
    org = Metal.MtlVector{NTuple{4,Float32}}([(-5f0, y, 0f0, 0f0) for y in ys])
    dir = Metal.MtlVector{NTuple{4,Float32}}([(1f0, 0f0, 0f0, 0f0) for _ in ys])
    out = Metal.MtlVector{NTuple{4,Float32}}(undef, nr)

    cb = Metal.MTL.MTLCommandBuffer(Metal.MTL.MTLCommandQueue(dev))
    Metal.MTL.MTLComputeCommandEncoder(cb) do cce
        Metal.MTL.set_function!(cce, pipe)
        Metal.MTL.set_acceleration_structure!(cce, hw.built.handle, 1)
        # Residency for the structures the TLAS points at: bound alone, its
        # BLASes are not resident and traversal silently misses everything.
        Metal.MTL.use!(cce, [b.handle for b in hw.blas_list], Metal.MTL.ReadUsage)
        Metal.MTL.set_buffer!(cce, out.data[], 0, 2)
        Metal.MTL.set_visible_function_table!(cce, tbl, 2)
        Metal.MTL.set_buffer!(cce, org.data[], 0, 4)
        Metal.MTL.set_buffer!(cce, dir.data[], 0, 5)
        Metal.MTL.dispatchThreads!(cce, Metal.MTL.MTLSize(nr,1,1), Metal.MTL.MTLSize(nr,1,1))
    end
    Metal.MTL.commit!(cb)
    Metal.MTL.wait_completed(cb)
    got = Array(out)

    # Against the same arithmetic on the host, per ray.
    for (i, y) in enumerate(ys)
        c = y*y - 0.75f0*0.75f0 + 25f0          # ex = -5, so b = -5 and c = ey^2 - r^2 + 25
        disc = 25f0 - c
        want_hit = disc >= 0f0
        @test (got[i][1] == 1f0) == want_hit
        want_hit && @test got[i][2] ≈ 5f0 - sqrt(disc) rtol=1f-6
    end
    # …and the two rejections really are rejections, not an empty traversal:
    # `y = 1` is inside the box, so a candidate WAS offered and the solve said no.
    @test got[5][1] == 0f0
    @test got[6][1] == 0f0
end

# ── The PORTABLE protocol, traced ───────────────────────────────────────────
#
# The testset above proves the mechanism with the solve written out by hand.
# This one goes through `Mantle.procedural_candidate` and the three portable
# verbs — the same code a payload in Hikari writes — so what it pins is the
# CONTRACT, not a trick.

"""A payload: one radius per box, in a device array reached by pointer."""
struct RadiusElems{A}
    r::A
end

struct RadiusHit
    t::Float32
    ξ₁::Float32
    ξ₂::Float32
    cell::UInt32
end

Mantle.procedural_miss(::RadiusElems) = RadiusHit(1f30, 0f0, 0f0, UInt32(0))

# Written exactly as `src/raytracing/accel.jl` documents: solve in OBJECT space,
# ask the QUERY for the ray, and commit only for a `t` that improves on `best`.
function Mantle.procedural_candidate(e::RadiusElems, best::RadiusHit)
    cell = Mantle.candidate_primitive_index()
    o, d = Mantle.candidate_object_ray()
    @inbounds rad = e.r[cell]
    ex = o[1] - 2f0 * Float32(cell - 1); ey = o[2]; ez = o[3]
    b = ex*d[1] + ey*d[2] + ez*d[3]
    c = ex*ex + ey*ey + ez*ez - rad*rad
    disc = b*b - c
    if disc >= 0f0
        t = -b - sqrt(disc)
        if t > 1f-4 && t < best.t
            Mantle.commit_intersection!(t)
            # `cell % UInt32`, not `UInt32(cell)`: the checked conversion's
            # `InexactError` path reaches for `kernel_state`, which a visible
            # function has no more than a graphics stage does.
            return RadiusHit(t, 0f0, 0f0, cell % UInt32)
        end
    end
    return best
end

"""A device array reached through a raw pointer, for a payload inside a visible function."""
struct PtrVec <: AbstractVector{Float32}
    p::Core.LLVMPtr{Float32, Metal.AS.Device}
end
# The length is never consulted; `femfloat`-style dispatch needs an
# `AbstractArray` to read the element type off, and a bare wrapper sends the
# whole call dynamic.
Base.size(::PtrVec) = (typemax(Int),)
Base.@propagate_inbounds Base.getindex(a::PtrVec, i::Integer) = unsafe_load(a.p, i)

struct RadiusOut
    v::NTuple{4,Float32}
end

"""
The visible function: rebuild the payload from its pointer, call the PORTABLE
candidate, hand the new best back as four floats.
"""
function radius_visible(bt::Float32, b1::Float32, b2::Float32, bcell::Float32,
                        pr::Core.LLVMPtr{Float32, Metal.AS.Device},
                        out::Core.LLVMPtr{RadiusOut,1})
    e = RadiusElems(PtrVec(pr))
    nb = Mantle.procedural_candidate(e, RadiusHit(bt, b1, b2, unsafe_trunc(UInt32, bcell)))
    Base.unsafe_store!(out, RadiusOut((nb.t, nb.ξ₁, nb.ξ₂, Float32(nb.cell))))
    return nothing
end

const RADIUS_TT = Tuple{Float32, Float32, Float32, Float32,
                        Core.LLVMPtr{Float32, Metal.AS.Device},
                        Core.LLVMPtr{RadiusOut,1}}

# The candidate builtins arrive as SEVEN SCALARS in the order
# `dx, dy, dz, ox, oy, oz, prim` — `stage_builtins!` sorts the globals it appends
# by NAME, and that is alphabetical, not declaration order. Scalars and not two
# `float3`s because MSL's `float3` is 16 bytes with a padding lane while AIR's
# `<3 x float>` is 12: the call is made, returns UNDEF, and the traversal commits
# nothing. Getting the ORDER wrong fails the same silent way.
const RADIUS_MSL = """
#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace raytracing;
kernel void trace_radius(
    instance_acceleration_structure accel [[buffer(0)]],
    device float4 *out [[buffer(1)]],
    visible_function_table<float4(float, float, float, float, device float*,
                                  float, float, float, float, float, float,
                                  uint)> tbl [[buffer(2)]],
    device const float4 *orig [[buffer(3)]],
    device const float4 *dirv [[buffer(4)]],
    device float *radii [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    ray r; r.origin = orig[tid].xyz; r.direction = dirv[tid].xyz;
    r.min_distance = 1e-4f; r.max_distance = 1e30f;
    intersection_query<instancing> q; q.reset(r, accel);

    float4 best = float4(1e30f, 0.0f, 0.0f, 0.0f);      // procedural_miss
    while (q.next()) {
        if (q.get_candidate_intersection_type() == intersection_type::bounding_box) {
            float3 o = q.get_candidate_ray_origin();
            float3 d = q.get_candidate_ray_direction();
            uint   p = q.get_candidate_primitive_id();
            float4 nb = tbl[0](best.x, best.y, best.z, best.w, radii,
                               d.x, d.y, d.z, o.x, o.y, o.z, p);
            // "the returned best improved" IS "commit_intersection! was called".
            if (nb.x < best.x) { best = nb; q.commit_bounding_box_intersection(nb.x); }
        }
    }
    bool hit = q.get_committed_intersection_type() == intersection_type::bounding_box;
    out[tid] = hit ? float4(1.0f, best.x, best.w, 0.0f) : float4(0.0f, 0.0f, -1.0f, 0.0f);
}
"""

@testset "Metal: the PORTABLE procedural protocol traces" begin
    be  = Metal.MetalBackend()
    dev = Metal.device()
    nb  = 4
    radii = Float32[0.75, 0.5, 0.9, 0.6]

    aabbs = [Mantle.AABB(Point3f(2f0*i - 1, -1, -1), Point3f(2f0*i + 1, 1, 1))
             for i in 0:(nb - 1)]
    blas = Mantle.build_accel!(Mantle.batchqueue(Mantle.Device())) do ctx
        Mantle.build_blas_aabb(ctx, aabbs)
    end
    hw = Mantle.HWTLAS{Raycore.Triangle{UInt32}}(be)
    push!(hw, blas, Mat4f(LinearAlgebra.I); instance_id = UInt32(0))
    Raycore.sync!(hw)
    drad = Metal.MtlVector{Float32}(radii)

    # `:candidate`, not `:visible`: a procedural candidate always takes the seven
    # candidate builtins, whatever its body reads, because the MSL above declares
    # the table's type by hand and cannot know.
    cfg = Metal.compiler_config(dev; stage = :candidate, name = "radius_visible")
    job = Metal.GPUCompiler.CompilerJob(
        Metal.methodinstance(typeof(radius_visible), RADIUS_TT), cfg)
    vfn = Metal.MTL.MTLFunction(
        Metal.MTL.MTLLibraryFromData(dev, Metal.compile_to_metallib(job).metallib),
        "radius_visible")
    @test vfn.functionType == Metal.MTL.MTLFunctionTypeVisible

    pfn  = Metal.MTL.MTLFunction(Metal.MTL.MTLLibrary(dev, RADIUS_MSL), "trace_radius")
    desc = Metal.MTLComputePipelineDescriptor()
    desc.computeFunction = pfn
    lf = Metal.MTL.MTLLinkedFunctions(); lf.functions = Metal.NSArray([vfn])
    desc.linkedFunctions = lf
    desc.maxCallStackDepth = 4
    pipe = Metal.MTLComputePipelineState(dev, desc)
    handle = Metal.MTL.function_handle(pipe, vfn)
    @test handle !== nothing
    tbl = Metal.MTL.MTLVisibleFunctionTable(
        pipe, Metal.MTL.MTLVisibleFunctionTableDescriptor(1))
    Metal.MTL.set_function!(tbl, handle, 0)

    ys  = Float32[0.0, 0.4, 0.8, 1.2]
    nr  = length(ys)
    org = Metal.MtlVector{NTuple{4,Float32}}([(-5f0, y, 0f0, 0f0) for y in ys])
    dir = Metal.MtlVector{NTuple{4,Float32}}([(1f0, 0f0, 0f0, 0f0) for _ in ys])
    out = Metal.MtlVector{NTuple{4,Float32}}(undef, nr)

    cb = Metal.MTL.MTLCommandBuffer(Metal.MTL.MTLCommandQueue(dev))
    Metal.MTL.MTLComputeCommandEncoder(cb) do cce
        Metal.MTL.set_function!(cce, pipe)
        Metal.MTL.set_acceleration_structure!(cce, hw.built.handle, 1)
        Metal.MTL.use!(cce, [b.handle for b in hw.blas_list], Metal.MTL.ReadUsage)
        Metal.MTL.set_buffer!(cce, out.data[], 0, 2)
        Metal.MTL.set_visible_function_table!(cce, tbl, 2)
        Metal.MTL.set_buffer!(cce, org.data[], 0, 4)
        Metal.MTL.set_buffer!(cce, dir.data[], 0, 5)
        Metal.MTL.set_buffer!(cce, drad.data[], 0, 6)
        Metal.MTL.dispatchThreads!(cce, Metal.MTL.MTLSize(nr,1,1), Metal.MTL.MTLSize(nr,1,1))
    end
    Metal.MTL.commit!(cb)
    Metal.MTL.wait_completed(cb)
    got = Array(out)

    # The nearest sphere along +x from -5, per ray, on the host. PER-BOX radii,
    # so this also checks the payload's device array really was read: a
    # traversal that ignored `e.r` would agree only for box 0.
    for (i, y) in enumerate(ys)
        bt = Inf32; bc = -1
        for p in 0:(nb - 1)
            ex = -5f0 - 2f0*p
            disc = ex*ex - (ex*ex + y*y - radii[p+1]^2)
            disc < 0f0 && continue
            t = -ex - sqrt(disc)
            (t > 1f-4 && t < bt) && (bt = t; bc = p)
        end
        if bc < 0
            @test got[i][1] == 0f0
        else
            @test got[i][1] == 1f0
            @test got[i][2] ≈ bt rtol=1f-5
            @test Int(got[i][3]) == bc + 1        # the payload reports 1-based
        end
    end
end

# ── The whole path: `Raycore.closest_hit` ───────────────────────────────────
#
# The two testsets above drive the traversal from MSL written out by hand, which
# pins the MECHANISM. This one goes through `src/metal/procedural.jl` — the
# plumbing a caller actually gets — and so pins the parts that file owns and
# nothing else checks: the candidate compiled as a `:candidate` stage, its
# registration as a TABLE-reachable function, the payload packed into the scene
# argument buffer, and the four-float round trip of the payload's best hit.
#
# Every one of those fails SILENTLY when wrong. A table entry pointing at an
# inlined function, a builtin order off by one, a dropped `best` slot — all of
# them link, dispatch, complete, and report a miss or an off-by-one cell.

Mantle.procedural_commit(::RadiusElems, ::RadiusHit, prim, t, empty) = empty
# The cell rides out in the barycentric's first slot, which is where a payload's
# own parameter travels — see `procedural_bary`.
Mantle.procedural_bary(::RadiusElems, b::RadiusHit) =
    StaticArrays.SVector{3,Float32}(Float32(b.cell), 0f0, 0f0)
Adapt.adapt_structure(to, e::RadiusElems) = RadiusElems(Adapt.adapt(to, e.r))

@kernel function closest_hit_probe!(hits, ts, cells, @Const(ys), accel, ox)
    i = @index(Global)
    r = Raycore.Ray(o = Point3f(ox, ys[i], 0f0), d = Vec3f(1, 0, 0), t_max = 1f30)
    hit, prim, t, bary, inst = Raycore.closest_hit(accel, r)
    hits[i]  = hit ? Int32(1) : Int32(0)
    ts[i]    = hit ? t : -1f0
    cells[i] = hit ? UInt32(bary[1]) : UInt32(0)
end

@testset "Metal: closest_hit traces procedural geometry" begin
    be  = Metal.MetalBackend()
    radii = Float32[0.75, 0.5, 0.9, 0.6]
    nb  = length(radii)

    aabbs = [Mantle.AABB(Point3f(2f0*i - 1, -1, -1), Point3f(2f0*i + 1, 1, 1))
             for i in 0:(nb - 1)]
    blas = Mantle.build_accel!(Mantle.batchqueue(Mantle.Device())) do ctx
        Mantle.build_blas_aabb(ctx, aabbs)
    end
    hw = Mantle.HWTLAS{Raycore.Triangle{UInt32}}(be)
    push!(hw, blas, Mat4f(LinearAlgebra.I); instance_id = UInt32(0))
    # Setting `procedural` is what makes `sync!` build the candidate's table and
    # lay the scene buffer out as `{accel, table, payload}` instead of one slot.
    hw.procedural = RadiusElems(Metal.MtlVector{Float32}(radii))
    Raycore.sync!(hw)
    @test hw.proc_table !== nothing
    @test length(hw.scene_buf) == 3

    # …and the accel's TYPE is what picks the traversal: `procedural === nothing`
    # is a static fact, so a triangles-only scene compiles to the call it always
    # made and pays nothing for this path existing.
    dev_accel = Metal.mtlconvert(Adapt.adapt(be, hw))
    @test dev_accel isa Mantle.MetalAdaptedProc
    @test !(dev_accel isa Mantle.MetalAdaptedTri)

    ys = Float32[0.0, 0.4, 0.8, 1.2]
    n  = length(ys)
    dy = KA.allocate(be, Float32, n); copyto!(dy, ys)
    dh = KA.zeros(be, Int32, n); dt = KA.zeros(be, Float32, n); dc = KA.zeros(be, UInt32, n)
    closest_hit_probe!(be, 64)(dh, dt, dc, dy, Adapt.adapt(be, hw), -5f0; ndrange = n)
    KA.synchronize(be)
    gh, gt, gc = Array(dh), Array(dt), Array(dc)

    for (i, y) in enumerate(ys)
        bt = Inf32; bc = 0
        for p in 0:(nb - 1)
            ex = -5f0 - 2f0*p
            disc = ex*ex - (ex*ex + y*y - radii[p + 1]^2)
            disc < 0f0 && continue
            t = -ex - sqrt(disc)
            (t > 1f-4 && t < bt) && (bt = t; bc = p + 1)
        end
        if bc == 0
            @test gh[i] == Int32(0)
        else
            @test gh[i] == Int32(1)
            @test gt[i] ≈ bt rtol=1f-5
            # 1-BASED, from the payload's own `cell`. Rebuilding this from the
            # primitive id instead is off by one, and `t` stays right — so a test
            # that checked only the distance would pass.
            @test Int(gc[i]) == bc
        end
    end
end
