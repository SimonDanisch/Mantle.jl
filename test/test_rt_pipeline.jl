"""
A ray-tracing PIPELINE (raygen, closest-hit and miss shaders written in Julia)
traced through the graph with `Mantle.trace!`.

The other way to trace is `Raycore.closest_hit` from an ordinary kernel
(`test_trace_closest_hit.jl`). A pipeline is what Hikari's hardware path takes
where the device has one (`trace_pass!` in its `rt-pipeline.jl`): the driver
invokes the hit shaders through a shader binding table. Metal has no pipeline of
this kind, it traverses from inside the shading kernel, so everything that traces
here is gated on `supports_rt_pipeline`.

The shaders follow Hikari's pattern. The raygen only traces; closest-hit and miss
take the raygen's arguments (`chit_miss_take_args = true`) and write the result
at the launch index themselves. `rt_hit_object_trace_ray` followed by
`rt_reorder_thread` and `rt_hit_object_execute_shader` is the one trace verb: on a
device without shader execution reordering the first is a plain trace that runs
the hit shader, and the other two do nothing.

Where each testset came from:

  * The triangle grid was the device half of `test/vulkan/test_handwritten_rt.jl`,
    which assembled the same three shaders in SPIR-V by hand and dispatched them
    through the Vulkan backend's own pipeline objects. Same scene, same rays,
    same margins.
  * The cull mask was `test_trace_cull_mask.jl`, written against the `cull_mask`
    keyword of `trace_closest_hits!`: a ray sees an instance only when its cull
    mask and the instance's mask share a bit.
  * The device-sized ray count is the shape of every Hikari trace (one ray per
    surviving path, a number only the GPU knows). The only tests that traced
    over a device-written count were the `mwe_*_per_iter.jl` reproducers, which
    hunted a device loss none of them reproduced and asserted only that the
    device survived; they are gone, and this keeps the indirect trace covered.
"""

using Test, KernelAbstractions
import Raycore
using GeometryBasics: Point3f, GLTriangleFace
import GeometryBasics
include(joinpath(@__DIR__, "testbackend.jl"))
const KA = KernelAbstractions

# The ray starts at `origins[i]` and goes along +z, from t = 0.001 to t = 100:
# the constants the hand-written raygen used.
function rtp_raygen(out, origins, mask)
    i = Int(Mantle.rt_launch_id_x()) + 1
    @inbounds o = origins[i]
    Mantle.rt_hit_object_trace_ray(UInt32(0), mask, UInt32(0), UInt32(0), UInt32(0),
                                   o[1], o[2], o[3], 1f-3, 0f0, 0f0, 1f0, 100f0)
    Mantle.rt_reorder_thread()
    Mantle.rt_hit_object_execute_shader()
    return nothing
end

function rtp_closesthit(out, origins, mask)
    i = Int(Mantle.rt_launch_id_x()) + 1
    @inbounds out[i] = Mantle.rt_ray_tmax()
    return nothing
end

function rtp_miss(out, origins, mask)
    i = Int(Mantle.rt_launch_id_x()) + 1
    @inbounds out[i] = -1f0
    return nothing
end

"""Trace one ray per origin over `n` (a count or a `DeviceRange`); what each ray
wrote, `-2` where no shader ran. The device caches the compiled pipeline by its
shaders, so a description per call compiles once."""
function rtp_trace(dev, hw, origins, mask, n = length(origins))
    pipe = Mantle.RayTracingPipeline(; raygen = rtp_raygen, closest_hit = rtp_closesthit,
                                     miss = rtp_miss, chit_miss_take_args = true)
    out = Mantle.Buffer(dev, fill(-2f0, length(origins)))
    org = Mantle.Buffer(dev, origins)
    g = Mantle.Graph(dev)
    Mantle.trace!(g, pipe, hw, (out, org, mask), n)
    Mantle.runonce!(g)
    KA.synchronize(TESTBACKEND)
    return Array(Mantle.storage(out))
end

"""A structure holding the one triangle `verts`, with `kw` for its instance."""
function rtp_structure!(hw, verts; kw...)
    mesh = GeometryBasics.normal_mesh(GeometryBasics.Mesh(verts, [GLTriangleFace(1, 2, 3)]))
    push!(hw, mesh; kw...)
    return hw
end

# The grid of the hand-written test: 16x16 rays from z = -1, x and y over
# [-0.5, 1.5), against the triangle (0,0,0), (1,0,0), (0,1,0). A hit is at t = 1.
const RTP_W = 16
rtp_xy(ix, iy) = (Float32(ix) / Float32(RTP_W) * 2f0 - 0.5f0, Float32(iy) / Float32(RTP_W) * 2f0 - 0.5f0)
rtp_grid() = [Point3f(rtp_xy(ix, iy)..., -1f0) for iy in 0:(RTP_W - 1) for ix in 0:(RTP_W - 1)]
rtp_unittriangle() = rtp_structure!(Mantle.HWTLAS{Raycore.Triangle{UInt32}}(TESTBACKEND),
                                    [Point3f(0, 0, 0), Point3f(1, 0, 0), Point3f(0, 1, 0)])

@testset "a ray-tracing pipeline implies hardware traversal" begin
    # Was `test_rayquery_device_probe.jl`, which read the Vulkan context's probe:
    # a driver with the ray-tracing pipeline extension has ray query too (RADV,
    # lavapipe, NVIDIA and AMD on Windows all do). The capabilities say the same.
    @test !Mantle.supports_rt_pipeline(TESTBACKEND) || Mantle.supports_hwtlas(TESTBACKEND)
end

if Mantle.supports_rt_pipeline(TESTBACKEND) && Mantle.supports_hwtlas(TESTBACKEND)

@testset "a pipeline traces a triangle: closest-hit and miss shaders run" begin
    dev = Mantle.Device(TESTBACKEND)
    hw = rtp_unittriangle()
    Raycore.sync!(hw)
    result = rtp_trace(dev, hw, rtp_grid(), UInt32(0xFF))

    # Rays clearly inside the triangle hit at t = 1, rays clearly outside miss.
    # Rays near an edge may go either way (hardware edge rules).
    eps = 0.15f0
    n_hits = n_misses = 0
    for iy in 0:(RTP_W - 1), ix in 0:(RTP_W - 1)
        x, y = rtp_xy(ix, iy)
        r = result[iy * RTP_W + ix + 1]
        if x > eps && y > eps && x + y < 1f0 - eps
            @test r ≈ 1f0 atol = 0.01f0
            n_hits += 1
        elseif x < -eps || y < -eps || x + y > 1f0 + eps
            @test r == -1f0
            n_misses += 1
        else
            @test isapprox(r, 1f0; atol = 0.01f0) || r == -1f0
        end
    end
    @test n_hits > 0
    @test n_misses > 0
end

@testset "the cull mask selects instances by their mask" begin
    # Two triangles stacked along the ray: instance A at z = 5 with mask 0x01,
    # instance B at z = 10 with mask 0x02.
    dev = Mantle.Device(TESTBACKEND)
    tri(z) = [Point3f(-1, -1, z), Point3f(1, -1, z), Point3f(0, 1, z)]
    hw = Mantle.HWTLAS{Raycore.Triangle{UInt32}}(TESTBACKEND)
    rtp_structure!(hw, tri(5f0); instance_id = UInt32(1), instance_mask = UInt8(0x01))
    rtp_structure!(hw, tri(10f0); instance_id = UInt32(2), instance_mask = UInt8(0x02))
    Raycore.sync!(hw)
    o = [Point3f(0, 0, 0)]

    # 0xFF sees both and hits the nearer, A.
    @test only(rtp_trace(dev, hw, o, UInt32(0xFF))) ≈ 5f0 atol = 1f-2
    # 0x01 sees only A.
    @test only(rtp_trace(dev, hw, o, UInt32(0x01))) ≈ 5f0 atol = 1f-2
    # 0x02 sees only B: the ray passes through A.
    @test only(rtp_trace(dev, hw, o, UInt32(0x02))) ≈ 10f0 atol = 1f-2
end

@testset "a trace over a device-sized count traces exactly that many rays" begin
    dev = Mantle.Device(TESTBACKEND)
    hw = rtp_unittriangle()
    Raycore.sync!(hw)
    origins = rtp_grid()
    n = length(origins)
    k = 100   # rows 0 to 5 and the start of row 6; row 5 crosses the triangle

    full = rtp_trace(dev, hw, origins, UInt32(0xFF))
    nrays = Mantle.Buffer(dev, Int32[k])
    got = rtp_trace(dev, hw, origins, UInt32(0xFF), Mantle.DeviceRange(nrays; max = n))

    # The first `k` rays are traced as the fixed-count trace traced them…
    @test any(r -> isapprox(r, 1f0; atol = 0.01f0), got[1:k])
    @test got[1:k] == full[1:k]
    # …and no shader ran for a ray past the count. The raygen does not bound
    # itself, as Hikari's does not: a trace launches one ray per counted element.
    @test all(==(-2f0), got[(k + 1):n])
end

else
    @info "no ray-tracing pipeline on this backend; skipping"
end
