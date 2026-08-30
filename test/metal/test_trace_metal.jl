"""
Hardware ray traversal on Metal: `Mantle.trace_closest_hits!`.

Traversal is the one kernel this backend cannot write in Julia.
`metal::raytracing::intersector<>` is a C++ class template the Metal frontend
instantiates and inlines — there is no `air.intersect*` symbol for a
`@typed_ccall` to name — so `src/metal/trace.jl` carries MSL source compiled at
runtime with `newLibraryWithSource:`.

That makes the Julia↔MSL struct layout an ABI with no compiler checking it,
which is what most of this file is about. `RTRay` and `RTHitResult` are
Raycore's, mirrored field-for-field in the MSL; a field inserted on either side
shears every hit record and nothing would report it.
"""

using Test, Mantle, Metal, Raycore
const MTL = Metal.MTL
const MEXT = Base.get_extension(Mantle, :MantleMetalExt)

# One BLAS from a flat Float32 vertex list, three floats per vertex.
function _blas_from(d, verts::Vector{Float32})
    @assert length(verts) % 9 == 0 "three vertices of three floats per triangle"
    buf = MTL.MTLBuffer(d.dev, sizeof(verts); storage = Metal.SharedStorage)
    unsafe_copyto!(convert(Ptr{Float32}, MTL.contents(buf)), pointer(verts), length(verts))
    return MEXT.build_blas(d, buf, length(verts) ÷ 9)
end

_ray(ox, oy, oz, dx, dy, dz; tmin = 0f0, tmax = 1f30) =
    Raycore.RTRay(Float32(ox), Float32(oy), Float32(oz), tmin,
                  Float32(dx), Float32(dy), Float32(dz), tmax)

_emptyhits(n) = MtlArray(fill(Raycore.RTHitResult(0, 0f0, 0, 0, 0f0, 0f0, 0, 0), n))

@testset "Metal: the hit-record ABI is the one MSL was written against" begin
    # `src/metal/trace.jl` asserts these at load; repeating them here is what
    # turns "the extension failed to load" into a message naming the cause.
    @test sizeof(Raycore.RTRay) == 32
    @test sizeof(Raycore.RTHitResult) == 32
    @test isbitstype(Raycore.RTRay) && isbitstype(Raycore.RTHitResult)
    @test fieldnames(Raycore.RTRay) ==
          (:origin_x, :origin_y, :origin_z, :tmin, :dir_x, :dir_y, :dir_z, :tmax)
    @test fieldnames(Raycore.RTHitResult) ==
          (:hit, :t, :primitive_id, :instance_custom_index, :bary_u, :bary_v, :instance_id, :_pad2)
    # No padding anywhere: the MSL structs are declared as plain scalars, so a
    # gap on the Julia side would misalign every field after it.
    @test all(fieldoffset(Raycore.RTHitResult, i) == 4(i - 1) for i in 1:8)
    @test all(fieldoffset(Raycore.RTRay, i) == 4(i - 1) for i in 1:8)
end

@testset "Metal: rays hit, miss, and report where" begin
    d = Mantle.Device(Mantle.MetalAPI())
    # Unit triangle in the z=0 plane, spanning (0,0)-(1,0)-(0,1).
    blas = _blas_from(d, Float32[0,0,0, 1,0,0, 0,1,0])
    tlas = Mantle.build_accel!(d, [blas], [Mantle.identity_transform()])

    # Three inside, three outside. (0.9, 0.9) is outside because 0.9+0.9 > 1 —
    # a bounding-box test would wrongly report it as a hit.
    inside  = [(0.2, 0.2), (0.1, 0.1), (0.3, 0.3)]
    outside = [(2.0, -2.0), (-1.0, 1.0), (0.9, 0.9)]
    pts = vcat(inside, outside)
    rays = MtlArray([_ray(x, y, 5, 0, 0, -1) for (x, y) in pts])
    hits = _emptyhits(length(pts))

    Mantle.trace_closest_hits!(hits, rays, tlas, length(pts))
    H = Array(hits)

    for i in 1:3
        @test H[i].hit == 1
        @test H[i].t ≈ 5f0 atol=1e-5
        @test H[i].primitive_id == 0
        # Barycentrics identify the point: for this triangle they are the xy.
        @test H[i].bary_u ≈ Float32(pts[i][1]) atol=1e-5
        @test H[i].bary_v ≈ Float32(pts[i][2]) atol=1e-5
    end
    for i in 4:6
        @test H[i].hit == 0
        # A miss reports tmax rather than a stale or uninitialised distance.
        @test H[i].t == 1f30
    end
end

@testset "Metal: the NEAREST hit wins" begin
    # Two parallel triangles. A ray from above must report the near one; a
    # traversal that returns any hit rather than the closest passes every test
    # with a single triangle in the scene.
    d = Mantle.Device(Mantle.MetalAPI())
    near = _blas_from(d, Float32[0,0,1, 1,0,1, 0,1,1])
    far  = _blas_from(d, Float32[0,0,0, 1,0,0, 0,1,0])
    tlas = Mantle.build_accel!(d, [far, near],
                               [Mantle.identity_transform(), Mantle.identity_transform()])

    rays = MtlArray([_ray(0.25, 0.25, 5, 0, 0, -1)])
    hits = _emptyhits(1)
    Mantle.trace_closest_hits!(hits, rays, tlas, 1)
    H = Array(hits)
    @test H[1].hit == 1
    @test H[1].t ≈ 4f0 atol=1e-5          # the z=1 triangle, not the z=0 one
    @test H[1].instance_id == 1           # …which is instance 1, not 0

    # Shooting the other way finds the far one first.
    rays2 = MtlArray([_ray(0.25, 0.25, -5, 0, 0, 1)])
    hits2 = _emptyhits(1)
    Mantle.trace_closest_hits!(hits2, rays2, tlas, 1)
    @test Array(hits2)[1].t ≈ 5f0 atol=1e-5
    @test Array(hits2)[1].instance_id == 0
end

@testset "Metal: tmin and tmax bound the search" begin
    d = Mantle.Device(Mantle.MetalAPI())
    blas = _blas_from(d, Float32[0,0,0, 1,0,0, 0,1,0])
    tlas = Mantle.build_accel!(d, [blas], [Mantle.identity_transform()])

    # The triangle is at t = 5 along this ray.
    hits = _emptyhits(3)
    rays = MtlArray([_ray(0.25, 0.25, 5, 0, 0, -1; tmax = 4f0),   # stops short
                     _ray(0.25, 0.25, 5, 0, 0, -1; tmin = 6f0),   # starts past
                     _ray(0.25, 0.25, 5, 0, 0, -1)])              # unbounded
    Mantle.trace_closest_hits!(hits, rays, tlas, 3)
    H = Array(hits)
    @test H[1].hit == 0
    @test H[2].hit == 0
    @test H[3].hit == 1
end

@testset "Metal: a batch big enough to span many threadgroups" begin
    # The dispatch is `dispatchThreads!` with a non-multiple count, so the
    # kernel's own `tid >= n` guard is what keeps the tail from writing past
    # the end. 10_007 is prime: every threadgroup size leaves a partial one.
    d = Mantle.Device(Mantle.MetalAPI())
    blas = _blas_from(d, Float32[0,0,0, 1,0,0, 0,1,0])
    tlas = Mantle.build_accel!(d, [blas], [Mantle.identity_transform()])

    n = 10_007
    # Alternate hit/miss so a kernel that wrote a constant would be caught.
    rs = [iseven(i) ? _ray(0.25, 0.25, 5, 0, 0, -1) : _ray(9, 9, 5, 0, 0, -1) for i in 1:n]
    hits = _emptyhits(n)
    Mantle.trace_closest_hits!(hits, MtlArray(rs), tlas, n)
    H = Array(hits)
    @test count(h -> h.hit == 1, H) == count(iseven, 1:n)
    @test all(H[i].hit == 1 for i in 2:2:n)
    @test all(H[i].hit == 0 for i in 1:2:n)
end

@testset "Metal: bad arguments are refused, not silently traced" begin
    d = Mantle.Device(Mantle.MetalAPI())
    blas = _blas_from(d, Float32[0,0,0, 1,0,0, 0,1,0])
    tlas = Mantle.build_accel!(d, [blas], [Mantle.identity_transform()])
    rays = MtlArray([_ray(0.25, 0.25, 5, 0, 0, -1)])
    hits = _emptyhits(1)

    @test_throws ArgumentError Mantle.trace_closest_hits!(hits, rays, tlas, 2)   # more rays than exist
    @test_throws ArgumentError Mantle.trace_closest_hits!(_emptyhits(0), rays, tlas, 1)
    # Wrong element type is an ABI mismatch, which is exactly what must not be
    # reinterpreted silently.
    @test_throws ArgumentError Mantle.trace_closest_hits!(Metal.zeros(Float32, 4), rays, tlas, 1)

    # Zero rays is legal and does nothing.
    @test Mantle.trace_closest_hits!(hits, rays, tlas, 0) === hits
end
