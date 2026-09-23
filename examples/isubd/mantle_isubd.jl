# Adaptive tessellation of curved FEM cells, on Mantle.
#
# The port of `reference.jl` (vendored from koehlerson/mantle_mwe). The
# reference's three passes map onto the backend like this:
#
#   A  update_keys   compute dispatch + prefix scan + scatter  — `refine!`
#   B  decode_keys   a MESH SHADER, three vertices per key     — `isubd_mesh`
#   C  shade         a fragment shader over the coefficients   — `isubd_frag`
#
# Pass B is why this example exists: a mesh stage is one workgroup per BATCH of
# keys, each invocation expanding its own key into a triangle, with no vertex
# buffer and no index buffer anywhere. The geometry never exists in memory.
#
# ── Why the math is written again here ──────────────────────────────────────
#
# The reference's estimator and key math are host code: `BASE_CORNERS` and
# `BASE_CELL` are const global `Vector`s, and `monomials` indexes a `Vector` of
# exponents inside a non-`Val` `ntuple`. None of that survives on a device — a
# const global `Vector` is a host pointer, and an `ntuple` whose length is not a
# compile-time constant does not unroll.
#
# The monomials are therefore written out, and the base triangles are passed as
# BUFFERS. Making them tuple constants instead does not work and should not:
# indexing a tuple with a runtime key index puts a comparison inside an LLVM
# constant expression, which the SPIR-V emitter refuses outright — and base
# triangles are mesh data anyway, which a second mesh would force into a buffer.
#
# It is the same arithmetic otherwise, and `test_isubd_mesh.jl` checks that by
# running both and comparing. That is the point of keeping the reference
# unedited beside it: it caught a swapped pair of base corners here already.

using Mantle, Lava, GeometryBasics, Raycore, LinearAlgebra, FileIO, Colors
import Hikari

# The isoparametric map lives in `Hikari/src/fem.jl` and is imported, not
# repeated. It was written here first and then a second time for the ray path,
# which is exactly the duplication this example exists to argue against: the
# mesh shader, the fragment shader and the ray tracer all evaluate the SAME
# element, so it has one home. `test_isubd_mesh.jl` compares these against an
# unedited CPU reference, and so is the gate on Hikari's copy.
using Hikari: monomials, eval_poly, eval_poly_grad,
              surfacepoint, surfacetangents, surfacenormal,
              intersect_element, element_aabb
const NMONO = Hikari.FEM_NMONO

"""The element's 2D position — the isoparametric map without the field."""
@inline eval_position(cx, cy, cell, a, b) =
    (eval_poly(cx, cell, a, b), eval_poly(cy, cell, a, b))

# ── The error estimator ─────────────────────────────────────────────────────
#
# Sample the deviation of the linear interpolation over the triangle, at all
# three EDGE midpoints and the centroid, for geometry and field, and take
# whichever is worse relative to its tolerance.
#
# Double precision throughout, deliberately: it measures deviations far below
# Float32 noise, and rounding them makes the criterion decide on garbage. The
# positions it feeds into rendering are Float32; the decision is not.
import KernelInterface
using KernelAbstractions: @kernel, @index, get_backend
using Lava: LavaDeviceArray

const KI = KernelInterface

# ── The cell data, as tuple constants ───────────────────────────────────────

const NCELLS = 2


# Base triangles: each cell fanned from its centre over its four element edges,
# so a base triangle's split edge is always an element edge.
#
# These are BUFFERS, not constants, and that is not only a device restriction:
# indexing a tuple with a runtime key index makes LLVM emit a comparison inside
# a constant expression, which the SPIR-V emitter refuses
# (`Unsupported ConstantExpr opcode: LLVMICmp`). But the shape is right anyway —
# base triangles are mesh data. Here there are eight of them because there are
# two hard-coded cells; in FerriteViz they come from the grid, and a constant
# would have to become a buffer the moment a second mesh existed.
#
# `basecorners` is flat `Float64`: corner `j` of triangle `t` at
# `(t - 1) * 6 + (j - 1) * 2 + 1`. Flat rather than an array of nested tuples so
# the indexing is the same arithmetic on a `Vector` and on a device array.

const NBASE = 4 * NCELLS

# `ncells`, not the constant: the reference comparison has two cells and the
# demo has as many as its mesh does. Four base triangles per cell either way.
function basetriangles(ncells::Integer = NCELLS)
    rim = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
    corners = Float64[]
    cells = Int32[]
    for c in 1:ncells, i in 1:4
        a, b = rim[i], rim[mod1(i + 1, 4)]
        # `(b, centre, a)`: the split edge is always `(v1, v3)`, so the rim edge
        # goes there — that is what makes a base triangle's split edge an element
        # edge. The other order gives eight plausible triangles that refine along
        # the fan diagonals instead.
        append!(corners, (b[1], b[2], 0.0, 0.0, a[1], a[2]))
        push!(cells, Int32(c))
    end
    return corners, cells
end

@inline function basecorner(basecorners, t::Integer, j::Integer)
    o = (t - 1) * 6 + (j - 1) * 2
    return (basecorners[o + 1], basecorners[o + 2])
end

# ── Keys ────────────────────────────────────────────────────────────────────
# One UInt64 per triangle: the base triangle, and the path of longest-edge
# bisections that produced it, led by a sentinel bit. Parent and children are
# shifts, so nothing is stored per triangle and a key reconstructs its own
# geometry — which is what lets pass B run with no buffers.

@inline key_base(k::UInt64) = Int(k >> 32)
@inline key_leb(k::UInt64) = k % UInt32
@inline make_key(b::Integer, leb::UInt32) = (UInt64(b) << 32) | UInt64(leb)
@inline root_key(b::Integer) = make_key(b, UInt32(1))
@inline key_depth(k::UInt64) = 31 - leading_zeros(key_leb(k))
@inline key_parent(k::UInt64) = make_key(key_base(k), key_leb(k) >> 1)
@inline key_child(k::UInt64, i) = make_key(key_base(k), key_leb(k) << 1 | UInt32(i))
@inline is_child0(k::UInt64) = iseven(key_leb(k))

"""
Corners of a key's triangle in its cell's reference coordinates.

One 3x3 barycentric bisection folded per path bit. The loop is bounded by the
key's depth, so it is a runtime trip count — fine on both sides, because the
body is branch-free arithmetic on three weights.
"""
@inline function key_corners(k::UInt64, basecorners)
    b = key_base(k)
    c1 = basecorner(basecorners, b, 1)
    c2 = basecorner(basecorners, b, 2)
    c3 = basecorner(basecorners, b, 3)
    w1 = (1.0, 0.0, 0.0); w2 = (0.0, 1.0, 0.0); w3 = (0.0, 0.0, 1.0)
    leb = key_leb(k)
    for i in (key_depth(k) - 1):-1:0
        m = ((w1[1] + w3[1]) / 2, (w1[2] + w3[2]) / 2, (w1[3] + w3[3]) / 2)
        if ((leb >> i) & 1) == UInt32(0)
            w1, w2, w3 = w1, m, w2
        else
            w1, w2, w3 = w2, m, w3
        end
    end
    mix(w) = (w[1] * c1[1] + w[2] * c2[1] + w[3] * c3[1],
              w[1] * c1[2] + w[2] * c2[2] + w[3] * c3[2])
    return mix(w1), mix(w2), mix(w3)
end



const SAMPLES = ((0.5, 0.0, 0.5), (0.5, 0.5, 0.0), (0.0, 0.5, 0.5),
                 (1 / 3, 1 / 3, 1 / 3))

@inline function excess_levels(k::UInt64, cx, cy, cf, basecorners, basecell, geo_tol, fld_tol)
    cell = Int(basecell[key_base(k)])
    p1, p2, p3 = key_corners(k, basecorners)
    x1 = eval_position(cx, cy, cell, p1...); f1 = eval_poly(cf, cell, p1...)
    x2 = eval_position(cx, cy, cell, p2...); f2 = eval_poly(cf, cell, p2...)
    x3 = eval_position(cx, cy, cell, p3...); f3 = eval_poly(cf, cell, p3...)
    # `ntuple(…, Val(…))` again, for the same reason as `eval_poly`: `for
    # (a, b, c) in SAMPLES` indexes the tuple at a runtime position, and that is
    # the third place this file hit `Unsupported ConstantExpr opcode: LLVMICmp`.
    # With `Val` each `si` is a literal and the whole sampling unrolls.
    per = ntuple(Val(length(SAMPLES))) do si
        a, b, c = SAMPLES[si]
        q1 = a * p1[1] + b * p2[1] + c * p3[1]
        q2 = a * p1[2] + b * p2[2] + c * p3[2]
        ex, ey = eval_position(cx, cy, cell, q1, q2)
        lx = a * x1[1] + b * x2[1] + c * x3[1]
        ly = a * x1[2] + b * x2[2] + c * x3[2]
        return (sqrt((ex - lx)^2 + (ey - ly)^2),
                abs(eval_poly(cf, cell, q1, q2) - (a * f1 + b * f2 + c * f3)))
    end
    geo_err = maximum(map(first, per))
    fld_err = maximum(map(last, per))
    # A level buys a factor two: the deviation is O(h^2) and an edge halves
    # every second bisection.
    return max(log2(max(geo_err, 1e-16) / geo_tol), log2(max(fld_err, 1e-16) / fld_tol))
end

# ── The demo's mesh ─────────────────────────────────────────────────────────
#
# `reference.jl`'s two cells are a TEST FIXTURE — two quadrilaterals with
# hand-written nodes, which is what the CPU comparison needs and not what a
# picture wants. The demo builds its own, and picks a shape whose BOUNDARY is
# curved: a tessellated rim shows chords and an exactly-solved one does not, so
# the argument this example exists to make is visible on the silhouette rather
# than only in the shading.

"""The nine Q2 nodes of a quadrilateral, in the reference order."""
const Q2_NODES = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0),
                  (0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, 0.0))

"""
Nodal values to monomial coefficients: the 9x9 change of basis, inverted once.

The same solve `reference.jl` does, and it has to be the same or the two would
be evaluating different elements — `monomials` here is Hikari's, in the order
that file's `EXPONENTS` lists.
"""
function q2basis()
    V = zeros(NMONO, NMONO)
    for (k, (a, b)) in enumerate(Q2_NODES)
        V[k, :] .= collect(monomials(a, b))
    end
    return inv(V)
end

"""
    annulus(; nθ, nr, r0, r1, lobes, amp) -> cx, cy, cf

A wavy annular plate, as `FEM_NMONO x ncells` coefficient blocks.

`nθ * nr` quadratic elements over `r ∈ [r0, r1]`, displaced out of plane by a
`lobes`-fold mode. Two curved boundaries and no straight edge anywhere, which
is the point: every silhouette in the frame is one a rasteriser has to
approximate and the tracer does not.

The field IS the displacement, so the colour reads as height — which is what a
mode shape looks like in every FEM post-processor.
"""
function annulus(; nθ = 20, nr = 3, r0 = 0.40, r1 = 1.15, lobes = 3, amp = 0.46)
    Vinv = q2basis()
    ncells = nθ * nr
    cx = zeros(NMONO, ncells); cy = zeros(NMONO, ncells); cf = zeros(NMONO, ncells)
    xs = zeros(NMONO); ys = zeros(NMONO); fs = zeros(NMONO)
    c = 0
    for j in 1:nr, i in 1:nθ
        c += 1
        for (k, (a, b)) in enumerate(Q2_NODES)
            # The node's place in the patch: `a`, `b` are in [-1,1] over this
            # cell, `u` and `v` in [0,1] over the whole annulus.
            u = (i - 1 + 0.5 * (a + 1)) / nθ
            v = (j - 1 + 0.5 * (b + 1)) / nr
            # CLOCKWISE in θ, so that `cross(∂S/∂ξ₁, ∂S/∂ξ₂)` — ξ₁ around, ξ₂
            # outward — comes out pointing UP. Anticlockwise gives the same ring
            # with its normals inverted, which is legal and which every
            # one-sided consumer renders black.
            θ = -2π * u
            r = r0 + (r1 - r0) * v
            xs[k] = r * cos(θ)
            ys[k] = r * sin(θ)
            # Flat at the inner rim, full wave at the outer one — so the lobes
            # are where the eye is, on the outline.
            fs[k] = 0.5 + amp * cos(lobes * θ) * v
        end
        cx[:, c] = Vinv * xs; cy[:, c] = Vinv * ys; cf[:, c] = Vinv * fs
    end
    return cx, cy, cf
end

# ── PASS A — update the key buffer ──────────────────────────────────────────
#
# Per key: split (two children), merge (child 0 emits the parent, child 1
# emits nothing), or keep. classify -> exclusive scan -> scatter, so the output
# keeps the input's order and the buffer stays sorted.

@kernel function classify_kernel!(counts, @Const(keys), @Const(cx), @Const(cy), @Const(cf),
                                  @Const(basecorners), @Const(basecell),
                                  geo_tol, fld_tol, max_depth)
    i = @index(Global)
    @inbounds begin
        k = keys[i]
        d = key_depth(k)
        if d < max_depth && excess_levels(k, cx, cy, cf, basecorners, basecell, geo_tol, fld_tol) > 0
            counts[i] = Int32(2)
        elseif d > 0 && excess_levels(key_parent(k), cx, cy, cf, basecorners, basecell, geo_tol, fld_tol) <= 0
            counts[i] = is_child0(k) ? Int32(1) : Int32(0)
        else
            counts[i] = Int32(1)
        end
    end
end

@kernel function scatter_kernel!(out, @Const(keys), @Const(counts), @Const(offsets),
                                 @Const(cx), @Const(cy), @Const(cf),
                                 @Const(basecorners), @Const(basecell),
                                 geo_tol, fld_tol, max_depth)
    i = @index(Global)
    @inbounds begin
        k = keys[i]
        n = counts[i]
        o = offsets[i]
        if n == Int32(2)
            out[o + 1] = key_child(k, 0)
            out[o + 2] = key_child(k, 1)
        elseif n == Int32(1)
            d = key_depth(k)
            merging = d > 0 &&
                      excess_levels(key_parent(k), cx, cy, cf, basecorners, basecell, geo_tol, fld_tol) <= 0 &&
                      !(d < max_depth && excess_levels(k, cx, cy, cf, basecorners, basecell, geo_tol, fld_tol) > 0)
            out[o + 1] = merging ? key_parent(k) : k
        end
    end
end

"""
    refine!(keys, cx, cy, cf; geo_tol, fld_tol, max_depth) -> keys'

One update pass. `keys` may live anywhere — the scan is Base's on a `Vector` and
Mantle's on a device array, and the kernels follow `get_backend`.

The total is the one scalar that crosses back to the host, because the output
buffer has to be SIZED. Everything else stays where it is.
"""
function refine!(keys, cx, cy, cf, basecorners, basecell; geo_tol, fld_tol, max_depth)
    n = length(keys)
    counts = similar(keys, Int32, n)
    backend = get_backend(keys)
    classify_kernel!(backend, 64)(counts, keys, cx, cy, cf, basecorners, basecell,
                                  geo_tol, fld_tol, max_depth; ndrange = n)
    # Exclusive, so slot `i` starts where everything before it ended.
    offsets = accumulate(+, counts) .- counts
    total = Int(scalarat(offsets, n)) + Int(scalarat(counts, n))
    out = similar(keys, total)
    scatter_kernel!(backend, 64)(out, keys, counts, offsets, cx, cy, cf,
                                 basecorners, basecell,
                                 geo_tol, fld_tol, max_depth; ndrange = n)
    return out
end

scalarat(a::AbstractArray, i::Integer) = a[i]
scalarat(a::Mantle.LavaArray, i::Integer) = only(Array(view(a, i:i)))

"""
    refine(keys0, cx, cy, cf; geo_tol, fld_tol, max_depth) -> keys

Run update passes until the key set stops changing.

A real frame does ONE pass and lets the mesh chase the criterion; running to a
fixed point is what makes a test deterministic.
"""
function refine(keys0, cx, cy, cf, basecorners, basecell; geo_tol, fld_tol, max_depth = 12)
    keys = keys0
    for _ in 1:(max_depth + 1)
        new = refine!(keys, cx, cy, cf, basecorners, basecell; geo_tol, fld_tol, max_depth)
        length(new) == length(keys) && Array(new) == Array(keys) && break
        keys = new
    end
    return keys
end

rootkeys(nbase::Integer = NBASE) = UInt64[root_key(b) for b in 1:nbase]

# ── PASS B — the mesh shader ────────────────────────────────────────────────
#
# One workgroup per BATCH of `MESH_THREADS` keys, each invocation expanding its
# own key into one triangle. No vertex buffer, no index buffer: the geometry is
# reconstructed from the key inside the stage and never exists in memory.
#
# One key per WORKGROUP would also work and is simpler, but it wastes the other
# 31 lanes and pays a workgroup launch per triangle. At 1106 triangles that is
# 1106 workgroups instead of 35.

const MESH_THREADS = 32

# HALF the stroke width, in pixels, because each face strokes INWARD and a
# shared edge is stroked by both of its triangles — the two halves meet and make
# one line of `2 * STROKE_HALFPX`. Drawing the full width from both sides, which
# is what a naive barycentric wireframe does, gives every interior edge twice
# the width and twice the darkening.
const STROKE_HALFPX = 0.55f0
const STROKE_COLOR = (0.06f0, 0.07f0, 0.10f0)

# The half-width travels in `flags[2]` rather than a bare on/off, because it is
# in RENDER-TARGET pixels: a caller that supersamples and filters down has to
# scale it or the line comes out half as wide as it asked for.
strokehalf(w::Bool) = w ? STROKE_HALFPX : 0.0f0
strokehalf(w::Real) = Float32(w)

# NOT dimmed any more. Dimming existed because the first background was a
# measured outdoor scene bright and busy enough to take the eye off the
# subject; a plain sky over a neutral ground does not, and leaving it at full
# strength is what makes the raster's background match what Hikari shows —
# which matters, because the demo flips between them in place.
const BG_DIM = 1.0f0
const EXPOSURE = 1.4f0

# The surface's specular response. Diffuse plus a broad Fresnel reflection is
# what a wet stone looks like; the thing that reads as a SURFACE with a shape is
# a tight highlight, because its position on screen is a direct readout of where
# the normal points. It also costs nothing next to the Newton solve.
# Sky-dominant, not sun-dominant, and that is a property of the ENVIRONMENT
# rather than a taste. These were tuned against a measured outdoor HDRI with a
# hard sun in it; a sky dome over a neutral ground lights a surface almost
# entirely by its hemisphere, so a big `SUN_DIFFUSE` over a small `SKY_AMBIENT`
# left everything facing away from one direction nearly black — while Hikari,
# integrating the actual sky, lit it. The two renderers flip back and forth in
# place, so a mismatch here reads as the renderers disagreeing.
const SUN_DIFFUSE = 0.45f0
# MUCH larger than the diffuse term, and it has to be. At 0.65 the highlight was
# measurably present — 4% of the visible surface sits inside the lobe — and
# invisible anyway: it arrived as a 65% boost on a diffuse-plus-sky floor near
# 1.0, and `x/(1+x)` turned that into about 10% on screen. A highlight reads
# because it is the part of the frame the tone curve is compressing, so it has
# to start well past where the rest of the surface sits.
const SUN_SPECULAR = 2.0f0
const SUN_TINT = (1.0f0, 0.965f0, 0.90f0)
const SKY_AMBIENT = 0.95f0

"""The fixed view direction the ORTHOGRAPHIC test uses; a ray has its own."""
const VIEWDIR = Vec3f(0.0, 0.62, -0.79)

# ── The environment ─────────────────────────────────────────────────────────
#
# A measured HDRI, sampled by BOTH renderers through the same function — which
# is the only way the two can be compared. A procedural sky would be easier and
# would defeat the purpose: it has no sun disc, no horizon and no structure, so
# everything it lights comes back with one soft value from every direction and
# the frame reads flat whatever the material does. What makes a curved surface
# read as a surface is a REFLECTION with edges in it.
#
# EQUIRECTANGULAR, and chosen for that. The other candidates on this machine
# (`*_equiarea.exr`, and the `sky.exr` the raven scene uses) are equal-area
# OCTAHEDRAL maps — a diamond of sky inscribed in a square, with the ground
# folded into the corners. Reading one of those as equirectangular gives a
# plausible sky that is wrong everywhere, which is worth knowing before
# swapping the file.

# A plain SKY, and chosen for being plain. The first pick was a measured
# outdoor HDRI with a gravel path and grass in it, and at full resolution it
# was sharp, photographic and completely at war with the subject: the eye went
# to the texture of the ground instead of to the silhouette the demo is about.
# A background's job here is to light the surface and to be somewhere for it to
# stand, not to compete.
#
# SQUARE, so it is already the equal-area octahedral map Hikari samples; the
# raster path wants equirectangular and gets it from `equalarea_to_equirect`.
# Reading one as the other gives a plausible-looking sky that is wrong in every
# direction, which is a bug that survives inspection.
# The sky is COMPUTED, not photographed.
#
# Three backgrounds got us here. A measured outdoor HDRI at full resolution is
# sharp and photographic and takes the eye straight to the gravel. `sky.exr` is
# calm and is a sky DOME — nothing below the horizon, so the lower two thirds of
# the frame is black. A hand-made neutral ground under it fixes that and is
# flat: no sun, so no highlight to travel across a glossy surface and nothing
# for the shape to cast a shadow into.
#
# Hosek-Wilkie gives a real sky with a real sun in it, and Hikari already has
# it. `Makie.SunSkyLight` is what the traced half is handed; the same call bakes
# the same sky for the raster half, so one atmosphere lights both.
#
# Placed so it REFLECTS INTO THE CAMERA, which is the part that is not a taste
# decision. The demo's eye sits at azimuth −36°; a sun at +58° put the specular
# lobe of every surface in the scene somewhere the camera was not, so gold had
# no highlight and glass no glint — the sun was in the sky and nothing could see
# it. Azimuth 140° is roughly opposite the eye, so the mirror direction off the
# lobes points at it and the highlight sweeps across them as the camera orbits.
const SUN_DIRECTION = let az = deg2rad(140.0), el = deg2rad(38.0)
    Vec3f(cos(el) * cos(az), cos(el) * sin(az), sin(el))
end
const SUN_INTENSITY = 1.0f0
const SUN_TURBIDITY = 2.2f0
# What `Hikari.sunsky_to_envlight` gives its own `SunLight`: a sun-to-sky
# illuminance ratio near 5:1, warmed slightly. Written out because that call
# returns the light as a spectrum and the frontend wants an RGB.
const SUNCOLOR = Colors.RGB{Float32}(5 * SUN_INTENSITY, 5 * SUN_INTENSITY * 0.95, 5 * SUN_INTENSITY * 0.85)

const ENV_W = 4096
const ENV_H = 2048

"""World units per ground tile — see the checker in [`demoenv`](@ref)."""
const GROUND_TILE = 0.34f0

# The SUN, as a disc IN the environment map.
#
# It was a `DirectionalLight` beside the sky, and a directional light is a delta
# light: it shades a surface and paints NOTHING along a reflected ray. So the
# sky a mirror saw had a dynamic range of 3.3:1 and a maximum of 0.29 — a flat
# wash — and gold came back dull, a clear coat had no highlight to be glossy
# with, and glass reflected nothing at all. A sun a surface can SEE has to be in
# the map.
#
# `SUN_ANGLE` is 3°, not the sun's real 0.27°. Bigger is softer: the highlight
# it makes has a readable size on a curved surface, and a few thousand texels of
# source instead of a handful is far less variance for the same energy.
# `SUN_RADIANCE` follows from wanting the same irradiance the directional light
# delivered — `E = L·Ω`, and `Ω = 2π(1 − cos 3°)` is 8.6e-3 sr.
# 2.5°, and the radiance follows: `Ω ≈ πθ²` for a small disc, so the angle and
# the radiance trade as `1/θ²` at fixed irradiance. Nine times the sun's real
# 0.27°, and that is a NOISE decision: a source this bright concentrated into a
# few hundred texels is a firefly generator at the sample counts a recording can
# afford. The glint's width comes from the BRDF lobe rather than from here — see
# `remap_roughness` in `isubd_gui.jl` — so softening the disc costs nothing
# visible and buys a clean frame.
const SUN_ANGLE = 2.5f0
const SUN_RADIANCE = 830.0f0

"""
    demoenv() -> equal-area octahedral Matrix{RGBSpectrum}

The world both renderers stand in: a Hosek-Wilkie sky with a GRADED ground
under it, and a sun to go with it ([`SUNCOLOR`](@ref)).

The ground is the part that took three tries. Hosek's own ground is a flat slab
of one colour; turning it off is worse, because the model then clamps to the
horizon and fills the entire lower hemisphere with a single dead grey — half
the frame, one value, and the subject looks pasted onto it. Grading it from the
horizon down to something darker is what makes the lower half read as ground
going away from you rather than as a wall.

It is also the reason the glass looks like glass. A refractive surface is only
visible as one when there is something behind it to BEND, and against a smooth
gradient every deflected ray lands on a neighbouring shade of the same colour —
which is a dark smudge, not a lens. The checker gives it edges to move and a
hue to shift.
"""
function demoenv()
    envlight, _ = Hikari.sunsky_to_envlight(direction = SUN_DIRECTION,
                                                   intensity = SUN_INTENSITY,
                                                   turbidity = SUN_TURBIDITY,
                                                   ground_enabled = false,
                                                   resolution = 1024)
    sq = copy(envlight.env_map.data)
    n = size(sq, 1)
    # The horizon's own colour, which is what the ground fades out of.
    acc = Hikari.RGBSpectrum(0f0); cnt = 0
    for j in 1:n, i in 1:n
        d = Hikari.equal_area_square_to_sphere(Point2f((i - 0.5f0)/n, (j - 0.5f0)/n))
        if abs(d[3]) < 0.02f0
            acc = Hikari.RGBSpectrum((acc.c .+ sq[j, i].c)...); cnt += 1
        end
    end
    hz = Hikari.RGBSpectrum((acc.c ./ max(cnt, 1))...)
    for j in 1:n, i in 1:n
        d = Hikari.equal_area_square_to_sphere(Point2f((i - 0.5f0)/n, (j - 0.5f0)/n))
        d[3] >= 0f0 && continue
        # `-d[3]` is 0 at the horizon and 1 straight down. Square it so most of
        # the fall happens well below the horizon and the line itself stays soft.
        t = (-d[3])^2
        f = 1f0 - 0.72f0 * t
        # A GRID on the ground, and it is not decoration. Glass is only visible
        # as glass when there is something behind it to bend: against a smooth
        # gradient a refractive surface looks like a dark smudge, because every
        # ray it deflects lands on a neighbouring shade of the same colour.
        # Straight lines to break make the refraction legible, and the same grid
        # shows up in the metal's reflection.
        #
        # Projected onto the ground plane rather than drawn in direction space,
        # so the lines converge at the horizon like a floor and not like a
        # painted dome.
        gx = d[1] / max(-d[3], 1f-3)
        gy = d[2] / max(-d[3], 1f-3)
        # Faded by DISTANCE ALONG THE GROUND, not by how far below the horizon
        # the direction points. A thin line is a high frequency in a 1024² map:
        # near the horizon a whole cell falls inside one texel and it turns into
        # speckle that looks exactly like sampling noise and never averages
        # away, because it is not noise.
        r = sqrt(gx * gx + gy * gy)
        fade = clamp(1f0 - r / 9f0, 0f0, 1f0)
        # TILES, and the contrast between them is the whole point.
        #
        # A grid of thin lines was here first and it was too weak to do its job:
        # at 26% over a grey ground, a refracted ray landed on a neighbouring
        # shade of the same colour and the glass came back a dark smudge. A
        # checker is the same information at a frequency the map can hold — one
        # tile is hundreds of texels, so it survives where a hairline speckles —
        # and at 2:1 contrast a bent ray lands somewhere visibly different. The
        # metal reflects it and the raster's Fresnel term picks it up too.
        # …and BAND-LIMITED, which a checker of `isodd(floor(gx) + floor(gy))` is
        # not. `sin·sin` is the same checker as a smooth function; dividing it
        # by a width that grows with distance and clamping is what filters it.
        # Without that, a tile near the horizon is smaller than a texel of a
        # 1024² map and its edge comes back as stair-stepped speckle — the same
        # failure the thin-line grid had, for the same reason. Washing out to
        # the mean at that size is what a filtered texture would do.
        sx = sinpi(gx / GROUND_TILE); sy = sinpi(gy / GROUND_TILE)
        w = clamp(0.05f0 * (1f0 + 0.35f0 * r), 0.05f0, 1f0)
        tile = 0.55f0 + 0.45f0 * (0.5f0 * (1f0 + clamp(sx * sy / w, -1f0, 1f0)))
        g = 1f0 - fade * (1f0 - tile)
        # …and slightly WARM against a blue sky, so refraction shifts hue as
        # well as brightness. Slightly: at (1.10, 1.00, 0.80) this went olive,
        # because the horizon it multiplies is already yellow under a Hosek sun.
        sq[j, i] = Hikari.RGBSpectrum(hz.c[1]*f*g*0.96f0, hz.c[2]*f*g*0.94f0, hz.c[3]*f*g*0.88f0)
    end
    # …and the sun, added rather than assigned: the disc sits ON the sky it
    # brightens, so the limb blends into it instead of ending on an edge.
    sd = normalize(SUN_DIRECTION)
    cosr = cos(deg2rad(SUN_ANGLE))
    for j in 1:n, i in 1:n
        d = Hikari.equal_area_square_to_sphere(Point2f((i - 0.5f0)/n, (j - 0.5f0)/n))
        c = d[1]*sd[1] + d[2]*sd[2] + d[3]*sd[3]
        c <= cosr && continue
        # Smoothstep across the outer third of the disc — a hard rim would
        # alias into the reflection as a ring of stair steps.
        t = clamp((c - cosr) / ((1f0 - cosr) * 0.34f0), 0f0, 1f0)
        w = SUN_RADIANCE * (t * t * (3f0 - 2f0 * t))
        sq[j, i] = Hikari.RGBSpectrum(sq[j, i].c[1] + w,
                                      sq[j, i].c[2] + w * 0.95f0,
                                      sq[j, i].c[3] + w * 0.86f0)
    end
    return sq
end

"""
    demosky() -> Matrix{RGBf}, equirectangular

[`demoenv`](@ref)'s world in the convention the raster path samples.
"""
function demosky()
    sq = demoenv()
    eq = Hikari.equalarea_to_equirect(sq)
    # `Colors.RGB{Float32}`, not Makie's `RGBf` alias: this example is loaded by
    # `test_isubd_mesh.jl`, which has no Makie, and the name resolved only
    # because the demo happened to have one loaded beside it.
    return map(c -> Colors.RGB{Float32}(c.c[1], c.c[2], c.c[3]), eq)
end

"""
    skyenv(path; w, h, gain) -> Vector{Float32}, flat RGB

Load and resample an equirectangular HDRI for the device.

Non-finite pixels are dropped to zero. An `Inf` in an environment map is an
infinitely bright direction, and anything that samples it comes back as a white
speck no averaging removes — the raven scene's notes make the same point about
the same class of file.
"""
function skyenv(; w = ENV_W, h = ENV_H, gain = 1.0f0)
    img = demosky()
    H, W = size(img)
    out = Array{Float32}(undef, 3 * w * h)
    for j in 1:h, i in 1:w
        si = clamp(round(Int, (i - 0.5) / w * W + 0.5), 1, W)
        sj = clamp(round(Int, (j - 0.5) / h * H + 0.5), 1, H)
        c = img[sj, si]
        f(x) = (v = Float32(x); isfinite(v) ? gain * max(v, 0f0) : 0f0)
        k = 3 * ((j - 1) * w + (i - 1))
        out[k+1] = f(Colors.red(c)); out[k+2] = f(Colors.green(c)); out[k+3] = f(Colors.blue(c))
    end
    return out
end

"""
The direction TO the light.

Not measured out of an image any more, because the sky is no longer an image:
`SUN_DIRECTION` is a parameter of the Hosek-Wilkie model that bakes it, so both
renderers are lit from the direction that actually made their sky. Recovering it
from pixels was the right answer when the background was a photograph — and a
photograph with a clipped sun, which made the recovery a guess.
"""
const LIGHTDIR = normalize(SUN_DIRECTION)

"""
    sampleenv(env, dx, dy, dz) -> (r, g, b)

The environment in direction `(dx, dy, dz)`, bilinearly, z up.

Bilinear rather than nearest because this is read for REFLECTIONS as well as for
the background, and a nearest reflection of a 1024-pixel sky across a curved
surface is a field of visible blocks that moves as it turns.
"""
@inline function sampleenv(env, dx::Float32, dy::Float32, dz::Float32)
    l = sqrt(dx*dx + dy*dy + dz*dz) + 1f-8
    x = dx / l; y = dy / l; z = dz / l
    u = 0.5f0 + atan(y, x) * 0.15915494f0        # 1/(2pi)
    v = acos(clamp(z, -1f0, 1f0)) * 0.31830987f0 # 1/pi, 0 at the zenith
    fx = clamp(u, 0f0, 1f0) * (ENV_W - 1) + 1f0
    fy = clamp(v, 0f0, 1f0) * (ENV_H - 1) + 1f0
    i0 = clamp(unsafe_trunc(Int32, fx), Int32(1), Int32(ENV_W - 1))
    j0 = clamp(unsafe_trunc(Int32, fy), Int32(1), Int32(ENV_H - 1))
    tx = fx - Float32(i0); ty = fy - Float32(j0)
    @inline function at(i, j)
        k = 3 * ((Int(j) - 1) * ENV_W + (Int(i) - 1))
        return (env[k+1], env[k+2], env[k+3])
    end
    a = at(i0, j0); b = at(i0 + Int32(1), j0)
    c = at(i0, j0 + Int32(1)); d = at(i0 + Int32(1), j0 + Int32(1))
    w00 = (1f0-tx)*(1f0-ty); w10 = tx*(1f0-ty); w01 = (1f0-tx)*ty; w11 = tx*ty
    return (a[1]*w00 + b[1]*w10 + c[1]*w01 + d[1]*w11,
            a[2]*w00 + b[2]*w10 + c[2]*w01 + d[2]*w11,
            a[3]*w00 + b[3]*w10 + c[3]*w01 + d[3]*w11)
end

"""Filmic-ish tone map, so an HDR value reaches the screen as a colour."""
@inline tonemap(x::Float32) = clamp((x / (1f0 + x))^(1f0/2.2f0), 0f0, 1f0)

"""
    orthocamera(bounds) -> Mat4f

The reference rasteriser's frame, as a matrix.

`reference.jl`'s `render` maps a fixed rectangle of the domain onto the image
and is not a perspective view of anything, so comparing this path against it
needs the same mapping — `px`/`py` from that function, written as a matrix, with
Vulkan's y DOWN.
"""
function orthocamera(bounds = BOUNDS)
    x0, x1, y0, y1 = bounds
    sx = 2 / (x1 - x0); sy = 2 / (y1 - y0)
    return Mat4f(sx, 0, 0, 0,
                 0, -sy, 0, 0,
                 0, 0, 0.5f0, 0,
                 -(x0 + x1) / (x1 - x0), (y0 + y1) / (y1 - y0), 0.5f0, 1)
end

"""
    camera(; azimuth, elevation, distance, target, fov, aspect) -> (mvp, eye)

One camera for both renderers. The raster path gets the matrix, the ray path
gets the eye and builds its rays from the same numbers, so an orbit moves both
identically and the A/B stays honest.
"""
function camera(; azimuth = 0.9, elevation = 0.55, distance = 3.4,
                  target = (0.0, 0.45, 0.20), fov = 0.8, aspect = 1.0,
                  near = 0.05, far = 40.0,
                  eye = (target[1] + distance * cos(elevation) * sin(azimuth),
                         target[2] - distance * cos(elevation) * cos(azimuth),
                         target[3] + distance * sin(elevation)))
    f = target .- eye; f = f ./ sqrt(sum(f .^ 2))
    up = (0.0, 0.0, 1.0)
    rgt = (f[2]*up[3]-f[3]*up[2], f[3]*up[1]-f[1]*up[3], f[1]*up[2]-f[2]*up[1])
    rgt = rgt ./ sqrt(sum(rgt .^ 2))
    u = (rgt[2]*f[3]-rgt[3]*f[2], rgt[3]*f[1]-rgt[1]*f[3], rgt[1]*f[2]-rgt[2]*f[1])
    view = Mat4f(rgt[1], u[1], -f[1], 0,
                 rgt[2], u[2], -f[2], 0,
                 rgt[3], u[3], -f[3], 0,
                 -(rgt[1]*eye[1]+rgt[2]*eye[2]+rgt[3]*eye[3]),
                 -(u[1]*eye[1]+u[2]*eye[2]+u[3]*eye[3]),
                  (f[1]*eye[1]+f[2]*eye[2]+f[3]*eye[3]), 1)
    tf = 1 / tan(fov / 2)
    # Vulkan clip space: y DOWN and z in [0, 1], which is why this is written
    # out rather than borrowed from an OpenGL-convention helper.
    proj = Mat4f(tf/aspect, 0, 0, 0,
                 0, -tf, 0, 0,
                 0, 0, far/(near-far), -1,
                 0, 0, near*far/(near-far), 0)
    # The basis the RAY side builds directions from, and the background
    # triangle in the raster side interpolates: columns are right, up, forward,
    # pre-scaled so a clip-space corner (+-1) maps straight to a ray.
    tanh = tan(fov / 2)
    basis = Mat4f(rgt[1]*tanh*aspect, rgt[2]*tanh*aspect, rgt[3]*tanh*aspect, 0,
                  u[1]*tanh,          u[2]*tanh,          u[3]*tanh,          0,
                  f[1],               f[2],               f[3],               0,
                  0, 0, 0, 1)
    return proj * view, eye, basis
end

# The reference's `render` bounds, which is what makes the two images
# comparable pixel for pixel.
const BOUNDS = (-1.05, 1.05, -0.35, 1.25)

"""
World position to clip space, through a full camera.

`BOUNDS` and the orthographic box map it used to do are gone from the raster
path: the two renderers now show the same WARPED 3D surface through the same
matrix, which is the only way the comparison between them is about tessellation
rather than about two different pictures.
"""
@inline function toclip(x, y, z, mvp::Mat4f)
    p = mvp * Vec4f(Float32(x), Float32(y), Float32(z), 1.0f0)
    return p
end

function isubd_mesh(out, keys, basecorners, basecell, cx, cy, cf, nkeys::Int32,
                    mvp::Mat4f, flags::Vec4f, viewdir::Vec3f, lightdir::Vec3f,
                    env, cambasis::Mat4f)
    g = KI.mesh_group_index()
    t = KI.mesh_thread_index()
    warp = Float64(flags[1])

    # Workgroup 1 is the BACKGROUND, the rest are geometry.
    #
    # In the same draw and the same pipeline, because a mesh stage decides what
    # geometry exists — so "one fullscreen triangle" is just another thing it
    # can emit. The alternative is a second pipeline and a second draw for three
    # vertices. It needs the depth buffer to sort it behind the surface, since
    # nothing orders one workgroup against another.
    # `flags[3]` turns the background off, which is what the reference
    # comparison in `test_isubd_mesh.jl` needs: it checks this path against a
    # CPU rasteriser that draws the field on white and knows nothing about a sky.
    if g == Int32(1)
        if flags[3] < 0.5f0
            KI.set_mesh_outputs!(out, Int32(0), Int32(0))
            return nothing
        end
        KI.set_mesh_outputs!(out, Int32(3), Int32(1))
        if t == Int32(1)
            # A triangle that covers the viewport, at the far plane. Its corner
            # ray directions are carried as a varying and interpolated, which is
            # exactly how a skybox is drawn.
            @inbounds for c in 1:3
                sx = c == 1 ? -1f0 : (c == 2 ? 3f0 : -1f0)
                sy = c == 1 ? -1f0 : (c == 2 ? -1f0 : 3f0)
                # MINUS sy against the up axis: Vulkan clip space has y DOWN,
                # so `sy = -1` is the TOP of the screen and must look UP. Taking
                # it as written flips the environment vertically — and only in
                # this path, so the sky ended up under the horizon here while
                # the traced view had it above.
                dx = cambasis[1,3] + sx*cambasis[1,1] - sy*cambasis[1,2]
                dy = cambasis[2,3] + sx*cambasis[2,1] - sy*cambasis[2,2]
                dz = cambasis[3,3] + sx*cambasis[3,1] - sy*cambasis[3,2]
                KI.set_mesh_vertex!(out, Int32(c),
                    (position = Vec4f(sx, sy, 0.999999f0, 1f0), xi = Vec2f(0, 0),
                     cell = 0f0, bary = Vec2f(1, 1), nrm = Vec3f(dx, dy, dz)))
            end
            KI.set_mesh_triangle!(out, Int32(1), Int32(1), Int32(2), Int32(3))
        end
        return nothing
    end

    gk = g - Int32(1)
    i = (gk - Int32(1)) * Int32(MESH_THREADS) + t
    nlocal = min(Int32(MESH_THREADS), nkeys - (gk - Int32(1)) * Int32(MESH_THREADS))
    KI.set_mesh_outputs!(out, Int32(3) * nlocal, nlocal)

    if t <= nlocal
        @inbounds k = keys[i]
        cell = Int(basecell[key_base(k)])
        c1, c2, c3 = key_corners(k, basecorners)
        v0 = (t - Int32(1)) * Int32(3)
        cellf = Float32(cell)

        # Position AND normal per corner. The normal comes from the surface
        # tangents — the same `surfacetangents` the ray path differentiates for
        # Newton — so the two renderers shade from the same geometry and the
        # only difference left is how finely this one samples it.
        p1 = surfacepoint(cx, cy, cf, cell, c1[1], c1[2], warp)
        p2 = surfacepoint(cx, cy, cf, cell, c2[1], c2[2], warp)
        p3 = surfacepoint(cx, cy, cf, cell, c3[1], c3[2], warp)
        n1 = surfacenormal(cx, cy, cf, cell, c1[1], c1[2], warp)
        n2 = surfacenormal(cx, cy, cf, cell, c2[1], c2[2], warp)
        n3 = surfacenormal(cx, cy, cf, cell, c3[1], c3[2], warp)

        KI.set_mesh_vertex!(out, v0 + Int32(1),
            (position = toclip(p1[1], p1[2], p1[3], mvp), xi = Vec2f(c1[1], c1[2]),
             cell = cellf, bary = Vec2f(1, 0), nrm = n1))
        KI.set_mesh_vertex!(out, v0 + Int32(2),
            (position = toclip(p2[1], p2[2], p2[3], mvp), xi = Vec2f(c2[1], c2[2]),
             cell = cellf, bary = Vec2f(0, 1), nrm = n2))
        KI.set_mesh_vertex!(out, v0 + Int32(3),
            (position = toclip(p3[1], p3[2], p3[3], mvp), xi = Vec2f(c3[1], c3[2]),
             cell = cellf, bary = Vec2f(0, 0), nrm = n3))
        KI.set_mesh_triangle!(out, t, v0 + Int32(1), v0 + Int32(2), v0 + Int32(3))
    end
    return nothing
end

# ── PASS C — the fragment shader ────────────────────────────────────────────
#
# The field, evaluated per pixel from the cell's coefficients and the
# interpolated reference coordinate. This is the whole argument for the split:
# the triangles resolve the GEOMETRY, and the colour is exact regardless of how
# few of them there are.
#
# `cell` arrives through a FLAT input — it is constant across its triangle, and
# flat takes the provoking vertex's value rather than interpolating, so a
# `Float32` carrying a small integer arrives unchanged.

"""
    shade(env, value, normal, viewdir, lightdir) -> (r, g, b)

The material, and the ONLY one: both renderers call this, so a difference in the
picture is a difference in the geometry and never in the look.

The field's colour is the diffuse albedo, lit by a sun plus the sky sampled
along the normal. On top of that sits an environment REFLECTION with a Fresnel
weight — which is what makes it read as a surface rather than as a coloured
shape, and what makes the orbit worth watching, because the reflected sky moves
across it while the diffuse term does not.

Tone mapped here rather than by the caller, because both callers would have to
do it identically and one of them would eventually not.
"""
@inline shade(env, v::Float32, n::Vec3f, viewdir::Vec3f, lightdir::Vec3f) =
    shade(env, colormap(v)..., n, viewdir, lightdir)

# The same material with the albedo already chosen. Two methods and not a
# keyword, because a caller either has a field VALUE — this path's colour ramp
# is its own — or a COLOUR, which is what a caller carrying the tracer's ramp
# has, and the ramp is the only difference between them.
@inline shade(env, r::Float32, g::Float32, b::Float32,
              n::Vec3f, viewdir::Vec3f, lightdir::Vec3f) =
    shade(env, r, g, b, n, viewdir, lightdir, EXPOSURE)

# …and with the exposure given. The standalone renderer owns the whole frame and
# `EXPOSURE` is what its picture is graded to; an OVERLAY lands on a film the
# path tracer graded, and the two have to agree or the switch between them reads
# as the brightness changing rather than the geometry.
@inline function shade(env, r::Float32, g::Float32, b::Float32,
                       n::Vec3f, viewdir::Vec3f, lightdir::Vec3f, exposure::Float32)
    ndl = max(n[1]*lightdir[1] + n[2]*lightdir[2] + n[3]*lightdir[3], 0f0)
    # `viewdir` points FROM the eye, so this is negated to face the surface.
    ndv = clamp(-(n[1]*viewdir[1] + n[2]*viewdir[2] + n[3]*viewdir[3]), 0f0, 1f0)

    # Sky along the normal: a cheap irradiance, and the reason shadowed parts
    # are blue rather than black.
    sr, sg, sb = sampleenv(env, n[1], n[2], n[3])

    # Mirror direction, for the reflection.
    d = viewdir[1]*n[1] + viewdir[2]*n[2] + viewdir[3]*n[3]
    rx = viewdir[1] - 2f0*d*n[1]; ry = viewdir[2] - 2f0*d*n[2]; rz = viewdir[3] - 2f0*d*n[3]
    er, eg, eb = sampleenv(env, rx, ry, rz)
    # Schlick, with a low base: a dielectric that goes mirror-like at grazing
    # angles, which is where the silhouette is and so where the two renderers
    # differ most. Kept WEAKER than a real dielectric's 1.0 at grazing, because
    # this surface is curved enough that most of it is near-grazing — at full
    # strength the reflection covers the whole object in one flat sheen and the
    # albedo underneath stops being visible at all.
    fres = 0.03f0 + 0.42f0 * (1f0 - ndv)^5

    # Blinn-Phong for the sun, and the half vector is built from `-viewdir`
    # because `viewdir` points away from the eye.
    hx = lightdir[1] - viewdir[1]
    hy = lightdir[2] - viewdir[2]
    hz = lightdir[3] - viewdir[3]
    hl = sqrt(hx*hx + hy*hy + hz*hz) + 1f-8
    ndh = max((n[1]*hx + n[2]*hy + n[3]*hz) / hl, 0f0)
    ndh2 = ndh * ndh; ndh4 = ndh2 * ndh2; ndh8 = ndh4 * ndh4
    ndh16 = ndh8 * ndh8; ndh32 = ndh16 * ndh16
    # `ndh^64`, by squaring. Written out because a runtime `^` on a device is a
    # library call and the exponent here is a constant either way.
    spec = SUN_SPECULAR * ndh32 * ndh32 * (ndl > 0f0 ? 1f0 : 0f0)

    # Exposure, and it matters more than it looks. At the previous levels the
    # lit surface landed at x = 2.3 in `x/(1+x)`, where the curve has almost no
    # slope left: the bright channels compressed together while the dark one was
    # lifted by the gamma, so a saturated yellow came back as pale chalk. Keeping
    # the surface near the middle of the curve is what gives it colour and form.
    sun = SUN_DIFFUSE * ndl
    lr = exposure * (r * (sun * SUN_TINT[1] + SKY_AMBIENT * sr) + fres * er + spec * SUN_TINT[1])
    lg = exposure * (g * (sun * SUN_TINT[2] + SKY_AMBIENT * sg) + fres * eg + spec * SUN_TINT[2])
    lb = exposure * (b * (sun * SUN_TINT[3] + SKY_AMBIENT * sb) + fres * eb + spec * SUN_TINT[3])
    return (tonemap(lr), tonemap(lg), tonemap(lb))
end

@inline function colormap(t)
    s = clamp(t, 0.0f0, 1.0f0)
    return (0.15f0 + 0.75f0 * s, 0.15f0 + 0.65f0 * s, 0.55f0 - 0.45f0 * s)
end

function isubd_frag(inputs, keys, basecorners, basecell, cx, cy, cf, nkeys::Int32,
                    mvp::Mat4f, flags::Vec4f, viewdir::Vec3f, lightdir::Vec3f,
                    env, cambasis::Mat4f)
    a = inputs.xi[1]
    b = inputs.xi[2]
    cell = Int(round(inputs.cell))
    nx = inputs.nrm[1]
    ny = inputs.nrm[2]
    nz = inputs.nrm[3]
    nl = sqrt(nx*nx + ny*ny + nz*nz) + 1.0f-8
    n = Vec3f(nx/nl, ny/nl, nz/nl)

    # `cell == 0` is the background triangle, whose `nrm` varying carries the
    # view ray rather than a normal.
    if cell == 0
        er, eg, eb = sampleenv(env, nx/nl, ny/nl, nz/nl)
        return Vec4f(tonemap(er*BG_DIM), tonemap(eg*BG_DIM), tonemap(eb*BG_DIM), 1.0f0)
    end

    v = Float32(eval_poly(cf, cell, Float64(a), Float64(b)))
    # `flags[4]` selects the RAW colormap over the material. The reference
    # rasteriser in `test_isubd_mesh.jl` knows nothing about lights or an
    # environment — it writes `colormap(value)` and nothing else — so comparing
    # against it means asking this stage for the same thing. The test is about
    # whether the mesh and the field are right; the material is the demo's.
    r, g, bl = flags[4] > 0.5f0 ? shade(env, v, n, viewdir, lightdir) : colormap(v)

    # Edge stroking, in the stage rather than as an overlay — the approach of
    # Makie PR #5727. Two things that port straight across:
    #
    #  * HALF the width, drawn INWARD from each face. A shared edge gets half
    #    from each of its two triangles and the halves meet as one full-width
    #    line. Drawing the full width from both sides, which is what the naive
    #    barycentric version does, gives every interior edge twice the width and
    #    twice the darkening.
    #  * Screen-space distance, so the line is one width at every zoom: the
    #    barycentric distance divided by its own derivative is a distance in
    #    pixels.
    #
    # Not ported: that PR's wing edges and `:boundary`/`:all` selection, which
    # exist to hide the diagonals triangulation introduces into quad meshes.
    # Every edge here IS a subdivision edge, so there is nothing to hide.
    if flags[2] > 0.0f0
        u1 = inputs.bary[1]
        u2 = inputs.bary[2]
        u3 = 1.0f0 - u1 - u2

        # Each edge's distance separately, THEN the minimum.
        #
        # Taking `min` of the barycentrics first and differentiating that is
        # wrong, and wrong in a visible way: the gradient of a `min` jumps where
        # two of them are equal, which is the line from each corner to the
        # opposite edge — so the stroke swelled into a blob at every vertex.
        # Dividing each barycentric by its OWN gradient gives a real distance in
        # pixels, and the nearest edge is then just the smallest of the three.
        d1 = u1 / max(sqrt(KernelInterface.dFdx(u1)^2 + KernelInterface.dFdy(u1)^2), 1f-8)
        d2 = u2 / max(sqrt(KernelInterface.dFdx(u2)^2 + KernelInterface.dFdy(u2)^2), 1f-8)
        d3 = u3 / max(sqrt(KernelInterface.dFdx(u3)^2 + KernelInterface.dFdy(u3)^2), 1f-8)
        d = min(d1, min(d2, d3))

        # Stroke OR face, with one pixel of coverage at the boundary — not a
        # gradient across the whole width. The previous version ramped the
        # brightness from 30% to 100% over the line's width, which has no flat
        # core and reads as a smudge rather than a line.
        cov = clamp(flags[2] - d + 0.5f0, 0.0f0, 1.0f0)
        r = r + (STROKE_COLOR[1] - r) * cov
        g = g + (STROKE_COLOR[2] - g) * cov
        bl = bl + (STROKE_COLOR[3] - bl) * cov
    end
    return Vec4f(r, g, bl, 1.0f0)
end

# ── Driving the two stages ──────────────────────────────────────────────────
#
# Both shaders are compiled against the SAME argument tuple, so the two stages
# agree on the push-constant layout — there is one push range and both stages
# read it. The fragment stage ignores the arguments it does not use; the
# alternative is two layouts over one block, which is a silent misread rather
# than an error.

const MESH_ARGTYPES = Tuple{Lava.LavaDeviceArray{UInt64,1}, Lava.LavaDeviceArray{Float64,1},
                            Lava.LavaDeviceArray{Int32,1}, Lava.LavaDeviceArray{Float64,2},
                            Lava.LavaDeviceArray{Float64,2}, Lava.LavaDeviceArray{Float64,2},
                            Int32, Mat4f, Vec4f, Vec3f, Vec3f,
                            Lava.LavaDeviceArray{Float32,1}, Mat4f}

meshconfig() = KI.MeshConfig(; max_vertices = 3 * MESH_THREADS,
                               max_primitives = MESH_THREADS,
                               topology = KI.TriangleList(), threads = MESH_THREADS)

"""
    IsubdRenderer(; width, height) -> renderer

The compiled shaders, the pipeline and the target, built ONCE.

Separate from [`draw!`](@ref) because none of it depends on the key set: a
refinement that changes the triangle count changes only the WORKGROUP COUNT of
the draw. Folding this into the per-frame call cost 24.7 ms a frame against
0.20 ms of actual drawing — it was shader compilation and pipeline creation,
measured, not the mesh stage.
"""
struct IsubdRenderer{P, F}
    pipeline::P
    framebuffer::F
    width::Int
    height::Int
end

function IsubdRenderer(; width::Integer = 700, height::Integer = 700)
    # WITH depth: the background workgroup emits a fullscreen triangle at the
    # far plane and nothing orders one mesh workgroup against another, so the
    # depth test is what puts it behind the surface.
    fb = Mantle.Framebuffer(Mantle.defaultbackend(), width, height;
                            depth = true, color_format = RGBA{Float32})
    pip = Mantle.MeshPipeline(
        mesh = Mantle.MeshShader(isubd_mesh;
                 outputs = (xi = Vec2f, cell = Mantle.Flat{Float32},
                            bary = Vec2f, nrm = Vec3f),
                 max_vertices = 3 * MESH_THREADS,
                 max_primitives = MESH_THREADS,
                 topology = Mantle.TriangleList(), threads = MESH_THREADS),
        fragment = Mantle.FragmentShader(isubd_frag),
        cull = Mantle.NoCull(), depth = Mantle.DepthLess())
    return IsubdRenderer(pip, fb, Int(width), Int(height))
end

"""
    draw!(r::IsubdRenderer, keys, cx, cy, cf, basecorners, basecell) -> pixels

One frame. The draw covers `cld(length(keys), MESH_THREADS)` workgroups, so a
refinement that changed the triangle count needs no reallocation, no new
pipeline and no upload — the key buffer IS the geometry. That is what the key
representation buys.
"""
function draw!(r::IsubdRenderer, keys, cx, cy, cf, basecorners, basecell;
               clear = (0.09f0, 0.09f0, 0.13f0, 1.0f0), mvp = camera()[1],
               viewdir = VIEWDIR, lightdir = LIGHTDIR, env, cambasis,
               warp = WARP, wireframe = false, background = true, shaded = true)
    nkeys = Int32(length(keys))
    args = (keys, basecorners, basecell, cx, cy, cf, nkeys, mvp,
            Vec4f(warp, strokehalf(wireframe), background ? 1 : 0, shaded ? 1 : 0),
            viewdir, lightdir,
            env, cambasis)
    # +1 for the background workgroup.
    groups = cld(Int(nkeys), MESH_THREADS) + 1
    fb = r.framebuffer
    Mantle.draw!(Mantle.Device(), r.pipeline, Mantle.OffscreenTarget(fb), groups;
                 args, frag_args = args, clear_color = clear)
    Mantle.flush!(Mantle.Device())
    return Mantle.readback_framebuffer(fb)
end

"""One-shot convenience: build a renderer, draw once, throw it away."""
render_isubd(keys, cx, cy, cf, basecorners, basecell; width = 700, height = 700, kw...) =
    draw!(IsubdRenderer(; width, height), keys, cx, cy, cf, basecorners, basecell; kw...)

"""Binary PPM, so comparing against the reference's output needs no image dependency."""
function write_ppm(path, px, w, h)
    open(path, "w") do io
        write(io, "P6\n$w $h\n255\n")
        for y in 1:h, x in 1:w
            p = px[x, y]
            write(io, UInt8(round(clamp(p.r, 0, 1) * 255)),
                      UInt8(round(clamp(p.g, 0, 1) * 255)),
                      UInt8(round(clamp(p.b, 0, 1) * 255)))
        end
    end
end

# ── PASS D — the same elements, RAY TRACED, with no triangles at all ────────
#
# The raster path subdivides until the flat triangles approximate the curve
# closely enough. A ray tracer does not have to: given a ray, solve for the
# point on the element it actually hits.
#
#     F(ξ₁, ξ₂, t) = S(ξ₁, ξ₂) − o − t·d = 0
#
# Three equations, three unknowns, Newton. `S` is the element's own surface —
# the SAME `eval_poly` the mesh shader and the fragment shader call, so there is
# no second geometry definition to keep in step. The acceleration structure
# holds ONE AABB PER ELEMENT: two boxes here, against 1106 triangles for the
# same picture, and the silhouette is exact at any zoom.
#
# The surface is the domain warped out of plane by the solution, which is what
# makes it a surface at all — `WARP = 0` gives the flat domain and still works.

const WARP = 0.6
function element_accel(backend, cx, cy, cf; warp = WARP)
    bq = Mantle.batchqueue(Mantle.Device())
    aabbs = [element_aabb(cx, cy, cf, c; warp) for c in 1:size(cf, 2)]
    blas = Mantle.build_accel!(bq) do c
        Mantle.build_blas_aabb(c, aabbs)
    end
    tlas = Mantle.HWTLAS{Raycore.Triangle{UInt32}}(backend)
    push!(tlas, blas, Mat4f(I); instance_id = UInt32(0))
    Raycore.sync!(tlas)
    return blas, tlas
end

"""Write an RGB triple of flat arrays as a binary PPM."""
function write_ppm_rgb(path, R, G, B, w, h)
    open(path, "w") do io
        write(io, "P6\n$w $h\n255\n")
        for k in 1:(w * h)
            write(io, UInt8(round(clamp(R[k], 0, 1) * 255)),
                      UInt8(round(clamp(G[k], 0, 1) * 255)),
                      UInt8(round(clamp(B[k], 0, 1) * 255)))
        end
    end
end
# ── The interactive trace ───────────────────────────────────────────────────
#
# An earlier version built the acceleration structure, uploaded the
# coefficients and uploaded six per-pixel ray arrays on every frame — all of it
# setup, none of it dependent on the frame, and 1061 ms each time. This one is
# handed the camera BASIS and computes each ray from its own pixel index, so an
# orbit costs a launch and nothing else.

"""
The traced frame: perspective rays, the shared shading, and SHADOW rays.

A shadow is the part the raster path cannot answer at all. It is a second ray
query from the hit point toward the light, solved against the same two AABBs by
the same Newton — so the surface shadows itself, exactly, with no shadow map, no
bias and no resolution.

Rays are computed from the camera basis per pixel rather than uploaded, so an
orbit costs a launch and nothing else.
"""
function isubd_trace_kernel(outr::LavaDeviceArray{Float32,1},
                            outg::LavaDeviceArray{Float32,1},
                            outb::LavaDeviceArray{Float32,1},
                            cx::LavaDeviceArray{Float64,2},
                            cy::LavaDeviceArray{Float64,2},
                            cf::LavaDeviceArray{Float64,2},
                            env::LavaDeviceArray{Float32,1},
                            eye::Vec3f, cright::Vec3f, cup::Vec3f, cfwd::Vec3f,
                            dims::Vec3f, warp::Float64, lightdir::Vec3f)
    i = Int(Lava.lava_global_invocation_id_x()) + 1
    w = Int(dims[1]); h = Int(dims[2]); tanf = dims[3]
    @inbounds begin
        px = (i - 1) % w; py = (i - 1) ÷ w
        sx = ((px + 0.5f0) / w * 2f0 - 1f0) * tanf
        sy = (1f0 - (py + 0.5f0) / h * 2f0) * tanf
        dx = cfwd[1] + sx*cright[1] + sy*cup[1]
        dy = cfwd[2] + sx*cright[2] + sy*cup[2]
        dz = cfwd[3] + sx*cright[3] + sy*cup[3]
        dl = sqrt(dx*dx + dy*dy + dz*dz)
        dx /= dl; dy /= dl; dz /= dl

        hit, t, a, b, cell = traceelements(cx, cy, cf, eye[1], eye[2], eye[3],
                                           dx, dy, dz, warp, 1f-3)
        if hit
            hx = eye[1] + t*dx; hy = eye[2] + t*dy; hz = eye[3] + t*dz
            n = surfacenormal(cx, cy, cf, cell, a, b, warp)
            v = Float32(eval_poly(cf, cell, a, b))
            r, g, bl = shade(env, v, n, Vec3f(dx, dy, dz), lightdir)
            # The shadow ray, offset along the normal so the surface does not
            # hit itself at t = 0.
            ox = hx + 1f-3*n[1]; oy = hy + 1f-3*n[2]; oz = hz + 1f-3*n[3]
            sh, _, _, _, _ = traceelements(cx, cy, cf, ox, oy, oz,
                                            lightdir[1], lightdir[2], lightdir[3],
                                            warp, 1f-3)
            if sh
                r *= 0.42f0; g *= 0.42f0; bl *= 0.48f0
            end
            outr[i] = r; outg[i] = g; outb[i] = bl
        else
            # The same environment the raster path's background triangle shows.
            er, eg, eb = sampleenv(env, dx, dy, dz)
            outr[i] = tonemap(er*BG_DIM); outg[i] = tonemap(eg*BG_DIM); outb[i] = tonemap(eb*BG_DIM)
        end
    end
    return nothing
end

"""
Nearest element hit along a ray — the traversal both the camera and the shadow
rays go through.
"""
@inline function traceelements(cx, cy, cf, ox, oy, oz, dx, dy, dz, warp, tmin)
    ray = Raycore.Ray(o = Point3f(ox, oy, oz), d = Vec3f(dx, dy, dz),
                      t_min = tmin, t_max = 100f0)
    Lava.lava_ray_query_init(ray)
    bt = 1f30; ba = 0.0; bb = 0.0; bc = 0
    while Lava.lava_ray_query_proceed()
        if Lava.lava_ray_query_get_type(false) == UInt32(1)
            cell = Int(Lava.lava_ray_query_get_primitive_index(false)) + 1
            qx = Float64(Lava.lava_ray_query_get_object_ray_origin(false, 1))
            qy = Float64(Lava.lava_ray_query_get_object_ray_origin(false, 2))
            qz = Float64(Lava.lava_ray_query_get_object_ray_origin(false, 3))
            rx = Float64(Lava.lava_ray_query_get_object_ray_direction(false, 1))
            ry = Float64(Lava.lava_ray_query_get_object_ray_direction(false, 2))
            rz = Float64(Lava.lava_ray_query_get_object_ray_direction(false, 3))
            c1, c2, c3 = surfacepoint(cx, cy, cf, cell, 0.0, 0.0, warp)
            t0 = (c1-qx)*rx + (c2-qy)*ry + (c3-qz)*rz
            h, t, a, b = intersect_element(cx, cy, cf, cell, qx,qy,qz, rx,ry,rz, t0, warp)
            if h && Float32(t) < bt
                bt = Float32(t); ba = a; bb = b; bc = cell
                Lava.lava_ray_query_generate_intersection(Float32(t))
            end
        end
    end
    committed = Lava.lava_ray_query_get_type(true) != UInt32(0) && bc != 0
    return (committed, bt, ba, bb, bc)
end
