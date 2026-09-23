# The RASTER half of the demo, as a RayMakie overlay.
#
# `RayMakie.mesh_overlay_dispatch!` dispatches on the plot's MATERIAL, which is
# the seam this file plugs into: a `Hikari.FEMMaterial` carries curved elements,
# so on the raster side it selects a MESH PIPELINE that subdivides them, the same
# way it selects procedural geometry on the traced side. One plot, one material,
# two renderers — and the choice is made by the same type in both.
#
# It lives with the demo rather than in RayMakie because the subdivision is the
# demo's subject: the keys, the estimator and the base triangles are the thing
# the talk is about, and RayMakie's business is only to hand them a draw.
#
# What is NOT drawn here is the background. In raster mode the tracer still runs
# — every ray escapes, and the environment light paints them — so the sky under
# this surface is the same sky the traced view has. The `isubd_mesh` stage's
# background workgroup exists because that path owns its whole framebuffer; this
# one is an overlay and owns only the surface.

using Makie: to_value

const FEM_OVERLAY_THREADS = MESH_THREADS

# Graded to the PATH TRACER's picture, not to the standalone renderer's. The
# flip is the demo's one cut and it has to show the geometry changing; a
# brightness step across it reads as a bug.
#
# Measured, not chosen: at the demo's framing the traced frame's mean luminance
# over the viewport is 0.579, and this path gives 0.449 at the standalone
# renderer's 1.4 — `shade`'s sky term is a hemisphere sample where the tracer
# integrates the whole environment, so it lands lower. `tonemap` is `x/(1+x)`
# and compresses hard, which is why closing a 0.13 gap takes this much: 2.0 →
# 0.495, 3.5 → 0.521, 5.0 → 0.537.
const FEM_OVERLAY_EXPOSURE = 3.5f0

# ── PASS B, as a portable mesh stage ────────────────────────────────────────
#
# The same expansion `isubd_mesh` does, written against `MeshShader`'s portable
# signature — `(out, args…)` with the emitter handed in — so Mantle compiles it
# and the graph draws it beside every other overlay.
#
# `wpos` is the one varying the standalone path has no use for: that one shades
# from a constant view direction, and a perspective camera in a Makie scene needs
# the direction per pixel, which is the surface point and the eye.
function fem_overlay_mesh(out, keys, basecorners, basecell, cx, cy, cf, ramp,
                          nkeys::Int32, mvp::Mat4f, params::Vec4f, stroke::Float32,
                          eye::Vec3f, lightdir::Vec3f, env)
    g = KI.mesh_group_index()
    t = KI.mesh_thread_index()
    warp = Float64(params[1])

    i = (g - Int32(1)) * Int32(FEM_OVERLAY_THREADS) + t
    nlocal = min(Int32(FEM_OVERLAY_THREADS), nkeys - (g - Int32(1)) * Int32(FEM_OVERLAY_THREADS))
    KI.set_mesh_outputs!(out, Int32(3) * nlocal, nlocal)

    if t <= nlocal
        @inbounds k = keys[i]
        cell = Int(basecell[key_base(k)])
        c1, c2, c3 = key_corners(k, basecorners)
        v0 = (t - Int32(1)) * Int32(3)
        cellf = Float32(cell)

        p1 = surfacepoint(cx, cy, cf, cell, c1[1], c1[2], warp)
        p2 = surfacepoint(cx, cy, cf, cell, c2[1], c2[2], warp)
        p3 = surfacepoint(cx, cy, cf, cell, c3[1], c3[2], warp)
        n1 = surfacenormal(cx, cy, cf, cell, c1[1], c1[2], warp)
        n2 = surfacenormal(cx, cy, cf, cell, c2[1], c2[2], warp)
        n3 = surfacenormal(cx, cy, cf, cell, c3[1], c3[2], warp)

        # Written out three times rather than through a local closure: the
        # order of the named tuple IS the varying numbering, so it is the
        # interface the fragment stage links against, and a closure is one more
        # thing between it and the emitter.
        # `bary` is what the EDGES are drawn from: two barycentrics interpolated
        # across the face, the third implied. See the stroke in
        # `fem_overlay_frag` — it is the reason the raster half is legible at
        # all, because a tessellation whose edges are invisible only shows at
        # the silhouette and that is not what the tolerance slider moves.
        KI.set_mesh_vertex!(out, v0 + Int32(1),
            (position = toclip(p1[1], p1[2], p1[3], mvp), xi = Vec2f(c1[1], c1[2]),
             cell = cellf, bary = Vec2f(1, 0), nrm = n1,
             wpos = Vec3f(Float32(p1[1]), Float32(p1[2]), Float32(p1[3]))))
        KI.set_mesh_vertex!(out, v0 + Int32(2),
            (position = toclip(p2[1], p2[2], p2[3], mvp), xi = Vec2f(c2[1], c2[2]),
             cell = cellf, bary = Vec2f(0, 1), nrm = n2,
             wpos = Vec3f(Float32(p2[1]), Float32(p2[2]), Float32(p2[3]))))
        KI.set_mesh_vertex!(out, v0 + Int32(3),
            (position = toclip(p3[1], p3[2], p3[3], mvp), xi = Vec2f(c3[1], c3[2]),
             cell = cellf, bary = Vec2f(0, 0), nrm = n3,
             wpos = Vec3f(Float32(p3[1]), Float32(p3[2]), Float32(p3[3]))))
        KI.set_mesh_triangle!(out, t, v0 + Int32(1), v0 + Int32(2), v0 + Int32(3))
    end
    return nothing
end

# ── PASS C ──────────────────────────────────────────────────────────────────
#
# The field, per pixel, from the cell's coefficients and the interpolated
# reference coordinate — which is the whole argument for the split: the triangles
# resolve the GEOMETRY and the colour is exact however few of them there are.
"""
The tracer's colour ramp, read from a buffer.

`Hikari.fem_ramp` reads the same 32 entries out of a TUPLE, which it has to:
they travel inside the material, and indexing a tuple at a runtime position is
what the SPIR-V emitter refuses. Here they are a device array and the index is
just an index — the interpolation is the same, and so is the colour, which is
the point: the two renderers must not disagree about what a value looks like.
"""
@inline function ramplookup(ramp, s::Float32)
    f = clamp(s, 0f0, 1f0) * Float32(Hikari.FEM_RAMP_N - 1)
    i0 = clamp(unsafe_trunc(Int32, f), Int32(0), Int32(Hikari.FEM_RAMP_N - 2))
    w = f - Float32(i0)
    @inbounds a = ramp[i0 + Int32(1)]
    @inbounds b = ramp[i0 + Int32(2)]
    return ((1f0 - w) * a[1] + w * b[1],
            (1f0 - w) * a[2] + w * b[2],
            (1f0 - w) * a[3] + w * b[3])
end

function fem_overlay_frag(inputs, keys, basecorners, basecell, cx, cy, cf, ramp,
                          nkeys::Int32, mvp::Mat4f, params::Vec4f, stroke::Float32,
                          eye::Vec3f, lightdir::Vec3f, env)
    cell = Int(round(inputs.cell))
    n = inputs.nrm
    nl = sqrt(n[1]*n[1] + n[2]*n[2] + n[3]*n[3]) + 1f-8
    nn = Vec3f(n[1]/nl, n[2]/nl, n[3]/nl)

    # FROM the eye, which is the direction `shade` expects.
    d = inputs.wpos - eye
    dl = sqrt(d[1]*d[1] + d[2]*d[2] + d[3]*d[3]) + 1f-8
    viewdir = Vec3f(d[1]/dl, d[2]/dl, d[3]/dl)

    # TWO-SIDED. An element is a surface, not the boundary of a solid, so which
    # way `cross(∂ξ₁, ∂ξ₂)` happens to point is a property of how the mesh was
    # written down and not of what the camera can see. The tracer already knows
    # this — a BSDF frames itself against the incoming ray — and a rasteriser
    # that does not renders every back-facing element black. Flipping the normal
    # to face the viewer is what a double-sided material means.
    if nn[1]*viewdir[1] + nn[2]*viewdir[2] + nn[3]*viewdir[3] > 0f0
        nn = Vec3f(-nn[1], -nn[2], -nn[3])
    end

    val = Float32(eval_poly(cf, cell, inputs.xi[1], inputs.xi[2]))
    vmin = params[2]; vmax = params[3]
    t = clamp((val - vmin) / max(vmax - vmin, 1f-8), 0f0, 1f0)

    cr, cg, cb = ramplookup(ramp, t)
    r, g, b = shade(env, cr, cg, cb, nn, viewdir, lightdir, params[4])

    # The subdivision edges, drawn in the stage — `isubd_frag`'s stroke, and the
    # same half-width-inward rule: a shared edge gets half from each of its two
    # triangles and the halves meet as one full-width line. Each barycentric is
    # divided by its OWN screen-space gradient, which makes it a distance in
    # PIXELS, so the line is one width at every zoom; taking `min` first and
    # differentiating that swells the stroke into a blob at every vertex.
    if stroke > 0f0
        u1 = inputs.bary[1]
        u2 = inputs.bary[2]
        u3 = 1f0 - u1 - u2
        d1 = u1 / max(sqrt(KI.dFdx(u1)^2 + KI.dFdy(u1)^2), 1f-8)
        d2 = u2 / max(sqrt(KI.dFdx(u2)^2 + KI.dFdy(u2)^2), 1f-8)
        d3 = u3 / max(sqrt(KI.dFdx(u3)^2 + KI.dFdy(u3)^2), 1f-8)
        d = min(d1, min(d2, d3))
        # Stroke OR face, with one pixel of coverage at the boundary — not a
        # gradient across the whole width, which has no flat core and reads as a
        # smudge rather than a line.
        cov = clamp(stroke - d + 0.5f0, 0f0, 1f0)
        r = r + (STROKE_COLOR[1] - r) * cov
        g = g + (STROKE_COLOR[2] - g) * cov
        b = b + (STROKE_COLOR[3] - b) * cov
    end
    return Vec4f(r, g, b, 1f0)
end

# ── The pipeline, and the objects it draws ──────────────────────────────────

"""
The mesh pipeline the FEM raster path draws through, built once per screen.

`DepthLessEq` and a depth attachment, like every other RayMakie overlay: the
overlay pass shares one depth buffer, and a surface that occludes itself needs
the test to sort its own triangles as well.
"""
function fem_overlay_pipeline!(screen)
    get!(screen.gfx_pipelines, :fem_elements) do
        Mantle.MeshPipeline(
            mesh = Mantle.MeshShader(fem_overlay_mesh;
                     outputs = (xi = Vec2f, cell = Mantle.Flat{Float32},
                                bary = Vec2f, nrm = Vec3f, wpos = Vec3f),
                     max_vertices = 3 * FEM_OVERLAY_THREADS,
                     max_primitives = FEM_OVERLAY_THREADS,
                     topology = Mantle.TriangleList(),
                     threads = FEM_OVERLAY_THREADS),
            fragment = Mantle.FragmentShader(fem_overlay_frag),
            cull = Mantle.NoCull(), depth = Mantle.DepthLessEq())
    end
end

"""
The environment the raster shading reflects, uploaded once per session.

The same sky the tracer has — `demosky` is `demoenv` in the convention
[`sampleenv`](@ref) reads — resampled to the flat `Float32` buffer that function
indexes. One per backend: it is 96 MB and it never changes.
"""
const FEM_OVERLAY_ENV = Base.RefValue{Any}(nothing)

function fem_overlay_env(backend)
    FEM_OVERLAY_ENV[] === nothing &&
        (FEM_OVERLAY_ENV[] = Mantle.devicearray(backend, skyenv()))
    return FEM_OVERLAY_ENV[]
end

"""
The device-side element data for one material, and the keys refined to its
tolerance.

Cached on the MATERIAL's element arrays rather than rebuilt per call: the
coefficients do not change while the demo runs, and the refinement is a handful
of GPU passes that only the tolerance moves.
"""
const FEM_OVERLAY_CACHE = IdDict{Any, Any}()

function fem_overlay_data(backend, elements, tolerance)
    entry = get(FEM_OVERLAY_CACHE, elements, nothing)
    if entry === nothing
        corners, cells = basetriangles(size(elements.cf, 2))
        entry = (cx = Mantle.devicearray(backend, vec(elements.cx)),
                 cy = Mantle.devicearray(backend, vec(elements.cy)),
                 cf = Mantle.devicearray(backend, vec(elements.cf)),
                 basecorners = Mantle.devicearray(backend, corners),
                 basecell = Mantle.devicearray(backend, cells),
                 keys = Base.RefValue{Any}(nothing),
                 tol = Base.RefValue{Float64}(NaN))
        FEM_OVERLAY_CACHE[elements] = entry
    end
    if entry.tol[] != tolerance
        k0 = Mantle.devicearray(backend, rootkeys(4 * size(elements.cf, 2)))
        entry.keys[] = refine(k0, entry.cx, entry.cy, entry.cf,
                              entry.basecorners, entry.basecell;
                              geo_tol = tolerance, fld_tol = tolerance)
        entry.tol[] = tolerance
    end
    return entry
end

"""
    mesh_overlay_dispatch!(::Hikari.FEMMaterial, screen, scene, plot, args, last)

The FEM element's raster path: a mesh stage, one workgroup per batch of keys.

No vertex buffer and no index buffer — the keys ARE the geometry, and a
tolerance change is a different key set and the same draw. That is the claim the
demo makes and this is where it is made.
"""
function RayMakie.mesh_overlay_dispatch!(material::Hikari.FEMMaterial, screen, scene,
                                         plot, args, last_robj)
    backend = screen.config.device
    e = material.elements
    data = fem_overlay_data(backend, e, Float64(material.tolerance))
    keys = data.keys[]
    nkeys = length(keys)

    mvp = RayMakie.plot_clip_matrix(plot) * Mat4f(args.model_f32c)
    eye = Vec3f(scene.camera.eyeposition[])
    # The tracer's range, not the coefficients' extrema: the material was told
    # what a value of 0 and a value of 1 look like, and a rasteriser that
    # normalises by something else shows a different picture of the same field.
    params = Vec4f(e.warp, material.field.vmin, material.field.vmax, FEM_OVERLAY_EXPOSURE)
    # In RENDER-TARGET pixels, so it scales with `px_per_unit`: a screen that
    # supersamples and filters down would otherwise draw the line half as wide
    # as it asked for.
    strokewidth = Float32(STROKE_HALFPX * screen.px_per_unit)
    env = fem_overlay_env(backend)

    # `cld(nkeys, threads)` workgroups, and no `+ 1`: the standalone renderer
    # spends its first group on a fullscreen background triangle, and an overlay
    # has a scene underneath it instead.
    groups = cld(nkeys, FEM_OVERLAY_THREADS)

    if last_robj isa RayMakie.RenderObject
        RayMakie.update_buffer!(last_robj, :keys, keys)
        last_robj.uniforms[:nkeys] = Int32(nkeys)
        last_robj.uniforms[:mvp] = mvp
        last_robj.uniforms[:params] = params
        last_robj.uniforms[:stroke] = strokewidth
        last_robj.uniforms[:eye] = eye
        # The ramp travels with the MATERIAL, so a menu swap changes it under an
        # object this path keeps — `update_buffer!` is what makes that reach the
        # draw, and it is the same call the keys take.
        RayMakie.update_buffer!(last_robj, :ramp,
            [Vec3f(c.c[1], c.c[2], c.c[3]) for c in material.field.ramp])
        last_robj.vertex_count = groups
        last_robj.visible = true
        return last_robj
    end

    return RayMakie.RenderObject(fem_overlay_pipeline!(screen);
        backend,
        arg_names = (:keys, :basecorners, :basecell, :cx, :cy, :cf, :ramp,
                     :nkeys, :mvp, :params, :stroke, :eye, :lightdir, :env),
        buffers = Dict{Symbol, Any}(
            # Its OWN key buffer, not the cache's: `update_buffer!` resizes and
            # copies into whatever is here, and handing it the array it would be
            # copying FROM is a buffer copy onto itself.
            :keys => Mantle.devicearray(backend, Array(keys)),
            :basecorners => data.basecorners,
            :basecell => data.basecell,
            :cx => data.cx, :cy => data.cy, :cf => data.cf,
            :ramp => Mantle.devicearray(backend,
                         [Vec3f(c.c[1], c.c[2], c.c[3]) for c in material.field.ramp]),
            :env => env,
        ),
        uniforms = Dict{Symbol, Any}(
            :nkeys => Int32(nkeys),
            :mvp => mvp,
            :params => params,
            :stroke => strokewidth,
            :eye => eye,
            :lightdir => Vec3f(LIGHTDIR),
        ),
        vertex_count = groups,
        instances = 1,
    )
end
