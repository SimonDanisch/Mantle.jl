import KernelInterface as KI

function fem_overlay_mesh(out, keys, basecorners, basecell, cx, cy, cf, ramp,
                          nkeys::Int32, mvp::Mat4f, params::Vec4f, stroke::Float32,
                          eye::Vec3f, lightdir::Vec3f, env)
    g = KI.mesh_group_index()
    t = KI.mesh_thread_index()
    warp = Float32(params[1])  # Apple GPUs have no Float64

    i = (g - Int32(1)) * Int32(meshthreads()) + t
    nlocal = min(Int32(meshthreads()), nkeys - (g - Int32(1)) * Int32(meshthreads()))
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
    # Not `Int(round(x))`: its InexactError cannot be built in a fragment stage.
    cell = Int(unsafe_trunc(Int32, inputs.cell + 0.5f0))
    n = inputs.nrm
    nl = sqrt(n[1]*n[1] + n[2]*n[2] + n[3]*n[3]) + 1f-8
    nn = Vec3f(n[1]/nl, n[2]/nl, n[3]/nl)

    d = inputs.wpos - eye
    dl = sqrt(d[1]*d[1] + d[2]*d[2] + d[3]*d[3]) + 1f-8
    viewdir = Vec3f(d[1]/dl, d[2]/dl, d[3]/dl)

    if nn[1]*viewdir[1] + nn[2]*viewdir[2] + nn[3]*viewdir[3] > 0f0
        nn = Vec3f(-nn[1], -nn[2], -nn[3])
    end

    val = Float32(eval_poly(cf, cell, inputs.xi[1], inputs.xi[2]))
    vmin = params[2]; vmax = params[3]
    t = clamp((val - vmin) / max(vmax - vmin, 1f-8), 0f0, 1f0)

    cr, cg, cb = ramplookup(ramp, t)
    r, g, b = shade(env, cr, cg, cb, nn, viewdir, lightdir, params[4])

    if stroke > 0f0
        u1 = inputs.bary[1]
        u2 = inputs.bary[2]
        u3 = 1f0 - u1 - u2
        # Each edge's pixel distance on its own, then the min; the other order blobs the vertices.
        d1 = u1 / max(sqrt(KI.dFdx(u1)^2 + KI.dFdy(u1)^2), 1f-8)
        d2 = u2 / max(sqrt(KI.dFdx(u2)^2 + KI.dFdy(u2)^2), 1f-8)
        d3 = u3 / max(sqrt(KI.dFdx(u3)^2 + KI.dFdy(u3)^2), 1f-8)
        d = min(d1, min(d2, d3))
        cov = clamp(stroke - d + 0.5f0, 0f0, 1f0)
        r = r + (strokecolor()[1] - r) * cov
        g = g + (strokecolor()[2] - g) * cov
        b = b + (strokecolor()[3] - b) * cov
    end
    return Vec4f(r, g, b, 1f0)
end

function fem_overlay_pipeline!(screen)
    get!(screen.gfx_pipelines, :fem_elements) do
        Mantle.MeshPipeline(
            mesh = Mantle.MeshShader(fem_overlay_mesh;
                     outputs = (xi = Vec2f, cell = Mantle.Flat{Float32},
                                bary = Vec2f, nrm = Vec3f, wpos = Vec3f),
                     max_vertices = 3 * meshthreads(),
                     max_primitives = meshthreads(),
                     topology = Mantle.TriangleList(),
                     threads = meshthreads()),
            fragment = Mantle.FragmentShader(fem_overlay_frag),
            cull = Mantle.NoCull(), depth = Mantle.DepthLessEq())
    end
end

overlay_env = Ref{Any}(nothing)

function fem_overlay_env(backend)
    overlay_env[] === nothing && (overlay_env[] = Mantle.devicearray(backend, skyenv()))
    return overlay_env[]
end

overlay_cache = IdDict{Any, Any}()

function fem_overlay_data(backend, elements, tolerance)
    entry = get(overlay_cache, elements, nothing)
    if entry === nothing
        corners, cells = basetriangles(size(elements.cf, 2))
        entry = (cx = Mantle.devicearray(backend, vec(elements.cx)),
                 cy = Mantle.devicearray(backend, vec(elements.cy)),
                 cf = Mantle.devicearray(backend, vec(elements.cf)),
                 basecorners = Mantle.devicearray(backend, corners),
                 basecell = Mantle.devicearray(backend, cells),
                 keys = Base.RefValue{Any}(nothing),
                 tol = Base.RefValue{Float64}(NaN))
        overlay_cache[elements] = entry
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

function RayMakie.mesh_overlay_dispatch!(material::Hikari.FEMMaterial, screen, scene,
                                         plot, args, last_robj)
    backend = screen.config.device
    e = material.elements
    data = fem_overlay_data(backend, e, Float64(material.tolerance))
    keys = data.keys[]
    nkeys = length(keys)

    mvp = RayMakie.plot_clip_matrix(plot) * Mat4f(args.model_f32c)
    eye = Vec3f(scene.camera.eyeposition[])
    exposure = 3.5f0
    params = Vec4f(e.warp, material.field.vmin, material.field.vmax, exposure)
    strokewidth = Float32(strokehalfpx() * screen.px_per_unit)
    env = fem_overlay_env(backend)

    groups = cld(nkeys, meshthreads())

    if last_robj isa RayMakie.RenderObject
        RayMakie.update_buffer!(last_robj, :keys, keys)
        last_robj.uniforms[:nkeys] = Int32(nkeys)
        last_robj.uniforms[:mvp] = mvp
        last_robj.uniforms[:params] = params
        last_robj.uniforms[:stroke] = strokewidth
        last_robj.uniforms[:eye] = eye
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
            :keys => Mantle.devicearray(backend, Array(keys)),  # not the cache's: updates copy from it
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
            :lightdir => lightdirection(),
        ),
        vertex_count = groups,
        instances = 1,
    )
end
