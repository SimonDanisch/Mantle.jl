import KernelInterface as KI

fem_arg_names() = (:keys, :basecorners, :basecell, :cx, :cy, :cf, :ramp,
                   :light_types, :light_colors, :light_parameters,
                   :nkeys, :model, :view, :projection, :world_normalmatrix, :params,
                   :eyeposition, :shading_mode, :ambient, :light_color, :light_direction,
                   :N_lights, :has_env, :env_sh, :diffuse, :specular, :shininess, :backlight,
                   :exposure, :tonemap, :white_point, :inv_gamma, :apply_gamma,
                   :strokewidth, :strokecolor, :resolution, :px_per_unit, :fxaa)

function fem_overlay_mesh(out, keys, basecorners, basecell, cx, cy, cf, ramp,
                          light_types, light_colors, light_parameters,
                          nkeys::Int32, model::Mat4f, view::Mat4f, projection::Mat4f,
                          world_normalmatrix::Mat3f, params::Vec4f,
                          eyeposition::Vec3f, shading_mode::Int32, ambient::Vec3f,
                          light_color::Vec3f, light_direction::Vec3f, N_lights::Int32,
                          has_env::Int32, env_sh::Mat{3,9,Float32}, diffuse::Vec3f, specular::Vec3f,
                          shininess::Float32, backlight::Float32, exposure::Float32, tonemap::Int32,
                          white_point::Float32, inv_gamma::Float32, apply_gamma::Int32,
                          strokewidth::Float32, strokecolor::Vec4f, resolution::Vec2f,
                          px_per_unit::Float32, fxaa::Int32)
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
        pvm = projection * view * model

        p1 = Vec3f(surfacepoint(cx, cy, cf, cell, c1[1], c1[2], warp)...)
        p2 = Vec3f(surfacepoint(cx, cy, cf, cell, c2[1], c2[2], warp)...)
        p3 = Vec3f(surfacepoint(cx, cy, cf, cell, c3[1], c3[2], warp)...)
        n1 = Vec3f(world_normalmatrix * surfacenormal(cx, cy, cf, cell, c1[1], c1[2], warp))
        n2 = Vec3f(world_normalmatrix * surfacenormal(cx, cy, cf, cell, c2[1], c2[2], warp))
        n3 = Vec3f(world_normalmatrix * surfacenormal(cx, cy, cf, cell, c3[1], c3[2], warp))

        KI.set_mesh_vertex!(out, v0 + Int32(1), fem_vertex(p1, n1, c1, cellf, p1, p2, p3, model, pvm))
        KI.set_mesh_vertex!(out, v0 + Int32(2), fem_vertex(p2, n2, c2, cellf, p1, p2, p3, model, pvm))
        KI.set_mesh_vertex!(out, v0 + Int32(3), fem_vertex(p3, n3, c3, cellf, p1, p2, p3, model, pvm))
        KI.set_mesh_triangle!(out, t, v0 + Int32(1), v0 + Int32(2), v0 + Int32(3))
    end
    return nothing
end

@inline function fem_vertex(p::Vec3f, n::Vec3f, xi, cellf::Float32, a::Vec3f, b::Vec3f, c::Vec3f,
                            model::Mat4f, pvm::Mat4f)
    clip = Vec4f(pvm * Vec4f(p[1], p[2], p[3], 1f0))
    w = model * Vec4f(p[1], p[2], p[3], 1f0)
    return (position = RayMakie.gl_to_clip_depth(clip), xi = Vec2f(xi[1], xi[2]), cell = cellf,
            nrm = n, wpos = Vec3f(w[1], w[2], w[3]) / w[4], clip = clip, c1 = a, c2 = b, c3 = c)
end

@inline function ramplookup(ramp, s::Float32)
    f = clamp(s, 0f0, 1f0) * Float32(Hikari.FEM_RAMP_N - 1)
    i0 = clamp(unsafe_trunc(Int32, f), Int32(0), Int32(Hikari.FEM_RAMP_N - 2))
    w = f - Float32(i0)
    @inbounds a = ramp[i0 + Int32(1)]
    @inbounds b = ramp[i0 + Int32(2)]
    return a * (1f0 - w) + b * w
end

# Every subdivision edge, half the width from each of its two triangles.
@inline function fem_stroke(color::Vec4f, inputs, pvm::Mat4f, resolution::Vec2f,
                            px_per_unit::Float32, strokewidth::Float32, strokecolor::Vec4f)
    strokewidth <= 0f0 && return color
    clip = inputs.clip
    scale = px_per_unit * resolution
    frag = Vec2f((0.5f0 * clip[1] / clip[4] + 0.5f0) * scale[1], (0.5f0 * clip[2] / clip[4] + 0.5f0) * scale[2])
    a = RayMakie.stroke_screen_space(inputs.c1, pvm, scale)
    b = RayMakie.stroke_screen_space(inputs.c2, pvm, scale)
    c = RayMakie.stroke_screen_space(inputs.c3, pvm, scale)
    edge(p, q) = RayMakie.edge_face_factor(frag, Vec2f(p[1], p[2]), Vec2f(q[1], q[2]), 0.5f0,
                                           strokewidth, px_per_unit)
    ff = min(edge(a, b), min(edge(b, c), edge(c, a)))
    return strokecolor * (1f0 - ff) + color * ff
end

function fem_overlay_frag(inputs, keys, basecorners, basecell, cx, cy, cf, ramp,
                          light_types, light_colors, light_parameters,
                          nkeys::Int32, model::Mat4f, view::Mat4f, projection::Mat4f,
                          world_normalmatrix::Mat3f, params::Vec4f,
                          eyeposition::Vec3f, shading_mode::Int32, ambient::Vec3f,
                          light_color::Vec3f, light_direction::Vec3f, N_lights::Int32,
                          has_env::Int32, env_sh::Mat{3,9,Float32}, diffuse::Vec3f, specular::Vec3f,
                          shininess::Float32, backlight::Float32, exposure::Float32, tonemap::Int32,
                          white_point::Float32, inv_gamma::Float32, apply_gamma::Int32,
                          strokewidth::Float32, strokecolor::Vec4f, resolution::Vec2f,
                          px_per_unit::Float32, fxaa::Int32)
    # Not `Int(round(x))`: its InexactError cannot be built in a fragment stage.
    cell = Int(unsafe_trunc(Int32, inputs.cell + 0.5f0))
    val = Float32(eval_poly(cf, cell, inputs.xi[1], inputs.xi[2]))
    rgb = ramplookup(ramp, (val - params[2]) / max(params[3] - params[2], 1f-8))

    if shading_mode != RayMakie.SHADING_NONE
        camdir = RayMakie._unit(inputs.wpos - eyeposition)
        n = RayMakie._unit(inputs.nrm)
        n = dot(n, camdir) > 0f0 ? -n : n  # two-sided: an element is a sheet
        lit = RayMakie.illuminate(inputs.wpos, camdir, n, rgb, shading_mode, ambient, light_color,
                                  light_direction, N_lights, light_types, light_colors,
                                  light_parameters, has_env, env_sh, diffuse, specular,
                                  shininess, backlight)
        rgb = RayMakie.film_mapping(lit, exposure, tonemap, white_point, inv_gamma, apply_gamma)
    end
    color = fem_stroke(Vec4f(rgb[1], rgb[2], rgb[3], 1f0), inputs, projection * view * model,
                       resolution, px_per_unit, strokewidth, strokecolor)
    return RayMakie.raster_output(color, fxaa)
end

function fem_overlay_pipeline!(screen)
    get!(screen.gfx_pipelines, :fem_elements) do
        Mantle.MeshPipeline(
            mesh = Mantle.MeshShader(fem_overlay_mesh;
                     outputs = (xi = Vec2f, cell = Mantle.Flat{Float32}, nrm = Vec3f, wpos = Vec3f,
                                clip = Vec4f, c1 = Mantle.Flat{Vec3f}, c2 = Mantle.Flat{Vec3f},
                                c3 = Mantle.Flat{Vec3f}),
                     max_vertices = 3 * meshthreads(),
                     max_primitives = meshthreads(),
                     topology = Mantle.TriangleList(),
                     threads = meshthreads()),
            fragment = Mantle.FragmentShader(fem_overlay_frag),
            cull = Mantle.NoCull(), depth = Mantle.DepthLessEq())
    end
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
                                         plot, args, changed, last_robj)
    backend = screen.config.device
    e = material.elements
    data = fem_overlay_data(backend, e, Float64(material.tolerance))
    keys = data.keys[]
    shading, lights = RayMakie.raster_shading(screen, plot, args)
    uniforms = merge((nkeys = Int32(length(keys)), model = Mat4f(args.model_f32c),
                      view = Mat4f(args.view), projection = Mat4f(args.projection),
                      world_normalmatrix = Mat3f(args.world_normalmatrix),
                      params = Vec4f(e.warp, material.field.vmin, material.field.vmax, 0f0)), shading)
    ramp = [Vec3f(c.c[1], c.c[2], c.c[3]) for c in material.field.ramp]
    groups = cld(length(keys), meshthreads())

    if last_robj isa RayMakie.RenderObject
        RayMakie.update_buffer!(last_robj, :keys, keys)
        RayMakie.update_buffer!(last_robj, :ramp, ramp)
        for (name, value) in pairs(lights)
            RayMakie.update_buffer!(last_robj, name, value)
        end
        for (name, value) in pairs(uniforms)
            last_robj.uniforms[name] = value
        end
        last_robj.vertex_count = groups
        last_robj.visible = true
        return last_robj
    end

    buffers = Dict{Symbol, Any}(
        :keys => Mantle.devicearray(backend, Array(keys)),  # not the cache's: updates copy from it
        :basecorners => data.basecorners, :basecell => data.basecell,
        :cx => data.cx, :cy => data.cy, :cf => data.cf,
        :ramp => Mantle.devicearray(backend, ramp))
    for (name, value) in pairs(lights)
        buffers[name] = Mantle.devicearray(backend, value)
    end
    return RayMakie.RenderObject(fem_overlay_pipeline!(screen);
        backend, fxaa = RayMakie.plot_fxaa(plot), arg_names = fem_arg_names(), buffers,
        uniforms = Dict{Symbol, Any}(pairs(uniforms)),
        vertex_count = groups, instances = 1)
end
