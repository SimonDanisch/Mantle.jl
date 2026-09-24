using Makie, RayMakie, Mantle, GeometryBasics, Colors, Printf, FileIO
using Makie: cam3d!, update_cam!, Consume
import Hikari

include(joinpath(@__DIR__, "isubd_overlay.jl"))

field_ramp = [RGBf(0.15 + 0.75t, 0.15 + 0.65t, 0.55 - 0.45t) for t in range(0, 1; length = 64)]
gold_ramp = [RGBf(0.62 + 0.38t, 0.36 + 0.50t, 0.13 + 0.42t) for t in range(0, 1; length = 64)]

# `remap_roughness = false` passes alpha directly; the remap makes these a wide, dull lobe.
# ThinDielectric, not Dielectric: an element has no thickness, and solid glass would only reflect.
fem_surfaces = [
    ("COATED",  Hikari.CoatedDiffuse(roughness = 0.035, thickness = 0.02, eta = 1.5,
                                     remap_roughness = false),              field_ramp),
    ("GOLD",    Hikari.Conductor(eta = (0.143, 0.375, 1.44), k = (3.98, 2.39, 1.60),
                                 roughness = 0.05, remap_roughness = false), gold_ramp),
    ("GLASS",   Hikari.ThinDielectric(eta = 2.4f0),                         field_ramp),
]

function isubd_gui(cx0, cy0, cf0; width = 760, height = 760, surface = 1, warp = 0.6)
    fig = Figure(size = (1120, 820), backgroundcolor = RGBf(0.96, 0.96, 0.97))

    cx, cy, cf = Float32.(cx0), Float32.(cy0), Float32.(cf0)

    env = demoenv()
    ls = LScene(fig[1, 1]; show_axis = false,
                scenekw = (; lights = [Makie.EnvironmentLight(1.0f0,
                                          map(c -> RGBf(c.c[1], c.c[2], c.c[3]), env))]))
    cam3d!(ls.scene; fov = Float32(0.8 * 180 / pi), projectiontype = Makie.Perspective)
    cc = Makie.cameracontrols(ls.scene)
    update_cam!(ls.scene, cc, Vec3f(2.55, -1.85, 1.30), Vec3f(0, 0, 0.25), Vec3f(0, 0, 1))

    femmat_for(i, tol) = Hikari.FEMMaterial(fem_surfaces[i][2], cx, cy, cf; warp,
                                            colormap = fem_surfaces[i][3],
                                            colorrange = (0, 1), tolerance = tol)
    cursurface = Base.RefValue(surface)
    curtol = Base.RefValue(2.0f-2)
    coarse = GeometryBasics.normal_mesh(Rect3f(Point3f(-1.3, -1.3, -0.1), Vec3f(2.6, 2.6, 0.9)))
    femplot = Base.RefValue{Any}(mesh!(ls, coarse; material = femmat_for(surface, curtol[])))

    panel = GridLayout(fig[1, 2]; tellheight = false)
    Label(panel[1, 1], "renderer"; fontsize = 14, halign = :left, color = RGBf(0.35, 0.35, 0.4))
    modebtn = Makie.Button(panel[2, 1]; label = "PATH TRACED", width = 260, height = 48,
                           fontsize = 20, buttoncolor = RGBf(0.95, 0.87, 0.82))
    Label(panel[3, 1], "material"; fontsize = 14, halign = :left, color = RGBf(0.35, 0.35, 0.4))
    matmenu = Menu(panel[4, 1]; options = first.(fem_surfaces),
                   default = fem_surfaces[surface][1], width = 260, fontsize = 16)
    Label(panel[5, 1], "tessellation tolerance"; fontsize = 15, halign = :left)
    tolsl = Makie.Slider(panel[6, 1]; range = range(log10(6e-2), log10(2e-3); length = 220),
                         startvalue = log10(2e-2), width = 260)
    Label(panel[7, 1], "drag to orbit, scroll to zoom —\nin EITHER renderer";
          fontsize = 13, halign = :left, justification = :left,
          color = RGBf(0.35, 0.35, 0.4), tellwidth = false)
    Label(panel[8, 1],
          "One `mesh!`, one camera, one set of\nlights. `material = FEMMaterial` puts the\n" *
          "$(size(cf, 2)) elements in as ONE BOX EACH for\nthe tracer and as subdivision keys for\n" *
          "the rasteriser — the material decides,\non both paths.";
          fontsize = 13, halign = :left, justification = :left,
          color = RGBf(0.3, 0.3, 0.35), tellwidth = false)
    colsize!(fig.layout, 2, Fixed(290))

    raster = Observable(false)
    on(modebtn.clicks) do _
        raster[] = !raster[]
        modebtn.label[] = raster[] ? "RASTER" : "PATH TRACED"
        modebtn.buttoncolor[] = raster[] ? RGBf(0.85, 0.87, 0.95) : RGBf(0.95, 0.87, 0.82)
        RayMakie.setrasterize!(Makie.getscreen(fig.scene), raster[])
        return nothing
    end
    on(matmenu.selection) do name
        i = findfirst(p -> p[1] == name, fem_surfaces)
        i === nothing && return nothing
        cursurface[] = i
        # A new plot, not `material[] = …`: the attribute is typed by the first material.
        delete!(ls.scene, femplot[])
        femplot[] = mesh!(ls, coarse; material = femmat_for(i, curtol[]))
        return nothing
    end
    on(tolsl.value) do lv
        curtol[] = Float32(10.0^lv)
        femplot[].material[] = femmat_for(cursurface[], curtol[])
        return nothing
    end

    return fig, (; modebtn, matmenu, tolsl, lscene = ls, cam = cc)
end

function warmup!(fig, ctrl; frames::Integer = 2, timeout::Real = 600)
    scr = Makie.getscreen(fig.scene)
    scr === nothing && throw(ArgumentError("warmup!: display the figure first"))
    function settle()
        f0 = scr.frames_presented[]
        t1 = time()
        while scr.frames_presented[] < f0 + frames
            RayMakie.renderloop_running(scr) ||
                error("warmup!: the render loop stopped; its log says why")
            time() - t1 > timeout && error("warmup!: no frame within $(timeout) s")
            sleep(0.05)
        end
    end
    t0 = time()
    start = ctrl.matmenu.selection[]
    for (name, _...) in fem_surfaces
        name == start && continue
        ctrl.matmenu.selection[] = name
        settle()
    end
    ctrl.matmenu.selection[] = start
    settle()
    for _ in 1:2
        ctrl.modebtn.clicks[] = ctrl.modebtn.clicks[] + 1
        settle()
    end
    return time() - t0
end
