# The demo: adaptive FEM tessellation, live, in a Makie window on RayMakie.
#
# One figure, two renderers behind one button:
#
#   RASTER      a mesh shader expands subdivision keys into triangles. The
#               tessellation slider moves the error tolerance, and the triangle
#               count moves with it.
#   PATH TRACED the same elements, solved per ray. Exact silhouette, real BSDF,
#               real shadows, real Monte Carlo noise.
#
# ONE plot either way: `mesh!` with `material = Hikari.FEMMaterial(...)`. The
# material carries the coefficients, and each renderer asks it for what it needs
# — boxes and an isoparametric solve for the tracer (`Hikari/src/fem.jl`),
# subdivision keys and a mesh stage for the rasteriser (`isubd_overlay.jl`). The
# button is a flag on the SCREEN, not a second program.
#
# Zoom in far enough and the difference stops being an argument: the raster's
# boundary is visibly polygonal and the traced one is not, at any zoom.
#
# Everything on screen is Makie, drawn by RayMakie — the widgets included.

using Makie, RayMakie, Mantle, GeometryBasics, Colors, Printf, FileIO
using Makie: cam3d!, update_cam!, Consume
import Hikari

include("isubd_overlay.jl")

"""The field's colour ramp — blue at 0, yellow at 1."""
const FIELD_RAMP = [RGBf(0.15 + 0.75t, 0.15 + 0.65t, 0.55 - 0.45t)
                    for t in range(0, 1; length = 64)]

"""
The same field as a METAL's reflectance: copper at 0, pale gold at 1.

A conductor's colour is its reflectance, and [`FIELD_RAMP`](@ref)'s dark end is
`(0.15, 0.15, 0.55)` — a metal that reflects 15% of the red reaching it and
most of the blue, which is not gold, it is tarnish. This is the same field read
as an alloy: warm across its whole range, so the surface is metal everywhere and
the solution is which metal.
"""
const GOLD_RAMP = [RGBf(0.62 + 0.38t, 0.36 + 0.50t, 0.13 + 0.42t)
                   for t in range(0, 1; length = 64)]

"""
The materials the demo cycles through, and the reason the field is a TEXTURE.

`FEMMaterial` wraps any Hikari material and merges the field into whatever that
one uses as its colour, so the solution can be matte, lacquered, gold or glass
without a single one of them knowing what a finite element is. Writing a colour
into the hit — the first design — would have made all four impossible: there is
no BSDF left to be gold with.

GLASS is the one that has no colour to merge into, and that is not a gap: a pane
of glass has no albedo. `merge_color_with_material(tex, ::ThinDielectric)`
returns it unchanged and the field shows in the other three.
"""
const FEM_SURFACES = [
    # No MATTE. A `Diffuse` and a `CoatedDiffuse` carrying the same ramp are the
    # same picture with a highlight on one of them, and two menu entries that
    # look alike read as the menu doing nothing. The coat is the one that keeps
    # the field AND shows the material, so it is the one that stays.
    # `remap_roughness = false`, and that is the whole reason these looked flat.
    # pbrt's remap is `alpha = sqrt(roughness)`, so the 0.04 this started at was
    # alpha 0.2 — a lobe far wider than a 1.5° sun, which is why sharpening the
    # sun changed nothing and polishing to 0.015 changed nothing either (alpha
    # 0.12). Passing alpha directly, 0.02 is a mirror finish: the sun arrives as
    # a glint and the floor as a floor.
    ("COATED",  Hikari.CoatedDiffuse(roughness = 0.035, thickness = 0.02, eta = 1.5,
                                     remap_roughness = false),              FIELD_RAMP),
    ("GOLD",    Hikari.Conductor(eta = (0.143, 0.375, 1.44), k = (3.98, 2.39, 1.60),
    #
    # Alpha 0.05 rather than a mirror's 0.02: a polished surface shows whatever
    # is AROUND it, and from half the orbit that is the dark ground — mirror
    # gold went olive as the camera came down. A wider lobe averages more of the
    # environment, so it stays metal from every angle and still takes the sun as
    # a streak.
                                 roughness = 0.05, remap_roughness = false), GOLD_RAMP),
    # `ThinDielectric`, and this is the whole answer to "why does the glass look
    # opaque". It was a `Dielectric` — the boundary of SOLID glass — and a FEM
    # element is a surface with no thickness, so a ray that refracted in had to
    # leave through the far side of the arch, which it meets at a grazing angle:
    # past the 41.8° critical angle, every one of them totally internally
    # reflects and never comes out. The sheet was a mirror, which is why it read
    # as a dark solid.
    #
    # Measured, not guessed. At `index = 1.01` (critical angle 82°) the sheet is
    # nearly invisible, so transmission was never broken; `max_depth = 64` is
    # pixel-identical to 16, so nothing was being truncated; and the same
    # `Dielectric(1.5)` on a SPHERE in this scene is textbook glass. Light was
    # being reflected away, not lost.
    #
    # A pane is two interfaces a hair apart: light enters and leaves undeviated,
    # with the multiple internal reflections summed analytically. That is what
    # `ThinDielectric` is, and it is what this surface physically is.
    # `eta = 2.4` is above window glass's 1.5, and deliberately: this surface is
    # broad and gently curved, so the camera sees almost all of it face-on,
    # where a thin dielectric reflects a few percent and is nearly invisible. A
    # higher index raises the Fresnel term at every angle, which is what gives
    # the panel an outline to read on a projector. Nothing else changes — it is
    # the same BSDF transmitting the same way, showing more of the sky.
    # The comparison that matters is a solid `Dielectric` beside this one: a FEM
    # element has no thickness, so `Dielectric` total-internal-reflects and
    # renders as a dark mirror. `ThinDielectric` is what a pane of glass is.
    ("GLASS",   Hikari.ThinDielectric(eta = 2.4f0),                         FIELD_RAMP),
]

"""
    isubd_gui(backend, cx, cy, cf; surface) -> (figure, controls)

ONE scene, one plot, one camera.

The plot is a `mesh!` whose material is `Hikari.FEMMaterial`, and the button is
`RayMakie.setrasterize!` — so the two renderers are two paths through the same
scene rather than two programs. The camera is Makie's, always live, identical in
either mode; the lights are the scene's; the material is the plot's.

That is worth stating because the first version was not this. It drove a second
renderer into a Mantle framebuffer, read the pixels back and showed them as an
`image!` in a 2D `Axis` beside a hidden `LScene` — which meant two cameras to
keep in step, no camera input at all while the raster showed, and a flip that
changed the material as well as the geometry.
"""
function isubd_gui(backend, cx0, cy0, cf0; width = 760, height = 760, surface = 1)
    fig = Figure(size = (1120, 820), backgroundcolor = RGBf(0.96, 0.96, 0.97))

    # Converted ONCE. `Hikari.femcoeffs` passes `Float32` data through, so every
    # material built below carries the SAME three arrays — which is what lets the
    # raster path keep its uploads and its refinement across a material swap and
    # across a tolerance move.
    cx, cy, cf = Float32.(cx0), Float32.(cy0), Float32.(cf0)

    env = demoenv()
    # ONE light, and it is the environment. The sun used to be a second,
    # `DirectionalLight` one — which shades a surface and paints nothing along a
    # reflected ray, so nothing in the scene could SEE it. It is a disc in the
    # map now (`demoenv`), which is what gives gold a highlight and glass
    # something to reflect; adding the directional light back would light
    # everything twice.
    ls = LScene(fig[1, 1]; show_axis = false,
                scenekw = (; lights = [Makie.EnvironmentLight(1.0f0,
                                          map(c -> RGBf(c.c[1], c.c[2], c.c[3]), env))]))
    cam3d!(ls.scene; fov = Float32(0.8 * 180 / pi), projectiontype = Makie.Perspective)
    cc = Makie.cameracontrols(ls.scene)
    update_cam!(ls.scene, cc, Vec3f(2.55, -1.85, 1.30), Vec3f(0, 0, 0.25), Vec3f(0, 0, 1))

    femmat_for(i, tol) = Hikari.FEMMaterial(FEM_SURFACES[i][2], cx, cy, cf; warp = WARP,
                                            colormap = FEM_SURFACES[i][3],
                                            colorrange = (0, 1), tolerance = tol)
    cursurface = Base.RefValue(surface)
    curtol = Base.RefValue(2.0f-2)
    # The triangles a backend that knows nothing about curved elements draws.
    # RayMakie replaces them on both paths — the material says how.
    coarse = GeometryBasics.normal_mesh(Rect3f(Point3f(-1.3, -1.3, -0.1), Vec3f(2.6, 2.6, 0.9)))
    femplot = Base.RefValue{Any}(mesh!(ls, coarse; material = femmat_for(surface, curtol[])))

    panel = GridLayout(fig[1, 2]; tellheight = false)
    Label(panel[1, 1], "renderer"; fontsize = 14, halign = :left, color = RGBf(0.35, 0.35, 0.4))
    modebtn = Button(panel[2, 1]; label = "PATH TRACED", width = 260, height = 48,
                     fontsize = 20, buttoncolor = RGBf(0.95, 0.87, 0.82))
    Label(panel[3, 1], "material"; fontsize = 14, halign = :left, color = RGBf(0.35, 0.35, 0.4))
    matmenu = Menu(panel[4, 1]; options = first.(FEM_SURFACES),
                   default = FEM_SURFACES[surface][1], width = 260, fontsize = 16)
    Label(panel[5, 1], "tessellation tolerance"; fontsize = 15, halign = :left)
    # The range is MEASURED against this mesh, not carried over: 60 elements
    # refine to 240 triangles at 6e-2 (each cell's four base triangles, and the
    # curved rim visibly polygonal), 1080 at 5e-3 and 3024 at 2e-3. Past that
    # the strokes touch and the surface fills in solid, so the track ends where
    # the triangles can still be counted.
    tolsl = Slider(panel[6, 1]; range = range(log10(6e-2), log10(2e-3); length = 220),
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
        i = findfirst(p -> p[1] == name, FEM_SURFACES)
        i === nothing && return nothing
        cursurface[] = i
        # Rebuilt rather than re-materialised: Makie types an attribute slot
        # from the recipe's default, and `material` defaults to a plain
        # `Diffuse`. Rebuilding is honest anyway — this material carries the
        # geometry, so replacing it replaces what is in the scene.
        delete!(ls.scene, femplot[])
        femplot[] = mesh!(ls, coarse; material = femmat_for(i, curtol[]))
        return nothing
    end
    # The tolerance is a field of the MATERIAL, so moving it is a material
    # update and not a new plot: same concrete type, so the trace path swaps it
    # in place and the raster path re-refines. The tracer ignores the number —
    # it solves the element — which is why only the raster picture moves.
    on(tolsl.value) do lv
        curtol[] = Float32(10.0^lv)
        femplot[].material[] = femmat_for(cursurface[], curtol[])
        return nothing
    end

    return fig, (; modebtn, matmenu, tolsl, lscene = ls, cam = cc)
end
