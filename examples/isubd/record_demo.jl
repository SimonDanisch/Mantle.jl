# Record the isubd demo as a video.
#
# Every edit on screen comes from a VISIBLE cursor driving the real widgets —
# `MouseTo`/`LeftDown`/`LeftUp`/`LeftClick` through `FakeInteraction`, never a
# hidden `observable[] = …`. A viewer has to be able to see what was clicked and
# why the picture changed; an edit that just happens reads as magic.
#
#     julia> include("record_demo.jl")
#
# Writes `isubd_demo.mp4` into the working directory (`ISUBD_DEMO_OUT` overrides).

using Mantle, Makie, RayMakie, GeometryBasics, Printf
import Hikari

const HERE = @__DIR__

# Its own module, not `include`s into Main: half a dozen loaded packages export
# `Button`, so `Main` is the one scope this cannot be loaded into.
#
# `reference.jl` is NOT loaded here. It is the CPU fixture the mesh-shader
# comparison in `test_isubd_mesh.jl` runs against — two hand-written cells —
# and the demo builds its own mesh, which is `Isubd.annulus`.
module Isubd
include(joinpath(@__DIR__, "mantle_isubd.jl"))
include(joinpath(@__DIR__, "isubd_gui.jl"))
end

include(joinpath(HERE, "..", "..", "..", "Makie", "docs", "fake_interaction.jl"))
using .FakeInteraction: Wait, MouseTo, LeftClick, LeftDown, LeftUp, Lazy, Scroll, relative_pos

# ACCUMULATE, which is what a path-traced viewport does: hold still and the
# image converges, move and it starts over. `Makie.recordframe!` reads the
# screen with `colorbuffer`'s defaults, and the default without this is a fresh
# fixed-budget render per frame — the same amount of noise in every frame,
# however long the camera has been still.
# `max_depth = 16`: glass needs several refraction events to get a ray through
# and out the other side, and a path that runs out of depth comes back BLACK —
# which is what the dark patch on the earlier glass shot was.
# …and TWELVE samples per read on top of that: accumulation converges a still
# camera over frames, but a moving one clears the film every frame, so what an
# orbit shows is whatever one read puts in. Six was enough before the sun went
# into the environment map; a 2.5° disc against a sky a thousand times dimmer is
# a much harder thing to estimate, and it showed as speckle on every orbit.
RayMakie.activate!(accumulate = true, samples = 12,
                   integrator = Hikari.VolPath(max_depth = 16, hw_accel = true))

const BACKEND = Mantle.defaultbackend()
# 60 quadratic elements over a wavy annulus — see `Isubd.annulus`. Curved
# inside and out, so every silhouette in the frame is one the rasteriser has to
# approximate and the tracer does not.
const CX, CY, CF = Isubd.annulus()

# COATED to open with — `FEM_SURFACES[1]`: both renderers show the field through the SAME ramp —
# the raster path reads it off the material — so the flip in the middle changes
# the geometry and nothing else, which is the claim being made. Gold and glass
# come later, once the tracer is the only one drawing.
fig, ctrl = Isubd.isubd_gui(BACKEND, CX, CY, CF;
                                  width = 700, height = 700, surface = 1)

blockcentre(b) = relative_pos(b, (0.5, 0.5))

"A point `frac` along a slider's track, at its vertical middle."
sliderpos(sl, frac) = relative_pos(sl, (clamp(frac, 0.02, 0.98), 0.5))

"""
Click-drag a slider's handle from `from` to `to`, both fractions along its track.

ONE `MouseTo` with an explicit duration, not a chain of short ones: a `MouseTo`
already interpolates the cursor every frame between where it is and where it is
going, so `duration` is how long the drag takes and the motion is continuous.
Stepping it instead — which this did — moves the cursor in `steps` jumps, and an
omitted duration is `0.6 s` MINIMUM per jump. Twelve of those is eight seconds
of stutter for one drag.
"""
function dragslider(sl, from, to; duration = 1.5)
    return Any[Lazy(_ -> MouseTo(sliderpos(sl, from))), LeftDown(), Wait(0.15),
               Lazy(_ -> MouseTo(sliderpos(sl, to), duration)), Wait(0.15),
               LeftUp(), Wait(0.35)]
end

"""
Pick option `i` from an open-able `Menu`, by clicking it.

`relative_pos` is relative to the block's bbox with y growing UP, so a negative
fraction lands below it — which is where a Makie menu drops its options, each
row one menu-height tall.
"""
function pickmenu(menu, i; hold = 0.3, travel = 0.45)
    return Any[Lazy(_ -> MouseTo(relative_pos(menu, (0.5, 0.5)), travel)),
               LeftClick(), Wait(hold),
               Lazy(_ -> MouseTo(relative_pos(menu, (0.5, -(i - 0.5))), travel)),
               LeftClick(), Wait(hold)]
end

"""
Orbit by DRAGGING IN THE PICTURE, the way anyone would.

One continuous `MouseTo`, for the reason [`dragslider`](@ref) gives: a camera is
not a stepped control, and the interpolation this is built on is already per
frame.
"""
function orbit(block, from, to; duration = 1.3, y = 0.55)
    return Any[Lazy(_ -> MouseTo(relative_pos(block, (from, y)), 0.5)),
               LeftDown(), Wait(0.08),
               Lazy(_ -> MouseTo(relative_pos(block, (to, y)), duration)),
               LeftUp(), Wait(0.1)]
end

# The camera does not move before the flip, and that is deliberate: a mesh being
# orbited is not what this demo is about. What the raster half gets instead is
# the TOLERANCE — the one control whose effect is the mesh shader's, and a
# control that needs a still camera to be read at all.
events = Any[
    Wait(0.6),

    # [1] The tracer, converging on the opening view.
    Wait(0.9),

    # [2] Flip. Same scene, same camera, same lights — only the path changes,
    #     and the curve becomes chords.
    Lazy(_ -> MouseTo(blockcentre(ctrl.modebtn), 0.5)), LeftClick(), Wait(1.0),

    # [3] The tessellation tolerance, in the rasteriser: the key set refines
    #     under the drag and the chords converge on the curve the tracer just
    #     showed. This is the mesh stage, and it is the only thing moving.
    # Coarsest to about three quarters: 240 triangles to roughly 1200. The
    # track deliberately stops before the strokes touch and the surface fills
    # in solid — that shows the triangle count winning rather than the
    # tolerance working.
    dragslider(ctrl.tolsl, 0.02, 0.75)..., Wait(0.8),

    # [4] Back into the tracer, and STAY there — each material with the camera
    #     moving, because a still frame of metal or glass says very little.
    Lazy(_ -> MouseTo(blockcentre(ctrl.modebtn), 0.5)), LeftClick(), Wait(0.7),

    # [5] Zoom, with the wheel, in the tracer. The silhouette is SOLVED per ray
    #     rather than approximated, so it stays exact however close this gets —
    #     which is the claim the tolerance slider makes from the other side.
    Lazy(_ -> MouseTo(relative_pos(ctrl.lscene, (0.45, 0.52)), 0.4)),
    Scroll((0.0, 4.0); duration = 1.1), Wait(0.6),

    # Three materials, each with the camera moving — a still frame of metal or
    # glass says very little. COATED is where the flip left it, so the tour is
    # the two it has not shown yet and then back.
    pickmenu(ctrl.matmenu, 2)..., orbit(ctrl.lscene, 0.40, 0.64)..., Wait(0.8),  # GOLD
    pickmenu(ctrl.matmenu, 3)..., orbit(ctrl.lscene, 0.64, 0.36)..., Wait(1.0),  # GLASS
    pickmenu(ctrl.matmenu, 1)..., orbit(ctrl.lscene, 0.36, 0.55)..., Wait(0.9)   # COATED
]

# OUT of the package. A recording is an artifact of running the demo, not part
# of the example, and a 3 MB mp4 in `examples/` is 3 MB in everyone's clone
# forever. `ISUBD_DEMO_OUT` overrides it; the default is the working directory.
const OUT = get(ENV, "ISUBD_DEMO_OUT", joinpath(pwd(), "isubd_demo.mp4"))
mkpath(dirname(OUT))
FakeInteraction.interaction_record(fig, OUT, events; fps = 30, px_per_unit = 1)
println("Saved: ", OUT)
