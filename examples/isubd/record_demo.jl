# Needs `Makie/docs/fake_interaction.jl` next to this checkout.
using Mantle, Makie, RayMakie, GeometryBasics

include(joinpath(@__DIR__, "mantle_isubd.jl"))
include(joinpath(@__DIR__, "isubd_gui.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "Makie", "docs", "fake_interaction.jl"))
using .FakeInteraction: Wait, MouseTo, LeftClick, LeftDown, LeftUp, Lazy, Scroll, relative_pos

backend = Mantle.defaultbackend()
RayMakie.activate!(device = backend, accumulate = true, samples = 12, max_depth = 16)

cx, cy, cf = annulus()
fig, ctrl = isubd_gui(cx, cy, cf; width = 700, height = 700, surface = 1)

blockcentre(b) = relative_pos(b, (0.5, 0.5))

sliderpos(sl, frac) = relative_pos(sl, (clamp(frac, 0.02, 0.98), 0.5))

function dragslider(sl, from, to; duration = 1.5)
    return Any[Lazy(_ -> MouseTo(sliderpos(sl, from))), LeftDown(), Wait(0.15),
               Lazy(_ -> MouseTo(sliderpos(sl, to), duration)), Wait(0.15),
               LeftUp(), Wait(0.35)]
end

function pickmenu(menu, i; hold = 0.3, travel = 0.45)
    return Any[Lazy(_ -> MouseTo(relative_pos(menu, (0.5, 0.5)), travel)),
               LeftClick(), Wait(hold),
               Lazy(_ -> MouseTo(relative_pos(menu, (0.5, -(i - 0.5))), travel)),
               LeftClick(), Wait(hold)]
end

function orbit(block, from, to; duration = 1.3, y = 0.55)
    return Any[Lazy(_ -> MouseTo(relative_pos(block, (from, y)), 0.5)),
               LeftDown(), Wait(0.08),
               Lazy(_ -> MouseTo(relative_pos(block, (to, y)), duration)),
               LeftUp(), Wait(0.1)]
end

events = Any[
    Wait(0.6),

    Wait(0.9),

    Lazy(_ -> MouseTo(blockcentre(ctrl.modebtn), 0.5)), LeftClick(), Wait(1.0),

    dragslider(ctrl.tolsl, 0.02, 0.75)..., Wait(0.8),

    Lazy(_ -> MouseTo(blockcentre(ctrl.modebtn), 0.5)), LeftClick(), Wait(0.7),

    Lazy(_ -> MouseTo(relative_pos(ctrl.lscene, (0.45, 0.52)), 0.4)),
    Scroll((0.0, 4.0); duration = 1.1), Wait(0.6),

    pickmenu(ctrl.matmenu, 2)..., orbit(ctrl.lscene, 0.40, 0.64)..., Wait(0.8),
    pickmenu(ctrl.matmenu, 3)..., orbit(ctrl.lscene, 0.64, 0.36)..., Wait(1.0),
    pickmenu(ctrl.matmenu, 1)..., orbit(ctrl.lscene, 0.36, 0.55)..., Wait(0.9)
]

out = get(ENV, "ISUBD_DEMO_OUT", joinpath(pwd(), "isubd_demo.mp4"))
mkpath(dirname(out))
FakeInteraction.interaction_record(fig, out, events; fps = 30, px_per_unit = 1)
println("Saved: ", out)
