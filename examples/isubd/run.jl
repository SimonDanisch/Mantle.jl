using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Mantle, Makie, RayMakie

include(joinpath(@__DIR__, "mantle_isubd.jl"))
include(joinpath(@__DIR__, "isubd_gui.jl"))

backend = Mantle.defaultbackend()
RayMakie.activate!(device = backend, accumulate = true, samples = 12, max_depth = 16)

cx, cy, cf = annulus()
fig, ctrl = isubd_gui(cx, cy, cf; width = 700, height = 700)
display(fig)

warmup!(fig, ctrl)
