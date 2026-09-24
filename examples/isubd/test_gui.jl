# The demo's two controls, driven the way a click drives them: the renderer
# toggle and the material menu, in every combination, headless.
#
# Every bug this pins was found by a person clicking in the window, and every
# one of them left the window up and frozen, looking like a hang:
#
#   GOLD, traced       `equal_area_sphere_to_square` took `sqrt(1 - |z|)` of a
#                      direction a hair past unit length; a mirror sends nearly
#                      every path to the environment. (Pinned exactly in
#                      Hikari's `test_environment_map_domain.jl`; here it is the
#                      demo that must survive it.)
#   RASTER, any        the graph asked a MeshPipeline for `.vertex`; then the
#                      mesh-output intrinsics were undeclared to the access walk;
#                      then the overlay's mesh stage took `warp` as Float64 and
#                      its fragment stage rounded with the checked `Int(round(…))`
#                      — neither of which a GPU stage without Float64 or
#                      `kernel_state` can compile.
#
# Run in the VulkanDev environment, like `test_slide.jl`:
#
#     julia> include("dev/Mantle/examples/isubd/test_gui.jl")

using Test
using Mantle, Makie, RayMakie, GeometryBasics, Hikari, Colors

backend = Mantle.defaultbackend()
RayMakie.activate!(device = backend, accumulate = false, samples = 2,
                   integrator = Hikari.VolPath(max_depth = 8, hw_accel = true))

module IsubdGuiTest
include(joinpath(@__DIR__, "mantle_isubd.jl"))
include(joinpath(@__DIR__, "isubd_gui.jl"))
end

"Luminance of every pixel, and whether any is NaN."
function lum(img)
    g = Float32[Float32(Gray(RGB{Float32}(red(p), green(p), blue(p)))) for p in img]
    return g, any(isnan, g)
end

@testset "isubd demo: renderer toggle x material menu" begin
    if !Mantle.supports_mesh_pipeline(backend)
        @info "no mesh pipeline on this backend: RASTER mode cannot draw; skipping"
        return
    end
    cx, cy, cf = IsubdGuiTest.annulus()
    fig, ctrl = IsubdGuiTest.isubd_gui(cx, cy, cf; width = 240, height = 240, surface = 1)
    first_img = Makie.colorbuffer(fig)          # creates the screen, traced
    screen = Makie.getscreen(fig.scene)
    @test screen isa RayMakie.Screen
    @test !screen.rasterize
    click!() = (ctrl.modebtn.clicks[] = ctrl.modebtn.clicks[] + 1)

    for (name, _...) in IsubdGuiTest.fem_surfaces
        ctrl.matmenu.selection[] = name

        traced = Makie.colorbuffer(screen)
        @test !screen.rasterize
        gt, nan_t = lum(traced)
        @test !nan_t
        @test maximum(gt) > 0.05                 # it drew something

        click!()                                 # -> RASTER, as the button does
        @test screen.rasterize
        raster = Makie.colorbuffer(screen)
        gr, nan_r = lum(raster)
        @test !nan_r
        @test size(raster) == size(traced)
        # The two renderers draw the same scene differently: the raster half
        # strokes its subdivision edges and the tracer shows the material. An
        # identical image would mean the toggle changed nothing.
        @test raster != traced

        click!()                                 # -> back to traced
        @test !screen.rasterize
        again = Makie.colorbuffer(screen)
        @test !lum(again)[2]

        # The TOLERANCE slider while traced, then RASTER. The slider hands the
        # plot a new `FEMMaterial`, and the traced path updated the scene's
        # material with the wrapper instead of the surface Hikari had stored for
        # it: a `MethodError` that the frame's poll caught and logged (the traced
        # view silently stopped updating), and that the switch to RASTER then
        # raised again outside any catch, killing the render loop. Found by
        # clicking RASTER in the live window with GLASS selected.
        empty!(RayMakie.POLL_ERROR_LOGGED)
        ctrl.tolsl.value[] = ctrl.tolsl.value[] - 0.3
        after_slider = Makie.colorbuffer(screen)
        @test isempty(RayMakie.POLL_ERROR_LOGGED)
        @test !lum(after_slider)[2]
        click!()                                 # -> RASTER, material just changed
        @test screen.rasterize
        @test !lum(Makie.colorbuffer(screen))[2]
        click!()                                 # -> back to traced
        @test isempty(RayMakie.POLL_ERROR_LOGGED)
    end
    close(screen)
end
