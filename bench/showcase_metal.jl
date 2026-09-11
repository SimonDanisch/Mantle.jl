# The Metal driver for `showcase_scene.jl`: the same twenty passes.
#
# Two things differ from the Vulkan driver, and only two: the device, and what
# the composite pass draws into. Everything that IS the demo — the scene, the
# shaders, the graph — is in the scene file, so this is not a second copy that
# can drift from the first.
#
#   SHOWCASE_WINDOW=1   open a window and present frames to it
#   otherwise           render offscreen and write a PNG
#
# The offscreen path is the default because it is the one that can be checked:
# it reads the frame back and saves it. Both drive the identical graph.

using Metal, GLFW
using FileIO, ImageIO
using ColorTypes: BGRA
include(joinpath(@__DIR__, "showcase_scene.jl"))

# Enough that the measurement is not dominated by whichever frame was odd.
const FRAMES = parse(Int, get(ENV, "SHOWCASE_FRAMES", "60"))
const WINDOWED = get(ENV, "SHOWCASE_WINDOW", "0") == "1"
const OUT = joinpath(@__DIR__, "showcase_metal.png")

dev = M.Device(M.MetalAPI())

if WINDOWED
    GLFW.WindowHint(GLFW.CLIENT_API, GLFW.NO_API)
    gw = GLFW.CreateWindow(W, H, "mantle: crystal field")
    # Sized so the FRAMEBUFFER is W by H, not the window.
    #
    # This scene is built at a fixed resolution: every transient is `W` by `H`
    # and the composite indexes `hdr[py * W + px + 1]` from `frag_coord`. On a
    # Retina display a window asked for in points gets a framebuffer twice that
    # in pixels, so the composite would run over a 2940-wide target while
    # striding a 1600-wide buffer, which reads far outside it. That is what a
    # smeared, distorted frame looked like.
    #
    # Dividing by the content scale makes the drawable pixel-exact instead:
    # nothing is scaled on the way to the screen, at the cost of a window that
    # is physically half the size on a Retina panel. Rendering at the
    # framebuffer size is the other answer, and it is four times the pixels.
    sx, sy = GLFW.GetWindowContentScale(gw)
    GLFW.SetWindowSize(gw, round(Int, W / sx), round(Int, H / sy))
    win = Mantle.attach!(Mantle.MetalWindow(BGRA{N0f8}, W, H; vsync = false), gw)
    # `attach!` adopts whatever the layer could actually be given, so this is a
    # real check and not a restatement of the line above.
    size(win) == (W, H) || error("the drawable is $(size(win)), the graph renders $((W, H))")
    S = showcase_plan(dev, graph -> M.Surface(graph, win))
else
    # A transient rather than a framebuffer: the composite is a pass IN the
    # graph, so its target has to be something the placer places.
    S = showcase_plan(dev, graph -> M.Transient.Image(graph, RGBA{N0f8}, (W, H)))
end

@info "kompiliert" round(M.peakbytes(S.plan) / 2^20; digits = 1) WINDOWED

# Warm up until nothing is compiling any more, and mean it: the FIRST frame
# spends ~250 ms in Julia's JIT inside the render passes alone, so a five-frame
# average measures the compiler. Recording settles at 0.02 ms per pass by the
# second frame; twenty is generous and cheap.
const WARMUP = 20
for i in 1:WARMUP
    showcase_frame!(S, Float32(i) / 60f0, 40f0, 16f0, false)
end
Metal.synchronize()

t0 = time()
for i in 1:FRAMES
    showcase_frame!(S, Float32(i) / 60f0, 40f0, 16f0, false)
end
Metal.synchronize()
dt = time() - t0

if WINDOWED
    @info "praesentiert" FRAMES round(1000dt / FRAMES; digits = 2) round(FRAMES / dt; digits = 1)
    GLFW.DestroyWindow(gw)
else
    save(OUT, permutedims(M.readback_target(S.screen)))
    @info "gerendert" FRAMES round(1000dt / FRAMES; digits = 2) round(FRAMES / dt; digits = 1) OUT
end
@info "sichtbar" Int(Array(S.counter)[1]) NINST
