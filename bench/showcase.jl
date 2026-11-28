# The Vulkan driver for `showcase_scene.jl`: a device, a window, and the loop.
#
# Everything that IS the demo — the scene, the shaders, the twenty passes — is
# in the scene file, so this and `showcase_metal.jl` cannot drift apart. What
# differs is three lines: which device, what the composite draws into, and
# whether there is a window to keep open.

using Lava                         # loads MantleVulkanExt
include(joinpath(@__DIR__, "showcase_scene.jl"))

dev = M.Device(M.VulkanAPI())
win = M.Window(W, H; title = "mantle: crystal field", vsync = false)
S = showcase_plan(dev, graph -> M.Surface(graph, win))


begin # ── the render loop ───────────────────────────────────────────────────────
    orbit = 40f0
    height = 16f0
    speed = 1f0
    freeze_cull = false
    stop = Threads.Atomic{Bool}(false)
    frames = Ref(0)
    nvisible = Ref(0)
    render = errormonitor(@async begin
        t, last = 0f0, time()
        while isopen(win) && !stop[]
            now = time(); t += speed * Float32(now - last); last = now
            showcase_frame!(S, t, orbit, height, freeze_cull)
            frames[] += 1
            frames[] % 30 == 0 && (nvisible[] = Int(Array(S.counter)[1]))
            sleep(0.001)
        end
    end)
end

# ── eval any of these while the window is up ─────────────────────────────────
#
# Every knob is followed by the line that puts it back, and those lines hold the
# defaults above rather than whatever they happened to be while this was written.

nvisible[]               # instances drawn, of NINST

S.sbiasref[] = 10f0        # every depth test passes: the same frame with no shadows
S.sbiasref[] = 0.0002f0    # too little to cover a PCF tap: acne on the grazing faces
S.sbiasref[] = 0.004f0     # back

S.aoamb[] = 0f0            # the same frame with no ambient occlusion
S.aoamb[] = 6f0            # far too much, which is the way to see where it acts
S.aoamb[] = 2.6f0          # back

S.aoradius[] = 1.5f0       # tight crevices only, rather than whole clusters
S.aoradius[] = 4.5f0       # back

S.bloomamt[] = 2.5f0       # more glow
S.bloomamt[] = 0.30f0      # back

S.threshold[] = 0.4f0      # what counts as bright
S.threshold[] = 1.6f0      # back

S.emitref[] = 25f0         # the sources themselves, much hotter
S.emitref[] = 9f0          # back

S.exposure[] = 2f0
S.exposure[] = 0.85f0      # back

S.fogref[] = 0.02f0        # thicker air
S.fogref[] = 0.0035f0      # back

S.suncolref[] = Vec4f(0.55, 0.72, 1.0, 2.5)   # a cold moon instead of a warm sun
S.suncolref[] = Vec4f(1.0, 0.55, 0.28, 6.5)   # back

S.sunref[] = Vec4f(normalize(Vec3f(-0.62, 0.045, 0.35))..., 0)   # lower, longer shadows
S.sunref[] = Vec4f(normalize(Vec3f(-0.62, 0.22, 0.35))..., 0)    # back

freeze_cull = true       # the frustum stops following the camera
freeze_cull = false

speed = 0f0              # hold still

M.timings(S.plan)          # per pass; a sum down the column is not the frame time

round(M.peakbytes(S.plan) / 2^20, digits = 1)   # what the transients cost together

begin # ── stop ──────────────────────────────────────────────────────────────
    stop[] = true
    wait(render)
    close(win)
end
