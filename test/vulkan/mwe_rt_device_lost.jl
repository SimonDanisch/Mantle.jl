# MWE: RT dispatch DEVICE_LOST after multiple renders with GC pressure
#
# The crash in the crown scene happens at 1600x1600 with 5 material types
# after zooming. This MWE simulates that by:
# - Using a larger resolution
# - Running many render iterations with camera changes
# - Adding GC pressure between iterations
#
# Run: julia --project=. dev/Lava/test/mwe_rt_DEVICE_LOST.jl

using Hikari, Lava, GeometryBasics
import KernelAbstractions as KA
import Adapt

backend = Mantle.LavaBackend()

# Use a test scene with multiple material types
sf = "dev/Hikari/test/pbrt/scenes/mat_conductor_gold_light_point.pbrt"
r = Hikari.load_pbrt(sf; backend=backend, samples=1, max_depth=5)

# Larger resolution to match crown scene
film = Adapt.adapt(backend, Hikari.Film(Point2f(800, 800)))
cam = r.camera

vp = Hikari.VolPath(samples=1, max_depth=5, hw_accel=true)

println("Phase 1: Initial renders at 800x800...")
for i in 1:5
    vp(r.scene, film, cam)
    adapted = Adapt.adapt(backend, r.scene)
    Hikari.fill_aux_buffers!(film, adapted, cam)
    GC.gc()  # Force GC to stress-test buffer lifetimes
    println("  sample $i OK")
end

println("\nPhase 2: Zoom + re-render (many iterations with GC)...")
for zoom in 1:10
    # New camera (zoom in progressively)
    dist = 3.0f0 - zoom * 0.15f0
    film_new = Adapt.adapt(backend, Hikari.Film(Point2f(800, 800)))
    cam_new = Hikari.PerspectiveCamera(
        Point3f(0, -dist, 1.5), Point3f(0, 0, 0.5), film_new; fov=45f0)

    Hikari.clear!(film_new)
    Hikari.clear!(vp)

    for s in 1:3
        vp(r.scene, film_new, cam_new)
        adapted = Adapt.adapt(backend, r.scene)
        Hikari.fill_aux_buffers!(film_new, adapted, cam_new)
    end
    GC.gc()
    println("  zoom $zoom (dist=$dist) OK")
end

println("\nALL OK — no DEVICE_LOST")
