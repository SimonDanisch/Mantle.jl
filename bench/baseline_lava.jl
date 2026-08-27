# Baseline: Lava's own path, points straight to the window. Establishes that the
# display and hardware work, and gives the number Mantle has to match.
using Lava, GeometryBasics, LinearAlgebra

const W, H = 1000, 750
const N = 300_000

function point_vertex(pos::AbstractVector{Vec3f}, col::AbstractVector{Vec4f},
                      siz::AbstractVector{Float32}, mvp::Mat4f)
    i = vertex_index()
    @inbounds p = pos[i]
    @inbounds c = col[i]
    @inbounds s = siz[1]
    set_point_size!(s)
    return (position = mvp * Vec4f(p[1], p[2], p[3], 1.0f0), color = c)
end

const POINTS = Rasterizer(
    vertex = point_vertex, fragment = inputs -> inputs.color,
    varyings = (color = Vec4f,),
    topology = PointList(), blend = Additive(), cull = NoCull(), depth = DepthOff(),
)

function camera(angle)
    eye = Vec3f(3.2f0 * cos(angle), 1.2f0, 3.2f0 * sin(angle))
    f = normalize(-eye); r = normalize(cross(f, Vec3f(0, 1, 0))); u = cross(r, f)
    view = Mat4f(r[1], u[1], -f[1], 0, r[2], u[2], -f[2], 0, r[3], u[3], -f[3], 0,
                 -dot(r, eye), -dot(u, eye), dot(f, eye), 1)
    tn = tan(60.0f0 * Float32(pi) / 360.0f0)
    proj = Mat4f(1 / (Float32(W/H) * tn), 0, 0, 0, 0, -1 / tn, 0, 0,
                 0, 0, 100.0f0 / (0.1f0 - 100.0f0), -1,
                 0, 0, (0.1f0 * 100.0f0) / (0.1f0 - 100.0f0), 0)
    proj * view
end

function run(frames = 600)
    pts = [normalize(Vec3f(randn(Float32), randn(Float32), randn(Float32))) *
           (1.0f0 + 0.05f0 * randn(Float32)) for _ in 1:N]
    cols = [0.06f0 * Vec4f(0.45f0 + 0.55f0*p[2], 0.55f0, 1.0f0 - 0.35f0*p[2], 1.0f0) for p in pts]

    gpos = LavaArray(pts); gcol = LavaArray(cols); gsiz = LavaArray(Float32[2.0f0])
    win = RenderWindow(W, H; title = "baseline", vsync = false)
    bq = Mantle.vk_context().default_bq

    times = Float64[]
    t0 = time()
    for k in 1:frames
        isopen(win) || break
        Mantle.GLFW.PollEvents()
        tf = time()
        acquire_next_image!(win)
        draw!(bq, POINTS, WindowTarget(win), N;
              args = (gpos, gcol, gsiz, camera(0.4f0 * Float32(time() - t0))),
              clear_color = (0.02f0, 0.02f0, 0.04f0, 1.0f0))
        present_frame!(bq, win)
        push!(times, time() - tf)
    end
    close(win)
    sort!(times)
    n = length(times)
    (n = n, median_ms = 1000 * times[n ÷ 2], p95_ms = 1000 * times[max(1, 19n ÷ 20)],
     fps = n / sum(times))
end
