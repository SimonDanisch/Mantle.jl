import Mantle
using GeometryBasics, LinearAlgebra, KernelAbstractions

# `using Lava` is gone (phase 2.8). This file is a benchmark scene: shaders, a
# graph and a window, all of it portable — it named a backend only because the
# shader intrinsics used to come from one, and they come from Mantle now.
#
# The note that stood here said Scalar, Buffer and draw! were each exported by
# more than one of Mantle, Lava and StaticArrays and had to be qualified. With
# the backend out of the picture that clash is gone too, and the observation it
# ended on — "a name that cannot be `using`-ed alongside the backend it drives
# is a poor name" — stopped being a problem rather than being solved.
const M = Mantle

const W, H = 1000, 750

function scatter_vertex(pos::AbstractVector{Vec3f}, col::AbstractVector{Vec4f},
                        siz::AbstractVector{Float32}, mvp::AbstractVector{Mat4f},
                        scol::Int32, ssiz::Int32)
    i = vertex_index()
    @inbounds p = pos[i]
    @inbounds c = col[1 + scol * (i - Int32(1))]
    @inbounds s = siz[1 + ssiz * (i - Int32(1))]
    @inbounds m = mvp[1]
    set_point_size!(s)
    return (position = m * Vec4f(p[1], p[2], p[3], 1.0f0), color = c)
end

const SCATTER = Rasterizer(
    vertex = scatter_vertex, fragment = inputs -> inputs.color,
    varyings = (color = Vec4f,),
    topology = PointList(), blend = Additive(), cull = NoCull(), depth = DepthOff(),
)

cloud(n) = [normalize(Vec3f(randn(Float32), randn(Float32), randn(Float32))) *
            (1.0f0 + 0.05f0 * randn(Float32)) for _ in 1:n]
tint(p) = 0.06f0 * Vec4f(0.45f0 + 0.55f0 * p[2], 0.55f0, 1.0f0 - 0.35f0 * p[2], 1.0f0)

function camera(angle)
    eye = Vec3f(3.2f0 * cos(angle), 1.2f0, 3.2f0 * sin(angle))
    f = normalize(-eye); r = normalize(cross(f, Vec3f(0, 1, 0))); u = cross(r, f)
    view = Mat4f(r[1], u[1], -f[1], 0, r[2], u[2], -f[2], 0, r[3], u[3], -f[3], 0,
                 -dot(r, eye), -dot(u, eye), dot(f, eye), 1)
    tn = tan(60.0f0 * Float32(pi) / 360.0f0)
    proj = Mat4f(1 / (Float32(W / H) * tn), 0, 0, 0, 0, -1 / tn, 0, 0,
                 0, 0, 100.0f0 / (0.1f0 - 100.0f0), -1,
                 0, 0, (0.1f0 * 100.0f0) / (0.1f0 - 100.0f0), 0)
    proj * view
end

drift(n) = [0.35f0 * Vec3f(randn(Float32), randn(Float32), randn(Float32)) for _ in 1:n]

@kernel function advect!(pos, vel, dt, radius::Float32)
    i = @index(Global)
    @inbounds begin
        v = vel[i]
        p = pos[i] + v * dt[1]
        r = sqrt(p[1] * p[1] + p[2] * p[2] + p[3] * p[3])
        # Reflect rather than clamp: clamping piles every escapee onto the shell
        # and the cloud slowly collapses into a sphere of stationary points.
        if r > radius
            n = p / r
            p = n * radius
            d = v[1] * n[1] + v[2] * n[2] + v[3] * n[3]
            vel[i] = v - 2f0 * d * n
        end
        pos[i] = p
    end
end

struct Scatter{P,C,S}
    positions::P
    color::C
    markersize::S
end

function bind(p, s::Scatter, mvp)
    M.use(p, mvp; read = true)
    pos = M.Attribute(p, s.positions)
    col = M.Attribute(p, s.color)
    siz = M.Attribute(p, s.markersize)
    (pos, col, siz, mvp, M.stride(col), M.stride(siz))
end

function build(dev, win, na, nb)
    seed_a, seed_b = cloud(na), cloud(nb)

    # A: per-point colour, one shared size.
    a = Scatter(M.Buffer(dev, seed_a), M.Buffer(dev, tint.(seed_a)), M.GPURef(dev, 2.0f0))
    # B: one shared colour, per-point size.
    b = Scatter(M.Buffer(dev, seed_b), M.GPURef(dev, Vec4f(0.10, 0.04, 0.01, 1)),
                M.Buffer(dev, 1.0f0 .+ 3.0f0 .* rand(Float32, nb)))

    a_vel = M.Buffer(dev, drift(na))
    b_vel = M.Buffer(dev, drift(nb))

    # Both are read again every frame (`demo` stores a new camera and frame
    # dt into them each loop), so both are `GPURef`s: the dispatches hold their
    # device address, and one store writes the pending value in place.
    mvp = M.GPURef(dev, camera(0.0f0))
    dt = M.GPURef(dev, 1.0f0 / 60)
    g = M.Graph(dev)
    screen = M.Surface(g, win)

    M.compute!(g, "advect") do p
        M.use(p, dt; read = true)
        for (s, vel) in ((a, a_vel), (b, b_vel))
            x = M.use(p, s.positions; read = true, write = true)
            v = M.use(p, vel; read = true, write = true)
            M.dispatch!(p, advect!, (x, v, dt, 1.6f0), length(s.positions))
        end
    end

    M.render!(g, "scatters", screen => M.Clear((0.02f0, 0.02f0, 0.04f0, 1.0f0))) do p
        for s in (a, b)
            M.draw!(p, SCATTER, bind(p, s, mvp), s.positions)
        end
    end
    (; a, b, mvp, dt, plan = M.Plan(g))
end

"""
    demo(; na, nb)

Open the window and run until it is closed. This is `docs/api/03-makie.jl` as far
as Mantle implements it: two scatters with swapped scalar and vector attributes,
one graph, one compiled plan, one pipeline between them.

`measure` is the same frame loop with a frame count and a stopwatch.
"""
function demo(; na = 200_000, nb = 100_000)
    dev = M.Device(M.defaultbackend())
    win = M.Window(W, H; title = "mantle: two scatters")
    s = build(dev, win, na, nb)
    @assert M.npipelines(s.plan) == 1 "a scalar and a vector attribute must not fork the shader"

    t0 = time(); frames = 0; last = t0; prev = t0
    while isopen(win)
        now = time()
        s.dt[] = Float32(min(now - prev, 0.05))     # read at submit; no rebuild
        prev = now
        s.mvp[] = camera(0.4f0 * Float32(now - t0))
        M.run!(s.plan)
        frames += 1
        if time() - last > 1
            @info "$(na + nb) points, $(round(Int, frames / (time() - last))) fps"
            frames = 0; last = time()
        end
    end
    close(win)
end

function measure(frames = 600; na = 200_000, nb = 100_000, sync = true)
    dev = M.Device(M.defaultbackend())
    win = M.Window(W, H; title = "mantle: two scatters")
    s = build(dev, win, na, nb)

    times = Float64[]; t0 = time()
    for _ in 1:frames
        isopen(win) || break
        tf = time()
        s.mvp[] = camera(0.4f0 * Float32(time() - t0))
        M.run!(s.plan)
        sync && KernelAbstractions.synchronize(M.backend(dev))
        push!(times, time() - tf)
    end
    img = M.screenshot(win)
    close(win)
    steady = times[min(end, 11):end]          # the first frames compile the shader
    q = sort(steady); n = length(q)
    (frames = length(times), pipelines = M.npipelines(s.plan),
     median_ms = round(1000 * q[n ÷ 2], digits = 3),
     p95_ms = round(1000 * q[max(1, 19n ÷ 20)], digits = 3),
     fps = round(n / sum(steady), digits = 0), img = img)
end
