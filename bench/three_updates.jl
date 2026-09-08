import Mantle
using GeometryBasics, LinearAlgebra, KernelAbstractions, ComputePipeline
using Lava   # only for the shader vocabulary Mantle does not wrap yet
M = Mantle #the attribute blocks at the bottom land on the next frame.

begin # ── packages: their own block, so the macros below are expandable ──────

    W, H = 1200, 800

    # A scalar attribute is stride 0, so it reads element 1 for every point —
    # scalar and per-element colour are one shader, not two.
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
    scatter_fragment(inputs) = inputs.color

    SCATTER = Rasterizer(vertex = scatter_vertex, fragment = scatter_fragment,
                        varyings = (color = Vec4f,), topology = PointList(),
                        blend = Additive(), cull = NoCull(), depth = DepthOff())

    # One step of the simulation, run on three different backends below.
    @kernel function advect!(pos, vel, dt, radius::Float32)
        i = @index(Global)
        @inbounds begin
            v = vel[i]
            p = pos[i] + v * dt[1]
            r = sqrt(p[1] * p[1] + p[2] * p[2] + p[3] * p[3])
            if r > radius
                n = p / r
                p = n * radius
                d = v[1] * n[1] + v[2] * n[2] + v[3] * n[3]
                vel[i] = v - 2.0f0 * d * n
            end
            pos[i] = p
        end
    end

    cloud(n) = [normalize(Vec3f(randn(Float32), randn(Float32), randn(Float32))) *
                (1.0f0 + 0.05f0 * randn(Float32)) for _ in 1:n]
    drift(n) = [0.35f0 * Vec3f(randn(Float32), randn(Float32), randn(Float32)) for _ in 1:n]
    tint(p) = 0.06f0 * Vec4f(0.45f0 + 0.55f0 * p[2], 0.55f0, 1.0f0 - 0.35f0 * p[2], 1.0f0)

    # The three clouds sit side by side; each draw gets its own matrix.
    function camera(angle, dx)
        eye = Vec3f(7.5f0 * cos(angle), 2.2f0, 7.5f0 * sin(angle))
        f = normalize(-eye); r = normalize(cross(f, Vec3f(0, 1, 0))); u = cross(r, f)
        view = Mat4f(r[1], u[1], -f[1], 0, r[2], u[2], -f[2], 0, r[3], u[3], -f[3], 0,
                    -dot(r, eye), -dot(u, eye), dot(f, eye), 1)
        tn = tan(60.0f0 * Float32(pi) / 360.0f0)
        proj = Mat4f(1 / (Float32(W / H) * tn), 0, 0, 0, 0, -1 / tn, 0, 0,
                    0, 0, 100.0f0 / (0.1f0 - 100.0f0), -1,
                    0, 0, (0.1f0 * 100.0f0) / (0.1f0 - 100.0f0), 0)
        proj * view * Mat4f(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, dx, 0, 0, 1)
    end
    # GLMakie holds a RenderObject per plot; here the device side is buffers plus
    # the positions in the schedule where new data for them lands.
    device_attribute(dev, v::AbstractVector) = M.Buffer(dev, v)
    device_attribute(dev, x) = M.GPURef(dev, x)
    # The write side of the same choice: a per-element attribute is a
    # whole-buffer store, a scalar one lands through the ref.
    landvalue!(b::M.Buffer, data) = (b[:] = data)
    landvalue!(r::M.GPURef, data) = (r[] = data)

    function build_plot(dev, g, args)
        buffers = Dict(k => device_attribute(dev, v) for (k, v) in pairs(args))
        (; buffers, fired = Dict(k => 0 for k in keys(args)))
    end

    # This is GLMakie's update_robjs!: walk the args, skip what did not change.
    function update_plot!(plot, args, changed)
        for name in keys(args)
            changed[name] || continue
            # A host array is staged and copied, a device array is copied device
            # to device and never sees the host, a scalar goes in place. The value
            # decides, not a flag.
            landvalue!(plot.buffers[name], args[name])
            plot.fired[name] += 1
        end
        return plot
    end

    function scatter(dev, g; attributes...)
        attr = ComputeGraph()
        for (k, v) in attributes
            add_input!(attr, k, v)
        end
        register_computation!(attr, collect(keys(attributes)), [:plot]) do args, changed, last
            isnothing(last) ? (build_plot(dev, g, args),) : (update_plot!(last.plot, args, changed),)
        end
        attr[:plot][]   # build the device side now, so the passes below can name it
        return attr
    end

    buffers(attr) = attr[:plot][].buffers
    fired(attr) = attr[:plot][].fired

    # GLMakie resolves its render object node per frame; this is that resolve.
    handover!(attr) = (attr[:plot][]; nothing)

    function draw_scatter!(p, attr, mvp)
        M.use(p, mvp; read = true)
        b = buffers(attr)
        col, siz = M.Attribute(p, b[:color]), M.Attribute(p, b[:markersize])
        args = (M.Attribute(p, b[:positions]), col, siz, mvp, M.stride(col), M.stride(siz))
        M.draw!(p, SCATTER, args, b[:positions])
    end
    n = 100_000
    dev = M.Device()
    win = M.Window(W, H; title = "mantle: three ways to update")
    graph = M.Graph(dev)
    screen = M.Surface(graph, win)
    # `dt` stays a plain host `Ref`: the raw launches below read it directly with
    # `dt[]`, which a `GPURef` cannot do (there is no `getindex` — reading a
    # device value is a download). `dtref` is what the in-graph sim's dispatch
    # holds instead, since a dispatch argument may not be a `Ref`.
    dt = Ref(1.0f0 / 60)
    dtref = M.GPURef(dev, dt[])

    a, b, c = cloud(n), cloud(n), cloud(n)
    mk(x) = scatter(dev, graph; positions = x, color = tint.(x), markersize = 2.0f0)
    sim, gpu, cpu = mk(a), mk(b), mk(c)

    # left: the graph owns the simulation, so this plot never hands anything over
    simvel = M.Buffer(dev, drift(n))
    M.compute!(graph, "advect") do p
        x = M.use(p, buffers(sim)[:positions]; read = true, write = true)
        v = M.use(p, simvel; read = true, write = true)
        M.use(p, dtref; read = true)
        M.dispatch!(p, advect!, (x, v, dtref, 1.4f0), n)
    end

    # middle: the same kernel on the device, outside the graph
    gpupos, gpuvel = M.Buffer(dev, copy(b)), M.Buffer(dev, drift(n))
    # right: the same kernel again, on the CPU backend over plain arrays
    cpupos, cpuvel = copy(c), drift(n)

    mvps = (M.GPURef(dev, camera(0.0f0, -2.2f0)), M.GPURef(dev, camera(0.0f0, 0.0f0)),
            M.GPURef(dev, camera(0.0f0, 2.2f0)))
    M.render!(graph, "three", screen => M.Clear((0.02f0, 0.02f0, 0.04f0, 1.0f0))) do p
        for (plt, mvp) in zip((sim, gpu, cpu), mvps)
            draw_scatter!(p, plt, mvp)
        end
    end

    plan = M.Plan(graph; profile = true)
    @assert M.npipelines(plan) == 1   # three plots, one shader
end

begin # ── the render loop, async like GLMakie's, so evaling keeps working ────
    simulate = true
    stop = Threads.Atomic{Bool}(false)
    frames = Ref(0)   # a Ref, not a plain global: the loop body is a closure
    # @async, not @spawn: Vulkan recording is single-threaded, and a sticky task
    # keeps it on the thread that owns the queue.
    rendertask = @async begin
        t0 = time()
        while isopen(win) && !stop[]
            t = Float32(time() - t0)
            for (mvp, dx) in zip(mvps, (-2.2f0, 0.0f0, 2.2f0))
                mvp[] = camera(0.3f0 * t, dx)
            end
            if simulate
                advect!(M.backend(dev))(M.storage(gpupos), M.storage(gpuvel), dt[], 1.4f0; ndrange = n)
                update!(gpu; positions = M.storage(gpupos))     # device array, no host round trip
                advect!(CPU())(cpupos, cpuvel, dt[], 1.4f0; ndrange = n)
                KernelAbstractions.synchronize(CPU())
                update!(cpu; positions = cpupos)                # host array, staged once
            end
            handover!(sim); handover!(gpu); handover!(cpu)
            M.run!(plan)
            frames[] += 1
            sleep(0.001)   # the yield that lets the REPL run between frames
        end
    end
    errormonitor(rendertask)
end

# ── eval any of these while the window is up ─────────────────────────────────

update!(cpu; color = fill(Vec4f(0.2, 0.02, 0.02, 1), n))   # one uniform colour

update!(cpu; markersize = 6.0f0)                            # a scalar attribute

simulate = false                                            # freeze, so the next one sticks

update!(cpu; positions = cloud(n))                          # a new host cloud

update!(gpu; positions = M.storage(M.Buffer(dev, cloud(n))))  # a new device cloud

dt[] = 1.0f0 / 240; dtref[] = dt[]           # dtref is what the in-graph sim reads

fired(cpu)      # per attribute, how often it was handed over: colour once, positions per frame

M.timings(plan) # per pass, host and gpu milliseconds

begin # ── stop ──────────────────────────────────────────────────────────────
    stop[] = true
    wait(rendertask)
    close(win)
end
