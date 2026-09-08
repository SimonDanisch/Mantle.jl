# A multi-pass graph with transients, which is what the allocator exists for.
#
#   render  points into an offscreen target
#   copy    that target into a transient buffer          (ColorAttachment -> CopySrc)
#   blur x3 a chain of compute passes, each writing a new transient
#   blit    the last one to the window
#
# The chain is what makes placement measurable: stage k reads t[k] and writes
# t[k+1], so only adjacent transients are ever live together and every other one
# can share bytes. Peak should be about two buffers, not five.

import Mantle
using Lava, GeometryBasics, LinearAlgebra, KernelAbstractions
const M = Mantle
# The backend, for the framebuffer readback this bench measures with.
const MVE = Base.get_extension(Mantle, :MantleVulkanExt)

const W, H = 1000, 750
const NPX = W * H
const STAGES = 3

# SCATTER, cloud, tint, camera. Guarded because the test file includes it too,
# and a second include silently redefines build/measure.
isdefined(Main, :SCATTER) || include(joinpath(@__DIR__, "two_scatters.jl"))

# The two layouts meet here: an image copy packs rows, and `blit!` reads a
# (height, width) matrix, so the transposition happens once, where the pixels
# arrive, rather than being carried down the chain and shearing the picture.
@kernel function unpack!(dst, @Const(src), w::Int32, h::Int32)
    i, j = @index(Global, NTuple)
    b = ((j - 1) * w + i - 1) * 4
    @inbounds dst[(i - 1) * h + j] = Vec4f(src[b + 3], src[b + 2], src[b + 1], 255f0) / 255f0
end

@kernel function blur!(dst, @Const(src), w::Int32, h::Int32)
    i, j = @index(Global, NTuple)
    acc = zero(Vec4f)
    @inbounds for dj in -1:1, di in -1:1
        acc += src[(clamp(i + di, 1, w) - 1) * h + clamp(j + dj, 1, h)]
    end
    @inbounds dst[(i - 1) * h + j] = acc / 9f0
end

function build_chain(dev, win, n)
    seed = cloud(n)
    pos = M.Buffer(dev, seed)
    col = M.Buffer(dev, tint.(seed))
    siz = M.GPURef(dev, 2.0f0)
    # Read again every frame (`measure_chain` stores a new camera into it each
    # loop), so a `GPURef`: the draw holds its device address.
    mvp = M.GPURef(dev, camera(0.0f0))

    fb = M.Framebuffer(M.backend(dev), W, H; depth = false)   # the portable spelling; the type is the backend's

    g = M.Graph(dev)

    M.render!(g, "points", fb => M.Clear((0.02f0, 0.02f0, 0.04f0, 1f0))) do p
        M.use(p, mvp; read = true)
        M.draw!(p, SCATTER, (M.Attribute(p, pos), M.Attribute(p, col),
                             M.Attribute(p, siz), mvp,
                             Int32(1), Int32(0)), pos)
    end

    raw = M.Transient.Buffer(g, UInt8, NPX * 4)
    stage = [M.Transient.Buffer(g, Vec4f, NPX) for _ in 1:(STAGES + 1)]

    M.compute!(g, "unpack") do p
        src = M.use(p, raw; read = true)
        dst = M.use(p, stage[1]; write = true)
        M.dispatch!(p, unpack!, (dst, src, Int32(W), Int32(H)), (W, H); group = (16, 16))
    end

    for k in 1:STAGES
        M.compute!(g, "blur $k") do p
            src = M.use(p, stage[k]; read = true)
            dst = M.use(p, stage[k + 1]; write = true)
            M.dispatch!(p, blur!, (dst, src, Int32(W), Int32(H)), (W, H); group = (16, 16))
        end
    end

    (; g, fb, raw, out = stage[end], mvp, plan = M.record!(M.Plan(g)))
end

function measure_chain(frames = 300; n = 200_000)
    dev = M.Device(M.VulkanAPI())
    win = RenderWindow(W, H; title = "mantle: transient chain", vsync = false)
    s = build_chain(dev, win, n)
    bq = dev.bq

    times = Float64[]
    t0 = time()
    for _ in 1:frames
        isopen(win) || break
        Mantle.GLFW.PollEvents()
        tf = time()
        s.mvp[] = camera(0.4f0 * Float32(time() - t0))
        acquire_next_image!(win)
        M.run!(s.plan)                               # render + compute chain
        MVE.copy_framebuffer!(M.storage(s.raw), s.fb) # ColorAttachment -> CopySrc
        blit!(bq, WindowTarget(win), M.storage(s.out))
        present_frame!(bq, win, MVE.oneshot(bq) do e
            MVE.presentready!(e, win)
        end)
        Mantle.flush!(bq, dev.ctx.device)
        push!(times, time() - tf)
    end
    img = readback_window(win)
    close(win)

    steady = times[min(end, 11):end]
    q = sort(steady); m = length(q)
    (frames = length(times),
     peak_mb = round(M.peakbytes(s.plan) / 2^20, digits = 2),
     naive_mb = round(Mantle.naivebytes(s.plan) / 2^20, digits = 2),
     median_ms = round(1000 * q[m ÷ 2], digits = 3),
     fps = round(m / sum(steady), digits = 0),
     img = img)
end

