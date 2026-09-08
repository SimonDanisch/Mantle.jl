# Transient render targets.
#
#   render A  -> target 1        clear, draw the cloud
#   copy   A  -> raw 1           ColorAttachment -> CopySrc, derived
#   render B  -> target 2        target 1 is dead here, so 2 can have its bytes
#   copy   B  -> raw 2
#   combine   -> out             one compute pass reading both
#
# Two 1000x750 BGRA targets are 3 MB each. Their intervals do not overlap, so the
# placer should hand the second the first's bytes and peak should be one target,
# not two. That is the claim this measures.

import Mantle
using Lava, GeometryBasics, LinearAlgebra, KernelAbstractions
using ColorTypes: BGRA
using ColorTypes.FixedPointNumbers: N0f8
const M = Mantle

const W, H = 1000, 750
const NPX = W * H

isdefined(Main, :SCATTER) || include(joinpath(@__DIR__, "two_scatters.jl"))

# The copies pack rows and `blit!` reads a (height, width) matrix, so this is
# where the two layouts meet: read across a row, write down a column.
@kernel function combine!(dst, @Const(a), @Const(b), w::Int32, h::Int32)
    i, j = @index(Global, NTuple)
    k = ((j - 1) * w + i - 1) * 4
    @inbounds dst[(i - 1) * h + j] = (Vec4f(a[k + 3], a[k + 2], a[k + 1], 255f0) +
                                      Vec4f(b[k + 3], b[k + 2], b[k + 1], 255f0)) / 510f0
end

function build_targets(dev, n; alias = true, points = cloud(n))
    seed = points
    pos = M.Buffer(dev, seed)
    col = M.Buffer(dev, tint.(seed))
    siz = M.GPURef(dev, 2.0f0)
    # Read again every frame (`measure_targets` stores a new camera into it each
    # loop), so a `GPURef`: the draw holds its device address.
    mvp = M.GPURef(dev, camera(0.0f0))

    g = M.Graph(dev)
    scatter(p) = begin
        M.use(p, mvp; read = true)
        M.draw!(p, SCATTER, (M.Attribute(p, pos), M.Attribute(p, col),
                             M.Attribute(p, siz), mvp,
                             Int32(1), Int32(0)), pos)
    end

    a = M.Transient.Image(g, BGRA{N0f8}, (W, H))
    b = M.Transient.Image(g, BGRA{N0f8}, (W, H))
    ra = M.Transient.Buffer(g, UInt8, NPX * 4)
    rb = M.Transient.Buffer(g, UInt8, NPX * 4)
    out = M.Transient.Buffer(g, Vec4f, NPX)

    M.render!(scatter, g, "A", a => M.Clear((0.02f0, 0.02f0, 0.04f0, 1f0)))
    M.copy!(g, "read A", ra, a)
    M.render!(scatter, g, "B", b => M.Clear((0.04f0, 0.02f0, 0.02f0, 1f0)))
    M.copy!(g, "read B", rb, b)
    M.compute!(g, "combine") do p
        M.dispatch!(p, combine!, (M.use(p, out; write = true),
                                  M.use(p, ra; read = true),
                                  M.use(p, rb; read = true),
                                  Int32(W), Int32(H)), (W, H); group = (16, 16))
    end
    (; g, a, b, out, mvp, plan = M.record!(M.Plan(g; alias)))
end

"""Bytes an arena holds, per arena, so the image saving is visible on its own."""
function arenas(plan)
    E = Mantle
    Dict(string(nameof(typeof(E.arena(plan.graph.transients[first(s.indices)])))) => s.bytes
         for s in plan.slabs)
end

function measure_targets(frames = 300; n = 200_000)
    dev = M.Device(M.VulkanAPI())
    win = RenderWindow(W, H; title = "mantle: transient targets", vsync = false)
    s = build_targets(dev, n)
    loose = build_targets(dev, n; alias = false)
    bq = dev.bq

    times = Float64[]
    t0 = time()
    for _ in 1:frames
        isopen(win) || break
        Mantle.GLFW.PollEvents()
        tf = time()
        s.mvp[] = camera(0.4f0 * Float32(time() - t0))
        acquire_next_image!(win)
        M.run!(s.plan)
        blit!(bq, WindowTarget(win), M.storage(s.out))
        present_frame!(bq, win)
        Mantle.flush!(bq, dev.ctx.device)
        push!(times, time() - tf)
    end
    img = readback_window(win)
    close(win)

    steady = times[min(end, 11):end]
    q = sort(steady); m = length(q)
    (frames = length(times),
     arenas = arenas(s.plan),
     loose_arenas = arenas(loose.plan),
     peak_mb = round(M.peakbytes(s.plan) / 2^20, digits = 2),
     loose_mb = round(M.peakbytes(loose.plan) / 2^20, digits = 2),
     median_ms = round(1000 * q[m ÷ 2], digits = 3),
     fps = round(m / sum(steady), digits = 0),
     img = img)
end
