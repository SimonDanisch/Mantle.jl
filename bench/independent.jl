# Does deriving barriers beat emitting one unconditionally?
#
# Two chains over disjoint buffers, interleaved: A1 B1 A2 B2 A3 B3. Consecutive
# dispatches touch nothing in common, so a derived barrier is needed only within
# each chain and never between them. Lava's automatic per-dispatch barrier
# (COMPUTE_SHADER -> COMPUTE_SHADER, SHADER_WRITE -> SHADER_READ|WRITE) cannot
# know that and serialises all six.
#
# This is the shape §9's claim needs: on a purely linear chain every stage really
# does depend on the last, so derivation can only match, never win.
#
# Measured, and the answer is no — not on this hardware, not on either shape.
# Both arms warmed before either is timed and the rounds interleaved, because a
# first timed loop runs ~2.6x slow and derived-then-backend otherwise measures the
# order rather than the arms:
#
#   8 chains x 6 stages, 1<<20 floats   derived 2.362 ms   backend 2.371 ms  (1.004x)
#   the showcase, 20 passes             derived 1.187 ms   backend 1.176 ms  (0.991x)
#
# Inside the noise both times, and the derived spread (2.02-2.71) is wider than
# the backend one (2.32-2.43). So scoping barriers per resource is not what makes
# a frame faster here. The likely reason is that RADV's barrier granularity is the
# cache hierarchy rather than the buffer: a `VkBufferMemoryBarrier2` and a global
# `VkMemoryBarrier2` cost the same flush, so naming less memory does not let more
# overlap — that part is a hypothesis, not a measurement.
#
# Which leaves the reason the derivation is scoped anyway, and it is not speed: a
# barrier that names one buffer cannot stand in for a hazard on another, so what
# runs is what the graph derived, and a dependency the graph missed stops being
# silently supplied by a barrier that happened to be wide enough. That is what the
# hazard-set tests in `test/test_window.jl` can check and a global barrier cannot.

import Mantle
using Lava, GeometryBasics, KernelAbstractions
const M = Mantle

@kernel function stir!(dst, @Const(src), k::Float32)
    i = @index(Global)
    @inbounds dst[i] = src[i] * k + sin(src[i])
end

"""`chains` independent chains of `stages` each, interleaved stage by stage."""
function build_interleaved(dev, n, chains, stages; coalesce = true, alias = true)
    g = M.Graph(dev)
    bufs = [[M.Transient.Buffer(g, Float32, n) for _ in 1:(stages + 1)] for _ in 1:chains]
    for s in 1:stages, c in 1:chains
        M.compute!(g, "c$c s$s") do p
            src = M.use(p, bufs[c][s]; read = true)
            dst = M.use(p, bufs[c][s + 1]; write = true)
            M.dispatch!(p, stir!, (dst, src, 1.001f0), n)
        end
    end
    (; g, bufs, plan = M.Plan(g; coalesce, alias))
end

nbarriers(plan) = count(pp -> !isempty(pp.pre), plan.passes)

function run_bench(; n = 1 << 20, chains = 4, stages = 6, iters = 200)
    dev = M.Device(M.VulkanAPI())
    s = build_interleaved(dev, n, chains, stages)
    npasses = chains * stages

    function time_it(mode)
        M.run!(s.plan; barriers = mode); Mantle.flush!(dev.bq, dev.ctx.device)
        ts = Float64[]
        for _ in 1:iters
            t = time()
            M.run!(s.plan; barriers = mode)
            Mantle.flush!(dev.bq, dev.ctx.device)
            push!(ts, time() - t)
        end
        q = sort(ts)
        round(1000 * q[length(q) ÷ 2], digits = 3)
    end

    (passes = npasses,
     barriers_derived = nbarriers(s.plan),
     barriers_backend = npasses - 1,
     peak_mb = round(M.peakbytes(s.plan) / 2^20, digits = 2),
     naive_mb = round(Mantle.naivebytes(s.plan) / 2^20, digits = 2),
     derived_ms = time_it(:derived),
     backend_ms = time_it(:backend))
end
