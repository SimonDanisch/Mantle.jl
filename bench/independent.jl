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
    dev = M.Device(Lava)
    s = build_interleaved(dev, n, chains, stages)
    npasses = chains * stages

    function time_it(mode)
        M.run!(s.plan; barriers = mode); Lava.flush!(dev.bq, dev.ctx.device)
        ts = Float64[]
        for _ in 1:iters
            t = time()
            M.run!(s.plan; barriers = mode)
            Lava.flush!(dev.bq, dev.ctx.device)
            push!(ts, time() - t)
        end
        q = sort(ts)
        round(1000 * q[length(q) ÷ 2], digits = 3)
    end

    (passes = npasses,
     barriers_derived = nbarriers(s.plan),
     barriers_backend = npasses - 1,
     peak_mb = round(M.peakbytes(s.plan) / 2^20, digits = 2),
     naive_mb = round(Base.get_extension(Mantle, :MantleLavaExt).naivebytes(s.plan) / 2^20, digits = 2),
     derived_ms = time_it(:derived),
     backend_ms = time_it(:backend))
end
