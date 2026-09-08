"""
A Mantle graph, compiled and run on Metal.

The point of the whole split, end to end: passes are declared, `compile!` runs
the seven phases, `Place` aliases the transients into an arena, and a frame is a
loop over baked callables. None of that is in the Metal backend — it is
`src/graph/` and `src/phases.jl`, shared with every other backend.

What Metal contributes is four methods: `materialize!`, `alignment`, a
`Barriers` no-op, and the pool primitives. If this file passes, the graph is
genuinely Mantle's.
"""

using Test, Mantle, Metal, KernelAbstractions
const KA = KernelAbstractions

@kernel cpu = false unsafe_indices = true function _scale!(o, @Const(a), s)
    i = @index(Global, Linear)
    @inbounds o[i] = a[i] * s
end

@kernel cpu = false unsafe_indices = true function _add!(o, @Const(x), @Const(y))
    i = @index(Global, Linear)
    @inbounds o[i] = x[i] + y[i]
end

@testset "Metal: a two-pass graph with a transient between them" begin
    dev = Mantle.Device(Mantle.MetalAPI())
    g = Mantle.Graph(dev)
    n = 1024

    inp = Mantle.Buffer(dev, collect(1.0f0:Float32(n)))
    out = Mantle.Buffer(dev, zeros(Float32, n))
    mid = Mantle.Transient.Buffer(g, Float32, n)

    Mantle.compute!(g, "scale") do p
        Mantle.use(p, inp; read = true)
        Mantle.use(p, mid; write = true)
        Mantle.dispatch!(p, _scale!, (mid, inp, 3.0f0), n)
    end
    Mantle.compute!(g, "add") do p
        Mantle.use(p, mid; read = true)
        Mantle.use(p, inp; read = true)
        Mantle.use(p, out; write = true)
        Mantle.dispatch!(p, _add!, (out, mid, inp), n)
    end

    @test length(Mantle.passes(g)) == 2
    @test length(g.transients) == 1

    plan = Mantle.Plan(g)
    # Every plan opens with the `updates` pass, on every backend: pending host
    # stores land there, ahead of anything that reads them. Spelled as the kinds
    # rather than a count, because a count says nothing about which pass moved.
    @test [p.pass.kind for p in plan.passes] == [:update, :compute, :compute]
    @test [p.pass.name for p in plan.passes] == ["updates", "scale", "add"]
    # The transient was placed into an arena, not allocated on its own.
    @test Mantle.peakbytes(plan) >= n * sizeof(Float32)
    # A KernelAbstractions backend has no argument memory and no profiler; the
    # plan says so rather than carrying an empty one.
    @test plan.args === nothing
    @test plan.profiler === nothing

    Mantle.run!(plan)
    Metal.synchronize()

    want = collect(1.0f0:Float32(n)) .* 3.0f0 .+ collect(1.0f0:Float32(n))
    @test Array(out) ≈ want

    # Re-runnable: a plan is a recording, not a one-shot. The second run must
    # produce the same answer from the same baked callables.
    Mantle.run!(plan)
    Metal.synchronize()
    @test Array(out) ≈ want
end

@testset "Metal: the ordering the graph derived is the ordering that ran" begin
    # `mid` is written by one pass and read by the next. If the scheduler
    # emitted them in the wrong order — or if `Barriers` being a no-op on this
    # backend were wrong — the read would see zeros, not the scaled values.
    #
    # Checked with a value that cannot arise by accident: reading `mid` before
    # the write gives `0 + inp`, which is `inp`, and reading after gives
    # `3*inp + inp`. The two differ everywhere except where `inp` is zero.
    dev = Mantle.Device(Mantle.MetalAPI())
    g = Mantle.Graph(dev)
    n = 256
    inp = Mantle.Buffer(dev, fill(2.0f0, n))
    out = Mantle.Buffer(dev, zeros(Float32, n))
    mid = Mantle.Transient.Buffer(g, Float32, n)

    Mantle.compute!(g, "producer") do p
        Mantle.use(p, inp; read = true)
        Mantle.use(p, mid; write = true)
        Mantle.dispatch!(p, _scale!, (mid, inp, 5.0f0), n)
    end
    Mantle.compute!(g, "consumer") do p
        Mantle.use(p, mid; read = true)
        Mantle.use(p, inp; read = true)
        Mantle.use(p, out; write = true)
        Mantle.dispatch!(p, _add!, (out, mid, inp), n)
    end

    plan = Mantle.Plan(g)
    Mantle.run!(plan)
    Metal.synchronize()
    @test all(≈(12.0f0), Array(out))      # 5*2 + 2, not 0 + 2
end

# ── a store lands, on the GPU side too ────────────────────────────────────────
#
# `run!(::Plan)` in `graph/kalaunch.jl` walked `pp.dispatches` and nothing else,
# and the update pass HAS no dispatches — landing the writes a store is holding
# is its entire body. So every store was silently dropped on every
# KernelAbstractions backend, Metal included. Only the Vulkan route, which
# records its passes itself, ever applied one.
#
# The host suite is what caught it, but the hole was in shared code, and this is
# the side where the write goes through mapped device memory instead of a plain
# `Vector` — and where a kernel, not a closure, is the thing that must observe
# it in the position the graph reserved.
@testset "Metal: a store lands before the pass that reads it" begin
    dev = Mantle.Device(Mantle.MetalAPI())
    g = Mantle.Graph(dev)
    n = 256

    src = Mantle.Buffer(dev, zeros(Float32, n))
    dst = Mantle.Buffer(dev, zeros(Float32, n))
    Mantle.compute!(g, "copy") do p
        Mantle.use(p, src; read = true)
        Mantle.use(p, dst; write = true)
        Mantle.dispatch!(p, _scale!, (dst, src, 1.0f0), n)
    end
    plan = Mantle.Plan(g)

    # The update pass is a pass, dispatchless or not, and it is FIRST — that
    # ordering is what makes the write visible to the reader in the same frame.
    @test length(plan.passes) == 2
    @test first(plan.passes).pass.kind === :update
    @test isempty(first(plan.passes).dispatches)

    # Not fired: nothing is written and the kernel copies the zeros.
    Mantle.run!(plan)
    Metal.synchronize()
    @test all(iszero, Array(dst))

    # Written: the store lands before the dispatch that declared the read. This
    # is the assertion the bug failed — `dst` stayed zero.
    src[:] = fill(3.0f0, n)
    Mantle.run!(plan)
    Metal.synchronize()
    @test Array(dst) == fill(3.0f0, n)

    # Consumed: a store that is not made again writes nothing next frame, so the
    # reader keeps seeing the last value rather than a stale one reappearing.
    src[:] = fill(7.0f0, n)
    Mantle.run!(plan)
    Metal.synchronize()
    @test Array(dst) == fill(7.0f0, n)
    Mantle.run!(plan)
    Metal.synchronize()
    @test Array(dst) == fill(7.0f0, n)

    # Unified memory, so the write is in place and the resource keeps its store
    # — the same as every backend, now that nothing renames.
    store = src.store
    src[:] = fill(2.0f0, n)
    Mantle.run!(plan)
    Metal.synchronize()
    @test src.store === store
    @test Array(dst) == fill(2.0f0, n)
end
