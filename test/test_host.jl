# The host backend: the same graph, the same five analyses, no GPU.
#
# This file is the reason the phases were lifted out of `MantleLavaExt`. Nothing
# here copies a scheduler; `MantleHostExt` supplies thirteen mostly one-line
# methods and two near-empty phases, and core does the rest.

using Test
import Mantle, KernelAbstractions
const M = Mantle
const KA = KernelAbstractions
using KernelAbstractions: @kernel, @index, @Const

const HOSTEXT = Base.get_extension(Mantle, :MantleHostExt)

@kernel function hostscale!(dst, @Const(src), a::Float32)
    i = @index(Global)
    @inbounds dst[i] = src[i] * a
end

"A forced chain: each pass reads what the previous wrote."
function buildhostchain(dev, n::Integer, links::Integer = 3)
    g = M.Graph(dev)
    src = M.Buffer(dev, fill(1.0f0, n))
    t = [M.Transient.Buffer(g, Float32, n) for _ in 1:links]
    prev = src
    for (k, dst) in enumerate(t)
        M.compute!(g, "chain$k") do p
            M.dispatch!(p, hostscale!, (M.use(p, dst; write = true),
                                        M.use(p, prev; read = true), 2.0f0), n)
        end
        prev = dst
    end
    return g, t
end

@testset "Host: the graph runs, in order, with no GPU present" begin
    n = 1024
    g, t = buildhostchain(M.Device(M.Host()), n)
    plan = M.Plan(g)
    M.run!(plan)

    # 1 * 2 * 2 * 2. Only correct if the passes ran in the scheduled order — and
    # nothing synchronises between them, because on this backend the order IS the
    # synchronisation.
    @test all(==(8.0f0), M.storage(t[3]))
end

@testset "Host: compile bakes, run! decides nothing" begin
    g, _ = buildhostchain(M.Device(M.Host()), 512)
    plan = M.Plan(g)
    @test length(plan.steps) == 3
    # Every step is a concrete `Launch{K,A,N}`: the kernel is specialised, the
    # arguments are bound and the ndrange fixed at compile. A closure that still
    # had to look something up would show up here as something else.
    @test all(s -> s isa HOSTEXT.Launch, plan.steps)
    @test isconcretetype(typeof(first(plan.steps)))
end

@testset "Host: transients really share the arena" begin
    n = 1024
    g, t = buildhostchain(M.Device(M.Host()), n)
    plan = M.Plan(g)
    @test HOSTEXT.naivebytes(plan) == 3 * n * sizeof(Float32)
    # Only adjacent links are ever live together, so the placer must overlap the
    # other pair. If this ever equals `naive`, the transients stopped sharing
    # bytes and the backend has quietly lost the ability to catch a placement
    # bug — which is the only reason it uses one arena instead of three arrays.
    @test M.peakbytes(plan) < HOSTEXT.naivebytes(plan)

    # …and the arena is genuinely one allocation the views point into.
    @test length(plan.arena) == M.peakbytes(plan)
    M.run!(plan)
    @test !all(iszero, plan.arena)
end

@testset "Host: alias = false is the bisection tool it claims to be" begin
    n = 1024
    g, _ = buildhostchain(M.Device(M.Host()), n)
    @test M.peakbytes(M.Plan(g; alias = false)) == 3 * n * sizeof(Float32)
end

@testset "Host: custom! bodies are steps too" begin
    ran = Ref(0)
    dev = M.Device(M.Host())
    g = M.Graph(dev)
    b = M.Buffer(dev, fill(1.0f0, 8))
    M.custom!(g, "by hand") do p
        M.use(p, b; read = true, write = true)
        () -> (ran[] += 1; nothing)
    end
    M.run!(M.Plan(g))
    @test ran[] == 1
end

# ── the thing that used to be silently broken ─────────────────────────────────
#
# `MantleLavaExt` defines `Device(::typeof(Lava))`, and `typeof(Lava)` is
# `Module` — so a `Device(::typeof(KernelAbstractions))` here would have been the
# SAME signature, and whichever extension loaded second would have replaced the
# other with no warning. Both load in any session with a GPU and a host graph,
# which is every session this work is aimed at.
#
# Dispatching on the backend tag instead makes them different methods. Asserted
# rather than reasoned about, because the failure mode is silent.
if Base.get_extension(Mantle, :MantleLavaExt) !== nothing
    @testset "Host and Lava devices coexist" begin
        @test M.Device(M.Host()) isa HOSTEXT.HostDevice
        @test M.Device(Lava) isa Base.get_extension(Mantle, :MantleLavaExt).LavaDevice
    end
end
