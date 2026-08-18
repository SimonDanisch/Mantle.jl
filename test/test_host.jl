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

    # …and the arena is genuinely one allocation the views point into. It is now
    # a POOL BLOCK, so it is at least the peak rather than exactly it — the plan
    # holds a region inside it, and that region is what matches the peak.
    @test length(plan.arena) >= M.peakbytes(plan)
    @test length(only(plan.regions)) == M.peakbytes(plan)
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

@testset "Host: an Update writes at the position the graph reserved" begin
    dev = M.Device(M.Host())
    g = M.Graph(dev)
    b = M.Buffer(dev, zeros(Float32, 4))
    ref = M.Update(g, b)
    seen = Float32[]
    M.custom!(g, "read it") do p
        M.use(p, b; read = true)
        () -> (append!(seen, copy(M.storage(b))); nothing)
    end
    plan = M.Plan(g)

    # Not fired: the update writes nothing, and the reader sees what was there.
    M.run!(plan)
    @test seen == zeros(Float32, 4)

    # Fired: the write lands BEFORE the pass that declared the read, which is the
    # whole claim — `ref(x)` itself only stores a reference.
    empty!(seen)
    ref(Float32[1, 2, 3, 4])
    M.run!(plan)
    @test seen == Float32[1, 2, 3, 4]

    # …and it is consumed, so the next frame writes nothing again.
    empty!(seen)
    M.run!(plan)
    @test seen == Float32[1, 2, 3, 4]

    # Host memory is directly addressable, so the write is in place and the
    # resource keeps its storage — where the Lava route renames into a fresh one.
    store = b.store
    ref(Float32[9, 9, 9, 9])
    M.run!(plan)
    @test b.store === store
    @test seen[(end - 3):end] == Float32[9, 9, 9, 9]
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

# ── DeviceRange ───────────────────────────────────────────────────────────────
#
# The count lives in device memory and is written by an earlier pass, so the
# dispatch size is not known when the graph is built. On this backend "device
# memory" is a Julia array, so the mechanism is visible without a GPU: the
# ndrange must be read at LAUNCH, and the pass that writes it must be ordered
# first — which it is because `dispatch!` registers the count as `Indirect`.

@kernel function hostcount!(n, @Const(src), thresh::Float32)
    i = @index(Global)
    @inbounds if src[i] > thresh
        # One thread decides; enough to prove the value reaches the ndrange.
        n[1] = Int32(i)
    end
end

@kernel function hostfill!(dst)
    i = @index(Global)
    @inbounds dst[i] = 1.0f0
end

@testset "Host: DeviceRange reads its count at launch, not at compile" begin
    dev = M.Device(M.Host())
    g = M.Graph(dev)
    src = M.Buffer(dev, Float32[i for i in 1:16])
    n = M.Buffer(dev, Int32[0])
    dst = M.Transient.Buffer(g, Float32, 16)

    M.compute!(g, "count") do p
        M.dispatch!(p, hostcount!, (M.use(p, n; write = true),
                                    M.use(p, src; read = true), 9.5f0), 16)
    end
    M.compute!(g, "fill") do p
        M.dispatch!(p, hostfill!, (M.use(p, dst; write = true),), M.DeviceRange(n))
    end

    plan = M.Plan(g)
    M.run!(plan)

    # `count` writes 16; the fill therefore covers all 16 elements. Had the
    # ndrange been resolved at compile it would have been the 0 the buffer held
    # then, and nothing would have been written.
    @test M.storage(n)[1] == Int32(16)
    @test all(==(1.0f0), M.storage(dst))
end

@testset "Host: a DeviceRange count is an Indirect read, so it orders the passes" begin
    dev = M.Device(M.Host())
    g = M.Graph(dev)
    n = M.Buffer(dev, Int32[4])
    dst = M.Transient.Buffer(g, Float32, 8)
    M.compute!(g, "fill") do p
        M.dispatch!(p, hostfill!, (M.use(p, dst; write = true),), M.DeviceRange(n))
    end
    # The usage the graph recorded for the count is `Indirect` — not a storage
    # read, because the stage and access it implies are different and a backend
    # emitting a storage barrier for it would order the wrong thing.
    usages = only(g.passes).usages
    @test any(u -> last(u) === M.Indirect, usages)
end

@testset "DeviceRange is core, not a backend's" begin
    # The record and the verb live in Mantle itself: both backends had
    # byte-identical copies of `Dispatch`/`dispatch!` before, which is one copy
    # per backend for the usage registration above to go missing from.
    @test isdefined(M, :Dispatch)
    @test isdefined(M, :DeviceRange)
    @test M.countresource(M.DeviceRange(1:3)) == 1:3
    @test M.countresource(1024) === nothing
    @test parentmodule(which(M.dispatch!, Tuple{Any,Any,Tuple,Any})) === M
end
