# The host backend: the same graph, the same five analyses, no GPU.
#
# This file is the reason the phases were lifted out of the Vulkan backend. Nothing
# here copies a scheduler; `MantleHostExt` supplies thirteen mostly one-line
# methods and two near-empty phases, and core does the rest.

using Test
import Mantle, KernelAbstractions
const M = Mantle
const KA = KernelAbstractions
using KernelAbstractions: @kernel, @index, @Const

# The host backend was `ext/MantleHostExt.jl` until the KernelAbstractions
# weakdep turned out to be unworkable; it lives in `src/host/host.jl` now, so
# `HostDevice` is a plain `Mantle` name. `get_extension` returned `nothing` here
# and every use of it threw — unnoticed, because `import Vulkan` in runtests.jl
# aborted the run before this file was reached on a driverless machine.

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
    g, t = buildhostchain(M.Device(M.HostAPI()), n)
    plan = M.Plan(g)
    M.run!(plan)

    # 1 * 2 * 2 * 2. Only correct if the passes ran in the scheduled order — and
    # nothing synchronises between them, because on this backend the order IS the
    # synchronisation.
    @test all(==(8.0f0), M.storage(t[3]))
end

@testset "Host: compile bakes, run! decides nothing" begin
    g, _ = buildhostchain(M.Device(M.HostAPI()), 512)
    plan = M.Plan(g)
    # A `Plan` holds its baked work per pass now, not in one flat `plan.steps`
    # list — the Place/Barriers phases need to know which pass a dispatch
    # belongs to. `run!` walks it in exactly this order.
    steps = collect(Iterators.flatten(pp.dispatches for pp in plan.passes))
    @test length(steps) == 3
    # Every step is a concrete `Launch{K,A,N,D}`: the kernel is specialised, the
    # arguments are bound and the ndrange fixed at compile. A closure that still
    # had to look something up would show up here as something else.
    @test all(s -> s isa M.Launch, steps)
    @test isconcretetype(typeof(first(steps)))
end

@testset "Host: transients really share the arena" begin
    n = 1024
    g, t = buildhostchain(M.Device(M.HostAPI()), n)
    plan = M.Plan(g)
    @test M.naivebytes(plan) == 3 * n * sizeof(Float32)
    # Only adjacent links are ever live together, so the placer must overlap the
    # other pair. If this ever equals `naive`, the transients stopped sharing
    # bytes and the backend has quietly lost the ability to catch a placement
    # bug — which is the only reason it uses one arena instead of three arrays.
    @test M.peakbytes(plan) < M.naivebytes(plan)

    # …and the arena is genuinely one allocation the views point into. It is now
    # a POOL BLOCK, so it is at least the peak rather than exactly it — the plan
    # holds a region inside it, and that region is what matches the peak.
    #
    # `plan.arena`/`plan.regions` until the graph moved into core: a plan is
    # placed in one arena PER KIND now, so it carries a slab (the shared region)
    # and the arena kind side by side. The block behind the slab is the single
    # allocation the old `arena` field named.
    slab = only(plan.slabs)
    @test slab.block.bytes >= M.peakbytes(plan)
    @test length(slab) == M.peakbytes(plan)
    M.run!(plan)
    # A `Region` is an offset into a block, not an array — `memoryof` is the
    # allocation and `offset` says where this plan's bytes start in it.
    bytes = view(M.memoryof(slab), M.offset(slab) .+ (1:length(slab)))
    @test !all(iszero, bytes)
end

@testset "Host: alias = false is the bisection tool it claims to be" begin
    n = 1024
    g, _ = buildhostchain(M.Device(M.HostAPI()), n)
    @test M.peakbytes(M.Plan(g; alias = false)) == 3 * n * sizeof(Float32)
end

@testset "Host: custom! bodies are steps too" begin
    ran = Ref(0)
    dev = M.Device(M.HostAPI())
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
    dev = M.Device(M.HostAPI())
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
# The Vulkan backend used to define `Device(::typeof(Lava))`, selecting on the
# MODULE — and `typeof(Lava)` is `Module`, so a `Device(::typeof(KernelAbstractions))`
# here would have been the SAME signature, and whichever extension loaded second
# would have replaced the other with no warning.
#
# Both dispatch on a backend MARKER now — `VulkanAPI()` and `HostAPI()` — so they
# are different methods by construction rather than by luck. Asserted rather than
# reasoned about, because the failure mode was silent.
include(joinpath(@__DIR__, "backend_probe.jl"))

@testset "Host and Vulkan devices coexist" begin
    @test M.Device(M.HostAPI()) isa M.HostDevice
    # Only this line needs a driver. What is being asserted — that two backend
    # MARKERS give two distinct methods rather than one silently replacing the
    # other — is worth checking on a machine with no Vulkan loader too, so the
    # Host half stays unconditional and the file keeps running there.
    if backend_loadable("Vulkan") !== nothing
        @test M.Device(M.VulkanAPI()) isa MVE.LavaDevice
    else
        @info "no Vulkan loader; the Vulkan half of the coexistence test is skipped"
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
    dev = M.Device(M.HostAPI())
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
    dev = M.Device(M.HostAPI())
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

# ── the ceiling, and why it is not a host readback ────────────────────────────
#
# `DeviceRange(count; max = capacity)` says two things: where the real count
# lives, and how large it can possibly get. A RECORDING backend uses the first
# (an indirect dispatch reads it on the device) and compiles against the second.
# A KernelAbstractions backend has no indirect dispatch, and it used to resolve
# the range by reading the count ON THE HOST — which means synchronising the
# device first, before every such dispatch.
#
# That is what `bakedrange` no longer does when a ceiling exists. Measured on an
# M5, killeroo-gold at 684x513/8spp: 320 of those synchronises per frame,
# 0.585 s of a 0.602 s frame. Dispatching the ceiling instead is defined to be
# equivalent, because the same contract the recording path relies on already
# requires the kernel to bound itself — see `dispatchrange`.
@testset "a DeviceRange with a ceiling bakes to the ceiling, not a readback" begin
    dev = M.Device(M.HostAPI())

    function chainplan(; ceiling)
        g = M.Graph(dev)
        src = M.Buffer(dev, Float32[i for i in 1:16])
        n = M.Buffer(dev, Int32[0])
        dst = M.Transient.Buffer(g, Float32, 16)
        M.compute!(g, "count") do p
            M.dispatch!(p, hostcount!, (M.use(p, n; write = true),
                                        M.use(p, src; read = true), 9.5f0), 16)
        end
        M.compute!(g, "fill") do p
            M.dispatch!(p, hostfill!, (M.use(p, dst; write = true),),
                        M.DeviceRange(n; max = ceiling))
        end
        (M.Plan(g), dst)
    end

    fillrange(plan) = only(last(plan.passes).dispatches).ndrange

    # With a ceiling: a plain Int, decided at compile. Nothing reads the count
    # on the host, so nothing has to wait for the device to catch up.
    withmax, dst = chainplan(ceiling = 16)
    @test fillrange(withmax) == 16
    @test fillrange(withmax) isa Integer
    M.run!(withmax)
    @test all(==(1.0f0), M.storage(dst))

    # Without one there is nothing to dispatch instead, so the readback stays —
    # `INDIRECT_CEILING` threads over a queue that might hold four is not a
    # trade worth making.
    nomax, _ = chainplan(ceiling = nothing)
    @test fillrange(nomax) isa M.DeferredRange
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

# A struct whose fields do not fill its size: `b` is one byte, but `a` forces
# 8-byte alignment, so seven bytes of every sixteen are padding. Shaped after the
# real cases — `LightBVHNode`, and most GPU work-item structs, which pack a few
# floats next to a `UInt32` flag.
struct HostPadded
    a::Int64
    b::Int8
end

@kernel function hostwritepadded!(dst)
    i = @index(Global)
    @inbounds dst[i] = HostPadded(Int64(i), Int8(i % 100))
end

@kernel function hostcopypadded!(dst, @Const(src))
    i = @index(Global)
    @inbounds dst[i] = src[i]
end

@testset "Host: a padded struct can be WRITTEN, not only read" begin
    # A `reinterpret(T, ::Vector{UInt8})` refuses `setindex!` for any `T` with
    # padding — "Padding of type … is not compatible with type UInt8" — because
    # the padding bytes have no defined value to write. Host storage used to be
    # exactly that, so a buffer of any padded type could be read and never
    # written: `upload!` threw, and so did every kernel that filled a work
    # queue. The whole point of this backend is to be the one the others are
    # checked against, and it could not hold most of their data.
    dev = M.Device(M.HostAPI())
    n = 64
    @test sizeof(HostPadded) > sizeof(Int64) + sizeof(Int8)   # it really is padded

    want = [HostPadded(Int64(i), Int8(0)) for i in 1:n]
    b = M.Buffer(dev, want)                       # the upload! path
    @test Array(b) == want

    g = M.Graph(dev)
    t = M.Transient.Buffer(g, HostPadded, n)      # and the materialize! path
    M.compute!(g, "write") do p
        M.dispatch!(p, hostwritepadded!, (M.use(p, t; write = true),), n)
    end
    M.compute!(g, "copy") do p
        M.dispatch!(p, hostcopypadded!, (M.use(p, b; write = true),
                                         M.use(p, t; read = true)), n)
    end
    plan = M.Plan(g)
    M.run!(plan)
    @test Array(b) == [HostPadded(Int64(i), Int8(i % 100)) for i in 1:n]

    M.free!(plan)
    M.free!(b)
end

@testset "Host: storage aliases the block rather than copying it" begin
    # The property the fix above had to keep. Storage that copied would pass
    # every read-back assertion in this file while making the arena inert —
    # two transients the placer overlapped would quietly stop corrupting each
    # other, which is the one thing this backend exists to prove.
    dev = M.Device(M.HostAPI())
    b = M.Buffer(dev, Float32[1, 2, 3, 4])
    M.storage(b)[2] = 99f0
    @test Array(b) == Float32[1, 99, 3, 4]
    M.free!(b)
end

@testset "Host: a Buffer keeps its shape" begin
    # A renderer's own buffers are not all vectors — a framebuffer, an albedo
    # layer, a depth layer — and `Buffer` was a device VECTOR, so anything with
    # two dimensions had to be allocated outside the pool and freed by hand.
    dev = M.Device(M.HostAPI())
    want = reshape(collect(1f0:12f0), 3, 4)
    b = M.Buffer(dev, want)
    @test size(b) == (3, 4)
    @test ndims(b) == 2
    @test length(b) == 12
    @test size(M.storage(b)) == (3, 4)
    @test Array(b) == want                      # comes back shaped, not flattened

    M.storage(b)[2, 3] = 99f0                   # and it is the same bytes
    @test Array(b)[2, 3] == 99f0
    M.free!(b)

    # The vector form is untouched: `len` is still a prefix a draw covers, and
    # `Array` still trims to it.
    v = M.Buffer(dev, Float32[1, 2, 3]; capacity = 8)
    @test size(v) == (8,)                       # the region
    @test length(v) == 3                        # what a draw covers
    @test Array(v) == Float32[1, 2, 3]
    @test M.capacity(v) == 8
    M.free!(v)
end

"""
Release everything already retired, so a `reclaim!` count afterwards is about the
one region a test just gave back rather than whatever the file left lying around.
"""
drain!(pool, dev) = (while M.reclaim!(pool, dev; wait = true) > 0 end; nothing)

@testset "Host: free! defers, and reclaim! is what releases" begin
    # `free!` has no precondition: it RETIRES the region, and `reclaim!` puts it
    # back on a free list once the device is done. That is what lets a caller
    # drop a resource mid-frame without knowing what is in flight, and what
    # lets the same call be a finalizer — it appends under a lock and never
    # touches a free list, which is the bug `trim!`'s docstring describes.
    dev = M.Device(M.HostAPI())
    pool = M.pool(dev)

    b = M.Buffer(dev, fill(1f0, 4096))
    reserved = M.reserved(pool)
    drain!(pool, dev)                             # so a count below is about `b`
    M.free!(b)                                    # retires; releases nothing yet
    @test M.reclaim!(pool, dev) == 1              # released here, not there
    @test M.reclaim!(pool, dev) == 0              # and idempotent

    # Really back on the free list: an identical request reuses it rather than
    # making the pool ask the device for more.
    b2 = M.Buffer(dev, fill(2f0, 4096))
    @test M.reserved(pool) == reserved
    @test Array(b2) == fill(2f0, 4096)
    M.free!(b2)

    # A second `free!` of the same resource must not double-release: `release!`
    # errors on an overlapping span, so this would throw rather than corrupt.
    b3 = M.Buffer(dev, fill(3f0, 128))
    drain!(pool, dev)
    M.free!(b3)
    @test M.reclaim!(pool, dev) == 1
    @test M.reclaim!(pool, dev) == 0
end
