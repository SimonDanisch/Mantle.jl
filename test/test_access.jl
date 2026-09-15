# What a kernel does to its arguments, read off the kernel.
#
# This is the analysis every declaration now comes from, so what it has to be is
# not "usually right": EXACT where it can be, and CONSERVATIVE where it cannot —
# claiming an access that does not happen costs a barrier, and missing one costs
# a race. Both halves are pinned here, on kernels small enough that the answer is
# obvious by reading them.
#
# The cases are the ones that were wrong at some point while it was written:
#
#   * a scalar argument came back read+write, because an argument was tainted
#     before its type was asked whether it could reach memory at all;
#   * an argument whose LENGTH a kernel reads came back read, for the same
#     reason one step later;
#   * a kernel written `f(queue, args...)` — which is every wavefront stage —
#     gave all twenty arguments whatever any one of them got, because the
#     inferred IR collapses a vararg tail into one argument;
#   * an atomic came back as an ordinary read-write, because Lava's atomics are
#     `llvmcall` and the module is named by a `GlobalRef` rather than inline.
using Test, Mantle, KernelAbstractions, Lava
import Atomix, Adapt
import KernelInterface as KI
const M = Mantle
const TESTBACKEND = M.VulkanAPI()

@kernel function acc_mixed!(dst, @Const(src), @Const(unused), scale::Float32)
    i = @index(Global)
    @inbounds dst[i] = src[i] * scale
end

@kernel function acc_atomic!(acc, @Const(src))
    i = @index(Global)
    @inbounds Atomix.@atomic acc[i] += src[i]
end

@kernel function acc_lengthonly!(dst, @Const(other))
    i = @index(Global)
    @inbounds dst[i] = Float32(length(other))
end

# Not inlined, so the walk has to follow the call to find the store.
@noinline function acc_fill_one!(v, i)
    @inbounds v[i] = 1f0
    return nothing
end
@kernel function acc_helper!(dst)
    i = @index(Global)
    acc_fill_one!(dst, i)
end

struct AccPair{A}
    a::A
    b::A
end
Adapt.@adapt_structure AccPair

@kernel function acc_struct!(q::AccPair, @Const(src))
    i = @index(Global)
    @inbounds q.a[i] = src[i]
end

# A work queue's append, which is the shape the wavefront stages have: claim a
# slot with an atomic, then store the payload at it. The payload store is an
# ORDINARY store and no two invocations can collide, which is what makes the
# whole queue unordered against another pass doing the same.
struct AccQueue{I,S}
    items::I
    size::S
end
Adapt.@adapt_structure AccQueue

@kernel function acc_append!(q::AccQueue, @Const(src))
    i = @index(Global)
    v = @inbounds src[i]
    slot = Atomix.@atomic q.size[1] += Int32(1)
    @inbounds q.items[slot] = v
end

# The higher-order shape: the kernel takes the real kernel and a tail of
# arguments, and splats them. Every stage of Hikari's integrator is this.
@kernel function acc_map!(f, n::Int32, args...)
    i = @index(Global)
    i <= n && f(i, args...)
end
acc_inner!(i, dst, src, spare) = (@inbounds dst[i] = src[i]; nothing)

"""`Touch` as a string: `R`, `W`, and `~` for a write nothing can collide with."""
show_t(t) = (t.read ? "R" : "-") * (t.write ? "W" : "-") * (t.write && t.atomic ? "~" : "")
touches(dev, k, args, nd; group = nothing) =
    show_t.(M.kerneltouches(dev, k, args, nd, group))

@testset "inferred access: exact where it can be" begin
    dev = M.Device(TESTBACKEND)
    n = 256
    dst = M.devicearray(dev, Float32, n)
    src = M.devicearray(dev, fill(1f0, n))
    other = M.devicearray(dev, Float32, n)

    # Written, read, untouched, and a scalar that has no bytes to touch.
    @test touches(dev, acc_mixed!, (dst, src, other, 2f0), n) == ["-W", "R-", "--", "--"]

    # An argument a kernel only asks the LENGTH of is not read: the size is a
    # by-value field of the array handle, not a load from its storage.
    @test touches(dev, acc_lengthonly!, (dst, other), n) == ["-W", "--"]

    # Through a call the optimiser left standing.
    @test touches(dev, acc_helper!, (dst,), n) == ["-W"]

    # Through a struct of arrays: `q.a` is written, and `q` is what the graph
    # declares, so the whole container is written and `src` is only read.
    q = AccPair(M.devicearray(dev, Float32, n), M.devicearray(dev, Float32, n))
    @test touches(dev, acc_struct!, (q, src), n) == ["-W", "R-"]
end

@testset "inferred access: an atomic is a write nothing can collide with" begin
    dev = M.Device(TESTBACKEND)
    n = 256
    acc = M.devicearray(dev, Float32, n)
    src = M.devicearray(dev, fill(1f0, n))
    @test touches(dev, acc_atomic!, (acc, src), n) == ["RW~", "R-"]

    # …and so is a plain store at an index claimed from one. The counter and the
    # payload are fields of ONE argument, which is what makes the disjointness
    # provable: a slot claimed from this queue's counter is a slot in this
    # queue's payload, and no two invocations — or two passes appending to it —
    # can be handed the same one.
    q = AccQueue(M.devicearray(dev, Float32, n), M.devicearray(dev, Int32, 1))
    @test touches(dev, acc_append!, (q, src), n) == ["RW~", "R-"]
end

@testset "inferred access: a vararg tail is not one argument" begin
    dev = M.Device(TESTBACKEND)
    n = 64
    dst = M.devicearray(dev, Float32, n)
    src = M.devicearray(dev, fill(2f0, n))
    spare = M.devicearray(dev, Float32, n)
    # `acc_map!` has FIVE arguments in the inferred IR — the kernel, the
    # iteration context, `f`, `n`, and one tuple holding the other three — while
    # the signature has seven. Without the element index, `dst`, `src` and
    # `spare` would all answer whatever any one of them got.
    @test touches(dev, acc_map!, (acc_inner!, Int32(n), dst, src, spare), n) ==
          ["--", "--", "-W", "R-", "--"]
end

# A kernel calling an intrinsic this backend does not implement — which is what
# `KI.localmemory` WAS, before its signature grew a call-site id. Every path
# through the body throws, so it infers to `Union{}` and its IR is unreachable
# from the first statement on.
function acc_missing_intrinsic end

function acc_uninferable!(dst, n::Int)
    i = KI.get_global_id().x
    v = acc_missing_intrinsic(Float32(i))
    @inbounds dst[i] = v
    return
end

@testset "inferred access: a kernel that does not infer is not untouched" begin
    dev = M.Device(TESTBACKEND)
    n = 64
    dst = M.devicearray(dev, Float32, n)
    # "No store" is the one answer that must never be a guess — it is the answer
    # that removes a barrier. An unreachable body says nothing, not nothing-done.
    @test touches(dev, acc_uninferable!, (dst, n), n) == ["RW", "RW"]
end

@testset "inferred access: the unknown widens, it does not vanish" begin
    dev = M.Device(TESTBACKEND)
    n = 16
    a = M.devicearray(dev, Float32, n)
    # `accessof` on a signature nothing can be inferred for answers `OPAQUE` for
    # every argument — never `NOTOUCH`, which would be a missing barrier.
    ts = M.accessof(nothing, sin, (Any,))
    @test all(t -> t.read && t.write, ts)
    @test M.usagetype(M.OPAQUE, M.BufferKind()) === Storage{BufferKind, ReadWrite}

    # And the pieces the walk decides with.
    @test M.carries(Float32) === false
    @test M.carries(Tuple{Int,Int}) === false
    @test M.carries(Ptr{Float32}) === true
    @test M.carries(Any) === true
    @test M.carries(M.LavaDeviceArray{Float32,1}) === true

    # A `Touch` with an ordinary write is ordered; one whose writes are all
    # atomic is not.
    @test M.usagetype(M.WRITE, M.BufferKind()) === Storage{BufferKind, WriteOnly}
    @test M.usagetype(M.ATOMIC, M.BufferKind()) ===
          Unordered{Storage{BufferKind, ReadWrite}}
    @test M.usagetype(M.READ, M.BufferKind()) === Storage{BufferKind, ReadOnly}
end

@testset "inferred access: what a dispatch declares" begin
    dev = M.Device(TESTBACKEND)
    n = 64
    g = M.Graph(dev)
    dst = M.Buffer(dev, zeros(Float32, n))
    src = M.Buffer(dev, fill(1f0, n))
    p = M.dispatch!(g, acc_mixed!, (dst, src, src, 2f0), n)

    byid = Dict(id => U for (id, U) in p.usages)
    @test byid[M.resourceid(g, dst)] === Storage{BufferKind, WriteOnly}
    @test byid[M.resourceid(g, src)] === Storage{BufferKind, ReadOnly}
    @test p.name == "acc_mixed!"          # the kernel names its own pass

    # A container is declared by its LEAVES, which is what a barrier is scoped to.
    q = AccPair(M.Buffer(dev, zeros(Float32, n)), M.Buffer(dev, zeros(Float32, n)))
    p2 = M.dispatch!(g, acc_struct!, (q, src), n; name = "struct")
    @test M.resourceid(g, q.a) in first.(p2.usages)
    @test M.resourceid(g, q.b) in first.(p2.usages)

    # A slice is an argument, and it is the slice the pass is recorded touching.
    sl = M.slice(g, dst, 1:32)
    p3 = M.dispatch!(g, acc_mixed!, (sl, src, src, 3f0), 32; name = "sliced")
    @test M.resourceid(g, sl) in first.(p3.usages)
    @test !(M.resourceid(g, dst) in first.(p3.usages))
end

@testset "inferred access: the device answers from a cache" begin
    dev = M.Device(TESTBACKEND)
    n = 64
    dst = M.devicearray(dev, Float32, n)
    src = M.devicearray(dev, fill(1f0, n))
    M.kerneltouches(dev, acc_mixed!, (dst, src, src, 1f0), n, nothing)
    c = M.accesscache(dev)
    @test c isa M.AccessCache
    # The second ask is a lookup, not an inference pass — the difference is
    # seconds per kernel on a wavefront stage, so it is worth pinning that it
    # happens at all. Measured rather than counted, because the cache is dropped
    # whole when the world age moves and a test file moves it constantly.
    t = @elapsed M.kerneltouches(dev, acc_mixed!, (dst, src, src, 1f0), n, nothing)
    @test t < 0.1
end
