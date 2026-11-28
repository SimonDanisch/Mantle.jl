# Each DeviceRange launch must observe its prepare kernel’s workgroup counts.
# Chained queue copies and fan-out exercise the dependency on an otherwise idle
# channel; submission ordering publishes queue payloads between kernels.
using Test, Mantle, KernelAbstractions
include(joinpath(@__DIR__, "testbackend.jl"))
const KA = KernelAbstractions
import Adapt
import Atomix

# A work queue: items and a device-side count, as Hikari's has. It must not
# depend on Hikari.
struct TestQueue{V, S}
    items::V
    size::S
end
Adapt.adapt_structure(to, q::TestQueue) =
    TestQueue(Adapt.adapt(to, q.items), Adapt.adapt(to, q.size))

@inline function queue_push!(q::TestQueue, item::Int32)
    # Reserve a slot; the following dispatch observes payloads after its barrier.
    idx = Atomix.@atomic :monotonic q.size[1] += Int32(1)
    if idx <= length(q.items)
        @inbounds q.items[idx] = item
    end
    return idx
end

@kernel function seed_queue!(q)
    i = @index(Global)
    queue_push!(q, Int32(i))
end

# Launched over `DeviceRange(src.size)`, so it bounds itself: a launch covers
# whole workgroups, and the threads past the count are real.
@kernel function copy_queue!(src, dst)
    i = @index(Global)
    if i <= src.size[1]
        @inbounds queue_push!(dst, src.items[i])
    end
end

@kernel function scatter3!(src, d1, d2, d3)
    i = @index(Global)
    if i <= src.size[1]
        v = @inbounds src.items[i]
        r = v % Int32(3)
        r == Int32(0) ? queue_push!(d1, v) :
        r == Int32(1) ? queue_push!(d2, v) : queue_push!(d3, v)
    end
end

makequeue(cap) = TestQueue(KA.allocate(TESTBACKEND, Int32, cap),
                           KA.allocate(TESTBACKEND, Int32, 1))

const N = 65536
const CAP = 100_000

@testset "an indirect dispatch is ordered after its own prepare" begin
    qa, qb, qmid = makequeue(CAP), makequeue(CAP), makequeue(CAP)
    for q in (qa, qb, qmid)
        fill!(q.size, Int32(0))
    end
    seed_queue!(TESTBACKEND, 256)(qa; ndrange = N)
    c! = copy_queue!(TESTBACKEND, 256)
    cur, nxt = qa, qb
    for _ in 1:40
        fill!(nxt.size, Int32(0))
        fill!(qmid.size, Int32(0))
        # stage 1: cur → qmid, sized by cur's count
        c!(cur, qmid; ndrange = Mantle.DeviceRange(cur.size))
        # stage 2: sized by qmid's count, which stage 1 just wrote
        c!(qmid, nxt; ndrange = Mantle.DeviceRange(qmid.size))
        cur, nxt = nxt, cur
    end
    KA.synchronize(TESTBACKEND)
    @test Int(Array(cur.size)[1]) == N
end

@testset "three indirect dispatches, each after its own prepare" begin
    # Same chain, but stage 2 fans out into THREE launches. Items are split
    # modulo 3 across destination queues and re-merged, so a dropped or racing
    # launch shows up as a wrong final count. This is the shape
    # `concurrent_indirect_group` existed for: it deferred all three prepares,
    # fused them into one dispatch and put one shared barrier behind them; a plan
    # does that from `emitprepares!`, and an undeclared launch does not get to.
    qa, qb = makequeue(CAP), makequeue(CAP)
    mids = (makequeue(CAP), makequeue(CAP), makequeue(CAP))
    for q in (qa, qb, mids...)
        fill!(q.size, Int32(0))
    end
    seed_queue!(TESTBACKEND, 256)(qa; ndrange = N)
    sc! = scatter3!(TESTBACKEND, 256)
    c! = copy_queue!(TESTBACKEND, 256)
    cur, nxt = qa, qb
    for _ in 1:40
        fill!(nxt.size, Int32(0))
        for q in mids
            fill!(q.size, Int32(0))
        end
        sc!(cur, mids...; ndrange = Mantle.DeviceRange(cur.size))
        for q in mids
            c!(q, nxt; ndrange = Mantle.DeviceRange(q.size))
        end
        cur, nxt = nxt, cur
    end
    KA.synchronize(TESTBACKEND)
    @test Int(Array(cur.size)[1]) == N
end
