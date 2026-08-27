using Test
using Lava, Mantle
import KernelAbstractions as KA
using KernelAbstractions: @kernel, @index

# Regression: inside a `concurrent_dispatch_group`, the group's barrier
# elision also dropped the barrier between a prepare-indirect kernel and the
# `vkCmdDispatchIndirect` that reads the VkDispatchIndirectCommand it wrote.
# That read raced the prepare's write: with a deep GPU pipeline the race was
# usually won (latent), but with an empty queue — e.g. right after a
# mid-pipeline flush — the indirect dispatch read stale group counts and
# silently ran with 0 (or garbage) workgroups.
#
# Surfaced through Hikari's volpath bounce loop (2026-06-10):
# `vp_shade_typed!` runs 12 prepare+indirect pairs inside one group; adding
# an early-exit synchronize to the loop made the race lose reliably and
# dropped ~15% of shadow_bumpgold's energy. Fixed by `force_pre_barrier=true`
# on every indirect dispatch record.
#
# The chain below reproduced the race deterministically pre-fix (final
# count 0 instead of 65536): each round copies a queue through a middle
# queue, with the second (indirect) copy recorded inside a group so its
# dispatch immediately follows its own prepare.

# Minimal WorkQueue clone — this is a Lava test; it must not depend on Hikari.
struct TestQueue{V, S}
    items::V
    size::S
end
import Adapt
Adapt.adapt_structure(to, q::TestQueue) =
    TestQueue(Adapt.adapt(to, q.items), Adapt.adapt(to, q.size))

import Atomix
@inline function queue_push!(q::TestQueue, item::Int32)
    idx = Atomix.@atomic q.size[1] += Int32(1)
    if idx <= length(q.items)
        @inbounds q.items[idx] = item
    end
    return idx
end

@kernel function seed_queue!(q)
    i = @index(Global)
    queue_push!(q, Int32(i))
end

@kernel function copy_queue!(src, dst)
    i = @index(Global)
    if i <= src.size[1]
        @inbounds queue_push!(dst, src.items[i])
    end
end

@testset "indirect dispatch inside concurrent_dispatch_group" begin
    backend = Mantle.LavaBackend()
    n = 65536
    cap = 100_000

    make_queue() = TestQueue(KA.allocate(backend, Int32, cap),
                             KA.allocate(backend, Int32, 1))
    qa, qb, qmid = make_queue(), make_queue(), make_queue()
    for q in (qa, qb, qmid)
        KA.fill!(q.size, Int32(0))
    end

    s! = seed_queue!(backend, 256)
    s!(qa; ndrange=n)

    c! = copy_queue!(backend, 256)
    cur, nxt = qa, qb
    for _ in 1:40
        KA.fill!(nxt.size, Int32(0))
        KA.fill!(qmid.size, Int32(0))
        # stage 1: cur → qmid (indirect dispatch, group count from cur.size)
        c!(cur, qmid; ndrange=cur.size)
        # stage 2 INSIDE a group: its prepare-indirect reads qmid.size and the
        # indirect dispatch must barrier against that prepare even though the
        # group elides inter-dispatch barriers.
        Mantle.concurrent_dispatch_group() do
            c!(qmid, nxt; ndrange=qmid.size)
        end
        cur, nxt = nxt, cur
    end
    KA.synchronize(backend)
    final = Int(Array(cur.size)[1])
    @test final == n
end

@testset "concurrent_indirect_group (deferred two-phase)" begin
    # Same chain, but stage 2 fans out into THREE indirect dispatches inside
    # a `concurrent_indirect_group` — the deferred two-phase form records all
    # prepares, one shared barrier, then the dispatches overlapped. Items are
    # split modulo 3 across destination mids and re-merged, so a dropped or
    # racing dispatch shows up as a wrong final count.
    backend = Mantle.LavaBackend()
    n = 65536
    cap = 100_000

    make_queue() = TestQueue(KA.allocate(backend, Int32, cap),
                             KA.allocate(backend, Int32, 1))
    qa, qb = make_queue(), make_queue()
    mids = (make_queue(), make_queue(), make_queue())
    for q in (qa, qb, mids...)
        KA.fill!(q.size, Int32(0))
    end

    s! = seed_queue!(backend, 256)
    s!(qa; ndrange=n)

    @kernel function scatter3!(src, d1, d2, d3)
        i = @index(Global)
        if i <= src.size[1]
            v = @inbounds src.items[i]
            r = v % Int32(3)
            r == Int32(0) ? queue_push!(d1, v) :
            r == Int32(1) ? queue_push!(d2, v) : queue_push!(d3, v)
        end
    end
    sc! = scatter3!(backend, 256)
    c! = copy_queue!(backend, 256)

    cur, nxt = qa, qb
    for _ in 1:40
        KA.fill!(nxt.size, Int32(0))
        for q in mids
            KA.fill!(q.size, Int32(0))
        end
        sc!(cur, mids...; ndrange=cur.size)
        Mantle.concurrent_indirect_group() do
            for q in mids
                c!(q, nxt; ndrange=q.size)
            end
        end
        cur, nxt = nxt, cur
    end
    KA.synchronize(backend)
    final = Int(Array(cur.size)[1])
    @test final == n
end
