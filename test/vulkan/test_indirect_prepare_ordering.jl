using Test
using Lava, Mantle
import KernelAbstractions as KA
using KernelAbstractions: @kernel, @index

# An unmodelled indirect dispatch reads its `VkDispatchIndirectCommand` with the
# command processor, and the kernel that wrote it ran a moment earlier in the
# same command buffer. Nothing derives that dependency — nothing declared what
# either touches — so `record_dispatch!` emits the barrier unconditionally, with
# `INDIRECT_COMMAND_READ` in the destination access mask.
#
# It was conditional once, and this file is what that cost. A
# `concurrent_dispatch_group` asserted by hand that the dispatches inside it were
# independent and elided the barriers between them, which dropped exactly this
# one. With a deep GPU pipeline the race was usually won (latent); with an empty
# queue — right after a mid-pipeline flush — the indirect dispatch read stale
# group counts and silently ran zero workgroups. Surfaced through Hikari's
# volpath bounce loop (2026-06-10): `vp_shade_typed!` runs 12 prepare+indirect
# pairs, and adding an early-exit synchronize made the race lose reliably and
# dropped ~15% of shadow_bumpgold's energy. It was patched with a
# `force_pre_barrier` flag — an exception to an exception — and the groups and
# the flag are both deleted now.
#
# The chains below reproduced the race deterministically pre-fix (final count 0
# instead of 65536): each round copies a queue through a middle queue, so the
# second (indirect) copy's dispatch immediately follows its own prepare.

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

@testset "an indirect dispatch is ordered after its own prepare" begin
    backend = Mantle.defaultbackend()
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
        # stage 2: its prepare-indirect reads qmid.size and the indirect
        # dispatch must be ordered after that prepare's write.
        c!(qmid, nxt; ndrange=qmid.size)
        cur, nxt = nxt, cur
    end
    KA.synchronize(backend)
    final = Int(Array(cur.size)[1])
    @test final == n
end

@testset "three indirect dispatches, each after its own prepare" begin
    # Same chain, but stage 2 fans out into THREE indirect dispatches. Items are
    # split modulo 3 across destination mids and re-merged, so a dropped or
    # racing dispatch shows up as a wrong final count. This was the shape
    # `concurrent_indirect_group` existed for — it deferred all three prepares,
    # fused them into one dispatch and put one shared barrier behind them; a
    # plan does that from `emitprepares!`, and an undeclared launch does not get
    # to.
    backend = Mantle.defaultbackend()
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
        for q in mids
            c!(q, nxt; ndrange=q.size)
        end
        cur, nxt = nxt, cur
    end
    KA.synchronize(backend)
    final = Int(Array(cur.size)[1])
    @test final == n
end
