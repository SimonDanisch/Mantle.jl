using Test
import Mantle, Lava
const M = Mantle
using KernelAbstractions: @kernel, @index, @Const

# A plan's indirect commands belong to the plan.
#
# They used to come from a slab ring on the `BatchQueue`, rewound by
# `reset_indirect_buffer_pool!` whenever the queue drained. A recording holds the
# address of its `VkDispatchIndirectCommand` for as long as it can be submitted,
# so the rewind put those bytes back on a free list under a live recording: two
# recorded plans running in one frame would write each other's
# workgroup counts, and the second one's prepare would land between the first
# one's prepare and its dispatch with nothing ordering them — a
# write-after-read the derived barriers cannot see, because the queue's scratch
# is not a resource the graph knows about.
#
# It could not be made to fail on demand, which is exactly why it is worth
# pinning structurally rather than behaviourally: the counts are laid out at
# compile now, in the plan's own argument memory, one per device-sized dispatch
# per slot. Two plans cannot collide because two plans cannot hold one region,
# and two slots cannot collide because the argument ring already says when a
# slot comes round again.

@kernel function pio_count!(n, @Const(src), thresh::Float32)
    i = @index(Global)
    @inbounds if i == 1
        c = Int32(0)
        for k in eachindex(src)
            src[k] > thresh && (c += Int32(1))
        end
        n[1] = c
    end
end

@kernel function pio_zero!(dst)
    i = @index(Global)
    @inbounds dst[i] = 0.0f0
end

@kernel function pio_mark!(dst, n)
    i = @index(Global)
    @inbounds if i <= n[1]
        dst[i] = 1.0f0
    end
end

"""A plan whose `nmark` marking dispatches each size themselves on the device,
over a count of `want`.

The marks land in PERSISTENT buffers, not transients. Two plans on one device
share an arena and every tenant is placed at offset zero, so two plans'
transients are the same bytes by construction — which is what the arena is for,
and would make "did plan `a` mark 1000 elements" a question about whichever plan
ran last."""
function markplan(dev, cap::Int, want::Int, nmark::Int)
    g = M.Graph(dev)
    src = M.Buffer(dev, Float32[k <= want ? 1.0f0 : 0.0f0 for k in 1:cap])
    n = M.Buffer(dev, Int32[0])
    ts = [M.Buffer(dev, zeros(Float32, cap)) for _ in 1:nmark]
    M.compute!(g, "count") do p
        M.dispatch!(p, pio_count!, (M.use(p, n; write = true),
                                    M.use(p, src; read = true), 0.5f0), 1)
    end
    for k in 1:nmark
        M.compute!(g, "zero$k") do p
            M.dispatch!(p, pio_zero!, (M.use(p, ts[k]; write = true),), cap)
        end
    end
    M.compute!(g, "mark") do p
        for k in 1:nmark
            M.dispatch!(p, pio_mark!, (M.use(p, ts[k]; write = true),
                                       M.use(p, n; read = true)),
                        M.DeviceRange(n; max = cap))
        end
    end
    (plan = M.record!(M.Plan(g)), outs = ts, count = n, src = src)
end

"""Every byte range this plan's indirect commands occupy."""
function indirectranges(pl)
    am = pl.args
    rs = Tuple{UInt64,UInt64}[]
    for v in am.indirect
        base = (v.buf[]::Any).address + UInt64(v.offset)
        push!(rs, (base, base + UInt64(3 * sizeof(UInt32))))
    end
    rs
end

@testset "a plan's indirect commands are laid out at compile" begin
    dev = M.Device(M.VulkanAPI())
    cap, nmark = 4096, 3
    p = markplan(dev, cap, 1000, nmark)
    # One per device-sized dispatch — and NOT one per direct dispatch: the
    # `zero` passes take a host ndrange and reserve nothing.
    @test length(p.plan.args.indirect) == nmark
    dispatches = collect(Iterators.flatten(pp.dispatches for pp in p.plan.passes))
    @test count(d -> d.indirect != 0, dispatches) == nmark
    @test sort(filter(!=(0), [d.indirect for d in dispatches])) == collect(1:nmark)
    M.free!(p.plan)
end

@testset "two recorded plans' indirect commands are disjoint" begin
    dev = M.Device(M.VulkanAPI())
    cap = 4096
    a = markplan(dev, cap, 1000, 2)
    b = markplan(dev, cap, 2500, 2)
    M.run!(a.plan); M.run!(b.plan); M.waitidle(dev)

    ra, rb = indirectranges(a.plan), indirectranges(b.plan)
    @test !isempty(ra) && length(ra) == length(rb)
    # Not one byte in common, over every slot of both — which is what makes
    # replaying them in either order mean the same thing.
    for (lo1, hi1) in ra, (lo2, hi2) in rb
        @test hi1 <= lo2 || hi2 <= lo1
    end

    # And the counts they read are their own, interleaved in one frame.
    for _ in 1:3
        M.run!(a.plan)
        M.run!(b.plan)
    end
    M.waitidle(dev)
    @test Array(a.count)[1] == Int32(1000)
    @test Array(b.count)[1] == Int32(2500)
    for t in a.outs
        @test count(==(1.0f0), Array(t)) == 1000
    end
    for t in b.outs
        @test count(==(1.0f0), Array(t)) == 2500
    end
    M.free!(a.plan); M.free!(b.plan)
end

@testset "a plan gives its argument memory back" begin
    dev = M.Device(M.VulkanAPI())
    sp = M.pool(dev)
    p = markplan(dev, 1024, 500, 2)
    M.run!(p.plan); M.waitidle(dev)
    held = length(p.plan.args.store)
    @test held > 0
    # Live bytes over every block of the arena, not the largest free span: with
    # a second, empty block already in the pool — any earlier plan in the
    # session leaves one — the largest span is that whole block before and
    # after, and says nothing about this plan's region coming back.
    live(kind) = sum((sum(values(b.live); init = 0) for b in M.blocksof(sp, kind)); init = 0)
    before = live(M.Unified())
    M.free!(p.plan)
    M.reclaim!(sp, dev)
    # The region is back, so the arena holds at least `held` fewer live bytes.
    # `free!` retires rather than releases, which is why the `reclaim!` above is
    # part of the test rather than an aside.
    @test live(M.Unified()) <= before - held
end
