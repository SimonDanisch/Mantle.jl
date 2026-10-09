using Test
import Mantle
const M = Mantle
using KernelAbstractions: @kernel, @index, @Const
include(joinpath(@__DIR__, "testbackend.jl"))

# A plan's indirect commands belong to the plan.
#
# From a slab ring on the queue, rewound whenever it drained, they would move
# under the recording that names them. A recording holds the address of its
# indirect command for as long as it can be submitted, so the rewind put those
# bytes back on a free list under a live recording: two recorded plans running in
# one frame would write each other's workgroup counts, and the second one's
# prepare would land between the first one's prepare and its dispatch with
# nothing ordering them, a write-after-read the derived barriers cannot see,
# because the queue's scratch is not a resource the graph knows about.
#
# The counts are laid out at compile now, in the plan's own argument memory. What
# is pinned here is what that buys: two recorded plans interleaved in one frame
# each mark exactly as many elements as their own count says, and a plan that is
# freed gives its memory back.

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
    M.dispatch!(g, pio_count!, (n,
                                    src, 0.5f0), 1; name = "count")
    for k in 1:nmark
        M.dispatch!(g, pio_zero!, (ts[k],), cap; name = "zero$k")
    end
    for k in 1:nmark
        M.dispatch!(g, pio_mark!, (ts[k], n), M.DeviceRange(n; max = cap); name = "mark")
    end
    (plan = M.record!(M.Plan(g)), outs = ts, count = n, src = src)
end

@testset "two recorded plans' device-sized dispatches read their own counts" begin
    dev = M.Device(TESTBACKEND)
    cap = 4096
    a = markplan(dev, cap, 1000, 2)
    b = markplan(dev, cap, 2500, 2)
    # Interleaved in one frame, three times over: each plan's marks follow its own
    # count, which they would not if the two shared an indirect command.
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

@testset "a plan gives its memory back" begin
    dev = M.Device(TESTBACKEND)
    sp = M.pool(dev)
    # Live bytes over every block the pool holds, of every kind.
    live() = sum(b -> sum(values(b.live); init = 0), Iterators.flatten(values(sp.blocks)); init = 0)
    # `free!` retires rather than releases, which is why `reclaim!` is part of the
    # cycle rather than an aside.
    function cycle!()
        p = markplan(dev, 1024, 500, 2)
        M.run!(p.plan); M.waitidle(dev)
        M.free!(p.plan)
        foreach(M.free!, (p.src, p.count, p.outs...))
        M.reclaim!(sp, dev)
        return nothing
    end
    cycle!()    # compiles, and grows the pool to what one plan needs
    before = live()
    for _ in 1:10
        cycle!()
    end
    # Ten plans built, run and freed: whatever one held, it gave back.
    @test live() <= before
end
