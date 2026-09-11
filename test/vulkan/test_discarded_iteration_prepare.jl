"""
The prepare of a discarded `repeat!` iteration still runs, and writes zero.

A device-sized dispatch or trace reads its count from an indirect slot that the
pass's fused prepare writes. The prepare folds the iteration's gate in: a
discarded iteration gets ZERO groups, so nothing runs for it on a backend that
cannot discard recorded work. That is only true if the prepare itself runs —
and it was emitted INSIDE the conditional-rendering scope, so on a device with
the extension it was discarded along with the iteration, and the slot kept
whatever the memory held.

For a dispatch that is harmless: the dispatch is discarded too. For a trace it
is not. `vkCmdTraceRaysIndirectKHR` is not subject to conditional rendering
(VK_KHR_ray_tracing_pipeline leaves the trace commands out of it), so the trace
of a discarded bounce ran, with a ray count read from a slot nothing had
written: garbage from a freed BLAS scratch region in a fresh session — a lost
device on the RTX at Hikari's second bounce whenever a scene's rays all left
by the first (RayMakie's `test_transform_update_hwtlas.jl`, a box under a
camera) — or a stale small count from the previous plan's argument memory,
which is why the same scene rendered when another had run before it, and why
black_hole's ~90 discarded rounds lost the device "twice, in long sessions".
The AMD driver runs its trace commands as compute, which conditional rendering
does discard, so RADV never showed it.

Pinned with a dispatch, not a trace, because the fault is a device loss and
this asserts the mechanism: the slot of the discarded iteration is written to
zero, not left holding the sentinel put there before the run. The prepare is
emitted before the predicate scope now, on every backend, which is where the
fold always belonged.
"""

using Test, Mantle, Lava, KernelAbstractions
const KA = KernelAbstractions

@kernel function dip_bump!(a, n)
    i = @index(Global)
    @inbounds i <= n[1] && (a[i] += Int32(1))
end

@testset "the prepare of a discarded iteration writes zero groups" begin
    dev = Mantle.Device(Mantle.VulkanAPI())
    Mantle.vk_context().conditional_rendering_available || error(
        "this device has no conditional rendering; the bug this pins needs it")
    n = 200
    out = Mantle.Buffer(dev, zeros(Int32, n))
    count = Mantle.Buffer(dev, Int32[n])          # the dispatch's device-side size
    trips = Mantle.GPURef(dev, Int32(1))          # the loop runs ONE of its two iterations
    g = Mantle.Graph(dev)
    Mantle.repeat!(g, 2, trips) do i
        Mantle.compute!(g, "bump-$i") do p
            Mantle.use(p, out; read = true, write = true)
            Mantle.use(p, count; read = true)
            Mantle.dispatch!(p, dip_bump!, (out, count), Mantle.DeviceRange(count); group = 64)
        end
    end
    pl = Mantle.record!(Base.invokelatest(Mantle.Plan, g))

    # One slot per iteration, in pass order.
    slots = Int[]
    for pp in pl.passes, d in pp.dispatches
        k = Mantle.indirectindex(d)
        k == 0 || push!(slots, k)
    end
    @test length(slots) == 2

    # A sentinel no prepare would write: whatever the slot holds after the run
    # is what the command processor read.
    sentinel = UInt32(0xFFFF0000)
    for k in slots
        KA.fill!(pl.args.indirect[k], sentinel)
    end
    KA.synchronize(Mantle.backend(dev))

    Mantle.run!(pl)
    Mantle.waitfor!(pl)

    first_slot = Array(pl.args.indirect[slots[1]])
    second_slot = Array(pl.args.indirect[slots[2]])
    @test first_slot[1] == UInt32(cld(n, 64))     # the iteration that ran: its groups
    @test first_slot[2:3] == UInt32[1, 1]
    # THE assertion: the discarded iteration's prepare ran and wrote zero. Before
    # the fix the slot still held the sentinel, which a trace would have read.
    @test second_slot[1] == UInt32(0)
    @test second_slot[2:3] == UInt32[1, 1]
    # And the body of the discarded iteration did not run: one bump, not two.
    @test Array(Mantle.storage(out)) == fill(Int32(1), n)
    Mantle.free!(pl)
end
