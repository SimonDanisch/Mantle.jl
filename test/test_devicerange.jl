# `DeviceRange` on a real device.
#
# The Host backend covers the CORE half — that the count is read at launch and
# registered as an `Indirect` usage. What it cannot cover is the half that only
# a GPU has: the count never crosses to the host at all. Lava compiles the
# kernel against a ceiling so `__validindex` lets every thread through, converts
# the element count to workgroup counts ON the device, and dispatches off that.
#
# So the assertion that matters here is not "it ran" but "it ran over exactly
# the count the device computed, having never been told that count". A kernel
# that writes one element per invocation makes that visible: the number of
# elements written IS the ndrange.

using Test
import Mantle
const M = Mantle
# The backend this run is for. `runtests.jl` includes this file once per
# available backend (`Mantle.eachbackend()`); a bare `include` from the REPL
# gets the default one. Nothing below names a backend, which is the point:
# these testsets check PORTABLE behaviour, on whichever backend is there.
const TESTBACKEND = isdefined(Main, :MANTLE_TEST_BACKEND) ?
    Main.MANTLE_TEST_BACKEND : M.defaultbackend()

using KernelAbstractions: @kernel, @index, @Const

@kernel function dr_setcount!(n, @Const(src), thresh::Float32)
    i = @index(Global)
    @inbounds if i == 1
        c = Int32(0)
        for k in eachindex(src)
            src[k] > thresh && (c += Int32(1))
        end
        n[1] = c
    end
end

# A kernel dispatched over a `DeviceRange` bounds ITSELF: the dispatch covers
# whole workgroups, so the invocations past the count are real. This is the
# shape every Hikari wavefront kernel already has (`i <= queue.size[1]`).
@kernel function dr_mark!(dst, n)
    i = @index(Global)
    @inbounds if i <= n[1]
        dst[i] = 1.0f0
    end
end

# Deliberately unguarded, to pin what the tail actually does rather than leave
# it to be rediscovered.
@kernel function dr_mark_unguarded!(dst)
    i = @index(Global)
    @inbounds dst[i] = 1.0f0
end

@kernel function dr_zero!(dst)
    i = @index(Global)
    @inbounds dst[i] = 0.0f0
end

@testset "DeviceRange dispatches over a count the host never sees" begin
    dev = M.Device(TESTBACKEND)
    cap = 4096
    for want in (1, 777, 4096)
        g = M.Graph(dev)
        # `want` values above the threshold, the rest below.
        src = M.Buffer(dev, Float32[k <= want ? 1.0f0 : 0.0f0 for k in 1:cap])
        n = M.Buffer(dev, Int32[0])
        dst = M.Transient.Buffer(g, Float32, cap)

        # A transient's bytes are whatever the arena last held, so the count
        # below has to be of marks this run made.
        M.dispatch!(g, dr_zero!, (dst,), cap; name = "zero")
        M.dispatch!(g, dr_setcount!, (n,
                                          src, 0.5f0), 1; name = "count")
        M.dispatch!(g, dr_mark!, (dst,
                                      n),
                        M.DeviceRange(n; max = cap), name = "mark")

        plan = M.record!(M.Plan(g))
        M.run!(plan)

        @test Array(n)[1] == Int32(want)
        # Exactly `want` elements marked: the dispatch covered the device's
        # count, not the ceiling it was compiled against.
        marked = count(==(1.0f0), Array(M.storage(dst)))
        @test marked == want
    end
end

@testset "the count is ordered before the dispatch that reads it" begin
    dev = M.Device(TESTBACKEND)
    g = M.Graph(dev)
    src = M.Buffer(dev, fill(1.0f0, 64))
    n = M.Buffer(dev, Int32[0])
    dst = M.Transient.Buffer(g, Float32, 64)
    M.dispatch!(g, dr_zero!, (dst,), 64; name = "zero")
    M.dispatch!(g, dr_setcount!, (n,
                                      src, 0.5f0), 1; name = "count")
    M.dispatch!(g, dr_mark!, (dst,
                                  n),
                    M.DeviceRange(n; max = 64), name = "mark")
    # Declared, not hoped for: the second pass reads `n` as `Indirect`, which is
    # the edge that orders it after the pass that wrote it. Without it the
    # dispatch would size itself off whatever the buffer held last frame.
    marks = only(filter(p -> any(u -> last(u) === M.Indirect, p.usages), g.passes))
    @test marks !== nothing

    M.run!(M.record!(M.Plan(g)))
    @test count(==(1.0f0), Array(M.storage(dst))) == 64
end

@testset "the tail is whole workgroups, and the guard is the kernel's job" begin
    # 777 elements with a group of 64 launches 832 invocations. Unguarded, all
    # 832 write — which is why `DeviceRange`'s docstring says the kernel must
    # bound itself, and why every wavefront kernel in Hikari already does.
    dev = M.Device(TESTBACKEND)
    cap, want, group = 4096, 777, 64
    g = M.Graph(dev)
    src = M.Buffer(dev, Float32[k <= want ? 1.0f0 : 0.0f0 for k in 1:cap])
    n = M.Buffer(dev, Int32[0])
    dst = M.Transient.Buffer(g, Float32, cap)
    M.dispatch!(g, dr_zero!, (dst,), cap; name = "zero")
    M.dispatch!(g, dr_setcount!, (n,
                                      src, 0.5f0), 1; name = "count")
    M.dispatch!(g, dr_mark_unguarded!, (dst,),
                    M.DeviceRange(n; max = cap); group = group, name = "mark")
    pl = M.record!(M.Plan(g))
    M.run!(pl)
    # How many invocations ran, observed through a kernel that does NOT bound
    # itself — which is the only way to see it, and why this kernel is named
    # unguarded.
    #
    # The answer is a property of the compiled dispatch and both answers are
    # correct. One sized on the device reads the count there and launches whole
    # workgroups over it (Vulkan's indirect dispatch). One that is not launches
    # the CEILING, deliberately: resolving the range by reading the count on the
    # host means synchronising before every such dispatch, which measured 320
    # synchronises and 0.585 s of a 0.602 s frame on an M5. The kernel must
    # bound itself either way.
    #
    # Asked of the dispatch, `devicesized`: whether its backend kept the indirect
    # slot core gave it (`indirectindex`). It was
    # asked through `recordsplans` until 2026-09-26, which is a different
    # property: ROCm records a plan (as a HIP graph) and has no indirect dispatch,
    # so it launched the ceiling where the test expected whole workgroups.
    mark = only(only(pp.dispatches) for pp in pl.passes if pp.pass.name == "mark")
    launched = M.devicesized(mark) ? cld(want, group) * group : cap
    @test count(==(1.0f0), Array(M.storage(dst))) == launched
end

@kernel function dr_mark_at!(dst, n, at::Int32)
    i = @index(Global)
    @inbounds if i <= n[at]
        dst[i] = 1.0f0
    end
end

# Counters of several queues in one buffer, each dispatch sized by its own (`at`).
# Read as element 1, the dispatch over the second launched one workgroup for a
# count of 5 and marked 64 of its 777.
@testset "a DeviceRange counts element `at` of its buffer" begin
    dev = M.Device(TESTBACKEND)
    cap, counts = 4096, Int32[5, 777, 3]
    n = M.Buffer(dev, counts)
    for (at, ceiling) in ((2, cap), (3, cap), (2, nothing))
        g = M.Graph(dev)
        dst = M.Buffer(dev, zeros(Float32, cap))
        M.dispatch!(g, dr_mark_at!, (dst, n, Int32(at)), M.DeviceRange(n; max = ceiling, at);
                    group = 64, name = "mark")
        pl = M.Plan(g)
        M.recordable(dev, pl) && M.record!(pl)
        M.run!(pl)
        M.waitfor!(pl)
        @test count(==(1.0f0), Array(M.storage(dst))) == counts[at]
        M.free!(pl)
    end
    # An eager launch reads the same element.
    dst = M.Buffer(dev, zeros(Float32, cap))
    k = dr_mark_at!(TESTBACKEND, 64)
    k(M.storage(dst), M.storage(n), Int32(2); ndrange = M.DeviceRange(M.storage(n); max = cap, at = 2))
    M.waitidle(dev)
    @test count(==(1.0f0), Array(M.storage(dst))) == counts[2]
    @test_throws ArgumentError M.DeviceRange(n; at = 0)
end

# A plain KernelInterface kernel, which takes its group from `KI.launch_geometry`
# rather than KernelAbstractions' partition (that one already refuses a group with
# more axes than the range).
import KernelInterface as KI
function dr_kimark!(dst, n)
    i = KI.get_global_id().x
    i <= n[1] && (@inbounds dst[i] = 1.0f0)
    return nothing
end

# A device count is one-dimensional. With group = (8, 8) one backend turned 300
# items into 5 workgroups of 64 covering x = 1:40, and another into 38 workgroups
# of 8 x 8 that ran every item eight times: each backend read the group's size
# differently (product vs x extent). Refused at `dispatch!` on every backend now.
@testset "a DeviceRange group extends along x only" begin
    dev = M.Device(TESTBACKEND)
    cap, want = 4096, 300
    n = M.Buffer(dev, Int32[want])
    for group in ((8, 8), (1, 64))
        g = M.Graph(dev)
        dst = M.Buffer(dev, zeros(Float32, cap))
        @test_throws ArgumentError M.dispatch!(g, dr_kimark!, (dst, n),
                                               M.DeviceRange(n; max = cap); group)
    end
    # A trailing axis of one is still a row.
    g = M.Graph(dev)
    dst = M.Buffer(dev, zeros(Float32, cap))
    M.dispatch!(g, dr_kimark!, (dst, n), M.DeviceRange(n; max = cap);
                group = (64, 1), name = "mark")
    pl = M.Plan(g)
    M.recordable(dev, pl) && M.record!(pl)
    M.run!(pl)
    M.waitfor!(pl)
    @test count(==(1.0f0), Array(M.storage(dst))) == want
    M.free!(pl)
end

# Without `max` a `DeviceRange` kernel is compiled against core's ceiling, and the
# prepare sizes the launch from the device count without clamping it. The ceiling
# was 1M items, so a count of 1.5M launched every invocation and KA's index check
# against the compiled range silently dropped all past 1,048,576, on every backend.
@testset "a device count past any fixed ceiling runs whole" begin
    dev = M.Device(TESTBACKEND)
    want = 1_500_000
    dst = M.Buffer(dev, zeros(Float32, want))
    n = M.Buffer(dev, Int32[want])
    g = M.Graph(dev)
    M.dispatch!(g, dr_mark!, (dst, n), M.DeviceRange(n); group = 256, name = "mark")
    pl = M.Plan(g; budget = Inf)
    M.recordable(dev, pl) && M.record!(pl)
    M.run!(pl)
    M.waitfor!(pl)
    @test count(==(1.0f0), Array(M.storage(dst))) == want
    M.free!(pl)
end
