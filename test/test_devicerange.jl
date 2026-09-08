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
import Mantle, Lava
const M = Mantle
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
    dev = M.Device(M.VulkanAPI())
    cap = 4096
    for want in (1, 777, 4096)
        g = M.Graph(dev)
        # `want` values above the threshold, the rest below.
        src = M.Buffer(dev, Float32[k <= want ? 1.0f0 : 0.0f0 for k in 1:cap])
        n = M.Buffer(dev, Int32[0])
        dst = M.Transient.Buffer(g, Float32, cap)

        # A transient's bytes are whatever the arena last held, so the count
        # below has to be of marks this run made.
        M.compute!(g, "zero") do p
            M.dispatch!(p, dr_zero!, (M.use(p, dst; write = true),), cap)
        end
        M.compute!(g, "count") do p
            M.dispatch!(p, dr_setcount!, (M.use(p, n; write = true),
                                          M.use(p, src; read = true), 0.5f0), 1)
        end
        M.compute!(g, "mark") do p
            M.dispatch!(p, dr_mark!, (M.use(p, dst; write = true),
                                      M.use(p, n; read = true)),
                        M.DeviceRange(n; max = cap))
        end

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
    dev = M.Device(M.VulkanAPI())
    g = M.Graph(dev)
    src = M.Buffer(dev, fill(1.0f0, 64))
    n = M.Buffer(dev, Int32[0])
    dst = M.Transient.Buffer(g, Float32, 64)
    M.compute!(g, "zero") do p
        M.dispatch!(p, dr_zero!, (M.use(p, dst; write = true),), 64)
    end
    M.compute!(g, "count") do p
        M.dispatch!(p, dr_setcount!, (M.use(p, n; write = true),
                                      M.use(p, src; read = true), 0.5f0), 1)
    end
    M.compute!(g, "mark") do p
        M.dispatch!(p, dr_mark!, (M.use(p, dst; write = true),
                                  M.use(p, n; read = true)),
                    M.DeviceRange(n; max = 64))
    end
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
    dev = M.Device(M.VulkanAPI())
    cap, want, group = 4096, 777, 64
    g = M.Graph(dev)
    src = M.Buffer(dev, Float32[k <= want ? 1.0f0 : 0.0f0 for k in 1:cap])
    n = M.Buffer(dev, Int32[0])
    dst = M.Transient.Buffer(g, Float32, cap)
    M.compute!(g, "zero") do p
        M.dispatch!(p, dr_zero!, (M.use(p, dst; write = true),), cap)
    end
    M.compute!(g, "count") do p
        M.dispatch!(p, dr_setcount!, (M.use(p, n; write = true),
                                      M.use(p, src; read = true), 0.5f0), 1)
    end
    M.compute!(g, "mark") do p
        M.dispatch!(p, dr_mark_unguarded!, (M.use(p, dst; write = true),),
                    M.DeviceRange(n; max = cap); group = group)
    end
    M.run!(M.record!(M.Plan(g)))
    @test count(==(1.0f0), Array(M.storage(dst))) == cld(want, group) * group
end
