# Successive runs do not overlap, across PLANS.
#
# A core requirement that no layer stated. Core derives dependencies WITHIN a
# frame — `passbarriers` into `emitbarriers!` — and stops at the command buffer
# boundary. Across submissions each backend was getting the ordering by a
# different accident: Metal from the `useResource` declaration plus an in-order
# queue, Vulkan from submission order with no `waits` on its `submit!`. Neither
# wrote it down, so on Metal the one call providing it could be deleted, the whole
# suite still passed, and it showed up as a wrong picture in a real scene.
#
# ONE plan accumulating into itself does NOT catch this — measured, correct at 64
# runs with the declaration removed. Two plans do: A writes a buffer, B reads it,
# alternating with no wait between them. If the submissions overlap, B reads `mid`
# while A is already writing the NEXT round into it, and those elements accumulate
# the wrong `k`. Verified by deleting that call from the source and running this
# in a fresh session: 1,027,648 wrong elements out of 2^20. It goes wrong in BOTH
# directions — 2076 against a wanted 2080 there, 2095 in another run — so bound it
# on each side rather than assuming which.
#
# Check EVERY element. The first one was correct in a failing run; only the spread
# showed it.

using Test
import Mantle
const M = Mantle
const TESTBACKEND = isdefined(Main, :MANTLE_TEST_BACKEND) ?
    Main.MANTLE_TEST_BACKEND : M.defaultbackend()

using KernelAbstractions: @kernel, @index, @Const

@kernel function ord_fill!(dst, k)
    i = @index(Global)
    @inbounds dst[i] = k[1]
end

@kernel function ord_accumulate!(acc, @Const(src))
    i = @index(Global)
    @inbounds acc[i] += src[i]
end

@testset "successive runs of two plans do not overlap" begin
    dev = M.Device(TESTBACKEND)
    # Large enough that a frame is real GPU work and there is room to overlap.
    n = 1 << 20
    mid = M.Buffer(dev, zeros(Float32, n))
    acc = M.Buffer(dev, zeros(Float32, n))
    k   = M.GPURef(dev, 1.0f0)

    ga = M.Graph(dev)
    M.compute!(ga, "write") do p
        M.dispatch!(p, ord_fill!, (M.use(p, mid; write = true),
                                   M.use(p, k; read = true)), n)
    end
    gb = M.Graph(dev)
    M.compute!(gb, "read") do p
        M.dispatch!(p, ord_accumulate!, (M.use(p, acc; read = true, write = true),
                                         M.use(p, mid; read = true)), n)
    end
    pa = M.record!(M.Plan(ga))
    pb = M.record!(M.Plan(gb))

    nrounds = 64                      # 16 is too few to overlap; 64 reproduces
    M.update!(acc, zeros(Float32, n))
    M.waitfor!(pb)
    for i in 1:nrounds
        k[] = Float32(i)
        M.run!(pa)
        M.run!(pb)                    # NO wait between them: that is the test
    end
    M.waitfor!(pb)

    want = Float32(sum(1:nrounds))
    got = Array(M.storage(acc))
    @test minimum(got) == want
    @test maximum(got) == want        # the failing case reads HIGH, not low
    @test count(!=(want), got) == 0
end
