"""
Elide the inter-dispatch barrier when two dispatches cannot alias.

`bq.barrier_elision` compares the device address ranges a dispatch's
arguments occupy against everything touched since the last barrier. Disjoint
ranges cannot alias, so no memory dependency exists and the barrier is dropped —
which needs no read/write annotation to be sound, and that is the whole point.

The risk being tested is the obvious one: eliding a barrier that was load-bearing
gives a read-before-write race that usually still produces the right answer. So
the dependent chain here is long, and every stage's result depends on every
previous stage, which is the shape that fails loudly when a barrier goes missing.
"""

using Test, KernelAbstractions, Lava
const KA = KernelAbstractions

@kernel function chain!(dst, @Const(src))
    i = @index(Global, Linear)
    @inbounds dst[i] = src[i] + 1.0f0
end

@kernel function selfadd!(a, c)
    i = @index(Global, Linear)
    @inbounds a[i] = a[i] + c
end

@testset "barrier elision" begin
    be = LavaBackend()
    n = 8192
    kc = chain!(be, 256)
    ks = selfadd!(be, 256)

    # A ping-pong chain: every dispatch reads what the previous one wrote, so
    # not one barrier here is elidable.
    x = KA.allocate(be, Float32, n)
    y = KA.allocate(be, Float32, n)
    depchain() = begin
        fill!(x, 0.0f0)
        for _ in 1:25
            kc(y, x; ndrange = n)
            kc(x, y; ndrange = n)
        end
    end

    depchain(); KA.synchronize(be)
    ref = collect(x)
    @test all(ref .== 50.0f0)

    Mantle.vk_context().default_bq.barrier_elision = true
    try
        for trial in 1:10
            depchain()
            KA.synchronize(be)
            got = collect(x)
            @test got == ref
        end
    finally
        Mantle.vk_context().default_bq.barrier_elision = false
    end

    # Independent buffers: nothing aliases, so every barrier after the first is
    # elidable and the results must still be exact.
    bufs = [KA.allocate(be, Float32, n) for _ in 1:8]
    for b in bufs; fill!(b, 0.0f0); end
    KA.synchronize(be)
    Mantle.vk_context().default_bq.barrier_elision = true
    try
        for _ in 1:4, b in bufs
            ks(b, 1.0f0; ndrange = n)
        end
        KA.synchronize(be)
    finally
        Mantle.vk_context().default_bq.barrier_elision = false
    end
    @test all(all(collect(b) .== 4.0f0) for b in bufs)

    # Mixed: a dependent chain on `x`/`y` interleaved with untouched buffers.
    fill!(x, 0.0f0); KA.synchronize(be)
    Mantle.vk_context().default_bq.barrier_elision = true
    try
        for i in 1:25
            kc(y, x; ndrange = n)
            ks(bufs[(i % 8) + 1], 1.0f0; ndrange = n)
            kc(x, y; ndrange = n)
        end
        KA.synchronize(be)
    finally
        Mantle.vk_context().default_bq.barrier_elision = false
    end
    @test collect(x) == ref
end
