"""
Two kernels are two pipelines, even when nothing about their arguments differs.

`launch_plan` caches a compiled pipeline under `typeof(all_args)`. That was
sufficient for as long as every kernel came from `@kernel`, whose launch puts a
`CompilerMetadata` in `all_args[1]` — but a macro-free `KernelInterface` kernel
is a plain function and that slot is `nothing`. Two such kernels with the same
signature and the same workgroup size then shared one cache entry, and whichever
compiled second was handed the first one's pipeline.

It is silent, and it is not silent in a small way. Mantle's own `gemv` generates
one kernel per `(nrows, block, subgroup)`, all of them
`(C, A, B, K::Int, N::Int)`: `gemvk_8_128_64` ran `gemvk_4_128_64`'s pipeline,
which writes four outputs per workgroup instead of eight, so **half the result
was never written** and came back as whatever the caller had left in it. It
reproduced only once both kernels had been compiled, which is what made it look
like a launch-ordering bug rather than a cache collision.

The two kernels below are the smallest thing that has that shape: identical
signatures, identical launch geometry, different bodies. If the key ever loses
the kernel again, the second assertion reads `1` where it wants `2`.
"""

using Test, Mantle, KernelAbstractions
const KI = Mantle.KI

# Same signature, same workgroup size, different constant written.
function lpk_writeone!(out, n::Int)
    i = KI.get_global_id().x
    i <= n || return nothing
    @inbounds out[i] = 1.0f0
    return nothing
end

function lpk_writetwo!(out, n::Int)
    i = KI.get_global_id().x
    i <= n || return nothing
    @inbounds out[i] = 2.0f0
    return nothing
end

# The gemv shape exactly: what a workgroup writes depends on a value baked into
# the kernel, so running the wrong one leaves part of the output untouched.
function lpk_writefew!(out, n::Int)
    i = KI.get_global_id().x
    KI.get_local_id().x <= 4 || return nothing
    i <= n || return nothing
    @inbounds out[i] = 1.0f0
    return nothing
end

function lpk_writemany!(out, n::Int)
    i = KI.get_global_id().x
    KI.get_local_id().x <= 8 || return nothing
    i <= n || return nothing
    @inbounds out[i] = 1.0f0
    return nothing
end

@testset "a launch plan is keyed on its kernel, not only its arguments" begin
    backend = LavaBackend()
    n, wg = 256, 32

    run(f) = begin
        o = KernelAbstractions.allocate(backend, Float32, n)
        fill!(o, 0.0f0)
        KI.Kernel(backend, f)(o, n; ndrange = n, workgroupsize = wg)
        KernelAbstractions.synchronize(backend)
        Array(o)
    end

    # Order matters: the second compile is the one that used to collide, so the
    # assertion that catches the bug is the second.
    @test all(==(1.0f0), run(lpk_writeone!))
    @test all(==(2.0f0), run(lpk_writetwo!))
    # …and back, in case the collision went the other way.
    @test all(==(1.0f0), run(lpk_writeone!))

    # The partial-write shape. `writefew` leaves 4 of every 32 written and
    # `writemany` 8, so a collision shows up as a count and not as a wrong value
    # — which is how it presented in `gemv`.
    few = run(lpk_writefew!)
    many = run(lpk_writemany!)
    @test count(==(1.0f0), few) == 4 * (n ÷ wg)
    @test count(==(1.0f0), many) == 8 * (n ÷ wg)
    @test count(==(1.0f0), many) == 2 * count(==(1.0f0), few)
end
