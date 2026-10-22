# Same-signature kernels and different parameters of one callable template must
# keep distinct pipeline entries. Partial writes expose reuse of the wrong variant.
using Test, Mantle, KernelAbstractions
include(joinpath(@__DIR__, "testbackend.jl"))
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
    backend = TESTBACKEND
    n, wg = 256, 32

    run(f) = begin
        o = KernelAbstractions.allocate(backend, Float32, n)
        fill!(o, 0.0f0)
        Mantle.kilaunch!(backend, f, o, n; ndrange = n, workgroupsize = wg)
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

    # One callable template with different parameters must keep distinct entries too.
    A = KernelAbstractions.allocate(backend,Float32,32); fill!(A,1f0)
    B = KernelAbstractions.allocate(backend,Float32,32,8); fill!(B,1f0)
    C = KernelAbstractions.allocate(backend,Float32,8)
    for nr in (4,8,4)
        fill!(C,-1f0)
        kern = Mantle.gemv_kcontig_kernel(nr,128,Mantle.subgroupwidth(A))
        Mantle.kilaunch!(backend,kern,C,A,B,32,8;ndrange=128,workgroupsize=128)
        got = Array(C)
        @test count(==(32f0),got) == nr
        @test count(==(-1f0),got) == 8-nr
    end
end
