"""
A workgroup passed in the kernel's TYPE computes the right global indices for
every shape below, including the ones with an INTERIOR unit extent.

What that guards against: with such a workgroup baked into the kernel's type, a
block index decode taking the wrong divisor for dimension 2 writes exactly
`min(1, blocks[3] / blocks[2])` of the output, silently. A *trailing* unit
extent is harmless, which is why ranks 1-3 look clean, and it is a codegen
fault rather than a KernelAbstractions one: the same kernel, workgroup and
ndrange are correct on `KA.CPU()`. `permutedims!` uses the typed spelling, so
`ndrange = (72, 256, 8, 16)` with `wg = (32, 4, 1, 1)` left 2 064 384 of
2 359 296 destination elements never written, with no error anywhere.

The Vulkan backend guards the affected shapes (`WORKGROUP_FALLBACK` in
`src/vulkan/array/ka_backend.jl` re-launches them through the dynamic path, at
roughly 2x the cost of the static one) because NVIDIA's driver still decodes them
wrong with the guard off — measured 2026-10-06 on an RTX 4000 Ada (595.99) and an
RTX 3070 Laptop (595.91), the same `min(1, b3/b2)` law, while lavapipe and RADV
run the same kernel exactly. This file asserts what a caller sees: every shape
covers its whole ndrange, whatever the backend does to get there. Whether the
guard can go is a question about one compiler and one driver, not about the
portable behaviour, and is not asked here.

The API rule still holds and is still worth following:

    kernel(backend, wg)(args...; ndrange)                      # typed
    kernel(backend)(args...; ndrange, workgroupsize = wg)      # keyword
    kernel(backend, wg, ndrange)(args...)                      # both-static
"""

using Test, Mantle, KernelAbstractions
include(joinpath(@__DIR__, "testbackend.jl"))
const KA = KernelAbstractions

@kernel function wgmark!(d)
    I = @index(Global, NTuple)
    @inbounds d[I...] = 1.0f0
end

"Fraction of `ndrange` actually written, launching `mode`."
function coverage(nd, wg, mode)
    back = TESTBACKEND
    d = KA.allocate(back, Float32, nd...)
    fill!(d, 0.0f0)
    if mode === :typed
        wgmark!(back, wg)(d; ndrange = nd)
    elseif mode === :keyword
        wgmark!(back)(d; ndrange = nd, workgroupsize = wg)
    else
        wgmark!(back, wg, nd)(d)
    end
    KA.synchronize(back)
    n = count(==(1.0f0), Array(d))
    d = nothing
    GC.gc()
    return n / prod(nd)
end

@testset "workgroup size: launch keyword vs kernel type" begin
    limit = Mantle.caps(TESTBACKEND).workgrouplimit

    @testset "the keyword form is correct at every rank" begin
        for (nd, wg) in (((1000,), (64,)),
                         ((250, 250), (32, 4)),
                         ((72, 64, 64), (32, 4, 1)),
                         ((72, 256, 8, 16), (32, 4, 1, 1)),   # rank 4: the failing shape
                         ((72, 8, 1, 1), (32, 4, 1, 1)),
                         ((8, 8, 8, 8, 8), (8, 1, 1, 1, 1)))
            @test coverage(nd, wg, :keyword) == 1.0
        end
    end

    @testset "both-static is correct too" begin
        @test coverage((72, 256, 8, 16), (32, 4, 1, 1), :bothstatic) == 1.0
    end

    @testset "the typed form is correct at every rank" begin
        @test coverage((1000,), (64,), :typed) == 1.0
        @test coverage((250, 250), (32, 4), :typed) == 1.0
        @test coverage((72, 64, 64), (32, 4, 1), :typed) == 1.0
        @test coverage((8, 8, 8, 8, 8), (8, 1, 1, 1, 1), :typed) == 1.0
    end

    @testset "workitems[3] == 1 is covered" begin
        # Sharpest form of the fault: at rank 4 the typed spelling was wrong IFF
        # the THIRD workgroup extent is 1, and `wg[4]` is irrelevant.
        ND = (64, 256, 8, 2)
        for wg in ((32, 4, 2, 1), (32, 4, 4, 1),
                   (32, 4, 1, 1), (32, 4, 1, 2), (32, 1, 1, 1), (32, 8, 1, 1))
            # `(32, 4, 4, 1)` is 512 work-items; a device that cannot launch it
            # has nothing to show here.
            prod(wg) <= limit || continue
            @test coverage(ND, wg, :typed) == 1.0
        end
    end

    @testset "an INTERIOR unit extent is covered too" begin
        # The same, generalised past rank 4.
        ND5 = (32, 128, 8, 4, 2)
        @test coverage(ND5, (16, 4, 1, 1, 1), :typed) == 1.0
        @test coverage(ND5, (16, 4, 2, 1, 1), :typed) == 1.0
        @test coverage(ND5, (16, 4, 2, 2, 1), :typed) == 1.0
    end

    @testset "the failure law does not fire" begin
        # The law was `min(1, b3/b2)` exactly — dimension 2 of the block grid
        # decoded with dimension 3's extent as its divisor. Every cell is whole,
        # including the ones with b3 < b2 where the law took output away.
        law(b2, b3) = coverage((64, 4b2, b3, 2), (32, 4, 1, 1), :typed)
        for (b2, b3) in ((2, 1), (4, 2), (8, 1), (16, 4), (64, 8),
                         (2, 2), (4, 4), (8, 16), (2, 8))
            @test law(b2, b3) == 1.0
        end
    end

    @testset "the typed form is correct on every shape that was wrong" begin
        for (nd, wg) in (((64, 256, 8, 2), (32, 4, 1, 1)), ((64, 256, 8, 2), (32, 4, 1, 2)),
                         ((64, 256, 8, 2), (32, 1, 1, 1)), ((72, 256, 8, 16), (32, 4, 1, 1)),
                         ((72, 8, 1, 1), (32, 4, 1, 1)), ((32, 128, 8, 4, 2), (16, 4, 1, 1, 1)),
                         ((64, 256, 8, 2), (32, 4, 2, 1)), ((250, 250), (32, 4)),
                         ((72, 64, 64), (32, 4, 1)), ((1000,), (64,)))
            @test coverage(nd, wg, :typed) == 1.0
        end
    end

    @testset "launchgroup fills the fast axis first" begin
        @test Mantle.launchgroup((4, 4, 288, 1024)) == (4, 4, 16, 1)
        @test Mantle.launchgroup((4096, 72, 8, 1)) == (256, 1, 1, 1)
        @test Mantle.launchgroup((16, 72, 4, 1024)) == (16, 16, 1, 1)
        @test Mantle.launchgroup((10,)) == (10,)
        for sz in ((4, 4, 288, 1024), (4096, 72, 8, 1), (16, 72, 4, 1024), (2, 2, 576, 1024))
            wg = Mantle.launchgroup(sz)
            @test all(wg .<= sz)                    # never larger than the axis
            @test prod(wg) <= 256
        end
    end

    @testset "no shaping rule may exceed the thread budget" begin
        # The one invariant that is not a performance preference: a workgroup
        # larger than the device's workgroup limit does not run slowly, it
        # does not complete. `staticgroup` giving every interior axis 2 threads
        # unconditionally is `2^(N-2)`, i.e. 65 536 at rank 18, and GPUArrays'
        # 18-d `permutedims` then hung for the full 120 s flush timeout,
        # failing the 354 assertions that came after it.
        shapes = Any[]
        for n in 1:20
            push!(shapes, ntuple(d -> d == 1 ? 4 : 2, n))       # the hanging shape
            push!(shapes, ntuple(_ -> 2, n))
            push!(shapes, ntuple(d -> d == n ? 1024 : 3, n))
            push!(shapes, ntuple(d -> isodd(d) ? 1 : 64, n))
        end
        for sz in shapes, f in (Mantle.launchgroup, Mantle.staticgroup)
            wg = f(sz)
            @test prod(wg) <= 256
            @test all(wg .>= 1)
            @test all(wg .<= sz)
        end
    end
end

# ── a SECOND trigger for the same silent fault ────────────────────────────────
#
# An interior unit extent is not the only shape that miscompiled: a workgroup in
# the kernel's TYPE, on an ndrange whose **last extent is 1**, at rank >= 5, also
# wrote part of its output and reported nothing. `(16,2,2,2,2,1)` has no interior
# unit extent, so the guard written for the first trigger never fired.
#
# Found through `permutedims!`, which returned wrong data for SAM 2's rank-6
# window-partition shapes — `(576,16,4,16,4,1)` and friends — while the ordinary
# broadcast of the same permutation stayed correct, because a broadcast uses a
# linear ndrange and never reaches this path. Measured written fractions were
# 0.500 and 0.062 on shapes differing only in extents, so the trigger is pinned
# here and no law is claimed about how much survives.
@kernel cpu=false function markall!(d)
    I = @index(Global, NTuple)
    @inbounds d[I...] = 1.0f0
end

@testset "trailing unit ndrange" begin
    backend = TESTBACKEND
    written(sz, wg) = begin
        d = KernelAbstractions.allocate(backend, Float32, sz...)
        fill!(d, 0.0f0)
        markall!(backend, wg)(d; ndrange = sz)
        KernelAbstractions.synchronize(backend)
        count(==(1.0f0), Array(d)) / length(d)
    end

    @testset "every element is written" begin
        for (sz, wg) in [((576, 16, 16, 4, 4, 1), (16, 2, 2, 2, 2, 1)),
                         ((288, 4, 4, 32, 32, 1), (16, 2, 2, 2, 2, 1)),
                         ((64, 4, 4, 4, 4, 1),    (16, 2, 2, 2, 2, 1)),
                         ((64, 4, 4, 4, 1),       (16, 2, 2, 2, 1)),
                         ((64, 4, 4, 4, 4, 2),    (16, 2, 2, 2, 2, 1))]   # control
            @test written(sz, wg) == 1.0
        end
    end

    @testset "permutedims! is correct on the shapes that found this" begin
        for (sz, perm) in [((576, 16, 4, 16, 4, 1), (1, 2, 4, 3, 5, 6)),
                           ((288, 4, 32, 4, 32, 1), (1, 2, 4, 3, 5, 6)),
                           ((72, 8, 256, 16),       (1, 3, 2, 4))]
            h = reshape(collect(Float32.(1:prod(sz))), sz)
            src = KernelAbstractions.allocate(backend, Float32, sz...)
            copyto!(src, h)
            dest = KernelAbstractions.allocate(backend, Float32,
                                               ntuple(d -> sz[perm[d]], length(sz))...)
            permutedims!(dest, src, perm)
            KernelAbstractions.synchronize(backend)
            @test Array(dest) == permutedims(h, perm)
        end
    end
end
