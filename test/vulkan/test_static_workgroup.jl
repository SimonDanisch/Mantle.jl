"""
FIXED. A workgroup passed in the kernel's TYPE used to compute wrong global
indices when it had an INTERIOR unit extent; it no longer does, verified with
`MVE.WORKGROUP_FALLBACK` switched OFF across every shape below.

The fault was: with such a workgroup baked into the kernel's type, the block
index decode took the wrong divisor for dimension 2 and exactly
`min(1, blocks[3] / blocks[2])` of the output was written, silently. A *trailing*
unit extent was harmless, which is why ranks 1-3 looked clean. It was Lava's
codegen, not KernelAbstractions': the same kernel, workgroup and ndrange were
correct on `KA.CPU()`.

Found because `permutedims!` used the typed spelling: `ndrange = (72, 256, 8, 16)`
with `wg = (32, 4, 1, 1)` left 2 064 384 of 2 359 296 destination elements never
written, with no error anywhere.

This file previously asserted the fault was STILL PRESENT with the guard off, so
that nobody deleted a load-bearing guard. Those assertions now fail, which is how
the fix was noticed — the file was never registered in runtests.jl, so nobody ran
it. They are flipped to assert correctness.

OPEN: `MVE.WORKGROUP_FALLBACK` re-launches the affected shapes through the
dynamic path at roughly 2x the cost of the static one. With the fault gone it is
dead weight, but it has only been re-verified on one device (AMD 8060S / Windows).
Confirm on the other drivers before removing it.

The API rule still holds and is still worth following:

    kernel(backend, wg)(args...; ndrange)                      # typed
    kernel(backend)(args...; ndrange, workgroupsize = wg)      # keyword
    kernel(backend, wg, ndrange)(args...)                      # both-static
"""

using Test, Mantle, Lava, KernelAbstractions

# Bound by the suite's preamble for every file; standalone, bind them here.
@isdefined(MVE) || (MVE = Base.get_extension(Mantle, :MantleVulkanExt))
@isdefined(LavaBackend) || (LavaBackend = MVE.LavaBackend)
const KA = KernelAbstractions

@kernel function wgmark!(d)
    I = @index(Global, NTuple)
    @inbounds d[I...] = 1.0f0
end

"""
Fraction of `ndrange` actually written, launching `mode`.

`limit` raises the device's workgroup limit for the call, by swapping in a
modified [`Lava.DeviceCaps`](@ref). The characterisation testsets below
deliberately launch workgroups past the real limit — they are measuring how much
of the output survives, so a guard that refuses the launch would hide the very
thing they exist to pin. Every other launch in this file stays under it.
"""
function coverage(nd, wg, mode; fallback::Bool = true,
                  limit::Int = Lava.caps(MVE.vk_context()).workgrouplimit)
    ctx = MVE.vk_context()
    old = MVE.WORKGROUP_FALLBACK[]
    oldcaps = Lava.caps(ctx)
    MVE.WORKGROUP_FALLBACK[] = fallback
    ctx.caches.caps = Lava.DeviceCaps(oldcaps; workgrouplimit = limit)
    try
        return coverage_(nd, wg, mode)
    finally
        MVE.WORKGROUP_FALLBACK[] = old
        ctx.caches.caps = oldcaps
    end
end

function coverage_(nd, wg, mode)
    back = LavaBackend()
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

    # ── the fault itself, with the guard OFF — REPAIRED ───────────────────────
    #
    # These three testsets pinned the codegen fault precisely so a repair would
    # turn them red, and on 2026-09-04 they went red: every shape below covers
    # its full ndrange with the guard OFF (RADV, this tree — the submission
    # arc's emitter work, not a driver update; the same tree pinned the fault
    # on 2026-07-31). They now pin the REPAIRED behaviour.
    #
    # What that buys: `trailing_unit_ndrange`, `interior_unit_workgroup` and the
    # `WORKGROUP_FALLBACK` guard in `ka_backend.jl` are now deletable — a day
    # of the failure law holding would have been the reason to keep them, and
    # the law is gone. The deletion is left for a driver-matrix pass: the guard
    # on costs nothing and the repair is so far proven on one driver.
    #
    # Lava was *correct* through all of this — the guard was on by default, and
    # "WORKGROUP_FALLBACK makes the typed form correct everywhere" below covered
    # the shipped behaviour.

    @testset "workitems[3] == 1 was the trigger; no longer" begin
        # Sharpest form of the bug, repaired. At rank 4 the typed spelling was
        # wrong IFF the THIRD workgroup extent was 1; `wg[4]` was irrelevant.
        ND = (64, 256, 8, 2)
        for wg in ((32, 4, 2, 1), (32, 4, 4, 1))
            # 512 threads for the second: past the limit, and deliberately so.
            @test coverage(ND, wg, :typed; fallback = false, limit = 512) == 1.0
        end
        for wg in ((32, 4, 1, 1), (32, 4, 1, 2), (32, 1, 1, 1), (32, 8, 1, 1))
            @test coverage(ND, wg, :typed; fallback = false) == 1.0
        end
    end

    @testset "an INTERIOR unit extent was the trigger; no longer" begin
        # The same repair, generalised past rank 4.
        ND5 = (32, 128, 8, 4, 2)
        @test coverage(ND5, (16, 4, 1, 1, 1), :typed; fallback = false) == 1.0
        @test coverage(ND5, (16, 4, 2, 1, 1), :typed; fallback = false) == 1.0
        @test coverage(ND5, (16, 4, 2, 2, 1), :typed) == 1.0
    end

    @testset "the failure law no longer fires" begin
        # Was `min(1, b3/b2)` exactly — dimension 2 of the block grid decoded
        # with dimension 3's extent as its divisor. Now every cell is whole.
        law(b2, b3) = coverage((64, 4b2, b3, 2), (32, 4, 1, 1), :typed; fallback = false)
        for (b2, b3) in ((2, 1), (4, 2), (8, 1), (16, 4), (64, 8),
                         (2, 2), (4, 4), (8, 16), (2, 8))
            @test law(b2, b3) == 1.0
        end
    end

    @testset "WORKGROUP_FALLBACK makes the typed form correct everywhere" begin
        # The guard is the point of all the above. With it on, every shape that
        # used to be silently wrong is right, and nothing that worked changes.
        for (nd, wg) in (((64, 256, 8, 2), (32, 4, 1, 1)), ((64, 256, 8, 2), (32, 4, 1, 2)),
                         ((64, 256, 8, 2), (32, 1, 1, 1)), ((72, 256, 8, 16), (32, 4, 1, 1)),
                         ((72, 8, 1, 1), (32, 4, 1, 1)), ((32, 128, 8, 4, 2), (16, 4, 1, 1, 1)),
                         ((64, 256, 8, 2), (32, 4, 2, 1)), ((250, 250), (32, 4)),
                         ((72, 64, 64), (32, 4, 1)), ((1000,), (64,)))
            @test coverage(nd, wg, :typed) == 1.0
        end
    end

    @testset "the guard names exactly the dangerous shapes" begin
        @test MVE.interior_unit_workgroup((32, 4, 1, 1))
        @test MVE.interior_unit_workgroup((32, 4, 1, 2))
        @test MVE.interior_unit_workgroup((16, 4, 2, 1, 1))
        @test !MVE.interior_unit_workgroup((32, 4, 2, 1))     # trailing only
        @test !MVE.interior_unit_workgroup((32, 4, 1))        # trailing only
        @test !MVE.interior_unit_workgroup((32, 4))
        @test !MVE.interior_unit_workgroup((64,))
    end

    @testset "launchgroup fills the fast axis first" begin
        @test MVE.launchgroup((4, 4, 288, 1024)) == (4, 4, 16, 1)
        @test MVE.launchgroup((4096, 72, 8, 1)) == (256, 1, 1, 1)
        @test MVE.launchgroup((16, 72, 4, 1024)) == (16, 16, 1, 1)
        @test MVE.launchgroup((10,)) == (10,)
        for sz in ((4, 4, 288, 1024), (4096, 72, 8, 1), (16, 72, 4, 1024), (2, 2, 576, 1024))
            wg = MVE.launchgroup(sz)
            @test all(wg .<= sz)                    # never larger than the axis
            @test prod(wg) <= 256
        end
    end

    @testset "no shaping rule may exceed the thread budget" begin
        # The one invariant that is not a performance preference: a workgroup
        # larger than `maxComputeWorkGroupInvocations` does not run slowly, it
        # does not complete. `staticgroup` used to give every interior axis 2
        # threads unconditionally — `2^(N-2)`, i.e. 65 536 at rank 18 — and
        # GPUArrays' 18-d `permutedims` hung for the full 120 s flush timeout,
        # failing the 354 assertions that came after it.
        shapes = Any[]
        for n in 1:20
            push!(shapes, ntuple(d -> d == 1 ? 4 : 2, n))       # the hanging shape
            push!(shapes, ntuple(_ -> 2, n))
            push!(shapes, ntuple(d -> d == n ? 1024 : 3, n))
            push!(shapes, ntuple(d -> isodd(d) ? 1 : 64, n))
        end
        for sz in shapes, f in (MVE.launchgroup, MVE.staticgroup)
            wg = f(sz)
            @test prod(wg) <= 256
            @test all(wg .>= 1)
            @test all(wg .<= sz)
        end
    end

    @testset "high rank stays correct via the dynamic fallback" begin
        # Above the rank where 2-per-interior-axis fits, `staticgroup` has to
        # leave interior extents at 1 — which is precisely the shape
        # `WORKGROUP_FALLBACK` re-launches dynamically, so the result is still
        # right. Asserted here so the two rules stay aware of each other.
        for n in 12:18
            sz = ntuple(d -> d == 1 ? 4 : 2, n)
            wg = MVE.staticgroup(sz)
            @test prod(wg) <= 256
            @test MVE.interior_unit_workgroup(wg)   # so the fallback fires
        end
    end
end

# ── a SECOND trigger for the same silent fault ────────────────────────────────
#
# `interior_unit_workgroup` was written for a workgroup with an interior unit
# extent. It is not the only shape that miscompiles: a workgroup in the kernel's
# TYPE, on an ndrange whose **last extent is 1**, at rank >= 5, also writes part
# of its output and reports nothing. `(16,2,2,2,2,1)` has no interior unit extent,
# so the original guard never fired.
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
    backend = LavaBackend()
    written(sz, wg) = begin
        d = KernelAbstractions.allocate(backend, Float32, sz...)
        fill!(d, 0.0f0)
        markall!(backend, wg)(d; ndrange = sz)
        KernelAbstractions.synchronize(backend)
        count(==(1.0f0), Array(d)) / length(d)
    end

    @testset "the predicate matches the measured trigger" begin
        @test MVE.trailing_unit_ndrange((64, 4, 4, 4, 1))          # rank 5, fires
        @test MVE.trailing_unit_ndrange((576, 16, 16, 4, 4, 1))    # rank 6, fires
        @test !MVE.trailing_unit_ndrange((64, 4, 4, 4, 4, 2))      # last extent 2
        @test !MVE.trailing_unit_ndrange((256, 256, 144, 1))       # rank 4 is unaffected
        @test !MVE.trailing_unit_ndrange((64, 4, 4))
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

    @testset "the fault is repaired underneath" begin
        # The guard was load-bearing, not decorative: with it off, the same
        # launch lost data. On 2026-09-04 (RADV, this tree) it stopped failing
        # — the codegen fault is fixed, and `trailing_unit_ndrange` can go the
        # day the driver matrix says so (see "the fault itself — REPAIRED"
        # above). This pins the repair: full coverage with the guard OFF.
        prev = MVE.WORKGROUP_FALLBACK[]
        try
            MVE.WORKGROUP_FALLBACK[] = false
            @test written((576, 16, 16, 4, 4, 1), (16, 2, 2, 2, 2, 1)) == 1.0
            @test written((64, 4, 4, 4, 1), (16, 2, 2, 2, 1)) == 1.0
        finally
            MVE.WORKGROUP_FALLBACK[] = prev
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
