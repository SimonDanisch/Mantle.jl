# A plain kernel runs whole workgroups, and guards its own tail.
#
# `@kernel` inserted `if index <= ndrange` for every kernel it built. A plain
# function over KernelInterface's intrinsics gets nothing inserted: the launch is
# `cld(ndrange, group)` groups, and the lanes past the ndrange run the body too.
# When Mantle's last `@kernel`s became plain functions, eight kept their bodies
# and lost that guard, so each wrote past its output whenever the ndrange was not
# a multiple of the group — the default group is 1024, so any size above that.
# Measured on `fill!`: 1500 elements wrote 548 more, into the first 512 of
# whatever array the pool had placed next.
#
# Each kernel below gets a window onto the front of a larger buffer, an ndrange
# the group does not divide, and a canary in the rest. The canary is what a lane
# past the ndrange would overwrite, and the only thing asserted: what these
# kernels compute is tested where they are.

using Test, Mantle
using Mantle: viewof
using GeometryBasics: Vec3f

# The backend this run is for: `runtests.jl` includes the file once per backend
# `Mantle.eachbackend()` reports (`foreachbackend`), and a bare `include` gets
# the default one. Every launch pads to whole workgroups, so every backend has
# the lanes this is about.
const TAILS_BACKEND = isdefined(Main, :MANTLE_TEST_BACKEND) ?
    Main.MANTLE_TEST_BACKEND : Mantle.defaultbackend()

# `n` elements the kernel gets, `pad` after them it must not touch, all `v`.
tailed(dev, n, pad, v) = (p = Mantle.Buffer(dev, fill(v, n + pad)); (p, viewof(p, n)))

tail(p, n) = Array(Mantle.storage(p))[(n + 1):end]

# 64 divides none of the ndranges below.
function runtails!(dev, f, args, n; group = 64)
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, f, args, n; group)
    Mantle.runonce!(g)
    return nothing
end

@testset "a plain kernel writes nothing past its ndrange — $(nameof(typeof(TAILS_BACKEND)))" begin
    dev = Mantle.todevice(TAILS_BACKEND)
    canary = ComplexF32(-12345, 678)

    @testset "fill_kernel!" begin
        n = 99
        p, a = tailed(dev, n, 64, 1f0)
        runtails!(dev, Mantle.fill_kernel!, (a, 7f0), n)
        got = Array(Mantle.storage(p))
        @test all(==(7f0), got[1:n])
        @test all(==(1f0), tail(p, n))
        Mantle.free!(p)
    end

    @testset "the real-FFT kernels" begin
        N, nb = 40, 3                    # 60, 63, 120 and 115 lanes: none a multiple of 64
        H = N ÷ 2
        # pack: H * nb complex out of 2H * nb reals
        src = Mantle.Buffer(dev, ones(Float32, N * nb))
        p, dst = tailed(dev, H * nb, 64, canary)
        runtails!(dev, Mantle.rfft_pack_kernel!, (dst, src), H * nb)
        @test all(==(canary), tail(p, H * nb))
        Mantle.free!(p)
        # post: (H + 1) * nb bins out of the half-length spectrum
        Z = Mantle.Buffer(dev, ones(ComplexF32, H * nb))
        p, dst = tailed(dev, (H + 1) * nb, 64, canary)
        runtails!(dev, Mantle.rfft_post_kernel!, (dst, Z, Val(N), Val(-1), 1f0), (H + 1) * nb)
        @test all(==(canary), tail(p, (H + 1) * nb))
        Mantle.free!(p)
        # extend: N * nb complex out of the NB-bin half spectrum
        NB = H + 1
        half = Mantle.Buffer(dev, ones(ComplexF32, NB * nb))
        p, dst = tailed(dev, N * nb, 64, canary)
        runtails!(dev, Mantle.irfft_extend_kernel!, (dst, half, Val(N), Val(NB)), N * nb)
        @test all(==(canary), tail(p, N * nb))
        Mantle.free!(p)
        # real: N * nb reals out of the inverse transform
        full = Mantle.Buffer(dev, ones(ComplexF32, N * nb))
        p, dst = tailed(dev, N * nb - 5, 64, -3f0)
        runtails!(dev, Mantle.irfft_real_kernel!, (dst, full, 0.5f0), N * nb - 5)
        @test all(==(-3f0), tail(p, N * nb - 5))
        Mantle.free!(p)
        foreach(Mantle.free!, (src, Z, half, full))
    end

    @testset "stft_frames_kernel!" begin
        nfft, frames, hop = 16, 7, 4
        x = Mantle.Buffer(dev, collect(Float32, 1:40))
        w = Mantle.Buffer(dev, ones(Float32, nfft))
        p, out = tailed(dev, nfft * frames, 64, -3f0)
        runtails!(dev, Mantle.stft_frames_kernel!,
                  (out, x, w, Val(nfft), hop, 40, Val(true)), nfft * frames)
        @test all(==(-3f0), tail(p, nfft * frames))
        foreach(Mantle.free!, (p, x, w))
    end

    # Past its last pair a lane would read the next pair in the parent, which is a
    # real overlapping one, and write its result into the canary.
    @testset "narrow_phase_kernel" begin
        tr = [(1f0, 0f0, 0f0, 0f0,  0f0, 1f0, 0f0, 0f0,  0f0, 0f0, 1f0, 0f0),
              (1f0, 0f0, 0f0, 1.9f0, 0f0, 1f0, 0f0, 0f0,  0f0, 0f0, 1f0, 0f0)]
        t = Mantle.Buffer(dev, tr)
        n = 5
        pp, pairs = tailed(dev, n, 64, (Int32(1), Int32(2)))
        sentinel = Mantle.EPAResult(Vec3f(99, 99, 99), 99f0, Vec3f(88, 88, 88), 77, false)
        rp, res = tailed(dev, n, 64, sentinel)
        runtails!(dev, Mantle.narrow_phase_kernel, (t, pairs, Mantle.UnitCube(), res), n)
        got = Array(Mantle.storage(rp))
        @test all(r -> isapprox(r.depth, 0.1f0; atol = 1f-3), got[1:n])
        @test all(==(sentinel), got[(n + 1):end])
        foreach(Mantle.free!, (t, pp, rp))
    end

    # The same over-read, which here bumps both grains' counters once per lane.
    @testset "narrow_phase_contacts_kernel" begin
        tr = [(1f0, 0f0, 0f0, 0f0,  0f0, 1f0, 0f0, 0f0,  0f0, 0f0, 1f0, 0f0),
              (1f0, 0f0, 0f0, 1.9f0, 0f0, 1f0, 0f0, 0f0,  0f0, 0f0, 1f0, 0f0)]
        t = Mantle.Buffer(dev, tr)
        n = 3
        pp, pairs = tailed(dev, n, 64, (Int32(1), Int32(2)))
        counters = Mantle.Buffer(dev, zeros(UInt32, 2))
        rec = Mantle.ContactRecord(typemax(UInt32), typemax(UInt32), Vec3f(0, 0, 0), Vec3f(0, 0, 0), 0f0)
        contacts = Mantle.Buffer(dev, fill(rec, 2 * 4))
        runtails!(dev, Mantle.narrow_phase_contacts_kernel,
                  (t, pairs, Mantle.UnitCube(), counters, contacts, Int32(4)), n)
        @test Array(Mantle.storage(counters)) == UInt32[n, n]
        foreach(Mantle.free!, (t, pp, counters, contacts))
    end
end
