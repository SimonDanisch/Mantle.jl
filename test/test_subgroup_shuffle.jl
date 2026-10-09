# The shuffle family: read another lane's value.
#
# `OpGroupNonUniformShuffle`/`ShuffleXor`/`ShuffleUp`/`ShuffleDown`/`Broadcast`
# were declared in Lava's `module.jl` for a long time with no emitter case and no
# Julia binding, so nothing could reach them; `OpGroupNonUniformRotateKHR` was not
# declared at all. These tests pin down the semantics that a reduction built on
# them depends on, and they check the *selector* convention: KernelInterface's
# `shfl` takes an ABSOLUTE, 0-based lane, while `get_sub_group_local_id` is
# 1-based like every KI index. The lane selector is not a Julia index, and the one
# between them has to be subtracted in exactly one place.
#
# Written against KernelInterface's portable spellings — `shfl` (absolute lane),
# `shfl_down` and `sub_group_reduce_add` — so it runs on every backend. The xor,
# up, rotate and reverse patterns are `shfl` with a computed selector, which is
# how a portable kernel writes them (DNNKernels' masked prefill reduces with
# `lane ⊻ delta`).
#
# Everything here is written against the device's real sub-group width rather
# than a hardcoded 32, since RDNA3 runs compute at 32 *or* 64 depending on how
# the driver compiled the shader and lavapipe runs 8; and the width a kernel sees
# has to be the one `Mantle.caps` reports, because that is what a kernel is sized
# from.

using Test, KernelAbstractions
import KernelInterface as KI
include(joinpath(@__DIR__, "testbackend.jl"))

const KA = KernelAbstractions

# This lane's 0-based index — the selector `shfl` takes — and the sub-group width.
@inline sglane() = KI.get_sub_group_local_id(UInt32) - UInt32(1)
@inline sgwidth() = KI.get_sub_group_size(UInt32)

@kernel function shufflekinds!(sz, lane, sh, sx, su, sd, rot)
    i = @index(Global, Linear)
    l = sglane()
    S = sgwidth()
    v = Float32(l)
    sz[i] = S
    lane[i] = l
    sh[i] = KI.shfl(v, UInt32(3))                          # absolute
    sx[i] = KI.shfl(v, xor(l, UInt32(1)))                  # butterfly partner
    # Up by one. Lane 0 has no source; it reads itself so the selector stays a
    # valid lane (Metal requires one), and the test asserts nothing there.
    su[i] = KI.shfl(v, max(l, UInt32(1)) - UInt32(1))
    sd[i] = KI.shfl_down(v, 1)                             # down by one
    rot[i] = KI.shfl(v, (l + UInt32(1)) % S)               # rotate: wraps
end

# A butterfly reduction: log2(size) xor steps, no shared memory and no barrier.
# This is the shape K3 wants for the flash softmax's row max/sum, so the test is
# that it agrees with `sub_group_reduce_add` exactly, not approximately — the
# values are integers small enough that Float32 sums them exactly in any order.
@kernel function butterfly!(out, ref, @Const(x))
    i = @index(Global, Linear)
    v = x[i]
    acc = v
    l = sglane()
    m = UInt32(1)
    while m < sgwidth()
        acc += KI.shfl(acc, xor(l, m))
        m <<= 1
    end
    out[i] = acc
    ref[i] = KI.sub_group_reduce_add(v)
end

# `shfl` with a per-lane (non-uniform) selector — the case that distinguishes it
# from a broadcast, which requires a uniform one.
@kernel function reverselanes!(out)
    i = @index(Global, Linear)
    l = sglane()
    out[i] = KI.shfl(Float32(l), sgwidth() - UInt32(1) - l)
end

# One kernel per operation, at the element type of `out`, so every type a
# backend lists can be run.
@kernel function shfl_t!(out)
    i = @index(Global, Linear)
    out[i] = KI.shfl(eltype(out)(sglane()), UInt32(2))
end

@kernel function shfl_down_t!(out)
    i = @index(Global, Linear)
    out[i] = KI.shfl_down(eltype(out)(sglane()), 1)
end

@kernel function reduce_t!(out)
    i = @index(Global, Linear)
    out[i] = KI.sub_group_reduce_add(eltype(out)(sglane() + UInt32(1)))
end

@testset "subgroup shuffle family" begin
    be = TESTBACKEND
    N = 256
    WG = 64

    sz   = KA.zeros(be, UInt32, N)
    lane = KA.zeros(be, UInt32, N)
    sh, sx, su, sd, rot = (KA.zeros(be, Float32, N) for _ in 1:5)
    shufflekinds!(be, WG)(sz, lane, sh, sx, su, sd, rot; ndrange=N)
    KA.synchronize(be)

    S = Int(Array(sz)[1])
    # The width the kernel ran at is the one the device reports, on every lane.
    @test S == Mantle.caps(be).subgroup
    @test S == KI.sub_group_size(be)
    @test all(==(S), Array(sz))
    hlane = Int.(Array(lane))
    # Lane index is 0-based and restarts at every subgroup boundary.
    @test hlane == [mod(i - 1, S) for i in 1:N]

    @testset "an absolute shuffle picks one lane for everyone" begin
        @test all(==(3.0f0), Array(sh))
    end

    @testset "selector xor 1 is the butterfly partner" begin
        got = Array(sx)
        @test got == Float32[xor(l, 1) for l in hlane]
    end

    @testset "up/down move by a delta inside the subgroup" begin
        gotu, gotd = Array(su), Array(sd)
        # Only the lanes whose source is IN RANGE are specified. KI leaves the
        # rest of `shfl_down` unspecified (not zero, not clamped), which is
        # exactly the trap a scan written on it has to mask for — so assert
        # nothing there.
        for (i, l) in enumerate(hlane)
            l >= 1     && @test gotu[i] == Float32(l - 1)
            l <= S - 2 && @test gotd[i] == Float32(l + 1)
        end
    end

    @testset "rotate wraps where shfl_down would fall off" begin
        # Defined for EVERY lane, including the last one, which reads lane 0.
        @test Array(rot) == Float32[mod(l + 1, S) for l in hlane]
    end

    @testset "a butterfly reduction agrees with sub_group_reduce_add" begin
        x = KA.allocate(be, Float32, N)
        copyto!(x, Float32.(1:N))
        out, ref = KA.zeros(be, Float32, N), KA.zeros(be, Float32, N)
        butterfly!(be, WG)(out, ref, x; ndrange=N)
        KA.synchronize(be)
        @test Array(out) == Array(ref)
        # And against the host, so a matching pair of wrong answers still fails.
        want = Float32[sum(Float32, (fld(i - 1, S) * S + 1):(fld(i - 1, S) * S + S))
                       for i in 1:N]
        @test Array(out) == want
    end

    @testset "a per-lane selector reverses the subgroup" begin
        out = KA.zeros(be, Float32, N)
        reverselanes!(be, WG)(out; ndrange=N)
        KA.synchronize(be)
        @test Array(out) == Float32[S - 1 - l for l in hlane]
    end

    # The type lists drive KI's own suite, so a type listed and not generated is
    # a kernel that fails to compile on first use. Every listed type runs here.
    @testset "every listed element type round-trips" begin
        for T in KI.shfl_types(be)
            out = KA.zeros(be, T, N)
            shfl_t!(be, WG)(out; ndrange=N)
            KA.synchronize(be)
            @test all(==(T(2)), Array(out))
        end
        downtypes = filter(T -> KI.supports_shuffle(be, T),
                           [Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64,
                            Float16, Float32, Float64])
        for T in downtypes
            out = KA.zeros(be, T, N)
            shfl_down_t!(be, WG)(out; ndrange=N)
            KA.synchronize(be)
            got = Array(out)
            @test all(got[i] == T(l + 1) for (i, l) in enumerate(hlane) if l <= S - 2)
        end
        for T in KI.sub_group_reduce_add_types(be)
            out = KA.zeros(be, T, N)
            reduce_t!(be, WG)(out; ndrange=N)
            KA.synchronize(be)
            @test all(==(T(S * (S + 1) ÷ 2)), Array(out))
        end
    end
end
