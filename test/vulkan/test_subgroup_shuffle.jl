# The shuffle family: read another lane's value.
#
# `OpGroupNonUniformShuffle`/`ShuffleXor`/`ShuffleUp`/`ShuffleDown`/`Broadcast`
# were declared in `module.jl` for a long time with no emitter case and no Julia
# binding, so nothing could reach them; `OpGroupNonUniformRotateKHR` was not
# declared at all. These tests pin down the semantics that a reduction built on
# them depends on, and they check the *selector* convention: SPIR-V lane indices
# are 0-based and `subgroup_lane()` deliberately does not convert to Julia's
# 1-based indexing, because it is a lane selector, not a Julia index.
#
# Everything here is written against the device's real `subgroup_size()` rather
# than a hardcoded 32, since RDNA3 runs compute at 32 *or* 64 depending on how
# the driver compiled the shader.

using Test, Lava, KernelAbstractions

const KA = KernelAbstractions

@kernel function shufflekinds!(sz, lane, sh, sx, su, sd, bc)
    i = @index(Global, Linear)
    l = Lava.subgroup_lane()
    v = Float32(l)
    sz[i] = Lava.subgroup_size()
    lane[i] = l
    sh[i] = Lava.subgroup_shuffle(v, UInt32(3))
    sx[i] = Lava.subgroup_shuffle_xor(v, UInt32(1))
    su[i] = Lava.subgroup_shuffle_up(v, UInt32(1))
    sd[i] = Lava.subgroup_shuffle_down(v, UInt32(1))
    bc[i] = Lava.subgroup_broadcast(v, UInt32(5))
end

@kernel function rotatekind!(out)
    i = @index(Global, Linear)
    out[i] = Lava.subgroup_rotate(Float32(Lava.subgroup_lane()), UInt32(1))
end

# A butterfly reduction: log2(size) shuffle_xor steps, no shared memory and no
# barrier. This is the shape K3 wants for the flash softmax's row max/sum, so
# the test is that it agrees with `subgroup_add` exactly, not approximately —
# both sum the same values in the same tree order.
@kernel function butterfly!(out, ref, @Const(x))
    i = @index(Global, Linear)
    v = x[i]
    acc = v
    m = UInt32(1)
    while m < Lava.subgroup_size()
        acc += Lava.subgroup_shuffle_xor(acc, m)
        m <<= 1
    end
    out[i] = acc
    ref[i] = Lava.subgroup_add(v)
end

# `subgroup_shuffle` with a per-lane (non-uniform) selector — the case that
# distinguishes it from `subgroup_broadcast`, which requires a uniform one.
@kernel function reverselanes!(out)
    i = @index(Global, Linear)
    last = Lava.subgroup_size() - UInt32(1)
    out[i] = Lava.subgroup_shuffle(Float32(Lava.subgroup_lane()), last - Lava.subgroup_lane())
end

@testset "subgroup shuffle family" begin
    be = LavaBackend()
    N = 256

    sz   = KA.zeros(be, UInt32, N)
    lane = KA.zeros(be, UInt32, N)
    sh, sx, su, sd, bc = (KA.zeros(be, Float32, N) for _ in 1:5)
    shufflekinds!(be, 64)(sz, lane, sh, sx, su, sd, bc; ndrange=N)
    KA.synchronize(be)

    S = Int(Array(sz)[1])
    @test S in (16, 32, 64)          # every subgroup width Vulkan permits here
    @test all(==(S), Array(sz))
    hlane = Int.(Array(lane))
    # Lane index is 0-based and restarts at every subgroup boundary.
    @test hlane == [mod(i - 1, S) for i in 1:N]

    @testset "absolute shuffle and broadcast pick one lane for everyone" begin
        @test all(==(3.0f0), Array(sh))
        @test all(==(5.0f0), Array(bc))
    end

    @testset "shuffle_xor is a butterfly partner" begin
        got = Array(sx)
        @test got == Float32[xor(l, 1) for l in hlane]
    end

    @testset "shuffle_up/down move by a delta inside the subgroup" begin
        gotu, gotd = Array(su), Array(sd)
        # Only the lanes whose source is IN RANGE are specified. SPIR-V leaves
        # the rest undefined (not zero, not clamped), which is exactly the trap
        # a scan written on these has to mask for — so assert nothing there.
        for (i, l) in enumerate(hlane)
            l >= 1     && @test gotu[i] == Float32(l - 1)
            l <= S - 2 && @test gotd[i] == Float32(l + 1)
        end
    end

    @testset "rotate wraps where shuffle_down would fall off" begin
        out = KA.zeros(be, Float32, N)
        rotatekind!(be, 64)(out; ndrange=N)
        KA.synchronize(be)
        got = Array(out)
        # Defined for EVERY lane, including the last one, which reads lane 0.
        @test got == Float32[mod(l + 1, S) for l in hlane]
    end

    @testset "a butterfly reduction agrees with subgroup_add" begin
        x = KA.allocate(be, Float32, N)
        copyto!(x, Float32.(1:N))
        out, ref = KA.zeros(be, Float32, N), KA.zeros(be, Float32, N)
        butterfly!(be, 64)(out, ref, x; ndrange=N)
        KA.synchronize(be)
        @test Array(out) == Array(ref)
        # And against the host, so a matching pair of wrong answers still fails.
        want = Float32[sum(Float32, (fld(i - 1, S) * S + 1):(fld(i - 1, S) * S + S))
                       for i in 1:N]
        @test Array(out) == want
    end

    @testset "a per-lane selector reverses the subgroup" begin
        out = KA.zeros(be, Float32, N)
        reverselanes!(be, 64)(out; ndrange=N)
        KA.synchronize(be)
        @test Array(out) == Float32[S - 1 - l for l in hlane]
    end

    @testset "every element type round-trips" begin
        for T in (Float32, Float64, Int32, UInt32, Int64, UInt64)
            @eval @kernel function shuffle_t!(out, ::Type{$T})
                i = @index(Global, Linear)
                out[i] = Lava.subgroup_shuffle($T(Lava.subgroup_lane()), UInt32(2))
            end
            out = KA.zeros(be, T, N)
            Base.invokelatest(shuffle_t!(be, 64), out, T; ndrange=N)
            KA.synchronize(be)
            @test all(==(T(2)), Array(out))
        end
    end
end
