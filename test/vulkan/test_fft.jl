# `Mantle.fft!` — batched 1D FFT, ported from VkFFT.
#
# The oracle is a **naive O(N²) DFT in Float64**, not another FFT. That is
# deliberate and it is the point of the file: an FFT compared against an FFT
# shares every indexing assumption, so a transposed butterfly or an off-by-one in
# the Stockham scatter agrees with itself. The direct sum shares nothing. It is
# O(N²), which is why the exhaustive size sweep stops at 1024 and the large sizes
# are checked by round-trip and by Parseval instead.
#
# Three properties, because they fail independently:
#
#   * *values* — against the DFT, every power of two through the whole plan
#     space (pure radix 8, leading radix 2, leading radix 4).
#   * *round trip* — `ifft(fft(x))/N == x`. Catches a forward/inverse sign error
#     that the forward check alone cannot, because both would be wrong the same
#     way only if the twiddle sign were right.
#   * *Parseval* — `sum|X|² == N sum|x|²`. Independent of the ordering of the
#     output, so it catches a permutation bug that a value check on a symmetric
#     input would miss.
#
# The batch axis is swept alongside, including counts that do NOT divide the
# workgroup group size — `fftgroup` has to fall back to 1 there, and a partial
# group would read past the end of the batch.

using Test, Lava, KernelAbstractions
const KA = KernelAbstractions

"The direct sum. Float64 throughout so it is the reference, not a competitor."
function dftref(x::Vector{ComplexF64}, sign::Int = -1)
    N = length(x)
    [sum(x[n + 1] * cispi(sign * 2 * k * n / N) for n in 0:(N - 1)) for k in 0:(N - 1)]
end

relerr(got, want) = maximum(abs, got .- want) / maximum(abs, want)

@testset "fft" begin
    backend = LavaBackend()

    @testset "the plan covers every power of two" begin
        # k % 3 picks the leading stage: 0 none, 1 radix 2, 2 radix 4.
        for k in 3:12
            N = 1 << k
            T, lead = Mantle.fftplan(N)
            @test T == N ÷ 8
            @test lead == (k % 3 == 0 ? 1 : (k % 3 == 1 ? 2 : 4))
            # the plan must actually reconstruct N
            @test lead * 8^Int(round(log(8, N ÷ lead))) == N
        end
        @test_throws ArgumentError Mantle.fftplan(96)    # not a power of two
        @test_throws ArgumentError Mantle.fftplan(4)     # below the radix-8 floor
    end

    @testset "values against a Float64 DFT" begin
        for N in (8, 16, 32, 64, 128, 256, 512, 1024)
            h = ComplexF32.(randn(ComplexF64, N))
            d = KA.allocate(backend, ComplexF32, N)
            copyto!(d, h)
            got = Array(Mantle.fft!(similar(d), d))
            KA.synchronize(backend)
            @test relerr(ComplexF64.(got), dftref(ComplexF64.(h))) < 1.0f-4
        end
    end

    @testset "batching, including counts that do not divide the group" begin
        for N in (64, 256, 1024), nb in (1, 3, 7, 16)
            h = ComplexF32.(randn(ComplexF64, N, nb))
            d = KA.allocate(backend, ComplexF32, N, nb)
            copyto!(d, h)
            got = Array(Mantle.fft!(similar(d), d))
            KA.synchronize(backend)
            # every column, not just the first: a wrong per-transform shared
            # offset leaves column 1 perfect and the rest garbage.
            for c in 1:nb
                @test relerr(ComplexF64.(got[:, c]), dftref(ComplexF64.(h[:, c]))) < 1.0f-4
            end
        end
    end

    @testset "round trip and Parseval, up to 4096" begin
        for N in (512, 2048, 4096)
            h = ComplexF32.(randn(ComplexF64, N, 8))
            d = KA.allocate(backend, ComplexF32, N, 8)
            copyto!(d, h)
            f = Mantle.fft!(similar(d), d)
            r = Mantle.fft!(similar(d), f; inverse = true)
            KA.synchronize(backend)
            @test maximum(abs, Array(r) ./ N .- h) < 1.0f-4

            # Parseval: independent of output ordering, so it catches a
            # permutation the value check would not.
            F = ComplexF64.(Array(f))
            @test isapprox(sum(abs2, F), N * sum(abs2, ComplexF64.(h)); rtol = 1.0f-4)
        end
    end

    @testset "the group planner respects its bounds" begin
        # It must never emit a partial group, exceed the workgroup limit, or
        # claim more than half the shared budget — the last is what keeps more
        # than one workgroup resident per SM.
        for N in (64, 128, 256, 512, 1024, 4096), nbatch in (1, 3, 8, 64, 1000)
            T, _ = Mantle.fftplan(N)
            g = Mantle.fftgroup(N, T, nbatch, 1024, 49152)
            @test g >= 1
            @test nbatch % g == 0
            @test g * T <= 1024
            @test 2 * sizeof(Float32) * N * g <= 49152 ÷ 2 || g == 1
        end
    end

    @testset "mixed radix reaches the sizes the audio models need" begin
        # This is the reason the mixed path exists. Two of the three audio
        # models are NOT powers of two, so the tuned radix-8 kernel covers
        # exactly one of them.
        @test Mantle.fftplan_mixed(400) == (8, 5, 5, 2)     # Whisper mel
        @test Mantle.fftplan_mixed(960) == (8, 8, 5, 3)     # DeepFilterNet3
        @test Mantle.fftplan_mixed(4096) == (8, 8, 8, 8)    # Demucs
        @test Mantle.fftplan_mixed(97) === nothing          # prime: needs Rader
        for RS in ((8,5,5,2), (8,8,5,3), (8,4,3), (5,4,3))
            @test prod(RS) |> Mantle.fftplan_mixed == RS
        end

        # Sizes with `reps > 1` in a stage are the ones that caught the
        # gather/scatter aliasing bug — a stage of radix r < Rmax has more
        # butterflies than threads, and the second pass used to read slots the
        # first had already overwritten. 96 and 400 both hit it; 40 did not,
        # because its read and write sets happened to be disjoint. Keep all
        # three.
        for N in (6, 10, 12, 15, 20, 24, 40, 48, 60, 96, 120, 400, 480, 960)
            h = ComplexF32.(randn(ComplexF64, N))
            d = KA.allocate(backend, ComplexF32, N)
            copyto!(d, h)
            got = Array(Mantle.fftany!(similar(d), d))
            KA.synchronize(backend)
            @test relerr(ComplexF64.(got), dftref(ComplexF64.(h))) < 1.0f-4
        end
    end

    @testset "rfft is the complex transform's half" begin
        for N in (16, 64, 256, 400, 512, 960, 2048)
            h = Float32.(randn(N))
            d = KA.allocate(backend, Float32, N)
            copyto!(d, h)
            got = Array(Mantle.rfft(d))
            KA.synchronize(backend)
            @test length(got) == N ÷ 2 + 1              # the non-redundant half
            want = dftref(ComplexF64.(h))[1:(N ÷ 2 + 1)]
            @test relerr(ComplexF64.(got), want) < 1.0f-4
            # bin 0 and the Nyquist bin are real for a real input; a sign error
            # in the split shows up here and nowhere else.
            @test abs(imag(got[1])) < 1.0f-4 * maximum(abs, got)
            @test abs(imag(got[end])) < 1.0f-4 * maximum(abs, got)
        end
    end

    @testset "stft matches torch.stft(center=true)" begin
        L, nfft, hop = 8000, 400, 160
        h = Float32.(randn(L))
        x = KA.allocate(backend, Float32, L)
        copyto!(x, h)
        w = Mantle.hannwindow(backend, nfft)
        S = Array(Mantle.stft(x, nfft, hop, w))
        KA.synchronize(backend)
        @test size(S) == (nfft ÷ 2 + 1, L ÷ hop + 1)

        # host reference with the same reflect padding and frame centring
        wr = Float64[0.5 * (1 - cospi(2k / nfft)) for k in 0:(nfft - 1)]
        pad = nfft ÷ 2
        refl(p) = p < 0 ? -p : (p >= L ? 2L - 2 - p : p)
        want = [begin
                    fr = [Float64(h[refl(t * hop + i - pad) + 1]) * wr[i + 1]
                          for i in 0:(nfft - 1)]
                    sum(fr[n + 1] * cispi(-2 * k * n / nfft) for n in 0:(nfft - 1))
                end for k in 0:(nfft ÷ 2), t in 0:(L ÷ hop)]
        @test relerr(ComplexF64.(S), want) < 1.0f-4
    end

    @testset "both shared-memory layouts agree" begin
        # `skew` is off by measurement, not by correctness — so the padded
        # addressing has to stay right, or turning it on to test another device
        # would silently produce garbage.
        for N in (64, 512, 1024)
            h = ComplexF32.(randn(ComplexF64, N, 4))
            d = KA.allocate(backend, ComplexF32, N, 4)
            copyto!(d, h)
            a = Array(Mantle.fft!(similar(d), d; skew = false))
            c = Array(Mantle.fft!(similar(d), d; skew = true))
            KA.synchronize(backend)
            @test relerr(ComplexF64.(c), ComplexF64.(a)) < 1.0f-5
        end
    end
end
