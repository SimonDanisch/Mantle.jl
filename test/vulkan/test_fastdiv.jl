"""
`FastDiv32`, and the broadcast paths that decompose their linear index with it.

The linear->Cartesian division chain is what these kernels are bound by, not
their memory traffic — isolated, over 2.36 M elements, a write-only kernel is
10.9 us and the same kernel plus a rank-6 `cart32` is 85.0, against a ~42 us
memory floor for the array. `FastDiv32` replaces each division with a high
multiply and a shift (`init_fastdiv_values` / `fastdiv` in llama.cpp's
`generic_unary_head.glsl`), which measured 2.4-3.1x on the permuted copies and
7.6% on the whole SAM 2 encode.

A wrong quotient here is not a wrong number, it is an **out-of-bounds index**, so
the arithmetic is checked exhaustively rather than sampled, and the kernels are
checked against the dividing path they replaced rather than against a reference
computed the same way.
"""

using Test, Lava, GPUArrays, KernelAbstractions
const KA = KernelAbstractions

# Every extent SAM 2 decomposes by, plus powers of two, primes, and the awkward
# small ones. `1` matters: it makes `L == 0` and `mp == 1`, the degenerate case.
const DIVISORS = sort(unique(vcat([1, 2, 3, 4, 8, 16, 32, 64, 72, 128, 144, 256, 288, 512,
                                   576, 1024, 1152, 2048, 4096, 65536, 1023, 4095],
                                  collect(1:64))))

"""
Count the `n < lim` where the magic-number quotient or remainder differs from `÷`.

A function, not a loop in the testset body, because a testset is a closure and
this ran 100x slower there — 164 M iterations took 3m49s inline and about two
seconds compiled.
"""
function sweepexact(d::Integer, lim::UInt32)
    f = MVE.FastDiv32(d)
    du = UInt32(d)
    wrongq = 0
    wrongr = 0
    n = UInt32(0)
    while n < lim
        hi = MVE.fastdiv(n, f)
        hi == n ÷ du || (wrongq += 1)
        (n - hi * du) == n % du || (wrongr += 1)
        n += UInt32(1)
    end
    return wrongq, wrongr
end

@testset "FastDiv32" begin
    @testset "quotient and remainder are exact" begin
        # 2^22 covers every extent-and-position pair the decoder reaches; the full
        # sweep to 2^25 — past the largest array the encoder builds (9.4 M) — was
        # run for all 78 divisors when this landed and is exact. Widen if you
        # touch `fastdiv`.
        lim = UInt32(1) << 22
        for d in DIVISORS
            @test sweepexact(d, lim) == (0, 0)
        end
    end

    @testset "rejects a zero divisor" begin
        @test_throws ArgumentError MVE.FastDiv32(0)
    end

    @testset "kernels agree with the dividing path" begin
        backend = LavaBackend()
        let
            # Ranks 2 to 7, the encoder's two permutation families, extents that
            # are not powers of two, and a length that is not a multiple of the
            # workgroup — the unrolled launch has a ragged tail and the index of
            # the tail elements is exactly what a bad quotient corrupts.
            shapes = [((63, 65), (2, 1)),
                      ((7, 11, 13), (3, 1, 2)),
                      ((72, 8, 256, 16), (1, 3, 2, 4)),
                      ((288, 4, 32, 4, 32, 1), (1, 2, 4, 3, 5, 6)),
                      ((576, 16, 4, 16, 4, 1), (1, 2, 4, 3, 5, 6)),
                      ((3, 5, 7, 11, 13, 2, 3), (1, 2, 4, 3, 6, 5, 7))]
            for (sz, perm) in shapes
                host = rand(Float32, sz...)
                a = KA.allocate(backend, Float32, sz...)
                copyto!(a, host)
                dsz = ntuple(i -> sz[perm[i]], length(sz))
                want = permutedims(host, perm)
                got = map((false, true)) do fd
                    d = KA.allocate(backend, Float32, dsz...)
                    fill!(d, 0.0f0)
                    # Driving `_copyto!` directly is what lets both index paths
                    # run without a process-wide switch: `d .= x` lowers to
                    # exactly this call with the default.
                    bc = Broadcast.instantiate(
                             Broadcast.broadcasted(identity, PermutedDimsArray(a, perm)))
                    GPUArrays._copyto!(d, bc; fastdiv = fd)
                    KA.synchronize(backend)
                    Array(d)
                end
                @test got[1] == want          # dividing path
                @test got[2] == want          # multiplying path
                @test got[1] == got[2]
            end
        end
    end
end
