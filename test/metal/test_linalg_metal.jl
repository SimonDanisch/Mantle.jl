"""
Metal's linear algebra (`src/metal/linalg/`): the tensor-ops GEMM and fused attention
kernels a graph dispatches, the MPSGraph product, attention and convolution a graph
calls, and one such call inside a recorded plan.

Kernels are launched from their own configurations, the way a graph records them. The
library calls commit their own command buffer, with launches on the same queue before
and after them, which is where a wrong order is silent: a call that ran ahead of the
launches filling its operands computes on zeros and nothing about the result looks wrong.
"""

using Test, Mantle, Metal, KernelAbstractions
# Mantle's own, the type its Metal GEMM declares BFloat16 operands with.
using Mantle: BFloat16

const LINALG = Mantle.MetalLinalg

"""Launch a kernel configuration the way a graph records it: `ndrange` and `group` count
threads."""
function linalg_launch(cfg)
    groups = map(÷, cfg.ndrange, cfg.group)
    Metal.@metal threads = cfg.group groups = groups cfg.kernel(cfg.args...)
end

"""`softmax(scale·QᵀK)·V` per (head, batch) plane, in Float64 over the operands given."""
function linalg_attnref(qh, kh, vh, scale)
    E, Lq, H, B = size(qh); Lk = size(kh, 2)
    out = zeros(Float64, E, Lq, H, B)
    for b in 1:B, h in 1:H, lq in 1:Lq
        s = [sum(Float64(qh[e, lq, h, b]) * Float64(kh[e, lk, h, b]) for e in 1:E) * scale
             for lk in 1:Lk]
        s .= exp.(s .- maximum(s)); s ./= sum(s)
        for e in 1:E
            out[e, lq, h, b] = sum(s[lk] * Float64(vh[e, lk, h, b]) for lk in 1:Lk)
        end
    end
    out
end

"""Run the fused attention kernel through its own config; `nothing` if declined."""
function linalg_fusedattention(qh, kh, vh, scale)
    q = MtlArray(qh); k = MtlArray(kh); v = MtlArray(vh)
    o = MtlArray(fill(eltype(qh)(NaN), size(qh)))
    cfg = LINALG.attention_kernel_config(Metal.device(), o, q, k, v; scale)
    cfg === nothing && return nothing
    linalg_launch(cfg)
    Metal.synchronize()
    Array(o)
end

"""`mpsattention!` through the same question a backend asks; `nothing` if declined."""
function linalg_mpsattention!(bq, o, q, k, v, scale)
    ops = LINALG.attentionoperands(o, q, k, v)
    ops === nothing && return nothing
    LINALG.mpsattention!(bq, o, ops.res..., ops.keys..., scale)
end

"""A `dims` array at `off` elements into `big`, which is what a suballocator hands out."""
linalg_slice(big, off, dims) = reshape(view(big, (off + 1):(off + prod(dims))), dims)

"""A zero device operand, for the tests that only ask which shapes are admitted."""
linalg_zeros(T, d...) = MtlArray(zeros(T, d...))

"""The convolution in Float64, in the reversed layout, from the definition."""
function linalg_convref(x, w, b, stride, pad, dil, groups)
    W, H, Cin, N = size(x); KW, KH, Cing, Cout = size(w)
    OW = (W + 2pad[1] - dil[1] * (KW - 1) - 1) ÷ stride[1] + 1
    OH = (H + 2pad[2] - dil[2] * (KH - 1) - 1) ÷ stride[2] + 1
    out = zeros(Float64, OW, OH, Cout, N)
    cpg = Cout ÷ groups
    for n in 1:N, co in 1:Cout
        g = (co - 1) ÷ cpg
        for oy in 1:OH, ox in 1:OW
            acc = b === nothing ? 0.0 : Float64(b[co])
            for ci in 1:Cing, ky in 1:KH, kx in 1:KW
                ix = (ox - 1) * stride[1] - pad[1] + (kx - 1) * dil[1] + 1
                iy = (oy - 1) * stride[2] - pad[2] + (ky - 1) * dil[2] + 1
                (1 <= ix <= W && 1 <= iy <= H) || continue
                acc += Float64(x[ix, iy, g * Cing + ci, n]) * Float64(w[kx, ky, ci, co])
            end
            out[ox, oy, co, n] = acc
        end
    end
    out
end

function linalg_setval!(x, v::Float32, n::Int32)
    i = Metal.thread_position_in_grid_1d()
    i <= n && (@inbounds x[i] = v)
    return
end

function linalg_trace!(out, c, n::Int32)
    if Metal.thread_position_in_grid_1d() == 1
        t = 0.0f0
        @inbounds for k in Int32(1):n
            t += c[k, k]
        end
        @inbounds out[1] = t
    end
    return
end

@kernel function linalg_fill!(a, v)
    i = @index(Global, Linear)
    @inbounds a[i] = v
end

@kernel function linalg_rowsum!(out, c, n)
    i = @index(Global, Linear)
    s = 0.0f0
    for j in Int32(1):n
        @inbounds s += c[i, j]
    end
    @inbounds out[i] = s
end

@testset "Metal: tensor-ops GEMM" begin
    dev = Metal.device()
    if !LINALG.tensorcapable(dev)
        @test_skip LINALG.tensorcapable(dev)
    else
        # BFloat16 operands need macOS 26.1; `tensoreltypes` declines them before.
        halves = LINALG.tensoroperand(BFloat16) ? (Float16, BFloat16) : (Float16,)
        ttol(T) = T === Float32 ? 1.0f-3 : 1.0f-2

        # NaN in every destination: the kernels write `C` without reading it, and a tile
        # one of them skipped cannot pass.
        @testset "$T $M×$N×$K" for T in (Float32, halves...),
                                   (M, N, K) in ((64, 64, 32), (128, 256, 64),
                                                 (512, 512, 128), (72, 96, 100))
            A = MtlArray(rand(T, M, K)); B = MtlArray(rand(T, K, N))
            C = MtlArray(fill(T(NaN), M, N))
            cfg = LINALG.gemm_kernel_config(dev, C, A, B)
            @test cfg.kernel === LINALG.gemm_tensor_kernel!
            @test cfg.fused == (; bias = false, epilogue = false)
            linalg_launch(cfg)
            Metal.synchronize()
            ref = Float32.(Array(A)) * Float32.(Array(B))
            @test isapprox(Float32.(Array(C)), ref; rtol = ttol(T))
        end

        # A half product into a single destination keeps the Float32 accumulator all the
        # way out. It used to fall to the SIMD kernel, 19x slower on an M5 at
        # 17408x512x5120, because the gate asked for one element type throughout.
        @testset "$T into Float32" for T in halves
            M, N, K = 128, 256, 512
            A = MtlArray(rand(T, M, K)); B = MtlArray(rand(T, K, N))
            C = MtlArray(fill(NaN32, M, N))
            cfg = LINALG.gemm_kernel_config(dev, C, A, B)
            @test cfg.kernel === LINALG.gemm_tensor_kernel!
            linalg_launch(cfg)
            Metal.synchronize()
            ref = Float32.(Array(A)) * Float32.(Array(B))
            @test maximum(abs, Array(C) .- ref) / maximum(abs, ref) < 1.0f-5
        end

        # Recorded DNN plans fold a row bias and an elementwise epilogue into the store.
        @testset "a row bias and an epilogue on the store" begin
            M, N, K = 128, 256, 64
            Ah = rand(Float16, M, K); Bh = rand(Float16, K, N); bh = rand(Float16, M)
            A, B, bias = MtlArray(Ah), MtlArray(Bh), MtlArray(bh)
            ref = Float32.(Ah) * Float32.(Bh) .+ Float32.(bh)
            C = MtlArray(fill(Float16(NaN), M, N))
            cfg = LINALG.gemm_kernel_config(dev, C, A, B; bias)
            @test cfg.fused == (; bias = true, epilogue = false)
            linalg_launch(cfg)
            Metal.synchronize()
            @test isapprox(Float32.(Array(C)), ref; rtol = 1.0f-2)

            C = MtlArray(fill(Float16(NaN), M, N))
            cfg = LINALG.gemm_kernel_config(dev, C, A, B; bias, epilogue = x -> -abs(x))
            @test cfg.fused == (; bias = true, epilogue = true)
            linalg_launch(cfg)
            Metal.synchronize()
            @test isapprox(Float32.(Array(C)), -abs.(ref); rtol = 1.0f-2)

            # A bias that is not one value per row is left to the caller.
            cfg = LINALG.gemm_kernel_config(dev, C, A, B; bias = MtlArray(rand(Float16, N)))
            @test cfg.fused.bias == false
        end

        # The contraction accumulates in a THREADGROUP Float32 tile, not in `C`. Pinned by
        # accuracy, because that is what the bug was: pointing `matmul2d` at the destination
        # rounded the running sum to `C`'s own precision once per K slice, so a 2304-deep
        # fp16 contraction came out at 4.8e-3 relative against a Float32 reference where the
        # storage floor is 8e-4. The bound below sits between them and fails on a
        # destination-accumulating kernel.
        @testset "a deep fp16 contraction keeps a Float32 accumulator" begin
            M, N, K = 128, 128, 2304
            Ah = rand(Float16, M, K); Bh = rand(Float16, K, N)
            C = MtlArray(fill(Float16(NaN), M, N))
            linalg_launch(LINALG.gemm_kernel_config(dev, C, MtlArray(Ah), MtlArray(Bh)))
            Metal.synchronize()
            ref = Float32.(Ah) * Float32.(Bh)
            @test maximum(abs, Float32.(Array(C)) .- ref) / maximum(abs, ref) < 2.0f-3
        end

        # Attention runs two strided-batched products: `q'k` with a transposed left operand
        # and the score scale, and `v p'` with a transposed right one and, for an
        # unnormalized softmax, a divisor per output column.
        @testset "strided-batched: transposes, scale and a column divisor" begin
            E, L, batch = 16, 16, 2
            qh = rand(Float16, E, L, batch); kh = rand(Float16, E, L, batch)
            scores = MtlArray(fill(Float16(NaN), L, L, batch))
            cfg = LINALG.batched_gemm_kernel_config(dev, scores, MtlArray(qh), MtlArray(kh);
                                                    transpose_a = true, alpha = 0.125f0)
            @test cfg !== nothing
            linalg_launch(cfg)
            Metal.synchronize()
            score_ref = zeros(Float32, L, L, batch)
            for b in 1:batch
                score_ref[:, :, b] .= 0.125f0 .* (Float32.(qh[:, :, b])' * Float32.(kh[:, :, b]))
            end
            @test isapprox(Float32.(Array(scores)), score_ref; rtol = 1.0f-2)

            vh = rand(Float16, E, L, batch); ph = rand(Float16, L, L, batch)
            v = MtlArray(vh); p = MtlArray(ph)
            out = MtlArray(fill(Float16(NaN), E, L, batch))
            cfg = LINALG.batched_gemm_kernel_config(dev, out, v, p; transpose_b = true)
            @test cfg !== nothing
            linalg_launch(cfg)
            Metal.synchronize()
            out_ref = zeros(Float32, E, L, batch)
            for b in 1:batch
                out_ref[:, :, b] .= Float32.(vh[:, :, b]) * Float32.(ph[:, :, b])'
            end
            @test isapprox(Float32.(Array(out)), out_ref; rtol = 1.0f-2)

            # One divisor per output COLUMN, laid out `(N, batch...)`.
            dh = rand(Float32, L, batch) .+ 1.0f0
            outd = MtlArray(fill(Float16(NaN), E, L, batch))
            cfg = LINALG.batched_gemm_kernel_config(dev, outd, v, p; transpose_b = true,
                                                    coldiv = MtlArray(dh))
            @test cfg !== nothing
            linalg_launch(cfg)
            Metal.synchronize()
            @test isapprox(Float32.(Array(outd)), out_ref ./ reshape(dh, 1, L, batch);
                           rtol = 1.0f-2)
            # …and a divisor of the wrong length is refused rather than read past.
            @test LINALG.batched_gemm_kernel_config(dev, outd, v, p; transpose_b = true,
                                                    coldiv = MtlArray(dh[:, 1])) === nothing
        end
    end

    # Element types the tensor kernels do not run go to Metal.jl's simdgroup GEMM, and
    # what neither covers is declined.
    @testset "the simdgroup fallback" begin
        A = MtlArray(rand(Float32, 70, 30)); B = MtlArray(rand(Float16, 30, 50))
        C = MtlArray(fill(NaN32, 70, 50))
        cfg = LINALG.gemm_kernel_config(dev, C, A, B)
        @test cfg.kernel === Metal.gemm_simd_kernel!
        linalg_launch(cfg)
        Metal.synchronize()
        @test isapprox(Array(C), Array(A) * Float32.(Array(B)); rtol = 1.0f-3)
        @test LINALG.gemm_kernel_config(dev, MtlArray(zeros(Int32, 4, 4)),
                                        MtlArray(zeros(Int32, 4, 4)),
                                        MtlArray(zeros(Int32, 4, 4))) === nothing
    end
end

@testset "Metal: fused attention" begin
    dev = Metal.device()
    if !LINALG.tensorcapable(dev)
        @test_skip LINALG.tensorcapable(dev)
    else
        E = 72                       # SAM 2.1's head width, and not a power of two
        scale = Float32(inv(sqrt(E)))

        # Two shapes on purpose: 128 keys is ONE tile, so the running max and sum are
        # written once and never corrected, and 256 is two, which is the only way the
        # rescale path runs at all.
        @testset "matches a Float64 reference — Lq=$Lq Lk=$Lk" for (Lq, Lk, H, B) in
                ((128, 128, 2, 1), (256, 256, 2, 2), (16, 128, 3, 1))
            qh = Float16.(reshape(0.4 .* sin.(range(0, 9, E*Lq*H*B)), E, Lq, H, B))
            kh = Float16.(reshape(0.4 .* cos.(range(0, 7, E*Lk*H*B)), E, Lk, H, B))
            vh = Float16.(reshape(0.4 .* sin.(range(0, 5, E*Lk*H*B)), E, Lk, H, B))
            got = linalg_fusedattention(qh, kh, vh, scale)
            @test got !== nothing
            ref = linalg_attnref(qh, kh, vh, Float64(scale))
            # A kernel that writes nothing also "matches" a zero reference, and the
            # destination starts as NaN so a tile it skips cannot pass either.
            @test all(isfinite, got)
            @test maximum(abs, Float64.(got) .- ref) / maximum(abs, ref) < 3e-3
        end

        # The RUNNING MAX, which a small random test cannot see. Scores here reach ~100, so
        # exponentiating them without subtracting a max gives `Inf` and then `NaN/NaN`; and
        # the largest score is in the LAST key tile, so the accumulator has to be rescaled
        # after it has already been written.
        @testset "the online softmax subtracts a running max" begin
            Lq, Lk, H, B = 32, 256, 1, 1
            qh = fill(Float16(2), E, Lq, H, B)
            kh = Float16.(reshape(repeat(range(0.5f0, 6.0f0; length = Lk)', E), E, Lk, H, B))
            vh = Float16.(reshape(0.3 .* sin.(range(0, 5, E*Lk*H*B)), E, Lk, H, B))
            raw = maximum(Float64(scale) * sum(Float64(qh[e,1,1,1]) * Float64(kh[e,lk,1,1])
                                               for e in 1:E) for lk in 1:Lk)
            @test raw > 90            # `exp` of this overflows Float32
            got = linalg_fusedattention(qh, kh, vh, scale)
            @test got !== nothing
            @test all(isfinite, got)
            ref = linalg_attnref(qh, kh, vh, Float64(scale))
            @test maximum(abs, Float64.(got) .- ref) / maximum(abs, ref) < 3e-3
        end

        # A PERMUTED operand read where it lies. SAM 2.1's projection leaves q/k/v as
        # `(E, H, L, B)` and attention wants `(E, L, H, B)`; dense, that is a 4.7 MiB
        # transpose per operand per op before the op can start. The answer has to be the
        # same to the bit as the one from the transposed copy, which is what this compares.
        @testset "a strided operand is read where it lies" begin
            Lq = Lk = 128; H = 3; B = 2
            raw = Float16.(reshape(0.4 .* sin.(range(0, 9, E*H*Lq*B)), E, H, Lq, B))
            qperm = permutedims(raw, (1, 3, 2, 4))          # (E, L, H, B), materialised
            kh = Float16.(reshape(0.4 .* cos.(range(0, 7, E*Lk*H*B)), E, Lk, H, B))
            vh = Float16.(reshape(0.4 .* sin.(range(0, 5, E*Lk*H*B)), E, Lk, H, B))
            dense = linalg_fusedattention(qperm, kh, vh, scale)
            @test dense !== nothing

            q = MtlArray(raw); k = MtlArray(kh); v = MtlArray(vh)
            o = MtlArray(fill(Float16(NaN), E, Lq, H, B))
            view_q = (; res = q, dims = (E, Lq, H, B),
                      strides = (1, E * H, E, E * H * Lq), offset = 0)
            cfg = LINALG.attention_kernel_config(dev, o, view_q, k, v; scale)
            @test cfg !== nothing
            linalg_launch(cfg)
            Metal.synchronize()
            @test all(isfinite, Array(o))
            @test Array(o) == dense

            # …and a view whose innermost axis is NOT contiguous is refused, because a tensor
            # descriptor carries a leading dimension and not a gap between elements.
            @test LINALG.attention_kernel_config(dev, o, (; res = q, dims = (E, Lq, H, B),
                                                          strides = (2, E * H, E, E * H * Lq),
                                                          offset = 0), k, v; scale) === nothing
        end

        @testset "which operands the config admits" begin
            mk(T, dims) = MtlArray(fill(T(0), dims...))
            ok = LINALG.attention_kernel_config(dev, mk(Float16, (E, 256, 2, 1)),
                                                mk(Float16, (E, 256, 2, 1)),
                                                mk(Float16, (E, 256, 2, 1)),
                                                mk(Float16, (E, 256, 2, 1)); scale = 0.1f0)
            @test ok !== nothing
            @test ok.group == (4 * 32, 1, 1)
            # A key run no tile divides, and a query run no tile divides.
            @test LINALG.attention_kernel_config(dev, mk(Float16, (E, 256, 2, 1)),
                                                 mk(Float16, (E, 256, 2, 1)),
                                                 mk(Float16, (E, 48, 2, 1)),
                                                 mk(Float16, (E, 48, 2, 1)); scale = 0.1f0) === nothing
            @test LINALG.attention_kernel_config(dev, mk(Float16, (E, 12, 2, 1)),
                                                 mk(Float16, (E, 12, 2, 1)),
                                                 mk(Float16, (E, 128, 2, 1)),
                                                 mk(Float16, (E, 128, 2, 1)); scale = 0.1f0) === nothing
            # A head width `matmul2d` cannot tile, mixed element types, and a plane with no
            # batch axis at all.
            @test LINALG.attention_kernel_config(dev, mk(Float16, (70, 128, 2, 1)),
                                                 mk(Float16, (70, 128, 2, 1)),
                                                 mk(Float16, (70, 128, 2, 1)),
                                                 mk(Float16, (70, 128, 2, 1)); scale = 0.1f0) === nothing
            @test LINALG.attention_kernel_config(dev, mk(Float16, (E, 128, 2, 1)),
                                                 mk(Float16, (E, 128, 2, 1)),
                                                 mk(Float32, (E, 128, 2, 1)),
                                                 mk(Float16, (E, 128, 2, 1)); scale = 0.1f0) === nothing
            @test LINALG.attention_kernel_config(dev, mk(Float16, (E, 128)), mk(Float16, (E, 128)),
                                                 mk(Float16, (E, 128)),
                                                 mk(Float16, (E, 128)); scale = 0.1f0) === nothing
            # Batch counts that disagree between the operands.
            @test LINALG.attention_kernel_config(dev, mk(Float16, (E, 128, 2, 1)),
                                                 mk(Float16, (E, 128, 2, 1)),
                                                 mk(Float16, (E, 128, 3, 1)),
                                                 mk(Float16, (E, 128, 3, 1)); scale = 0.1f0) === nothing
        end
    end
end

@testset "Metal: MPSGraph product and attention" begin
    bq = Metal.global_queue(Metal.device())

    @testset "the product is ordered against the launches around it" begin
        n = 64
        A = MtlArray(zeros(Float32, n, n))
        B = MtlArray(zeros(Float32, n, n))
        C = MtlArray(fill(-1.0f0, n, n))
        tr = MtlArray(zeros(Float32, 1))
        # No synchronisation between the launches and the product: set both operands,
        # multiply, read the product back with a third launch. A product that ran ahead
        # of the two launches that fill its operands gives a product of zeros and a trace
        # read from it.
        Metal.@metal threads=256 groups=cld(n*n, 256) queue=bq linalg_setval!(A, 1.0f0, Int32(n*n))
        Metal.@metal threads=256 groups=cld(n*n, 256) queue=bq linalg_setval!(B, 1.0f0, Int32(n*n))
        LINALG.mpsgemm!(bq, C, A, B)
        Metal.@metal threads=1 groups=1 queue=bq linalg_trace!(tr, C, Int32(n))
        Metal.synchronize(bq)
        @test all(==(Float32(n)), Array(C))
        @test Array(tr)[1] == Float32(n * n)
    end

    @testset "a suballocated operand is read where it lies" begin
        M, N, K = 64, 48, 32
        Ah = rand(Float32, M, K)
        Bh = rand(Float32, K, N)
        big = MtlArray(zeros(Float32, 1 << 18))
        A = linalg_slice(big, 1024, (M, K))
        B = linalg_slice(big, 1024 + M * K, (K, N))
        C = linalg_slice(big, 1024 + M * K + K * N, (M, N))
        copyto!(A, Ah); copyto!(B, Bh)
        LINALG.mpsgemm!(bq, C, A, B)
        Metal.synchronize(bq)
        @test Array(C) ≈ Ah * Bh
        # An ignored offset would have written the product at the start of the buffer,
        # over bytes nobody in this product owns.
        @test all(iszero, Array(view(big, 1:1024)))
    end

    # Bound with packed rows, an operand whose innermost extent is not a multiple of
    # sixteen bytes binds at an offset as well: 3 Float32 is 12 bytes.
    @testset "rows that are not sixteen bytes bind at an offset" begin
        M, N, K = 3, 5, 7
        Ah = rand(Float32, M, K); Bh = rand(Float32, K, N); bh = rand(Float32, M)
        big = MtlArray(fill(42.0f0, 256))
        A = linalg_slice(big, 3, (M, K))
        B = linalg_slice(big, 3 + M * K + 1, (K, N))
        bias = linalg_slice(big, 3 + M * K + 1 + K * N, (M,))
        C = linalg_slice(big, 3 + M * K + 1 + K * N + M + 2, (M, N))
        copyto!(A, Ah); copyto!(B, Bh); copyto!(bias, bh)
        @test LINALG.supported(LINALG.mpsgemm!, C, A, B, bias)
        LINALG.mpsgemm!(bq, C, A, B, bias)
        Metal.synchronize(bq)
        @test Array(C) ≈ Ah * Bh .+ bh
        tail = 3 + M * K + 1 + K * N + M + 2 + M * N
        @test all(==(42.0f0), Array(view(big, (tail + 1):256)))
    end

    @testset "a row bias is folded into the product" begin
        M, N, K = 32, 48, 16
        Ah = rand(Float32, M, K); Bh = rand(Float32, K, N); bh = rand(Float32, M)
        C = MtlArray(zeros(Float32, M, N))
        LINALG.mpsgemm!(bq, C, MtlArray(Ah), MtlArray(Bh), MtlArray(bh))
        Metal.synchronize(bq)
        @test Array(C) ≈ Ah * Bh .+ bh
    end

    # `matmul2d` accumulates in Float32 and stores once, which is why a caller may ask for
    # a Float32 destination from Float16 operands and get the accumulator it wanted.
    # MPSGraph's product has no compute type — its output dtype follows its inputs — so the
    # same ask is a HALF product cast afterwards. Taking it moved SAM 2.1's encoder
    # `add_129` from 0.48 to 1.12, past the band the Vulkan reference sits in, and nothing
    # about the result looked wrong.
    @testset "a destination wider than the operands is refused" begin
        half(dims...) = linalg_zeros(Float16, dims...)
        wide(dims...) = linalg_zeros(Float32, dims...)
        @test LINALG.supported(LINALG.mpsgemm!, half(64, 64), half(64, 64), half(64, 64), nothing)
        @test !LINALG.supported(LINALG.mpsgemm!, wide(64, 64), half(64, 64), half(64, 64), nothing)
        @test LINALG.supported(LINALG.mpsgemm!, wide(64, 64), wide(64, 64), wide(64, 64), nothing)
    end

    @testset "the activation on the store" begin
        # A&S 7.1.26, in Float64 where its 1.5e-7 is far below what fp16 can show.
        function erf_ref(x::Float64)
            a = abs(x)
            t = 1.0 / (1.0 + 0.3275911a)
            y = 1.0 - (((((1.061405429t - 1.453152027)t + 1.421413741)t -
                         0.284496736)t + 0.254829592)t) * exp(-a * a)
            x < 0 ? -y : y
        end
        gelu_ref(x) = 0.5x * (1 + erf_ref(x / sqrt(2.0)))
        gelutanh_ref(x) = 0.5x * (1 + tanh(0.7978845608028654 * (x + 0.044715x^3)))
        # The product is `xs[i]` in every column, so the activation is exercised over the
        # whole range — including below -2, where `1 + erf` cancels and a half evaluation
        # of it keeps a couple of significant bits.
        Mw, Nw, Kw = 512, 8, 16
        Ah = zeros(Float16, Mw, Kw); Bh = zeros(Float16, Kw, Nw)
        xs = Float32.(range(-8, 8; length = Mw))
        Ah[:, 1] .= Float16.(xs); Bh[1, :] .= Float16(1)
        A, B = MtlArray(Ah), MtlArray(Bh)
        for (act, ref) in ((:identity, identity), (:relu, x -> max(x, 0.0)),
                           (:gelu, gelu_ref), (:gelu_tanh, gelutanh_ref))
            C = MtlArray(fill(Float16(NaN), Mw, Nw))
            LINALG.mpsgemm!(bq, C, A, B, nothing, act)
            Metal.synchronize(bq)
            got = Float64.(Array(C)[:, 1])
            want = ref.(Float64.(xs))
            @test all(isfinite, got)
            # Half's own rounding at this scale, and nothing more: an activation evaluated
            # in half rather than single is an order of magnitude worse.
            @test maximum(abs, got .- want) / maximum(abs, want) < 1e-3
        end
        @test_throws ArgumentError LINALG.mpsgemm!(
            bq, MtlArray(zeros(Float16, Mw, Nw)), A, B, nothing, :softplus)

        # An INTEGER operand is the one that used to end the process rather than the
        # call: the graph hands its placeholders straight to `mps.matmul`, which has no
        # integer node, and MPSGraph answers by calling `abort()`.
        let Ai = MtlArray(fill(Int8(1), Mw, 8)), Bi = MtlArray(fill(Int8(1), 8, Nw))
            @test !LINALG.supported(LINALG.mpsgemm!, MtlArray(zeros(Float16, Mw, Nw)), Ai, Bi, nothing)
            @test_throws ArgumentError LINALG.mpsgemm!(bq, MtlArray(zeros(Float16, Mw, Nw)), Ai, Bi)
        end
    end

    @testset "which operands the product admits" begin
        ok(dims...) = linalg_zeros(Float32, dims...)
        @test LINALG.supported(LINALG.mpsgemm!, ok(8, 8), ok(8, 8), ok(8, 8), nothing)
        # An activation the graph has no node for is not a shape it admits.
        @test !LINALG.supported(LINALG.mpsgemm!, ok(8, 8), ok(8, 8), ok(8, 8), nothing, :softplus)
        @test LINALG.supported(LINALG.mpsgemm!, ok(8, 8), ok(8, 8), ok(8, 8), nothing, :gelu)
        # Rank, and a contraction length that does not match.
        @test !LINALG.supported(LINALG.mpsgemm!, ok(8, 8, 2), ok(8, 8), ok(8, 8), nothing)
        @test !LINALG.supported(LINALG.mpsgemm!, ok(8, 8), ok(8, 4), ok(8, 8), nothing)
        # A bias that is not one value per row.
        @test !LINALG.supported(LINALG.mpsgemm!, ok(8, 8), ok(8, 8), ok(8, 8), ok(8, 8))
        @test !LINALG.supported(LINALG.mpsgemm!, ok(8, 8), ok(8, 8), ok(8, 8), ok(4))
        # An empty operand, on which MPS aborts the process.
        @test !LINALG.supported(LINALG.mpsgemm!, ok(8, 0), ok(8, 4), ok(4, 0), nothing)
    end

    # Each call is a command buffer of its own, holding one graph. Metal aborts the
    # process past 32 residency sets in one command buffer, and every distinct graph adds
    # some; reaching the end with every product right is the test.
    @testset "many distinct graphs back to back" begin
        ops = [(MtlArray(rand(Float32, 16k, 64)), MtlArray(rand(Float32, 64, 40)),
                MtlArray(zeros(Float32, 16k, 40))) for k in 1:20]
        for (A, B, C) in ops
            LINALG.mpsgemm!(bq, C, A, B)
        end
        Metal.synchronize(bq)
        @test all(((A, B, C),) -> Array(C) ≈ Array(A) * Array(B), ops)
    end

    @testset "fused attention" begin
        E, Lq, Lk, H, B = 72, 128, 256, 3, 2
        scale = Float32(inv(sqrt(E)))
        qh = Float16.(reshape(0.4 .* sin.(range(0, 9, E*Lq*H*B)), E, Lq, H, B))
        kh = Float16.(reshape(0.4 .* cos.(range(0, 7, E*Lk*H*B)), E, Lk, H, B))
        vh = Float16.(reshape(0.4 .* sin.(range(0, 5, E*Lk*H*B)), E, Lk, H, B))
        o = MtlArray(fill(Float16(NaN), E, Lq, H, B))
        q, k, v = MtlArray(qh), MtlArray(kh), MtlArray(vh)
        @test linalg_mpsattention!(bq, o, q, k, v, scale) !== nothing
        Metal.synchronize(bq)
        got = Array(o)
        # The destination starts as NaN, so a plane the op skipped cannot pass.
        @test all(isfinite, got)
        ref = linalg_attnref(qh, kh, vh, Float64(scale))
        @test maximum(abs, Float64.(got) .- ref) / maximum(abs, ref) < 3e-3

        # …and which operands it admits.
        mk(T, dims) = MtlArray(zeros(T, dims...))
        @test LINALG.attentionoperands(mk(Float16, (72, 8)), mk(Float16, (72, 8)),
                                       mk(Float16, (72, 8)), mk(Float16, (72, 8))) === nothing
        @test LINALG.attentionoperands(mk(Float16, (72, 8, 1, 1)), mk(Float16, (72, 8, 1, 1)),
                                       mk(Float32, (72, 8, 1, 1)),
                                       mk(Float16, (72, 8, 1, 1))) === nothing
        @test LINALG.attentionoperands(mk(Float16, (72, 8, 1, 1)), mk(Float16, (70, 8, 1, 1)),
                                       mk(Float16, (70, 8, 1, 1)),
                                       mk(Float16, (70, 8, 1, 1))) === nothing
    end

    # The RESULT type is asked separately from the operand type. A graph that projects in
    # fp16 and declares an fp32 result is an ordinary export -- it is what Qwen-Image 2.1
    # does -- and requiring all four to agree declined 32 of its attention ops into the
    # three-pass path, ~85 s of a 144 s step.
    #
    # The wider destination is a WIDENED fp16 answer and not an fp32 accumulator: the op's
    # result dtype follows its inputs. That is pinned below, because it is the one thing
    # about this gate somebody will assume the other way round, and because it is exactly
    # the property that makes the product's gate refuse the same pair. What must NOT be
    # admitted is q, k and v disagreeing with EACH OTHER, which the testset above pins.
    @testset "the attention result type need not be the operand type" begin
        E, Lq, Lk, H, B = 72, 64, 96, 2, 2
        scale = Float32(inv(sqrt(E)))
        qh = Float16.(reshape(0.4 .* sin.(range(0, 9, E*Lq*H*B)), E, Lq, H, B))
        kh = Float16.(reshape(0.4 .* cos.(range(0, 7, E*Lk*H*B)), E, Lk, H, B))
        vh = Float16.(reshape(0.4 .* sin.(range(0, 5, E*Lk*H*B)), E, Lk, H, B))
        q, k, v = MtlArray(qh), MtlArray(kh), MtlArray(vh)
        ref = linalg_attnref(qh, kh, vh, Float64(scale))

        # NaN rather than zeros: a plane the op skipped reads as an answer otherwise,
        # which is how a read-before-write hides.
        o32 = MtlArray(fill(Float32(NaN), E, Lq, H, B))
        @test linalg_mpsattention!(bq, o32, q, k, v, scale) !== nothing
        o16 = MtlArray(fill(Float16(NaN), E, Lq, H, B))
        @test linalg_mpsattention!(bq, o16, q, k, v, scale) !== nothing
        Metal.synchronize(bq)
        g32, g16 = Array(o32), Array(o16)
        @test all(isfinite, g32)
        @test all(isfinite, g16)
        e32 = maximum(abs, Float64.(g32) .- ref) / maximum(abs, ref)
        e16 = maximum(abs, Float64.(g16) .- ref) / maximum(abs, ref)
        @test e32 < 3e-3
        # The SAME number, not a smaller one: the op computed in fp16 either way and the
        # fp32 destination holds that answer widened. Pinned as an equality so that a
        # future MPSGraph which really does accumulate in single fails here and gets the
        # docstring corrected, rather than quietly making a `<=` look prescient.
        @test e32 == e16

        # fp32 operands with an fp16 destination is the same door, the other way, and
        # there the operand type is what carries the precision: computed in single and
        # narrowed once at the end, it beats the fp16 operands above.
        o = MtlArray(fill(Float16(NaN), E, Lq, H, B))
        @test linalg_mpsattention!(bq, o, MtlArray(Float32.(qh)), MtlArray(Float32.(kh)),
                                   MtlArray(Float32.(vh)), scale) !== nothing
        o32 = MtlArray(fill(Float32(NaN), E, Lq, H, B))
        @test linalg_mpsattention!(bq, o32, MtlArray(Float32.(qh)), MtlArray(Float32.(kh)),
                                   MtlArray(Float32.(vh)), scale) !== nothing
        Metal.synchronize(bq)
        @test all(isfinite, Array(o))
        @test maximum(abs, Float64.(Array(o)) .- ref) / maximum(abs, ref) < 3e-3
        @test maximum(abs, Float64.(Array(o32)) .- ref) / maximum(abs, ref) < e32
    end

    # A projection leaves q, k and v as three windows on ONE buffer. Both have to come
    # out the same: the library reads the window where it lies, by slicing the dense
    # array inside the graph, and a copy of the same bytes into a packed array is what
    # that has to agree with.
    @testset "attention reads three windows of one buffer" begin
        E, H, L, B = 16, 2, 16, 2
        scale = Float32(inv(sqrt(E)))
        # `(E, H, 3, L, B)`, which is the layout a fused qkv projection writes.
        ph = Float16.(reshape(0.3 .* sin.(range(0, 11, E*H*3*L*B)), E, H, 3, L, B))
        parent = MtlArray(ph)
        st = (1, E, E * H, E * H * 3, E * H * 3 * L)
        window(j) = (; res = parent, dims = (E, L, H, B),
                     strides = (st[1], st[4], st[2], st[5]), offset = (j - 1) * st[3])
        packed(j) = Float16.(permutedims(ph[:, :, j, :, :], (1, 3, 2, 4)))

        o  = MtlArray(fill(Float16(NaN), E, L, H, B))
        od = MtlArray(fill(Float16(NaN), E, L, H, B))
        @test linalg_mpsattention!(bq, o, window(1), window(2), window(3), scale) !== nothing
        @test linalg_mpsattention!(bq, od, MtlArray(packed(1)), MtlArray(packed(2)),
                                   MtlArray(packed(3)), scale) !== nothing
        Metal.synchronize(bq)
        @test all(isfinite, Array(o))
        @test Array(o) == Array(od)
        ref = linalg_attnref(packed(1), packed(2), packed(3), Float64(scale))
        @test maximum(abs, Float64.(Array(o)) .- ref) / maximum(abs, ref) < 3e-3
    end

    @testset "which strided views are windows on a dense array" begin
        parent = MtlArray(zeros(Float16, 16 * 2 * 3 * 16 * 2))
        view4(st, off) = (; res = parent, dims = (16, 16, 2, 2), strides = st, offset = off)
        # SAM 2.1's own: the middle axis of `(E, H, 3, L, B)` is the gap.
        p = LINALG.packoperand(view4((1, 96, 16, 1536), 32))
        @test p !== nothing
        @test p.memdims == (16, 2, 3, 16, 2)
        @test p.tags == (1, 3, 0, 2, 4)
        @test p.starts == [1]
        # A stride that is not a multiple of what precedes it has no dense array to be a
        # window on, and an offset that does not land on a gap names no window.
        @test LINALG.packoperand(view4((1, 100, 16, 1600), 0)) === nothing
        @test LINALG.packoperand(view4((1, 96, 16, 1536), 33)) === nothing
        # A dense array is its own window, with no gaps and nothing to reorder.
        d = LINALG.packoperand(MtlArray(zeros(Float16, 4, 5)))
        @test d.memdims == (4, 5) && isempty(d.gapat) && d.tags == (1, 2)
    end
end

# Apple's direct 2-D convolution, in the reversed layout a declared graph uses: `x` is
# `(W, H, Cin, N)`, `w` is `(KW, KH, Cin ÷ groups, Cout)`, `out` is `(OW, OH, Cout, N)`.
# MPSGraph sees those as `NCHW` and `OIHW` with no transposition, which is the whole reason
# this is cheap to reach for — and the thing a wrong descriptor gets silently wrong, so
# every geometry below is checked against a reference rather than against another Metal
# path.
@testset "Metal: MPSGraph convolution" begin
    bq = Metal.global_queue(Metal.device())
    conv = LINALG.mpsconv2d!

    @testset "geometry — $KW×$KH s$(st) p$(pd) d$(dl) g$gr" for
            (W, Cin, Cout, KW, KH, st, pd, dl, gr) in
            ((32,  8, 16, 3, 3, (1, 1), (1, 1), (1, 1), 1),
             (64, 16,  8, 7, 7, (4, 4), (3, 3), (1, 1), 1),   # SAM 2.1's stem shape
             (32,  8,  8, 1, 1, (1, 1), (0, 0), (1, 1), 1),
             (32, 16, 16, 3, 3, (2, 2), (1, 1), (2, 2), 1),   # strided AND dilated
             (32, 12,  8, 3, 5, (1, 2), (1, 2), (1, 1), 1),   # asymmetric in x and y
             (16, 16,  8, 3, 3, (1, 1), (1, 1), (1, 1), 4))   # grouped
        xh = Float16.(0.2f0 .* randn(Float32, W, W, Cin, 1))
        wh = Float16.(0.2f0 .* randn(Float32, KW, KH, Cin ÷ gr, Cout))
        bh = Float16.(0.3f0 .* randn(Float32, Cout))
        want = linalg_convref(xh, wh, bh, st, pd, dl, gr)
        o = MtlArray(fill(Float16(NaN), size(want, 1), size(want, 2), Cout, 1))
        @test LINALG.supported(conv, o, MtlArray(xh), MtlArray(wh), MtlArray(bh); groups = gr)
        conv(bq, o, MtlArray(xh), MtlArray(wh), MtlArray(bh), st, pd, dl, gr)
        Metal.synchronize(bq)
        got = Float64.(Array(o))
        # The destination starts as NaN, so an output the op did not cover cannot pass,
        # and a descriptor that got the geometry wrong lands nowhere near the reference.
        @test all(isfinite, got)
        @test maximum(abs, got .- want) / maximum(abs, want) < 2e-3
    end

    @testset "the activation and the bias on the store" begin
        xh = Float16.(0.4f0 .* randn(Float32, 24, 24, 8, 1))
        wh = Float16.(0.4f0 .* randn(Float32, 3, 3, 8, 8))
        bh = Float16.(0.5f0 .* randn(Float32, 8))
        want = linalg_convref(xh, wh, bh, (1, 1), (1, 1), (1, 1), 1)
        o = MtlArray(fill(Float16(NaN), 24, 24, 8, 1))
        conv(bq, o, MtlArray(xh), MtlArray(wh), MtlArray(bh), (1, 1), (1, 1), (1, 1), 1, :relu)
        Metal.synchronize(bq)
        @test maximum(abs, Float64.(Array(o)) .- max.(want, 0.0)) / maximum(abs, want) < 2e-3
        # No bias, and the same convolution: the bias placeholder is the only thing that
        # goes away, and it must not shift the rest.
        o2 = MtlArray(fill(Float16(NaN), 24, 24, 8, 1))
        conv(bq, o2, MtlArray(xh), MtlArray(wh), nothing, (1, 1), (1, 1), (1, 1), 1)
        Metal.synchronize(bq)
        w0 = linalg_convref(xh, wh, nothing, (1, 1), (1, 1), (1, 1), 1)
        @test maximum(abs, Float64.(Array(o2)) .- w0) / maximum(abs, w0) < 2e-3
    end

    # A weight's innermost extent is its KERNEL width, which is never a multiple of
    # sixteen bytes; bound with packed rows it is read where it lies all the same, here
    # as a suballocation would hand it over — every operand at an offset into one buffer.
    @testset "operands with unaligned rows, at offsets" begin
        W, Cin, Cout = 20, 3, 8
        xh = Float16.(0.3f0 .* randn(Float32, W, W, Cin, 1))
        wh = Float16.(0.2f0 .* randn(Float32, 7, 7, Cin, Cout))
        bh = Float16.(0.2f0 .* randn(Float32, Cout))
        want = linalg_convref(xh, wh, bh, (2, 2), (3, 3), (1, 1), 1)
        big = MtlArray(fill(Float16(42), 1 << 14))
        x = linalg_slice(big, 5, size(xh))
        w = linalg_slice(big, 5 + length(xh) + 3, size(wh))
        b = linalg_slice(big, 5 + length(xh) + 3 + length(wh) + 1, size(bh))
        o = linalg_slice(big, 5 + length(xh) + 3 + length(wh) + 1 + length(bh) + 7, size(want))
        copyto!(x, xh); copyto!(w, wh); copyto!(b, bh)
        @test LINALG.supported(conv, o, x, w, b)
        conv(bq, o, x, w, b, (2, 2), (3, 3), (1, 1), 1)
        Metal.synchronize(bq)
        got = Float64.(Array(o))
        @test all(isfinite, got)
        @test maximum(abs, got .- want) / maximum(abs, want) < 2e-3
    end

    @testset "which operands it admits" begin
        mk = linalg_zeros
        @test LINALG.supported(conv, mk(Float16, 8, 8, 4, 1), mk(Float16, 8, 8, 4, 1),
                               mk(Float16, 3, 3, 4, 4), nothing)
        # A destination wider than the operands, for the reason the product refuses one:
        # there is no compute type to ask for.
        @test !LINALG.supported(conv, mk(Float32, 8, 8, 4, 1), mk(Float16, 8, 8, 4, 1),
                                mk(Float16, 3, 3, 4, 4), nothing)
        # Rank, a channel count that does not match the weight on either side, an
        # activation with no node, and a bias that is not one value per output channel.
        @test !LINALG.supported(conv, mk(Float16, 8, 8, 4), mk(Float16, 8, 8, 4),
                                mk(Float16, 3, 3, 4, 4), nothing)
        @test !LINALG.supported(conv, mk(Float16, 8, 8, 8, 1), mk(Float16, 8, 8, 4, 1),
                                mk(Float16, 3, 3, 4, 4), nothing)
        @test !LINALG.supported(conv, mk(Float16, 8, 8, 4, 1), mk(Float16, 8, 8, 6, 1),
                                mk(Float16, 3, 3, 4, 4), nothing)
        @test !LINALG.supported(conv, mk(Float16, 8, 8, 4, 1), mk(Float16, 8, 8, 4, 1),
                                mk(Float16, 3, 3, 4, 4), nothing, :softplus)
        @test !LINALG.supported(conv, mk(Float16, 8, 8, 4, 1), mk(Float16, 8, 8, 4, 1),
                                mk(Float16, 3, 3, 4, 4), mk(Float16, 8))
    end

    # SAM 2.1's stem at its real size, against the SAME graph in single. This is the
    # measurement that settles whether the library convolution costs accuracy: the answer
    # is half's own output rounding and not one bit more, so `add_129` moving when the stem
    # changes route is a different valid rounding rather than a loss.
    @testset "the stem is at half's rounding floor" begin
        Cout = 144
        xh = Float16.(0.3f0 .* randn(Float32, 1024, 1024, 3, 1))
        wh = Float16.(0.05f0 .* randn(Float32, 7, 7, 3, Cout))
        bh = Float16.(0.2f0 .* randn(Float32, Cout))
        o16 = MtlArray(fill(Float16(NaN), 256, 256, Cout, 1))
        o32 = MtlArray(fill(NaN32, 256, 256, Cout, 1))
        conv(bq, o16, MtlArray(xh), MtlArray(wh), MtlArray(bh), (4, 4), (3, 3), (1, 1), 1)
        conv(bq, o32, MtlArray(Float32.(xh)), MtlArray(Float32.(wh)), MtlArray(Float32.(bh)),
             (4, 4), (3, 3), (1, 1), 1)
        Metal.synchronize(bq)
        a, b = Float64.(Array(o16)), Float64.(Array(o32))
        sc = maximum(abs, b)
        @test all(isfinite, a)
        # Half's own spacing at that magnitude, with a hair of slack. An accumulator that
        # was half rather than single would be an order of magnitude over this.
        @test maximum(abs, a .- b) / sc < 1.2 * Float64(eps(Float16(sc))) / 2 / sc
    end

    # A TRANSPOSED convolution is the same node's data gradient, and the layouts line up
    # without a transposition: ATen's transposed weight is `(KW, KH, Cout ÷ groups, Cin)`,
    # which reversed is MPSGraph's `OIHW` for the forward convolution that maps this node's
    # OUTPUT back to its input.
    #
    # Checked against that forward convolution rather than against a reimplementation:
    # `conv_transpose2d(x, W)` scatters exactly what `conv(y, W)` gathers, so summing the
    # product both ways has to agree. RIFE's decoder upsamples with seven of these and they
    # were 438 ms of its 581 ms frame on the gather kernel; 21 ms here.
    @testset "a transposed convolution is the data gradient" begin
        mk = linalg_zeros
        # Which axis carries the output channels is the one thing that moves.
        @test LINALG.supported(conv, mk(Float16, 16, 16, 6, 1), mk(Float16, 8, 8, 4, 1),
                               mk(Float16, 4, 4, 6, 4), nothing; transposed = true, groups = 1)
        # The FORWARD reading of the same operands is not a legal shape, so the flag is
        # doing the work rather than both happening to pass.
        @test !LINALG.supported(conv, mk(Float16, 16, 16, 6, 1), mk(Float16, 8, 8, 4, 1),
                                mk(Float16, 4, 4, 6, 4), nothing)

        Cin, Cout, W, H = 4, 6, 8, 8
        xh = 0.3f0 .* randn(Float32, W, H, Cin, 1)
        wh = 0.2f0 .* randn(Float32, 4, 4, Cout, Cin)
        ot = MtlArray(fill(NaN32, 2W, 2H, Cout, 1))
        conv(bq, ot, MtlArray(xh), MtlArray(wh), nothing, (2, 2), (1, 1), (1, 1), 1,
             :identity; transposed = true)
        Metal.synchronize(bq)
        y = Array(ot)
        @test all(isfinite, y)

        # `<y, conv_transpose(x, W)> == <conv(y, W), x>`, which is what "data gradient"
        # means and needs no second implementation of either.
        yr = randn(Float32, 2W, 2H, Cout, 1)
        fwd = MtlArray(fill(NaN32, W, H, Cin, 1))
        conv(bq, fwd, MtlArray(yr), MtlArray(wh), nothing, (2, 2), (1, 1), (1, 1), 1)
        Metal.synchronize(bq)
        lhs = sum(Float64.(yr) .* Float64.(y))
        rhs = sum(Float64.(Array(fwd)) .* Float64.(xh))
        @test isapprox(lhs, rhs; rtol = 1e-4)
    end
end

# A library call in a recorded plan commits its own command buffer between the frame's
# dispatches: the recording stops at the call, the frame's work so far is committed, the
# call's command buffer follows, and the rest of the frame after it. A dispatch on each
# side of the call, read back, is what shows the order.
@testset "Metal: a library product in a recorded plan keeps its place" begin
    dev = Mantle.Device(Mantle.MetalAPI())
    if !Mantle.runscalls(dev)
        @test_skip Mantle.runscalls(dev)
    else
        n = 64
        g = Mantle.Graph(dev)
        A = Mantle.Transient.Buffer(g, Float32, n, n)
        B = Mantle.Transient.Buffer(g, Float32, n, n)
        C = Mantle.Transient.Buffer(g, Float32, n, n)
        out = Mantle.Buffer(dev, zeros(Float32, n))
        Mantle.dispatch!(g, linalg_fill!, (A, 1.0f0), n * n; name = "fill A")
        Mantle.dispatch!(g, linalg_fill!, (B, 2.0f0), n * n; name = "fill B")
        call = Mantle.librarygemm(dev, C, A, B, nothing, identity)
        @test call !== nothing
        Mantle.dispatch!(g, call, (C, A, B, nothing); name = "mpsgemm")
        Mantle.dispatch!(g, linalg_rowsum!, (out, C, Int32(n)), n; name = "row sums")
        plan = Mantle.record!(Mantle.Plan(g; budget = Inf))

        # A is 1, B is 2: every element of C is 2n, every row sum 2n².
        for frame in 1:2
            fill!(Mantle.storage(out), 0.0f0)
            Mantle.run!(plan)
            Mantle.waitfor!(plan)
            @test all(==(Float32(2n * n)), Array(Mantle.storage(out)))
        end
        Mantle.free!(plan)
    end
end
