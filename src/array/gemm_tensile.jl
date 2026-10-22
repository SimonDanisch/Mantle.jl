# rocBLAS's fp16 NN GEMM for RDNA3 (Tensile MT128x128x16_MI16x16x16x1, gfx1100),
# ported from its disassembly: a 128 x 128 tile per workgroup, k steps of 16, four
# 64 x 64 subgroups of 4 x 4 cooperative matrices, two LDS buffers (one barrier per
# step). About 91% of rocBLAS on an RX 7900 XTX, from three choices: lanes address
# through 32-bit byte offsets from a uniform base (indexing makes ACO recompute
# 64-bit addresses every step); each step stores the tile fetched a step earlier
# before fetching (ACO hoists LDS stores above the WMMAs anyway); workgroups group
# along N. The rest of the gap is not reachable from SPIR-V.

"""
Whether `coopmat_gemm!` takes the Tensile kernel for the products
`tensile_gemm_ok` accepts. On because it beats every staged tiling on the RX 7900
XTX and the RTX 4000 Ada; a benchmark of the staged family passes `tensile = false`.
"""
const GEMM_TENSILE_DEFAULT = true

"""Eight fp16 values as one 128-bit global load."""
const GemmU4 = NTuple{4,VecElement{UInt32}}

const TENSILE_BM = 128               # output rows per workgroup
const TENSILE_BN = 128               # output columns per workgroup
const TENSILE_BK = 16                # k per step: one cooperative matrix deep
const TENSILE_WG = 4 * COOPMAT_SUBGROUP

"""
B's LDS row in fp16: the step's 16 values plus 24 of padding. A cooperative-matrix
load needs a stride that is a 16-byte multiple
(VUID-RuntimeSpirv-OpCooperativeMatrixLoadKHR-08986); of those, 40 measured best
on gfx1100, while 32 puts a subgroup's rows on the same banks. Tensile's own
padding has no single stride, so a cooperative-matrix load cannot express it.
"""
const TENSILE_LDB = 40

"""
Column blocks per group of workgroups: eight consecutive workgroups take eight
column blocks of one row block and share its A tile in cache. 4 and 16 are within
noise; grouping row blocks instead is slower.
"""
const TENSILE_GROUP = 8

"""
Output tile `(tm, tn)` of workgroup `wg` on an `NM x NN` grid of 128 x 128 tiles:
in groups of `TENSILE_GROUP` column blocks, walked down N first. The last group
is narrower when `NN` is not a multiple of the group.
"""
@inline function tensile_tile(wg, ::Val{NM}, ::Val{NN}) where {NM,NN}
    G = TENSILE_GROUP
    r, grp = splitidx(wg, Val(G * NM))
    first = grp * G
    tail = NN % G
    ni, tm = tail != 0 && first + G > NN ? splitidx(r, Val(tail)) : splitidx(r, Val(G))
    return tm, first + ni
end

"""
Sixteen bytes at byte offset `off` from `p`: one 128-bit load. `p` is a device
address on Lava and an LLVM pointer on ROCm; `off` is zero-extended, which is
what lets the compiler keep `p` uniform and `off` per lane.
"""
@inline gemmload16(p::Ptr, off::UInt32) = Core.Intrinsics.pointerref(reinterpret(Ptr{GemmU4}, p + off), 1, 16)
@inline gemmload16(p::Core.LLVMPtr{T,A}, off::UInt32) where {T,A} =
    unsafe_load(reinterpret(Core.LLVMPtr{GemmU4,A}, p + off), 1, Val(16))

"""
Store a 128-bit load into `f16vec4` LDS at element `i` as its two 64-bit halves,
which the compilers fuse back into one store. `f16vec4` because every backend can
load a cooperative matrix from it.
"""
@inline function stage16!(s, i, v::GemmU4)
    lo = Base.llvmcall("""
        %s = shufflevector <4 x i32> %0, <4 x i32> poison, <2 x i32> <i32 0, i32 1>
        %h = bitcast <2 x i32> %s to <4 x half>
        ret <4 x half> %h""", GemmV4, Tuple{GemmU4}, v)
    hi = Base.llvmcall("""
        %s = shufflevector <4 x i32> %0, <4 x i32> poison, <2 x i32> <i32 2, i32 3>
        %h = bitcast <2 x i32> %s to <4 x half>
        ret <4 x half> %h""", GemmV4, Tuple{GemmU4}, v)
    @inbounds s[i] = lo
    @inbounds s[i + 1] = hi
    return nothing
end

let
    # The three pieces of a k step, spliced into the kernel below so the body is
    # written once and its locals stay plain SSA values, as the emitter needs.
    SA = TENSILE_BK * TENSILE_BM ÷ 4       # one A buffer, in f16vec4
    LB = TENSILE_LDB ÷ 4                   # one B row, in f16vec4
    SB = TENSILE_BN * LB                   # one B buffer, in f16vec4

    # Four 128-bit loads of k step `kstep`: A rows m0.., columns ak and ak + 8;
    # B columns n0 + bn and n0 + bn + 64, the 8 k values at bc.
    fetch = quote
        pa = pA + (m0 + kstep * $TENSILE_BK * M) * sizeof(Float16)
        pb = pB + (n0 * K + kstep * $TENSILE_BK) * sizeof(Float16)
        a_1 = gemmload16(pa, oa_1)
        a_2 = gemmload16(pa, oa_2)
        b_1 = gemmload16(pb, ob_1)
        b_2 = gemmload16(pb, ob_2)
    end
    # The fetched chunks into LDS buffer `buf`.
    stage = quote
        stage16!(sA, 1 + buf * $SA + ak * $(TENSILE_BM ÷ 4) + 2ac, a_1)
        stage16!(sA, 1 + buf * $SA + (ak + 8) * $(TENSILE_BM ÷ 4) + 2ac, a_2)
        stage16!(sB, 1 + buf * $SB + bn * $LB + 2bc, b_1)
        stage16!(sB, 1 + buf * $SB + (bn + 64) * $LB + 2bc, b_2)
    end
    # One k step out of LDS buffer `buf`: four A and four B fragments, their 4 x 4
    # products. A is m-major in LDS, i.e. a column-major A operand.
    multiply = quote
        Base.Cartesian.@nexprs 4 i -> fa_i = AcceleratedMatrix{Float16,GEMM_TILE,GEMM_TILE,MatrixA}(
            sA, 1 + buf * $SA + (wm + GEMM_TILE * (i - 1)) ÷ 4, $(TENSILE_BM ÷ 4))
        Base.Cartesian.@nexprs 4 j -> fb_j = AcceleratedMatrix{Float16,GEMM_TILE,GEMM_TILE,MatrixB}(
            sB, 1 + buf * $SB + (wn + GEMM_TILE * (j - 1)) * $LB, $LB)
        Base.Cartesian.@nexprs 4 j -> Base.Cartesian.@nexprs 4 i -> c_i_j = muladd(fa_i, fb_j, c_i_j)
    end

    @eval begin
        """
            coopmat_gemm_tensile_kernel!(C, A, B, bias, epi, Val(M), Val(N), Val(K))

        `C = epi.(A * B .+ bias)` for fp16 `A` (`M x K`) and `B` (`K x N`), fp32
        accumulation, `C` fp16 or fp32. One workgroup of `TENSILE_WG` lanes per
        128 x 128 tile, under the constraints `tensile_gemm_ok` checks. The k loop
        runs steps in pairs, one per LDS buffer, so buffer offsets are constants.
        Each step stores the tile fetched one step earlier into the buffer the
        previous step's barrier freed, fetches the next, multiplies, synchronises.
        """
        function coopmat_gemm_tensile_kernel!(C, A, B, bias, epi,
                                              ::Val{M}, ::Val{N}, ::Val{K}) where {M,N,K}
            sA = KI.localmemory(GemmV4, Val((2 * $SA,)), Val(1))
            sB = KI.localmemory(GemmV4, Val((2 * $SB,)), Val(2))
            tid = KI.get_local_id().x - 1
            tm, tn = tensile_tile(KI.get_group_id().x - 1, Val(M ÷ $TENSILE_BM), Val(N ÷ $TENSILE_BN))
            m0 = tm * $TENSILE_BM
            n0 = tn * $TENSILE_BN
            s = tid ÷ COOPMAT_SUBGROUP
            wm = (s & 1) * 64          # the subgroup's 64 x 64 of the tile
            wn = (s >> 1) * 64
            ac = tid & 15              # A: 8 rows from 8ac, column ak (and ak + 8)
            ak = tid >> 4
            bn = tid >> 1              # B: 8 k values from 8bc, column bn (and bn + 64)
            bc = tid & 1
            pA = pointer(A)
            pB = pointer(B)
            oa_1 = ((8ac + ak * M) * sizeof(Float16)) % UInt32
            oa_2 = ((8ac + (ak + 8) * M) * sizeof(Float16)) % UInt32
            ob_1 = ((bn * K + 8bc) * sizeof(Float16)) % UInt32
            ob_2 = (((bn + 64) * K + 8bc) * sizeof(Float16)) % UInt32

            bp = bias === nothing ? bias : pointer(bias)
            Base.Cartesian.@nexprs 4 j -> Base.Cartesian.@nexprs 4 i ->
                c_i_j = accinit(bias, bp, 1 + m0 + wm + GEMM_TILE * (i - 1))

            nk = Int32(K ÷ $TENSILE_BK)
            kstep = Int32(0); $fetch
            buf = 0; $stage
            kstep = Int32(1); $fetch
            KI.barrier()
            k = Int32(0)
            while k < nk - Int32(2)
                buf = 1; $stage
                kstep = k + 2; $fetch
                buf = 0; $multiply
                KI.barrier()
                buf = 0; $stage
                kstep = k + 3; $fetch
                buf = 1; $multiply
                KI.barrier()
                k += Int32(2)
            end
            buf = 1; $stage
            buf = 0; $multiply
            KI.barrier()
            buf = 1; $multiply

            Base.Cartesian.@nexprs 4 j -> Base.Cartesian.@nexprs 4 i ->
                accstore!(C, 1 + (m0 + wm + GEMM_TILE * (i - 1)) + (n0 + wn + GEMM_TILE * (j - 1)) * M, M,
                          c_i_j, epi)
            return nothing
        end
    end
end

"""
    tensile_gemm_ok(C, A, B, M, N, K, nbatch, splitk) -> Bool

Whether this product can run `coopmat_gemm_tensile_kernel!`, which masks nothing,
reads 16 bytes at a time and walks the whole of K in one plane:

  * fp16 `A` and `B`, fp16 or fp32 `C`;
  * one batch, `splitk == 1`;
  * `M` and `N` multiples of 128 and `K` of 32 (k steps run in pairs);
  * `A` and `B` 16-byte aligned (`basealignment`), since the loads step from their
    bases;
  * each operand under 4 GiB, since lanes address it with 32-bit byte offsets.
"""
@inline function tensile_gemm_ok(C, A, B, M::Int, N::Int, K::Int, nbatch::Int, splitk::Int)
    eltype(A) === Float16 && eltype(B) === Float16 || return false
    eltype(C) === Float16 || eltype(C) === Float32 || return false
    nbatch == 1 && splitk == 1 || return false
    M % TENSILE_BM == 0 && N % TENSILE_BN == 0 && K % (2 * TENSILE_BK) == 0 || return false
    basealignment(A) >= 16 && basealignment(B) >= 16 || return false
    return widemul(M, K) * sizeof(Float16) < typemax(UInt32) &&
           widemul(K, N) * sizeof(Float16) < typemax(UInt32)
end

"""The one launch of the Tensile kernel for `C = epilogue.(A * B .+ bias)`."""
tensile_gemm_launch(C, A, B, bias, epilogue, M::Int, N::Int, K::Int) =
    ArrayLaunch(coopmat_gemm_tensile_kernel!, (C, A, B, bias, epilogue, Val(M), Val(N), Val(K)),
                (M ÷ TENSILE_BM) * (N ÷ TENSILE_BN) * TENSILE_WG, TENSILE_WG)
