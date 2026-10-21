# Core GEMM kernels and shape selection use KernelInterface operations.
# Backends route their own array mul! methods here or to native libraries.
# Cooperative matrices require Float16 16×16×16 support and 32-lane subgroups;
# scalar staged and strided kernels cover other supported types/shapes.
# Staging/blocking derives from reference/mul_mm (llama.cpp, MIT).
const GEMM_TILE = 16
const GEMM_BLOCK = 4       # register block: 4×4 accumulators per subgroup
const GEMM_WORKGROUP = 64  # two subgroups for the register-blocked kernel
const GEMM_MAXBLOCK = 4    # largest register block `@nexprs` is unrolled for

# Staged products keep one A and one B fragment live. Hoisting all fragments
# increased registers and spilled accumulators; retain the stagedmma load order.
const GEMM_BK = 32
const GEMM_PAD = 4                  # bank-conflict padding, in elements

"""Staged tiling `(STM, STN, WM, WN, BK, PAD)`: `WM*WN` subgroups each own
`STM*STN` 16×16 output tiles. The workgroup covers `(16*STM*WM, 16*STN*WN)`
and advances K by BK; PAD adds shared-memory leading-dimension padding."""
const GemmTiling = NTuple{6,Int}

@inline gemm_bm(c::GemmTiling) = GEMM_TILE * c[1] * c[3]
@inline gemm_bn(c::GemmTiling) = GEMM_TILE * c[2] * c[4]
@inline gemm_wg(c::GemmTiling) = c[3] * c[4] * COOPMAT_SUBGROUP
@inline gemm_bk(c::GemmTiling) = c[5]
@inline gemm_lda(c::GemmTiling) = gemm_bm(c) + c[6]
@inline gemm_ldb(c::GemmTiling) = gemm_bk(c) + c[6]

"""Compiled staged tilings in selection order. `gemm_tiling` takes the first
exactly dividing, non-aliasing entry; forced tilings still require exact extents.
BK=16 covers reductions that cannot use BK=32. See `test_gemm_staged.jl`."""
const GEMM_TILINGS = GemmTiling[
    # Prefer the measured 128×128 block; BK=16 also covers K not divisible by 32.
    (4, 2, 2, 4, 32, 8),    # 128 x 128 x 32, 8 warps — the reference block
    (4, 2, 2, 4, 16, 8),    # 128 x 128 x 16, 8 warps — llama.cpp's own BK
    (3, 2, 2, 4, 32, 8),    #  96 x 128, 8 warps
    (2, 2, 2, 4, 32, 8),    #  64 x 128, 8 warps
    (2, 2, 2, 2, 32, 8),    #  64 x  64, 4 warps
    (1, 2, 2, 4, 32, 8),    #  32 x 128, 8 warps
    (1, 1, 2, 4, 32, 8),    #  32 x  64, 8 warps — one tile per warp, widest reach
]

# Core owns the staged tile capability beside its implementation.
staged_gemm_tile() = GEMM_TILE

"""Historical tuning marker; results are device/workload-specific, not a proof of optimality."""
const GEMM_TILING_SWEEP_DONE = true


"""Whether the staged block exactly divides M, N and K. Its loads are unmasked."""
@inline gemm_divides(c::GemmTiling, M::Int, N::Int, K::Int) =
    M % gemm_bm(c) == 0 && N % gemm_bn(c) == 0 && K % gemm_bk(c) == 0

"""Reject non-power-of-two row blocks at K divisible by 256. The 96-row block
measured a sharp regression there; cache-set aliasing is suspected, not proven.
The rule's extension to other non-power-of-two blocks is unqualified."""
@inline gemm_aliasing(c::GemmTiling, K::Int) = !ispow2(gemm_bm(c)) && K % 256 == 0

# Shared default for launch selection and staged eligibility.
const GEMM_STAGED_DEFAULT = true

"""Default off: double buffering did not improve the measured workloads."""
const GEMM_DOUBLEBUF_DEFAULT = false

"""Default off: vec4 staging did not improve the measured workloads."""
const GEMM_VEC4_DEFAULT = false

"""Use Int32 staging indices when all addressed matrix extents fit (gemm_fits32)."""
const GEMM_NARROW_DEFAULT = true

@inline function gemm_tiling(M::Int, N::Int, K::Int; forced = nothing)
    # Forced tilings must divide the shape; no bounds masking exists in this path.
    forced === nothing || return gemm_divides(forced, M, N, K) ? forced : nothing
    for c in GEMM_TILINGS
        gemm_divides(c, M, N, K) && !gemm_aliasing(c, K) && return c
    end
    return nothing
end

"""Choose a gathering tiling, or nothing. Forced tilings may also select
CONV_GATHER_TILINGS, which exist only in the gathering kernel table."""
@inline function gather_gemm_tiling(M::Int, N::Int, K::Int, nbatch::Int, splitk::Int;
                                    staged::Bool = GEMM_STAGED_DEFAULT, tiling = nothing)
    tiling === nothing && return staged_gemm_tiling(M, N, K, nbatch, splitk; staged)
    staged && nbatch == 1 && splitk == 1 || return nothing
    gemm_divides(tiling, M, N, K) && haskey(GEMM_STAGED_GATHER_KERNELS, tiling) || return nothing
    return tiling
end

"""Which tiling this call may use, or `nothing` for the register-blocked kernel."""
@inline function staged_gemm_tiling(M::Int, N::Int, K::Int, nbatch::Int, splitk::Int;
                                    staged::Bool = GEMM_STAGED_DEFAULT, tiling = nothing)
    staged && nbatch == 1 || return nothing
    # Staged kernels write one plane; selecting them with split K leaves stale partials.
    splitk == 1 || return nothing
    c = gemm_tiling(M, N, K; forced = tiling)
    c === nothing && return nothing
    return haskey(GEMM_STAGED_KERNELS, c) ? c : nothing
end

"""Maximum extra padded width relative to tile-rounded N; raising it requires small-N measurements."""
const GEMM_PAD_SLACK = 0.25

"""    gemm_padn(M, N, K; tile=GEMM_TILE, slack=GEMM_PAD_SLACK, faster=1/16) -> Int

Round N to a usable staged block width within slack of tile-rounded N.
Prefer the first tiling within faster of the narrowest eligible width;
otherwise return tile-rounded N. Uses the selector's divisibility/aliasing rules."""
function gemm_padn(M::Int, N::Int, K::Int;
                   tile::Int = GEMM_TILE, slack::Real = GEMM_PAD_SLACK, faster::Real = 1 / 16)
    base = cld(N, tile) * tile
    # The padded width each applicable tiling needs, in the table's speed order.
    widths = Int[]
    for c in GEMM_TILINGS
        M % gemm_bm(c) == 0 && K % gemm_bk(c) == 0 && !gemm_aliasing(c, K) || continue
        haskey(GEMM_STAGED_KERNELS, c) || continue
        push!(widths, cld(N, gemm_bn(c)) * gemm_bn(c))
    end
    isempty(widths) && return base
    narrowest = minimum(widths)
    narrowest <= base * (1 + slack) || return base
    return first(w for w in widths if w <= narrowest * (1 + faster))
end

# These kernels require 32-lane subgroups; native pipeline setup must request
# that width. A wider native wave cannot be treated as two independent subgroups.
const COOPMAT_SUBGROUP = 32

# Register-blocked kernels use the same 32-lane subgroup contract.
const GEMM_SUBGROUP = COOPMAT_SUBGROUP


# Register-blocked kernels require exact output/K tile extents. With split K,
# each split writes its own plane for later reduction. Generate literal block
# sizes: conditional fragment definitions caused poor native code generation.
"""Initialize an fp32 accumulator from zero or row bias broadcast with stride 0.
Bias loads retain their element type before conversion. Stride-0 broadcast is
qualified by `test_coopmat_epilogue.jl`; it is not a portable specification guarantee."""
@inline accinit(::Nothing, ptr, off) = zero(AcceleratedMatrix{Float32,GEMM_TILE,GEMM_TILE,Accumulator})
@inline function accinit(bias, ptr, off)
    # Load bias in its element type before converting the values to fp32.
    b = AcceleratedMatrix{eltype(bias),GEMM_TILE,GEMM_TILE,Accumulator}(ptr, off, 0)
    return convert(AcceleratedMatrix{Float32,GEMM_TILE,GEMM_TILE,Accumulator}, b)
end

"""Store an fp32 accumulator into C, converting once to its element type."""
@inline accstore!(C, off, ld, c::AcceleratedMatrix{Float32,GEMM_TILE,GEMM_TILE,Accumulator}) =
    accstore!(C, off, ld, c, identity)

"""Store a tile after conversion to eltype(C), then apply f per component.
That order preserves an exported graph's rounding between matmul and activation;
identity skips the component walk."""
@inline function accstore!(C, off, ld,
                           c::AcceleratedMatrix{Float32,GEMM_TILE,GEMM_TILE,Accumulator},
                           f::F) where {F}
    m = convert(AcceleratedMatrix{eltype(C),GEMM_TILE,GEMM_TILE,Accumulator}, c)
    if F !== typeof(identity)
        n = coopmat_length(AcceleratedMatrix{eltype(C),GEMM_TILE,GEMM_TILE,Accumulator})
        for i in Int32(0):(n - Int32(1))
            m = coopmat_setcomp(m, i, f(coopmat_getcomp(m, i)))
        end
    end
    copyto!(pointer(C), off, ld, m)
end

const GEMM_BLOCK_KERNELS = Dict{Int,Any}()

for BLK in (1, 2, 4)
    kname = Symbol("coopmat_gemm_kernel_", BLK, "!")
    @eval begin
        function $kname(C, A, B, bias, epi,
                                ::Val{M}, ::Val{N}, ::Val{K},
                                ::Val{KPER}) where {M,N,K,KPER}
            lane = KI.get_global_id().x - 1
            g = lane ÷ 32
            tiles_m = M ÷ (GEMM_TILE * $BLK)
            ntiles = tiles_m * (N ÷ (GEMM_TILE * $BLK))
            # Batch is outermost so consecutive subgroups walk one matrix's tiles.
            splitk = (K ÷ GEMM_TILE) ÷ KPER
            per = ntiles * splitk
            bat = g ÷ per
            r = g % per
            t = r % ntiles
            sk = r ÷ ntiles
            tm = (t % tiles_m) * (GEMM_TILE * $BLK)
            tn = (t ÷ tiles_m) * (GEMM_TILE * $BLK)
            aoff = bat * M * K
            boff = bat * K * N
            coff = bat * M * N * splitk

            # KPER is static and divides K into equal tile counts, allowing fixed-trip loops.
            klo = sk * KPER * GEMM_TILE

            # Zero, or the bias broadcast down the tile's rows — `bias` is
            # indexed by the output ROW, so the offset follows `tm` and `i` and
            # is independent of `j`.
            bp = bias === nothing ? bias : pointer(bias)
            Base.Cartesian.@nexprs $BLK j -> Base.Cartesian.@nexprs $BLK i ->
                c_i_j = accinit(bias, bp, 1 + tm + (i - 1) * GEMM_TILE)

            # Two-way unroll overlaps independent loads; the tail handles odd KPER.
            kt = 0
            while kt + 2 <= KPER
                Base.Cartesian.@nexprs 2 u -> begin
                    k0_u = klo + (kt + (u - 1)) * GEMM_TILE
                    Base.Cartesian.@nexprs $BLK i -> a_i_u =
                        AcceleratedMatrix{Float16,GEMM_TILE,GEMM_TILE,MatrixA}(
                            pointer(A), 1 + aoff + tm + (i - 1) * GEMM_TILE + k0_u * M, M)
                    Base.Cartesian.@nexprs $BLK j -> b_j_u =
                        AcceleratedMatrix{Float16,GEMM_TILE,GEMM_TILE,MatrixB}(
                            pointer(B), 1 + boff + k0_u + (tn + (j - 1) * GEMM_TILE) * K, K)
                    Base.Cartesian.@nexprs $BLK j -> Base.Cartesian.@nexprs $BLK i ->
                        c_i_j = muladd(a_i_u, b_j_u, c_i_j)
                end
                kt += 2
            end
            while kt < KPER
                k0 = klo + kt * GEMM_TILE
                Base.Cartesian.@nexprs $BLK i -> a_i =
                    AcceleratedMatrix{Float16,GEMM_TILE,GEMM_TILE,MatrixA}(
                        pointer(A), 1 + aoff + tm + (i - 1) * GEMM_TILE + k0 * M, M)
                Base.Cartesian.@nexprs $BLK j -> b_j =
                    AcceleratedMatrix{Float16,GEMM_TILE,GEMM_TILE,MatrixB}(
                        pointer(B), 1 + boff + k0 + (tn + (j - 1) * GEMM_TILE) * K, K)
                Base.Cartesian.@nexprs $BLK j -> Base.Cartesian.@nexprs $BLK i ->
                    c_i_j = muladd(a_i, b_j, c_i_j)
                kt += 1
            end

            Base.Cartesian.@nexprs $BLK j -> Base.Cartesian.@nexprs $BLK i ->
                accstore!(C, 1 + coff + tm + (i - 1) * GEMM_TILE +
                             (tn + (j - 1) * GEMM_TILE) * M + sk * M * N,
                          M, c_i_j, epi)
            return nothing
        end
        GEMM_BLOCK_KERNELS[$BLK] = $kname
    end
end

# Staging fills one A/B tile per workgroup. Consecutive lanes read consecutive
# global addresses. Shared packing widens stores and matrix loads together;
# widening global loads alone did not improve measured streaming throughput.
const GEMM_STAGED_KERNELS = Dict{GemmTiling,Any}()


"""Vec2-staged twins of GEMM_STAGED_KERNELS. A packs along m, B along k;
shared leading dimensions count packed pairs, preserving global coalescing."""
const GEMM_STAGED_V2_KERNELS = Dict{GemmTiling,Any}()

"""Whether all staging/store indices fit Int32. Products widen before comparison
to avoid host overflow; strict < leaves room for 1-based end indices."""
@inline gemm_fits32(M::Integer, N::Integer, K::Integer) =
    widemul(Int(M), Int(K)) < typemax(Int32) && widemul(Int(K), Int(N)) < typemax(Int32) &&
    widemul(Int(M), Int(N)) < typemax(Int32)

"""Int32-index variants of vec2 staging, generated from the shared wide/narrow body.
The selector uses gemm_fits32; larger extents retain Int indexing."""
const GEMM_STAGED_V2N_KERNELS = Dict{GemmTiling,Any}()

"""Double-buffered narrow vec2 kernels. Alternating shared tiles allow one
barrier per k-block, at twice the shared storage and additional phase registers.
Selected explicitly with doublebuf; default off after native comparisons."""
const GEMM_STAGED_DB_KERNELS = Dict{GemmTiling,Any}()

"""Register-prefetched, single-shared-buffer narrow vec2 kernels. Read the next
tile before consuming the current tile, then retire, overwrite and publish it.
Plain GEMM retains MVector queues; gathering needs explicit SSA bindings.
Keep these encodings distinct unless native and consumer measurements support a change."""
const GEMM_STAGED_PREFETCH_KERNELS = Dict{GemmTiling,Any}()

"""Vec4-staged twins, generated only when all packed dimensions and per-lane
loads divide exactly by four. Selected explicitly with vec4; default off.
`test_coopmat_shared.jl` covers vector shared-memory size/alignment."""
const GEMM_STAGED_V4_KERNELS = Dict{GemmTiling,Any}()

"""Diagnostic variants: :nostage reuses only the first k-block (intentionally wrong);
:all stages every block. Never selected by normal GEMM dispatch. ABI retained for
JuliaVision/tools/gemm_ablate.jl."""
const GEMM_STAGED_ABL_KERNELS = Dict{GemmTiling,Any}()

"A 2-wide fp16 vector, i.e. `f16vec2`; see `Lava.coopmat_vecwidth`."
const GemmV2 = NTuple{2,VecElement{Float16}}

"""Four packed fp16 components for 64-bit shared accesses."""
const GemmV4 = NTuple{4,VecElement{Float16}}

# `apair` itself is declared in `src/array/indexing.jl`: DNNKernels extends it.
# The kernels that read through it are below.

"""
Gathering twins of `GEMM_STAGED_PREFETCH_KERNELS`: the same schedule, with the
A operand computed by an [`apair`](@ref) loader instead of read from a matrix.
"""
const GEMM_STAGED_GATHER_KERNELS = Dict{GemmTiling,Any}()

"""Quote staged products with one live A/B fragment. Caller supplies sA/sB, sm/sn
and c_mt_nt; optional offsets select the active half of a double buffer.
This runs at definition time; regenerate the @eval kernels after editing it."""
function stagedmma(STM, STN, NKT, LDA, LDB, width; aoffset=0, boffset=0)
    return quote
        Base.Cartesian.@nexprs $NKT u -> begin
            kt = (u - 1) * GEMM_TILE
            Base.Cartesian.@nexprs $STM mt -> begin
                a = AcceleratedMatrix{Float16,GEMM_TILE,GEMM_TILE,MatrixA}(
                    sA, 1 + $aoffset + (sm + mt - 1) * $(GEMM_TILE ÷ width) + kt * $LDA, $LDA)
                Base.Cartesian.@nexprs $STN nt -> begin
                    b = AcceleratedMatrix{Float16,GEMM_TILE,GEMM_TILE,MatrixB}(
                        sB, 1 + $boffset + (kt ÷ $width) + (sn + nt - 1) * GEMM_TILE * $LDB, $LDB)
                    c_mt_nt = muladd(a, b, c_mt_nt)
                end
            end
        end
    end
end

for (ci, cfg) in enumerate(GEMM_TILINGS)
    STM, STN, WM, WN, BK, PAD = cfg
    BM, BN = GEMM_TILE * STM * WM, GEMM_TILE * STN * WN
    WG = WM * WN * COOPMAT_SUBGROUP
    LDA, LDB = BM + PAD, BK + PAD
    NKT = BK ÷ GEMM_TILE
    (BM * BK) % WG == 0 && (BK * BN) % WG == 0 ||
        error("gemm tiling $cfg: staging does not divide the workgroup")
    # splitidx avoids native integer-division defects for non-power-of-two blocks;
    # BM need not divide WG. See test_shared_index_division.jl.
    AREPS, BREPS = (BM * BK) ÷ WG, (BK * BN) ÷ WG
    kname = Symbol("coopmat_gemm_staged_kernel_", ci, "!")
    @eval begin
        # Launches use exact whole workgroups and derive indices from group/local IDs.
        function $kname(
                                          C, A, B, bias, epi,
                                          ::Val{M}, ::Val{N}, ::Val{K}) where {M,N,K}
            sA = KI.localmemory(Float16, Val(($LDA * $BK,)), Val(1))
            sB = KI.localmemory(Float16, Val(($LDB * $BN,)), Val(2))

            tid = KI.get_local_id().x - 1
            blk = KI.get_group_id().x - 1
            nblk_m = M ÷ $BM
            tm = (blk % nblk_m) * $BM
            tn = (blk ÷ nblk_m) * $BN

            s = tid ÷ 32                # subgroup within the workgroup
            sm = (s % $WM) * $STM       # its first tile row, in tiles
            sn = (s ÷ $WM) * $STN       # its first tile column, in tiles

            bp = bias === nothing ? bias : pointer(bias)
            Base.Cartesian.@nexprs $STN nt -> Base.Cartesian.@nexprs $STM mt ->
                c_mt_nt = accinit(bias, bp, 1 + tm + (sm + mt - 1) * GEMM_TILE)

            # The K-block trip count is static; splitidx handles staging-index correctness.
            for kb in 0:(K ÷ $BK - 1)
                k0 = kb * $BK
                # One element per lane per step, and it stays that way — see the
                # note on wide staging loads above.
                #
                # A block: rows [tm, tm+BM) × cols [k0, k0+BK), column-major both sides.
                @inbounds for r in 0:($AREPS - 1)
                    idx = tid + r * $WG
                    i, kk = splitidx(idx, Val($BM))
                    sA[1 + i + kk * $LDA] = A[1 + (tm + i) + (k0 + kk) * M]
                end
                # B block: rows [k0, k0+BK) × cols [tn, tn+BN).
                @inbounds for r in 0:($BREPS - 1)
                    idx = tid + r * $WG
                    kk, j = splitidx(idx, Val($BK))
                    sB[1 + kk + j * $LDB] = B[1 + (k0 + kk) + (tn + j) * K]
                end
                KI.barrier()

                # stagedmma bounds live fragments while reusing accumulator tiles.
                $(stagedmma(STM,STN,NKT,LDA,LDB,1))
                KI.barrier()   # nothing may refill shared until every subgroup is done
            end

            Base.Cartesian.@nexprs $STN nt -> Base.Cartesian.@nexprs $STM mt ->
                accstore!(C,
                          1 + (tm + (sm + mt - 1) * GEMM_TILE) +
                              (tn + (sn + nt - 1) * GEMM_TILE) * M,
                          M, c_mt_nt, epi)
            return nothing
        end
        GEMM_STAGED_KERNELS[$cfg] = $kname
    end

    # Packed pairs halve staging counts and cooperative-load strides. The guard
    # keeps every pair, tile offset and workgroup transfer divisible by two.
    if iseven(BM) && iseven(BK) && iseven(LDA) && iseven(LDB) &&
       ((BM * BK) ÷ 2) % WG == 0 && ((BK * BN) ÷ 2) % WG == 0
        LDA2, LDB2 = LDA ÷ 2, LDB ÷ 2
        AREPS2, BREPS2 = (BM * BK) ÷ 2 ÷ WG, (BK * BN) ÷ 2 ÷ WG
        BM2, BK2 = BM ÷ 2, BK ÷ 2
        # Index width and optional ablation change code generation, not the schedule.
        for (suffix, Index, table, ablation) in (
            ("_v2!", Int, :GEMM_STAGED_V2_KERNELS, false),
            ("_v2n!", Int32, :GEMM_STAGED_V2N_KERNELS, false),
            ("_abl!", Int32, :GEMM_STAGED_ABL_KERNELS, true))
            vname = Symbol("coopmat_gemm_staged_kernel_", ci, suffix)
            extra_args = ablation ? (:(::Val{ABL}),) : ()
            parameters = ablation ? (:M,:N,:K,:ABL) : (:M,:N,:K)
            staging = quote
                @inbounds for r in $Index(0):$Index($AREPS2 - 1)
                    idx = tid + r * $Index($WG)
                    p, kk = splitidx(idx, Val($BM2))
                    g = $Index(1) + (tm + $Index(2) * $Index(p)) +
                        (k0 + $Index(kk)) * $Index(M)
                    sA[1 + Int(p) + Int(kk) * $LDA2] =
                        (VecElement(A[g]), VecElement(A[g + $Index(1)]))
                end
                @inbounds for r in $Index(0):$Index($BREPS2 - 1)
                    idx = tid + r * $Index($WG)
                    q, j = splitidx(idx, Val($BK2))
                    g = $Index(1) + (k0 + $Index(2) * $Index(q)) +
                        (tn + $Index(j)) * $Index(K)
                    sB[1 + Int(q) + Int(j) * $LDB2] =
                        (VecElement(B[g]), VecElement(B[g + $Index(1)]))
                end
            end
            if ablation
                staging = :(if ABL !== :nostage || kb == $Index(0); $staging; end)
            end
            @eval begin
                function $vname(C, A, B, bias, epi, ::Val{M}, ::Val{N}, ::Val{K},
                                $(extra_args...)) where {$(parameters...)}
                    sA = KI.localmemory(GemmV2, Val(($LDA2 * $BK,)), Val(1))
                    sB = KI.localmemory(GemmV2, Val(($LDB2 * $BN,)), Val(2))

                    tid = $Index(KI.get_local_id().x - 1)
                    blk = $Index(KI.get_group_id().x - 1)
                    nblk_m = $Index(M ÷ $BM)
                    tm = (blk % nblk_m) * $Index($BM)
                    tn = (blk ÷ nblk_m) * $Index($BN)

                    s = tid ÷ $Index(32)
                    sm = (s % $Index($WM)) * $Index($STM)
                    sn = (s ÷ $Index($WM)) * $Index($STN)

                    bp = bias === nothing ? bias : pointer(bias)
                    Base.Cartesian.@nexprs $STN nt -> Base.Cartesian.@nexprs $STM mt ->
                        c_mt_nt = accinit(bias, bp, 1 + tm + (sm + mt - 1) * GEMM_TILE)

                    for kb in $Index(0):$Index(K ÷ $BK - 1)
                        k0 = kb * $Index($BK)
                        $staging
                        KI.barrier()

                        $(stagedmma(STM,STN,NKT,LDA2,LDB2,2))
                        KI.barrier()
                    end

                    Base.Cartesian.@nexprs $STN nt -> Base.Cartesian.@nexprs $STM mt ->
                        accstore!(C,
                                  1 + Int(tm + (sm + mt - 1) * $Index(GEMM_TILE)) +
                                      (tn + (sn + nt - 1) * GEMM_TILE) * M,
                                  M, c_mt_nt, epi)
                    return nothing
                end
                $table[$cfg] = $vname
            end
        end

        # ── ncnn schedule: register prefetch + one shared tile ─────────────
        kp = Symbol("coopmat_gemm_staged_kernel_", ci, "_prefetch!")
        @eval begin
            function $kp(
                                              C, A, B, bias, epi,
                                              ::Val{M}, ::Val{N}, ::Val{K}) where {M,N,K}
                sA = KI.localmemory(GemmV2, Val(($LDA2 * $BK,)), Val(1))
                sB = KI.localmemory(GemmV2, Val(($LDB2 * $BN,)), Val(2))

                tid = Int32(KI.get_local_id().x - 1)
                blk = Int32(KI.get_group_id().x - 1)
                nblk_m = Int32(M ÷ $BM)
                tm = (blk % nblk_m) * Int32($BM)
                tn = (blk ÷ nblk_m) * Int32($BN)

                s = tid ÷ Int32(32)
                sm = (s % Int32($WM)) * Int32($STM)
                sn = (s ÷ Int32($WM)) * Int32($STN)

                bp = bias === nothing ? bias : pointer(bias)
                Base.Cartesian.@nexprs $STN nt -> Base.Cartesian.@nexprs $STM mt ->
                    c_mt_nt = accinit(bias, bp, 1 + tm + (sm + mt - 1) * GEMM_TILE)

                # Function-storage arrays are scalar-replaced by Lava's normal
                # pipeline. They are the one-tile register queue, not another
                # workgroup allocation.
                nextA = MVector{$AREPS2,GemmV2}(undef)
                nextB = MVector{$BREPS2,GemmV2}(undef)

                # Prologue: fetch block zero to registers, then publish it to
                # the sole shared-memory tile.
                @inbounds for r in Int32(0):Int32($AREPS2 - 1)
                    idx = tid + r * Int32($WG)
                    p, kk = splitidx(idx, Val($BM2))
                    g = Int32(1) + tm + Int32(2) * Int32(p) + Int32(kk) * Int32(M)
                    nextA[Int(r) + 1] = (VecElement(A[g]), VecElement(A[g + Int32(1)]))
                end
                @inbounds for r in Int32(0):Int32($BREPS2 - 1)
                    idx = tid + r * Int32($WG)
                    q, j = splitidx(idx, Val($BK2))
                    g = Int32(1) + Int32(2) * Int32(q) + (tn + Int32(j)) * Int32(K)
                    nextB[Int(r) + 1] = (VecElement(B[g]), VecElement(B[g + Int32(1)]))
                end
                @inbounds for r in Int32(0):Int32($AREPS2 - 1)
                    idx = tid + r * Int32($WG)
                    p, kk = splitidx(idx, Val($BM2))
                    sA[1 + Int(p) + Int(kk) * $LDA2] = nextA[Int(r) + 1]
                end
                @inbounds for r in Int32(0):Int32($BREPS2 - 1)
                    idx = tid + r * Int32($WG)
                    q, j = splitidx(idx, Val($BK2))
                    sB[1 + Int(q) + Int(j) * $LDB2] = nextB[Int(r) + 1]
                end
                KI.barrier()

                nkb = Int32(K ÷ $BK)
                for kb in Int32(0):(nkb - Int32(1))
                    # Issue the following tile's global reads before the MMAs.
                    # Their values remain live while the current shared tile is
                    # consumed, giving the scheduler independent memory work.
                    if kb + Int32(1) < nkb
                        k0 = (kb + Int32(1)) * Int32($BK)
                        @inbounds for r in Int32(0):Int32($AREPS2 - 1)
                            idx = tid + r * Int32($WG)
                            p, kk = splitidx(idx, Val($BM2))
                            g = Int32(1) + tm + Int32(2) * Int32(p) +
                                (k0 + Int32(kk)) * Int32(M)
                            nextA[Int(r) + 1] =
                                (VecElement(A[g]), VecElement(A[g + Int32(1)]))
                        end
                        @inbounds for r in Int32(0):Int32($BREPS2 - 1)
                            idx = tid + r * Int32($WG)
                            q, j = splitidx(idx, Val($BK2))
                            g = Int32(1) + k0 + Int32(2) * Int32(q) +
                                (tn + Int32(j)) * Int32(K)
                            nextB[Int(r) + 1] =
                                (VecElement(B[g]), VecElement(B[g + Int32(1)]))
                        end
                    end

                    $(stagedmma(STM,STN,NKT,LDA2,LDB2,2))

                    # The last product has no successor to publish.
                    if kb + Int32(1) < nkb
                        KI.barrier()
                        @inbounds for r in Int32(0):Int32($AREPS2 - 1)
                            idx = tid + r * Int32($WG)
                            p, kk = splitidx(idx, Val($BM2))
                            sA[1 + Int(p) + Int(kk) * $LDA2] = nextA[Int(r) + 1]
                        end
                        @inbounds for r in Int32(0):Int32($BREPS2 - 1)
                            idx = tid + r * Int32($WG)
                            q, j = splitidx(idx, Val($BK2))
                            sB[1 + Int(q) + Int(j) * $LDB2] = nextB[Int(r) + 1]
                        end
                        KI.barrier()
                    end
                end

                Base.Cartesian.@nexprs $STN nt -> Base.Cartesian.@nexprs $STM mt ->
                    accstore!(C,
                              1 + Int(tm + (sm + mt - 1) * Int32(GEMM_TILE)) +
                                  (tn + (sn + nt - 1) * GEMM_TILE) * M,
                              M, c_mt_nt, epi)
                return nothing
            end
            GEMM_STAGED_PREFETCH_KERNELS[$cfg] = $kp
        end

        # Alternate shared halves: fill the next tile while multiplying the current one.
        # One barrier both publishes writes and retires reads before halves swap.
        kdb = Symbol("coopmat_gemm_staged_kernel_", ci, "_db!")
        @eval begin
            function $kdb(
                                              C, A, B, bias, epi,
                                              ::Val{M}, ::Val{N}, ::Val{K}) where {M,N,K}
                sA = KI.localmemory(GemmV2, Val((2 * $LDA2 * $BK,)), Val(1))
                sB = KI.localmemory(GemmV2, Val((2 * $LDB2 * $BN,)), Val(2))

                tid = Int32(KI.get_local_id().x - 1)
                blk = Int32(KI.get_group_id().x - 1)
                nblk_m = Int32(M ÷ $BM)
                tm = (blk % nblk_m) * Int32($BM)
                tn = (blk ÷ nblk_m) * Int32($BN)

                s = tid ÷ Int32(32)
                sm = (s % Int32($WM)) * Int32($STM)
                sn = (s ÷ Int32($WM)) * Int32($STN)

                bp = bias === nothing ? bias : pointer(bias)
                Base.Cartesian.@nexprs $STN nt -> Base.Cartesian.@nexprs $STM mt ->
                    c_mt_nt = accinit(bias, bp, 1 + tm + (sm + mt - 1) * GEMM_TILE)

                nkb = Int32(K ÷ $BK)

                # Prologue: block 0 into buffer 0, so the loop always has a tile
                # to compute on and only ever stages the NEXT one.
                @inbounds for r in Int32(0):Int32($AREPS2 - 1)
                    idx = tid + r * Int32($WG)
                    p, kk = splitidx(idx, Val($BM2))
                    g = Int32(1) + (tm + Int32(2) * Int32(p)) + Int32(kk) * Int32(M)
                    sA[1 + Int(p) + Int(kk) * $LDA2] =
                        (VecElement(A[g]), VecElement(A[g + Int32(1)]))
                end
                @inbounds for r in Int32(0):Int32($BREPS2 - 1)
                    idx = tid + r * Int32($WG)
                    q, j = splitidx(idx, Val($BK2))
                    g = Int32(1) + Int32(2) * Int32(q) + (tn + Int32(j)) * Int32(K)
                    sB[1 + Int(q) + Int(j) * $LDB2] =
                        (VecElement(B[g]), VecElement(B[g + Int32(1)]))
                end
                KI.barrier()

                for kb in Int32(0):(nkb - Int32(1))
                    rda = (kb & Int32(1)) * Int32($LDA2 * $BK)
                    rdb = (kb & Int32(1)) * Int32($LDB2 * $BN)
                    wra = (Int32(1) - (kb & Int32(1))) * Int32($LDA2 * $BK)
                    wrb = (Int32(1) - (kb & Int32(1))) * Int32($LDB2 * $BN)
                    # Unconditional staging keeps all invocations at the barrier. On the last step,
                    # repeat the final tile into the unused half rather than branch around the loads.
                    k0 = min(kb + Int32(1), nkb - Int32(1)) * Int32($BK)

                    @inbounds for r in Int32(0):Int32($AREPS2 - 1)
                        idx = tid + r * Int32($WG)
                        p, kk = splitidx(idx, Val($BM2))
                        g = Int32(1) + (tm + Int32(2) * Int32(p)) +
                            (k0 + Int32(kk)) * Int32(M)
                        sA[1 + Int(wra) + Int(p) + Int(kk) * $LDA2] =
                            (VecElement(A[g]), VecElement(A[g + Int32(1)]))
                    end
                    @inbounds for r in Int32(0):Int32($BREPS2 - 1)
                        idx = tid + r * Int32($WG)
                        q, j = splitidx(idx, Val($BK2))
                        g = Int32(1) + (k0 + Int32(2) * Int32(q)) +
                            (tn + Int32(j)) * Int32(K)
                        sB[1 + Int(wrb) + Int(q) + Int(j) * $LDB2] =
                            (VecElement(B[g]), VecElement(B[g + Int32(1)]))
                    end

                    $(stagedmma(STM,STN,NKT,LDA2,LDB2,2; aoffset=:(Int(rda)),boffset=:(Int(rdb))))
                    # ONE barrier: it both publishes the tile just staged and
                    # retires the reads of the tile just consumed.
                    KI.barrier()
                end

                Base.Cartesian.@nexprs $STN nt -> Base.Cartesian.@nexprs $STM mt ->
                    accstore!(C,
                              1 + Int(tm + (sm + mt - 1) * Int32(GEMM_TILE)) +
                                  (tn + (sn + nt - 1) * GEMM_TILE) * M,
                              M, c_mt_nt, epi)
                return nothing
            end
            GEMM_STAGED_DB_KERNELS[$cfg] = $kdb
        end
    end

    # Vec4 staging requires exact packed dimensions and per-lane tile coverage.
    # Cooperative-matrix strides count vectors, not scalar components.
    if LDA % 4 == 0 && LDB % 4 == 0 && BM % 4 == 0 && BK % 4 == 0 &&
       (BM * BK) % (4 * WG) == 0 && (BK * BN) % (4 * WG) == 0
        LDA4, LDB4 = LDA ÷ 4, LDB ÷ 4
        AREPS4, BREPS4 = (BM * BK) ÷ 4 ÷ WG, (BK * BN) ÷ 4 ÷ WG
        BM4, BK4 = BM ÷ 4, BK ÷ 4
        kv4 = Symbol("coopmat_gemm_staged_kernel_", ci, "_v4!")
        @eval begin
            function $kv4(
                                              C, A, B, bias, epi,
                                              ::Val{M}, ::Val{N}, ::Val{K}) where {M,N,K}
                sA = KI.localmemory(GemmV4, Val(($LDA4 * $BK,)), Val(1))
                sB = KI.localmemory(GemmV4, Val(($LDB4 * $BN,)), Val(2))

                tid = Int32(KI.get_local_id().x - 1)
                blk = Int32(KI.get_group_id().x - 1)
                nblk_m = Int32(M ÷ $BM)
                tm = (blk % nblk_m) * Int32($BM)
                tn = (blk ÷ nblk_m) * Int32($BN)

                s = tid ÷ Int32(32)
                sm = (s % Int32($WM)) * Int32($STM)
                sn = (s ÷ Int32($WM)) * Int32($STN)

                bp = bias === nothing ? bias : pointer(bias)
                Base.Cartesian.@nexprs $STN nt -> Base.Cartesian.@nexprs $STM mt ->
                    c_mt_nt = accinit(bias, bp, 1 + tm + (sm + mt - 1) * GEMM_TILE)

                for kb in Int32(0):Int32(K ÷ $BK - 1)
                    k0 = kb * Int32($BK)
                    # A: quads along `m`, the contiguous axis, so consecutive
                    # lanes still read consecutive global addresses.
                    @inbounds for r in Int32(0):Int32($AREPS4 - 1)
                        idx = tid + r * Int32($WG)
                        p, kk = splitidx(idx, Val($BM4))
                        g = Int32(1) + (tm + Int32(4) * Int32(p)) +
                            (k0 + Int32(kk)) * Int32(M)
                        sA[1 + Int(p) + Int(kk) * $LDA4] =
                            (VecElement(A[g]), VecElement(A[g + Int32(1)]),
                             VecElement(A[g + Int32(2)]), VecElement(A[g + Int32(3)]))
                    end
                    # B: quads along `k`, contiguous here for the same reason.
                    @inbounds for r in Int32(0):Int32($BREPS4 - 1)
                        idx = tid + r * Int32($WG)
                        q, j = splitidx(idx, Val($BK4))
                        g = Int32(1) + (k0 + Int32(4) * Int32(q)) +
                            (tn + Int32(j)) * Int32(K)
                        sB[1 + Int(q) + Int(j) * $LDB4] =
                            (VecElement(B[g]), VecElement(B[g + Int32(1)]),
                             VecElement(B[g + Int32(2)]), VecElement(B[g + Int32(3)]))
                    end
                    KI.barrier()

                    $(stagedmma(STM,STN,NKT,LDA4,LDB4,4))
                    KI.barrier()
                end

                Base.Cartesian.@nexprs $STN nt -> Base.Cartesian.@nexprs $STM mt ->
                    accstore!(C,
                              1 + Int(tm + (sm + mt - 1) * Int32(GEMM_TILE)) +
                                  (tn + (sn + nt - 1) * GEMM_TILE) * M,
                              M, c_mt_nt, epi)
                return nothing
            end
            GEMM_STAGED_V4_KERNELS[$cfg] = $kv4
        end
    end


end

"""Gather-only tilings for output channels divisible by 144: 192×144 and 96×288
blocks, twelve subgroups with 3×3 tiles each. Wider column blocks avoid repeated
im2col gathering and excessive output padding. Plain GEMM does not select these."""
const CONV_GATHER_TILINGS = GemmTiling[
    (3, 3, 4, 3, 32, 8),    # 192 x 144, 12 subgroups
    (3, 3, 2, 6, 32, 8),    #  96 x 288, 12 subgroups
]

# The gathering kernels, for the product's tilings and for the convolution-only
# ones above. Numbered through both lists so the product's keep their names.
for (ci, cfg) in enumerate(vcat(GEMM_TILINGS, CONV_GATHER_TILINGS))
    STM, STN, WM, WN, BK, PAD = cfg
    BM, BN = GEMM_TILE * STM * WM, GEMM_TILE * STN * WN
    WG = WM * WN * COOPMAT_SUBGROUP
    LDA, LDB = BM + PAD, BK + PAD
    NKT = BK ÷ GEMM_TILE
    vec2 = iseven(BM) && iseven(BK) && iseven(LDA) && iseven(LDB) &&
           ((BM * BK) ÷ 2) % WG == 0 && ((BK * BN) ÷ 2) % WG == 0
    if !vec2
        ci > length(GEMM_TILINGS) &&
            error("CONV_GATHER_TILINGS entry $cfg does not stage in vec2 pairs")
        continue
    end
    LDA2, LDB2 = LDA ÷ 2, LDB ÷ 2
    AREPS2, BREPS2 = (BM * BK) ÷ 2 ÷ WG, (BK * BN) ÷ 2 ÷ WG
    BM2, BK2 = BM ÷ 2, BK ÷ 2

    # Gather uses an explicitly unrolled register queue. Plain prefetch keeps
    # MVector storage: sharing that queue encoding regressed its measured path.
    # Their cooperative-matrix products share stagedmma without changing queues.
    kg = Symbol("coopmat_gemm_staged_kernel_", ci, "_gather!")
    @eval begin
        function $kg(
                                          C, A, B, bias, epi, ald,
                                          ::Val{M}, ::Val{N}, ::Val{K}) where {M,N,K}
            sA = KI.localmemory(GemmV2, Val(($LDA2 * $BK,)), Val(1))
            sB = KI.localmemory(GemmV2, Val(($LDB2 * $BN,)), Val(2))

            tid = Int32(KI.get_local_id().x - 1)
            blk = Int32(KI.get_group_id().x - 1)
            nblk_m = Int32(M ÷ $BM)
            tm = (blk % nblk_m) * Int32($BM)
            tn = (blk ÷ nblk_m) * Int32($BN)

            s = tid ÷ Int32(32)
            sm = (s % Int32($WM)) * Int32($STM)
            sn = (s ÷ Int32($WM)) * Int32($STN)

            bp = bias === nothing ? bias : pointer(bias)
            Base.Cartesian.@nexprs $STN nt -> Base.Cartesian.@nexprs $STM mt ->
                c_mt_nt = accinit(bias, bp, 1 + tm + (sm + mt - 1) * GEMM_TILE)

            # Prologue: fetch block zero to registers, then publish it to
            # the sole shared-memory tile.
            @inbounds Base.Cartesian.@nexprs $AREPS2 r -> begin
                pkk_r = splitidx(tid + Int32(r - 1) * Int32($WG), Val($BM2))
                nextA_r = apair(ald, A, tm + Int32(2) * Int32(pkk_r[1]),
                                Int32(pkk_r[2]), Int32(M))
            end
            @inbounds Base.Cartesian.@nexprs $BREPS2 r -> begin
                qj_r = splitidx(tid + Int32(r - 1) * Int32($WG), Val($BK2))
                g_r = Int32(1) + Int32(2) * Int32(qj_r[1]) +
                      (tn + Int32(qj_r[2])) * Int32(K)
                nextB_r = (VecElement(B[g_r]), VecElement(B[g_r + Int32(1)]))
            end
            @inbounds Base.Cartesian.@nexprs $AREPS2 r -> begin
                sA[1 + Int(pkk_r[1]) + Int(pkk_r[2]) * $LDA2] = nextA_r
            end
            @inbounds Base.Cartesian.@nexprs $BREPS2 r -> begin
                sB[1 + Int(qj_r[1]) + Int(qj_r[2]) * $LDB2] = nextB_r
            end
            KI.barrier()

            nkb = Int32(K ÷ $BK)
            for kb in Int32(0):(nkb - Int32(1))
                # Issue the following tile's global reads before the MMAs.
                if kb + Int32(1) < nkb
                    k0 = (kb + Int32(1)) * Int32($BK)
                    @inbounds Base.Cartesian.@nexprs $AREPS2 r -> begin
                        nextA_r = apair(ald, A, tm + Int32(2) * Int32(pkk_r[1]),
                                        k0 + Int32(pkk_r[2]), Int32(M))
                    end
                    @inbounds Base.Cartesian.@nexprs $BREPS2 r -> begin
                        gk_r = Int32(1) + k0 + Int32(2) * Int32(qj_r[1]) +
                               (tn + Int32(qj_r[2])) * Int32(K)
                        nextB_r = (VecElement(B[gk_r]), VecElement(B[gk_r + Int32(1)]))
                    end
                end

                $(stagedmma(STM,STN,NKT,LDA2,LDB2,2))

                # The last product has no successor to publish.
                if kb + Int32(1) < nkb
                    KI.barrier()
                    @inbounds Base.Cartesian.@nexprs $AREPS2 r -> begin
                        sA[1 + Int(pkk_r[1]) + Int(pkk_r[2]) * $LDA2] = nextA_r
                    end
                    @inbounds Base.Cartesian.@nexprs $BREPS2 r -> begin
                        sB[1 + Int(qj_r[1]) + Int(qj_r[2]) * $LDB2] = nextB_r
                    end
                    KI.barrier()
                end
            end

            Base.Cartesian.@nexprs $STN nt -> Base.Cartesian.@nexprs $STM mt ->
                accstore!(C,
                          1 + Int(tm + (sm + mt - 1) * Int32(GEMM_TILE)) +
                              (tn + (sn + nt - 1) * GEMM_TILE) * M,
                          M, c_mt_nt, epi)
            return nothing
        end
        GEMM_STAGED_GATHER_KERNELS[$cfg] = $kg
    end
end

"""    gemmaccum(T) -> Type

Float16 destinations accumulate in Float32 and round once on store.
Other destination types retain their precision. See `test_gemm_fp16_accum.jl`."""
@inline gemmaccum(::Type{Float16}) = Float32
@inline gemmaccum(::Type{T}) where {T} = T

# The strided kernel covers small/ragged products and Float64/complex output.
# Operands are dense bases plus strides, so transposes need no copy or device-side
# wrapper construction. K is supplied by the host.
function strided_gemm_kernel!(C, A, B, ::Val{K},
                                      co, cr, cc, ao, ar, ac, bo, br, bc,
                                      α, β, M, ntot) where {K}
    # Flat launch: an N-D `ndrange` is partitioned into N-D workgroups, so
    # consecutive lanes stop walking consecutive memory. Worth 2.3 ms of a
    # 31.7 ms inference step when applied to the convolution's im2col and
    # epilogue.
    lin = KI.get_global_id().x
    if lin <= ntot
    i = (Int32(lin) - Int32(1)) % Int32(M) + Int32(1)
    j = (Int32(lin) - Int32(1)) ÷ Int32(M) + Int32(1)
    @inbounds begin
        # Float16 destinations reduce in Float32; round only once when storing C.
        T = gemmaccum(eltype(C))
        acc = zero(T)
        ai = ao + (i - 1) * ar
        bj = bo + (j - 1) * bc
        # Explicit unrolling avoids dependence on backend loop-unroll heuristics.
        k = 0
        while k + 4 <= K
            Base.Cartesian.@nexprs 4 d -> begin
                kd = k + (d - 1)
                acc = muladd(T(A[ai + kd * ac]), T(B[bj + kd * br]), acc)
            end
            k += 4
        end
        while k < K
            acc = muladd(T(A[ai + k * ac]), T(B[bj + k * br]), acc)
            k += 1
        end
        ci = co + (i - 1) * cr + (j - 1) * cc
        # `α`/`β` arrive in the destination's type; widen them so the epilogue
        # stays in the accumulator's precision, and convert once at the store.
        C[ci] = eltype(C)(iszero(β) ? acc * T(α) :
                          muladd(acc, T(α), T(C[ci]) * T(β)))
    end
    end
    return nothing
end

# Scalar staged GEMM uses Float32 shared tiles with explicit strides and scalar
# register accumulators. WARP decomposes indices; no subgroup operation requires
# it to match the native subgroup width.
"""Scalar GEMM block dimensions, also used by DNNKernels.planewise_worth."""
const SGEMM_BM, SGEMM_BN, SGEMM_BK = 64, 64, 32

"""Minimum scalar GEMM tile count, also used by DNNKernels.planewise_worth."""
const SGEMM_MINTILES = 16

"""Maximum ratio of padded to useful scalar GEMM output area; excludes gross waste."""
const SGEMM_MAXWASTE = 4

const SGEMM_WM, SGEMM_WN = 32, 32
# The balanced 2/2/2 register block reuses four A and four B loads for 16 FMAs.
const SGEMM_WMITER, SGEMM_TM, SGEMM_TN = 2, 2, 2
const SGEMM_WARP = 64
const SGEMM_BKSTEP = 4
# Scalar shared tiles pad by two elements; cooperative-matrix padding differs.
const SGEMM_SHF = SGEMM_BK + 2
const SGEMM_NUMWARPS = (SGEMM_BM ÷ SGEMM_WM) * (SGEMM_BN ÷ SGEMM_WN)
const SGEMM_WG = SGEMM_NUMWARPS * SGEMM_WARP
const SGEMM_WNITER = (SGEMM_WM * SGEMM_WN) ÷
                     (SGEMM_WARP * SGEMM_TM * SGEMM_TN * SGEMM_WMITER)
const SGEMM_WSUBM = SGEMM_WM ÷ SGEMM_WMITER
const SGEMM_WSUBN = SGEMM_WN ÷ SGEMM_WNITER
const SGEMM_AREPS = (SGEMM_BM * SGEMM_BK) ÷ SGEMM_WG
const SGEMM_BREPS = (SGEMM_BK * SGEMM_BN) ÷ SGEMM_WG
const SGEMM_NKSTEP = SGEMM_BK ÷ SGEMM_BKSTEP

@eval begin
    # M/N/K and FAST specialize address/bounds arithmetic. FAST requires exact tiles
    # and unit row strides; the general path zero-fills partial shared tiles and guards
    # stores. Precompile representative shapes to avoid compilation on first use.
    function scalar_gemm_staged_kernel!(
            C, A, B, ::Val{M}, ::Val{N}, ::Val{K}, ::Val{FAST},
            co, cr, cc_, ao, ar, ac, bo, br, bc, α, β, nblk_m) where {M,N,K,FAST}
        bufa = KI.localmemory(Float32, Val(($SGEMM_BM * $SGEMM_SHF,)), Val(1))
        bufb = KI.localmemory(Float32, Val(($SGEMM_BN * $SGEMM_SHF,)), Val(2))

        tid = KI.get_local_id().x - 1
        blk = KI.get_group_id().x - 1
        ir = (blk % nblk_m) * $SGEMM_BM        # this workgroup's first row of C
        ic = (blk ÷ nblk_m) * $SGEMM_BN        # ...and its first column

        tiw    = tid % $SGEMM_WARP
        warp_i = tid ÷ $SGEMM_WARP
        tiwr   = tiw % ($SGEMM_WSUBM ÷ $SGEMM_TM)
        tiwc   = tiw ÷ ($SGEMM_WSUBM ÷ $SGEMM_TM)
        warp_r = warp_i % ($SGEMM_BM ÷ $SGEMM_WM)
        warp_c = warp_i ÷ ($SGEMM_BM ÷ $SGEMM_WM)

        Base.Cartesian.@nexprs $SGEMM_WNITER wsic -> Base.Cartesian.@nexprs $SGEMM_TN cn ->
            Base.Cartesian.@nexprs $SGEMM_WMITER wsir -> Base.Cartesian.@nexprs $SGEMM_TM jj ->
                s_wsic_cn_wsir_jj = zero(eltype(C))

        for kb in 0:(cld(K, $SGEMM_BK) - 1)
            k0 = kb * $SGEMM_BK
            # A block: rows [ir, ir+BM) x cols [k0, k0+BK); `m` runs fastest, so
            # consecutive lanes read consecutive addresses when `ar == 1`.
            @inbounds for r in 0:($SGEMM_AREPS - 1)
                idx = tid + r * $SGEMM_WG
                m, kk = splitidx(idx, Val($SGEMM_BM))
                i = ir + m; k = k0 + kk
                bufa[1 + m * $SGEMM_SHF + kk] = if FAST
                    Float32(A[ao + i + k * ac])
                else
                    (i < M) & (k < K) ? Float32(A[ao + i * ar + k * ac]) : 0.0f0
                end
            end
            # B block: rows [k0, k0+BK) x cols [ic, ic+BN); `k` runs fastest.
            @inbounds for r in 0:($SGEMM_BREPS - 1)
                idx = tid + r * $SGEMM_WG
                kk, n = splitidx(idx, Val($SGEMM_BK))
                k = k0 + kk; j = ic + n
                bufb[1 + n * $SGEMM_SHF + kk] = if FAST
                    Float32(B[bo + k + j * bc])
                else
                    (k < K) & (j < N) ? Float32(B[bo + k * br + j * bc]) : 0.0f0
                end
            end
            KI.barrier()

            @inbounds Base.Cartesian.@nexprs $SGEMM_NKSTEP ii -> begin
                kof = (ii - 1) * $SGEMM_BKSTEP
                # cache_a[WMITER*TM], four k-values each: the reference's V4.
                Base.Cartesian.@nexprs $SGEMM_WMITER wsir ->
                    Base.Cartesian.@nexprs $SGEMM_TM jj -> begin
                        ab_wsir_jj = 1 + (warp_r * $SGEMM_WM + (wsir - 1) * $SGEMM_WSUBM +
                                          tiwr * $SGEMM_TM + (jj - 1)) * $SGEMM_SHF + kof
                        a_wsir_jj_0 = bufa[ab_wsir_jj]
                        a_wsir_jj_1 = bufa[ab_wsir_jj + 1]
                        a_wsir_jj_2 = bufa[ab_wsir_jj + 2]
                        a_wsir_jj_3 = bufa[ab_wsir_jj + 3]
                    end
                # Reload one B vector at a time to bound live registers.
                Base.Cartesian.@nexprs $SGEMM_WNITER wsic ->
                    Base.Cartesian.@nexprs $SGEMM_TN cn -> begin
                        bb = 1 + (warp_c * $SGEMM_WN + (wsic - 1) * $SGEMM_WSUBN +
                                  tiwc * $SGEMM_TN + (cn - 1)) * $SGEMM_SHF + kof
                        b0 = bufb[bb]; b1 = bufb[bb + 1]
                        b2 = bufb[bb + 2]; b3 = bufb[bb + 3]
                        Base.Cartesian.@nexprs $SGEMM_WMITER wsir ->
                            Base.Cartesian.@nexprs $SGEMM_TM jj ->
                                # `dot_product()`: the reference's 4-deep fma chain.
                                s_wsic_cn_wsir_jj =
                                    muladd(a_wsir_jj_0, b0,
                                    muladd(a_wsir_jj_1, b1,
                                    muladd(a_wsir_jj_2, b2,
                                    muladd(a_wsir_jj_3, b3, s_wsic_cn_wsir_jj))))
                    end
            end
            KI.barrier()   # nothing may refill shared until every warp is done
        end

        @inbounds Base.Cartesian.@nexprs $SGEMM_WNITER wsic ->
            Base.Cartesian.@nexprs $SGEMM_WMITER wsir -> begin
                drw_wsic_wsir = ir + warp_r * $SGEMM_WM +
                                (wsir - 1) * $SGEMM_WSUBM + tiwr * $SGEMM_TM
                dcw_wsic_wsir = ic + warp_c * $SGEMM_WN +
                                (wsic - 1) * $SGEMM_WSUBN + tiwc * $SGEMM_TN
                Base.Cartesian.@nexprs $SGEMM_TN cn ->
                    Base.Cartesian.@nexprs $SGEMM_TM jj -> begin
                        i = drw_wsic_wsir + (jj - 1)
                        j = dcw_wsic_wsir + (cn - 1)
                        if FAST | ((i < M) & (j < N))
                            ci = FAST ? co + i + j * cc_ : co + i * cr + j * cc_
                            C[ci] = iszero(β) ? s_wsic_cn_wsir_jj * α :
                                    muladd(s_wsic_cn_wsir_jj, α, C[ci] * β)
                        end
                    end
            end
        return nothing
    end
end

"""    gemmstrides(A) -> (base, offset, rowstride, colstride) | nothing

Describe a strided window: `A[i,j] == base[offset+(i-1)*rowstride+(j-1)*colstride]`.
Offset is 1-based; strides count elements. Backends describe dense arrays, core
unwraps strided views, and unrepresentable layouts return nothing for materialization."""
function gemmstrides(A::Union{PermutedDimsArray{<:Any,2,(2,1)},
                              LinearAlgebra.Transpose,LinearAlgebra.Adjoint{<:Real}})
    p = gemmstrides(parent(A))
    p === nothing ? nothing : (p[1], p[2], p[4], p[3])
end
gemmstrides(A::PermutedDimsArray{T,2,(1,2)}) where {T} = gemmstrides(parent(A))
# Complex adjoints also conjugate, which cannot be expressed by strides alone.


gemmstrides(::AbstractArray) = nothing

# Declared resources are contiguous; placement resolves storage later.
# Their dense strides follow size without requiring a native array yet.
gemmstrides(x::Union{Buffer,TransientBuffer,ResourceView,BufferRange}) =
    (x, 1, 1, size(x, 1))

# Materialize layouts that gemmstrides cannot express. Use broadcast: generic
# copyto! on wrappers may scalar-index device arrays from the host.
function densify(a::AbstractArray)
    d = KernelAbstractions.allocate(KernelAbstractions.get_backend(a), eltype(a), size(a))
    d .= a
    return d
end


"""    coopmat_gemm_shape(M, N, K; cores=48, nbatch=1) -> (BLK, SPLITK)

Choose the widest dividing register block and enough equal K splits to target
four subgroups per core across all batches. Narrow the block only if needed;
SPLITK divides the number of K tiles and is bounded by GEMM_MAXSPLIT."""
function coopmat_gemm_shape(M::Int, N::Int, K::Int; cores::Int = 48, nbatch::Int = 1)
    # Batch parallelism counts toward the target; unnecessary splits add scratch traffic.
    target = max(1, cld(4cores, nbatch))
    nk = K ÷ GEMM_TILE
    # Only splits that divide the k-tile count, so every split has the same
    # static trip count (see the kernel).
    divisors = [s for s in 1:min(GEMM_MAXSPLIT, nk) if nk % s == 0]
    fallback = (1, 1)
    first = true
    for blk in (GEMM_MAXBLOCK, 2, 1)
        span = GEMM_TILE * blk
        (M % span == 0 && N % span == 0) || continue
        ntiles = (M ÷ span) * (N ÷ span)
        want = cld(target, ntiles)
        splitk = something(findfirst(>=(want), divisors), length(divisors))
        splitk = divisors[splitk]
        first && (fallback = (blk, splitk); first = false)
        ntiles * splitk >= target && return (blk, splitk)
    end
    fallback
end

const GEMM_MAXSPLIT = 8    # past this the partial-sum traffic outweighs the parallelism

"""Sum S partial planes of n elements per batch; input layout is [batch][split][plane]."""
function splitk_reduce_kernel!(C, Cp, ::Val{S}, n) where {S}
    i = KI.get_global_id().x
    i <= length(C) || return nothing   # the launch is whole workgroups
    @inbounds begin
        b = (i - 1) ÷ n                 # 0 for an unbatched call
        j = (i - 1) % n + 1 + b * S * n
        acc = Cp[j]
        for s in 1:(S - 1)
            acc += Cp[j + s * n]
        end
        C[i] = acc
    end
    return nothing
end

"""    splitscratch(C, M, N, splitk)

Allocate fp32 partial planes on C's device. Backends may retain and grow scratch
until native users retire; Vulkan uses its device cache. Callers with explicit
workspace pass partials instead."""
splitscratch(C, M::Int, N::Int, splitk::Int) =
    splitscratch(KernelAbstractions.get_backend(C), C, M, N, splitk)

# Backends whose allocator tracks submitted array arguments can hand the launch
# an ordinary temporary.  The launch tuple owns it through submission; the
# backend owns retirement after that. Vulkan retains scratch in its device cache
# until queued users retire.
function splitscratch(backend, C, M::Int, N::Int, splitk::Int)
    return KernelAbstractions.allocate(backend, Float32, M, N, splitk)
end


# Cooperative-matrix extents must already be tile-aligned; callers own padding.
"""    coopmat_gemm_launches(C, A, B, M, N, K; ...) -> Vector{ArrayLaunch}

Select GEMM without submitting. Return one launch, or partial products followed
by reduction for split K. The caller owns explicit partials; otherwise splitscratch
provides storage. Bias and nonlinear epilogues require splitk=1."""
function coopmat_gemm_launches(C, A, B, M::Int, N::Int, K::Int;
                       nbatch::Int = 1,
                       blk_split = coopmat_gemm_shape(M, N, K; nbatch),
                       partials = nothing, reduce::Bool = true, bias = nothing,
                       epilogue = identity,
                       # Per-call selection overrides retain process-independent policy.
                       staged::Bool = GEMM_STAGED_DEFAULT,
                       vec2::Bool   = true,
                       narrow_ok::Bool = GEMM_NARROW_DEFAULT,
                       prefetch::Bool = true,
                       doublebuf::Bool = GEMM_DOUBLEBUF_DEFAULT,
                       vec4::Bool = GEMM_VEC4_DEFAULT,
                       tensile::Bool = GEMM_TENSILE_DEFAULT,
                       tiling = nothing,
                       # Only gathering kernels interpret aload; other kernels require a plain A matrix.
                       aload = nothing)
    # A bias goes into the accumulator's initial value, so it must land exactly
    # once — with `splitk > 1` every plane would carry its own copy and the
    # reduction would sum them. The caller keeps its own epilogue there.
    bias === nothing || blk_split[2] == 1 ||
        throw(ArgumentError("coopmat_gemm!: bias needs splitk == 1, got $(blk_split[2])"))
    blk, splitk = blk_split
    # rocBLAS's kernel takes every product it accepts; see `gemm_tensile.jl`. A
    # forced tiling, a loader or `staged = false` asks for another kernel by name.
    tensile && staged && tiling === nothing && aload === nothing &&
        tensile_gemm_ok(C, A, B, M, N, K, nbatch, splitk) &&
        return [tensile_gemm_launch(C, A, B, bias, epilogue, M, N, K)]
    # The staged kernel masks nothing and does not split K, so it takes only the
    # shapes it divides exactly and only where the plan wants a single plane —
    # which includes all four of SAM 2's dominant `addmm` shapes, i.e. 72.7% of
    # the encoder's arithmetic.
    c = aload === nothing ? staged_gemm_tiling(M, N, K, nbatch, splitk; staged, tiling) :
                            gather_gemm_tiling(M, N, K, nbatch, splitk; staged, tiling)
    # A loader requires a supported staged tiling; falling through would misread A.
    aload === nothing || c !== nothing ||
        throw(ArgumentError("coopmat_gemm!: a loader needs the staged kernel, and \
                             M=$M N=$N K=$K splitk=$splitk has no staged tiling"))
    if c !== nothing
        wg = gemm_wg(c)
        # Narrow staging is legal only while all matrix indices fit Int32.
        narrow = narrow_ok && gemm_fits32(M, N, K)
        if aload !== nothing
            # `B` and `C` are the only matrices a gathering call has; `A` is the
            # loader's own addressing of whatever it reads. `M*K` is the size of
            # an im2col matrix that is never formed, and at `288 -> 288` over
            # 1024x1024 it is 2.7e9, which the plain check refused.
            (narrow_ok && widemul(K, N) < typemax(Int32) && widemul(M, N) < typemax(Int32)) ||
                throw(ArgumentError("coopmat_gemm!: a loader needs 32-bit indexing \
                                     of B and C, got M=$M N=$N K=$K"))
            haskey(GEMM_STAGED_GATHER_KERNELS, c) ||
                throw(ArgumentError("coopmat_gemm!: no gathering kernel for tiling $c"))
            wgg = gemm_wg(c)
            return [ArrayLaunch(GEMM_STAGED_GATHER_KERNELS[c],
                               (C, A, B, bias, epilogue, aload, Val(M), Val(N), Val(K)),
                               (M ÷ gemm_bm(c)) * (N ÷ gemm_bn(c)) * wgg, wgg)]
        end
        kern = if !vec2
            GEMM_STAGED_KERNELS[c]
        elseif vec4 && haskey(GEMM_STAGED_V4_KERNELS, c)
            GEMM_STAGED_V4_KERNELS[c]
        elseif narrow && doublebuf
            # The double-buffered kernel is a narrow-index one, so it needs both
            # gates; falling back through the narrow kernel keeps a shape that
            # only fails `doublebuf` on the next-best path rather than the worst.
            get(GEMM_STAGED_DB_KERNELS, c,
                get(GEMM_STAGED_V2N_KERNELS, c,
                    get(GEMM_STAGED_V2_KERNELS, c, GEMM_STAGED_KERNELS[c])))
        elseif narrow && prefetch && haskey(GEMM_STAGED_PREFETCH_KERNELS, c)
            GEMM_STAGED_PREFETCH_KERNELS[c]
        elseif narrow
            get(GEMM_STAGED_V2N_KERNELS, c,
                get(GEMM_STAGED_V2_KERNELS, c, GEMM_STAGED_KERNELS[c]))
        else
            get(GEMM_STAGED_V2_KERNELS, c, GEMM_STAGED_KERNELS[c])
        end
        return [ArrayLaunch(kern, (C, A, B, bias, epilogue, Val(M), Val(N), Val(K)),
                           (M ÷ gemm_bm(c)) * (N ÷ gemm_bn(c)) * wg, wg)]
    end
    span = GEMM_TILE * blk
    ntiles = (M ÷ span) * (N ÷ span)
    dst = splitk == 1 ? C :
          (partials === nothing ? splitscratch(C, M, N, splitk * nbatch) : partials)
    # A split-K plane is a PARTIAL sum, so an activation on it would be applied
    # to a fraction of the dot product and then summed — wrong, and silently so.
    # The epilogue belongs to whoever reduces the planes.
    epilogue === identity || splitk == 1 ||
        throw(ArgumentError("coopmat_gemm!: an epilogue needs splitk == 1, got $splitk"))
    block = ArrayLaunch(GEMM_BLOCK_KERNELS[blk],
                       (dst, A, B, bias, epilogue, Val(M), Val(N), Val(K),
                        Val((K ÷ GEMM_TILE) ÷ splitk)),
                       ntiles * splitk * 32 * nbatch, GEMM_WORKGROUP)
    splitk == 1 && return [block]
    # `reduce=false` hands the partial planes back untouched, for a caller whose
    # own epilogue can sum them — a convolution already reads every element of
    # the result to add bias and scatter it, so folding the sum in there saves a
    # dispatch and a full write-plus-read of `M*N*splitk` floats per convolution.
    reduce || return [block]
    return [block, ArrayLaunch(splitk_reduce_kernel!,
                              (C, dst, Val(splitk), M * N),
                              M * N * nbatch, 0)]
end

"""    coopmat_gemm!(C, A, B, M, N, K; ...) -> C

Submit the selected launches on C's backend/device."""
function coopmat_gemm!(C, A, B, M::Int, N::Int, K::Int; kw...)
    runlaunches!(KernelAbstractions.get_backend(C),
                 coopmat_gemm_launches(C, A, B, M, N, K; kw...))
    return C
end

"""    coopmat_gemm_dispatch!(g, C, A, B, M, N, K; name, ...) -> C

Declare one graph pass per selected launch. Pass graph-owned transient partials
when splitting K so placement can account for the workspace."""
function coopmat_gemm_dispatch!(g, C, A, B, M::Int, N::Int, K::Int;
                                name::AbstractString = "gemm", kw...)
    dispatchlaunches!(g, coopmat_gemm_launches(C, A, B, M, N, K; kw...); name)
    return C
end


# Tile count bounds insufficient parallelism; padding ratio bounds wasted work.
# These are measured heuristics, not universal device thresholds. Boundary cases
# are covered by test_gemm_staged_scalar.jl.
"""Whether Float32 output with Float16/Float32 inputs has enough scalar GEMM
tiles and acceptable padding waste. Float64/complex retain the strided kernel;
its Float32 shared buffers cannot preserve their precision."""
@inline function staged_gemm_ok(C, A, B, M, N)
    eltype(C) === Float32 || return false
    (eltype(A) === Float32 || eltype(A) === Float16) || return false
    (eltype(B) === Float32 || eltype(B) === Float16) || return false
    tm, tn = cld(M, SGEMM_BM), cld(N, SGEMM_BN)
    tm * tn >= SGEMM_MINTILES || return false
    return (tm * SGEMM_BM) * (tn * SGEMM_BN) <= SGEMM_MAXWASTE * M * N
end

# Split K widens small GEMV row grids. Consecutive lanes still walk rows;
# partials and their reduction use Float32, converting once on final store.
const GEMV_NOSPLIT_ROWS = 16384

# Target launch width for split GEMV; more splits increase reduction overhead.
const GEMV_TARGET_THREADS = 1 << 16

# Shortest K-chunk worth giving a thread. Below this the per-thread prologue and
# the extra plane cost more than the parallelism buys.
const GEMV_MIN_CHUNK = 64

"""    gemv_split(M, K) -> S

Split K to widen a small row grid, bounded by GEMV_MIN_CHUNK. Return 1 when
M already fills the grid (GEMV_NOSPLIT_ROWS)."""
@inline function gemv_split(M::Int, K::Int)
    M >= GEMV_NOSPLIT_ROWS && return 1
    S = 1
    while S * M < GEMV_TARGET_THREADS && cld(K, 2S) >= GEMV_MIN_CHUNK
        S *= 2
    end
    S
end

"""Use split-K GEMV only for Float16/Float32 operands and destination; its partials
are Float32. Other types retain the strided kernel's precision and complex arithmetic."""
gemv_split(C, A, B, M::Int, K::Int) =
    all(T -> T === Float32 || T === Float16, (eltype(C), eltype(A), eltype(B))) ?
        gemv_split(M, K) : 1

function gemv_splitk_kernel!(P, A, B,
                                               ao, ar, ac, bo, br,
                                               M::Int32, K::Int32, KC::Int32, ntot::Int32)
    lin = KI.get_global_id().x
    if lin <= ntot
        l = Int32(lin) - Int32(1)
        i = l % M + Int32(1)        # row; consecutive lanes -> consecutive rows
        s = l ÷ M                   # 0-based split
        @inbounds begin
            k1 = min(s * KC + KC, K)
            ai = ao + (i - Int32(1)) * ar
            # FOUR INDEPENDENT ACCUMULATORS, not one. `acc = muladd(_, _, acc)`
            # serialises on the FMA latency and a GEMV has no other arithmetic to
            # hide it behind; the unroll in `strided_gemm_kernel!` shortens the
            # loop but keeps the chain. Summed pairwise at the end.
            a0 = 0f0; a1 = 0f0; a2 = 0f0; a3 = 0f0
            k = s * KC
            while k + Int32(4) <= k1
                a0 = muladd(Float32(A[ai + k * ac]), Float32(B[bo + k * br]), a0)
                k1_ = k + Int32(1)
                a1 = muladd(Float32(A[ai + k1_ * ac]), Float32(B[bo + k1_ * br]), a1)
                k2_ = k + Int32(2)
                a2 = muladd(Float32(A[ai + k2_ * ac]), Float32(B[bo + k2_ * br]), a2)
                k3_ = k + Int32(3)
                a3 = muladd(Float32(A[ai + k3_ * ac]), Float32(B[bo + k3_ * br]), a3)
                k += Int32(4)
            end
            while k < k1
                a0 = muladd(Float32(A[ai + k * ac]), Float32(B[bo + k * br]), a0)
                k += Int32(1)
            end
            P[lin] = (a0 + a1) + (a2 + a3)
        end
    end
    return nothing
end

function gemv_reduce_kernel!(C, P, co, cr,
                                               M::Int32, S::Int32, α, β)
    i = KI.get_global_id().x
    if i <= M
        @inbounds begin
            acc = P[i]
            for s in Int32(1):(S - Int32(1))
                acc += P[i + s * M]
            end
            ci = co + (Int32(i) - Int32(1)) * cr
            C[ci] = eltype(C)(iszero(β) ? acc * Float32(α) :
                              muladd(acc, Float32(α), Float32(C[ci]) * Float32(β)))
        end
    end
    return nothing
end

"""    scalar_gemm_launches(C, A, B, M, N, K, α, β; partials=nothing) -> Vector{ArrayLaunch}

Select strided, staged or split-K GEMV launches without submitting. Operands must
have representable gemmstrides; callers materialize other layouts explicitly.
The caller owns explicit split-K partials; otherwise splitscratch provides them."""
function scalar_gemm_launches(C, A, B, M, N, K, α, β; partials = nothing)
    a = gemmstrides(A)
    b = gemmstrides(B)
    c = gemmstrides(C)
    (a === nothing || b === nothing) && throw(ArgumentError(
        "scalar_gemm_launches: an operand is not dense, and making it dense is " *
        "an allocation a declaration cannot make. Materialise it into a " *
        "transient first — `densify` is what the immediate path calls here."))
    # Before `staged_gemm_ok`, which declines `N == 1` anyway — this is the
    # decode path and it wants a kernel of its own.
    if N == 1
        S = gemv_split(C, A, B, M, K)
        if S > 1
            KC = cld(K, S)
            P = partials === nothing ? splitscratch(C, M, 1, S) : partials
            return [ArrayLaunch(gemv_splitk_kernel!,
                        (P, a[1], b[1], a[2], a[3], a[4], b[2], b[3],
                         Int32(M), Int32(K), Int32(KC), Int32(M * S)), M * S, 256),
                    ArrayLaunch(gemv_reduce_kernel!,
                        (c[1], P, c[2], c[3], Int32(M), Int32(S),
                         eltype(C)(α), eltype(C)(β)), M, 256)]
        end
    end
    if staged_gemm_ok(C, A, B, M, N)
        nblk_m = cld(M, SGEMM_BM)
        nblk_n = cld(N, SGEMM_BN)
        # The reference's `is_aligned && is_in_bounds`, decided once on the host:
        # exact tiling in all three extents and unit row stride on every operand.
        fast = M % SGEMM_BM == 0 && N % SGEMM_BN == 0 && K % SGEMM_BK == 0 &&
               a[3] == 1 && b[3] == 1 && c[3] == 1
        return [ArrayLaunch(scalar_gemm_staged_kernel!,
                    (c[1], a[1], b[1], Val(M), Val(N), Val(K), Val(fast),
                     c[2], c[3], c[4], a[2], a[3], a[4], b[2], b[3], b[4],
                     eltype(C)(α), eltype(C)(β), nblk_m),
                    nblk_m * nblk_n * SGEMM_WG, SGEMM_WG)]
    end
    # Use 256 threads; the native maximum underfills small output grids.
    return [ArrayLaunch(strided_gemm_kernel!,
                (c[1], a[1], b[1], Val(K), c[2], c[3], c[4],
                 a[2], a[3], a[4], b[2], b[3], b[4], α, β, M, M * N),
                M * N, 256)]
end

"""Submit scalar GEMM on C's device, densifying unrepresentable operand layouts first."""
function gemmlaunch!(C, A, B, M, N, K, α, β)
    Ad = gemmstrides(A) === nothing ? densify(A) : A
    Bd = gemmstrides(B) === nothing ? densify(B) : B
    runlaunches!(KernelAbstractions.get_backend(gemmstrides(C)[1]),
                 scalar_gemm_launches(C, Ad, Bd, M, N, K, α, β))
    return C
end

"""    scalar_gemm_dispatch!(g, C, A, B, M, N, K, α, β; name, partials=nothing) -> C

Declare one graph pass per selected scalar GEMM launch."""
function scalar_gemm_dispatch!(g, C, A, B, M, N, K, α, β;
                               name::AbstractString = "gemm", partials = nothing)
    dispatchlaunches!(g, scalar_gemm_launches(C, A, B, M, N, K, α, β; partials);
                      name)
    return C
end
