# Flash attention on coopmat2, the whole algorithm: workgroup-scope matrices, the
# online softmax, many key blocks, and no shared memory anywhere.
#
# `mwe_flash_cm2.jl` proved the primitive CHAIN on one key block at subgroup
# scope. This is the kernel that chain was for — `flash_attn_cm2.comp`'s
# structure, ported to our `(E, L, H, B)` layout:
#
#     M = max(M, rowmax(S))      P = exp(S - M)      eM = exp(Mold - M)
#     L = eM*L + rowsum(P)       O = O*eM + P·V      out = O / L
#
# Two things it does differently from the reference, both forced and both worth
# knowing:
#
#   * **`max(rowmax, Mold)` is `Mold + relu(rowmax - Mold)`.** The reference gets
#     the elementwise maximum of two matrices from `coopMatPerElementNV(M, rowmax,
#     Max, Mold)` — a per-element op with ANOTHER MATRIX as an extra operand,
#     which our `coopmat_perelement` cannot pass (extras are scalars and
#     pointers). The identity needs only ops we have, and `eM = exp(-relu(d))`
#     falls out of the same `d`.
#   * **The scale is applied to `S`, not to `Q`.** The reference scales `Q` while
#     it is still fp32, because ggml's `q` is fp32 in memory. Ours is fp16, so
#     scaling there would round the operand; scaling `S` is exact and costs one
#     per-element pass over `Br x Bc` against a `Br x Bc x EP` product.
#
# The probe below runs first because the shape-changing smear (`Br x Bc` reduced
# and resized to `Br x EP`) is the one primitive nothing has tested yet, and if
# it is wrong the kernel is wrong in a way that reads as an arithmetic bug.
#
#     julia --project=. dev/Lava/test/mwe_flash_cm2_online.jl
using Lava, KernelAbstractions, Printf, LinearAlgebra
const KA = KernelAbstractions
const WM = Lava.WorkgroupMatrix

# Tiling. `EP` is `E` rounded up to the N granularity this device reports for a
# 256-invocation workgroup (M 32, N 32, K 16 for fp16 x fp16 -> fp32) — the
# clamping layout fills the columns past `E` with zeros, so the padding costs
# arithmetic and nothing else.
const BR = 32
const BC = 64
const NT = 256

# Top-level, not closures: `coopmat_perelement` and `coopmat_reduce` name the
# function in the instruction and there is no operand slot for an environment.
pneg(::UInt32, ::UInt32, e::Float32) = -e
# The running maximum's starting value. `-3e38` and not `-Inf`, so the first
# block's `Mold - M` subtracts finite numbers instead of producing `Inf - Inf`.
pninf(::UInt32, ::UInt32, ::Float32) = -3.0f38
pexp(::UInt32, ::UInt32, e::Float32) = exp(e)
# `max(rowmax, Mold)` as `rowmax + relu(Mold - rowmax)`, both halves of it. The
# anchor matters and the other way round is broken: `Mold + relu(rowmax - Mold)`
# is the same identity in exact arithmetic, but on the FIRST key block
# `Mold = -3e38` and `rowmax - Mold` rounds to `3e38`, so the sum comes back as
# 0 rather than as `rowmax` — the running maximum is then wrong by whatever the
# first block's scores were. Anchored on `rowmax` the first block is exact,
# because `relu(-3e38) == 0` contributes nothing.
pnrelu(::UInt32, ::UInt32, e::Float32) = max(-e, 0.0f0)     # relu(-d)
pexpnrelu(::UInt32, ::UInt32, e::Float32) = exp(-max(e, 0.0f0))  # exp(Mold - M)
pscale(::UInt32, ::UInt32, e::Float32, s::Float32) = e * s
precip(::UInt32, ::UInt32, e::Float32) = e == 0.0f0 ? 0.0f0 : 1.0f0 / e
# Padding elements are the ones the clamping load filled with zero: without this
# they enter the row maximum as 0 and the row sum as `exp(-M)`, which is a wrong
# answer rather than a missing one. `-3.0f38` and not `-Inf`, so that `Mold - M`
# is a subtraction of finite numbers rather than `Inf - Inf`.
pmask(r::UInt32, c::UInt32, e::Float32, nr::UInt32, nc::UInt32) =
    (r >= nr || c >= nc) ? -3.0f38 : e
pmask0(r::UInt32, c::UInt32, e::Float32, nr::UInt32, nc::UInt32) =
    (r >= nr || c >= nc) ? 0.0f0 : e

rmax(x::Float32, y::Float32) = max(x, y)
rsum(x::Float32, y::Float32) = x + y
# The reference's `smearReduce`. Only sound on a matrix already uniform along the
# reduced axis — the combine order is unspecified — which every use here is.
rfirst(x::Float32, y::Float32) = x

"""2-D clamping layout over one `(E, L)` slab: `nl x ne` of the tensor starting
at column `l0`. The tensor's dimensions are the REVERSE of Julia's, its last
being the fastest-varying one, so `(Lfull, Efull)` describes an `(E, L)` array."""
@inline slab(Lfull::Int32, Efull::Int32, l0::Int32, nl, ne) =
    Lava.tensor_slice(
        Lava.tensor_setdim(
            Lava.tensor_layout(Val(2), Val(Lava.TENSOR_CLAMP_CONSTANT)),
            (Lfull, Efull)),
        (l0, Int32(0)), (Int32(nl), Int32(ne)))

# ── The probe: a reduce that CHANGES SHAPE ───────────────────────────────────
@kernel cpu = false unsafe_indices = true function smearprobe!(out, @Const(x),
                                                              ::Val{R}, ::Val{C},
                                                              ::Val{C2}) where {R,C,C2}
    m = Lava.tensor_load(Lava.coopmat_zero(WM{Float32,R,C,Lava.Accumulator}),
                         UInt64(pointer(x)), slab(Int32(R), Int32(C), Int32(0), R, C))
    s = Lava.coopmat_reduce(rsum, WM{Float32,R,C2,Lava.Accumulator}, m,
                            Val(Lava.CoopMatReduce.Row))
    Lava.tensor_store(s, UInt64(pointer(out)), slab(Int32(R), Int32(C2), Int32(0), R, C2))
end

# ── The kernel ───────────────────────────────────────────────────────────────
@kernel cpu = false unsafe_indices = true function flashcm2!(
        out, @Const(q), @Const(k), @Const(v), scale::Float32,
        Lq::Int32, Lk::Int32, H::Int32, ::Val{E}, ::Val{EP}) where {E,EP}
    grp = @index(Group, NTuple)
    qb, h, b = grp[1], grp[2], grp[3]
    q0 = Int32((qb - 1) * BR)

    # Slab bases. Dense `(E, L, H, B)`, so a head's slab is contiguous.
    qoff = (Int32(h - 1) * Lq + Int32(b - 1) * Lq * H) * Int32(E)
    koff = (Int32(h - 1) * Lk + Int32(b - 1) * Lk * H) * Int32(E)
    qa = UInt64(pointer(q)) + UInt64(qoff) * 2
    ka = UInt64(pointer(k)) + UInt64(koff) * 2
    va = UInt64(pointer(v)) + UInt64(koff) * 2
    oa = UInt64(pointer(out)) + UInt64(qoff) * 4      # fp32 destination

    # Q for this block, once, and it stays in registers for every key block.
    mq = Lava.tensor_load(Lava.coopmat_zero(WM{Float16,BR,EP,Lava.MatrixA}), qa,
                          slab(Lq, Int32(E), q0, BR, EP))

    # The transposing view: K's slab is `(E, Lk)` exactly as Q's is, so as the
    # B operand it is the wrong way round. V wants `(Bc, EP)` and is already right.
    vt = Lava.tensor_view(Val(2), Val((1, 0)))

    acc = Lava.coopmat_zero(WM{Float32,BR,EP,Lava.Accumulator})
    lsum = Lava.coopmat_zero(WM{Float32,BR,BC,Lava.Accumulator})
    mrun = Lava.coopmat_perelement(pninf,
               Lava.coopmat_zero(WM{Float32,BR,BC,Lava.Accumulator}))

    nkb = cld(Lk, Int32(BC))
    for j in Int32(0):(nkb - Int32(1))
        j0 = j * Int32(BC)
        lk = slab(Lk, Int32(E), j0, BC, EP)
        mk = Lava.tensor_load(Lava.coopmat_zero(WM{Float16,EP,BC,Lava.MatrixB}),
                              ka, lk, vt)
        s = Lava.coopmat_muladd(mq, mk,
                                Lava.coopmat_zero(WM{Float32,BR,BC,Lava.Accumulator}))
        s = Lava.coopmat_perelement(pscale, s, scale)
        # Rows past `Lq` and columns past `Lk` were filled with zero by the
        # clamping load; they must not reach the row maximum.
        nr = UInt32(max(Int32(0), min(Int32(BR), Lq - q0)))
        nc = UInt32(max(Int32(0), min(Int32(BC), Lk - j0)))
        s = Lava.coopmat_perelement(pmask, s, nr, nc)

        rowmax = Lava.coopmat_reduce(rmax, WM{Float32,BR,BC,Lava.Accumulator}, s,
                                     Val(Lava.CoopMatReduce.Row))
        # max(rowmax, Mold) without a matrix-valued per-element operand, which is
        # the one thing `coopmat_perelement` cannot express:
        #   d = rowmax - Mold ;  M = rowmax + relu(-d) ;  eM = exp(Mold - M) = exp(-relu(d))
        d = Lava.coopmat_add(rowmax, Lava.coopmat_perelement(pneg, mrun))
        mrun = Lava.coopmat_add(rowmax, Lava.coopmat_perelement(pnrelu, d))
        em = Lava.coopmat_perelement(pexpnrelu, d)

        p = Lava.coopmat_perelement(pexp,
                Lava.coopmat_add(s, Lava.coopmat_perelement(pneg, mrun)))
        p = Lava.coopmat_perelement(pmask0, p, nr, nc)
        rowsum = Lava.coopmat_reduce(rsum, WM{Float32,BR,BC,Lava.Accumulator}, p,
                                     Val(Lava.CoopMatReduce.Row))
        lsum = Lava.coopmat_add(Lava.coopmat_mul(em, lsum), rowsum)

        # `eM` resized from `Br x Bc` to `Br x EP` by a reduce that reduces
        # nothing — sound because `eM` is already uniform along its row.
        emd = Lava.coopmat_reduce(rfirst, WM{Float32,BR,EP,Lava.Accumulator}, em,
                                  Val(Lava.CoopMatReduce.Row))
        acc = Lava.coopmat_mul(acc, emd)

        pa = Lava.coopmat_convert(WM{Float16,BR,BC,Lava.MatrixA}, p)
        mv = Lava.tensor_load(Lava.coopmat_zero(WM{Float16,BC,EP,Lava.MatrixB}), va, lk)
        acc = Lava.coopmat_muladd(pa, mv, acc)
    end

    ld = Lava.coopmat_reduce(rfirst, WM{Float32,BR,EP,Lava.Accumulator}, lsum,
                             Val(Lava.CoopMatReduce.Row))
    acc = Lava.coopmat_mul(acc, Lava.coopmat_perelement(precip, ld))
    Lava.tensor_store(acc, oa, slab(Lq, Int32(E), q0, BR, EP))
end

# ── Host side ────────────────────────────────────────────────────────────────
function cpuattn(qh, kh, vh, scale)
    E, Lq = size(qh)
    Lk = size(kh, 2)
    s = (Float32.(qh)' * Float32.(kh)) .* scale        # Lq x Lk
    s .-= maximum(s, dims = 2)
    p = exp.(s)
    p ./= sum(p, dims = 2)
    (p * Float32.(vh)')                                # Lq x E
end

function run(; E = 72, EP = 96, Lq = 128, Lk = 256, H = 2, B = 1, tol = 2e-2)
    back = LavaBackend()
    scale = Float32(1 / sqrt(E))
    qh = Float16.(0.3 .* sin.(reshape(range(0, 31, E * Lq * H * B), E, Lq, H, B)))
    kh = Float16.(0.3 .* cos.(reshape(range(0, 27, E * Lk * H * B), E, Lk, H, B)))
    vh = Float16.(0.3 .* sin.(reshape(range(0, 19, E * Lk * H * B), E, Lk, H, B)))
    Q = KA.allocate(back, Float16, E, Lq, H, B); copyto!(Q, qh)
    K = KA.allocate(back, Float16, E, Lk, H, B); copyto!(K, kh)
    V = KA.allocate(back, Float16, E, Lk, H, B); copyto!(V, vh)
    O = KA.allocate(back, Float32, E, Lq, H, B); fill!(O, Float32(NaN))

    nq = cld(Lq, BR)
    flashcm2!(back, NT)(O, Q, K, V, scale, Int32(Lq), Int32(Lk), Int32(H),
                        Val(E), Val(EP); ndrange = (nq * NT, H, B))
    KA.synchronize(back)
    got = Array(O)

    worst = 0.0
    for b in 1:B, h in 1:H
        want = cpuattn(qh[:, :, h, b], kh[:, :, h, b], vh[:, :, h, b], scale)   # Lq x E
        e = maximum(abs.(got[:, :, h, b]' .- want)) / maximum(abs.(want))
        worst = max(worst, e)
    end
    @printf("E=%-3d EP=%-3d Lq=%-5d Lk=%-5d H=%d  max rel err %.3e  %s\n",
            E, EP, Lq, Lk, H, worst, (isfinite(worst) && worst < tol) ? "OK" : "MISMATCH")
    isfinite(worst) && worst < tol
end

ctx = MVE.vk_context()
if !(ctx.coopmat2.workgroup_scope && ctx.coopmat2.tensor_addressing &&
     ctx.coopmat2.per_element_operations && ctx.coopmat2.reductions)
    @info "device lacks a coopmat2 sub-feature this needs — nothing to run"
else
    back0 = LavaBackend()
    # Probe: does a Row reduce whose destination has a DIFFERENT column count
    # smear correctly? Nothing else tests it, and the kernel leans on it twice.
    # An `R x C` MATRIX lives in a `(C, R)` array — the array's column index is
    # the matrix's row index, which is the mapping `mwe_tensor_gemm_nonsquare.jl`
    # pinned. Getting that backwards here would test the transpose, not the smear.
    let R = 32, C = 64, C2 = 96
        x = Float32.(reshape(1:(R * C), C, R) ./ 97)      # (C, R) = matrix R x C
        X = KA.allocate(back0, Float32, C, R); copyto!(X, x)
        Y = KA.allocate(back0, Float32, C2, R); fill!(Y, Float32(NaN))
        smearprobe!(back0, NT)(Y, X, Val(R), Val(C), Val(C2); ndrange = NT)
        KA.synchronize(back0)
        y = Array(Y)
        want = repeat(sum(x, dims = 1), C2, 1)            # row sums, smeared
        e = maximum(abs.(y .- want)) / maximum(abs.(want))
        @printf("shape-changing row reduce %dx%d -> %dx%d: max rel err %.3e  %s\n",
                R, C, R, C2, e, (isfinite(e) && e < 1e-5) ? "OK" : "MISMATCH")
    end
    println()
    ok = run()                                   # every extent divides its tile
    ok &= run(; Lq = 100, Lk = 200)              # ragged: clamped loads + masking
    ok &= run(; E = 64, EP = 64, Lq = 64, Lk = 128)
    println()
    ok && println("Flash attention runs entirely in workgroup-scope matrices: " *
                  "no shared memory, no staging, one matrix per operand.")
end
