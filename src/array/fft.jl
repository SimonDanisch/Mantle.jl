# Batched FFT/STFT through portable KI kernels and device capabilities.
# Stockham register/shared-memory arrangement derives from VkFFT (MIT,
# Dmitrii Tolmachev). Numeric tilings are Mantle's measured choices.
# Keep real/imaginary shared arrays separate. One shuffle buffer needs uniform
# barriers both before gathering and between all gathers and all scatters.

"""
Radix-R DFT in registers with separate real/imaginary tuples.
SIGN=-1 is forward and +1 inverse; Val specialization folds the ±i rotations.

"""
@inline function fftbutterfly(::Val{2}, xr::NTuple{2}, xi::NTuple{2}, ::Val{SIGN}) where {SIGN}
    return (xr[1] + xr[2], xr[1] - xr[2]), (xi[1] + xi[2], xi[1] - xi[2])
end

@inline function fftbutterfly(::Val{4}, xr::NTuple{4}, xi::NTuple{4}, ::Val{SIGN}) where {SIGN}
    ar, ai = xr[1] + xr[3], xi[1] + xi[3]
    br, bi = xr[1] - xr[3], xi[1] - xi[3]
    cr, ci = xr[2] + xr[4], xi[2] + xi[4]
    dr, di = xr[2] - xr[4], xi[2] - xi[4]
    # e = ∓i * d. Forward (-1): -i*(dr,di) = (di,-dr).
    er, ei = SIGN < 0 ? (di, -dr) : (-di, dr)
    return (ar + cr, br + er, ar - cr, br - er),
           (ai + ci, bi + ei, ai - ci, bi - ei)
end

# Radix 8 as two radix-4s plus the W8 twiddles (Cooley-Tukey):
#   y[k] = E[k] + W8^k O[k],  y[k+4] = E[k] - W8^k O[k]
# with E = DFT4(x0,x2,x4,x6) and O = DFT4(x1,x3,x5,x7). Only W8^1 and W8^3 are
# irrational — W8^0 = 1 and W8^2 = ∓i is a swap-and-negate — so the whole stage
# costs four real multiplies of twiddle beyond the two radix-4s.

@inline function fftbutterfly(::Val{8}, xr::NTuple{8}, xi::NTuple{8}, ::Val{SIGN}) where {SIGN}
    er, ei = fftbutterfly(Val(4), (xr[1], xr[3], xr[5], xr[7]),
                                  (xi[1], xi[3], xi[5], xi[7]), Val(SIGN))
    or, oi = fftbutterfly(Val(4), (xr[2], xr[4], xr[6], xr[8]),
                                  (xi[2], xi[4], xi[6], xi[8]), Val(SIGN))
    c = 0.7071067811865476f0            # cos(pi/4) = sin(pi/4), the only W8 irrational
    sg = Float32(SIGN)
    # W8^0 = 1
    t1r, t1i = or[1], oi[1]
    # W8^1 = (c, sg*c)
    t2r = or[2] * c - oi[2] * (sg * c)
    t2i = or[2] * (sg * c) + oi[2] * c
    # W8^2 = (0, sg) — i.e. ∓i, a swap and a negate, no multiply
    t3r = -sg * oi[3]
    t3i = sg * or[3]
    # W8^3 = (-c, sg*c)
    t4r = or[4] * (-c) - oi[4] * (sg * c)
    t4i = or[4] * (sg * c) + oi[4] * (-c)
    return (er[1] + t1r, er[2] + t2r, er[3] + t3r, er[4] + t4r,
            er[1] - t1r, er[2] - t2r, er[3] - t3r, er[4] - t4r),
           (ei[1] + t1i, ei[2] + t2i, ei[3] + t3i, ei[4] + t4i,
            ei[1] - t1i, ei[2] - t2i, ei[3] - t3i, ei[4] - t4i)
end

"""
    fft_kernel!(dst, src, Val(N), Val(LEAD), Val(SIGN))

NB transforms share a workgroup; each transform uses N÷8 threads and eight
register points per thread. A leading radix-2/4 stage covers powers not divisible
by eight; it runs first because every twiddle is one at that stage. Remaining
stages use radix eight. SKEW optionally pads shared indices for bank conflicts.

"""
function fft_kernel!(
        dst, src, ::Val{N}, ::Val{LEAD}, ::Val{SIGN}, ::Val{SKEW},
        ::Val{NB}) where {N,LEAD,SIGN,SKEW,NB}
    # Shared extents must be compile-time constants. NB packs small transforms;
    # SKEW adds i÷32 padding to shared indices.
    sre = KI.localmemory(Float32, Val((NB * (SKEW ? N + N ÷ 32 : N),)), Val(1))
    sim = KI.localmemory(Float32, Val((NB * (SKEW ? N + N ÷ 32 : N),)), Val(2))
    @inline sx(i) = SKEW ? i + i ÷ 32 : i

    lid = KI.get_local_id().x - 1          # 0 .. NB*T-1
    blk = KI.get_group_id().x - 1
    T = N ÷ 8
    sub = lid ÷ T                            # which transform inside the group
    tid = lid % T                            # 0 .. T-1 within that transform
    base = (blk * NB + sub) * N              # its slice of the global array
    soff = sub * (SKEW ? N + N ÷ 32 : N)     # and of shared

    # ── load, coalesced: consecutive lanes read consecutive elements.
    @inbounds for k in 0:7
        i = tid + k * T
        z = src[base + i + 1]
        sre[soff + sx(i) + 1] = real(z)
        sim[soff + sx(i) + 1] = imag(z)
    end

    Ns = 1
    @inbounds if LEAD > 1
        # Ns == 1: every twiddle is 1, so this is butterflies and nothing else.
        # `M` butterflies in the stage, `8 ÷ LEAD` of them per thread.
        M = N ÷ LEAD
        KI.barrier()
        gr = ntuple(Val(8)) do e
            q, m = (e - 1) ÷ LEAD, (e - 1) % LEAD
            sre[soff + sx(tid + q * T + m * M) + 1]
        end
        gi = ntuple(Val(8)) do e
            q, m = (e - 1) ÷ LEAD, (e - 1) % LEAD
            sim[soff + sx(tid + q * T + m * M) + 1]
        end
        KI.barrier()
        # Unroll tuple indices so register values do not spill to runtime-indexed stacks.
        ntuple(Val(8 ÷ LEAD)) do qq
            q = qq - 1
            xr = ntuple(m -> gr[q * LEAD + m], Val(LEAD))
            xi = ntuple(m -> gi[q * LEAD + m], Val(LEAD))
            yr, yi = fftbutterfly(Val(LEAD), xr, xi, Val(SIGN))
            j = tid + q * T
            for m in 1:LEAD
                sre[soff + sx(j * LEAD + m - 1) + 1] = yr[m]
                sim[soff + sx(j * LEAD + m - 1) + 1] = yi[m]
            end
            zero(Float32)
        end
        Ns = LEAD
    end

    @inbounds while Ns < N
        KI.barrier()
        xr = ntuple(k -> sre[soff + sx(tid + (k - 1) * T) + 1], Val(8))
        xi = ntuple(k -> sim[soff + sx(tid + (k - 1) * T) + 1], Val(8))

        # ── ONE sincos per stage. The eight twiddles are consecutive powers of a
        # single root, so the rest are complex multiplies off `w`.
        idx = tid % Ns
        ang = Float32(SIGN) * 2.0f0 * Float32(pi) * Float32(idx) / Float32(Ns * 8)
        s1, c1 = sincos(ang)
        # powers by recurrence, in registers
        p2c, p2s = c1 * c1 - s1 * s1, 2.0f0 * s1 * c1
        p3c, p3s = p2c * c1 - p2s * s1, p2s * c1 + p2c * s1
        p4c, p4s = p2c * p2c - p2s * p2s, 2.0f0 * p2s * p2c
        p5c, p5s = p4c * c1 - p4s * s1, p4s * c1 + p4c * s1
        p6c, p6s = p3c * p3c - p3s * p3s, 2.0f0 * p3s * p3c
        p7c, p7s = p6c * c1 - p6s * s1, p6s * c1 + p6c * s1
        cs = (1.0f0, c1, p2c, p3c, p4c, p5c, p6c, p7c)
        sn = (0.0f0, s1, p2s, p3s, p4s, p5s, p6s, p7s)
        tr = ntuple(k -> xr[k] * cs[k] - xi[k] * sn[k], Val(8))
        ti = ntuple(k -> xr[k] * sn[k] + xi[k] * cs[k], Val(8))

        yr, yi = fftbutterfly(Val(8), tr, ti, Val(SIGN))

        # ── the barrier that lets one buffer do the work of two.
        KI.barrier()
        out0 = (tid ÷ Ns) * (Ns * 8) + idx
        for k in 1:8
            sre[soff + sx(out0 + (k - 1) * Ns) + 1] = yr[k]
            sim[soff + sx(out0 + (k - 1) * Ns) + 1] = yi[k]
        end
        Ns *= 8
    end

    KI.barrier()
    @inbounds for k in 0:7
        i = tid + k * T
        dst[base + i + 1] = ComplexF32(sre[soff + sx(i) + 1], sim[soff + sx(i) + 1])
    end
    return nothing
end

"""Default shared-index padding. Off in the measured workloads; retain the option
for device-specific tuning rather than assuming fewer bank conflicts is faster."""
const FFT_SKEW = false

"""
    fftplan(N) -> (T, lead)

Threads per transform and the radix of the leading stage (1 = none).

`N` must be a power of two and at least 8. `k = log2(N)` decides the plan:
`k % 3` is 0 for pure radix 8, 1 for a leading radix 2, 2 for a leading radix 4 —
see [`fft_kernel!`](@ref) for why the odd stage leads rather than trails.
"""
function fftplan(N::Integer)
    (N > 0 && count_ones(N) == 1) ||
        throw(ArgumentError("fft!: length $N is not a power of two"))
    k = trailing_zeros(N)
    k >= 3 || throw(ArgumentError(
        "fft!: length $N is below the radix-8 minimum of 8"))
    r = k % 3
    return (N ÷ 8, r == 0 ? 1 : (r == 1 ? 2 : 4))
end

"""
    fftgroup(N, T, nbatch, limit, sharedbudget) -> Int

Transforms per workgroup: target 32 threads, then bound by batch divisibility,
thread limit and half the shared-memory budget. One large transform may use more
than half the budget. These are measured defaults, not a universal optimum.

"""
function fftgroup(N::Int, T::Int, nbatch::Int, limit::Int, sharedbudget::Int)
    want = max(1, cld(32, T))                       # one warp; measured, see above
    byshared = max(1, (sharedbudget ÷ 2) ÷ (2 * sizeof(Float32) * N))
    bylimit = max(1, limit ÷ T)
    nb = min(want, byshared, bylimit, nbatch)
    while nb > 1 && nbatch % nb != 0                # only whole groups
        nb -= 1
    end
    return nb
end

"""
    fft!(dst, src; inverse = false) -> dst

Batched 1D FFT along the **first** dimension, out of place.

`src` is `(N, batch...)` of `ComplexF32` with `N` a power of two; every trailing
dimension is a separate transform. Column-major, so the transform axis is the
contiguous one — which is what makes the load coalesced.

The inverse is **unnormalised**, matching `AbstractFFTs.bfft`: `ifft` is this
divided by `N`, and leaving the scale out keeps it off the critical path for
callers who fold it into a later op.
"""
function fft!(dst::AbstractGPUArray{ComplexF32}, src::AbstractGPUArray{ComplexF32};
              inverse::Bool = false, skew::Bool = FFT_SKEW,
              group::Union{Nothing,Int} = nothing)
    be = get_backend(src)
    N,T,lead,nb,nbatch = fft_config(dst,src,caps(be);skew,group)
    nbatch == 0 && return dst
    kilaunch!(be,fft_kernel!,fft_args(dst,src,N,lead,inverse,skew,nb)...;
              ndrange=T*nbatch,workgroupsize=T*nb)
    return dst
end

"""
    fft(src; inverse = false) -> AbstractGPUArray

Allocating [`fft!`](@ref).
"""
fft(src::AbstractGPUArray{ComplexF32}; inverse::Bool = false, skew::Bool = FFT_SKEW) =
    fft!(similar(src), src; inverse, skew)

# ─────────────────────────────────────────────────────────── real-input transform
#
# Every audio model here feeds the FFT real samples, and a real transform is
# half the work and half the shared memory of the complex one it is built from.
# This is VkFFT's `vkFFT_R2C_even_decomposition.h`.
#
# The packing step is **free**. The decomposition wants
# `z[j] = x[2j] + i x[2j+1]`, and a `Float32` array reinterpreted as
# `ComplexF32` is exactly that — no kernel, no copy, just a different view of the
# same bytes. So `rfft!` is one half-length complex FFT plus one cheap pass.

"""
    rfft_post_kernel!(dst, Z, Val(N), Val(SIGN), scale)

Split the half-length spectrum back into the real transform's `N ÷ 2 + 1` bins.

`Z = FFT_{N/2}(z)` holds the even and odd sub-spectra summed; separating them is
the conjugate-symmetric pair

    e[k] = (Z[k] + conj(Z[H-k])) / 2        the even-indexed samples' transform
    o[k] = (Z[k] - conj(Z[H-k])) / 2i       the odd-indexed samples'

and `X[k] = e[k] + W_N^k o[k]`. The `mod` on both indices is what makes `k = 0`
and `k = H` fall out without their own branch: at `k = 0` both reads hit `Z[0]`,
so `o[0]` is real and `X[0] = e[0] + o[0]` is the DC bin.
"""
function rfft_post_kernel!(
        dst, Z, ::Val{N}, ::Val{SIGN}, scale::Float32) where {N,SIGN}
    H = N ÷ 2
    g = KI.get_global_id().x - 1
    g < length(dst) || return nothing   # the launch is whole workgroups
    k = g % (H + 1)                 # bin
    c = g ÷ (H + 1)                 # which transform
    @inbounds begin
        zk = Z[c * H + (k % H) + 1]
        zm = Z[c * H + ((H - k) % H) + 1]
        er = 0.5f0 * (real(zk) + real(zm))
        ei = 0.5f0 * (imag(zk) - imag(zm))
        # (zk - conj(zm)) / 2i  =  -i * (zk - conj(zm)) / 2
        dr = 0.5f0 * (real(zk) - real(zm))
        di = 0.5f0 * (imag(zk) + imag(zm))
        or, oi = di, -dr
        ang = Float32(SIGN) * Float32(pi) * Float32(k) / Float32(H)
        sn, cs = sincos(ang)
        dst[c * (H + 1) + k + 1] = scale * ComplexF32(er + or * cs - oi * sn,
                                                      ei + or * sn + oi * cs)
    end
    return nothing
end

"""
    rfft!(dst, src) -> dst

Batched **real** 1D FFT along the first dimension.

`src` is `(N, batch...)` of `Float32` with `N` even and `N ÷ 2` factorising over
the supported radices; `dst` is `(N ÷ 2 + 1, batch...)` of `ComplexF32` — the non-redundant
half, the same convention as `AbstractFFTs.rfft` and `torch.fft.rfft`.

Costs one `N ÷ 2` complex transform plus a linear pass, so roughly half of
`fft!` on the same `N`.
"""
function rfft!(dst::AbstractGPUArray{ComplexF32}, src::AbstractGPUArray{Float32};
               skew::Bool = FFT_SKEW)
    N,H,nbatch = rfft_geometry(dst,src)
    nbatch == 0 && return dst
    # The pack, for free: N reals ARE H complex in the interleave the
    # decomposition wants.
    z = reshape(reinterpret(ComplexF32, reshape(src, length(src))), H, nbatch)
    # `fftany!`, not `fft!`: H is 200 for Whisper's 400-point mel and 480 for
    # DeepFilterNet3's 960, neither a power of two.
    Z = fftany!(similar(z), z; skew)
    backend = get_backend(src)
    kilaunch!(backend, rfft_post_kernel!, dst, Z, Val(N), Val(-1), 1f0;
              ndrange = (H + 1) * nbatch)
    return dst
end

"""
    rfft(src) -> AbstractGPUArray{ComplexF32}

Allocating [`rfft!`](@ref).
"""
function rfft(src::AbstractGPUArray{Float32}; skew::Bool = FFT_SKEW)
    N = size(src, 1)
    nbatch = length(src) ÷ N
    dst = similar(src, ComplexF32, (N ÷ 2 + 1, Base.tail(size(src))...))
    rfft!(reshape(dst, N ÷ 2 + 1, nbatch), src; skew)
    return dst
end

# Mixed radices cover non-power-of-two audio lengths such as 400 and 960.
# Smaller stages run multiple guarded butterflies per thread while keeping
# all stage gathers before any scatter. The tuned power-of-two kernel stays distinct.

"""
    fftplan_mixed(N; radices = (8, 5, 4, 3, 2)) -> Tuple

The radix sequence for `N`, largest first, or `nothing` if `N` does not
factorise over `radices`.

Largest-first is not arbitrary: a bigger radix means fewer stages, and a stage
costs a shared-memory round trip and two barriers. The residue is taken by the
smaller radices, so `960` plans as `(8, 8, 5, 3)` — four stages — rather than the
ten a pure radix-2 plan would need.
"""
function fftplan_mixed(N::Integer; radices = (8, 5, 4, 3, 2))
    n = Int(N)
    n >= 2 || throw(ArgumentError("fft!: length $N is too small"))
    all(r->r isa Integer && r>=2,radices) ||
        throw(ArgumentError("fft!: radix choices must be integers >= 2"))
    rs = Int[]
    for r in radices
        while n % r == 0
            push!(rs, r)
            n ÷= r
        end
    end
    n == 1 || return nothing
    return Tuple(rs)
end

struct MixedFFTKernel{RS,SIGN} <: Function end

"""
    fftmixed_kernel(RS, sign) -> kernel

Return a zero-size callable specialized by radix sequence and direction.
Its generated method emits literal Float32 DFT coefficients and fixed indices;
there are no runtime transcendental coefficient calculations or eval-defined
methods. Julia owns specialization caching, including downstream precompilation.
Barriers stay outside guarded butterflies so every workgroup lane reaches them.

"""
fftmixed_kernel(RS::Tuple, sign::Int) = MixedFFTKernel{RS,sign}()

@generated function (kern::MixedFFTKernel{RS,SIGN})(dst,src) where {RS,SIGN}
    sign = SIGN
    RMAX = maximum(RS)
    N = prod(RS)
    TT = N ÷ RMAX
    stages = Expr[]
    Ns = 1
    for r in RS
        B = N ÷ r
        reps = cld(RMAX, r)
        wc = [Float32(cospi(2 * sign * (k - 1) * (m - 1) / r)) for k in 1:r, m in 1:r]
        ws = [Float32(sinpi(2 * sign * (k - 1) * (m - 1) / r)) for k in 1:r, m in 1:r]
        # Gather/compute every rep before any scatter: Stockham is not in-place.
        # Keep each rep's register values live across the uniform barrier.
        gathers = Expr[]; scatters = Expr[]
        for q in 0:(reps - 1)
            J, OK = Symbol(:j_, q), Symbol(:ok_, q)
            ex = Expr[]
            push!(ex, :($J = tid + $(q * TT)))
            push!(ex, :($OK = $J < $B))
            for k in 1:r
                push!(ex, :($(Symbol(:xr_, q, :_, k)) = $OK ? sre[$J + $((k - 1) * B) + 1] : 0.0f0))
                push!(ex, :($(Symbol(:xi_, q, :_, k)) = $OK ? sim[$J + $((k - 1) * B) + 1] : 0.0f0))
            end
            push!(ex, :($(Symbol(:idx_, q)) = $J % $Ns))
            push!(ex, :($(Symbol(:ang_, q)) = $(Float32(sign * 2 * pi / (Ns * r))) *
                                              Float32($(Symbol(:idx_, q)))))
            push!(ex, :(($(Symbol(:sn_, q, :_1)), $(Symbol(:cs_, q, :_1))) =
                        sincos($(Symbol(:ang_, q)))))
            for k in 2:(r - 1)
                push!(ex, :($(Symbol(:cs_, q, :_, k)) = $(Symbol(:cs_, q, :_, k-1)) * $(Symbol(:cs_, q, :_1)) -
                                                        $(Symbol(:sn_, q, :_, k-1)) * $(Symbol(:sn_, q, :_1))))
                push!(ex, :($(Symbol(:sn_, q, :_, k)) = $(Symbol(:sn_, q, :_, k-1)) * $(Symbol(:cs_, q, :_1)) +
                                                        $(Symbol(:cs_, q, :_, k-1)) * $(Symbol(:sn_, q, :_1))))
            end
            for k in 1:r
                xr, xi = Symbol(:xr_, q, :_, k), Symbol(:xi_, q, :_, k)
                tr, ti = Symbol(:tr_, q, :_, k), Symbol(:ti_, q, :_, k)
                if k == 1
                    push!(ex, :($tr = $xr)); push!(ex, :($ti = $xi))
                else
                    c, sq = Symbol(:cs_, q, :_, k-1), Symbol(:sn_, q, :_, k-1)
                    push!(ex, :($tr = $xr * $c - $xi * $sq))
                    push!(ex, :($ti = $xr * $sq + $xi * $c))
                end
            end
            for k in 1:r
                re = Expr(:call, :+); im = Expr(:call, :+)
                for m in 1:r
                    c, sb = wc[k, m], ws[k, m]
                    tr, ti = Symbol(:tr_, q, :_, m), Symbol(:ti_, q, :_, m)
                    if c != 0
                        push!(re.args, c == 1 ? tr : :($tr * $c))
                        push!(im.args, c == 1 ? ti : :($ti * $c))
                    end
                    if sb != 0
                        push!(re.args, :(-($ti * $sb)))
                        push!(im.args, :($tr * $sb))
                    end
                end
                push!(ex, :($(Symbol(:yr_, q, :_, k)) = $re))
                push!(ex, :($(Symbol(:yi_, q, :_, k)) = $im))
            end
            push!(gathers, Expr(:block, ex...))
            sc = Expr[]
            for k in 1:r
                push!(sc, :(sre[$(Symbol(:out0_, q)) + $((k - 1) * Ns) + 1] = $(Symbol(:yr_, q, :_, k))))
                push!(sc, :(sim[$(Symbol(:out0_, q)) + $((k - 1) * Ns) + 1] = $(Symbol(:yi_, q, :_, k))))
            end
            push!(scatters, quote
                if $OK
                    $(Symbol(:out0_, q)) = ($J ÷ $Ns) * $(Ns * r) + $(Symbol(:idx_, q))
                    $(sc...)
                end
            end)
        end
        body = Expr[Expr(:block, gathers...),
                    :(KI.barrier()),          # UNIFORM, between all reads and all writes
                    Expr(:block, scatters...)]
        push!(stages, quote
            KI.barrier()
            $(body...)
        end)
        Ns *= r
    end
    load = Expr[]; store = Expr[]
    for k in 0:(RMAX - 1)
        push!(load, quote
            let i = tid + $(k * TT)
                if i < $N
                    z = src[base + i + 1]
                    sre[i + 1] = real(z)
                    sim[i + 1] = imag(z)
                end
            end
        end)
        push!(store, quote
            let i = tid + $(k * TT)
                if i < $N
                    dst[base + i + 1] = ComplexF32(sre[i + 1], sim[i + 1])
                end
            end
        end)
    end
    return quote
        sre = KI.localmemory(Float32, Val(($N,)), Val(1))
        sim = KI.localmemory(Float32, Val(($N,)), Val(2))
        tid = KI.get_local_id().x - 1
        blk = KI.get_group_id().x - 1
        base = blk * $N
        @inbounds begin
            $(load...)
            $(stages...)
            KI.barrier()
            $(store...)
        end
        return nothing
    end
end

"""
    fftmixed!(dst, src, RS; inverse = false) -> dst

Batched transform over the radix sequence `RS`, for lengths the power-of-two
kernel cannot take. `prod(RS)` must equal `size(src, 1)`.

Uses one transform per workgroup without skew or batch grouping. The literal
butterfly generator supports radix sequences beyond the tuned power-of-two path.

"""
function fftmixed!(dst::AbstractGPUArray{ComplexF32}, src::AbstractGPUArray{ComplexF32},
                   RS::Tuple; inverse::Bool = false)
    be = get_backend(src)
    T,nbatch = fftmixed_config(dst,src,RS,caps(be))
    nbatch == 0 && return dst
    kern = fftmixed_kernel(RS,inverse ? 1 : -1)
    kilaunch!(be,kern,dst,src;ndrange=T*nbatch,workgroupsize=T)
    return dst
end

"""
    fftany!(dst, src; inverse = false) -> dst

Select the tuned power-of-two kernel or a supported mixed-radix sequence.
skew controls the power-of-two shared layout; the mixed layout has no skew option.

"""
function fftany!(dst::AbstractGPUArray{ComplexF32}, src::AbstractGPUArray{ComplexF32};
                 inverse::Bool = false, skew::Bool = FFT_SKEW)
    RS = fft_radices(size(src,1))
    return RS === nothing ? fft!(dst,src;inverse,skew) : fftmixed!(dst,src,RS;inverse)
end

function rfft_geometry(dst,src)
    N = size(src,1)
    N > 0 && iseven(N) || throw(ArgumentError("rfft!: length $N must be positive and even"))
    H = N ÷ 2
    nbatch = length(src) ÷ N
    size(dst,1) == H+1 && length(dst) == (H+1)*nbatch || throw(DimensionMismatch(
        "rfft!: destination $(size(dst)) must hold $(H+1) bins for $nbatch batches"))
    return N,H,nbatch
end

# Configuration is shared; eager calls keep Val arguments on the direct launch path.
@inline fft_args(dst,src,N,lead,inverse,skew,nb) =
    (dst,src,Val(N),Val(lead),Val(inverse ? 1 : -1),Val(skew),Val(nb))

function fft_config(dst,src,c; skew::Bool=FFT_SKEW, group=nothing)
    size(dst) == size(src) || throw(DimensionMismatch(
        "fft!: destination $(size(dst)) does not match source $(size(src))"))
    N = size(src,1)
    T,lead = fftplan(N)
    nbatch = length(src) ÷ N
    nbatch == 0 && return (N,T,lead,0,0)
    T <= c.workgrouplimit || throw(ArgumentError(
        "fft!: N=$N needs $T threads, above this device's limit of $(c.workgrouplimit). " *
        "A multi-pass decomposition is not implemented."))
    nb = group === nothing ? fftgroup(N,T,nbatch,c.workgrouplimit,c.sharedbudget) : group
    nb > 0 && nbatch % nb == 0 && T*nb <= c.workgrouplimit || throw(ArgumentError(
        "fft!: group $nb must divide $nbatch batches and fit $(c.workgrouplimit) threads"))
    shared = 2sizeof(Float32)*nb*(skew ? N+N÷32 : N)
    shared <= c.sharedbudget || throw(ArgumentError(
        "fft!: group $nb needs $shared shared bytes, above this device's budget of $(c.sharedbudget)"))
    return N,T,lead,nb,nbatch
end

function fftmixed_config(dst,src,RS::Tuple,c)
    N = size(src,1)
    !isempty(RS) && all(r->r isa Integer && r>=2,RS) && prod(RS) == N || throw(ArgumentError(
        "fftmixed!: radix sequence $RS must contain radices >= 2 and multiply to $N"))
    size(dst) == size(src) || throw(DimensionMismatch(
        "fftmixed!: destination $(size(dst)) does not match source $(size(src))"))
    nbatch = length(src) ÷ N
    nbatch == 0 && return (0,0)
    T = N ÷ maximum(RS)
    T <= c.workgrouplimit || throw(ArgumentError(
        "fftmixed!: N=$N needs $T threads, above this device's limit of $(c.workgrouplimit)"))
    shared = 2sizeof(Float32)*N
    shared <= c.sharedbudget || throw(ArgumentError(
        "fftmixed!: N=$N needs $shared shared bytes, above this device's budget of $(c.sharedbudget)"))
    return T,nbatch
end

function fft_radices(N)
    count_ones(N) == 1 && N >= 8 && return nothing
    RS = fftplan_mixed(N)
    RS === nothing && throw(ArgumentError(
        "fftany!: length $N does not factorise over radices (8,5,4,3,2); " *
        "Rader and Bluestein transforms are not implemented"))
    return RS
end

function fftany_launches(dst,src,c; inverse::Bool=false, skew::Bool=FFT_SKEW)
    RS = fft_radices(size(src,1))
    if RS === nothing
        N,T,lead,nb,nbatch = fft_config(dst,src,c;skew)
        nbatch == 0 && return ()
        return (ArrayLaunch(fft_kernel!,fft_args(dst,src,N,lead,inverse,skew,nb),T*nbatch,T*nb),)
    end
    T,nbatch = fftmixed_config(dst,src,RS,c)
    nbatch == 0 && return ()
    return (ArrayLaunch(fftmixed_kernel(RS,inverse ? 1 : -1),(dst,src),T*nbatch,T),)
end

# ─────────────────────────────────────────────────────── declared graph forms

"""Pack adjacent real samples as the complex input used by the real-FFT split."""
function rfft_pack_kernel!(dst, src)
    i = KI.get_global_id().x
    i <= length(dst) || return nothing
    @inbounds dst[i] = ComplexF32(src[2i - 1], src[2i])
    return nothing
end

"""Restore the redundant half of a real signal's Hermitian spectrum."""
function irfft_extend_kernel!(
        dst, src, ::Val{N}, ::Val{NB}) where {N,NB}
    g = KI.get_global_id().x - 1
    g < length(dst) || return nothing
    k = g % N
    c = g ÷ N
    @inbounds dst[g + 1] = k < NB ? src[c * NB + k + 1] :
                                    conj(src[c * NB + (N - k) + 1])
    return nothing
end

"""Take and scale the real component after an unnormalised inverse FFT."""
function irfft_real_kernel!(
        dst, src, scale::Float32)
    i = KI.get_global_id().x
    i <= length(dst) || return nothing
    @inbounds dst[i] = real(src[i]) * scale
    return nothing
end

"""
    fftany_dispatch!(graph, dst, src; inverse = false, name = "fft") -> dst

Declare a batched complex FFT along the first dimension.  This is the graph
form of [`fftany!`](@ref): it selects the same tuned power-of-two or generated
mixed-radix kernel, but records the launch rather than executing it.
"""
function fftany_dispatch!(g, dst, src; inverse::Bool = false, skew::Bool = FFT_SKEW,
                          name::AbstractString = "fft")
    dispatchlaunches!(g,fftany_launches(dst,src,caps(backend(g.dev)); inverse,skew);name)
    return dst
end

"""
    rfft_dispatch!(graph, dst, src, packed, spectrum; name = "rfft") -> dst

Declare a real forward FFT. `packed` and `spectrum` are `(N/2, batch...)`
complex scratch resources supplied by the caller, so their lifetimes remain
visible to the graph placer.
"""
function rfft_dispatch!(g, dst, src, packed, spectrum;
                        scale::Real = 1f0,
                        name::AbstractString = "rfft")
    N,H,nbatch = rfft_geometry(dst,src)
    size(packed,1) == H || throw(DimensionMismatch(
        "rfft_dispatch!: packed scratch first dimension is $(size(packed,1)), expected $H"))
    length(packed) == H * nbatch || throw(DimensionMismatch(
        "rfft_dispatch!: packed scratch has $(length(packed)) elements, expected $(H * nbatch)"))
    size(spectrum) == size(packed) || throw(DimensionMismatch(
        "rfft_dispatch!: spectrum scratch $(size(spectrum)) does not match packed $(size(packed))"))
    nbatch == 0 && return dst
    dispatch!(g, rfft_pack_kernel!, (packed, src), H * nbatch;
              name = "$name/pack")
    fftany_dispatch!(g, spectrum, packed; name = "$name/fft")
    dispatch!(g, rfft_post_kernel!,
              (dst, spectrum, Val(N), Val(-1), Float32(scale)),
              (H + 1) * nbatch; name = "$name/post")
    return dst
end

"""
    irfft_dispatch!(graph, dst, src, full, transformed; scale = 1, name = "irfft")

Declare an inverse real FFT from the non-redundant half spectrum. `full` and
`transformed` are full-length complex scratch resources. The underlying inverse
FFT is unnormalised; `scale` is folded into the final real-component store.
"""
function irfft_dispatch!(g, dst, src, full, transformed;
                         scale::Real = 1f0,
                         name::AbstractString = "irfft")
    N = size(dst, 1)
    NB = size(src, 1)
    N >= 2 || throw(ArgumentError("irfft_dispatch!: length $N must be at least 2"))
    length(src) == NB*(length(dst)÷N) || throw(DimensionMismatch(
        "irfft_dispatch!: source and destination batch counts differ"))
    size(full,1) == N || throw(DimensionMismatch(
        "irfft_dispatch!: scratch first dimension is $(size(full,1)), expected $N"))
    NB == N ÷ 2 + 1 || throw(DimensionMismatch(
        "irfft_dispatch!: half spectrum has $NB bins, expected $(N ÷ 2 + 1) for length $N"))
    size(full) == size(transformed) || throw(DimensionMismatch(
        "irfft_dispatch!: scratch shapes $(size(full)) and $(size(transformed)) differ"))
    length(full) == length(dst) || throw(DimensionMismatch(
        "irfft_dispatch!: full scratch has $(length(full)) elements, expected $(length(dst))"))
    length(dst) == 0 && return dst
    dispatch!(g, irfft_extend_kernel!, (full, src, Val(N), Val(NB)), length(full);
              name = "$name/extend")
    fftany_dispatch!(g, transformed, full; inverse = true, name = "$name/fft")
    dispatch!(g, irfft_real_kernel!, (dst, transformed, Float32(scale)), length(dst);
              name = "$name/real")
    return dst
end

# ────────────────────────────────────────────────────────────────────────── STFT

"""
    stft_frames_kernel!(frames, x, window, Val(NFFT), hop, len, Val(CENTER))

Cut `x` into overlapping windowed frames, `(NFFT, T)`, ready for [`rfft!`](@ref).

`CENTER` matches `torch.stft(center=true)`: the signal is reflect-padded by
`NFFT ÷ 2` so frame `t` is centred on sample `t * hop`. The reflection is
computed per sample rather than materialised, so no padded copy of the input
exists — which matters because the padded signal is the largest array in a mel
front end and it would be read exactly once.
"""
function stft_frames_kernel!(
        frames, x, window, ::Val{NFFT}, hop::Int, len::Int,
        ::Val{CENTER}) where {NFFT,CENTER}
    g = KI.get_global_id().x - 1
    g < length(frames) || return nothing
    i = g % NFFT                 # position within the frame
    t = g ÷ NFFT                 # which frame
    @inbounds begin
        p = t * hop + i - (CENTER ? NFFT ÷ 2 : 0)
        # reflect at both ends, without repeating the edge sample — the same
        # convention as torch's 'reflect' pad.
        if p < 0
            p = -p
        end
        if p >= len
            p = 2 * len - 2 - p
        end
        v = (p >= 0 && p < len) ? x[p + 1] : 0.0f0
        frames[g + 1] = v * window[i + 1]
    end
    return nothing
end

"""
    stft(x, nfft, hop, window; center = true) -> AbstractGPUArray{ComplexF32}

Short-time Fourier transform of a real signal, `(nfft ÷ 2 + 1, frames)`.

Matches `torch.stft(x, nfft, hop, window=window, center=center,
return_complex=true)` — reflect padding, frame `t` centred on `t * hop`, and the
non-redundant half of each spectrum.

`nfft` need not be a power of two: Whisper's mel is 400 and DeepFilterNet3's is
960, and [`fftany!`](@ref) covers both.

**Single channel only.** `x` is a vector; a stereo or batched signal has to be
looped, which is fine for Whisper (mono by definition) and not for Demucs
(stereo). Batching it is a change to this function and not to the transform —
`rfft!` underneath is already batched over its trailing dimensions — but it is
not written, and doing it blind would be guessing at the layout the caller wants.
"""
function stft(x::AbstractGPUArray{Float32}, nfft::Int, hop::Int,
              window::AbstractGPUArray{Float32}; center::Bool = true)
    length(window) == nfft || throw(DimensionMismatch(
        "stft: window has $(length(window)) samples, expected nfft = $nfft"))
    len = length(x)
    nframes = center ? len ÷ hop + 1 : (len - nfft) ÷ hop + 1
    nframes > 0 || throw(ArgumentError("stft: signal of $len samples is shorter than nfft = $nfft"))
    backend = get_backend(x)
    frames = similar(x, Float32, nfft, nframes)
    kilaunch!(backend, stft_frames_kernel!, frames, x, window, Val(nfft), hop, len,
              Val(center); ndrange = nfft * nframes)
    return rfft(frames)
end

"""
    hannwindow(backend, n; periodic = true) -> AbstractGPUArray{Float32}

The Hann window `torch.hann_window` produces: periodic by default, which is what
every STFT in this repo wants (a symmetric window is for filter design, and using
it here shifts every bin slightly).
"""
function hannwindow(backend, n::Int; periodic::Bool = true)
    d = periodic ? n : n - 1
    h = Float32[0.5f0 * (1 - cospi(Float32(2 * k / d))) for k in 0:(n - 1)]
    w = KernelAbstractions.allocate(backend, Float32, n)
    copyto!(w, h)
    return w
end
