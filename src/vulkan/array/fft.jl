# Batched 1D FFT, ported from VkFFT (`dev/VkFFT`, MIT, Dmitrii Tolmachev).
#
# What was taken from it is the ARRANGEMENT, which is not the obvious one. The
# obvious design keeps the whole transform in `@localmem` and butterflies it in
# place. VkFFT does not: its working set lives in **registers**
# (`logicalStoragePerThread` — `vkFFT_RegisterBoost.h`), `fftDim /
# logicalStoragePerThread` threads cooperate on one transform, and shared memory
# is only the *shuffle buffer* between stages
# (`appendRegistersToShared`/`appendSharedToRegisters`, `vkFFT_RadixShuffle.h`).
# Registers are faster than shared and, more importantly, holding only one shared
# array instead of the whole working set is what decides how many transforms fit
# per workgroup.
#
# Constants are NOT taken from it. VkFFT's radix and register choices are tuned
# across its whole device matrix; per `feedback-port-sota-kernels` the structure
# ports and every number gets re-measured here.
#
# ## One buffer, two barriers
#
# Stockham is not in-place: a stage reads with stride `T` and writes with the
# permuted stride `Ns`. The textbook fix is two buffers, ping-ponged — which for
# N = 4096 is 2 x 4096 x 8 = 64 KB and does **not fit** in this device's 49152 B
# (`caps(ctx).sharedbudget`).
#
# One buffer suffices with a barrier *between the read and the write*: every
# thread has finished gathering its R points into registers before any thread
# writes. That is two barriers per stage instead of one, and it is what makes
# N = 4096 land at 32 KB and fit at all.
#
# ## Where it lands
#
# Against a **same-size** copy kernel, interleaved, 2^21 complex points total.
# "First" is the initial radix-4 version before any of the tuning below.
#
#     N      group  threads   ms       GB/s    % of copy   first
#     64     4      32        0.108    310     42.6%       30.2%
#     128    2      32        0.073    459     61.5%       45.6%
#     256    1      32        0.069    485     59.2%       59.8%
#     512    1      64        0.072    466     57.6%       21.1%   <- Whisper
#     1024   1      128       0.076    444     55.0%       38.3%   <- DeepFilterNet3
#     2048   1      256       0.118    284     36.2%       33.9%
#     4096   1      512       0.186    181     23.0%       16.0%   <- Demucs
#
# Same-size matters: an earlier table compared everything against one 128 MB
# copy and reported 128% for the small shapes, which only meant the FFT working
# set was cache-resident and the copy was not.
#
# **N >= 2048 is the known weak point and it is shared memory.** All of 1024,
# 2048 and 4096 run four stages, so stage count does not explain the fall-off;
# shared occupancy does. The `skew` experiment measured the sensitivity
# directly — a 5.6% larger shared allocation cost 7.3% at N = 4096, 3.6% at
# N = 1024, and was a *gain* at N = 256. Single-pass Stockham needs N complex in
# shared and there is no way around that; the fix is VkFFT's multi-pass
# decomposition (`numAxisUploads > 1`, 4096 = 64 x 64 through global memory),
# which is not implemented. Demucs is the only model here that wants 4096 and it
# is not ported, so this is documented rather than built.
#
# ## Real and imaginary in separate arrays
#
# Not a `Complex` array. Each is then a stride-1 access per lane where an
# interleaved layout strides by 2, and it keeps `Complex` out of `@localmem`
# entirely — see `lava-localmem-silent-miscompile` for how quietly that class of
# thing fails here.

"""
The radix-R DFT on R points held in registers, real and imaginary separate.

`SIGN` is -1 for the forward transform and +1 for the inverse, and it appears
only as the sense of the multiply-by-i — which is why it is a `Val` rather than a
runtime factor: at -1 the rotation is `(a,b) -> (b,-a)`, six of the eight
radix-4 flops fold away, and a runtime sign would leave a select in the inner
loop of every stage.
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

One transform per workgroup, `T = N ÷ 8` threads, **eight points per thread**,
radix 8 throughout with one leading radix-2 or radix-4 stage when `N` is not a
power of eight. `LEAD` is that leading radix, or 1 for none.

## Eight points per thread, not four

This is VkFFT's `registerBoost` (`vkFFT_RegisterBoost.h`): the working set per
thread is a *tuning parameter*, and raising it removes whole stages. The cost of
a stage here is not the butterfly — it is the shared-memory round trip and the
two barriers around it. At four points per thread `N = 4096` took six stages and
measured **16% of a same-size copy**; the arithmetic was never the problem, the
6x shared traffic over global was.

Radix 8 makes that four stages, and `N = 512` three.

## Why the odd stage goes first

`N = 2^k` is a power of eight only when `k % 3 == 0`. Otherwise one stage of
radix 2 (`k % 3 == 1`) or radix 4 (`k % 3 == 2`) makes up the difference, and it
goes **first** because at `Ns = 1` every twiddle is exactly 1 — the angle is
`idx/(Ns*R)` and `idx = tid % 1 = 0`. Putting it last would cost a full twiddle
pass. Each thread runs `8 ÷ LEAD` of those butterflies, so no lane idles.
"""
@kernel cpu=false unsafe_indices=true function fft_kernel!(
        dst, @Const(src), ::Val{N}, ::Val{LEAD}, ::Val{SIGN}, ::Val{SKEW},
        ::Val{NB}) where {N,LEAD,SIGN,SKEW,NB}
    # `Val` parameters, not locals: a `@localmem` whose size comes from a local
    # variable compiles, runs, and writes NOTHING on this backend.
    # `SKEW` pads the shared arrays so that a stride-8 write does not serialise.
    # The Stockham store address is `(tid ÷ Ns) * (Ns * 8) + idx`; at `Ns = 1`
    # that is `tid * 8`, which puts a whole warp on 4 of the 32 banks. Adding
    # `i ÷ 32` spreads them. Costs N ÷ 32 floats per array.
    # `NB` transforms per workgroup. At N = 64 one transform needs T = 8
    # threads, so a workgroup would be a quarter of a warp and three lanes in
    # four would idle — measured 30.2% of copy against 59.8% at N = 256. Packing
    # several transforms into the workgroup fixes the shape of the launch
    # without touching the algorithm.
    sre = @localmem Float32 (NB * (SKEW ? N + N ÷ 32 : N),)
    sim = @localmem Float32 (NB * (SKEW ? N + N ÷ 32 : N),)
    @inline sx(i) = SKEW ? i + i ÷ 32 : i

    lid = @index(Local, Linear) - 1          # 0 .. NB*T-1
    blk = @index(Group, Linear) - 1
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
        @synchronize
        gr = ntuple(Val(8)) do e
            q, m = (e - 1) ÷ LEAD, (e - 1) % LEAD
            sre[soff + sx(tid + q * T + m * M) + 1]
        end
        gi = ntuple(Val(8)) do e
            q, m = (e - 1) ÷ LEAD, (e - 1) % LEAD
            sim[soff + sx(tid + q * T + m * M) + 1]
        end
        @synchronize
        # `ntuple(Val(...))` rather than `for q in 0:...`: the body indexes the
        # register tuples `gr`/`gi`, and a tuple indexed by a RUNTIME value is a
        # stack slot on this backend — the whole working set would spill to
        # scratch memory. `ntuple` with a `Val` is unrolled and inlined, so every
        # tuple index below is a literal. The returned tuple is discarded.
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
        @synchronize
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
        @synchronize
        out0 = (tid ÷ Ns) * (Ns * 8) + idx
        for k in 1:8
            sre[soff + sx(out0 + (k - 1) * Ns) + 1] = yr[k]
            sim[soff + sx(out0 + (k - 1) * Ns) + 1] = yi[k]
        end
        Ns *= 8
    end

    @synchronize
    @inbounds for k in 0:7
        i = tid + k * T
        dst[base + i + 1] = ComplexF32(sre[soff + sx(i) + 1], sim[soff + sx(i) + 1])
    end
end

"""
Whether the shared arrays are padded to break bank conflicts on the Stockham
write.

**Measured OFF.** The theory was good and the measurement disagreed. At `Ns = 1`
the store address is `tid * 8`, which puts a warp on 4 of the 32 banks, so the
padding should have paid — and against a same-size copy, interleaved, it went

    256x6000    62.3% -> 63.9%   (+2.6%)
    512x3000    61.2% -> 58.2%   (-4.9%)
    1024x1500   58.4% -> 56.3%   (-3.6%)
    4096x512    23.1% -> 21.4%   (-7.3%)

i.e. a loss on three of four shapes. The conflicting stage is one of three or
four, the skew costs an extra shift-and-add on *every* shared access including
the conflict-free ones, and at N = 4096 it pushes the shared allocation from
32 KB to 33.8 KB. Kept as a switch because the sign may differ on another
device — `fft!(...; skew = true)` — but the default is what measured here.

Caveat on reading those numbers: run-to-run spread on this benchmark is ~5-10%,
so the three losses are near the noise floor individually. What makes them worth
acting on is that they are consistent in sign across shapes and the win is not.
"""
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

How many transforms share a workgroup.

One transform per workgroup is `T = N ÷ 8` threads, and at small `N` that is a
fraction of a warp — `N = 64` gives **8**, so three lanes in four idle. Packing
transforms into the workgroup fixes the launch shape.

**The target is ONE warp, not two.** That is measured, and the first guess here
was `cld(64, T)`, which is wrong by a factor of two — it made `N = 256` slower.
Swept against a same-size copy, interleaved, every candidate a separately
compiled module:

    N=64   T=8    1:29.6%  2:44.5%  4:46.6%  8:42.3%  16:39.7%
    N=128  T=16   1:45.3%  2:57.9%  4:58.0%  8:57.3%  16:37.9%
    N=256  T=32   1:53.4%  2:51.5%  4:51.4%  8:30.5%  16:20.5%
    N=512  T=64   1:54.7%  2:52.2%  4:33.8%  8:22.4%
    N=1024 T=128  1:51.7%  2:32.9%  4:21.8%

The optimum is the smallest group reaching 32 threads and it falls off hard
above ~64 — a bigger group is more shared memory for the same work, and past one
warp there is nothing left to fill.

Bounded above by three more things: it must divide the batch (no partial group is
emitted), it must fit the workgroup limit, and it must fit shared memory — where
the budget asks for **half** the device maximum, because a workgroup claiming the
whole 48 KB is the only one resident on its SM.
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
function fft!(dst::LavaArray{ComplexF32}, src::LavaArray{ComplexF32};
              inverse::Bool = false, skew::Bool = FFT_SKEW,
              group::Union{Nothing,Int} = nothing)
    size(dst) == size(src) || throw(DimensionMismatch(
        "fft!: destination $(size(dst)) does not match source $(size(src))"))
    N = size(src, 1)
    nbatch = length(src) ÷ N
    T, lead = fftplan(N)
    backend = get_backend(src)
    ctx = vk_context(src)
    lim = workgroup_limit(ctx)
    T <= lim || throw(ArgumentError(
        "fft!: N=$N needs $T threads, above this device's limit of $lim. " *
        "A larger radix, or a multi-pass decomposition, is what VkFFT reaches " *
        "for here (`numAxisUploads > 1`); neither is implemented yet."))
    nb = group === nothing ? fftgroup(N, T, nbatch, lim, max_shared_memory(ctx)) : group
    kern = fft_kernel!(backend)
    kern(dst, src, Val(N), Val(lead), Val(inverse ? 1 : -1), Val(skew), Val(nb);
         ndrange = T * nb * (nbatch ÷ nb), workgroupsize = T * nb)
    return dst
end

"""
    fft(src; inverse = false) -> LavaArray

Allocating [`fft!`](@ref).
"""
fft(src::LavaArray{ComplexF32}; inverse::Bool = false, skew::Bool = FFT_SKEW) =
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
    rfft_post_kernel!(dst, Z, Val(N), Val(SIGN))

Split the half-length spectrum back into the real transform's `N ÷ 2 + 1` bins.

`Z = FFT_{N/2}(z)` holds the even and odd sub-spectra summed; separating them is
the conjugate-symmetric pair

    e[k] = (Z[k] + conj(Z[H-k])) / 2        the even-indexed samples' transform
    o[k] = (Z[k] - conj(Z[H-k])) / 2i       the odd-indexed samples'

and `X[k] = e[k] + W_N^k o[k]`. The `mod` on both indices is what makes `k = 0`
and `k = H` fall out without their own branch: at `k = 0` both reads hit `Z[0]`,
so `o[0]` is real and `X[0] = e[0] + o[0]` is the DC bin.
"""
@kernel cpu=false unsafe_indices=true function rfft_post_kernel!(
        dst, @Const(Z), ::Val{N}, ::Val{SIGN}) where {N,SIGN}
    H = N ÷ 2
    g = @index(Global, Linear) - 1
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
        dst[c * (H + 1) + k + 1] = ComplexF32(er + or * cs - oi * sn,
                                              ei + or * sn + oi * cs)
    end
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
function rfft!(dst::LavaArray{ComplexF32}, src::LavaArray{Float32};
               skew::Bool = FFT_SKEW)
    N = size(src, 1)
    iseven(N) || throw(ArgumentError("rfft!: length $N must be even"))
    H = N ÷ 2
    nbatch = length(src) ÷ N
    size(dst, 1) == H + 1 || throw(DimensionMismatch(
        "rfft!: destination first dimension is $(size(dst, 1)), expected $(H + 1)"))
    # The pack, for free: N reals ARE H complex in the interleave the
    # decomposition wants.
    z = reshape(reinterpret(ComplexF32, reshape(src, length(src))), H, nbatch)
    # `fftany!`, not `fft!`: H is 200 for Whisper's 400-point mel and 480 for
    # DeepFilterNet3's 960, neither a power of two.
    Z = fftany!(similar(z), z)
    backend = get_backend(src)
    rfft_post_kernel!(backend)(dst, Z, Val(N), Val(-1);
                               ndrange = (H + 1) * nbatch)
    return dst
end

"""
    rfft(src) -> LavaArray{ComplexF32}

Allocating [`rfft!`](@ref).
"""
function rfft(src::LavaArray{Float32}; skew::Bool = FFT_SKEW)
    N = size(src, 1)
    nbatch = length(src) ÷ N
    dst = similar(src, ComplexF32, (N ÷ 2 + 1, Base.tail(size(src))...))
    rfft!(reshape(dst, N ÷ 2 + 1, nbatch), src; skew)
    return dst
end


# ─────────────────────────────────────────────── mixed radix, non-powers of two
#
# The tuned kernel above is powers of two only, and **that covers one of the
# three audio models**. Measured from their own configs:
#
#     Whisper mel      n_fft = 400  = 2^4 * 5^2      NOT a power of two
#     DeepFilterNet3   n_fft = 960  = 2^6 * 3 * 5    NOT a power of two
#     Demucs htdemucs  n_fft = 4096 = 2^12           power of two
#
# Nor can the "one leading stage then radix 8" shape be stretched to reach them:
# `400 = L * 8^m` has no integer solution for a radix `L`, and `960 = 15 * 8^2`
# wants a leading 15, which is two stages and not one. They need a genuine
# **sequence** of radices, which is what VkFFT's `stageRadix[]` is.
#
# So there are two kernels, deliberately. The power-of-two one is tuned and
# measured (see the table at the top); this one is general and is not. Splitting
# them keeps the tuning from being diluted by a shape it was never measured on —
# and `fft!` dispatches on `count_ones(N) == 1`, so the fast path stays the fast
# path.
#
# The thread count is `T = N ÷ Rmax`. A stage of radix `r < Rmax` has `N ÷ r`
# butterflies, which is more than `T`, so each thread runs `cld(Rmax, r)` of them
# under a bounds guard. VkFFT does the same thing — the `PfIf_lt_start` around
# each stage in `vkFFT_RegisterBoost.h` is that guard.

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

"""
    fftmixed_kernel(RS, sign) -> kernel

Generate the transform for radix sequence `RS` and direction `sign`.

## Every constant is a literal, on purpose

The first version called a generic `fftbutterfly(::Val{R}, ...)` that built its
twiddles with `cospi`/`sinpi` and relied on inference to fold them. **It does not
fold.** The inner `for m in 1:R` is not unrolled — `R` being a type parameter
makes the loop *bound* a constant, not the loop — so `m` stays a runtime value
and `cospi` survives as a Float64 transcendental. On this backend that is not
merely slow, it is a compile error (`unsupported dynamic function invocation`),
followed by a page of cascading dynamic-dispatch failures that name `ntuple` and
`Val` and point nowhere near the cause.

So the DFT matrix is computed **here**, in Julia, at generation time, and spliced
in as literal `Float32`s. Nothing is left to fold, and zero coefficients are
dropped from the expression rather than multiplied by.

That also makes `sign` a generation parameter rather than a `Val`, since the
twiddles depend on it: two kernels per sequence, each compiled once and cached.

The stage structure follows the same rule. `@synchronize` must sit in **uniform**
control flow, and a stage of radix `r < Rmax` leaves some lanes with no butterfly
— so the barrier is lifted out of the `if ok` guard, and every index around it is
a literal.
"""
const FFT_MIXED_KERNELS = Dict{Tuple{Tuple,Int},Any}()

function fftmixed_kernel(RS::Tuple, sign::Int)
    get!(FFT_MIXED_KERNELS, (RS, sign)) do
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
            # ── ALL gathers before ANY scatter.
            #
            # With `reps > 1` the naive "gather/compute/scatter" per rep is
            # WRONG, and wrong in a way that passes small cases: rep 1 reads
            # slots rep 0 has already overwritten. At N = 96 stage 2 (r=4,
            # Ns=8), rep 0 at j=4 writes slot 12 and rep 1 at j=12 reads it.
            # Two-stage plans happened to have disjoint sets and passed, which
            # is exactly how this survived the first round of testing.
            #
            # Stockham is not in place: every read of a stage must precede every
            # write of that stage. So the reps are split into a gather+compute
            # phase, one barrier, then a scatter phase — each rep keeping its own
            # registers across it.
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
                        :(@synchronize),          # UNIFORM, between all reads and all writes
                        Expr(:block, scatters...)]
            push!(stages, quote
                @synchronize
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
        # A DETERMINISTIC name, not `gensym`. `frozen_key` hashes
        # `string(nameof(F))` (`runtime/frozen_cache.jl`), and a gensym carries a
        # per-session counter — `##fftmixed#277` one run, `#281` the next — so
        # every mixed kernel would miss the frozen cache and recompile on every
        # load. That is precisely the cost the Runner packages exist to remove.
        kname = Symbol("fftmixed_", join(RS, "_"), sign > 0 ? "_inv" : "_fwd")
        @eval begin
            @kernel cpu=false unsafe_indices=true function $kname(dst, @Const(src))
                sre = @localmem Float32 ($N,)
                sim = @localmem Float32 ($N,)
                tid = @index(Local, Linear) - 1
                blk = @index(Group, Linear) - 1
                base = blk * $N
                @inbounds begin
                    $(load...)
                    $(stages...)
                    @synchronize
                    $(store...)
                end
            end
            $kname
        end
    end
end


"""
    fftmixed!(dst, src, RS; inverse = false) -> dst

Batched transform over the radix sequence `RS`, for lengths the power-of-two
kernel cannot take. `prod(RS)` must equal `size(src, 1)`.

Correctness-first and deliberately untuned: no grouping, no skew, a generic
butterfly for radix 3 and 5. It exists because Whisper's mel (400) and
DeepFilterNet3 (960) cannot be computed at all without it, and tuning it before
either is wired up would be tuning against a guess.
"""
function fftmixed!(dst::LavaArray{ComplexF32}, src::LavaArray{ComplexF32},
                   RS::Tuple; inverse::Bool = false)
    N = size(src, 1)
    prod(RS) == N || throw(ArgumentError(
        "fftmixed!: radix sequence $RS multiplies to $(prod(RS)), not $N"))
    size(dst) == size(src) || throw(DimensionMismatch(
        "fftmixed!: destination $(size(dst)) does not match source $(size(src))"))
    nbatch = length(src) ÷ N
    T = N ÷ maximum(RS)
    backend = get_backend(src)
    lim = workgroup_limit(vk_context(src))
    T <= lim || throw(ArgumentError(
        "fftmixed!: N=$N needs $T threads, above this device's limit of $lim"))
    # BOTH calls need `invokelatest`, not just the launch. `fftmixed_kernel`
    # `@eval`s the kernel, so the method `@kernel` defines for it is newer than
    # this function's world — and `kern(backend)`, which CONSTRUCTS the
    # `KA.Kernel`, is one of those methods. Wrapping only the launch gets a
    # `MethodError: method too new to be called from this world context`, which
    # names the right problem in an easy place to misread.
    kern = fftmixed_kernel(RS, inverse ? 1 : -1)
    k = Base.invokelatest(kern, backend)
    Base.invokelatest(k, dst, src; ndrange = T * nbatch, workgroupsize = T)
    return dst
end

"""
    fftany!(dst, src; inverse = false) -> dst

Dispatch on the length: the tuned power-of-two kernel where it applies, the
general mixed-radix one otherwise. This is what a caller who does not control
`N` should use — `fft!` stays the fast path with a hard requirement, so nothing
silently falls off it.
"""
function fftany!(dst::LavaArray{ComplexF32}, src::LavaArray{ComplexF32};
                 inverse::Bool = false)
    N = size(src, 1)
    if count_ones(N) == 1 && N >= 8
        return fft!(dst, src; inverse)
    end
    RS = fftplan_mixed(N)
    RS === nothing && throw(ArgumentError(
        "fftany!: length $N does not factorise over radices (8,5,4,3,2); " *
        "Rader or Bluestein is what VkFFT reaches for here and neither is ported"))
    return fftmixed!(dst, src, RS; inverse)
end

"""
    FFT_PREGENERATED

Non-power-of-two transform lengths whose kernels are built when **Lava loads**
rather than on first use.

[`fftmixed_kernel`](@ref) is a real code generator — the butterfly for a radix
sequence is spliced together as literals, so it cannot be a `Val`-parameterised
method the way an ordinary sized kernel can. That leaves an `@eval` on the
runtime path, and an `@eval` on the runtime path **cannot be precompiled**: a
downstream package whose `@compile_workload` runs an iSTFT dies with

    Evaluation into the closed module `Lava` breaks incremental compilation

so the workload silently skips and every first call in a fresh process pays the
compile the Runner packages exist to remove. `KokoroRunner` found this: its
vocoder is an iSTFT at `n_fft = 20`, and nothing in its workload was ever frozen.

Generating them here moves the `@eval` to module load, where it is ordinary. The
lengths are the ones the shipped models transform:

  * **10, 20** — Kokoro's iSTFTNet vocoder (`n_fft = 20`, hop 5), forward and
    inverse. The 10 is the half-length complex transform the real-input trick
    uses.
  * **400** — Whisper's mel front end (`n_fft = 400`, hop 160), which plans as
    `(8, 5, 5, 2)`.

Powers of two are absent on purpose: `fftany!` sends those to the tuned `fft!`,
which is an ordinary method and needs nothing here.

A length not on this list still works — the runtime generator is the fallback —
it just cannot be frozen into a package image. Adding a model that transforms a
new length means adding it here, and the symptom if it is forgotten is a slow
first call rather than a wrong answer.
"""
const FFT_PREGENERATED = (10, 20, 400)

for n in FFT_PREGENERATED
    rs = fftplan_mixed(n)
    rs === nothing && error("FFT_PREGENERATED: $n does not factorise over the radices")
    for sign in (-1, 1)
        fftmixed_kernel(rs, sign)
    end
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
@kernel cpu=false unsafe_indices=true function stft_frames_kernel!(
        frames, @Const(x), @Const(window), ::Val{NFFT}, hop::Int, len::Int,
        ::Val{CENTER}) where {NFFT,CENTER}
    g = @index(Global, Linear) - 1
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
end

"""
    stft(x, nfft, hop, window; center = true) -> LavaArray{ComplexF32}

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
function stft(x::LavaArray{Float32}, nfft::Int, hop::Int,
              window::LavaArray{Float32}; center::Bool = true)
    length(window) == nfft || throw(DimensionMismatch(
        "stft: window has $(length(window)) samples, expected nfft = $nfft"))
    len = length(x)
    nframes = center ? len ÷ hop + 1 : (len - nfft) ÷ hop + 1
    nframes > 0 || throw(ArgumentError("stft: signal of $len samples is shorter than nfft = $nfft"))
    backend = get_backend(x)
    frames = similar(x, Float32, nfft, nframes)
    stft_frames_kernel!(backend)(frames, x, window, Val(nfft), hop, len, Val(center);
                                 ndrange = nfft * nframes)
    return rfft(frames)
end

"""
    hannwindow(backend, n; periodic = true) -> LavaArray{Float32}

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
