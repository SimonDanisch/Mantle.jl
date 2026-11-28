# Fused scaled-dot-product attention on `tensor_ops::matmul2d`.
#
# The three-pass form — scores, softmax, apply — has to MATERIALISE the `Lq x Lk` score
# matrix, and at SAM 2.1's 4096-token global block that is 256 MiB written by the first
# pass, read and rewritten by the second and read again by the third. A gigabyte of traffic
# for 38.7 GFLOP of arithmetic, which is why all three passes sit within a factor of two of
# this machine's memory bandwidth and nowhere near its compute.
#
# Fused, the score tile never leaves threadgroup memory: one threadgroup owns `BQ` queries
# of one (head, batch), walks the keys in tiles of `BK`, and keeps a running max, a running
# sum and an `E x BQ` Float32 output accumulator — the standard online-softmax recurrence.
# Both products are `matmul2d`, and the second one reads its right operand out of
# THREADGROUP memory, which the run helpers support per operand.
#
# Measured on an M5, fp16, E = 72, against the three declared passes it replaces (each
# timed inside the same recorded plan):
#
#   Lq = Lk = 4096, 8 heads     22.16 -> 12.99 ms   1.71x
#   Lq = Lk =  256, 128 windows  1.44 ->  1.02 ms   1.41x
#
# `BQ = 16, BK = 128` with four simdgroups is the best of 40 tilings tried, on BOTH shapes,
# so the tiling is not a per-shape choice here. Bigger `BQ` loses to its accumulator's
# threadgroup footprint (occupancy) and smaller `BK` to the score product's aspect ratio.

"""
    attention_kernel!(O, Q, K, V, olay, qlay, klay, vlay, H, scale, Lq, Lk,
                      Val(E), Val(BQ), Val(BK), Val(NSIMD))

`O = softmax(scale · QᵀK) V`; one threadgroup computes `BQ` queries of one (head, batch)
plane.

The first two axes of each operand hold its matrix and every trailing axis is flattened
into the grid's y — `Q`/`O` are `(E, Lq, batch...)` and `K`/`V` are `(E, Lk, batch...)`.
Each operand is read through its layout `(offset, lda, head stride, batch stride)`, in
elements; `H` is the head count that splits the grid's y into head and batch.
"""
function attention_kernel!(O::MtlDeviceArray, Q::MtlDeviceArray,
                           K::MtlDeviceArray, V::MtlDeviceArray,
                           olay::NTuple{4, Int32}, qlay::NTuple{4, Int32},
                           klay::NTuple{4, Int32}, vlay::NTuple{4, Int32},
                           H::UInt32, scale::Float32, Lq::UInt32, Lk::UInt32,
                           ::Val{E}, ::Val{BQ}, ::Val{BK},
                           ::Val{NSIMD}) where {E, BQ, BK, NSIMD}
    tgid = threadgroup_position_in_grid_3d()
    qb = (unsafe_trunc(Int32, tgid.x) - Int32(1)) * Int32(BQ)
    hb = unsafe_trunc(Int32, tgid.y) - Int32(1)
    # Each operand is `(offset, lda, head stride, batch stride)` in elements, so a PERMUTED
    # view is read where it lies instead of being copied first. SAM 2.1's qkv projection
    # leaves q/k/v as `(E, H, L, B)` and attention wants `(E, L, H, B)`: dense, that is three
    # 4.7 MiB transposes per op before the op can start. The dense case is this same code with
    # the packed strides, so there is one path, not two.
    hh = hb % unsafe_trunc(Int32, H)
    bb = hb ÷ unsafe_trunc(Int32, H)
    @inline plane(A, lay, L) = MtlDeviceArray(
        (E, Int(L)), pointer(A, Int(lay[1] + hh * lay[3] + bb * lay[4]) + 1))
    Qp = plane(Q, qlay, Lq)
    tK = MtlInlineTensor(plane(K, klay, Lk), (E, Int(Lk)), (1, Int(klay[2])))
    tV = MtlInlineTensor(plane(V, vlay, Lk), (E, Int(Lk)), (1, Int(vlay[2])))
    Op = plane(O, olay, Lq)
    olda = Int(olay[2])

    # The score tile in Float32 and the probabilities in the operand type, because the
    # apply product's right operand has to match `V`'s element type.
    Sf = MtlThreadGroupArray(Float32, (BQ, BK))
    Ph = MtlThreadGroupArray(eltype(V), (BQ, BK))
    Oa = MtlThreadGroupArray(Float32, (E, BQ))
    mv = MtlThreadGroupArray(Float32, (BQ,))   # running row max
    lv = MtlThreadGroupArray(Float32, (BQ,))   # running row sum
    cv = MtlThreadGroupArray(Float32, (BQ,))   # this tile's rescale factor
    # The query tile, staged ONCE. The score product is inside the key loop, so a device-memory
    # q tile is re-read `Lk/BK` times — 32 times at 4096 keys — and through a permuted view
    # those reads are `BQ` scattered runs rather than one block. Staged, the layout is paid
    # once: measured on SAM 2.1's encoder this is what keeps reading q/k/v where they lie
    # cheaper than transposing them first.
    Qs = MtlThreadGroupArray(eltype(Q), (E, BQ))

    mSf = view(MtlInlineTensor(Sf), (Int32(1), Int32(1)), (Int32(BQ), Int32(BK)))
    mPh = view(MtlInlineTensor(Ph), (Int32(1), Int32(1)), (Int32(BQ), Int32(BK)))
    mOa = view(MtlInlineTensor(Oa), (Int32(1), Int32(1)), (Int32(E), Int32(BQ)))
    mQs = view(MtlInlineTensor(Qs), (Int32(1), Int32(1)), (Int32(E), Int32(BQ)))

    # `S = Qᵀ K` for this tile, and `O += V Pᵀ`. The contraction lengths are the tile's own
    # (E and BK), so both descriptors are fully static.
    score = TensorOpsMatmul2D{matmul2d_descriptor(BQ, BK, E; transpose_left = true,
                                                  mode = matmul2d_multiply),
                              Int32(NSIMD)}()
    apply = TensorOpsMatmul2D{matmul2d_descriptor(E, BQ, BK; transpose_right = true,
                                                  mode = matmul2d_multiply_accumulate),
                              Int32(NSIMD)}()

    nthr = NSIMD * 32
    tid = Int(thread_index_in_threadgroup()) - 1
    for lin in tid:nthr:(E * BQ - 1)
        @inbounds Oa[lin % E + 1, lin ÷ E + 1] = 0.0f0
    end
    # Strided over the rows rather than one row per thread: `nthr` may be smaller than `BQ`,
    # and a row no thread owns keeps whatever threadgroup memory held, which comes out as a
    # NaN in the answer rather than as a launch that fails.
    for r in (tid + 1):nthr:BQ
        @inbounds mv[r] = -Inf32
        @inbounds lv[r] = 0.0f0
    end
    for lin in tid:nthr:(E * BQ - 1)
        i = lin % E; j = lin ÷ E
        @inbounds Qs[i + 1, j + 1] = Qp[i + 1 + Int(qlay[2]) * (Int(qb) + j)]
    end
    threadgroup_barrier(MemoryFlagThreadGroup)

    # Dynamic trip count: a compile-time constant one crashes Apple's back-end (see
    # `matmul2d_descriptor`).
    ntiles = unsafe_trunc(Int32, Lk ÷ UInt32(BK))
    for t in Int32(0):(ntiles - Int32(1))
        koff = t * Int32(BK)
        score(mQs, view(tK, (Int32(1), koff + Int32(1)), (Int32(E), Int32(BK))), mSf)
        threadgroup_barrier(MemoryFlagThreadGroup)
        # One thread per query row: the max over this tile, then the exponentials against
        # the running max. Splitting a row across threads was tried and is not faster — the
        # reduction's two extra barriers cost what the parallelism saves.
        for r in (tid + 1):nthr:BQ
            @inbounds mold = mv[r]
            mx = mold
            for j in 1:BK
                @inbounds mx = max(mx, Sf[r, j] * scale)
            end
            # `exp(-Inf - mx)` is 0, which is what the first tile needs.
            corr = exp(mold - mx)
            a = 0.0f0
            for j in 1:BK
                @inbounds e = exp(Sf[r, j] * scale - mx)
                @inbounds Ph[r, j] = eltype(V)(e)
                a += e
            end
            @inbounds mv[r] = mx
            @inbounds lv[r] = lv[r] * corr + a
            @inbounds cv[r] = corr
        end
        threadgroup_barrier(MemoryFlagThreadGroup)
        # The accumulator is rescaled to the new max BEFORE the product adds to it.
        for lin in tid:nthr:(E * BQ - 1)
            i = lin % E; j = lin ÷ E
            @inbounds Oa[i + 1, j + 1] *= cv[j + 1]
        end
        threadgroup_barrier(MemoryFlagThreadGroup)
        apply(view(tV, (Int32(1), koff + Int32(1)), (Int32(E), Int32(BK))), mPh, mOa)
        threadgroup_barrier(MemoryFlagThreadGroup)
    end

    for lin in tid:nthr:(E * BQ - 1)
        i = lin % E; j = lin ÷ E
        # `Op` carries its own leading dimension too, so the result can land straight in a
        # permuted destination.
        @inbounds Op[i + 1 + olda * (Int(qb) + j)] = eltype(O)(Oa[i + 1, j + 1] / lv[j + 1])
    end
    return
end

"""
    attentionlayout(x) -> (array, dims, (offset, lda, head stride, batch stride)) or nothing

One attention operand, dense or strided.

`x` is either an array — packed, offset zero — or a `(; res, dims, strides, offset)` view of
one, which is how a permuted operand arrives. Four dimensions `(E, L, H, B)` or three
`(E, L, HB)`, and `E` must be the contiguous axis: `matmul2d` reads the plane through a
tensor descriptor, which carries a leading dimension but not a gap between elements.
"""
function attentionlayout(x::NamedTuple)
    dims = x.dims; st = x.strides
    length(dims) == length(st) || return nothing
    first(st) == 1 || return nothing
    length(dims) == 3 && return (x.res, dims, (Int(x.offset), Int(st[2]), Int(st[3]), 0))
    length(dims) == 4 &&
        return (x.res, dims, (Int(x.offset), Int(st[2]), Int(st[3]), Int(st[4])))
    return nothing
end

function attentionlayout(x)
    d = size(x)
    length(d) == 3 && return (x, d, (0, d[1], d[1] * d[2], 0))
    length(d) == 4 && return (x, d, (0, d[1], d[1] * d[2], d[1] * d[2] * d[3]))
    return nothing
end

"""
    attention_kernel_config(dev, O, Q, K, V; scale, bq, bk, nsimd)
        -> (; kernel, args, ndrange, group) or nothing

The fused attention kernel for `O = softmax(scale · QᵀK) V`, one matrix per trailing-axis
batch entry, for a graph to dispatch; `nothing` when `dev` or these operands are not
covered.

`Q`/`O` are `(E, Lq, H, B)` and `K`/`V` are `(E, Lk, H, B)` — or three-dimensional with the
heads and the batch already flattened — all with the same element type; the answer is
written in `O`'s. Any operand may instead be a strided view (see [`attentionlayout`](@ref)),
which the kernel reads where it lies. The head width `E` is a compile-time parameter of the
kernel — it sizes the accumulator — so a new `E` compiles a new kernel.

`bq` and `bk` are the query and key tiles to try, first that divides the sequence wins:
`BQ = 16`, `BK = 128` and `nsimd = 4` simdgroups is the best of 40 combinations on both of
SAM 2.1's attention shapes; the shorter `BK`s are there so a 64- or 32-long key run still
fuses rather than falling back to three passes, and they were measured at 1.2-1.5x as well.
"""
function attention_kernel_config(dev::MTL.MTLDevice, O, Q, K, V; scale,
                                 bq = (16, 8), bk = (128, 64, 32), nsimd::Int = 4)
    tensorcapable(dev) || return nothing
    lo = attentionlayout(O); lq = attentionlayout(Q)
    lk = attentionlayout(K); lv = attentionlayout(V)
    (lo === nothing || lq === nothing || lk === nothing || lv === nothing) && return nothing
    Ores, Odims, olay = lo; Qres, Qdims, qlay = lq
    Kres, Kdims, klay = lk; Vres, Vdims, vlay = lv
    T = eltype(Qres)
    (eltype(Kres) === T && eltype(Vres) === T) || return nothing
    tensoreltypes(T, T, T) || return nothing
    # One batch shape for all four, because one grid index walks all of them.
    (Odims[3:end] == Qdims[3:end] == Kdims[3:end] == Vdims[3:end]) || return nothing

    E, Lq = Qdims[1], Qdims[2]
    Lk = Kdims[2]
    (Kdims[1] == E && Vdims[1] == E && Vdims[2] == Lk) || return nothing
    (Odims[1] == E && Odims[2] == Lq) || return nothing
    nheads = Qdims[3]
    nbatch = prod(Qdims[3:end])
    # `matmul2d` tiles in eights on every axis, and `E` is one of its contraction lengths.
    E % TENSOR_MULT == 0 || return nothing

    tq = tiledivisor(Lq, bq)
    tk = tiledivisor(Lk, bk)
    (tq == 0 || tk == 0) && return nothing
    # Sf + Ph + Oa + the three row vectors + Qs, against what a threadgroup gets.
    tgmem = tq * tk * sizeof(Float32) + tq * tk * sizeof(T) +
            E * tq * sizeof(Float32) + 3 * tq * sizeof(Float32) + E * tq * sizeof(T)
    tgmem <= TENSOR_TGMEM || return nothing

    # The default four simdgroups are 128 threads, under every Metal device's threadgroup
    # limit.
    threads = nsimd * 32
    # `Int` tile parameters, not `Int32`: the kernel builds `(E, Int(Lq))` shape tuples and a
    # mixed `Tuple{Int32, Int64}` matches no `MtlDeviceArray` constructor — which reaches
    # Mantle's access walk as "every path through it throws" rather than as a method error.
    args = (Ores, Qres, Kres, Vres,
            map(Int32, olay), map(Int32, qlay), map(Int32, klay), map(Int32, vlay),
            UInt32(nheads), Float32(scale), UInt32(Lq), UInt32(Lk),
            Val(Int(E)), Val(Int(tq)), Val(Int(tk)), Val(Int(nsimd)))
    return (; kernel = attention_kernel!, args,
            ndrange = ((Lq ÷ tq) * threads, nbatch, 1), group = (threads, 1, 1))
end
