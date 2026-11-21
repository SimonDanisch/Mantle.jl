# MPSGraph graphs for Mantle's library calls: a product with a fused row bias and
# activation, Apple's fused attention over operands that may be windows on a dense array,
# and 2-D convolution and its transpose with a fused channel bias and activation.
#
# Each call commits ONE MPS command buffer to the queue it is given (`encodegraph!`).
# Operands are bound with `MPSGraphTensorData(::MtlArray)`, which takes the array's offset
# in its buffer and packed rows, so a suballocated operand of any shape binds where it lies.

## graphs and their cache

"""
    LibraryGraph(graph, inputs, output)

A built graph: the placeholders a call binds, in order, and the tensor it writes.
"""
struct LibraryGraph
    graph::MPSGraph
    inputs::Vector{MPSGraphTensor}
    output::MPSGraphTensor
    # Concurrent encoding of one graph is not documented to be safe.
    lock::ReentrantLock
end

LibraryGraph(graph::MPSGraph, inputs::Vector{MPSGraphTensor}, output::MPSGraphTensor) =
    LibraryGraph(graph, inputs, output, ReentrantLock())

"""The graphs built so far, by key (`GemmKey`, `AttentionKey`, `Conv2DKey`)."""
struct GraphCache
    graphs::Dict{Any,LibraryGraph}
    lock::ReentrantLock
end

const GRAPHS = GraphCache(Dict{Any,LibraryGraph}(), ReentrantLock())

"""
    librarygraph(key) -> LibraryGraph

The graph `key` describes, built by `buildgraph(key)` on first use and kept. A graph names
no array and no device: MPSGraph compiles it for the device of the command buffer it is
encoded into, and keeps that compilation itself.
"""
librarygraph(key) = Base.@lock GRAPHS.lock get!(() -> buildgraph(key), GRAPHS.graphs, key)

"""
    tensordata(arr, dims = size(arr)) -> MPSGraphTensorData

`arr`'s elements as a packed `dims` array, from where `arr` starts in its buffer. `dims`
may cover fewer elements than `arr` has: the dense array a window lies in is bound as the
leading part of the array the window was declared on.
"""
function tensordata(arr::MtlArray, dims::Dims = size(arr))
    dims == size(arr) && return MPSGraphTensorData(arr)
    n = prod(dims)
    lead = n == length(arr) ? arr : view(vec(arr), 1:n)
    return MPSGraphTensorData(reshape(lead, dims))
end

"""
    encodegraph!(bq, g::LibraryGraph, output, inputs)

Run `g` in one MPS command buffer committed to `bq`, reading `inputs` (`MPSGraphTensorData`
in the order of `g.inputs`) and writing `output`.

The command buffer is made from `bq` itself rather than from its `MTLCommandQueue`, and
that conversion commits `bq`'s open batch first: the graph runs after the work encoded on
`bq` before this call and before the work encoded after it, by the queue's commit order.
That costs a command buffer for the graph and, when `bq` has work open, one for the batch.
"""
function encodegraph!(bq::BatchedCommandQueue, g::LibraryGraph,
                      output::MPSGraphTensorData, inputs::Tuple)
    length(inputs) == length(g.inputs) || throw(ArgumentError(
        "encodegraph!: the graph binds $(length(g.inputs)) inputs, got $(length(inputs))"))
    feeds = Dict{MPSGraphTensor,MPSGraphTensorData}(zip(g.inputs, inputs))
    results = Dict{MPSGraphTensor,MPSGraphTensorData}(g.output => output)
    # Committed under `global_queue!(bq)`, Metal.jl records `bq` as the queue using the
    # buffers bound above. Without it they are recorded for this task's own queue over
    # the same `MTLCommandQueue`, and the next launch on `bq` that uses one waits for it.
    Metal.global_queue!(bq) do
        cmdbuf = MPSCommandBuffer(bq)
        Base.@lock g.lock MPS.encode!(cmdbuf, g.graph, NSDictionary(feeds),
                                      NSDictionary(results), nil, default_exec_desc())
        MTL.commit!(cmdbuf)
    end
    return nothing
end

# Every check below is asked of graph resources as often as of arrays — a backend answers
# while declaring, before anything is placed — so none of them is more specific than
# `eltype`, `ndims` and `size`.
describable(xs...) =
    all(x -> applicable(ndims, x) && applicable(eltype, x) && applicable(size, x), xs)

# MPS aborts the process on an empty buffer.
nonempty(xs...) = all(x -> all(>(0), size(x)), xs)

## activations

"""The activations a graph puts on the store of a product or a convolution."""
const ACTIVATIONS = (:identity, :relu, :gelu, :gelu_tanh)

"""
    activation(graph, t, kind, T) -> MPSGraphTensor

One named activation as graph nodes, evaluated in `t`'s type `T`.

`:gelu` is the erf formulation — torch's default — and `:gelu_tanh` its `approximate
= "tanh"` variant. Apple's `erf` is a real one, so this is not bit-identical to a
hand-written Abramowitz-Stegun approximation of the same expression; it is closer to
the reference the models are checked against, which is the comparison that matters.
"""
function activation(graph::MPSGraph, t::MPSGraphTensor, kind::Symbol, T::DataType)
    kind === :identity && return t
    kind === :relu && return reLUWithTensor(graph, t)
    constant(v) = constantWithScalar(graph, v, T)
    if kind === :gelu
        # 0.5x (1 + erf(x / sqrt 2))
        inner = erfWithTensor(graph, multiplicationWithPrimaryTensor(
            graph, t, constant(0.7071067811865476), "gelu_scale"), "gelu_erf")
        s = additionWithPrimaryTensor(graph, inner, constant(1.0), "gelu_one")
        return multiplicationWithPrimaryTensor(
            graph, multiplicationWithPrimaryTensor(graph, t, constant(0.5), "gelu_halfx"),
            s, "gelu")
    end
    if kind === :gelu_tanh
        # 0.5x (1 + tanh(sqrt(2/pi) (x + 0.044715 x³)))
        x2 = multiplicationWithPrimaryTensor(graph, t, t, "gelu_x2")
        x3 = multiplicationWithPrimaryTensor(graph, x2, t, "gelu_x3")
        inner = additionWithPrimaryTensor(graph, t, multiplicationWithPrimaryTensor(
            graph, x3, constant(0.044715), "gelu_c3"), "gelu_inner")
        th = tanhWithTensor(graph, multiplicationWithPrimaryTensor(
            graph, inner, constant(0.7978845608028654), "gelu_sqrt2pi"), "gelu_tanh")
        s = additionWithPrimaryTensor(graph, th, constant(1.0), "gelu_one")
        return multiplicationWithPrimaryTensor(
            graph, multiplicationWithPrimaryTensor(graph, t, constant(0.5), "gelu_halfx"),
            s, "gelu")
    end
    throw(ArgumentError("activation: no graph node for :$kind"))
end

## product

"""What makes two product graphs the same graph."""
struct GemmKey
    size_a::Dims{2}
    size_b::Dims{2}
    Tab::DataType
    Tc::DataType
    # `nothing` for a product with no bias, otherwise the bias element type: the bias is
    # a placeholder of its own and its cast is part of the graph.
    Tbias::Union{DataType,Nothing}
    act::Symbol
end

function buildgraph(key::GemmKey)
    graph = MPSGraph()
    M = key.size_a[1]
    placeA = placeholderTensor(graph, key.size_a, key.Tab, "A")
    placeB = placeholderTensor(graph, key.size_b, key.Tab, "B")
    # Julia's `A * B` is MPSGraph's `B * A`: the shapes are reversed, so MPS sees
    # `(N, K) * (K, M) = (N, M)`, which is Julia's `(M, N)`.
    product = matrixMultiplicationWithPrimaryTensor(graph, placeB, placeA)
    # The EPILOGUE in Float32 where there is an activation, and in the operand type
    # where there is not. `gelu` of a value rounded to half loses accuracy twice —
    # inside `erf`, and again at the `1 + erf` that follows, where an argument below
    # -2 cancels to a couple of significant bits — which is why torch evaluates it in
    # `opmath_type` and why the native kernel accumulates in Float32 and rounds once.
    # Not applied to a plain product, whose structure is measured.
    wide = key.act !== :identity && key.Tab !== Float32
    epi = wide ? castTensor(graph, product, Float32, "wide") : product
    Te = wide ? Float32 : key.Tab
    inputs = [placeA, placeB]
    if key.Tbias !== nothing
        placeBias = placeholderTensor(graph, (M,), key.Tbias, "bias")
        push!(inputs, placeBias)
        # A length-`M` bias against MPS's `(N, M)`: the trailing axis matches and
        # MPS broadcasts the leading one, which is Julia's "one value per ROW".
        b = key.Tbias === Te ? placeBias : castTensor(graph, placeBias, Te, "castbias")
        epi = additionWithPrimaryTensor(graph, epi, b)
    end
    epi = activation(graph, epi, key.act, Te)
    result = key.Tc === Te ? epi : castTensor(graph, epi, key.Tc, "castC")
    return LibraryGraph(graph, inputs, result)
end

"""
    mpsgemm!(bq, C, A, B, bias = nothing, act = :identity) -> C

`C = act.(A * B .+ bias)` with MPSGraph, in one command buffer committed to `bq` (see
[`encodegraph!`](@ref)). `act` is one of `ACTIVATIONS`; [`supported`](@ref) says which
operands are covered. The graph is built once per shape, types and activation, and kept.
"""
function mpsgemm!(bq::BatchedCommandQueue, C::MtlMatrix, A::MtlMatrix, B::MtlMatrix,
                  bias = nothing, act::Symbol = :identity)
    if !supported(mpsgemm!, C, A, B, bias, act)
        b = bias === nothing ? "no bias" : "a bias of $(size(bias))"
        throw(ArgumentError(
            "mpsgemm!: MPSGraph does not cover $(size(A)) * $(size(B)) into $(size(C)), " *
            "$(eltype(A))/$(eltype(B))/$(eltype(C)), $b, activation :$act; ask " *
            "`supported(mpsgemm!, ...)` first."))
    end
    key = GemmKey(size(A), size(B), eltype(A), eltype(C),
                  bias === nothing ? nothing : eltype(bias), act)
    inputs = bias === nothing ? (tensordata(A), tensordata(B)) :
                                (tensordata(A), tensordata(B), tensordata(bias))
    encodegraph!(bq, librarygraph(key), tensordata(C), inputs)
    return C
end

"""
    supported(mpsgemm!, C, A, B, bias, act = :identity) -> Bool

Whether [`mpsgemm!`](@ref) covers these operands: matrices of one real float type, an
optional bias of one value per row, and an activation the graph has a node for.
"""
function supported(::typeof(mpsgemm!), C, A, B, bias, act::Symbol = :identity)
    act in ACTIVATIONS || return false
    describable(C, A, B) || return false
    ndims(C) == ndims(A) == ndims(B) == 2 || return false
    Tab = eltype(A)
    eltype(B) === Tab || return false
    # Real floats only. The graph feeds its placeholders straight into `mps.matmul`,
    # which has no integer node — "operand #0 must be tensor of floating point values" —
    # and MPSGraph answers that by failing its own module verification and calling
    # `abort()`, which takes the Julia process with it, with no exception to catch.
    (Tab === Float16 || Tab === Float32 || Tab === BFloat16) || return false
    # A destination WIDER than the operands is refused, and this is the one that
    # matters rather than a tidiness rule. `matrixMultiplicationWithPrimaryTensor`
    # has no compute type: its output dtype follows its inputs, so a half product
    # asked for in single is a half product cast afterwards, and the accumulator
    # the caller wanted is gone. A convolution's im2col GEMM asks for exactly that —
    # `(Float16, Float16) -> Float32`, on the grounds that the native kernel
    # accumulates directly into single — and taking it here moved SAM 2.1's encoder
    # `add_129` from 0.48 to 1.12, past the band the Vulkan reference sits in. The
    # only way to give MPSGraph a wide accumulator is to widen its operands, which
    # is four times the traffic for a product the caller already has a kernel for.
    eltype(C) === Tab || return false
    size(A, 2) == size(B, 1) && size(C) == (size(A, 1), size(B, 2)) || return false
    nonempty(C, A, B) || return false
    if bias !== nothing
        describable(bias) || return false
        ndims(bias) == 1 && length(bias) == size(C, 1) || return false
    end
    return true
end

## attention

# Apple's fused scaled dot-product attention, as one graph node.
#
# `scaledDotProductAttentionWithQueryTensor:` is QKᵀ → scale → softmax → ·V in one
# op, and on an M5 it is around 3.5x a hand-written fused kernel over the same
# operands (72-wide heads, 4096 keys, fp16).
#
# Julia's `(E, Lq, H, B)` is MPSGraph's `(B, H, Lq, E)` — the layout Apple's SDPA
# documents — so no transposition is needed anywhere; `placeholderTensor` reverses
# the shape it is given and the tensor data describes the same bytes innermost-first.
#
# A projection leaves q, k and v interleaved in one buffer — SAM 2.1's are three
# windows on the middle axis of one `(E, H, 3, L, B)` array — so the operand
# attention wants is a strided view and not an array. Materialising it costs a full
# copy of every operand of every attention op, which is most of what the library
# saves. It does not have to be copied: such a view is a DENSE array with axes the
# window does not span, so the array goes to MPSGraph whole and the narrowing happens
# in the graph — one `sliceTensor` per gap, a reshape to drop them, and the transposes
# that put the axes back in the caller's order. All of it is graph structure, built
# once per layout and cached with the graph.

"""
An operand that is a window on a dense array: which array, how it lies in memory,
and where the window starts.

`memdims` is the dense shape innermost-first, `tags` says which of the caller's axes
each memory axis is (`0` for a GAP the window does not span), `gapat` indexes the
gaps within `memdims`, and `starts` is where the window begins along each of them.
"""
struct PackedOperand{R}
    res::R
    dims::Tuple                 # the shape the caller asked for
    memdims::Tuple
    tags::Tuple
    gapat::Vector{Int}
    starts::Vector{Int}
end

"""Everything about an operand except its bytes, which is what a graph is built on."""
packedkey(p::PackedOperand) =
    (p.dims, p.memdims, p.tags, Tuple(p.gapat), Tuple(p.starts))

"""
    packoperand(x) -> PackedOperand or nothing

One operand, dense or strided, as the dense array it lives in.

`x` is either an array — the whole of it — or a `(; res, dims, strides, offset)` view
of one, which is the shape a strided operand arrives in (`offset` in elements).
`nothing` when the view is not a window on a dense array: a stride that is not a
multiple of what precedes it has no dense array to be a window on, and neither has
an offset that does not land on a gap.
"""
function packoperand(x::NamedTuple)
    dims = Tuple(Int.(x.dims)); strides = Tuple(Int.(x.strides)); offset = Int(x.offset)
    length(dims) == length(strides) || return nothing
    # An axis of one extent occupies nothing and has no meaningful stride, so it takes
    # no part in the layout; the reshape at the end puts it back.
    real = [i for i in eachindex(dims) if dims[i] != 1]
    isempty(real) && return nothing
    order = sort(real; by = i -> strides[i])
    memdims = Int[]; tags = Int[]; gapat = Int[]; gapstride = Int[]
    acc = 1
    for p in order
        s = strides[p]
        if s != acc
            (s > acc && s % acc == 0) || return nothing
            push!(memdims, s ÷ acc); push!(tags, 0)
            push!(gapat, length(memdims)); push!(gapstride, acc)
            acc = s
        end
        push!(memdims, dims[p]); push!(tags, p)
        acc *= dims[p]
    end
    # Where the window starts, one index per gap. Largest stride first, so the
    # decomposition is the only one there is.
    starts = zeros(Int, length(gapat))
    rest = offset
    for gi in length(gapat):-1:1
        idx, rest = divrem(rest, gapstride[gi])
        idx < memdims[gapat[gi]] || return nothing
        starts[gi] = idx
    end
    rest == 0 || return nothing
    # The dense array the window is on has to fit in the elements the resource owns.
    prod(memdims) <= length(x.res) || return nothing
    return PackedOperand(x.res, dims, Tuple(memdims), Tuple(tags), gapat, starts)
end

function packoperand(x)
    describable(x) || return nothing
    ndims(x) >= 1 || return nothing
    d = size(x)
    return PackedOperand(x, d, d, ntuple(identity, length(d)), Int[], Int[])
end

"""
Narrow and reorder a packed operand's placeholder into the tensor the caller asked
for.

Built from the KEY alone — no array — because it is graph structure and is cached
with the graph. A dense operand costs no node at all.
"""
function packedplaceholder(graph::MPSGraph, key, t::MPSGraphTensor, name)
    dims, memdims, tags, gapat, starts = key
    M = length(memdims)
    # One index along each gap. The axis is KEPT, with extent one, so the axis
    # numbers of the slices after it do not move.
    for (gi, at) in enumerate(gapat)
        t = sliceTensor(graph, t, mps_axis(M, at), starts[gi], 1, "$(name)_s$gi")
    end
    isempty(gapat) || (t = reshapeTensor(graph, t,
        convert(MPSShape, reverse([memdims[i] for i in 1:M if tags[i] != 0])),
        "$(name)_drop"))
    # What is left is the caller's axes, in MEMORY order. Selection sort into the
    # caller's order, one transpose per swap.
    ord = [tags[i] for i in 1:M if tags[i] != 0]
    R = length(ord)
    for i in 1:R
        j = i
        for l in (i + 1):R
            ord[l] < ord[j] && (j = l)
        end
        j == i && continue
        t = transposeTensor(graph, t, mps_axis(R, i), mps_axis(R, j), "$(name)_t$i")
        ord[i], ord[j] = ord[j], ord[i]
    end
    # …and the axes of one extent, which took no part in any of the above.
    R == length(dims) || (t = reshapeTensor(graph, t, convert(MPSShape, reverse(dims)),
                                            "$(name)_fill"))
    return t
end

"""
What makes two attention graphs the same graph: the operand layouts, the two types and
the scale. No array, and no address.

`eltyp` is what q, k and v are; `outtyp` is what the caller asked the answer to be.
They differ where a graph projects in fp16 and declares an fp32 result, which is what
Qwen-Image 2.1 exports.

Worth being exact about what the wider destination is and is not, because the
product's gate refuses the same pair for a reason that sounds like it applies here.
`scaledDotProductAttentionWithQueryTensor` has no compute type either: its result
dtype follows its INPUTS, so fp16 operands give an fp16-accurate answer and the cast
below widens it rather than accumulating in single. Measured against a Float64
reference at `(128, 256/384, 4, 1)`: fp16 operands read 7.06e-4 into an fp32
destination and 7.06e-4 into an fp16 one — the same number — where fp32 operands
read 9.08e-7.

It is admitted anyway, which `supported(mpsgemm!, ...)`'s pair is not, because of what
the caller does when it is refused. A refused product falls back to a kernel that
accumulates in single, so refusing keeps an accumulator. A refused ATTENTION falls
back to `DNNKernels`' `threepass!`, whose score matrix is `eltype(q)` — fp16, the
same precision this gives — so refusing keeps nothing and costs the whole op.
"""
struct AttentionKey
    q::Any
    k::Any
    v::Any
    eltyp::DataType
    outtyp::DataType
    scale::Float32
end

function buildgraph(key::AttentionKey)
    graph = MPSGraph()
    T = key.eltyp
    qph = placeholderTensor(graph, key.q[2], T, "q")
    kph = placeholderTensor(graph, key.k[2], T, "k")
    vph = placeholderTensor(graph, key.v[2], T, "v")
    out = scaledDotProductAttentionWithQueryTensor(
        graph,
        packedplaceholder(graph, key.q, qph, "q"),
        packedplaceholder(graph, key.k, kph, "k"),
        packedplaceholder(graph, key.v, vph, "v"),
        key.scale)
    key.outtyp === T || (out = castTensor(graph, out, key.outtyp, "castout"))
    return LibraryGraph(graph, [qph, kph, vph], out)
end

"""
    attentionoperands(o, q, k, v) -> (; keys, res) or nothing

Whether Apple's attention covers these operands as they lie, and if so everything about
them a graph is built from: one layout key and one array per operand.

Rank four throughout with the key and value runs agreeing, one element type across
q, k and v, a result that is fp16 or fp32 and need not match them, and a destination
that is an array rather than a view. `nothing` is how a caller learns to keep its own
kernel.
"""
function attentionoperands(o, q, k, v)
    po = packoperand(o)
    po === nothing && return nothing
    # The ANSWER is written, so it has to be an array and not a window on one.
    isempty(po.gapat) && po.dims == po.memdims || return nothing
    ps = map(packoperand, (q, k, v))
    any(isnothing, ps) && return nothing
    # MPS takes tensors of rank one to sixteen.
    all(p -> 1 <= length(p.memdims) <= 16, ps) || return nothing
    # The OPERAND type and the RESULT type are asked separately. They agree in most
    # graphs and the cast costs nothing there; where they do not, requiring them to
    # agree is what sent an fp16 attention with an fp32 result to the three-pass
    # path. The wider destination does NOT buy a wider accumulator — see
    # `AttentionKey` for the measurement, and for why that is worth taking here and
    # not in `supported(mpsgemm!, ...)`.
    Te = eltype(first(ps).res)
    To = eltype(o)
    (Te === Float16 || Te === Float32) || return nothing
    (To === Float16 || To === Float32) || return nothing
    all(p -> eltype(p.res) === Te, ps) || return nothing
    pq, pk, pv = ps
    length(po.dims) == length(pq.dims) == length(pk.dims) == length(pv.dims) == 4 ||
        return nothing
    pk.dims == pv.dims || return nothing
    po.dims == pq.dims || return nothing
    pq.dims[1] == pk.dims[1] || return nothing                     # the head width
    pq.dims[3] == pk.dims[3] && pq.dims[4] == pk.dims[4] || return nothing
    all(>(0), po.dims) && all(>(0), pk.dims) || return nothing
    return (; keys = map(packedkey, ps), res = map(p -> p.res, ps))
end

"""
    mpsattention!(bq, o, qres, kres, vres, qkey, kkey, vkey, scale) -> o

`o = softmax(scale · qᵀk) · v` per `(head, batch)` plane, Apple's fused op, in one command
buffer committed to `bq` (see [`encodegraph!`](@ref)).

The three keys come from [`attentionoperands`](@ref) and say how each operand lies; the
three arrays are the dense arrays those layouts are windows on, which is what makes
this callable with addresses fixed when a plan was recorded. The graph is built once per
layout, types and scale, and kept.
"""
function mpsattention!(bq::BatchedCommandQueue, o::MtlArray, qres::MtlArray,
                       kres::MtlArray, vres::MtlArray, qkey, kkey, vkey, scale::Real)
    key = AttentionKey(qkey, kkey, vkey, eltype(qres), eltype(o), Float32(scale))
    inputs = (tensordata(qres, qkey[2]), tensordata(kres, kkey[2]),
              tensordata(vres, vkey[2]))
    encodegraph!(bq, librarygraph(key), tensordata(o), inputs)
    return o
end

## convolution

# Apple's 2-D convolution, as one graph node.
#
# `convolution2DWithSourceTensor:weightsTensor:descriptor:` is a direct convolution:
# no im2col matrix to materialise, no padded reduction axis, no epilogue pass to sum
# split-K planes. Reaching for it is worth most of what a `7x7x3` stem costs — SAM
# 2.1's is 20.0 MiB of im2col written and read back before the product starts.
#
# The layouts line up without a transposition. A reversed-layout `(W, H, C, N)` is
# MPSGraph's `(N, C, H, W)`, which is its `NCHW`, and `(KW, KH, Cin, Cout)` is
# `(Cout, Cin, KH, KW)`, which is its `OIHW`. Both are what the descriptor is told.
#
# A TRANSPOSED convolution is the same node's data gradient, and needs no separate
# kernel: `conv_transpose2d(x, W)` is exactly the gradient with respect to the input
# of the forward convolution that would map the transpose's OUTPUT back to `x`. That
# forward convolution's weight in `OIHW` is `(Cin, Cout ÷ groups, KH, KW)`, which is
# what ATen already hands over -- reversed, `(KW, KH, Cout ÷ groups, Cin)` -- so the
# operands line up here too and only the channel test changes sides.
#
# It is worth the wiring. RIFE's decoder upsamples with seven `4x4` stride-2
# transposes, and on the implicit-GEMM kernel they ran at 0.03-0.04 TFLOP/s and were
# **75% of the entire frame** (438 ms of 581).

"""What makes two convolution graphs the same graph."""
struct Conv2DKey
    dims_x::NTuple{4,Int}
    dims_w::NTuple{4,Int}
    dims_o::NTuple{4,Int}
    Txw::DataType
    To::DataType
    Tbias::Union{DataType,Nothing}
    stride::NTuple{2,Int}
    pad::NTuple{2,Int}
    dilation::NTuple{2,Int}
    groups::Int
    act::Symbol
    transposed::Bool
end

function buildgraph(key::Conv2DKey)
    graph = MPSGraph()
    px = placeholderTensor(graph, key.dims_x, key.Txw, "x")
    pw = placeholderTensor(graph, key.dims_w, key.Txw, "w")
    # `paddingLeft`/`Right` are along X and `Top`/`Bottom` along Y, and ATen's
    # padding is symmetric in each.
    desc = MPSGraphConvolution2DOpDescriptor(
        key.stride, (key.pad[1], key.pad[1], key.pad[2], key.pad[2]),
        key.dilation, key.groups)
    # The descriptor is the FORWARD convolution's in both cases. For the transpose
    # that forward convolution runs the other way -- from this node's output shape to
    # its input -- and the same stride, padding, dilation and groups describe it.
    t = key.transposed ?
        convolution2DDataGradientWithIncomingGradientTensor(
            graph, px, pw, convert(MPSShape, reverse(collect(key.dims_o))), desc) :
        convolution2DWithSourceTensor(graph, px, pw, desc)
    # The EPILOGUE in Float32 whenever there is one at all, which is stricter than the
    # product. A convolution's other lowering accumulates into a Float32 scratch and
    # converts ONCE, in its epilogue pass — so adding the bias in half after MPS has
    # already rounded the product is a second rounding the caller did not have, and
    # SAM 2.1's stem has a bias and no activation.
    wide = (key.act !== :identity || key.Tbias !== nothing) && key.Txw !== Float32
    Te = wide ? Float32 : key.Txw
    wide && (t = castTensor(graph, t, Float32, "wide"))
    inputs = [px, pw]
    if key.Tbias !== nothing
        Cout = key.dims_o[3]
        place_bias = placeholderTensor(graph, (Cout,), key.Tbias, "bias")
        push!(inputs, place_bias)
        # One value per output CHANNEL. A `(Cout,)` tensor against MPS's
        # `(N, Cout, H, W)` would broadcast along W, so it is reshaped to the axis
        # it belongs on — `(1, 1, Cout, 1)` reversed, which is `(1, Cout, 1, 1)`.
        b = reshapeTensor(graph, place_bias, convert(MPSShape, [1, Cout, 1, 1]),
                          "biasnchw")
        key.Tbias === Te || (b = castTensor(graph, b, Te, "castbias"))
        t = additionWithPrimaryTensor(graph, t, b, "biasadd")
    end
    t = activation(graph, t, key.act, Te)
    key.To === Te || (t = castTensor(graph, t, key.To, "castout"))
    return LibraryGraph(graph, inputs, t)
end

"""
    mpsconv2d!(bq, out, x, w, bias, stride, pad, dilation, groups, act = :identity;
               transposed = false) -> out

`out = act.(conv(x, w) .+ bias)`, Apple's direct convolution — or with `transposed` its
data gradient, the transposed convolution — in one command buffer committed to `bq` (see
[`encodegraph!`](@ref)). Operands are in the reversed layout `supported(mpsconv2d!, ...)`
describes, `stride`, `pad` and `dilation` are `(x, y)` and padding is symmetric. The graph
is built once per shape, types, geometry and activation, and kept.
"""
function mpsconv2d!(bq::BatchedCommandQueue, out::MtlArray, x::MtlArray, w::MtlArray,
                    bias, stride::NTuple{2,Int}, pad::NTuple{2,Int},
                    dilation::NTuple{2,Int}, groups::Int, act::Symbol = :identity;
                    transposed::Bool = false)
    supported(mpsconv2d!, out, x, w, bias, act; transposed, groups) || throw(ArgumentError(
        "mpsconv2d!: MPSGraph does not cover $(size(x)) * $(size(w)) into " *
        "$(size(out)), $(eltype(x))/$(eltype(out)), groups $groups, activation :$act, " *
        "transposed $transposed; ask `supported(mpsconv2d!, ...)` first."))
    key = Conv2DKey(size(x), size(w), size(out), eltype(x), eltype(out),
                    bias === nothing ? nothing : eltype(bias),
                    stride, pad, dilation, groups, act, transposed)
    inputs = bias === nothing ? (tensordata(x), tensordata(w)) :
                                (tensordata(x), tensordata(w), tensordata(bias))
    encodegraph!(bq, librarygraph(key), tensordata(out), inputs)
    return out
end

"""
    supported(mpsconv2d!, out, x, w, bias, act = :identity; transposed = false, groups = 1)
        -> Bool

Whether Apple's convolution covers these operands as they lie.

`x` is `(W, H, Cin, N)`, `w` is `(KW, KH, Cin ÷ groups, Cout)` and `out` is
`(OW, OH, Cout, N)`; transposed, `w` is `(KW, KH, Cout ÷ groups, Cin)`. One element type
for the input and the weight, an activation the graph has a node for, and a destination
no wider than them — `convolution2DWithSource` has no compute type any more than the
product does, so a half convolution asked for in single is a half convolution cast
afterwards.
"""
function supported(::typeof(mpsconv2d!), out, x, w, bias, act::Symbol = :identity;
                   transposed::Bool = false, groups::Int = 1)
    act in ACTIVATIONS || return false
    describable(out, x, w) || return false
    ndims(out) == ndims(x) == ndims(w) == 4 || return false
    T = eltype(x)
    eltype(w) === T || return false
    (T === Float16 || T === Float32) || return false
    eltype(out) === T || return false
    # Which axis of the weight carries the OUTPUT channels is the one thing the
    # transpose changes: ATen's forward weight is `(KW, KH, Cin ÷ groups, Cout)` and
    # its transposed weight is `(KW, KH, Cout ÷ groups, Cin)`.
    if transposed
        size(x, 3) == size(w, 4) || return false
        size(out, 3) == size(w, 3) * groups || return false
    else
        size(x, 3) == size(w, 3) * groups || return false
        size(out, 3) == size(w, 4) || return false
    end
    size(out, 4) == size(x, 4) || return false
    nonempty(out, x, w) || return false
    if bias !== nothing
        describable(bias) || return false
        ndims(bias) == 1 && length(bias) == size(out, 3) || return false
    end
    return true
end
