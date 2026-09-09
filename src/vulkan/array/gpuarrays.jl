# Full GPUArrays.jl interface for LavaArray
#
# Provides broadcasting, scalar indexing, and the Adapt pipeline.
# Device-side representation: LavaDeviceArray{T,N} (isbits, BDA pointer + dims).

import GPUArrays
import Adapt

# ── Adapt.jl integration ──

# LavaAdaptor struct is declared in array/lavaarray.jl (needed earlier by
# ka_backend.jl method signatures).  This file defines the Adapt rules.
#
# The adaptor is the single place LavaArray → LavaDeviceArray (pointer strip)
# happens during kernel-arg conversion, so it is also the single place buffer-
# lifetime pinning fires.  Adapt.jl's recursion handles wrapper structs,
# Broadcasted, NamedTuple — every LavaArray anywhere inside lands here.

"""
    adapt_storage(::LavaAdaptor, ::LavaArray) → LavaDeviceArray

Pure strip: wrap the GPU buffer address in a device-visible struct. **No pin
side effect**. Callers must separately invoke `pin_leaves!(batch, args...)`
before submitting a command buffer that uses the stripped result — otherwise
the underlying `VkManagedBuffer` can be GC'd before the GPU reads from it.

(AMDGPU's adapt is also pure; they get away without an explicit pin pass
because `@roc` uses `GC.@preserve vars...` lexically around a synchronous
dispatch. Lava's dispatches are batched and submitted later, so `@preserve`
isn't enough — hence the explicit `pin_leaves!` step.)
"""
function Adapt.adapt_storage(::LavaAdaptor, a::LavaArray{T,N}) where {T,N}
    return LavaDeviceArray{T,N}(Ptr{T}(bda_address(a)), a.dims)
end
Adapt.adapt_storage(::LavaAdaptor, x) = x  # Scalars pass through

# Type-based adapt: Array → LavaArray (used by GPUArrays test suite's compare())
Adapt.adapt_storage(::Type{<:LavaArray}, xs::AT) where {AT<:AbstractArray} =
    isbitstype(AT) ? xs : convert(LavaArray, xs)
Adapt.adapt_storage(::Type{<:LavaArray{T}}, xs::AT) where {T, AT<:AbstractArray} =
    isbitstype(AT) ? xs : convert(LavaArray{T}, xs)
Adapt.adapt_storage(::Type{Array}, xs::LavaArray) = convert(Array, xs)

# ── GPU-safe RefValue (isbits replacement for mutable Base.RefValue) ──

"""LavaRefValue: isbits wrapper replacing Base.RefValue for GPU kernels."""
struct LavaRefValue{T} <: Ref{T}
    x::T
end
Base.getindex(r::LavaRefValue) = r.x

# Generic RefValue → LavaRefValue (isbits, GPU-safe)
Adapt.adapt_structure(to::LavaAdaptor, r::Base.RefValue) = LavaRefValue(Adapt.adapt(to, r[]))

# RefValue{Type/DataType} → zero-size type wrapper (types are not values on GPU)
struct LavaRefType{T} <: Ref{DataType} end
Base.getindex(::LavaRefType{T}) where {T} = T
Adapt.adapt_structure(::LavaAdaptor, r::Base.RefValue{<:Union{DataType, Type}}) =
    LavaRefType{r[]}()

# Broadcasted with Type as function (e.g. T.(args...)) → lambda wrapper
Adapt.adapt_structure(to::LavaAdaptor,
                      bc::Base.Broadcast.Broadcasted{Style, <:Any, Type{T}}) where {Style, T} =
    Base.Broadcast.Broadcasted{Style}((x...) -> T(x...), Adapt.adapt(to, bc.args), bc.axes)

# ── Broadcast style ──

struct LavaArrayStyle{N} <: GPUArrays.AbstractGPUArrayStyle{N} end

LavaArrayStyle{M}(::Val{N}) where {M,N} = LavaArrayStyle{N}()

Base.Broadcast.BroadcastStyle(::Type{<:LavaArray{T,N}}) where {T,N} = LavaArrayStyle{N}()

# Promote structured matrix styles (Diagonal, Tridiagonal, etc.) to LavaArrayStyle
# so that broadcasting with Diagonal{T, LavaArray{T,1}} runs on GPU.
Base.Broadcast.BroadcastStyle(::LavaArrayStyle{N}, ::LinearAlgebra.StructuredMatrixStyle{T}) where {N,T} = LavaArrayStyle{N}()
Base.Broadcast.BroadcastStyle(::LinearAlgebra.StructuredMatrixStyle{T}, ::LavaArrayStyle{N}) where {N,T} = LavaArrayStyle{N}()

# Wrapped GPU arrays (Transpose, Adjoint, SubArray, PermutedDimsArray,
# ReshapedArray) must use LavaArrayStyle. Without it the wrapper falls back to
# DefaultArrayStyle, which iterates and therefore hits the scalar-indexing
# guard — so `PermutedDimsArray(a, perm) .+ 1` threw rather than running.
#
# **Wrappers compose to arbitrary depth, so the check has to recurse.** Matching
# on a *direct* `LavaArray` parent, or on a union one level deep, leaves anything
# deeper on `DefaultArrayStyle`. That is not hypothetical: Kokoro's duration
# encoder transposes on nearly every line, the graph converter folds each permute
# into buffer metadata, and a layer norm arrived holding a **five-deep**
# `PermutedDimsArray`. It reduced to a `MethodError` about ambiguity, which names
# neither the depth nor the wrapper.

"""
    lavabacked(T) -> Bool

Whether an array type bottoms out in a `LavaArray` through any stack of view
wrappers. Recursive, and resolved at compile time for a concrete type.
"""
lavabacked(::Type{<:LavaArray}) = true
lavabacked(::Type{<:PermutedDimsArray{<:Any, <:Any, <:Any, <:Any, P}}) where {P} = lavabacked(P)
lavabacked(::Type{<:SubArray{<:Any, <:Any, P}}) where {P} = lavabacked(P)
lavabacked(::Type{<:Base.ReshapedArray{<:Any, <:Any, P}}) where {P} = lavabacked(P)
lavabacked(::Type{<:LinearAlgebra.Transpose{<:Any, P}}) where {P} = lavabacked(P)
lavabacked(::Type{<:LinearAlgebra.Adjoint{<:Any, P}}) where {P} = lavabacked(P)
lavabacked(::Type) = false

# Non-Lava wrappers get exactly the answer Base would have given, so these add a
# case rather than change one.
lavastyle(::Type{W}, ::Val{N}) where {W, N} =
    lavabacked(W) ? LavaArrayStyle{N}() : Base.Broadcast.DefaultArrayStyle{N}()

Base.Broadcast.BroadcastStyle(::Type{W}) where {T, N, W <: PermutedDimsArray{T, N}} =
    lavastyle(W, Val(N))
Base.Broadcast.BroadcastStyle(::Type{W}) where {T, N, W <: SubArray{T, N}} =
    lavastyle(W, Val(N))
Base.Broadcast.BroadcastStyle(::Type{W}) where {T, N, W <: Base.ReshapedArray{T, N}} =
    lavastyle(W, Val(N))
Base.Broadcast.BroadcastStyle(::Type{W}) where {T, W <: LinearAlgebra.Transpose{T}} =
    lavastyle(W, Val(2))
Base.Broadcast.BroadcastStyle(::Type{W}) where {T, W <: LinearAlgebra.Adjoint{T}} =
    lavastyle(W, Val(2))

"""
    lavaroot(x) -> LavaArray

The device array underneath any stack of view wrappers.
"""
lavaroot(x::LavaArray) = x
lavaroot(x::Union{PermutedDimsArray, SubArray, Base.ReshapedArray,
                  LinearAlgebra.Transpose, LinearAlgebra.Adjoint}) = lavaroot(parent(x))

"""
    dense(x) -> LavaArray

Materialise a wrapped device array into a dense one, on the device.

**`Array(x)` does not work on a deeply wrapped device array, and this is why.**
`Base.copyto!(::Array, ::PermutedDimsArray)` reaches `copyto_unaliased!`, which
*iterates* the source — the scalar-indexing guard fires, and the message blames
"an iterating implementation of a method" without naming the wrapper. GPUArrays
covers `AbstractGPUArray`, and a `PermutedDimsArray` over a `PermutedDimsArray`
over a `LavaArray` is not one.

Broadcast, on the other hand, works at any depth (see `lavabacked`), so
`y .= x` is the escape hatch and this wraps it. `Array(dense(x))` is the
supported way to bring a graph output home; a non-wrapped array passes through
untouched.
"""
dense(x::LavaArray) = x
function dense(x::AbstractArray)
    lavabacked(typeof(x)) || return x
    y = similar(lavaroot(x), eltype(x), size(x))
    y .= x                      # broadcast, NOT copyto! — copyto! iterates too
    return y
end

const AnyLavaArray{T} = Union{LavaArray{T},
                              SubArray{T, <:Any, <:LavaArray},
                              PermutedDimsArray{T, <:Any, <:Any, <:Any, <:LavaArray},
                              Base.ReshapedArray{T, <:Any, <:LavaArray},
                              LinearAlgebra.Transpose{T, <:LavaArray},
                              LinearAlgebra.Adjoint{T, <:LavaArray}}

"""
Reducing a `PermutedDimsArray` into a device array is **ambiguous** without this.

`Base.PermutedDimsArrays` specialises `mapreducedim!` on the *source* being a
`PermutedDimsArray`, GPUArrays specialises it on the *destination* being a GPU
array, and neither is more specific than the other:

    mapreducedim!(f, op, R::AnyGPUArray, A::AbstractArray)              # GPUArrays
    mapreducedim!(f, op, B::AbstractArray, A::PermutedDimsArray)        # Base

So `sum(permuted; dims)` throws `MethodError: ... is ambiguous` rather than
running. Fixing both argument positions at once is strictly more specific than
either candidate, which is what resolves it; the body is GPUArrays' — Base's
would iterate and trip the scalar-indexing guard.

Found on Kokoro, whose duration encoder transposes on nearly every line: the
graph converter folds each permute into buffer metadata, they nest, and the
layer norm that reduces the result arrived holding a **five-deep**
`PermutedDimsArray`. `Base.tail`-style unwrapping is not needed — the parent
only has to bottom out in something GPUArrays can reduce.
"""
Base.mapreducedim!(f, op, R::LavaArray, A::PermutedDimsArray) =
    GPUArrays.mapreducedim!(f, op, R, A)

# ...and the same again, matching Base's *exact* constraints plus the device
# destination. Specialising only on `R` is NOT enough: Base's method also pins
# `op` to a union of seven reducers and requires `ndims(R) == ndims(A)`, so it
# is more specific in two places where the method above is more specific in one,
# and neither wins. This one is more specific in all three, which is what
# actually settles it — `sum(::PermutedDimsArray; dims)` reaches here.
const REDUCERS = Union{typeof(&), typeof(+), typeof(Base._extrema_rf),
                       typeof(Base.add_sum), typeof(max), typeof(min), typeof(|)}
Base.mapreducedim!(f, op::REDUCERS, R::LavaArray{T, N},
                   A::PermutedDimsArray{S, N, perm, iperm}) where {T, S, N, perm, iperm} =
    GPUArrays.mapreducedim!(f, op, R, A)

# ── broadcast: always launch over a flat index space ──
#
# GPUArrays' `_copyto!` launches with `ndrange = size(dest)`, so an N-D
# destination gets an N-D index space and KernelAbstractions partitions it into
# N-D workgroups — at which point consecutive lanes no longer walk consecutive
# memory. The cost is not subtle. `D .= A .+ B` over 16.7M Float16:
#
#   dest 1-D          315 GB/s     (CUDA.jl: 325)
#   dest (512,512,64)  54 GB/s     (CUDA.jl: 204)
#
# Same kernel, same data, same card — only the launch geometry differs. CUDA.jl
# never sees this because it overrides broadcast with its own flat grid-stride
# loop rather than going through the KA path.
#
# So: flatten the range, and recover the Cartesian index inside the kernel when
# the broadcast genuinely needs one. That div/mod chain is far cheaper than
# losing coalescing. This is the single largest thing in Lava for anything
# elementwise — `add`, `relu`, `sigmoid`, `_to_copy`, `cat` and every kernel
# epilogue in DNNKernels were all running at a quarter of the achievable bandwidth.
# Each thread handles `U` elements, `WG` apart, which is `contig_copy.comp`'s
# shape from `reference/llama.cpp-vulkan` (128 threads x 4, "num_threads *
# num_iter must equal 512"). The stride is what keeps it coalesced: iteration `u`
# has the whole workgroup on one contiguous run, so all `U` runs are wide loads,
# not a per-thread gather of `U` neighbours.
#
# It buys memory-level parallelism, not fewer instructions — one thread with 8
# loads in flight hides latency a thread with 1 cannot. Measured on the encoder's
# permuted copies (interleaved, 2265 MHz, bit-exact), against the same kernel at
# one element per thread:
#
#     (72,8,256,16)      104 -> 114 GB/s
#     (72,4,16,1024)     113 -> 123
#     (576,16,4,16,4,1)   73 ->  76
#     (256,256,144,1)     45 -> 125
#
# Raising the *workgroup* instead does nothing (64 and 256 measure the same); the
# ceiling is the device's `maxComputeWorkGroupInvocations`, see `DeviceCaps`.
# (The "above 256 is unsafe on this driver" that used to be written here was the
# pipeline-cache hash collision, not the device — see `workgroup_limit`.)
@kernel cpu=false function lava_broadcast_flat!(dest, bc, n, ::Val{U}, ::Val{WG}) where {U, WG}
    l = @index(Local, Linear)
    g = @index(Group, Linear)
    base = (g - 1) * (WG * U) + l
    @inbounds for u in 0:(U - 1)
        I = base + u * WG
        if I <= n
            dest[I] = bc[I]
        end
    end
end

"""
    broadcastlaunch(n; wg, maxunroll, mingroups) -> (workgroupsize, Val(U), ndrange)

Launch geometry for a flat broadcast over `n` elements.

`maxunroll` is elements per thread; 1 restores one-element-per-thread. It was a
module-level `Ref` so it could be flipped **inside one session**, because that is
the only comparison this project accepts: an isolated benchmark showed 5-11% here
and a dispatch-timing profile showed the elementwise bucket dropping 18 ms, while
the free-running encode did not move at all across sessions. An argument does
that without making the setting process-wide. Each new value recompiles the
broadcast kernels, so warm up after each flip before timing.

`U` is backed off while the grid would be too small to occupy the device — 8
elements per thread over a 4096-element array is 2 workgroups, and a kernel that
cannot fill the card is slow however well it reads memory. `mingroups` is a few
per SM (48 here), so the back-off only triggers on arrays small enough that the
whole launch is noise anyway.

`ndrange` is rounded up to a whole number of workgroups; the kernels bound every
access with `n` themselves, so the tail threads simply do nothing.
"""
@inline function broadcastlaunch(n::Int; wg::Int = 256, maxunroll::Int = 8,
                                 mingroups::Int = 192)
    u = maxunroll
    while u > 1 && cld(n, wg * u) < mingroups
        u >>= 1
    end
    return wg, Val(u), wg * cld(n, wg * u)
end

# ── division by a runtime-constant extent, without dividing ──────────────────
#
# **The division chain is the cost of these kernels, which this document said it
# was not.** Isolated, with no memory traffic in the way — one kernel that only
# writes, one that also runs `cart32` — over 2.36 M elements:
#
#     rank 2, 1 division      11.7 -> 33.5 us
#     rank 4, 3 divisions     11.7 -> 59.6 us
#     rank 6, 5 divisions     10.9 -> 85.0 us
#
# Linear in the number of divisions, ~15 us each, and at rank 6 that is **74 us
# against a ~42 us memory floor** for the same array — the arithmetic outweighs
# the traffic. The earlier note (kept below) concluded the opposite from an
# end-to-end delta of -0.5 ms; it was measuring a kernel where the access pattern
# happened to hide it.
#
# `reference/llama.cpp-vulkan`'s `generic_unary_head.glsl` solves exactly this,
# and its `copy.comp` is otherwise the same algorithm as ours: it never divides,
# it multiplies by a magic number. `init_fastdiv_values` on the host, `fastdiv`
# in the shader.
struct FastDiv32
    d::UInt32
    mp::UInt32
    L::UInt32
end

"""
    FastDiv32(d)

Magic number and shift for unsigned division by `d`, so the kernel can divide
with a high multiply and a shift instead of the real thing.

Straight from `init_fastdiv_values` in llama.cpp's `ggml-vulkan.cpp`:
`L = ceil(log2(d))`, `mp = 2^32 * (2^L - d) / d + 1`, and then
`n / d == (mulhi(n, mp) + n) >> L`. Verified exact against `÷` for every extent
SAM 2's encoder decomposes by, and exhaustively over 0:20e6 for the two hottest.
"""
function FastDiv32(d::Integer)
    du = UInt32(d)
    du == 0 && throw(ArgumentError("FastDiv32: divisor must be positive"))
    L = 0x00000000
    while L < 0x20 && (UInt32(1) << L) < du
        L += 0x1
    end
    mp = UInt32(((UInt64(1) << 32) * ((UInt64(1) << L) - UInt64(du)) ÷ UInt64(du)) + 1)
    return FastDiv32(du, mp, L)
end

"""`n ÷ f.d`, as a 32-bit high multiply and a shift."""
@inline function fastdiv(n::UInt32, f::FastDiv32)
    # The widening multiply is the point: NVIDIA has `mul.hi.u32` and no integer
    # divide at all, so this is ~5 cycles where the divide was ~25.
    return (UInt32((UInt64(n) * UInt64(f.mp)) >> 32) + n) >> f.L
end

"""
    broadcastextents(sz; fastdiv = false)

Whether the broadcast kernels decompose the linear index by multiplying
([`FastDiv32`](@ref)) or by dividing. `false` restores the divisions.

Here for the same reason as `broadcastlaunch`'s `maxunroll`: so the claim can be
re-measured **inside one session** rather than against a number from another day.
Two cross-session readings of the unroll disagreed by 38 ms in opposite
directions, and both were noise.
"""
const BROADCAST_FASTDIV_DEFAULT = true

"Extents in the form `cart32` should decompose them with."
@inline broadcastextents(sz::Dims; fastdiv::Bool = BROADCAST_FASTDIV_DEFAULT) =
    fastdiv ? map(FastDiv32, sz) : map(Int32, sz)

# The kernels hand `cart32` an unsigned position; the dividing path still wants
# the signed extents it was written for.
@inline cart32(q::UInt32, sz::Tuple{Int32, Vararg{Int32}}) = cart32(Int32(q), sz)

@inline cart32(q::UInt32, ::Tuple{}) = ()
@inline function cart32(q::UInt32, sz::Tuple{FastDiv32, Vararg{FastDiv32}})
    f = first(sz)
    hi = fastdiv(q, f)
    # One multiply recovers the remainder, so an axis costs a mulhi, a mul and
    # two adds. The last axis's `hi` is dead and the compiler drops it.
    return (Int(q - hi * f.d) + 1, cart32(hi, Base.tail(sz))...)
end

# Narrowing the linear->Cartesian `divrem` to 32 bits here was tried and
# reverted. It is the same trick that took `im2col_kernel!` (four divisions per
# element) from 35.7 ms to 32.7 on a whole step, but handing these kernels a
# `CartesianIndices` with `Int32` axes silently produced wrong results — a view
# operand off by 252, a permuted operand off by 18 — and widening the recovered
# index back to `Int` immediately afterwards did not fix it, so the fault is in
# the narrowed `CartesianIndices` itself rather than in what `Broadcasted` does
# with the index type. Worth revisiting with a hand-rolled 32-bit decomposition
# instead of `CartesianIndices`, but not without these four cases as a test.
"""
Cartesian index for 0-based linear position `q`, given the extents as `Int32`.

Hand-rolled rather than `CartesianIndices[...]`: the divisions happen in 32-bit,
which NVIDIA has hardware for and 64-bit it does not, but the components are
widened to `Int` as they are built. Handing a `CartesianIndices` with `Int32`
axes to these kernels instead is *miscompiled* by Lava — see
`test/test_int32_cartesian_miscompile.jl` — while this form is correct.

The recursion is over the tuple's type, so it unrolls completely; SPIR-V's ban
on recursive call graphs does not apply.
"""
@inline cart32(q::Int32, ::Tuple{}) = ()
@inline function cart32(q::Int32, sz::Tuple{Int32,Vararg{Int32}})
    s = first(sz)
    (Int(q % s) + 1, cart32(q ÷ s, Base.tail(sz))...)
end

@kernel cpu=false function lava_broadcast_flat_cartesian!(dest, bc, sz, n,
                                                          ::Val{U}, ::Val{WG}) where {U, WG}
    l = @index(Local, Linear)
    g = @index(Group, Linear)
    base = (g - 1) * (WG * U) + l
    @inbounds for u in 0:(U - 1)
        I = base + u * WG
        if I <= n
            J = CartesianIndex(cart32(UInt32(I) - UInt32(1), sz))
            dest[J] = bc[J]
        end
    end
end

"""Destination dense, source not: index `dest` linearly and only `bc` by index."""
@kernel cpu=false function lava_broadcast_flat_mixed!(dest, bc, sz, n,
                                                      ::Val{U}, ::Val{WG}) where {U, WG}
    l = @index(Local, Linear)
    g = @index(Group, Linear)
    base = (g - 1) * (WG * U) + l
    @inbounds for u in 0:(U - 1)
        I = base + u * WG
        if I <= n
            dest[I] = bc[CartesianIndex(cart32(UInt32(I) - UInt32(1), sz))]
        end
    end
end

# ── the permuted-copy path is bandwidth-bound, not index-bound ───────────────
#
# `dest .= PermutedDimsArray(a, perm)` is **203 of 203** dispatches on this path
# for SAM 2's encoder — 40.1 ms, the third-largest kernel family, moving ~2.4 GB
# at 60 GB/s where a plain copy of the same bytes runs at 150-270. In 178 of
# them `perm[1] == 1` (Hiera's window partition `(1,2,4,3,5,6)`, attention's
# head/token swap `(1,3,2,4)`), so the copy is a permutation of whole contiguous
# rows and the general kernel's per-element `cart32` decomposition plus
# `PermutedDimsArray` re-indexing looks like pure overhead.
#
# It is not. A kernel that decomposes only the *row* index and adds the column —
# written, bit-exact on every shape the encoder uses, and verified firing on 90
# dispatches per encode — is worth **-0.5 ms end to end**, which is noise. Both
# kernels touch memory in the same order, and that order is what the time is.
#
# Recorded rather than kept: the addressing is not the cost, so anything that
# helps has to change the access pattern — a tiled staging transpose, as
# `DNNKernels`' `toLE_tiled` does — rather than the index arithmetic.
# `ctx.diag.broadcast_probe` below is what identified the shapes.


"""
Can every operand be read with the destination's own linear index?

`IndexStyle(::Broadcasted)` is too conservative for the case that dominates a
DNN — every operand the same dense shape as the destination, no extrusion — and
answering it directly is what gets those onto the linear kernel. Operands are
inspected before `preprocess` wraps them in `Extruded`; the `Extruded` method is
there for when one is passed anyway.
"""
flatok(dest, x) = true                                  # scalars, refs, functions
flatok(dest, x::AbstractArray) =
    size(x) == size(dest) && IndexStyle(x) === IndexLinear()
# A Tuple is a broadcast *container* with its own axes, not a scalar, so it must
# not fall through to the `true` above. `flat1` reshapes array leaves to vectors
# of `length(dest)` but leaves a tuple at its own length, so flattening a tree
# containing one produces `OneTo(length(dest))` against `OneTo(length(tuple))`:
#
#   broadcast!(f, out, arr, (a, b, c))   # out 3x10
#   DimensionMismatch: a has axes OneTo(30) and b has axes OneTo(3)
#
# Tuples are never the destination's shape in any useful case here, so the flat
# path simply does not apply; the general paths below handle them correctly.
flatok(dest, x::Tuple) = false
flatok(dest, x::Broadcast.Extruded) = flatok(dest, x.x)
flatok(dest, x::Broadcast.Broadcasted) = all(a -> flatok(dest, a), x.args)

"""
Rebuild a broadcast tree over 1-D reshapes of its operands.

Launching over a flat range is not enough on its own: `getindex(::Broadcasted,
::Integer)` on an N-D tree is defined as `bc[CartesianIndices(bc)[i]]`, so the
div/mod chain comes back per element even inside a linear kernel. Reshaping the
leaves to vectors makes the tree genuinely 1-D and the index arithmetic
disappears. Only valid when every leaf already has the destination's shape,
which is what `flatok` establishes.
"""
flat1(x) = x
flat1(x::AbstractArray) = reshape(x, length(x))
flat1(bc::Broadcast.Broadcasted) = Broadcast.broadcasted(bc.f, map(flat1, bc.args)...)

"""
    ctx.diag.broadcast_probe :: Union{Nothing,Dict}

Set to a dict to record which broadcast kernel each `.=` takes, keyed by
`(path, dest size, leaf sizes)`. Off by default and free when off.

The three paths differ by roughly 3x in achieved bandwidth and which one a
broadcast lands on is decided by operand *shapes*, invisibly. On SAM 2's encoder
`lava_broadcast_flat_mixed!` — the one that pays a `cart32` division chain per
element — is 203 dispatches and 40 ms, the third-largest kernel family, and this
is how to find out what is feeding it.
"""

leafsizes(x) = Any[]
leafsizes(x::AbstractArray) = Any[(size(x), nameof(typeof(x)), IndexStyle(x) === IndexLinear())]
leafsizes(x::PermutedDimsArray{T,N,perm}) where {T,N,perm} = Any[(size(parent(x)), :Permuted, perm)]
leafsizes(x::Broadcast.Extruded) = leafsizes(x.x)
leafsizes(bc::Broadcast.Broadcasted) = reduce(vcat, map(leafsizes, bc.args); init = Any[])

function probe_broadcast!(path, dest, bc)
    # `dest` is a LavaArray, so the device that owns this broadcast is reachable
    # without asking which one happens to be global.
    p = vk_context(dest).diag.broadcast_probe
    p === nothing && return
    key = (path, size(dest), Tuple(unique(leafsizes(bc))))
    p[key] = get(p, key, 0) + 1
    return
end

# `fastdiv` is optional and defaulted, so GPUArrays' own two-argument call is
# unchanged — but a test can drive both index paths without a global. It was
# `BROADCAST_FASTDIV[]`, set and restored around the comparison.
function GPUArrays._copyto!(dest::AnyLavaArray, bc::Broadcast.Broadcasted;
                            fastdiv::Bool = BROADCAST_FASTDIV_DEFAULT)
    axes(dest) == axes(bc) || Broadcast.throwdm(axes(dest), axes(bc))
    isempty(dest) && return dest
    n = length(dest)
    # `KA.get_backend(dest)`, NOT `LavaBackend()`. An unpinned backend resolves
    # its queue through `vk_context()`, so on a second device this dispatches on
    # whichever context happens to be global — the work lands on the wrong GPU
    # and the buffer's own device never sees it. `get_backend` derives the
    # context from the array's buffer, which has always carried it.
    backend = KernelAbstractions.get_backend(dest)
    wg, u, nd = broadcastlaunch(n)
    if ndims(dest) > 1 && IndexStyle(dest) === IndexLinear() && flatok(dest, bc)
        probe_broadcast!(:flat, dest, bc)
        flat = Broadcast.instantiate(flat1(bc))
        d1 = reshape(dest, n)
        lava_broadcast_flat!(backend)(d1, Broadcast.preprocess(d1, flat), n, u, Val(wg);
                                      ndrange = nd, workgroupsize = wg)
        return dest
    end
    linear = ndims(dest) == 1
    bc = Broadcast.preprocess(dest, bc)
    # Extents as magic numbers, so `cart32` multiplies instead of dividing. Built
    # per launch on the host: three `UInt32` per axis and one `÷` each, against
    # the ~15 us per axis a real division costs across a 2.4 M-element dispatch.
    sz = broadcastextents(size(dest); fastdiv)
    if linear
        probe_broadcast!(:linear1d, dest, bc)
        lava_broadcast_flat!(backend)(dest, bc, n, u, Val(wg);
                                      ndrange = nd, workgroupsize = wg)
    elseif IndexStyle(dest) === IndexLinear()
        probe_broadcast!(:mixed, dest, bc)
        lava_broadcast_flat_mixed!(backend)(dest, bc, sz, n, u, Val(wg);
                                            ndrange = nd, workgroupsize = wg)
    else
        probe_broadcast!(:cartesian, dest, bc)
        lava_broadcast_flat_cartesian!(backend)(dest, bc, sz, n, u, Val(wg);
                                                ndrange = nd, workgroupsize = wg)
    end
    return dest
end

# GPUArrays implements `repeat` for `AnyGPUArray` (host/base.jl `repeat_inner`
# / `repeat_outer`), but `AnyGPUArray` only recognises a *single* layer of
# wrapping, so a `ReshapedArray{PermutedDimsArray{LavaArray}}` misses it and
# falls back to `Base._RepeatInnerOuter.repeat_outer`, which indexes
# elementwise and trips the scalar-indexing guard. Detaching the wrapper first
# gets it back onto the device kernels; `repeat` allocates its output anyway, so
# the copy costs nothing that was not already being paid.
const NestedLavaWrapper = Union{
    Base.ReshapedArray{<:Any, <:Any, <:Union{SubArray{<:Any, <:Any, <:LavaArray},
                                             PermutedDimsArray{<:Any, <:Any, <:Any, <:Any, <:LavaArray}}},
    SubArray{<:Any, <:Any, <:Union{Base.ReshapedArray{<:Any, <:Any, <:LavaArray},
                                   PermutedDimsArray{<:Any, <:Any, <:Any, <:Any, <:LavaArray}}},
    PermutedDimsArray{<:Any, <:Any, <:Any, <:Any, <:Union{Base.ReshapedArray{<:Any, <:Any, <:LavaArray},
                                                          SubArray{<:Any, <:Any, <:LavaArray}}},
}

function Base.repeat(x::NestedLavaWrapper; inner=nothing, outer=nothing)
    dense = similar(x)
    dense .= x
    repeat(dense; inner, outer)
end

# Allocate broadcast output, on the device the inputs live on. The style carries
# no device, so the first device array in the argument tree decides: a broadcast
# over arrays on a second device used to allocate its result on the default
# device and then record the kernel reading the inputs on the result's queue.
firstdevicearray(x::LavaArray) = x
firstdevicearray(bc::Base.Broadcast.Broadcasted) = firstdevicearray(bc.args)
firstdevicearray(x::Base.Broadcast.Extruded) = firstdevicearray(x.x)
firstdevicearray(::Tuple{}) = nothing
function firstdevicearray(t::Tuple)
    a = firstdevicearray(first(t))
    return a === nothing ? firstdevicearray(Base.tail(t)) : a
end
firstdevicearray(x::SubArray) = firstdevicearray(parent(x))
firstdevicearray(x::Base.ReshapedArray) = firstdevicearray(parent(x))
firstdevicearray(x::PermutedDimsArray) = firstdevicearray(parent(x))
firstdevicearray(x::LinearAlgebra.Transpose) = firstdevicearray(parent(x))
firstdevicearray(x::LinearAlgebra.Adjoint) = firstdevicearray(parent(x))
# A wrapper this walk does not know is not an error: GPUArrays' own suite
# broadcasts over ad-hoc wrapper types, and the walk cannot enumerate every one.
# When no device array is found the result lands on the default device — exactly
# what this did before it grew device awareness — so a wrapper we cannot see
# through costs correctness only across devices, which is where the caller should
# be unwrapping anyway.
firstdevicearray(::Any) = nothing
function Base.similar(bc::Base.Broadcast.Broadcasted{LavaArrayStyle{N}}, ::Type{T}, dims) where {T,N}
    a = firstdevicearray(bc)
    bq = a === nothing ? vk_context().default_bq : queueof(a)
    # dims can be axes (OneTo) or plain integers
    LavaArray{T}(undef, map(Base.to_dim, dims); bq)
end

# Let GPUArrays choose between linear (1D) and cartesian (multi-D) broadcast kernels.
# LavaDeviceArray supports both linear and CartesianIndex indexing.

# ── Scalar indexing ──
# GPUArrays.jl provides Base.getindex(::AbstractGPUArray, ::Int) and
# Base.setindex!(::AbstractGPUArray, v, ::Int) which delegate to copyto!.
# Our copyto! implementations (below) handle the actual data transfer.

# ── copyto! variants for CPU↔GPU transfers ──

# ── No staging copy in either direction ──
#
# Both of these used to allocate a `Vector{UInt8}` the size of the transfer and
# `unsafe_copyto!` every byte into it, purely to hand `upload!`/`download!` the
# `Vector{UInt8}` their signatures ask for. That is a second full copy of every
# byte that crosses the bus, and on Kokoro it was **3.6 MB of host allocation per
# utterance** (measured, `--track-allocation`).
#
# `copy_buffer!` underneath them takes a raw pointer and a byte count — `upload!`
# is only a wrapper — and `download_typed!` already called it directly with a
# user array's pointer. So do these. `GC.@preserve` keeps the host array alive
# across the call, which is exactly what the wrapper was doing for its temporary.
function Base.copyto!(dest::LavaArray{T}, doffs::Integer,
                      src::Array{T}, soffs::Integer, n::Integer) where T
    n == 0 && return dest
    # Records into the active batch and flushes; `sync_access!` takes care of any
    # prior cross-queue writer to `dest.buf[]` via timeline semaphore.
    GC.@preserve src copy_buffer!(:upload, dest.buf[],
                                  Ptr{UInt8}(pointer(src, soffs)), n * sizeof(T);
                                  offset = dest.offset + (Int(doffs) - 1) * sizeof(T))
    return dest
end

function Base.copyto!(dest::Array{T}, doffs::Integer,
                      src::LavaArray{T}, soffs::Integer, n::Integer) where T
    n == 0 && return dest
    # Records a copy into the active batch of whichever queue last wrote
    # `src.buf[]` and flushes — `sync_access!` inserts any cross-queue wait.
    GC.@preserve dest copy_buffer!(:download, src.buf[],
                                   Ptr{UInt8}(pointer(dest, doffs)), n * sizeof(T);
                                   offset = src.offset + (Int(soffs) - 1) * sizeof(T))
    return dest
end

function Base.copyto!(dest::LavaArray{T}, doffs::Integer,
                      src::LavaArray{T}, soffs::Integer, n::Integer) where T
    n == 0 && return dest
    # Two devices: there is no path between their memories, and a buffer from
    # one named in a command on the other is a handle the driver takes as a
    # fault. Staged through the host instead, synchronously; a caller that wants
    # the transfer off the critical path keeps the data on one device.
    if (src.buf[].ctx::VkContext) !== (dest.buf[].ctx::VkContext)
        staging = Vector{T}(undef, n)
        copyto!(staging, 1, src, soffs, n)
        copyto!(dest, doffs, staging, 1, n)
        return dest
    end
    # Direct GPU→GPU copy via vkCmdCopyBuffer (no CPU staging roundtrip).
    src_offset = pool_offset(src.buf[]) + src.offset + (Int(soffs) - 1) * sizeof(T)
    dst_offset = pool_offset(dest.buf[]) + dest.offset + (Int(doffs) - 1) * sizeof(T)
    nbytes = n * sizeof(T)
    bq = (dest.buf[].ctx::VkContext).default_bq
    # Pin the ARRAYS, not the `VkManagedBuffer`s that `cmd_copy_buffer!` sees.
    #
    # It pins what it is given, and it is given `src.buf[]` — so it can only
    # reach `pin!(::VkManagedBuffer)`, which pushes the object into
    # `batch.pinned` and makes it REACHABLE. Reachability is not what decides
    # whether the memory is still ours: the `DataRef` refcount is, and the
    # array's finalizer releases that. So for
    #
    #     copyto!(dst, Adapt.adapt(backend, host_array))
    #
    # — no reference to the source survives the call — the GC collects the
    # source, its ref releases, the pooled block goes back, and the recorded
    # `vkCmdCopyBuffer` names memory the pool has handed onward. `submit!`
    # catches it as `sync_access!: buffer is not ALIVE`.
    #
    # `pin!(::LavaArray)` is the level that does both halves: it retains the
    # `DataRef` into the one-shot's `pinned_refs` and takes a buffer pin, and
    # `release_pinned_refs!` drops both when the submission completes or
    # fails. Nothing new is needed — every kernel argument already gets exactly
    # this lifetime through `LavaAdaptor`; only the copy path was pinning a
    # level too low.
    oneshot!(bq; tag = :copy) do e
        # DELETED in phase 1.1 (lifetime) / 1.2 (object pools): see docs/mantle-owns-it.md
        # DELETED in phase 1.1 (lifetime) / 1.2 (object pools): see docs/mantle-owns-it.md
        cmd_copy_buffer!(e, src.buf[], dest.buf[], nbytes;
                         src_off=src_offset, dst_off=dst_offset)
    end
    # No wait: this is device→device, so nothing on the host needs the result,
    # and the next closed buffer on this queue opens with the barrier that
    # orders it behind the copy. The flush that used to be here drained the
    # whole GPU on every `copy(::LavaArray)`.
    return dest
end

# ── fill! ──

function Base.fill!(a::LavaArray{T}, val) where T
    length(a) == 0 && return a
    v = convert(T, val)
    @kernel cpu=false function fill_kernel!(A, v)
        I = @index(Global)
        @inbounds A[I] = v
    end
    # From the array, not the global context. `fill!` on an array belonging to a
    # second device was dispatching on whichever context was global: the write
    # landed on the wrong queue, the array read back as zeros, and Lava's own
    # `sync_access!` guard caught it later as "buffer was last written on a
    # VulkanBatchQueue from a DIFFERENT VkContext" — a long way from the cause.
    k = fill_kernel!(KernelAbstractions.get_backend(a))
    k(a, v; ndrange=length(a))
    return a
end

# ── resize! ──
#
# Capacity-aware resize for LavaArrays, both 1-D (standard Base.resize! shape)
# and N-D (non-standard but the natural extension; textures are 2-D/3-D and
# `copyto_texture!` needs it).  Invariants:
#
# * The backing VkManagedBuffer is pool-allocated with rounded-up capacity.
#   If the new element count still fits, we just update `dims` — no Vulkan
#   alloc, no copy.  Elements beyond old length are left undefined, matching
#   `Base.resize!(::Vector, n)` semantics.
# * On genuine growth, allocate a fresh pool chunk, copy the first min(old,new)
#   elements, and retire the old DataRef via `GPUArrays.unsafe_free!`.  The
#   retirement enters `bq.deferred_frees` gated on the batch timeline — the
#   caller does NOT need a `synchronize` to make it safe w.r.t. in-flight work.
# * Throws on a freed DataRef (surfaces caller bugs; no self-heal).

function Base.resize!(a::LavaArray{T,N}, new_dims::Dims{N}) where {T,N}
    all(>=(0), new_dims) || throw(ArgumentError("new size must be non-negative"))
    a.buf.freed && throw(ArgumentError(
        "resize!(::LavaArray{$T,$N}): the backing DataRef was already freed. " *
        "This array was unsafe_free!'d elsewhere — typically an HW-accel rebuild " *
        "that released state, or a cached slot whose owner was torn down."))
    Tuple(a.dims) == new_dims && return a
    old_len = length(a)
    new_len = prod(new_dims)

    buf = a.buf[]
    capacity_bytes = buf.size - a.offset
    if new_len * sizeof(T) <= capacity_bytes
        a.dims = new_dims
        return a
    end

    ctx = buf.ctx::VkContext
    bq = ctx.default_bq
    new_buf = pool_alloc(bq, max(new_len * sizeof(T), 16))
    if old_len > 0 && new_len > 0
        copy_len = min(old_len, new_len) * sizeof(T)
        src_off = pool_offset(buf) + a.offset
        oneshot!(bq; tag = :copy) do e
            cmd_copy_buffer!(e, buf, new_buf, copy_len;
                             src_off=src_off, dst_off=pool_offset(new_buf))
        end
        # The copy pinned `buf` and stamped its `last_write` at the submit that
        # closed the one-shot, so the `unsafe_free!` below sees the buffer as in
        # use and routes through `deferred_frees`. The flush that stood here
        # closed a window the open batch had, between a recorded pin and a
        # submit that had not happened yet.
    end
    new_ref = GPUArrays.DataRef(new_buf) do b
        vk_free!(b)
    end
    old_ref = a.buf
    a.buf = new_ref
    a.dims = new_dims
    a.offset = 0
    GPUArrays.unsafe_free!(old_ref)
    return a
end

# Variadic / scalar dispatch forwarders (N-D and 1-D calling conventions).
Base.resize!(a::LavaArray{T,N}, new_dims::Vararg{Integer,N}) where {T,N} =
    resize!(a, Dims{N}(Int.(new_dims)))
Base.resize!(a::LavaArray{T,1}, n::Integer) where T = resize!(a, (Int(n),))

# ── append! ──

function Base.append!(a::LavaArray{T,1}, items::AbstractVector) where T
    items_arr = collect(T, items)
    old_len = length(a)
    new_len = old_len + length(items_arr)
    resize!(a, new_len)
    # Upload the new items to the end
    if !isempty(items_arr)
        bytes = Vector{UInt8}(reinterpret(UInt8, vec(items_arr)))
        upload!(a.buf[], bytes; offset=old_len * sizeof(T))
    end
    return a
end

# ── norm override ──
# GPUArrays' norm uses closures that capture variables as Core.Box (non-isbits),
# causing compilation failures in the rescaling branch. Override with separate
# helper functions to ensure all captured variables are isbits.

import LinearAlgebra

# Float16/Float32: accumulate in Float64 to avoid overflow entirely (no rescaling needed)
function lava_norm_p2_widen(v::LavaArray{T}) where T
    s = sum(x -> Float64(abs(x))^2, v; init=Float64(0))
    return typeof(float(LinearAlgebra.norm(zero(T))))(sqrt(s))
end

function lava_norm_p1_widen(v::LavaArray{T}) where T
    s = sum(x -> Float64(abs(x)), v; init=Float64(0))
    return typeof(float(LinearAlgebra.norm(zero(T))))(s)
end

function lava_norm_pp_widen(v::LavaArray{T}, spp::Float64) where T
    # Use exp2(p*log2(x)) instead of x^p to avoid ^ operator dispatch issues on GPU
    s = sum(x -> exp2(spp * log2(Float64(abs(x)))), v; init=Float64(0))
    return typeof(float(LinearAlgebra.norm(zero(T))))(exp2(inv(spp) * log2(s)))
end

# Float64: rescale by maxabs to avoid overflow.
#
# Subtlety: writing `abs(x) / maxabs` in the source is NOT enough. GPU drivers
# (and LLVM's `arcp` fast-math) lower a Float64 `x / m` to `x * (1/m)`, so when
# `maxabs` is near `floatmax(Float64)` the reciprocal `1/maxabs` is a subnormal
# (~1.1e-308, below the 2.2e-308 normal floor). Under flush-denormals-to-zero —
# the default on lavapipe AND RDNA3 (RADV) — that reciprocal becomes 0, then
# `abs(x) * 0 = 0`, `sum = 0`, and the rescaled norm collapses to 0. (Verified:
# `sum(x -> x/m, v)` returns exactly 0.0 iff `1/m` is subnormal, 2.0 otherwise.)
#
# Fix: divide by `sqrt(maxabs)` TWICE instead of by `maxabs` once.
# `(x / √m) / √m == x / m`, but `1/√m` is always normal (even for m == floatmax,
# `1/√floatmax ≈ 7.5e-155`), so no subnormal reciprocal is ever formed. This is
# correct for the overflow case (maxabs near floatmax), the underflow case
# (maxabs near floatmin), and normal magnitudes alike.
function lava_norm_p2_rescale(v::LavaArray{T}, maxabs::Float64) where T
    sm = sqrt(maxabs)
    s = sum(x -> ((abs(x) / sm) / sm)^2, v; init=Float64(0))
    return typeof(float(LinearAlgebra.norm(zero(T))))(maxabs * sqrt(s))
end

function lava_norm_pp_rescale(v::LavaArray{T}, maxabs::Float64, spp::Float64) where T
    # Use exp2(p*log2(x)) instead of x^p to avoid ^ operator dispatch issues on GPU
    s = sum(x -> exp2(spp * log2(abs(x) / maxabs)), v; init=Float64(0))
    return typeof(float(LinearAlgebra.norm(zero(T))))(maxabs * exp2(inv(spp) * log2(s)))
end

function LinearAlgebra.norm(v::LavaArray{T}, p::Real=2) where T
    RT = typeof(float(LinearAlgebra.norm(zero(T))))
    isempty(v) && return zero(RT)
    # `Base.count`, spelled out: this module imports Mantle's `count` (a
    # resource's element count, for `Buffer`, `GPURef` and `Attr`), so the bare
    # name here was Mantle's and the 0-norm of every `LavaArray` was a
    # `MethodError` — the GPUArrays testsuite's `0-norm` cases, 60 of them.
    p == 0 && return convert(RT, Base.count(!iszero, v))
    p == Inf && return convert(RT, maximum(abs, v))
    p == -Inf && return convert(RT, minimum(abs, v))
    # Non-trivial p-norms rely on transcendental math in the reduction path.
    # These are currently not numerically stable across Vulkan drivers (e.g. RDNA3.5),
    # so use CPU fallback for correctness/portability.
    (p != 1 && p != 2) && return convert(RT, LinearAlgebra.norm(Array(v), p))

    # Float16/Float32/ComplexF16/ComplexF32: accumulate in Float64 (no overflow possible)
    if RT === Float32 || RT === Float16
        p == 2 && return lava_norm_p2_widen(v)
        p == 1 && return lava_norm_p1_widen(v)
        return lava_norm_pp_widen(v, Float64(p))
    end

    # Float64/ComplexF64: rescale by max to avoid overflow
    maxabs = convert(Float64, maximum(abs, v))
    (iszero(maxabs) || isinf(maxabs)) && return convert(RT, maxabs)

    # Try without rescaling first (faster, only when no overflow AND no underflow)
    if p == 2 && isfinite(length(v) * maxabs^2) && !iszero(maxabs^2)
        s = sum(abs2, v; init=Float64(0))
        return convert(RT, sqrt(s))
    end

    p == 2 && return lava_norm_p2_rescale(v, maxabs)
    p == 1 && return convert(RT, sum(abs, v; init=Float64(0)))
    return lava_norm_pp_rescale(v, maxabs, Float64(p))
end

# ── permutedims: superseded, and the reasoning is why ──
#
# There was a `lava_permutedims_kernel!` here: an N-D ndrange so the grid hands
# the kernel its Cartesian coordinates and the `ndims - 1` integer divisions
# GPUArrays' generic kernel pays per element disappear. The premise was that the
# permute is bound by that division chain.
#
# **It is not, and removing the divisions made it slower.** Measured against the
# ordinary broadcast — which pays the full `cart32` chain *and* re-indexes a
# `PermutedDimsArray` — the division-free kernel lost by 2.6-5.9x on every shape
# SAM 2's encoder runs (table in `lavapermutedims!` below). What it bought in
# arithmetic it gave back many times over in geometry: an N-D ndrange gets an N-D
# workgroup, consecutive lanes stop walking consecutive memory, and that is the
# same effect the note at the top of this file measures at 54 GB/s against 315.
#
# Access order is the cost. Anything that beats the broadcast has to change the
# order — tile it, or widen what one workgroup covers — and it belongs in the
# broadcast path so both spellings get it.

@inline groupfill(::Tuple{}, left::Int) = ()
@inline function groupfill(sz::Tuple{Int, Vararg{Int}}, left::Int)
    t = min(first(sz), left)
    (t, groupfill(Base.tail(sz), max(1, left ÷ t))...)
end

"""
    launchgroup(sz, target = 256) -> Dims

Workgroup shape for an `ndrange` of `sz`: fill from the fastest axis up, moving
to the next only once the current one is exhausted.

Filling in axis order is what makes the group a *contiguous* run of memory —
every axis before the first partially-taken one is complete, so `(4, 4, 288,
1024)` yields `(4, 4, 16, 1)`, covering 256 consecutive elements. Left to
KernelAbstractions, which derives the group from the ndrange alone, that same
shape launches with a **4-thread** workgroup: 4 of 32 lanes in the warp doing
work.

**Pass the result as a launch keyword**, `kernel(backend)(args...; ndrange,
workgroupsize = launchgroup(ndrange))`, and *not* as `kernel(backend, wg)`.
Those two spellings are not equivalent:

  * `kernel(backend, wg)` puts the size in the kernel's type, so the iteration
    space it builds pairs a `blocks` field with a zero-size `workitems::Nothing`.
    For a **rank-4** ndrange that combination computes wrong global indices —
    `ndrange = (72, 256, 8, 16)`, `wg = (32, 4, 1, 1)` writes 294 912 of
    2 359 296 elements and reports no error. Ranks 1-3 and rank 5 are fine, as
    is `kernel(backend, wg, ndrange)` with both static, which is what makes the
    fault easy to miss.
  * The keyword form keeps both extents as fields and is correct at every rank
    tested. `test_static_workgroup.jl` pins the difference.
"""
@inline launchgroup(sz::Dims, target::Int = 256) = groupfill(sz, target)

"""
    staticgroup(sz, target = 256) -> Dims

Like [`launchgroup`](@ref) but never returns an interior unit extent, so the
result can go in the kernel's TYPE without tripping `WORKGROUP_FALLBACK` — which
keeps the index arithmetic compile-time constant.

Each interior axis is given 2 first (or its own extent, if smaller), and whatever
is left of the budget goes to the fastest axis. That trades some of `launchgroup`'s
contiguity for the constant-folded indexing; only worth it where measured.

**Only as far as the budget reaches.** Two threads on every interior axis is
`2^(N-2)` threads, which passes `target` at rank 10 and is 65 536 at rank 18 —
past `maxComputeWorkGroupInvocations` on every device, so the dispatch simply
never completes. GPUArrays' 18-d `permutedims` test hung there for the full
120 s flush timeout and took the following 354 assertions of the suite with it.
Above that rank the reservation stops and the trailing axes stay at 1, which is
exactly what `WORKGROUP_FALLBACK` exists to catch: the launch is re-issued
dynamically and is correct, just without the constant-folded indices.
"""
function staticgroup(sz::Dims{N}, target::Int = 256) where {N}
    N <= 2 && return launchgroup(sz, target)
    w = ones(Int, N)
    # Reserve 2 on each interior axis so none stays at 1, while it still fits…
    for d in 2:(N - 1)
        c = min(2, sz[d])
        c > 1 && c * prod(w) > target && break
        w[d] = c
    end
    # …give the fastest axis whatever that leaves…
    w[1] = min(sz[1], max(1, target ÷ prod(w)))
    # …and spend the remainder outward, so a short leading axis (`(4,4,288,…)`
    # would otherwise get a 16-thread group) still fills the budget.
    for d in 1:(N - 1)
        room = target ÷ prod(w)
        room <= 1 && break
        w[d] = min(sz[d], w[d] * room)
    end
    return ntuple(d -> w[d], Val(N))
end


"""
    permutedims!(dest::AnyLavaArray, src::AnyLavaArray, perm)

`dest[I...] = src[I[perm]...]`, one thread per destination element.

More specific than GPUArrays' method, which this exists to replace; see the note
above for why.
"""
# Two methods, matching GPUArrays' own signatures so neither is ambiguous with
# them: it defines one for `NTuple` and one for the general case.
Base.permutedims!(dest::AnyLavaArray, src::AnyLavaArray, perm::NTuple{N, T}) where {N, T} =
    lavapermutedims!(dest, src, perm)
Base.permutedims!(dest::AnyLavaArray, src::AnyLavaArray, perm::AbstractVector) =
    lavapermutedims!(dest, src, Tuple(perm))

function lavapermutedims!(dest::AnyLavaArray, src::AnyLavaArray, perm::Tuple)
    N = ndims(src)
    length(perm) == N ||
        throw(ArgumentError("permutedims!: perm has $(length(perm)) entries for a $N-d array"))
    isperm(perm) || throw(ArgumentError("permutedims!: $perm is not a permutation"))
    ndims(dest) == N ||
        throw(DimensionMismatch("permutedims!: destination is $(ndims(dest))-d, source $N-d"))
    for d in 1:N
        size(dest, d) == size(src, perm[d]) ||
            throw(DimensionMismatch("permutedims!: destination axis $d is $(size(dest, d)), " *
                                    "source axis $(perm[d]) is $(size(src, perm[d]))"))
    end
    N == 0 && return dest
    # One path, not two spellings. `dest .= PermutedDimsArray(src, perm)` and
    # `permutedims!(dest, src, perm)` are the same operation, and until now they
    # ran different kernels: the broadcast flattens the launch and recovers the
    # Cartesian index with `cart32`, this one launched an N-D ndrange with an
    # N-D workgroup. That is precisely the geometry the note at the top of this
    # file measures at a quarter of the achievable bandwidth — consecutive lanes
    # stop walking consecutive memory — and the dedicated kernel lost to the
    # generic one on every shape SAM 2's encoder runs:
    #
    #   src shape              perm             bcast   permutedims!
    #   (72, 8, 256, 16)       (1,3,2,4)         34.3     13.2 GB/s
    #   (72, 4, 16, 1024)      (1,3,2,4)         35.8      9.6
    #   (576,16,4,16,4,1)      (1,2,4,3,5,6)     33.2      5.6
    #   (288,4,32,4,32,1)      (1,2,4,3,5,6)     19.0      4.7
    #   (256,256,144,1)        (3,1,2,4)         17.8     16.9
    #
    # (interleaved, one session, bit-exact both ways). 2.6-5.9x, for deleting a
    # kernel rather than writing one. It also retires two traps that were only
    # ever this launch's: the rank-6 miscompile that made `permutedims!` write a
    # fraction of its output, and the rank-18 `staticgroup` hang.
    #
    # Anything faster than the broadcast belongs *in* the broadcast path, where
    # both spellings get it.
    dest .= PermutedDimsArray(src, perm)
    return dest
end
