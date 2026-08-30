# `KernelInterface`'s host half, for Metal.
#
# The mirror of `src/vulkan/array/kernelinterface_host.jl`, and deliberately the
# same shape: every one of these either forwards to KernelAbstractions, which
# Metal.jl already implements, or reads a field of [`caps`](@ref). Nothing here
# decides anything — a backend that computes a launch geometry in this file has
# taken work that belongs to `KernelInterface.launch`.
#
# The device half — `get_global_id`, `shfl_down`, `sub_group_reduce_add` and the
# rest — is NOT here. Metal.jl's own `@metal` compiler provides those
# intrinsics, where the Vulkan backend needs Lava to emit SPIR-V for them. That
# asymmetry is why Metal is a weak dependency on its own and Vulkan is one
# paired with Lava.

const MB = Metal.MetalBackend

KI.synchronize(b::MB) = KA.synchronize(b)
KI.get_backend(a::MtlArray) = KA.get_backend(a)

KI.allocate(b::MB, ::Type{T}, dims::Tuple;
            unified::Union{Nothing,Bool} = nothing) where {T} =
    KA.allocate(b, T, dims)

# Both true and both structural rather than a capability flag: Apple silicon has
# one physical memory, so every allocation is unified, and Metal's atomics are
# core to the shading language rather than an extension.
KI.supports_unified(::MB) = true
KI.supports_atomics(::MB) = true

# ── Device limits, all out of `caps` ──────────────────────────────────────────
#
# Read from the cached `DeviceCaps` rather than the `MTLDevice`, so the portable
# route and the direct one cannot disagree. That is the invariant
# `test_array_algorithm_portability.jl` pins on the Vulkan side, and the reason
# `gemv` can size a workgroup without naming a backend.

KI.sub_group_size(b::MB) = caps(b).subgroup
KI.max_work_group_size(b::MB) = caps(b).workgrouplimit
KI.multiprocessor_count(b::MB) = caps(b).cores

"""
The largest threadgroup a specific kernel can be launched with.

The device's limit is an upper bound; a kernel that uses many registers gets
less. Metal reports the real number per pipeline, so unlike the Vulkan backend —
which has to take the device limit and hope — this can answer exactly.
"""
function KI.kernel_max_work_group_size(b::MB, kernel; max_work_items::Int = typemax(Int))
    n = try
        Int(kernel.pipeline.maxTotalThreadsPerThreadgroup)
    catch
        # Not every callable handed here is a compiled Metal kernel; fall back to
        # the device limit rather than throwing, which is what the Vulkan backend
        # always does.
        caps(b).workgrouplimit
    end
    return min(n, max_work_items)
end

KI.matrix_shapes(b::MB) = KI.matrix_shapes(caps(b))
KI.supports(b::MB, s::MatrixShape) = KI.supports(caps(b), s)
KI.bestshape(b::MB, ab, acc; scope::MatrixScope = SubgroupScope()) =
    KI.bestshape(caps(b), ab, acc; scope)

# ── The shuffle and reduction families ────────────────────────────────────────
#
# Which element types Metal's SIMD-group intrinsics cover. Narrower than
# Vulkan's six: `simd_shuffle_down` and `simd_sum` are float and 32-bit integer,
# and there is **no Float64 at all** — Apple GPUs do not have it. That is the
# divergence `KI_REDUCE_ADD_TYPES` was given its own list for rather than being
# tied to `KI_SHFL_TYPES`.
const MTL_SHFL_TYPES = (Float32, Float16, Int32, UInt32)
const MTL_REDUCE_ADD_TYPES = (Float32, Float16, Int32, UInt32)

KI.shfl_down_types(::MB) = collect(MTL_SHFL_TYPES)
KI.sub_group_reduce_add_types(::MB) = collect(MTL_REDUCE_ADD_TYPES)

# ── Argument conversion ───────────────────────────────────────────────────────

"""
What a kernel receives in place of a host-side array.

`MtlArray` → `MtlDeviceArray`, which is Metal.jl's own adaptation and the exact
counterpart of `LavaArray` → `LavaDeviceArray`. Everything else passes through:
a scalar is a scalar on both sides.
"""
KI.argconvert(::MB, a::MtlArray) = Metal.mtlconvert(a)
KI.argconvert(::MB, x) = x
