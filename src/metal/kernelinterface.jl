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
# `KI.get_backend(::MtlArray)` is NOT here: Metal.jl defines it now, and it is
# typed on the ARRAY rather than on a backend, so two definitions is a
# redefinition and Mantle stopped precompiling outright. Metal's answers
# `MetalInterface.MetalBackend`, its own KernelInterface backend, which is a
# different type from the `MetalKernels.MetalBackend` every method below is
# written for. Nothing in this tree calls it, so ceding it costs nothing --
# but the two backend types are a seam worth knowing about before anything
# starts routing through `KI.get_backend`.

KI.allocate(b::MB, ::Type{T}, dims::Tuple;
            unified::Union{Nothing,Bool} = nothing) where {T} =
    KA.allocate(b, T, dims)

# Both true and both structural rather than a capability flag: Apple silicon has
# one physical memory, so every allocation is unified, and Metal's atomics are
# core to the shading language rather than an extension.
KI.supports_unified(::MB) = true
KI.supports_atomics(::MB) = true
KI.supports_float64(::MB) = false

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
KI.shfl_types(::MB) = collect(MTL_SHFL_TYPES)
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

# `Metal.mtlfunction` is the compiler; `KI.Kernel` supplies KI's portable launch
# vocabulary around the resulting pipeline.  Keep the compiled HostKernel in the
# wrapper so `kernel_max_work_group_size` can use the pipeline's real limit and an
# immediate launch does not compile a second time.
function KI.kernel_function(backend::MB, @nospecialize(f), @nospecialize(tt) = Tuple{};
                            name = nothing, kwargs...)
    isempty(kwargs) || throw(ArgumentError(
        "unsupported Metal KernelInterface compiler keywords: $(keys(kwargs))"))
    host = Metal.mtlfunction(Metal.mtlconvert(f), tt; name)
    return KI.Kernel(backend, host)
end

function KI.kernel_max_work_group_size(k::KI.Kernel{<:MB};
                                       max_work_items::Int = typemax(Int))::Int
    return min(k.kern.maxthreads, max_work_items)
end

function (k::KI.Kernel{<:MB})(args...;
                              numworkgroups = (), workgroupsize = (), ndrange = (),
                              max_work_group_size::Int = typemax(Int))
    KI.check_launch_args(numworkgroups, workgroupsize, ndrange)

    # KI explicitly defines an empty dimension as a no-op.  An explicit zero
    # workgroup count has the same meaning and Metal must not encode it.
    (length(ndrange) > 0 && prod(ndrange) == 0) && return nothing
    (length(numworkgroups) > 0 && prod(numworkgroups) == 0) && return nothing

    # A `KI.Kernel` here holds EITHER a compiled `HostKernel` or the plain function
    # it was built from. `kernel_function` above hands back the first; core builds
    # the second directly -- `KI.Kernel(backend, f)` in `array/gemv.jl` and
    # `array/fft.jl`, which is the documented spelling for a macro-free kernel that
    # is not its own constructor. Both have to launch, and everything below this
    # point needs the compiled one: `auto_launch_sizes` reads the pipeline's
    # `maxthreads`, and the call itself takes `groups`/`threads` keywords that a
    # plain function has no method for.
    kern = k.kern isa Metal.HostKernel ? k.kern :
        Metal.mtlfunction(Metal.mtlconvert(k.kern),
                          Tuple{map(a -> Core.Typeof(KI.argconvert(k.backend, a)), args)...})
    groups, threads = KI.auto_launch_sizes(
        KI.Kernel(k.backend, kern), numworkgroups, workgroupsize, ndrange,
        max_work_group_size)
    kern(args...; groups, threads)
    return nothing
end
