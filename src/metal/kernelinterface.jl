# `KernelInterface`'s host half, for Metal.
#
# Metal.jl implements KernelInterface itself: the device queries, the limits,
# `argconvert`, `kernel_function` and `launch` are its methods on
# `Metal.MetalBackend`, and defining them here again would overwrite them, which
# Julia refuses during precompilation. What is left is what Mantle adds on top.
#
# The device half — `get_global_id`, `shfl_down`, `sub_group_reduce_add` and the
# rest — is Metal.jl's too. Its own `@metal` compiler provides those intrinsics,
# where the Vulkan backend needs Lava to emit SPIR-V for them. That asymmetry is
# why Metal is a weak dependency on its own and Vulkan is one paired with Lava.

const MB = Metal.MetalBackend

# ── The matrix vocabulary, out of `caps` ──────────────────────────────────────
#
# Read from the cached `DeviceCaps` rather than the `MTLDevice`, so the portable
# route and the direct one cannot disagree. That is the invariant
# `test_array_algorithm_portability.jl` pins on the Vulkan side, and the reason
# `gemv` can size a workgroup without naming a backend.

KI.matrix_shapes(b::MB) = KI.matrix_shapes(caps(b))
KI.supports(b::MB, s::MatrixShape) = KI.supports(caps(b), s)
KI.bestshape(b::MB, ab, acc; scope::MatrixScope = SubgroupScope()) =
    KI.bestshape(caps(b), ab, acc; scope)

# ── A kernel around a plain function ──────────────────────────────────────────
#
# `KI.kernel_function` compiles, and the `KI.Kernel` it returns holds Metal.jl's
# `HostKernel`, which is what Metal.jl's `KI.launch` reads. Two callers build the
# wrapper around the plain function instead, `KI.Kernel(backend, f)`, because the
# argument types are only known at the launch: `record.jl` and DNNKernels' tuning
# harness. These two methods compile such a kernel for the arguments it is
# launched with, and answer its workgroup limit from the device until then.

KI.max_work_group_size(k::KI.Kernel{MB, <:Function}) = KI.max_work_group_size(k.backend)

function KI.launch(k::KI.Kernel{MB, <:Function}, groups::Dims{3}, items::Dims{3},
                   args::Tuple; kwargs...)
    tt = Tuple{map(a -> Core.Typeof(KI.argconvert(k.backend, a)), args)...}
    KI.launch(KI.kernel_function(k.backend, k.kern, tt), groups, items, args; kwargs...)
end
