# Mantle's additions to KernelInterface for Metal. Metal.jl defines the device
# queries, `argconvert`, `kernel_function`, `launch` and the device intrinsics
# itself; redefining them here is a method overwrite, which fails precompilation.

const MB = Metal.MetalBackend

# Read from the cached `DeviceCaps`, not the `MTLDevice`, so the portable and the
# direct query cannot disagree (`test_array_algorithm_portability.jl`).
KI.matrix_shapes(b::MB) = KI.matrix_shapes(caps(b))
KI.supports(b::MB, s::MatrixShape) = KI.supports(caps(b), s)
KI.bestshape(b::MB, ab, acc; scope::MatrixScope = SubgroupScope()) =
    KI.bestshape(caps(b), ab, acc; scope)

# `KI.Kernel(backend, f)` around a plain function: its argument types are known
# only at launch, so it compiles for the launch arguments and answers its
# workgroup limit from the device until then.
KI.max_work_group_size(k::KI.Kernel{MB, <:Function}) = KI.max_work_group_size(k.backend)

function KI.launch(k::KI.Kernel{MB, <:Function}, groups::Dims{3}, items::Dims{3},
                   args::Tuple; kwargs...)
    # A top-level `Buffer` or `GPURef` resolves to the array over its region, as in
    # `dispatch!`; Metal.jl's compiler rejects it whole as a non-bitstype argument.
    args = argmap(storage, args)
    tt = Tuple{map(a -> Core.Typeof(KI.argconvert(k.backend, a)), args)...}
    KI.launch(KI.kernel_function(k.backend, k.kern, tt), groups, items, args; kwargs...)
end
