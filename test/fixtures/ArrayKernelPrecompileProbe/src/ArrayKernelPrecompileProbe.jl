module ArrayKernelPrecompileProbe
import Mantle

# Previously unlisted kernel configurations must work while a consumer precompiles.
const kernels = (Mantle.fftmixed_kernel((5,5,5),-1),
                 Mantle.fftmixed_kernel((5,5,5),1),
                 Mantle.gemv_kcontig_kernel(3,96,32),
                 Mantle.gemv_ncontig_kernel(24,192,3))
end
