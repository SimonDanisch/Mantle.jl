module ArrayKernelPrecompileProbe
import Mantle

# Previously unlisted kernel configurations must work while a consumer precompiles.
const kernels = (Mantle.fftmixed_kernel((5,5,5),-1),
                 Mantle.fftmixed_kernel((5,5,5),1),
                 Mantle.gemv_kcontig_kernel(3,96,32),
    Mantle.gemv_ncontig_kernel(24,192,3),
    (Mantle.blocked_gemm_kernel(b) for b in (1,2,4))...,
    (Mantle.staged_gemm_kernel(c,f) for
        f in (:scalar,:vec2,:narrow,:ablation,:prefetch,:doublebuf,:vec4,:gather)
        for c in Mantle.staged_gemm_tilings(f))...)
end
