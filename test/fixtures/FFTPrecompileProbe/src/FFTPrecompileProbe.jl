module FFTPrecompileProbe
import Mantle

# A previously unlisted transform size must work while a consumer precompiles.
const kernels = (Mantle.fftmixed_kernel((5,5,5),-1),
                 Mantle.fftmixed_kernel((5,5,5),1))
end
