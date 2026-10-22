# ROCm qualification

Recording requires the project's split-capture AMDGPU API. The qualified source
is [SimonDanisch/AMDGPU.jl, mantle/ka-0.10](https://github.com/SimonDanisch/AMDGPU.jl/tree/4594aa162f98f6547dcd6563e06176b480d75708)
at `4594aa162f98f6547dcd6563e06176b480d75708` (2.8.0).
Registered AMDGPU 2.9.0 runs kernels but lacks HIP.begincapture!/endcapture!;
it cannot run this recording implementation unchanged.

In a disposable copy of the workspace environment, develop that source before
starting Julia. Preserve the workspace's local dependency paths. Import AMDGPU
and Mantle normally; never include the extension by hand. Then select a device:

```julia
using AMDGPU, Mantle
AMDGPU.device!(first(AMDGPU.devices()))
Mantle.defaultdevice!(Mantle.Device(Mantle.ROCmAPI()))
global MANTLE_TEST_BACKEND = Mantle.defaultbackend()
include("test/rocm/test_rocm.jl")
```

Qualified 2026-10-22 on Linux, Julia 1.12.7, RX 7900 XTX/gfx1100: 77 backend
assertions and 1,329 shared GEMM assertions across test_gemm_staged_scalar.jl,
test_gemm_staged.jl and test_coopmat_epilogue.jl. Recording/replay, stream and
argument ownership, borrowed offsets, allocation-free replay, hipBLASLt and
portable GEMM/GEMV/FFT/STFT pass. MIOpen was unavailable; these results do not
qualify MIOpen or other GPU/OS combinations.
