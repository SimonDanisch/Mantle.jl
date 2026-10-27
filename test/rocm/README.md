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
global MANTLE_TEST_BACKEND = Mantle.Device(Mantle.ROCmAPI())
@assert nameof(typeof(MANTLE_TEST_BACKEND)) === :ROCmDevice
include("test/rocm/test_rocm.jl")
```

defaultbackend() still chooses the platform's preferred API. Select the ROCm
device explicitly for qualification. The [qualification script](../../experiments/review-2026-10-20/qualify-rocm.jl)
asserts its backend and runs the backend, array and recording checks together.
The earlier 1,329 “ROCm” GEMM count accidentally exercised Vulkan and is withdrawn.
Qualified 2026-10-22 on Linux, Julia 1.12.7, RX 7900 XTX/gfx1100: 2,019 assertions
pass, including all staged GEMM families, GEMV, FFT/STFT and recording/store
ownership. One graphics case skips through supports_graphics.
MIOpen was unavailable; qualification does not cover MIOpen or other GPU/OS combinations.
