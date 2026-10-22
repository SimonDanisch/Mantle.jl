# Shared FFT configuration and generated kernels — 2026-10-22

Baseline: `9b008fa`. Eager and graph FFTs share radix selection, launch arguments,
group/thread/shared-memory constraints and real-transform geometry. Eager calls
consume concrete configuration directly; graphs build ArrayLaunch descriptors.
This preserves eager allocation behavior while deleting repeated selection policy.

MixedFFTKernel specializes a generated callable on radices/direction. It replaces
runtime eval, the mutable FFT_MIXED_KERNELS dictionary and FFT_PREGENERATED list.
Julia's specialization machinery owns the generated bodies; arbitrary supported
sizes can be requested while consumers precompile. Literal coefficients, uniform
barriers and distinct tuned power-of-two/mixed layouts remain.

Qualification also fixes invalid forced groups silently omitting batches (the
old 64×3/group=2 launch leaves the third column untouched), ignored rfft skew,
unchecked real-transform batch/scratch geometry and invalid radix choices.
Zero-batch transforms emit no FFT dispatch. Unsupported sizes still fail explicitly.

1,013 FFT assertions pass across AMD/NVIDIA Vulkan and ROCm, including direct
Float64 DFT references, inverse/Parseval checks, recorded complex/real replay,
mixed layouts, empty batches and configuration refusals. Both Vulkan validation
and synchronization drains are empty. Thirty paired old/new checks preserve
mixed-transform output bytes across 14 configurations on both Vulkan devices;
23 source-portability checks and 12 paired output/allocation checks pass.

[Allocation probe](fft-config-probe.jl) and [results](fft-config-allocations.tsv):
100 warm calls, minimum of three samples. Power-of-two allocation is unchanged
within queue-pool variation; mixed calls save about 96 host bytes/call. GPU
throughput/native driver allocation is not measured. Native Metal is unqualified.

The [consumer fixture](../../test/fixtures/FFTPrecompileProbe/Project.toml) requests
the previously unlisted 125-point transform during package precompilation. The
old implementation fails with evaluation into closed module Mantle; the new one
passes. In a disposable workspace environment, develop this fixture with
Pkg.PRESERVE_ALL and run Pkg.precompile(["FFTPrecompileProbe"]).
A fresh process passes 338 fixture/native FFT assertions.

All 115 source files parse. Prose edits preserve definitions. Source falls
50,338 → 50,102 physical lines; implementation rises 22,816 → 22,822 (six net
lines for the added checks). Cache/selection owners are deleted without claiming
a net implementation-line reduction.
