# Shared diagonal multiplication — 2026-10-22

Baseline: `b06b12c`. Vulkan's three diagonal mul! intersections now forward to
GPUArrays.generic_diagonal_mul!. GPUArrays' existing five-argument methods use
the same helper; all three algorithm bodies remain textually identical there.
The native backend no longer copies scaling, dimension checks, conjugation or
diagonal/off-diagonal initialization.

GPUArrays adds 13 physical / nine implementation lines on the published
`sd/mantle-diagonal-core` branch (`61e30cea215e93202261d3f0b38df991f36c9081`).
Mantle's source dependency pins that revision. Direct invoke delegation was
rejected after exposing extra allocation on plain-matrix operands; the final
ordinary dispatch uses the shared helper without that regression.

[Probe](diagonal-probe.jl): 56 baseline output arrays match byte for byte across
AMD/NVIDIA Vulkan, including complex values, transpose/adjoint layouts, both
zero/nonzero beta and diagonal-pair NaN initialization. Both validation
drains are empty (58 assertions). Existing diagonal/dense/non-numeric regressions
pass 226 assertions; expanded wrapped/dimension regressions pass 94.
ROCm passes 144 related assertions. A fresh environment fetches the published
dependency, precompiles the consumer and passes 149 native assertions plus two
dependency-source checks. Native Metal is unqualified.

[Allocations](diagonal-allocations.tsv): six paired measurements pass byte,
validation and allocation checks (18 assertions). Minimum drained allocation over 30 calls
is unchanged at 496/528/864 bytes for left/right/both products on both GPUs.
The 100-call burst measurements retain queue-capacity effects; neither set
measures GPU throughput or driver allocation.

All 115 Mantle source files parse; prose edits preserve definitions. Mantle
falls 49,691 → 49,597 physical and 22,776 → 22,752 implementation lines. Including
the GPUArrays helper, the net implementation decrease is 15 lines; three copied
algorithm bodies are removed, not three dispatch intersections.
