# Shared Vulkan draw preparation/packing — 2026-10-22

Baseline: `4ae893b`. Graph and manual draws now share shader-pair preparation,
pipeline publication and packed-layout selection. Raster/mesh wrappers return
the shader already obtained, removing a second lookup. Graph callers keep exact
native attachment formats, including sRGB, plus their existing binding cells,
argument offsets and ownership. The unused vector-returning pack_gfx_args and
redundant post-adaptation convert_args are deleted; no workspace caller uses them.

Both paths now use pack_gfx_args! for leaf retention, adaptation and packing.
A negative control on the old manual packer found zero holders, no retained nested
array wrapper and no registered backing buffer. Its assumption that Adapt pinned
arrays was false. The permanent native regression frees a nested array before
submission, drains retirement, verifies the rendered pixels, then checks exactly
when the backing buffer becomes dead. This passes on AMD and NVIDIA.

Graphics/lifetime/publication suites pass 392 assertions with two existing
native-geometry skips. Four additional assertions preserve sRGB/UNORM formats;
62 source-boundary assertions pass with one existing pure-method triage broken.
A fresh package load independently passes 34 native lifetime/publication/format
checks. Validation and synchronization validation are enabled; drains are empty.

The [paired probe](draw-preparation-probe.jl) checks identical compiled fields and
warm allocation (10 assertions). [Interleaved medians](draw-preparation-timings.tsv)
use 15 pairs of 100 calls after warmup: host allocation falls 1,984 → 1,904 bytes
per call. Both arms use current stage publication, isolating draw preparation.
These are compilation/setup measurements, not frame throughput or GPU speedups.

Prose edits preserve JuliaSyntax definition trees. All 115 source files parse.
Source decreases 51,251 → 50,995 physical and 22,943 → 22,893 implementation
lines: 50 implementation lines removed. Native Metal/ROCm execution is unqualified.
