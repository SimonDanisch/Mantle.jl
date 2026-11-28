# Shared ray-tracing packing and native setup

Baseline: `0068bd1`. One packtrace! body now retains shader/argument leaves,
adapts values and packs bytes for eager direct/indirect and recorded traces.
Callers still choose one-shot scratch or plan-owned memory; Ref-backed graph
arguments are resolved at recording time. The existing core leaf walk covers
all stages of the driver-free RayTracingPipeline description.

One emit_trace! body owns native bindings, TLAS/pipeline holds, timestamps and
dispatch accounting. Typed tracecommand! methods encode direct dimensions or
an indirect GPU address, including view offsets. Indirect command storage stays
held through completion; counts are never downloaded.

Compute, graphics and ray pipelines share shader-module construction. Compute
and ray tracing share acceleration-structure descriptor allocation/writes,
while retaining their different ownership: compute submission/recording versus
TLAS/layout cache. The obsolete no-op RT invalidation hook is removed.

677 assertions pass on RX 7900 XTX / RTX 4000 Ada and host fixtures: eager
before/after byte comparisons, forced native RT rebuilds, recorded direct and
device-count traces, cull masks, hardware/software hits, access tracking,
compute-cache invalidation, graphics creation/rendering and hold lifetime.
Validation and synchronization validation are enabled; final drains are empty.
A permanent host regression checks every shader stage's captured array holds.

All 115 source files parse. Source shrinks 53,649 → 53,476 physical lines and
23,339 → 23,265 implementation lines. This does not finish Builder execution,
cache retention or token-retirement ownership; their deletion gates remain in
consolidation-order.md. No speedup or complete-platform qualification is claimed.
