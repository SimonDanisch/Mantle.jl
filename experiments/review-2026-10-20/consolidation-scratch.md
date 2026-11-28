# Independent split workspace

Baseline: `15a44e4`. Vulkan's split-K cache and retained growth history are
deleted. Core already allocates ordinary partial arrays on the destination's
device; each launch description now owns independent defaults on Vulkan too.
Explicit caller-provided partials and kernel selection/math are unchanged.

The frozen old specialization reproduces corruption on AMD/NVIDIA: two programs
for all-one/all-two inputs share partials. Interleaving their block/reduction
launches returns 2,048 instead of 1,024 for the first program. Owned defaults
produce both expected results. Recording/emission retain original array owners;
no backend cache or new lease protocol is needed.

Qualification across AMD Vulkan, NVIDIA Vulkan and explicitly selected ROCm:
30 workspace/replay assertions, 286 scalar GEMM assertions and 68 split-GEMV
assertions pass. Six Vulkan retirement checks prove dropped partials return
their pool regions; native validation is empty. Warm recorded replay allocates
zero Julia bytes on all three backends. Native Metal remains unqualified.

Two fixture issues were debugged: measuring an abstract plan in the preparation
closure boxed 64 bytes, while a warmed specialized call allocates zero. The
existing alpha=2 GEMV check divided error by the alpha=1 reference scale;
normalizing by its actual expected output preserves the same relative tolerance.
ROCm's ordinary mul! uses its library implementation, so those results are not
presented as evidence for Mantle's split kernel; the interleaved fixture selects
Mantle's kernel explicitly.

Cost is explicit: preparation/release grows from 80 to 608 Julia bytes and from
0.032–0.036 to 0.783–0.786 microseconds. Paired completed-call measurements are
1.01–1.04 times the old policy for the six measured cases. They include forced
synchronization, explicit partial release and draining; they are not throughput
or recorded replay measurements. Independent ownership fixes correctness; it
is not a speedup. Target Builder/transient workspace integration remains open.

[Probe](split-workspace-probe.jl) replays the old specialization directly from
git. [Preparation](split-workspace-results.tsv) and
[completed calls](split-workspace-completed-results.tsv) preserve measurements.
Production source loses 14 implementation lines; prose deletion is separate.
P0.A10 and the broader lifetime/Builder migration remain unfinished.
