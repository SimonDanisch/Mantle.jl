# Remove redundant native argument packers — 2026-10-22

Baseline: `00a99ef`. Vulkan's device-array aggregate packer duplicated the
generic isbits aggregate branch, including padding, self-reference and pointer
patch registration. Its UInt64 packer duplicated the primitive branch. Both
specializations are deleted. Ptr retains explicit 64-bit BDA conversion;
VkManagedBuffer retains allocation holds and lifecycle validation.

24 paired byte/offset checks against the deleted methods pass for ranks 1–3
and UInt64 on both devices, including nonaligned initial inline offsets.
3,492 AMD/NVIDIA native assertions pass: argument packing, nested tuple pointer
patches, struct broadcast, recorded relocation and dispatch allocation checks.
Validation/synchronization drains are empty. The runtime now has three pack_arg!
methods. This changes no ABI, pointer-patch depth rule or resource ownership.

Historical launch/slab/cache stories are replaced with current contracts;
JuliaSyntax confirms those prose edits preserve definitions. All 115 files parse.
Source falls 50,708 → 50,570 physical and 22,844 → 22,824 implementation lines
(20 implementation lines removed, with no prose reclassification).
