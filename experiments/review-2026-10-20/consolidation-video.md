# Shared native video sessions

Baseline: `1f30265` (including the concurrent sum-ordering repair). H.264 and H.265 now use `CodecDecoder` and
`vulkan/runtime/video_common.jl` for session state, native capability queries,
memory binding, images, Annex-B splitting, feed grouping, upload packing,
batch draining and cleanup. Codec parsing, picture/reference structures and
reference-marking rules remain separate. Existing public streaming verbs remain.

Mapped input grows transactionally and releases its previous allocation;
per-growth memory/profile entries no longer accumulate in PIN. Direct upload
writes prefixes and slice bytes into mapped memory without per-slice staging.
Session memory prefers device-local placement while obeying driver eligibility;
bitstream memory still requires HOST_VISIBLE | HOST_COHERENT. NVIDIA requires
host-only memory for one session binding: requiring local memory there was an
implementation error, fixed without a transfer fallback or design change.

Both batch wrappers share prefix truncation and cleanup. H.265 maxframes=5 now
returns five frames (previously zero because reorder holdback never flushed).
With chroma=false, both wrappers return an empty UV vector. Closed decoders
release retained resources and reject further feed/decode operations.

## Qualification

307 assertions pass: 90 host parser/packing checks; 75 AMD and 108 NVIDIA
session checks; 24 existing H.264 assertions; ten native memory-policy checks.
Both GPUs match all Y/UV bytes for 24 H.264 type-2 frames; NVIDIA also matches
24 H.265 frames, with and without UV output. Checks cover incremental decoding,
input growth, five/zero-frame batches, close and memory requirements. Validation
and synchronization validation are enabled; both final drains are empty.
AMD's existing H.265 DISTINCT refusal remains explicit.

The unchanged decoder first qualified the existing H.264 fixture on both GPUs
and H.265 on NVIDIA. The new H.264 fixture initially had 48 FFmpeg reference
frames for 24 encoded frames: disabling FFmpeg frame duplication fixes the
reference. No pixel tolerance or native behavior was weakened.
A warm 16×4096-byte host packing check allocates 128 bytes versus 69,168 bytes
for the previous staging/offset-copy body; this is not an end-to-end speed claim.

All 115 source files parse. Source shrinks 54,828 → 54,533 physical lines;
implementation lines shrink 23,347 → 23,340. The small implementation-line delta
reflects expansion of formerly compressed native initialization into shared,
reviewable helpers; the physical reduction includes prose cleanup. These are
not claims of hundreds of deleted implementation lines. Replacement fault
injection, asynchronous video retirement and H.265 DISTINCT remain open.
