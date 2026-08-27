# GPU compute kernel composing GJK + EPA over pairs of instance transforms.
#
# narrow_phase_kernel: one thread per (i, j) pair index.  Reads transforms[i]
# and transforms[j], runs gjk(); on overlap runs epa() and writes the
# EPAResult into results[k] (where k is the linear pair index).  On
# separation, writes the NO_CONTACT sentinel (depth == 0f0).
#
# This is the P4.4 building block: P4.5 will add a ContactRecord struct +
# atomic compaction; P4.6 will produce the `pairs` buffer from a broad-phase
# rayQuery sweep; P4.7 stitches the three together end-to-end.  For now the
# `pairs` buffer is supplied externally (CPU test harnesses or future
# pair-discovery kernels).
#
# Convention: pair indices are 1-based Julia indices (transforms is a Julia
# AbstractVector, indexed from 1).  Both bodies share the same `shape`
# argument; mixed-shape narrow-phase needs an enum-dispatch design that's
# out of scope for P4.4.

"""
    ContactRecord

Per-contact record produced by the narrow-phase pipeline and consumed by the
XPBD solver (P5).  Layout mirrors the design spec section 3.4:

- `i`, `j`: 1-based grain indices for the contact pair.
- `n_hat`: contact normal pointing from B (j) toward A (i), unit length.
- `p`: contact point on A's surface in world space.
- `depth`: penetration depth (>= 0) along `n_hat`.

Tightly packed (no padding) at 36 bytes; `isbitstype` so it can live in a
GPU-resident `LavaArray`.
"""
struct ContactRecord
    i::UInt32
    j::UInt32
    n_hat::Vec3f
    p::Vec3f
    depth::Float32
end

"""
    NO_CONTACT

Sentinel `EPAResult` written into the result slot for pairs that do not
overlap.  Distinguished from a valid contact by `r.depth == 0f0`; downstream
consumers (P4.5 contact compaction, P4.7 integration test) test on
`r.depth == 0f0` to skip non-contacts.

`converged` is set to `true` because there's nothing to converge -- a future
reader who wants "did EPA fail to converge?" must check `r.depth > 0f0` AND
`r.converged == false` together.
"""
const NO_CONTACT = EPAResult(Vec3f(0f0, 0f0, 0f0), 0f0, Vec3f(0f0, 0f0, 0f0), 0, true)

"""
    narrow_phase_kernel(transforms, pairs, shape, results)

`KernelAbstractions.@kernel`: one thread per pair index `k`.  Reads
`pairs[k] = (i, j)` (1-based indices into `transforms`), looks up the row-
major 3x4 transforms `T_A = transforms[i]` and `T_B = transforms[j]`, runs
`gjk(shape, shape, T_A, T_B)`.  If GJK reports overlap, runs `epa(...)` and
writes the resulting `EPAResult` into `results[k]`; otherwise writes the
`NO_CONTACT` sentinel (`depth == 0f0`).

Preconditions (not validated -- this is a kernel; violations produce wrong
output, not errors):
- `length(results) == length(pairs)`.
- Every `(i, j) ∈ pairs` satisfies `1 ≤ i, j ≤ length(transforms)`.

Output sentinel: `r.depth == 0f0` ⇔ no overlap for that pair.

Both bodies share `shape`.  Mixed-shape narrow-phase (cube vs sphere etc.)
is a future enhancement; for now the kernel is specialized at compile time
on the concrete `shape` type, which lets `support()` inline cleanly into
the GPU compile.

Element types:
- `transforms`:  `AbstractVector{NTuple{12, Float32}}` (matches
  `LavaInstanceRecord.transform` layout).
- `pairs`:       `AbstractVector{NTuple{2, Int32}}` (4-byte ints; halves
  the buffer footprint vs `Int64`).
- `shape`:       `ConvexShape` subtype, e.g. `UnitCube()`.
- `results`:     `AbstractVector{EPAResult}` of length `== length(pairs)`.
"""
# Keeps its CPU variant: `test_narrow_phase_kernel.jl` is a GPU-vs-CPU parity
# test and `cpu=false` deletes the side it compares against.
@kernel function narrow_phase_kernel(
        @Const(transforms),
        @Const(pairs),
        shape,
        results)
    k = @index(Global)
    @inbounds pair = pairs[k]
    i = pair[1]
    j = pair[2]
    @inbounds T_A = transforms[i]
    @inbounds T_B = transforms[j]
    g = gjk(shape, shape, T_A, T_B)
    if g.overlap
        @inbounds results[k] = epa(shape, shape, T_A, T_B, g.simplex)
    else
        @inbounds results[k] = NO_CONTACT
    end
end

"""
    narrow_phase_contacts_kernel(transforms, pairs, shape,
                                  counters, contacts, max_contacts)

Fused narrow-phase + per-grain compaction kernel.  For each pair index `k`,
runs `gjk(shape, shape, T_A, T_B)`; on overlap runs `epa(...)` and atomically
inserts a `ContactRecord` into BOTH grain `i`'s and grain `j`'s slot list.

Layout:
- `counters::AbstractVector{UInt32}` of length `n_grains`.  Each slot is the
  per-grain attempted contact count; the live count is `min(counter, max)`.
- `contacts::AbstractVector{ContactRecord}` of length `n_grains * max`,
  laid out grain-major: grain `g`'s slots occupy
  `(g-1)*max + 1 ... g*max`.
- `max_contacts::Int32` is the per-grain slot limit (typ. 12).

Both copies of the contact carry the same `(i, j)` pair indices (the
record's orientation is *pair-indexed*, not self/other) — the XPBD solver
matches by checking `rec.i == self_grain ? own_A : own_B`.

Counter overflow: if a grain's counter exceeds `max_contacts`, those extra
contacts are dropped (no out-of-bounds writes).  Overshooting counters are
fine for read-side consumers, which clamp via `min(counter, max)`.

Preconditions (kernel; violations → wrong output, not errors):
- `length(counters) >= max(grain id in pairs)`.
- `length(contacts) >= length(counters) * max_contacts`.

Element types:
- `transforms`:  `AbstractVector{NTuple{12, Float32}}`.
- `pairs`:       `AbstractVector{NTuple{2, Int32}}` (1-based grain indices).
- `shape`:       `ConvexShape` subtype.
- `counters`:    `AbstractVector{UInt32}`, **must be zero-initialised**.
- `contacts`:    `AbstractVector{ContactRecord}` of length
                 `length(counters) * max_contacts`.
- `max_contacts`: `Int32`.
"""
# Same as `narrow_phase_kernel`: `test_narrow_phase_contacts.jl` runs it on CPU.
@kernel function narrow_phase_contacts_kernel(
        @Const(transforms),
        @Const(pairs),
        shape,
        counters,
        contacts,
        max_contacts::Int32)
    k = @index(Global)
    @inbounds pair = pairs[k]
    i = pair[1]
    j = pair[2]
    @inbounds T_A = transforms[i]
    @inbounds T_B = transforms[j]
    g = gjk(shape, shape, T_A, T_B)
    if g.overlap
        r = epa(shape, shape, T_A, T_B, g.simplex)
        if r.depth > 0f0
            rec = ContactRecord(UInt32(i), UInt32(j),
                                r.normal, r.contact, r.depth)
            # Atomic slot allocation for grain i.  @atomic returns the new
            # (post-increment) value, which is the 1-based slot we own.
            slot_i = Atomix.@atomic counters[i] += UInt32(1)
            if slot_i <= max_contacts % UInt32
                idx_i = (Int32(i) - Int32(1)) * max_contacts + Int32(slot_i)
                @inbounds contacts[idx_i] = rec
            end
            # Same for grain j.  Both copies carry identical (i, j) pair
            # indices; orientation is preserved.
            slot_j = Atomix.@atomic counters[j] += UInt32(1)
            if slot_j <= max_contacts % UInt32
                idx_j = (Int32(j) - Int32(1)) * max_contacts + Int32(slot_j)
                @inbounds contacts[idx_j] = rec
            end
        end
    end
end
