# GJK/EPA kernels over explicit 1-based pairs and core affine transforms.
"""ContactRecord

Pair-indexed contact: 1-based UInt32 grain indices i/j, EPA closest-face normal
n_hat, world-space point p on shape A and penetration depth. Both grain lists
store identical orientation. Plain 36-byte data; see EPAResult for normal direction."""
struct ContactRecord
    i::UInt32
    j::UInt32
    n_hat::Vec3f
    p::Vec3f
    depth::Float32
end

"""NO_CONTACT

Zero-depth sentinel for separated pairs. converged=true means no EPA work
was needed; check positive depth before interpreting convergence failures."""
const NO_CONTACT = EPAResult(Vec3f(0f0, 0f0, 0f0), 0f0, Vec3f(0f0, 0f0, 0f0), 0, true)

"""narrow_phase_kernel(transforms, pairs, shape, results)

One invocation per 1-based (i,j) pair: run GJK/EPA and write EPAResult or
NO_CONTACT. Transforms are row-major NTuple{12,Float32}; both bodies use shape.
Require length(results)==length(pairs) and valid pair indices."""
function narrow_phase_kernel(
        transforms,
        pairs,
        shape,
        results)
    k = KI.get_global_id().x
    # The launch covers whole workgroups; guard the tail.
    k <= length(pairs) || return nothing
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
    return nothing
end

# Reserve a unique slot; consumers read contacts only after dispatch completion.
@inline function append_contact!(counters, contacts, grain, max_contacts::Int32, rec)
    slot = Atomix.@atomic :monotonic counters[grain] += UInt32(1)
    if slot <= max_contacts % UInt32
        idx = (Int32(grain) - Int32(1)) * max_contacts + Int32(slot)
        @inbounds contacts[idx] = rec
    end
    return nothing
end

"""narrow_phase_contacts_kernel(transforms, pairs, shape, counters, contacts, max_contacts)

Run GJK/EPA and append each positive-depth contact to both grains' lists, in
grain-major fixed slots; requires length(contacts) >= length(counters) *
max_contacts and valid pair indices. Counters are zero-initialized UInt32 attempt
counts; live counts clamp to max_contacts and overflow drops records. The atomic
increment only reserves a slot: read counters and records after the dispatch
completes, not from concurrent invocations."""
function narrow_phase_contacts_kernel(
        transforms,
        pairs,
        shape,
        counters,
        contacts,
        max_contacts::Int32)
    k = KI.get_global_id().x
    # A lane past the last pair would read a garbage pair and bump two garbage
    # counters.
    k <= length(pairs) || return nothing
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
            append_contact!(counters, contacts, i, max_contacts, rec)
            append_contact!(counters, contacts, j, max_contacts, rec)
        end
    end
    return nothing
end
