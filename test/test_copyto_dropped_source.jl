# A device-to-device `copyto!` whose SOURCE is dropped before the batch submits.
#
#     copyto!(dst, Adapt.adapt(backend, host_array))   # nothing holds the source
#
# The temporary had no reference left after the call, so the GC collected it, its
# `DataRef` released, the pooled block went back — and the recorded
# `vkCmdCopyBuffer` still named it. `submit!` caught it as
#
#     a use-after-free: the buffer's state was no longer ALIVE at submit
#
# Found on the Vulkan backend, and the history below is its; what is asserted —
# the copy lands, and the temporaries' memory comes back — is every backend's.
#
# ── The cause, kept because it is the part that took the time ────────────────
#
# `cmd_copy_buffer!` pinned both ends and its docstring said a
# `VkManagedBuffer` needs no help from the caller. It pinned what it was GIVEN,
# and `copyto!` handed it `src.buf[]` — already dereferenced — so the only `pin!`
# it could reach was the `VkManagedBuffer` one, which pushes the object into
# `batch.pinned` and makes it REACHABLE. Reachability is not what decides
# whether the memory is still ours; the `DataRef` refcount is, and the array's
# finalizer releases that.
#
# `pin!(::LavaArray)` is the level that does both halves — retain the `DataRef`
# into `batch.pinned_refs`, take a buffer pin — and `release_pinned_refs!` drops
# both at `reclaim_batch!` or on a failed submit. Every kernel argument has
# always had exactly that lifetime via `LavaAdaptor`. Only the copy path pinned
# one level too low, so the fix is to pin the arrays in `copyto!`, where they are
# still in scope.
#
# Worth remembering: a first diagnosis concluded the retain/release list did not
# exist and that this needed its own pass over the lifetime layer. It did exist.
#
# ── Scope ────────────────────────────────────────────────────────────────────
#
# Host-side copies were never affected: `copyto!(device, host)` and its inverse
# go through `copy_buffer!`, which records AND flushes, so nothing can be
# collected between the record and the submit. Only device→device from a value
# nobody holds. That is why the rest of the suite never hit it, and why this file
# has to exist rather than being covered by a render test.

using Test, Mantle, KernelAbstractions, Adapt
import KernelAbstractions as KA
include(joinpath(@__DIR__, "testbackend.jl"))

"The bug: the source is a temporary with no reference after the call."
function copy_from_dropped_source(n = 4096)
    backend = TESTBACKEND
    dst = KA.allocate(backend, Float32, n)
    KA.fill!(dst, 0f0)
    copyto!(dst, Adapt.adapt(backend, fill(3f0, n)))   # source unreferenced from here
    GC.gc(true); GC.gc(true)
    KA.synchronize(backend)                            # Vulkan's submit! asserts it here
    return count(==(3f0), Array(dst))
end

"The same copy with the source held — the control, and what always worked."
function copy_from_held_source(n = 4096)
    backend = TESTBACKEND
    dst = KA.allocate(backend, Float32, n)
    KA.fill!(dst, 0f0)
    src = Adapt.adapt(backend, fill(3f0, n))
    copyto!(dst, src)
    GC.gc(true); GC.gc(true)
    KA.synchronize(backend)
    GC.@preserve src nothing
    return count(==(3f0), Array(dst))
end

"""
Bytes the device's pool has handed out and not had back, once everything dropped
so far has been collected and the device has finished with it.

The collection, the launch and the waits are what make that "so far": a dropped
array is handed back by its finalizer, a backend runs the destroys it has been
asked for where it does its own bookkeeping (a launch, and the synchronize after
it), and the pool releases a retired region once the device is past it.
"""
function livebytes(dev, scratch)
    GC.gc(true)
    fill!(scratch, 0f0)
    KA.synchronize(TESTBACKEND)
    sp = Mantle.pool(dev)
    Mantle.reclaim!(sp, dev)
    Mantle.waitidle(dev)
    Mantle.reclaim!(sp, dev)
    return sum(b -> sum(((lo, hi),) -> hi - lo, b.live; init = 0),
               Iterators.flatten(values(sp.blocks)); init = 0)
end

@testset "device→device copyto! from a dropped source" begin
    # Both arms, because a fix that pinned nothing and a fix that pinned
    # everything for ever would each pass one of them alone.
    @test copy_from_dropped_source() == 4096
    @test copy_from_held_source() == 4096

    # The pin has to be RELEASED, not merely taken. A retain with no matching
    # release shows up here: the regions those temporaries sat in would never
    # come back, and the bytes the pool has handed out would climb once per
    # iteration. Measured on the device's pool, which is where a backend array
    # lives when the backend pools them; one that does not (Metal.jl's own
    # arrays) leaves the pool untouched and this reads equal.
    dev = Mantle.Device(TESTBACKEND)
    scratch = KA.allocate(TESTBACKEND, Float32, 1024)
    baseline = livebytes(dev, scratch)
    for _ in 1:20
        copy_from_dropped_source(1024)
    end
    @test livebytes(dev, scratch) <= baseline
end
