# The CPU-side view of a built acceleration structure.
#
# `AdaptedAccel` (was `HWAdaptedAccel`) reaches its backend's HWTLAS through a
# type parameter, so it never names one — which is why it is here and the
# `HWTLAS` it wraps is only declared.
"""
    AdaptedAccel{H, T, O, Tri, S} <: Raycore.AbstractAdaptedAccel

GPU-adapted form of `VulkanTLAS`.  Carries the kernel-side data needed by
`Raycore.closest_hit(::AdaptedAccel, ray)` / `any_hit` (which lower to
`OpRayQueryInitializeKHR` etc.):

  * `triangles` — flat array of all BLAS triangles, concatenated.
  * `offsets`   — per-instance offset into `triangles`, indexed by `gl_InstanceID + 1`.
  * `empty`     — sentinel triangle returned on miss.
  * `hwtlas`    — CPU-side `VulkanTLAS` reference for callers that need it
                  (descriptor binding, sync, RT pipeline path); `nothing` in
                  the kernel-form produced by `Adapt.adapt`.

Constructed by `sync!(hwtlas)` from the live VulkanTLAS state.  The kernel-form
(returned by `Adapt.adapt(LavaAdaptor, accel)`) drops `hwtlas` (Nothing) and
adapts the array fields to `LavaDeviceArray`.
"""
struct AdaptedAccel{H, T, O, Tri, S} <: Raycore.AbstractAdaptedAccel
    hwtlas::H
    triangles::T
    offsets::O
    empty::Tri
    # How the KERNEL reaches the acceleration structure, when the backend needs
    # it in-band rather than bound beside the dispatch.
    #
    # Vulkan binds its TLAS as a descriptor, so a ray query names nothing and
    # this stays `nothing` — a ghost type, so the kernel-side struct is byte-for
    # byte what it was. Metal has no descriptor for an acceleration structure
    # reachable from a Julia kernel: traversal is a linked MSL function that
    # takes the structure through an argument buffer, and this carries the
    # pointer to it.
    scene::S
end

# Four-argument form for a backend that does not need `scene`. Keeping it means
# adding the field did not touch a single existing construction site.
AdaptedAccel(hwtlas, triangles, offsets, empty) =
    AdaptedAccel(hwtlas, triangles, offsets, empty, nothing)



"""
    AccelBuildContext(queue)

Scratch state reused across acceleration-structure builds: the queue the builds
are submitted on, and the objects that must outlive the build.

Was `ASBuildContext` in the Vulkan backend and then an abstract type here. It is
one shared struct: both fields are portable, the second trivially and the first
once the submission channel stopped being per-backend. Building is the expensive
part of ray tracing and every backend wants to amortise it, so the context is
API rather than an implementation detail one of them happens to have.
"""
mutable struct AccelBuildContext{Q,E}
    bq::Q
    # Where the builds are written: the backend's emitter over the closed
    # command buffer `build_accel!` opened for them, submitted on the way out.
    into::E
    preserves::Vector{Any}
end
AccelBuildContext(bq, into) = AccelBuildContext(bq, into, Any[])
