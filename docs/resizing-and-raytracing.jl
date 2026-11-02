# Arrays, growth and acceleration structures: target design sketch.
# app_* names are omitted consumer operations/kernels.
# Not executable. Public signatures: api.md; full protocols: execution-contracts.md.
# internals.jl owns graph/region compilation, command DAGs, locks and submission.

# A. Public use: explicit operation targets are independent of storage lifetime.
dev = Device()
g = Graph(dev)
positions = MantleArray(g, zeros(Float32, 3, 1024)) # initialized persistent state
velocity = MantleArray(dev, zeros(Float32, 3, 1024))
workspace = MantleArray(g, Float32, 1024; scratch=true)
# Persistent construction initializes once; explicit fill!/copy nodes run per replay.
# Scratch data construction instead initializes each execution of its owning region.
dispatch!(g, app_advance!, (positions, velocity), launchrange(positions))

# Persistent leaves can be named by another graph with no payload/size copy.
consumer = Graph(dev)
checksum = MantleArray(consumer, zeros(Float32, 1))
dispatch!(consumer, app_checksum_kernel!, (checksum, positions), 1)
# Consumer's node retains the SAME positions root after free!(g).
# Using workspace in consumer rejects its scratch lifetime escape before mutation.

# GPU size has one query API, and one standard mutation API.
function publish_count!(active, count)
    if KI.get_global_id().x == 1
        resize!(active, count[1])
    end
    nothing
end
active = MantleArray(g, Int32, 0; capacity=1024)
# app_compact! is an illustrative application graph builder; its implementation
# uses flags/scan/checked scatter then publishes resize!(active, selected_count).
# length(active) inside a following kernel is the prepared live count.
# launchrange(active) is launch geometry, not a second metadata-query API.

# B. Logical array state plus canonical allocation ownership.
# Ordinary Mantle views share this logical state; independent imported headers
# can share allocation_owner/payload synchronization without merging dimensions.
mutable struct ArrayState <: Resource
    storage::Storage                  # Region/Committed/Reserve/Evicted/Imported/etc.
    requested_dims::Tuple
    applied_dims::Tuple
    capacity::Int                     # logical-root element units; back before descriptor publication
    lengthsource::LengthSource        # HostLength / DeviceLength
    resizable::Bool
    cell::CellStorage                 # Cell, or plan-owned fixed scratch metadata
    lifetime::Lifetime                # Persistent / RunLocal; runtime property
    scratch_owner::WeakRef            # owning Graph/run, not a different lifetime per Group
    hostsource::Union{Nothing,AbstractArray}
    allocation_owner::AllocationOwner # canonical payload identity/byte bounds + physical owner
    sync::SyncState                   # canonical payload synchronization for data aliases
    holds::HoldCount
    dependents::Dependents
    readers::ReaderSet                # immutable counted composite backreferences
end

struct Imported
    allocation::ExternalAllocationHold
    offset::Int
    bytes::Int
end
# Root registry key includes native allocation/context/generation; never pointer alone.
# Imported subviews share root+sync and retain allocation owner, not a mutable header.
# Imports cannot be moved/evicted/arena-aliased or resized beyond this bound.

# Fixed scratch has plan-owned placement. Resizable scratch uses plan-owned metadata
# whose address/initial dims are reset at run entry. It must be initialized on each
# executable path before content reads; metadata cannot retain a prior run's result.
# Graph replacement retires old placement but never copies/resets persistent roots.
# Unplaced/evicted capacity is a backing requirement, not resident-byte accounting.
# Placement/restore backs it before publishing descriptors; virtual spare is excluded.

# C. Scheduled host changes.
abstract type Change end
struct Resize{E} <: Change
    sequence::UInt64
    target::ArrayState
    dims::Tuple
    keepbytes::E                     # known bytes or prepared min(old live bytes, new bytes)
end
struct Store{M} <: Change
    sequence::UInt64
    target::ArrayState
    upload::Region
    mapping::M                        # actual root coverage, not strided bounding span
end
struct BuildAccel <: Change
    sequence::UInt64
    target::AccelStructure
    mode::BuildMode
    geometry_version::GeometryVersion
end

function Base.resize!(a::MantleArray, dims::Tuple)
    checkroot(a)                      # view resize rejects
    validate_dimensions(a, dims)      # checked product/ranges, readonly and imported bound
    queuechange!(a.state) do snapshot
        Resize(nextsequence(a), a.state, dims, preservedbytes(snapshot, dims, sizeof(eltype(a))))
    end
    a
end
Base.resize!(a::MantleArray, n::Integer) =
    resize!(a, growlast(requested_or_explicit_shape(a), n))
# preservedbytes retains an ordered prepared extent when the old live size is GPU
# written; it must not compute a host Int from stale requested_dims. The shared
# copy recipe prepares the old descriptor and copies min(old live bytes,new bytes)
# before publishing new dimensions/address. Known extents use native copies;
# unknown extents use the core bounded copy kernel or a qualified native equivalent.
# requested_or_explicit_shape never downloads a GPU count under a lock.
# Host lengthsource queries of device metadata use an explicit snapshot path.

# Device lowering of Base.resize! is compiler-owned. A shape-writer effect:
#   validates 0 <= product(dims) <= capacity;
#   updates that invocation's local descriptor;
#   writes only size metadata through its inferred SizeWrite capability;
#   leaves storage address unchanged, allocates nothing, and is not a grid barrier.
# Other invocations see entry descriptors. Following dispatches see prepared dims.
# The capability is ABI-internal, not a public metadata-pointer argument.
# A view cannot publish root dimensions. Checked invalid sizes signal an error;
# @inbounds does not grant an allocation. No clamping or implicit retry is added.

function prepare_changes!(uses)
    snapshots = locked_snapshots(uses)  # include pending/member/namer versions
    prepared = Prepared()
    for snapshot in snapshots
        prepare_storage_and_commands!(prepared, snapshot)  # outside record locks
    end
    prepared
end

function apply_changes!(target, uses, prepared, ownrecipe)
    lockexpandeduses(uses) do locked
        validatesnapshots(locked, prepared) || return Again()
        # Missing or outgrown prepared capacity returns before claim.
        steps = ordered_steps(locked, prepared)
        commands = lower_core_recipes(steps, ownrecipe)
        # Commands have explicit IDs/effects/predecessors, including:
        # copy old contents -> metadata publish -> descriptor prepare -> consumers;
        # flags -> scan -> checked scatter -> count publish;
        # geometry preparation -> native build -> query/trace.
        validate_command_dag!(commands)
        token = submitprepared!(target, commands, reservedpublication(locked))
        commit_accepted_changes!(locked, steps, token)
        token
    end
end
# Claim/accept/publication failure rules are execution-contracts §2.
# No live root/membership reads occur inside native record/emit functions.
# An accepted sparse bind followed by later preparation failure is tracked/retired;
# it cannot be rolled back or released as unused tokenless storage.

# Store source is copied once at the host call into upload memory before locking.
# Under root locks validate known destination mapping against the ordered state.
# GPU-written logical bounds defer the full-range check to prepared GPU metadata.
# Only known-valid extents can select inline emitstore!/copy; otherwise use the
# shared checked-store recipe (not stale requested_dims or an unchecked scatter).
# Consecutive compatible resizes coalesce; stores coalesce only by actual coverage
# and never across a held resize/deferred version. Changes keep device-wide order.

# D. Capacity and dependencies.
# Capacity plans physical backing and maximum scratch. It never determines the
# kernel-visible live length. Host growth:
#  - within bound: metadata write only;
#  - qualified QueueMapping reserve: acquire pages before locks, bind with token;
#  - otherwise: prepare larger committed region, ordered copy, metadata publish.
# CUDA/HIP host VMM alone does not provide queue-ordered mapping; initially committed.
# Trim commits reduce backed capacity; coherent GPU metadata checkpoints precede
# host page-binding preparation where needed. Restore publishes HostCopy.shape,
# not stale requested_dims, before later ordered Resize steps. Unknown GPU logical
# store bounds use a prepared checked kernel, never unconditional inline/copy.
# Shrink retains capacity; explicit array trimming is deferred, not initial public API.
# Handle moves rerecord only dependent parts; selection-domain growth re-emits the
# retained RegionNode recipe, candidate command sequence and scratch together.

# Internal SizedBy(f,a) evaluates requested host shape or the GPU size's CAPACITY.
# It is not a public alternate size query. Algorithm scratch is explicitly scratch.
function reduction_recipe!(sink, result, input)
    partial = scratch_array(sink, eltype(result), SizedBy(nblocks, input))
    emit_partials!(sink, partial, input)
    # Partial's live blocks are derived from the same input descriptor, not assumed
    # to equal its capacity. Unwritten scratch tail cannot participate in reduction.
    emit_finish!(sink, result, partial, input)
end

# E. Stable compaction without a public size handle.
function compact_recipe!(sink, active, particles, predicate)
    flags = scratch_array(sink, UInt32, SizedBy(identity, particles))
    offsets = scratch_array(sink, UInt32, SizedBy(identity, particles))
    emit_flags!(sink, flags, particles, predicate)
    emit_exclusive_scan!(sink, offsets, flags, particles)
    # Capacity-bounded descriptor is an internal view of the SAME root allocation.
    # It cannot escape this producer or become another hazard/ownership identity.
    producer = checked_producer_descriptor(sink, active)
    emit_scatter!(sink, producer, particles, flags, offsets)
    emit_count_publish!(sink, active, offsets, flags, particles) # one item, even input=0
end
# emit_count_publish!'s body uses resize!, whose inferred effect schedules the
# subsequent descriptor preparation. Consumers depend on both payload and size.
# A GPU-produced local view is ordinary view(parent, GPU bounds) INSIDE a kernel.
# Multi-kernel consumers retain their bounds in descriptors/slots; no integer-only
# host helper is allowed to recover them by a hidden scalar readback.

# F. Acceleration structures: one core owner, with geometry-version dependencies.
# Logical scene state has no ray-type stride. Table/layout-specific native binding
# views have independent accepted-version tracking and share logical geometry.
# Per-geometry/instance selectors are ordinary resources, never native SBT offsets.
mutable struct TLAS{D<:Device} <: AccelStructure
    device::D
    hardware::Union{Nothing,Bool}      # explicit requirement, not a per-backend policy
    members::MemberSet
    ranges::RangeTable
    geometry_versions::GeometryVersions
    bindings::NativeBindings          # layout/path-keyed native objects and versions
    sync::SyncState
    holds::HoldCount
end
# Each native binding owns storage, builtcount, accepted-build versions and dirty
# sources; it stores scratch REQUIREMENTS, not a permanently reserved workspace.
# Core places build/refit workspace by operation liveness in its arena/temp pool.
# Logical scene versions are not cleared by one binding's build.
# Keys/geometry/instance counts stay runtime values; no graph-sized type parameters.
# BLAS has the same version/lifetime policy over its geometry, no backend copy.
# A run-local composite explicitly opts into scratch lifetime if it names scratch
# members. Persistent composites reject scratch members before taking holds.
# Structures/ranges retain the geometry allocation and record dependencies.

function pendupdate_locked!(structure, source, version)
    update_source_version!(structure.geometry_versions, source, version)
    mark_binding_views_stale!(structure.bindings, source, version)
    notify_naming_plans!(structure)
end
function build_accepted_locked!(structure, binding, required, count, token)
    record_accepted_build!(binding, required, count, token)
    remove_covered_dirty_versions!(binding.dirty, required)
end
# A graph writing geometry declares no manual refit just to avoid stale queries.
# Core inserts an update at its first dependent read, even in the same segment.
# Graph external writes publish transitive dirty versions per accepted prefix.
# No blanket "this graph builds it somewhere" exclusion is legal.
# A later failed build leaves the earlier write dirty for another consumer.

# Member changes:
# source positions/vertices -> eligible Update;
# count/topology/index changes -> Build under native rules;
# hidden range -> mask zero; deleted range -> inactive (Build);
# growth -> prepared storage/scratch replacement and token-safe retirement.
# Host-known totals build exactly. GPU totals use capacity count with valid mask-zero
# TLAS padding; BLAS padding uses degenerate geometry, never NaN activity toggles.
# Every native Update preserves builtcount/activity/topology constraints.
# Native exact-device-count alternatives require independent legality qualification.

function emit_build_recipe!(sink, structure, binding, required)
    offsets = emit_offsets!(sink, structure, binding, required)
    records = emit_records!(sink, structure, binding, required; after=exits(offsets))
    build = emit_native_build!(sink, frozen_build(binding, required);
                               after=exits(records))
    build
end
# These are core command-DAG entries. Backend emitbuild! receives immutable prepared
# state and complete dependencies. It cannot compute membership, dirtiness or order.
# Native representation-only builds on disabled iterations must be proven valid/
# observationally equivalent; logical data writes still obey the condition.

# G. Images and consumer composition.
# Sampled/storage/subresource views retain canonical Image root identity. A native
# generation has a fresh table entry on restore/resize; old entry/image retire with
# every old use. Sampler weak interning and lifetime are independent from images.
# See execution-contracts §§7/8/11/16 for formats, filters, feedback and resolves.

simulation = Graph(dev)
state = MantleArray(simulation, zeros(Float32, 3, 1024))
solver = Group(simulation, "timestep")
solver_state = (state, velocity)
# Application method emits nested Newton/Krylov/line-search regions through its
# explicit operation targets. No backend-specific solver copies or graph partitioning.
app_declare_solver!(solver, solver_state)
repeat!(solver, 4)

frame = Graph(dev; priority=1, share_scratch=false)
# A renderer's root uses order against simulation, while independent frame work can
# overlap subject to budget/hazards. Priority does not secretly choose arena sharing.
app_declare_render!(frame, state)
run!(simulation)
run!(frame)
