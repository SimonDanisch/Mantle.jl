# Design sketch; not executable runtime code.
# api.md owns signatures; execution-contracts.md owns full failure/lifetime rules.
# Helpers below denote single core algorithms, not backend policy hooks.

abstract type Device <: KA.Backend end
abstract type Resource end
abstract type Node end

struct MantleArray{T,N,D<:Device} <: AbstractGPUArray{T,N}
    device::D
    state::ArrayState
    view::Union{Nothing,ArrayView{N}}
end
struct ArrayView{N}
    offset::Int                       # byte offset from canonical allocation root
    dims::NTuple{N,Int}
    strides::NTuple{N,Int}             # byte strides; native element strides convert explicitly
end

# ArrayState/storage/resize are in resizing-and-raytracing.jl.
# Data hazards/ownership use the canonical allocation root; logical-size metadata
# uses its own header identity/epoch. Independent imported headers may share data
# without sharing dimensions. Ordinary Mantle aliases retain their logical parent.
# One frontier algorithm handles both data-byte and metadata uses.
# Persistent, scratch and imported are runtime storage/lifetime facts.
# Typed payloads preserve recipe/argument ABI, not graph topology or resource IDs.
# One indexed core walk uses typed function barriers; baked replay walks no nodes.
struct RegionNode{F,A,O} <: Node
    recipe::F                         # same emission body used by eager work
    args::A
    outputs::O                        # stable public roots, not replacement temporaries
    origin::Origin
end

struct LoopNode{B,C} <: Node
    body::B                           # RegionId/indexed region, NOT a tuple type of child nodes
    maximum::Int
    count::C
end
struct Bound{R,E}
    resource::R
    epoch::E                          # descriptor slots belonging to one use epoch
end

# ---------------------------------------------------------------------------
# 1. Explicit targets and atomic declaration
# ---------------------------------------------------------------------------

KA.get_backend(a::MantleArray) = a.device
location(args...) = commondomain(discoveredresources(args))
# location resolves only the owning Device, including callable/keyword resources.
# Arrays never carry a graph/group destination. Reuse them directly in any node.
# adapt(dev,...) handles device/data conversion; Graph/Group adaptation rejects.
# KA.allocate/similar default persistent; graph workspace explicitly names target.

function declare!(target, declaration)
    graph = graphof(target)
    declaring(graph) do
        snapshot = declarationsnapshot(target, declaration)
        # Include all generated metadata prepares, callable captures and child uses.
        # Reject freed/lent/foreign roots, illegal controls, readonly writes and
        # lifetime escape BEFORE publishing any node or changing holds.
        expansion = validateexpand(snapshot)
        reservation = reservepublication(expansion)
        publishdeclaration!(graph, target, expansion, reservation)
    end
end
# A failure before publish leaves nodes/holds unchanged. Allocation failure during
# publication cannot leave half a kernel+prepare pair: publication is reserved first.
# Internal prepares/copies/builds are region children, not forbidden user node kinds.

# Dense same-element-type matrix multiplication is the expanded example.
# Other shapes/operator types require their own qualified recipe methods.
struct MatMulRecipe end
recipe(::typeof(mul!), ::Tuple{<:MantleArray{T,2},
                              <:MantleArray{T,2},
                              <:MantleArray{T,2}}) where {T} = MatMulRecipe()

LinearAlgebra.mul!(C::MantleArray, A::MantleArray, B::MantleArray) =
    _mul!(location(C, A, B), C, A, B)
call!(target::Union{Graph,Group}, ::typeof(mul!), C::MantleArray, A, B) =
    _mul!(target, C, A, B)

function _mul!(sink, C, A, B)
    validateuses(sink, (C, A, B))
    validate_mul!(C, A, B)                 # shapes/alias/type policy; no GPU-size download
    operation!(sink, mul!, (C, A, B), (C,))
    C                                    # this definition explicitly returns C
end

function operation!(target::Union{Graph,Group}, f, args, outputs)
    node = RegionNode(recipe(f, args), args, outputs, Origin(f, args))
    declare!(target, node)                 # retain intent; no kernel launch here
end
function operation!(dev::Device, f, args, outputs)
    node = RegionNode(recipe(f, args), args, outputs, Origin(f, args))
    executeeager!(materialize(eagercontext(dev), node))
end

# Minimal Candidate layout for this illustration; production metadata also
# carries version, cost and workspace. These are code-owned values, not a registry.
struct Candidate{F,R}
    emit::F
    requires::R
end

# This method connects MatMulRecipe to its implementations. Candidate metadata
# also includes version/cost/workspace; only one candidate is expanded here.
candidates(::MatMulRecipe) = (
    Candidate(emit_generic_matmul!, generic_matmul_requirements),
)
# Tiled/tensor/native-library candidates are additional code-owned entries using
# the SAME selector; no candidates(dev,f) policy table in each backend.

function materialize(context, node::RegionNode)
    impl = select(candidates(node.recipe), enabledfacts(context),
                  node.args, requirements(context))
    # select filters requirements/precision/layout/alias/workspace/control first,
    # then uses common cost/tuning evidence. Unknown/illegal recipes reject.
    impl.emit(context, node.args...)
    context                              # whole emitted DAG, not last emitter return
end

function emit_generic_matmul!(context, C, A, B)
    emit_kernel!(context, generic_matmul_kernel!, (C, A, B), launchrange(C))
end
# emit_kernel! writes into the private compiled command DAG; it neither appends
# declarations to the user's Graph nor immediately submits a kernel.

function generic_matmul_kernel!(C, A, B)
    id = KI.get_global_id()
    i, j = id.x, id.y
    i <= size(C, 1) && j <= size(C, 2) || return
    value = zero(eltype(C))
    for k in 1:size(A, 2)
        value += A[i, k] * B[k, j]
    end
    C[i, j] = value                       # empty inner dimension writes zero
    nothing
end
# Arithmetic illustration only: reuse/refactor the existing shared GPUArrays/KA
# kernel body in production rather than keep another generic GEMM copy.
# Requirement predicates/shape preparation/dispatch mechanics are omitted helpers;
# they are shared core code, not per-device numerical implementations.

Base.copyto!(dst::MantleArray, bc::Broadcast.Broadcasted) =
    _broadcast!(location(dst, bc), dst, bc)  # dotted syntax stays eager

call!(target::Union{Graph,Group}, ::typeof(copyto!), dst::MantleArray,
      bc::Broadcast.Broadcasted) = _broadcast!(target, dst, bc)

function _broadcast!(sink, dst, bc)
    validateuses(sink, (dst, bc))
    operation!(sink, broadcast!, (dst, bc), (dst,))
    dst
end
# The known broadcast!(f,dst,args...) frontend shares this implementation too.
# A library helper emits call!(target,known_function,args...) and raw dispatch!
# explicitly. Ordinary Julia/library code is not automatically captured.

Base.Array(a::MantleArray) = fetch(snapshotrecipe!(a.device, a))
call!(target::Union{Graph,Group}, ::Type{Array}, a::MantleArray) =
    snapshotrecipe!(target, a)
# Same private copy/hold/visibility recipe. A recorded Array call returns its
# declared ReadbackFuture; the ordinary blocking constructor fetches internally.
# Readback(previous_future) re-requests that output; no Readback(array) operation.

# ---------------------------------------------------------------------------
# 2. Compile: structured regions -> one explicit command DAG
# ---------------------------------------------------------------------------

function compile(g::Graph)
    s = snapshotspecification(g)
    materialize_regions!(s)            # retained operation recipes; select now
    normalizecontrols!(s)             # nested loops/conditions still present
    formpasses!(s)
    applyoptinrewrites!(s)
    choose_region_encodings!(s)        # core selects native/capacity/expanded construction
    lower_command_recipes!(s)          # EVERY internal command has identity and edges
    ensurebuilds!(s)                   # geometry version -> build -> AS read
    prepare_metadata!(s)               # inferred resize! writes -> subsequent descriptors
    dependencies!(s)
    costs!(s)
    schedule!(s)                      # kind/family and logical lane
    hostcuts!(s)
    presentspan!(s)
    revalidate_command_dag!(s)         # inserted/split/moved commands participate
    liveness!(s)
    renderstores!(s)
    place!(s)
    sharing!(s)
    revalidate_schedule!(s)
    barriers!(s)
    submissions!(s)
    summary!(s)
    bindings!(s)
    Plan(s)
end

# Capability facts select a native loop/conditional region before fallback expansion.
# Portable bounded expansion resets the stop latch per enclosing invocation,
# gives iterations distinct argument/control slots, and connects every body tail
# before its next head. Internal PrepareNodes share the enclosing predicate.
# Native regions have the same externally visible effects/control contract.
function expand_bounded_loop!(sink, loop::LoopNode)
    control = startcontrol!(sink, loop)
    previous = exits(control)
    for i in 1:loop.maximum
        head = emit_loop_head!(sink, control, i, loop.count; after=previous)
        body = materializeiteration!(sink, loop.body, control, i; after=exits(head))
        previous = exits(body)         # all tails, including metadata/library work
    end
    finishcontrol!(sink, control; after=previous)
end

# A false control skips logical writes and RNG advancement. A native AS build
# may run unconditionally only when core proves representation-only equivalence
# and native legality; software build kernels can use ordinary predication.
# Conditional render clear lowering is a shared operation recipe (§16).

function dependencies!(s)
    frontier = AccessFrontiers()
    for command in commands(s)
        uses = mergeduses(command)     # include captures, aliases, metadata and slot epochs
        before = snapshot(frontier, uses)
        predecessors = conflictingfrontier(before, uses)
        # Check all uses first; aliases and AnyWrite cannot create self edges.
        addedges!(s.dag, predecessors, command)
        after = advancefrontier(before, command, uses)
        mergefrontier!(frontier, conditional(command) ? unionpaths(before, after) : after)
    end
    validateacyclic!(s.dag)
end
# Read depends on prior writers; write depends on writers and outstanding readers.
# Conditional union retains effects needed on either path. Unordered atomic groups
# still conflict with ordinary accesses. Proven spans split interval frontiers;
# unknown spans use the root. A linear chain does not retain all historical uses.
# The same hazard algebra supplies barriers and external-use conflict summaries.

function sharing!(s)
    for (old, new) in aliasedlifetimes(s.placement)
        prior = maximalusers(s.dag, old)
        following = minimalusers(s.dag, new)
        addfrontierorder!(s.dag, prior, following)
    end
end
# addfrontierorder! adds all required pairs or a join/entry equivalent.
# A single last index in a topological list is NOT a completion frontier.
# Reuse remains legal when an earlier-listed reader is slow on a different queue.

function ensurebuilds!(s)
    versions = geometryversions(s)
    for access in ordered_as_accesses(s)
        required = requiredgeometry(access, versions)
        coveredbyacceptedbuild(access, required) ||
            insertderivedbuild!(s, access, required)
    end
end
# Fixed graph writes compile to explicit write -> build/refit -> trace/query edges.
# Dynamic members retain conservative membership dependencies. Native commands are
# frozen from prepared snapshots. Graph edits/member changes invalidate relevant
# recipes/recordings, not a second per-backend dirty-state algorithm.

function costs!(s)
    for command in commands(s)
        command.estimate = estimatewithprovenance(s.device, command)
        enforcevalidatedconstraints!(command)
        # Estimates guide scheduling/eligible partitions, never prove device loss.
        partition_if_legal!(s, command)
    end
end

# ---------------------------------------------------------------------------
# 3. Recording and native boundary
# ---------------------------------------------------------------------------

function record!(pl::Plan)
    for segment in segments(pl)
        builder = Builder(pl.device, segment.kind)
        for command in commands(segment)
            emit!(builder, command, Deps(predecessors(command), barrier(command)))
        end
        installrecording!(segment, finish(builder))
    end
end
# Builder receives immutable command descriptors and stable command IDs.
# A multi-command head/prefix/build recipe already exists in the core DAG.
# Native DAG emitters translate predecessor IDs; linear APIs encode the derived
# barriers. Missing an internal edge is an error, not permission to infer order.
# Inter-segment predecessors are represented by segment waits, never dropped.

function emitdispatch!(b::CudaBuilder, node, cmd::CompiledDispatch, deps)
    predecessors = local_native_predecessors(b, deps)
    handle = cuGraphAddKernelNode(b.graph, predecessors, kernelparams(cmd))
    register_native_command!(b, cmd.id, handle)
end
function emitdispatch!(b::VulkanBuilder, node, cmd::CompiledDispatch, deps)
    emitbarrier!(b, deps.barrier)
    bindcompiled!(b, cmd)
    emitlaunch!(b, cmd.extents)
end

# record! may acquire prepared native/host command objects, but no device memory,
# compile, upload, submit, pressure or live ArrayState/membership read.
# Library native recording consumes a closed eligible recipe and frozen bindings.

# ---------------------------------------------------------------------------
# 4. Submission, accepted prefixes, and coherent reads
# ---------------------------------------------------------------------------

function submit!(channel, parts, waits, signals, publication)
    packet = preparesubmit(channel, parts, waits, signals)   # fallible preclaim
    st = channelstate(channel)
    lock(st.lock) do
        checkhealthy(st.domain)
        token = Token(channel, nexttoken!(st))
        try
            submitnative!(channel, token.value, packet)
            publishaccepted!(publication, token)             # reserved/no allocation
        catch err
            poison!(st.domain, err)
            quarantine!(st.domain, packet, token)
            rethrow()
        end
        token
    end
end
# Enclosing ordered plan/root locks remain held through accepted publication.
# An enqueue exception after claim has ambiguous acceptance: no rollback/recycle.
# Prepared-but-unused objects clean up; accepted objects retire by actual tokens.

function pendreaders_locked!(written, version, publication)
    for root in written, reader in transitive_readers(root)
        penddirtyversion!(reader, root, version, publication)
    end
end
function publishbuild_locked!(structure, buildversion, token)
    acceptbuildversion!(structure, buildversion, token)
    consumeonlycoveredversions!(structure, buildversion)
end
# Never suppress dirtiness merely because builtby(plan) contains the structure.
# A write after that build, or before an unsubmitted later build, remains dirty.

function run!(g::Graph)
    lockgraph(g) do
        checklive(g)
        retrycompletedpresent!(g) && return Submitted()
        applywindows!(g)
        planinvalid(g) && replaceplan!(g)
        updatepieces!(g)
        runplan!(g.plan)
    end
end
# runplan! is core's prepare/lock/validate/claim/accept/publish stage loop.
# It preserves earlier accepted stages on failure. Retry waits release complete
# lock sets; WaitFor releases graph/arena holds before waiting/restarting stage 1.
# Run roots wait on the graph's previous-run snapshot across all lanes.
# Presentation happens only after all stages accepted and outside record locks.

function replaceplan!(g)
    new = compile(g)
    record!(new)
    registervalidated!(new)
    old = g.plan
    inheritlastrun!(new, old)
    rebindpendingoutputs!(g, new)      # Graph owns requests; old-plan retirement cannot cancel them
    g.plan = new
    retireoldplan!(old, new)
end
# If compile/record/register fails, discard only the unaccepted NEW plan and
# leave g marked for retry. Persistent output arrays and RNG state never reset.

function waitfor(t::Token; timeout=Inf)
    checkhealthy(t.state.domain)
    waitfor(t.channel, t.value, timeout) || throw(WaitTimeout(t, timeout))
    t
end
# Native wait errors still propagate DeviceLost. Deadline expiry leaves all tokens/
# holds valid and pending; it does NOT poison a healthy domain.
# Native driver waits are GC-safe, outside plan/root/channel/pool/window locks.

# Host read:
#  - capture pending/applied dimensions, view mapping, versions and writer tokens
#    under ordered naming-plan/root locks; allocate outside locks;
#  - apply changes/copy payload+device metadata in one ordered submission;
#  - wait outside locks, invalidate host mapping, use captured shape;
#  - direct unified memcpy revalidates an idle snapshot and excludes new writers.
# Readback futures use recorded snapshot nodes and do not submit at fetch.
# Readback(previous) copies only the stable output descriptor, not a future chain.
# Completed requests leave the Graph pending set; live futures own captured results.
# Detailed retirement, capacity-retry and HostCopyJob protocols remain in §§1/12.

# ---------------------------------------------------------------------------
# 5. Ownership, pool, and queues
# ---------------------------------------------------------------------------

# Creator, graph, composite, external lease and submitted future are explicit holds.
# Zero holds cannot resurrect. free! marks creator aliases unusable; graphs holding
# the root remain valid. Data+Cell+descriptors retire against every last use.
# Graph/Group references live on node declarations/uses, never array device handles.
# Scratch metadata points only into its live plan/run; exports reject before mutation.
# Imports hold the allocation owner and drop that hold, never rawfree foreign bytes.

# One acquire! pressure path, in order:
# drop idle native caches -> unmap idle scratch -> trim empty blocks -> collect ->
# wait retirements -> bounded pageable eviction -> OutOfDeviceMemory.
# Direct restore retains §12/F74's debug-first/no-speculative-staging rule.
# QueueMapping and host VMM are separate facts; initially committed CUDA/HIP arenas.

# Queue mapping:
# schedule! assigns family/kind + logical lane, run maps lanes to native endpoints.
# Endpoint aliases share ChannelState lock/timeline/counter/failure domain.
# priority selects urgency, share_scratch selects arena sharing independently.
# Independent placement can overlap equal-priority graphs; one graph stays serial.
# Proven disjoint spans may avoid data conflicts while metadata uses still order.

# Lock order:
# graph -> arena holds -> window -> plans -> root resources -> ChannelState.
# Plan/root IDs use (device identity, local ID) when explicit transport spans devices.
# Pool/cache/table locks are leaves. No callback/compile/pressure/GPU wait under
# plan/root/channel locks. Finalizers enqueue idempotent releases only.
# Explicit foreign stream leases never hold a Mantle lock across vendor code.

# api.md §4 is the native vocabulary; execution-contracts defines its extensions.
# A backend never walks a Mantle plan, decides lifetimes, or chooses a fallback.
