# Consumer evidence and migration gates

Inspected directly on 2026-10-20. These are source-grounded migrations of the
representative integration boundaries below, not executed ports or a claim that
every numerical kernel/model in these packages was reviewed. Line references
refer to the consumer checkout at review time. Mantle's new API does not exist
at runtime yet. These packages already have real Mantle integrations and acceptance baselines.
The migration gates below concern changed refactor boundaries, not restarting
those ports. Existing numerical/performance evidence is indexed in evidence-index.md.
Pattern tests alone do not close phase 11.

## Source-to-contract map

| Real path | Current behavior / cost or hazard | Target and acceptance gate |
|---|---|---|
| [RayMakie frame_plan!](../../../Makie/RayMakie/src/overlay_rendering.jl:331) | Computes a signature over source, plots, shadow state and extent; frees/rebuilds a frame on mismatch. Rebinds DrawBindings and deliberately leaves window plans unrecorded. Uses depth, FXAA flags, shadow passes and picking metadata, not just flat coloured rectangles. | One graph per screen, per-plot Group pieces and GPURef/array stores. Window generations select recorded per-image work. Preserve colour/depth/metadata attachments and premultiplied alpha. Run existing overlay/picking/FXAA/shadow tests at two sizes, dynamic plot insertion and two screens. RM-01–17 are core gates, not the whole Makie gate. |
| [RayMakie scene render](../../../Makie/RayMakie/src/screen.jl:645) | Explicit Hikari.sync! is required: syncing TLAS alone previously freed an AS still baked into integrator plans. Empty geometry still needs environment illumination. | Core AS cells/retirement and reader updates replace native-handle invalidation. Add/remove/move geometry while a trace is in flight; compare raster/traced composite, empty-scene environment and allocations. HK-09/RM RT cases must be followed by real scene tests. |
| [RayMakie glyph path](../../../Makie/RayMakie/src/overlay/scatter.jl:158) and [atlas dirty state](../../../Makie/RayMakie/src/screen.jl:291) | Text/scatter require actual SDF/atlas contents; a fixed-size atlas can change without changing dimensions. Dirty callback is explicitly separate from atlas size. | Typed sampled-image ABI and sample intrinsic in api.md §2.6; content stores are independent of growth. RM-15 now actually samples and changes texels after growth. Run real text/scatter tests with new glyph insertion, camera zoom, atlas growth and callback cleanup after closing one of two screens. |
| [Raycore mutation boundary](../../../Raycore/src/instanced-bvh.jl:2635) and [Hikari scene sync](../../../Hikari/src/scene.jl:351) | Raycore mutators update backing buffers and dirty/refit flags; sync! is the commit boundary. Hikari additionally updates scene bounds and notifies integrators on topology changes. Closest/any-hit default masks are both 0xff. | Mantle owns shared BLAS/TLAS build/update/retirement; Raycore keeps geometry/traversal adapters and hit conventions. Preserve scene bounds and notification semantics. Test topology versus transform-only changes, default/explicit masks, empty traversal and software/hardware hit equivalence; no backend-specific second member/lifetime store. |
| [Hikari sample_graph](../../../Hikari/src/integrators/volpath/graph.jl:417), [paired rounds](../../../Hikari/src/integrators/volpath/graph.jl:328) | Bounded device-count loop, two rounds per body to restore ping-pong direction; odd final round; sample counter increments on device. Comments at the file's start still describe an obsolete host loop: executable code is the evidence. | Group containing two rounds, repeat!(group,max_depth÷2;while_nonzero=count), optional odd round, final accumulation. No per-round readback. HK-08 tests termination, but real scenes must compare odd/even depths, zero/one live ray, media/shadows and empty queues on hardware and software traversal. |
| [Hikari per-material queues](../../../Hikari/src/integrators/volpath/per_material_queues.jl:1) | TypedHitRef stores one Int32 index per material queue, not a copied 304-byte hit per material. Generated dispatch specializes shading for each material type. | Keep compact typed indices and ordinary Julia specialization. Adding data of an existing type grows/stores arrays; introducing a new type specializes the shade kernel and recompiles affected plans. Preserve both typed queue shading and the actual per-material closest-hit/SBT path in rt-pipeline.jl:530–575; do not return to an all-material hit copy. Measure peak memory and compile counts when adding a type. |
| [DNNKernels emitgraph!](../../../JuliaVision/DNNKernels/src/emit.jl:134), [launch!](../../../JuliaVision/DNNKernels/src/launch.jl:257), [view planning](../../../JuliaVision/DNNKernels/src/plan.jl:62) | Modules already compose into a caller graph and bind resources without copies. Helper signatures currently require Graph. Escaping outputs include view roots; exported liveness is explicitly unsafe as placement input. Some permute/reshape chains require a materializing copy. | Generalize emission to target=Graph/Group plus graph owner for allocations. Preserve explicit workgroup sizes and required view copies. Derive liveness from actual uses, retain escaping roots, and propagate target to every internal launch. DNN-12 checks composed modules in a repeated Group with persistent state. Real exporter models still need view/quantization/precision comparisons. |
| [DNN memory read](../../../JuliaVision/DNNKernels/src/memory.jl:331), [memory ownership](../../../JuliaVision/DNNKernels/src/memory.jl:47) | Affinity → in-place top-k normalization → readout; capacity-sized intermediate and GPURef live-bank extent. Bank arrays persist across several module calls. | Graph temporaries for similarity/sums, persistent MantleArrays for bank/output, one live-count GPURef. No host bank-size read between passes; no forced materialization of resident outputs. DNN-10/12 plus real SAM2 frame variants and reference masks. |
| [MatAnyone propagation](../../../JuliaVision/MatAnyoneRunner/src/MatAnyoneRunner.jl:350) | Mask ingest and inference are separate calls; the initial seed resets recurrent memory, later corrections must not. Warmup repeatedly advances state. Device alpha is copied into persistent per-frame planes. | Preserve state transition order; combine only repeated calls with the same semantics. Test first seed, warmup=0/1/2, later correction, before-first-seed transparency and downstream device consumers. Compare alpha against the frozen driver, not merely a toy affine model. |
| [Bonsai session](../../../JuliaVision/BonsaiRunner/src/BonsaiRunner.jl:283), [KV growth](../../../JuliaVision/BonsaiRunner/src/BonsaiRunner.jl:351), [step/prefill](../../../JuliaVision/BonsaiRunner/src/BonsaiRunner.jl:937) | Quantized K/V plus scales and recurrent arrays are session-owned. Growth waits idle, copies all layers into replacement buffers, rebuilds step and clears prefill plans. Per-token host input is two GPURefs; greedy path currently synchronizes the whole backend before a 4-byte read. | Resize the same session arrays, keep quantization/block layout valid and let dependencies retire old storage. Recompile only capacity-dependent kernels. Array(best) waits for its writer, not unrelated device work. Test both sides of each page boundary, mixed prefill/decode, two sessions sharing weights, deterministic tokens and GC/close memory recovery. DNN-10 is necessary but does not validate quantized-cache layout. |
| [Editor composition](../../../VideoEditor/src/gpugraph.jl:1664), [layer plan](../../../VideoEditor/src/gpugraph.jl:1524) | Caches structural graphs, updates per-frame parameters, runs one composition then unconditionally synchronizes the backend before a callback. Source shape must be known before building the chain. | Keep structural caching; use device-owned sources/outputs, graph temporaries and stores. A consuming Mantle graph gets resource edges; a host callback/read waits only for its output. Test CPU/GPU sources, recurrent model output, four layers, premultiplied crop/alpha and resolution changes. Retain waits at genuinely foreign consumers until an explicit fence/lease contract exists. |
| [Editor decode ring](../../../VideoEditor/src/gpustream.jl:433), [eviction/close](../../../VideoEditor/src/gpustream.jl:493) | Direct vk_context and VideoDecode session; GOP feed/reordering state; unsafe_free! and synchronize prevent in-flight plane reuse. A bounded ring keeps current/prefetched GOPs; abandoned feed has parser/session state. | Device-explicit decoder boundary; decoded planes, session and DPB slots are resources with access tokens. free! retires plane creator holds; don't reuse a DPB slot while downstream work reads it. Test seek/reseek, B-frame display order, abandoned feeds, concurrent conversion, bounded resident bytes and close with work in flight. Codec conformance remains in the video package. |
| [Editor GL preview](../../../VideoEditor/src/glbridge.jl:234) | Imports Vulkan memory fds into GL, double-buffers, switches the actual GL context through GLMakie. Cross-driver/context synchronization is real integration work. | Preferred all-Mantle preview uses a Window and normal edges. Retaining GL preview requires an explicit external-image acquire/release fence protocol and OS-specific handles. Ordinary vendor-array withview does not implement GL interop. Do not delete the bridge or synchronization on the assumption that one process implies one memory/queue domain. |

## Migration sketches

These are target API sketches; `declare_round!`, `emit_model!`, and operation
kernels are consumer code, not new Mantle API. The graph owner and declaration
target are deliberately distinct.

Hikari replaces the do-block loop, Plan and manual record step:

```julia
g = Graph(dev)
declare_setup!(g, state, refs)
rounds = Group(g, "paired bounce")
declare_round!(rounds, state.a, state.b, refs)
declare_round!(rounds, state.b, state.a, refs)
repeat!(rounds, maxdepth ÷ 2; while_nonzero = state.a_count)
isodd(maxdepth) && declare_round!(g, state.a, state.b, refs)
declare_accumulate!(g, state, refs)
# Per frame/sample: stores into refs, then run!(g).
```

DNN module emit retains its exporter lowering, but threads an explicit target
through every operation and allocates scratch on the graph owner:

```julia
function emit_module!(target, graph, model, input, output)
    tmp = MantleArray(graph, Float32, model.hidden_shape...)
    dispatch!(target, model.first, (tmp, input, model.weights), length(tmp))
    dispatch!(target, model.second, (output, tmp), length(output))
    output
end
# Recurrent/KV state and outputs shared across graphs live on dev.
g = Graph(dev)
body = Group(g, "model step")
emit_module!(body, g, module_a, input, intermediate)
emit_module!(body, g, module_b, intermediate, output)
repeat!(body, iterations)
```

Bonsai's growth orchestration becomes resize! on each cache array (using its
actual packed layout), and run! applies ordered moves. The session still owns
position, token and recurrent state. Quantized packing kernels/candidates remain
DNN/LLM code; a generic float resize example is not a numerical port.

RayMakie's stable frame drops its per-frame draw-signature walk as the mechanism
for discovering edits: edit callbacks update arrays/refs or add/delete/sort
plot Groups. Core patches the pass list once per run. A new material/type,
attachment contract or capacity specialization may still trigger a recompile;
"all edits compile nothing" is not promised. A sorted list rewrite is O(plots),
so keep RM-06's measured scaling gate and report its cost.

The editor closes a decoded ring by dropping its creator holds and retiring
sessions/DPB resources after every decoding and downstream use token; it need
not idle unrelated compute. Keep the GL bridge's synchronization until either
its external-fence protocol is implemented or preview is ported to Mantle.

## Acceptance and limits

For each row, phase 11 records: source revision; migrated entry points; numerical
or image reference result; warmup versus steady allocations; native submits and
host waits; peak mapped/reserved/opaque bytes; compile/record counts; baseline
and replacement timing distributions on the same device and data. Consumer
benchmarks must complete equivalent work, not time only enqueue. Costs moved
into another package still count. No final line-count or speed result follows
from these shorter sketches.

The core suite gains DNN-12; RM-15 now samples real pixels and checks updated
contents. HK-10 now covers two actual closest-hit groups and callback-only holds; TL-43
checks numeric hit/mask/filter/interval semantics. DECL-12/13 check captured-only
resources and foreign-device rejection. Existing HK-08/09 and DNN-10 cover
further traced requirements.
The matrix above identifies real integration gates that remain unavailable
until implementation: material specialization/SBT equivalence, actual model
precision and quantization, decoder conformance/DPB reuse and GL handoff. These
must not be relabelled as passed by host-only tests or suite parsing.


## Additional RT source finding

Hikari's rt-pipeline.jl:530–575 constructs VPClosesthitTyped{Material} tuples,
assigns zero-based logical slots and caches programs in a global IdDict.
The earlier statement that material types are never hit groups was wrong.
execution-contracts.md §7 preserves that native path alongside typed shading
queues and moves program caching to Mantle core. The same file's alpha-test
comment at lines 155–161 explicitly says its initial HW RT shortcut treats alpha
textures as opaque. Do not carry that wrong rendering behavior into the target
or use it as the numerical reference: qualify alpha/media/light semantics and
select the query/filter path when the native candidate does not implement them.
