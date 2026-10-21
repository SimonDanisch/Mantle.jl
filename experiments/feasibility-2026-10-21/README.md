# Native foundation feasibility check — 2026-10-21

[recorded-slots.jl](recorded-slots.jl) is a bounded native vertical slice using
Vulkan.jl 0.7.0 and Lava's compiler, with no import of Mantle's current graph,
array, pool or queue implementation. It is experimental code, not the production
replacement or a claim that phase 1 is green.

Run in the workspace Julia environment:

```julia
include("dev/Mantle/experiments/feasibility-2026-10-21/recorded-slots.jl")
for name in ("NAVI31", "RTX 4000", "llvmpipe"), mode in (false, true)
    RecordedSlots.probe(name; indirect=mode)
end
```

The [source-hashed results](results.txt) record 22 passing assertions for each
of six device/mode combinations: 132 total. Khronos validation and synchronization
validation are enabled. Captured validation errors are test failures; the final
captured count, including a GC after the runs, is zero. Test summary times are
host test durations, not GPU performance measurements.

The slice has a fixed native buffer owned by its pool, suballocated data/cell/
argument regions, one serialized timeline queue, a graph, graph resource holds,
compiled kernels and one persistent body command buffer. A GPU head copies each
array's address/length Cell into plan argument descriptors and writes indirect
launch commands. The ordinary body kernel reads its prepared descriptor.

It actually executes and checks:

- repeated two-kernel dependent updates and back-to-back submissions;
- GPU-written lengths 0, 1, 129 and capacity 256, with exact expected contents;
- native indirect launch and capacity launch with entry bounds checks;
- a queued copy/cell-address update that moves data without rerecording the body;
- preclaim preparation failure without advancing the queue value or poisoning;
- creator release/GC while graph holds remain, followed by another correct run;
- graph release dropping its hold. The body record count stays one.

The first setup run exposed two probe setup defects: the compiler declares
VariablePointers, so those native features must be enabled; Vulkan.jl's generated
empty device-layer list passes a non-null pointer and triggers a current VUID.
The final probe enables the required features and locally normalizes that pointer
before create_device. It does not ignore these validation errors or modify a
production dependency. These are not presented as newly discovered Mantle bugs.

Scope limits: Int32 vectors, fixed capacity, one queue, host-coherent mapped
memory, no production type/layout/alias inference, no CodeInstance cache integration,
no asynchronous region reclamation or pool reuse, no multithreaded lock proof,
no presentation, and no Metal/CUDA/HIP native execution in this probe. free!(graph)
waits before reclaiming its command buffer. The fixed buffer keeps all subregions
allocated until teardown, so creator/graph hold checks do not prove safe reuse of
retired storage. Pool allocation and wait behavior here are not production speed
or nonblocking API claims. Capacity execution on Vulkan proves the mechanism,
not performance or integration on another native API.

Existing architectural and numerical evidence predates this probe. See the
[evidence index](../review-2026-10-20/evidence-index.md); do not restart those
migrations or replace their acceptance baselines with this small experiment.
