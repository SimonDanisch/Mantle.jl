# One short example per problem `recording-plan.md` solves.
#
# The API here is the planned one: it does not run against the code at 90e582f.
# Names the plan has not decided are marked "(name open)" or "(form open)".
# Functions that are not Mantle's (`shade!`, `lookat`, `samgraph`, model
# loaders, kernels) belong to the application. Every array an example uses is
# created in that example, where its location matters.

using Mantle, LinearAlgebra, GeometryBasics
using NNlib
using NNlib: gelu

# ── Devices, not backends ─────────────────────────────────────────────────────

# 1. No GPU (CI): lavapipe is a LavaDevice on Mesa's CPU driver. Where the
#    system's Mesa does not install it, the JLL makes it visible; it is loaded
#    before the first Device(), which creates the Vulkan instance.
using Lavapipe_jll

# 2. One GPU: nothing to choose. A LavaDevice on GPU 0 (a MetalDevice on macOS).
dev = Device()
Device() === dev                # the same object for the same hardware device

# 3. Several GPUs: devices() returns the devices themselves.
devices()                       # [LavaDevice(AMD …), LavaDevice(NVIDIA …), LavaDevice(lavapipe)]
nv = devices()[2]

# 4. CUDA and HIP are add-ons; their devices appear after loading the package.
using CUDA                      # devices() now also lists CudaDevice(…)

# ── Eager: arrays on a device run now ─────────────────────────────────────────

# 5. Each launch runs now: the kernel from its `Mantle.Kernel` at the code's
#    CodeInstance (B5.11), one submission. No batching.
a = MantleArray(dev, Float32, 1024)
fill!(a, 1f0)
b = a .* 2f0                    # b: a new MantleArray on dev

# 6. A KernelAbstractions kernel on a device runs now as well: the device is
#    the KA backend.
@kernel function scale!(x, s)
    i = @index(Global); x[i] *= s
end
scale!(dev)(a, 2f0; ndrange = length(a))

# 7. Calls on the same memory are ordered by that memory's sync state; calls
#    on unrelated memory do not wait for each other.
c = MantleArray(dev, Float32, 1024)
fill!(a, 0f0); fill!(c, 3f0)    # independent
a .+= c                         # waits for both fills

# 8. Reading on the host waits for exactly the work that wrote `a`.
host = Array(a)

# 9. Temporaries of an eager operation (a reduction's partials) are pool
#    regions, released once the GPU has used them.
total = sum(a)

# ── Graphs: arrays on a graph declare ─────────────────────────────────────────

# 10. An array on a graph declares: a call on it adds kernel nodes. Its
#     contents last only during a run; what must survive runs is a device
#     array the graph names.
g = Graph(dev)
x = MantleArray(g, Float32, 1024)
fill!(x, 0f0)                   # a node of g; nothing runs now

# 11. `y` and `x` live on g, so this adds `y = W * x` to g. The kernel is
#     chosen here, from dev's selection table for this shape. W lives on dev:
#     g names it (a hold), every run reads it.
W = MantleArray(dev, rand(Float32, 512, 1024))
y = MantleArray(g, Float32, 512)
mul!(y, W, x)

# 12. Where the arguments live decides when a call runs.
kvcache = MantleArray(dev, zeros(Float16, 64, 20, 32, 448))  # a transformer's key/value cache
fill!(kvcache, 0f0)             # only dev arrays: runs now, once
out = MantleArray(dev, Float32, 512)
copyto!(out, y)                 # y lives on g: a copy kernel node; every run writes `out`
prev = MantleArray(dev, Float32, 512)          # kept between runs: a device array g names
prev .= 0.9f0 .* prev .+ 0.1f0 .* y

# 13. A raw kernel in a graph: argument tuple plus an ndrange.
dispatch!(g, scale!, (x, 2f0), 1024)

# 14. Code that reads elements throws at the read (B4.3); code that only
#     launches (KA kernels, broadcasts, GPUArrays' generic code) declares into
#     the graph.
function myhelper!(v)
    for i in eachindex(v); v[i] = v[i]^2; end
    return v
end
myhelper!(x)                    # throws at the read v[i]: a graph array has no data at declaration
x .= x .^ 2                     # broadcasting declares: one kernel node

run!(g); Array(out)             # compiles, records, runs; reads the output

# ── Operations are Julia methods ──────────────────────────────────────────────

# 15. An operation is a method on MantleArray that launches its kernels where
#     its arguments live, over launchrange(dst): direct while the array is
#     fixed, indirect after its first resize!. Eager and declared are the same code.
function resizeplanar!(dst::MantleArray, img::MantleArray; mean, std)
    dispatch!(location(dst, img), resizeplanar_kernel!, (dst, img, mean, std), launchrange(dst))
    return dst
end

# 16. Rewrites are opt-in per graph. A DNN graph asks for epilogue fusion: the
#     broadcast is merged into the GEMM where the selected candidate implements
#     exactly bias + gelu, so yd is written once. A graph that asks for nothing
#     (RayMakie's) runs no rewrite.
gd   = Graph(dev; rewrites = (FuseEpilogues(),))  # (names open)
xd   = MantleArray(dev, Float32, 1024)        # per-run host data: a device array written by stores (B4.9)
yd   = MantleArray(gd, Float32, 512)
bias = MantleArray(dev, zeros(Float32, 512))
yout = MantleArray(dev, Float32, 512)
mul!(yd, W, xd)
yd .= gelu.(yd .+ bias)         # one kernel with FuseEpilogues, two without
copyto!(yout, yd)
xd[:] = rand(Float32, 1024); run!(gd)

# 17. The same mul! selects per device: our kernel on Vulkan, an MLX kernel or
#     an MPSGraph executable on Metal, CUTLASS or cuBLAS on CUDA. In a Group
#     and in a when! launch only kernel candidates qualify.

# 18. Host work in the middle of a call: a host node, its writes declared.
#     Mantle submits the work before it, waits, calls `align` on this thread,
#     then submits the rest, which starts by copying the cells again. The
#     arrays it touches are placed in host-visible arena memory; another graph's
#     run on gh's arena part waits on the host until gh's run has submitted its
#     last stage (host function included), then on its tokens. The cut is
#     visible in submissions(gh).
gh        = Graph(dev)
durations = MantleArray(gh, Int32, 512)
idx       = MantleArray(gh, Int32, 8192)
phonemes  = MantleArray(dev, Float32, 512, 80) # per-run host data: a device array written by stores (B4.9)
predictdurations!(durations, phonemes)
dispatch!(gh, HostCall(align), (durations, idx); writes = (idx,))
frames = MantleArray(dev, Float32, 80, 8192)
expandframes!(frames, idx)
phonemes[:] = embedphonemes("hello"); run!(gh)

# ── One call, one graph ───────────────────────────────────────────────────────

# 19. Exported modules are called like functions on graph arrays: all of them
#     and the work between them are one graph. A loop with a fixed count is a
#     launch group, compiled once (B4.2): a value that holds GPU launches only,
#     declared into it by passing the group to the functions that dispatch:
#     what gets the group is repeated. A call without it (a plain `mul!(C, A, B)`)
#     is not repeated: on graph arrays it declares into their graph and runs once
#     per run, on device arrays it runs now. Only the arguments decide; there is
#     no "current graph" (plan rule 5; B4.2, B5.1).
#     Before each iteration the loop head writes the iteration index into the
#     argument slots of the group's launches that take it. A group is not a
#     location: allocations belong to the graph, not the group (B4.4), so the
#     transformer's output and temporaries are gq's transients.
textencoder, transformer, vae = loadqwen(dev)   # models with weights on dev (name open)
gq     = Graph(dev)
ids    = MantleArray(dev, Int32, 77)            # per-run host data: a device array written by stores (B4.9)
noise  = MantleArray(dev, randn(Float16, 64, 64, 16))
sigmas = MantleArray(dev, sigmaschedule(50))    # per-step constants
h      = textencoder(adapt(Mantle.Device(gq), ids))   # ids aliased into gq, so the call declares (B4.4)
lat    = MantleArray(gq, Float16, 64, 64, 16)
copyto!(lat, noise)
step   = Group(gq, "step")                      # a value; launches only
i      = iteration(step)                        # the loop index the loop head writes (name open)
v      = transformer(step, lat, h, sigmas, i)   # its launches go into step; v is gq's
eulerstep!(step, lat, v, sigmas, i)
repeat!(step, 50)                               # attaches the group: 50 iterations per run of gq
image = MantleArray(dev, Float16, 1024, 1024, 3)
copyto!(image, vae(lat))
ids[:] = tokenize("a red bicycle")              # a store; gq's first run applies it
run!(gq)

# 20. A loop the device ends: generate until a stop token. `running` is an
#     integer count on the device (B4.1); the loop ends on the GPU once it is 0.
gw2     = Graph(dev)
tokens  = MantleArray(dev, Int32, 448)          # output: the host reads it after the run
running = MantleArray(gw2, Int32, 1)
logits  = MantleArray(gw2, Float32, 51865)
fill!(running, Int32(1))
dec     = Group(gw2, "decode")                  # a value; launches only
decodestep!(dec, logits, kvcache, tokens, iteration(dec))
sampletoken!(dec, tokens, running, logits, iteration(dec))   # writes 0 into running at the stop token
repeat!(dec, 448; while_nonzero = running)

# 21. A length only the device knows: a resizable array. The kernel that
#     decides it writes it into the array's cell; a prepare kernel then writes
#     the launch commands and sizes of the launches over it.
gr0  = Graph(dev)
rays = MantleArray(gr0, Ray, 1920 * 1080)
hits = MantleArray(gr0, Hit, 0; capacity = 1920 * 1080)
film = MantleArray(dev, RGBA{Float32}, 1920, 1080)
generaterays!(rays)
intersect!(hits, rays)          # writes length(hits) on the device
shade!(film, hits)              # launches over launchrange(hits): indirect, from the prepare kernel
run!(gr0)

gv    = Graph(dev)
mel   = MantleArray(dev, Float32, 512, 8192)  # per-run host data: a device array written by stores (B4.9)
audio = MantleArray(dev, Float32, 0; capacity = 8192 * 300)   # an output of unknown length
vocoder!(audio, adapt(Mantle.Device(gv), mel))                 # mel aliased into gv, so the call declares (B4.4)
mel[:] = melspectrogram("hello world")
run!(gv); length(audio)         # waits for vocoder!'s write, reads the cell back (B4.10)

# 22. A launch that runs only when a flag is set, without recompiling. One
#     launch per when!; the flag is a Bool on the device (a GPURef{Bool} or a
#     one-element Bool array), enforced by the method's types: any other flag
#     is a MethodError (B4.1).
gm2   = Graph(dev)
bank  = MantleArray(dev, Float32, 256, 4096)    # MatAnyone's memory, kept between calls
feats = MantleArray(gm2, Float32, 256, 4096)
frame = MantleArray(dev, Float32, 3, 480, 854)  # per-run host data: a device array written by stores (B4.9)
encodeframe!(feats, frame)
firstframe = GPURef(dev, true)
when!(gm2, firstframe, initmemory_kernel!, (bank, feats), size(bank))   # one launch; runs where firstframe is true
readmemory!(bank, feats)

# ── Compile, record, run ──────────────────────────────────────────────────────

# 23. The graph manages its plan: the first run! compiles (kernels, order,
#     placement, queues, submissions) and records once; later runs only run.
frame[:] = decodeframe("clip.mp4", 1); run!(gm2)   # compiles, records, runs initmemory_kernel!
firstframe[] = false                               # a store into its slot, applied by gm2's next run
frame[:] = decodeframe("clip.mp4", 2); run!(gm2)   # initmemory_kernel!'s count is written times false: skipped on the GPU; same recording

# 24. Per-frame values: a value that changes every frame (a camera matrix) is a
#     one-element device array written by a store. A store of up to 64 KiB is
#     inline: the call copies its bytes into an upload region, and the next
#     submission that touches the array writes them in front of its own work
#     (emitstore!), with no host wait (B3.18). A value GPURef is for a scalar a
#     kernel takes by value: one slot in device memory whose address every
#     plan bakes; `r[] = v` is a scheduled store; it marks no
#     plan (B4.9).
gc       = Graph(dev)
cam      = MantleArray(dev, [Mat4f(I)])         # a one-element array
exposure = GPURef(dev, 1f0)                     # a scalar the kernel takes by value
verts    = MantleArray(dev, rand(Point3f, 1000); capacity = 4096)   # resizable
depth    = MantleArray(gc, Float32, 1920, 1080) # a transient: the calls declare into gc
img2     = MantleArray(dev, RGBA{Float32}, 1920, 1080)   # output, shown after the run
pts      = MantleArray(dev, Point3f, 64)        # per-run host data: a device array written by stores (B4.9)
rasterize!(depth, verts, cam)
shade!(img2, depth, verts, pts, cam, exposure)
pts[:] = rand(Point3f, 64)
cam[1:1] = [lookat(Vec3f(0, 0, 5), Vec3f(0), Vec3f(0, 1, 0))]; run!(gc)
cam[1:1] = [lookat(Vec3f(1, 0, 5), Vec3f(0), Vec3f(0, 1, 0))]; run!(gc)   # inline store, applied by this run; no host wait
exposure[] = 1.5f0; run!(gc)    # a store into the slot, applied by this run; no wait

# 25. `pts` is a device array gc names; `pts[:] = data` is a scheduled store
#     applied by gc's next run.
pts[:] = rand(Point3f, 64)
run!(gc)

# 26. A store into a device array is pending: the data is copied now into an
#     upload region (written inline up to 64 KiB, copied from the region
#     beyond; B3.18), and gc's next run applies it in front of its own work
#     (after the work still using verts). Per-frame values are one-element
#     arrays written by a store (example 24).
verts[1:3] = [Point3f(0), Point3f(1), Point3f(2)]
run!(gc)

# 27. Growing an array a graph uses: pending, like a store. The first resize!
#     of a fixed array makes it resizable (gc recompiles once, since it
#     launched over the array directly): if the new size fits its storage
#     nothing moves; beyond it, a large array on a device with sparse residency
#     moves into a reserve, any other into a pool region of twice the need.
#     Later ones map pages (or move to a doubled region) and write the size
#     into the cell, and gc's next run copies it in its head node: nothing is
#     re-recorded.
resize!(img2, 10)               # img2 was fixed: it shrank, so it stays where it is; gc recompiles at its next run
resize!(verts, 2000)            # within verts' capacity of 4096: a cell write (and pages, where it is a reserve)
run!(gc)

# 28. Submissions are cut by an estimated cost against the device's budget, at
#     compile. Nothing is measured and re-recorded.
submissions(gq)

# 29. Two graphs sharing an array: decode's run waits for encode's complete
#     last run, found through that array (B3.15). No full barrier, no extra
#     command buffer.
features = MantleArray(dev, Float16, 256, 64, 64)
encode = encodegraph(features); decode = decodegraph(features)
run!(encode); run!(decode)      # decode reads what encode wrote

# ── Memory ────────────────────────────────────────────────────────────────────
#
# Byte counts in these examples are illustrations, not measurements.
# Introspection names (reserved, mapped, peakbytes, naivebytes, budget) are open.

# 30. One pool per device; everything comes from it. The device type only
#     allocates raw blocks of a kind; the pool hands out regions of them.
m = MantleArray(dev, Float32, 16 * 2^20)        # 64 MB region, kind Buffers
t = Texture2D(dev, RGBA{Float16}, 4096, 4096)   # 128 MB region, kind Images
reserved(dev)                   # blocks held from the driver, all kinds

# 31. Within one graph, transients whose lifetimes do not overlap share memory.
gp  = Graph(dev)
p1  = MantleArray(gp, Float32, 16 * 2^20)       # 64 MB
p2  = MantleArray(gp, Float32, 16 * 2^20)
p3  = MantleArray(gp, Float32, 16 * 2^20)
res = MantleArray(dev, Float32, 16 * 2^20)
fill!(p1, 1f0)                  # node 1
p2 .= p1 .+ 1f0                 # node 2: last use of p1
p3 .= p2 .* 2f0                 # node 3: p3 can take p1's bytes
copyto!(res, p3)                # node 4
run!(gp)
naivebytes(gp), peakbytes(gp)   # 192 MB if each had its own, 128 MB placed

# 32. Across graphs, transients share one arena part from offset 0: it costs
#     the largest graph's peak, not the sum.
sam = samgraph(dev)             # peak 900 MB of transients
ray = raygraph(dev)             # peak 300 MB
run!(sam); run!(ray)            # the arena part needs 900 MB, not 1200 MB
run!(ray)                       # ray waits on sam's work in the shared arena part

# 33. Graphs core expects to run at the same time (the editor's frame beside a
#     DNN, by priority; open) get separate arena parts: 1200 MB then, in
#     exchange for overlap. Core decides; either is correct.

# 34. Sparse arenas grow by mapping pages and never move anything.
big = biggraph(dev)             # peak 2 GB
run!(big)                       # maps pages up to 2 GB; sam's and ray's addresses unchanged

# 35. An idle graph holds address space, not memory. Under pressure, pages only
#     idle graphs use are unmapped and those graphs are marked; a marked graph's
#     next run maps its pages back before it submits.
run!(ray); run!(ray)            # big idle: pages beyond what sam and ray use may be unmapped
mapped(dev)                     # physical pages mapped, all arenas
run!(big)                       # maps what it needs, then submits

# 36. Memory is never reused while the GPU may still use it. Every submission
#     has a token the GPU signals when it has finished; memory is retired with
#     the tokens of its last uses and released once they have passed.
a3 = MantleArray(dev, Float32, 2^24)            # region R
a3 .+= 1f0                                      # submission 7 uses R
a3 = nothing; GC.gc()           # the finalizer retires R with token 7
a4 = MantleArray(dev, Float32, 2^24)            # R if 7 has passed, other memory if not

# 37. Most memory is the graph's and goes with the graph. Its transients are
#     arena offsets: a run allocates and frees nothing. Its argument block and
#     recordings are released together once its last run has finished, and it
#     stops counting toward the arena's size.
free!(big)                      # or drop `big` and let the GC collect it
run!(big)                       # error: the graph was freed

# 38. A device array shared by graphs is counted: one hold for its creator, one
#     per graph that names it. It is retired when the last hold goes.
feat = MantleArray(dev, Float16, 256, 64, 64)
e1 = encodegraph(feat); d1 = decodegraph(feat)  # three holds: creator, e1, d1
free!(feat)                     # fine while e1 and d1 still use it; `feat` can no longer be used here
free!(e1); free!(d1)            # last hold gone: retired, released once both last runs are done

# 39. A KV cache that grows: resizable, the sequence as its last dimension, so
#     its contents are kept. Its capacity is reserved as address space; pages
#     are mapped as it grows. Runs submitted before see the old size, runs
#     after see the new one.
kv = MantleArray(dev, Float16, 64, 20, 32, 448; capacity = 64 * 20 * 32 * 4096)
resize!(kv, 896)                # (64, 20, 32, 896): pages mapped, a cell write
resize!(kv, 8192)               # beyond the capacity: a move; the parts that recorded its handle are recorded
                                # again (open decision 17, ruled (c), B3.11); plans whose temporaries were sized
                                # from its capacity recompile once (B4.10)

# 40. Pressure: an allocation that would exceed the device's budget makes core
#     release passed work, then, in order and retrying after each step: unmap
#     idle graphs' pages, trim empty blocks, collect garbage so dropped arrays
#     retire, wait for retiring storage oldest first, then (ruled B3.9) evict
#     the least recently used idle resources and wait for their eviction
#     copies, and only then throw OutOfDeviceMemory. It never flushes a queue.
budget(dev)                     # VK_EXT_memory_budget on Vulkan, cuMemGetInfo on CUDA
huge = MantleArray(dev, Float32, 2^31)          # may trigger the steps above

# 40b. Eviction (ruled B3.9). Two models that each need 90% of the device,
#      run one after the other with no host wait. Their transients
#      share one arena part (the larger one's peak). When dnn_2's run cannot
#      get memory, acquire!'s pressure steps end by evicting dnn_1's idle
#      weights to host memory: no plan that names them is running, though
#      dnn_1's work may still be in flight on the GPU; the eviction copies are
#      ordered after it through the weights' sync states, and acquire!
#      waits for the copies outside every lock. dnn_1's next run restores them
#      (and may evict dnn_2's). Nothing recompiles: addresses reach the plans
#      through cells, and resources a plan recorded by handle or a composite
#      holds are never evicted. On a device whose readback memory is the same
#      heap as device memory (the Apple M5) eviction frees nothing and is
#      skipped.
dnn_1 = whisperlarge(dev)       # weights + peak transients ≈ 90% of the device
dnn_2 = qwenimage(dev)          # ≈ 90% as well
run!(dnn_1)                     # works
run!(dnn_2)                     # works: dnn_1's weights evicted if both do not fit, also while dnn_1 is in flight
run!(dnn_1)                     # works: restored before the run, in the same submission

# 41. A weight that does not have to stay resident is placed like a transient
#     and uploaded before its first use in each run (open).
Wbig = MantleArray(dev, rand(Float16, 4096, 4096); streamed = true)   # (open decision 6)

# ── Vendor code ───────────────────────────────────────────────────────────────

# 42. Handing memory to vendor code: a zero-copy view in the vendor's type, in a
#     block with an explicit stream. withview takes a hold on the array,
#     applies its pending changes (and any restore) in a submission of its
#     own, and makes the stream wait on that and on Mantle's accesses;
#     Mantle's later uses wait for the stream's work, also if the block throws,
#     and the hold is dropped with the stream's token. While lent, Mantle calls
#     on the array throw. Vendor allocations come from the vendor's pool;
#     Mantle sees them only through the device's budget.
cud = devices()[4]                               # a CudaDevice
xin = MantleArray(cud, Float32, 32, 32, 3, 1)
withview(CuArray, xin, CUDA.stream()) do ca     # (name open)
    NNlib.conv(ca, CUDA.rand(Float32, 3, 3, 3, 8))
end

# ── Ray tracing ───────────────────────────────────────────────────────────────

# 43. One core TLAS for hardware and software ray tracing (resizing-and-
#     raytracing.jl has the full API). Builds and refits are operations,
#     ordered like everything else, never in a Group or a when! launch.
gt       = Graph(dev)
tlas     = TLAS(dev)                            # hardware where the device has it
vertices = MantleArray(dev, rand(Point3f, 300))
faces    = MantleArray(dev, [TriangleFace{UInt32}(i, i + 1, i + 2) for i in 1:3:298])
blas     = BLAS(dev, vertices, faces)
sphere   = BLAS(dev, MantleArray(dev, spherevertices(16)), MantleArray(dev, spherefaces(16)))
h        = push!(tlas, blas, Mat4f(I))          # one instance; a build is pending
parts    = MantleArray(dev, rand(Point3f, 500)) # particle positions
r        = push!(tlas, sphere, parts; scale = 0.01f0)
acc      = MantleArray(dev, RGBA{Float32}, 1920, 1080)   # accumulates across runs
pipeline = RayTracingPipeline(; raygen, closest_hit, miss)   # a value, like the raster pipelines
refit!(gt, tlas)                                # every run: records rewritten from parts, Update()
trace!(gt, pipeline, tlas, (acc, cam), (1920, 1080))   # ordered after the refit
run!(gt)

# 44. Particles move and change in number; gt is never recompiled. The same
#     length: the refit picks the new positions up. A new length: copy! of
#     parts (a resize that does not keep the old contents, then a store) locks
#     parts and the TLAS, its reader (ruled B3.10), together and pends the
#     stores and a Build with the exact count; the submission that applies the
#     Build grows the TLAS's instance capacity if 801 instances exceed it
#     (B3.13: the call never allocates). The change marks the BuildRecording of
#     gt's refit, so gt's next run applies the Build in front of its work and
#     re-records only that small command buffer for the new count (Vulkan;
#     Metal reads the count from a cell).
copy!(parts, rand(Point3f, 500)); run!(gt)
copy!(parts, rand(Point3f, 800)); run!(gt)      # parts' first resize!: a small array, so a move into a pool
                                                # region of twice the need; the range table reads its cell,
                                                # so gt does not recompile
delete!(tlas, h); run!(gt)                      # a Build; blas released once that build has run

# ── Graphics ──────────────────────────────────────────────────────────────────

# 44b. One pipeline for a value and per-element attributes. Attributes are
#      GPURefs of device arrays (device-only; a view as target throws); shaders
#      read attribute(a, i) = a[min(i, length(a))]. Switching between one colour
#      and a colour per point is a resize and a store, pending. A pipeline is a
#      value: equal stages and state give one pipeline, so it is built where it
#      is drawn.
scatterpipeline() = Rasterizer(; vertex = VertexShader(scatter_vertex; outputs = (color = Vec4f,)),
                                 fragment = FragmentShader(inputs -> inputs.color),
                                 topology = PointList(), blend = Additive(), depth = DepthOff())
gs      = Graph(dev)
win2    = Window(dev, 1200, 800)
pts     = GPURef(dev, MantleArray(dev, rand(Point3f, 100_000); capacity = 1 << 18))
colors  = GPURef(dev, MantleArray(dev, [Vec4f(1, 0, 0, 1)]))     # one colour for all points
sizes   = GPURef(dev, MantleArray(dev, rand(Float32, 100_000) .* 4f0))   # one size per point
camera  = MantleArray(dev, [Mat4f(I)])                              # a uniform: a one-element array
rect    = MantleArray(dev, [Vec4f(0, 0, 1, 1)])                     # the scene's rectangle in the target
visible = MantleArray(dev, [true])                                  # the draw's flag: a one-element Bool array
render!(gs, win2 => Clear((0f0, 0f0, 0f0, 1f0))) do p
    when!(p, visible, scatterpipeline(), (pts, colors, sizes, camera, rect), launchrange(pts))   # one draw (B4.1)
end
run!(gs)                                         # compiles: one pipeline
copy!(colors[], rand(Vec4f, 100_000)); run!(gs)  # per-point colours: same pipeline, nothing recompiles
camera[1:1] = [lookat(Vec3f(0, 0, 5), Vec3f(0), Vec3f(0, 1, 0))]  # a store, inline in the next submission
visible[1:1] = [false]; run!(gs)                 # hidden: the head writes the draw's count times false; no new plan
pts[] = MantleArray(dev, rand(Point3f, 300_000)) # another device array: a rebind (a cell write), no copy
run!(gs)                                         # draws 300 000 points; nothing recompiles

# 44c. A simulation drawn without copies (recording-plan.md, "RayMakie on the
#      core model"). The plot points at the simulation's device arrays.
#      Readers (ruled B3.10): a run that writes arrays pends the updates of the
#      structures built from them. The BLAS over lengths a kernel writes is
#      built over the geometry capacity, with degenerate primitives
#      past the device length written by a prepare kernel, so a change of the
#      vertices is an Update. Vulkan's update may not change which vertices
#      form a triangle, so new faces from simulate! need a Build, or a BLAS
#      over geometry the prepare kernel expands without indices (to be
#      confirmed, recording-plan.md, Acceleration structures).
simg     = Graph(dev)
vertices = MantleArray(dev, Point3f, 0; capacity = 1 << 20)
faces    = MantleArray(dev, TriangleFace{UInt32}, 0; capacity = 1 << 21)
simulate!(simg, vertices, faces)                 # kernels that write both, lengths included (lengthof)
blas2 = BLAS(dev, vertices, faces)               # holds both arrays; registered in their readers
push!(tlas, blas2, Mat4f(I))                     # a new instance: a Build pending
run!(simg)    # writes the arrays; pends blas2's update and, through blas2, tlas's refit
run!(gt)      # tlas's pending Build and refit merge into one Build; applied in front of the trace,
              # after simg's write (gt locks blas2's arrays with tlas and waits on them); nothing recompiles

# ── Windows ───────────────────────────────────────────────────────────────────

# 45. The window is a resource; acquire and present are part of run!.
win  = Window(dev, 1280, 720)
gwin = Graph(dev)
blur = Image(gwin, RGBA{Float16}, win)          # sized like the window; placed at its capacity (the monitor),
                                                # pages mapped for the window's size where the device has sparse residency
render!(gwin, blur => Clear((0f0, 0f0, 0f0, 1f0))) do p
    draw!(p, shader, (verts, cam), launchrange(verts))
end
blurpass!(win, blur)                            # writes the swapchain image
run!(gwin)                      # compiles; Vulkan: the present segment once per swapchain image
run!(gwin)                      # acquire, submit that image's segment, present

# 46. Resize comes from the window's event and only marks the graph. Its next
#     run makes the new swapchain generation (the old one retired by its last
#     run's tokens; open decision 10), recreates blur at the new size on the
#     same arena memory, and re-records the segments that draw into blur and
#     the per-image present segments. Beyond the window's capacity (the
#     largest monitor's size) the graph recompiles at that run.
run!(gwin)                      # after the user dragged the window larger: nothing else to do

# ── Threads and devices ───────────────────────────────────────────────────────

# 47. No owning thread: run and free a graph from any thread; one graph's runs
#     are serialised by its lock.
task = Threads.@spawn run!(g)
run!(g)
wait(task)

# 48. Two devices in one process never mix: every array names its device.
d1, d2 = devices()[1], devices()[2]
a1 = MantleArray(d1, Float32, 10); a2 = MantleArray(d2, Float32, 10)
a2 .= a1                        # error: arguments from two devices
copyto!(a2, a1)                 # explicit cross-device copy (open)
