# Target API examples, not runnable integration tests.
# Julia pseudocode: app_* names are omitted application operations/kernels;
# named pipelines/data are supplied by the application, never extra Mantle entries.
# Mantle entry signatures are defined only in api.md.
using Mantle
using LinearAlgebra: mul!
using Random: rand!
using ColorTypes: RGBA
import KernelInterface as KI

# 1. Eager and declared work share the same operation body.
function scale_kernel!(a, scale)
    i = KI.get_global_id().x
    i <= length(a) || return
    a[i] *= scale
    nothing
end

dev = Device()
a = MantleArray(dev, ones(Float32, 1024))
dispatch!(dev, scale_kernel!, (a, 2f0), launchrange(a))
host = Array(a)

# The shared operation layer preserves ordinary eager numerical signatures.
A = MantleArray(dev, A_cpu)
B = MantleArray(dev, B_cpu)
C = MantleArray(dev, Float32, size(A_cpu, 1), size(B_cpu, 2))
mul!(C, A, B)
host_C = Array(C)

# 2. Construction/naming in a graph never changes the array's owning device.
g = Graph(dev)
positions = MantleArray(g, zeros(Float32, 3, 1024))  # initialized ONCE
velocity = MantleArray(dev, ones(Float32, 3, 1024))
call!(g, broadcast!, +, positions, positions, velocity)   # explicitly recorded, runs on EVERY replay
run!(g)
run!(g)                                           # state accumulates; no implicit reset

# 3. Scratch is explicit and initialized each run where it is read.
tmp = MantleArray(g, Float32, 1024; scratch=true)
call!(g, fill!, tmp, 0f0)
app_accumulate_work!(g, tmp, positions)               # application operation
result = MantleArray(g, Float32, 1024)              # persistent result
call!(g, copyto!, result, tmp)
snapshot = call!(g, Array, result)
run!(g)
host_result = fetch(snapshot)                      # no GPU submission by fetch

# 4. Share a persistent leaf, including its GPU size, with another graph.
other = Graph(dev)
dispatch!(other, scale_kernel!, (result, 0.5f0), launchrange(result))
free!(g)                                          # result/other retain storage
run!(other)
# dispatch!(other, ..., (tmp,), ...) would reject a scratch lifetime escape.

# 5. Group membership belongs to explicitly targeted operations, never arrays.
model_graph = Graph(dev)
state = MantleArray(model_graph, zeros(Float32, 512))
weights = MantleArray(dev, ones(Float32, 512, 512))
step = Group(model_graph, "model step")
work = MantleArray(step, Float32, 512; scratch=true)
recorded_work = call!(step, mul!, work, weights, state)
@assert recorded_work === work                    # return specified by this shared definition
call!(step, broadcast!, (w, x) -> tanh(w + x), state, work, state)
repeat!(step, 8)
run!(model_graph)

# 6. Nested FEM/Newton/Krylov regions retain library composition.
fem = Graph(dev)
u = MantleArray(fem, zeros(Float64, 3, 128))
running = GPURef(dev, Int32(1))
timestep = Group(fem, "timestep")
newton = Group(timestep, "Newton")
app_declare_newton_rhs!(newton, u)
dispatch!(newton, app_reset_count_kernel!, (running, Int32(1)), 1)
krylov = Group(newton, "Krylov")
residual = MantleArray(krylov, Float64, 3, 128; scratch=true)
app_declare_krylov_step!(krylov, u, running, residual) # writes this iteration's residual
r2 = MantleArray(krylov, Float64, 1; scratch=true)
call!(krylov, sum!, abs2, r2, residual)              # ordinary array, result stays on GPU
dispatch!(krylov, app_update_convergence_kernel!, (running, r2, 1e-12), 1)
repeat!(krylov, 100; while_nonzero=running)
linesearch = Group(newton, "line search")
app_declare_line_search!(linesearch, u)
repeat!(linesearch, 12)
repeat!(newton, 8)
repeat!(timestep, 4)
# FP64 is selected only where the compiler/device supports the required semantics.
# Solvers are consumer algorithms; Mantle owns only their regions/resources/order.

# 7. Device resizing publishes the logical size without a public metadata handle.
function publish_size_kernel!(output, computed_count)
    if KI.get_global_id().x == 1
        resize!(output, computed_count[1])
    end
    nothing
end
function consume_kernel!(output, input)
    i = KI.get_global_id().x
    i <= length(input) || return
    output[i] = input[i]
    nothing
end

particles = Graph(dev)
source = MantleArray(particles, Float32.(1:4096) ./ 4096f0)
active = MantleArray(particles, Float32, 0; capacity=4096)
out = MantleArray(particles, Float32, 4096)
app_compact!(particles, active, source, x -> x > 0.5f0) # application builder: checked scatter + final resize!
dispatch!(particles, consume_kernel!, (out, active), launchrange(active))
active_snapshot = call!(particles, Array, active)                # records size + matching payload
run!(particles)
selected = fetch(active_snapshot)
selected_count = length(selected)                 # size of the same coherent snapshot
# length(active) in a kernel is logical size; host inspection uses the readback.
# Capacity plans scratch; no hidden host Int(counter) is permitted.

# 8. A GPU view is created by ordinary device code.
function subrange_kernel!(out, parent, bounds)
    i = KI.get_global_id().x
    first, count = bounds[1], bounds[2]
    v = view(parent, first:first+count-1)
    i <= length(v) || return
    out[i] = v[i]
    nothing
end
# Multiple dispatches receive the same bound resources through prepared arguments.

# 9. Recompilation can change algorithms without changing persistent outputs.
# A mul! declaration retains its RegionNode recipe. Growth across a candidate's
# shape domain reselects and rebuilds workspace/commands together. Native handles
# recorded by value are refreshed/retired; there is no second backend graph walker.
resize!(state, 1024)
state[513:1024] = zeros(Float32, 512)              # initialize the newly exposed tail
copy!(weights, ones(Float32, 1024, 1024))           # explicit replacement weights
resize!(work, 1024)                                # mul! initializes all scratch each iteration
run!(model_graph)                                  # one coherent shape change, then replay
# Unchanged state prefix survives; shape-invalid calls report their origin.

# 10. Conditional shadow updates include the clear and preserve skipped output.
frame = Graph(dev; priority=1, share_scratch=false)
shadow = Image(frame, Float32, 1024, 1024)         # persistent depth image
dirty = GPURef(dev, true)
update_shadow = Group(frame, "shadow update")
clear!(update_shadow, shadow, 1f0)
draw!(update_shadow, shadow, shadow_pipeline, shadow_args, shadow_count)
when!(update_shadow, dirty)
dispatch!(update_shadow, app_clear_flag_kernel!, (dirty,), 1)
# Active path removes stale depth then redraws; inactive path preserves shadow.
# Core chooses native control or conditional clear draws, not consumer branches.

# 11. GPU-driven culling builds a bounded indirect command array.
commands = MantleArray(frame, DrawCommand, 0; capacity=10000)
app_cull_commands!(frame, commands, meshes, camera)    # publishes resize!(commands,n)
draw!(frame, attachments, raster_pipeline, draw_args, commands;
      count=launchrange(commands), indices=index_array)
# Each command carries counts and offsets. Execution sets bound any GPU-selected
# program/resource choices; generated commands cannot introduce hidden resources.

# 12. Same-graph geometry writes are followed by a derived update before tracing.
rays = Graph(dev)
vertices = MantleArray(rays, initial_vertices)
faces = MantleArray(rays, initial_faces)
blas = BLAS(dev, vertices, faces)
tlas = TLAS(dev)
push!(tlas, blas, instance_transform)
app_deform!(rays, vertices)
trace!(rays, ray_program, tlas, trace_args, (1920, 1080))
# No manual refit call is needed to repair the deform -> trace ordering.
# A later failed stage cannot clear dirty geometry before its matching build.

# 12b. The same alpha/intersection functions serve queries and logical programs.
function app_alpha_ok(hit, ray, scene)
    app_alpha_sample(scene, hit, ray) >= 0.5f0
end
function app_sphere_intersections(report, candidate, scene)
    # Domain helper returns finite roots and Float32 UVs, including near AND far.
    # The object ray is already transformed; do not transform/normalize it again.
    for (t, u, v) in app_sphere_roots(candidate.objectray, candidate.primitiveid, scene)
        report(t, (UInt32(candidate.primitiveid), u, v, 1f0)) || return nothing
    end
    nothing
end
sphere_attributes = Tuple{UInt32,Float32,Float32,Float32}
hit_functions = HitFunctionTable((
    TriangleHitGroup(app_shade_triangle; any_hit=app_alpha_ok),
    ProceduralHitGroup(app_sphere_intersections, app_shade_sphere;
                       any_hit=app_alpha_ok, attributes=sphere_attributes),
))
# app_shade_*(hit, ray, payload, scene) -> the declared payload type.
# Geometry count is runtime data, not a tuple type carrying scene topology.
geometry = [TriangleGeometry(vertices, faces), ProceduralGeometry(sphere_bounds)]
mixed_blas = BLAS(dev, geometry)
query_scene = TLAS(dev)
range = push!(query_scene, mixed_blas, instance_transform; hitgroup=(1, 2))
function app_query_kernel!(out, accel, table, origins, directions, scene)
    i = KI.get_global_id().x
    i <= length(out) || return
    # Nothing attributes by default: this result needs only IDs/distance/barycentrics.
    out[i] = rayquery(accel, origins[i], directions[i], (scene,); hitgroups=table)
    nothing
end
# Typed procedural result from the shorthand, using a sphere-only accel:
function app_sphere_query(origin, direction, accel, scene)
    h = rayquery(accel, origin, direction, (scene,);
                 intersect=app_sphere_intersections, accept=app_alpha_ok,
                 attributes=Tuple{UInt32,Float32,Float32,Float32})
    if h.hit
        app_consume_sphere_attributes(hitattributes(h), scene) # concrete tuple
    end
    nothing
end
# A mixed query with non-Nothing result attributes supplies a capture function
# normalizing both triangle and procedural candidates into one declared schema.
program = RayTracingPipeline(raygen=app_raygen, hitgroups=hit_functions,
                             miss=(app_miss,), payload=NTuple{3,Float32}, maxdepth=2)
# app_miss(ray,payload,scene) -> NTuple{3,Float32}; raygen receives scene explicitly.
trace!(rays, program, query_scene, (scene_data,), (1920, 1080))
# No device/backend branch. Shared selection chooses query/typed dispatch or an
# eligible native program; Hikari's existing wavefront kernel need not use program.

# 13. Explicit RNG state survives replay/recompile.
stochastic = Graph(dev)
rng = RNG(dev, 1234)
noise = MantleArray(stochastic, Float32, 1024)
call!(stochastic, rand!, rng, noise)
# run!(stochastic) advances this stream; Random.seed! explicitly resets it.
# Private tuning state never consumes rng, and queue placement selects no RNG.

# 14. Foreign storage enters the same array model.
function import_and_record(dev, foreign_array, stream)
    imported = MantleArray(dev, foreign_array; copy=false, stream=stream)
    graph = Graph(dev)
    call!(graph, broadcast!, *, imported, imported, 2f0)
    graph, imported
end
# The retained allocation owner, initial stream readiness, alias identity and
# completion handoff are part of import/withview. Mantle never pool-frees it.
# Resize beyond the imported allocation bound rejects rather than moving it.

# 15. Image views carry resource/format/subresource information.
texture = Image(dev, RGBA{Float32}, 128, 128, 64; mips=4)
mip = view(texture; mip=2)                       # second Julia mip; native level 1
# KI.imagestore!/imageload/imageatomic! are planned typed intrinsic families;
# sample(mip, sampler, coords) selects only a qualified format/dimension path.
# Existing 2-D sampling support is not evidence for all these new paths.

# 16. Dynamic graph readbacks use the public Graph lifecycle.
# A request includes warmup if that is the next run; rearm before each wanted run.
# Repeated fetch returns the same captured snapshot, never the newest device state.
read_graph = Graph(dev)
dynamic = MantleArray(read_graph, Float32, 0; capacity=1024)
app_produce_dynamic!(read_graph, dynamic)
future = call!(read_graph, Array, dynamic)
run!(read_graph)
values = fetch(future)
# After run!, Array(dynamic) is the ordinary blocking snapshot of accepted work.
next_future = Readback(future)                   # bind next execution
run!(read_graph)
free!(read_graph)
next_values = fetch(next_future)                  # submitted future retains snapshot

# 17. Explicit device transfer is the only mixed-device operation here.
function transfer_graph(dstdev, dst, src)
    graph = Graph(dstdev)
    call!(graph, copyto!, dst, src)                       # asynchronous recorded transport
    graph
end
# Native peer/external candidates or a visible host-cut bridge share snapshot,
# destination publication and holds. devicecopy!(dst,src) is synchronous convenience.
# Ordinary dst .= src with different devices still rejects.

# 18. Windows have one presenting graph and separately retired presentation.
window = Window(dev, 1280, 720)
presenting = Graph(dev)
color = Image(presenting, RGBA{Float32}, window; scratch=true)
clear!(presenting, color, (0f0, 0f0, 0f0, 1f0))
draw!(presenting, window, frame_pipeline, (color,), 6) # final window draw establishes presenting graph
frame_snapshot = call!(presenting, Array, window)   # after final writer, before present
run!(presenting)
frame_pixels = fetch(frame_snapshot)               # no future-frame wait or new submission
# Resize within capacity refreshes native generations/recordings. Beyond it
# recompiles while persistent resources survive. Rendering tokens are not WSI fences.

# 19. No target means eager, regardless of which graphs already name an array.
# For same-device persistent A, B, C:
# mul!(C, A, B) executes eagerly; call!(step, mul!, C, A, B) records and returns C itself.
# KA.get_backend(C) stays the same Mantle Device in both cases. No alias retargeting.
