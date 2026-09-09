# Rasterization commands.
#
# These were `vk_begin_pass!`, `vk_draw_in_pass!`, `vk_set_viewport!` and the
# rest: Mantle verbs with a driver's name bolted on the front. Beginning a
# render pass and issuing a draw are not things Vulkan invented, and a caller
# writing `vk_draw_in_pass!` on a Mac would be naming the wrong API for the
# hardware it is running on.
#
# Declared here, implemented per backend. The prefix is gone rather than swapped
# for `mtl_`/`vk_` at the call site, which is the point: there is one verb.

"""
    begin_pass!(target; clear = nothing)

Begin a render pass against `target`, a [`RenderTarget`](@ref).

Must be matched by [`end_pass!`](@ref). Draws issued between the two go to this
target; draws issued outside a pass are an error, not a no-op, because a driver
that accepts them produces nothing and says nothing.
"""
function begin_pass! end

"""
    end_pass!(target)

End the render pass begun by [`begin_pass!`](@ref).
"""
function end_pass! end

"""
    draw_in_pass!(pipeline, vertices; instances = 1)

Issue a non-indexed draw inside an open pass.
"""
function draw_in_pass! end

"""
    draw_indexed_in_pass!(pipeline, indices; instances = 1)

Issue an indexed draw inside an open pass.
"""
function draw_indexed_in_pass! end

"""
    draw_indirect_in_pass!(pipeline, buffer)

Issue a draw whose parameters the device reads from `buffer`, as
[`DrawIndirectCommand`](@ref) records.

The count comes from the device too, which is the reason to use this at all: a
culling pass can decide how much to draw without a round trip to the host.
"""
function draw_indirect_in_pass! end

"""
    set_viewport!(target, x, y, width, height)

Set the viewport rectangle for subsequent draws.
"""
function set_viewport! end

"""
    reset_device!(device)

Tear down and re-create the device's transient state.

The recovery path after a device loss, and the reason it is API rather than
internal: a caller that has just lost the device is the only one who knows
whether re-creating it is the right response.
"""
function reset_device! end

"""
    blit!(dst, src)

Copy an image to another, converting format and scaling as needed.
"""
function blit! end

"""
    present_frame!(submitter, window)

Show what was drawn, once the work behind it completes.

`submitter` is whatever this backend submits through — a [`BatchQueue`](@ref) on
one, the device itself on the other. That is the only part that differs, and it
differs because the two have genuinely different submission machinery; the verb,
the ordering and the moment are the same.

Ordered behind the frame rather than issued immediately: presenting an image the
GPU has not finished writing shows a torn frame, and nothing reports it.
"""
function present_frame! end

"""
    acquire_next_image!(window)

Take the image the next frame renders into.

After [`beginframe!`](@ref) and before `refit!`, both from `beforeframe!`: on
both backends this can BLOCK — it is where the frame rate comes from — and it is
the last place a resize is noticed, so the attachments that follow the surface
are refit after it. Acquiring later, with the frame already refit, rebuilt a
swapchain under a depth target sized for the old one: a lost device two frames
later, on NVIDIA, one resize in three.
"""
function acquire_next_image! end

"""
    transition_image!(image, from, to)

Move `image` between usage states.

Explicit because the graph derives these automatically for anything it knows
about; this is the escape hatch for an image it does not, such as one shared
with another API through an [`ExternalImage`](@ref).
"""
function transition_image! end

"""
    readback_framebuffer(fb) -> Array

Read a [`Framebuffer`](@ref)'s colour attachment back to the host.

Synchronises: there is no way to read pixels the device has not finished
writing, so unlike the rest of the frame loop this one waits.
"""
function readback_framebuffer end

"""
    readback_window(window) -> Array

Read the last presented image of `window` back to the host.
"""
function readback_window end

"""
    readback_target(target) -> Matrix

Read a render target the graph placed back to the host, as a `Matrix{T}` of its
element type.

The counterpart to [`readback_framebuffer`](@ref) for a
[`Transient.Image`](@ref): a framebuffer is a resource the CALLER made and can
keep in host-visible memory, while a transient lives wherever the placer put it
— on a backend that is device-only memory, so this goes through a copy.

Synchronises, for the same reason the other two do.
"""
function readback_target end

"""
    supports_graphics(backend) -> Bool

Whether `backend` can rasterize: render passes, graphics pipelines, and the
vertex/fragment stages the verbs above record into.

`false` by default, so a backend opts in rather than inheriting a claim it
cannot honour. Vulkan says `true`. Metal says `false` and will keep saying it
for as long as Metal.jl compiles Julia to compute kernels only — there is no
`MTLRenderPipelineState` wrapper and no vertex/fragment path for
`@device_override`d Julia to compile into, so this is a missing capability in
the toolchain, not a switch someone forgot to flip.

Callers that can degrade should ask. RayMakie composites its overlays through a
graphics pipeline and has a direct-readback path for scenes with none, so it
asks here and takes the readback path rather than failing on a Mac.
"""
supports_graphics(backend) = false

"""
    supports_geometry_stage(backend) -> Bool
    supports_tessellation(backend) -> Bool

Whether this backend has the stage at all.

`false` by default, so a backend opts in — and both defaults are the honest
answer for Metal, which has no geometry stage (Apple's replacement is the mesh
pipeline, which Mantle does not describe) and no tessellation through Mantle.

They exist because `GraphicsPipeline` HAS a `geometry` field and a
`tess_control`/`tess_eval` pair, so a caller can build one that a device cannot
run — and the only way to find out was to compile it and read the error. A
capability a caller cannot ask about is one they find out about from a shader
compile, which is the wrong place and the wrong time. RayMakie's overlay is the
caller that needs the answer: its scatter path emits geometry.
"""
supports_geometry_stage(backend) = false
@doc (@doc supports_geometry_stage) supports_tessellation(backend) = false

"""
    supports_mesh_pipeline(backend) -> Bool

Whether this backend can run a [`MeshPipeline`](@ref).

The pair to compare against [`supports_geometry_stage`](@ref), and the reason
that one answering `false` is not the end of the sentence. Metal has no geometry
stage, but Apple's replacement for it is the mesh pipeline, and Khronos arrived
at the same two stages under different names in `VK_EXT_mesh_shader` — so this
is the capability a caller with a geometry body actually wants to ask about. A
geometry stage is expressible on a mesh pipeline; the reverse is not.

`false` by default, so a backend opts in.
"""
supports_mesh_pipeline(backend) = false

"""
    use_bindings!(bq, pipeline, bindings)

Make `bindings` — the resource set built by [`bind_textures`](@ref) — the one
the next draw on `bq` reads from.

Separate from `bind_textures` because that one BUILDS a set and this one
SELECTS it: a set is built once when the textures change and selected once per
draw. Keeping the two apart is what lets a renderer rebuild an atlas without
re-recording every draw that samples it.
"""
function use_bindings! end

# ── executing a render pass on a KernelAbstractions backend ──────────────────
#
# The graph already carries everything a render pass is: `Pass.targets` and
# `Pass.loads` are the colour attachments and what to do with them,
# `Pass.depth` / `Pass.depth_load` the depth one, and `Pass.draws` the draws in
# order. What was missing is a way to EXECUTE that on a backend which records
# per draw rather than into a batch queue.
#
# So these four verbs, and nothing more. Mantle decides which pass runs, in what
# order, with which attachments and which draws; a backend opens a pass, records
# what it is handed, and closes it. Anything shaped like scheduling belongs on
# this side of the line, which is why `begin_render_pass!` takes the attachments
# already resolved rather than the `Pass`.
#
# The Vulkan backend does NOT go through these: it records whole frames into a
# `BatchQueue` and has its own path in `src/vulkan/graph.jl`. These exist for
# backends that have no batch queue — see `supports_batch_queue`.

"""
    compile_draw(device, pipeline, color_formats, depth_format, vert_args, frag_args)

Compile a [`GraphicsPipeline`](@ref) for the attachments it will be drawn into.

Called once per draw at COMPILE time, not per frame: the formats and the
arguments are fixed by then, and they are the whole of what a compiled pipeline
is specialised on. A backend is expected to cache.

`color_formats` are ELEMENT TYPES, the same vocabulary a caller names a
`Transient.Image` in; the backend lowers them the way it lowers every other
format. `vert_args`/`frag_args` are the RESOLVED arguments — the values, not
their types — because what a stage's device signature is differs per backend:
one derives it by adapting each value the way a kernel launch would, and only
the value can answer that.
"""
function compile_draw end

"""
    begin_render_pass!(device, targets, loads, depth, depth_load) -> handle

Open a render pass over the given attachments and hand back whatever the backend
needs to record into.

`targets` and `loads` are parallel, one [`LoadOp`](@ref) per colour attachment;
`depth` is `nothing` when the pass has none. The handle is opaque to Mantle and
is passed straight back to `record_draw!` and `end_render_pass!`.
"""
function begin_render_pass! end

"""
    record_draw!(handle, compiled, args, count; instances = 1, indices = nothing)

Record one draw into an open pass.

`count` is what `drawover` resolved: a number, or a `Commands` naming a buffer of
`DrawIndirectCommand` the GPU wrote — in which case the backend must issue an
INDIRECT draw, because the whole point of that form is that the host never
learns the count.
"""
# `instances` and `indices` are keywords with defaults, so the graph — which never
# uses either — calls this exactly as it did. They are here because a
# hand-recorded pass does: RayMakie's overlay draws instanced sprites and indexed
# line strips, and those were `draw_in_pass!`/`draw_indexed_in_pass!`, a second
# verb family only one backend implemented. One primitive with two options beats
# two primitives, and beats a family.
function record_draw! end

# ── The frame, for a graph that reaches a window ─────────────────────────────
#
# Three verbs, called by `run!` around the passes for every surface the graph
# has. The SEQUENCE is Mantle's — poll, acquire, record, present, in that order
# and no other — and what each step means is the backend's: a swapchain image
# and two semaphores on one, a `CAMetalDrawable` on the other.
#
# The Vulkan backend does not come through here either; it records whole frames
# itself. These exist so a backend without a batch queue can still drive a
# window, which until now it could not: the shared `run!` walked passes and
# nothing else, so a `Surface` in the graph was an attachment nobody acquired.


"""
    beginframe!(window)

Bring `window` up to date before anything is recorded: pump the platform's
event queue, and pick up a resize.

Once per frame and HERE rather than inside `isopen`, because a predicate that
also pumps events is a surprise, and a frame loop that has to remember to poll
is one that stops responding the day someone forgets.
"""
function beginframe! end

# `acquire_next_image!` and `present_frame!` are the other two, and they are
# NOT declared here — they are older than this block, up beside `blit!`. This
# file briefly declared an `acquire!`/`present!` pair beside them, which was two
# portable APIs for one operation, and `acquire!` is worse than redundant: it is
# the POOL's verb (`acquire!(pool, dev, kind, transients, bytes)`), so a window
# and an allocator shared a name in one exported namespace.

"""
    copy_target!(device, dst, src)

Copy a render target's pixels into a buffer, on the device.

The body of a `copy!` pass. `src` is an attachment the graph placed and `dst` a
[`Transient.Buffer`](@ref) of its element type, sized `width * height` — the
graph checked both when the pass was declared, so this only has to move bytes.

It exists as a verb because a render target is not addressable memory on either
backend: Vulkan needs `vkCmdCopyImageToBuffer` and Metal a blit encoder, and a
deferred renderer reads its whole g-buffer this way once per frame. No host
round-trip is involved and none is allowed — the destination is device memory a
compute pass reads next.
"""
function copy_target! end

"""
    end_render_pass!(handle)

Close the pass opened by [`begin_render_pass!`](@ref) and submit it.
"""
function end_render_pass! end
