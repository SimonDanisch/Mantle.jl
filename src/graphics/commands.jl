# Rasterization commands: declared here without a `vk_`/`mtl_` prefix,
# implemented by a backend.

"""
    begin_pass!(emitter, color_views, color_images, extent; clear_color, depth_view, ...)

Begin a render pass on `emitter`; must be matched by [`end_pass!`](@ref). A draw
outside a pass is an error, since a driver that accepts one draws nothing and
reports nothing.
"""
function begin_pass! end

"""
    end_pass!(emitter)

End the render pass begun by [`begin_pass!`](@ref).
"""
function end_pass! end

"""
    draw_in_pass!(emitter, pipeline, count; instances = 1)

Issue a non-indexed draw inside an open pass.
"""
function draw_in_pass! end

"""
    draw_indexed_in_pass!(emitter, pipeline, count; indices_buffer, instances = 1)

Issue an indexed draw inside an open pass.
"""
function draw_indexed_in_pass! end

"""
    draw_indirect_in_pass!(emitter, pipeline, commands; first = 1, count = 1)

Issue draws whose parameters the device reads from `commands`, as
[`DrawIndirectCommand`](@ref) records, so a culling pass can set the vertex
counts without a host round trip.
"""
function draw_indirect_in_pass! end

"""
    set_viewport!(emitter, x, y, width, height)

Set the viewport rectangle for subsequent draws.
"""
function set_viewport! end

"""
    present_frame!(device, window)

Show what was drawn, ordered behind the frame's work: presenting an image the
GPU has not finished writing shows a torn frame and reports nothing. Takes the
device, not a channel; the backend picks the channel.
"""
function present_frame! end

"""
    acquire_next_image!(window)

Take the image the next frame renders into. Called by `beforeframe!` after
[`beginframe!`](@ref) and before `refit!`: it can block (it paces the frame
rate) and is the last place a resize is noticed. Acquiring after the refit
rebuilds the swapchain under a depth target sized for the old one, which loses
the device on NVIDIA.
"""
function acquire_next_image! end

"""
    checkacquirable(acquirer)

Refuse a second image while `acquirer`, the task holding the window's image, has
not presented it; `nothing` is a window between frames. Compiling a kernel
mid-frame yields, so a second task can arrive between acquire and present. On
Vulkan its acquire would block forever in a foreign call; on Metal two frames
would draw at once.
"""
checkacquirable(::Nothing) = nothing
function checkacquirable(acquirer::Task)
    who = acquirer === current_task() ?
          "the same task that is asking for another one" :
          "another task ($acquirer) that has not presented it yet"
    error("acquire_next_image!: this window already holds an image, acquired by " *
          who * ". A window is rendered by one task: pass it around, or hand it " *
          "over once the frame that owns it has been presented.")
end

"""
    readback_framebuffer(fb) -> Array

Read a [`Framebuffer`](@ref)'s colour attachment back to the host. Waits for the
device to finish writing it.
"""
function readback_framebuffer end

"""
    readback_window(window) -> Array

Read an image of `window` back to the host between frames: acquire one, copy it,
give it back. The platform picks the image, so it holds an earlier frame; the
Vulkan backend hands back an image not drawn since a resize and acquires again.
Inside a frame [`checkreadable`](@ref) refuses, since the copy would have to go
in the frame's own submission.
"""
function readback_window end

"""
    checkreadable(acquirer)

Refuse a readback while `acquirer` holds the window's image (see
[`readback_window`](@ref)); `nothing` is a window between frames.
"""
checkreadable(::Nothing) = nothing
checkreadable(::Task) =
    error("readback_window: the window holds an acquired image; read back after " *
          "the frame that owns it has been presented")

"""
    readback_target(plan, target) -> Matrix

Read a render target `plan` placed back to the host after a run, as a `Matrix`
of its element type, through a copy since a transient may live in device-only
memory. Takes the plan because between runs the image rests in the layout of its
last usage, which only the plan records. Synchronises; inside a frame, use a
`copy!` pass into a [`Transient.Buffer`](@ref).
"""
function readback_target(pl::Plan, t)
    rest = restingusage(pl, t)
    rest === nothing && throw(ArgumentError(
        "readback_target: the plan never uses this target, so it holds nothing to read"))
    return readback_target(pl.graph.dev, t, rest)
end

"""
    readback_target(device, target, usage) -> Matrix

The backend's half of [`readback_target`](@ref): copy `target` to the host, given
`usage`, the usage the plan left it in.
"""
readback_target(dev::Device, t, ::Type) = throw(ArgumentError(
    "readback_target: $(typeof(dev)) cannot read back a render target"))

"""
    restingusage(plan, r) -> Union{Type,Nothing}

The destination of the last transition `plan` makes on `r`, in scheduled order:
the layout `r` rests in between runs. `nothing` when the plan never touches `r`.
"""
function restingusage(pl::Plan, r)
    id = get(pl.graph.ids, r, nothing)
    id === nothing && return nothing
    rest = nothing
    for pp in pl.passes, t in pp.pre
        t.resource == id && (rest = t.to)
    end
    return rest
end

"""
    supports_graphics(backend) -> Bool

Whether `backend` can rasterize: render passes, graphics pipelines, vertex and
fragment stages. `false` by default so a backend opts in. Callers that can
degrade ask; RayMakie takes its direct-readback path when it is `false`.

The untyped default would answer `false` for a device nobody wrote a method for,
so the `::Device` method routes to the device's backend. A backend may still
define a more specific method for its device.
"""
supports_graphics(backend) = false
supports_graphics(dev::Device) = supports_graphics(backend(dev))

"""
    supports_geometry_stage(backend) -> Bool
    supports_tessellation(backend) -> Bool

Whether this backend has the stage, so a caller learns it before a shader
compile fails. `false` by default (Metal has neither). Without the stage a
geometry pipeline can still be drawn through [`lower_geometry_to_mesh`](@ref),
as Metal's `compile_draw` does; [`supports_mesh_pipeline`](@ref) asks that.
"""
supports_geometry_stage(backend) = false
@doc (@doc supports_geometry_stage) supports_tessellation(backend) = false

# Route a device to its backend's answer; see `supports_graphics`.
supports_geometry_stage(dev::Device) = supports_geometry_stage(backend(dev))
supports_tessellation(dev::Device) = supports_tessellation(backend(dev))

"""
    supports_mesh_pipeline(backend) -> Bool

Whether this backend can run a [`MeshPipeline`](@ref) (Metal mesh pipelines,
`VK_EXT_mesh_shader`). A geometry stage is expressible on a mesh pipeline through
[`lower_geometry_to_mesh`](@ref), so this is the question for a caller holding a
geometry body. `false` by default.
"""
supports_mesh_pipeline(backend) = false
supports_mesh_pipeline(dev::Device) = supports_mesh_pipeline(backend(dev))

"""
    use_bindings!(handle, compiled, bindings)

Select `bindings`, a set built by [`bind_textures`](@ref), for the next draw on
`handle`. Separate from building so a set is built once when its textures change
and selected per draw; an atlas rebuild does not re-record the draws sampling it.
"""
function use_bindings! end

# Per-pass verbs for a backend that records pass by pass. Mantle decides which
# pass runs, in what order, with which attachments and draws; the backend opens
# a pass, records what it is handed and closes it, so `begin_render_pass!` takes
# resolved attachments rather than the `Pass`.

"""
    compile_draw(device, pipeline, color_formats, depth_format, vert_args, frag_args;
                 bindings = nothing)

Compile a [`GraphicsPipeline`](@ref) for the attachments it draws into. Called
once per draw at compile time, not per frame; a backend is expected to cache.

`color_formats` are element types, as a `Transient.Image` is named.
`vert_args`/`frag_args` are the resolved values, not their types, because a
backend derives a stage's device signature by adapting each value as a launch
would. `bindings` is the texture table from [`bind_textures`](@ref) or `nothing`;
Vulkan builds the pipeline layout around its descriptor set layout, Metal ignores
it. The result holds the compiled pipeline in a `pipeline` field, which
[`npipelines`](@ref) counts.
"""
function compile_draw end

"""
    checkdepthtarget(depth, hasdepth::Bool)

Refuse a depth test against a render target without a depth attachment instead
of drawing untested. Core calls it before every `compile_draw`, for raster and
mesh pipelines alike.
"""
function checkdepthtarget(depth, hasdepth::Bool)
    depth isa DepthOff || hasdepth || throw(ArgumentError(
        "$(typeof(depth)) needs a depth attachment, and this render target has none. " *
        "Give the target a depth view, or build the pipeline with `depth = DepthOff()`."))
    return nothing
end

"""
    begin_render_pass!(device, targets, loads, depth, depth_load) -> handle

Open a render pass over the given attachments. `targets` and `loads` are
parallel, one [`LoadOp`](@ref) per colour attachment; `depth` is `nothing` when
the pass has none. The handle is opaque to Mantle and goes back to
`record_draw!` and `end_render_pass!`.
"""
function begin_render_pass! end

"""
    record_draw!(handle, compiled, args, count; instances = 1, indices = nothing)

Record one draw into an open pass. `count` is a number, or a `Commands` naming a
buffer of `DrawIndirectCommand` the GPU wrote, for which the backend must issue
an indirect draw because the host never learns the count.
"""
function record_draw! end

"""
    argidentity(x)

What a recorded draw baked about `x`, or `nothing`. `record_draw!` bakes the
address of the packed argument block, so a device array that `resize!`
reallocated is invisible to an older plan; a caller caching plans compares this
value, which changes exactly then. Everything but a device array answers
`nothing`: a uniform rebinds through the draw's cell, and keying on its value
would rebuild a plan per camera move. Backends implement it for their array type.
"""
argidentity(@nospecialize(x)) = nothing

# Frame verbs, called by `run!` around the passes for every surface the graph
# has, in the order poll, acquire, record, present. What each step means is the
# backend's: a swapchain image and two semaphores, or a `CAMetalDrawable`.

"""
    beginframe!(window)

Bring `window` up to date before anything is recorded: pump the platform's event
queue and pick up a resize. Here rather than inside `isopen`, so a predicate does
not pump events and no frame loop has to remember to poll.
"""
function beginframe! end

# `acquire_next_image!` and `present_frame!` above are the other two. Not
# `acquire!`: that is the pool's verb.

"""
    copy_target!(device, dst, src; region = nothing)

Copy a render target's pixels into a buffer on the device, the body of a `copy!`
pass. `src` is an attachment the graph placed, `dst` a
[`Transient.Buffer`](@ref) of its element type sized `width * height`; the graph
checked both when the pass was declared. `region` limits the copy to the pass's
viewport. A verb because a render target is not addressable memory on any
backend; no host round trip is allowed.
"""
function copy_target! end

# A core method rather than a bare declaration: `test_core_names_no_backend.jl`
# reads a bare `function f end` as a name only a backend defines.
copy_target!(dev::Device, dst, src) = throw(ArgumentError(
    "copy_target!: $(typeof(dev)) does not implement the `copy!` pass. A render " *
    "target is not addressable memory, so each backend moves those bytes its own " *
    "way — `vkCmdCopyImageToBuffer` on Vulkan, a blit encoder on Metal."))

"""
    end_render_pass!(handle)

Close the pass opened by [`begin_render_pass!`](@ref) and submit it.
"""
function end_render_pass! end
