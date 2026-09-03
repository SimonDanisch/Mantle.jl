# High-level graphics API for Lava.jl
#
# GraphicsPipeline — the central type for configuring and dispatching
# graphics shaders. Supports lazy compilation, caching, and multiple
# draw dispatch via RenderTarget types.


# ── Lazy Compilation ──


# The graphics shader cache is a `VkContext` field; a reset makes a new one.

"""
Everything that changes the created `VkPipeline` but is not the shader pair.

Blend, cull, topology and depth mode are singletons, so `typeof(pipeline)` carries
them exactly; `varyings` and the geometry/tessellation configs are values, so they
are hashed as values. Leaving state out of the key made two pipelines that differ
only in, say, blend mode or depth mode share one compiled pipeline — whichever was
compiled first won, and the second draw silently rendered with the wrong state.
"""
pipeline_state_key(p::GraphicsPipeline) = (typeof(p), p.varyings, p.geometry, p.tess_control)

"""Return (vert_shader::LavaGfxShader, compiled::VulkanCompiledGraphicsPipeline)."""
function ensure_compiled_with_shader!(pipeline::GraphicsPipeline,
                              vert_fn, frag_fn, tt_vertex, tt_fragment;
                              color_format=VK.FORMAT_B8G8R8A8_SRGB,
                              depth_format=VK.FORMAT_UNDEFINED,
                              descriptor_set_layout=nothing)
    vert = get_or_compile_gfx(vert_fn, tt_vertex, :vertex)
    compiled = ensure_compiled!(pipeline, vert_fn, frag_fn, tt_vertex, tt_fragment;
        color_format, depth_format, descriptor_set_layout)
    return vert, compiled
end

function ensure_compiled!(pipeline::GraphicsPipeline, vert_fn, frag_fn, tt_vertex, tt_fragment;
                              color_format=VK.FORMAT_B8G8R8A8_SRGB,
                              depth_format=VK.FORMAT_UNDEFINED,
                              descriptor_set_layout=nothing,
                              ctx::VkContext = vk_context())
    # Cache key includes type tuples — different arg types get different compiled
    # pipelines — and the pipeline state, which is the rest of what is baked in.
    cache_key = hash((vert_fn, frag_fn, tt_vertex, tt_fragment, color_format, depth_format,
                       pipeline_state_key(pipeline), descriptor_set_layout !== nothing))
    cached = get(ctx.caches.gfx_pipelines, cache_key, nothing)
    cached !== nothing && return cached::VulkanCompiledGraphicsPipeline

    # Compile vertex shader
    vert = get_or_compile_gfx(vert_fn, tt_vertex, :vertex)

    # Compile fragment shader
    frag = get_or_compile_gfx(frag_fn, tt_fragment, :fragment)

    # Optional stages
    geom_spirv = nothing
    geom_config = nothing
    if pipeline.geometry !== nothing
        geom_fn, geom_cfg = pipeline.geometry
        geom = get_or_compile_gfx(geom_fn, tt_vertex, :geometry; config=geom_cfg)
        geom_spirv = geom.spirv_bytes
        geom_config = geom_cfg
    end

    tc_spirv = nothing
    te_spirv = nothing
    tess_cfg = nothing
    if pipeline.tess_control !== nothing
        tc_fn, tc_cfg = pipeline.tess_control
        tc = get_or_compile_gfx(tc_fn, tt_vertex, :tess_control; config=tc_cfg)
        tc_spirv = tc.spirv_bytes
        tess_cfg = tc_cfg
    end
    if pipeline.tess_eval !== nothing
        te = get_or_compile_gfx(pipeline.tess_eval, tt_vertex, :tess_eval;
            config=tess_cfg)
        te_spirv = te.spirv_bytes
    end

    compiled = create_graphics_pipeline(vert.spirv_bytes, frag.spirv_bytes;
        blend=pipeline.blend, cull=pipeline.cull,
        topology=pipeline.topology, depth=pipeline.depth,
        color_format=color_format, depth_format=depth_format,
        push_constant_size=max(vert.push_info.push_size, frag.push_info.push_size),
        geometry_spirv=geom_spirv,
        tess_ctrl_spirv=tc_spirv, tess_eval_spirv=te_spirv,
        tess_config=tess_cfg,
        descriptor_set_layout=descriptor_set_layout)

    ctx.caches.gfx_pipelines[cache_key] = compiled
    return compiled
end

function get_or_compile_gfx(@nospecialize(f), @nospecialize(tt), stage::Symbol;
                            config=nothing, ctx::VkContext = vk_context())
    key = hash((f, tt, stage, config))
    cached = get(ctx.caches.gfx_shaders, key, nothing)
    cached !== nothing && return cached
    shader = lava_compile_gfx_shader(f, tt; stage, config)
    ctx.caches.gfx_shaders[key] = shader
    return shader
end

# ── Draw API ──

"""Convert args to device-side values (LavaArray → LavaDeviceArray, scalars pass through)."""
convert_args(args::Tuple) = map(arg -> arg isa LavaArray ? LavaDeviceArray(arg) : arg, args)
convert_args(::Tuple{}) = ()

"""
Build the full vertex output NamedTuple type from a varyings spec.
E.g. `(normal=Vec3f, uv=Vec2f)` → `@NamedTuple{position::Vec4f, normal::Vec3f, uv::Vec2f}`
"""
function varyings_to_output_type(varyings::NamedTuple)
    names = (:position, keys(varyings)...)
    types = (Vec4f, values(varyings)...)
    return NamedTuple{names, Tuple{types...}}
end

"""
Resolve vertex/fragment functions and type tuples, wrapping NamedTuple-returning
shaders with VertexWrapper/FragmentWrapper as needed.
Returns (vert_fn, vert_tt, frag_fn, frag_tt).
"""
function resolve_shader_pair(pipeline, vert_tt::Type, frag_tt::Type)
    if pipeline.varyings !== nothing
        vout = varyings_to_output_type(pipeline.varyings)
        wrapped_vert = VertexWrapper{typeof(pipeline.vertex)}()
        wrapped_frag = FragmentWrapper{typeof(pipeline.fragment), vout}()
        return wrapped_vert, vert_tt, wrapped_frag, frag_tt
    else
        # Legacy API: user calls gfx_output/gfx_input directly
        return pipeline.vertex, vert_tt, pipeline.fragment, frag_tt
    end
end

"""
    draw!(pipeline::GraphicsPipeline, target::RenderTarget, vertex_count;
          args=(), frag_args=(), instances=1,
          clear_color=(0f0, 0f0, 0f0, 1f0))

Draw using the given graphics pipeline to the render target.
Device-side type tuples are inferred automatically from args.
"""
function draw!(bq::VulkanBatchQueue, pipeline::GraphicsPipeline, target::WindowTarget, vertex_count::Integer;
               args=(), frag_args=(), instances::Integer=1,
               clear_color::Union{Nothing, NTuple{4, Float32}}=(0.0f0, 0.0f0, 0.0f0, 1.0f0))
    win = target.window

    converted_vert = convert_args(args)
    converted_frag = convert_args(frag_args)
    vert_tt = typeof(converted_vert)
    frag_tt = typeof(converted_frag)

    vert_fn, vert_tt, frag_fn, frag_tt = resolve_shader_pair(pipeline, vert_tt, frag_tt)

    # A window target has no depth attachment, so the pipeline must not declare one.
    vert_shader, compiled = ensure_compiled_with_shader!(pipeline,
        vert_fn, frag_fn, vert_tt, frag_tt;
        color_format=win.format, depth_format=VK.FORMAT_UNDEFINED)

    view = win.views[win.current_image_idx + 1]
    image = win.images[win.current_image_idx + 1]

    push_data = isempty(args) ? UInt8[] : pack_gfx_args(bq, args, vert_shader.push_info)

    vk_draw!(bq, compiled, view, image, win.extent, vertex_count;
        push_data, instances, clear_color)
end

function draw!(bq::VulkanBatchQueue, pipeline::GraphicsPipeline, target::OffscreenTarget, vertex_count::Integer;
               args=(), frag_args=(), instances::Integer=1,
               clear_color::Union{Nothing, NTuple{4, Float32}}=(0.0f0, 0.0f0, 0.0f0, 1.0f0),
               depth_clear::Union{Nothing, Float32}=1.0f0,
               descriptor_set_layout=nothing,
               descriptor_set=nothing)
    fb = target.fb

    converted_vert = convert_args(args)
    converted_frag = convert_args(frag_args)
    vert_tt = typeof(converted_vert)
    frag_tt = typeof(converted_frag)

    vert_fn, vert_tt, frag_fn, frag_tt = resolve_shader_pair(pipeline, vert_tt, frag_tt)

    vert_shader, compiled = ensure_compiled_with_shader!(pipeline,
        vert_fn, frag_fn, vert_tt, frag_tt;
        color_format=fb.color_format,
        depth_format=fb.depth_view === nothing ? VK.FORMAT_UNDEFINED : fb.depth_format,
        descriptor_set_layout)

    push_data = isempty(args) ? UInt8[] : pack_gfx_args(bq, args, vert_shader.push_info)

    vk_draw!(bq, compiled, fb.color_view, fb.color_image,
        VK.Extent2D(UInt32(fb.width), UInt32(fb.height)),
        vertex_count;
        push_data, instances,
        depth_view=fb.depth_view,
        depth_image=fb.depth_image,
        depth_clear,
        clear_color, descriptor_set)
end

"""
    pack_gfx_args_bda(bq, args, push_info) -> UInt64

Pack a draw's arguments and return the buffer device address to push, rather than
an eight-byte `Vector` holding it.

The push constant is one BDA and always has been; wrapping it in a heap array
allocated 48 bytes of header plus the payload *per draw per frame*, which at a
hundred plots is most of a frame's garbage. `pack_gfx_args` still returns the
vector for callers written against it.
"""
function pack_gfx_args_bda(bq::VulkanBatchQueue, args, push_info::PushConstantInfo)
    (push_info.push_size == 0 || isempty(args)) && return UInt64(0)
    batch = ensure_active_batch!(bq)
    adaptor = LavaAdaptor(batch)
    converted = map(a -> Adapt.adapt(adaptor, a), args)
    byval_sizes = push_info.byval_llvm_sizes
    total_size = push_info.arg_buffer_size + compute_inline_extra_from_byval(byval_sizes)
    arg_buf = get_arg_buffer(batch, total_size)
    pack_args_direct!(batch, arg_buf.mapped_ptr, arg_buf.address, push_info.arg_offsets,
                      push_info.arg_buffer_size, byval_sizes, converted)
    return arg_buf.address
end

function pack_gfx_args(bq::VulkanBatchQueue, args, push_info::PushConstantInfo)
    push_info.push_size == 0 && return UInt8[]
    isempty(args) && return UInt8[]

    batch = ensure_active_batch!(bq)
    # LavaAdaptor is the single point that strips LavaArray → LavaDeviceArray
    # AND pins the original into batch.pinned.  Adapt.jl's recursion handles
    # wrapper structs / Broadcasted / NamedTuple, so any nested LavaArray
    # gets both its pointer strip and its pin in the same pass.
    adaptor = LavaAdaptor(batch)
    converted = map(a -> Adapt.adapt(adaptor, a), args)

    # Use the same arg buffer packing as compute: inline byval structs into
    # the arg buffer with self-referencing BDA pointers.
    offsets = push_info.arg_offsets      # precomputed: this ran per draw per frame
    byval_sizes = push_info.byval_llvm_sizes

    inline_extra = compute_inline_extra_from_byval(byval_sizes)
    total_size = push_info.arg_buffer_size + inline_extra

    arg_buf = get_arg_buffer(batch, total_size)

    pack_args_direct!(batch, arg_buf.mapped_ptr, arg_buf.address, offsets,
                       push_info.arg_buffer_size, byval_sizes, converted)

    # Push constant = BDA of arg buffer
    push_data = Vector{UInt8}(undef, 8)
    GC.@preserve push_data begin
        unsafe_store!(Ptr{UInt64}(pointer(push_data)), arg_buf.address)
    end
    return push_data
end

# Empty-args variant
function pack_gfx_args(::VulkanBatchQueue, args, ::Nothing=nothing)
    isempty(args) && return UInt8[]
    error("pack_gfx_args requires push_info for non-empty args.")
end

# ── Blit: Fullscreen Display of GPU Buffer ──

# Built-in vertex shader for fullscreen triangle (3 vertices, no buffer)
function blit_vertex()
    vid = vertex_index() - Int32(1)  # 0-based for bit tricks
    # Fullscreen triangle: covers entire NDC [-1,1]×[-1,1]
    # vid=0: (-1,-1), vid=1: (3,-1), vid=2: (-1,3)
    x = Float32(Int32(vid & Int32(1)) * 4 - 1)
    y = Float32(Int32((vid >> Int32(1)) & Int32(1)) * 4 - 1)
    set_position!(Vec4f(x, y, 0.0f0, 1.0f0))
    # Pass UV coordinates
    u = (x + 1.0f0) * 0.5f0
    v = (y + 1.0f0) * 0.5f0
    gfx_output(0, Vec2f(u, v))
    return nothing
end

# Convert any RGBA-like color to Vec4f for fragment output
to_vec4f(v::Vec4f) = v
to_vec4f(c) = Vec4f(c.r, c.g, c.b, c.alpha)

# Built-in fragment shader for blitting a GPU buffer to screen.
# Buffer is in Julia column-major layout: element [row, col] is at linear index col * height + row + 1.
# Screen coords: fx = column (x), fy = row (y, 0 = top in Vulkan).
function blit_fragment(buffer, width::Int32, height::Int32)
    fx = frag_coord_x()
    fy = frag_coord_y()
    ix = unsafe_trunc(Int32, fx)   # column (0-based)
    iy = unsafe_trunc(Int32, fy)   # row (0-based)
    # Column-major indexing: col * height + row + 1
    idx = ix * height + iy + Int32(1)
    pixel = to_vec4f(buffer[idx])
    gfx_output(0, pixel)
    return nothing
end


# Likewise the blit pipeline — see `DeviceCaches.blit`.

"""
What shape a blit source has.

A matrix is `(height, width)` — `img[row, col]`, which is what everything that
calls an array an image means by it, and what `KernelAbstractions.allocate(be, T,
h, w)` gives. A vector is that matrix flattened, so the row index varies fastest
and pixel `(x, y)` is at `x * height + y + 1`.

A `(width, height)` matrix is the transposition, and it is what
`copy_image_to_buffer!` produces: an image copy packs rows, so its column index
varies fastest. Reading one and blitting it with the same index is a picture that
is sheared rather than wrong-looking, so the matrix case says so here. The vector
case cannot tell the two apart and only checks there are enough pixels.
"""
function checkblitsize(a::LavaArray{<:Any,2}, w::Integer, h::Integer)
    size(a) == (h, w) && return nothing
    # Only the transposition is an error. A size that merely disagrees is what a
    # resize looks like between the swapchain following the window and the caller
    # reallocating, and one frame of the wrong size there is not worth a throw.
    size(a) == (w, h) && throw(DimensionMismatch(
        "blit source is $(size(a)) for a $(w)x$(h) target, which is the transpose of " *
        "the $(h)x$(w) it wants. A source is a (height, width) matrix; an image read " *
        "back with `copy_image_to_buffer!` packs rows and so comes out the other way."))
    length(a) >= w * h || throw(DimensionMismatch(
        "blit source holds $(length(a)) pixels and a $(w)x$(h) target needs $(w * h)"))
    nothing
end
checkblitsize(a::LavaArray{<:Any,1}, w::Integer, h::Integer) =
    length(a) >= w * h ? nothing : throw(DimensionMismatch(
        "blit source holds $(length(a)) pixels and a $(w)x$(h) target needs $(w * h)"))

"""
    blit!(bq, target::RenderTarget, source::LavaArray; clear=true)

Display a GPU array on screen using a fullscreen blit.
The source array should contain RGBA Float32 pixels (or any 4-component type).
Its layout is `(height, width)` — see `checkblitsize`.
"""
function blit!(bq::VulkanBatchQueue, target::RenderTarget, source::LavaArray;
               clear::Bool=true)
    if target isa WindowTarget
        win = target.window
        w, h = size(win)
        color_format = win.format
        view = win.views[win.current_image_idx + 1]
        image = win.images[win.current_image_idx + 1]
        extent = win.extent
    elseif target isa OffscreenTarget
        fb = target.fb
        w, h = fb.width, fb.height
        color_format = fb.color_format
        view = fb.color_view
        image = fb.color_image
        extent = VK.Extent2D(UInt32(w), UInt32(h))
    else
        error("blit! only supports WindowTarget and OffscreenTarget")
    end
    checkblitsize(source, w, h)

    # Create or reuse blit pipeline. `bq.ctx`, not `vk_context()`: this pipeline
    # is a device-owned handle and the queue we are recording on names its device.
    ctx = bq.ctx::VkContext
    if ctx.caches.blit === nothing
        ctx.caches.blit = GraphicsPipeline(;
            vertex=blit_vertex,
            fragment=blit_fragment,
            blend=Opaque(),
            cull=NoCull(),
            depth=DepthOff(),
        )
    end

    pipeline = ctx.caches.blit

    # Fragment shader takes: (buffer::LavaDeviceArray, width::Int32, height::Int32)
    # Use the fragment shader's push_info for arg packing since vertex has no args.
    frag_args = (source, Int32(w), Int32(h))
    converted_frag = convert_args(frag_args)
    frag_tt = typeof(converted_frag)

    _, compiled = ensure_compiled_with_shader!(pipeline,
        pipeline.vertex, pipeline.fragment, Tuple{}, frag_tt;
        color_format=color_format)

    # Pack fragment args via the fragment shader's push_info
    frag_shader = get_or_compile_gfx(pipeline.fragment, frag_tt, :fragment)
    push_data = pack_gfx_args(bq, frag_args, frag_shader.push_info)

    clear_color = clear ? (0.0f0, 0.0f0, 0.0f0, 1.0f0) : nothing

    vk_draw!(bq, compiled, view, image, extent, 3;
        push_data, clear_color)
end

"""
    present_frame!(bq::VulkanBatchQueue, win::VulkanWindow)

Submit recorded draw commands and present to screen.
"""
function present_frame!(bq::VulkanBatchQueue, win::VulkanWindow)
    batch = bq.active_batch
    batch === nothing && error("present_frame! called without an active recording batch")
    cmd = batch.cmd_buf

    # Transition swapchain image to PRESENT_SRC before presenting
    image = win.images[win.current_image_idx + 1]
    transition_image!(cmd, image,
        VK.IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK.IMAGE_LAYOUT_PRESENT_SRC_KHR,
        VK.PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK.PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        VK.ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK.AccessFlag(0))

    # End command buffer
    throw_if_error(bq, "vkEndCommandBuffer", VK.end_command_buffer(cmd))

    fi = win.current_frame

    # ensure_active_batch! pre-assigned batch.signal_value = next_timeline + 1.
    # We must signal the timeline semaphore here so any `last_write` entries
    # set by dispatches in this batch can be observed as "complete".
    # Otherwise `wait_for_write(buf)` will hang forever on a value that
    # never arrives.
    bq.next_timeline += 1
    @assert batch.signal_value == bq.next_timeline "present_frame! signal desync"

    wait_infos = [
        # Wait for collected cross-queue deps:
        [VK.SemaphoreSubmitInfo(s, v, UInt32(0); stage_mask=stage)
         for (s, v, stage) in batch.wait_semaphores]...,
        # Wait for swapchain image availability before color attachment writes:
        VK.SemaphoreSubmitInfo(win.image_available[fi], UInt64(0), UInt32(0);
            stage_mask=VK.PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT),
    ]
    signal_infos = [
        # Signal timeline for in-Lava lifetime tracking:
        VK.SemaphoreSubmitInfo(bq.timeline_sem, batch.signal_value, UInt32(0);
            stage_mask=VK.PIPELINE_STAGE_2_ALL_COMMANDS_BIT),
        # Signal render_finished for the subsequent present operation. Indexed by
        # IMAGE, not frame slot — see the comment where these are created.
        VK.SemaphoreSubmitInfo(win.render_finished[win.current_image_idx + 1], UInt64(0), UInt32(0);
            stage_mask=VK.PIPELINE_STAGE_2_ALL_COMMANDS_BIT),
    ]
    # Sealed segments first, then the one still being recorded — the same order
    # `submit!` uses, and for the same reason: `maybe_split_cb!` ends the current
    # CB mid-frame and starts a fresh one, so everything recorded before the split
    # lives in `sealed_cmd_bufs`. Submitting only `batch.cmd_buf` here dropped it,
    # silently: no validation error, because nothing is invalid about commands
    # that are simply never handed to the queue. What it looked like was a frame
    # that did nothing, once every `cb_split_threshold` dispatches — and for a
    # graph whose first pass clears a counter that a later pass accumulates into,
    # a lost clear means the counter never restarts and the consumer indexes off
    # the end of its buffer.
    cb_infos = [VK.CommandBufferSubmitInfo(cb, UInt32(0)) for cb in batch.sealed_cmd_bufs]
    push!(cb_infos, VK.CommandBufferSubmitInfo(cmd, UInt32(0)))
    # Moved out of `sealed_cmd_bufs`, or the next frame submits them again: this
    # batch is reused across frames and `reclaim_batch!` is what normally empties
    # that list, which does not happen between two presents. Re-submitting a
    # sealed segment executes it twice — harmless for a clear, wrong for anything
    # that accumulates. They cannot go straight back to `free_cmd_bufs` either;
    # the GPU is still reading them until the fence. `reclaim_batch!` drains this.
    #
    # A buffer a `Recording` lent this batch is dropped rather than moved: the
    # recording owns it and submits it again. A windowed plan cannot be recorded
    # today, so `borrowed` is empty on every path that reaches here.
    for cb in batch.sealed_cmd_bufs
        any(x -> x === cb, batch.borrowed) && continue
        push!(batch.submitted_cmd_bufs, cb)
    end
    empty!(batch.sealed_cmd_bufs)
    empty!(batch.borrowed)
    submit_info = VK.SubmitInfo2(wait_infos, cb_infos, signal_infos)
    queue_submit_2!(bq, [submit_info]; fence=win.in_flight[fi])

    # Store batch in window's per-frame slot — it will be reclaimed in
    # acquire_next_image! after the fence wait confirms GPU completion.
    # Do NOT push to free_batches here: the GPU is still using this command buffer.
    # In the one place that means "on its way to the device". A present submits
    # without going through `submit!` — the batch goes into the window's frame
    # slot rather than `bq.in_flight` — so nothing recorded it, and `flush!`
    # computed a target that excluded the frame currently being drawn. See
    # `graph/submission.jl`.
    submitted!(bq, batch.signal_value; tag = :present)
    bq.active_batch = nothing
    win.frame_batches[fi] = batch
    empty!(batch.wait_semaphores)

    drain_deferred_frees!(bq)
    drain_deferred_as_frees!(bq)

    # Present
    present!(win)
end

# ── Mantle's shader builtins, on this backend ────────────────────────────────
#
# The counterpart to the Metal backend's block, and the reason both exist: a
# shader names `Mantle.vertex_index()` and the compiler that is running decides
# what that means. `@lava_device_override` is Lava's wrapper around the same
# method-table overlay `@device_override` is on the other side.
#
# Generated from `Mantle.SHADER_BUILTINS`, so a builtin added to that list and
# forgotten here is a `MethodError` naming it rather than a shader that reads
# the wrong thing.
#
# `Lava.$f`, QUALIFIED, and that is not tidiness. Written bare, the right-hand
# side resolves through this module's bindings: for the builtins in the
# `using Lava:` list at the top of `vulkan.jl` that happens to be Lava's, and for
# `frag_coord` — which is not in that list — it is MANTLE's, the very function
# being overridden. `Mantle.frag_coord(dim) = Mantle.frag_coord(dim)`: infinite
# recursion in any fragment shader that reads its own position, and nothing says
# so until such a shader is compiled.
#
# Found by `test_lava_import_completeness.jl`, which is exactly the question it
# asks — "a Lava name this backend uses and did not import". Qualifying is the
# fix rather than extending the import list, because `frag_coord` is exported by
# both packages: importing it would make the bare name ambiguous and break the
# other uses in this file, whereas the qualified form cannot be read two ways.
for f in Mantle.SHADER_BUILTINS
    f === :frag_coord && continue
    @eval @lava_device_override Mantle.$f() = Lava.$f()
end
@lava_device_override Mantle.frag_coord(dim::Integer = 1) = Lava.frag_coord(dim)
