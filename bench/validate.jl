# Synchronization validation over the Mantle window path.
#
# Run deliberately, not from the test suite: sync validation costs ~212 ms/frame
# against ~0.16 ms without, so this is seconds per frame, not a routine check.
#
#   include("bench/validate.jl"); syncdevice!(); validate()
#
# Mantle emits the swapchain barrier itself (vk_begin_pass! is called with
# transition=false), so a wrong stage, access mask or layout surfaces here rather
# than as intermittent tearing that any full sync hides.

import Mantle
using Lava, GeometryBasics, LinearAlgebra
const M = Mantle
include(joinpath(@__DIR__, "two_scatters.jl"))
# At top level, not inside `validate_targets`: a kernel defined by a runtime
# `include` is invisible to `record!`, which was compiled in an earlier world.
include(joinpath(@__DIR__, "targets.jl"))

"""
    syncdevice!()

Rebuild the default device with sync validation on. Call it before anything here.

Validation is a property of the `VkInstance`, so it is chosen when the device is
built and cannot be switched on afterwards — which is also why the seven `LAVA_*`
environment variables this file used to ask for are gone. **Every `LavaArray`
alive becomes invalid**, so this goes first, before any of the builders below.
"""
syncdevice!() = vk_reset_device!(debug = DebugConfig(sync_val = true))

"""
Refuse to report a clean run out of a device that is not instrumented.

The failure this exists for is silent in exactly the wrong direction: a device
built without the layer produces no messages, which reads the same as a device
that produced none.
"""
function requiresync()
    ctx = Lava.vk_context()
    ctx.debug.sync_val || error(
        "this device has no synchronization validation. It is fixed at device " *
        "creation, so call `syncdevice!()` first — and note that invalidates " *
        "every LavaArray that already exists.")
    nothing
end

function validate(frames = 60; na = 20_000, nb = 10_000)
    requiresync()
    Lava.clear_validation_messages!()
    dev = M.Device(Lava)
    win = RenderWindow(W, H; title = "validate", vsync = false)
    s = build(dev, win, na, nb)

    for k in 1:frames
        Lava.GLFW.PollEvents()
        s.mvp[] = camera(0.01f0 * k)
        M.run!(s.plan)
    end
    Lava.flush!(dev.bq, dev.ctx.device)
    during = copy(Lava.get_validation_messages())

    img = readback_window(win)
    close(win)
    after = Lava.get_validation_messages()

    bg = img[1, 1]
    (frames = frames,
     during_render = length(during),
     after_readback = length(after),
     lit_fraction = round(count(p -> p != bg, img) / length(img), digits = 3),
     messages = unique(after))
end

"""
The transient-target path under sync validation.

Two placed render targets that share bytes, each copied out and then combined.
This is where a missing barrier is most likely: the second target's first write
lands on memory the first was still being copied from, and per-resource tracking
cannot see that the two are the same memory.
"""
function validate_targets(frames = 20; n = 50_000)
    requiresync()
    Lava.clear_validation_messages!()
    dev = M.Device(Lava)
    win = RenderWindow(W, H; title = "validate targets", vsync = false)
    s = build_targets(dev, n)
    for k in 1:frames
        Lava.GLFW.PollEvents()
        s.mvp[] = camera(0.01f0 * k)
        acquire_next_image!(win)
        M.run!(s.plan)
        blit!(dev.bq, WindowTarget(win), M.storage(s.out))
        present_frame!(dev.bq, win)
        Lava.flush!(dev.bq, dev.ctx.device)
    end
    msgs = Lava.get_validation_messages()
    close(win)
    (frames = frames,
     image_arena = only(b.bytes for b in s.plan.slabs if b.bytes < 8_000_000),
     messages = length(msgs), detail = unique(msgs))
end

"""
The update path under sync validation.

This is where a missing barrier would be: the update pass writes a buffer the
render pass reads as an attribute, so `CopyDst -> Vertices` has to be derived and
emitted. Both routes are exercised — a rename writes a fresh store and a partial
write goes in place.

"Attribute" and not "vertex attribute": this backend binds no vertex buffers, so
the read is a storage load in the vertex shader and `Vertices` lowers to that.
"""
function validate_updates(frames = 20; n = 20_000)
    requiresync()
    Lava.clear_validation_messages!()
    dev = M.Device(Lava)
    win = RenderWindow(W, H; title = "validate updates", vsync = false)

    pts = cloud(n)
    sc = Scatter(M.Buffer(dev, pts), M.Buffer(dev, tint.(pts)), M.Scalar(dev, 2f0))
    mvp = Ref(camera(0f0))
    g = M.Graph(dev)
    screen = M.Surface(g, win)
    whole = M.Update(g, sc.positions)                    # rename route
    part  = M.Update(g, sc.color; range = 1:100)         # in-place route
    M.render!(g, "plot", screen => M.Clear((0.02f0, 0.02f0, 0.04f0, 1f0))) do p
        M.draw!(p, SCATTER, bind(p, sc, mvp), sc.positions)
    end
    plan = M.Plan(g)

    tinted = [Vec4f(0.1, 0.02, 0.02, 1) for _ in 1:100]
    ondevice = M.Buffer(dev, cloud(n))                   # a device-side source
    for k in 1:frames
        Lava.GLFW.PollEvents()
        mvp[] = camera(0.01f0 * k)
        # All three routes: whole-buffer from the host, partial from the host,
        # and whole-buffer from a device array, which copies device to device and
        # never stages.
        r = mod1(k, 3)
        r == 1 ? whole(cloud(n)) :
        r == 2 ? part(tinted) :
                 whole(M.storage(ondevice))
        M.run!(plan)
        Lava.flush!(dev.bq, dev.ctx.device)
    end
    msgs = Lava.get_validation_messages()
    close(win)
    (frames = frames, messages = length(msgs), detail = unique(msgs))
end

"""
The depth attachment under sync validation.

The depth target is a transient, so it gets its layout from a barrier Mantle
derived from the `Depth` usage rather than from anything Lava emitted — the pass
is begun with `transition = false` for both attachments. A wrong aspect mask,
stage or layout is a validation message here and nothing visible on screen, since
RADV renders a wrong-but-tolerated barrier exactly like a right one.
"""
function validate_depth(frames = 20; n = 20_000)
    requiresync()
    Lava.clear_validation_messages!()
    dev = M.Device(Lava)
    win = RenderWindow(W, H; title = "validate depth", vsync = false)

    depth_frag(inputs) = inputs.color
    zpipe = Rasterizer(vertex = scatter_vertex, fragment = depth_frag,
                       varyings = (color = Vec4f,), topology = PointList(),
                       blend = Opaque(), cull = NoCull(), depth = DepthLess())

    pts = cloud(n)
    sc = Scatter(M.Buffer(dev, pts), M.Buffer(dev, tint.(pts)), M.Scalar(dev, 4f0))
    mvp = Ref(camera(0f0))
    g = M.Graph(dev)
    screen = M.Surface(g, win)
    z = M.Transient.Image(g, Float32, (W, H))
    M.render!(g, "plot", screen => M.Clear((0.02f0, 0.02f0, 0.04f0, 1f0)),
              z => M.Clear(1f0)) do p
        M.draw!(p, zpipe, bind(p, sc, mvp), sc.positions)
    end
    plan = M.Plan(g)

    for k in 1:frames
        Lava.GLFW.PollEvents()
        mvp[] = camera(0.01f0 * k)
        M.run!(plan)
        Lava.flush!(dev.bq, dev.ctx.device)
    end
    msgs = Lava.get_validation_messages()
    close(win)
    (frames = frames, messages = length(msgs), detail = unique(msgs))
end

"""
Several colour attachments under sync validation.

Each attachment is a separate image with its own derived barrier, and the pass
begins with `transition = false`, so a missing or wrong one for the second target
is a validation message and nothing else — the first target would still look
right on screen.
"""
function validate_mrt(frames = 20; n = 20_000)
    requiresync()
    Lava.clear_validation_messages!()
    dev = M.Device(Lava)
    win = RenderWindow(W, H; title = "validate mrt", vsync = false)

    two_frag(inputs) = (inputs.color, Vec4f(0, 1, 0, 1))
    mrt = Rasterizer(vertex = scatter_vertex, fragment = two_frag,
                     varyings = (color = Vec4f,), topology = PointList(),
                     blend = Opaque(), cull = NoCull(), depth = DepthLess())

    pts = cloud(n)
    sc = Scatter(M.Buffer(dev, pts), M.Buffer(dev, tint.(pts)), M.Scalar(dev, 4f0))
    mvp = Ref(camera(0f0))
    g = M.Graph(dev)
    screen = M.Surface(g, win)
    second = M.Transient.Image(g, BGRA{N0f8}, (W, H))
    z = M.Transient.Image(g, Float32, (W, H))
    raw = M.Transient.Buffer(g, UInt8, W * H * 4)
    M.render!(g, "gbuffer", screen => M.Clear((0.02f0, 0.02f0, 0.04f0, 1f0)),
              second => M.Clear((0f0, 0f0, 0f0, 1f0)), z => M.Clear(1f0)) do p
        M.draw!(p, mrt, bind(p, sc, mvp), sc.positions)
    end
    M.copy!(g, "read second", raw, second)
    plan = M.Plan(g)

    for k in 1:frames
        Lava.GLFW.PollEvents()
        mvp[] = camera(0.01f0 * k)
        M.run!(plan)
        Lava.flush!(dev.bq, dev.ctx.device)
    end
    msgs = Lava.get_validation_messages()
    close(win)
    (frames = frames, messages = length(msgs), detail = unique(msgs))
end

"""The detector has to be known to fire, or a clean run means nothing."""
function selftest()
    Lava.clear_validation_messages!()
    dev = M.Device(Lava)
    try
        Lava.Vulkan.Buffer(dev.ctx.device, 0, Lava.Vulkan.BUFFER_USAGE_STORAGE_BUFFER_BIT,
                           Lava.Vulkan.SHARING_MODE_EXCLUSIVE, UInt32[])
    catch e
        e isa Lava.Vulkan.VulkanError || rethrow()
    end
    sleep(0.2)
    n = length(Lava.get_validation_messages())
    n > 0 || error("validation layer is not reporting; a clean result would be meaningless")
    n
end


"""
    validate_matrix()

Emit every barrier the lowering can produce into a real command buffer and let
the validator check it: 42x42 memory barriers, plus image barriers over the
colour and depth usages against images created with every relevant usage flag.

This is §4's promised barrier matrix. Checking it against a table written by the
same hand that wrote the lowering would prove nothing; checking it against the
validator finds stage/access pairs the spec forbids.

Each image pair is primed into its `from` layout first, because `oldLayout` must
be UNDEFINED or the image's actual current layout, not whatever we claim.
"""
function validate_matrix()
    requiresync()
    be = M.Vulkan()
    dev = M.Device(Lava)
    bq = dev.bq
    acc = (M.ReadOnly, M.WriteOnly, M.ReadWrite)
    all_usages = Type[M.Vertices, M.Indices, M.Indirect, M.Uniform, M.Sampled, M.Present,
                      M.CopySrc, M.CopyDst, M.TraceRead, M.TraceBuild,
                      M.ColorAttachment{true}, M.ColorAttachment{false},
                      (M.Storage{K,A} for K in (M.BufferKind, M.ImageKind) for A in acc)...,
                      (M.Depth{D,S,X} for D in acc for S in (acc..., M.NoAccess)
                                      for X in (true, false))...]

    Lava.clear_validation_messages!()
    b = Lava.ensure_active_batch!(bq)
    nmem = 0
    for from in all_usages, to in all_usages
        Lava.Vulkan._cmd_pipeline_barrier_2(b.cmd_buf,
            Lava.Vulkan._DependencyInfo([Lava.Vulkan._MemoryBarrier2(;
                src_stage_mask = M.stages(be, from, M.Src()),
                src_access_mask = M.access(be, from, M.Src()),
                dst_stage_mask = M.stages(be, to, M.Dst()),
                dst_access_mask = M.access(be, to, M.Dst()))], [], []))
        nmem += 1
    end
    Lava.flush!(bq, dev.ctx.device)
    nimg = sweep_images(dev, bq, be)
    Lava.flush!(bq, dev.ctx.device)
    (memory_barriers = nmem, image_barriers = nimg,
     messages = length(Lava.get_validation_messages()),
     detail = unique(Lava.get_validation_messages()))
end

"""An image with every usage flag the layouts in question require."""
function scratch_image(dev, fmt, usage)
    d = dev.ctx.device
    img = Lava.Vulkan.Image(d, Lava.Vulkan.IMAGE_TYPE_2D, fmt,
        Lava.Vulkan.Extent3D(64, 64, 1), UInt32(1), UInt32(1),
        Lava.Vulkan.SAMPLE_COUNT_1_BIT, Lava.Vulkan.IMAGE_TILING_OPTIMAL, usage,
        Lava.Vulkan.SHARING_MODE_EXCLUSIVE, UInt32[], Lava.Vulkan.IMAGE_LAYOUT_UNDEFINED)
    req = Lava.Vulkan.get_image_memory_requirements(d, img)
    mp = Lava.Vulkan.get_physical_device_memory_properties(dev.ctx.physical_device)
    idx = findfirst(eachindex(mp.memory_types)) do i
        (req.memory_type_bits >> (i - 1)) & 1 == 1 &&
        (mp.memory_types[i].property_flags & Lava.Vulkan.MEMORY_PROPERTY_DEVICE_LOCAL_BIT) !=
            Lava.Vulkan.MemoryPropertyFlag(0)
    end
    mem = Lava.Vulkan.DeviceMemory(d, req.size, UInt32(idx - 1))
    Lava.Vulkan.bind_image_memory(d, img, mem, 0)
    (img, mem)
end

"""
Every layout transition the lowering can produce, on images that support them.

Each pair is primed into its `from` layout first: `oldLayout` must be UNDEFINED
or the image's actual current layout, so emitting arbitrary pairs back to back
trips VUID-VkImageMemoryBarrier2-oldLayout-01197 for reasons that have nothing to
do with the table under test.
"""
function sweep_images(dev, bq, be)
    acc = (M.ReadOnly, M.WriteOnly, M.ReadWrite)
    coloru = Type[M.Sampled, M.CopySrc, M.CopyDst, M.ColorAttachment{true}, M.ColorAttachment{false},
                  (M.Storage{M.ImageKind,A} for A in acc)...]
    depthu = Type[(M.Depth{D,S,X} for D in acc for S in (acc..., M.NoAccess)
                                  for X in (true, false))...]

    color, _ = scratch_image(dev, Lava.Vulkan.FORMAT_R8G8B8A8_UNORM,
        Lava.Vulkan.IMAGE_USAGE_COLOR_ATTACHMENT_BIT | Lava.Vulkan.IMAGE_USAGE_SAMPLED_BIT |
        Lava.Vulkan.IMAGE_USAGE_STORAGE_BIT | Lava.Vulkan.IMAGE_USAGE_TRANSFER_SRC_BIT |
        Lava.Vulkan.IMAGE_USAGE_TRANSFER_DST_BIT)
    depth, _ = scratch_image(dev, Lava.Vulkan.FORMAT_D32_SFLOAT_S8_UINT,
        Lava.Vulkan.IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | Lava.Vulkan.IMAGE_USAGE_SAMPLED_BIT |
        Lava.Vulkan.IMAGE_USAGE_TRANSFER_SRC_BIT | Lava.Vulkan.IMAGE_USAGE_TRANSFER_DST_BIT)

    n = 0
    for (img, usages, aspect) in ((color, coloru, Lava.Vulkan.IMAGE_ASPECT_COLOR_BIT),
                                  (depth, depthu, Lava.Vulkan.IMAGE_ASPECT_DEPTH_BIT |
                                                  Lava.Vulkan.IMAGE_ASPECT_STENCIL_BIT))
        b = Lava.ensure_active_batch!(bq)
        rng = Lava.Vulkan._ImageSubresourceRange(aspect, UInt32(0), UInt32(1), UInt32(0), UInt32(1))
        anystage = Lava.Vulkan.PipelineStageFlag2(Lava.Vulkan.PIPELINE_STAGE_2_ALL_COMMANDS_BIT)
        noaccess = Lava.Vulkan.AccessFlag2(0)
        for from in usages, to in usages
            oldl = M.layout(be, from, M.Src())
            newl = M.layout(be, to, M.Dst())
            newl == Lava.Vulkan.IMAGE_LAYOUT_UNDEFINED && continue
            if oldl != Lava.Vulkan.IMAGE_LAYOUT_UNDEFINED
                Lava.Vulkan._cmd_pipeline_barrier_2(b.cmd_buf,
                    Lava.Vulkan._DependencyInfo([], [], [Lava.Vulkan._ImageMemoryBarrier2(
                        Lava.Vulkan.IMAGE_LAYOUT_UNDEFINED, oldl,
                        Lava.Vulkan.QUEUE_FAMILY_IGNORED, Lava.Vulkan.QUEUE_FAMILY_IGNORED,
                        img, rng; src_stage_mask = anystage, src_access_mask = noaccess,
                        dst_stage_mask = anystage, dst_access_mask = noaccess)]))
            end
            Lava.Vulkan._cmd_pipeline_barrier_2(b.cmd_buf,
                Lava.Vulkan._DependencyInfo([], [], [Lava.Vulkan._ImageMemoryBarrier2(
                    oldl, newl, Lava.Vulkan.QUEUE_FAMILY_IGNORED,
                    Lava.Vulkan.QUEUE_FAMILY_IGNORED, img, rng;
                    src_stage_mask = M.stages(be, from, M.Src()),
                    src_access_mask = M.access(be, from, M.Src()),
                    dst_stage_mask = M.stages(be, to, M.Dst()),
                    dst_access_mask = M.access(be, to, M.Dst()))]))
            n += 1
        end
    end
    n
end
