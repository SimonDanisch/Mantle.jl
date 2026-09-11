"""
Mantle's portable graphics API, on Metal.

`supports_graphics(::MetalBackend)` answered `false` for one reason — Metal.jl
compiled Julia to compute kernels and nothing else, so there was no vertex or
fragment program for a pipeline to be made of. Metal.jl compiles both now, and
this is the rest: the concrete resources behind Mantle's abstract `Framebuffer`,
the translation of the portable pipeline state, and `draw!`.

What is asserted is that a caller writes the SAME thing it writes on the other
backend — a `GraphicsPipeline` of two Julia functions plus state, pointed at an
`OffscreenTarget` — and gets the right pixels.
"""

using Test, Mantle, Metal
using Metal: vertex_index
# The composite format's colorant, for the readback assertions.
using ColorTypes: RGBA, BGRA
using ColorTypes.FixedPointNumbers: N0f8
const MTLg = Metal.MTL

# The PORTABLE spelling, the one `bench/showcase.jl` uses: a vertex stage
# RETURNS `(position = …, varyings…)` and a fragment stage takes the varyings as
# its first argument and returns its colour. Neither shape is what AIR wants —
# a stage writes through a trailing pointer and reads each varying as its own
# tagged parameter — and `MetalVertexStage`/`MetalFragmentStage` are that
# translation, the counterpart to Lava's `VertexWrapper`/`FragmentWrapper`.
mantle_gfx_vertex(verts::Core.LLVMPtr{NTuple{4,Float32},1}) =
    (position = unsafe_load(verts, Int(vertex_index())), tint = (0f0, 1f0, 0f0, 1f0))
# The fragment reads the varying rather than repeating the constant, so a
# vertex-to-fragment link that silently failed would show as a black triangle.
mantle_gfx_fragment(inputs) = inputs.tint

@testset "Metal answers the two capability questions separately" begin
    be = Metal.MetalBackend()
    # It rasterises now…
    @test Mantle.supports_graphics(be)
    # …but has no command pool or fence to build a `BatchQueue` from, and
    # `allocate_batch_queue!` says so rather than handing back a stub that
    # records nothing. A caller that needs to record INTO a batch queue has to
    # ask the second question; the two came apart when Metal learned to draw.
    @test !Mantle.supports_batch_queue(be)
    @test_throws ArgumentError Mantle.allocate_batch_queue!(be)
end

@testset "a portable GraphicsPipeline draws on Metal" begin
    be = Metal.MetalBackend()
    fb = Mantle.Framebuffer(be, 64, 64; depth = false)
    @test fb isa Mantle.Framebuffer
    @test (fb.width, fb.height) == (64, 64)

    pipe = Mantle.GraphicsPipeline(; vertex = Mantle.VertexShader(mantle_gfx_vertex; outputs = (tint = NTuple{4,Float32},)),
                                     fragment = Mantle.FragmentShader(mantle_gfx_fragment),
                                     cull = Mantle.NoCull())

    verts = NTuple{4,Float32}[(-0.9f0, -0.9f0, 0f0, 1f0), (0.9f0, -0.9f0, 0f0, 1f0),
                              (0f0, 0.9f0, 0f0, 1f0)]
    vbuf = MTLg.MTLBuffer(Metal.device(), sizeof(verts), pointer(verts);
                          storage = Metal.SharedStorage)

    Mantle.draw!(be, pipe, Mantle.OffscreenTarget(fb), 3; buffers = (vbuf,),
                 vert_tt = Tuple{Core.LLVMPtr{NTuple{4,Float32},1}})

    # `width` x `height`, one 4-tuple per pixel, which is the SHAPE
    # `readback_framebuffer` promises and what the Vulkan side answers. It used
    # to be a `4 x width x height` byte array here and only here, so the one
    # portable caller — RayMakie's compositor — worked on Vulkan and hit a
    # `BoundsError` on Metal.
    px = Mantle.readback_framebuffer(fb)
    @test px isa Matrix{NTuple{4,UInt8}}
    @test size(px) == (64, 64)
    green = count(q -> q[2] > 0x80, px)
    # About 40 % of the frame, which is this triangle's area in NDC. A blank or
    # a fully-covered target both fail here.
    @test 1000 < green < 3000
    # …and the pixels outside it kept the clear colour, so the draw was bounded
    # by the geometry rather than covering everything.
    @test count(q -> q[2] <= 0x80, px) > 1000

    # Compiling the same pipeline again must hit the cache: a pipeline rebuilt
    # per draw is the difference between a frame and a stall.
    second = Mantle.Framebuffer(be, 64, 64; depth = false)
    Mantle.draw!(be, pipe, Mantle.OffscreenTarget(second), 3; buffers = (vbuf,),
                 vert_tt = Tuple{Core.LLVMPtr{NTuple{4,Float32},1}})
    @test Mantle.readback_framebuffer(second) == px
end

@testset "the portable pipeline state reaches Metal" begin
    # One method per state value rather than a lookup table, so an unhandled one
    # is a missing method at compile time instead of a `KeyError` mid-frame.
    @test Mantle.mtl_cull(Mantle.NoCull())    == MTLg.MTLCullModeNone
    @test Mantle.mtl_cull(Mantle.CullBack())  == MTLg.MTLCullModeBack
    @test Mantle.mtl_cull(Mantle.CullFront()) == MTLg.MTLCullModeFront
    @test Mantle.mtl_primitive(Mantle.TriangleList()) == MTLg.MTLPrimitiveTypeTriangle
    @test Mantle.mtl_primitive(Mantle.LineList())     == MTLg.MTLPrimitiveTypeLine
    @test Mantle.mtl_depth_compare(Mantle.DepthLess()) == MTLg.MTLCompareFunctionLess
    @test !Mantle.depth_writes(Mantle.DepthOff())
    @test Mantle.depth_writes(Mantle.DepthLess())

    # Metal has no geometry stage at all — Apple's replacement is the mesh
    # pipeline — so asking for one has to fail loudly rather than silently drop
    # the stage and render something plausible.
    @test_throws ErrorException Mantle.compile_pipeline(
        Mantle.GraphicsPipeline(;
            vertex = Mantle.VertexShader(mantle_gfx_vertex),
            fragment = Mantle.FragmentShader(mantle_gfx_fragment),
            geometry = Mantle.GeometryShader(mantle_gfx_vertex; max_vertices = 3)),
        MTLg.MTLPixelFormat[MTLg.MTLPixelFormatBGRA8Unorm], nothing, Tuple{}, Tuple{})
end

@testset "2.7: a capability is asked, not discovered from a compile error" begin
    # `GraphicsPipeline` has a `geometry` field and a tess pair, so a caller can
    # build one this device cannot run. Before this the only way to find out was
    # to compile it and read the message — the wrong place and the wrong time,
    # and the reason RayMakie's geometry-emitting scatter path had no way to
    # choose a different one.
    be = Metal.MetalBackend()
    @test Mantle.supports_graphics(be)
    @test !Mantle.supports_geometry_stage(be)
    @test !Mantle.supports_tessellation(be)
    # The default is `false`, so a backend that answers nothing answers no —
    # which is the safe direction for a capability.
    @test !Mantle.supports_geometry_stage(:something_that_is_not_a_backend)
end

# ── KernelInterface's mesh vocabulary on Metal ───────────────────────────────
#
# A mesh body written ONLY in the portable verbs — no `Metal.set_*_mesh`
# anywhere — and drawn. What this pins beyond "it compiles" is the two
# conversions the Metal overrides exist to do, both of which fail silently:
#
#   * SLOTS AND INDICES COUNT FROM ONE in KernelInterface and from zero in AIR.
#     Off by one gives a degenerate or misplaced primitive, never an error.
#   * A VERTEX IS A NAMEDTUPLE and AIR wants the position through one intrinsic
#     and every other field through a NUMBERED one. A wrong number reads a
#     different field in the fragment stage than the mesh stage wrote, which is
#     why the assertion below is the per-primitive colour and not just coverage.

const KIV = @NamedTuple{position::NTuple{4,Float32}, uv::NTuple{2,Float32}}
const KIP = @NamedTuple{colour::NTuple{4,Float32}}
const KIObj = Metal.MeshPtr{KIV, KIP, 4, 2, :triangle}
const KIm = parentmodule(Mantle.MeshEmitter)

function ki_mesh(out::KIObj)
    KIm.set_mesh_vertex!(out, 1, (position = (-1f0, -1f0, 0f0, 1f0), uv = (0f0, 0f0)))
    KIm.set_mesh_vertex!(out, 2, (position = ( 3f0, -1f0, 0f0, 1f0), uv = (2f0, 0f0)))
    KIm.set_mesh_vertex!(out, 3, (position = (-1f0,  3f0, 0f0, 1f0), uv = (0f0, 2f0)))
    KIm.set_mesh_triangle!(out, 1, 1, 2, 3)
    KIm.set_mesh_primitive_data!(out, 1, (colour = (0f0, 1f0, 0f0, 1f0),))
    KIm.set_mesh_outputs!(out, 3, 1)
    return nothing
end

# The two builtins, and the once-per-threadgroup rule `set_mesh_outputs!`
# carries: called from every thread it races, and the count the driver reads is
# whichever invocation wrote last.
function ki_mesh_grouped(out::KIObj)
    if KIm.mesh_thread_index() == Int32(1)
        s = 1f0 / Float32(KIm.mesh_group_index())
        KIm.set_mesh_vertex!(out, 1, (position = (-s, -s, 0f0, 1f0), uv = (0f0, 0f0)))
        KIm.set_mesh_vertex!(out, 2, (position = (3f0 * s, -s, 0f0, 1f0), uv = (2f0, 0f0)))
        KIm.set_mesh_vertex!(out, 3, (position = (-s, 3f0 * s, 0f0, 1f0), uv = (0f0, 2f0)))
        KIm.set_mesh_triangle!(out, 1, 1, 2, 3)
        KIm.set_mesh_primitive_data!(out, 1, (colour = (0f0, 1f0, 0f0, 1f0),))
        KIm.set_mesh_outputs!(out, 3, 1)
    end
    return nothing
end

struct KIFragOut
    colour::NTuple{4,Float32}
end

# Reads BOTH planes, so a per-primitive write that landed in the wrong numbered
# slot shows up as the wrong colour rather than as nothing at all.
function ki_frag(uv::Metal.Varying{:uv, NTuple{2,Float32}},
                 col::Metal.Varying{:colour, NTuple{4,Float32}},
                 out::Core.LLVMPtr{KIFragOut,1})
    c = col.value
    u = uv.value
    Base.unsafe_store!(out, KIFragOut((u[1] * 0f0, c[2], u[2] * 0f0, 1f0)))
    return nothing
end

"""Draw `meshfn` over `groups` x `threads` into a 32x32 target, read it back."""
function ki_draw_mesh(meshfn, fragfn; groups::Int = 1, threads::Int = 1)
    MTL = Metal.MTL
    dev = Metal.device()
    W = H = 32

    pd = MTL.MTLMeshRenderPipelineDescriptor()
    pd.meshFunction = meshfn
    pd.fragmentFunction = fragfn
    pd.colorAttachments[1].pixelFormat = MTL.MTLPixelFormatRGBA8Unorm
    pd.maxTotalThreadsPerMeshThreadgroup = threads
    pipe = MTL.MTLRenderPipelineState(dev, pd)

    td = MTL.MTLTextureDescriptor(MTL.MTLPixelFormatRGBA8Unorm, W, H, false)
    td.usage = MTL.MTLTextureUsageRenderTarget
    td.storageMode = MTL.MTLStorageModeShared
    tex = MTL.MTLTexture(dev, td)

    rp = MTL.MTLRenderPassDescriptor()
    ca = rp.colorAttachments[1]
    ca.texture = tex
    ca.loadAction  = MTL.MTLLoadActionClear
    ca.storeAction = MTL.MTLStoreActionStore
    ca.clearColor  = MTL.MTLClearColor(0.0, 0.0, 0.0, 1.0)

    cb  = MTL.MTLCommandBuffer(MTL.MTLCommandQueue(dev))
    enc = MTL.MTLRenderCommandEncoder(cb, rp)
    MTL.set_pipeline!(enc, pipe)
    MTL.set_viewport!(enc, MTL.MTLViewport(0, 0, W, H, 0, 1))
    MTL.draw_mesh_threadgroups!(enc, MTL.MTLSize(groups, 1, 1),
                                     MTL.MTLSize(1, 1, 1), MTL.MTLSize(threads, 1, 1))
    MTL.endEncoding!(enc)
    MTL.commit!(cb)
    MTL.wait_completed(cb)

    px = Matrix{NTuple{4,UInt8}}(undef, W, H)
    GC.@preserve px MTL.getBytes!(pointer(px), tex, W * 4,
                                  MTL.MTLRegion(MTL.MTLOrigin(0,0,0), MTL.MTLSize(W,H,1)))
    return px
end

"""Compile a mesh or fragment stage and keep its library reachable."""
const KI_LIBS = Any[]
function ki_stage(f, tt, stage::Symbol, name::String)
    dev = Metal.device()
    job = Metal.GPUCompiler.CompilerJob(Metal.methodinstance(typeof(f), tt),
                                        Metal.compiler_config(dev; stage, name))
    lib = Metal.MTL.MTLLibraryFromData(dev, Metal.compile_to_metallib(job).metallib)
    push!(KI_LIBS, lib)
    return Metal.MTL.MTLFunction(lib, name)
end

@testset "the portable mesh vocabulary compiles and draws on Metal" begin
    ms = ki_stage(ki_mesh, Tuple{KIObj}, :mesh, "ki_mesh")
    fs = ki_stage(ki_frag, Tuple{Metal.Varying{:uv, NTuple{2,Float32}},
                                 Metal.Varying{:colour, NTuple{4,Float32}},
                                 Core.LLVMPtr{KIFragOut,1}}, :fragment, "ki_frag")
    @test ms.functionType == Metal.MTL.MTLFunctionTypeMesh

    px = ki_draw_mesh(ms, fs)
    # A fullscreen triangle, and every pixel carries the per-primitive colour
    # the body declared — (0, 1, 0, 1) — which is what says the numbered field
    # slot and the one-based conversions are both right.
    @test all(==((0x00, 0xff, 0x00, 0xff)), px)
end

@testset "mesh_thread_index and mesh_group_index count from one" begin
    ms = ki_stage(ki_mesh_grouped, Tuple{KIObj}, :mesh, "ki_mesh_grouped")
    fs = ki_stage(ki_frag, Tuple{Metal.Varying{:uv, NTuple{2,Float32}},
                                 Metal.Varying{:colour, NTuple{4,Float32}},
                                 Core.LLVMPtr{KIFragOut,1}}, :fragment, "ki_frag")

    lit(px) = count(q -> q[2] > 0x80, px)
    # Group 1 scales by 1/1 and covers the frame. If `mesh_group_index` were
    # zero-based this would divide by zero and cover nothing.
    @test lit(ki_draw_mesh(ms, fs; groups = 1)) == 32 * 32
    # Group 2 draws a quarter-size triangle INSIDE group 1's, so two groups
    # still cover the frame — and the assertion that matters is that adding a
    # group does not lose the first one.
    @test lit(ki_draw_mesh(ms, fs; groups = 2)) == 32 * 32
    # Four threads, one primitive: only thread 1 writes, and `mesh_thread_index`
    # being one-based is what makes that the first invocation rather than none.
    @test lit(ki_draw_mesh(ms, fs; groups = 1, threads = 4)) == 32 * 32
end

# ── the two paths agree about clip space ─────────────────────────────────────

"""Position straight out of a buffer, so the vertex path adds no arithmetic."""
function ki_ref_vertex(verts::Core.LLVMPtr{NTuple{4,Float32},1})
    return (position = unsafe_load(verts, Int(Mantle.vertex_index())),)
end
ki_solid_fragment(_) = (colour = (0f0, 1f0, 0f0, 1f0),)

# The same three positions the vertex path reads, emitted by a mesh stage.
# Deliberately OFF CENTRE in y, so an unmirrored clip space moves the triangle
# to the other half instead of leaving it where it was.
function ki_mesh_halfheight(out::KIObj)
    KIm.set_mesh_vertex!(out, 1, (position = (-0.9f0, -0.9f0, 0f0, 1f0), uv = (0f0, 0f0)))
    KIm.set_mesh_vertex!(out, 2, (position = ( 0.9f0, -0.9f0, 0f0, 1f0), uv = (1f0, 0f0)))
    KIm.set_mesh_vertex!(out, 3, (position = ( 0.0f0,  0.1f0, 0f0, 1f0), uv = (0.5f0, 1f0)))
    KIm.set_mesh_triangle!(out, 1, 1, 2, 3)
    KIm.set_mesh_primitive_data!(out, 1, (colour = (0f0, 1f0, 0f0, 1f0),))
    KIm.set_mesh_outputs!(out, 3, 1)
    return nothing
end

"""Draw a mesh stage into a Mantle `Framebuffer`, so the readback matches `draw!`'s."""
function ki_draw_mesh_into(fb, meshfn, fragfn)
    MTL = Metal.MTL
    dev = Metal.device()
    pd = MTL.MTLMeshRenderPipelineDescriptor()
    pd.meshFunction = meshfn
    pd.fragmentFunction = fragfn
    pd.colorAttachments[1].pixelFormat = fb.color_format
    pd.maxTotalThreadsPerMeshThreadgroup = 1
    pipe = MTL.MTLRenderPipelineState(dev, pd)

    rp = MTL.MTLRenderPassDescriptor()
    ca = rp.colorAttachments[1]
    ca.texture = fb.color
    ca.loadAction  = MTL.MTLLoadActionClear
    ca.storeAction = MTL.MTLStoreActionStore
    ca.clearColor  = MTL.MTLClearColor(0.0, 0.0, 0.0, 1.0)

    cb  = MTL.MTLCommandBuffer(MTL.MTLCommandQueue(dev))
    enc = MTL.MTLRenderCommandEncoder(cb, rp)
    MTL.set_pipeline!(enc, pipe)
    MTL.draw_mesh_threadgroups!(enc, MTL.MTLSize(1, 1, 1),
                                     MTL.MTLSize(1, 1, 1), MTL.MTLSize(1, 1, 1))
    MTL.endEncoding!(enc)
    MTL.commit!(cb)
    MTL.wait_completed(cb)
    return Mantle.readback_framebuffer(fb)
end

@testset "a mesh stage lands in the same clip space as a vertex stage" begin
    # Mantle's clip space is Vulkan's, +y DOWN, and Metal's is the other one.
    # The vertex wrapper mirrors y on the way out; a mesh stage has to do the
    # same, and if it does not the image is upside down with nothing in the
    # pipeline objecting. Comparing the two paths pins it without this test
    # having to know which way up Metal's readback is.
    be = Metal.MetalBackend()
    verts = NTuple{4,Float32}[(-0.9f0, -0.9f0, 0f0, 1f0), (0.9f0, -0.9f0, 0f0, 1f0),
                              (0f0, 0.1f0, 0f0, 1f0)]
    vbuf = Metal.MTL.MTLBuffer(Metal.device(), sizeof(verts), pointer(verts);
                               storage = Metal.SharedStorage)

    vfb = Mantle.Framebuffer(be, 32, 32; depth = false)
    Mantle.draw!(be, Mantle.GraphicsPipeline(;
                     vertex = Mantle.VertexShader(ki_ref_vertex),
                     fragment = Mantle.FragmentShader(ki_solid_fragment),
                     cull = Mantle.NoCull()),
                 Mantle.OffscreenTarget(vfb), 3; buffers = (vbuf,),
                 vert_tt = Tuple{Core.LLVMPtr{NTuple{4,Float32},1}})
    vpx = Mantle.readback_framebuffer(vfb)

    mfb = Mantle.Framebuffer(be, 32, 32; depth = false)
    ms = ki_stage(ki_mesh_halfheight, Tuple{KIObj}, :mesh, "ki_mesh_halfheight")
    fs = ki_stage(ki_frag, Tuple{Metal.Varying{:uv, NTuple{2,Float32}},
                                 Metal.Varying{:colour, NTuple{4,Float32}},
                                 Core.LLVMPtr{KIFragOut,1}}, :fragment, "ki_frag")
    mpx = ki_draw_mesh_into(mfb, ms, fs)

    # The triangle is off centre, so the two halves differ and the comparison
    # has something to catch.
    top = count(q -> q[2] > 0x80, @view vpx[:, 1:16])
    bot = count(q -> q[2] > 0x80, @view vpx[:, 17:32])
    @test top != bot
    @test 100 < top + bot < 32 * 32
    # …and the mesh stage put it in the same place, pixel for pixel.
    @test mpx == vpx
end

# ── a portable MeshPipeline, end to end ──────────────────────────────────────
#
# The testsets above drive the mesh stage through `MTLMeshRenderPipelineDescriptor`
# built by hand. This one goes through Mantle: a `MeshPipeline` of stage structs,
# `draw!`, `readback_framebuffer`. Nothing here names an AIR intrinsic or a Metal
# descriptor, which is the point — the same description has to be all a caller
# writes.

function mp_mesh(out)
    KIm.set_mesh_vertex!(out, 1, (position = (-1f0, -1f0, 0f0, 1f0), uv = (0f0, 0f0)))
    KIm.set_mesh_vertex!(out, 2, (position = ( 3f0, -1f0, 0f0, 1f0), uv = (2f0, 0f0)))
    KIm.set_mesh_vertex!(out, 3, (position = (-1f0,  3f0, 0f0, 1f0), uv = (0f0, 2f0)))
    KIm.set_mesh_triangle!(out, 1, 1, 2, 3)
    KIm.set_mesh_primitive_data!(out, 1, (colour = (0f0, 1f0, 0f0, 1f0),))
    KIm.set_mesh_outputs!(out, 3, 1)
    return nothing
end

# Reads the per-primitive plane and returns it, which is the portable fragment
# spelling — one value per colour attachment.
mp_frag(inputs) = inputs.colour

# Half height in Mantle's clip space, to catch a mesh pipeline that forgot the
# y mirror the vertex path applies.
function mp_mesh_lower(out)
    KIm.set_mesh_vertex!(out, 1, (position = (-1f0, -1f0, 0f0, 1f0), uv = (0f0, 0f0)))
    KIm.set_mesh_vertex!(out, 2, (position = ( 3f0, -1f0, 0f0, 1f0), uv = (2f0, 0f0)))
    KIm.set_mesh_vertex!(out, 3, (position = (-1f0,  0f0, 0f0, 1f0), uv = (0f0, 1f0)))
    KIm.set_mesh_triangle!(out, 1, 1, 2, 3)
    KIm.set_mesh_primitive_data!(out, 1, (colour = (0f0, 1f0, 0f0, 1f0),))
    KIm.set_mesh_outputs!(out, 3, 1)
    return nothing
end

mp_pipeline(f) = Mantle.MeshPipeline(;
    mesh = Mantle.MeshShader(f;
        outputs = (uv = NTuple{2,Float32}, colour = Mantle.Flat{NTuple{4,Float32}}),
        max_vertices = 4, max_primitives = 2,
        topology = Mantle.TriangleStrip(), threads = 1),
    fragment = Mantle.FragmentShader(mp_frag),
    cull = Mantle.NoCull(), depth = Mantle.DepthOff())

@testset "a portable MeshPipeline draws on Metal" begin
    be = Metal.MetalBackend()
    # The capability now answers `true`, and for the backend and the device
    # alike — a caller may hold either.
    @test Mantle.supports_mesh_pipeline(be)

    p = mp_pipeline(mp_mesh)

    # The object's type is DERIVED from the declaration: the `Flat` split gives
    # the two planes, `MeshConfig` the bounds, and a strip topology collapses to
    # the primitive KIND because a mesh stage has no input stream to group.
    Obj = Mantle.mesh_object_of(p)
    @test Obj === Metal.MeshObject{@NamedTuple{position::NTuple{4,Float32},
                                               uv::NTuple{2,Float32}},
                                   @NamedTuple{colour::NTuple{4,Float32}},
                                   4, 2, :triangle}

    fb = Mantle.Framebuffer(be, 32, 32; depth = false)
    Mantle.draw!(be, p, Mantle.OffscreenTarget(fb), 1)
    px = Mantle.readback_framebuffer(fb)
    # Every pixel carries the per-primitive colour the body declared, so both
    # the coverage and the flat plane are asserted at once.
    @test all(==((0x00, 0xff, 0x00, 0xff)), px)

    # Compiling the same pipeline again hits the cache rather than rebuilding.
    fb2 = Mantle.Framebuffer(be, 32, 32; depth = false)
    Mantle.draw!(be, p, Mantle.OffscreenTarget(fb2), 1)
    @test Mantle.readback_framebuffer(fb2) == px
end

@testset "a MeshPipeline uses Mantle's clip space" begin
    # The triangle covers y in [-1, 0], which is one half in Mantle's space
    # (Vulkan's, +y down). Which half it lands in on screen is Metal's business;
    # what this pins is that it is the SAME half a vertex stage would put it in,
    # and that it is not the whole frame.
    be = Metal.MetalBackend()
    fb = Mantle.Framebuffer(be, 32, 32; depth = false)
    Mantle.draw!(be, mp_pipeline(mp_mesh_lower), Mantle.OffscreenTarget(fb), 1)
    px = Mantle.readback_framebuffer(fb)
    lit(v) = count(q -> q[2] > 0x80, v)
    top, bot = lit(@view px[:, 1:16]), lit(@view px[:, 17:32])
    @test top + bot > 200          # it drew something
    @test top + bot < 32 * 32      # …and not everything
    @test (top == 0) != (bot == 0) # …all of it on one side
end

@testset "an object stage is refused rather than ignored" begin
    # Metal.jl emits no `air.object` program yet. A pipeline that declares one
    # has to say so at compile: silently dispatching the mesh stage from the host
    # would run the right shader over the wrong grid.
    p = Mantle.MeshPipeline(;
        mesh = Mantle.MeshShader(mp_mesh;
            outputs = (uv = NTuple{2,Float32}, colour = Mantle.Flat{NTuple{4,Float32}}),
            max_vertices = 4, max_primitives = 2,
            topology = Mantle.TriangleStrip(), threads = 1),
        fragment = Mantle.FragmentShader(mp_frag),
        object = Mantle.ObjectShader(identity; threads = 1))
    be = Metal.MetalBackend()
    fb = Mantle.Framebuffer(be, 8, 8; depth = false)
    @test_throws ErrorException Mantle.draw!(be, p, Mantle.OffscreenTarget(fb), 1)
end

@testset "a readback waits for the pass that filled the target" begin
    # `readback_framebuffer` promises to synchronise, and `getBytes!` is a HOST
    # copy out of the texture — it neither joins the queue nor blocks on it. With
    # no wait, a readback taken straight after a pass reads whatever the texture
    # held, which for a fresh one is zeros.
    #
    # A race, so it hid for a long time: RayMakie's compositor synchronises
    # elsewhere in the frame and the window is narrow. Back to back it is lost
    # every time, and the whole composited image comes back transparent black.
    # No `synchronize` here on purpose — that is the thing being tested.
    be = Metal.MetalBackend()
    fb = Mantle.Framebuffer(be, 8, 8; depth = false,
                            color_format = BGRA{N0f8})
    orange = Metal.MtlVector{RGBA{Float32}}(fill(RGBA{Float32}(1f0, 0.5f0, 0f0, 1f0), 64))
    Mantle.blit!(be, Mantle.OffscreenTarget(fb), orange; clear = true)
    px = Mantle.readback_framebuffer(fb)
    # BGRA on the wire, so the blue channel is first and the red last.
    @test all(==((0x00, 0x80, 0xff, 0xff)), px)
end

# ── a geometry pipeline, lowered onto the mesh stage ─────────────────────────
#
# Metal has no geometry stage, so `compile_draw` lowers a `GraphicsPipeline` that
# declares one with `Mantle.lower_geometry_to_mesh` and compiles the mesh pipeline
# that comes out. Nothing in the CALLER changes: the same description, the same
# `pass!`, the same vertex count — which is the point, and the reason
# `supports_geometry_stage` answering `false` no longer means an overlay is lost.
#
# The lowering's own arithmetic is asserted on the host in `test/test_lowering.jl`,
# over `HostMeshOutput`, where the emitted vertices and indices can be read back.
# What is asserted HERE is that the resulting pixels are right: the quad lands
# where the input point is, the smooth plane interpolates across it, and the flat
# plane carries one value per primitive.

lg_vertex(vid::Mantle.VertexIndex, points) =
    (position = Vec4f(points[vid.value][1], points[vid.value][2], 0f0, 1f0),
     tint = Float32(vid.value))
lg_vertex(points) = lg_vertex(Mantle.VertexIndex(Mantle.vertex_index()), points)

function lg_geometry(gs, prim, points)
    c = prim.position[1]
    t = prim.tint[1]
    for k in Int32(1):Int32(4)
        dx = (k == Int32(1) || k == Int32(2)) ? -0.25f0 : 0.25f0
        dy = (k == Int32(1) || k == Int32(3)) ? -0.25f0 : 0.25f0
        Mantle.emit!(gs, (position = Vec4f(c[1] + dx, c[2] + dy, c[3], c[4]),
                          uv = (dx, dy), tint = (t / 4f0, 0f0, 0f0, 1f0)))
    end
    Mantle.endprimitive!(gs)
    return nothing
end

# The smooth plane in two channels and the flat one in the third, so one frame
# shows both and shows which vertex and which primitive each value came from.
lg_frag_uv(inputs, points) = (abs(inputs.uv[1]) * 4f0, abs(inputs.uv[2]) * 4f0,
                              0.5f0, 1f0)
lg_frag_tint(inputs, points) = inputs.tint

lg_pipeline(frag) = Mantle.GraphicsPipeline(;
    vertex = Mantle.VertexShader(lg_vertex; outputs = (tint = Float32,)),
    geometry = Mantle.GeometryShader(lg_geometry;
                                    outputs = (uv = NTuple{2,Float32},
                                               tint = Mantle.Flat{NTuple{4,Float32}}),
                                    input = Mantle.PointList(),
                                    output = Mantle.TriangleStrip(), max_vertices = 4),
    fragment = Mantle.FragmentShader(frag),
    topology = Mantle.PointList(), cull = Mantle.NoCull(), depth = Mantle.DepthOff())

# The `lines` shape: four input vertices per primitive, out of an index buffer.
lg_seg_vertex(vid::Mantle.VertexIndex, points) =
    (position = Vec4f(points[vid.value][1], points[vid.value][2], 0f0, 1f0),
     tint = Float32(vid.value))

function lg_seg_geometry(gs, prim, points)
    a = prim.position[2]
    b = prim.position[3]
    cx = 0.5f0 * (a[1] + b[1]); cy = 0.5f0 * (a[2] + b[2])
    t = (prim.tint[1] + prim.tint[4]) / 16f0
    for k in Int32(1):Int32(4)
        dx = (k == Int32(1) || k == Int32(2)) ? -0.1f0 : 0.1f0
        dy = (k == Int32(1) || k == Int32(3)) ? -0.1f0 : 0.1f0
        Mantle.emit!(gs, (position = Vec4f(cx + dx, cy + dy, 0f0, 1f0),
                          uv = (dx, dy), tint = (t, 0f0, 0f0, 1f0)))
    end
    Mantle.endprimitive!(gs)
    return nothing
end

lg_seg_pipeline() = Mantle.GraphicsPipeline(;
    vertex = Mantle.VertexShader(lg_seg_vertex; outputs = (tint = Float32,)),
    geometry = Mantle.GeometryShader(lg_seg_geometry;
                                    outputs = (uv = NTuple{2,Float32},
                                               tint = Mantle.Flat{NTuple{4,Float32}}),
                                    input = Mantle.LineStripAdjacency(),
                                    output = Mantle.TriangleStrip(), max_vertices = 4),
    fragment = Mantle.FragmentShader(lg_frag_tint),
    topology = Mantle.LineStripAdjacency(), cull = Mantle.NoCull(),
    depth = Mantle.DepthOff())

"""
Record one draw of `p` through `pass!` — the path RayMakie's overlays take.

`compile_draw` and `record_draw!` and nothing mesh-specific: a caller that has a
geometry pipeline writes exactly this on either backend.
"""
function lg_draw(p, args, count; n = 64, indices = nothing, instances = 1)
    be = Metal.MetalBackend()
    dev = Mantle.todevice(be)
    fb = Mantle.Framebuffer(be, n, n; depth = false)
    target = Mantle.OffscreenTarget(fb)
    compiled = Mantle.compile_draw(dev, p, (Mantle.blittarget(target),), nothing,
                                   args, args)
    Mantle.pass!(dev, target; clear = (0f0, 0f0, 0f0, 1f0)) do pr
        Mantle.viewport!(pr, 0, 0, n, n)
        Mantle.draw!(pr, compiled, args, count; indices, instances)
    end
    return Mantle.readback_framebuffer(fb)
end

"""The pixels that are not the clear colour, and the block they occupy."""
function lg_covered(px)
    idx = findall(q -> q != (0x00, 0x00, 0x00, 0xff), px)
    isempty(idx) && return (n = 0, rows = (0, 0), cols = (0, 0), colours = eltype(px)[])
    r = extrema(getindex.(idx, 1))
    c = extrema(getindex.(idx, 2))
    return (n = length(idx), rows = r, cols = c, colours = sort(unique(px[idx])))
end

@testset "a geometry pipeline draws on a backend with no geometry stage" begin
    be = Metal.MetalBackend()
    # The capability still answers honestly about the STAGE…
    @test !Mantle.supports_geometry_stage(be)
    # …and the one a caller with a geometry body should ask says yes.
    @test Mantle.supports_mesh_pipeline(be)

    # One point at the origin: a quad of 0.5 in NDC is a quarter of each axis, so
    # 16x16 of a 64x64 frame, centred.
    pts = Metal.MtlVector{Vec2f}([Vec2f(0, 0)])
    px = lg_draw(lg_pipeline(lg_frag_uv), (pts,), 1)
    cov = lg_covered(px)
    @test cov.n == 16 * 16
    @test cov.rows == (25, 40) && cov.cols == (25, 40)

    # The SMOOTH plane interpolates: `uv` runs from -0.25 at one edge to +0.25 at
    # the other, so `abs(uv) * 4` sweeps nearly the whole range and is smallest in
    # the middle. This is what the per-vertex plane being addressed per FIELD and
    # per SLOT buys — with the two AIR operands the other way round only the first
    # vertex's `uv` landed and the interpolation collapsed to a corner ramp.
    sub = px[cov.rows[1]:cov.rows[2], cov.cols[1]:cov.cols[2]]
    ch2 = map(q -> q[2], sub)
    ch3 = map(q -> q[3], sub)
    @test maximum(ch2) > 0xd0 && maximum(ch3) > 0xd0
    @test minimum(ch2) < 0x20 && minimum(ch3) < 0x20
    # Symmetric about the centre in both directions, which a one-corner ramp is
    # not: the first row equals the last, and the first column the last.
    @test ch2[1, :] == ch2[end, :]
    @test ch3[:, 1] == ch3[:, end]
    # The blue channel is the fragment's own constant, so every covered pixel has
    # it — a check that the frame is the shader's and not something left over.
    @test all(q -> q[1] == 0x80, sub)
end

@testset "a lowered draw runs one threadgroup per input primitive" begin
    # Three points, three quads, and each carries its own vertex index as a
    # per-primitive value. A lowering that ignored `mesh_group_index` would draw
    # one quad three times over; one that got the flat plane wrong would draw
    # three quads of one colour.
    pts = Metal.MtlVector{Vec2f}([Vec2f(-0.5, -0.5), Vec2f(0.5, 0.5), Vec2f(-0.5, 0.5)])
    px = lg_draw(lg_pipeline(lg_frag_tint), (pts,), 3)
    cov = lg_covered(px)
    @test cov.n == 3 * 16 * 16
    # `tint = vertex index / 4`, so 0.25, 0.5 and 0.75 — and BGRA on the wire puts
    # the red channel third.
    @test cov.colours == [(0x00, 0x00, 0x40, 0xff), (0x00, 0x00, 0x80, 0xff),
                          (0x00, 0x00, 0xbf, 0xff)]
end

@testset "an indexed lowered draw fetches its own vertex indices" begin
    # `LineStripAdjacency` over `0 0 1 2 3 3`: a four-wide window sliding one
    # index at a time, which is how RayMakie's `lines` walks a polyline. A mesh
    # pipeline has no index buffer, so the lowered stage reads it as one of its
    # own arguments — the one part of this that is more than a rearrangement.
    be = Metal.MetalBackend()
    dev = Mantle.todevice(be)
    pts = Metal.MtlVector{Vec2f}([Vec2f(-0.6, -0.6), Vec2f(-0.2, -0.2),
                                  Vec2f(0.2, 0.2), Vec2f(0.6, 0.6)])
    ib = Mantle.indexbuffer(dev, UInt32[0, 0, 1, 2, 3, 3])
    px = lg_draw(lg_seg_pipeline(), (pts,), 6; indices = ib)
    cov = lg_covered(px)
    # Six indices give three segments, each a 0.2-wide quad: 6 or 7 pixels a side
    # on a 64-pixel axis.
    @test 3 * 30 < cov.n < 3 * 50
    # One colour per segment, and they are the OUTER two vertices of each window:
    # (1+3)/16, (1+4)/16, (2+4)/16 one-based.
    @test cov.colours == [(0x00, 0x00, 0x40, 0xff), (0x00, 0x00, 0x50, 0xff),
                          (0x00, 0x00, 0x60, 0xff)]
    # …and the three sit along the diagonal, at the midpoints of the segments.
    centres = [(sum(getindex.(findall(==(c), px), 1)) / count(==(c), px),
                sum(getindex.(findall(==(c), px), 2)) / count(==(c), px))
               for c in cov.colours]
    @test all(abs(c[1] - c[2]) < 1 for c in centres)
    @test issorted(first.(centres))
end

@testset "a lowered draw refuses what it cannot record" begin
    be = Metal.MetalBackend()
    dev = Mantle.todevice(be)
    pts = Metal.MtlVector{Vec2f}([Vec2f(0, 0)])
    # Instancing would put the instance on the mesh grid's second axis, which needs
    # a portable name for `instance_index()` inside a mesh stage. There is none, so
    # this says so rather than drawing one instance and reporting success.
    @test_throws ErrorException lg_draw(lg_pipeline(lg_frag_tint), (pts,), 1;
                                       instances = 2)

    # A vertex body that cannot be handed its index names itself, rather than
    # failing as a `MethodError` from inside a shader compilation.
    noindex(points) = (position = Vec4f(0f0, 0f0, 0f0, 1f0), tint = 0f0)
    p = Mantle.GraphicsPipeline(;
        vertex = Mantle.VertexShader(noindex; outputs = (tint = Float32,)),
        geometry = Mantle.GeometryShader(lg_geometry;
                                        outputs = (uv = NTuple{2,Float32},
                                                   tint = Mantle.Flat{NTuple{4,Float32}}),
                                        input = Mantle.PointList(),
                                        output = Mantle.TriangleStrip(), max_vertices = 4),
        fragment = Mantle.FragmentShader(lg_frag_tint),
        topology = Mantle.PointList(), cull = Mantle.NoCull(), depth = Mantle.DepthOff())
    err = try
        lg_draw(p, (pts,), 1)
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("VertexIndex", err.msg)

    # Fewer vertices than one primitive needs draws nothing, and that is not an
    # error: a `lines` plot with one point assembles no segment. `drawMeshThreadgroups`
    # refuses a zero grid, so the count has to be checked rather than passed on.
    ib = Mantle.indexbuffer(dev, UInt32[0, 0, 1])
    blank = lg_draw(lg_seg_pipeline(), (pts,), 3; indices = ib)
    @test all(==((0x00, 0x00, 0x00, 0xff)), blank)
end

# ── Sampling a bound texture, and the layout it is bound in ─────────────────
#
# `KernelInterface.sample_texture_2d` names a SLOT, and on this backend the
# texture reaches the intrinsic through a lowering rather than an argument. What
# is asserted here is what a caller can see: which texel a coordinate lands on,
# and that the FILTERING is the sampler's rather than the shader's.

# A fullscreen triangle whose `uv` sweeps 0..1 across the target.
function tex_vertex()
    vid = vertex_index() - Int32(1)
    x = Float32(Int32(vid & Int32(1)) * 4 - 1)
    y = Float32(Int32((vid >> Int32(1)) & Int32(1)) * 4 - 1)
    return (position = Vec4f(x, Mantle.clip_y(y), 0f0, 1f0),
            uv = (0.5f0 * (x + 1f0), 0.5f0 * (y + 1f0)))
end

tex_fragment(inputs) =
    (Mantle.sample_texture_2d(UInt32(0), inputs.uv[1], inputs.uv[2], UInt32(0)),
     0f0, 0f0, 1f0)

tex_pipeline() = Mantle.GraphicsPipeline(;
    vertex = Mantle.VertexShader(tex_vertex; outputs = (uv = NTuple{2,Float32},)),
    fragment = Mantle.FragmentShader(tex_fragment; textures = 1),
    cull = Mantle.NoCull(), depth = Mantle.DepthOff())

"""Draw the sampling quad over `data` into an n x n target and read it back."""
function tex_draw(data; n = 4, filter = :nearest)
    be = Metal.MetalBackend()
    dev = Mantle.todevice(be)
    tex = Mantle.Texture2D(be, data)
    sampler = Mantle.Sampler(be; filter, wrap = :clamp)
    bindings = Mantle.bind_textures([Mantle.SampledTexture(tex, sampler)])
    fb = Mantle.Framebuffer(be, n, n; depth = false)
    target = Mantle.OffscreenTarget(fb)
    compiled = Mantle.compile_draw(dev, tex_pipeline(), (Mantle.blittarget(target),),
                                   nothing, (), ())
    Mantle.pass!(dev, target; clear = (0f0, 0f0, 0f0, 1f0)) do pr
        Mantle.viewport!(pr, 0, 0, n, n)
        Mantle.bindings!(pr, compiled, bindings)
        Mantle.draw!(pr, compiled, (), 3)
    end
    px = Mantle.readback_framebuffer(fb)
    # The red channel carries component 0; the readback is BGRA, and its second
    # index counts from the top while `uv.y = 0` is the bottom of the target.
    return [px[x, n + 1 - y][3] / 255 for x in 1:n, y in 1:n]
end

@testset "a texture is bound as data[x, y]" begin
    # NON-SQUARE on purpose: 4 wide and 2 tall is the case a square atlas cannot
    # tell apart, and it is the one both uploads used to get wrong — the Vulkan
    # side read the dimensions one way round and copied the bytes the other, so
    # only a square texture or a single row came out whole.
    data = Float32[(16 * (x - 1) + 2 * (y - 1)) / 100 for x in 1:4, y in 1:2]
    got = tex_draw(data; n = 4)

    # Four columns of the target over four texel columns; the two texel rows each
    # cover half the height. `data[x, y]`: x is the FIRST index.
    for x in 1:4, y in 1:2
        rows = y == 1 ? (1:2) : (3:4)     # y = 1 is v ≈ 0, the bottom half
        for r in rows
            @test isapprox(got[x, r], data[x, y]; atol = 0.01)
        end
    end
end

@testset "the filtering is the sampler's" begin
    # Two texels, black and white. A LINEAR sampler mixes them across the middle
    # of the target and a NEAREST one cannot: no intermediate value exists in the
    # texture, so anything between the two came from the texture unit.
    data = Float32[0f0 1f0]'          # 2 wide, 1 tall — data[x, y]
    near = tex_draw(data; n = 16, filter = :nearest)
    lin  = tex_draw(data; n = 16, filter = :linear)
    @test all(v -> v < 0.01 || v > 0.99, near)
    @test any(v -> 0.2 < v < 0.8, lin)
    # …and the ends still read as the texels themselves.
    @test lin[1, 8] < 0.1 && lin[end, 8] > 0.9
end

@testset "a negative viewport height flips what it draws" begin
    # Vulkan's spelling for "clip-space +y at the top", and the only thing that
    # can undo the mirror this backend's vertex stage applies. `y` names the
    # BOTTOM edge when the height is negative, so the rect is the same either way
    # and only the orientation differs.
    pts = Metal.MtlVector{Vec2f}([Vec2f(0f0, 0.5f0)])
    n = 64

    function draw_at(vp)
        be = Metal.MetalBackend()
        dev = Mantle.todevice(be)
        fb = Mantle.Framebuffer(be, n, n; depth = false)
        target = Mantle.OffscreenTarget(fb)
        p = lg_pipeline(lg_frag_uv)
        compiled = Mantle.compile_draw(dev, p, (Mantle.blittarget(target),), nothing,
                                       (pts,), (pts,))
        Mantle.pass!(dev, target; clear = (0f0, 0f0, 0f0, 1f0)) do pr
            Mantle.viewport!(pr, vp...)
            Mantle.draw!(pr, compiled, (pts,), 1)
        end
        return lg_covered(Mantle.readback_framebuffer(fb))
    end

    up   = draw_at((0, 0, n, n))          # portable default: clip +y is DOWN
    down = draw_at((0, n, n, -n))         # flipped: clip +y is UP

    # The same quad, the same size, in opposite halves — and both INSIDE the
    # target, which is what `abs(height)` with the given `y` did not manage: it
    # put the rect one whole target below and nothing came out where it should.
    #
    # `lg_covered` names its extents after a matrix, and the readback is indexed
    # the other way round: its FIRST index is the column and its second the row.
    # So `cols` is the vertical extent here.
    @test up.n == down.n == 16 * 16
    @test up.cols[1] > n ÷ 2            # +0.5 goes down…
    @test down.cols[2] < n ÷ 2          # …and up when the height is negative
    @test up.rows == down.rows          # x is untouched either way
end
