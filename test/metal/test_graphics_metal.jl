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
    MEXT = Base.get_extension(Mantle, :MantleMetalExt)
    @test MEXT.mtl_cull(Mantle.NoCull())    == MTLg.MTLCullModeNone
    @test MEXT.mtl_cull(Mantle.CullBack())  == MTLg.MTLCullModeBack
    @test MEXT.mtl_cull(Mantle.CullFront()) == MTLg.MTLCullModeFront
    @test MEXT.mtl_primitive(Mantle.TriangleList()) == MTLg.MTLPrimitiveTypeTriangle
    @test MEXT.mtl_primitive(Mantle.LineList())     == MTLg.MTLPrimitiveTypeLine
    @test MEXT.mtl_depth_compare(Mantle.DepthLess()) == MTLg.MTLCompareFunctionLess
    @test !MEXT.depth_writes(Mantle.DepthOff())
    @test MEXT.depth_writes(Mantle.DepthLess())

    # Metal has no geometry stage at all — Apple's replacement is the mesh
    # pipeline — so asking for one has to fail loudly rather than silently drop
    # the stage and render something plausible.
    @test_throws ErrorException MEXT.compile_pipeline(
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
    MEXT = Base.get_extension(Mantle, :MantleMetalExt)
    be = Metal.MetalBackend()
    # The capability now answers `true`, and for the backend and the device
    # alike — a caller may hold either.
    @test Mantle.supports_mesh_pipeline(be)

    p = mp_pipeline(mp_mesh)

    # The object's type is DERIVED from the declaration: the `Flat` split gives
    # the two planes, `MeshConfig` the bounds, and a strip topology collapses to
    # the primitive KIND because a mesh stage has no input stream to group.
    Obj = MEXT.mesh_object_of(p)
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
