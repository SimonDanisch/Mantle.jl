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

    pipe = Mantle.GraphicsPipeline(; vertex = mantle_gfx_vertex,
                                     fragment = mantle_gfx_fragment,
                                     cull = Mantle.NoCull(),
                                     varyings = (tint = NTuple{4,Float32},))

    verts = NTuple{4,Float32}[(-0.9f0, -0.9f0, 0f0, 1f0), (0.9f0, -0.9f0, 0f0, 1f0),
                              (0f0, 0.9f0, 0f0, 1f0)]
    vbuf = MTLg.MTLBuffer(Metal.device(), sizeof(verts), pointer(verts);
                          storage = Metal.SharedStorage)

    Mantle.draw!(be, pipe, Mantle.OffscreenTarget(fb), 3; buffers = (vbuf,),
                 vert_tt = Tuple{Core.LLVMPtr{NTuple{4,Float32},1}})

    px = reshape(Mantle.readback_framebuffer(fb), 4, :)
    green = count(j -> px[2, j] > 0x80, 1:size(px, 2))
    # About 40 % of the frame, which is this triangle's area in NDC. A blank or
    # a fully-covered target both fail here.
    @test 1000 < green < 3000
    # …and the pixels outside it kept the clear colour, so the draw was bounded
    # by the geometry rather than covering everything.
    @test count(j -> px[2, j] <= 0x80, 1:size(px, 2)) > 1000

    # Compiling the same pipeline again must hit the cache: a pipeline rebuilt
    # per draw is the difference between a frame and a stall.
    second = Mantle.Framebuffer(be, 64, 64; depth = false)
    Mantle.draw!(be, pipe, Mantle.OffscreenTarget(second), 3; buffers = (vbuf,),
                 vert_tt = Tuple{Core.LLVMPtr{NTuple{4,Float32},1}})
    @test reshape(Mantle.readback_framebuffer(second), 4, :) == px
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
        Mantle.GraphicsPipeline(; vertex = mantle_gfx_vertex,
                                  fragment = mantle_gfx_fragment,
                                  geometry = (mantle_gfx_vertex, nothing)),
        MTLg.MTLPixelFormat[MTLg.MTLPixelFormatBGRA8Unorm], nothing, Tuple{}, Tuple{})
end
