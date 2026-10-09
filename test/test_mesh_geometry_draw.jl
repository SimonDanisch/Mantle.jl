# Mesh pipelines and geometry pipelines, drawn and read back.
#
# Two capabilities, asked separately: `supports_mesh_pipeline` for a
# `MeshPipeline`, and `supports_geometry_stage || supports_mesh_pipeline` for a
# `GraphicsPipeline` with a geometry stage — a device with no geometry stage runs
# one through `Mantle.lower_geometry_to_mesh`, and nothing in the CALLER changes:
# the same description, the same `pass!`, the same vertex count. That is the
# reason `supports_geometry_stage` answering `false` does not mean an overlay is
# lost, and why the pixels asserted here are the same on a device that has the
# stage and one that lowers it.
#
# Written against the Metal backend first, where both mesh conversions fail
# silently: KernelInterface counts slots and indices from one and AIR from zero
# (off by one is a degenerate or misplaced primitive, never an error), and a
# vertex is a NamedTuple whose fields AIR wants through numbered intrinsics (a
# wrong number reads a different field in the fragment stage than the mesh stage
# wrote, which is why the per-primitive colour is asserted and not just coverage).
#
# The lowering's own arithmetic is asserted on the host in `test_lowering.jl`,
# over `HostMeshOutput`; what is asserted HERE is that the pixels are right.

using Test, Mantle
using Mantle: Vec2f, Vec4f
using ColorTypes: BGRA
using ColorTypes.FixedPointNumbers: N0f8
include(joinpath(@__DIR__, "testbackend.jl"))

n8(b::UInt8) = reinterpret(N0f8, b)
lit(v) = count(q -> q.g > n8(0x80), v)

"""Draw `p` immediately into an `n` x `n` BGRA target and read it back."""
function drawn(dev, p, count; n = 32, args = ())
    fb = Mantle.Framebuffer(TESTBACKEND, n, n; depth = false, color_format = BGRA{N0f8})
    Mantle.draw!(dev, p, Mantle.OffscreenTarget(fb), count; args)
    return Mantle.readback_framebuffer(fb)
end

# ── a portable MeshPipeline ──────────────────────────────────────────────────

function mp_mesh(out)
    Mantle.set_mesh_outputs!(out, 3, 1)
    Mantle.set_mesh_vertex!(out, 1, (position = (-1f0, -1f0, 0f0, 1f0), uv = (0f0, 0f0)))
    Mantle.set_mesh_vertex!(out, 2, (position = ( 3f0, -1f0, 0f0, 1f0), uv = (2f0, 0f0)))
    Mantle.set_mesh_vertex!(out, 3, (position = (-1f0,  3f0, 0f0, 1f0), uv = (0f0, 2f0)))
    Mantle.set_mesh_triangle!(out, 1, 1, 2, 3)
    Mantle.set_mesh_primitive_data!(out, 1, (colour = (0f0, 1f0, 0f0, 1f0),))
    return nothing
end

# Reads the per-primitive plane and returns it, which is the portable fragment
# spelling — one value per colour attachment.
mp_frag(inputs) = inputs.colour

# Half height in Mantle's clip space, to catch a mesh pipeline that forgot the
# y mirror the vertex path applies.
function mp_mesh_lower(out)
    Mantle.set_mesh_outputs!(out, 3, 1)
    Mantle.set_mesh_vertex!(out, 1, (position = (-1f0, -1f0, 0f0, 1f0), uv = (0f0, 0f0)))
    Mantle.set_mesh_vertex!(out, 2, (position = ( 3f0, -1f0, 0f0, 1f0), uv = (2f0, 0f0)))
    Mantle.set_mesh_vertex!(out, 3, (position = (-1f0,  0f0, 0f0, 1f0), uv = (0f0, 1f0)))
    Mantle.set_mesh_triangle!(out, 1, 1, 2, 3)
    Mantle.set_mesh_primitive_data!(out, 1, (colour = (0f0, 1f0, 0f0, 1f0),))
    return nothing
end

# The SAME pipeline as `mp_mesh`, with `colour` written in the VERTEX tuple
# instead of through `set_mesh_primitive_data!`. Both are legal and both must
# draw the same frame — see the testset below for why that is not a stylistic
# choice.
function mp_mesh_flat_via_vertex(out)
    green = (0f0, 1f0, 0f0, 1f0)
    Mantle.set_mesh_outputs!(out, 3, 1)
    Mantle.set_mesh_vertex!(out, 1, (position = (-1f0, -1f0, 0f0, 1f0), uv = (0f0, 0f0), colour = green))
    Mantle.set_mesh_vertex!(out, 2, (position = ( 3f0, -1f0, 0f0, 1f0), uv = (2f0, 0f0), colour = green))
    Mantle.set_mesh_vertex!(out, 3, (position = (-1f0,  3f0, 0f0, 1f0), uv = (0f0, 2f0), colour = green))
    Mantle.set_mesh_triangle!(out, 1, 1, 2, 3)
    return nothing
end

# Deliberately OFF CENTRE in y, so an unmirrored clip space moves the triangle
# to the other half instead of leaving it where it was. The same three positions
# `ref_vertex` reads out of a buffer.
const HALF_POSITIONS = NTuple{4,Float32}[(-0.9f0, -0.9f0, 0f0, 1f0), (0.9f0, -0.9f0, 0f0, 1f0),
                                         (0f0, 0.1f0, 0f0, 1f0)]
function mp_mesh_halfheight(out)
    Mantle.set_mesh_outputs!(out, 3, 1)
    Mantle.set_mesh_vertex!(out, 1, (position = (-0.9f0, -0.9f0, 0f0, 1f0), uv = (0f0, 0f0)))
    Mantle.set_mesh_vertex!(out, 2, (position = ( 0.9f0, -0.9f0, 0f0, 1f0), uv = (1f0, 0f0)))
    Mantle.set_mesh_vertex!(out, 3, (position = ( 0.0f0,  0.1f0, 0f0, 1f0), uv = (0.5f0, 1f0)))
    Mantle.set_mesh_triangle!(out, 1, 1, 2, 3)
    Mantle.set_mesh_primitive_data!(out, 1, (colour = (0f0, 1f0, 0f0, 1f0),))
    return nothing
end

# Position straight out of a buffer, so the vertex path adds no arithmetic.
ref_vertex(verts) = (position = verts[Int(Mantle.vertex_index())],)
ref_fragment(_) = (0f0, 1f0, 0f0, 1f0)

# The two builtins. `set_mesh_outputs!` from every invocation with the same
# counts, and first: Vulkan reads it from uniform control flow before any output
# is written, and on Metal the invocations race to write one value. Only
# invocation ONE writes the primitive, which is what makes `mesh_thread_index`
# counting from one the first invocation rather than none; group G draws its
# triangle scaled by 1/G, so a zero-based `mesh_group_index` divides by zero and
# covers nothing.
function mp_mesh_grouped(out)
    Mantle.set_mesh_outputs!(out, 3, 1)
    if Mantle.mesh_thread_index() == Int32(1)
        s = 1f0 / Float32(Mantle.mesh_group_index())
        Mantle.set_mesh_vertex!(out, 1, (position = (-s, -s, 0f0, 1f0), uv = (0f0, 0f0)))
        Mantle.set_mesh_vertex!(out, 2, (position = (3f0 * s, -s, 0f0, 1f0), uv = (2f0, 0f0)))
        Mantle.set_mesh_vertex!(out, 3, (position = (-s, 3f0 * s, 0f0, 1f0), uv = (0f0, 2f0)))
        Mantle.set_mesh_triangle!(out, 1, 1, 2, 3)
        Mantle.set_mesh_primitive_data!(out, 1, (colour = (0f0, 1f0, 0f0, 1f0),))
    end
    return nothing
end

mp_pipeline(f; threads = 1) = Mantle.MeshPipeline(;
    mesh = Mantle.MeshShader(f;
        outputs = (uv = NTuple{2,Float32}, colour = Mantle.Flat{NTuple{4,Float32}}),
        max_vertices = 4, max_primitives = 2,
        topology = Mantle.TriangleStrip(), threads),
    fragment = Mantle.FragmentShader(mp_frag),
    cull = Mantle.NoCull(), depth = Mantle.DepthOff())

@testset "a portable MeshPipeline" begin
    dev = Mantle.Device(TESTBACKEND)
    if !Mantle.supports_mesh_pipeline(dev)
        @info "no mesh pipeline on this backend; skipping"
        @test_skip Mantle.supports_mesh_pipeline(dev)
        return
    end

    @testset "draws the colour its body declared" begin
        p = mp_pipeline(mp_mesh)
        px = drawn(dev, p, 1)
        # Every pixel carries the per-primitive colour the body declared, so both
        # the coverage and the flat plane are asserted at once.
        @test all(==(BGRA{N0f8}(0, 1, 0, 1)), px)
        # Compiling the same pipeline again hits the cache rather than rebuilding.
        @test drawn(dev, p, 1) == px
    end

    @testset "a Flat output may be written per vertex" begin
        # Mantle's contract is Vulkan's: EVERY declared output is written by
        # `set_mesh_vertex!` and `Flat` only says how the varying is interpolated —
        # "flatness says WHERE a value is delivered, never what it is"
        # (`src/vulkan/graphics/api.jl`). So a portable mesh shader puts its flat
        # fields in the vertex tuple, and `set_mesh_primitive_data!` is the other,
        # equally legal spelling for a stage that would rather name the primitive.
        #
        # AIR has no flat qualifier on mesh vertex data — a mesh stage's flat
        # varying IS per-primitive data — so Metal splits the declaration into two
        # planes and has to bridge the two spellings. It did not: `colour` was in
        # the primitive plane and not in the vertex one, so the `@generated` writer
        # raised "`colour` is not a field of ..." from inside its generator, which
        # surfaced as `unsupported dynamic function invocation` naming neither
        # `colour` nor `Flat` — unnoticed until a real shader (`examples/isubd`)
        # tried it.
        px = drawn(dev, mp_pipeline(mp_mesh_flat_via_vertex), 1)
        # The SAME frame the `set_mesh_primitive_data!` spelling draws.
        @test all(==(BGRA{N0f8}(0, 1, 0, 1)), px)
    end

    @testset "a MeshPipeline uses Mantle's clip space" begin
        # The triangle covers y in [-1, 0], which is one half in Mantle's space
        # (Vulkan's, +y down). Which half it lands in on screen is the backend's
        # business; what this pins is that it is not the whole frame and that all
        # of it is on one side.
        px = drawn(dev, mp_pipeline(mp_mesh_lower), 1)
        top, bot = lit(@view px[:, 1:16]), lit(@view px[:, 17:32])
        @test top + bot > 200          # it drew something
        @test top + bot < 32 * 32      # …and not everything
        @test (top == 0) != (bot == 0) # …all of it on one side
    end

    @testset "a mesh stage lands in the same clip space as a vertex stage" begin
        # Mantle's clip space is Vulkan's, +y DOWN, and Metal's is the other one.
        # The vertex wrapper mirrors y on the way out; a mesh stage has to do the
        # same, and if it does not the image is upside down with nothing in the
        # pipeline objecting. Comparing the two paths pins it without this test
        # having to know which way up a readback is.
        vbuf = Mantle.Buffer(dev, HALF_POSITIONS)
        vpx = drawn(dev, Mantle.GraphicsPipeline(;
                             vertex = Mantle.VertexShader(ref_vertex),
                             fragment = Mantle.FragmentShader(ref_fragment),
                             cull = Mantle.NoCull(), depth = Mantle.DepthOff()), 3; args = (vbuf,))
        mpx = drawn(dev, mp_pipeline(mp_mesh_halfheight), 1)
        # The triangle is off centre, so the two halves differ and the comparison
        # has something to catch.
        top = lit(@view vpx[:, 1:16])
        bot = lit(@view vpx[:, 17:32])
        @test top != bot
        @test 100 < top + bot < 32 * 32
        # …and the mesh stage put it in the same place, pixel for pixel.
        @test mpx == vpx
        Mantle.free!(vbuf)
    end

    @testset "mesh_thread_index and mesh_group_index count from one" begin
        # Group 1 scales by 1/1 and covers the frame.
        @test lit(drawn(dev, mp_pipeline(mp_mesh_grouped), 1)) == 32 * 32
        # Group 2 draws a quarter-size triangle INSIDE group 1's, so two groups
        # still cover the frame — and the assertion that matters is that adding a
        # group does not lose the first one.
        @test lit(drawn(dev, mp_pipeline(mp_mesh_grouped), 2)) == 32 * 32
        # Four threads, one primitive: only thread 1 writes it.
        @test lit(drawn(dev, mp_pipeline(mp_mesh_grouped; threads = 4), 1)) == 32 * 32
    end
end

# ── a geometry pipeline, on every device that can draw one ───────────────────

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
lg_seg_vertex(points) = lg_seg_vertex(Mantle.VertexIndex(Mantle.vertex_index()), points)

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

# A vertex body that cannot be handed its index.
lg_noindex(points) = (position = Vec4f(0f0, 0f0, 0f0, 1f0), tint = 0f0)

"""
Record one draw of `p` through `pass!` — the path RayMakie's overlays take.

`compile_draw` and `record_draw!` and nothing geometry-specific: a caller that
has a geometry pipeline writes exactly this on every backend.
"""
function lg_draw(dev, p, args, count; n = 64, indices = nothing, instances = 1,
                 viewport = (0, 0, n, n))
    fb = Mantle.Framebuffer(TESTBACKEND, n, n; depth = false, color_format = BGRA{N0f8})
    target = Mantle.OffscreenTarget(fb)
    compiled = Mantle.compile_draw(dev, p, (Mantle.blittarget(target),), nothing,
                                   args, args)
    Mantle.pass!(dev, target; clear = (0f0, 0f0, 0f0, 1f0)) do pr
        Mantle.viewport!(pr, viewport...)
        Mantle.draw!(pr, compiled, args, count; indices, instances)
    end
    return Mantle.readback_framebuffer(fb)
end

"""The pixels that are not the clear colour, and the block they occupy."""
function lg_covered(px)
    idx = findall(q -> q != BGRA{N0f8}(0, 0, 0, 1), px)
    isempty(idx) && return (n = 0, rows = (0, 0), cols = (0, 0), colours = eltype(px)[])
    r = extrema(getindex.(idx, 1))
    c = extrema(getindex.(idx, 2))
    # `by`, because a colorant has no `isless` — and should not: there is no
    # meaningful order on colours, only on their channels. Sorting by the tuple
    # makes the order explicit and keeps these assertions stable.
    return (n = length(idx), rows = r, cols = c,
            colours = sort(unique(px[idx]); by = q -> (q.r, q.g, q.b, q.alpha)))
end

@testset "a geometry pipeline" begin
    dev = Mantle.Device(TESTBACKEND)
    if !(Mantle.supports_geometry_stage(dev) || Mantle.supports_mesh_pipeline(dev))
        @info "no geometry stage and nothing to lower it onto; skipping"
        @test_skip Mantle.supports_geometry_stage(dev)
        return
    end
    # Lowered onto the mesh stage exactly when the device has no geometry stage.
    lowered = !Mantle.supports_geometry_stage(dev)

    @testset "a point becomes a quad" begin
        # One point at the origin: a quad of 0.5 in NDC is a quarter of each axis,
        # so 16x16 of a 64x64 frame, centred.
        pts = Mantle.devicearray(TESTBACKEND, [Vec2f(0, 0)])
        px = lg_draw(dev, lg_pipeline(lg_frag_uv), (pts,), 1)
        cov = lg_covered(px)
        @test cov.n == 16 * 16
        @test cov.rows == (25, 40) && cov.cols == (25, 40)

        # The SMOOTH plane interpolates: `uv` runs from -0.25 at one edge to +0.25
        # at the other, so `abs(uv) * 4` sweeps nearly the whole range and is
        # smallest in the middle. This is what the per-vertex plane being
        # addressed per FIELD and per SLOT buys — with the two AIR operands the
        # other way round only the first vertex's `uv` landed and the
        # interpolation collapsed to a corner ramp.
        sub = px[cov.rows[1]:cov.rows[2], cov.cols[1]:cov.cols[2]]
        ch2 = map(q -> q.g, sub)
        ch3 = map(q -> q.r, sub)
        @test maximum(ch2) > n8(0xd0) && maximum(ch3) > n8(0xd0)
        @test minimum(ch2) < n8(0x20) && minimum(ch3) < n8(0x20)
        # Symmetric about the centre in both directions, which a one-corner ramp
        # is not: the first row equals the last, and the first column the last.
        @test ch2[1, :] == ch2[end, :]
        @test ch3[:, 1] == ch3[:, end]
        # The blue channel is the fragment's own constant, so every covered pixel
        # has it — a check that the frame is the shader's and not something left
        # over.
        @test all(q -> q.b == n8(0x80), sub)
    end

    @testset "one primitive per input primitive" begin
        # Three points, three quads, and each carries its own vertex index as a
        # per-primitive value. A lowering that ignored `mesh_group_index` would
        # draw one quad three times over; one that got the flat plane wrong would
        # draw three quads of one colour.
        pts = Mantle.devicearray(TESTBACKEND, [Vec2f(-0.5, -0.5), Vec2f(0.5, 0.5), Vec2f(-0.5, 0.5)])
        px = lg_draw(dev, lg_pipeline(lg_frag_tint), (pts,), 3)
        cov = lg_covered(px)
        @test cov.n == 3 * 16 * 16
        # `tint = vertex index / 4`, so 0.25, 0.5 and 0.75 — as RED, which is what
        # the shader wrote. Where red SITS in the attachment is the colorant's
        # business, not this assertion's.
        @test cov.colours == [BGRA{N0f8}(n8(0x40), 0, 0, 1), BGRA{N0f8}(n8(0x80), 0, 0, 1),
                              BGRA{N0f8}(n8(0xbf), 0, 0, 1)]
    end

    @testset "an indexed draw fetches its own vertex indices" begin
        # `LineStripAdjacency` over `0 0 1 2 3 3`: a four-wide window sliding one
        # index at a time, which is how RayMakie's `lines` walks a polyline. A
        # mesh pipeline has no index buffer, so a lowered stage reads it as one of
        # its own arguments — the one part of the lowering that is more than a
        # rearrangement.
        pts = Mantle.devicearray(TESTBACKEND, [Vec2f(-0.6, -0.6), Vec2f(-0.2, -0.2),
                                               Vec2f(0.2, 0.2), Vec2f(0.6, 0.6)])
        ib = Mantle.indexbuffer(dev, UInt32[0, 0, 1, 2, 3, 3])
        px = lg_draw(dev, lg_seg_pipeline(), (pts,), 6; indices = ib)
        cov = lg_covered(px)
        # Six indices give three segments, each a 0.2-wide quad: 6 or 7 pixels a
        # side on a 64-pixel axis.
        @test 3 * 30 < cov.n < 3 * 50
        # One colour per segment, and they are the OUTER two vertices of each
        # window: (1+3)/16, (1+4)/16, (2+4)/16 one-based.
        @test cov.colours == [BGRA{N0f8}(n8(0x40), 0, 0, 1), BGRA{N0f8}(n8(0x50), 0, 0, 1),
                              BGRA{N0f8}(n8(0x60), 0, 0, 1)]
        # …and the three sit along the diagonal, at the midpoints of the segments.
        centres = [(sum(getindex.(findall(==(c), px), 1)) / count(==(c), px),
                    sum(getindex.(findall(==(c), px), 2)) / count(==(c), px))
                   for c in cov.colours]
        @test all(abs(c[1] - c[2]) < 1 for c in centres)
        @test issorted(first.(centres))
    end

    @testset "fewer vertices than one primitive draws nothing" begin
        # Not an error: a `lines` plot with one point assembles no segment. A
        # lowered draw has to check the count rather than pass it on, because
        # Metal's `drawMeshThreadgroups` refuses a zero grid.
        pts = Mantle.devicearray(TESTBACKEND, [Vec2f(0, 0)])
        ib = Mantle.indexbuffer(dev, UInt32[0, 0, 1])
        blank = lg_draw(dev, lg_seg_pipeline(), (pts,), 3; indices = ib)
        @test all(==(BGRA{N0f8}(0, 0, 0, 1)), blank)
    end

    @testset "a lowered draw refuses what it cannot express" begin
        # Only where the stage is lowered: a native geometry stage takes both.
        if !lowered
            @test_skip lowered
        else
            pts = Mantle.devicearray(TESTBACKEND, [Vec2f(0, 0)])
            # Instancing would put the instance on the mesh grid's second axis,
            # which needs a portable name for `instance_index()` inside a mesh
            # stage. There is none, so this says so rather than drawing one
            # instance and reporting success.
            @test_throws ErrorException lg_draw(dev, lg_pipeline(lg_frag_tint), (pts,), 1;
                                               instances = 2)
            # A vertex body that cannot be handed its index names itself, rather
            # than failing as a `MethodError` from inside a shader compilation.
            p = Mantle.GraphicsPipeline(;
                vertex = Mantle.VertexShader(lg_noindex; outputs = (tint = Float32,)),
                geometry = Mantle.GeometryShader(lg_geometry;
                                                outputs = (uv = NTuple{2,Float32},
                                                           tint = Mantle.Flat{NTuple{4,Float32}}),
                                                input = Mantle.PointList(),
                                                output = Mantle.TriangleStrip(), max_vertices = 4),
                fragment = Mantle.FragmentShader(lg_frag_tint),
                topology = Mantle.PointList(), cull = Mantle.NoCull(), depth = Mantle.DepthOff())
            err = try
                lg_draw(dev, p, (pts,), 1)
            catch e
                e
            end
            @test err isa ErrorException
            @test occursin("VertexIndex", err.msg)
        end
    end

    @testset "a negative viewport height flips what it draws" begin
        # Vulkan's spelling for "clip-space +y at the top", and the only thing
        # that can undo the mirror Metal's vertex stage applies. `y` names the
        # BOTTOM edge when the height is negative, so the rect is the same either
        # way and only the orientation differs.
        n = 64
        pts = Mantle.devicearray(TESTBACKEND, [Vec2f(0f0, 0.5f0)])
        up   = lg_covered(lg_draw(dev, lg_pipeline(lg_frag_uv), (pts,), 1;
                                  viewport = (0, 0, n, n)))     # clip +y is DOWN
        down = lg_covered(lg_draw(dev, lg_pipeline(lg_frag_uv), (pts,), 1;
                                  viewport = (0, n, n, -n)))    # clip +y is UP

        # The same quad, the same size, in opposite halves — and both INSIDE the
        # target, which is what `abs(height)` with the given `y` did not manage:
        # it put the rect one whole target below and nothing came out where it
        # should.
        #
        # `lg_covered` names its extents after a matrix, and the readback is
        # indexed the other way round: its FIRST index is the column and its
        # second the row. So `cols` is the vertical extent here.
        @test up.n == down.n == 16 * 16
        @test up.cols[1] > n ÷ 2            # +0.5 goes down…
        @test down.cols[2] < n ÷ 2          # …and up when the height is negative
        @test up.rows == down.rows          # x is untouched either way
    end
end
