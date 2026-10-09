"""
A render GRAPH, drawn by Julia shaders: attachments the COMPILER placed, draws
compiled by the `Pipelines` phase, and a frame executed by Mantle's own
render-pass sequencing. `test_graphics_pipeline.jl` is the other half — one
pipeline, one target the caller made, drawn immediately.

Written against the Metal backend first, where three things had to become
portable for it: `TransientImage` lived in `src/vulkan/graph.jl` with `VK.Format`
and `VK.Image` fields, so a render target was a Vulkan concept; the image arena
had to become a placement heap, because a texture over buffer memory is LINEAR
and an Apple GPU cannot render into one; and a graph draw's arguments arrive as
device arrays, so they compile and bind through the same ABI a kernel launch
uses. The same assertions now run on every backend with graphics.

Targets are read back through the graph — a `copy!` pass into a persistent
`Buffer` — which every backend runs. A persistent buffer and not a transient:
nothing in the graph reads the copy, so two transient readouts may share memory
and the second copy lands in both.

The window and presentation-surface halves need a display and are not here.
"""

using Test, Mantle, ColorTypes, KernelAbstractions
using ColorTypes.FixedPointNumbers: N0f8
using GeometryBasics: Vec4f, Vec3f
include(joinpath(@__DIR__, "testbackend.jl"))
const M = Mantle

# Indexed, NOT `unsafe_load`: the point is that a graph draw's argument is a
# device ARRAY with its length, the same thing a kernel gets, so a shader
# written once compiles on either backend. That indexing carries a bounds check,
# which is what forced a graphics stage to compile at debug level 0 on Metal.
rg_vertex(verts) = (position = verts[Int(M.vertex_index())], tint = (0f0, 1f0, 0f0, 1f0))
rg_fragment(inputs) = inputs.tint

const RG_DEV = M.Device(TESTBACKEND)
const RG_PIPE = M.GraphicsPipeline(; vertex = M.VertexShader(rg_vertex; outputs = (tint = NTuple{4,Float32},)),
                                     fragment = M.FragmentShader(rg_fragment),
                                     cull = M.NoCull())
const RG_TRI = M.Buffer(RG_DEV, NTuple{4,Float32}[(-0.9f0, -0.9f0, 0f0, 1f0),
                                                  ( 0.9f0, -0.9f0, 0f0, 1f0),
                                                  ( 0f0,    0.9f0, 0f0, 1f0)])
# The pixels of RG_TRI on a 64x64 target. No pixel centre lies on one of its
# edges, so no fill-rule tie can move the count between backends.
const RG_COVERED = 1682

"""Read `img` out through the graph: a copy pass into a persistent buffer."""
function readout!(g, img; name = "read")
    out = M.Buffer(RG_DEV, eltype(img), prod(size(img)))
    M.copy!(g, name, out, img)
    return out
end

"""What `readout!` copied, as a `(width, height)` matrix: `[x, y]`, row 1 on top."""
pixels(out, w = 64, h = 64) = reshape(Array(M.storage(out)), w, h)

greens(img) = count(p -> ColorTypes.green(p) > 0.5, img)

# ── the g-buffer shape ───────────────────────────────────────────────────────
#
# `bench/showcase.jl`'s deferred pass, reduced to what makes it different from
# the triangle: three colour attachments written in one pass, and varyings
# spelled `Vec4f`/`Vec3f` rather than `NTuple`.

mrt_vertex(verts) = (position = verts[Int(M.vertex_index())],
                     albedo = Vec4f(1f0, 0f0, 0f0, 1f0),
                     normal = Vec3f(0f0, 1f0, 0f0))
# One value per attachment. A single colour is ALSO a four-tuple, so telling the
# two apart is on the element type — `r isa Tuple` read one `NTuple{4,Float32}`
# as four render targets, threw inside the `NamedTuple` constructor, and left a
# fragment stage that was `noreturn` and drew nothing.
mrt_fragment(inputs) = (inputs.albedo,
                        Vec4f(0f0, 0f0, 1f0, 1f0),
                        Vec4f(inputs.normal[1], inputs.normal[2], inputs.normal[3], 1f0))

const RG_MRT = M.GraphicsPipeline(; vertex = M.VertexShader(mrt_vertex; outputs = (albedo = Vec4f, normal = Vec3f)),
                                    fragment = M.FragmentShader(mrt_fragment),
                                    cull = M.NoCull())

rg_depth_only(verts) = (position = verts[Int(M.vertex_index())],)
rg_no_colour(inputs) = nothing

const RG_SHADOW = M.GraphicsPipeline(; vertex = M.VertexShader(rg_depth_only),
                                       fragment = M.FragmentShader(rg_no_colour),
                                       cull = M.NoCull(),
                                       depth = M.DepthLess())

@kernel function rg_setcount!(cmd, n::Int32)
    cmd[1] = M.DrawIndirectCommand(UInt32(n), UInt32(1), UInt32(0), UInt32(0))
end

@kernel function rg_touch!(x)
    i = @index(Global)
    @inbounds x[i] += 1f0
end

# Something a target follows the size of, as a window does. `target_extent` is
# all a tracking transient asks of its source.
mutable struct ResizableSource
    extent::Tuple{Int,Int}
end
M.target_extent(s::ResizableSource) = s.extent

@testset "a render graph" begin
    if !M.supports_graphics(RG_DEV)
        @info "no graphics pipeline on this backend; skipping"
        @test_skip M.supports_graphics(RG_DEV)
        return
    end

    @testset "a transient render target is portable" begin
        g = M.Graph(RG_DEV)
        color = M.Transient.Image(g, RGBA{N0f8}, (64, 64))
        z     = M.Transient.Image(g, Float32, (64, 64))

        # The type is core's, and only the element type is spelled by the caller.
        @test color isa M.TransientImage{RGBA{N0f8}}
        @test z isa M.TransientImage{Float32}
        @test eltype(color) === RGBA{N0f8}
        @test size(color) == (64, 64)
        @test M.target_extent(color) == (64, 64)
        @test M.arena(color) === M.Images()
        # Interpolated on both sides, not spelled out: whether Julia prints the
        # element type as `RGBA{N0f8}` or `RGBA{FixedPointNumbers.N0f8}` depends on
        # which modules the session happens to have loaded, and what `describe` is
        # for is the FORMAT around it.
        @test M.describe(color) == "Transient.Image($(eltype(color)), (64, 64))"
        @test occursin("N0f8", M.describe(color))

        # `Float32` is depth on every backend. Metal's format table answered
        # `R32Float`, a COLOUR format, while the one caller hardcoded a depth
        # format beside it and so never noticed.
        @test M.isdepth(z)
        @test !M.isdepth(color)

        # What the placer reads, asked of the driver before any memory exists.
        @test M.nbytes(color) > 64 * 64 * 4 - 1
        @test M.alignment(RG_DEV, color) > 0
        @test ispow2(M.alignment(RG_DEV, color))
        # Unbound until the plan places it: the whole reason the object exists
        # this early is to be ASKED how much room it wants.
        @test M.target_view(color) === nothing
    end

    @testset "the graph places a target and draws Julia shaders into it" begin
        g = M.Graph(RG_DEV)
        color = M.Transient.Image(g, RGBA{N0f8}, (64, 64))
        M.render!(g, "tri", color => M.Clear((0f0, 0f0, 0f0, 1f0))) do p
            M.draw!(p, RG_PIPE, (RG_TRI,), 3)
        end
        out = readout!(g, color)
        plan = M.record!(M.Plan(g))

        # The `Pipelines` phase compiled the draw. It used to push an empty
        # `CompiledDraw[]` on Metal, so a render pass in a graph did nothing at
        # all on any backend without a batch queue.
        #
        # The first pass is the host-update one: a draw DECLARES what its stages
        # touch, so the vertex buffer is a usage of the render pass, and a pass
        # that uses a host-writable resource gets the update pass that orders host
        # stores against it. `test_window.jl` pins the same shape as
        # `["updates", "plot"]`.
        @test [pp.pass.name for pp in plan.passes] == ["updates", "tri", "read"]
        @test length(only(pp for pp in plan.passes if pp.pass.name == "tri").draws) == 1

        M.run!(plan)
        img = pixels(out)
        @test size(img) == (64, 64)
        @test img isa Matrix{RGBA{N0f8}}
        @test greens(img) == RG_COVERED
        @test img[32, 32] == RGBA{N0f8}(0, 1, 0, 1)
        # …and the clear survived outside it, so the draw was bounded by geometry
        # rather than covering the target.
        @test count(p -> ColorTypes.green(p) <= 0.5, img) == 4096 - RG_COVERED

        # Running the same plan again must give the same frame: a target that came
        # back at a different offset would render into someone else's bytes.
        M.run!(plan)
        @test pixels(out) == img
        M.free!(plan)
        M.free!(out)
    end

    @testset "two targets that never overlap, in one arena, both render" begin
        # The whole point of placing render targets: `a` is finished before `b`
        # starts, so the compiler is free to give them the same bytes. On Metal
        # this needed the placement heap — `Automatic` picks its own offsets, so
        # the placement would have been advisory.
        g = M.Graph(RG_DEV)
        a = M.Transient.Image(g, RGBA{N0f8}, (64, 64))
        b = M.Transient.Image(g, RGBA{N0f8}, (64, 64))
        M.render!(g, "first",  a => M.Clear((1f0, 0f0, 0f0, 1f0))) do p
            M.draw!(p, RG_PIPE, (RG_TRI,), 3)
        end
        M.render!(g, "second", b => M.Clear((0f0, 0f0, 1f0, 1f0))) do p
            M.draw!(p, RG_PIPE, (RG_TRI,), 3)
        end
        out = readout!(g, b)
        plan = M.record!(M.Plan(g))
        M.run!(plan)
        # Both were written by their own pass, so whatever the aliasing decided,
        # the second pass's clear is what `b` holds outside the triangle.
        ib = pixels(out)
        @test greens(ib) == RG_COVERED
        @test count(==(RGBA{N0f8}(0, 0, 1, 1)), ib) == 4096 - RG_COVERED
        M.free!(plan)
        M.free!(out)
    end

    @testset "a tracking target refits without naming a driver" begin
        # `refit!` was Vulkan's, and recreated a `VkImage`. It is core's now and
        # the backend supplies only `remakeimage!`.
        g = M.Graph(RG_DEV)
        src = ResizableSource((32, 32))
        t = M.Transient.Image(g, RGBA{N0f8}, src)
        @test size(t) == (32, 32)
        # The device is an explicit argument (`graph/build.jl`), so a refit names
        # the device it reallocates on rather than reaching for a global.
        @test !M.refit!(RG_DEV, t)           # nothing moved
        src.extent = (48, 48)
        @test M.refit!(RG_DEV, t)
        @test size(t) == (48, 48)
        @test M.nbytes(t) >= 48 * 48 * 4
        @test M.target_view(t) === nothing   # unbound again; the placement is void
    end

    @testset "three attachments in one pass" begin
        g = M.Graph(RG_DEV)
        albedo = M.Transient.Image(g, RGBA{N0f8}, (64, 64))
        matter = M.Transient.Image(g, RGBA{N0f8}, (64, 64))
        normal = M.Transient.Image(g, RGBA{N0f8}, (64, 64))
        M.render!(g, "gbuffer", albedo => M.Clear((0f0, 0f0, 0f0, 1f0)),
                                matter => M.Clear((0f0, 0f0, 0f0, 1f0)),
                                normal => M.Clear((0f0, 0f0, 0f0, 1f0))) do p
            M.draw!(p, RG_MRT, (RG_TRI,), 3)
        end
        # Every render target needs its offset aligned to what the driver asked
        # for, and so does the REGION they sit in. Reserved at `reserve!`'s
        # default 256 while a Metal texture wants 2048, the second target landed
        # 128 bytes short of its boundary.
        oa = readout!(g, albedo; name = "read albedo")
        om = readout!(g, matter; name = "read matter")
        on = readout!(g, normal; name = "read normal")
        plan = M.record!(M.Plan(g))

        M.run!(plan)
        ia, im, inm = pixels(oa), pixels(om), pixels(on)
        # Each attachment got ITS OWN colour: a pipeline that wrote target 0 three
        # times, or dropped two of the three, fails here rather than looking right.
        @test ia[32, 32] == RGBA{N0f8}(1, 0, 0, 1)
        @test im[32, 32] == RGBA{N0f8}(0, 0, 1, 1)
        @test inm[32, 32] == RGBA{N0f8}(0, 1, 0, 1)
        @test count(p -> ColorTypes.red(p)   > 0.5, ia)  == RG_COVERED
        @test count(p -> ColorTypes.blue(p)  > 0.5, im)  == RG_COVERED
        @test count(p -> ColorTypes.green(p) > 0.5, inm) == RG_COVERED
        M.free!(plan)
        foreach(M.free!, (oa, om, on))
    end

    @testset "a depth-only pass writes depth and a copy pass reads it back" begin
        # A shadow pass has no colour attachment at all, and Metal spells that as
        # a pipeline with NO fragment function — compiling a stage that returns an
        # empty struct is not something AIR has.
        #
        # `copy!` is the other half: five of showcase's twenty passes are copies,
        # and the shared graph did not execute them at all. `run!` walked
        # dispatches and draws, so on any backend without a batch queue the
        # deferred half of that frame read an empty g-buffer.
        g = M.Graph(RG_DEV)
        z   = M.Transient.Image(g, Float32, (64, 64))
        zpx = M.Transient.Buffer(g, Float32, 64 * 64)
        tri = M.Buffer(RG_DEV, NTuple{4,Float32}[(-0.9f0, -0.9f0, 0.25f0, 1f0),
                                                 ( 0.9f0, -0.9f0, 0.25f0, 1f0),
                                                 ( 0f0,    0.9f0, 0.25f0, 1f0)])
        M.render!(g, "shadow", z => M.Clear(1f0)) do p
            M.draw!(p, RG_SHADOW, (tri,), 3)
        end
        M.copy!(g, "read shadow", zpx, z)
        plan = M.record!(M.Plan(g))
        M.run!(plan)

        d = Array(M.storage(zpx))
        @test length(d) == 64 * 64
        # 0.25 inside the triangle, the clear outside, and the same pixels.
        @test minimum(d) ≈ 0.25f0
        @test maximum(d) ≈ 1.0f0
        @test count(x -> x < 0.9f0, d) == RG_COVERED
        M.free!(plan)
        M.free!(tri)
    end

    @testset "an indirect draw reads a count a compute pass wrote" begin
        # The form where the host never learns the count. Two things this pins:
        #
        #   * the indirect buffer must be made RESIDENT for the render encoder on
        #     Metal. Nothing in the shader names it — the command processor reads
        #     it — so it is easy to miss, and missing it does not fail: the GPU
        #     reads whatever is mapped, takes the vertex count from it, and asks
        #     for however many that is. Four billion vertices is a command buffer
        #     that never completes.
        #   * the compute pass that writes it has to have run first, which is the
        #     graph's ordering.
        function indirectplan()
            g = M.Graph(RG_DEV)
            color = M.Transient.Image(g, RGBA{N0f8}, (64, 64))
            cmd = M.Buffer(RG_DEV, M.DrawIndirectCommand, 1)
            M.dispatch!(g, rg_setcount!, (cmd, Int32(3)), 1; name = "count")
            M.render!(g, "tri", color => M.Clear((0f0, 0f0, 0f0, 1f0))) do p
                M.draw!(p, RG_PIPE, (RG_TRI,), cmd)
            end
            out = readout!(g, color)
            return M.record!(M.Plan(g)), cmd, out
        end
        plan, cmd, out = indirectplan()
        M.run!(plan)
        @test Array(cmd)[1].vertices == 3
        @test greens(pixels(out)) == RG_COVERED
        M.free!(plan)

        # Again, three times, and the reason is a hazard that only shows from the
        # SECOND plan on. On Metal `useResource` is not only about residency: it
        # is how the driver learns that a dispatch touched those bytes, and a
        # resource that is resident but undeclared is one it believes nobody used
        # — so it may overlap the command buffer behind this one with it. A
        # dispatch that handed the encoder arguments already converted to raw
        # addresses skipped that declaration, and this draw read the count from
        # BEFORE the dispatch, in seven runs of eight. The one that passed is why
        # a single run is not enough of a test.
        for _ in 1:3
            plan2, _, out2 = indirectplan()
            M.run!(plan2)
            @test greens(pixels(out2)) == RG_COVERED
            M.free!(plan2)
        end
    end

    @testset "the FIRST run of a plan is already correct" begin
        # Not a tautology. With Metal's image arena heap and its textures created
        # `Untracked`, on the reasoning that Mantle's graph emitted the dependency,
        # the driver was free to overlap a readback with the render pass feeding
        # it, and did, exactly once per process: the first frame came back as the
        # untouched texture, alpha and all, and every frame after it was right.
        g = M.Graph(RG_DEV)
        color = M.Transient.Image(g, RGBA{N0f8}, (64, 64))
        M.render!(g, "tri", color => M.Clear((0f0, 0f0, 0f0, 1f0))) do p
            M.draw!(p, RG_PIPE, (RG_TRI,), 3)
        end
        out = readout!(g, color)
        plan = M.record!(M.Plan(g))
        M.run!(plan)
        first_frame = pixels(out)
        # The clear alone would fail this: an untouched target reads as zeros, and
        # the clear colour has alpha 1.
        @test all(p -> ColorTypes.alpha(p) == 1, first_frame)
        @test greens(first_frame) == RG_COVERED
        M.run!(plan)
        @test pixels(out) == first_frame
        M.free!(plan)
    end

    @testset "an arena serves tenants that need more alignment than it defaults to" begin
        # Two targets in one arena, each wanting the alignment the driver asked
        # for. `reserve!` defaults to 256 and its fast path handed back a kept
        # region without re-checking, so on Metal (2048) the second target landed
        # 128 bytes short of its boundary and was refused. Sizes that are NOT a
        # multiple of the alignment are what expose it.
        g = M.Graph(RG_DEV)
        a = M.Transient.Image(g, RGBA{N0f8}, (64, 64))
        b = M.Transient.Image(g, RGBA{N0f8}, (64, 64))
        M.render!(g, "a", a => M.Clear((0f0, 0f0, 0f0, 1f0))) do p
            M.draw!(p, RG_PIPE, (RG_TRI,), 3)
        end
        M.render!(g, "b", b => M.Clear((0f0, 0f0, 0f0, 1f0))) do p
            M.draw!(p, RG_PIPE, (RG_TRI,), 3)
        end
        out = readout!(g, b)
        plan = M.record!(M.Plan(g))
        for t in (a, b)
            @test ispow2(M.alignment(RG_DEV, t))
        end
        M.run!(plan)
        @test greens(pixels(out)) == RG_COVERED
        M.free!(plan)
        M.free!(out)
    end

    @testset "a profiled plan times its passes" begin
        # `makeprofiler` answered `nothing` for anything but Vulkan, so
        # `Plan(g; profile = true)` quietly produced a plan with no profiler and
        # `timings` did not exist to be called. The host side is something every
        # backend can measure, and it is what says where a frame goes when there
        # are no GPU timestamps.
        g = M.Graph(RG_DEV)
        color = M.Transient.Image(g, RGBA{N0f8}, (64, 64))
        M.render!(g, "tri", color => M.Clear((0f0, 0f0, 0f0, 1f0))) do p
            M.draw!(p, RG_PIPE, (RG_TRI,), 3)
        end
        out = readout!(g, color)

        plain = M.record!(M.Plan(g))
        # Asking a plan that was not built for it says so, rather than returning
        # zeros that look like a fast frame.
        @test_throws ArgumentError M.timings(plain)
        M.free!(plain)

        plan = M.record!(M.Plan(g; profile = true))
        for _ in 1:3
            M.run!(plan)
        end
        M.waitidle(RG_DEV)
        ts = M.timings(plan)
        @test length(ts) == length(plan.passes)
        # The host-update pass leads, for the same reason it does above.
        @test [t.name for t in ts] == ["updates", "tri", "read"]
        @test [t.kind for t in ts] == [:update, :render, :copy]
        @test all(t -> t.samples <= M.NSAMPLES, ts)
        if M.timestamps(RG_DEV)
            # Timestamps came back for the passes that record commands, and they
            # are times rather than zeros. The update pass lands its stores as a
            # transfer the profiler does not bracket.
            for t in ts
                t.kind === :update && continue
                @test t.samples > 0
                @test !isnan(t.gpu_ms) && t.gpu_ms >= 0
            end
        else
            # No timestamp queries: every pass is still timed on the host, which
            # is the time `run!` spends in it, including whatever the backend
            # waits for inside it.
            @test all(t -> t.samples == 3, ts)
            @test all(t -> t.host_ms >= 0, ts)
        end
        # …and the frame still renders while being measured.
        @test greens(pixels(out)) == RG_COVERED
        M.free!(plan)
        M.free!(out)
    end

    @testset "a recorded pass reports no GPU time rather than zero" begin
        # Core profiles a pass where it is emitted, which for a recording is
        # `record!`: nothing has run there. On a device without timestamp queries
        # `timings` said `0.0` GPU ms for every dispatch of such a plan, and a
        # declared GEMV read as free.
        g = M.Graph(RG_DEV)
        x = M.Buffer(RG_DEV, zeros(Float32, 16))
        M.dispatch!(g, rg_touch!, (x,), 16; name = "touch")
        plan = M.record!(M.Plan(g; profile = true))
        for _ in 1:3
            M.run!(plan)
        end
        M.waitidle(RG_DEV)
        @test Array(M.storage(x)) == fill(3f0, 16)
        t = only(t for t in M.timings(plan) if t.name == "touch")
        if M.timestamps(RG_DEV)
            @test !isnan(t.gpu_ms)
        else
            @test isnan(t.gpu_ms)
            @test t.samples == 0
        end
        M.free!(plan)
        M.free!(x)
    end

    @testset "a pass leaves nothing open, and the frame is still correct" begin
        # Keeping ONE command buffer open across consecutive render and copy
        # passes, committed only where the graph said a dispatch was about to
        # follow, was the wrong trade: what is open depends on every call since it
        # was opened, so nothing can be scheduled around it, and batching is the
        # graph's job rather than the queue's.
        #
        # What is testable is the contract that replaced it: a readback needs no
        # submit from the caller and nothing is left dangling between passes, so a
        # second run over the same plan, with nothing submitted by hand in
        # between, still lands.
        g = M.Graph(RG_DEV)
        a = M.Transient.Image(g, RGBA{N0f8}, (64, 64))
        px = M.Transient.Buffer(g, RGBA{N0f8}, 64 * 64)
        b = M.Transient.Image(g, RGBA{N0f8}, (64, 64))
        M.render!(g, "one", a => M.Clear((0f0, 0f0, 0f0, 1f0))) do p
            M.draw!(p, RG_PIPE, (RG_TRI,), 3)
        end
        M.copy!(g, "read one", px, a)
        M.render!(g, "two", b => M.Clear((0f0, 0f0, 0f0, 1f0))) do p
            M.draw!(p, RG_PIPE, (RG_TRI,), 3)
        end
        out = readout!(g, b; name = "read two")
        plan = M.record!(M.Plan(g))

        M.run!(plan)
        @test greens(pixels(out)) == RG_COVERED
        M.run!(plan)
        @test greens(pixels(out)) == RG_COVERED
        M.free!(plan)
        M.free!(out)
    end

    @testset "a dispatch between two render passes keeps commit order" begin
        # Command buffers on one queue run in COMMIT order, so drawing that was
        # still open when a dispatch was committed would have run AFTER it; the
        # order has to be the order the walk emitted, which is the order the graph
        # derived. Asserted through the result.
        g = M.Graph(RG_DEV)
        a = M.Transient.Image(g, RGBA{N0f8}, (64, 64))
        scratch = M.Buffer(RG_DEV, zeros(Float32, 16))
        M.render!(g, "before", a => M.Clear((0f0, 0f0, 0f0, 1f0))) do p
            M.draw!(p, RG_PIPE, (RG_TRI,), 3)
        end
        M.dispatch!(g, rg_touch!, (scratch,), 16; name = "between")
        M.render!(g, "after", a => M.Keep) do p
            M.draw!(p, RG_PIPE, (RG_TRI,), 3)
        end
        out = readout!(g, a)
        plan = M.record!(M.Plan(g))

        M.run!(plan)
        @test Array(M.storage(scratch))[1] == 1f0
        # `Keep` on the second pass: it drew over what the first left, so the count
        # is one triangle and not two overlaid or a cleared target.
        @test greens(pixels(out)) == RG_COVERED
        M.free!(plan)
        M.free!(out)
        M.free!(scratch)
    end

    @testset "clip space is Mantle's, not the backend's" begin
        # The bug a person found by looking at a window: the scene was rendered
        # upside down. **Mantle's clip space is Vulkan's — +y is DOWN the screen**
        # — and Metal's is the other one, so the same `position` came out
        # mirrored.
        #
        # Nothing automated caught it. The pixel counts were identical, the
        # window/offscreen comparison agreed perfectly (both were equally wrong),
        # and a mirrored scene still looks like a scene. This asserts the
        # ORIENTATION rather than the count.
        g = M.Graph(RG_DEV)
        color = M.Transient.Image(g, RGBA{N0f8}, (64, 64))
        # Apex at NDC +0.9, base at −0.9. Narrow end and wide end are the marker.
        M.render!(g, "tri", color => M.Clear((0f0, 0f0, 0f0, 1f0))) do p
            M.draw!(p, RG_PIPE, (RG_TRI,), 3)
        end
        out = readout!(g, color)
        plan = M.record!(M.Plan(g))
        M.run!(plan)
        img = pixels(out)
        width(y) = count(x -> ColorTypes.green(img[x, y]) > 0.5, 1:64)

        # Row 1 of a readback is the TOP of the image. Under Mantle's convention
        # the apex (+y) belongs at the BOTTOM, so the triangle must be narrow at
        # large y and wide at small y.
        @test width(6) > 40          # base, near the top
        @test width(58) < 10         # apex, near the bottom
        @test width(6) > width(32) > width(58)

        # `clip_y` is the same transform, exposed so a shader that does the
        # viewport step BY HAND — a shadow lookup turning `sunvp * world` into a
        # texel row — applies the one the rasteriser applied. It is its own
        # inverse.
        @test M.clip_y(M.clip_y(0.37f0)) == 0.37f0
        @test abs(M.clip_y(0.37f0)) == 0.37f0
        M.free!(plan)
        M.free!(out)
    end
end
