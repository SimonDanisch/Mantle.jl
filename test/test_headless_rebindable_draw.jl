# A headless plan whose draw reads a `DrawBinding` cannot be recorded: the host
# rewrites the cell between runs (`recordable`), so every `run!` walks the plan
# into a one-shot instead. Two things were wrong with that walk on Vulkan, and
# RayMakie's composited frame, which is exactly this plan, hit both:
#
#   * `closerun!` asserted the plan held recording parts, and an unrecorded plan
#     holds `nothing`, so every run was a `TypeError`.
#   * the backend's `emitdraw!` drew with the count, index buffer, instances and
#     texture table the plan was compiled with, where core's interpreted walk
#     reads them from the binding. The arguments did follow the cell, so a
#     rebound colour changed and a rebound count did not.
#
# The same assertions on every backend with graphics.

using Test, Mantle
using Mantle: Vec4f
using ColorTypes: BGRA
using ColorTypes.FixedPointNumbers: N0f8

# A full-screen triangle; `vertex_index` counts from one.
function hr_vertex(level::Float32)
    v = vertex_index()
    x = v == Int32(2) ? 3f0 : -1f0
    y = v == Int32(3) ? 3f0 : -1f0
    return (position = Vec4f(x, y, 0f0, 1f0),)
end

hr_fragment(inputs, level::Float32) = Vec4f(0f0, level, 0f0, 1f0)

const HR_PIPE = Mantle.Rasterizer(; vertex = Mantle.VertexShader(hr_vertex),
                                    fragment = Mantle.FragmentShader(hr_fragment),
                                    topology = Mantle.TriangleList(),
                                    blend = Mantle.Opaque(),
                                    cull = Mantle.NoCull(),
                                    depth = Mantle.DepthOff())

@testset "a headless plan with a rebindable draw runs unrecorded" begin
    be = Mantle.defaultbackend()
    if !Mantle.supports_graphics(be)
        @info "no graphics pipeline on this backend; skipping"
        @test_skip false
        return
    end
    dev = Mantle.todevice(be)
    W, H = 16, 16
    cell = Mantle.DrawBinding(dev, (1f0,), 3)
    g = Mantle.Graph(dev)
    img = Mantle.Transient.Image(g, BGRA{N0f8}, (W, H))
    Mantle.render!(g, "draw", img => Mantle.Clear((1f0, 0f0, 0f0, 1f0))) do p
        Mantle.draw!(p, HR_PIPE, cell; frag_args = Mantle.drawargs(cell))
    end
    out = Mantle.Transient.Buffer(g, BGRA{N0f8}, W * H)
    Mantle.copy!(g, "read", out, img)
    plan = Mantle.Plan(g)
    @test !Mantle.recordable(plan)
    pixels() = unique(Array(Mantle.storage(out)))

    Mantle.run!(plan)
    @test pixels() == [BGRA{N0f8}(0, 1, 0, 1)]
    # The argument, read on the next run.
    Mantle.rebind!(cell, dev, (0.2f0,), 3)
    Mantle.run!(plan)
    @test pixels() == [BGRA{N0f8}(0, 0.2, 0, 1)]
    # And the count: no vertices, so the clear colour, at a level the draw would
    # have changed.
    Mantle.rebind!(cell, dev, (1f0,), 0)
    Mantle.run!(plan)
    @test pixels() == [BGRA{N0f8}(1, 0, 0, 1)]
    Mantle.free!(plan)
end
