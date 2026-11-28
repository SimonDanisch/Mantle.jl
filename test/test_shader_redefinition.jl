# A shader is compiled from the code its functions have NOW.
#
# The graphics shader cache was keyed on the stage function and its argument
# types, for the life of the device. A method redefined afterwards (Revise, or
# `@eval` as here) changed nothing on screen: the first SPIR-V was drawn for
# the rest of the session, even after the edit was inlined into the stage.

using Test, Mantle
using Mantle: Vec4f
using ColorTypes: BGRA
using ColorTypes.FixedPointNumbers: N0f8

# A full-screen triangle; `vertex_index` counts from one.
function rs_vertex()
    v = vertex_index()
    x = v == Int32(2) ? 3f0 : -1f0
    y = v == Int32(3) ? 3f0 : -1f0
    return (position = Vec4f(x, y, 0f0, 1f0),)
end

rs_colour() = Vec4f(0f0, 1f0, 0f0, 1f0)
# What changes is a callee, not the stage function itself: the edit reaches the
# shader only through inlining.
rs_fragment(inputs) = rs_colour()

const RS_PIPE = Mantle.Rasterizer(; vertex = Mantle.VertexShader(rs_vertex),
                                    fragment = Mantle.FragmentShader(rs_fragment),
                                    topology = Mantle.TriangleList(),
                                    blend = Mantle.Opaque(),
                                    cull = Mantle.NoCull(),
                                    depth = Mantle.DepthOff())

"""Draw `RS_PIPE` into a fresh image and return its centre pixel."""
function rs_draw(dev; W = 8, H = 8)
    g = Mantle.Graph(dev)
    img = Mantle.Transient.Image(g, BGRA{N0f8}, (W, H))
    Mantle.render!(g, "redefinition", img => Mantle.Clear((0f0, 0f0, 0f0, 1f0))) do p
        Mantle.draw!(p, RS_PIPE, (), 3)
    end
    out = Mantle.Transient.Buffer(g, BGRA{N0f8}, W * H)
    Mantle.copy!(g, "read", out, img)
    Mantle.run!(Mantle.record!(Mantle.Plan(g)))
    return reshape(Array(Mantle.storage(out)), W, H)[W ÷ 2, H ÷ 2]
end

@testset "a redefined shader callee is drawn" begin
    be = Mantle.defaultbackend()
    if !Mantle.supports_graphics(be)
        @info "no graphics pipeline on this backend; skipping"
        return
    end
    dev = Mantle.todevice(be)
    @test rs_draw(dev) == BGRA{N0f8}(0, 1, 0, 1)
    @eval rs_colour() = Vec4f(0f0, 0f0, 1f0, 1f0)
    # In the world that has the new method; this block's own world predates it.
    @test Base.invokelatest(rs_draw, dev) == BGRA{N0f8}(0, 0, 1, 1)
end
