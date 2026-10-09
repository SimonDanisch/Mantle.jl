"""
`Mantle.indexbuffer` reaches this backend's allocation, and what it returns draws.

Vulkan refuses a `vkCmdBindIndexBuffer` whose buffer lacks
`BUFFER_USAGE_INDEX_BUFFER_BIT`, and the bit has to be asked for at allocation —
which is why the portable `indexbuffer` exists at all and why that backend
overrides it. Two ways to get it wrong, and both have happened:

  * the override never fires, because a caller handed in a KA backend and the
    method is on the DEVICE. `indexbuffer(::Backend, …)` normalises first for
    exactly this reason;
  * the override fires and calls an allocation that does not exist. It called
    `alloc_index_buffer(indices)` while that function took `(backend, indices)`,
    so every indexed draw in RayMakie was a `MethodError` — six call sites, none
    of them covered.

Both spellings are checked here because both are what a caller writes, and each
is DRAWN with: an indexed draw is the only end-to-end statement that the buffer
really is an index buffer.
"""

using Test, Mantle, KernelAbstractions
using GeometryBasics: Vec4f
using ColorTypes: RGBA
using ColorTypes.FixedPointNumbers: N0f8
include(joinpath(@__DIR__, "testbackend.jl"))

# A unit quad in the lower-left quarter of clip space. Which of its four corners
# a triangle uses is the index buffer's to say, so the lit area reads off as how
# much of the quad the indices named.
function ib_quad_vertex(_::Int32)
    v = vertex_index() - Int32(1)
    fx = (v == Int32(1) || v == Int32(2)) ? 1f0 : 0f0
    fy = (v == Int32(2) || v == Int32(3)) ? 1f0 : 0f0
    (position = Vec4f(-1f0 + fx, -1f0 + fy, 0.5f0, 1f0),
     color = Vec4f(1, 1, 1, 1))
end
ib_fragment(inputs) = inputs.color
const IB_QUAD = Rasterizer(; vertex = VertexShader(ib_quad_vertex; outputs = (color = Vec4f,)),
                             fragment = FragmentShader(ib_fragment),
                             topology = TriangleList(), blend = Opaque(),
                             cull = NoCull(), depth = DepthOff())

"""Pixels lit by one indexed draw of the quad with `ib`, `n` indices of it."""
function litcount(dev, ib, n; N = 64)
    g = Mantle.Graph(dev)
    img = Mantle.Transient.Image(g, RGBA{N0f8}, (N, N))
    out = Mantle.Transient.Buffer(g, UInt32, N * N)
    Mantle.render!(g, "indexed", img => Mantle.Clear((0f0, 0f0, 0f0, 1f0));
                   viewport = (0, 0, N, N)) do p
        Mantle.draw!(p, IB_QUAD, (Int32(0),), n; indices = ib)
    end
    Mantle.copy!(g, "read", out, img)
    plan = Mantle.record!(Mantle.Plan(g))
    Mantle.run!(plan)
    KernelAbstractions.synchronize(Mantle.backend(dev))
    lit = count(!=(0xff000000), Array(Mantle.storage(out)))
    Mantle.free!(plan)
    return lit
end

@testset "indexbuffer allocates an index buffer, from either spelling" begin
    dev = Mantle.Device(TESTBACKEND)
    N = 64
    quad = UInt32[0, 1, 2, 0, 2, 3]

    for from in (TESTBACKEND, dev)
        ib = Mantle.indexbuffer(from, quad)
        # An ARRAY, which is what both backends answer with and what a caller
        # keeps beside its vertex arrays — not a `Mantle.Buffer`.
        @test ib isa AbstractVector{UInt32}
        @test length(ib) == length(quad)
        @test Array(ib) == quad

        if Mantle.supports_graphics(dev)
            # The whole quad: exactly the quarter of clip space it covers.
            @test litcount(dev, ib, length(quad); N) == (N ÷ 2)^2
            # HALF the indices is half the quad, so `count` is an index count. Not
            # exact: the two triangles share a diagonal and both fill rules put
            # those pixels somewhere.
            half = litcount(dev, Mantle.indexbuffer(from, quad[1:3]), 3; N)
            @test 0.4 * (N ÷ 2)^2 < half < 0.6 * (N ÷ 2)^2
        else
            @info "no graphics pipeline on this device; the draw is not checked"
        end
    end
end
