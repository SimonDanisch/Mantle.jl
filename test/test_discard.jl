# `discard()` is KernelInterface's, and it throws a fragment away.
#
# It was never declared there. Lava defined it by overriding
# `KernelInterface.discard() = …`, and a qualified definition CREATES the
# binding in the module it names — so `discard` existed whenever Lava was
# loaded and not otherwise. On a machine with no Vulkan, Mantle's
# `using KernelInterface: …, discard` found nothing (a warning, not an error, in
# `using X: name`), RayMakie imported the same nothing, and every overlay
# fragment stage that calls `discard()` failed to compile with "unsupported use
# of an undefined name".
#
# Two halves. The declaration needs no GPU and runs everywhere; the draw runs on
# every backend with graphics, with the same assertions.

using Test, Mantle
import KernelInterface
using Mantle: Vec4f
using ColorTypes: BGRA
using ColorTypes.FixedPointNumbers: N0f8

@testset "discard is declared by KernelInterface" begin
    @test isdefined(KernelInterface, :discard)
    # Mantle re-exports KernelInterface's function, not a copy of its own.
    @test Mantle.discard === KernelInterface.discard
    # On the host it SAYS something, like every other shader builtin, rather
    # than being a MethodError. Before the declaration, the only method on a
    # Vulkan machine was Lava's device overlay, so this was a MethodError there
    # and an UndefVarError everywhere else.
    err = try
        KernelInterface.discard()
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("discard", err.msg)
end

# A full-screen triangle; `vertex_index` counts from one.
function dc_vertex()
    v = vertex_index()
    x = v == Int32(2) ? 3f0 : -1f0
    y = v == Int32(3) ? 3f0 : -1f0
    return (position = Vec4f(x, y, 0f0, 1f0),)
end

# Everything left of x = 16 is discarded, so it keeps the clear colour.
# Execution continues past `discard()` and a colour is still returned; it is
# simply not written.
function dc_fragment(inputs)
    frag_coord_x() < 16f0 && discard()
    return Vec4f(0f0, 1f0, 0f0, 1f0)
end

const DC_PIPE = Mantle.Rasterizer(; vertex = Mantle.VertexShader(dc_vertex),
                                    fragment = Mantle.FragmentShader(dc_fragment),
                                    topology = Mantle.TriangleList(),
                                    blend = Mantle.Opaque(),
                                    cull = Mantle.NoCull(),
                                    depth = Mantle.DepthOff())

@testset "a discarded fragment keeps what was there" begin
    be = Mantle.defaultbackend()
    if !Mantle.supports_graphics(be)
        @info "no graphics pipeline on this backend; skipping"
        return
    end
    dev = Mantle.todevice(be)
    W, H = 32, 32
    g = Mantle.Graph(dev)
    img = Mantle.Transient.Image(g, BGRA{N0f8}, (W, H))
    Mantle.render!(g, "discard", img => Mantle.Clear((1f0, 0f0, 0f0, 1f0))) do p
        Mantle.draw!(p, DC_PIPE, (), 3)
    end
    out = Mantle.Transient.Buffer(g, BGRA{N0f8}, W * H)
    Mantle.copy!(g, "read", out, img)
    Mantle.run!(Mantle.Plan(g))
    px = reshape(Array(Mantle.storage(out)), W, H)

    red, green = BGRA{N0f8}(1, 0, 0, 1), BGRA{N0f8}(0, 1, 0, 1)
    # Left and right by COLUMN, which no backend's y convention can swap.
    @test all(==(red), @view px[1:16, :])
    @test all(==(green), @view px[17:32, :])
end
