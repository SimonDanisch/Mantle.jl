# Headless rendering records on graphics devices; stores preserve addresses
# and replay on every recording backend.
using Test, Mantle, KernelAbstractions
using GeometryBasics: Vec4f
using ColorTypes: RGBA
include(joinpath(@__DIR__, "testbackend.jl"))
const KA = KernelAbstractions
const M = Mantle

function tri_vertex()
    v = Mantle.vertex_index() - Int32(1)
    x = v == Int32(1) ? 3f0 : -1f0
    y = v == Int32(2) ? 3f0 : -1f0
    (position = Vec4f(x, y, 0f0, 1f0), color = Vec4f(0.25f0, 0.5f0, 0.75f0, 1f0))
end
tri_fragment(inputs) = inputs.color
const TRI = M.Rasterizer(; vertex = M.VertexShader(tri_vertex; outputs = (color = Vec4f,)),
                           fragment = M.FragmentShader(tri_fragment),
                           topology = M.TriangleList(),
                           blend = M.Opaque(),
                           cull = M.NoCull(),
                           depth = M.DepthOff())

@kernel function s2g_copy!(dst, @Const(src))
    i = @index(Global)
    @inbounds dst[i] = src[i]
end

@testset "a headless render pass records and runs" begin
    dev = M.Device(TESTBACKEND)
    if !M.supports_graphics(dev)
        @test_skip M.supports_graphics(dev)
    else
        N = 64
        g = M.Graph(dev)
        img = M.Transient.Image(g, RGBA{Float16}, (N, N))
        out = M.Buffer(dev, zeros(RGBA{Float16}, N * N))
        M.render!(g, "tri", img => M.Clear((0f0, 0f0, 0f0, 1f0))) do p
            M.draw!(p, TRI, (), 3)
        end
        M.copy!(g, "read", out, img)
        pl = M.Plan(g)
        M.record!(pl)
        @test M.recorded(pl) == M.recordable(dev, pl)
        M.run!(pl)
        KA.synchronize(dev)
        px = Array(M.storage(out))
        @test all(p -> Float32(p.r) ≈ 0.25f0 && Float32(p.g) ≈ 0.5f0, px)
        fill!(M.storage(out), RGBA{Float16}(0, 0, 0, 0))
        KA.synchronize(dev)
        M.run!(pl)
        KA.synchronize(dev)
        @test Array(M.storage(out)) == px
        M.free!(pl)
    end
end

@testset "a whole-buffer store writes in place, so its plan records" begin
    dev = M.Device(TESTBACKEND)
    n = 128
    g = M.Graph(dev)
    src = M.Buffer(dev, zeros(Float32, n))
    dst = M.Buffer(dev, zeros(Float32, n))
    M.dispatch!(g, s2g_copy!, (dst,
                                   src), n; name = "copy")
    pl = M.record!(M.Plan(g))
    @test Mantle.recordable(pl)
    store = src.store
    for v in (3f0, 7f0, 11f0, 13f0)
        src[:] = fill(v, n)
        M.run!(pl)
        KA.synchronize(dev)
        @test all(==(v), Array(M.storage(dst)))
    end
    @test M.recorded(pl)
    @test src.store === store
    M.free!(pl)
end

@testset "a ranged store writes in place, so its plan records" begin
    dev = M.Device(TESTBACKEND)
    n = 128
    g = M.Graph(dev)
    src = M.Buffer(dev, zeros(Float32, n))
    dst = M.Buffer(dev, zeros(Float32, n))
    M.dispatch!(g, s2g_copy!, (dst,
                                   src), n; name = "copy")
    pl = M.Plan(g)
    @test Mantle.recordable(pl)
    M.record!(pl)
    for v in (3f0, 7f0, 11f0, 13f0)
        src[1:n] = fill(v, n)
        M.run!(pl)
        KA.synchronize(dev)
        @test all(==(v), Array(M.storage(dst)))
    end
    M.free!(pl)
end

@testset "two ranged stores before one run both land, in order" begin
    dev = M.Device(TESTBACKEND)
    n = 128
    g = M.Graph(dev)
    src = M.Buffer(dev, zeros(Float32, n))
    dst = M.Buffer(dev, zeros(Float32, n))
    M.dispatch!(g, s2g_copy!, (dst,
                                   src), n; name = "copy")
    pl = M.Plan(g)
    M.record!(pl)
    src[1:64] = fill(3f0, 64)
    src[33:96] = fill(7f0, 64)
    @test M.isdirty(src)
    M.run!(pl)
    KA.synchronize(dev)
    got = Array(M.storage(dst))
    @test all(==(3f0), got[1:32])
    @test all(==(7f0), got[33:96])
    @test all(==(0f0), got[97:end])
    @test !M.isdirty(src)
    M.free!(pl)
end
