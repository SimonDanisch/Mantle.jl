# A texture made from DEVICE data samples the same as one made from host data,
# and an existing texture takes new texels in place.
#
# Three things were wrong on Metal, and each left a picture with no error:
#
#   * `Texture2D` tested `data isa DenseMatrix` to decide it had a host pointer.
#     A GPU array IS a `DenseMatrix` (`AbstractGPUArray <: DenseArray`) whose
#     `pointer` is DEVICE memory, and the CPU-side upload read texels from it:
#     the texture came out zero, and RayMakie's image shader discards a zero
#     alpha, so an `image!` of a device array drew nothing at all.
#   * The upload ran on a command buffer opened BESIDE the device's open batch,
#     so it could run before the kernel that had just written the array — which
#     is exactly where RayMakie's device texels come from (a broadcast).
#   * There was no `upload_texture_data!`, and `eltype` of a texture was `Any`,
#     so nothing ever "fit" and every animated texture was rebuilt from scratch.
#
# Portable: gated on `supports_graphics`, the same assertions everywhere.

using Test, Mantle
using Mantle: Vec2f, Vec4f
using ColorTypes: BGRA
using ColorTypes.FixedPointNumbers: N0f8

const TU_N = 16

# A full-screen triangle carrying uv; the fragment returns the sampled texel.
function tu_vertex()
    v = vertex_index()
    x = v == Int32(2) ? 3f0 : -1f0
    y = v == Int32(3) ? 3f0 : -1f0
    return (position = Vec4f(x, y, 0f0, 1f0),
            uv = Vec2f((x + 1f0) * 0.5f0, (y + 1f0) * 0.5f0))
end
function tu_fragment(inputs)
    u = inputs.uv[1]; v = inputs.uv[2]
    return Vec4f(sample_texture_2d(UInt32(0), u, v, UInt32(0)),
                 sample_texture_2d(UInt32(0), u, v, UInt32(1)),
                 sample_texture_2d(UInt32(0), u, v, UInt32(2)), 1f0)
end
const TU_PIPE = Mantle.GraphicsPipeline(;
    vertex = Mantle.VertexShader(tu_vertex; outputs = (uv = Vec2f,)),
    fragment = Mantle.FragmentShader(tu_fragment; textures = 1),
    blend = Mantle.Opaque(), topology = Mantle.TriangleList(),
    cull = Mantle.NoCull(), depth = Mantle.DepthOff())

# What `tex` looks like, drawn through the graph at one pixel per texel.
function tu_draw(dev, be, tex)
    b = Mantle.bind_textures([Mantle.SampledTexture(tex, Mantle.Sampler(be; filter = :nearest,
                                                                           wrap = :clamp))])
    g = Mantle.Graph(dev)
    img = Mantle.Transient.Image(g, BGRA{N0f8}, (TU_N, TU_N))
    Mantle.render!(g, "sample", img => Mantle.Clear((0f0, 0f0, 0f0, 1f0))) do p
        Mantle.draw!(p, TU_PIPE, (), 3; bindings = b)
    end
    out = Mantle.Transient.Buffer(g, BGRA{N0f8}, TU_N * TU_N)
    Mantle.copy!(g, "read", out, img)
    Mantle.run!(Mantle.Plan(g))
    return reshape(Array(Mantle.storage(out)), TU_N, TU_N)
end

# A pattern that differs along BOTH axes, so a zero texture, a transpose and a
# stale upload all show.
tu_texels(k) = [(Float32(x) / TU_N, Float32(y) / TU_N, Float32(k), 1f0) for x in 1:TU_N, y in 1:TU_N]

@testset "a texture from device data samples as one from host data" begin
    be = Mantle.defaultbackend()
    if !Mantle.supports_graphics(be)
        @info "no graphics pipeline on this backend; skipping"
        return
    end
    dev = Mantle.todevice(be)
    host = tu_texels(0.5f0)

    th = Mantle.Texture2D(be, host)
    # `eltype` is the texture's, asked the way RayMakie asks whether new
    # texels fit: it was `Any` on Metal.
    @test eltype(th) == eltype(host)
    @test size(th) == size(host)
    ref = tu_draw(dev, be, th)
    @test any(p -> p.r > 0 || p.g > 0, ref)          # the reference itself drew

    # Device data that a KERNEL has just written, uploaded without anything in
    # between: the ordering bug needs both halves.
    d = Mantle.devicearray(be, tu_texels(0f0))
    d .= Mantle.devicearray(be, host)                  # a broadcast writes it
    td = Mantle.Texture2D(be, d)
    @test tu_draw(dev, be, td) == ref
end

@testset "upload_texture_data! writes into the existing texture" begin
    be = Mantle.defaultbackend()
    Mantle.supports_graphics(be) || return
    dev = Mantle.todevice(be)
    a, b, c = tu_texels(0f0), tu_texels(1f0), tu_texels(0.25f0)
    ref_b = tu_draw(dev, be, Mantle.Texture2D(be, b))
    ref_c = tu_draw(dev, be, Mantle.Texture2D(be, c))
    @test ref_b != ref_c

    t = Mantle.Texture2D(be, a)
    Mantle.upload_texture_data!(t, b)                  # from the host
    @test tu_draw(dev, be, t) == ref_b
    dc = Mantle.devicearray(be, a)
    dc .= Mantle.devicearray(be, c)                    # from a kernel's output
    Mantle.upload_texture_data!(t, dc)
    @test tu_draw(dev, be, t) == ref_c

    # A size mismatch is a new texture, not an upload.
    @test_throws DimensionMismatch Mantle.upload_texture_data!(t, tu_texels(0f0)[1:8, :])
end
