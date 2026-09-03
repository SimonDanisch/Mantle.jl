# Which plans can be recorded, and that both answers are right.
#
# `run!` records on the first run of any plan it can record, so "can it" has to be
# decided somewhere and be right. Two things say no today, and each is a step of
# this refactor that has not happened yet:
#
#   * a SURFACE — a swapchain image is a different image every frame and a
#     recording names one (step 4 renders into a fixed offscreen target instead);
#   * a whole-buffer `Update` — it lands by RENAMING, because writing the contents
#     in place would overwrite bytes the previous run may still be reading, and a
#     recording holds the address it was written with (step 3 puts a per-run value
#     inline in the command buffer, so nothing has to move).
#
# A RANGED update always writes in place, so its plan records. That difference is
# the whole of `renameable`, and it is worth pinning from the outside: the two
# routes produce the same numbers and differ only in whether a store moves.
#
# The render pass here is also the only coverage the emitter's `begin_pass!` /
# `set_viewport!` / `draw_in_pass!` / `end_pass!` have. `test_window.jl` is where
# they belong and it cannot reach them: it aborts in its first testset compiling
# `scatter_vertex` (`KernelError: kernel returns a value of type Any`, identical at
# HEAD — verified by stashing), which takes every testset after it down too.

using Test, Mantle, KernelAbstractions
import Lava, Vulkan   # the extension trigger; `import`, since Lava exports names Mantle does
using GeometryBasics: Vec4f
using ColorTypes: RGBA
# `runtests.jl` binds these too, to the same values — a `const` re-bound to what
# it already holds is not a redefinition.
const KA = KernelAbstractions
const M = Mantle
const MVE = Base.get_extension(Mantle, :MantleVulkanExt)

function tri_vertex()
    v = Mantle.vertex_index() - Int32(1)
    x = v == Int32(1) ? 3f0 : -1f0
    y = v == Int32(2) ? 3f0 : -1f0
    (position = Vec4f(x, y, 0f0, 1f0), color = Vec4f(0.25f0, 0.5f0, 0.75f0, 1f0))
end
tri_fragment(inputs) = inputs.color
# `Mantle.vertex_index`, QUALIFIED: `runtests.jl` says `using Mantle` and this
# file is included into the same Main, where Lava may also be `using`ed — and two
# modules exporting one name leaves the bare form resolving to neither.
const TRI = M.Rasterizer(vertex = tri_vertex, fragment = tri_fragment,
                         varyings = (color = Vec4f,), topology = M.TriangleList(),
                         blend = M.Opaque(), cull = M.NoCull(), depth = M.DepthOff())

@kernel function s2g_copy!(dst, @Const(src))
    i = @index(Global)
    @inbounds dst[i] = src[i]
end

@testset "a headless render pass records and runs" begin
    dev = M.Device(M.VulkanAPI())
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
    @test M.recorded(pl)
    M.run!(pl)
    KA.synchronize(M.backend(dev))
    px = Array(M.storage(out))
    @test all(p -> Float32(p.r) ≈ 0.25f0 && Float32(p.g) ≈ 0.5f0, px)
    fill!(M.storage(out), RGBA{Float16}(0, 0, 0, 0))
    KA.synchronize(M.backend(dev))
    M.run!(pl)
    KA.synchronize(M.backend(dev))
    @test Array(M.storage(out)) == px
    M.free!(pl)
end

@testset "a renaming Update keeps its plan out of a recording" begin
    dev = M.Device(M.VulkanAPI())
    n = 128
    g = M.Graph(dev)
    src = M.Buffer(dev, zeros(Float32, n))
    dst = M.Buffer(dev, zeros(Float32, n))
    ref = M.Update(g, src)
    M.compute!(g, "copy") do p
        M.dispatch!(p, s2g_copy!, (M.use(p, dst; write = true),
                                   M.use(p, src; read = true)), n)
    end
    pl = M.Plan(g)
    # A whole-buffer Update renames, so this plan cannot be recorded — and says so.
    @test !MVE.recordable(pl)
    @test_throws ArgumentError M.record!(pl)
    for v in (3f0, 7f0, 11f0, 13f0)
        ref(fill(v, n))
        M.run!(pl)
        KA.synchronize(M.backend(dev))
        @test all(==(v), Array(M.storage(dst)))
    end
    @test !M.recorded(pl)
    M.free!(pl)
end

@testset "a ranged Update writes in place, so its plan records" begin
    dev = M.Device(M.VulkanAPI())
    n = 128
    g = M.Graph(dev)
    src = M.Buffer(dev, zeros(Float32, n))
    dst = M.Buffer(dev, zeros(Float32, n))
    ref = M.Update(g, src; range = 1:n)
    M.compute!(g, "copy") do p
        M.dispatch!(p, s2g_copy!, (M.use(p, dst; write = true),
                                   M.use(p, src; read = true)), n)
    end
    pl = M.Plan(g)
    @test MVE.recordable(pl)
    M.record!(pl)
    for v in (3f0, 7f0, 11f0, 13f0)
        ref(fill(v, n))
        M.run!(pl)
        KA.synchronize(M.backend(dev))
        @test all(==(v), Array(M.storage(dst)))
    end
    M.free!(pl)
end

# The SURFACE half of `recordable` is not asserted here: it needs a real window,
# and a window in the main suite is what `test_window.jl` runs in its own process
# with a deadline for. `record!`'s refusal names it, and step 4 removes it.
