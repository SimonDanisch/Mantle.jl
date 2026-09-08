# Which plans can be recorded, and that both answers are right.
#
# `run!` records on the first run of any plan it can record, so "can it" has to be
# decided somewhere and be right. ONE thing says no now — a SURFACE, since a
# swapchain image is a different image every frame and a recording names one, and
# step 4 renders into a fixed offscreen target instead.
#
# There were two. A whole-buffer `Update` used to land by RENAMING, which a
# recording cannot follow because it holds the address it was written with, so a
# ranged update recorded and a whole-buffer one did not. Both write in place now,
# and the two testsets below pin that from the outside: same numbers, same
# recording, and the store the plan was compiled against is still the store.
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
# `KA` and `M` are `const` in whichever test file gets there first, and a `const`
# re-bound to what it already holds is not a redefinition.
#
# `MVE` is bound by `runtests.jl` — a `const` now, because kernels read through
# it (`MVE.GEMM_TILE`) and a non-const Main global is a type-unstable global
# access that GPUCompiler rejects. Files bind it themselves only when the
# harness has not, which is what lets them work standalone too.
const KA = KernelAbstractions
const M = Mantle
@isdefined(MVE) || (MVE = Base.get_extension(Mantle, :MantleVulkanExt))

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

# A whole-buffer store used to RENAME: the contents landed in a fresh store
# and the resource was pointed at it, which a recording cannot follow, because it
# holds the address it was written with. That made this plan unrecordable, and it
# was one of the two reasons `record!` could refuse one.
#
# Both routes are in place now — see `emitstore!` — so the plan records and
# every value reaches the recording that reads it.
@testset "a whole-buffer store writes in place, so its plan records" begin
    dev = M.Device(M.VulkanAPI())
    n = 128
    g = M.Graph(dev)
    src = M.Buffer(dev, zeros(Float32, n))
    dst = M.Buffer(dev, zeros(Float32, n))
    M.compute!(g, "copy") do p
        M.dispatch!(p, s2g_copy!, (M.use(p, dst; write = true),
                                   M.use(p, src; read = true)), n)
    end
    pl = M.record!(M.Plan(g))
    @test MVE.recordable(pl)
    # The store the plan was compiled against, so a rename would show up as a
    # different object rather than as a wrong number.
    store = src.store
    for v in (3f0, 7f0, 11f0, 13f0)
        src[:] = fill(v, n)
        M.run!(pl)
        KA.synchronize(M.backend(dev))
        @test all(==(v), Array(M.storage(dst)))
    end
    @test M.recorded(pl)
    @test src.store === store
    M.free!(pl)
end

@testset "a ranged store writes in place, so its plan records" begin
    dev = M.Device(M.VulkanAPI())
    n = 128
    g = M.Graph(dev)
    src = M.Buffer(dev, zeros(Float32, n))
    dst = M.Buffer(dev, zeros(Float32, n))
    M.compute!(g, "copy") do p
        M.dispatch!(p, s2g_copy!, (M.use(p, dst; write = true),
                                   M.use(p, src; read = true)), n)
    end
    pl = M.Plan(g)
    @test MVE.recordable(pl)
    M.record!(pl)
    for v in (3f0, 7f0, 11f0, 13f0)
        src[1:n] = fill(v, n)
        M.run!(pl)
        KA.synchronize(M.backend(dev))
        @test all(==(v), Array(M.storage(dst)))
    end
    M.free!(pl)
end

# Two stores to different ranges before one run are two writes, in order: a
# `Buffer` keeps a LIST of what is waiting, where a `GPURef` keeps the last
# value. The second range overlaps the first, so the order is visible.
@testset "two ranged stores before one run both land, in order" begin
    dev = M.Device(M.VulkanAPI())
    n = 128
    g = M.Graph(dev)
    src = M.Buffer(dev, zeros(Float32, n))
    dst = M.Buffer(dev, zeros(Float32, n))
    M.compute!(g, "copy") do p
        M.dispatch!(p, s2g_copy!, (M.use(p, dst; write = true),
                                   M.use(p, src; read = true)), n)
    end
    pl = M.Plan(g)
    M.record!(pl)
    src[1:64] = fill(3f0, 64)
    src[33:96] = fill(7f0, 64)
    @test M.isdirty(src)
    M.run!(pl)
    KA.synchronize(M.backend(dev))
    got = Array(M.storage(dst))
    @test all(==(3f0), got[1:32])
    @test all(==(7f0), got[33:96])
    @test all(==(0f0), got[97:end])
    @test !M.isdirty(src)
    M.free!(pl)
end

# The SURFACE half of `recordable` is not asserted here: it needs a real window,
# and a window in the main suite is what `test_window.jl` runs in its own process
# with a deadline for. `record!`'s refusal names it, and step 4 removes it.
