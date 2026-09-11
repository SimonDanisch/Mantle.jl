# A recorded plan survives the storage underneath it MOVING.
#
# A recording's command buffer holds the ADDRESS of the plan's argument memory,
# and the argument memory holds the addresses of everything else — so a resource
# that moves is not a new recording, it is eight bytes per packed pointer,
# written by `cmd_update_buffer` in the next run's own submission. The table
# that says where each address landed is baked once, at `record!`, from the same
# pack that wrote the blob (`recpatch!`); a move is announced where it happens
# (`resize!` → `resource_moved!`, arena growth → `arena_moved!`), and
# `notify_move!` turns it into pending patches. Nothing is walked per run.
#
# What cannot be patched is an image: the commands name the `VkImage` and its
# view directly. An images arena growing under a recorded plan drops the
# recording instead, and the next `run!` writes it again.
#
# Both halves are pinned here from the outside: same plan object, right numbers.

using Test, Mantle, KernelAbstractions
import Lava   # the extension trigger
using ColorTypes: RGBA
using GeometryBasics: Vec4f

const KA = KernelAbstractions
const M = Mantle

@kernel function _movepatch_copy!(dst, @Const(src))
    i = @index(Global)
    @inbounds dst[i] = src[i]
end

"""out[i] = src[i], n lanes, as a recordable one-pass plan."""
function _movepatch_plan(dev, src, out, n)
    g = M.Graph(dev)
    M.compute!(g, "copy") do p
        s = M.use(p, src; read = true)
        d = M.use(p, out; write = true)
        M.dispatch!(p, _movepatch_copy!, (d, s), n)
    end
    M.record!(M.Plan(g))
end

# A chain of transients (stage k reads t[k], writes t[k+1]) so the plan has
# arena-placed storage at all; `out` ends at `nstage + 2`.
@kernel function _movepatch_bump!(dst, @Const(src))
    i = @index(Global)
    @inbounds dst[i] = src[i] + 1f0
end

function _movepatch_chainplan(dev, n, nstage)
    g = M.Graph(dev)
    seed = M.Buffer(dev, zeros(Float32, n))
    t = [M.Transient.Buffer(g, Float32, n) for _ in 1:(nstage + 1)]
    M.compute!(g, "seed") do p
        s = M.use(p, seed; read = true); d = M.use(p, t[1]; write = true)
        M.dispatch!(p, _movepatch_bump!, (d, s), n)
    end
    for k in 1:nstage
        M.compute!(g, "s$k") do p
            s = M.use(p, t[k]; read = true); d = M.use(p, t[k + 1]; write = true)
            M.dispatch!(p, _movepatch_bump!, (d, s), n)
        end
    end
    out = M.Buffer(dev, zeros(Float32, n))
    M.compute!(g, "out") do p
        s = M.use(p, t[end]; read = true); d = M.use(p, out; write = true)
        M.dispatch!(p, _movepatch_bump!, (d, s), n)
    end
    (; out, plan = M.record!(M.Plan(g)), keep = (seed, t))
end

"""Two transients alive at once — the placement that forces the arena to grow."""
function _movepatch_fatplan(dev, n)
    g = M.Graph(dev)
    seed = M.Buffer(dev, zeros(Float32, 16))
    t1 = M.Transient.Buffer(g, Float32, n)
    t2 = M.Transient.Buffer(g, Float32, n)
    M.compute!(g, "a") do p
        s = M.use(p, seed; read = true); d = M.use(p, t1; write = true)
        M.dispatch!(p, _movepatch_bump!, (d, s), 16)
    end
    M.compute!(g, "b") do p
        s = M.use(p, t1; read = true); d = M.use(p, t2; write = true)
        M.dispatch!(p, _movepatch_bump!, (d, s), 16)
    end
    (; plan = M.record!(M.Plan(g)), keep = (seed, t1, t2))
end

function _movepatch_tri_vertex()
    v = Mantle.vertex_index() - Int32(1)
    x = v == Int32(1) ? 3f0 : -1f0
    y = v == Int32(2) ? 3f0 : -1f0
    (position = Vec4f(x, y, 0f0, 1f0), color = Vec4f(0.25f0, 0.5f0, 0.75f0, 1f0))
end
_movepatch_tri_fragment(inputs) = inputs.color
const _MOVEPATCH_TRI = M.Rasterizer(; vertex = M.VertexShader(_movepatch_tri_vertex; outputs = (color = Vec4f,)),
                                      fragment = M.FragmentShader(_movepatch_tri_fragment),
                                      topology = M.TriangleList(),
                                      blend = M.Opaque(),
                                      cull = M.NoCull(),
                                      depth = M.DepthOff())

@testset "a resize!d buffer under a recorded plan is patched, not re-recorded" begin
    dev = M.Device(M.VulkanAPI())
    be = M.backend(dev)
    n = 64
    src = M.Buffer(dev, ones(Float32, n))          # capacity == n, so a grow moves it
    out = M.Buffer(dev, zeros(Float32, n))
    pl = Base.invokelatest(_movepatch_plan, dev, src, out, n)

    M.run!(pl)                                      # recorded at build
    KA.synchronize(be)
    @test Array(M.storage(out)) == ones(Float32, n)
    rec = pl.recording
    @test rec !== nothing

    # Move the store out from under the recording, then change the CONTENTS, so
    # a run that still read the old region reads the wrong numbers, not a copy
    # of the right ones.
    M.resize!(src, 4n)
    copyto!(M.storage(src), fill(2f0, 4n))
    M.run!(pl)
    KA.synchronize(be)

    @test pl.recording === rec                      # patched: the same recording ran
    @test Array(M.storage(out)) == fill(2f0, n)     # …and it read the NEW store
    M.free!(pl)
end

@testset "a buffers arena growing under a recorded plan patches it" begin
    dev = M.Device(M.VulkanAPI())
    be = M.backend(dev)
    n = 250_000
    small = Base.invokelatest(_movepatch_chainplan, dev, n, 3)
    M.run!(small.plan)                               # recorded at build
    KA.synchronize(be)
    @test Array(M.storage(small.out)) == fill(Float32(3 + 2), n)
    rec = small.plan.recording
    @test rec !== nothing

    # Grow the shared buffers arena underneath the recording: `big` needs far
    # more than `small`'s peak, so its `Plan` re-reserves the arena.
    big = Base.invokelatest(_movepatch_fatplan, dev, 8_000_000)
    M.run!(big.plan)
    KA.synchronize(be)

    # The same recording still answers — patched to the arena's new base.
    M.run!(small.plan)
    KA.synchronize(be)
    @test small.plan.recording === rec
    @test Array(M.storage(small.out)) == fill(Float32(3 + 2), n)
    M.free!(small.plan)
    M.free!(big.plan)
end

@testset "an images arena growing under a recorded plan re-records it" begin
    dev = M.Device(M.VulkanAPI())
    be = M.backend(dev)
    N = 64
    g = M.Graph(dev)
    img = M.Transient.Image(g, RGBA{Float16}, (N, N))
    out = M.Buffer(dev, zeros(RGBA{Float16}, N * N))
    M.render!(g, "tri", img => M.Clear((0f0, 0f0, 0f0, 1f0))) do p
        M.draw!(p, _MOVEPATCH_TRI, (), 3)
    end
    M.copy!(g, "read", out, img)
    pl = M.record!(M.Plan(g))
    M.run!(pl)
    KA.synchronize(be)
    @test pl.recording !== nothing
    px = Array(M.storage(out))
    @test all(p -> Float32(p.r) ≈ 0.25f0 && Float32(p.g) ≈ 0.5f0, px)

    # Grow the images arena out from under the recording. The commands name the
    # old VkImage and view — nothing to patch — so the recording is DROPPED…
    g2 = M.Graph(dev)
    big_img = M.Transient.Image(g2, RGBA{Float16}, (4096, 4096))
    M.render!(g2, "big", big_img => M.Clear((0f0, 0f0, 0f0, 1f0))) do p
        M.draw!(p, _MOVEPATCH_TRI, (), 3)
    end
    pl2 = M.record!(M.Plan(g2))
    @test pl.recording === nothing

    # …and the next run writes it again, against the new placement.
    M.run!(pl)
    KA.synchronize(be)
    @test pl.recording !== nothing
    @test Array(M.storage(out)) == px
    M.free!(pl)
    M.free!(pl2)
end
