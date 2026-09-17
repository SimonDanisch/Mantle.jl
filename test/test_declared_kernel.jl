# A dispatch's kernel is a plain Julia function, on every backend.
#
# `dispatch!` requires no `@kernel`, which as an unstated rule is just "the path
# that compiles one is the only path there is". `kernelfor(k, nothing, backend)`
# is `k(backend)`, which is what KernelAbstractions' macro generates — a
# CONSTRUCTOR that answers a backend and returns a launchable object. Hand it a
# kernel written against `KernelInterface`'s intrinsics instead and the call is a
# `MethodError: no method matching ew2!(::CPU)`, four frames below the
# `dispatch!` that caused it.
#
# The predicate that separates the two is `buildskernel`, and it is ONE
# predicate in core on purpose. Three of them is what it looks like otherwise:
# `hasmethod` in `compile_dispatch`, the same question as `kifunction`,
# and core's own `bake` did not ask at all, so the backend a graph fell back to
# was the one that could not run the kernel. Three copies of a decision about
# what a dispatch MEANS is the shape of thing `docs/mantle-owns-it.md` is about:
# the backends deliver verbs, they do not each decide what a kernel is.
#
# Portable, via `foreachbackend`, and that matters more here than usual. Every
# assertion below passed on one backend while failing on another at some point
# in this file's first hour: the values on Lava (no KI path in
# `compile_dispatch`), the same values on the host (no `KI.kernel_function` for
# `KernelAbstractions.CPU` — still true, and the test states it as a named
# refusal rather than skipping), and `peakbytes` on ROCm (a transient with no
# interval, because a `range`d usage interns as the `BufferRange` and the
# parent's `touch!` was discarded).

using Test, Mantle
using KernelAbstractions
import KernelInterface as KI

const M = Mantle
const BE = Main.MANTLE_TEST_BACKEND

# Two things about the intrinsics that cost a compile each to learn:
# `get_global_id()` returns `@NamedTuple{x::Int, y::Int, z::Int}` and is
# 1-BASED, and there is no implicit ndrange bounds check the way `@kernel` had,
# so a kernel guards its own tail.
function bcast_f!(out, n::Int, a, b, bstride::Int, f)
    i = KI.get_global_id().x
    i <= n || return
    @inbounds out[i] = f(a[i], b[(i - 1) ÷ bstride + 1])
    return
end

# The KA counterpart, so the predicate is pinned against both kinds rather than
# against one kind and an assumption.
KernelAbstractions.@kernel function ka_bcast_f!(out, @Const(a))
    i = @index(Global, Linear)
    @inbounds out[i] = a[i] + 1.0f0
end

# Two workgroup buffers, same element type, same shape.
#
# `KI.localmemory(T, Val(Dims))` had no call-site id, and a backend lowers
# workgroup memory to a module-level global keyed by what it is handed, so these
# two were ONE buffer: AMDGPU named both `alloc_special_shmem`, Metal and POCL
# both `threadgroup_memory` / `local_memory`, and Lava declined to implement the
# function at all rather than implement it wrong. The second write landed on the
# first tile, silently. A tiled matmul is the case that matters: it stages two
# tiles and both are the same type and shape, which is why KernelAbstractions'
# own `performant_matmul` example had the bug.
#
# `lo` holds each workitem's local id and `hi` holds it negated, so their
# difference is `2l` if the buffers are distinct and `0` if they alias.
function twotiles!(out, n::Int, ::Val{WG}) where {WG}
    lo = KI.localmemory(Float32, Val((WG,)), Val(1))
    hi = KI.localmemory(Float32, Val((WG,)), Val(2))
    l = KI.get_local_id().x
    i = KI.get_global_id().x
    @inbounds lo[l] = Float32(l)
    @inbounds hi[l] = -Float32(l)
    KI.barrier()
    i <= n || return
    @inbounds out[i] = lo[l] - hi[l]
    return
end

# Nothing to launch and nothing to touch: the two dispatches `dispatch!` refuses.
#
# A KA kernel rather than a macro-free one so this reaches every backend --
# `kisupported` is false on the host, and the refusals are core's, not a
# backend's. The body has to do something with its argument or the optimiser
# deletes the whole kernel, and `donotdelete` is the cheapest way to say so
# without touching memory.
KernelAbstractions.@kernel function ka_touchesnothing!(n::Int)
    i = @index(Global, Linear)
    Base.donotdelete(i + n)
end

@testset "a dispatch with nothing to do is refused — $(nameof(typeof(BE)))" begin
    dev = M.Device(BE)

    # ── an empty ndrange ─────────────────────────────────────────────────────
    #
    # `KernelAbstractions.partition` divides by the group size and then by the
    # ndrange, so a zero ndrange is a `DivideError` from inside the launch with
    # nothing in the trace naming the dispatch. SAM 2's encoder has a `(1, 0,
    # 256)` tensor and reached it.
    g = M.Graph(dev)
    out = M.Transient.Buffer(g, Float32, 4)
    err = try
        M.dispatch!(g, ka_bcast_f!, (out, out), 0; name = "empty")
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("empty", err.msg)

    # ── a kernel the walk reports as touching nothing ────────────────────────
    #
    # "No store" is the one answer that must never be a guess, because it is the
    # one that removes a barrier, and a walk that misses every access returns a
    # complete-looking summary rather than refusing. `coopmat_gemm_kernel_4!`
    # did exactly that (see `test_access.jl`): all nine arguments untouched,
    # destination included, and its output transient reached `Place` with no
    # interval.
    err2 = try
        M.dispatch!(g, ka_touchesnothing!, (32,), 32; name = "untouched")
        nothing
    catch e
        e
    end
    @test err2 isa ArgumentError
    @test occursin("touch none", err2.msg)
end

@testset "a plain function is a dispatch kernel — $(nameof(typeof(BE)))" begin
    # ── the predicate ────────────────────────────────────────────────────────
    #
    # Structural, and it deletes itself with the last `@kernel`: what separates
    # the two is that the macro's function answers a BACKEND, which a kernel
    # body does not. `isa Function` cannot tell them apart, because the macro
    # generates a function too.
    @test !M.buildskernel(bcast_f!, BE)
    @test M.buildskernel(ka_bcast_f!, BE)

    n, C = 24, 3
    stride = n ÷ C
    ah = collect(Float32, 1:n)
    bh = Float32[10, 20, 30]
    slope = 0.5f0
    want = [max(ah[i] + bh[(i - 1) ÷ stride + 1], 0.0f0) * slope for i in 1:n]

    # A closure over a scalar, which is an argument WITH data rather than a
    # singleton: the slope travels in the closure and needs no operand.
    f = (x, y) -> max(x + y, 0.0f0) * slope

    dev = M.Device(BE)
    if !M.kisupported(dev, bcast_f!)
        # Stated, not skipped. `KernelInterface` has no backend for
        # `KernelAbstractions.CPU` (only POCL), so a macro-free kernel cannot be
        # compiled for the host at all, and `kikernel` says so by name. The
        # refusal is the behaviour under test here.
        #
        # `kisupported` and not a `hasmethod` written out here, because a device
        # can have two backend objects and only `kibackend` knows which one KI
        # speaks to: asked of `backend(dev)`, this branch was taken on ROCm,
        # whose KI backend is a different type in another module.
        err = try
            M.Plan(let g = M.Graph(dev)
                out = M.Transient.Buffer(g, Float32, n)
                M.dispatch!(g, bcast_f!, (out, n, M.Buffer(dev, ah),
                                          M.Buffer(dev, bh), stride, f), n;
                            name = "bcast")
                g
            end)
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("macro-free kernel", err.msg)
        @test occursin("KI.kernel_function", err.msg)
        return
    end

    a = M.Buffer(dev, ah)
    b = M.Buffer(dev, bh)
    g = M.Graph(dev)
    out = M.Transient.Buffer(g, Float32, n)
    M.dispatch!(g, bcast_f!,
                    (out, n,
                     a,
                     b, stride, f),
                    n; name = "bcast")
    pl = M.Plan(g)
    M.record!(pl)
    M.run!(pl)
    M.waitidle(dev)

    @test Array(M.storage(out)) == want
    # The output is the only transient, so the placer's high-water mark is its
    # bytes and nothing else. Zero here would mean the transient got no
    # interval, which is what a `range`d usage did to its parent.
    @test M.peakbytes(pl) == n * sizeof(Float32)

    # A recorded plan replays without recompiling: the kernel was compiled at
    # `Pipelines`, so a second `run!` is a submission and nothing else.
    fill!(M.storage(out), 0.0f0)
    M.run!(pl)
    M.waitidle(dev)
    @test Array(M.storage(out)) == want

    M.free!(pl)

    # ── the group the caller asked for is the group that runs ────────────────
    #
    # `compile_dispatch` read the workgroup size `KA.launch_config` RETURNED
    # rather than the caller's, and that call substitutes a default of its own,
    # so the occupancy autotune never ran: 29% of SAM 2.1's encoder, recorded.
    # The KI path takes the group through `auto_launch_sizes`, which honours a
    # requested size, so the values must not depend on what was asked for.
    g2 = M.Graph(dev)
    out2 = M.Transient.Buffer(g2, Float32, n)
    M.dispatch!(g2, bcast_f!,
                    (out2, n,
                     a,
                     b, stride, f),
                    n; group = 8, name = "bcast")
    pl2 = M.Plan(g2)
    M.record!(pl2)
    M.run!(pl2)
    M.waitidle(dev)
    @test Array(M.storage(out2)) == want
    M.free!(pl2)

    # ── two workgroup buffers are two buffers ────────────────────────────────
    #
    # See `twotiles!`. Reached only where a macro-free kernel compiles at all,
    # which is what the early return above covers.
    WG = 32
    g3 = M.Graph(dev)
    out3 = M.Transient.Buffer(g3, Float32, WG)
    M.dispatch!(g3, twotiles!, (out3, WG, Val(WG)), WG;
                group = WG, name = "twotiles")
    pl3 = M.Plan(g3)
    M.record!(pl3)
    M.run!(pl3)
    M.waitidle(dev)
    @test Array(M.storage(out3)) == Float32[2l for l in 1:WG]
    M.free!(pl3)
end
