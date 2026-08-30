using Mantle, Test

include(joinpath(@__DIR__, "backend_probe.jl"))


# Relative to this checkout, not to the tree it was first written in: `dev/Mantle`
# and `dev/minimalloc` are siblings wherever the project is checked out, and an
# absolute path here silently reads another tree's benchmarks — or fails on a
# machine that has only one of them.
const BENCH = normpath(joinpath(@__DIR__, "..", "..", "minimalloc", "benchmarks"))

# The corpus is an EXTERNAL checkout, and its absence used to take the whole
# suite down with it: the four testsets below erred, the enclosing
# `@testset "Mantle"` threw at its `end`, and the exception propagated out of
# this file — so the `sync` testset and all six GPU test files never ran. That
# reads as "the suite failed", not as "six files were skipped", which is the
# worse of the two failure modes by a distance.
#
# Guarded rather than deleted: with the corpus present these are the only thing
# pinning how GOOD the packing is, as opposed to merely legal (see
# `docs/design.md`). Clone it beside this checkout to get them back:
#     git clone https://github.com/google/minimalloc ../../minimalloc
const HAVE_BENCH = isdir(BENCH)
HAVE_BENCH || @warn """
    minimalloc benchmarks not found at $BENCH — the packing-QUALITY ratchet is \
    NOT running. Everything else in this suite still is. Clone google/minimalloc \
    as a sibling of this checkout to enable it."""

@testset "Mantle" begin

@testset "spans are half-open" begin
    @test length(Span(3, 9)) == 6
    @test isempty(Span(4, 4))
    @test_throws ArgumentError Span(9, 3)
    # Qualified: GeometryBasics also exports `overlaps`, so an unqualified call
    # is ambiguous the moment anything has loaded it — which is any session that
    # ran a bench first.
    @test Mantle.overlaps(Span(0, 3), Span(2, 5))
    @test !Mantle.overlaps(Span(0, 3), Span(3, 5))   # touching is not overlapping
    @test Mantle.from_closed(0, 2) == Span(0, 3)
end

@testset "csv: the end-column trap" begin
    dir = mktempdir()
    upper = joinpath(dir, "u.csv"); write(upper, "id,lower,upper,size\nb,0,3,4\n")
    ending = joinpath(dir, "e.csv"); write(ending, "id,lower,end,size\nb,0,3,4\n")
    @test readproblem(upper, 10).items[1].live == Span(0, 3)
    @test readproblem(ending, 10).items[1].live == Span(0, 4)   # inclusive, +1
end

@testset "csv: gaps" begin
    dir = mktempdir()
    f = joinpath(dir, "g.csv")
    write(f, "id,lower,upper,size,gaps\nb,0,20,4,9-11 12-14@2:13\n")
    g = readproblem(f, 10).items[1].gaps
    @test length(g) == 2
    @test g[1].during == Span(9, 11) && g[1].window === nothing
    @test g[2].during == Span(12, 14) && g[2].window == OffsetWindow(2, 13)
end

@testset "segments apply gaps" begin
    i = Item("x", Span(0, 10), 4; gaps = [Gap(Span(4, 6), nothing)])
    @test segments(i) == [(Span(0, 4), 4), (Span(6, 10), 4)]
    j = Item("y", Span(0, 10), 4; gaps = [Gap(Span(4, 6), OffsetWindow(0, 1))])
    @test segments(j) == [(Span(0, 4), 4), (Span(4, 6), 1), (Span(6, 10), 4)]
end

HAVE_BENCH && @testset "the README instance is solved optimally" begin
    p = readproblem(joinpath(BENCH, "examples", "input.12.csv"), 12)
    pl = place(p)
    @test [pl.offsets[i.id] for i in p.items] == [8, 8, 4, 4, 0]
    @test pl.height == 12
    @test maxload(p) == 12
    @test first(fragmentation(p, pl)) == 1.0
end

"""No two items that are ever simultaneously live may share a byte."""
function valid(p::Problem, pl::Placement)
    for a in p.items, b in p.items
        a.id < b.id || continue
        Mantle.conflicts(a, b) || continue
        oa, ob = pl.offsets[a.id], pl.offsets[b.id]
        oa < ob + b.size && ob < oa + a.size && return false
    end
    true
end

HAVE_BENCH && @testset "placement is valid on every benchmark" begin
    for f in readdir(joinpath(BENCH, "challenging"); join = true)
        p = readproblem(f)
        for strategy in (LowestFit(), BestFit())
            pl = place(p, strategy)
            @test valid(p, pl)
            @test pl.height >= maxload(p)
            @test all(i -> pl.offsets[i.id] % i.alignment == 0, p.items)
        end
    end
end

"""minimalloc names each instance `<letter>.<capacity>.csv` — the height it is
   known to be solvable at, which is the only external number in the corpus."""
capacity(f) = parse(Int, split(basename(f), ".")[2])

HAVE_BENCH && @testset "the gap to minimalloc's capacity does not grow" begin
    # `valid` and `height >= maxload` say the packing is legal and not below the
    # lower bound. Neither says anything about how *good* it is, so packing could
    # get arbitrarily worse and every placement test would still pass.
    #
    # minimalloc solves all twelve of these at 1048576 with an exact search; a
    # greedy first-fit does not, and is not meant to. Measured, the overshoot is
    # 1.23x (D) to 1.41x (I). Pinned a little above that: this is a ratchet, so a
    # change that packs worse fails here, and one that packs better fails too and
    # gets the bound lowered on purpose rather than by accident.
    worst = 0.0
    for f in readdir(joinpath(BENCH, "challenging"); join = true)
        p, cap = readproblem(f), capacity(f)
        for strategy in (LowestFit(), BestFit())
            worst = max(worst, place(p, strategy).height / cap)
        end
    end
    @test worst <= 1.45
    @test worst > 1.0        # if this ever fails we are matching an exact solver
end

HAVE_BENCH && @testset "lowest fit is not worse than best fit" begin
    ratios = map(readdir(joinpath(BENCH, "challenging"); join = true)) do f
        p = readproblem(f)
        (place(p, LowestFit()).height / maxload(p),
         place(p, BestFit()).height / maxload(p))
    end
    mean(xs) = sum(xs) / length(xs)
    lowest, best = mean(first.(ratios)), mean(last.(ratios))
    @test lowest <= best
    @test lowest < 1.36            # pins the measured 1.3426
    @test count(r -> r[1] < r[2], ratios) > count(r -> r[2] < r[1], ratios)
end

@testset "pinned items keep their offset" begin
    p = Problem([Item("a", Span(0, 10), 4; offset = 16),
                 Item("b", Span(0, 10), 4),
                 Item("c", Span(0, 10), 4)], 64)
    pl = place(p)
    @test pl.offsets["a"] == 16
    @test valid(p, pl)
end

@testset "alignment is honoured" begin
    p = Problem([Item("a", Span(0, 10), 3),
                 Item("b", Span(0, 10), 3; alignment = 256)], 1024)
    pl = place(p)
    @test pl.offsets["b"] % 256 == 0
    @test valid(p, pl)
end

# Two items whose windows are disjoint can interlock: MiniMalloc packs the Tetris
# case into 3 units where a size-only placer needs 4 (minimalloc_test.cc:91-99).
# Gaps currently narrow the lower bound but not the placement, so we take 4.
# `conflicts` compares only the time segments of two items and ignores the offset
# window a `Gap` carries, so two items that could interleave within each other's
# bytes are forced disjoint. Reaching 3 means testing candidate offsets against
# the window — the collision is a function of both offsets, not a static property
# of the pair, so `blocked` would become forbidden *offset* intervals rather than
# occupied byte ranges, and both strategies would move with it.
#
# Left alone on purpose. Nothing generates gaps: the Place phase builds its items
# as `Item(id, span, size; alignment)` and never passes any, and no benchmark CSV
# has a gaps column — the whole corpus is `id,lower,upper,size`. So this would be
# a rewrite of a placer the benchmarks do exercise, to improve one they do not.
@testset "gaps do not yet tighten placement" begin
    p = Problem([Item("a", Span(0, 10), 2; gaps = [Gap(Span(0, 5), OffsetWindow(0, 1))]),
                 Item("b", Span(0, 10), 2; gaps = [Gap(Span(5, 10), OffsetWindow(1, 2))])], 16)
    @test place(p).height == 4
    @test_broken place(p).height == 3
end

end

# ── everything below that needs a Vulkan driver ───────────────────────────────
#
# `Vulkan` is a weakdep of Mantle now, so this file is reached on a machine that
# has no loader at all. See `backend_probe.jl` for why that used to be fatal.
const _VULKAN_OK = backend_loadable("Vulkan") !== nothing
_VULKAN_OK || @info "Mantle tests: no Vulkan driver; skipping the Vulkan backend and the sync lowering"

if _VULKAN_OK

# `import`, not `using`: Mantle exports `Vulkan` as the name of its backend, and
# the package is also called Vulkan, so `using Vulkan` alongside `using Mantle`
# makes the bare name ambiguous. Everything here names it qualified anyway.
import Vulkan

@testset "sync" begin
    be = Mantle.VulkanAPI()
    img, buf = ImageKind(), BufferKind()
    A2(x) = Vulkan.AccessFlag2(x)
    S2(x) = Vulkan.PipelineStageFlag2(x)

    @testset "lowering takes a side" begin
        # Depth writes complete late, reads begin early. One value cannot serve
        # both (rps_vk_runtime_backend.cpp:168).
        @test Mantle.stages(be, Depth{WriteOnly,WriteOnly,false}, Src()) ==
              S2(Vulkan.PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT)
        @test Mantle.stages(be, Depth{WriteOnly,WriteOnly,false}, Dst()) ==
              S2(Vulkan.PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT)
        @test Mantle.layout(be, Present, Src()) == Vulkan.IMAGE_LAYOUT_UNDEFINED
        @test Mantle.layout(be, Present, Dst()) == Vulkan.IMAGE_LAYOUT_PRESENT_SRC_KHR
    end

    @testset "TOP_OF_PIPE is unspellable as a source" begin
        top = S2(Vulkan.PIPELINE_STAGE_2_TOP_OF_PIPE_BIT)
        for U in (Vertices, Indices, Indirect, Uniform, Sampled, Present, CopySrc,
                  CopyDst, ColorAttachment{true}, ColorAttachment{false}, Depth{WriteOnly,WriteOnly,false},
                  Storage{ImageKind,ReadWrite}, TraceRead, TraceBuild)
            @test Mantle.stages(be, U, Src()) != top
        end
    end

    @testset "a discarding load op drops the read bit" begin
        @test Mantle.access(be, ColorAttachment{true}, Dst()) == A2(Vulkan.ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT)
        @test (Mantle.access(be, ColorAttachment{false}, Dst()) &
               A2(Vulkan.ACCESS_2_COLOR_ATTACHMENT_READ_BIT)) != A2(0)
    end

    @testset "four depth layouts, not two" begin
        ls = [Mantle.layout(be, Depth{a,b,false}, Dst()) for a in (ReadOnly, WriteOnly),
                                                             b in (ReadOnly, WriteOnly)]
        @test length(unique(ls)) == 4
    end

    @testset "an image with no stencil aspect is the fifth case" begin
        # The four above are the separateDepthStencilLayouts ones. A depth-only
        # image has no stencil state to name, and the combined layout is what
        # every device can take, feature or no feature.
        @test Mantle.layout(be, Depth{ReadWrite,NoAccess,false}, Dst()) ==
              Vulkan.IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        @test Mantle.layout(be, Depth{ReadOnly,NoAccess,false}, Dst()) ==
              Vulkan.IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
        @test Mantle.layout(be, Depth{ReadWrite,NoAccess,false}, Dst()) !=
              Mantle.layout(be, Depth{ReadWrite,ReadOnly,false}, Dst())
    end

    @testset "a discarding depth load op drops the read bit" begin
        # Same rule as the colour attachment, and the same reason: a cleared depth
        # buffer reads nothing from before the pass, so the barrier into it can
        # come from UNDEFINED — which is what a transient sharing bytes needs.
        cleared = Depth{ReadWrite,NoAccess,true}
        kept = Depth{ReadWrite,NoAccess,false}
        @test Mantle.access(be, cleared, Dst()) ==
              A2(Vulkan.ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT)
        @test (Mantle.access(be, kept, Dst()) &
               A2(Vulkan.ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT)) != A2(0)
        @test discards(cleared) && !discards(kept)
        @test !reads(cleared) && reads(kept)
        @test writes(cleared) && writes(kept)
    end

    @testset "hazard rules" begin
        @test isempty(transitions(be, buf, [Storage{BufferKind,ReadOnly},
                                            Storage{BufferKind,ReadOnly}]))
        # UAV to UAV races even when the access did not change
        @test length(transitions(be, buf, [Storage{BufferKind,WriteOnly},
                                           Storage{BufferKind,WriteOnly}])) == 1
        # unordered must be declared on both sides
        u = Unordered{Storage{BufferKind,WriteOnly}}
        @test isempty(transitions(be, buf, [u, u]))
        @test length(transitions(be, buf, [u, Storage{BufferKind,WriteOnly}])) == 1
    end

    @testset "layout changes between two reads, but only for images" begin
        @test Mantle.needs_transition(be, img, Sampled, CopySrc)
        @test !Mantle.needs_transition(be, buf, Sampled, CopySrc)
        @test !Mantle.needs_transition(be, img, Sampled, Sampled)
    end

    @testset "transitions chain from the current state" begin
        ts = transitions(be, img, [ColorAttachment{true}, Sampled, CopySrc])
        @test length(ts) == 2
        @test ts[2].from === Sampled          # not ColorAttachment: the sample moved the layout
        @test ts[2].to === CopySrc
    end

    @testset "a write waits on every outstanding reader" begin
        ts = transitions(be, img, [ColorAttachment{true}, Sampled, CopySrc,
                                   Storage{ImageKind,WriteOnly}])
        last = ts[end]
        @test last.from === CopySrc                    # layout comes from one state
        @test Set(last.waits) == Set([CopySrc, Sampled])  # ordering from all readers
    end

    @testset "every lowered usage lowers on every side" begin
        # The exhaustiveness check §3 promised. `applicable` with the usage passed
        # as a value against a ::Type{<:U} parameter: asking about Type{U} is a
        # different question and answers false for everything.
        acc = (ReadOnly, WriteOnly, ReadWrite)
        usages = Type[Vertices, Indices, Indirect, Uniform, Sampled, Present,
                      CopySrc, CopyDst, TraceRead, TraceBuild,
                      ColorAttachment{true}, ColorAttachment{false},
                      (Storage{K,A} for K in (BufferKind, ImageKind) for A in acc)...,
                      # The stencil half takes NoAccess too — an image without a
                      # stencil aspect — and both load ops.
                      (Depth{D,S,X} for D in acc for S in (acc..., NoAccess)
                                    for X in (true, false))...]
        @test length(usages) == 42
        for U in usages, side in (Src(), Dst()), f in (stages, access, layout)
            @test applicable(f, be, U, side)
        end
    end

    @testset "no stage that is only legal on one side appears on the other" begin
        acc = (ReadOnly, WriteOnly, ReadWrite)
        usages = Type[Vertices, Indices, Indirect, Uniform, Sampled, Present,
                      CopySrc, CopyDst, TraceRead, TraceBuild, ColorAttachment{true}, ColorAttachment{false},
                      (Storage{K,A} for K in (BufferKind, ImageKind) for A in acc)...,
                      (Depth{D,S,X} for D in acc for S in (acc..., NoAccess)
                                    for X in (true, false))...]
        top = S2(Vulkan.PIPELINE_STAGE_2_TOP_OF_PIPE_BIT)
        bottom = S2(Vulkan.PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT)
        for U in usages
            @test Mantle.stages(be, U, Src()) != top       # no dependency with anything
            @test Mantle.stages(be, U, Dst()) != bottom    # likewise
        end
    end

    @testset "an attribute is pulled, not bound" begin
        # The Vulkan backend's pipelines declare an empty vertex input state and
        # the vertex shader loads its attributes through a buffer device address,
        # so `Vertices` is a storage read in VERTEX_SHADER. It used to lower to
        # VERTEX_ATTRIBUTE_INPUT / VERTEX_ATTRIBUTE_READ, which describes a fetch
        # that never happens: the stage was survivable, because it precedes
        # VERTEX_SHADER and so the execution dependency still covered the shader,
        # but the access mask was not — it makes a write visible to attribute
        # fetch and says nothing about the storage read that actually occurs.
        #
        # The two also have to move together: VERTEX_ATTRIBUTE_READ alongside
        # VERTEX_SHADER is VUID-VkMemoryBarrier2-srcAccessMask-03902, which is
        # what caught the half-done version of this.
        @test Mantle.stages(be, Vertices, Src()) == S2(Vulkan.PIPELINE_STAGE_2_VERTEX_SHADER_BIT)
        @test Mantle.access(be, Vertices, Src()) == A2(Vulkan.ACCESS_2_SHADER_STORAGE_READ_BIT)
    end

    @testset "no usage lowers to every stage" begin
        # `ALL_COMMANDS` on both sides of a barrier is a full pipeline drain. It
        # is never *wrong*, which is the problem: a derivation whose answer is
        # always "wait for everything" has stopped deriving, and a mistake in it
        # cannot be observed. Every usage here names the stages it actually
        # touches, so this asserts the derivation still has something to say.
        #
        # The last holdout was the aliasing barrier, and it is not a usage at all
        # any more — it is assembled from what the vacating transient was doing.
        every = S2(Vulkan.PIPELINE_STAGE_2_ALL_COMMANDS_BIT)
        acc3 = (ReadOnly, WriteOnly, ReadWrite)
        usages = Type[Vertices, Indices, Indirect, Uniform, Sampled, Present, Undefined,
                      CopySrc, CopyDst, TraceRead, TraceBuild,
                      ColorAttachment{true}, ColorAttachment{false},
                      (Storage{K,A} for K in (BufferKind, ImageKind) for A in acc3)...,
                      (Depth{D,S,X} for D in acc3 for S in (acc3..., NoAccess)
                                    for X in (true, false))...]
        for U in usages, side in (Src(), Dst())
            @test Mantle.stages(be, U, side) != every
        end
    end

    @testset "WebGPU has no barriers" begin
        @test isempty(transitions(Mantle.WebGPUAPI(), img, [ColorAttachment{true}, Sampled, CopySrc]))
    end

    @testset "a load op says whether the target's contents survive" begin
        # Three answers, not two. `Discard` exists because it is the only route to
        # LOAD_OP_DONT_CARE, and because `discards` is what lets the barrier into
        # the attachment come from UNDEFINED and drop the read access bit.
        @test !Mantle.discards(Mantle.Keep)
        @test Mantle.discards(Mantle.Discard)
        @test Mantle.discards(Mantle.Clear((0f0, 0f0, 0f0, 1f0)))

        # ...which is exactly the parameter of the usage a render pass declares.
        @test Mantle.discards(ColorAttachment{true})
        @test !Mantle.discards(ColorAttachment{false})
        @test !Mantle.reads(ColorAttachment{true})     # nothing is loaded, so nothing is read
        @test Mantle.reads(ColorAttachment{false})
    end
end

end  # if _VULKAN_OK — the sync lowering names Vulkan enums in every assertion

# CPU-only, so they go first and fail fast. These sat next to this file without
# being included by it, which meant every regression they pin was unguarded —
# the host backend could not write a struct with padding for as long as it has
# existed, and `test_host.jl` is the file that would have said so.
#
# Outside the driver gate on purpose: that they need no GPU is the assertion.
include(joinpath(@__DIR__, "test_pool.jl"))
include(joinpath(@__DIR__, "test_host.jl"))

if _VULKAN_OK
# Needs a GPU but no display — every graph in it is headless, which is also the
# only kind `bake!` takes — so it runs before the window tests rather than inside
# their DISPLAY guard.
#
# FIRST among the files that build a plan on `Device(VulkanAPI())`, and that is a
# constraint rather than a preference: it counts the tenants of that device's
# shared arena, and `Device(VulkanAPI())` is cached per context, so any earlier file
# that compiled a plan is still a tenant until a GC reaps it. Adding an include
# above this line that touches the Lava device breaks it.
include(joinpath(@__DIR__, "test_arena_bake.jl"))
include(joinpath(@__DIR__, "test_compile_golden.jl"))
# Same shape: headless, GPU-only. A `DeviceRange` is the one ndrange whose value
# never reaches the host, so the Host backend cannot pin the half that matters.
include(joinpath(@__DIR__, "test_devicerange.jl"))
include(joinpath(@__DIR__, "test_window.jl"))
end  # if _VULKAN_OK

# ── the Metal backend ─────────────────────────────────────────────────────────
#
# Gated on Metal being loadable AND functional, not on the OS: a Mac without a
# usable GPU (a CI container, a VM) must skip rather than error, and `Metal` is
# resolvable on any platform while `Metal.functional()` is the honest question.
#
# These sat in `test/metal/` without being included by anything, which meant
# every regression they pin was unguarded — the same way `test_host.jl` was
# before the note above. Two of them were failing when they were finally run.
const METAL_TESTS = joinpath(@__DIR__, "metal")

const _METAL_OK = let
    M = backend_loadable("Metal")
    # `functional()` on top of loadability, because Metal.jl imports fine on a
    # machine with no usable device — which Vulkan does not, so only this side
    # needs the second question.
    #
    # `invokelatest`, because `Base.require` defines `functional` in a world
    # NEWER than this top-level statement, which was fixed when it began. A
    # direct call is a `MethodError: the applicable method may be too new` on
    # the very machine the gate exists to serve.
    M === nothing ? false : Base.invokelatest(M.functional)::Bool
end

if _METAL_OK
    @testset "Metal backend" begin
        for f in ("test_pool_metal.jl", "test_kernelinterface_metal.jl",
                  "test_graph_metal.jl", "test_raytracing_metal.jl",
                  "test_trace_metal.jl", "test_hwtlas_metal.jl",
                  "test_residency_metal.jl", "test_graphics_metal.jl",
                  "test_render_graph_metal.jl")
            @testset "$f" begin
                include(joinpath(METAL_TESTS, f))
            end
        end
    end
else
    @info "Mantle tests: no usable Metal device; skipping the Metal backend"
end

# ── the Vulkan backend ────────────────────────────────────────────────────────
#
# 128 files that were Lava's test suite until 2026-08-27, when the runtime moved
# into `src/vulkan/`. Every one of them needs a device, a queue, a pool or a
# `LavaArray` — which is exactly the rule that decided what moved. What stayed
# with Lava runs on a machine with no Vulkan driver at all.
#
# The `mwe_*.jl` files beside them are standalone reproducers and are not driven
# from here, the same as before: each one is run on its own while a bug is being
# chased.
const VULKAN_TESTS = joinpath(@__DIR__, "vulkan")

if _VULKAN_OK
@testset "Vulkan backend" begin

        # Source-and-bindings only, no device. First, so it is reported before
        # anything that can take a device down with it — and because what it
        # catches is a name that would otherwise throw far away from its cause.
        @testset "Lava import completeness" begin
            include(joinpath(VULKAN_TESTS, "test_lava_import_completeness.jl"))
            # The same boundary from the other side: what each package EXPORTS
            # has to match what it defines. Both directions went wrong in the
            # move and neither failed at load.
            include(joinpath(VULKAN_TESTS, "test_no_stale_exports.jl"))
            # And how much of the array algorithms is still Vulkan's — a ratchet
            # on step 5, which asks where they should live.
            include(joinpath(VULKAN_TESTS, "test_array_algorithm_portability.jl"))
        end

        # ── Tier 3a3: tensor addressing actually loads (GPU) ──
        # Compiling and validating is not enough here: the instruction validated
        # twice while still wrong. This runs it and checks the values and the
        # orientation.
        @testset "Tier 3a3: coopmat2 tensor load" begin
            include(joinpath(VULKAN_TESTS, "test_tensor_load.jl"))
            # What the load substitutes OUT of range, which is a value the caller
            # chooses and not always zero — `attn_flash_cm2!` gets a whole row
            # reduction deleted by asking for one.
            include(joinpath(VULKAN_TESTS, "test_tensor_clampvalue.jl"))
            # And what a PRODUCT of two tensor-loaded operands computes, which the
            # load test cannot say: the load returns the transpose of its block, so
            # the product is `P' * Q'`. A GEMM cannot be routed through this until
            # that is pinned, and all four candidate orientations look sane.
            include(joinpath(VULKAN_TESTS, "test_tensor_gemm.jl"))
            # And the claim the port rests on: a clamping layout bounds-checks the
            # load, so an extent that divides nothing is legal and out-of-range reads
            # come back as exact zeros. That is what retires `gemm_padn`,
            # `GEMM_BLOCK`, `padtile`/`crsextent` and `gemm_divides`.
            include(joinpath(VULKAN_TESTS, "test_tensor_clamp.jl"))
            # And that a shape the device does NOT report is usable at all: every
            # KHR shape here has M == 16, so 64x16 exercises coopmat2's flexible
            # dimensions. A hard-coded shape list in `KNOWN_INTRINSICS` used to
            # reject it before the emitter — which handles it correctly — ever saw
            # it, so this guards a gate, not an instruction.
            include(joinpath(VULKAN_TESTS, "test_coopmat_flexible_dims.jl"))
            # The clamp test above covers READS. This is the other half: a clamping
            # layout must bounds-check the STORE too, or a tensor GEMM can consume
            # unpadded operands and still not write an unpadded result. Asserts
            # two-sided — in-range elements land, and nothing outside the extent
            # moves — because a store that trampled its neighbours would pass a
            # one-sided "the right values are there" check.
            include(joinpath(VULKAN_TESTS, "test_tensor_store.jl"))
            # Workgroup scope, and the two kernels built on it. The GEMM is not
            # routed to yet — `coopmat_gemm!` still runs the staged kernel — so this
            # is the only thing keeping it honest while it waits to be measured.
            include(joinpath(VULKAN_TESTS, "test_workgroup_scope.jl"))
            include(joinpath(VULKAN_TESTS, "test_gemm_cm2.jl"))
        end


        # ── Tier 3a: Workgroup barrier-skip fix (GPU; catches lavapipe deadlock) ──
        @testset "Tier 3a: Barrier skip fix" begin
            include(joinpath(VULKAN_TESTS, "test_barrier_skip.jl"))
        end


        # ── Tier 3a2: norm FTZ rescaling regression (GPU; Float64/ComplexF64) ──
        @testset "Tier 3a2: norm FTZ rescaling" begin
            include(joinpath(VULKAN_TESTS, "test_norm_overflow.jl"))
        end


        # ── Tier 3a3: workgroup (shared) memory stress (GPU) ──
        @testset "Tier 3a3: shared-memory stress" begin
            include(joinpath(VULKAN_TESTS, "test_shared_memory_stress.jl"))
        end


        # ── Tier 3a4: OpSelect-of-Workgroup-pointers type-dedup regression (GPU) ──
        @testset "Tier 3a4: OpSelect Workgroup pointer dedup" begin
            include(joinpath(VULKAN_TESTS, "test_select_width_mismatch.jl"))
        end


        # ── Tier 3a5: constant lookup table indexed at runtime (GPU) ──
        @testset "Tier 3a5: constant table storage class" begin
            include(joinpath(VULKAN_TESTS, "test_const_table_index.jl"))
        end


        # ── Tier 3: GPU Execution ──
        @testset "Tier 3: GPU Execution" begin
            include(joinpath(VULKAN_TESTS, "test_handwritten_spirv.jl"))
            include(joinpath(VULKAN_TESTS, "test_handwritten_rt.jl"))
            # After test_handwritten_rt.jl: it reuses those shaders to prove a refit
            # moved the acceleration structure, not merely the vertex buffer.
            include(joinpath(VULKAN_TESTS, "test_blas_refit.jl"))
            include(joinpath(VULKAN_TESTS, "test_rayquery_vs_cpu.jl"))
            include(joinpath(VULKAN_TESTS, "test_aabb_blas_overlap.jl"))
            include(joinpath(VULKAN_TESTS, "test_instance_masks.jl"))
        end

            @testset "Tier 3b: Struct Broadcast" begin
                include(joinpath(VULKAN_TESTS, "test_struct_broadcast.jl"))
            end

            @testset "Narrow phase (CPU)" begin
                include(joinpath(VULKAN_TESTS, "test_convex_shape.jl"))
                include(joinpath(VULKAN_TESTS, "test_gjk.jl"))
                include(joinpath(VULKAN_TESTS, "test_epa.jl"))
                include(joinpath(VULKAN_TESTS, "test_narrow_phase_kernel.jl"))
                include(joinpath(VULKAN_TESTS, "test_contact_record.jl"))
                include(joinpath(VULKAN_TESTS, "test_narrow_phase_contacts.jl"))
            end


        # ── Tier 3c: Atomics & Batched Dispatch ──
        @testset "Tier 3c: Atomics & Dispatch" begin
            include(joinpath(VULKAN_TESTS, "test_atomics_and_dispatch.jl"))
        end


        # ── Tier 3c2: On-kernel printf (DebugPrintf SPIR-V emission) ──
        # Only the spirv-val emission test runs here; the live-output test resets the
        # device and is opt-in via LAVA_PRINTF_LIVE=1.
        @testset "Tier 3c2: @lava_printf" begin
            include(joinpath(VULKAN_TESTS, "test_lava_printf.jl"))
        end


        # ── Tier 3v: hardware H.264 video decode (skips without a video-decode queue) ──
        @testset "Tier 3v: H.264 hardware decode" begin
            include(joinpath(VULKAN_TESTS, "test_video_decode.jl"))
        end


        @testset "video decode layout" begin
            include(joinpath(VULKAN_TESTS, "test_video_decode_layout.jl"))
        end


        @testset "double-indirect MVector access" begin
            include(joinpath(VULKAN_TESTS, "test_double_indirect.jl"))
        end


        @testset "Int32 CartesianIndex into Broadcasted" begin
            include(joinpath(VULKAN_TESTS, "test_int32_cartesian_miscompile.jl"))
        end


        @testset "workgroup size limit" begin
            include(joinpath(VULKAN_TESTS, "test_workgroup_limit.jl"))
        end


        @testset "fastdiv index decomposition" begin
            include(joinpath(VULKAN_TESTS, "test_fastdiv.jl"))
        end


        @testset "cooperative-matrix epilogue" begin
            include(joinpath(VULKAN_TESTS, "test_coopmat_epilogue.jl"))
        end


        @testset "shared stores through a divided index" begin
            include(joinpath(VULKAN_TESTS, "test_shared_index_division.jl"))
        end


        @testset "staged GEMM" begin
            include(joinpath(VULKAN_TESTS, "test_gemm_staged.jl"))
        end


        @testset "scalar GEMM accumulator width" begin
            include(joinpath(VULKAN_TESTS, "test_gemm_fp16_accum.jl"))
        end


        # The scalar half of the same port: `mul!`'s fp32 path, which had no tiling
        # at all and ran at 0.448 TFLOP/s where this reaches 4.301.
        @testset "staged scalar GEMM" begin
            include(joinpath(VULKAN_TESTS, "test_gemm_staged_scalar.jl"))
        end


        @testset "whole-struct copy alignment" begin
            include(joinpath(VULKAN_TESTS, "test_struct_copy_alignment.jl"))
        end


        @testset "systematic struct alignment" begin
            include(joinpath(VULKAN_TESTS, "test_struct_alignment_systematic.jl"))
        end


        @testset "pipeline cache avoids driver compilation" begin
            include(joinpath(VULKAN_TESTS, "test_pipeline_cache_no_compile.jl"))
        end


        # Straight after the reset above, and before `test_static_workgroup.jl` —
        # which is where this crashed, because that file calls `GC.gc()` explicitly
        # and so ran whichever finalizer the reset had stranded.
        @testset "a buffer may outlive a device reset" begin
            include(joinpath(VULKAN_TESTS, "test_device_reset_finalizer.jl"))
        end


        @testset "static workgroup indexing" begin
            include(joinpath(VULKAN_TESTS, "test_static_workgroup.jl"))
        end


        # The one field the multi-device work rests on. Needs no second GPU: what it
        # pins is that "this device" and "whichever is current" stay distinguishable.
        @testset "a backend knows its device" begin
            include(joinpath(VULKAN_TESTS, "test_backend_context.jl"))
        end


        # Two live devices in one process — the real GPU and lavapipe, which the
        # loader enumerates together, so this needs no second card. Asserts both
        # compute correctly AND that one kernel compiles twice: a shared pipeline
        # can still give the right answer by luck. See the file for the five
        # separate pieces of module-scope device state it found.
        @testset "two devices in one process" begin
            include(joinpath(VULKAN_TESTS, "twodevice_probe.jl"))
            probe()
        end


        # Catches an invalid module that the driver accepts and runs correctly, so it
        # is invisible to every other test unless the device carries
        # `DebugConfig(validation = true)`.
        @testset "no OpBitcast on a logical pointer" begin
            include(joinpath(VULKAN_TESTS, "test_logical_pointer_bitcast.jl"))
        end


        # Only meaningful on a device whose subgroup width is not fixed — on wave32
        # hardware "requested" and "default" are the same number and nothing is proved.
        @testset "pinned subgroup width" begin
            include(joinpath(VULKAN_TESTS, "test_subgroup_size_pinning.jl"))
        end


        # Both halves of the coopmat 32-lane pin: that it is not needed here, and
        # that the refusal fires when it would be. The second half is unreachable on
        # any device present, so it drives the capability cache instead.
        @testset "coopmat 32-lane pin" begin
            include(joinpath(VULKAN_TESTS, "test_coopmat_subgroup_pin.jl"))
        end


        # A handled allocation failure must absorb its own validation messages, or it
        # aborts whatever unrelated code calls check_validation_errors! next.
        @testset "tolerated alloc failure" begin
            include(joinpath(VULKAN_TESTS, "test_tolerated_alloc_failure.jl"))
        end


        # The frozen path is the one the runners ship, and the profiler could not see
        # it: 0 kernels reported against 45 live dispatches.
        @testset "frozen kernels are visible to the profiler" begin
            include(joinpath(VULKAN_TESTS, "test_frozen_kernels_visible.jl"))
        end


        # `vk_context` had methods for three of AnyLavaArray's six wrappers.
        @testset "vk_context through array wrappers" begin
            include(joinpath(VULKAN_TESTS, "test_vk_context_wrappers.jl"))
        end


        # The buffer-lifetime case behind the intermittent flush hang. Asserted on
        # the state machine, not by provoking the hang — see the file.
        @testset "free during recording" begin
            include(joinpath(VULKAN_TESTS, "test_free_during_recording.jl"))
        end


        # ── Previously unregistered test files ──
        #
        # These existed in test/ but were never included here. That is not neutral:
        # of the files found unregistered, three characterised bugs that had since been
        # FIXED without anyone noticing, one exercised an API deleted three months
        # earlier, one never waited on its own spawned task, and two had gone stale
        # against a refactor. Registered so they cannot rot again.
        #
        # Still deliberately out: the heavy stress/CI entry points.
        # (test_struct_alignment_systematic.jl was excluded here while the
        # whole-struct-copy bug was open; that is fixed, and it is registered above.)
        @testset "barrier elision" begin
            include(joinpath(VULKAN_TESTS, "test_barrier_elision.jl"))
        end

        @testset "broadcast paths" begin
            include(joinpath(VULKAN_TESTS, "test_broadcast_paths.jl"))
        end

        @testset "closest_hit via ray query" begin
            include(joinpath(VULKAN_TESTS, "test_closesthit_via_rayquery.jl"))
        end

        @testset "coopmat shared memory" begin
            include(joinpath(VULKAN_TESTS, "test_coopmat_shared.jl"))
        end

        @testset "batched coopmat GEMM" begin
            include(joinpath(VULKAN_TESTS, "test_gemm_batched.jl"))
        end

        @testset "AdaptedAccel via ray query" begin
            include(joinpath(VULKAN_TESTS, "test_hwadapted_via_rayquery.jl"))
        end

        @testset "hwtlas batch delete" begin
            include(joinpath(VULKAN_TESTS, "test_hwtlas_batch_delete.jl"))
        end

        @testset "hwtlas batch triangles" begin
            include(joinpath(VULKAN_TESTS, "test_hwtlas_batch_triangles.jl"))
        end

        @testset "hwtlas instance buffer" begin
            include(joinpath(VULKAN_TESTS, "test_hwtlas_instance_buffer.jl"))
        end

        @testset "hwtlas push instances" begin
            include(joinpath(VULKAN_TESTS, "test_hwtlas_push_instances.jl"))
        end

        @testset "hwtlas refit" begin
            include(joinpath(VULKAN_TESTS, "test_hwtlas_refit.jl"))
        end

        @testset "hwtlas sync batch" begin
            include(joinpath(VULKAN_TESTS, "test_hwtlas_sync_batch.jl"))
        end

        @testset "instance record" begin
            include(joinpath(VULKAN_TESTS, "test_instance_record.jl"))
        end

        @testset "permutedims" begin
            include(joinpath(VULKAN_TESTS, "test_permutedims.jl"))
        end

        @testset "phase 1 — pin/sync_access" begin
            include(joinpath(VULKAN_TESTS, "test_phase1_pin.jl"))
        end

        @testset "phase 2 — error surfacing" begin
            include(joinpath(VULKAN_TESTS, "test_phase2_errors.jl"))
        end

        @testset "phase 3 — lifecycle" begin
            include(joinpath(VULKAN_TESTS, "test_phase3_lifecycle.jl"))
        end

        @testset "phase 4 — single writer" begin
            include(joinpath(VULKAN_TESTS, "test_phase4_singlethread.jl"))
        end

        @testset "phase 5 — transfer path" begin
            include(joinpath(VULKAN_TESTS, "test_phase5_copy.jl"))
        end

        @testset "phase 6 — graphics dispatch" begin
            include(joinpath(VULKAN_TESTS, "test_phase6_graphics.jl"))
        end

        # `test_pool_sizeclass.jl` was here, and it is deleted with the thing it
        # tested. It checked that `size_class` was idempotent on its own output —
        # `pool_alloc` looked a class up from the REQUEST and `return_to_pool!`
        # looked it up again from the size handed out, so a chunk returning to the
        # wrong list would be given to a caller who asked for more than it holds.
        # `Mantle.carve!` splits at exactly the requested length and `release!`
        # checks the block's own `live` ledger, so neither the rounding nor the
        # round-trip it had to be consistent about exists any more. The property
        # that replaced it — every live region disjoint, in range and aligned — is
        # in `test/test_pool.jl`, and runs with no device at all.

        @testset "source mapping" begin
            include(joinpath(VULKAN_TESTS, "test_source_mapping.jl"))
        end

        @testset "tlas allow_update" begin
            include(joinpath(VULKAN_TESTS, "test_tlas_allow_update.jl"))
        end

        @testset "tlas refit" begin
            include(joinpath(VULKAN_TESTS, "test_tlas_refit.jl"))
        end

        @testset "tlas refit integration" begin
            include(joinpath(VULKAN_TESTS, "test_tlas_refit_integration.jl"))
        end

        @testset "trace cull mask" begin
            include(joinpath(VULKAN_TESTS, "test_trace_cull_mask.jl"))
        end

        @testset "typepun trunc/bitcast" begin
            include(joinpath(VULKAN_TESTS, "test_typepun_trunc_bitcast.jl"))
        end

            @testset "Tier 3d: SPIR-V Pattern Correctness & Stress" begin
                include(joinpath(VULKAN_TESTS, "test_spirv_pattern_correctness.jl"))
                include(joinpath(VULKAN_TESTS, "test_loop_unswitch_miscompile.jl"))
                include(joinpath(VULKAN_TESTS, "test_psb_chain_fold.jl"))
                include(joinpath(VULKAN_TESTS, "test_repeat_inner_3d.jl"))
                include(joinpath(VULKAN_TESTS, "test_multiindex_getindex.jl"))
            end

            @testset "Tier 3e: GPU Memory Safety" begin
                include(joinpath(VULKAN_TESTS, "test_gpu_memory_safety.jl"))
            end

            @testset "Tier 3e2: BAR Memcpy Sync" begin
                include(joinpath(VULKAN_TESTS, "test_bar_memcpy_sync.jl"))
            end

            @testset "Tier 3e3: MultiTypeSet Surgical" begin
                include(joinpath(VULKAN_TESTS, "test_multitypeset_surgical.jl"))
            end

            @testset "Tier 3f: Caching & Allocations" begin
                include(joinpath(VULKAN_TESTS, "test_caching_and_allocations.jl"))
            end


        # ── Tier 3h: Disk Cache & Two-Tier Caching ──
        @testset "Tier 3h: Kernel Cache" begin
            include(joinpath(VULKAN_TESTS, "test_disk_cache.jl"))
            # Fixed per-kernel compile overhead (ungated IR dump, subprocess poll
            # quantisation). Lives here because it is about what a compile costs
            # when the cache does NOT save you.
            include(joinpath(VULKAN_TESTS, "test_compile_overhead.jl"))
        end


        # ── Tier 3g: Graphics Pipeline ──
        @testset "Tier 3g: Graphics Pipeline" begin
            include(joinpath(VULKAN_TESTS, "test_graphics_pipeline.jl"))
        end

            @testset "Tier 4: GPUArrays TestSuite" begin
                include(joinpath(VULKAN_TESTS, "test_gpuarrays.jl"))
            end

            @testset "HW HWTLAS — stress + correctness" begin
                include(joinpath(VULKAN_TESTS, "test_hwtlas_stress.jl"))
            end


            @testset "HW HWTLAS — mesh update" begin
                include(joinpath(VULKAN_TESTS, "test_hwtlas_mesh_update.jl"))
            end


            @testset "HW HWTLAS — UAF safety" begin
                include(joinpath(VULKAN_TESTS, "test_hwtlas_uaf_safety.jl"))
            end


            @testset "pinned buffer lifetime" begin
                include(joinpath(VULKAN_TESTS, "test_pinned_buffer_lifetime.jl"))
            end


            @testset "workgroup zero-init" begin
                include(joinpath(VULKAN_TESTS, "test_workgroup_zero_init.jl"))
            end


            @testset "pool trim" begin
                include(joinpath(VULKAN_TESTS, "test_pool_trim.jl"))
                # The other end of the same allocator: what happens when the
                # device says no. Only reachable by asking for more VRAM than
                # exists, so nothing else covers it.
                include(joinpath(VULKAN_TESTS, "test_pool_alloc_oom.jl"))
            end


            # Was never included, and had been throwing `UndefVarError` on the first
            # line of every testset since the deferred-free list went per-VulkanBatchQueue.
            @testset "rapid alloc/free" begin
                include(joinpath(VULKAN_TESTS, "test_rapid_alloc_free.jl"))
            end


            @testset "Diagonal mul! disambiguation" begin
                include(joinpath(VULKAN_TESTS, "test_diagonal_mul.jl"))
            end


            @testset "device capabilities" begin
                include(joinpath(VULKAN_TESTS, "test_device_caps.jl"))
            end


            @testset "debug configuration" begin
                include(joinpath(VULKAN_TESTS, "test_debug_config.jl"))
                include(joinpath(VULKAN_TESTS, "test_dispatch_allocation.jl"))
            end


            @testset "batched 1D FFT" begin
                include(joinpath(VULKAN_TESTS, "test_fft.jl"))
            end


            @testset "batch-1 GEMV" begin
                include(joinpath(VULKAN_TESTS, "test_gemv.jl"))
            end


            @testset "coopmat shape query" begin
                include(joinpath(VULKAN_TESTS, "test_coopmat_shape.jl"))
            end


            @testset "coopmat GEMM subgroup guard" begin
                include(joinpath(VULKAN_TESTS, "test_coopmat_gemm_subgroup.jl"))
            end


            @testset "subgroup shuffle family" begin
                include(joinpath(VULKAN_TESTS, "test_subgroup_shuffle.jl"))
            end


            @testset "coopmat per-element and component-wise ops" begin
                include(joinpath(VULKAN_TESTS, "test_coopmat_perelement.jl"))
            end


            @testset "coopmat component-wise add" begin
                include(joinpath(VULKAN_TESTS, "test_coopmat_add.jl"))
            end


            @testset "coopmat reductions (NV)" begin
                include(joinpath(VULKAN_TESTS, "test_coopmat_reduce.jl"))
            end


            # test_frozen_cache.jl shipped unregistered, so the compute-side frozen
            # cache had no coverage in CI. It restores FROZEN_VERSION by plain
            # assignment, so a throw part-way through would leave the cache ENABLED
            # for every later testset here — silently, since a frozen hit looks like
            # a normal launch. Restore it from a finally block instead.
            @testset "frozen kernel cache (compute)" begin
                _fv, _fr = Lava.FROZEN_VERSION[], Lava.FROZEN_RECORDING[]
                try
                    include(joinpath(VULKAN_TESTS, "test_frozen_cache.jl"))
                finally
                    Lava.FROZEN_VERSION[]   = _fv
                    Lava.FROZEN_RECORDING[] = _fr
                end
            end


            @testset "HW HWTLAS — nonblocking sync!" begin
                include(joinpath(VULKAN_TESTS, "test_hwtlas_nonblocking_sync.jl"))
            end

            @testset "arg-slab mid-recording sweep" begin
                include(joinpath(VULKAN_TESTS, "test_argslab_midrecording_sweep.jl"))
            end

            @testset "indirect in concurrent group" begin
                include(joinpath(VULKAN_TESTS, "test_indirect_in_concurrent_group.jl"))
            end

            @testset "indirect in concurrent group" begin
                include(joinpath(VULKAN_TESTS, "test_indirect_in_concurrent_group.jl"))
            end
            include(joinpath(VULKAN_TESTS, "test_crossqueue_sync.jl"))
            # A recorded copy has to outlive the value that was copied FROM, even
            # when nothing holds it. Its own file because no render test reaches it:
            # only device→device from a temporary is affected.
            @testset "copyto! from a dropped source" begin
                include(joinpath(VULKAN_TESTS, "test_copyto_dropped_source.jl"))
            end

            # Who owns a queue from `allocate_batch_queue!`. A dropped one used to
            # let its timeline semaphore be finalized while buffers still named it,
            # and the buffer finalizer's `vkGetSemaphoreCounterValue` then segfaulted
            # inside the driver.
            @testset "batch queue lifetime" begin
                include(joinpath(VULKAN_TESTS, "test_batch_queue_lifetime.jl"))
            end

            # A `transpose(::LavaArray)` destination is not a `LavaArray` and used to
            # miss every `mapreducedim!` method here. Its own file because the
            # assertions have to read the destination's parent storage, which is the
            # only place `adjoint`'s conjugation is visible.
            @testset "mapreducedim! into transposed destinations" begin
                include(joinpath(VULKAN_TESTS, "test_mapreduce_transposed.jl"))
            end

            # Lava's GEMM kernels need a numeric element type; anything else belongs
            # to `GPUArrays.generic_matmatmul!`. Its own file because it defines a
            # struct at top level to be the non-numeric element type.
            @testset "mul! with a non-numeric element type" begin
                include(joinpath(VULKAN_TESTS, "test_mul_nonnumeric_eltype.jl"))
            end

            @testset "compute" begin
                include(joinpath(VULKAN_TESTS, "mwe_alloc_dispatch_free_loop.jl"))
            end

            @testset "RT direct" begin
                include(joinpath(VULKAN_TESTS, "mwe_rt_alloc_dispatch_free_loop.jl"))
            end

            @testset "RT indirect + busy" begin
                include(joinpath(VULKAN_TESTS, "mwe_indirect_rt_busy_loop.jl"))
            end

            @testset "12 distinct kernels" begin
                include(joinpath(VULKAN_TESTS, "mwe_distinct_kernels_per_iter.jl"))
            end

            @testset "SoA workqueue" begin
                include(joinpath(VULKAN_TESTS, "mwe_soa_workqueue_per_iter.jl"))
            end

            @testset "VolPath shape" begin
                include(joinpath(VULKAN_TESTS, "mwe_volpath_shape_per_iter.jl"))
            end

            @testset "GPU-AV clean" begin
                include(joinpath(VULKAN_TESTS, "test_gpuav_clean.jl"))
            end
end

end  # if _VULKAN_OK
