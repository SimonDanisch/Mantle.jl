using Mantle, Test

include(joinpath(@__DIR__, "backend_probe.jl"))

# Loadable AND registered. Loadability alone stopped being the question when
# VulkanCore, Vulkan and Lava were made to compile NOTHING on a platform with no
# loader: before that a Mac could not precompile Vulkan at all, so
# `backend_loadable` answered `nothing` and this gate held. Now all three import
# cleanly and register no backend — which is the point of them being cheap
# dependencies — and this gate went TRUE, running 130 Vulkan files that every
# one of them errored out of. 5557 errors, on a machine with no Vulkan driver,
# reported as if Mantle were broken.
#
# So ask what `_METAL_OK` asks: is there a usable DEVICE. Mantle's own probe is
# the answer — the Vulkan backend registers only when one is there — and
# `invokelatest` for the same reason the Metal gate needs it, the modules above
# were required in a newer world than this statement.
_VULKAN_OK = backend_loadable("Vulkan") !== nothing &&
             backend_loadable("Lava") !== nothing &&
             :vulkan in Base.invokelatest(Mantle.availablebackends)

# BOTH backends are probed here, before anything runs.
#
# Decided here and not below the portable tests: asked later,
# `Mantle.eachbackend()` is empty when they run, and `foreachbackend` skips
# every one of them with "no usable backend" on a
# machine whose backend was working fine. A probe that runs after the thing it
# gates is not a gate.
global _METAL_OK = let
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

# ONE outer testset around everything, and the reason is the failure mode the
# comment below already describes — from the other side.
#
# `@testset` only throws at the END of the OUTERMOST one. Several of those at
# top level, with bare `include`s between them, means the first one with a
# failure ended the run and everything after it never ran. Measured on
# 2026-08-30: nine errors in `sync` — one ambiguous name — and the 130 Vulkan
# backend files below reported nothing at all, which reads as "the suite
# stopped" when it should read "one testset failed and here is the rest".
#
# Nested, they record and the run continues; the summary at the bottom is still
# one number and the process still exits non-zero. The `const`s inside are
# `global` for it: a testset body is a local scope, and the included files name
# `VULKAN_TESTS` and `_VULKAN_OK`.
@testset "Mantle.jl" begin

# Relative to this checkout, not to the tree it was first written in: `dev/Mantle`
# and `dev/minimalloc` are siblings wherever the project is checked out, and an
# absolute path here silently reads another tree's benchmarks — or fails on a
# machine that has only one of them.
global BENCH = normpath(joinpath(@__DIR__, "..", "..", "minimalloc", "benchmarks"))

# The corpus is an EXTERNAL checkout, and unguarded its absence takes the whole
# suite down: the four testsets below err, the enclosing `@testset "Mantle"`
# throws at its `end`, and the exception propagates out of this file, so the
# `sync` testset and all six GPU test files never run. That reads as "the suite
# failed", not as "six files were skipped", which is the worse of the two
# failure modes by a distance.
#
# Guarded rather than deleted: with the corpus present these are the only thing
# pinning how GOOD the packing is, as opposed to merely legal (see
# `docs/design.md`). Clone it beside this checkout to get them back:
#     git clone https://github.com/google/minimalloc ../../minimalloc
global HAVE_BENCH = isdir(BENCH)
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
# `Vulkan` loads with Mantle but the DRIVER need not be there, so this file is
# reached on a machine that has no loader at all. See `backend_probe.jl` for
# why that must not be fatal.
#
# BOTH, because the backend is the pair: Vulkan is the driver and Lava is the
# Julia→SPIR-V compiler that feeds it. Asking only about Vulkan let the gate open
# on a machine where the compiler was missing — `Mantle.stages` in the `sync`
# testset just below is defined in `src/vulkan/lowering.jl` and needs both.
_VULKAN_OK || @info "Mantle tests: no Vulkan driver and compiler pair; skipping the Vulkan backend and the sync lowering"

# The backend names the test files use WITHOUT qualifying: a file that says plain
# `LavaArray` relies on `using Mantle` exporting it, and Mantle does not export
# its backend's names.
#
# Here rather than file by file, because the files are `include`d into `Main` and
# one import serves all of them — 54 of them name `LavaBackend` alone. Derived by
# parsing every driven test file for free identifiers that ONLY the extension
# defines, so the list is what is actually used and not a guess.
if _VULKAN_OK
    using Mantle: LavaArray, LavaBackend, VulkanFramebuffer, VulkanWindow,
                VulkanCompiledGraphicsPipeline, DebugConfig, copy_framebuffer!,
                fastdiv
end

# The barrier lowering per API marker: host tables, no device and no driver, so they
# run on every machine. Every build compiles the Vulkan tree, the Mac's through the
# loader JLL, and these ask nothing of a driver.
#
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
        # so `Vertices` is a storage read in VERTEX_SHADER and NOT
        # VERTEX_ATTRIBUTE_INPUT / VERTEX_ATTRIBUTE_READ, which describes a fetch
        # that never happens: the stage is survivable, because it precedes
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

    # The Vulkan lowering of what a render pass declares. The behaviour each one
    # leads to (a kept target keeps its pixels, a depth test decides) is asserted
    # on every backend in `test_window.jl`; these are the tables underneath.
    @testset "a traced usage adds the ray-tracing stage and nothing else" begin
        RT = S2(Vulkan.PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR)
        T = Traced{Storage{BufferKind,ReadWrite}}
        S = Storage{BufferKind,ReadWrite}
        for d in (Src(), Dst())
            @test (Mantle.stages(be, T, d) & RT) == RT
            @test (Mantle.stages(be, S, d) & RT) == S2(0)
            @test Mantle.access(be, T, d) == Mantle.access(be, S, d)
        end
    end

    @testset "a load op lowers to its attachment op" begin
        @test Mantle.loadop(Mantle.Keep) == Vulkan.ATTACHMENT_LOAD_OP_LOAD
        @test Mantle.loadop(Mantle.Discard) == Vulkan.ATTACHMENT_LOAD_OP_DONT_CARE
        @test Mantle.loadop(Mantle.Clear((0f0, 0f0, 0f0, 1f0))) == Vulkan.ATTACHMENT_LOAD_OP_CLEAR
    end

    @testset "a Float32 target is depth, by format and by aspect" begin
        @test Mantle.vkformat(be, Float32) == Vulkan.FORMAT_D32_SFLOAT
        @test Mantle.aspect(Float32) == Vulkan.IMAGE_ASPECT_DEPTH_BIT
    end

    # NOT a barrier per buffer: a recording cannot bake a handle for anything that
    # can move, and baking makes everything movable, so a pass barrier is one
    # `VkMemoryBarrier2` per distinct `(waits, to)` hazard and no buffer barriers.
    # The tuples are unique and never unioned: a union would make the shader's
    # writes visible to a copy nothing performs.
    @testset "a pass barrier carries mask tuples, not buffers" begin
        R, W = Storage{BufferKind,ReadOnly}, Storage{BufferKind,WriteOnly}
        # Two copies made visible to a shader read are one hazard; the third, made
        # visible to a shader write, is another.
        ts = [Transition(1, CopyDst, [CopyDst], R), Transition(2, CopyDst, [CopyDst], R),
              Transition(3, CopyDst, [CopyDst], W)]
        dep = Mantle.build_pass_barrier(nothing, ts)
        v = dep.vks
        @test Int(v.memoryBarrierCount) == length(barrierhazards(ts)) == 2
        @test Int(v.bufferMemoryBarrierCount) == 0
        GC.@preserve dep begin
            mb = unsafe_wrap(Array, v.pMemoryBarriers, Int(v.memoryBarrierCount))
            @test allunique((m.srcStageMask, m.srcAccessMask, m.dstStageMask, m.dstAccessMask)
                            for m in mb)
        end
        # Four slices of one buffer, the same hazard each: one barrier.
        same = [Transition(k, CopyDst, [CopyDst], R) for k in 1:4]
        @test Int(Mantle.build_pass_barrier(nothing, same).vks.memoryBarrierCount) == 1
        @test Mantle.build_pass_barrier(nothing, Transition[]) === nothing
    end

    @testset "an image barrier lowers from the state it leaves" begin
        # A depth target re-cleared each frame still waits for the last frame's
        # writes, from an UNDEFINED layout because its contents are discarded.
        D = Depth{ReadWrite,NoAccess,true}
        b = Mantle.ImageBarrier(nothing, Transition(1, D, [D], D))
        @test b.old == Vulkan.IMAGE_LAYOUT_UNDEFINED
        @test b.src_access != Vulkan.AccessFlag2(0)
        @test b.src_stage != Vulkan.PipelineStageFlag2(0)
        # A colour target read back is in the transfer-source layout.
        c = Mantle.ImageBarrier(nothing, Transition(1, ColorAttachment{true},
                                                    [ColorAttachment{true}], CopySrc))
        @test c.new == Vulkan.IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
    end
end

# CPU-only, so they go first and fail fast.
#
# Outside the driver gate on purpose: that they need no GPU is the assertion.
include(joinpath(@__DIR__, "test_pool.jl"))
# The Vulkan half is gated inside on `_VULKAN_OK`.
include(joinpath(@__DIR__, "test_basealignment.jl"))
# Also outside the driver gate, and for a stronger reason than "needs no GPU":
# it reads the extension SOURCE, so it checks the Metal extension's import list
# on a Linux box and the Vulkan one on a Mac. Gating it on a loaded backend
# would confine each half to the machine that cannot be the one to catch it.
# Also needs no GPU: the stages a pipeline is made of, the interfaces between
# them, and the names a shader body reaches the mesh stage through. The emitter
# arithmetic those names lower to is tested in KernelInterface, over its host
# output object.
include(joinpath(@__DIR__, "test_pipeline_stages.jl"))
# Also needs no GPU, and that is the point of the design rather than a happy
# accident: everything the geometry-to-mesh lowering does except read one builtin
# is portable code over an output object, so `runprimitive` plus
# `KernelInterface.HostMeshOutput` runs what a mesh stage runs and shows the
# vertices, indices and per-primitive values a frame can only imply.
include(joinpath(@__DIR__, "test_lowering.jl"))

# The isubd example's pass A (adaptive refinement) against a CPU reference written
# independently of it (`examples/isubd/reference.jl`, vendored from
# koehlerson/mantle_mwe), key sets compared as EXACT integers. Unguarded and not
# under `vulkan/`: it asks `Mantle.defaultbackend()`, so it runs everywhere.
include(joinpath(@__DIR__, "test_isubd_mesh.jl"))
# A MeshPipeline drawn through the GRAPH, and `discard()` declared where both
# backends can reach it. Unguarded for the same reason as the file above: each
# gates itself on the capability, so the same assertions run on every backend
# that has it. Both were found by RayMakie's RASTER mode on a Mac.
include(joinpath(@__DIR__, "test_mesh_pipeline_graph.jl"))
include(joinpath(@__DIR__, "test_discard.jl"))
include(joinpath(@__DIR__, "test_headless_rebindable_draw.jl"))
# A shader drawn after a method it inlines was redefined (Revise).
include(joinpath(@__DIR__, "test_shader_redefinition.jl"))
# A texture from device data, and one updated in place. Found by RayMakie's
# `test_device_arrays.jl`: an `image!` of a device array drew nothing on Metal.
include(joinpath(@__DIR__, "test_texture_upload.jl"))
# More stage arguments than Metal's 31-entry buffer table; RayMakie's mesh
# shader has 51.
include(joinpath(@__DIR__, "test_many_stage_args.jl"))
# A varying spelled `NTuple{N,Float32}`, which Metal takes and Lava had no
# method for.
include(joinpath(@__DIR__, "test_tuple_varyings.jl"))
include(joinpath(@__DIR__, "test_integer_attachments.jl"))
include(joinpath(@__DIR__, "test_core_names_no_backend.jl"))
# The guards for `docs/mantle-owns-it.md`. Mostly `@test_broken`: they are
# written before the refactor deletes anything, so each one fails today and
# turns into an "Unexpectedly Pass" the moment its phase lands.
include(joinpath(@__DIR__, "test_mantle_owns_it.jl"))

include(joinpath(@__DIR__, "test_contact_record.jl"))
include(joinpath(@__DIR__, "test_convex_shape.jl"))
include(joinpath(@__DIR__, "test_gjk.jl"))
include(joinpath(@__DIR__, "test_epa.jl"))
include(joinpath(@__DIR__, "test_backend_vocabulary.jl"))

# And how much of the array algorithms is still Vulkan's — a ratchet
# on where they should live.
include(joinpath(@__DIR__, "test_array_algorithm_portability.jl"))
include(joinpath(@__DIR__, "test_lava_import_completeness.jl"))
# The same boundary from the other side: what each package EXPORTS
# has to match what it defines. Both directions went wrong in the
# move and neither failed at load.
include(joinpath(@__DIR__, "test_no_stale_exports.jl"))

include(joinpath(@__DIR__, "test_geometry_types.jl"))
include(joinpath(@__DIR__, "test_hold_lifetime.jl"))
include(joinpath(@__DIR__, "test_no_ambient_allocation.jl"))

include(joinpath(@__DIR__, "test_submit_channel.jl"))
include(joinpath(@__DIR__, "test_catch_returns_no_sentinel.jl"))
include(joinpath(@__DIR__, "foreachbackend.jl"))

# NOT gated on `_VULKAN_OK` any more. Everything below runs once per backend
# `Mantle.eachbackend()` reports, which on a machine with no Vulkan driver is
# the whole point: these files check PORTABLE behaviour, and gating them on one
# driver is how they came to hardcode `VulkanAPI()` 57 times between them.
# The one Vulkan-specific include below keeps a guard of its own.
# Needs a GPU but no display — every graph in it is headless, which is also the
# only kind `bake!` takes — so it runs before the window tests rather than inside
# their DISPLAY guard.
#
# FIRST among the files that build a plan on the backend's device, and that is a
# constraint rather than a preference: it counts the tenants of that device's
# shared arena, and the device is cached per process, so any earlier file that
# compiled a plan is still a tenant until a GC reaps it. Adding an include above
# this line that builds a plan breaks it.
foreachbackend(joinpath(@__DIR__, "test_arena_recording.jl"))
# What every declaration is derived from. Before the files that build plans,
# because a wrong answer there is a wrong barrier in every one of them, and this
# is the file that says what right looks like. Split in two: the type questions
# need no device, and the walk needs the interpreter a backend compiles with.
include(joinpath(@__DIR__, "test_access.jl"))
foreachbackend(joinpath(@__DIR__, "test_compile_golden.jl"))
# Same shape: headless, GPU-only. A `DeviceRange` is the one ndrange whose value
# never reaches the host, so the Host backend cannot pin the half that matters.
foreachbackend(joinpath(@__DIR__, "test_devicerange.jl"))
foreachbackend(joinpath(@__DIR__, "test_run_ordering.jl"))
# A dispatch's kernel is a plain Julia function, and one core predicate decides
# whether it is that or a `@kernel` constructor. Per-backend because the decision
# was being made three times and the three did not agree.
foreachbackend(joinpath(@__DIR__, "test_declared_kernel.jl"))
# A plain kernel guards its own tail; eight lost the guard with `@kernel` and
# wrote past their outputs. Every backend's launch pads to whole workgroups.
foreachbackend(joinpath(@__DIR__, "test_kernel_tails.jl"))
# And a library call declared as a pass member: `mul!` with no ndrange. The two
# GPU backends answer `runscalls` differently and the file asserts both sides.
foreachbackend(joinpath(@__DIR__, "test_declared_call.jl"))
# How each argument of a dispatch reaches the kernel: which take a slot, which
# bind an allocation, which the compiled kernel has no parameter for at all. One
# rule in core, checked against GPUCompiler's own: two rules in two backends is
# only ever one of them in a build.
foreachbackend(joinpath(@__DIR__, "test_argument_packing.jl"))

# ── device tests from test/vulkan and test/metal, now on every backend ──

foreachbackend(joinpath(@__DIR__, "test_broadcast_paths.jl"))
foreachbackend(joinpath(@__DIR__, "test_diagonal_mul.jl"))
foreachbackend(joinpath(@__DIR__, "test_mul_nonnumeric_eltype.jl"))
foreachbackend(joinpath(@__DIR__, "test_norm_overflow.jl"))
foreachbackend(joinpath(@__DIR__, "test_permutedims.jl"))
foreachbackend(joinpath(@__DIR__, "test_mapreduce_transposed.jl"))
# …and survives a branch: `when!` records every branch once and the
# run submits only the pieces its conditions ask for.
foreachbackend(joinpath(@__DIR__, "test_when.jl"))
foreachbackend(joinpath(@__DIR__, "test_basealignment_device.jl"))
foreachbackend(joinpath(@__DIR__, "test_buffer_alias.jl"))
foreachbackend(joinpath(@__DIR__, "test_download_readback.jl"))
foreachbackend(joinpath(@__DIR__, "test_narrow_phase_contacts.jl"))
foreachbackend(joinpath(@__DIR__, "test_narrow_phase_kernel.jl"))
# Forgetting `free!` is not a leak: every persistent resource and
# every plan carries a finalizer, and the release happens on the
# owning thread at the next `reclaim!`.
foreachbackend(joinpath(@__DIR__, "test_dropped_resources.jl"))
_VULKAN_OK && include(joinpath(@__DIR__, "vulkan", "test_indirect_prepare_ordering.jl"))

# Recording a plan that has already run: the recording is one
# command buffer and belongs to no argument slot.
foreachbackend(joinpath(@__DIR__, "test_record_after_run.jl"))
foreachbackend(joinpath(@__DIR__, "test_record_does_not_execute.jl"))
# Which plans can be recorded at all, and that a render pass and both
# `Update` routes come out the same either way.
foreachbackend(joinpath(@__DIR__, "test_recordable_plans.jl"))
# A recorded plan survives its storage moving: a resized buffer and
# a grown buffers arena are patched, a grown images arena re-records.
foreachbackend(joinpath(@__DIR__, "test_recorded_move_patch.jl"))
# What a recording has to survive between the run that wrote it and
# the runs that submit it: a collection, ad hoc work on the same
# queue, and input rewritten in place.
foreachbackend(joinpath(@__DIR__, "test_recording_lifecycle.jl"))
# The point of all of the above: `run!` of a recorded plan, with
# nothing pending, allocates zero bytes.
foreachbackend(joinpath(@__DIR__, "test_run_allocates_nothing.jl"))
# And where a `DeviceRange`'s workgroup counts live: in the plan, laid out at
# compile beside its arguments, rather than in a slab ring the queue rewinds.
# The path is spelled out because `VULKAN_TESTS` is not bound until further down.
foreachbackend(joinpath(@__DIR__, "test_plan_indirect_ownership.jl"))
foreachbackend(joinpath(@__DIR__, "test_device_selection.jl"))
foreachbackend(joinpath(@__DIR__, "test_fastdiv.jl"))
foreachbackend(joinpath(@__DIR__, "test_fft.jl"))
foreachbackend(joinpath(@__DIR__, "test_gemv.jl"))
# Under `test/vulkan/` and guarded, because it is that backend's: it builds a
# `Mantle.LavaBackend()` and asserts `Mantle.gemv_split`, a rule with no core
# default and no other backend's answer. It sat in the section above, whose
# stated assertion is that its files need no GPU, so on a machine with no Vulkan
# driver it threw from the first line of its testset rather than being skipped.
foreachbackend(joinpath(@__DIR__, "test_gemv_splitk.jl"))
foreachbackend(joinpath(@__DIR__, "test_launch_plan_key.jl"))
foreachbackend(joinpath(@__DIR__, "test_workgroup_zero_init.jl"))
foreachbackend(joinpath(@__DIR__, "test_select_width_mismatch.jl"))
foreachbackend(joinpath(@__DIR__, "test_shared_memory_stress.jl"))
foreachbackend(joinpath(@__DIR__, "test_struct_alignment_systematic.jl"))
foreachbackend(joinpath(@__DIR__, "test_struct_copy_alignment.jl"))
foreachbackend(joinpath(@__DIR__, "test_typepun_trunc_bitcast.jl"))
foreachbackend(joinpath(@__DIR__, "test_multiindex_getindex.jl"))
foreachbackend(joinpath(@__DIR__, "test_double_indirect.jl"))
foreachbackend(joinpath(@__DIR__, "test_const_table_index.jl"))
foreachbackend(joinpath(@__DIR__, "test_coopmat_epilogue.jl"))
foreachbackend(joinpath(@__DIR__, "test_gemm_fp16_accum.jl"))
foreachbackend(joinpath(@__DIR__, "test_gemm_staged.jl"))
foreachbackend(joinpath(@__DIR__, "test_gemm_staged_scalar.jl"))
foreachbackend(joinpath(@__DIR__, "test_bar_memcpy_sync.jl"))

foreachbackend(joinpath(@__DIR__, "test_coopmat_add.jl"))
foreachbackend(joinpath(@__DIR__, "test_coopmat_gemm_subgroup.jl"))
foreachbackend(joinpath(@__DIR__, "test_coopmat_phi_cycle.jl"))
foreachbackend(joinpath(@__DIR__, "test_coopmat_shape.jl"))
foreachbackend(joinpath(@__DIR__, "test_coopmat_shared.jl"))
foreachbackend(joinpath(@__DIR__, "test_gemm_batched.jl"))
foreachbackend(joinpath(@__DIR__, "test_gemm_cm2.jl"))
# Tensor addressing and the cooperative-matrix extensions, on every backend that
# answers `KernelInterface.supports_tensor_addressing` and friends (Vulkan with
# `VK_NV_cooperative_matrix2`); the rest skip on the predicate, not on a name.
# The load: compiling and validating is not enough, the instruction validated twice
# while still wrong, so this runs it and checks the values and the orientation.
foreachbackend(joinpath(@__DIR__, "test_tensor_load.jl"))
# What the load substitutes OUT of range, a value the caller chooses —
# `attn_flash_cm2!` gets a whole row reduction deleted by asking for one.
foreachbackend(joinpath(@__DIR__, "test_tensor_clampvalue.jl"))
# What a PRODUCT of two tensor-loaded operands computes: the load returns the
# transpose of its block, so the product is `P' * Q'`.
foreachbackend(joinpath(@__DIR__, "test_tensor_gemm.jl"))
# A clamping layout bounds-checks the load, so an extent that divides nothing is
# legal and out-of-range reads come back as exact zeros.
foreachbackend(joinpath(@__DIR__, "test_tensor_clamp.jl"))
# A shape the device does NOT report is usable: a 64x16 operand where every reported
# shape has M == 16.
foreachbackend(joinpath(@__DIR__, "test_coopmat_flexible_dims.jl"))
# The other half of the clamp: a clamping layout bounds-checks the STORE too,
# asserted two-sided (in-range elements land, nothing outside moves).
foreachbackend(joinpath(@__DIR__, "test_tensor_store.jl"))
foreachbackend(joinpath(@__DIR__, "test_coopmat_perelement.jl"))
foreachbackend(joinpath(@__DIR__, "test_coopmat_reduce.jl"))
foreachbackend(joinpath(@__DIR__, "test_coopmat_components.jl"))

foreachbackend(joinpath(@__DIR__, "test_aabb_blas_overlap.jl"))
foreachbackend(joinpath(@__DIR__, "test_atomics_and_dispatch.jl"))
foreachbackend(joinpath(@__DIR__, "test_float_conversion_exact.jl"))
foreachbackend(joinpath(@__DIR__, "test_gpuarrays.jl"))
foreachbackend(joinpath(@__DIR__, "test_int32_cartesian_miscompile.jl"))
foreachbackend(joinpath(@__DIR__, "test_logical_pointer_bitcast.jl"))
foreachbackend(joinpath(@__DIR__, "test_loop_unswitch_miscompile.jl"))
foreachbackend(joinpath(@__DIR__, "test_psb_chain_fold.jl"))
foreachbackend(joinpath(@__DIR__, "test_spirv_pattern_correctness.jl"))
foreachbackend(joinpath(@__DIR__, "test_struct_broadcast.jl"))
foreachbackend(joinpath(@__DIR__, "test_vector_pointer_access.jl"))

foreachbackend(joinpath(@__DIR__, "test_trace_closest_hit.jl"))
foreachbackend(joinpath(@__DIR__, "test_rt_pipeline.jl"))
foreachbackend(joinpath(@__DIR__, "test_math_intrinsics.jl"))
foreachbackend(joinpath(@__DIR__, "test_workgroup_struct_accesschain.jl"))
foreachbackend(joinpath(@__DIR__, "test_kernel_print.jl"))
foreachbackend(joinpath(@__DIR__, "test_array_capability_queries.jl"))

foreachbackend(joinpath(@__DIR__, "test_holdleaves_stops_at_tlas.jl"))
foreachbackend(joinpath(@__DIR__, "test_hwtlas_batch_delete.jl"))
foreachbackend(joinpath(@__DIR__, "test_hwtlas_batch_triangles.jl"))
foreachbackend(joinpath(@__DIR__, "test_hwtlas_device.jl"))
foreachbackend(joinpath(@__DIR__, "test_hwtlas_mesh_update.jl"))
foreachbackend(joinpath(@__DIR__, "test_hwtlas_nonblocking_sync.jl"))
foreachbackend(joinpath(@__DIR__, "test_hwtlas_stress.jl"))
foreachbackend(joinpath(@__DIR__, "test_hwtlas_uaf_safety.jl"))
# Instances a kernel writes (`Raycore.instance_buffer`, `Raycore.refit!`), per-ray
# cull masks, and BLAS refit: the portable API that replaced Vulkan's instance
# records and hardcoded 0xFF masks.
foreachbackend(joinpath(@__DIR__, "test_hwtlas_instance_buffer.jl"))
foreachbackend(joinpath(@__DIR__, "test_hwtlas_push_instances.jl"))
foreachbackend(joinpath(@__DIR__, "test_hwtlas_refit.jl"))
foreachbackend(joinpath(@__DIR__, "test_hwtlas_sync_batch.jl"))
foreachbackend(joinpath(@__DIR__, "test_hwtlas_refit_stress.jl"))
foreachbackend(joinpath(@__DIR__, "test_instance_writer_kernel.jl"))
foreachbackend(joinpath(@__DIR__, "test_instance_masks.jl"))
foreachbackend(joinpath(@__DIR__, "test_blas_refit.jl"))
foreachbackend(joinpath(@__DIR__, "test_indexbuffer.jl"))

foreachbackend(joinpath(@__DIR__, "test_workgroup_limit.jl"))
foreachbackend(joinpath(@__DIR__, "test_shared_index_division.jl"))
foreachbackend(joinpath(@__DIR__, "test_shared_vector_access.jl"))
foreachbackend(joinpath(@__DIR__, "test_subgroup_size_pinning.jl"))
foreachbackend(joinpath(@__DIR__, "test_subgroup_shuffle.jl"))
# Workgroup scope, and the two kernels built on it. The GEMM is not
# routed to yet — `coopmat_gemm!` still runs the staged kernel — so this
# is the only thing keeping it honest while it waits to be measured.
foreachbackend(joinpath(@__DIR__, "test_workgroup_scope.jl"))
foreachbackend(joinpath(@__DIR__, "test_static_workgroup.jl"))

foreachbackend(joinpath(@__DIR__, "test_kernelinterface.jl"))

foreachbackend(joinpath(@__DIR__, "test_alloc_budget.jl"))
foreachbackend(joinpath(@__DIR__, "test_argument_memory_isolation.jl"))
foreachbackend(joinpath(@__DIR__, "test_caching_and_allocations.jl"))
foreachbackend(joinpath(@__DIR__, "test_copyto_dropped_source.jl"))
foreachbackend(joinpath(@__DIR__, "test_dispatch_allocation.jl"))
foreachbackend(joinpath(@__DIR__, "test_free_during_recording.jl"))
foreachbackend(joinpath(@__DIR__, "test_gpu_memory_safety.jl"))
foreachbackend(joinpath(@__DIR__, "test_kernel_cache.jl"))
foreachbackend(joinpath(@__DIR__, "test_multitypeset_surgical.jl"))
# Recording a plan big enough to allocate while it is being written.
foreachbackend(joinpath(@__DIR__, "test_no_pool_trim_while_recorded.jl"))
# The other end of the same allocator: what happens when the
# device says no. Only reachable by asking for more VRAM than
# exists, so nothing else covers it.
foreachbackend(joinpath(@__DIR__, "test_pool_alloc_oom.jl"))
foreachbackend(joinpath(@__DIR__, "test_pool_trim.jl"))
foreachbackend(joinpath(@__DIR__, "test_rapid_alloc_free.jl"))
foreachbackend(joinpath(@__DIR__, "test_backend_context.jl"))
foreachbackend(joinpath(@__DIR__, "test_backend_equality.jl"))
foreachbackend(joinpath(@__DIR__, "test_batch_queue_lifetime.jl"))
foreachbackend(joinpath(@__DIR__, "test_device_caps.jl"))
foreachbackend(joinpath(@__DIR__, "test_device_identity.jl"))
foreachbackend(joinpath(@__DIR__, "test_discarded_iteration_prepare.jl"))
foreachbackend(joinpath(@__DIR__, "test_partitioned_recording.jl"))
foreachbackend(joinpath(@__DIR__, "test_phase3_lifecycle.jl"))
foreachbackend(joinpath(@__DIR__, "test_recorded_run_semantics.jl"))
foreachbackend(joinpath(@__DIR__, "test_repeat.jl"))
foreachbackend(joinpath(@__DIR__, "test_repeat_inner_3d.jl"))
foreachbackend(joinpath(@__DIR__, "test_traced_usages.jl"))
foreachbackend(joinpath(@__DIR__, "test_waitidle_submits.jl"))
foreachbackend(joinpath(@__DIR__, "test_graphics_pipeline.jl"))
foreachbackend(joinpath(@__DIR__, "test_video_decode.jl"))

foreachbackend(joinpath(@__DIR__, "test_access_kernels.jl"))
foreachbackend(joinpath(@__DIR__, "test_device_array_wrappers.jl"))
foreachbackend(joinpath(@__DIR__, "test_two_devices.jl"))
foreachbackend(joinpath(@__DIR__, "test_mapreduce_struct_eltypes.jl"))
foreachbackend(joinpath(@__DIR__, "test_uploads_back_to_back.jl"))
foreachbackend(joinpath(@__DIR__, "test_mesh_geometry_draw.jl"))
foreachbackend(joinpath(@__DIR__, "test_texture_sampling.jl"))
foreachbackend(joinpath(@__DIR__, "test_render_graph.jl"))
foreachbackend(joinpath(@__DIR__, "test_trace_hwtlas.jl"))
foreachbackend(joinpath(@__DIR__, "test_trace_hit_record.jl"))
foreachbackend(joinpath(@__DIR__, "test_trace_procedural.jl"))
foreachbackend(joinpath(@__DIR__, "test_recorded_plan_runs.jl"))
foreachbackend(joinpath(@__DIR__, "test_pool_on_device.jl"))
foreachbackend(joinpath(@__DIR__, "test_stored_device_array.jl"))

# ── the window tests, in their own process, on a clock ────────────────────────
#
# `test_window.jl` can BLOCK, and its own header says so and says to run it alone
# with a timeout. It was `include`d here anyway, which is the worst of the two:
# measured 2026-08-30, a full run reached it, sat in `glfwCreateWindow` at 100 %
# CPU and never came back — and because it produces no output before the hang and
# everything before it is buffered, the run looks idle rather than stuck. That
# cost an hour and a half before anyone looked at a stack.
#
# The hang is not Mantle's: on this XWayland session a VISIBLE GLFW window never
# receives its `VisibilityNotify`, so `_glfwCreateWindowX11` spins forever.
# Hidden windows take 0.07 s. GLMakie's precompile workload dies the same way.
#
# A subprocess with a deadline turns that into a reported failure, which is what
# the file is worth: skipping it is how it rotted through the runtime move
# unnoticed, and blocking on it is how the rest of the suite stops existing.
# Once per backend, like the portable files above — a window, a surface, a
# resize and a presentation are the same question on every backend, and this
# file asked it on Vulkan alone for 2,021 lines.
# BOTH window files go in a child with a deadline, and the portable one is here
# rather than in `foreachbackend` above for the reason the note gives: it opens
# a window, so it can block, and a file that can block must not run in this
# process. It did, and the whole suite stopped at it — 100 % of one core inside
# `_glfwCreateWindowX11`, uninterruptible (a SIGINT cannot land inside a C call),
# with everything before it still in the stdout buffer. One child per FILE, so a
# hang in one does not hide what the other would have said.
@testset "windows (separate process): $(nameof(typeof(WINDOW_BE)))" for WINDOW_BE in Mantle.eachbackend()
    # Both files on every backend, each in a child of its own with a deadline, so a
    # hang in the long one does not hide the short one's results.
    for wf in ("test_window_portable.jl", "test_window.jl")
        log = joinpath(mktempdir(), "window.log")
        # The child picks the backend by NAME and re-derives the object, because a
        # backend object does not survive being interpolated into a command line.
        #
        # And it has to LOAD one first: `using Mantle` alone registers nothing —
        # a backend arrives with its package, which is what `backend_probe.jl`
        # is for on this side too. Without it `eachbackend()` is empty in the
        # child and `only` says "Collection is empty" from a process whose
        # output nobody reads as being about a missing backend.
        # `repr` of a STRING, not of the Symbol `nameof` answers: the child
        # compares `string(nameof(typeof(b)))` against this, and a `String` is
        # never `==` a `Symbol` — the comparison read false for every backend
        # and the child died on an empty `only`.
        bename = string(nameof(typeof(WINDOW_BE)))
        child = """using Mantle, Test
            include($(repr(joinpath(@__DIR__, "backend_probe.jl"))))
            foreach(backend_loadable, ("Lava", "Metal"))
            found = [b for b in Mantle.eachbackend() if string(nameof(typeof(b))) == $(repr(bename))]
            isempty(found) && error("child: no backend named $bename here; loaded: " *
                                    string(collect(Mantle.eachbackend())))
            @eval Main MANTLE_TEST_BACKEND = \$(only(found))
            include($(repr(joinpath(@__DIR__, wf))))"""
        cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) -e $child`
        proc = run(pipeline(cmd; stdout = log, stderr = log); wait = false)
        deadline = time() + 600
        while process_running(proc) && time() < deadline
            sleep(1)
        end
        blocked = process_running(proc)
        blocked && kill(proc, Base.SIGKILL)
        wait(proc)
        blocked && @warn """$wf did not finish in 600 s — almost certainly \
                            blocked creating a visible window. This is a display-stack \
                            problem, not a Mantle one; see the note above."""
        # Not `success(...)`: a failure has to say what failed, and the child's
        # output is the only place that is written down.
        ok = !blocked && success(proc)
        ok || println(read(log, String))
        @test ok
    end
end

# ── the Metal backend ─────────────────────────────────────────────────────────
#
# Gated on Metal being loadable AND functional, not on the OS: a Mac without a
# usable GPU (a CI container, a VM) must skip rather than error, and `Metal` is
# resolvable on any platform while `Metal.functional()` is the honest question.
#
# These sat in `test/metal/` without being included by anything, which meant
# every regression they pin was unguarded. Two of them were failing when they were finally run.
global METAL_TESTS = joinpath(@__DIR__, "metal")

# `_METAL_OK` is decided at the top, beside `_VULKAN_OK`.

if _METAL_OK
    @testset "Metal backend" begin
        for f in ("test_kernelinterface_metal.jl",)
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
global VULKAN_TESTS = joinpath(@__DIR__, "vulkan")

if _VULKAN_OK
@testset "Vulkan backend" begin
        # The automatic trim, its rate limit and its GC budget: Vulkan's
        # `pool_alloc` policy. Core's pool trims only when asked (`test_pool_trim.jl`).
        @testset "pool trim policy" begin
            include(joinpath(VULKAN_TESTS, "test_pool_trim_policy.jl"))
        end

        # ── Tier 3a: Workgroup barrier-skip fix (GPU; catches lavapipe deadlock) ──
        @testset "Tier 3a: Barrier skip fix" begin
            include(joinpath(VULKAN_TESTS, "test_barrier_skip.jl"))
        end

        # ── Tier 3: GPU Execution ──
        # Gone. The compiler halves of the files that were here — hand-built SPIR-V
        # modules, `@lava_printf` emission, source maps and compile errors — are in
        # Lava's own suite, which needs no device; BLAS refit and instance masks are
        # portable (`test_blas_refit.jl`, `test_instance_masks.jl` in `test/`).

        @testset "pipeline cache avoids driver compilation" begin
            include(joinpath(VULKAN_TESTS, "test_pipeline_cache_no_compile.jl"))
        end

        # Straight after the reset above, and before `test_static_workgroup.jl` —
        # which is where this crashed, because that file calls `GC.gc()` explicitly
        # and so ran whichever finalizer the reset had stranded.
        @testset "a buffer may outlive a device reset" begin
            include(joinpath(VULKAN_TESTS, "test_device_reset_finalizer.jl"))
        end

        # Two live devices in one process — the real GPU and lavapipe, which the
        # loader enumerates together, so this needs no second card. Asserts both
        # compute correctly AND that one kernel compiles twice: a shared pipeline
        # can still give the right answer by luck. See the file for the five
        # separate pieces of module-scope device state it found.
        @testset "two devices in one process" begin
            # Two devices computing side by side: `test_two_devices.jl`, on every
            # backend. What stays here is the exit below.
            # And that the process can then EXIT. A passing probe is not enough:
            # the crash is in the shutdown finalizer sweep, after every summary
            # has printed. Nothing inside this process can observe that, so the
            # check
            # is a subprocess and an exit code.
            include(joinpath(VULKAN_TESTS, "test_twodevice_shutdown.jl"))
        end

        # What baking is FOR, and three things it silently was not: it executed
        # the plan as it recorded it, it froze `Ref` arguments at capture, and it
        # pinned one argument slot for life so the repack that fixed the second
        # raced the device. All three are numbers a run produces, so both files
        # assert on buffer contents rather than on bookkeeping.
        # A latent fault in the arena allocator that only a second driver could
        # show: the memory was not allocated for device addresses, and NVIDIA
        # returns a usable address anyway. Asserted through the validation layer,
        # which is the loader's and so reports it on every GPU.
        @testset "arena memory is allocated for device addresses" begin
            include(joinpath(VULKAN_TESTS, "test_arena_memory_device_address.jl"))
        end

        @testset "recording" begin

            include(joinpath(VULKAN_TESTS, "test_alloc_debug_log.jl"))
            # No open command buffer on the queue: every call closes and submits
            # what it wrote, and a run is one submission.
            include(joinpath(VULKAN_TESTS, "test_closed_command_buffers.jl"))
        end

        # A handled allocation failure must absorb its own validation messages, or it
        # aborts whatever unrelated code calls check_validation_errors! next.
        @testset "tolerated alloc failure" begin
            include(joinpath(VULKAN_TESTS, "test_tolerated_alloc_failure.jl"))
        end

        # And what the driver says about a pipeline is asked of the context that
        # made it, not of a global flag that only describes the next one.
        @testset "pipeline executable properties per context" begin
            include(joinpath(VULKAN_TESTS, "test_pipeline_exec_ir.jl"))
        end

        # `test_phase6_graphics.jl` was here, and it is deleted with the thing
        # it tested: a static source check that `vk_draw!` did not hand-roll its
        # dispatch barrier. `vk_draw!` is gone, and the property it asserted now
        # holds structurally — every draw goes through `begin_pass!`, which is
        # where the barrier lives.

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

            @testset "debug configuration" begin
                include(joinpath(VULKAN_TESTS, "test_debug_config.jl"))
            end

            include(joinpath(VULKAN_TESTS, "test_crossqueue_sync.jl"))

            @testset "GPU-AV clean" begin
                include(joinpath(VULKAN_TESTS, "test_gpuav_clean.jl"))
            end
end

end  # if _VULKAN_OK

end  # @testset "Mantle.jl"
