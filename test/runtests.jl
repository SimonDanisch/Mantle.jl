using Mantle, Test


# Relative to this checkout, not to the tree it was first written in: `dev/Mantle`
# and `dev/minimalloc` are siblings wherever the project is checked out, and an
# absolute path here silently reads another tree's benchmarks — or fails on a
# machine that has only one of them.
const BENCH = normpath(joinpath(@__DIR__, "..", "..", "minimalloc", "benchmarks"))

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

@testset "the README instance is solved optimally" begin
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

@testset "placement is valid on every benchmark" begin
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

@testset "the gap to minimalloc's capacity does not grow" begin
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

@testset "lowest fit is not worse than best fit" begin
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

# `import`, not `using`: Mantle exports `Vulkan` as the name of its backend, and
# the package is also called Vulkan, so `using Vulkan` alongside `using Mantle`
# makes the bare name ambiguous. Everything here names it qualified anyway.
import Vulkan

@testset "sync" begin
    be = Mantle.Vulkan()
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
        @test isempty(transitions(Mantle.WebGPU(), img, [ColorAttachment{true}, Sampled, CopySrc]))
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
# Needs a GPU but no display — every graph in it is headless, which is also the
# only kind `bake!` takes — so it runs before the window tests rather than inside
# their DISPLAY guard.
include(joinpath(@__DIR__, "test_arena_bake.jl"))
# Same shape: headless, GPU-only. A `DeviceRange` is the one ndrange whose value
# never reaches the host, so the Host backend cannot pin the half that matters.
include(joinpath(@__DIR__, "test_devicerange.jl"))
include(joinpath(@__DIR__, "test_window.jl"))
