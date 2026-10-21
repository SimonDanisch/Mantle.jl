# The suite for Mantle's new API (docs/api.md, docs/recording-plan.md,
# docs/internals.jl, docs/examples.jl, docs/resizing-and-raytracing.jl).
#
# Every case has an id (its testset name). A case belongs to the ledger phase that makes it
# pass (refactor-ledger.md); cases of later phases are skipped, so the suite
# is green on every finished phase and grows with the refactor.
#
# Run it in the main environment of the project, in a persistent session
# (never Pkg.test: its --check-bounds=yes changes Lava's codegen):
#
#     include("dev/Mantle/test/suite/runsuite.jl")
#     runsuite(; phase = 3)                       # every case up to phase 3
#     runsuite(; phase = 3, areas = ["ordering"]) # one area
#
# Long cases (fuzz, 10^4 iterations, leaks over many cycles) are tagged `long`
# and run only with `long = true`, at the end of a phase (they are a gate, not
# a step).

using Test, Mantle

include(joinpath(@__DIR__, "helpers.jl"))
include(joinpath(@__DIR__, "faultdevice.jl"))

"""The suite's areas, in the order they run: one file each."""
const AREAS = [
    # core: graphs, ordering, windows, memory
    "devices", "graphs", "mutation", "declaration", "repeat", "when", "host",
    "arena", "run", "entry", "ordering", "inbox", "threads", "pieces",
    "windows", "eviction", "holds", "eager", "reads", "views", "creation", "split",
    # resizing, stores, ray tracing
    "scheduling", "resize", "stores", "copy", "fixed", "lengthof", "coalescing",
    "applying", "changes_during_runs", "free", "tlas", "blas", "gpuref", "textures",
    # consumer patterns, written against the new API only
    "hikari_patterns", "raymakie_patterns", "dnn_patterns", "editor_patterns", "waterlily",
    # gates
    "fuzz",
]

"""
    runsuite(; phase, areas = AREAS, devices = suitedevices(), long = false, driverpairs = ())

Every case of `areas` whose phase is at most `phase`, on each device of
`devices` (a case that needs a capability a device lacks is skipped there).
"""
function runsuite(; phase::Integer, areas = AREAS, devices = suitedevices(), long::Bool = false, driverpairs = ())
    phase >= 0 || throw(ArgumentError("phase must be nonnegative"))
    isempty(devices) && throw(ArgumentError("the suite requires at least one device"))
    isempty(areas) && throw(ArgumentError("the suite requires at least one area"))
    for (cu, vk) in driverpairs
        cu in devices && vk in devices && cu !== vk || throw(ArgumentError("driver pair must name two suite devices"))
        caps(cu).api === :cuda && caps(vk).api === :vulkan || throw(ArgumentError("driverpairs contain (CUDA, Vulkan) fixture pairs"))
    end
    SUITE.phase[] = phase
    SUITE.long[] = long
    @testset "Mantle suite (phase ≤ $phase)" begin
        for dev in devices
            SUITE.device[] = dev
            @testset "$(devicename(dev))" begin
                executed = 0
                for area in areas
                    @testset "$area" begin
                        # Area helpers and constants have overlapping names.
                        # Give each device/area pass its own definitions: a
                        # previous area or a second runsuite call must not
                        # replace a helper's behavior or redefine a struct.
                        scope = Module(gensym(:MantleSuiteArea))
                        Base.include(scope, joinpath(@__DIR__, "helpers.jl"))
                        Base.include(scope, joinpath(@__DIR__, "faultdevice.jl"))
                        config = getfield(scope, :SUITE)
                        config.phase[] = phase
                        config.long[] = long
                        config.device[] = dev
                        config.fixtures[] = (; devices = Tuple(devices), driverpairs = Tuple(driverpairs))
                        Base.include(scope, joinpath(@__DIR__, "areas", area * ".jl"))
                        executed += config.executed[]
                    end
                end
                # Phase 0 writes the specification. Every later full phase
                # gate must execute cases on each joined device; all skips
                # cannot count as a passing implementation gate.
                if phase >= 1 && areas == AREAS
                    @test executed > 0
                end
            end
        end
    end
end
