# The suite for Mantle's new API (docs/api.md, docs/recording-plan.md,
# docs/internals.jl, docs/resizing-and-raytracing.jl, decisions B2/B3).
#
# Every case comes from the two catalogues in
# experiments/plan-review/catalogue-core.md and catalogue-resize-rt.md and
# carries its catalogue id. A case belongs to the ledger phase that makes it
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
    # core: graphs, ordering, windows, memory (catalogue-core.md)
    "devices", "graphs", "mutation", "declaration", "repeat", "when", "host",
    "arena", "run", "entry", "ordering", "inbox", "threads", "pieces",
    "windows", "eviction", "holds", "eager", "reads", "views", "creation", "split",
    # resizing, stores, ray tracing (catalogue-resize-rt.md)
    "scheduling", "resize", "stores", "copy", "fixed", "lengthof", "coalescing",
    "applying", "changes_during_runs", "free", "tlas", "blas", "gpuref", "textures",
    # consumer patterns, written against the new API only
    "hikari_patterns", "raymakie_patterns", "dnn_patterns", "editor_patterns", "waterlily",
    # gates
    "fuzz",
]

"""
    runsuite(; phase, areas = AREAS, devices = suitedevices(), long = false)

Every case of `areas` whose phase is at most `phase`, on each device of
`devices` (a case that needs a capability a device lacks is skipped there).
"""
function runsuite(; phase::Integer, areas = AREAS, devices = suitedevices(), long::Bool = false)
    SUITE.phase[] = phase
    SUITE.long[] = long
    @testset "Mantle suite (phase ≤ $phase)" begin
        for dev in devices
            SUITE.device[] = dev
            @testset "$(devicename(dev))" begin
                for area in areas
                    @testset "$area" begin
                        include(joinpath(@__DIR__, "areas", area * ".jl"))
                    end
                end
            end
        end
    end
end
