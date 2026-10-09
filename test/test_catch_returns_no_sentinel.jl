# Phase 2 — error surfacing: a failure is an error, never a sentinel value.
#
# A `catch` that answers `typemax(UInt64)` turns "the timeline query failed" into
# "the device is infinitely far ahead", and every caller then reads a lost device
# as finished work. Phase 2 removed them; this keeps them out.
#
# No device: a scan of the source. BOTH packages — the invariant is about error
# handling, which came to Mantle with the runtime, and scanning only Lava after
# the move meant scanning the half that never had the sentinel. Lava is found
# through Mantle's manifest entry, without loading it.
#
# This was `test/vulkan/test_phase2_errors.jl`. Its other two testsets asserted
# that the Vulkan backend's `query_timeline`, `safe_fin_log` and `@vk_checked`
# exist and answer on a healthy device; those are one backend's internals, and
# every device test that waits on the device exercises the query.

using Test, Mantle

@testset "no typemax(UInt64) sentinel in catch bodies" begin
    # Each package's entry file, whose directory is its `src`.
    dirs = [dirname(pathof(Mantle))]
    lava = Base.locate_package(Base.PkgId(Base.UUID("3a680b1f-cb25-4bee-9cf7-bc880b76dc8c"), "Lava"))
    @test lava !== nothing
    lava === nothing || push!(dirs, dirname(lava))
    bad = String[]
    for dir in dirs, (root, _, files) in walkdir(dir), f in files
        endswith(f, ".jl") || continue
        path = joinpath(root, f)
        lines = readlines(path)
        for i in 1:length(lines)-3
            occursin(r"^\s*catch\s*$", lines[i]) || continue
            # look at next non-blank/non-comment line
            j = i + 1
            while j <= length(lines) && occursin(r"^\s*($|#)", lines[j])
                j += 1
            end
            j <= length(lines) || continue
            if occursin(r"typemax\(UInt64\)", lines[j])
                push!(bad, "$path:$i")
            end
        end
    end
    @test isempty(bad)
end
