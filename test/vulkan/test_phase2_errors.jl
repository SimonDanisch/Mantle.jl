using Test, Lava, Mantle
@testset "Phase 2 — error surfacing" begin

@testset "query_timeline exists and returns current counter on healthy device" begin
    ctx = Mantle.vk_context()
    bq = ctx.default_bq
    @test isdefined(Lava, :query_timeline)

    # Healthy-device query path: must return a UInt64 without throwing.
    current = Mantle.query_timeline(bq)
    @test current isa UInt64
end

@testset "safe_fin_log and @vk_checked exist" begin
    @test isdefined(Lava, :safe_fin_log)
    @test isdefined(Lava, Symbol("@vk_checked"))

    # safe_fin_log should not throw on a normal string.
    @test Mantle.safe_fin_log("test: safe_fin_log smoke\n") === nothing
end

@testset "no typemax(UInt64) sentinel in catch bodies" begin
    # Static: iterate Lava src files; assert no `catch` is followed directly
    # by a `typemax(UInt64)` return-sentinel.  This is the invariant Phase 2
    # enforces.
    srcdir = dirname(dirname(pathof(Lava))) * "/src"
    bad = String[]
    for (root, _, files) in walkdir(srcdir), f in files
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

end  # @testset
