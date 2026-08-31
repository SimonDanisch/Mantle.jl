using Test, Lava, Mantle
@testset "Phase 2 — error surfacing" begin

@testset "query_timeline exists and returns current counter on healthy device" begin
    ctx = MVE.vk_context()
    bq = ctx.default_bq
    @test isdefined(MVE, :query_timeline)

    # Healthy-device query path: must return a UInt64 without throwing.
    current = MVE.query_timeline(bq)
    @test current isa UInt64
end

@testset "safe_fin_log and @vk_checked exist" begin
    @test isdefined(MVE, :safe_fin_log)
    # `Symbol("@vk_checked")` — a macro, so the plain `isdefined(Lava, :name)`
    # sweep that moved the rest of these to Mantle did not match it.
    @test isdefined(Mantle, Symbol("@vk_checked"))

    # safe_fin_log should not throw on a normal string.
    @test MVE.safe_fin_log("test: safe_fin_log smoke\n") === nothing
end

@testset "no typemax(UInt64) sentinel in catch bodies" begin
    # Static: iterate Lava src files; assert no `catch` is followed directly
    # by a `typemax(UInt64)` return-sentinel.  This is the invariant Phase 2
    # enforces.
    # BOTH packages. The invariant is about error handling, which came to Mantle
    # with the runtime — scanning only Lava after the move meant scanning the
    # half that never had the sentinel.
    srcdirs = [joinpath(dirname(dirname(pathof(m))), "src") for m in (Lava, Mantle)]
    bad = String[]
    for srcdir in srcdirs, (root, _, files) in walkdir(srcdir), f in files
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
