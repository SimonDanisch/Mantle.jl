using Test, Lava, Mantle
@testset "Phase 4 — single-writer enforcement + GC counter fix" begin

@testset "VulkanBatchQueue has owning_thread set to construction thread" begin
    bq = MVE.vk_context().default_bq
    @test hasfield(MVE.VulkanBatchQueue, :owning_thread)
    @test bq.owning_thread == Threads.threadid()
end

@testset "cross-thread dispatch trips the assert" begin
    # Only meaningful under `julia -t N` with N > 1.
    if Threads.nthreads() > 1
        bq = MVE.vk_context().default_bq
        result = Ref{Any}(nothing)
        # Open a one-shot from a different thread.
        #
        # `Threads.@spawn begin ... end |> wait` does NOT wait: a macro consumes
        # as much of the expression as it can, so that parses as
        # `@spawn (begin ... end |> wait)` — the wait runs INSIDE the task and
        # nothing joins it. `result[]` was then read before the task had run and
        # came back `nothing`, failing against a Lava that behaves correctly.
        task = Threads.@spawn begin
            try
                MVE.oneshot(bq) do e end
                result[] = :no_assert   # should not happen
            catch e
                result[] = e
            end
        end
        wait(task)
        @test result[] isa AssertionError
    else
        @info "Skipping cross-thread test; run with `julia -t 2+` to exercise"
    end
end

@testset "live_bytes is atomic" begin
    @test MVE.mempolicy(MVE.vk_context()).live_bytes isa Threads.Atomic{Int}
end

@testset "VkContext has no public nothing-default_bq path" begin
    # The inner constructor takes a raw primary queue and builds default_bq
    # itself; there's no way to get a VkContext with default_bq unset after
    # the constructor returns.
    #
    # `<:`, not `===`. The assertion is that the field admits no `nothing` — it
    # was written as identity against the bare `VulkanBatchQueue`, which also pinned
    # the field to the UnionAll and broke when `VulkanBatchQueue` gained its `{C}`
    # parameter. `VulkanBatchQueue{VkContext}` satisfies the intent MORE strongly (it
    # is concrete), so test the property, not one spelling of it.
    T = fieldtype(MVE.VkContext, :default_bq)
    @test T <: MVE.VulkanBatchQueue
    @test !(Nothing <: T)
end

end  # @testset
