using Test, Lava, Mantle
@testset "Phase 6 — graphics dispatch routed through record_dispatch!" begin

@testset "vk_draw! no longer hand-rolls its dispatch barrier" begin
    # Static check: vk_draw!'s source should not contain the manual
    # `MemoryBarrier(...) + cmd_pipeline_barrier(...)` pattern that handled
    # prior-dispatch → graphics sync.  Image-layout transitions (depth
    # attachment, transition_image! helper) are still allowed.
    # `graphics/pipeline.jl` came to Mantle with the runtime, under `src/vulkan/`.
    # Read from source because the assertion is about what `vk_draw!` does NOT
    # contain, which no runtime observation can show.
    src = read(joinpath(dirname(pathof(Mantle)),
                        "vulkan", "graphics", "pipeline.jl"), String)

    # Extract the vk_draw! body (from the function header to its matching end)
    m = match(r"function vk_draw!\(bq::VulkanBatchQueue,(.*?)(?=\nfunction )"s, src)
    @test m !== nothing
    vk_draw_body = m.captures[1]

    # Old pattern: a standalone MemoryBarrier(C_NULL, ...) for dispatch sync.
    # New code only uses ImageMemoryBarrier (for image layout transitions).
    @test !occursin(r"Vulkan\.MemoryBarrier\(C_NULL", vk_draw_body)

    # vk_draw! must go through record_dispatch!
    @test occursin("record_dispatch!(bq", vk_draw_body)
end

end  # @testset
