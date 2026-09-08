using Test, Lava, Mantle
@testset "Phase 6 — a draw emits into the frame's command buffer" begin

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

    # Extract the vk_draw! body (from the function header to the next function).
    # It takes the frame's emitter: the one-shot or recording it writes into
    # opened with the global head barrier that orders it behind earlier work,
    # so there is nothing for the draw to synchronise by hand.
    m = match(r"function vk_draw!\(e::Emitter,(.*?)(?=\nfunction )"s, src)
    @test m !== nothing
    vk_draw_body = m.captures[1]

    # Old pattern: a standalone MemoryBarrier(C_NULL, ...) for dispatch sync.
    # Only ImageMemoryBarrier (a depth attachment's layout transition) remains.
    @test !occursin(r"Vulkan\.MemoryBarrier\(C_NULL", vk_draw_body)
    @test !occursin(r"VK\.MemoryBarrier\(C_NULL", vk_draw_body)
end

end  # @testset
