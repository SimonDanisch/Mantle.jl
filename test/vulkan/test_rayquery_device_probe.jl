"""
Does this driver actually expose VK_KHR_ray_query?

Split out of Lava's `spirv/test_rayquery.jl` on 2026-08-27. Everything else in
that file reads `ctx.features` — a record with no Vulkan in it — and so
runs on a machine with no driver at all. This testset reads a `VkContext` field,
which is exactly what could not come along.

Worth keeping rather than folding into the emitter tests: the emitter's guard is
only as good as what the probe reports, and a driver that gains or loses the
extension changes what every ray-query kernel compiles to.
"""

using Test, Mantle, Lava

@testset "Ray Query - device probe" begin
    ctx = MVE.vk_context()
    @test hasfield(typeof(ctx), :ray_query_available)

    # Soft check: if the RT pipeline is available, ray query almost certainly is
    # too — RADV, lavapipe, NVIDIA and AMDGPU-Windows all support it.
    if ctx.rt_pipeline_properties !== nothing
        @test ctx.ray_query_available == true
    end

    # And what the probe found is what the emitter was told. `bind_context!` sets
    # both together; this is the join between the two files.
    @test ctx.features.ray_query === ctx.ray_query_available
end
