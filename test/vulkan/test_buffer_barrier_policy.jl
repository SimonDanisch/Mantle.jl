using Test, Mantle

@testset "native global buffer barrier policy" begin
    @test Mantle.global_buffer_barriers(Mantle.VulkanAPI())
    @test Mantle.global_buffer_barriers(Mantle.MetalAPI())
    @test !Mantle.global_buffer_barriers(Mantle.WebGPUAPI())
    @test !Mantle.global_buffer_barriers(Mantle.ROCmAPI())
end
