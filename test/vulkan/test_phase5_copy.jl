using Test, Lava, Mantle
@testset "Phase 5 — transfer path unification" begin

@testset "legacy transfer symbols are gone" begin
    @test !isdefined(Lava, :one_shot_copy)
    @test !isdefined(Lava, :append_copy_and_flush!)
    @test !hasfield(Mantle.VulkanBatchQueue, :xfer_cmd_buf)
    @test !hasfield(Mantle.VulkanBatchQueue, :xfer_fence)
end

@testset "cmd_copy_buffer! is the single entry point" begin
    @test isdefined(Mantle, :cmd_copy_buffer!)
end

@testset "upload/download roundtrip via new path" begin
    a = LavaArray{Float32,1}(undef, (8,))
    src = collect(1.0f0:8.0f0)
    Mantle.upload!(a, src)
    result = Array(a)
    @test result == src
end

@testset "GPU→GPU copyto! uses the new path" begin
    src = LavaArray{Int32,1}(undef, (16,))
    Mantle.upload!(src, collect(Int32, 1:16))
    dst = LavaArray{Int32,1}(undef, (16,))
    copyto!(dst, src)
    @test Array(dst) == Array(src)
end

end  # @testset
