# GPUArrays and AcceleratedKernels on Mantle's WebGPU backend: their generic,
# KernelAbstractions-based implementations, unchanged.
isdefined(Main, :dev) || include(joinpath(@__DIR__, "setup.jl"))
import AcceleratedKernels as AK

upload(x::Array{T}) where {T} = copyto!(KA.allocate(dev, T, size(x)...), x)

@testset "WebGPU arrays" begin
    a, b = rand(Float32, 1000), rand(Float32, 1000)
    da, db = upload(a), upload(b)
    @testset "broadcast" begin
        @test Array(da .+ db .* 2f0) ≈ a .+ b .* 2f0
        @test Array(sin.(da) .+ Ref(1f0)) ≈ sin.(a) .+ 1f0
        dc = similar(da)
        dc .= max.(da, db)
        @test Array(dc) == max.(a, b)
        M = rand(Float32, 30, 40)
        dM = upload(M)
        @test Array(dM .* reshape(upload(collect(Float32, 1:40)), 1, 40)) ≈ M .* reshape(1:40, 1, 40)
    end
    @testset "map and copies" begin
        @test Array(map(x -> x * x, da)) ≈ a .* a
        @test Array(copy(da)) == a
    end
    @testset "reductions" begin
        @test sum(da) ≈ sum(a)
        @test maximum(da) == maximum(a)
        M = rand(Float32, 64, 33)
        dM = upload(M)
        @test Array(sum(dM; dims = 1)) ≈ sum(M; dims = 1)
        @test Array(sum(dM; dims = 2)) ≈ sum(M; dims = 2)
        @test any(da .> 0.99f0) == any(a .> 0.99f0)
    end
    @testset "AcceleratedKernels" begin
        x = rand(Float32, 10_000)
        dx = upload(x)
        @test Array(AK.accumulate(+, dx; init = 0f0)) ≈ accumulate(+, x)
        s = upload(copy(x))
        AK.sort!(s)
        @test Array(s) == sort(x)
        i = rand(Int32(1):Int32(100), 5000)
        @test AK.reduce(+, upload(i); init = Int32(0)) == sum(i)
    end
end
