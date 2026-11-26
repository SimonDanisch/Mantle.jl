# Every WebGPU backend test of this experiment, on one device.
include(joinpath(@__DIR__, "setup.jl"))
@testset "Mantle WebGPU" begin
    for f in ("first_kernels.jl", "atomics.jl", "arrays.jl", "heaps.jl", "grids.jl")
        include(joinpath(@__DIR__, f))
    end
end
