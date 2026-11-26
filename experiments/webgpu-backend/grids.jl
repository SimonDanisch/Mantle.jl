# Grids of more workgroups than a WebGPU dimension holds (65535): a `@kernel`'s is
# split for Lava's index recovery, a plain KernelInterface kernel is printed for the
# split 1-D grid.
import KernelAbstractions as KA
import KernelInterface as KI
using Test

function grids_ki!(a)
    i = KI.get_global_id().x
    i <= length(a) && (@inbounds a[i] = i % Int32)
    return nothing
end

KA.@kernel function grids_ka!(a)
    i = KA.@index(Global, Linear)
    @inbounds a[i] = i % Int32
end

@testset "grids past 65535 workgroups" begin
    n = 70_001 * 64 + 5                      # 70002 groups of 64
    a = KA.zeros(dev, Int32, n)
    KI.Kernel(dev, grids_ki!)(a; ndrange = n, workgroupsize = 64)
    @test Array(a) == Int32.(1:n)
    b = KA.zeros(dev, Int32, n)
    grids_ka!(dev, 64)(b; ndrange = n)
    @test Array(b) == Int32.(1:n)
end
