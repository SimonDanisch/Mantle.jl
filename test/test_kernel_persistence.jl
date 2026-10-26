# A kernel compiled in one process is not compiled again in the next.
#
# Mantle guarantees it of every backend (`kernelcompiles`); each backend's compiler
# keeps what it compiles in CompilerCaching's object cache, keyed by the code and
# not by the session's `CodeInstance`. Two fresh processes launch the same kernel:
# the first compiles it, which shows the kernel is one no run has seen (`K` is a
# literal drawn per run, the same in both), and the second compiles nothing.

using Test, Mantle, KernelAbstractions
include(joinpath(@__DIR__, "testbackend.jl"))
include(joinpath(@__DIR__, "freshprocess.jl"))

@testset "a kernel compiled in one process is not compiled in the next" begin
    K = rand(10_000:99_999)
    code = """
        @kernel function kp_scale!(d, ::Val{K}) where {K}
            i = @index(Global)
            @inbounds d[i] = d[i] * 2.0f0 + Float32(K)
        end
        dev = Mantle.Device(BACKEND)
        a = Mantle.devicearray(BACKEND, ones(Float32, 64))
        KA.synchronize(BACKEND)
        Mantle.resetkernelcompiles!(dev)
        kp_scale!(BACKEND, 64)(a, Val($K); ndrange = 64)
        KA.synchronize(BACKEND)
        @testset "launch" begin
            @test Array(a) == fill(2.0f0 + $K, 64)
        end
        println("KERNEL COMPILES: ", Mantle.kernelcompiles(dev).misses)
        """
    compiles(out) = parse(Int, match(r"KERNEL COMPILES: (\d+)", out)[1])
    first_status, first_out = infreshprocess(code)
    reportfresh(first_status, first_out)
    second_status, second_out = infreshprocess(code)
    reportfresh(second_status, second_out)
    @test first_status == second_status == 0
    @test compiles(first_out) >= 1
    @test compiles(second_out) == 0
end
