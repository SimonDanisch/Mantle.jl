# `KernelAbstractions.@print` in a kernel compiles, and the kernel still computes.
#
# `@print` is the portable way to print from a kernel: KA lowers it to
# `KernelAbstractions.__print`, which each compiler overrides for the device —
# Lava routes it to `NonSemantic.DebugPrintf` (`device/printf.jl`), Metal.jl to
# `os_log` through `KI._print`. Both build the format from the argument types
# when the kernel compiles (Lava throws for a type it has no specifier for), so
# the scalar kinds Lava's printf test prints (32- and 64-bit integers, Float32,
# Float64) are printed here. That the format reaches the module is Lava's
# `test/spirv/test_printf.jl`, on the SPIR-V.
#
# What is printed is not checked. Seeing it needs the backend's own channel —
# on Vulkan the validation layer's debug-printf, which resets the device — and
# there is no portable way to read it back. What a caller does see is whether a
# kernel with a print in it still compiles and still writes its result.

using Test, Mantle, KernelAbstractions
include(joinpath(@__DIR__, "testbackend.jl"))
const KA = KernelAbstractions

@kernel function print_scalars!(out)
    i = @index(Global)
    KernelAbstractions.@print("tid=", UInt32(i), " i=", Int32(i), " big=", Int64(i) * 3,
                              " val=", Float32(i) * 10f0, "\n")
    @inbounds out[i] = Float32(i)
end

@kernel function print_literal!(out)
    i = @index(Global)
    KernelAbstractions.@print("hello from a kernel\n")
    @inbounds out[i] = Float32(2i)
end

@kernel function print_float64!(out)
    i = @index(Global)
    KernelAbstractions.@print("d=", Float64(i), "\n")
    @inbounds out[i] = Float64(i)
end

@testset "a kernel that prints still computes" begin
    n = 4
    out = Mantle.devicearray(TESTBACKEND, zeros(Float32, n))
    print_scalars!(TESTBACKEND)(out; ndrange = n)
    KA.synchronize(TESTBACKEND)
    @test Array(out) == Float32[1, 2, 3, 4]

    out = Mantle.devicearray(TESTBACKEND, zeros(Float32, n))
    print_literal!(TESTBACKEND)(out; ndrange = n)
    KA.synchronize(TESTBACKEND)
    @test Array(out) == Float32[2, 4, 6, 8]

    # A Float64 argument only where the device computes in Float64 at all.
    for T in testeltypes(Float64)
        out = Mantle.devicearray(TESTBACKEND, zeros(T, n))
        print_float64!(TESTBACKEND)(out; ndrange = n)
        KA.synchronize(TESTBACKEND)
        @test Array(out) == T[1, 2, 3, 4]
    end
end
