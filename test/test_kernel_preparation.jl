using Test, Mantle
using KernelAbstractions
import KernelInterface as KI

# The reported array backend and compiler backend can be different (ROCm).
# Conversion can also change the callable type, just as it changes array types.
struct PreparationCompiler
    launches::Vector{Any}
end

KernelAbstractions.@kernel function preparation_group!(out)
    i = @index(Global)
    @inbounds out[i] = 10out[i] + Int32(@groupsize()[1])
end

@testset "array launches share KA construction and preserve group/order" begin
    be = KernelAbstractions.CPU()
    out = zeros(Int32, 64)
    launches = (Mantle.ArrayLaunch(preparation_group!, (out,), 64, 8),
                Mantle.ArrayLaunch(preparation_group!, (out,), 64, 16))
    @test Mantle.runlaunches!(be, launches) === nothing
    @test out == fill(Int32(96), 64)
    fill!(out, 0)
    Mantle.runlaunches!(be, (Mantle.ArrayLaunch(preparation_group!, (out,), 64),))
    @test all(>(0), out)
end
struct PreparationDevice <: Mantle.Device
    compiler::PreparationCompiler
end
struct PreparationBackend
    device::PreparationDevice
end
Mantle.backend(d::PreparationDevice) = PreparationBackend(d)
Mantle.todevice(b::PreparationBackend) = b.device
Mantle.kibackend(d::PreparationDevice) = d.compiler

struct HostCallable
    offset::Float32
end
struct ConvertedCallable
    offset::Float32
end
struct HostArgument
    data::Vector{Float32}
end
struct ConvertedArgument
    data::Vector{Float32}
end
(f::HostCallable)(a) = (a.data .+= f.offset; nothing)
KI.argconvert(::PreparationCompiler, f::HostCallable) = ConvertedCallable(f.offset)
KI.argconvert(::PreparationCompiler, a::HostArgument) = ConvertedArgument(a.data)

struct PreparedKernel{T}
    compiler::PreparationCompiler
    callable::ConvertedCallable
end
function KI.kernel_function(be::PreparationCompiler, f::ConvertedCallable, ::Type{T}) where {T}
    T === Tuple{ConvertedArgument} || error("compiled the host argument ABI")
    return PreparedKernel{T}(be, f)
end
function (k::PreparedKernel{T})(a::HostArgument; ndrange, workgroupsize=nothing) where {T}
    converted = KI.argconvert(k.compiler, a)
    for i in 1:ndrange
        converted.data[i] += k.callable.offset
    end
    push!(k.compiler.launches, (ndrange, workgroupsize))
    return nothing
end

struct UnsupportedPreparationDevice <: Mantle.Device end
Mantle.backend(::UnsupportedPreparationDevice) = :no_ki_backend

@testset "eager and declared kernels use the converted callable and argument ABI" begin
    compiler = PreparationCompiler(Any[])
    dev = PreparationDevice(compiler)
    be = Mantle.backend(dev)
    f = HostCallable(2f0)
    arg = HostArgument(zeros(Float32, 4))
    @test Mantle.kisupported(dev, f)
    # The declared path asks for a kernel now and launches it later.
    kernel = Mantle.kikernel(f, dev, (arg,))
    kernel(arg; ndrange=4, workgroupsize=2)
    @test arg.data == fill(2f0, 4)
    @test compiler.launches == [(4, 2)]
    # Array library eager submission must use that same compiler and ABI.
    @test Mantle.kilaunch!(be, f, arg; ndrange=4, workgroupsize=2) === nothing
    @test arg.data == fill(4f0, 4)
    @test compiler.launches[end] == (4, 2)
    Mantle.kilaunch!(be, f, arg; ndrange=2)
    @test arg.data == Float32[6, 6, 4, 4]
    @test compiler.launches[end] == (2, nothing)
    @test !Mantle.kisupported(UnsupportedPreparationDevice(), f)
    @test_throws r"macro-free kernel" Mantle.kikernel(f, UnsupportedPreparationDevice(), (arg,))
end
