using Test, Mantle
using KernelAbstractions
import KernelInterface as KI

# An array can report its own package's backend, which resolves to its device
# (Metal.jl's `MetalBackend`, AMDGPU.jl's `ROCBackend`); the device compiles.
# Conversion can also change the callable type, just as it changes array types.

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
    launches::Vector{Any}
end
struct PreparationBackend
    device::PreparationDevice
end
Mantle.todevice(b::PreparationBackend) = b.device

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
KI.argconvert(::PreparationDevice, f::HostCallable) = ConvertedCallable(f.offset)
KI.argconvert(::PreparationDevice, a::HostArgument) = ConvertedArgument(a.data)

struct PreparedKernel{T}
    device::PreparationDevice
    callable::ConvertedCallable
end
function KI.kernel_function(d::PreparationDevice, f::ConvertedCallable, ::Type{T}) where {T}
    T === Tuple{ConvertedArgument} || error("compiled the host argument ABI")
    return PreparedKernel{T}(d, f)
end
function (k::PreparedKernel{T})(a::HostArgument; ndrange, workgroupsize=nothing) where {T}
    converted = KI.argconvert(k.device, a)
    for i in 1:ndrange
        converted.data[i] += k.callable.offset
    end
    push!(k.device.launches, (ndrange, workgroupsize))
    return nothing
end

# A device that answers no KernelInterface verb.
struct UnsupportedPreparationDevice <: Mantle.Device end

@testset "eager and declared kernels use the converted callable and argument ABI" begin
    dev = PreparationDevice(Any[])
    be = PreparationBackend(dev)
    f = HostCallable(2f0)
    arg = HostArgument(zeros(Float32, 4))
    @test Mantle.kisupported(dev, f)
    # The declared path asks for a kernel now and launches it later.
    kernel = Mantle.kikernel(f, dev, (arg,))
    kernel(arg; ndrange=4, workgroupsize=2)
    @test arg.data == fill(2f0, 4)
    @test dev.launches == [(4, 2)]
    # Array library eager submission must use that same compiler and ABI.
    @test Mantle.kilaunch!(be, f, arg; ndrange=4, workgroupsize=2) === nothing
    @test arg.data == fill(4f0, 4)
    @test dev.launches[end] == (4, 2)
    Mantle.kilaunch!(be, f, arg; ndrange=2)
    @test arg.data == Float32[6, 6, 4, 4]
    @test dev.launches[end] == (2, nothing)
    @test !Mantle.kisupported(UnsupportedPreparationDevice(), f)
    @test_throws r"macro-free kernel" Mantle.kikernel(f, UnsupportedPreparationDevice(), (arg,))
end

struct ContextPreparationDevice <: Mantle.Device end
struct ConvertedAccessContext{C} end
Mantle.argtype(::ContextPreparationDevice, ctx) = ConvertedAccessContext{typeof(ctx)}
Mantle.kernelinterpreter(::ContextPreparationDevice, f, tt) = tt

@testset "KA access signatures use the converted context ABI" begin
    dev = ContextPreparationDevice()
    for (dims,group) in ((64,nothing),(64,8),((16,16),(4,4)))
        interp,body,tt = Mantle.kakernelaccesssignature(
            dev,preparation_group!,(Vector{Int32},),dims,group)
        @test tt[1] <: ConvertedAccessContext
        @test tt[2] === Vector{Int32}
        @test interp === Tuple{tt...}
        @test body === Mantle.kernelfor(preparation_group!,group,dev).f
    end
end
