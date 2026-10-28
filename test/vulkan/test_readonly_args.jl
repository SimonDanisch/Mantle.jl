# Arguments a dispatch only reads reach Lava as `Lava.Const`, whose loads Lava declares
# NonWritable, unless a written argument reaches the same buffer: NonWritable is a
# promise about the memory, and `x .= x .+ 1` reads and writes one buffer through two
# arguments. Pinned here: which arguments are wrapped, that every launch path still
# computes the same results, and that a wrapped argument packs as the array it wraps,
# which `packdispatch!` relies on to keep recording free of the access analysis.

using Test, Mantle
import Mantle.Lava
import KernelInterface as KI
include(joinpath(@__DIR__, "..", "testbackend.jl"))

if TESTBACKEND isa Mantle.LavaDevice

const M = Mantle

function ro_add_one!(dst, src)
    i = KI.get_global_id().x
    i <= length(dst) && (@inbounds dst[i] = src[i] + 1f0)
    return
end

"The argument types `readonlyargs` hands the compiler for `f` over `args`."
function wrappedtypes(dev, f, args)
    raw = M.resolve(dev, args)
    adapted = M.devargs(M.LavaAdaptor(nothing), raw)
    touches = M.argument_usage(dev, f, args, length(first(args)), nothing)
    return map(typeof, M.readonlyargs(nothing, touches, raw, adapted))
end

isconst(T::Type) = occursin("Lava.Const", string(T))

@noinline ro_source_value!(src, i) = @inbounds src[i]
function ro_changing_source!(dst, src)
    i = KI.get_global_id().x
    i <= length(dst) && (@inbounds dst[i] = ro_source_value!(src, i))
    return nothing
end
ro_launch_changing!(dev, dst, src) = KI.@launch dev ndrange=16 ro_changing_source!(dst, src)

@testset "read-only summaries follow an edited callee" begin
    dev = M.Device(TESTBACKEND)
    dst = M.devicearray(dev, zeros(Float32, 16))
    src = M.devicearray(dev, Float32.(1:16))
    @test M.isreadonly(last(M.kitouches(dev, ro_changing_source!, (dst, src), 16)))
    # The eager fast path must reject the old Summary before deciding to wrap src as Const.
    @eval @noinline function ro_source_value!(src, i)
        @inbounds src[i] += 1f0
        return @inbounds src[i]
    end
    touches = Base.invokelatest(M.kitouches, dev, ro_changing_source!, (dst, src), 16)
    @test !M.isreadonly(last(touches))
    Base.invokelatest(ro_launch_changing!, dev, dst, src)
    @test Array(dst) == Float32.(2:17)
    @test Array(src) == Float32.(2:17)
end

@testset "read-only arguments: which are wrapped" begin
    dev = M.Device(TESTBACKEND)
    a = M.devicearray(dev, zeros(Float32, 16))
    b = M.devicearray(dev, ones(Float32, 16))
    @test wrappedtypes(dev, ro_add_one!, (a, b)) ==
          (Lava.LavaDeviceArray{Float32,1}, Lava.Const{Float32,1})
    # The written buffer read through a second argument is not read-only memory.
    @test !any(isconst, wrappedtypes(dev, ro_add_one!, (a, a)))
    # Nor through a view of it.
    @test !any(isconst, wrappedtypes(dev, ro_add_one!, (a, view(a, 1:16))))
end

@testset "read-only arguments: results on every launch path" begin
    dev = M.Device(TESTBACKEND)
    a = M.devicearray(dev, zeros(Float32, 16))
    b = M.devicearray(dev, Float32.(1:16))

    KI.@launch dev ndrange=16 ro_add_one!(a, b)
    @test Array(a) == Float32.(2:17)
    KI.@launch dev ndrange=16 ro_add_one!(a, a)
    @test Array(a) == Float32.(3:18)

    # Broadcasting is the eager KA path, and the common way into it.
    c = M.devicearray(dev, zeros(Float32, 16))
    c .= b .* 2f0
    @test Array(c) == Float32.(1:16) .* 2
    @test any(k -> occursin("lava_broadcast_flat!", string(k)) && isconst(k),
              keys(M.launch_plan_cache(dev)))
    c .= c .+ b
    @test Array(c) == Float32.(1:16) .* 3

    g = M.Graph(dev)
    out = M.devicearray(dev, zeros(Float32, 16))
    M.dispatch!(g, ro_add_one!, (out, b), 16)
    M.runonce!(g)
    @test Array(out) == Float32.(2:17)
end

@testset "a read-only argument packs as the array it wraps" begin
    # The bytes: a pipeline compiled for `Const` reads the slot `packdispatch!` fills
    # with the unwrapped array. Pointer patch points only matter to a `Recording`, and
    # recording packs that unwrapped array, so they are the array's by construction;
    # `devicepointeroffsets(Const)` is empty (a pointer two struct levels deep), which
    # is why recording must not pack the wrapper.
    for (T, N) in ((Float32, 1), (UInt32, 2), (Float16, 3))
        A, C = Lava.LavaDeviceArray{T,N}, Lava.Const{T,N}
        @test sizeof(C) == sizeof(A)
        @test Base.datatype_alignment(C) == Base.datatype_alignment(A)
        a = A(Ptr{T}(UInt(0x1000)), ntuple(i -> 2i, N))
        @test reinterpret(UInt8, [C(a)]) == reinterpret(UInt8, [a])
    end
end

end # LavaDevice
