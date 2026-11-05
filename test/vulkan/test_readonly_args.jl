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

    alias_graph = M.Graph(dev)
    M.dispatch!(alias_graph, ro_add_one!, (a, view(a, 1:16)), 16)
    M.runonce!(alias_graph)
    @test Array(a) == Float32.(4:19)

    recorded_graph = M.Graph(dev)
    M.dispatch!(recorded_graph, ro_add_one!, (a, a), 16)
    plan = M.record!(M.Plan(recorded_graph))
    try
        M.run!(plan)
        @test Array(a) == Float32.(5:20)
        M.run!(plan)
        @test Array(a) == Float32.(6:21)
    finally
        M.free!(plan)
    end
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

@testset "planned read-only arguments preserve storage aliases" begin
    dev = M.Device(TESTBACKEND)
    graph = Mantle.Graph(dev)
    a = Mantle.devicearray(dev, zeros(Float32, 16))
    b = Mantle.devicearray(dev, ones(Float32, 16))
    c = Mantle.devicearray(dev, fill(2f0, 16))
    adapt = Mantle.LavaAdaptor(nothing)
    isconst(x) = x isa Lava.Const
    function wrapped(raw, touches)
        adapted = Mantle.devargs(adapt, raw)
        return Mantle.readonlyargs(graph, touches, raw, adapted)
    end

    # The aliases include views and a nested aggregate. Aliasing one member
    # blocks the entire aggregate from Const, but leaves independent reads alone.
    for aliased in (a, view(a, 1:16), view(a, 9:16))
        raw = (a, (source = aliased, other = b), b)
        touches = [Mantle.WRITE, Mantle.READ, Mantle.READ]
        actual = wrapped(raw, touches)
        @test !isconst(actual[1])
        @test !isconst(actual[2].source)
        @test !isconst(actual[2].other)
        @test isconst(actual[3])
    end

    # Every written argument participates, including ones after the read.
    for raw in ((a, b, a, b, c), (a, b, view(a, 1:8), view(b, 9:16), c))
        actual = wrapped(raw, [Mantle.WRITE, Mantle.WRITE, Mantle.READ, Mantle.READ, Mantle.READ])
        @test !isconst(actual[3])
        @test !isconst(actual[4])
        @test isconst(actual[5])
    end
    actual = wrapped((a, b), [Mantle.READ, Mantle.WRITE])
    @test isconst(actual[1])
    @test !isconst(actual[2])
    actual = wrapped((a, a), [Mantle.READ, Mantle.WRITE])
    @test !isconst(actual[1])

    actual = wrapped((a, (source = b,)), [Mantle.READ, Mantle.READ])
    @test isconst(actual[1])
    @test isconst(actual[2].source)
    adapted = Mantle.devargs(adapt, (a, b))
    @test Mantle.readonlyargs(graph, [Mantle.WRITE, Mantle.NOTOUCH], (a, b), adapted) === adapted
end

@testset "read-only arguments distinguish disjoint pooled byte ranges" begin
    dev = M.Device(TESTBACKEND)
    owner = M.Buffer(dev, Float32.(1:32))
    try
        parent = M.storage(owner)
        dst, src = view(parent, 1:16), view(parent, 17:32)
        @test M.bufferof(dst) === M.bufferof(src)
        @test !Base.mightalias(dst, src)
        adapt = M.LavaAdaptor(nothing)
        for graph in (nothing, M.Graph(dev))
            wrapped(raw, touches) = M.readonlyargs(graph, touches, raw, M.devargs(adapt, raw))
            # Adjacent ranges may be readonly even when their allocation owner
            # is shared. Reinterpretation compares bytes, not element indices.
            for read in (src, reinterpret(UInt32, src), reinterpret(UInt8, src))
                actual = wrapped((dst, read), [M.WRITE, M.READ])
                @test isconst(typeof(actual[2]))
                @test !(actual[1] isa Lava.Const)
            end
            for read in (dst, view(parent, 16:31),
                         view(reinterpret(UInt8, parent), 64:67))
                @test !(wrapped((dst, read), [M.WRITE, M.READ])[2] isa Lava.Const)
            end
            empty = view(parent, 1:0)
            @test isconst(typeof(wrapped((dst, empty), [M.WRITE, M.READ])[2]))
            @test isconst(typeof(wrapped((empty, dst), [M.WRITE, M.READ])[2]))
            # A later write and an alias buried in an aggregate still block it.
            @test isconst(typeof(wrapped((src, dst), [M.READ, M.WRITE])[1]))
            actual = wrapped(((source = src, alias = dst), dst), [M.READ, M.WRITE])
            @test !(actual[1].source isa Lava.Const)
            @test !(actual[1].alias isa Lava.Const)
        end
        KI.@launch dev ndrange=16 ro_add_one!(dst, src)
        @test Array(dst) == Float32.(18:33)
        g = M.Graph(dev)
        M.dispatch!(g, ro_add_one!, (dst, src), 16)
        M.runonce!(g)
        @test Array(dst) == Float32.(18:33)
    finally
        M.free!(owner)
    end
end

end # LavaDevice
