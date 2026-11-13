# Arguments a dispatch only reads are compiled as read-only, and their loads at addresses
# every invocation shares NonWritable, unless a written argument reaches the same
# buffer: NonWritable is a promise about the
# memory, and `x .= x .+ 1` reads and writes one buffer through two arguments. The
# compile job carries a flag per argument (`readonlymask`), and the argument types stay
# the ones the access walk inferred. They were `Lava.Const` wrappers, and Julia inferred
# every kernel twice: once for the walk, once for the compile. Pinned here: which
# arguments are read-only, that the compile finds the walk's inference, that their
# uniform loads are NonWritable, and that every launch path still computes the same
# results.

using Test, Mantle
import Mantle.Lava
import Mantle.Lava: GPUCompiler
import KernelInterface as KI
include(joinpath(@__DIR__, "..", "testbackend.jl"))

if TESTBACKEND isa Mantle.LavaDevice

const M = Mantle

function ro_add_one!(dst, src)
    i = KI.get_global_id().x
    i <= length(dst) && (@inbounds dst[i] = src[i] + 1f0)
    return
end

"The first `n` bits of `mask`, as a tuple."
bits(mask::UInt64, n::Int) = ntuple(i -> isodd(mask >> (i - 1)), n)

"Which of `args` an eager launch of `f` compiles as read-only."
function readonly(dev, f, args)
    touches = M.argument_usage(dev, f, args, length(first(args)), nothing)
    return bits(M.readonlymask(nothing, touches, M.resolve(dev, args)), length(args))
end

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

@testset "read-only arguments: which are" begin
    dev = M.Device(TESTBACKEND)
    a = M.devicearray(dev, zeros(Float32, 16))
    b = M.devicearray(dev, ones(Float32, 16))
    @test readonly(dev, ro_add_one!, (a, b)) == (false, true)
    # The written buffer read through a second argument is not read-only memory.
    @test readonly(dev, ro_add_one!, (a, a)) == (false, false)
    # Nor through a view of it.
    @test readonly(dev, ro_add_one!, (a, view(a, 1:16))) == (false, false)
end

function ro_one_inference!(dst, src)
    i = KI.get_global_id().x
    i <= length(dst) && (@inbounds dst[i] = 2f0 * src[i])
    return
end

@testset "the compile infers what the access walk inferred" begin
    dev = M.Device(TESTBACKEND)
    dst = M.devicearray(dev, zeros(Float32, 16))
    src = M.devicearray(dev, Float32.(1:16))
    # The walk, as an eager launch runs it.
    touches = M.kitouches(dev, ro_one_inference!, (dst, src), 16)
    mask = M.readonlymask(nothing, touches, (dst, src))
    @test bits(mask, 2) == (false, true)
    tt = Tuple{map(a -> M.devicetype(dev, a), (dst, src))...}
    job = Lava.lava_kernel_job(ro_one_inference!, tt; workgroup_size = (64, 1, 1),
                               features = dev.features, readonly = mask)
    interp = GPUCompiler.get_interpreter(job)
    world = GPUCompiler.get_inference_world(interp)
    cache = Core.Compiler.WorldView(Core.Compiler.code_cache(interp), world, world)
    # Before compiling: the walk inferred the very signature the compile infers.
    @test Core.Compiler.haskey(cache, job.source)
    KI.@launch dev ndrange=16 ro_one_inference!(dst, src)
    @test Array(dst) == 2f0 .* Float32.(1:16)
end

# `src[1]` is one address for every invocation, `src[i]` one per invocation.
function ro_add_first!(dst, src)
    i = KI.get_global_id().x
    i <= length(dst) && (@inbounds dst[i] = src[i] + src[1])
    return
end

@testset "a read-only argument's uniform loads are NonWritable" begin
    dev = M.Device(TESTBACKEND)
    a = M.devicearray(dev, zeros(Float32, 16))
    b = M.devicearray(dev, ones(Float32, 16))
    nonwritable(f, flags) = occursin("NonWritable", Lava.disassemble_spirv(
        Lava.compile_or_lookup(Lava.lava_kernel_job(f, Tuple{map(x -> M.devicetype(dev, x), (a, b))...};
            workgroup_size = (64, 1, 1), features = dev.features, readonly = flags)).spirv_bytes))
    @test nonwritable(ro_add_first!, 0b10)
    @test !nonwritable(ro_add_first!, 0)
    # Per invocation only: nothing to gain from NonWritable but a driver free to move it.
    @test !nonwritable(ro_add_one!, 0b10)
    KI.@launch dev ndrange=16 ro_add_first!(a, b)
    @test Array(a) == fill(2f0, 16)
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
    @test any(((k, plans),) -> occursin("lava_broadcast_flat!", string(k)) &&
                               any(p -> (p::M.LaunchPlan).readonly != 0, plans),
              M.launch_plan_cache(dev))
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

@testset "planned read-only arguments preserve storage aliases" begin
    dev = M.Device(TESTBACKEND)
    graph = Mantle.Graph(dev)
    a = Mantle.devicearray(dev, zeros(Float32, 16))
    b = Mantle.devicearray(dev, ones(Float32, 16))
    c = Mantle.devicearray(dev, fill(2f0, 16))
    flags(raw, touches) = bits(Mantle.readonlymask(graph, touches, raw), length(raw))

    # The aliases include views and a nested aggregate. Aliasing one member
    # makes the whole aggregate writable, but leaves independent reads alone.
    for aliased in (a, view(a, 1:16), view(a, 9:16))
        raw = (a, (source = aliased, other = b), b)
        @test flags(raw, [Mantle.WRITE, Mantle.READ, Mantle.READ]) == (false, false, true)
    end

    # Every written argument participates, including ones after the read.
    for raw in ((a, b, a, b, c), (a, b, view(a, 1:8), view(b, 9:16), c))
        @test flags(raw, [Mantle.WRITE, Mantle.WRITE, Mantle.READ, Mantle.READ, Mantle.READ]) ==
              (false, false, false, false, true)
    end
    @test flags((a, b), [Mantle.READ, Mantle.WRITE]) == (true, false)
    @test flags((a, a), [Mantle.READ, Mantle.WRITE]) == (false, false)
    @test flags((a, (source = b,)), [Mantle.READ, Mantle.READ]) == (true, true)
    @test flags((a, b), [Mantle.WRITE, Mantle.NOTOUCH]) == (false, false)
end

@testset "read-only arguments distinguish disjoint pooled byte ranges" begin
    dev = M.Device(TESTBACKEND)
    owner = M.Buffer(dev, Float32.(1:32))
    try
        parent = M.storage(owner)
        dst, src = view(parent, 1:16), view(parent, 17:32)
        @test M.bufferof(dst) === M.bufferof(src)
        @test !Base.mightalias(dst, src)
        for graph in (nothing, M.Graph(dev))
            flags(raw, touches) = bits(M.readonlymask(graph, touches, raw), length(raw))
            # Adjacent ranges may be readonly even when their allocation owner
            # is shared. Reinterpretation compares bytes, not element indices.
            for read in (src, reinterpret(UInt32, src), reinterpret(UInt8, src))
                @test flags((dst, read), [M.WRITE, M.READ]) == (false, true)
            end
            for read in (dst, view(parent, 16:31),
                         view(reinterpret(UInt8, parent), 64:67))
                @test flags((dst, read), [M.WRITE, M.READ]) == (false, false)
            end
            empty = view(parent, 1:0)
            @test flags((dst, empty), [M.WRITE, M.READ])[2]
            @test flags((empty, dst), [M.WRITE, M.READ])[2]
            # A later write and an alias buried in an aggregate still block it.
            @test flags((src, dst), [M.READ, M.WRITE])[1]
            @test !flags(((source = src, alias = dst), dst), [M.READ, M.WRITE])[1]
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
