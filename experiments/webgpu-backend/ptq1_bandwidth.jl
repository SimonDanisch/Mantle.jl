include(joinpath(@__DIR__, "bonsai_full.jl"))
import KernelInterface as KI
using Random, Statistics

# Each group reads one contiguous span of the tiled weight buffer. Keep one
# checksum per thread so the compiler must execute the loads.
function read_weights!(out, data, n::Int32)
    words = reinterpret(UInt32, data)
    i = Int32(KI.get_group_id().x - 1) * Int32(4096) + Int32(KI.get_local_id().x)
    bits = UInt32(0)
    @inbounds for k in Int32(0):Int32(15)
        j = i + Int32(256) * k
        j <= n && (bits ⊻= words[j])
    end
    @inbounds out[Int32(KI.get_global_id().x)] = bits
    return nothing
end

function timed_dispatch!(declare, dev; samples)
    graph = Mantle.Graph(dev)
    for _ in 1:samples
        declare(graph)
    end
    webgpu = Mantle.syncbackend(dev) isa Mantle.WebGPUAPI
    plan = Mantle.record!(Mantle.Plan(graph; profile = !webgpu, budget = Inf))
    try
        Mantle.run!(plan)
        Mantle.waitidle(dev)
        if webgpu
            ext = Base.get_extension(Mantle, :MantleWebGPUExt)
            times = ext.profiledispatches(dev) do
                Mantle.run!(plan)
            end
            @assert length(times.samples) == samples
            return median(row.ns for row in times.samples) / 1e6
        end
        Mantle.run!(plan)
        Mantle.waitidle(dev)
        costs = [t.gpu_ms for t in Mantle.timings(plan) if t.kind === :compute]
        @assert length(costs) == samples && all(isfinite, costs)
        return median(costs)
    finally
        Mantle.free!(plan)
    end
end

function benchmark_weight(dev, file, name; samples, configs = ())
    D = BR.DNNKernels
    A = D.ptq1matrix(dev, file, name)
    nbytes = length(A.data)
    nwords = nbytes ÷ 4
    threads = cld(nwords, 4096) * 256
    x = Mantle.Buffer(dev, randn(MersenneTwister(52), Float32, A.k))
    y = Mantle.Buffer(dev, Float32, (A.m,))
    partial = Mantle.Buffer(dev, UInt32, (threads,))
    try
        product_ms = timed_dispatch!(dev; samples) do graph
            D.ptq1_rows!(graph, y, A, x; name = "matvec")
        end
        stream_ms = timed_dispatch!(dev; samples) do graph
            Mantle.dispatch!(graph, read_weights!, (partial, A.data, Int32(nwords)), threads;
                             group = 256, name = "stream")
        end
        output = Array(y)
        variants = map(configs) do (parts, double_buffer)
            group = D.PTQ1_TILE * parts
            args = (y, A.data, x, A.data, Int32(A.m), Int32(A.k), Val(false),
                    Val(parts), Val(double_buffer))
            ms = timed_dispatch!(dev; samples) do graph
                Mantle.dispatch!(graph, D.ptq1_mul_rows_kernel!, args, cld(A.m, D.PTQ1_TILE) * group;
                                 group, name = "matvec")
            end
            @assert isapprox(output, Array(y); rtol = 1f-6)
            return (; parts, double_buffer, ms)
        end
        return (; shape = size(A), nbytes, product_ms, stream_ms, output, variants,
                checksum = reduce(xor, Array(partial)))
    finally
        foreach(Mantle.free!, (A.data, x, y, partial))
    end
end

"""Time real Bonsai projections and a read of the same packed weights, on both devices."""
function compare_ptq1(webgpu, vulkan; samples = 11,
                     names = ("blk.0.ffn_gate.weight", "blk.0.ffn_down.weight", "output.weight"),
                     configs = ())
    model = BR.Bonsai2(BR.checkpointpath(); device = vulkan, loadweights = false, progress = false)
    try
        for name in names
            v = benchmark_weight(vulkan, model.file, name; samples)
            w = benchmark_weight(webgpu, model.file, name; samples, configs)
            @assert v.checksum == w.checksum
            @assert isapprox(v.output, w.output; rtol = 1f-6)
            println((; name, v.shape, v.nbytes, vulkan_ms = (v.product_ms, v.stream_ms),
                     webgpu_ms = (w.product_ms, w.stream_ms), maxdiff = maximum(abs, v.output .- w.output)))
            isempty(w.variants) || println("WebGPU variants: ", w.variants)
        end
    finally
        close(model)
    end
end
