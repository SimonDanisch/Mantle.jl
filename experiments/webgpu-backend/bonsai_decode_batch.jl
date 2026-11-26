include(joinpath(@__DIR__, "bonsai_full.jl"))
include(joinpath(@__DIR__, "bonsai_layers.jl"))
using Statistics

function feedback_token!(tokens, token, position, best, iteration)
    @inbounds begin
        id = best[1]
        tokens[iteration[1]] = id
        token[1] = id
        position[1] += Int32(1)
    end
    return nothing
end

function declare_decode!(s, graph, scratch, layers)
    if layers === nothing
        BR._declarestep!(s, graph, scratch)
    else
        declarestep!(s, graph, scratch, layers)
    end
end

function decode_session(model, layers)
    layers === nothing && return BR.session(model; kv_page = 256)
    s = partialsession(model, layers)
    try
        g = BR._neuralgraph(s, model.backend)
        s.scratch = BR._stepscratch(g, s.logits, BR.kv_capacity(s))
        declare_decode!(s, g, s.scratch, layers)
        s.plan = BR._recordneural(g)
        return s
    catch
        close(s)
        rethrow()
    end
end

function record_decode_batch(s, steps, layers)
    g = Mantle.Graph(s.model.backend)
    q = BR._stepscratch(g, s.logits, BR.kv_capacity(s))
    tokens = Mantle.Transient.Buffer(g, Int32, (steps,))
    Mantle.repeat!(g, steps) do i
        declare_decode!(s, g, q, layers)
        Mantle.dispatch!(g, feedback_token!, (tokens, s.tokenref, s.positionref, s.best, i), 1;
                         group = 1, name = "decode_feedback")
    end
    pending = Mantle.Readback(g, tokens)
    plan = Mantle.record!(Mantle.Plan(g; budget = Inf))
    return (; g, tokens, pending, plan)
end

function run_host_decode!(s, seed, steps)
    ids = Vector{Int32}(undef, steps)
    elapsed = @elapsed for i in eachindex(ids)
        seed = BR._greedy(s, BR.step!(s, seed))
        ids[i] = seed
    end
    return (; ids, elapsed)
end

function run_device_decode!(s, seed, batch, future)
    dev = s.model.backend
    before = Mantle.submissions(dev)
    elapsed = @elapsed begin
        s.tokenref[] = Int32(seed)
        s.positionref[] = Int32(s.position)
        Mantle.run!(batch.plan)
        ids = fetch(future)
    end
    s.position += length(ids)
    return (; ids, elapsed, submissions = Mantle.submissions(dev) - before)
end

"""
Compare fixed-size greedy decode with host feedback and with GPU feedback in one
submission. `layers = nothing` uses the full model. The experiment runs a fixed
number of steps, including if an EOS token appears, to compare the exact work.
"""
function compare_decode_batch(dev; layers = 0:3, steps = 8, samples = 4)
    steps >= 2 && samples >= 1 || throw(ArgumentError("use at least two steps and one sample"))
    model = BR.Bonsai2(BR.checkpointpath(); device = dev, context = 256,
                       loadweights = layers === nothing, progress = false)
    host = device = batch = nothing
    try
        layers === nothing || loadlayers!(model, layers)
        prompt = BR.encode(model.tokenizer, "The capital of France is")
        length(prompt) + steps * (samples + 1) <= model.maxcontext ||
            throw(ArgumentError("prompt and samples exceed the benchmark's context"))
        host = decode_session(model, layers)
        device = decode_session(model, layers)
        for token in prompt
            BR.step!(host, token)
            BR.step!(device, token)
        end
        seed = BR._greedy(host, host.logits)
        @assert seed == BR._greedy(device, device.logits)
        batch = record_decode_batch(device, steps, layers)
        Mantle.waitidle(dev)
        host_ms, device_ms = Float64[], Float64[]
        for sample in 0:samples
            future = sample == 0 ? batch.pending : Mantle.Readback(batch.g, batch.tokens)
            h = run_host_decode!(host, seed, steps)
            d = run_device_decode!(device, seed, batch, future)
            @assert h.ids == d.ids
            @assert d.submissions == 1
            seed = last(h.ids)
            sample == 0 && continue  # Warm the decode and readback paths.
            push!(host_ms, 1000 * h.elapsed / steps)
            push!(device_ms, 1000 * d.elapsed / steps)
        end
        host_cost, device_cost = median(host_ms), median(device_ms)
        return (; steps, host_ms_per_token = host_cost, device_ms_per_token = device_cost,
                speedup = host_cost / device_cost, checked_tokens = steps * (samples + 1))
    finally
        batch === nothing || Mantle.free!(batch.plan)
        host === nothing || close(host)
        device === nothing || close(device)
        close(model)
    end
end
