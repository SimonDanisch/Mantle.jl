# Bonsai's decode step for its first layers, with only those layers' weights loaded
# (about 1 GB of the 5.95 GB): every kernel kind a decode step runs, recurrent and
# attention layers, the PTQ1 products and the greedy pick, on any device.
# `decode(device, layers, tokens)` returns each step's logits and greedy token, so
# a WebGPU run can be compared with a Vulkan one.
#
# BonsaiRunner's own step builder declares all 64 layers; `declarestep!` below is
# its loop over a range of layers. The last boundary norms with the next layer's
# weight (loaded for that), not `output_norm`, which is the same computation.
using BonsaiRunner, Mantle
const BR = BonsaiRunner
const DK = BR.DNNKernels

"""Whether a decode over `layers` reads tensor `name`: all but other layers' blocks, and the next layer's norm."""
function wanted(name::AbstractString, layers)
    m = match(r"^blk\.(\d+)\.", name)
    m === nothing && return true
    il = parse(Int, m.captures[1])
    return il in layers || (il == last(layers) + 1 && endswith(name, "attn_norm.weight"))
end

"""Upload the tensors `wanted` for `layers`, as `BonsaiRunner.loadweights!` uploads all of them."""
function loadlayers!(model, layers)
    for name in sort!(collect(keys(model.file.tensors)))
        wanted(name, layers) || continue
        t = model.file[name]
        model.weights[name] = if t.typeid == DK.GGML_TYPE_PTQ1_0
            BR.ptq1matrix(model.backend, model.file, t)
        elseif t.typeid == DK.GGML_TYPE_BF16
            Mantle.Buffer(Mantle.todevice(model.backend), DK.gguftensor(model.file, t))
        else
            BR._upload(model.backend, DK.gguffloat(model.file, t))
        end
    end
    Mantle.waitidle(Mantle.todevice(model.backend))
    return model
end

"""A session over `layers` only: their recurrent state and KV caches, `BonsaiRunner.session` without its recording."""
function partialsession(model, layers; capacity::Int = 256)
    recurrent = Dict{Int, Tuple{Any, Any}}()
    kv = Dict{Int, Any}()
    for il in layers
        if (il + 1) % 4 == 0
            kv[il] = BR._q8cache(model, capacity)
        else
            recurrent[il] = (BR._zeros(model, Float32, 3, 10240), BR._zeros(model, Float32, 128, 128, 48))
        end
    end
    dev = Mantle.Device(model.backend)
    return BR.BonsaiSession(model, 0, recurrent, kv, BR._zeros(model, Float32, 248320),
                            Mantle.Buffer(dev, Int32, (1,)), nothing, Mantle.GPURef(dev, Int32(0)),
                            Mantle.GPURef(dev, Int32(0)), nothing, capacity, Dict{Int, Any}())
end

"""`BonsaiRunner._declarestep!` over `layers`."""
function declarestep!(s, g, q, layers)
    embedding = BR._w(s, "token_embd.weight")
    BR._dispatch!(g, DK.ptq1_getrows_kernel!,
        (q[:x], embedding.data, s.tokenref, Int32(embedding.m), Int32(embedding.k), Int32(1)),
        embedding.k; group = 256, name = "token_embedding")
    BR._hadamard!(s, g, q[:x], s.model.signs[BR.WIDTH]; inverse = true)
    xs = (q[:x], q[:xalt])
    cur = 1
    BR._boundary!(s, g, q, xs[cur], nothing, nothing, "blk.0.attn_norm.weight", "blk.0.attn_norm")
    for il in layers
        prefix = "blk.$il."
        branch = (il + 1) % 4 == 0 ? BR._attention!(s, g, q, il, q[:norm]) :
                                     BR._recurrent!(s, g, q, il, q[:norm])
        BR._boundary!(s, g, q, xs[cur], xs[3 - cur], branch,
                      prefix * "post_attention_norm.weight", prefix * "attn_residual_norm")
        cur = 3 - cur
        branch = BR._ffn!(s, g, q, il, q[:norm])
        BR._boundary!(s, g, q, xs[cur], xs[3 - cur], branch,
                      "blk.$(il + 1).attn_norm.weight", prefix * "ffn_residual_norm")
        cur = 3 - cur
    end
    BR._ptq!(s, g, q[:logits], "output.weight", q[:had5120])
    BR._declaregreedy!(s, g, q[:logits])
    return q[:logits]
end

"""
    decode(device, tokens; layers = 0:3) -> (; logits, best, kernels)

The partial model's decode step over `tokens` (zero-based ids), one recorded plan
replayed per token: each step's logits and greedy token, and the kernel of every
dispatch of the plan.
"""
function decode(device, tokens::AbstractVector{<:Integer}; layers = 0:3, path = BR.checkpointpath())
    model = BR.Bonsai2(path; device, context = 256, loadweights = false, progress = false)
    loadlayers!(model, layers)
    s = partialsession(model, layers)
    dev = Mantle.Device(model.backend)
    g = BR._neuralgraph(s, dev)
    q = BR._stepscratch(g, s.logits, BR.kv_capacity(s))
    declarestep!(s, g, q, layers)
    plan = BR._recordneural(g)
    logits = Vector{Float32}[]
    best = Int32[]
    for (pos, t) in enumerate(tokens)
        s.tokenref[] = Int32(t)
        s.positionref[] = Int32(pos - 1)
        Mantle.run!(plan)
        push!(logits, Array(s.logits))
        push!(best, only(Array(s.best)))
    end
    kernels = [string(nameof(d.kernel)) for p in plan.graph.passes for d in p.dispatches]
    return (; logits, best, kernels)
end
