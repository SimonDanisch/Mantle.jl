# The whole of Bonsai 2 27B on the WebGPU device: the README's question, greedy.
# The prompt goes through the decode step one token at a time (`step!`), the path
# `bonsai_layers.jl` checks against Vulkan; the batched prefill is the next step.
using BonsaiRunner, Mantle, ElectronCall, Bonito
const BR = BonsaiRunner

if Sys.islinux() && !haskey(ENV, "DISPLAY")
    ENV["DISPLAY"] = ":1"
    ENV["XAUTHORITY"] = joinpath("/run/user/1000", first(filter(startswith("xauth_"), readdir("/run/user/1000"))))
end
Sys.islinux() && (ENV["FONTCONFIG_FILE"] = joinpath(@__DIR__, "fonts.conf"))

const QUESTION = "In three sentences: why does a Vulkan compute shader need a pipeline barrier " *
                 "between two dispatches that share a buffer?"

"""
Load Bonsai on `dev` and answer `QUESTION`, greedy, timed. The device is the
caller's (`Mantle.webgpudevice(; memory = 8 * 2^30)`), so it stays reachable when
this throws.
"""
function answer(dev; max_tokens = 120)
    tload = @elapsed (model = BR.Bonsai2(BR.checkpointpath(); device = dev, context = 4096, progress = false))
    tk = model.tokenizer
    ids = BR.encode(tk, BR.chatprompt(["user" => QUESTION]; thinking = false))
    tsession = @elapsed (s = BR.session(model; kv_page = 1024))
    tprompt = @elapsed begin
        local logits
        for id in ids
            logits = BR.step!(s, id)
        end
        out = [BR._greedy(s, logits)]
    end
    tdecode = @elapsed while length(out) < max_tokens && out[end] != tk.eos
        logits = BR.step!(s, out[end])
        push!(out, BR._greedy(s, logits))
    end
    return (; model, s, answer = BR.decode(tk, out), prompt_tokens = length(ids), tokens = length(out),
            load_s = tload, session_s = tsession, prompt_tok_s = length(ids) / tprompt,
            decode_tok_s = (length(out) - 1) / tdecode)
end
