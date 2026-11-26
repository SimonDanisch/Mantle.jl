# Bonsai's first layers on Vulkan and on WebGPU, the same tokens: each step's
# greedy token and how far the logits are apart. `bonsai_layers.jl` builds them.
include(joinpath(@__DIR__, "bonsai_layers.jl"))
using ElectronCall, Bonito

if !haskey(ENV, "DISPLAY")
    ENV["DISPLAY"] = ":1"
    ENV["XAUTHORITY"] = joinpath("/run/user/1000", first(filter(startswith("xauth_"), readdir("/run/user/1000"))))
end
ENV["FONTCONFIG_FILE"] = joinpath(@__DIR__, "fonts.conf")

const CKPT = BR.checkpointpath()
const VK = Mantle.Device(Mantle.VulkanAPI())
const TOKENS = BR.encode(BR.Bonsai2(CKPT; device = VK, loadweights = false, progress = false).tokenizer,
                         "The capital of France is")
const REF = decode(VK, TOKENS; path = CKPT)
const WG = Mantle.webgpudevice(; memory = 3 * 2^30)

"""The WebGPU run against Vulkan's: greedy tokens, and the largest logit difference per step."""
function compare()
    out = decode(WG, TOKENS; path = CKPT)
    return (; webgpu = out.best, vulkan = REF.best,
            maxdiff = [maximum(abs, a .- b) for (a, b) in zip(out.logits, REF.logits)],
            scale = [maximum(abs, b) for b in REF.logits])
end
