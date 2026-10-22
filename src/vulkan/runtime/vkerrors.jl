# Vulkan call errors and the checks that report them.

"""
    LavaVulkanError <: Exception

A failed Vulkan call: its name, `VkResult`, message and an optional suggestion.
"""
struct LavaVulkanError <: Exception
    call::String
    vk_result::Int32
    message::String
    suggestion::String
end

function Base.showerror(io::IO, e::LavaVulkanError)
    print(io, "LavaVulkanError in ", e.call, ":\n")
    print(io, "  VkResult: ", e.vk_result, " — ", e.message, "\n")
    if !isempty(e.suggestion)
        print(io, "  Suggestion: ", e.suggestion)
    end
end

"""
    safe_fin_log(msg::String)

Print from a finalizer through `jl_safe_printf`, which cannot yield. `msg` is
passed as the argument of a `"%s"` format, so `%` in it is safe; it must carry
its own trailing newline.
"""
@inline safe_fin_log(msg::String) =
    ccall(:jl_safe_printf, Cvoid, (Cstring, Cstring), "%s", msg)

"""
    @vk_checked ctx site vk_call

`unwrap` the `Result` of `vk_call`, throwing on a failed `VkResult`, then throw
any validation errors `ctx` captured (`check_validation_errors!`), so an error
is reported at the call that produced it.

    pipelines, _ = @vk_checked ctx "vkCreateGraphicsPipelines" VK.create_graphics_pipelines(dev, [ci])
"""
macro vk_checked(ctx, site, call)
    quote
        r = unwrap($(esc(call)))
        check_validation_errors!($(esc(site)), $(esc(ctx)))
        r
    end
end

