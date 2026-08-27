# Vulkan-side errors, and the two helpers that only make sense next to a device.
#
# Split out of what is now Lava's `runtime/errors.jl`: `LavaError`,
# `LavaCompilationError`, `SourceMap`, `validate_spirv` and `disassemble_spirv`
# stayed there because they are the compiler's. These did not.
#
# `@vk_checked` is the reason the split had to happen rather than being tidy: it
# expands to a call to `check_validation_errors!` WITHOUT escaping the name, so
# the call resolves in the macro's DEFINING module. Left in Lava, every use of it
# here would have looked for Lava's `check_validation_errors!` — which is now
# this package's — and found nothing.

"""
    LavaVulkanError <: Exception

Error from a Vulkan API call with context.
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
    safe_fin_log(msg)

Finalizer-safe logging. Uses `jl_safe_printf` which is a raw ccall that
cannot yield — the only form of diagnostic output allowed inside a Julia
finalizer.  `msg` must be a `String` that already contains any trailing
newline; no interpolation at call time (that would trigger dispatch /
world-age / allocation).  Prefer short fixed strings built by the caller.
"""
# `msg` goes through a "%s" FORMAT, not as the format string itself.
# `jl_safe_printf(const char *fmt, ...)` interprets its first argument as a
# printf format, so a message containing `%` — a path, a driver string, an
# exception text like "100% of heap" — was undefined behaviour: `%s` would read
# a nonexistent vararg off the stack, inside a finalizer, where there is no way
# to recover. Passing the string as an ARGUMENT makes any content safe, and
# costs nothing.
@inline safe_fin_log(msg::String) =
    ccall(:jl_safe_printf, Cvoid, (Cstring, Cstring), "%s", msg)

"""
    @vk_checked site_str vk_call

Wrap a Vulkan API call that returns a `ResultTypes.Result`: `unwrap` it
(propagating VkResult errors as exceptions), then flush accumulated
validation-layer messages via `check_validation_errors!(site_str)`.  Ensures
no validation error survives past the create/allocate call that produced
it and leaks into an unrelated later frame's diagnostic.

Usage:
    pipeline = @vk_checked "create_graphics_pipeline" VK.create_graphics_pipelines(dev, [ci])[1]
"""
macro vk_checked(site, call)
    quote
        r = unwrap($(esc(call)))
        check_validation_errors!($(esc(site)))
        r
    end
end

