# Mantle's Metal backend, beside `src/vulkan/`. It needs neither Lava nor SPIR-V:
# Metal.jl compiles the kernels.
#
# Metal buffers have no usage flags, only a storage mode, so the storage mode is
# the only constraint on sharing. On Apple silicon `Shared` memory is unified, so
# `upload!` is a `copyto!` without staging or barriers. One `MTLSharedEvent`
# counts submissions and backs `fence`/`passed`/`waitfor`.

using Metal
using Metal: MtlArray
# Device intrinsics must be `@device_override`s, not plain methods: `KI.barrier`
# has a host method that errors, and a plain method would replace it on the host.
using Metal: @device_override
# GLFW owns the window; `@objc`/`id` put a `CAMetalLayer` on its view.
using GLFW
# ObjectiveC 7 gives each `@objc` call its own autorelease pool and Metal.jl returns
# retained wrappers, so builds must not be wrapped in a global `@autoreleasepool`.
using Metal.ObjectiveC: @objc, id
# `MTLSetReportFailureBlock` is looked up by name and receives an `NSString` block (`debug.jl`).
import Libdl
using Metal.ObjectiveC: @objcblock
using Metal.ObjectiveC.Foundation: NSString, NSBlock
const OCObject = Metal.ObjectiveC.Object
const MTL = Metal.MTL

include("device.jl")
include("debug.jl")
include("memory.jl")
include("caps.jl")
include("kernelinterface.jl")
include("device_intrinsics.jl")
include("ka.jl")
include("raytracing.jl")
# After raytracing.jl: needs `MetalTLAS`.
include("hwtlas.jl")

include("graphics.jl")

# After graphics.jl: needs `Metal.MeshPtr` and the AIR intrinsics.
include("mesh.jl")
# After graphics.jl: needs `mtlformat`.
include("images.jl")
# After graphics.jl: needs `mtlformat`, `framebuffer!` and `commit!`.
include("window.jl")

# Last: recording a plan into an indirect command buffer needs every other file.
include("record.jl")

# No pipeline thread: Metal.jl compiles on the calling thread. The `atexit` hook
# retires every live device before Julia's final finalizer pass, so no resource
# is released through a driver that is already gone.
function Mantle.initbackend!(::MetalAPI)
    register_backend!(; name = :metal, priority = 90) do
        Metal.functional() ? Device(MetalAPI()) : nothing
    end
    atexit() do
        for r in METAL.live
            d = r.value
            d isa MetalDevice && retire!(d)
        end
    end
    return nothing
end

# `staged_gemm_tile` is core's (`array/gemm.jl`); a method here would be an
# overwrite. Metal devices decline the staged GEMM by their 8x8 `caps` tile.

# Forwarded so the count is real: core's default `(; hits = 0, misses = 0)` would
# read as "nothing compiled".
kernelcompiles(::MetalDevice) = Metal.compile_stats()
resetkernelcompiles!(::MetalDevice) = (Metal.reset_compile_stats!(); nothing)
