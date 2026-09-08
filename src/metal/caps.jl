# What this device can do, as `KernelInterface.DeviceCaps`.
#
# One struct, filled by every backend, read by the portable kernels. `gemv.jl`
# sizes its workgroup from `workgrouplimit` and `fft.jl` its staging from
# `sharedbudget`, and neither of them knows which GPU answered — that is the
# whole reason `caps` exists rather than each algorithm asking a driver.
#
# Two of the ten fields Metal does not report, and they are `0` rather than a
# guess. `DeviceCaps`' own documentation says `0` means "not known", and a
# plausible-looking invention is worse than an absence: a scheduler that trusts
# a made-up core count schedules for a device that does not exist.

"""
Threads per SIMD group.

32 on every Apple GPU, but read from a compiled pipeline rather than assumed —
`threadExecutionWidth` is the number the hardware will actually run, and the
same query is what Vulkan's `subgroup_size_control` answers. A kernel that
hard-codes 32 is wrong on the first device that disagrees, which is exactly how
the Vulkan backend got a wave32 assumption wrong on lavapipe.
"""
function simd_width(d::MetalDevice)
    kernel = Metal.@metal launch = false _caps_probe()
    return Int(kernel.pipeline.threadExecutionWidth)
end

# A kernel that does nothing, compiled only so the driver will tell us how wide
# it made the SIMD group and how many threads it will allow in a threadgroup.
_caps_probe() = return

"""
The largest threadgroup this device will launch, in threads.

`maxTotalThreadsPerThreadgroup` on the pipeline, not the device: the device's
`maxThreadsPerThreadgroup` is a per-dimension bound (1024×1024×1024 here), while
the pipeline's is the product limit a launch is actually checked against, and it
falls when a kernel uses more registers. Reading the device's would let a kernel
be sized past what it can be launched with.
"""
function max_threads(d::MetalDevice)
    kernel = Metal.@metal launch = false _caps_probe()
    return Int(kernel.pipeline.maxTotalThreadsPerThreadgroup)
end

"""
    matrixshapes(dev) -> Vector{MatrixShape}

The cooperative-matrix tiles this device implements.

**8×8×8, where Vulkan's is 16×16×16.** Metal's `simdgroup_matrix` is an 8×8 tile
and Apple exposes no other, so a kernel written against a hard-coded 16 does not
merely run slower here — it does not run. This is the field that makes tile size
a device query instead of a constant, and it is why `bestshape` exists.

Accumulation is `Float32` for every input type, which is what
`simdgroup_multiply_accumulate` provides.
"""
function matrixshapes(::MetalDevice)
    shapes = MatrixShape[]
    # Through `Metal.BFloat16`: the type comes from BFloat16s.jl, which is
    # Metal.jl's dependency and not Mantle's, so this extension cannot `using`
    # it. Metal binds the name, which is the reachable spelling.
    #
    # The three are what `Metal.simdgroup_load` actually has methods for — the
    # authoritative list, rather than what the Metal docs say the hardware can
    # do. A shape advertised here that Metal.jl cannot emit is a kernel that
    # fails to compile at first use.
    for ab in (Float16, Float32, Metal.BFloat16)
        push!(shapes, MatrixShape(ab, Float32, 8, 8, 8, SubgroupScope()))
    end
    return shapes
end

"""
    caps(dev) -> DeviceCaps

Everything the portable kernels ask about this device.

Built once and cached on the device: `threadExecutionWidth` needs a compiled
pipeline to read, so an uncached `caps` would compile a probe kernel on every
query — and `gemv` asks per launch.
"""
function caps(d::MetalDevice)
    d.caps === nothing || return d.caps
    mtl = d.dev
    w = simd_width(d)
    d.caps = DeviceCaps(
        true,                                   # coopmat: simdgroup_matrix
        8,                                      # tile: 8×8, not Vulkan's 16
        w,                                      # subgroup
        w,                                      # coopmatsubgroup — same group
        Int(mtl.maxThreadgroupMemoryLength),    # sharedbudget
        max_threads(d),                         # workgrouplimit
        0,                                      # cores: Metal does not report it
        0,                                      # warps: nor resident simdgroups
        NTuple{4,Int}[],                        # wggran: no granularity table
        matrixshapes(d),
    )
    return d.caps
end

# DELETED in phase 1.6: `_DEVICE` and `_device_for`, a second cache for "one
# device per process" beside `METAL_DEVICE` in `device.jl`. It built a second
# `MetalDevice` — a second `Pool`, a second `MTLCommandQueue`, a second
# `MTLSharedEvent` — from a capability query. `device.jl:65` forbids it and
# `graphics.jl` records two Metal queues as the cause of a hang.
caps(b::Metal.MetalBackend) = caps(Device(MetalAPI()))
