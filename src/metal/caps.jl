# `KernelInterface.DeviceCaps` for Metal. A field Metal does not report is `0`
# ("not known"), never a guess.

"""
    simd_width(d) -> Int

Threads per SIMD group, read from a compiled pipeline's `threadExecutionWidth`
rather than assumed to be 32.
"""
function simd_width(d::MetalDevice)
    kernel = Metal.@metal launch = false caps_probe()
    return Int(kernel.pipeline.threadExecutionWidth)
end

# Empty kernel, compiled only to read the pipeline's SIMD width and threadgroup limit.
caps_probe() = return

"""
    max_threads(d) -> Int

Largest threadgroup in threads: the pipeline's `maxTotalThreadsPerThreadgroup`,
which a launch is checked against and which falls as register use grows, not the
device's per-dimension `maxThreadsPerThreadgroup`.
"""
function max_threads(d::MetalDevice)
    kernel = Metal.@metal launch = false caps_probe()
    return Int(kernel.pipeline.maxTotalThreadsPerThreadgroup)
end

"""
    matrixshapes(dev) -> Vector{MatrixShape}

The cooperative-matrix tiles: `simdgroup_matrix` is 8×8×8 only (Vulkan's is
16×16×16), accumulating in `Float32` for every input type.
"""
function matrixshapes(::MetalDevice)
    shapes = MatrixShape[]
    # The types `Metal.simdgroup_load` has methods for; a shape Metal.jl cannot
    # emit fails to compile at first use. `Metal.BFloat16` because BFloat16s.jl
    # is Metal.jl's dependency, not Mantle's.
    for ab in (Float16, Float32, Metal.BFloat16)
        push!(shapes, MatrixShape(ab, Float32, 8, 8, 8, SubgroupScope()))
    end
    return shapes
end

"""
    caps(dev) -> DeviceCaps

Cached on the device: reading `threadExecutionWidth` compiles a probe kernel,
and `gemv` asks per launch.
"""
function caps(d::MetalDevice)
    d.caps === nothing || return d.caps
    mtl = d.dev
    w = simd_width(d)
    d.caps = DeviceCaps(
        true,                                   # coopmat: simdgroup_matrix
        8,                                      # tile: 8×8
        w,                                      # subgroup
        w,                                      # coopmatsubgroup, same group
        Int(mtl.maxThreadgroupMemoryLength),    # sharedbudget
        max_threads(d),                         # workgrouplimit
        Metal.num_gpu_cores(),                  # cores: the IORegistry's gpu-core-count
        0,                                      # warps: Metal does not report it
        NTuple{4,Int}[],                        # wggran: no workgroup-scope matrices
        matrixshapes(d),
    )
    return d.caps
end

# The backend's one device per process: a new `MetalDevice` here would bring a
# second `Pool` and `MTLCommandQueue`, unordered against the first.
caps(b::Metal.MetalBackend) = caps(Device(b))
