"""
Matrix products, attention and convolution on Metal, for Mantle's graphs.

Kernels a graph dispatches: the Metal 4 tensor-ops GEMM (`gemm_kernel_config`,
`batched_gemm_kernel_config`) and fused attention (`attention_kernel_config`), written
against Metal.jl's exported device API, with Metal.jl's own simdgroup GEMM kernel where
the tensor path does not apply. Calls a graph makes: MPSGraph products, attention and
convolutions (`mpsgemm!`, `mpsattention!`, `mpsconv2d!`), built from the wrappers in
`Metal.MPSGraphs` and `MetalBindings`, each committing one MPS command buffer to the
queue it is given.
"""
module MetalLinalg

using Metal
using Metal: MTL, MPS, MPSGraphs, MtlDeviceArray, BatchedCommandQueue,
             MemoryFlagThreadGroup
using Metal.MPS: MPSCommandBuffer, MPSShape
using Metal.MPSGraphs: MPSGraph, MPSGraphTensor, MPSGraphTensorData, placeholderTensor,
                       castTensor, reshapeTensor, transposeTensor, constantWithScalar,
                       matrixMultiplicationWithPrimaryTensor,
                       multiplicationWithPrimaryTensor, additionWithPrimaryTensor,
                       scaledDotProductAttentionWithQueryTensor,
                       MPSGraphConvolution2DOpDescriptor, convolution2DWithSourceTensor,
                       convolution2DDataGradientWithIncomingGradientTensor, mps_axis,
                       default_exec_desc
using Metal.ObjectiveC: nil
using Metal.ObjectiveC.Foundation: NSDictionary
using BFloat16s: BFloat16
using ..MetalBindings: sliceTensor, reLUWithTensor, erfWithTensor, tanhWithTensor

include("tensorgemm.jl")  # tensor-ops GEMM, and the simdgroup GEMM fallback
include("attention.jl")   # fused attention on tensor ops
include("graphs.jl")      # MPSGraph product, attention and convolution

end
