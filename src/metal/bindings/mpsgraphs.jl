# MPSGraph operations that Metal.jl does not wrap. Each follows the wrappers in
# `Metal.MPSGraphs`: named after Apple's selector, the graph first and the node name
# last, taking and returning Metal.jl's `MPSGraph` and `MPSGraphTensor`.

using Metal.MPSGraphs: MPSGraph, MPSGraphTensor
using Metal.ObjectiveC.Foundation: NSString, NSInteger, NSUInteger

"""
    sliceTensor(graph, tensor, dimension, start, length, name = "slice") -> MPSGraphTensor

`length` entries of `dimension`, starting at `start`. `dimension` is an MPSGraph axis,
zero-based and outermost first (`Metal.MPSGraphs.mps_axis` converts a Julia one). The
sliced axis is kept with the given length, so the axis numbers after it do not change.
"""
function sliceTensor(graph::MPSGraph, tensor::MPSGraphTensor, dimension::Integer,
                     start::Integer, len::Integer, name = "slice")
    @objc [graph::id{MPSGraph} sliceTensor:tensor::id{MPSGraphTensor}
                                 dimension:dimension::NSUInteger
                                     start:start::NSInteger
                                    length:len::NSInteger
                                      name:name::id{NSString}]::MPSGraphTensor
end

"""
    reLUWithTensor(graph, tensor, name = "relu") -> MPSGraphTensor

`max(x, 0)` elementwise.
"""
function reLUWithTensor(graph::MPSGraph, tensor::MPSGraphTensor, name = "relu")
    @objc [graph::id{MPSGraph} reLUWithTensor:tensor::id{MPSGraphTensor}
                                        name:name::id{NSString}]::MPSGraphTensor
end

"""
    erfWithTensor(graph, tensor, name = "erf") -> MPSGraphTensor

The error function elementwise.
"""
function erfWithTensor(graph::MPSGraph, tensor::MPSGraphTensor, name = "erf")
    @objc [graph::id{MPSGraph} erfWithTensor:tensor::id{MPSGraphTensor}
                                       name:name::id{NSString}]::MPSGraphTensor
end

"""
    tanhWithTensor(graph, tensor, name = "tanh") -> MPSGraphTensor

`tanh` elementwise.
"""
function tanhWithTensor(graph::MPSGraph, tensor::MPSGraphTensor, name = "tanh")
    @objc [graph::id{MPSGraph} tanhWithTensor:tensor::id{MPSGraphTensor}
                                        name:name::id{NSString}]::MPSGraphTensor
end
