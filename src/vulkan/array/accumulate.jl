# Prefix scans for LavaArray
#
# `accumulate`/`cumsum` had no device implementation: GPUArrays.jl does not
# provide a scan (unlike `mapreduce`, which it does), so every call on a
# `LavaArray` fell through to Base's sequential version and died in
# `Scalar indexing is disallowed`. Anything above Mantle that needed a running
# total therefore had to copy to the host and back — which is what RayMakie's
# line topology did until this file existed.
#
# AcceleratedKernels is the same dependency `mapreduce.jl` already reaches for
# on its non-Float32 path, and its scan is written against KernelAbstractions,
# so ONE implementation serves Vulkan here and Metal through `MtlArray`'s own
# KA backend. There is no second scan to keep in step.
#
# What a scan needs beyond an operator is a NEUTRAL ELEMENT — the value a
# partial block starts from. `AK.neutral_element` knows the standard operators;
# a custom `op` has to bring its own, and passing a wrong one gives a wrong
# answer rather than an error, so it is required rather than guessed.
#
# Known limit, measured 2026-09-22: a scan whose element type is a TUPLE fails
# SPIR-V validation. So the textbook segmented scan — carrying `(value, flag)`
# pairs — does not compile here, and a segmented scan has to be expressed with
# scalar payloads (RayMakie's line lengths do it as `+` scan minus a `max` scan).

import AcceleratedKernels as AK

"""
    Base.accumulate!(op, dest::AnyLavaArray, src; init, dims) -> dest

Device prefix scan, delegating to AcceleratedKernels.

`init` defaults to `op`'s neutral element where AcceleratedKernels knows one
(`+`, `*`, `min`, `max`); for any other operator it must be given, because a
block-parallel scan has no way to derive it and no way to detect a wrong one.
"""
function Base.accumulate!(op, dest::AnyLavaArray, src::AbstractArray;
                          init = AK.neutral_element(op, eltype(dest)),
                          dims::Union{Nothing,Int} = nothing)
    dest === src || copyto!(dest, src)
    AK.accumulate!(op, dest; init, neutral = init, dims)
    return dest
end

function Base.accumulate(op, src::AnyLavaArray;
                         init = AK.neutral_element(op, eltype(src)),
                         dims::Union{Nothing,Int} = nothing)
    return AK.accumulate(op, src; init, neutral = init, dims)
end

Base.cumsum(src::AnyLavaArray; dims::Union{Nothing,Int} = nothing) =
    accumulate(+, src; dims)
Base.cumsum!(dest::AnyLavaArray, src::AbstractArray; dims::Union{Nothing,Int} = nothing) =
    accumulate!(+, dest, src; dims)
Base.cumprod(src::AnyLavaArray; dims::Union{Nothing,Int} = nothing) =
    accumulate(*, src; dims)
Base.cumprod!(dest::AnyLavaArray, src::AbstractArray; dims::Union{Nothing,Int} = nothing) =
    accumulate!(*, dest, src; dims)
