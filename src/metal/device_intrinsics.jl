# `KernelInterface`'s device half, for Metal.
#
# WHAT IS NOT HERE any more: the lane identity queries, `localmemory`,
# `shfl_down` and the two barriers. Metal.jl implements `KernelInterface`
# itself now, so keeping a copy here was not redundancy, it was a SHADOW --
# these are `@device_override`s on the same names, and the zero-arg forms took
# precedence over Metal's, which meant Metal's implementation never ran and KI
# 0.2's typed `get_global_id(::Type{T})` never got the chance to convert.
#
# The copies were also wrong where they differed. KI declares
# `get_sub_group_size()::UInt32` and Metal's `threads_per_simdgroup()` answers
# `UInt32`; this file wrapped it in `Int`. Lava's half does not wrap it either,
# so removing these is what makes the two backends agree with the contract and
# with each other.
#
# `shfl` and `sub_group_reduce_add` went the same way once Metal.jl implemented
# KernelInterface 0.4 (JuliaGPU/Metal.jl#994, merged into our fork 2026-10-15):
# Metal's `KI.shfl` takes the 1-based lane on its eight native types, and its
# `sub_group_reduce_add` is `simd_sum` for any type. Our overrides for four of
# those types were more specific than Metal's methods, so they would have
# replaced them silently, without Metal's lane wrapping. The rule for this
# backend stands: a capability Metal has and Metal.jl does not wrap is a gap to
# close there, not to work around here.

# ── The output pointer of a shader stage ─────────────────────────────────────

"""
    StageOut{T}

The trailing output-pointer argument every Metal shader stage wrapper here is
compiled with — vertex, fragment, the mesh pipeline's fragment and the procedural
candidate function. Metal's `stage_return!` replaces it with a stack slot and
returns the slot's contents as the stage's value.

Address space 0, the slot's own. With the device address space (`1`) the slot was
`addrspacecast` into it, and a private alloca seen through a device pointer is
not the same memory on Apple GPUs: the body's stores went through the cast, the
return loaded the slot, and the two met only where LLVM folded the cast away. A
fragment comparing two `Vec3f` with `==` — Base's `AbstractArray` method, whose
bounds-check exits keep the slot from being promoted — returned zeros for every
pixel, clear colour included, with no error anywhere (`test_tuple_varyings.jl`).
"""
const StageOut{T} = Core.LLVMPtr{T,0}
