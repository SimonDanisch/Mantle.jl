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
# What REMAINS is what Metal.jl does not provide: `shfl` and
# `sub_group_reduce_add`. Both are one Metal instruction and the rule for this
# backend still holds -- a capability Metal has and Metal.jl does not wrap is a
# gap to close there, not to work around here.

# ── The shuffle family ────────────────────────────────────────────────────────
#
# `simd_shuffle_down` is Metal's, one instruction, for the four element types
# `MTL_SHFL_TYPES` lists. Float64 is absent because Apple GPUs do not have it —
# the divergence that made `KI_REDUCE_ADD_TYPES` a separate list from
# `KI_SHFL_TYPES` in the first place.
for T in (Float32, Float16, Int32, UInt32)
    @eval @device_override @inline KI.shfl(val::$T, lane::Integer) =
        Metal.simd_shuffle(val, lane + 1)
end

# ── The reduction ─────────────────────────────────────────────────────────────

# `sub_group_reduce_add` is `Metal.simd_sum`: ONE instruction, not a butterfly.
#
# MSL has had `simd_sum` all along; Metal.jl simply had not wrapped it, so the
# first version of this file was a log2(width) XOR-shuffle loop — five
# instructions at width 32 to do what the hardware does in one. The fix belonged
# upstream, and `air.simd_sum` and its `product`/`min`/`max`/`and`/`or`/`xor`
# siblings are wrapped in `Metal/src/device/intrinsics/simd.jl` now.
#
# That is the rule for this backend generally: a capability Metal has and
# Metal.jl does not wrap is a gap to close there, not to work around here.
# Hardware ray tracing will need a great deal more of it.
for T in (Float32, Float16, Int32, UInt32)
    @eval @device_override @inline KI.sub_group_reduce_add(val::$T) = Metal.simd_sum(val)
end
