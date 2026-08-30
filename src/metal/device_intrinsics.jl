# `KernelInterface`'s device half, for Metal.
#
# The counterpart of Lava's `device/kernelinterface.jl`. Each of these is one
# call site the Metal compiler lowers to a SIMD-group instruction; outside a
# kernel they mean nothing, exactly as on the Vulkan side.
#
# Shorter than Lava's because Metal.jl already provides the primitives — there
# is no SPIR-V to emit, so this file only has to say which Metal spelling
# answers which portable name.

# ── Lane identity ─────────────────────────────────────────────────────────────
#
# KI is 1-BASED, Metal is 0-based. The `+ 1` is not cosmetic: `shfl_down` and
# the reductions index off these, and Lava's half carries the same correction
# for the same reason.
@device_override @inline KI.get_sub_group_size() = Int(Metal.threads_per_simdgroup())
@device_override @inline KI.get_sub_group_local_id() =
    Int(Metal.thread_index_in_simdgroup()) + 1

# ── The shuffle family ────────────────────────────────────────────────────────
#
# `simd_shuffle_down` is Metal's, one instruction, for the four element types
# `MTL_SHFL_TYPES` lists. Float64 is absent because Apple GPUs do not have it —
# the divergence that made `KI_REDUCE_ADD_TYPES` a separate list from
# `KI_SHFL_TYPES` in the first place.
for T in (Float32, Float16, Int32, UInt32)
    @eval @device_override @inline KI.shfl_down(val::$T, offset::Integer) =
        Metal.simd_shuffle_down(val, offset)
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

# ── Barriers ──────────────────────────────────────────────────────────────────
#
# `@device_override`, not a plain method: KI gives `barrier` a HOST method that
# errors, so a plain definition would shadow it and a host-side call would reach
# a device instruction on the CPU. Lava's half carries the identical note.
@device_override @inline KI.barrier() = KA.@synchronize()
@device_override @inline KI.sub_group_barrier() = Metal.simdgroup_barrier(Metal.MemoryFlagThreadGroup)
