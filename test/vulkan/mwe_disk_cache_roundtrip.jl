# mwe_disk_cache_roundtrip.jl — does a deserialized LavaGPUKernel actually
# behave like the freshly-compiled one it was serialized from?
#
# The "MCP restart" crash isolated to the SPIR-V disk cache load. But the
# claim "AMDVLK can't handle previously-serialized SPIR-V" is implausible
# — vkCreateShaderModule has been battle-tested for years. Much more
# likely there's a bug in how Lava reuses the deserialized kernel.
#
# This MWE exercises the serialize→deserialize roundtrip in one session,
# no MCP restart, no atexit hooks. If the roundtripped kernel still works,
# the bug is somewhere in the MCP-restart path. If it fails here, the bug
# is in the serialization/reuse path itself.

using Lava, Test
import Serialization

# Simple kernel
function rt_kernel!(buf::LavaDeviceArray{Float32, 1})
    i = Int(Lava.lava_global_invocation_id_x()) + 1
    if i <= length(buf)
        @inbounds buf[i] = Float32(i) * 2f0
    end
    return nothing
end

backend = LavaBackend()
bq = backend.bq
N = 32

# Run 1: cold compile, capture the LavaGPUKernel from the in-memory cache
println("=== Run 1: cold compile + dispatch ===")
buf = Mantle.LavaArray(zeros(Float32, N))
Mantle.lava_launch!(bq, rt_kernel!, buf; ndrange=N, workgroup_size=(64, 1, 1))
Mantle.vk_flush!(bq)
ref = Array(buf)
println("  result[1:8] = ", ref[1:8])
@assert ref == Float32.(2 .* (1:N))

# Find the compiled kernel in the linked cache
linked = nothing
for (key, v) in Mantle.vk_context().caches.linked
    if v.compiled.entry_name == "main" && length(v.compiled.spirv_bytes) > 0
        linked = v
    end
end
@assert linked !== nothing "couldn't find compiled rt_kernel in LINKED_KERNEL_CACHE"
fresh = linked.compiled
println("  fresh kernel: $(length(fresh.spirv_bytes)) bytes spirv, entry_name=$(fresh.entry_name)")

# ── Roundtrip: serialize, deserialize, compare field-by-field ─────────────
println("\n=== Roundtrip: serialize → deserialize ===")
io = IOBuffer()
entry_for_disk = (
    spec_types  = nothing,           # not relevant for this MWE; usually source.specTypes
    workgroup_size = fresh.workgroup_size,
    kernel = Lava.LavaGPUKernel(
        fresh.spirv_bytes, fresh.entry_name, fresh.workgroup_size,
        fresh.push_info, "", fresh.enable_ray_query),
)
Serialization.serialize(io, entry_for_disk)
seekstart(io)
loaded = Serialization.deserialize(io)
deserialized = loaded.kernel

# Field-by-field
println("  spirv_bytes equal: ", deserialized.spirv_bytes == fresh.spirv_bytes,
        " (len fresh=", length(fresh.spirv_bytes), " des=", length(deserialized.spirv_bytes), ")")
println("  entry_name equal: ", deserialized.entry_name == fresh.entry_name,
        " (\"", deserialized.entry_name, "\" vs \"", fresh.entry_name, "\")")
println("  workgroup_size equal: ", deserialized.workgroup_size == fresh.workgroup_size)
println("  enable_ray_query equal: ", deserialized.enable_ray_query == fresh.enable_ray_query)
println("  push_info.wrapper_name equal: ", deserialized.push_info.wrapper_name == fresh.push_info.wrapper_name)
println("  push_info.push_size equal: ", deserialized.push_info.push_size == fresh.push_info.push_size)
println("  push_info.arg_buffer_size equal: ", deserialized.push_info.arg_buffer_size == fresh.push_info.arg_buffer_size)
println("  push_info.arg_layout equal: ", deserialized.push_info.arg_layout == fresh.push_info.arg_layout)
println("    fresh arg_layout: ", fresh.push_info.arg_layout)
println("    des   arg_layout: ", deserialized.push_info.arg_layout)
println("  push_info.byval_llvm_sizes equal: ", deserialized.push_info.byval_llvm_sizes == fresh.push_info.byval_llvm_sizes)
println("    fresh byval_sizes: ", fresh.push_info.byval_llvm_sizes)
println("    des   byval_sizes: ", deserialized.push_info.byval_llvm_sizes)

# ── Clear PIPELINE_CACHE so link_kernel must call vkCreateComputePipelines
# again from the deserialized bytes. Without this, get_compute_pipeline keys
# by hash(spirv_bytes,…) and we'd just reuse the existing VkPipeline from
# the fresh path — never actually exercising the driver on the cached bytes.
println("\n=== Clear PIPELINE_CACHE + build pipeline from deserialized bytes ===")
n_cached = length(Mantle.vk_context().caches.pipelines)
empty!(Mantle.vk_context().caches.pipelines)
empty!(Mantle.vk_context().caches.pipeline_order)
println("  cleared $n_cached entries from PIPELINE_CACHE")
ctx = Mantle.vk_context()
linked_des = Mantle.link_kernel(ctx, deserialized)
println("  link_kernel OK; pipeline=$(typeof(linked_des.pipeline))")
println("  offsets = ", linked_des.offsets)
println("  byval_sizes = ", linked_des.byval_sizes)

# Manually inject it into the linked-kernel cache under a fake key so
# lava_launch! picks it up — actually we can just call dispatch directly.
# Simplest: use the same lava_launch! path but force-replace LINKED_KERNEL_CACHE
# entry. Easier: just call get_or_dispatch path with our linked kernel.
# Actually simplest of all: call vkCmdDispatch ourselves via Lava's lower-level API.

# Just compile-and-dispatch a fresh kernel via lava_launch! again (will hit
# in-memory cache), then SWAP the cached linked kernel with our reconstructed
# one and dispatch. If the reconstructed kernel works, results match.
buf2 = Mantle.LavaArray(zeros(Float32, N))
# Find the cache key whose linked kernel matches `linked` (the one we captured earlier)
key_to_replace = nothing
for (key, v) in Mantle.vk_context().caches.linked
    if v === linked
        key_to_replace = key
    end
end
@assert key_to_replace !== nothing
println("  replacing caches.linked[$key_to_replace] with reconstructed kernel")
Mantle.vk_context().caches.linked[key_to_replace] = linked_des

Mantle.lava_launch!(bq, rt_kernel!, buf2; ndrange=N, workgroup_size=(64, 1, 1))
Mantle.vk_flush!(bq)
got = Array(buf2)
println("  result[1:8] = ", got[1:8])
println("  match fresh: ", got == ref)
@assert got == ref "deserialized kernel produced different output: $(got[1:8]) vs $(ref[1:8])"

println("\n✓ Roundtrip preserves correctness — no crash, no wrong output.")
println("So the MCP-restart crash is somewhere else; deserialize itself is sound.")
