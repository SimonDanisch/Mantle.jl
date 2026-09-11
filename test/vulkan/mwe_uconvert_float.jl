# MWE: OpUConvert emitting float result type for type-punned struct field access
#
# Bug: When a StaticMultiTypeSet dispatches over a Conductor{..., Texture{Float32}, ...},
# the type-punned packed bit field access generates `OpUConvert %float %ulong` instead of
# `OpUConvert %uint %ulong` followed by `OpBitcast %float %uint`.
#
# Root cause: The Pointee Type Map (PTM) resolves the trunc i64→i32 result type to `float`
# because the only user of the trunc is a `bitcast i32 to float`. The emit_conversion!
# function then uses this PTM-resolved type as the OpUConvert result type, which is invalid
# (OpUConvert requires integer result type).
#
# The fix should be in the load/trunc handling to ensure OpUConvert ALWAYS produces integer
# result type, regardless of what the PTM says.
#
# To reproduce:
#   julia --project=/sim/Programmieren/VulkanDev test/mwe_uconvert_float.jl
#
# Expected: renders successfully
# Actual: SPIR-V validation error: "Expected unsigned int scalar or vector type as Result Type: UConvert"

using Hikari, Lava, GeometryBasics, FileIO
import KernelAbstractions as KA

ENV["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json"
backend = Mantle.LavaBackend()

# This scene uses Conductor{..., Texture{Float32, 2, Matrix{Float32}}, ...} (checkerboard roughness)
scene_file = joinpath(@__DIR__, "..", "dev", "Hikari", "test", "pbrt", "scenes", "tex_conductor_checker_rough_light_point.pbrt")
fb = Hikari.render_pbrt(scene_file; backend=backend, samples=1)
println("OK: $(sum(Float64(red(p)) for p in Array(fb)) / length(fb))")
