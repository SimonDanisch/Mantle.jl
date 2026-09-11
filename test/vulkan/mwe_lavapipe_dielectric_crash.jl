# MWE: lavapipe segfault from max(abs(dot(reflected,n)), 1f-6) in large kernel
#
# The crashing pattern:  R / max(abs(dot(wi, n)), 1f-6)  where wi = reflect(wo, n)
# The working pattern:   R / cos_theta_o                  (reuse precomputed cos)
#
# Both are mathematically identical (dot(reflect(wo,n), n) == dot(wo,n)).
# Crashes lavapipe but works on AMD RADV.
#
# Run: julia --project=. dev/Lava/test/mwe_lavapipe_dielectric_crash.jl

ENV["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json"

using Lava, Mantle
using GeometryBasics
import KernelAbstractions as KA
using KernelAbstractions: @kernel, @index, synchronize
using LinearAlgebra: dot, normalize

backend = Mantle.defaultbackend()
println("Backend: lavapipe")

# The crashing pattern: reflect → recompute cos → max guard → divide
@kernel function crash_kernel!(out, @Const(wo_arr), @Const(n_arr), @Const(R_arr))
    i = @index(Global)
    wo = wo_arr[i]
    n = n_arr[i]
    R = R_arr[i]
    wi = -wo + 2f0 * dot(wo, n) * n    # reflect
    cos_i = abs(dot(wi, n))              # recompute cos from reflected dir
    out[i] = R / max(cos_i, 1f-6)       # divide with guard
end

# The working pattern: use precomputed cos directly
@kernel function working_kernel!(out, @Const(wo_arr), @Const(n_arr), @Const(R_arr))
    i = @index(Global)
    wo = wo_arr[i]
    n = n_arr[i]
    R = R_arr[i]
    cos_o = abs(dot(wo, n))              # cos from original direction
    out[i] = R / cos_o                   # same value, no max guard needed
end

N = 256
wo = KA.allocate(backend, Vec3f, N)
ns = KA.allocate(backend, Vec3f, N)
Rs = KA.allocate(backend, Float32, N)
out = KA.allocate(backend, Float32, N)

copyto!(wo, [normalize(Vec3f(rand(Float32)*2-1, rand(Float32)*2-1, 0.5f0 + rand(Float32)*0.5f0)) for _ in 1:N])
copyto!(ns, fill(Vec3f(0f0, 0f0, 1f0), N))
copyto!(Rs, fill(0.04f0, N))

println("Testing crash pattern: R / max(abs(dot(reflect(wo,n), n)), 1f-6)...")
k1 = crash_kernel!(backend)
k1(out, wo, ns, Rs; ndrange=N)
synchronize(backend)
r1 = Array(out)
println("  OK! result[1] = $(r1[1])")

println("Testing working pattern: R / abs(dot(wo, n))...")
k2 = working_kernel!(backend)
k2(out, wo, ns, Rs; ndrange=N)
synchronize(backend)
r2 = Array(out)
println("  OK! result[1] = $(r2[1])")

println("Max diff: $(maximum(abs.(r1 .- r2)))")
println("\nBoth patterns work in isolation. The crash only occurs in the full")
println("122KB evaluate_materials kernel with 3+ material types.")
