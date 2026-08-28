# Lava.jl CI Test Suite (fast)
#
# Runs in ~5 minutes on lavapipe or NVIDIA (~340s).
# Covers critical paths without running the full 8000+ GPUArrays tests.
#
# Usage:
#   julia --project test/test_ci.jl
#   LAVA_FULL_TESTS=1 julia --project test/runtests.jl  # full suite
#
# What's included:
#   1. SPIR-V emission checks (CPU only, fast)
#   2. Golden file comparison (CPU only, fast)
#   3. GPU smoke tests (basic compute, structs, atomics, tuples, reductions)
#   4. Mapreduce with complex structs (NTuple, multi-field, mixed types)
#   5. GPUArrays broadcasting subset (closures, fusion, tuples, multi-dim)

using Test
using Lava, Mantle
using KernelAbstractions
using GPUArrays

# `spirv_test_utils.jl` is Lava's: it disassembles and pattern-matches SPIR-V,
# which is compiler territory, and it moved to `Lava/test/` with the rest of the
# emission suite. Reached through `pkgdir(Lava)` rather than a path relative to
# this file, so it does not have to be re-derived if either tree moves again.
include(joinpath(pkgdir(Lava), "test", "spirv_test_utils.jl"))
import .SPIRVTestUtils: check, check_not, check_dag, check_sequence, check_count,
    check_regex, normalize_spirv, compare_golden, compile_and_disasm,
    spirv_opt_roundtrip, check_vendor_safety, compile_with_llc

let ctx = Mantle.vk_context()
    has_rt = ctx.rt_pipeline_properties !== nothing
    @info "Lava CI suite" device=ctx.device_name has_rt
end

@testset "Lava.jl CI" begin

    # ── Tier 1: SPIR-V Emission (CPU only) ──────────────────────────────
    # These are fast (~5s total) and catch compiler bugs without GPU.
    @testset "SPIR-V Emission" begin
        # `test/spirv/` went to Lava with the emitter, for the same reason its
        # helpers did: these compile a kernel and check the disassembly, and
        # need no device at all. They are `Lava/test/runtests.jl`'s Tier 1 now,
        # and this suite keeps running them because "fast CI" wants the compiler
        # checks and the GPU smoke tests in ONE invocation.
        spirv_dir = joinpath(pkgdir(Lava), "test", "spirv")
        for f in sort(readdir(spirv_dir; join=true))
            endswith(f, ".jl") || continue
            @info "Running $(basename(f))..."
            include(f)
        end
    end

    # Golden file tests skipped in CI — SPIR-V IDs differ across platforms
    # (LLVM CSE produces different output on NVIDIA vs lavapipe).
    # The SPIR-V emission pattern tests above catch structural issues.

    # ── Tier 3: GPU Smoke Tests ─────────────────────────────────────────
    # Targeted tests that exercise critical code paths.

    @testset "GPU Compute" begin
        # Basic vector add
        a = LavaArray(Float32[1, 2, 3, 4, 5])
        b = LavaArray(Float32[10, 20, 30, 40, 50])
        c = a .+ b
        @test Array(c) == Float32[11, 22, 33, 44, 55]

        # Broadcast fusion
        x = LavaArray(rand(Float32, 100))
        y = LavaArray(rand(Float32, 100))
        z = @. x * 2.0f0 + y - 1.0f0
        xc, yc = Array(x), Array(y)
        @test Array(z) ≈ @. xc * 2.0f0 + yc - 1.0f0

        # 2D broadcast with different shapes
        m = LavaArray(rand(Float32, 4, 8))
        v = LavaArray(rand(Float32, 1, 8))
        r = m .+ v
        @test Array(r) ≈ Array(m) .+ Array(v)
    end

    @testset "Tuple Broadcast" begin
        # This was the lavapipe segfault bug — regression test
        N = 10
        out = LavaArray(zeros(Float32, 3, N))
        arr = LavaArray(rand(Float32, 3, N))
        a = LavaArray(rand(Float32, N))
        b = LavaArray(rand(Float32, N))
        c = LavaArray(rand(Float32, N))

        broadcast!(out, arr, (a, b, c)) do xx, yy
            xx + first(yy)
        end

        cpu_out = similar(Array(arr))
        broadcast!(cpu_out, Array(arr), (Array(a), Array(b), Array(c))) do xx, yy
            xx + first(yy)
        end
        @test Array(out) ≈ cpu_out

        # Also test last()
        broadcast!(out, arr, (a, b, c)) do xx, yy
            xx + last(yy)
        end
        broadcast!(cpu_out, Array(arr), (Array(a), Array(b), Array(c))) do xx, yy
            xx + last(yy)
        end
        @test Array(out) ≈ cpu_out
    end

    @testset "Reductions" begin
        x = LavaArray(rand(Float32, 1024))
        @test sum(x) ≈ sum(Array(x)) rtol=1e-4
        @test minimum(x) ≈ minimum(Array(x))
        @test maximum(x) ≈ maximum(Array(x))

        # 2D reduction
        m = LavaArray(rand(Float32, 32, 64))
        @test Array(sum(m; dims=1)) ≈ sum(Array(m); dims=1) rtol=1e-4
        @test Array(sum(m; dims=2)) ≈ sum(Array(m); dims=2) rtol=1e-4

        # Large reduction (multi-pass)
        big = LavaArray(rand(Float32, 100_000))
        @test sum(big) ≈ sum(Array(big)) rtol=1e-3
    end

    @testset "Mapreduce with Structs" begin
        # Complex struct types in mapreduce — a frequent source of regressions.
        # Tests the full pipeline: struct layout, BDA passing, shared memory, subgroup reductions.

        # NTuple field (common in spectral rendering)
        struct SpectrumVal
            wavelengths::NTuple{4, Float32}
        end
        data = [SpectrumVal(ntuple(j -> rand(Float32), 4)) for _ in 1:512]
        ga = LavaArray(data)
        # sum of first wavelength
        gpu_sum = mapreduce(x -> x.wavelengths[1], +, ga)
        cpu_sum = mapreduce(x -> x.wavelengths[1], +, data)
        @test gpu_sum ≈ cpu_sum rtol=1e-3

        # Multi-field struct
        struct Vec3f32
            x::Float32
            y::Float32
            z::Float32
        end
        vecs = [Vec3f32(rand(Float32), rand(Float32), rand(Float32)) for _ in 1:1024]
        gv = LavaArray(vecs)
        # Sum of magnitudes squared
        gpu_mag = mapreduce(v -> v.x^2 + v.y^2 + v.z^2, +, gv)
        cpu_mag = mapreduce(v -> v.x^2 + v.y^2 + v.z^2, +, vecs)
        @test gpu_mag ≈ cpu_mag rtol=1e-3

        # Struct with mixed types (common in Hikari WorkItems)
        struct WorkItem
            origin_x::Float32
            origin_y::Float32
            origin_z::Float32
            pixel_index::Int32
            weight::Float32
        end
        items = [WorkItem(rand(Float32), rand(Float32), rand(Float32),
                          Int32(i), rand(Float32)) for i in 1:256]
        gi = LavaArray(items)
        gpu_wsum = mapreduce(w -> w.weight, +, gi)
        cpu_wsum = mapreduce(w -> w.weight, +, items)
        @test gpu_wsum ≈ cpu_wsum rtol=1e-3

        # Minimum/maximum with struct accessor
        gpu_min = mapreduce(w -> w.weight, min, gi)
        cpu_min = mapreduce(w -> w.weight, min, items)
        @test gpu_min ≈ cpu_min

        # 2D reduction with structs
        mat = LavaArray(reshape([Vec3f32(rand(Float32), rand(Float32), rand(Float32))
                                  for _ in 1:128], 8, 16))
        gpu_col = Array(mapreduce(v -> v.x, +, mat; dims=1))
        cpu_col = mapreduce(v -> v.x, +, Array(mat); dims=1)
        @test gpu_col ≈ cpu_col rtol=1e-3

        # ComplexF32 reduction (uses struct-like layout internally)
        cx = LavaArray(rand(ComplexF32, 512))
        @test sum(cx) ≈ sum(Array(cx)) rtol=1e-3
        @test mapreduce(abs2, +, cx) ≈ mapreduce(abs2, +, Array(cx)) rtol=1e-3
    end

    @testset "KernelAbstractions" begin

        @kernel function vadd_ka(a, b, c)
            i = @index(Global)
            @inbounds c[i] = a[i] + b[i]
        end

        a = LavaArray(rand(Float32, 256))
        b = LavaArray(rand(Float32, 256))
        c = LavaArray(zeros(Float32, 256))
        vadd_ka(Mantle.LavaBackend())(a, b, c; ndrange=256)
        KernelAbstractions.synchronize(Mantle.LavaBackend())
        @test Array(c) ≈ Array(a) .+ Array(b)

        # Shared memory + synchronize
        @kernel function reduce_ka(input, output)
            lid = @index(Local)
            gid = @index(Group)
            shared = @localmem Float32 (64,)
            @inbounds shared[lid] = input[(gid - 1) * 64 + lid]
            @synchronize()
            if lid == 1
                s = 0.0f0
                for j in 1:64
                    s += shared[j]
                end
                @inbounds output[gid] = s
            end
        end

        input = LavaArray(ones(Float32, 256))
        output = LavaArray(zeros(Float32, 4))
        reduce_ka(Mantle.LavaBackend())(input, output; ndrange=256, workgroupsize=64)
        KernelAbstractions.synchronize(Mantle.LavaBackend())
        @test all(Array(output) .≈ 64.0f0)
    end

    @testset "Barrier deadlock fix" begin
        # Regression test: error paths must not skip barriers.
        # replace_unreachable! converts trap+unreachable to early returns,
        # which can skip @synchronize() calls. fix_barrier_skipping_paths!
        # redirects these paths through the barrier-containing continuation.
        # Without the fix, this deadlocks on lavapipe (software Vulkan).

        @kernel function barrier_error_kernel(A, kill_idx)
            i = @index(Global)
            @synchronize()
            if i == kill_idx[1]
                error("dead")
            end
            @synchronize()
            A[i] = Int32(i)
        end

        backend = Mantle.LavaBackend()
        A = LavaArray(zeros(Int32, 128))
        kill = LavaArray(Int32[64])
        barrier_error_kernel(backend)(A, kill; ndrange=128, workgroupsize=128)
        KernelAbstractions.synchronize(backend)
        result = Array(A)

        # Dead thread falls through and writes its value
        @test result[64] == Int32(64)
        # All other threads write correctly
        @test all(result[i] == Int32(i) for i in 1:128 if i != 64)
    end

    @testset "Struct Broadcast" begin
        include(joinpath(@__DIR__, "test_struct_broadcast.jl"))
    end

    @testset "Atomics & Dispatch" begin
        include(joinpath(@__DIR__, "test_atomics_and_dispatch.jl"))
    end

    @testset "Memory Safety" begin
        include(joinpath(@__DIR__, "test_gpu_memory_safety.jl"))
    end

    @testset "Graphics Pipeline" begin
        include(joinpath(@__DIR__, "test_graphics_pipeline.jl"))
    end

    # ── Tier 4: GPUArrays Subset ────────────────────────────────────────
    # Run only the most important groups that catch real bugs.
    # broadcasting = tuple/closure/fusion bugs
    # base = copyto!/fill!/conversions
    # reductions/mapreduce = complex kernel + shared memory + multi-pass
    @testset "GPUArrays Subset" begin
        gpuarrays_testsuite = joinpath(dirname(dirname(pathof(GPUArrays))), "test", "testsuite.jl")
        include(gpuarrays_testsuite)
        TestSuite.supported_eltypes(::Type{<:LavaArray}) = (Int16, Int32, Int64,
                                                             Float16, Float32, Float64,
                                                             ComplexF16, ComplexF32, ComplexF64,
                                                             Complex{Int16}, Complex{Int32}, Complex{Int64})
        GPUArrays.allowscalar(false)

        # broadcasting is the most complex group: closures, fusion, tuples, multi-dim
        # base and reductions are covered by the smoke tests above
        ci_groups = ["broadcasting"]

        for name in ci_groups
            @testset "$name" begin
                TestSuite.tests[name](LavaArray)
            end
        end
    end

end
