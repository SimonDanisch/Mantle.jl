# test_atomics_and_dispatch.jl
#
# Tests for:
# 1. Atomic memory semantics (Vulkan memory model: MakeAvailable/MakeVisible)
# 2. Batched dispatch (record_dispatch! pattern, push_constants!, barriers)
# 3. Cross-workgroup atomic visibility (the BVH refit pattern)
#
# These tests verify both SPIR-V emission patterns (Tier 1) and GPU execution (Tier 3).

using Test
using Lava, Mantle
using KernelAbstractions
using Atomix

# ═══════════════════════════════════════════════════════════════════════
# Tier 1: SPIR-V Emission — Atomic Memory Semantics
# ═══════════════════════════════════════════════════════════════════════

@testset "Atomic Memory Semantics (SPIR-V)" begin

    @testset "monotonic atomicrmw → Relaxed (no MakeAvailable/Visible)" begin
        function monotonic_add(counter)
            Atomix.@atomic counter[1] += Int32(1)
            return nothing
        end
        result = Lava.lava_compile_gpu(monotonic_add,
            Tuple{Lava.LavaDeviceArray{Int32,1}}; workgroup_size=(64,1,1), validate=true)
        d = Lava.disassemble_spirv(result.spirv_bytes)
        @test occursin("OpAtomicIAdd", d)
        # Monotonic → Relaxed semantics — no MakeAvailable/MakeVisible flags
        @test !occursin("MakeVisible", d)
        @test !occursin("MakeAvailable", d)
    end

    @testset "monotonic f32 atomic → hardware OpAtomicFAddEXT" begin
        function f32_fadd(counter)
            Atomix.@atomic counter[1] += 1.0f0
            return nothing
        end
        result = Lava.lava_compile_gpu(f32_fadd,
            Tuple{Lava.LavaDeviceArray{Float32,1}}; workgroup_size=(64,1,1), validate=true)
        d = Lava.disassemble_spirv(result.spirv_bytes)
        # Float32 atomics use VK_EXT_shader_atomic_float's OpAtomicFAddEXT —
        # no more CAS loop (the previous emulation went through cmpxchg).
        @test occursin("OpAtomicFAddEXT", d)
        @test occursin("AtomicFloat32AddEXT", d)
        @test !occursin("OpAtomicCompareExchange", d)
        @test !occursin("MakeVisible", d)
        @test !occursin("MakeAvailable", d)
    end

    @testset "no CrossDevice scope" begin
        function any_atomic(counter, data)
            i = Lava.lava_global_invocation_id_x()
            Atomix.@atomic counter[1] += Int32(1)
            Atomix.@atomic counter[1] += data[i]
            return nothing
        end
        result = Lava.lava_compile_gpu(any_atomic,
            Tuple{Lava.LavaDeviceArray{Int32,1}, Lava.LavaDeviceArray{Int32,1}};
            workgroup_size=(64,1,1), validate=true)
        d = Lava.disassemble_spirv(result.spirv_bytes)
        @test !occursin("CrossDevice", d)
        @test !occursin("OpUnreachable", d)
    end

    # shared memory atomics tested in test_compute_memory.jl (requires @kernel context)
end

# ═══════════════════════════════════════════════════════════════════════
# Tier 3: GPU Execution — Atomic Correctness
# ═══════════════════════════════════════════════════════════════════════

@testset "Atomic GPU Execution" begin

    @testset "atomic counter (Int32)" begin
        @kernel function atomic_counter_i32!(counter)
            Atomix.@atomic counter[1] += Int32(1)
        end

        N = 4096
        counter = MVE.LavaArray(Int32[0])
        atomic_counter_i32!(MVE.LavaBackend())(counter; ndrange=N)
        MVE.vk_flush!(MVE.vk_context())
        @test Array(counter)[1] == Int32(N)
    end

    @testset "atomic counter (UInt32)" begin
        @kernel function atomic_counter_u32!(counter)
            Atomix.@atomic counter[1] += UInt32(1)
        end

        N = 4096
        counter = MVE.LavaArray(UInt32[0])
        atomic_counter_u32!(MVE.LavaBackend())(counter; ndrange=N)
        MVE.vk_flush!(MVE.vk_context())
        @test Array(counter)[1] == UInt32(N)
    end

    @testset "atomic counter (Float32 CAS)" begin
        @kernel function atomic_counter_f32!(counter)
            Atomix.@atomic counter[1] += 1.0f0
        end

        N = 1024
        counter = MVE.LavaArray(Float32[0])
        atomic_counter_f32!(MVE.LavaBackend())(counter; ndrange=N)
        MVE.vk_flush!(MVE.vk_context())
        @test Array(counter)[1] ≈ Float32(N)
    end

    @testset "atomic returns unique values" begin
        @kernel function atomic_unique!(counter, results)
            i = @index(Global)
            old = Atomix.@atomic counter[1] += Int32(1)
            @inbounds results[i] = old
        end

        N = 2048
        counter = MVE.LavaArray(Int32[0])
        results = MVE.LavaArray(zeros(Int32, N))
        atomic_unique!(MVE.LavaBackend())(counter, results; ndrange=N)
        MVE.vk_flush!(MVE.vk_context())

        r = Array(results)
        @test Array(counter)[1] == Int32(N)
        @test length(unique(r)) == N
        # Atomix.@atomic += returns the new value (1..N)
        @test sort(r) == collect(Int32(1):Int32(N))
    end

    @testset "cross-workgroup atomic visibility (BVH refit pattern)" begin
        # Simulates BVH bottom-up refit:
        # - N leaf threads write data[i] then atomically increment flags[parent]
        # - When flags[parent] == 2, the second-arriving thread reads BOTH children
        # - With broken visibility (Relaxed-only), second thread sees stale data
        @kernel function refit_pattern!(data, flags, results, parents, siblings)
            i = @index(Global)
            n = @uniform @groupsize()[1] * @ndrange()[1] ÷ @groupsize()[1]

            # Write leaf data (non-atomic)
            @inbounds data[i] = Float32(i * 10)

            # Atomic increment on parent flag
            @inbounds parent = parents[i]
            @inbounds sibling = siblings[i]
            old = Atomix.@atomic flags[parent] += Int32(1)

            if old == Int32(1)
                # Second thread: both children done, read sibling data
                @inbounds sibling_val = data[sibling]
                @inbounds my_val = data[i]
                @inbounds results[parent] = my_val + sibling_val
            end
        end

        N = 64
        P = N ÷ 2

        data = MVE.LavaArray(zeros(Float32, N))
        flags = MVE.LavaArray(zeros(Int32, P))
        results = MVE.LavaArray(zeros(Float32, P))

        # Pair leaves: (1,2)→parent 1, (3,4)→parent 2, ...
        parents_h = Int32[div(i - 1, 2) + 1 for i in 1:N]
        siblings_h = Int32[i % 2 == 1 ? i + 1 : i - 1 for i in 1:N]
        parents_d = MVE.LavaArray(parents_h)
        siblings_d = MVE.LavaArray(siblings_h)

        refit_pattern!(MVE.LavaBackend())(data, flags, results, parents_d, siblings_d; ndrange=N)
        MVE.vk_flush!(MVE.vk_context())

        r = Array(results)
        expected = Float32[(2k - 1) * 10 + (2k) * 10 for k in 1:P]
        @test r == expected
    end
end

# ═══════════════════════════════════════════════════════════════════════
# Tier 3: GPU Execution — Batched Dispatch
# ═══════════════════════════════════════════════════════════════════════

@testset "Batched Dispatch" begin

    @testset "multiple dispatches in one batch" begin
        a = MVE.LavaArray(Float32[1, 2, 3, 4])
        b = MVE.LavaArray(Float32[10, 20, 30, 40])
        c = a .+ b
        d = c .* Float32(2)
        MVE.vk_flush!(MVE.vk_context())
        @test Array(d) == Float32[22, 44, 66, 88]
    end

    @testset "flush counter increments" begin
        before = MVE.vk_context().diag.flush_counter[]
        a = MVE.LavaArray(Float32[1, 2, 3])
        _ = a .+ Float32(1)
        MVE.vk_flush!(MVE.vk_context())
        @test MVE.vk_context().diag.flush_counter[] > before
    end

    @testset "dispatch counter increments" begin
        MVE.vk_context().diag.dispatch_logging = true
        before = MVE.vk_context().diag.total_dispatches[]
        a = MVE.LavaArray(Float32[1, 2, 3])
        _ = a .+ Float32(1)
        MVE.vk_flush!(MVE.vk_context())
        @test MVE.vk_context().diag.total_dispatches[] > before
        MVE.vk_context().diag.dispatch_logging = false
    end

    @testset "KA.synchronize flushes GPU work" begin
        a = MVE.LavaArray(ones(Float32, 64))
        before = MVE.vk_context().diag.flush_counter[]
        b = a .+ Float32(1)
        c = b .+ Float32(1)
        KernelAbstractions.synchronize(MVE.LavaBackend())
        # synchronize triggers a flush
        @test MVE.vk_context().diag.flush_counter[] - before >= 1
        @test Array(c) == fill(Float32(3), 64)
    end

    @testset "empty flush is no-op" begin
        before = MVE.vk_context().diag.flush_counter[]
        MVE.vk_flush!(MVE.vk_context())
        @test MVE.vk_context().diag.flush_counter[] == before
    end

    @testset "barrier between dispatches preserves ordering" begin
        @kernel function write_val!(A, val::Float32)
            i = @index(Global)
            @inbounds A[i] = val
        end
        @kernel function read_add!(B, A)
            i = @index(Global)
            @inbounds B[i] = A[i] + 1.0f0
        end

        N = 256
        A = MVE.LavaArray(zeros(Float32, N))
        B = MVE.LavaArray(zeros(Float32, N))
        write_val!(MVE.LavaBackend())(A, 42.0f0; ndrange=N)
        read_add!(MVE.LavaBackend())(B, A; ndrange=N)
        MVE.vk_flush!(MVE.vk_context())
        @test all(Array(B) .== 43.0f0)
    end

    @testset "data kept alive across GC + flush" begin
        @kernel function fill_index!(A)
            i = @index(Global)
            @inbounds A[i] = Float32(i)
        end

        A = MVE.LavaArray(zeros(Float32, 1024))
        fill_index!(MVE.LavaBackend())(A; ndrange=1024)
        GC.gc()
        MVE.vk_flush!(MVE.vk_context())
        result = Array(A)
        @test result[1] == 1.0f0
        @test result[1024] == 1024.0f0
    end

    @testset "dispatch log records entries" begin
        MVE.vk_context().diag.dispatch_logging = true
        empty!(MVE.vk_context().diag.dispatch_log)
        a = MVE.LavaArray(Float32[1, 2, 3])
        _ = a .+ Float32(1)
        MVE.vk_flush!(MVE.vk_context())
        @test !isempty(MVE.vk_context().diag.dispatch_log)
        MVE.vk_context().diag.dispatch_logging = false
    end
end

# ═══════════════════════════════════════════════════════════════════════
# Tier 3: GPU Execution — Multi-dispatch Patterns
# ═══════════════════════════════════════════════════════════════════════

@testset "Multi-dispatch Patterns" begin

    @testset "chain of 10 dispatches" begin
        a = MVE.LavaArray(ones(Float32, 128))
        for _ in 1:10
            a = a .+ Float32(1)
        end
        MVE.vk_flush!(MVE.vk_context())
        @test all(Array(a) .== 11.0f0)
    end

    @testset "interleaved compute and reduction" begin
        a = MVE.LavaArray(Float32[1, 2, 3, 4, 5, 6, 7, 8])
        b = a .* Float32(2)
        MVE.vk_flush!(MVE.vk_context())
        @test sum(b) ≈ 72.0f0
    end

    @testset "large dispatch without splitting" begin
        N = 128 * 256 * 2
        a = MVE.LavaArray(ones(Float32, N))
        b = a .+ Float32(1)
        KernelAbstractions.synchronize(MVE.LavaBackend())
        @test all(Array(b) .== 2.0f0)
    end
end

# ═══════════════════════════════════════════════════════════════════════
# Tier 3: GPU Execution — Atomix with CartesianIndex + Float32 subtract
# ═══════════════════════════════════════════════════════════════════════

@testset "KA @private (Scratchpad)" begin
    @kernel function private_accum!(A)
        I = @index(Global)
        priv = @private Float32 (4,)
        @inbounds for k in 1:4
            priv[k] = Float32(I * k)
        end
        @inbounds A[I] = priv[1] + priv[4]
    end

    a = MVE.LavaArray(zeros(Float32, 64))
    private_accum!(MVE.LavaBackend(), 64)(a; ndrange=64)
    MVE.vk_flush!(MVE.vk_context())
    result = Array(a)
    @test result[1] == 1f0 + 4f0
    @test result[10] == 10f0 + 40f0
end

@testset "Int64/UInt64 atomics" begin
    @testset "Int64 atomic add" begin
        @kernel function atomic_add_i64!(counter)
            Atomix.@atomic counter[1] += Int64(1)
        end
        N = 2048
        c = MVE.LavaArray(Int64[0])
        atomic_add_i64!(MVE.LavaBackend())(c; ndrange=N)
        MVE.vk_flush!(MVE.vk_context())
        @test Array(c)[1] == Int64(N)
    end

    @testset "UInt64 atomic add" begin
        @kernel function atomic_add_u64!(counter)
            Atomix.@atomic counter[1] += UInt64(1)
        end
        N = 1024
        c = MVE.LavaArray(UInt64[0])
        atomic_add_u64!(MVE.LavaBackend())(c; ndrange=N)
        MVE.vk_flush!(MVE.vk_context())
        @test Array(c)[1] == UInt64(N)
    end
end

@testset "Float64 atomics" begin
    @testset "Float64 atomic add" begin
        @kernel function atomic_add_f64!(counter)
            Atomix.@atomic counter[1] += 1.0
        end
        N = 1024
        c = MVE.LavaArray(Float64[0.0])
        atomic_add_f64!(MVE.LavaBackend())(c; ndrange=N)
        MVE.vk_flush!(MVE.vk_context())
        @test Array(c)[1] ≈ Float64(N)
    end

    @testset "Float64 atomic subtract" begin
        @kernel function atomic_sub_f64!(counter)
            Atomix.@atomic counter[1] -= 1.0
        end
        N = 1024
        c = MVE.LavaArray(Float64[Float64(N)])
        atomic_sub_f64!(MVE.LavaBackend())(c; ndrange=N)
        MVE.vk_flush!(MVE.vk_context())
        @test Array(c)[1] ≈ 0.0
    end
end

@testset "Atomix CartesianIndex and Float32 subtract" begin

    @testset "Float32 atomic subtract" begin
        @kernel function atomic_sub_f32!(counter)
            Atomix.@atomic counter[1] -= 1.0f0
        end

        N = 1024
        counter = MVE.LavaArray(Float32[Float32(N)])
        atomic_sub_f32!(MVE.LavaBackend())(counter; ndrange=N)
        MVE.vk_flush!(MVE.vk_context())
        @test Array(counter)[1] ≈ 0.0f0
    end

    @testset "Atomix with CartesianIndex on 3D array" begin
        @kernel function atomic_add_3d!(A, idx_array)
            i = @index(Global)
            @inbounds ci = idx_array[i]
            Atomix.@atomic A[ci] += 1.0f0
        end

        dims = (4, 4, 4)
        A = MVE.LavaArray(zeros(Float32, dims))
        # All threads write to the same CartesianIndex
        target = CartesianIndex(2, 3, 1)
        N = 512
        idx_array = MVE.LavaArray(fill(target, N))
        atomic_add_3d!(MVE.LavaBackend())(A, idx_array; ndrange=N)
        MVE.vk_flush!(MVE.vk_context())
        result = Array(A)
        @test result[2, 3, 1] ≈ Float32(N)
        @test sum(result) ≈ Float32(N)  # only one cell was touched
    end

    @testset "Atomix subtract with CartesianIndex on 2D array" begin
        @kernel function atomic_sub_2d!(A, idx_array, val)
            i = @index(Global)
            @inbounds ci = idx_array[i]
            Atomix.@atomic A[ci] -= val
        end

        dims = (8, 8)
        N = 256
        target = CartesianIndex(4, 5)
        A = MVE.LavaArray(fill(Float32(N), dims))
        idx_array = MVE.LavaArray(fill(target, N))
        atomic_sub_2d!(MVE.LavaBackend())(A, idx_array, 1.0f0; ndrange=N)
        MVE.vk_flush!(MVE.vk_context())
        result = Array(A)
        @test result[4, 5] ≈ 0.0f0
        # All other cells untouched
        result[4, 5] = Float32(N)
        @test all(result .== Float32(N))
    end

    @testset "Atomix with N integer indices on 2D array (no CartesianIndex)" begin
        # `@atomic A[i, j] += v` expands to modifyindex_atomic!(A, ..., i, j) —
        # requires the Vararg{Integer,N} override, not the CartesianIndex one.
        @kernel function atomic_add_ij!(A, is, js)
            k = @index(Global, Linear)
            @inbounds i = is[k]
            @inbounds j = js[k]
            Atomix.@atomic A[i, j] += UInt32(1)
        end

        dims = (6, 5)
        A = MVE.LavaArray(zeros(UInt32, dims))
        # All threads hit (3, 4) with Int32 indices like the solar kernel does.
        N = 1024
        is = MVE.LavaArray(fill(Int32(3), N))
        js = MVE.LavaArray(fill(Int32(4), N))
        atomic_add_ij!(MVE.LavaBackend())(A, is, js; ndrange=N)
        MVE.vk_flush!(MVE.vk_context())
        result = Array(A)
        @test result[3, 4] == UInt32(N)
        @test sum(result) == UInt32(N)
    end
end
