# Regression tests for `fix_barrier_skipping_paths!`.
#
# Bug (no-inline path): an error path (`error(...)`) that returns early skips a
# workgroup `@synchronize()` that the other invocations still reach. Per the
# Vulkan spec every invocation must reach every control barrier, so this is a
# deadlock on lavapipe (software) and drops the dead invocation's later writes on
# real hardware. `fix_barrier_skipping_paths!` redirects such a path through the
# barrier-containing continuation instead of returning.
#
# The detection broke specifically on the no-inline path: each `@synchronize`
# survives as its own wrapper function (`call @llvm.spv...barrier; ret`) rather
# than an inlined intrinsic, so the pass saw zero barrier blocks and did nothing.
# `function_contains_barrier` now looks through such wrapper calls.

using Test
using Lava, Mantle
using LLVM
using KernelAbstractions
const KA = KernelAbstractions

@testset "fix_barrier_skipping_paths!" begin
    # ── Unit test: detection looks through wrapper-function barriers (no GPU) ──
    @testset "function_contains_barrier sees wrapped barriers" begin
        ir = """
        declare void @llvm.spv.group.memory.barrier.with.group.sync()

        define internal void @sync_wrapper() {
          call void @llvm.spv.group.memory.barrier.with.group.sync()
          ret void
        }

        define internal void @plain_helper() {
          ret void
        }

        define internal void @calls_wrapper() {
          call void @sync_wrapper()
          ret void
        }
        """
        LLVM.Context() do ctx
            mod = parse(LLVM.Module, ir)
            memo = Dict{LLVM.Function,Bool}()
            barrier = "llvm.spv.group.memory.barrier.with.group.sync"
            fns = LLVM.functions(mod)
            # Direct barrier wrapper, a transitive caller, and a plain helper.
            @test Lava.function_contains_barrier(fns["sync_wrapper"], barrier, memo)
            @test Lava.function_contains_barrier(fns["calls_wrapper"], barrier, memo)
            @test !Lava.function_contains_barrier(fns["plain_helper"], barrier, memo)
        end
    end

    # ── End-to-end: a dead invocation must still reach the 2nd barrier and
    #    write its value (would deadlock on lavapipe without the fix). ──
    @testset "error path participates in later barrier" begin
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
        KA.synchronize(backend)
        r = Array(A)

        # The "dead" invocation falls through the barrier and writes its value …
        @test r[64] == Int32(64)
        # … and every other invocation is unaffected (no deadlock, no corruption).
        @test all(r[i] == Int32(i) for i in 1:128 if i != 64)
    end
end
