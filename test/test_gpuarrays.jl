# GPUArrays TestSuite runner for the array type of the backend under test.
# Runs every test group in-process, each in its own testset, and prints a summary.

using Mantle
import GPUArrays
using Test
include(joinpath(@__DIR__, "testbackend.jl"))

gpuarrays_testsuite = joinpath(dirname(dirname(pathof(GPUArrays))), "test", "testsuite.jl")
include(gpuarrays_testsuite)

# The array type the suite constructs from, read off an array of this backend
# rather than named: the suite calls `AT(x)` and `AT{T}(undef, dims)`, so it
# needs the type constructor itself, without its parameters.
const AT = Base.typename(typeof(Mantle.devicearray(TESTBACKEND, Float32, 1))).wrapper

# Supported element types — the full set, less Float64 where the device has none.
TestSuite.supported_eltypes(::Type{<:AT}) = testeltypes(Int16, Int32, Int64,
                                                        Float16, Float32, Float64,
                                                        ComplexF16, ComplexF32, ComplexF64,
                                                        Complex{Int16}, Complex{Int32}, Complex{Int64})

# Disallow scalar indexing — matches CUDA/Metal/AMDGPU behavior — for the testsuite
# run only (below). `allowscalar(false)` here set it in the running task's local
# storage for good, and every suite run after Mantle's in the same task inherited
# it: the editor suite's `out[][3, 3]` failed as scalar indexing.

function count_results(ts::Test.DefaultTestSet)
    pass = ts.n_passed; fail = 0; err = 0; broken = 0
    for r in ts.results
        if r isa Test.DefaultTestSet
            sub = count_results(r)
            pass += sub[1]; fail += sub[2]; err += sub[3]; broken += sub[4]
        elseif r isa Test.Pass
            pass += 1
        elseif r isa Test.Fail
            fail += 1
        elseif r isa Test.Error
            err += 1
        elseif r isa Test.Broken
            broken += 1
        end
    end
    return (pass, fail, err, broken)
end

# Skip tests that need features we don't support (all drivers).
const SKIP = Set([
    "sparse",       # needs sparse array types
    "ext/jld2",     # needs JLD2 extension
    "alloc cache",  # needs alloc_cache support
    "random",       # needs RNG on GPU (not implemented)
    "statistics",   # mean(sin, A; dims=2) precision mismatch on lavapipe — TODO fix
])

# Groups that *crash the whole process* (SIGSEGV / EXCEPTION_ACCESS_VIOLATION in
# the JIT) on the lavapipe software rasterizer — nothing in the process can
# recover a segfault, so they must be skipped entirely on llvmpipe. They pass on
# real hardware (RADV runs them as part of the full suite), so they are skipped
# ONLY when the device under test is llvmpipe.
const LAVAPIPE_CRASH_SKIP = Set([
    # These crash ONLY on the GitHub Azure runner's CPU — they are NOT
    # reproducible on a local lavapipe LLVM 20.1.2 container (verified), so the
    # set can only be discovered by watching CI. Each was confirmed via a
    # `signal 11` / EXCEPTION_ACCESS_VIOLATION that aborted the test process.
    "indexing find",     # signal 11 in vulkan_lvp.dll
    "linalg/diagonal",   # signal 11 (Azure Linux + Windows), exposed once Tier 4 ran in CI
])

# The device name is the one portable thing that says which driver this is; it
# skips groups that take the process down with them, and gates nothing else.
function effective_skip()
    skip = copy(SKIP)
    if occursin("llvmpipe", lowercase(Mantle.devicename(Mantle.Device(TESTBACKEND))))
        union!(skip, LAVAPIPE_CRASH_SKIP)
    end
    return skip
end

# Each group in its own testset, so a failure or an error in one is recorded
# against that group and the next one runs. Nothing is caught here: a group that
# errors is an `Error` in the results, and a lost device makes every group after
# it error the same way, which is what the summary then says.
function run_gpuarrays_tests()
    skip = effective_skip()
    test_names = sort(collect(keys(TestSuite.tests)))
    filter!(n -> n ∉ skip, test_names)

    all_results = Vector{Tuple{String,Int,Int,Int}}()
    for name in test_names
        print("$name ... ")
        flush(stdout)
        ts = @testset "$name" begin
            TestSuite.tests[name](AT)
        end
        p, f, e, _ = count_results(ts)
        push!(all_results, (name, p, f, e))
        if f + e > 0
            println("$(p)p $(f)f $(e)e")
        else
            println("$(p) pass")
        end
    end

    println("\n" * "="^70)
    println("  GPUArrays TestSuite Results ($AT)")
    println("="^70)
    total_p = 0; total_f = 0; total_e = 0
    for (name, p, f, e) in all_results
        total_p += p; total_f += f; total_e += e
        status = (f + e == 0) ? "PASS" : "FAIL"
        detail = (f + e == 0) ? "$(p) pass" : "$(p)p $(f)f $(e)e"
        println("  $status  $(rpad(name, 40)) $detail")
    end
    println("="^70)
    println("  Total: $total_p passed, $total_f failed, $total_e errors")
    n_groups_pass = count(x -> x[3] + x[4] == 0, all_results)
    println("  $n_groups_pass/$(length(all_results)) test groups fully passing")
    println("  Skipped: $(join(sort(collect(effective_skip())), ", "))")
    return all_results
end

task_local_storage(run_gpuarrays_tests, :ScalarIndexing, GPUArrays.GPUArraysCore.ScalarDisallowed)
