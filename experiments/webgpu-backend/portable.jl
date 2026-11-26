# Mantle's portable test files (the ones `runtests.jl` runs through `foreachbackend`)
# on the WebGPU device. Include `setup.jl` first, then `portable(files)`.
const MANTLE_TESTS = joinpath(pkgdir(Mantle), "test")
const PORTABLE_FILES = [
    "test_arena_recording.jl", "test_compile_golden.jl", "test_recorded_buffer_sync.jl",
    "test_devicerange.jl", "test_run_ordering.jl", "test_declared_kernel.jl",
    "test_kernel_preparation.jl", "test_kernel_tails.jl", "test_dynamic_tuple_getindex.jl",
    "test_declared_call.jl", "test_argument_packing.jl", "test_broadcast_paths.jl",
    "test_diagonal_mul.jl", "test_mul_nonnumeric_eltype.jl", "test_norm_overflow.jl",
    "test_permutedims.jl", "test_mapreduce_transposed.jl", "test_basealignment_device.jl",
    "test_buffer_alias.jl", "test_buffer_resize_length.jl", "test_download_readback.jl",
    "test_narrow_phase_contacts.jl", "test_narrow_phase_kernel.jl", "test_dropped_resources.jl",
    "test_submission_count.jl", "test_readback.jl", "test_kernel_error.jl",
    "test_pool_trim_policy.jl", "test_tolerated_alloc_failure.jl",
]

Mantle.defaultdevice!(dev)
@eval Main MANTLE_TEST_BACKEND = $dev

"""Each file in a module of its own, as `foreachbackend` runs it."""
function portable(files = PORTABLE_FILES)
    @testset "Mantle portable tests on WebGPU" begin
        for f in files
            @testset "$f" begin
                path = joinpath(MANTLE_TESTS, f)
                @eval Main module $(gensym(:PortableRun))
                    using Test
                    include($path)
                end
            end
        end
    end
end
