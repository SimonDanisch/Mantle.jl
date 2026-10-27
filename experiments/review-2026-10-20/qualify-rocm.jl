# Run in the disposable SDK environment described in test/rocm/README.md.
using AMDGPU, Mantle, Test

AMDGPU.device!(first(AMDGPU.devices()))
global MANTLE_TEST_BACKEND = Mantle.Device(Mantle.ROCmAPI())
@assert nameof(typeof(MANTLE_TEST_BACKEND)) === :ROCmDevice
println((backend=typeof(MANTLE_TEST_BACKEND),device=AMDGPU.device()))

@testset "explicit ROCm qualification" begin
    for file in ("rocm/test_rocm.jl", "test_kernel_preparation.jl",
                 "test_gemm_staged_scalar.jl", "test_gemm_staged.jl",
                 "test_coopmat_epilogue.jl", "test_gemm_fp16_accum.jl",
                 "test_gemv.jl", "test_launch_plan_key.jl", "test_fft.jl",
                 "test_pending_stores.jl", "test_recordable_plans.jl",
                 "test_run_allocates_nothing.jl")
        @testset "$file" begin
            m = Module(gensym(:ROCmQualification))
            Core.eval(m,:(include(path)=Base.include(@__MODULE__,path)))
            Base.include(m,joinpath(@__DIR__,"../../test",file))
        end
    end
end
