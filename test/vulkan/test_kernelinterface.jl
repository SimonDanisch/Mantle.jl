"""
Lava's implementation of `KernelInterface` (KI).

KI is the cross-backend device/host/launch contract — the thing a kernel written
once has to mean the same on Lava, Metal and CUDA — so the assertions here are
about the CONTRACT, not about Lava. Each one is a rule KI states that a backend
can get wrong quietly:

  * indexing is **1-based**, where SPIR-V's builtins are 0-based. An off-by-one
    here is a kernel that skips element 1 and reads one past the end, on every
    backend that trusts the interface.
  * a **zero-sized `ndrange`** is a no-op returning `nothing`, not an error.
    Launching over an empty array is ordinary.
  * `ndrange` and `numworkgroups` are **mutually exclusive** — one says how much
    work there is, the other says how it is cut up.
  * a `workgroupsize` above the device's limit is an **error at the call**, not a
    dispatch the driver rejects later with a validation message.
  * the device queries answer from the DEVICE. `sub_group_size` is 32 on NVIDIA,
    32 or 64 on RDNA3 depending on how the driver compiled the shader, 8 on
    lavapipe, and nothing may hard-code it.

Both launch forms are exercised. `KI.@kernel` is sugar over `argconvert` /
`kernel_function` / calling the `Kernel`, and a backend can satisfy the macro
while leaving one of the three wrong — the macro-less form is what catches that.
"""

using Test, Lava, Mantle
import KernelInterface as KI

# A kernel per index query, each writing what it read so the host can check the
# base. Plain functions: KI kernels are not `KA.@kernel` definitions, they are
# ordinary code compiled for the device, and that difference is the point of the
# macro-less form below.
function ki_fill_kernel(out)
    i = KI.get_global_id().x
    @inbounds i <= length(out) && (out[i] = Float32(1))
    return nothing
end

function ki_global_id_kernel(out)
    i = KI.get_global_id().x
    @inbounds i <= length(out) && (out[i] = UInt32(i))
    return nothing
end

function ki_local_id_kernel(out)
    i = KI.get_global_id().x
    @inbounds i <= length(out) && (out[i] = UInt32(KI.get_local_id().x))
    return nothing
end

function ki_group_id_kernel(out)
    i = KI.get_global_id().x
    @inbounds i <= length(out) && (out[i] = UInt32(KI.get_group_id().x))
    return nothing
end

function ki_num_groups_kernel(out)
    i = KI.get_global_id().x
    @inbounds i <= length(out) && (out[i] = UInt32(KI.get_num_groups().x))
    return nothing
end

@testset "KernelInterface" begin
    backend = LavaBackend()

    @testset "launch, macro form" begin
        a = LavaArray(zeros(Float32, 64))
        KI.@kernel backend ndrange = 64 ki_fill_kernel(a)
        KI.synchronize(backend)
        @test all(Array(a) .== 1.0f0)
    end

    @testset "launch, macro-less form" begin
        # The three calls `KI.@kernel` expands to, spelled out. A backend can
        # satisfy the macro and still have `argconvert` or `kernel_function`
        # wrong, because the macro is free to route around them.
        a = LavaArray(zeros(Float32, 64))
        args = (a,)
        converted = map(x -> KI.argconvert(backend, x), args)
        tt = Tuple{map(Core.Typeof, converted)...}
        kernel = KI.kernel_function(backend, ki_fill_kernel, tt)
        kernel(args...; ndrange = 64)
        KI.synchronize(backend)
        @test all(Array(a) .== 1.0f0)

        # `argconvert` is a pure strip to the device-side form. It must NOT pin
        # or allocate: KI uses it only to derive the type to compile against, and
        # the pinning happens at launch, against the original arguments.
        @test converted[1] isa Lava.LavaDeviceArray{Float32, 1}
        @test size(converted[1]) == size(a)
        @test KI.argconvert(backend, 7) === 7
    end

    # SPIR-V's builtins are 0-based and KI's are 1-based, so every one of these
    # is a `+ 1` that has to be there exactly once.
    @testset "indexing is 1-based" begin
        n, wg = 64, 16

        got = LavaArray(zeros(UInt32, n))
        KI.@kernel backend ndrange = n workgroupsize = wg ki_global_id_kernel(got)
        KI.synchronize(backend)
        @test Array(got) == UInt32.(1:n)

        got = LavaArray(zeros(UInt32, n))
        KI.@kernel backend ndrange = n workgroupsize = wg ki_local_id_kernel(got)
        KI.synchronize(backend)
        # 1:wg, repeated once per workgroup. A 0-based leak shows up as a zero.
        @test Array(got) == UInt32.(repeat(1:wg, n ÷ wg))

        got = LavaArray(zeros(UInt32, n))
        KI.@kernel backend ndrange = n workgroupsize = wg ki_group_id_kernel(got)
        KI.synchronize(backend)
        @test Array(got) == UInt32.(repeat(1:(n ÷ wg), inner = wg))

        # A COUNT is a count in both numbering schemes, so this one must NOT
        # gain a one — the mirror-image mistake to the three above.
        got = LavaArray(zeros(UInt32, n))
        KI.@kernel backend ndrange = n workgroupsize = wg ki_num_groups_kernel(got)
        KI.synchronize(backend)
        @test all(Array(got) .== UInt32(n ÷ wg))
    end

    @testset "launch argument rules" begin
        a = LavaArray(zeros(Float32, 4))
        kernel = KI.kernel_function(backend, ki_fill_kernel,
                                    Tuple{Lava.LavaDeviceArray{Float32, 1}})

        # Empty work is not an error. `cld` would turn a zero extent into one
        # workgroup, which is a dispatch that writes nothing but still runs.
        @test kernel(a; ndrange = 0) === nothing
        @test kernel(a; ndrange = (4, 0)) === nothing
        @test kernel(a; numworkgroups = 0) === nothing
        @test all(Array(a) .== 0.0f0)

        # One says how much work there is, the other how it is cut up.
        @test_throws ArgumentError kernel(a; ndrange = 4, numworkgroups = 1)

        # Over the device's limit, refused here rather than by the driver.
        limit = KI.max_work_group_size(backend)
        @test_throws ArgumentError kernel(a; numworkgroups = 1, workgroupsize = limit + 1)
        # `max_work_group_size` also bounds it from the call site.
        @test_throws ArgumentError kernel(a; ndrange = 4, workgroupsize = 4,
                                          max_work_group_size = 2)
    end

    @testset "device queries" begin
        # Answered from the device, so the assertions are on shape and
        # plausibility rather than on a number this machine happens to report.
        @test KI.get_backend(LavaArray(zeros(Float32, 1))) isa LavaBackend
        @test KI.supports_unified(backend)
        @test KI.supports_atomics(backend)

        sg = KI.sub_group_size(backend)
        @test sg isa Integer && sg > 0 && ispow2(sg)
        @test sg == Lava.caps(backend).subgroup

        limit = KI.max_work_group_size(backend)
        @test limit >= 64                       # Vulkan's floor is 128
        @test limit == Lava.caps(backend).workgrouplimit
        @test KI.multiprocessor_count(backend) == Lava.caps(backend).cores

        # Per kernel, because register pressure can lower it; the device limit
        # until the driver gives a better number, and never above it.
        @test KI.kernel_max_work_group_size(backend, ki_fill_kernel) == limit
        @test KI.kernel_max_work_group_size(backend, ki_fill_kernel;
                                            max_work_items = 32) == 32

        # `allocate` goes through the same path as `KA.allocate`.
        a = KI.allocate(backend, Float32, (8,))
        @test a isa LavaArray{Float32, 1}
        @test size(a) == (8,)
    end

    @testset "shfl_down" begin
        # KI asks the backend which element types its shuffle covers, and the
        # answer drives KI's own suite — so the list must be the one Lava
        # actually generates methods for, not a hopeful superset.
        types = KI.shfl_down_types(backend)
        @test Set(types) == Set(Lava.KI_SHFL_TYPES)
        for T in types
            @test hasmethod(KI.shfl_down, Tuple{T, Int})
        end
    end

    @testset "cooperative matrices" begin
        # The vocabulary that used to be its own package. `matrix_shapes` is the
        # backend hook it gained in the move, and it has to agree with the
        # `DeviceCaps` accessors rather than being a second opinion.
        c = Lava.caps(backend)
        shapes = KI.matrix_shapes(backend)
        @test shapes == (c.coopmat ? c.shapes : KI.MatrixShape[])

        # `bestshape` returns `nothing` rather than a fabricated tile, on a
        # device without matrix hardware and for a type pair it does not list.
        @test KI.bestshape(backend, Float64, Float64) === nothing
        best = KI.bestshape(backend, Float16, Float32)
        if c.coopmat && !isempty(shapes)
            @test best === nothing || (KI.supports(backend, best) && best.acc === Float32)
        else
            @test best === nothing
        end
    end

    # Not implemented, and each for a reason that is a separate task rather than
    # an oversight. Asserted so that "missing" stays a decision:
    #
    #   * `get_local_size`/`get_global_size` need `WorkgroupSize` as an
    #     `OpConstantComposite`; Lava emits an Input variable, which spirv-val
    #     rejects (see test_builtin_validation.jl).
    #   * `get_max_sub_group_size` is `SubgroupMaxSize`, which SPIR-V puts under
    #     the Kernel (OpenCL) capability — illegal in a Vulkan module. It needs a
    #     specialization constant fed from the host.
    #   * `sub_group_barrier` is `OpControlBarrier Subgroup Subgroup`; Lava's
    #     only barrier is workgroup-scoped, and aliasing them would synchronise
    #     the wrong set of invocations while appearing to work.
    # Which of KI's device functions Lava implements, and by which of the two
    # mechanisms — because "implemented" is not one question here.
    #
    # A KI function with no host method (a bare `function f end`) is implemented
    # with a PLAIN method: there is nothing to shadow. One that KI gives a host
    # method — `barrier` errors, `_print` prints with `Base.print`, `localmemory`
    # errors — has to be a `@lava_device_override`, or the host behaviour KI
    # promises is replaced by an `llvmcall` of a SPIR-V intrinsic on the CPU.
    #
    # So `methods()` answers for the first group and says nothing about the
    # second: an overlay method lives in `Lava.lava_method_table`, which
    # GPUCompiler consults during compilation and `methods()` never sees.
    @testset "device-side coverage" begin
        overridden(f) = any(Base.MethodList(Lava.lava_method_table)) do m
            Base.unwrap_unionall(m.sig).parameters[1] === typeof(f)
        end

        @testset "plain methods, no host fallback to shadow" begin
            for f in (KI.get_global_id, KI.get_local_id, KI.get_group_id,
                      KI.get_num_groups, KI.get_sub_group_size,
                      KI.get_num_sub_groups, KI.get_sub_group_id,
                      KI.get_sub_group_local_id, KI.shfl_down)
                @test any(m -> m.module === Lava, methods(f))
            end
        end

        @testset "device overrides, host behaviour left intact" begin
            @test overridden(KI.barrier)
            @test overridden(KI._print)
            # Off-device they are still KI's: the barrier reports itself rather
            # than executing, and printing still prints.
            @test_throws ErrorException KI.barrier()
            @test all(m -> m.module === KI, methods(KI.barrier))
            @test all(m -> m.module === KI, methods(KI._print))
        end

        # Each of these is a separate task, not an oversight, and each would be
        # WRONG rather than merely absent if faked:
        #
        #   * `get_local_size`/`get_global_size` need `WorkgroupSize` as an
        #     `OpConstantComposite`; Lava emits an Input variable, which
        #     spirv-val rejects (see test_builtin_validation.jl).
        #   * `get_max_sub_group_size` is `SubgroupMaxSize`, which SPIR-V puts
        #     under the Kernel (OpenCL) capability — illegal in a Vulkan module.
        #     It needs a specialization constant fed from the host.
        #   * `sub_group_barrier` is `OpControlBarrier Subgroup Subgroup`; Lava's
        #     only barrier is workgroup-scoped, and aliasing them would
        #     synchronise the wrong set of invocations while appearing to work.
        #   * `localmemory` has no per-call-site id in KI's signature, so the
        #     only key is `(T, Dims)` and two identical calls in one kernel would
        #     silently share one buffer. `KA.@localmem` carries the id and works.
        @testset "gaps stay gaps" begin
            for f in (KI.get_local_size, KI.get_global_size, KI.get_max_sub_group_size)
                @test isempty(methods(f))        # KI declares these with no body
                @test !overridden(f)
            end
            for f in (KI.sub_group_barrier, KI.localmemory)
                @test !overridden(f)
                @test all(m -> m.module === KI, methods(f))
            end
            @test_throws ErrorException KI.sub_group_barrier()
        end
    end
end
