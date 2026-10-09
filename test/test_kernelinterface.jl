"""
`KernelInterface` (KI) on the backend under test.

KI is the cross-backend device/host/launch contract — the thing a kernel written
once has to mean the same on Vulkan, Metal and CUDA — so the assertions here are
about the CONTRACT, not about one backend. Each one is a rule KI states that a
backend can get wrong quietly:

  * indexing is **1-based**, where SPIR-V's builtins are 0-based. An off-by-one
    here is a kernel that skips element 1 and reads one past the end, on every
    backend that trusts the interface.
  * a **zero-sized `ndrange`** is a no-op returning `nothing`, not an error.
    Launching over an empty array is ordinary.
  * `ndrange` and `numgroups` are **mutually exclusive** — one says how much
    work there is, the other says how it is cut up.
  * a `workgroupsize` above the device's limit is an **error at the call**, not a
    dispatch the driver rejects later with a validation message.
  * the device queries answer from the DEVICE. `sub_group_size` is 32 on NVIDIA,
    32 or 64 on RDNA3 depending on how the driver compiled the shader, 8 on
    lavapipe, and nothing may hard-code it. And KI's answer and `Mantle.caps`'
    answer are two handles on one device: they must not differ.
  * a 1-D workgroup no wider than `sub_group_size` is ONE sub-group, numbered
    from 1, and the sub-group queries say so.
  * two workgroup buffers of one type and shape are two buffers when their ids
    differ, and a barrier makes one work-item's store visible to the others.

Both launch forms are exercised. `KI.@launch` is sugar over `argconvert` /
`kernel_function` / calling the `Kernel`, and a backend can satisfy the macro
while leaving one of the three wrong — the macro-less form is what catches that.
"""

using Test, Mantle
import KernelInterface as KI
include(joinpath(@__DIR__, "testbackend.jl"))

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

# Every sub-group query, launched with workgroups exactly one sub-group wide. The
# reduction is outside the bounds check: every lane of the sub-group must reach it.
function ki_subgroup_kernel(sz, lane, nsg, sgid, red)
    i = KI.get_global_id().x
    l = KI.get_sub_group_local_id(Int32)
    r = KI.sub_group_reduce_add(Float32(l))
    @inbounds if i <= length(sz)
        sz[i] = KI.get_sub_group_size(Int32)
        lane[i] = l
        nsg[i] = KI.get_num_sub_groups(Int32)
        sgid[i] = KI.get_sub_group_id(Int32)
        red[i] = r
    end
    return nothing
end

# The workgroup size of `ki_localmem_kernel`, which its buffers are declared at.
const KI_WG = 64

# Two workgroup buffers of the same element type and the same shape, told apart
# only by their id, and a barrier between writing one's own slot and reading a
# NEIGHBOUR's. If the ids were decoration the two buffers would be one and the
# second store would land on the first tile; without the barrier the read could
# see the slot before its owner wrote it.
function ki_localmem_kernel(outa, outb)
    a = KI.localmemory(Float32, Val((KI_WG,)), Val(1))
    b = KI.localmemory(Float32, Val((KI_WG,)), Val(2))
    t = KI.get_local_id().x
    g = KI.get_global_id().x
    @inbounds a[t] = Float32(t)
    @inbounds b[t] = -Float32(t)
    KI.barrier()
    @inbounds outa[g] = a[KI_WG + 1 - t]
    @inbounds outb[g] = b[KI_WG + 1 - t]
    return nothing
end

@testset "KernelInterface" begin
    backend = TESTBACKEND
    c = Mantle.caps(backend)

    @testset "launch, macro form" begin
        a = Mantle.devicearray(backend, zeros(Float32, 64))
        KI.@launch backend ndrange = 64 ki_fill_kernel(a)
        KI.synchronize(backend)
        @test all(Array(a) .== 1.0f0)
    end

    @testset "launch, macro-less form" begin
        # The three calls `KI.@launch` expands to, spelled out. A backend can
        # satisfy the macro and still have `argconvert` or `kernel_function`
        # wrong, because the macro is free to route around them.
        a = Mantle.devicearray(backend, zeros(Float32, 64))
        args = (a,)
        converted = map(x -> KI.argconvert(backend, x), args)
        tt = Tuple{map(Core.Typeof, converted)...}
        kernel = KI.kernel_function(backend, ki_fill_kernel, tt)
        kernel(args...; ndrange = 64)
        KI.synchronize(backend)
        @test all(Array(a) .== 1.0f0)

        # `argconvert` is a pure strip to the device-side form. It must NOT pin
        # or allocate: KI uses it only to derive the type to compile against, and
        # the pinning happens at launch, against the original arguments. What a
        # caller can see of that: a different array type of the same element type
        # and shape, a bits value a kernel takes by value, and the same value
        # however often it is asked (KI may call it more than once per launch).
        d = converted[1]
        @test d isa AbstractVector{Float32}
        @test !(d isa typeof(a))
        @test isbits(d)
        @test size(d) == size(a)
        @test KI.argconvert(backend, a) === d
        @test KI.argconvert(backend, 7) === 7
    end

    # SPIR-V's builtins are 0-based and KI's are 1-based, so every one of these
    # is a `+ 1` that has to be there exactly once.
    @testset "indexing is 1-based" begin
        n, wg = 64, 16

        got = Mantle.devicearray(backend, zeros(UInt32, n))
        KI.@launch backend ndrange = n workgroupsize = wg ki_global_id_kernel(got)
        KI.synchronize(backend)
        @test Array(got) == UInt32.(1:n)

        got = Mantle.devicearray(backend, zeros(UInt32, n))
        KI.@launch backend ndrange = n workgroupsize = wg ki_local_id_kernel(got)
        KI.synchronize(backend)
        # 1:wg, repeated once per workgroup. A 0-based leak shows up as a zero.
        @test Array(got) == UInt32.(repeat(1:wg, n ÷ wg))

        got = Mantle.devicearray(backend, zeros(UInt32, n))
        KI.@launch backend ndrange = n workgroupsize = wg ki_group_id_kernel(got)
        KI.synchronize(backend)
        @test Array(got) == UInt32.(repeat(1:(n ÷ wg), inner = wg))

        # A COUNT is a count in both numbering schemes, so this one must NOT
        # gain a one — the mirror-image mistake to the three above.
        got = Mantle.devicearray(backend, zeros(UInt32, n))
        KI.@launch backend ndrange = n workgroupsize = wg ki_num_groups_kernel(got)
        KI.synchronize(backend)
        @test all(Array(got) .== UInt32(n ÷ wg))
    end

    @testset "launch argument rules" begin
        a = Mantle.devicearray(backend, zeros(Float32, 4))
        kernel = KI.kernel_function(backend, ki_fill_kernel,
                                    Tuple{Core.Typeof(KI.argconvert(backend, a))})

        # Empty work is not an error. `cld` would turn a zero extent into one
        # workgroup, which is a dispatch that writes nothing but still runs.
        @test kernel(a; ndrange = 0) === nothing
        @test kernel(a; ndrange = (4, 0)) === nothing
        @test kernel(a; numgroups = 0) === nothing
        @test all(Array(a) .== 0.0f0)

        # One says how much work there is, the other how it is cut up.
        @test_throws ArgumentError kernel(a; ndrange = 4, numgroups = 1)

        # Over the device's limit, refused here rather than by the driver.
        limit = KI.max_work_group_size(backend)
        @test_throws ArgumentError kernel(a; numgroups = 1, workgroupsize = limit + 1)
        # A per-axis size above the device's per-axis limit, likewise.
        dims = KI.max_work_group_dims(backend)
        @test_throws ArgumentError kernel(a; numgroups = 1, workgroupsize = (1, 1, dims[3] + 1))
        # `max_work_group_size` bounds the size KI picks; it has to be positive.
        @test_throws ArgumentError kernel(a; ndrange = 4, max_work_group_size = 0)
    end

    @testset "device queries" begin
        # Answered from the device, so the assertions are on shape, plausibility
        # and agreement with `Mantle.caps` rather than on a number this machine
        # happens to report.
        @test KI.get_backend(Mantle.devicearray(backend, zeros(Float32, 1))) isa typeof(backend)
        @test KI.supports_unified(backend)
        # Mantle's own kernels use Atomix on every backend
        # (`test_atomics_and_dispatch.jl`); a backend answering `false` here is
        # one those kernels should not have run on.
        @test KI.supports_atomics(backend)
        # Stated, not inherited: before the Vulkan backend declared this, asking
        # threw `MethodError` instead of answering. And KI's answer and Mantle's
        # are two handles on one device, so they agree.
        @test KI.supports_float64(backend) == Mantle.supports_float64(backend)

        sg = KI.sub_group_size(backend)
        @test sg isa Integer && sg > 0 && ispow2(sg)
        @test sg == c.subgroup

        limit = KI.max_work_group_size(backend)
        @test limit >= 64                       # Vulkan's floor is 128
        @test limit == c.workgrouplimit
        @test KI.multiprocessor_count(backend) == c.cores

        # Per kernel, because register pressure can lower it; never above the
        # device's. (The Vulkan backend answers the device limit until the driver
        # gives a per-kernel number; Metal answers the pipeline's own.)
        a = Mantle.devicearray(backend, zeros(Float32, 1))
        kernel = KI.kernel_function(backend, ki_fill_kernel,
                                    Tuple{Core.Typeof(KI.argconvert(backend, a))})
        @test 0 < KI.max_work_group_size(kernel) <= limit
        @test KI.launch_configuration(kernel; max_work_group_size = 32).workgroupsize == 32

        # Per axis, and per launch, from the device's limits.
        dims = KI.max_work_group_dims(backend)
        @test dims isa NTuple{3, Int} && all(>(0), dims) && prod(dims) >= limit
        groups = KI.max_num_groups(backend)
        @test groups isa NTuple{3, Int} && all(>(0), groups)

        # `allocate` goes through the same path as `KA.allocate`.
        b = KI.allocate(backend, Float32, (8,))
        @test b isa AbstractVector{Float32}
        @test KI.get_backend(b) isa typeof(backend)
        @test size(b) == (8,)
    end

    @testset "a workgroup one sub-group wide is one sub-group" begin
        @test KI.supports_subgroups(backend)
        S = KI.sub_group_size(backend)
        n = 2S
        sz, lane, nsg, sgid = (Mantle.devicearray(backend, zeros(Int32, n)) for _ in 1:4)
        red = Mantle.devicearray(backend, zeros(Float32, n))
        KI.@launch backend numgroups = 2 workgroupsize = S ki_subgroup_kernel(sz, lane, nsg, sgid, red)
        KI.synchronize(backend)
        @test all(==(S), Array(sz))
        # 1-based, restarting at every sub-group. A 0-based leak is a zero.
        @test Array(lane) == Int32.(repeat(1:S, 2))
        @test all(==(1), Array(nsg))
        @test all(==(1), Array(sgid))
        # EVERY lane holds the total: a reduction and an inclusive scan differ
        # only on the other lanes, so checking lane 1 alone would pass a scan.
        @test all(==(Float32(S * (S + 1) ÷ 2)), Array(red))
    end

    @testset "shuffle and reduce type lists" begin
        # KI asks the backend which element types its shuffles cover, and the
        # answer drives KI's own suite — so the lists must agree with each other
        # and be what the backend generates, not a hopeful superset.
        # `test_subgroup_shuffle.jl` runs every listed type on the device.
        types = filter(T -> KI.supports_shuffle(backend, T),
                       [Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64,
                        Float16, Float32, Float64])
        @test Float32 in types
        @test Set(KI.shfl_types(backend)) == Set(types)
        @test Float32 in KI.sub_group_reduce_add_types(backend)
        # A Float64 shuffle or sum is a Float64 computation: listed only where the
        # device has Float64.
        if !KI.supports_float64(backend)
            @test !(Float64 in types)
            @test !(Float64 in KI.sub_group_reduce_add_types(backend))
        end
    end

    @testset "cooperative matrices" begin
        # The vocabulary, in KI rather than a package of its own.
        # `matrix_shapes` is the backend hook beside it, and it has to agree
        # with the `DeviceCaps` accessors rather than being a second opinion.
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

    @testset "barrier, and workgroup memory keyed by its id" begin
        outa = Mantle.devicearray(backend, zeros(Float32, 2KI_WG))
        outb = Mantle.devicearray(backend, zeros(Float32, 2KI_WG))
        KI.@launch backend numgroups = 2 workgroupsize = KI_WG ki_localmem_kernel(outa, outb)
        KI.synchronize(backend)
        want = Float32.(repeat(KI_WG:-1:1, 2))
        @test Array(outa) == want
        # The second buffer kept its own values: had the ids been one buffer, both
        # outputs would read whichever store landed last.
        @test Array(outb) == -want
    end

    # Every backend implements KI's device functions as OVERLAYS, in its own
    # compiler's method table. They were plain methods on the Vulkan side until
    # 2026-09-26, which made them the methods EVERY backend's compiler found
    # wherever it had no override of its own: ROCm compiled `Mantle.gemv!`'s
    # `sub_group_reduce_add` to `_lava_subgroup_reduce_add_f32`, an unknown
    # function in a GCN module, and AMDGPU had added an override of `shfl` only to
    # keep `_lava_subgroup_shuffle_f32` out. So the only methods anyone can see
    # globally are KI's own — whichever backends this process has loaded.
    #
    # One that KI gives a host method — `barrier` errors, `_print` prints with
    # `Base.print`, `localmemory` errors — would additionally have its host
    # behaviour replaced by a device intrinsic called on the CPU.
    @testset "device functions are overlays, host behaviour left intact" begin
        for f in (KI.get_global_id, KI.get_local_id, KI.get_group_id,
                  KI.get_num_groups, KI.get_local_size, KI.get_global_size,
                  KI.get_sub_group_size, KI.get_max_sub_group_size,
                  KI.get_num_sub_groups, KI.get_sub_group_id,
                  KI.get_sub_group_local_id, KI.shfl_down, KI.shfl,
                  KI.sub_group_reduce_add, KI.barrier, KI.sub_group_barrier,
                  KI.localmemory, KI._print)
            @test all(m -> m.module === KI, methods(f))
        end
        # Off-device they are still KI's: a barrier or a workgroup buffer
        # reports itself rather than executing.
        @test_throws ErrorException KI.barrier()
        @test_throws ErrorException KI.sub_group_barrier()
        @test_throws ErrorException KI.localmemory(Float32, (2, 2), 1)
    end

    # Not exercised on the device here: `get_local_size`, `get_global_size`,
    # `get_max_sub_group_size` and `sub_group_barrier`. The Vulkan backend does
    # not implement them, each for a reason that is a separate task rather than
    # an oversight, and each would be WRONG rather than merely absent if faked:
    #
    #   * `get_local_size`/`get_global_size` need `WorkgroupSize` as an
    #     `OpConstantComposite`; Lava emits an Input variable, which spirv-val
    #     rejects.
    #   * `get_max_sub_group_size` is `SubgroupMaxSize`, which SPIR-V puts under
    #     the Kernel (OpenCL) capability — illegal in a Vulkan module. It needs a
    #     specialization constant fed from the host.
    #   * `sub_group_barrier` is `OpControlBarrier Subgroup Subgroup`; Lava's
    #     only barrier is workgroup-scoped, and aliasing them would synchronise
    #     the wrong set of invocations while appearing to work.
end
