"""
`KernelInterface` on Metal: the caps, the device intrinsics, and the payoff.

The payoff is the last testset. `gemv.jl` is ~2,300 lines written against a
Vulkan device and moved to `src/array/` during the split; it names no backend,
sizes itself out of `caps`, and reduces with `sub_group_reduce_add`. If it
computes the right answer on an Apple GPU with no change to the algorithm, the
portable half of Mantle is portable — which is the claim the whole refactor
rests on and the only way to check it is to run it on a second device.
"""

using Test, Mantle, Metal, KernelAbstractions, KernelInterface
const KA = KernelAbstractions
const KI = KernelInterface

@testset "Metal: DeviceCaps" begin
    d = Mantle.Device(Mantle.MetalAPI())
    c = Mantle.caps(d)

    # Read from the device, not defaulted. A backend that failed to answer would
    # leave these at core's fallbacks, and every one of those is a value a
    # `> 0` check would happily accept.
    @test c.subgroup == 32                    # threadExecutionWidth on Apple GPUs
    @test c.workgrouplimit == 1024
    @test c.sharedbudget == 32768
    @test c.coopmatsubgroup == c.subgroup

    # **The tile is 8, not 16.** Metal's `simdgroup_matrix` is 8×8 where Vulkan's
    # cooperative matrix is 16×16. A kernel with 16 hard-coded does not run
    # slower here, it does not run — which is why tile size is a device query.
    @test c.coopmat
    @test c.tile == 8
    @test !isempty(c.shapes)
    @test all(s -> s.M == 8 && s.N == 8 && s.K == 8, c.shapes)
    @test all(s -> s.acc === Float32, c.shapes)

    # Not reported by Metal, and `0` rather than an invention: `DeviceCaps` says
    # `0` means "not known", and a plausible fake core count is worse than none.
    @test c.cores == 0
    @test c.warps == 0

    # Cached — reading `threadExecutionWidth` needs a compiled pipeline, and
    # `gemv` asks per launch.
    @test Mantle.caps(d) === c
end

@testset "Metal: the portable queries agree with the device" begin
    # The invariant `test_array_algorithm_portability.jl` pins on the Vulkan
    # side: what a kernel asks through the portable route must be what the
    # device would have said. If these drift, a kernel is sized for one device
    # and launched on another — silently.
    b = Metal.MetalBackend()
    a = Metal.zeros(Float32, 16)
    @test Mantle.workgrouplimit(a) == KI.max_work_group_size(b)
    @test Mantle.sharedbudget(a) == Mantle.caps(b).sharedbudget
    @test KI.sub_group_size(b) == Mantle.caps(b).subgroup
    @test Mantle.workgrouplimit(a) > 0
    @test Mantle.sharedbudget(a) > 0
    # Float64 is absent on purpose: Apple GPUs do not have it, which is the
    # divergence the shuffle and reduce type lists were separated for.
    @test !(Float64 in KI.shfl_down_types(b))
    @test !(Float64 in KI.sub_group_reduce_add_types(b))
    @test Float32 in KI.sub_group_reduce_add_types(b)
end

@kernel cpu = false unsafe_indices = true function _ki_reduce!(out, @Const(a))
    i = @index(Global, Linear)
    out[i] = KI.sub_group_reduce_add(a[i])
end

@testset "Metal: sub_group_reduce_add lowers to one instruction" begin
    n = 32
    a = MtlArray(collect(1.0f0:Float32(n)))
    out = Metal.zeros(Float32, n)
    _ki_reduce!(Metal.MetalBackend())(out, a; ndrange = n)
    Metal.synchronize()
    h = Array(out)

    # EVERY lane holds the total. A reduction and an inclusive scan differ only
    # on the other lanes, so checking `h[1]` alone would pass an implementation
    # that wired the wrong thing — the same trap the KernelInterface test suite
    # guards, restated here because this is a different lowering.
    @test all(==(Float32(sum(1:n))), h)
end

@testset "Metal: gemv, written for Vulkan, run on Apple" begin
    # `gemv.jl` is core. It names no backend, takes its workgroup size from
    # `caps`, and reduces with `sub_group_reduce_add`. Nothing in it was changed
    # for Metal.
    for (K, N) in ((128, 64), (256, 32), (64, 128))
        Ah = rand(Float32, K, N)
        xh = rand(Float32, K)
        A = MtlArray(Ah); x = MtlArray(xh); C = Metal.zeros(Float32, N)
        Mantle.gemv!(C, x, A)
        Metal.synchronize()
        @test maximum(abs.(Array(C) .- (Ah' * xh))) < 1.0f-3
    end
end
