# coopmat2 FLEXIBLE DIMENSIONS: a cooperative matrix whose shape is not one the
# device reports.
#
# This device reports 15 KHR shapes and EVERY ONE has M == 16, so a 64x16 matrix
# is only legal because `flexible_dimensions` is enabled at device creation
# (`PhysicalDeviceCooperativeMatrix2FeaturesNV`). It used to be rejected before
# reaching the emitter, by a hard-coded ((16,16),(16,8),(8,8)) list in
# `KNOWN_INTRINSICS` — the GATE decided which shapes existed, not the hardware.
#
# The first assertion below is the one that matters and the easiest to fake: a
# kernel that computes nothing still "passes" a test that only checks it ran. So
# the shape is asserted to be absent from the device list FIRST (otherwise this
# tests the KHR path under a flexible-sounding name), and the result is compared
# against a CPU reference elementwise.
using Test, Lava, KernelAbstractions
const KA = KernelAbstractions
const AMg = Lava.AcceleratedMatrix

@kernel cpu = false unsafe_indices = true function flexdim_mul!(C, @Const(A), @Const(B))
    a = Lava.coopmat_load(AMg{Float16,64,16,Lava.MatrixA}, pointer(A), 1, 64)
    b = Lava.coopmat_load(AMg{Float16,16,16,Lava.MatrixB}, pointer(B), 1, 16)
    c = Lava.coopmat_muladd(a, b, Lava.coopmat_zero(AMg{Float32,64,16,Lava.Accumulator}))
    Mantle.copyto!(pointer(C), 1, 64, c)
end

@testset "coopmat2 flexible dimensions" begin
    ctx = Mantle.vk_context()
    if !ctx.coopmat_available || !ctx.coopmat2.flexible_dimensions
        @info "flexible dimensions unavailable — skipping"
    else
        # The premise. If some device DOES report a 64-row shape, this test is
        # exercising the ordinary KHR path and proves nothing about flexibility.
        @test !any(s -> s.M == 64 && s.N == 16 && s.K == 16, ctx.coopmat_shapes)
        @test !Mantle.coopmat_shape(ctx, Float16, 64, 16, 16)

        back = LavaBackend()
        # A cooperative matrix is subgroup-scoped, so the launch is exactly one
        # subgroup wide — asked rather than assumed (32 on Ada, 64 on RDNA 3.5).
        WG = Int(Mantle.device_subgroup_size(ctx))
        a = Float16.(randn(Float32, 64, 16) .* 0.1f0)
        b = Float16.(randn(Float32, 16, 16) .* 0.1f0)
        Ad = KA.allocate(back, Float16, 64, 16); copyto!(Ad, a)
        Bd = KA.allocate(back, Float16, 16, 16); copyto!(Bd, b)
        Cd = KA.allocate(back, Float32, 64, 16); fill!(Cd, 77f0)   # sentinel, not 0
        flexdim_mul!(back, (WG,))(Cd, Ad, Bd; ndrange = (WG,))
        KA.synchronize(back)
        out = Array(Cd)

        ref = Float32.(a) * Float32.(b)
        @test size(out) == (64, 16)
        # The sentinel makes "wrote nothing" distinguishable from "wrote zeros" —
        # a zeroed buffer would pass an approx test against a near-zero reference.
        @test !any(==(77f0), out)
        @test maximum(abs.(out .- ref)) / maximum(abs.(ref)) < 1e-5
    end
end
