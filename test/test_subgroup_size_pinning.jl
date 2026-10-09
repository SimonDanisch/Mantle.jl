# A kernel runs at the sub-group width the device reports for it, and the kernel
# observes that width.
#
# There are two numbers, because there are two kinds of kernel.
# `Mantle.caps(b).subgroup` is the width a plain kernel runs at, and
# `coopmatsubgroup` is the width a kernel holding a cooperative matrix gets. They
# differ where the backend pins cooperative-matrix kernels: the Vulkan backend's
# cooperative-matrix kernels are written against a 32-lane subgroup, and Lava
# pins 32 when the module declares CooperativeMatrixKHR and the device's default
# is wider (`src/vulkan/runtime/pipeline.jl`).
#
# This is the precondition for making plan objects carry a subgroup width. On an
# RTX 4000 Ada it says little: the card reports 32 and can only be 32, so "the
# width the kernel sees" and "the device default" are the same number and nothing
# distinguishes a correct pin from no pin at all. RDNA 3.5 reports a DEFAULT of 64
# with size control min=32 max=64, so there a plain kernel runs at 64 and a
# cooperative-matrix kernel at 32, and the two answers are distinguishable.
#
# What is asserted, for each kind of kernel:
#   1. the width the kernel OBSERVES equals the width the device reports for it;
#   2. a width-INDEPENDENT result is identical at both widths (the pin does not
#      change what the kernel computes);
#   3. a width-DEPENDENT result matches the observed width exactly — a subgroup
#      reduction over 64 lanes sums 64 values at width 64 and 32 at width 32, so
#      this fails if the pin is silently ignored while the size query reports
#      the requested number.
#
# (3) is the one that matters. Without it a driver could report the requested
# width and still schedule the other one, which is precisely the failure a plan
# object storing an unpinned width would produce.

using Test, KernelAbstractions
import KernelInterface as KI
using KernelInterface: AcceleratedMatrix, Accumulator, coopmat_getcomp
include(joinpath(@__DIR__, "testbackend.jl"))
const KA = KernelAbstractions

const WG_SP = 64          # one wave64, or two wave32 subgroups

# What each lane sees, written out per lane. The same body for both kernels, so
# the comparison cannot be between two different programs.
@inline function sgp_observe!(sz, lane, red, indep, x, i)
    r = KI.sub_group_reduce_add(x[i])                  # width-DEPENDENT
    @inbounds begin
        sz[i]    = KI.get_sub_group_size(Int32)
        lane[i]  = KI.get_sub_group_local_id(Int32) - Int32(1)
        red[i]   = r
        indep[i] = x[i] * 2.0f0                        # width-INDEPENDENT
    end
    return nothing
end

@kernel cpu=false unsafe_indices=true function sgp_plain!(sz, lane, red, indep, @Const(x))
    i = KI.get_local_id().x
    sgp_observe!(sz, lane, red, indep, x, i)
end

# `cm` exists only to make the kernel hold a cooperative matrix, which is what the
# pin keys on. Its value is written out so it cannot be optimised away.
@kernel cpu=false unsafe_indices=true function sgp_coop!(sz, lane, red, indep, cm, @Const(x),
                                                         ::Val{T}) where {T}
    i = KI.get_local_id().x
    sgp_observe!(sz, lane, red, indep, x, i)
    m = AcceleratedMatrix{Float32,T,T,Accumulator}(pointer(cm), 1, T)
    @inbounds cm[i] = coopmat_getcomp(m, Int32(0))
end

"Run one workgroup of `WG_SP` lanes through `launch` and read back what each saw."
function sgp_run(launch, be)
    x = KA.allocate(be, Float32, WG_SP); copyto!(x, Float32.(1:WG_SP))
    sz    = KA.zeros(be, Int32, WG_SP)
    lane  = KA.zeros(be, Int32, WG_SP)
    red   = KA.zeros(be, Float32, WG_SP)
    indep = KA.zeros(be, Float32, WG_SP)
    launch(sz, lane, red, indep, x)
    KA.synchronize(be)
    return (sz = Int.(Array(sz)), lane = Int.(Array(lane)),
            red = Array(red), indep = Array(indep))
end

"The three per-width assertions: observed width, lane numbering, reduction extent."
function checkwidth(r, w)
    # 1. the kernel observes the width the device reports, on every lane
    @test all(==(w), r.sz)
    # lane ids restart at every subgroup boundary
    @test r.lane == [mod(i - 1, w) for i in 1:WG_SP]
    # 3. the width-dependent reduction really used that many lanes
    want = [sum(Float32, (fld(i - 1, w) * w + 1):(fld(i - 1, w) * w + w)) for i in 1:WG_SP]
    @test r.red == want
end

@testset "a kernel runs at the sub-group width reported for it" begin
    be = TESTBACKEND
    c = Mantle.caps(be)
    @info "sub-group widths" plain = c.subgroup coopmat = c.coopmatsubgroup

    plain = sgp_run(be) do sz, lane, red, indep, x
        sgp_plain!(be, WG_SP)(sz, lane, red, indep, x; ndrange = WG_SP)
    end
    @testset "a plain kernel" begin
        checkwidth(plain, c.subgroup)
    end

    if !c.coopmat
        @info "no cooperative matrices here; the cooperative-matrix width is not exercised"
    else
        coop = sgp_run(be) do sz, lane, red, indep, x
            # Large enough for the tile the kernel loads, not only for its lanes.
            cm = KA.zeros(be, Float32, max(WG_SP, c.tile^2))
            sgp_coop!(be, WG_SP)(sz, lane, red, indep, cm, x, Val(c.tile); ndrange = WG_SP)
        end
        @testset "a cooperative-matrix kernel" begin
            checkwidth(coop, c.coopmatsubgroup)
        end

        # 2. the pin changes the width, not the arithmetic
        @test coop.indep == plain.indep
        if c.coopmatsubgroup != c.subgroup
            @test coop.sz != plain.sz        # the two runs really differed
            # And the reductions must NOT agree, which is what proves the
            # width-dependent assertion above had teeth.
            @test coop.red != plain.red
        end
    end
end
