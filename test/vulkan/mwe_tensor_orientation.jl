# Which tensor-layout dimension maps to which array axis?
#
# `test_tensor_load.jl` establishes that a `(64, 64)` layout over a column-major
# Julia array yields the TRANSPOSE of the leading block. That is true and it is
# not enough: with a square tensor AND a square tile, "transposed" pins the
# element mapping but not which layout dimension is which — swapping both the
# dims and the axes is indistinguishable.
#
# A GEMM has to get that right. So: a RECTANGULAR tensor and a RECTANGULAR tile,
# where every wrong combination produces either a mismatch or an out-of-range
# read rather than a coincidence.
#
#     src is (SRC_R, SRC_C) column-major, values 1:SRC_R*SRC_C
#     tile is TILE_M x TILE_N with TILE_M != TILE_N
#
# Run directly:  julia --project=. dev/Lava/test/mwe_tensor_orientation.jl
using Lava, KernelAbstractions, Printf
const KA = KernelAbstractions
const AMo = Lava.AcceleratedMatrix

const SRC_R, SRC_C = 64, 32      # column-major: SRC_R is the fast axis
const TILE_M, TILE_N = 16, 8     # deliberately not square

# Arm A: layout dims (SRC_R, SRC_C) — i.e. dim0 = the SLOW axis in memory terms
@kernel cpu = false function tload_rc!(out, @Const(src))
    l  = Lava.tensor_layout(Val(2), Val(Lava.TENSOR_CLAMP_CONSTANT))
    l2 = Lava.tensor_setdim(l, (Int32(SRC_R), Int32(SRC_C)))
    l3 = Lava.tensor_slice(l2, (Int32(0), Int32(0)), (Int32(TILE_M), Int32(TILE_N)))
    z  = Lava.coopmat_zero(AMo{Float32,TILE_M,TILE_N,Lava.Accumulator})
    m  = Lava.tensor_load(z, UInt64(pointer(src)), l3)
    Mantle.copyto!(pointer(out), 1, TILE_M, m)
end

# Arm B: layout dims (SRC_C, SRC_R) — the other assignment
@kernel cpu = false function tload_cr!(out, @Const(src))
    l  = Lava.tensor_layout(Val(2), Val(Lava.TENSOR_CLAMP_CONSTANT))
    l2 = Lava.tensor_setdim(l, (Int32(SRC_C), Int32(SRC_R)))
    l3 = Lava.tensor_slice(l2, (Int32(0), Int32(0)), (Int32(TILE_M), Int32(TILE_N)))
    z  = Lava.coopmat_zero(AMo{Float32,TILE_M,TILE_N,Lava.Accumulator})
    m  = Lava.tensor_load(z, UInt64(pointer(src)), l3)
    Mantle.copyto!(pointer(out), 1, TILE_M, m)
end

back = LavaBackend()
ctx = Mantle.vk_context()
WG = Mantle.device_subgroup_size(ctx)
src = KA.allocate(back, Float32, SRC_R, SRC_C)
copyto!(src, Float32.(reshape(1:(SRC_R * SRC_C), SRC_R, SRC_C)))
s = Array(src)

"""What the tile should be if layout dim `d1` (the fast one) walks array axis `ax`."""
function expected(ax::Symbol)
    # `copyto!` writes the matrix column-major into a TILE_M x TILE_N block, and
    # the load's row index is the matrix's row. `:rows` means the layout's fast
    # dimension follows the array's fast (column-major first) axis.
    ax === :rows ? [s[c, r] for r in 1:TILE_M, c in 1:TILE_N] :
                   [s[r, c] for r in 1:TILE_M, c in 1:TILE_N]
end

for (name, k) in (("dims=(SRC_R,SRC_C)", tload_rc!), ("dims=(SRC_C,SRC_R)", tload_cr!))
    out = KA.allocate(back, Float32, TILE_M, TILE_N); fill!(out, -1.0f0)
    k(back, (Int(WG),))(out, src; ndrange = (Int(WG),))
    KA.synchronize(back)
    o = Array(out)
    @printf("%-22s  all-written=%-5s  == s[r,c]:%-5s  == s[c,r]:%-5s  o[1,2]=%.0f o[2,1]=%.0f\n",
            name, !any(==(-1.0f0), o), o == expected(:cols), o == expected(:rows),
            o[1, 2], o[2, 1])
end
println()
println("s[1,1..3] = ", s[1, 1:3], "   s[1..3,1] = ", s[1:3, 1])
println("A GEMM must use whichever arm reproduces the operand it means; the other")
println("reads a differently-shaped region and can still look finite.")
