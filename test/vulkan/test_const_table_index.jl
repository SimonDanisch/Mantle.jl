# A constant lookup table indexed by a runtime value.
#
# Julia writes `kern[i]` on a constant `SVector` as a pointer one element BEFORE
# the table plus the index — the 1-based fold — and when the table is hoisted to
# a module-scope constant global, LLVM emits the "before" pointer as a CONSTANT
# EXPRESSION:
#
#   %p = getelementptr float,
#          ptr addrspace(1) getelementptr ([5 x float], ptr addrspace(1) @_j_const_1,
#                                          i64 -1, i64 4),
#          i64 %i
#
# The emitter read the address space off that expression to pick a storage class.
# Address space 1 is Julia's for both device pointers and constant globals, so
# the table got `PhysicalStorageBuffer` while the instruction GEP built on top of
# it — which asks `get_pointer_storage_class` and therefore knows better — got
# `Private`. One pointer chain, two storage classes:
#
#   The result pointer storage class and base pointer storage class in
#   OpAccessChain do not match.
#
# It takes the whole shape below to reproduce: a 5x5 neighbourhood with clamped
# coordinates and TWO indexed reads of the table per tap. With less, LLVM either
# folds the table away or materialises it inside the function, and the constant
# expression is never emitted — a first attempt at this test "passed" for that
# reason and pinned nothing.
#
# Found via Hikari's a-trous denoiser, whose 5x5 B-spline weights are exactly
# this and which had therefore never compiled on any driver.

@testset "a constant table indexed at runtime" begin

@inline function tap_weight(i::Int32)
    kern = SVector{5,Float32}(0.0625f0, 0.25f0, 0.375f0, 0.25f0, 0.0625f0)
    return kern[i]
end

@kernel inbounds = true function const_table_filter!(
        dst, @Const(src), @Const(w::Int32), @Const(h::Int32), @Const(step::Int32))
    idx = @index(Global)
    if idx <= w * h
        row = ((idx - Int32(1)) % h) + Int32(1)
        col = ((idx - Int32(1)) ÷ h) + Int32(1)
        acc = 0f0
        wsum = 0f0
        for dy_i in Int32(1):Int32(5)
            for dx_i in Int32(1):Int32(5)
                q_row = clamp(row + (dy_i - Int32(3)) * step, Int32(1), h)
                q_col = clamp(col + (dx_i - Int32(3)) * step, Int32(1), w)
                weight = tap_weight(dx_i) * tap_weight(dy_i)
                acc += src[q_row, q_col] * weight
                wsum += weight
            end
        end
        dst[row, col] = acc / wsum
    end
end

function reference_filter(src::Matrix{Float32}, step::Int)
    h, w = size(src)
    kern = (0.0625f0, 0.25f0, 0.375f0, 0.25f0, 0.0625f0)
    dst = similar(src)
    for col in 1:w, row in 1:h
        acc = 0f0; wsum = 0f0
        for dy_i in 1:5, dx_i in 1:5
            q_row = clamp(row + (dy_i - 3) * step, 1, h)
            q_col = clamp(col + (dx_i - 3) * step, 1, w)
            weight = kern[dx_i] * kern[dy_i]
            acc += src[q_row, q_col] * weight
            wsum += weight
        end
        dst[row, col] = acc / wsum
    end
    return dst
end

backend = LavaBackend()
h, w = 32, 24
src = Float32[sin(0.3f0i) * cos(0.2f0j) + 0.5f0 for i in 1:h, j in 1:w]

for step in (1, 2, 4)
    d_src = LavaArray(src)
    d_dst = KernelAbstractions.allocate(backend, Float32, (h, w))
    KernelAbstractions.fill!(d_dst, 0f0)
    const_table_filter!(backend)(d_dst, d_src, Int32(w), Int32(h), Int32(step);
                                 ndrange = w * h)
    KernelAbstractions.synchronize(backend)
    got = Array(d_dst)
    want = reference_filter(src, step)
    # The weights the GPU read have to be the table's, at the index asked for —
    # a storage class that validated but pointed elsewhere would show up here as
    # well as in `spirv-val`.
    @test maximum(abs.(got .- want)) < 1f-5
end

end
