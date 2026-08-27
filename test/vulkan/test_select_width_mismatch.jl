# Regression test for an OpSelect-of-Workgroup-pointers type-dedup bug.
#
# A clamped boundary ternary in a shared-memory kernel —
#     l = lid == 1     ? a[lid] : a[lid - 1]
#     r = lid == STILE ? a[lid] : a[lid + 1]
# is lowered by LLVM into an `OpSelect` between two Workgroup pointers (select the
# base pointer, then access [0]). The emitter used to map the bitcast's pointee
# `[N x T]` through the regular `map_type!` path while the shared global mapped it
# through `map_workgroup_type!`, minting two structurally-identical-but-distinct
# `[N x T]` type ids. The OpSelect then had an operand whose type ≠ its result
# type → SPIR-V validation error "Expected both objects to be of Result Type: Select".
#
# Fix: `map_pointer_type_for_value!` routes Workgroup pointees through
# `map_workgroup_type!` (matching the store/access-chain paths), so the bitcast
# reuses the global's deduplicated type id.

using Test
using Lava, Mantle
using KernelAbstractions
const KA = KernelAbstractions

@testset "OpSelect of Workgroup pointers (clamped ternary)" begin
    M = 128
    @kernel function _clamped_ternary!(out)
        lid = @index(Local)
        sh = @localmem Float32 (128,)
        @inbounds sh[lid] = Float32(lid)
        @synchronize()
        @inbounds begin
            l = lid == 1   ? sh[lid] : sh[lid - 1]
            r = lid == 128 ? sh[lid] : sh[lid + 1]
            out[lid] = (l + sh[lid] + r) / 3f0
        end
    end

    out = Mantle.LavaArray(zeros(Float32, M))
    _clamped_ternary!(Mantle.LavaBackend())(out; ndrange = M, workgroupsize = M)
    KA.synchronize(Mantle.LavaBackend())
    got = Array(out)

    # CPU reference: clamped 3-point average of v = 1:M.
    v = Float32.(1:M)
    ref = similar(v)
    for i in 1:M
        l = i == 1 ? v[i] : v[i-1]
        r = i == M ? v[i] : v[i+1]
        ref[i] = (l + v[i] + r) / 3f0
    end
    @test got ≈ ref
end
