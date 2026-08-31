# Dynamic-index double-indirection through MVectors.
#
# Pattern: load an NTuple from an MVector{N, NTuple{K,Int32}} via a runtime index,
# then use that NTuple's component as a second runtime index into a different
# MVector. This is the core of epa()'s polytope walk:
#
#     v    = face_v[best_idx]   # NTuple{3, Int32}
#     vert = verts_m[v[1]]      # Vec3f
#
# It used to miscompile: the SPIR-V emit path constant-folded too aggressively and
# wrote a fixed wrong byte pattern (Vec3f(0,0,0)) instead of the indexed value.
# Discovered bringing up narrow_phase_kernel. The emit fixes that landed since
# resolved it, but nothing guarded it: the original file was an MWE gated with
# `@test_skip`, was never registered in runtests.jl, and had bit-rotted to the
# point of not compiling (`ntuple(f, 8)` without a Val length heap-allocates, and
# its CPU half scalar-indexed a LavaArray on the host, which is disallowed).
#
# Every index is checked, not just the one that happened to be reported, since a
# constant-folding bug can be right for some indices and wrong for others.

using Test, Lava, KernelAbstractions, GeometryBasics, StaticArrays
using .MVE: LavaArray, LavaBackend
using GeometryBasics: Vec3f

@inline function double_indirect_body(idxs::AbstractVector, i::Integer)
    face_v = MVector{8, NTuple{3, Int32}}(ntuple(j -> (Int32(j), Int32(j+1), Int32(j+2)), Val(8)))
    verts  = MVector{16, Vec3f}(ntuple(j -> Vec3f(Float32(j*10), 0f0, 0f0), Val(16)))
    @inbounds dyn_idx = idxs[i]
    @inbounds v       = face_v[dyn_idx]
    @inbounds vert    = verts[v[1]]
    return vert
end

@kernel function double_indirect_kernel(out, @Const(idxs))
    i = @index(Global)
    @inbounds out[i] = double_indirect_body(idxs, i)
end

# Host reference, computed with plain tuples.
function double_indirect_ref(k)
    face_v = ntuple(j -> (Int32(j), Int32(j+1), Int32(j+2)), Val(8))
    verts  = ntuple(j -> Vec3f(Float32(j*10), 0f0, 0f0), Val(16))
    return verts[face_v[k][1]]
end

@testset "double-indirect MVector access" begin
    # One index per launch: each is a separate runtime value reaching the same
    # chained access, which is what the folding bug keyed on.
    for k in 1:8
        idxs = LavaArray(Int32[k])
        out  = LavaArray([Vec3f(0, 0, 0)])
        double_indirect_kernel(LavaBackend())(out, idxs; ndrange = 1)
        MVE.vk_flush!(MVE.vk_context().default_bq)
        @test Array(out)[1] == double_indirect_ref(k)
    end

    # ...and all of them in ONE launch, so the index is genuinely per-lane rather
    # than uniform across the dispatch.
    idxs = LavaArray(Int32.(collect(1:8)))
    out  = LavaArray(fill(Vec3f(0, 0, 0), 8))
    double_indirect_kernel(LavaBackend())(out, idxs; ndrange = 8)
    MVE.vk_flush!(MVE.vk_context().default_bq)
    @test Array(out) == [double_indirect_ref(k) for k in 1:8]
end

