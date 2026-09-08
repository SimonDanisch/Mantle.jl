# Regression test for a narrow-index miscompile.
#
# ── 2026-08-02: re-audited under Rule 0, and the earlier diagnosis was wrong ──
#
# The verdict "NVIDIA's shader compiler" **stands**, and it is now established by
# the strongest instrument rather than by elimination. But every *mechanism* this
# header used to assert has been disproved by direct measurement, so read the new
# section, not the old story.
#
# What settles it: the same computation **written by hand in GLSL and compiled by
# glslangValidator 16.4.0** — not by Lava — over the same argument buffer, run
# through the same Lava dispatch, produces the **identical** wrong answer on the
# RTX 4000 Ada, and is exact on lavapipe. That is one file of Khronos-produced
# SPIR-V failing on one driver and passing on another; our emitter is not in the
# loop at all. (`tools/` + the GLSL sources are described in
# `dev/JuliaVision/plans/projects/lava-core/REPORT.md`.)
#
#   producer  \  consumer      NVIDIA RTX 4000 Ada 595.336   lavapipe / LLVM 22.1.8
#   Lava (this kernel)                    WRONG                       exact
#   glslangValidator 16.4.0               WRONG                       exact
#
# The old two-driver argument had a hole — nobody had checked that lavapipe ran
# the *same bytes*, and Lava's module generation is device-dependent. With a
# hand-written GLSL module it is literally the same file on both, so the hole is
# closed.
#
# ── The wrong value, located exactly ──
#
# Reading back intermediates from inside the failing kernel (four values per
# invocation, one dispatch, so no probe can differ from another by dead-code
# elimination):
#
#   q1 = z ÷ e1          read directly:  CORRECT
#   q2 = q1 ÷ e2         read directly:  CORRECT
#   c2 = (q1 + 1) - q2*e2                WRONG — equals (q2 + 1) - q2*e2
#   the same expression in Int32, same dispatch:  CORRECT
#
# So `q1` is right when read on its own, and delivers **`q2`'s value** where it is
# used again *after* the second division. For `(5,5,3)` that makes c2 = 1-4(k-1)
# = 1, -3, -7 instead of j, which is exactly the observed delivered index
# `(c1, c3, 1)` and the observed flat offset `(c1-1) + d1*(c3-1)`.
#
# ── Disproved, each by an experiment, not an argument ──
#
# All of these were previously asserted in this file. Each was tested by swapping
# the *emitted module* under an otherwise identical dispatch and re-running:
#
#   * **not the Int32 narrowing.** Turning `shl 32 / ashr 32` into `shl 0 /
#     ashr 0` — the wide computation, same instruction count — is still wrong.
#     "It is `CartesianIndices` under a narrow index" was never the mechanism;
#     `cis[I]` escapes only because Julia emits a *different* decomposition for
#     it (one `udiv` by `d1*d2` plus a `udiv` of the remainder — not a chained
#     pair), not because it is 64-bit.
#   * **not `OpSDiv` vs `OpUDiv`.** Rewriting every division is still wrong.
#   * **not the divide-by-zero control flow.** Deleting both guards for a single
#     straight-line block is still wrong; so is keeping them and deleting the
#     redundant re-tests in the merge blocks.
#   * **not the struct layout and not our argument packing.** A GLSL memory-dump
#     shader reads the `Broadcasted` at the same offsets the module uses and gets
#     dims (5,5,3), keeps 0x010101, defaults (1,1,1), axes (5,5,3), extents
#     (5,5,3), ndrange 75 — all correct.
#   * **not `Extruded`'s selects and not the address arithmetic.** `c1`, `c2`,
#     `c3`, `dims[1]`, `dims[2]` each read back correct; rewriting the address as
#     three explicit multiplies changes nothing.
#   * **`spirv-val --target-env vulkan1.3` passes.** GPU-assisted validation
#     (`Mantle.enable_gpu_av`) reports nothing on the failing dispatch — but
#     `MVE.verify_gpu_av()` fails on this layer, so GPU-AV cannot see BDA
#     accesses here and its silence is **not** evidence.
#
# ── What does NOT reproduce it, which is why it looked like `CartesianIndices` ──
#
# The chained-division shape alone is not sufficient. A Julia kernel that does
# only `q1 = z ÷ e1; q2 = q1 ÷ e2; out[I] = (q1+1) - q2*e2` with runtime divisors
# is **correct** at Int64 and Int32, and so is the same thing hand-written in
# GLSL — even with `q1` additionally stored. The fault needs the surrounding
# register pressure, which is why it tracked rank ≥ 3 and `Extruded` and why
# every attempt to instrument it made it disappear. Treat it as a scheduling /
# register-allocation fault around chained 64-bit division, not as a rule about
# any one instruction.
#
# ── Consequence for kernels ──
#
# Any kernel doing two *dependent* 64-bit integer divisions and re-using the
# first quotient afterwards is exposed. `MVE.splitidx` / `cart32` (magic-number
# division, no `OpSDiv`) is the standing workaround and is why Lava's broadcast
# path is safe; narrowing the arithmetic to `Int32` also avoids it. Both cases
# below were `@test_broken` waiting on a driver fix, and on 2026-09-04 they
# announced it: promoted to `@test`.
#
# Symptom when it regresses in the other direction: `Mantle.lava_broadcast_flat_*`
# in `array/gpuarrays.jl` may go back to `cis[I % Int32]`.
#
# ── Why it is worth fixing rather than avoiding ──
#
# Narrowing this arithmetic is a real speedup — the same narrowing inside
# `DNNKernels.im2col_kernel!` (four integer divisions per element) moved a whole
# inference step from 35.7 ms to 32.7. Lava's broadcast kernels cannot take it
# until this is fixed, and any other kernel reaching for a 32-bit index is
# exposed to the same silent wrong answer.

using Test, Lava, KernelAbstractions
using .MVE: FastDiv32, cart32
const KA = KernelAbstractions

# Four decompositions of the same linear index into the same `Broadcasted`. The
# point of running all four is to separate the three things that could be at
# fault — the 32-bit width, the division, and `CartesianIndices` — since only
# `:narrowci` has all three.
@kernel function bcast_index!(dest, bc, cis, n, fd, szi, ::Val{MODE}) where {MODE}
    I = @index(Global, Linear)
    if I <= n
        J = MODE === :wide     ? (@inbounds cis[I]) :                        # 64-bit, Base
            MODE === :narrowci ? (@inbounds cis[I % Int32]) :                # 32-bit, Base
            MODE === :handdiv  ? CartesianIndex(cart32(Int32(I) - Int32(1), szi)) :
                                 CartesianIndex(cart32(UInt32(I) - UInt32(1), fd))
        @inbounds dest[I] = bc[J]
    end
end

@testset "Int32 linear index into a Broadcasted" begin
    be = LavaBackend()
    host = reshape(collect(1f0:105f0), 7, 5, 3)
    A = KA.allocate(be, Float32, 7, 5, 3); copyto!(A, host)

    for (name, src, sz, want) in (
            ("view",     view(A, 2:6, :, :),          (5, 5, 3), host[2:6, :, :] .* 3),
            ("permuted", PermutedDimsArray(A, (2, 1, 3)), (5, 7, 3),
             permutedims(host, (2, 1, 3)) .* 3))
        n = prod(sz)
        cis = CartesianIndices(sz)
        fd = map(FastDiv32, sz)
        szi = map(Int32, sz)
        # `preprocess` is what `_copyto!` does, and it is load-bearing here: it
        # wraps the operands in `Extruded`, and without it one of these two
        # shapes computes the right answer even with the narrow index.
        dummy = KA.allocate(be, Float32, sz...)
        bc = Broadcast.preprocess(dummy,
                Broadcast.instantiate(Broadcast.broadcasted(*, src, 3f0)))
        run(mode) = begin
            out = KA.allocate(be, Float32, n); fill!(out, 0f0)
            bcast_index!(be, 64)(out, bc, cis, n, fd, szi, Val(mode); ndrange = n)
            KA.synchronize(be)
            reshape(Array(out), sz)
        end
        @test run(:wide) ≈ want              # 64-bit through Base: correct
        # Base's `CartesianIndices` under a narrow index: was wrong, and the
        # `@test_broken` announced the fix on 2026-09-04 (RADV, this tree).
        @test run(:narrowci) ≈ want
        # The two that isolate it. `:handdiv` is 32-bit AND divides and is exact,
        # which rules out both the width and the division; `:magic` is the form
        # Lava's broadcast kernels actually use.
        @test run(:handdiv) ≈ want
        @test run(:magic) ≈ want
    end
end

@testset "the boundary: rank 2 is exact, rank 3 preprocessed is not" begin
    # The smallest form of the bug — no view, no permutation, no arithmetic —
    # and the rank below it, which must stay correct. Pinning both means a change
    # in *either* direction shows up: a fix (rank 3 starts passing) or a spread
    # (rank 2 starts failing).
    be = LavaBackend()
    narrow(src, hsrc, sz) = begin
        n = prod(sz)
        dummy = KA.allocate(be, Float32, sz...)
        bc = Broadcast.preprocess(dummy, Broadcast.instantiate(
                 Broadcast.broadcasted(identity, src)))
        out = KA.allocate(be, Float32, n); fill!(out, 0f0)
        bcast_index!(be, 64)(out, bc, CartesianIndices(sz), n,
                             map(FastDiv32, sz), map(Int32, sz), Val(:narrowci);
                             ndrange = n)
        KA.synchronize(be)
        reshape(Array(out), sz) ≈ hsrc
    end
    h2 = reshape(collect(1f0:25f0), 5, 5)
    A2 = KA.allocate(be, Float32, 5, 5); copyto!(A2, h2)
    @test narrow(A2, h2, (5, 5))                 # rank 2: exact

    h3 = reshape(collect(1f0:75f0), 5, 5, 3)
    A3 = KA.allocate(be, Float32, 5, 5, 3); copyto!(A3, h3)
    @test narrow(A3, h3, (5, 5, 3))              # rank 3 + Extruded: was the bug,
                                                 # fixed — the `@test_broken` announced it
end

