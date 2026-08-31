"""
A shared-memory store whose index goes through a real `OpUDiv` loses stores.

This is the bug that made a 96 x 128 GEMM tiling silently drop 4 of every 32
k-terms per row, and it took a long way round because almost everything about it
points somewhere else. The shader is valid, `spirv-val` passes, the barrier is
emitted in the right block with the right semantics, the accesses are tagged
`NonPrivatePointer`, the driver reports no register spill, and the staged block
still comes out wrong.

**One variable.** Staging a 96-row block through a 104-wide shared array, with a
cooperative-matrix `muladd` in scope, counting how many of 3072 elements survive:

                        K = 32   K = 64  K = 128  K = 256
      while + OpUDiv      3072      240      256      240
      while + fastdiv     3072     3072     3072     3072
      for   + OpUDiv      3072      240      256      240
      for   + fastdiv     3072     3072     3072     3072

The loop form does not matter. The division does. `i, kk = (idx % 96, idx ÷ 96)`
emits `OpUDiv`; `splitidx(idx, Val(96))` emits a high multiply and a shift, and
the two modules are otherwise **identical opcode for opcode** — that diff, on
`BM = 112` (lossy) against `BM = 128` (exact, because 128 folds to a mask and a
shift), is what finally identified it.

Three further conditions are each necessary, so a smaller repro will not show it:
the stored value must come from a **global load** (storing a computed constant is
exact at every geometry), the enclosing loop must run **more than one iteration**
(everything is exact at K = 32), and a cooperative-matrix `muladd` must be in
scope (deleting it makes every geometry exact). Total workgroup memory is not a
variable — 4 KB to 32 KB are all exact when the index folds to shifts.

What is lost, with the two k-blocks given distinguishable values: of 3072 slots,
0 hold the previous block's value, ~2200 hold a value from the right block but
the wrong row, and ~390 are never written. Nothing is sunk past the barrier;
stores are dropped. And it cannot be instrumented — recording the store and load
indices to global shows both correct and injective **and makes the corruption
disappear** — which is why this is a black-box table rather than a diagnosis.

Whether the dropped stores come from our SPIR-V or the driver's compilation of it
is still open: our module contains a *rolled* loop with a single `OpStore`, so
whatever unrolls it and drops stores is downstream of us. `splitidx` removes the
`OpUDiv` and with it the whole question, which is why `gemm.jl` uses it for every
staging index rather than only where a divisor is awkward.

## 2026-08-02: three candidate mechanisms measured and DISPROVED

The condition above says "a cooperative-matrix `muladd` must be in scope", which
invites three explanations. All three were tested by building the same kernel
without a coopmat and reading the driver's own register count
(`VK_KHR_pipeline_executable_properties`), and all three are wrong.

    variant                                     survivors/3072   registers
    coopmat, muladd in the k-loop                    240             64
    no coopmat, 0 live fp32 accumulators            3072             26
    no coopmat, 32 live fp32 accumulators           3072             60
    no coopmat, 48 live fp32 accumulators           3072             76
    coopmat present, muladd OUTSIDE the loop        3072             37
    coopmat, muladd in loop, acc NOT live
      across the barrier                             240             64

1. **Not register pressure.** The no-coopmat kernel at 48 accumulators uses
   **76 registers — more than the coopmat kernel's 64 — and loses nothing.**
   Worth stating because the sibling bug next door
   (`test_int32_cartesian_miscompile.jl`) *is* register pressure, so the obvious
   guess here is that they share a mechanism. They do not.

2. **Not the capability, and not the fragments.** Declaring
   `CooperativeMatrixKHR`, constructing `MatrixA`/`MatrixB`/`Accumulator` and
   doing a `muladd` — all present — is exact when the `muladd` sits *after* the
   k-loop instead of inside it.

3. **Not a cooperative-matrix value living across the barrier.** Producing and
   consuming the accumulator inside one iteration, so nothing coopmat-shaped
   crosses `@synchronize`, still loses exactly as much.

What survives all three: **an `OpCooperativeMatrixMulAddKHR` executed in the same
loop iteration as the divided-index shared store.** Tighter than "in scope".

## 2026-08-02, later: SETTLED — the module is valid and NVIDIA miscompiles it

This section previously said the bug "cannot be settled on this machine", because
"lavapipe reports `coopmat available: false` (subgroup size 8)". **That was wrong,
and it closed the cheapest route for no reason.** lavapipe reports
`coopmat_available = true` with four 8x8x8 shapes, `Float16` among them, at
subgroup scope. It is a second, independent cooperative-matrix consumer, and it
was sitting on this box the whole time.

Running this file's kernel on both — same `BM = 96`, `LDA = 104`, `BK = 32`,
`WG = 256`, same global load addressed by the divided values, same `fa`/`fb` from
a global pointer outside the loop — with only the cooperative matrix's extent
taken from the device:

    device      form       K=32    K=64   K=128   K=256
    NVIDIA      udiv       3072     256     240     240    DROPS
    NVIDIA      fastdiv    3072    3072    3072    3072    exact
    lavapipe    udiv       3072    3072    3072    3072    exact
    lavapipe    fastdiv    3072    3072    3072    3072    exact

**The pattern is compiled correctly by an independent stack.** Our SPIR-V is
therefore runnable as written, and what loses the stores is NVIDIA's compilation
of it — the same conclusion the sibling bug in
`test_int32_cartesian_miscompile.jl` reached the same way, by varying the
consumer.

Two honest limits on that. The modules are **not byte-identical**: lavapipe has
no 16x16x16, so its tile is 8x8, and this varies the extent along with the
consumer. And llvmpipe is a software rasteriser whose compilation strategy shares
nothing with a GPU driver's, so "it does not perform the transform that breaks"
is a weaker statement than "the transform is illegal". What it does establish is
that the four conditions are not intrinsically unsafe, which is what the
`@test_broken` above is waiting on.

The other two routes remain, and both would strengthen it:

  * a **third consumer**: the AMD laptop's Radeon 8060S (RDNA 3.5, RADV) reports
    14 shapes including this kernel's exact `Float16 x Float16 -> Float32`, so it
    can run the module byte-identically where lavapipe cannot;
  * **glslang as an independent producer** — write this kernel in GLSL, compile
    it with glslang, and run that module on the same NVIDIA driver. Correct there
    means our SPIR-V differs from glslang's in a way that matters.

A note on the earlier attempt, because it cost an hour: a reworded reproducer —
operands loaded from shared inside the loop, global load addressed linearly
rather than by the divided values — came out **exact on NVIDIA**, i.e. carried no
bug at all. The conditions in the prose above are necessary, not sufficient. Copy
the kernel.

**Audit of every `@localmem` kernel in Lava and DNNKernels**, since a bug that
needs four coincidences is one nobody finds by testing the obvious thing. The
question per kernel is whether a *shared-store index* goes through a
non-power-of-two `%` or `÷` — a division feeding a **global** address is fine:

    kernel                        localmem  coopmat  divided store index
    gemm.jl staged                  yes       yes    was yes -> splitidx
    flash.jl attn_flash!            yes       no     was yes, E=72 -> splitidx
    conv_implicit.jl                yes       no     no: BS_CRS/BS_NPQ are 16/32/128/256,
                                                     all powers of two. Its real
                                                     divisions (by KW*KH = 9) address
                                                     `w`, not shared.
    conv_coopmat.jl                 no        yes    n/a — stages nothing
    attention.jl toLE_tiled_*!      yes       no     no: `tile[tx, ty+4j]`, both from
                                                     `@index(Local, NTuple)`
    attention.jl coopmat kernels    no        yes    n/a
    layernorm.jl, ops.jl            yes       no     no: `red[t+1]`, `sh[lt]`

Two were affected and both are fixed. `flash.jl`'s was the one that had been
sitting as a documented "unexplained" blocker.
"""

using Test, Lava, KernelAbstractions
using Lava: AcceleratedMatrix, MatrixA, MatrixB, Accumulator
using .MVE: splitidx
const KA = KernelAbstractions

const SID_BM, SID_LDA, SID_BK, SID_WG = 96, 104, 32, 256

# `TS` is the cooperative matrix's extent, and 8 exists so this runs on lavapipe
# — a second, independent consumer, which is what settled where the dropped
# stores come from. Nothing else differs between the two.
for (name, fast) in (("udiv", false), ("fastdiv", true)), TS in (16, 8)
    kn = Symbol("sid_", name, "_", TS, "!")
    split = fast ? :(splitidx(idx, Val($SID_BM))) :
                   :((idx % $SID_BM, idx ÷ $SID_BM))
    @eval @kernel cpu=false unsafe_indices=true function $kn(C, dump, @Const(A), @Const(G),
                                                             ::Val{M}, ::Val{K}) where {M,K}
        sh = @localmem Float16 ($SID_LDA * $SID_BK,)
        tid = @index(Local, Linear) - 1
        acc = zero(AcceleratedMatrix{Float32,$TS,$TS,Accumulator})
        fa = AcceleratedMatrix{Float16,$TS,$TS,MatrixA}(pointer(G), 1, $TS)
        fb = AcceleratedMatrix{Float16,$TS,$TS,MatrixB}(pointer(G), 1, $TS)
        for kb in 0:(K ÷ $SID_BK - 1)
            k0 = kb * $SID_BK
            @inbounds for r in 0:(($SID_BM * $SID_BK) ÷ $SID_WG - 1)
                idx = tid + r * $SID_WG
                i, kk = $split
                sh[1 + i + kk * $SID_LDA] = A[1 + i + (k0 + kk) * M]
            end
            @synchronize
            acc = muladd(fa, fb, acc)
            @synchronize
        end
        # Linear read-back: no divided index on this side, so the check cannot be
        # confounded by the very arithmetic under test.
        @inbounds for r in 0:(($SID_LDA * $SID_BK) ÷ $SID_WG)
            j = tid + r * $SID_WG
            j < $SID_LDA * $SID_BK && (dump[1 + j] = sh[1 + j])
        end
        copyto!(pointer(C), 1, $TS, convert(AcceleratedMatrix{Float16,$TS,$TS,Accumulator}, acc))
    end
end

"Elements of the staged block that survived, out of `SID_BM * SID_BK`."
function sid_survivors(backend, kern, K; TS::Int = 16)
    hA = Float16.(reshape(0:(SID_BM * K - 1), SID_BM, K) .% 2048)
    A = KA.allocate(backend, Float16, SID_BM, K); copyto!(A, hA)
    G = KA.allocate(backend, Float16, 256); fill!(G, Float16(0.5))
    C = KA.allocate(backend, Float16, TS, TS); fill!(C, Float16(-1))
    dump = KA.allocate(backend, Float16, SID_LDA * SID_BK); fill!(dump, Float16(-7))
    kern(backend, SID_WG)(C, dump, A, G, Val(SID_BM), Val(K); ndrange = SID_WG)
    KA.synchronize(backend)
    # every k-block restages the same slots, so the last one wins
    want = hA[:, (K - SID_BK + 1):K]
    count(reshape(Array(dump), SID_LDA, SID_BK)[1:SID_BM, :] .== want)
end

@testset "shared stores through a divided index" begin
    backend = LavaBackend()
    if !MVE.coopmat_gemm_available()
        @info "skipping: no cooperative-matrix support on this device"
    else
        total = SID_BM * SID_BK
        @testset "splitidx keeps every store, at every trip count" begin
            for K in (32, 64, 128, 256)
                @test sid_survivors(backend, sid_fastdiv_16!, K) == total
            end
        end

        # The plain `%` / `÷` form is what `splitidx` exists to avoid. Asserted as
        # broken rather than deleted: if a driver update fixes it this turns into
        # a failure, which is the signal that `splitidx` could be relaxed.
        @testset "the plain division form still drops stores" begin
            @test sid_survivors(backend, sid_udiv_16!, 32) == total   # one trip is fine
            for K in (64, 128, 256)
                @test_broken sid_survivors(backend, sid_udiv_16!, K) == total
            end
        end
    end
end

# The second consumer, and the testset that settled where the dropped stores come
# from. It needs no second card: the Vulkan loader enumerates lavapipe alongside
# the real GPU on every machine here, and lavapipe DOES have cooperative matrices
# — four 8x8x8 shapes — which is the fact this file previously got wrong.
@testset "the same pattern on a second cooperative-matrix consumer" begin
    lp = try
        MVE.VkContext(select = devs -> only(filter(MVE.islavapipe, devs)))
    catch e
        @info "no lavapipe device; skipping the second-consumer check" exception=e
        nothing
    end
    if lp === nothing
    elseif !MVE.coopmat_shape(lp, Float16, 8, 8, 8)
        @info "lavapipe has no 8x8x8 Float16 cooperative matrix here" lp.device_name
        MVE.mark_device_lost!(lp)
    else
        try
            back = LavaBackend(lp)
            total = SID_BM * SID_BK
            # The control first: if `splitidx` were lossy here the comparison
            # below would mean nothing.
            for K in (32, 64, 128, 256)
                @test sid_survivors(back, sid_fastdiv_8!, K; TS = 8) == total
            end
            # And the form that drops on NVIDIA at every trip count above one.
            # Exact here — which is the whole result. If this ever starts
            # failing, a second stack has begun losing stores too and the bug
            # moves back to being ours.
            for K in (32, 64, 128, 256)
                @test sid_survivors(back, sid_udiv_8!, K; TS = 8) == total
            end
        finally
            # Nothing else can retire a context built with `VkContext(; select)`,
            # and its buffers' finalizers would otherwise run against a torn-down
            # device at exit.
            MVE.mark_device_lost!(lp)
        end
    end
end

@testset "splitidx agrees with ÷ and % on the host" begin
    for N in (1, 2, 3, 7, 16, 31, 32, 48, 96, 104, 112, 128, 160, 255, 256, 1000)
        for idx in vcat(0:300, [1023, 1024, 4095, 4096, 65535, 65536, 1_000_000])
            @test splitidx(idx, Val(N)) == (idx % N, idx ÷ N)
        end
    end
end

