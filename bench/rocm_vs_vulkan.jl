# The same Mantle graph on two backends: Lava/Vulkan and AMDGPU/ROCm.
#
#     julia --project=. bench/rocm_vs_vulkan.jl
#
# One device, two drivers, so what differs is the runtime and not the hardware:
# a Radeon 8060S (gfx1151, Strix Halo, 20 WGPs, 32 MB Infinity Cache, 112 GiB
# unified) reached through RADV + Lava's SPIR-V on one side and through HIP +
# AMDGPU.jl on the other. `ext/MantleROCmExt.jl` is the backend under test.
#
# What each section is FOR, since three different claims get measured here:
#
#   `chain`     the graph's memory planning, and the cost of a dispatch.
#   `gemms`     ROCm's libraries against Lava's own kernels, at SAM 2.1's
#               shapes, which is the reason to want a ROCm backend at all.
#   `mixed`     a plan whose passes are BOTH — rocBLAS beside KernelAbstractions
#               kernels, captured into one HIP graph.
#
# ── Measured 2026-09-14 ───────────────────────────────────────────────────────
#
# Two clean-process runs of this script. Where they disagreed the range is
# given: this part clocks between 600 MHz and 2900 MHz and a figure quoted to
# three digits from one run would be quoting the clock.
#
# *Memory planning is worth more than a byte count.* The 8-link chain at 16 MiB
# a link has a 128 MiB footprint unaliased and 32 MiB aliased, and 32 MiB is
# what fits in this part's Infinity Cache:
#
#                    footprint          time           effective
#     ROCm  alias      32 MiB    0.520..0.573 ms    468..516 GB/s
#     ROCm  no alias   128 MiB   1.098..1.100 ms         245 GB/s
#     Lava  alias      32 MiB    0.492..0.582 ms    461..546 GB/s
#     Lava  no alias   128 MiB          1.112 ms         241 GB/s
#
# 2.1x on both backends, from the same core analysis, on a device whose measured
# DRAM rate for this kernel is 279 GB/s. The aliasing is not saving memory here
# so much as buying cache residency.
#
# *A dispatch costs more on HIP than on a recorded command buffer.* 64 passes of
# a 4096-element kernel, p50 of 50 runs:
#
#                            latency    throughput    per dispatch    allocated
#     ROCm, walked           0.160 ms    0.141 ms      2.20 us       40,976 B
#     ROCm, hipGraph         0.150 ms    0.124 ms      1.93 us            0 B
#     Lava, recorded         0.102 ms    0.062 ms      0.97 us            0 B
#
# **The graph is worth 12% of the host cost and all of the allocation.** The
# walk cost 6.22 us a dispatch while this backend still let KernelAbstractions
# re-derive the iteration space and look the kernel up per launch, and capturing
# it then looked like a 3.2x win. Compiling in `compile_dispatch` — where
# Mantle's architecture puts it, and what `src/metal/record.jl` does — took the
# walk to 2.2 us. What the 4 us were was host work, not submission.
#
# The allocation column is the part that does not shrink with tuning: a walk
# builds a kernel object and a launch configuration per dispatch, ~640 bytes
# apiece, and a replay is one `hipGraphLaunch` that touches no host memory. On
# SAM 2.1's encoder that is 10.5 MB a call against zero — and a replay that
# allocates is a replay a garbage collection can interrupt while the previous
# launch is still in flight.
#
# Lava's recorded command buffer is still 2x cheaper per dispatch. The chain
# above says that is submission cost and nothing else: on work that is
# memory-bound rather than launch-bound the two backends are level. SAM 2.1's
# encoder is the extreme case — captured and walked both replay in 322 ms,
# because 1,082 launches hide entirely behind the work.
#
# *ROCm's GEMM is twice Lava's, at the same numerics.* SAM 2.1 large's encoder
# is 195 `addmm`s, 1606 GFLOP, fp16 in with fp32 accumulation; the shapes below
# are read out of the exported graph. Summed over all 195:
#
#     Lava  coopmat_gemm!          94.8..98.1 ms    16.4..16.9 TFLOP/s
#     rocBLAS rocblas_gemm_ex      41.9..43.1 ms    37.3..38.3 TFLOP/s
#
# Same accuracy — 0.00391 and 0.00390 max error against an fp32 reference, on a
# scale of 11.2. `rocblas_hgemm` is faster still (46.6 TFLOP/s) and is NOT the
# comparison: it accumulates in fp16 and its error is 22x worse (0.0877).
#
# *Half of that win is the epilogue Lava fuses and rocBLAS cannot.* Lava's GEMM
# adds the bias and applies the activation inside its store; `gemm_ex` has no
# epilogue, so those 2554 MiB of outputs have to be read and written again:
#
#     read+write floor (no arithmetic)  18.9..22.6 ms   237..283 GB/s
#     bias + gelu, linear index               28.6 ms       188 GB/s
#     bias + gelu, 2-D index                  38.2 ms       140 GB/s
#
# So ~96 ms against 61..72 ms — 1.4..1.6x, not 2.3x. Getting the whole 2.3x
# needs a fused epilogue, which is hipBLASLt: `libhipblaslt.so.1` is on this
# machine and AMDGPU.jl has no bindings for it.
#
# *A library call and a kernel can live in one plan and one captured graph.*
# Four passes — rocBLAS, KA epilogue, rocBLAS, KA epilogue — recorded once and
# replayed: correct to 4.5e-05 against an fp32 reference, and a second input
# uploaded into the same buffer comes back right through the SAME recording
# (4.1e-05, against 0.0503 for the first input's answer), which is what says the
# graph holds the address and not the value.

using Mantle, KernelAbstractions, LinearAlgebra
import AMDGPU
using KernelAbstractions: @kernel, @index, @Const
const M = Mantle
const RB = AMDGPU.rocBLAS

# The backend under test. An extension once Mantle declares the weak dependency;
# until then this is how it is loaded.
isdefined(Main, :MantleROCmExt) ||
    include(joinpath(@__DIR__, "..", "ext", "MantleROCmExt.jl"))
const RE = Main.MantleROCmExt
RE.__init__()

const rdev = M.Device(M.ROCmAPI())
const vdev = M.Device(M.VulkanAPI())

"""
Two numbers, because a per-run wait and a frame loop are different questions.

`latency` waits after every run, which is what an interactive caller pays.
`throughput` submits `iters` runs and waits once, which is what a pipelined loop
gets. On this backend they differ by more than they should — see the note on
`blocking` in `waitidle`.
"""
function bench(f, dev; warm = 20, iters = 50)
    for _ in 1:warm; f(); end
    M.waitidle(dev)
    ts = Float64[]
    for _ in 1:iters
        t0 = time_ns(); f(); M.waitidle(dev); push!(ts, (time_ns() - t0) / 1e6)
    end
    sort!(ts)
    t0 = time_ns(); for _ in 1:iters; f(); end; M.waitidle(dev)
    return (latency = ts[(end + 1) ÷ 2], throughput = (time_ns() - t0) / 1e6 / iters)
end

timeit(f, dev; iters = 10) = (f(); M.waitidle(dev);
    begin t0 = time_ns(); for _ in 1:iters; f(); end; M.waitidle(dev)
          (time_ns() - t0) / 1e6 / iters end)

# ── the chain: memory planning and dispatch cost ─────────────────────────────

@kernel function tinyadd!(dst, @Const(src))
    i = @index(Global)
    @inbounds dst[i] = src[i] + 1.0f0
end

"""
`links` passes, each reading what the previous wrote.

A forced chain, so the schedule is a line and only two transients are ever live
together: what `Place` and `Aliasing` do with that is the thing being measured,
and the result is checkable — the last transient holds `links` exactly when
every pass ran in order on the bytes the placer gave it.
"""
function chainplan(dev, n::Integer, links::Integer; alias::Bool = true, record::Bool = true)
    g = M.Graph(dev)
    src = M.Buffer(dev, fill(0.0f0, n))
    t = [M.Transient.Buffer(g, Float32, n) for _ in 1:links]
    prev = src
    for (k, dst) in enumerate(t)
        M.dispatch!(g, tinyadd!, (dst,
                                      prev), n; name = "l$k")
        prev = dst
    end
    pl = M.Plan(g; alias)
    record && M.recordsplans(dev) && M.record!(pl)
    return g, t, pl
end

function chain(n = 1 << 22, links = 8)
    gb = links * n * 4 * 2 / 1e9
    println("\n── chain: $links passes over $((n * 4) >> 20) MiB ──")
    for (label, dev) in (("ROCm", rdev), ("Lava", vdev)), alias in (true, false)
        _, t, pl = chainplan(dev, n, links; alias)
        M.run!(pl)
        ok = all(==(Float32(links)), Array(M.storage(t[end])))
        r = bench(() -> M.run!(pl), dev; iters = 30)
        println("  ", rpad(label, 5), " alias=", rpad(alias, 6),
                " footprint ", lpad(M.peakbytes(pl) >> 20, 4), " MiB  ",
                lpad(round(r.throughput; digits = 3), 7), " ms  ",
                lpad(round(gb / (r.throughput / 1e3); digits = 1), 6), " GB/s  correct=", ok)
    end
end

"""
What a dispatch costs, walked and baked.

The walked ROCm row does not go through `run!`, because this backend answers
`recordsplans` with `true` and `run!` then refuses a recordable plan that was
never recorded — deliberately, since handing one back unrecorded is what made a
replay of SAM 2.1's encoder silently stale. So the row spells the walk out in the
verbs core would have used for it: open a run, put the plan's passes in it, close
it. That is `execute!`'s interpreted branch, and it is what the Metal and host
backends do on every frame.
"""
function dispatchcost(n = 4096, links = 64)
    println("\n── dispatch cost: $links passes over $n elements ──")
    rows = Tuple{String,Any,Any,Any}[]
    _, tw, pw = chainplan(rdev, n, links; record = false)
    walk = function ()
        e = M.openrun(rdev, pw)
        M.emitplan!(e, pw)
        M.closerun!(rdev, pw, e)
        return nothing
    end
    push!(rows, ("ROCm walked", rdev, walk, tw))
    _, tg, pg = chainplan(rdev, n, links)
    push!(rows, ("ROCm hipGraph", rdev, () -> M.run!(pg), tg))
    _, tv, pv = chainplan(vdev, n, links)
    push!(rows, ("Lava recorded", vdev, () -> M.run!(pv), tv))
    for (label, dev, f, t) in rows
        f(); M.waitidle(dev)
        ok = all(==(Float32(links)), Array(M.storage(t[end])))
        r = bench(f, dev)
        println("  ", rpad(label, 15), " latency ", lpad(round(r.latency; digits = 3), 6),
                " ms  throughput ", lpad(round(r.throughput; digits = 3), 6), " ms  ",
                lpad(round(r.throughput / links * 1000; digits = 2), 5), " us/dispatch  correct=", ok)
    end
end

# ── the GEMMs: ROCm's library against Lava's kernels ─────────────────────────

# Every `addmm` shape in SAM 2.1 large's encoder and how many times it occurs,
# as `(count, torch M, K, N)`, read out of `sam2_encoder.json`. DNNKernels runs
# them in the reversed layout, so the Julia call is M = torch N (out features),
# N = torch M (tokens), K = K; the four biggest are 72.7% of the arithmetic.
const SAM2_GEMMS = [
    (36, 4096, 576, 2304), (36, 4096, 2304, 576), (35, 4096, 576, 1728),
    (36, 4096, 576, 576), (6, 16384, 288, 1152), (6, 16384, 1152, 288),
    (4, 1024, 1152, 4608), (4, 1024, 4608, 1152), (5, 16384, 288, 864),
    (3, 1024, 1152, 3456), (2, 65536, 144, 576), (2, 65536, 576, 144),
    (2, 65536, 144, 432), (1, 65536, 144, 864), (6, 16384, 288, 288),
    (1, 16384, 288, 1728), (1, 4096, 576, 3456), (4, 1024, 1152, 1152),
    (2, 65536, 144, 144), (1, 65536, 144, 288), (1, 16384, 288, 576),
    (1, 4096, 576, 1152)]

"""
`C = A*B`, fp16 in and **fp32 accumulation**, through `rocblas_gemm_ex`.

Not `LinearAlgebra.mul!`, which does not reach rocBLAS for this element type:
the dispatcher in AMDGPU's `blas/highlevel.jl` gates on `ROCBLASFloat`, which is
`Union{Float32, Float64, ComplexF32, ComplexF64}` — `Float16` falls through to
`GPUArrays.generic_matmatmul!` and runs at 0.67 TFLOP/s against 42.7 here, a
factor of 64, silently. (`ROCBLASFloatWithHalf` exists in the same file and
`rocblas_hgemm` is wrapped, so the intent was there.)

Not `rocblas_hgemm` either, which is what that dispatcher would have reached:
HGEMM accumulates in fp16 and is 16x less accurate (0.0626 max error against
0.0039), so it is not the same computation as Lava's coopmat GEMM and cannot be
compared with it. `gemm_ex` with `compute_type = f32_r` is.
"""
function gemm_ex!(c, a, b, m::Int, n::Int, k::Int)
    (; handle) = RB.lib_state()
    alpha = Ref(1.0f0); beta = Ref(0.0f0)
    RB.rocblas_gemm_ex_64(
        handle, RB.rocblas_operation_none, RB.rocblas_operation_none,
        Int64(m), Int64(n), Int64(k), alpha,
        a, RB.rocblas_datatype_f16_r, Int64(max(1, stride(a, 2))),
        b, RB.rocblas_datatype_f16_r, Int64(max(1, stride(b, 2))), beta,
        c, RB.rocblas_datatype_f16_r, Int64(max(1, stride(c, 2))),
        c, RB.rocblas_datatype_f16_r, Int64(max(1, stride(c, 2))),
        RB.rocblas_datatype_f32_r, RB.rocblas_gemm_algo_standard, Int32(0), UInt32(0))
    return c
end

"""Three matrices of a GEMM, over pool regions on `dev`, as the backend's own
arrays — which is the whole point of `deviceview`: rocBLAS reads a Mantle region
with no copy and no knowledge that it is one."""
function gemmarrays(dev, m, n, k)
    A = M.Buffer(dev, zeros(Float16, m * k))
    B = M.Buffer(dev, zeros(Float16, k * n))
    C = M.Buffer(dev, zeros(Float16, m * n))
    return (reshape(M.storage(A), m, k), reshape(M.storage(B), k, n),
            reshape(M.storage(C), m, n))
end

function gemms()
    println("\n── SAM 2.1 encoder GEMMs: rocBLAS against Lava's cooperative matrices ──")
    tl = tr = 0.0
    gflop = 0.0
    for (cnt, mt, kt, nt) in SAM2_GEMMS
        m, n, k = nt, mt, kt
        a, b, c = gemmarrays(vdev, m, n, k)
        l = timeit(() -> Mantle.coopmat_gemm!(c, a, b, m, n, k), vdev)
        ar, br, cr = gemmarrays(rdev, m, n, k)
        RE.adopt!(rdev)
        r = timeit(() -> gemm_ex!(cr, ar, br, m, n, k), rdev)
        tl += cnt * l; tr += cnt * r; gflop += cnt * 2 * m * n * k / 1e9
        println("  ", lpad(cnt, 3), " x  M=", lpad(m, 5), " N=", lpad(n, 5), " K=", lpad(k, 4),
                "   Lava ", lpad(round(l; digits = 3), 7), " ms   rocBLAS ",
                lpad(round(r; digits = 3), 7), " ms   ", round(l / r; digits = 2), "x")
        GC.gc(false)
    end
    println("  TOTAL ", round(gflop; digits = 1), " GFLOP:  Lava ", round(tl; digits = 1),
            " ms (", round(gflop / (tl / 1e3) / 1000; digits = 1), " TFLOP/s)   rocBLAS ",
            round(tr; digits = 1), " ms (", round(gflop / (tr / 1e3) / 1000; digits = 1), " TFLOP/s)")
end

"""
Both engines against an fp32 reference, because a speed comparison between two
GEMMs that do not compute the same thing is not a comparison. This is what says
`gemm_ex` with fp32 accumulation is Lava's equal and `rocblas_hgemm` is not.
"""
function gemmaccuracy(m = 2304, n = 4096, k = 576)
    println("\n── accuracy: both engines against an fp32 reference ──")
    ah = rand(Float16, m, k) .- Float16(0.5)
    bh = rand(Float16, k, n) .- Float16(0.5)
    ref = Float32.(ah) * Float32.(bh)
    A = M.Buffer(vdev, vec(ah)); B = M.Buffer(vdev, vec(bh)); C = M.Buffer(vdev, zeros(Float16, m * n))
    a = reshape(M.storage(A), m, k); b = reshape(M.storage(B), k, n); c = reshape(M.storage(C), m, n)
    Mantle.coopmat_gemm!(c, a, b, m, n, k); M.waitidle(vdev)
    Ar = M.Buffer(rdev, vec(ah)); Br = M.Buffer(rdev, vec(bh)); Cr = M.Buffer(rdev, zeros(Float16, m * n))
    ar = reshape(M.storage(Ar), m, k); br = reshape(M.storage(Br), k, n); cr = reshape(M.storage(Cr), m, n)
    RE.adopt!(rdev); gemm_ex!(cr, ar, br, m, n, k)
    Cr2 = M.Buffer(rdev, zeros(Float16, m * n)); cr2 = reshape(M.storage(Cr2), m, n)
    RB.gemm!('N', 'N', one(Float16), ar, br, zero(Float16), cr2)
    M.waitidle(rdev)
    err(x) = maximum(abs.(Float32.(Array(x)) .- ref))
    println("  scale ", round(maximum(abs.(ref)); digits = 3),
            "   Lava coopmat ", err(c), "   rocBLAS gemm_ex ", err(cr),
            "   rocBLAS hgemm ", err(cr2))
end

# ── the epilogue, and then a plan that mixes both ────────────────────────────

"""
What the epilogue rocBLAS cannot fuse would cost.

Lava's `coopmat_gemm!` starts its accumulators from the bias and applies the
activation as it converts to fp16 in registers, so bias and gelu are free.
`rocblas_gemm_ex` has no epilogue, so the same work is a second pass over every
output: 2554 MiB of them across the encoder's 195 GEMMs.

`floor` is the same pass with no arithmetic in it — read, add one, write — which
bounds what any epilogue kernel can achieve here and says how much of the
`biasgelu` number is the kernel's own fault rather than the memory system's.
"""
function epilogue()
    println("\n── the epilogue rocBLAS has to pay separately ──")
    bytes = 0
    tepi = tfloor = 0.0
    for (cnt, mt, kt, nt) in SAM2_GEMMS
        m, n = nt, mt
        C = M.Buffer(rdev, zeros(Float16, m * n))
        B = M.Buffer(rdev, zeros(Float16, m))
        c = M.storage(C); b = M.storage(B)
        k1 = biasgelu!(AMDGPU.ROCBackend()); k2 = addone!(AMDGPU.ROCBackend())
        RE.adopt!(rdev)
        tepi += cnt * timeit(() -> k1(c, b, Int32(m); ndrange = length(c)), rdev)
        tfloor += cnt * timeit(() -> k2(c; ndrange = length(c)), rdev)
        bytes += cnt * m * n * 2
        GC.gc(false)
    end
    rate(t) = round(2 * bytes / (t / 1e3) / 1e9; digits = 1)
    println("  outputs: ", round(bytes / 2^20; digits = 0), " MiB")
    println("  bias + gelu        ", lpad(round(tepi; digits = 1), 6), " ms   ", rate(tepi), " GB/s")
    println("  read+write floor   ", lpad(round(tfloor; digits = 1), 6), " ms   ", rate(tfloor), " GB/s")
end

@kernel function biasgelu!(c, @Const(bias), rows::Int32)
    i = @index(Global, Linear)
    @inbounds begin
        x = Float32(c[i]) + Float32(bias[(i - Int32(1)) % rows + Int32(1)])
        c[i] = Float16(0.5f0 * x * (1f0 + tanh(0.7978845608f0 * (x + 0.044715f0 * x^3))))
    end
end

"""The same traffic with no arithmetic: what `biasgelu!` is measured against."""
@kernel function addone!(c)
    i = @index(Global, Linear)
    @inbounds c[i] = c[i] + Float16(1)
end

"""
A library call dressed as a KernelAbstractions kernel, so a pass can declare it.

Three lines, because `bake` asks a dispatch's kernel for two things only —
`kernelfor` builds it for a backend and the `Launch` calls it with an `ndrange`
— and a callable can answer both. What that buys is the thing `custom!` was
deleted for being unable to do safely: a pass that says what it READS and WRITES
in the graph's own vocabulary, and whose body is rocBLAS.

It is sound on this backend for a reason that did not hold on the one `custom!`
was removed from. A recorded pass there had to be handed the queue, because the
body wrote commands; here the body submits to a stream that is being CAPTURED,
so the graph gets the node without the body knowing anything about recording.
"""
struct LibOp{F}
    f::F
end
(op::LibOp)(backend) = op
(op::LibOp)(args...; ndrange = nothing) = (op.f(args...); nothing)
Mantle.callgroup(::LibOp, group) = nothing

gemmop(m, n, k) = (c, a, b) ->
    gemm_ex!(reshape(c, m, n), reshape(a, m, k), reshape(b, k, n), m, n, k)

gelu(x) = 0.5f0 .* x .* (1 .+ tanh.(0.7978845608f0 .* (x .+ 0.044715f0 .* x .^ 3)))

"""
`gelu(W2 * gelu(W1*X .+ b1) .+ b2)` as a four-pass Mantle graph: two rocBLAS
GEMMs and two KernelAbstractions epilogues, planned and aliased by core and
captured whole into one HIP graph.

Checked two ways. Against an fp32 reference, and then against a SECOND input
uploaded into the same buffer and run through the SAME recording — a captured
graph has to hold the address and not the value, and this is what says it does.
"""
function mixed(m1 = 1152, n = 1024, k = 1152, m2 = 1152)
    println("\n── mixed: rocBLAS and KernelAbstractions in one captured graph ──")
    w1 = (rand(Float16, m1, k) .- Float16(0.5)) .* Float16(0.1)
    b1 = (rand(Float16, m1) .- Float16(0.5)) .* Float16(0.1)
    x  = (rand(Float16, k, n) .- Float16(0.5)) .* Float16(0.1)
    w2 = (rand(Float16, m2, m1) .- Float16(0.5)) .* Float16(0.1)
    b2 = (rand(Float16, m2) .- Float16(0.5)) .* Float16(0.1)
    reference(xx) = gelu(Float32.(w2) *
        Float32.(Float16.(gelu(Float32.(w1) * Float32.(xx) .+ Float32.(b1)))) .+ Float32.(b2))

    g = M.Graph(rdev)
    W1 = M.Buffer(rdev, vec(w1)); B1 = M.Buffer(rdev, vec(b1)); X = M.Buffer(rdev, vec(x))
    W2 = M.Buffer(rdev, vec(w2)); B2 = M.Buffer(rdev, vec(b2))
    h = M.Transient.Buffer(g, Float16, m1 * n)
    o = M.Transient.Buffer(g, Float16, m2 * n)
    M.dispatch!(g, LibOp(gemmop(m1, n, k)),
                    (h, W1,
                     X), 1; name = "gemm1")
    M.dispatch!(g, biasgelu!, (h,
                                   B1, Int32(m1)), m1 * n; name = "epi1")
    M.dispatch!(g, LibOp(gemmop(m2, n, m1)),
                    (o, W2,
                     h), 1; name = "gemm2")
    M.dispatch!(g, biasgelu!, (o,
                                   B2, Int32(m2)), m2 * n; name = "epi2")
    pl = M.Plan(g)
    M.record!(pl)
    M.run!(pl); M.waitidle(rdev)
    got() = Float32.(reshape(Array(M.storage(o)), m2, n))
    println("  first input:  max err ", maximum(abs.(got() .- reference(x))))
    x2 = (rand(Float16, k, n) .- Float16(0.5)) .* Float16(0.1)
    M.upload!(rdev, X.store, 1, vec(x2)); M.waitidle(rdev)
    M.run!(pl); M.waitidle(rdev)
    println("  second input: max err ", maximum(abs.(got() .- reference(x2))),
            "   (against the first input's reference: ",
            round(maximum(abs.(got() .- reference(x))); digits = 4), ")")
    r = bench(() -> M.run!(pl), rdev)
    gflop = (2 * m1 * n * k + 2 * m2 * n * m1) / 1e9
    println("  per run: latency ", round(r.latency; digits = 3), " ms  throughput ",
            round(r.throughput; digits = 3), " ms  (",
            round(gflop / (r.throughput / 1e3) / 1000; digits = 1), " TFLOP/s over the two GEMMs)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("ROCm  : ", M.devicename(rdev))
    println("Vulkan: ", M.devicename(vdev))
    chain()
    dispatchcost()
    gemmaccuracy()
    gemms()
    epilogue()
    mixed()
end
