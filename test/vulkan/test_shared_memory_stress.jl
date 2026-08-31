# Stress tests for workgroup ("local"/shared) memory: `@localmem` + `@synchronize`.
#
# Shared memory has historically been the fragile corner of the SPIR-V emitter
# (struct AccessChain, 8-bit/Bool layout, barrier-skipping error paths). The
# existing tests cover struct *shapes* (alignment) and a single reduction. This
# file stresses *numerical correctness* of real shared-memory algorithms across:
#   - element types (Float32/64, Int32/64, Bool, GeometryBasics.Vec3f, structs)
#   - many workgroup sizes (incl. non-power-of-two and partial last block)
#   - heavy barrier traffic (tree reduction, Hillis-Steele scan, iterative stencil)
#   - cross-lane patterns (shared-memory transpose)
#   - multiple @localmem buffers in one kernel (double-buffering)
# Every case is checked against a CPU reference.

using Test
using Lava, Mantle
using KernelAbstractions
using GeometryBasics: Vec3f, Vec
const KA = KernelAbstractions

const SMBACKEND = MVE.LavaBackend()

# ── A. Tree reduction (sum) — multiple barriers, types × workgroup sizes ──────
# One block per `TILE` elements; log2(TILE) barrier-synchronised halving steps.
for TILE in (32, 64, 128, 256), T in (Float32, Float64, Int32, Int64)
    fname = Symbol("_smr_reduce_", T, "_", TILE, "!")
    @eval @kernel function $fname(input, output)
        gid = @index(Group)
        lid = @index(Local)
        sh = @localmem $T ($TILE,)
        @inbounds sh[lid] = input[(gid - 1) * $TILE + lid]
        @synchronize()
        s = $TILE ÷ 2
        while s >= 1
            if lid <= s
                @inbounds sh[lid] = sh[lid] + sh[lid + s]
            end
            @synchronize()
            s ÷= 2
        end
        if lid == 1
            @inbounds output[gid] = sh[1]
        end
    end
end

@testset "tree reduction (sum) × type × workgroup size" begin
    nblocks = 7
    for TILE in (32, 64, 128, 256), T in (Float32, Float64, Int32, Int64)
        n = TILE * nblocks
        cpu = T <: Integer ? rand(T(1):T(9), n) : rand(T, n)
        g = MVE.LavaArray(cpu)
        out = MVE.LavaArray(zeros(T, nblocks))
        k = getfield(@__MODULE__, Symbol("_smr_reduce_", T, "_", TILE, "!"))(SMBACKEND)
        k(g, out; ndrange = n, workgroupsize = TILE)
        KA.synchronize(SMBACKEND)
        ref = [sum(@view cpu[(b-1)*TILE+1 : b*TILE]) for b in 1:nblocks]
        got = Array(out)
        if T <: Integer
            @test got == ref
        else
            @test got ≈ ref rtol=(T === Float32 ? 1f-3 : 1e-9)
        end
    end
end

# ── B. Hillis-Steele inclusive scan (prefix sum) — double-buffered shared mem ──
# Two @localmem buffers ping-pong; read-all/barrier/write-all/barrier per step.
for TILE in (64, 128), T in (Float32, Int32)
    fname = Symbol("_smr_scan_", T, "_", TILE, "!")
    @eval @kernel function $fname(input, output)
        gid = @index(Group)
        lid = @index(Local)
        a = @localmem $T ($TILE,)
        b = @localmem $T ($TILE,)
        @inbounds a[lid] = input[(gid - 1) * $TILE + lid]
        @synchronize()
        offset = 1
        while offset < $TILE
            if lid > offset
                @inbounds b[lid] = a[lid] + a[lid - offset]
            else
                @inbounds b[lid] = a[lid]
            end
            @synchronize()
            @inbounds a[lid] = b[lid]
            @synchronize()
            offset *= 2
        end
        @inbounds output[(gid - 1) * $TILE + lid] = a[lid]
    end
end

@testset "Hillis-Steele inclusive scan × type × workgroup size" begin
    nblocks = 5
    for TILE in (64, 128), T in (Float32, Int32)
        n = TILE * nblocks
        cpu = T <: Integer ? rand(T(0):T(4), n) : rand(T, n)
        g = MVE.LavaArray(cpu)
        out = MVE.LavaArray(zeros(T, n))
        k = getfield(@__MODULE__, Symbol("_smr_scan_", T, "_", TILE, "!"))(SMBACKEND)
        k(g, out; ndrange = n, workgroupsize = TILE)
        KA.synchronize(SMBACKEND)
        ref = similar(cpu)
        for blk in 1:nblocks
            rng = (blk-1)*TILE+1 : blk*TILE
            ref[rng] = cumsum(cpu[rng])
        end
        got = Array(out)
        if T <: Integer
            @test got == ref
        else
            @test got ≈ ref rtol=1f-3
        end
    end
end

# ── C. Shared-memory tile transpose (cross-lane) ──────────────────────────────
const TT = 16  # 16x16 tile, 256 threads — exercises 2-D @localmem multi-index
@kernel function _smr_transpose!(input, output)
    gI = @index(Group, Cartesian)
    lI = @index(Local, Cartesian)
    tile = @localmem Float32 (TT, TT)
    lx = lI[1]; ly = lI[2]
    gx = (gI[1] - 1) * TT + lx
    gy = (gI[2] - 1) * TT + ly
    @inbounds tile[lx, ly] = input[gy, gx]       # coalesced read, into shared
    @synchronize()
    # write transposed: swap block coords, read tile with swapped local coords
    ox = (gI[2] - 1) * TT + lx
    oy = (gI[1] - 1) * TT + ly
    @inbounds output[oy, ox] = tile[ly, lx]
end

@testset "shared-memory tile transpose" begin
    m = TT * 3; n = TT * 4
    cpu = rand(Float32, m, n)               # m rows, n cols
    g = MVE.LavaArray(cpu)
    out = MVE.LavaArray(zeros(Float32, n, m))
    _smr_transpose!(SMBACKEND)(g, out; ndrange = (n, m), workgroupsize = (TT, TT))
    KA.synchronize(SMBACKEND)
    @test Array(out) ≈ permutedims(cpu)
end

# ── D. Vec3f in shared memory (vector element type) ───────────────────────────
const VTILE = 64
@kernel function _smr_vec3_reduce!(input, output)
    gid = @index(Group)
    lid = @index(Local)
    sh = @localmem Vec3f (VTILE,)
    @inbounds sh[lid] = input[(gid - 1) * VTILE + lid]
    @synchronize()
    s = VTILE ÷ 2
    while s >= 1
        if lid <= s
            @inbounds sh[lid] = sh[lid] + sh[lid + s]
        end
        @synchronize()
        s ÷= 2
    end
    if lid == 1
        @inbounds output[gid] = sh[1]
    end
end

@testset "Vec3f reduction in shared memory" begin
    nblocks = 4
    n = VTILE * nblocks
    cpu = [Vec3f(rand(), rand(), rand()) for _ in 1:n]
    g = MVE.LavaArray(cpu)
    out = MVE.LavaArray(fill(Vec3f(0), nblocks))
    _smr_vec3_reduce!(SMBACKEND)(g, out; ndrange = n, workgroupsize = VTILE)
    KA.synchronize(SMBACKEND)
    ref = [reduce(+, cpu[(b-1)*VTILE+1 : b*VTILE]) for b in 1:nblocks]
    got = Array(out)
    @test all(isapprox(got[b], ref[b]; rtol=1f-3) for b in 1:nblocks)
end

# ── E. Bool / 8-bit shared memory (WorkgroupMemoryExplicitLayout8BitAccess) ───
const BTILE = 128
@kernel function _smr_bool_count!(input, output)
    gid = @index(Group)
    lid = @index(Local)
    flags = @localmem Bool (BTILE,)
    counts = @localmem Int32 (BTILE,)
    @inbounds flags[lid] = input[(gid - 1) * BTILE + lid] > 0.5f0
    @synchronize()
    @inbounds counts[lid] = flags[lid] ? Int32(1) : Int32(0)
    @synchronize()
    s = BTILE ÷ 2
    while s >= 1
        if lid <= s
            @inbounds counts[lid] = counts[lid] + counts[lid + s]
        end
        @synchronize()
        s ÷= 2
    end
    if lid == 1
        @inbounds output[gid] = counts[1]
    end
end

@testset "Bool (8-bit) shared memory — count predicate per block" begin
    nblocks = 6
    n = BTILE * nblocks
    cpu = rand(Float32, n)
    g = MVE.LavaArray(cpu)
    out = MVE.LavaArray(zeros(Int32, nblocks))
    _smr_bool_count!(SMBACKEND)(g, out; ndrange = n, workgroupsize = BTILE)
    KA.synchronize(SMBACKEND)
    ref = [Int32(count(>(0.5f0), cpu[(b-1)*BTILE+1 : b*BTILE])) for b in 1:nblocks]
    @test Array(out) == ref
end

# ── F. Iterative stencil with many barriers (double-buffered ping-pong) ───────
# Repeated clamped 3-point averaging inside shared memory; STEPS×2 barriers, two
# buffers. Combines three things in one kernel: the loop+barrier+double-buffer
# pattern AND the clamped boundary `?:` (LLVM lowers `lid==1 ? a[lid] : a[lid-1]`
# to an OpSelect of Workgroup pointers — exercises the workgroup pointer type-dedup
# fix in `map_pointer_type_for_value!`, here inside a barrier loop).
const STILE = 128
const STEPS = 16
@kernel function _smr_stencil!(input, output)
    gid = @index(Group)
    lid = @index(Local)
    a = @localmem Float32 (STILE,)
    b = @localmem Float32 (STILE,)
    @inbounds a[lid] = input[(gid - 1) * STILE + lid]
    @synchronize()
    for _ in 1:STEPS
        @inbounds begin
            l = lid == 1 ? a[lid] : a[lid - 1]       # clamped boundary
            r = lid == STILE ? a[lid] : a[lid + 1]
            b[lid] = (l + a[lid] + r) / 3f0
        end
        @synchronize()
        @inbounds a[lid] = b[lid]
        @synchronize()
    end
    @inbounds output[(gid - 1) * STILE + lid] = a[lid]
end

@testset "iterative shared-memory stencil (clamped, 32 barriers, double-buffered)" begin
    nblocks = 3
    n = STILE * nblocks
    cpu = rand(Float32, n)
    g = MVE.LavaArray(cpu)
    out = MVE.LavaArray(zeros(Float32, n))
    _smr_stencil!(SMBACKEND)(g, out; ndrange = n, workgroupsize = STILE)
    KA.synchronize(SMBACKEND)
    # CPU reference: same per-block clamped 3-point average, STEPS times
    ref = copy(cpu)
    for blk in 1:nblocks
        rng = (blk-1)*STILE+1 : blk*STILE
        v = cpu[rng]
        for _ in 1:STEPS
            nv = similar(v)
            for i in 1:STILE
                l = i == 1 ? v[i] : v[i-1]
                r = i == STILE ? v[i] : v[i+1]
                nv[i] = (l + v[i] + r) / 3f0
            end
            v = nv
        end
        ref[rng] = v
    end
    @test Array(out) ≈ ref rtol=1f-3
end
