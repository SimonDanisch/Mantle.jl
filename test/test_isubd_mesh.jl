# Pass A of the isubd example, the adaptive refinement of curved FEM cells,
# against its CPU reference.
#
# The reference is `examples/isubd/reference.jl`, vendored unedited from
# koehlerson/mantle_mwe. Keys are compared EXACTLY: they are integers, and a
# topology that is one triangle different is a different mesh. That includes
# every intermediate state of a coarsening sequence, not only the fixed point.
#
# The drawing half is not here. The example draws through RayMakie, and
# `examples/isubd/test_gui.jl` drives that path.
#
# It asks `Mantle.defaultbackend()` for a device and runs on every platform. It
# used to live under `test/vulkan/`, importing `Lava`, and never ran on a Mac.

using Test
using Mantle
using Lavapipe_jll
import KernelAbstractions as KA

module IsubdRef
include(joinpath(@__DIR__, "..", "examples", "isubd", "reference.jl"))
end
module IsubdGpu
include(joinpath(@__DIR__, "..", "examples", "isubd", "mantle_isubd.jl"))
end

# The reference's two kernels as host loops. No device can run them: they read
# `BASE_CORNERS` and `BASE_CELL`, host `Vector` globals, which only POCL reached
# because it runs in host memory, and POCL crashed in its barrier scheduler on a
# Mac. The bodies are the kernels' own, and everything they call is the vendored
# file's; `ref_refine` is its `refine` with `update_keys` replaced.
function refcount(k, cx, cy, cf, geo_tol, fld_tol, max_depth)
    d = IsubdRef.key_depth(k)
    if d < max_depth && IsubdRef.excess_levels(k, cx, cy, cf, geo_tol, fld_tol) > 0
        return 2                                        # split
    elseif d > 0 && IsubdRef.excess_levels(IsubdRef.key_parent(k), cx, cy, cf, geo_tol, fld_tol) <= 0
        return IsubdRef.is_child0(k) ? 1 : 0            # merge: one of the pair emits
    else
        return 1                                        # keep
    end
end

function ref_update_keys(keys, cx, cy, cf; geo_tol, fld_tol, max_depth)
    counts = [refcount(k, cx, cy, cf, geo_tol, fld_tol, max_depth) for k in keys]
    offsets, total = IsubdRef.exclusive_scan(counts)
    out = Vector{UInt64}(undef, total)
    for (k, n, o) in zip(keys, counts, offsets)
        if n == 2
            out[o + 1] = IsubdRef.key_child(k, 0)
            out[o + 2] = IsubdRef.key_child(k, 1)
        elseif n == 1
            d = IsubdRef.key_depth(k)
            merging = d > 0 && IsubdRef.excess_levels(IsubdRef.key_parent(k), cx, cy, cf, geo_tol, fld_tol) <= 0 &&
                      !(d < max_depth && IsubdRef.excess_levels(k, cx, cy, cf, geo_tol, fld_tol) > 0)
            out[o + 1] = merging ? IsubdRef.key_parent(k) : k
        end
    end
    return out
end

function ref_refine(cx, cy, cf; geo_tol, fld_tol, max_depth = 12)
    keys = UInt64[IsubdRef.root_key(b) for b in 1:IsubdRef.NBASE]
    for _ in 1:(max_depth + 1)
        new = ref_update_keys(keys, cx, cy, cf; geo_tol, fld_tol, max_depth)
        new == keys && break
        keys = new
    end
    return keys
end

# The port in Float64 runs on lavapipe, a CPU Vulkan device: its kernels take
# the base triangles as arguments, so unlike the reference's they run anywhere,
# and an Apple GPU has no Float64.
const LVP = Mantle.backend(Mantle.Device("lavapipe"))
lvp(x::AbstractArray) = (d = KA.allocate(LVP, eltype(x), size(x)); copyto!(d, x); d)

@testset "isubd pass A: refinement against its CPU reference" begin
    backend = Mantle.defaultbackend()
    todevice(x) = Mantle.devicearray(backend, x)
    cx, cy, cf = IsubdRef.build_coefficients()
    basecorners, basecell = IsubdGpu.basetriangles()
    # Float32 on the DEVICE, Float64 on the host reference, deliberately.
    # `build_coefficients` is the vendored reference's and stays double —
    # it is the thing being compared against. The kernels take single
    # precision, because an Apple GPU has no
    # Float64 at all; uploading the reference's arrays unconverted asks for
    # a double-precision device array and fails before any kernel runs.
    dcx, dcy, dcf = todevice(Float32.(cx)), todevice(Float32.(cy)), todevice(Float32.(cf))
    dbc, dbcell = todevice(basecorners), todevice(basecell)
    roots() = todevice(IsubdGpu.rootkeys())

    # The three configurations the reference's own run reports.
    CASES = [("coarse", 2.0e-2, 5.0e-2, 112),
             ("fine",   2.0e-3, 5.0e-3, 1106),
             ("geometry only", 2.0e-2, 1.0e9, 70)]

    @testset "the ported math agrees with the reference" begin
        # Deliberately on the CPU device and in Float64, as the reference is:
        # this isolates the PORT (tuple constants rewritten as buffers,
        # unrolled monomials) from the GPU. A failure here is arithmetic.
        lcx, lcy, lcf = lvp(cx), lvp(cy), lvp(cf)
        lbc, lbcell = lvp(basecorners), lvp(basecell)
        for (label, geo_tol, fld_tol, expect) in CASES
            r = ref_refine(cx, cy, cf; geo_tol, fld_tol)
            g = IsubdGpu.refine(lvp(IsubdGpu.rootkeys()), lcx, lcy, lcf, lbc, lbcell;
                                geo_tol, fld_tol)
            @test length(r) == expect
            @test Array(g) == r
        end
    end

    @testset "pass A on the device" begin
        for (label, geo_tol, fld_tol, expect) in CASES
            r = ref_refine(cx, cy, cf; geo_tol, fld_tol)
            g = IsubdGpu.refine(roots(), dcx, dcy, dcf, dbc, dbcell; geo_tol, fld_tol)
            @test length(g) == expect
            # Exact: keys are integers and the scan is not a float sum.
            @test Array(g) == r
        end
    end

    @testset "an update COARSENS as well as refines" begin
        # The half of pass A that refining from the root never reaches: a
        # key whose parent no longer needs splitting merges, and child 0 is
        # the one that emits it. Driven from the FINE set with the coarse
        # tolerances, which is what a frame does when the camera pulls back.
        fine = IsubdGpu.refine(roots(), dcx, dcy, dcf, dbc, dbcell;
                               geo_tol = 2.0e-3, fld_tol = 5.0e-3)
        @test length(fine) == 1106

        hostfine = Array(fine)
        gpu = fine
        cpu = hostfine
        for pass in 1:13
            gpu = IsubdGpu.refine!(gpu, dcx, dcy, dcf, dbc, dbcell;
                                   geo_tol = 2.0e-2, fld_tol = 5.0e-2, max_depth = 12)
            cpu = ref_update_keys(cpu, cx, cy, cf;
                                  geo_tol = 2.0e-2, fld_tol = 5.0e-2, max_depth = 12)
            # Every intermediate state matches, not just the fixed point —
            # a merge rule that is wrong for one pass and self-correcting by
            # the next would pass an end-state-only check.
            @test Array(gpu) == cpu
        end
        @test length(gpu) < 1106            # it really did coarsen
        @test Array(gpu) ==
              ref_refine(cx, cy, cf; geo_tol = 2.0e-2, fld_tol = 5.0e-2)
    end
end
