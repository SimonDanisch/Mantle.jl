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

# The reference's kernels on lavapipe, a CPU Vulkan device, in Float64 as the
# reference computes. `KA.CPU()` ran them on POCL, which crashed in its barrier
# scheduler on a Mac. The vendored file stays unedited: these are its
# `update_keys` and `refine` as written, except that the arrays live on the
# device and its scan, a host loop, reads the counts back.
const LVP = Mantle.backend(Mantle.Device("lavapipe"))
lvp(x::AbstractArray) = (d = KA.allocate(LVP, eltype(x), size(x)); copyto!(d, x); d)

function lvp_update_keys(keys, cx, cy, cf; geo_tol, fld_tol, max_depth)
    dkeys = lvp(keys)
    counts = KA.allocate(LVP, Int, length(keys))
    IsubdRef.classify_kernel!(LVP, 64)(counts, dkeys, cx, cy, cf, geo_tol, fld_tol, max_depth;
                                       ndrange = length(keys))
    offsets, total = IsubdRef.exclusive_scan(Array(counts))
    out = KA.allocate(LVP, UInt64, total)
    IsubdRef.scatter_kernel!(LVP, 64)(out, dkeys, counts, lvp(offsets), cx, cy, cf,
                                      geo_tol, fld_tol, max_depth; ndrange = length(keys))
    return Array(out)
end

function lvp_refine(cx, cy, cf; geo_tol, fld_tol, max_depth = 12)
    keys = UInt64[IsubdRef.root_key(b) for b in 1:IsubdRef.NBASE]
    for _ in 1:(max_depth + 1)
        new = lvp_update_keys(keys, cx, cy, cf; geo_tol, fld_tol, max_depth)
        new == keys && break
        keys = new
    end
    return keys
end

@testset "isubd pass A: refinement against its CPU reference" begin
    backend = Mantle.defaultbackend()
    todevice(x) = Mantle.devicearray(backend, x)
    cx, cy, cf = IsubdRef.build_coefficients()
    lcx, lcy, lcf = lvp(cx), lvp(cy), lvp(cf)
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
        lbc, lbcell = lvp(basecorners), lvp(basecell)
        for (label, geo_tol, fld_tol, expect) in CASES
            r = lvp_refine(lcx, lcy, lcf; geo_tol, fld_tol)
            g = IsubdGpu.refine(lvp(IsubdGpu.rootkeys()), lcx, lcy, lcf, lbc, lbcell;
                                geo_tol, fld_tol)
            @test length(r) == expect
            @test Array(g) == r
        end
    end

    @testset "pass A on the device" begin
        for (label, geo_tol, fld_tol, expect) in CASES
            r = lvp_refine(lcx, lcy, lcf; geo_tol, fld_tol)
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
            cpu = lvp_update_keys(cpu, lcx, lcy, lcf;
                                  geo_tol = 2.0e-2, fld_tol = 5.0e-2, max_depth = 12)
            # Every intermediate state matches, not just the fixed point —
            # a merge rule that is wrong for one pass and self-correcting by
            # the next would pass an end-state-only check.
            @test Array(gpu) == cpu
        end
        @test length(gpu) < 1106            # it really did coarsen
        @test Array(gpu) ==
              lvp_refine(lcx, lcy, lcf; geo_tol = 2.0e-2, fld_tol = 5.0e-2)
    end
end
