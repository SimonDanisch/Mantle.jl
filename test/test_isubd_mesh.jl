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
using KernelAbstractions: CPU

module IsubdRef
include(joinpath(@__DIR__, "..", "examples", "isubd", "reference.jl"))
end
module IsubdGpu
include(joinpath(@__DIR__, "..", "examples", "isubd", "mantle_isubd.jl"))
end

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
        # Deliberately on host arrays: this isolates the PORT (tuple
        # constants rewritten as buffers, unrolled monomials) from the
        # device. A failure here is arithmetic, not Vulkan.
        hostkeys = IsubdGpu.rootkeys()
        for (label, geo_tol, fld_tol, expect) in CASES
            r = IsubdRef.refine(CPU(), cx, cy, cf; geo_tol, fld_tol)
            g = IsubdGpu.refine(hostkeys, cx, cy, cf, basecorners, basecell;
                                geo_tol, fld_tol)
            @test length(r) == expect
            @test g == r
        end
    end

    @testset "pass A on the device" begin
        for (label, geo_tol, fld_tol, expect) in CASES
            r = IsubdRef.refine(CPU(), cx, cy, cf; geo_tol, fld_tol)
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
            cpu = IsubdRef.update_keys(CPU(), cpu, cx, cy, cf;
                                       geo_tol = 2.0e-2, fld_tol = 5.0e-2, max_depth = 12)
            # Every intermediate state matches, not just the fixed point —
            # a merge rule that is wrong for one pass and self-correcting by
            # the next would pass an end-state-only check.
            @test Array(gpu) == cpu
        end
        @test length(gpu) < 1106            # it really did coarsen
        @test Array(gpu) ==
              IsubdRef.refine(CPU(), cx, cy, cf; geo_tol = 2.0e-2, fld_tol = 5.0e-2)
    end
end
