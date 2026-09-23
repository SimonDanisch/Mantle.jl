# Adaptive FEM tessellation through a MESH PIPELINE, against its CPU reference.
#
# The reference is `examples/isubd/reference.jl`, vendored unedited from
# koehlerson/mantle_mwe — a self-contained reduction of what FerriteViz wants to
# do on the GPU, including a software rasteriser that produces "the image a GPU
# implementation has to reproduce". This file is that comparison.
#
# What it covers, in the order the passes run:
#
#   A  the key set        refinement AND coarsening, against the reference's
#   B  the mesh shader    three vertices per key, no vertex or index buffer
#   C  the fragment stage the field evaluated per pixel from the coefficients
#
# The keys are compared EXACTLY — they are integers, and a topology that is one
# triangle different is a different mesh. The images are compared per pixel with
# a tolerance, because the reference rasterises in Float64 and a GPU does not.
#
# Why images and not just triangle counts: a count says nothing about whether
# the triangles are in the right PLACE. The first run here had the right count
# and the wrong base corners.

using Test
using Mantle, Lava, LinearAlgebra
using KernelAbstractions: CPU

const ISUBD_DIR = joinpath(@__DIR__, "..", "..", "examples", "isubd")

module IsubdRef
include(joinpath(@__DIR__, "..", "..", "examples", "isubd", "reference.jl"))
end
module IsubdGpu
include(joinpath(@__DIR__, "..", "..", "examples", "isubd", "mantle_isubd.jl"))
end

"""
How many pixels disagree by more than `tol`, and the mean disagreement.

A COUNT rather than a maximum, and that is the point. The reference mesh has
T-vertex pinholes — single white pixels where non-conforming refinement leaves a
crack, documented in its own README — and the two rasterisers place them a pixel
apart. One such pixel makes a max-difference 0.765 on an image that is otherwise
exact to 1e-5, so a max assertion tests pinhole placement rather than whether
the mesh is right.

Both numbers, because either alone is fooled: a count ignores how wrong the
outliers are, and a mean hides a small patch of completely wrong pixels.
"""
function pixeldisagreement(gpu, ref, w, h; tol = 0.02)
    nbad = 0; total = 0.0
    for y in 1:h, x in 1:w
        p = gpu[x, y]; r = ref[y, x]
        d = max(abs(p.r - r[1]), abs(p.g - r[2]), abs(p.b - r[3]))
        d > tol && (nbad += 1)
        total += d
    end
    return (nbad = nbad, mean = total / (w * h))
end

"The reference's own image for a tolerance pair."
function reference_image(cx, cy, cf, geo_tol, fld_tol)
    keys = IsubdRef.refine(CPU(), cx, cy, cf; geo_tol, fld_tol)
    pos, xi, cellid = IsubdRef.decode(CPU(), keys, cx, cy)
    img, _ = IsubdRef.render(pos, xi, cellid, cf; per_fragment = true)
    return keys, img
end

@testset "isubd — adaptive FEM tessellation on a mesh pipeline" begin
    backend = Mantle.defaultbackend()

    if !Mantle.supports_mesh_pipeline(backend)
        @info "isubd: this device has no mesh pipeline; skipping" backend = nameof(typeof(backend))
    else
        todevice(x) = Mantle.devicearray(backend, x)
        cx, cy, cf = IsubdRef.build_coefficients()
        basecorners, basecell = IsubdGpu.basetriangles()
        dcx, dcy, dcf = todevice(cx), todevice(cy), todevice(cf)
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

        @testset "the drawn image matches the reference rasteriser" begin
            W = H = 700
            renderer = IsubdGpu.IsubdRenderer(; width = W, height = H)
            env = todevice(IsubdGpu.skyenv())
            for (label, geo_tol, fld_tol, expect) in CASES
                refkeys, refimg = reference_image(cx, cy, cf, geo_tol, fld_tol)
                keys = IsubdGpu.refine(roots(), dcx, dcy, dcf, dbc, dbcell; geo_tol, fld_tol)
                # Flat, orthographic, no background, no stroke: the reference
                # rasteriser draws the field on white in a fixed rectangle of
                # the domain and knows nothing about a 3D camera or a sky. The
                # DEMO's perspective view is the same shaders with different
                # arguments, which is the point of them being arguments.
                px = IsubdGpu.draw!(renderer, keys, dcx, dcy, dcf, dbc, dbcell;
                                    mvp = IsubdGpu.orthocamera(), warp = 0.0,
                                    background = false, wireframe = false, shaded = false,
                                    env = env, cambasis = IsubdGpu.Mat4f(1.0I),
                                    clear = (1f0, 1f0, 1f0, 1f0))
                # Measured: 0 or 1 disagreeing pixels and a mean around 1e-6.
                # `25` is loose enough for pinhole placement to move and far
                # tighter than any real error — a mesh one triangle wrong
                # disagrees over thousands of pixels, and a wrong transform over
                # all of them.
                d = pixeldisagreement(px, refimg, W, H)
                @test d.nbad <= 25
                @test d.mean < 1.0e-4
                @test count(p -> p.r < 0.99 || p.g < 0.99 || p.b < 0.99, px) > 10_000
            end
        end

        @testset "one renderer serves any triangle count" begin
            # The key buffer IS the geometry, so a refinement changes only the
            # draw's workgroup count — no reallocation, no new pipeline, no
            # upload. Reusing one renderer across two very different counts is
            # what proves it.
            renderer = IsubdGpu.IsubdRenderer(; width = 256, height = 256)
            env2 = todevice(IsubdGpu.skyenv())
            counts = Int[]
            for (label, geo_tol, fld_tol, _) in CASES
                keys = IsubdGpu.refine(roots(), dcx, dcy, dcf, dbc, dbcell; geo_tol, fld_tol)
                px = IsubdGpu.draw!(renderer, keys, dcx, dcy, dcf, dbc, dbcell;
                                    mvp = IsubdGpu.orthocamera(), warp = 0.0,
                                    background = false, wireframe = false, shaded = false,
                                    env = env2, cambasis = IsubdGpu.Mat4f(1.0I),
                                    clear = (1f0, 1f0, 1f0, 1f0))
                push!(counts, count(p -> p.r < 0.99 || p.g < 0.99 || p.b < 0.99, px))
            end
            @test all(>(1000), counts)
            # The silhouette tightens with refinement, so the coarse mesh covers
            # STRICTLY more of the frame than the fine one — a cheap check that
            # the three draws actually differed.
            @test counts[1] != counts[2]
        end
    end
end
