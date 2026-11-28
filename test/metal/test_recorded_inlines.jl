# Recorded Metal kernels call no Julia function: every callee is inlined.
#
# Apple's compiler intermittently gets an out-of-line call wrong in a large kernel.
# In Hikari's general closure shading kernel the hit triangle's normal transform,
# `Raycore.transform(::Mat3x4f, ::Triangle)` mapping a closure over the 48-byte
# instance matrix and returning through `sret`, left whole 32-lane groups with an
# empty BSDF: a different set of lanes on each launch of the same kernel on the
# same data, so renders on the M5 were not repeatable. The same call in a kernel
# this small comes out right, so the defect itself is not reproducible here; this
# pins what the fix relies on, with that very call.
module RecordedInlinesRegression
using Mantle, Metal, Test, KernelAbstractions, Raycore, StaticArrays
using GeometryBasics: Point3f, Point2f, Vec3f

@kernel function recorded_transform!(out, ms, tris)
    i = @index(Global)
    @inbounds out[i] = Vec3f(Raycore.transform(ms[i], tris[i]).normals[2])
end

# Julia functions a module's IR calls (intrinsics and Metal.jl's exception runtime,
# reached only when a kernel throws, excluded).
juliacalls(ir) = unique([m.captures[1] for m in eachmatch(r"call [^@\n]*@(julia_[\w.]+)", ir)
                         if !occursin(r"exception|store_string|claim_output|throw", m.captures[1])])

@testset "recorded Metal kernels call no Julia function" begin
    n = 64
    ms = [Raycore.Mat3x4f(ntuple(k -> Float32(k == 1 || k == 6 || k == 11) + 0.01f0 * k, 12)) for _ in 1:n]
    tri = Raycore.Triangle{UInt32}(SVector(Point3f(0, 0, 0), Point3f(1, 0, 0), Point3f(0, 1, 0)),
                                    SVector(Normal3f(0, 0, 1), Normal3f(0, 1, 0), Normal3f(1, 0, 0)),
                                    SVector(Vec3f(1, 0, 0), Vec3f(1, 0, 0), Vec3f(1, 0, 0)),
                                    SVector(Point2f(0, 0), Point2f(1, 0), Point2f(0, 1)), UInt32(0))
    want = [Vec3f(Raycore.transform(m, tri).normals[2]) for m in ms]

    # Compiled the way Metal.jl launches by default, the normal transform stays a
    # call: what this test looks for is there to be found.
    out = MtlArray(fill(Vec3f(0), n))
    plain = sprint() do io
        Metal.@device_code_llvm io = io recorded_transform!(Metal.MetalBackend(), 32)(
            out, MtlArray(ms), MtlArray(fill(tri, n)); ndrange = n)
    end
    Metal.synchronize()
    @test !isempty(juliacalls(plain))

    dev = Mantle.Device(Mantle.MetalAPI())
    dout = Mantle.Buffer(dev, fill(Vec3f(0), n))
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, recorded_transform!, (dout, Mantle.Buffer(dev, ms), Mantle.Buffer(dev, fill(tri, n))), n;
                     name = "recorded transform")
    # The plan compiles its dispatches as it is built.
    local pl
    recorded = sprint() do io
        Metal.@device_code_llvm io = io (pl = Mantle.record!(Mantle.Plan(g)))
    end
    try
        @test occursin("recorded_transform", recorded)
        @test isempty(juliacalls(recorded))
        Mantle.run!(pl); Mantle.waitfor!(pl)
        @test all(isapprox.(Array(Mantle.storage(dout)), want; atol = 1f-5))
    finally
        Mantle.free!(pl)
    end
end
end
