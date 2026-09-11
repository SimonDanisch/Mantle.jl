"""
The kernel-argument hold walk stops at a `VulkanTLAS`, for EVERY owner.

`holdleaves!` recurses into an argument's fields until it finds an array to
hold, and a `VulkanTLAS` is a cycle: it holds its submission channel, the
channel holds the context, the context holds the channel. It is also
unnecessary — the arrays a ray query walks are exposed directly as fields of the
`AdaptedAccel` the kernel is handed — so there is a method that stops the walk.

That method was written for a single closed-command-buffer owner, from when a
one-shot was the only thing a claim could go into. A `Recording` is the other
one, so a hardware-RT plan fell through to the generic walker the first time it
was RECORDED rather than launched, and the walk reached `VK.Instance` — whose
`destructor` field is a closure over the instance itself. 53 320 frames and a
`StackOverflowError`, from Hikari's hardware trace pass.

Structural rather than behavioural on purpose: the failure is a stack overflow,
which corrupts the session it happens in, so what is worth pinning is that the
stop exists for both owners rather than that a render survives.
"""

using Test, GeometryBasics, StaticArrays, LinearAlgebra
using Raycore, Lava, Mantle

@testset "holdleaves! stops at a VulkanTLAS for every owner" begin
    backend = Mantle.LavaBackend()
    bq = backend.dispatch_bq
    hwtlas = Mantle.VulkanTLAS(backend)
    push!(hwtlas, GeometryBasics.normal_mesh(Sphere(Point3f(0, 0, 0), 1f0)),
          SMatrix{4,4,Float32}(I); instance_id = UInt32(1))
    Raycore.sync!(hwtlas)

    # Both owners answer, and both answer by doing nothing. A missing method
    # here is not a `MethodError` — it is the generic walker, and the generic
    # walker on this type does not return.
    o = Mantle.oneshot(bq) do e end
    @test Mantle.holdleaves!(o, hwtlas) === nothing
    @test isempty(o.sync)

    rec = Mantle.recording!(bq)
    @test Mantle.holdleaves!(rec, hwtlas) === nothing
    @test isempty(rec.sync)
    # …and through the wrapper a kernel is actually handed.
    @test Mantle.holdleaves!(rec, Mantle.AdaptedAccel(hwtlas)) === nothing
    Mantle.release!(rec)

    Mantle.release!(bq, o)
end
