"""
The kernel-argument pin walker stops at a `VulkanTLAS`, for EVERY owner.

`pin_leaves!` recurses into an argument's fields until it finds a `LavaArray` to
pin, and a `VulkanTLAS` is a cycle: it holds its batch queue, the queue holds the
context, the context holds the queue. It is also unnecessary — the arrays a ray
query walks are exposed directly as fields of the `AdaptedAccel` the kernel is
handed — so there is a method that stops the walk.

That method was written for a single closed-command-buffer owner, from when a
one-shot was the only thing a pin could go into. A [`Recording`](@ref) is the
other one, so a hardware-RT plan fell through to the generic walker the first
time it was RECORDED rather than launched, and the walk reached `VK.Instance` — whose
`destructor` field is a closure over the instance itself. 53 320 frames of
`pin_leaves!(::Recording, ::VK.Instance)` and a `StackOverflowError`, from
`Hikari`'s hardware trace pass.

Structural rather than behavioural on purpose: the failure is a stack overflow,
which corrupts the session it happens in, so what is worth pinning is that the
stop exists for both owners rather than that a render survives.
"""

using Test, GeometryBasics, StaticArrays, LinearAlgebra
using Raycore, Lava, Mantle

@testset "pin_leaves! stops at a VulkanTLAS for every owner" begin
    backend = MVE.LavaBackend()
    bq = backend.dispatch_bq
    hwtlas = MVE.VulkanTLAS(backend)
    push!(hwtlas, GeometryBasics.normal_mesh(Sphere(Point3f(0, 0, 0), 1f0)),
          SMatrix{4,4,Float32}(I); instance_id = UInt32(1))
    Raycore.sync!(hwtlas)

    # Both owners answer, and both answer by doing nothing. A missing method
    # here is not a `MethodError` — it is the generic walker, and the generic
    # walker on this type does not return.
    batch = MVE.oneshot(bq) do e end
    @test MVE.pin_leaves!(batch, hwtlas) === nothing
    @test isempty(batch.pinned)

    rec = MVE.recording!(bq)
    @test MVE.pin_leaves!(rec, hwtlas) === nothing
    @test isempty(rec.pinned)
    # …and through the wrapper a kernel is actually handed.
    @test MVE.pin_leaves!(rec, Mantle.AdaptedAccel(hwtlas)) === nothing
    MVE.seal!(rec)
    MVE.release!(rec)

    MVE.recycle!(bq, batch)
end
