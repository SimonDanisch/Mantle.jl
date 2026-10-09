"""
A device array reached through a STORED handle must be readable.

The launch path takes care of the arrays a kernel is handed. There is a second
way an array reaches a kernel: converted to its device-side handle ahead of time
(`KernelInterface.argconvert`) and STORED somewhere the launch path never walks —
a table of handles, an argument buffer, a struct field. Raycore's `store_texture`
builds its texture table exactly that way.

On Metal a resource is only mapped for a dispatch if something says so: the
encoder is asked (`useResource`), or the resource is in a residency set. Without
it the GPU reads unmapped memory, and it does not fault: it returns zeros,
intermittently, depending on what else happens to be mapped. The symptom was
image textures and environment maps rendering black — killeroo_gold's background
vanished and its mean dropped from 0.56 to 0.02 — and it looked for a long time
like a nondeterministic race in the integrator. Metal.jl makes the converted
buffer persistently resident now; this pins the behaviour rather than the
mechanism, on every backend: a kernel that dereferences a stored handle sees the
data.
"""

using Test, Mantle, KernelAbstractions
import KernelInterface as KI
include(joinpath(@__DIR__, "testbackend.jl"))
const KA = KernelAbstractions

# Reads through a handle the kernel was never handed as an argument.
@kernel function stored_read!(out, @Const(table), n::Int32)
    i = @index(Global)
    if i <= n
        @inbounds out[i] = table[1][i]
    end
end

@testset "a stored device-array handle reads its data" begin
    be = TESTBACKEND
    n = 4096
    data = Float32.(1:n)
    src = Mantle.devicearray(be, data)

    # Bake the handle the way `store_texture` does — outside any launch.
    table = Mantle.devicearray(be, [KI.argconvert(be, src)])
    out = KA.zeros(be, Float32, n)

    # Eight rounds, each churning device memory and then reading through the
    # stored handle again.
    #
    # One round is not enough: "not resident" means the driver is FREE to unmap,
    # not that it has, so on an idle machine a single round missed the bug about
    # a third of the time. That is inherent to the failure, not a weakness of the
    # check — and it is why the bug survived so long in renders. Eight independent
    # rounds bring the miss probability under a thousandth.
    #
    # `src` is preserved because nothing below mentions it, and a stored handle is
    # not a reference the collector can see. Without it the `GC.gc(true)` frees the
    # very array the table names, and every round would be measuring
    # use-after-free. A caller that stores a handle is required to keep the array
    # alive; Raycore's `store_texture` does exactly that with its texture table.
    ok = trues(8)
    GC.@preserve src for round in 1:8
        for _ in 1:16
            junk = KA.zeros(be, Float32, 4 * 1024 * 1024)
            fill!(junk, 1f0)
        end
        GC.gc(true)
        fill!(out, 0f0)
        stored_read!(be, 256)(out, table, Int32(n); ndrange = n)
        KA.synchronize(be)
        ok[round] = Array(out) == data
    end
    @test all(ok)
    # Name the failure mode: unmapped memory reads as zeros, so a regression shows
    # up as an all-zero round rather than as an error.
    @test count(ok) == 8
end

@testset "converting the same array again is the same handle" begin
    # `store_texture` converts one array per texture and a scene has many, so the
    # conversion runs repeatedly on one device. Converting the same array twice
    # must give the same handle and not throw.
    be = TESTBACKEND
    a = Mantle.devicearray(be, Float32[1, 2, 3, 4])
    @test KI.argconvert(be, a) === KI.argconvert(be, a)
    b = Mantle.devicearray(be, Float32[5, 6])
    @test KI.argconvert(be, b) !== nothing
end
