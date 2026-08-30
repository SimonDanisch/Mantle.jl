"""
Buffers reached through a BAKED GPU address must be made resident.

Metal only maps a resource for a dispatch if something tells it to: either the
encoder is asked (`useResource`) or the resource is in a residency set. The
launch path handles kernel arguments, because
`Adapt.adapt_storage(::Metal.Adaptor, ::MTLBuffer)` calls `use!` when an encoder
is in scope.

There is a second way a buffer reaches a kernel, and it has no encoder:
converting an array to its raw GPU address ahead of time and STORING that
address somewhere the launch path never walks — a pointer table, an argument
buffer, a struct field. `Metal.mtlconvert` outside a launch does exactly this.
Raycore's `store_texture` builds its texture table that way, via
`KA.argconvert`.

Without residency the GPU reads unmapped memory. It does not fault: it returns
zeros, intermittently, depending on what else happens to be mapped. The symptom
was image textures and environment maps rendering black — killeroo_gold's
background vanished and its mean dropped from 0.56 to 0.02 — and it looked for
a long time like a nondeterministic race in the integrator.

`Metal.make_persistently_resident!`, called from that no-encoder branch, is the
fix. This test pins the behaviour rather than the mechanism: a kernel that
dereferences a stored address must see the data.
"""

using Test, Metal, KernelAbstractions
const KA = KernelAbstractions
const MTL = Metal.MTL

# Reads through a pointer the kernel was never handed as an array.
function _deref_kernel(out, table, n)
    i = Metal.thread_position_in_grid_1d()
    if i <= n
        p = reinterpret(Core.LLVMPtr{Float32, Metal.AS.Device}, table[1])
        @inbounds out[i] = unsafe_load(p, i)
    end
    return
end

@testset "Metal: a baked GPU address is backed by residency" begin
    n = 4096
    data = Float32.(1:n)
    src = MtlArray(data)

    # Bake the address the way `store_texture` does — no encoder in scope.
    ptr = Metal.mtlconvert(src)
    addr = Base.bitcast(UInt64, pointer(ptr))
    @test addr != 0

    table = MtlArray(UInt64[addr])
    out = Metal.zeros(Float32, n)

    # Eight rounds, each churning device memory and then reading through the
    # baked address again.
    #
    # One round is not enough: "not resident" means the driver is FREE to unmap,
    # not that it has, so on an idle machine a single round misses the bug about
    # a third of the time. That is inherent to the failure, not a weakness of
    # the check — and it is why the bug survived so long in renders. Eight
    # independent rounds bring the miss probability under a thousandth.
    ok = trues(8)
    for round in 1:8
        for _ in 1:16
            junk = Metal.zeros(Float32, 4 * 1024 * 1024)
            junk .= 1f0
        end
        GC.gc(true)
        fill!(out, 0f0)
        Metal.@metal threads=256 groups=cld(n, 256) _deref_kernel(out, table, Int32(n))
        Metal.synchronize()
        ok[round] = Array(out) == data
    end

    @test all(ok)
    # Name the failure mode: unresident memory reads as zeros, so a regression
    # shows up as an all-zero round rather than as an error.
    @test count(ok) == 8
end

@testset "Metal: residency registration is idempotent and cheap to repeat" begin
    # `store_texture` bakes one address per texture and a scene has many, so the
    # no-encoder branch runs repeatedly on the same device. Registering the same
    # buffer twice must not throw or corrupt the set.
    a = MtlArray(Float32[1, 2, 3, 4])
    p1 = Metal.mtlconvert(a)
    p2 = Metal.mtlconvert(a)
    @test pointer(p1) == pointer(p2)
    b = MtlArray(Float32[5, 6])
    @test Metal.mtlconvert(b) !== nothing
end
