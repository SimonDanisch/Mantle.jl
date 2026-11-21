using Test, Mantle, Metal, Raycore, GeometryBasics

function dropped_hwtlas(be)
    t = Mantle.HWTLAS{Raycore.Triangle{UInt32}}(be)
    push!(t, GeometryBasics.normal_mesh(Tesselation(Sphere(Point3f(0), 1f0), 400)))
    push!(t, GeometryBasics.normal_mesh(Tesselation(Sphere(Point3f(3, 0, 0), 1f0), 400)))
    Raycore.sync!(t)
    push!(t, GeometryBasics.normal_mesh(Rect3f(Point3f(-5), Vec3f(1))))
    Raycore.sync!(t)
    return nothing
end

# The device's residency set holds what its acceleration structures reach by address.
resident_allocations(be) = Int(Mantle.residency(be).set.allocationCount)

# Collected wrappers leave the set at the next drain: a recorded replay in flight may
# still reach their memory by address (`MetalBindings.forget_after_drain!`).
function settle(be)
    Mantle.waitidle(be)
    GC.gc(true); GC.gc(true)
    Mantle.waitidle(be)
    Mantle.MetalBindings.trim!(Mantle.residency(be))
end

@testset "Metal: a dropped HWTLAS gives its memory back" begin
    be = Mantle.Device(Mantle.MetalAPI())
    dev = Metal.device()
    dropped_hwtlas(be)
    settle(be)
    resident, allocated = resident_allocations(be), dev.currentAllocatedSize

    for _ in 1:4
        dropped_hwtlas(be)
    end
    settle(be)
    @test resident_allocations(be) == resident
    @test dev.currentAllocatedSize == allocated
end
