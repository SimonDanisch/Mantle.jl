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

resident_allocations() = sum(s -> Int(s.allocationCount), values(Metal.queue_residency_sets); init = 0)

function settle()
    Metal.synchronize()
    GC.gc(true); GC.gc(true)
    Metal.trim_residency!()
end

@testset "Metal: a dropped HWTLAS gives its memory back" begin
    be = Metal.MetalBackend()
    dev = Metal.device()
    dropped_hwtlas(be)
    settle()
    resident, allocated = resident_allocations(), dev.currentAllocatedSize

    for _ in 1:4
        dropped_hwtlas(be)
    end
    settle()
    @test resident_allocations() == resident
    @test dev.currentAllocatedSize == allocated
end
