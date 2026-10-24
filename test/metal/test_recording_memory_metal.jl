using Test, Mantle, Metal, KernelAbstractions

@kernel function recorded_fill!(a, n, v)
    i = @index(Global)
    @inbounds if i <= n[1]
        a[i] = v
    end
end

function dropped_recording(dst, n)
    g = Mantle.Graph(Mantle.Device(Metal.MetalBackend()))
    Mantle.dispatch!(g, recorded_fill!, (dst, n, 1f0), Mantle.DeviceRange(n; max = length(dst)); name = "fill")
    pl = Mantle.record!(Mantle.Plan(g))
    Mantle.run!(pl)
    Mantle.free!(pl)
    return nothing
end

resident_allocations() = sum(s -> Int(s.allocationCount), values(Metal.queue_residency_sets); init = 0)
managed_records() = length(Mantle.Device(Metal.MetalBackend()).managed)

function settle()
    Mantle.waitidle(Mantle.Device(Metal.MetalBackend()))
    GC.gc(true); GC.gc(true)
    Metal.trim_residency!()
end

@testset "Metal: a freed recording gives its memory back" begin
    mtl = Metal.device()
    dst = KernelAbstractions.allocate(Metal.MetalBackend(), Float32, 1024)
    n = Mantle.Buffer(Mantle.Device(Metal.MetalBackend()), Int32[length(dst)])
    dropped_recording(dst, n)
    settle()
    resident, allocated, records = resident_allocations(), mtl.currentAllocatedSize, managed_records()

    for _ in 1:4
        dropped_recording(dst, n)
    end
    settle()
    @test resident_allocations() == resident
    @test mtl.currentAllocatedSize == allocated
    @test managed_records() == records
    @test all(==(1f0), Array(dst))
end
