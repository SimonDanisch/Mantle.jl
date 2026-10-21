using Test, Mantle, GeometryBasics, LinearAlgebra
import Raycore

function check_tlas_mutation_allocations(ctx)
    @testset "queued GPU transforms remain allocation-free" begin
        mesh = Mesh(Point3f[(0,0,0), (1,0,0), (0,1,0)], GLTriangleFace[(1,2,3)])
        t = Mantle.VulkanTLAS(Mantle.LavaBackend(ctx))
        h = push!(t, mesh, fill(Mat4f(I), 2);
                  instance_ids=UInt32[17,29], instance_mask=UInt8(0x42))
        transforms = Mantle.LavaArray(fill(Mantle.identity_transform(), 2); bq=t.bq)
        Raycore.update_transforms!(t, h, transforms)
        Raycore.update_transforms!(t, h, transforms)
        @test (@allocated Raycore.update_transforms!(t, h, transforms)) == 0
        Raycore.sync!(t)
        records = Array(Raycore.instance_buffer(t, h))
        @test getproperty.(records, :id) == UInt32[17,29]
        @test getproperty.(records, :mask) == UInt32[0x42,0x42]
        @test isempty(Mantle.validationmessages!(Mantle.lavadevice(ctx)))
    end
end
