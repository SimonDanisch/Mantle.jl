using Test, Mantle, Raycore, GeometryBasics, LinearAlgebra, GPUArrays

function check_accel_barriers(ctx)
    dev = Mantle.lavadevice(ctx)
    if !Mantle.supports_rt_pipeline(dev)
        @test_skip Mantle.supports_rt_pipeline(dev)
        return nothing
    end
    Mantle.defaultdevice!(dev)
    be = Mantle.backend(dev)
    @testset "pooled and ordinary AS builds preserve dependency contracts" begin
        vertices = [[(x-0.4f0,-0.4f0,0f0),(x+0.4f0,-0.4f0,0f0),(x,0.4f0,0f0)]
                    for x in (0f0,3f0,6f0)]
        indices = [UInt32[0,1,2] for _ in vertices]
        masks = UInt8[1,2,3]
        xf = Mat4f(0.75,0,0,0, 0,1.25,0,0, 0,0,1,0, 1,2,1,1)
        sw = Raycore.TLAS(be)
        handles = Raycore.TLASHandle[]
        for (i,vs) in enumerate(vertices)
            mesh = GeometryBasics.normal_mesh(GeometryBasics.Mesh(
                Point3f.(vs),[GLTriangleFace(1,2,3)]))
            push!(handles,push!(sw,mesh,xf;instance_id=UInt32(10+i),instance_mask=masks[i]))
        end
        Raycore.sync!(sw)
        ch = Mantle.allocate_batch_queue!(dev)
        before = Mantle.driver(ch).next_timeline
        combined = Mantle.HardwareAccel(sw;ctx,bq=ch)
        @test Mantle.driver(ch).next_timeline > before
        blases = Mantle.build_accel!(dev.bq) do build
            [Mantle.build_blas(build,vs,is) for (vs,is) in zip(vertices,indices)]
        end
        tlas = Mantle.build_accel!(dev.bq) do build
            Mantle.build_tlas(build,blases;custom_indices=UInt32[11,12,13],
                             transforms=fill(Tuple(Raycore.mat4_to_mat3x4(xf)),3),masks)
        end
        separate = Mantle.HardwareAccel(tlas,combined.triangle_data,combined.blas_offsets,
                                       combined.per_instance_tri_offsets;bq=dev.bq)
        rays = Mantle.devicearray(be,[Raycore.RTRay(0.75f0*x+1f0,2f0,5f0,0f0,0f0,0f0,-1f0,10f0)
                                     for x in (0f0,3f0,6f0,20f0)])
        out = Mantle.devicearray(be,Raycore.RTHitResult,4)
        raycount = Mantle.devicearray(be,Int32[4])
        hidden = nothing
        try
            for accel in (combined,separate), indirect in (false,true), mask in UInt32[0xff,1,2]
                if indirect
                    Mantle.trace_closest_hits_indirect!(out,rays,accel,raycount;cull_mask=mask)
                else
                    Mantle.trace_closest_hits!(out,rays,accel,4;cull_mask=mask)
                end
                result = Array(out)
                want = vcat((masks .& UInt8(mask)) .!= 0,false)
                @test (getproperty.(result,:hit) .!= 0) == want
                @test getproperty.(result[findall(want)],:t) ≈ fill(4f0,count(want))
                @test getproperty.(result,:instance_custom_index) ==
                      ifelse.(want,UInt32[11,12,13,0],UInt32(0))
            end
            @test Raycore.set_visible!(sw,last(handles),false)
            hidden = Mantle.HardwareAccel(sw;ctx,bq=ch)
            for indirect in (false,true)
                indirect ? Mantle.trace_closest_hits_indirect!(out,rays,hidden,raycount) :
                           Mantle.trace_closest_hits!(out,rays,hidden,4)
                result = Array(out)
                @test getproperty.(result,:hit) == UInt32[1,1,0,0]
                @test getproperty.(result,:instance_custom_index) == UInt32[11,12,0,0]
            end
        finally
            Mantle.waitidle(dev)
            hidden === nothing || Mantle.unsafe_free!(hidden.tlas)
            foreach(Mantle.unsafe_free!,(rays,out,raycount,tlas,combined.tlas,blases...))
            Mantle.release_batch_queue!(ch)
        end
        Mantle.drain!(dev.bq)
        @test isempty(Mantle.validationmessages!(dev))
    end
    @testset "host and offset GPU vertices support in-place BLAS refits" begin
        vertices = [(0f0,0f0,0f0),(1f0,0f0,0f0),(0f0,1f0,0f0)]
        moved = [(x,y,2f0) for (x,y,_) in vertices]
        rays = Mantle.LavaArray([Raycore.RTRay(0.25f0,0.25f0,-1f0,0f0,0f0,0f0,1f0,10f0)];bq=dev.bq)
        out = Mantle.LavaArray{Raycore.RTHitResult}(undef,(1,);bq=dev.bq)
        try
            for gpu_input in (false,true)
                vb = Mantle.LavaArray(vcat([(9f0,9f0,9f0)],vertices,[(8f0,8f0,8f0)]);
                                     bq=dev.bq,extra_usage=Mantle.AS_INPUT_USAGE)
                ib = Mantle.LavaArray(UInt32[99,0,1,2,99];bq=dev.bq,extra_usage=Mantle.AS_INPUT_USAGE)
                vs = GPUArrays.derive(NTuple{3,Float32},vb,(3,),1)
                ids = GPUArrays.derive(UInt32,ib,(3,),1)
                blas = Mantle.build_accel!(dev.bq) do build
                    gpu_input ? Mantle.build_blas(build,vs,ids;allow_update=true) :
                                Mantle.build_blas(build,vertices,UInt32[0,1,2];allow_update=true)
                end
                tlas = Mantle.build_accel!(dev.bq) do build
                    Mantle.build_tlas(build,[blas];allow_update=true)
                end
                accel = Mantle.HardwareAccel(tlas,Float32[],UInt32[0],UInt32[0];bq=dev.bq)
                try
                    Mantle.trace_closest_hits!(out,rays,accel,1)
                    @test only(Array(out)).t ≈ 1f0
                    Mantle.build_accel!(dev.bq) do build
                        Mantle.refit_blas!(build,blas,moved)
                        Mantle.refit_tlas!(build,tlas,tlas.instance_buf,1)
                    end
                    Mantle.trace_closest_hits!(out,rays,accel,1)
                    @test only(Array(out)).t ≈ 3f0
                    if gpu_input
                        @test Mantle.bda_address(blas.vertex_arr) == Mantle.bda_address(vs)
                        @test Array(vb) == vcat([(9f0,9f0,9f0)],moved,[(8f0,8f0,8f0)])
                    end
                    @test_throws ArgumentError Mantle.build_accel!(dev.bq) do build
                        Mantle.refit_blas!(build,blas,vcat(moved,moved))
                    end
                finally
                    foreach(Mantle.unsafe_free!,(tlas,blas,vs,ids,vb,ib))
                end
            end
        finally
            Mantle.unsafe_free!(rays); Mantle.unsafe_free!(out)
        end
        Mantle.waitidle(dev); Mantle.drain!(dev.bq)
        @test isempty(Mantle.validationmessages!(dev))
    end
end
