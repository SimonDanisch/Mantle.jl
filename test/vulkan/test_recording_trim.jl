using Test, Mantle, KernelAbstractions

@kernel function recording_trim_add!(out, @Const(input))
    i=@index(Global)
    out[i]=input[i]+1f0
end

function check_recording_trim(backend)
    @testset "explicit trim releases cached commands and preserves live plans" begin
        dev=Mantle.Device(backend)
        input=Mantle.Buffer(dev,fill(2f0,256))
        output=Mantle.Buffer(dev,zeros(Float32,256))
        graph=Mantle.Graph(dev)
        temporary=Mantle.Transient.Buffer(graph,Float32,256)
        Mantle.dispatch!(graph,recording_trim_add!,(temporary,input),256)
        Mantle.dispatch!(graph,recording_trim_add!,(output,temporary),256)
        plan=Base.invokelatest(Mantle.Plan,graph)
        try
            Mantle.record!(plan)
            Mantle.run!(plan);Mantle.waitfor!(plan)
            @test Array(output)==fill(4f0,256)
            @test Mantle.recorded(plan)
            # An unrelated native-file upload leaves an empty large block.
            scratch=Mantle.LavaArray(zeros(UInt32,20*1024*1024);bq=dev.default_bq)
            Mantle.waitidle(dev)
            Mantle.unsafe_free!(scratch)
            GC.gc()
            before=Mantle.reserved(Mantle.pool(dev))
            Mantle.trim_gpu_pool!()
            @test Mantle.recorded(plan) # default keeps reusable commands
            reclaimed=Mantle.trim_gpu_pool!(;invalidate_recordings=true)
            @test reclaimed[2]>0
            @test Mantle.reserved(Mantle.pool(dev))<before
            @test !Mantle.recorded(plan)
            @test plan.stale
            fill!(Mantle.storage(input),7f0)
            Mantle.run!(plan);Mantle.waitfor!(plan)
            @test Mantle.recorded(plan)
            @test Array(output)==fill(9f0,256)
            # A trim inside an open recording must not pop its hold frame.
            Mantle.oneshot!(dev.default_bq) do _
                @test Mantle.trim_gpu_pool!(;invalidate_recordings=true)==(0,0)
            end
            other=Mantle.allocate_batch_queue!(dev)
            try
                Mantle.oneshot!(other) do _
                    @test Mantle.trim_gpu_pool!(;invalidate_recordings=true)==(0,0)
                end
            finally
                Mantle.waitidle(dev)
                Mantle.release_batch_queue!(other)
            end
            Mantle.run!(plan);Mantle.waitfor!(plan)
            @test Array(output)==fill(9f0,256)
        finally
            Mantle.free!(plan);Mantle.free!(input);Mantle.free!(output)
        end
    end
end
