# Include inside DirectHostProbe; performance diagnostic, not production policy.
function alignment_cost(match; bytes=64<<20)
    d=Native(match)
    lease=nothing; dst=nothing; readback=nothing; qp=nothing
    commands=VK.CommandBuffer[]
    start_errors=length(ERRORS)
    results=NamedTuple[]
    try
        mktempdir() do dir
            lease, weak, expected, _=makelease(d,:private_anon,bytes,joinpath(dir,"unused"))
            dst=region(d,bytes);readback=region(d,bytes;host=true)
            qp=unwrap(VK.create_query_pool(d.device,VK.QUERY_TYPE_TIMESTAMP,2))
            variants=(
                (:aligned,[VK.BufferCopy(0,0,bytes)],bytes),
                (:exact,[VK.BufferCopy(0,0,bytes-12)],bytes-12),
                (:split,[VK.BufferCopy(0,0,bytes-16),VK.BufferCopy(bytes-16,bytes-16,4)],bytes-12))
            samples=Dict(name=>Float64[] for (name,_,_) in variants)
            @testset "copy alignment: $(d.name)" begin
                for (name,regions,payload) in variants
                    cb=command(d) do cmd
                        VK.cmd_reset_query_pool(cmd,qp,0,2)
                        barrier(cmd,VK.PIPELINE_STAGE_2_HOST_BIT,VK.ACCESS_2_HOST_WRITE_BIT,
                                VK.PIPELINE_STAGE_2_COPY_BIT,VK.ACCESS_2_TRANSFER_READ_BIT)
                        VK.cmd_write_timestamp_2(cmd,qp,0;stage=VK.PIPELINE_STAGE_2_ALL_COMMANDS_BIT)
                        VK.cmd_copy_buffer(cmd,lease.native.buffer,dst.buffer,regions)
                        VK.cmd_write_timestamp_2(cmd,qp,1;stage=VK.PIPELINE_STAGE_2_ALL_COMMANDS_BIT)
                        barrier(cmd,VK.PIPELINE_STAGE_2_COPY_BIT,VK.ACCESS_2_TRANSFER_WRITE_BIT,
                                VK.PIPELINE_STAGE_2_COPY_BIT,VK.ACCESS_2_TRANSFER_READ_BIT)
                        VK.cmd_copy_buffer(cmd,dst.buffer,readback.buffer,[VK.BufferCopy(0,0,payload)])
                        barrier(cmd,VK.PIPELINE_STAGE_2_COPY_BIT,VK.ACCESS_2_TRANSFER_WRITE_BIT,
                                VK.PIPELINE_STAGE_2_HOST_BIT,VK.ACCESS_2_HOST_READ_BIT)
                    end
                    push!(commands,cb)
                end
                # Alternate traversal order so clocks/cache history are not
                # confounded with a variant's position in one sequential run.
                for round in 1:9
                    order=isodd(round) ? eachindex(variants) : reverse(eachindex(variants))
                    for index in order
                        name,_,payload=variants[index]
                        cb=commands[index]
                        lease.last=submit(d,cb);waitdone(d,lease.last)
                        ticks=zeros(UInt64,2)
                        GC.@preserve ticks unwrap(VK.get_query_pool_results(d.device,qp,0,2,
                            sizeof(ticks),Ptr{Cvoid}(pointer(ticks)),8;
                            flags=VK.QUERY_RESULT_64_BIT | VK.QUERY_RESULT_WAIT_BIT))
                        round>2 && push!(samples[name],(ticks[2]-ticks[1])*d.period/1e6)
                        if round==9
                            @test unsafe_wrap(Vector{UInt8},readback.pointer,payload;own=false)==view(expected,1:payload)
                        end
                    end
                end
                for (name,_,payload) in variants
                    value=median(samples[name])
                    println("ALIGNMENT device=",d.name," variant=",name," import_bytes=",bytes,
                            " payload_bytes=",payload," upload_gpu_ms=",value,
                            " min_ms=",minimum(samples[name])," max_ms=",maximum(samples[name]))
                    push!(results,(;device=d.name,name,bytes,payload,gpu_ms=value))
                end
                @test length(ERRORS)==start_errors
            end
        end
    finally
        waitdone(d)
        isempty(commands) || VK.free_command_buffers(d.device,d.commands,commands)
        lease===nothing || retire!(d,lease)
        qp===nothing || finalize(qp)
        dst===nothing || release(d,dst)
        readback===nothing || release(d,readback)
        cleanup(d)
    end
    results
end
