# Include inside DirectHostProbe. Diagnostic controls, never a production fallback.
function raw_import_result(d, source, index)
    C = VK.VkCore
    host = Ref(C.VkImportMemoryHostPointerInfoEXT(
        C.VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT, Ptr{Cvoid}(C_NULL),
        C.VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, Ptr{Cvoid}(pointer(source.bytes))))
    output = Ref{C.VkDeviceMemory}()
    result = GC.@preserve source host begin
        info = Ref(C.VkMemoryAllocateInfo(C.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            Ptr{Cvoid}(Base.unsafe_convert(Ptr{typeof(host[])}, host)), UInt64(length(source.bytes)), UInt32(index)))
        C.vkAllocateMemory(d.device, info, C_NULL, output)
    end
    result == C.VK_SUCCESS && C.vkFreeMemory(d.device, output[], C_NULL)
    result
end

# Another existing direct-upload construction: memcpy into the final CPU-visible,
# device-local allocation. No source import and no intermediate upload allocation.
function mapped_destination(match; bytes=4<<20)
    d = Native(match)
    dst = nothing; readback = nothing; cb = nothing
    start_errors = length(ERRORS)
    try
        mktempdir() do dir
            path = joinpath(dir,"readonly.bin")
            source, expected = makesource(:file_private_ro,bytes,path)
            flags = VK.MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK.MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    VK.MEMORY_PROPERTY_HOST_COHERENT_BIT
            dst = region(d,bytes;host=true,required_flags=flags)
            readback = region(d,bytes;host=true)
            cb = command(d) do cmd
                barrier(cmd,VK.PIPELINE_STAGE_2_HOST_BIT,VK.ACCESS_2_HOST_WRITE_BIT,
                        VK.PIPELINE_STAGE_2_COPY_BIT,VK.ACCESS_2_TRANSFER_READ_BIT)
                VK.cmd_copy_buffer(cmd,dst.buffer,readback.buffer,[VK.BufferCopy(0,0,bytes)])
                barrier(cmd,VK.PIPELINE_STAGE_2_COPY_BIT,VK.ACCESS_2_TRANSFER_WRITE_BIT,
                        VK.PIPELINE_STAGE_2_HOST_BIT,VK.ACCESS_2_HOST_READ_BIT)
            end
            samples=Float64[]
            @testset "mapped final destination: $(d.name)" begin
                for _ in 1:5
                    started=time_ns()
                    GC.@preserve source unsafe_copyto!(dst.pointer,pointer(source.bytes),bytes)
                    push!(samples,(time_ns()-started)/1e6)
                    value=submit(d,cb);waitdone(d,value)
                    @test unsafe_wrap(Vector{UInt8},readback.pointer,bytes;own=false)==expected
                    @test source.bytes==expected
                end
                @test read(path)==expected
                @test length(ERRORS)==start_errors
            end
            println("MAPPED_DESTINATION device=",d.name," bytes=",bytes,
                    " cpu_copy_ms_median=",median(samples[2:end]),
                    " explicit_upload_staging_bytes=0 validation_errors=",length(ERRORS)-start_errors)
            (;device=d.name,bytes,cpu_ms=median(samples[2:end]))
        end
    finally
        waitdone(d)
        cb===nothing || VK.free_command_buffers(d.device,d.commands,[cb])
        dst===nothing || release(d,dst)
        readback===nothing || release(d,readback)
        cleanup(d)
    end
end
function diagnose(match; bytes=4096)
    d = Native(match)
    println("DIAGNOSE ",d.name," bytes=",bytes)
    try
        mktempdir() do dir
            for kind in (:private_anon, :file_private_rw, :file_shared_rw, :file_private_ro, :file_shared_ro)
                source, expected = makesource(kind, bytes, joinpath(dir,string(kind)*".bin"))
                rawprops = VK.initialize_core(VK.VkCore.VkMemoryHostPointerPropertiesEXT, [])
                props = Ref(rawprops)
                query = VK.VkCore.vkGetDeviceProcAddr(d.device, "vkGetMemoryHostPointerPropertiesEXT")
                @assert query != C_NULL
                code = GC.@preserve source VK.VkCore.vkGetMemoryHostPointerPropertiesEXT(d.device,
                    VK.VkCore.VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
                    Ptr{Cvoid}(pointer(source.bytes)), props, query)
                println("SOURCE ",kind," raw_properties=",code," type_bits=",props[].memoryTypeBits,
                        " ptr_alignment=",UInt(pointer(source.bytes)) % d.host_import_alignment)
                usage = VK.BUFFER_USAGE_TRANSFER_SRC_BIT
                external = VK.get_physical_device_external_buffer_properties(d.physical,
                    VK.PhysicalDeviceExternalBufferInfo(usage, VK.EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT))
                println("copy_only_external_features=",external.external_memory_properties.external_memory_features)
                # A copy-only buffer removes storage/BDA use and allocation flags.
                buffer = VK.Buffer(d.device, bytes, usage, VK.SHARING_MODE_EXCLUSIVE, UInt32[];
                    next=VK.ExternalMemoryBufferCreateInfo(; handle_types=VK.EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT))
                req = VK.get_buffer_memory_requirements(d.device, buffer)
                println("copy_only_requirements size=",req.size," alignment=",req.alignment," bits=",req.memory_type_bits)
                bits = props[].memoryTypeBits & req.memory_type_bits
                for index in 0:Int(d.memory_properties.memory_type_count)-1
                    bits & (UInt32(1)<<index) == 0 && continue
                    result = raw_import_result(d, source, index)
                    println("RAW_ALLOCATE kind=",kind," type=",index,
                        " flags=",d.memory_properties.memory_types[index+1].property_flags," result=",result)
                    if kind === :file_private_ro
                        # Owned fixture only: keep its bytes/file unchanged, vary CPU protection.
                        r = GC.@preserve source ccall(:mprotect, Cint,
                            (Ptr{Cvoid}, Csize_t, Cint), pointer(source.bytes), bytes, 3)
                        @assert r == 0
                        try
                            println("PRIVATE_RO_WITH_RW_PROTECTION type=",index," result=",raw_import_result(d,source,index))
                            # Owned private fixture: fault COW pages without changing bytes.
                            GC.@preserve source for off in 0:4096:bytes-1
                                p = pointer(source.bytes)+off
                                unsafe_store!(p,unsafe_load(p))
                            end
                            println("PRIVATE_RO_RW_COW_TOUCHED type=",index," result=",raw_import_result(d,source,index))
                            @assert source.bytes == expected
                        finally
                            r = GC.@preserve source ccall(:mprotect, Cint,
                                (Ptr{Cvoid}, Csize_t, Cint), pointer(source.bytes), bytes, 1)
                            @assert r == 0
                        end
                    end
                end
                finalize(buffer)
            end
        end
    finally
        GC.gc()
        cleanup(d)
    end
end
