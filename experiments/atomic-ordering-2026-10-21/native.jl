# Native visibility tests for the production Lava compiler; no Mantle import.
# Reuse the previous probe's Vulkan allocation/submission fixture; enable
# shaderInt8 as well because Bool loop temporaries declare the Int8 capability.
module AtomicVisibility
using Test, SHA, Dates
import Lava
import KernelInterface as KI
const fixture_path=joinpath(@__DIR__,"../feasibility-2026-10-21/recorded-slots.jl")
const fixture_source=replace(read(fixture_path,String),
    ":vulkan_memory_model,:vulkan_memory_model_device_scope,:scalar_block_layout" =>
    ":vulkan_memory_model,:vulkan_memory_model_device_scope,:shader_int_8,:scalar_block_layout")
include_string(@__MODULE__,fixture_source,fixture_path)
const N = RecordedSlots
const V = N.V
const UA = Lava.UnsafeAtomics

@inline pattern(i,j,epoch) = Int32(i*31 + j*197 + epoch*997)
function message(flags::Ptr{Int32},payload::Ptr{Int32},out::Ptr{Int32},epoch::Int32,::Val{mode}) where mode
    KI.get_local_id().x == 1 || return nothing
    group=Int32(KI.get_group_id().x)
    i=(group+Int32(1))÷Int32(2)
    flag=flags+4*(i-1)
    if isodd(group)
        for j in Int32(1):Int32(8)
            unsafe_store!(payload,pattern(i,j,epoch),8*(i-1)+j)
        end
        if mode === :fence
            UA.fence(UA.release,UA.device)
            UA.store!(flag,Int32(1),UA.monotonic,UA.device)
        elseif mode === :cas
            UA.cas!(flag,Int32(0),Int32(1),UA.release,UA.monotonic,UA.device)
        else
            UA.store!(flag,Int32(1),UA.release,UA.device)
        end
    else
        # Bounded polling: a consumer can finish even if its producer has not
        # been scheduled. No assumption of cross-workgroup forward progress.
        for attempt in Int32(1):Int32(4096)
            ready = if mode === :cas
                UA.cas!(flag,Int32(1),Int32(1),UA.acquire,UA.monotonic,UA.device).success
            elseif mode === :fence
                UA.load(flag,UA.monotonic,UA.device) == 1
            else
                UA.load(flag,UA.acquire,UA.device) == 1
            end
            if ready
                mode === :fence && UA.fence(UA.acquire,UA.device)
                ok=true
                for j in Int32(1):Int32(8)
                    ok &= unsafe_load(payload,8*(i-1)+j) == pattern(i,j,epoch)
                end
                unsafe_store!(out,ok ? Int32(1) : Int32(2),i)
                return nothing
            end
        end
        unsafe_store!(out,Int32(0),i)
    end
    nothing
end

# Every workgroup publishes a payload, then joins a release sequence. The
# last RMW's acquire must see all payloads. No spinning or dispatch barrier.
function last_arriver(counter::Ptr{Int32},payload::Ptr{Int32},out::Ptr{Int32},count::Int32,epoch::Int32)
    KI.get_local_id().x == 1 || return nothing
    i=Int32(KI.get_group_id().x)
    unsafe_store!(payload,pattern(i,Int32(1),epoch),i)
    old=first(UA.modify!(counter,+,Int32(1),UA.acq_rel,UA.device))
    if old == count-1
        ok=true
        for j in Int32(1):count
            ok &= unsafe_load(payload,j) == pattern(j,Int32(1),epoch)
        end
        unsafe_store!(out,ok ? Int32(1) : Int32(2))
    end
    nothing
end

function gpu_storage(d)
    pd=only(filter(p->V.get_physical_device_properties(p).device_name==d.name,
        V.unwrap(V.enumerate_physical_devices(d.instance))))
    buffer=V.unwrap(V.create_buffer(d.device,d.bytes,
        V.BUFFER_USAGE_STORAGE_BUFFER_BIT|V.BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT|
        V.BUFFER_USAGE_TRANSFER_SRC_BIT|V.BUFFER_USAGE_TRANSFER_DST_BIT,
        V.SHARING_MODE_EXCLUSIVE,UInt32[]))
    req=V.get_buffer_memory_requirements(d.device,buffer)
    properties=V.get_physical_device_memory_properties(pd)
    mt=findfirst(i->((req.memory_type_bits>>(i-1))&1==1) &&
        (properties.memory_types[i].property_flags & V.MEMORY_PROPERTY_DEVICE_LOCAL_BIT)!=V.MemoryPropertyFlag(0),
        1:properties.memory_type_count)
    mt===nothing && error("No device-local memory for native visibility test")
    memory=V.unwrap(V.allocate_memory(d.device,req.size,mt-1;
        next=V.MemoryAllocateFlagsInfo(0;flags=V.MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT)))
    V.unwrap(V.bind_buffer_memory(d.device,buffer,memory,0))
    address=V.get_buffer_device_address(d.device,V.BufferDeviceAddressInfo(buffer))
    println("STORAGE ",d.name," type=",mt-1," flags=",properties.memory_types[mt].property_flags)
    (; buffer,memory,address)
end

function probe(match;count=257,repeats=32)
    before=N.VALIDATION_ERRORS[]
    d=N.Native(match)
    shaders=N.Shader[]
    gpu=gpu_storage(d)
    try
        @testset "atomic visibility: $(d.name)" begin
            flags=N.region!(d,4count); payload=N.region!(d,32count); output=N.region!(d,4count)
            fp=Ptr{Int32}(gpu.address+flags); pp=Ptr{Int32}(gpu.address+payload); op=Ptr{Int32}(gpu.address+output)
            for mode in (:loadstore,:fence,:cas,:release_sequence)
                f,tt = if mode === :release_sequence
                    (last_arriver,Tuple{Ptr{Int32},Ptr{Int32},Ptr{Int32},Int32,Int32})
                else
                    (message,Tuple{Ptr{Int32},Ptr{Int32},Ptr{Int32},Int32,Val{mode}})
                end
                shader=N.Shader(d,f,tt); push!(shaders,shader)
                # Inspect the executed, validated module as well as its output.
                asm=Lava.disassemble_spirv(shader.code.spirv_bytes)
                @test occursin("VulkanMemoryModelDeviceScope",asm)
                @test occursin(mode === :fence ? "OpMemoryBarrier" :
                    mode === :cas ? "OpAtomicCompareExchange" :
                    mode === :release_sequence ? "OpAtomicIAdd" : "OpAtomicLoad",asm)
                write(joinpath(@__DIR__,"executed-$(mode).spvasm"),asm)
                observed=0
                for epoch in Int32(1):Int32(repeats)
                    N.waitvalue(d)
                    args= mode === :release_sequence ? (fp,pp,op,Int32(count),epoch) : (fp,pp,op,epoch)
                    offset=N.arguments!(d,shader,args)
                    cb=N.command(d) do cb
                        V.cmd_fill_buffer(cb,gpu.buffer,flags,4count,UInt32(0))
                        V.cmd_fill_buffer(cb,gpu.buffer,payload,32count,typemax(UInt32))
                        V.cmd_fill_buffer(cb,gpu.buffer,output,4count,typemax(UInt32))
                        N.barrier(cb)
                        N.launch(cb,shader,d.address+offset,mode === :release_sequence ? count : 2count)
                        N.barrier(cb)
                        for (off,size) in ((flags,4count),(payload,32count),(output,4count))
                            V.cmd_copy_buffer(cb,gpu.buffer,d.buffer,[V.BufferCopy(off,off,size)])
                        end
                        N.barrier(cb)
                    end
                    N.submit!(d,[cb]); N.waitvalue(d)
                    result=copy(unsafe_wrap(Vector{Int32},Ptr{Int32}(d.mapped+output),mode === :release_sequence ? 1 : count))
                    @test all(x->x in (0,1),result)
                    if mode === :release_sequence
                        @test only(result)==1
                        @test unsafe_load(Ptr{Int32}(d.mapped+flags))==count
                    else
                        observed+=Base.count(==(1),result)
                        @test all(==(1),unsafe_wrap(Vector{Int32},Ptr{Int32}(d.mapped+flags),count))
                    end
                    @test all(i->unsafe_load(Ptr{Int32}(d.mapped+payload),i)==
                        pattern(mode === :release_sequence ? Int32(i) : Int32(cld(i,8)),
                                mode === :release_sequence ? Int32(1) : Int32(mod1(i,8)),epoch),
                        1:(mode === :release_sequence ? count : 8count))
                    V.free_command_buffers(d.device,d.commands,[cb])
                end
                mode !== :release_sequence && (@test observed>0)
                println("DEVICE ",d.name," mode=",mode," dispatches=",repeats,
                    " observed_messages=",mode === :release_sequence ? count*repeats : observed,
                    " validation_errors=",N.VALIDATION_ERRORS[]-before)
            end
            @test N.VALIDATION_ERRORS[]==before
        end
    finally
        N.waitvalue(d)
        for s in shaders
            finalize(s.pipeline); finalize(s.layout); finalize(s.modulehandle)
        end
        finalize(gpu.buffer);finalize(gpu.memory)
        N.cleanup(d)
    end
    @test N.VALIDATION_ERRORS[]==before
    nothing
end

function run()
    path=joinpath(@__DIR__,"native-results.txt")
    open(path,"w") do io
        redirect_stdout(io) do
            redirect_stderr(io) do
                println("Atomic ordering native qualification ",now()," Julia=",VERSION)
                println("Mantle loaded=",any(m->nameof(m)==:Mantle,values(Base.loaded_modules)))
                @testset "F33 native visibility" begin
                    for match in ("NAVI31","RTX 4000","RAPHAEL","llvmpipe")
                        probe(match)
                    end
                    GC.gc()
                    @test N.VALIDATION_ERRORS[]==0
                end
                for file in ("native.jl","../feasibility-2026-10-21/recorded-slots.jl",
                    "../../../Lava/src/compiler/spirv/emit.jl","../../../Lava/src/compiler/spirv/module.jl","../../../Lava/src/compiler/spirv/types.jl",
                    "../../src/vulkan/runtime/device.jl",
                    "../../../Lava/src/compiler/target.jl","../../../Lava/src/device/atomics.jl")
                    println("SHA256 ",bytes2hex(sha256(read(joinpath(@__DIR__,file))))," ",file)
                end
            end
        end
    end
    println("Native results saved: ",path)
    nothing
end
end
