# Bounded native foundation probe, not the replacement Mantle implementation.
# No old Mantle import. Fixed host-coherent pool; Int32 vectors; one queue.
# Run via Julia in the workspace env: include(...); RecordedSlots.probe("NAVI31")
module RecordedSlots
import Vulkan as V
import Lava
import KernelInterface as KI
using Test

const GROUP = 64
const VALIDATION_ERRORS = Ref(0)
function debugcallback(severity, typ, data, user)
    severity == V.DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT && (VALIDATION_ERRORS[] += 1)
    if severity == V.DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT ||
       severity == V.DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
        V.default_debug_callback(severity, typ, data, user)
    end
    UInt32(0)
end
const CALLBACK = @cfunction(debugcallback, UInt32,
    (V.DebugUtilsMessageSeverityFlagEXT, V.DebugUtilsMessageTypeFlagEXT,
     Ptr{V.VkCore.VkDebugUtilsMessengerCallbackDataEXT}, Ptr{Cvoid}))

mutable struct Native
    instance::V.Instance
    messenger::V.DebugUtilsMessengerEXT
    device::V.Device
    queue::V.Queue
    commands::V.CommandPool
    timeline::V.Semaphore
    buffer::V.Buffer
    memory::V.DeviceMemory
    address::UInt64
    mapped::Ptr{UInt8}
    bytes::Int
    cursor::Int
    next::UInt64
    name::String
    lock::ReentrantLock
    poisoned::Bool
end
function Native(match; bytes=1<<20)
    inst = V.unwrap(V.create_instance(["VK_LAYER_KHRONOS_validation"], ["VK_EXT_debug_utils","VK_EXT_validation_features"];
        next=V.ValidationFeaturesEXT([V.VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT],V.ValidationFeatureDisableEXT[]),
        application_info=V.ApplicationInfo(v"0.1",v"0.1",v"1.3")))
    msg = V.DebugUtilsMessengerEXT(inst, CALLBACK)
    pd = only(filter(p -> occursin(match,V.get_physical_device_properties(p).device_name),
                     V.unwrap(V.enumerate_physical_devices(inst))))
    props = V.get_physical_device_properties(pd)
    fam = UInt32(findfirst(q -> (q.queue_flags & V.QUEUE_COMPUTE_BIT) != V.QueueFlag(0),
                          V.get_physical_device_queue_family_properties(pd))-1)
    feats = V.PhysicalDeviceFeatures2(V.PhysicalDeviceFeatures(:shader_int_64);
        next=V.PhysicalDeviceVulkan12Features(:buffer_device_address,:timeline_semaphore,
             :vulkan_memory_model,:vulkan_memory_model_device_scope,:scalar_block_layout;
             next=V.PhysicalDeviceVulkan13Features(:synchronization2,:maintenance4)))
    # VariablePointers is declared by this compiler even for the small kernels.
    feats11=V.PhysicalDeviceVulkan11Features(:variable_pointers,:variable_pointers_storage_buffer;next=feats.next)
    feats=V.PhysicalDeviceFeatures2(V.PhysicalDeviceFeatures(:shader_int_64);next=feats11)
    # Normalize the generated binding's empty device-layer pointer, without
    # masking the VUID or changing Vulkan.jl globally. Keep all conversion deps.
    low=V._DeviceCreateInfo(V.DeviceCreateInfo([V.DeviceQueueCreateInfo(fam,[1f0])],String[],String[];next=feats))
    raw=low.vks
    fixed=typeof(raw)((name===:ppEnabledLayerNames ? Ptr{Cstring}(C_NULL) : getfield(raw,name)
                      for name in fieldnames(typeof(raw)))...)
    dev=V.unwrap(V._create_device(pd,V._DeviceCreateInfo(fixed,low.deps)))
    queue = V.get_device_queue(dev,fam,0)
    commands = V.unwrap(V.create_command_pool(dev,fam;flags=V.COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT))
    timeline = V.unwrap(V.create_semaphore(dev;next=V.SemaphoreTypeCreateInfo(V.SEMAPHORE_TYPE_TIMELINE,0)))
    usage = V.BUFFER_USAGE_STORAGE_BUFFER_BIT | V.BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            V.BUFFER_USAGE_TRANSFER_SRC_BIT | V.BUFFER_USAGE_TRANSFER_DST_BIT | V.BUFFER_USAGE_INDIRECT_BUFFER_BIT
    buf = V.unwrap(V.create_buffer(dev,bytes,usage,V.SHARING_MODE_EXCLUSIVE,UInt32[]))
    req = V.get_buffer_memory_requirements(dev,buf)
    mp = V.get_physical_device_memory_properties(pd)
    want = V.MEMORY_PROPERTY_HOST_VISIBLE_BIT | V.MEMORY_PROPERTY_HOST_COHERENT_BIT
    mt = findfirst(i -> ((req.memory_type_bits >> (i-1)) & 1 == 1) &&
        (mp.memory_types[i].property_flags & want) == want,1:mp.memory_type_count)
    mt === nothing && error("No host-coherent memory fixture")
    mem = V.unwrap(V.allocate_memory(dev,req.size,mt-1;
        next=V.MemoryAllocateFlagsInfo(0;flags=V.MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT)))
    V.unwrap(V.bind_buffer_memory(dev,buf,mem,0))
    ptr = V.unwrap(V.map_memory(dev,mem,0,bytes))
    Native(inst,msg,dev,queue,commands,timeline,buf,mem,
        V.get_buffer_device_address(dev,V.BufferDeviceAddressInfo(buf)),Ptr{UInt8}(ptr),
        bytes,0,0,props.device_name,ReentrantLock(),false)
end
function region!(d::Native,n;align=16)
    off = cld(d.cursor,align)*align
    off+n <= d.bytes || error("fixed probe pool exhausted")
    d.cursor=off+n
    off
end
function put!(d::Native,off,x::T) where T
    unsafe_store!(Ptr{T}(d.mapped+off),x)
end
function waitvalue(d::Native,v=d.next)
    d.poisoned && error("poisoned native failure domain")
    v==0 && return
    V.unwrap(V.wait_semaphores(d.device,V.SemaphoreWaitInfo([d.timeline],[v]),typemax(UInt64)))
end
function command(f,d::Native)
    cb=only(V.unwrap(V.allocate_command_buffers(d.device,
        V.CommandBufferAllocateInfo(d.commands,V.COMMAND_BUFFER_LEVEL_PRIMARY,1))))
    V.unwrap(V.begin_command_buffer(cb,V.CommandBufferBeginInfo(;flags=V.COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT)))
    f(cb)
    V.unwrap(V.end_command_buffer(cb))
    cb
end
barrier(cb)=V.cmd_pipeline_barrier_2(cb,V.DependencyInfo([V.MemoryBarrier2(;
    src_stage_mask=V.PIPELINE_STAGE_2_ALL_COMMANDS_BIT,src_access_mask=V.ACCESS_2_MEMORY_WRITE_BIT,
    dst_stage_mask=V.PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
    dst_access_mask=V.ACCESS_2_MEMORY_READ_BIT | V.ACCESS_2_MEMORY_WRITE_BIT)],[],[]))
function submit!(d::Native,cbs;after=d.next)
    d.poisoned && error("poisoned native failure domain")
    # Construct all fallible descriptors before claim. Serialized queue identity.
    lock(d.lock) do
        value=d.next+1
        waits=after==0 ? V.SemaphoreSubmitInfo[] : [V.SemaphoreSubmitInfo(d.timeline,after,0;
            stage_mask=V.PIPELINE_STAGE_2_ALL_COMMANDS_BIT)]
        packet=[V.SubmitInfo2(waits,[V.CommandBufferSubmitInfo(cb,0) for cb in cbs],
            [V.SemaphoreSubmitInfo(d.timeline,value,0;stage_mask=V.PIPELINE_STAGE_2_ALL_COMMANDS_BIT)])]
        d.next=value
        try
            V.unwrap(V.queue_submit_2(d.queue,packet))
        catch
            d.poisoned=true
            rethrow()
        end
        value
    end
end
struct Shader
    code
    modulehandle::V.ShaderModule
    layout::V.PipelineLayout
    pipeline::V.Pipeline
end
function Shader(d,f,tt)
    code=Lava.lava_compile_gpu(f,tt;workgroup_size=(GROUP,1,1))
    words=collect(reinterpret(UInt32,code.spirv_bytes))
    sm=V.unwrap(V.create_shader_module(d.device,sizeof(words),words))
    layout=V.unwrap(V.create_pipeline_layout(d.device,[],[V.PushConstantRange(V.SHADER_STAGE_COMPUTE_BIT,0,8)]))
    info=V.ComputePipelineCreateInfo(V.PipelineShaderStageCreateInfo(V.SHADER_STAGE_COMPUTE_BIT,sm,code.entry_name),layout,-1)
    pipes,_=V.unwrap(V.create_compute_pipelines(d.device,[info]))
    Shader(code,sm,layout,only(pipes))
end
function launch(cb,s::Shader,addr,groups)
    V.cmd_bind_pipeline(cb,V.PIPELINE_BIND_POINT_COMPUTE,s.pipeline)
    pc=Ref(UInt64(addr))
    GC.@preserve pc V.cmd_push_constants(cb,s.layout,V.SHADER_STAGE_COMPUTE_BIT,0,8,Ptr{Cvoid}(Base.unsafe_convert(Ptr{UInt64},pc)))
    V.cmd_dispatch(cb,groups,1,1)
end
function arguments!(d,s,args)
    off=region!(d,s.code.push_info.arg_buffer_size)
    for (i,arg) in enumerate(args)
        @assert s.code.push_info.byval_llvm_sizes[i]==0
        put!(d,off+s.code.push_info.arg_offsets[i],arg)
    end
    off
end
# Header is address,length. Each map entry is descriptor destination, header,
# and indirect-command destination. The ordinary kernel never reads the header.
function gather!(mapping::Ptr{UInt64},n::Int32)
    i=Int(KI.get_global_id().x)
    i<=n || return nothing
    dst=Ptr{UInt64}(unsafe_load(mapping,3i-2))
    src=Ptr{UInt64}(unsafe_load(mapping,3i-1))
    cmd=Ptr{UInt64}(unsafe_load(mapping,3i))
    len=unsafe_load(src,2)
    unsafe_store!(dst,unsafe_load(src,1),1)
    unsafe_store!(dst,len,2)
    unsafe_store!(cmd,UInt64(cld(len,UInt64(GROUP))) | (UInt64(1)<<32),1)
    unsafe_store!(cmd,UInt64(1),2)
    nothing
end
function increment!(a,delta::Int32)
    i=Int(KI.get_global_id().x)
    i<=length(a) || return nothing
    @inbounds a[i]+=delta
    nothing
end
function lengthwriter!(cell::Ptr{UInt64},n::UInt64)
    KI.get_global_id().x==1 || return nothing
    unsafe_store!(cell,n,2)
    nothing
end
mutable struct ArrayState
    device::Native
    storage::Int
    cell::Int
    capacity::Int
    holds::Int
    last::UInt64
end
mutable struct MantleArray
    state::ArrayState
    creator::Bool
end
function MantleArray(d::Native,n)
    waitvalue(d)
    storage=region!(d,4n);cell=region!(d,16)
    for i in 1:n;put!(d,storage+4(i-1),Int32(0));end
    put!(d,cell,d.address+storage);put!(d,cell+8,UInt64(n))
    MantleArray(ArrayState(d,storage,cell,n,1,0),true)
end
mutable struct Graph
    device::Native
    states::Vector{ArrayState}
    nodes::Vector{Tuple{ArrayState,Int32}}
    work::V.CommandBuffer
    haswork::Bool
    records::Int
    last::UInt64
    retained::Vector{Any}
    live::Bool
end
# A command buffer is initialized by compile!, not a fabricated native handle.
Graph(d::Native)=Graph(d,ArrayState[],Tuple{ArrayState,Int32}[],command(_->nothing,d),false,0,0,Any[],true)
function dispatch!(g::Graph,a::MantleArray,delta::Int32)
    g.live && a.creator || throw(ArgumentError("released creator/graph"))
    a.state.device===g.device || throw(ArgumentError("mixed device arguments"))
    g.haswork && error("probe declarations are fixed after compile")
    if !(a.state in g.states);push!(g.states,a.state);a.state.holds+=1;end
    push!(g.nodes,(a.state,delta));g
end
function compile!(g::Graph;indirect=false)
    d=g.device;waitvalue(d)
    head=Shader(d,gather!,Tuple{Ptr{UInt64},Int32})
    body=Shader(d,increment!,Tuple{Lava.LavaDeviceArray{Int32,1},Int32})
    mapoff=region!(d,24length(g.nodes));args=Int[];cmds=Int[]
    for (i,(st,delta)) in enumerate(g.nodes)
        off=region!(d,body.code.push_info.arg_buffer_size+16)
        desc=cld(off+body.code.push_info.arg_buffer_size,16)*16
        d.cursor=max(d.cursor,desc+16)
        put!(d,off+body.code.push_info.arg_offsets[1],d.address+desc)
        put!(d,off+body.code.push_info.arg_offsets[2],delta)
        cmd=region!(d,16);push!(cmds,cmd);push!(args,off)
        put!(d,mapoff+24(i-1),d.address+desc)
        put!(d,mapoff+24(i-1)+8,d.address+st.cell)
        put!(d,mapoff+24(i-1)+16,d.address+cmd)
    end
    ha=arguments!(d,head,(Ptr{UInt64}(d.address+mapoff),Int32(length(g.nodes))))
    V.free_command_buffers(d.device,d.commands,[g.work])
    g.work=command(d) do cb
        launch(cb,head,d.address+ha,cld(length(g.nodes),GROUP));barrier(cb)
        for (i,(st,delta)) in enumerate(g.nodes)
            if indirect
                # Bind the body and push its immutable argument-block address.
                V.cmd_bind_pipeline(cb,V.PIPELINE_BIND_POINT_COMPUTE,body.pipeline)
                pc=Ref(d.address+args[i])
                GC.@preserve pc V.cmd_push_constants(cb,body.layout,V.SHADER_STAGE_COMPUTE_BIT,0,8,Ptr{Cvoid}(Base.unsafe_convert(Ptr{UInt64},pc)))
                V.cmd_dispatch_indirect(cb,d.buffer,cmds[i])
            else
                launch(cb,body,d.address+args[i],cld(st.capacity,GROUP))
            end
            barrier(cb)
        end
    end
    append!(g.retained,(head,body));g.haswork=true;g.records+=1;g
end
function run!(g::Graph;prefix=nothing,prepare=()->nothing)
    g.live || throw(ArgumentError("freed graph"))
    prepare() # explicit preclaim fault seam
    g.haswork || compile!(g)
    cbs=prefix===nothing ? [g.work] : [prefix,g.work]
    token=submit!(g.device,cbs;after=g.last)
    g.last=token
    foreach(st->st.last=token,g.states)
    token
end
function array(a::MantleArray)
    a.creator || throw(ArgumentError("released creator"))
    st=a.state;waitvalue(st.device,st.last)
    n=Int(unsafe_load(Ptr{UInt64}(st.device.mapped+st.cell+8)))
    n<=st.capacity || error("invalid GPU length")
    copy(unsafe_wrap(Vector{Int32},Ptr{Int32}(st.device.mapped+st.storage),n))
end
function free!(a::MantleArray)
    a.creator || return
    a.creator=false;a.state.holds-=1
end
function free!(g::Graph)
    g.live || return
    waitvalue(g.device,g.last) # probe-only blocking reclamation, outside queue lock
    V.free_command_buffers(g.device.device,g.device.commands,[g.work])
    foreach(st->st.holds-=1,g.states)
    g.live=false
end
function cleanup(d::Native)
    V.unwrap(V.device_wait_idle(d.device))
    V.unmap_memory(d.device,d.memory)
    finalize(d.buffer);finalize(d.memory);finalize(d.timeline);finalize(d.commands)
    finalize(d.device);finalize(d.messenger);finalize(d.instance)
end
function probe(match;indirect=false)
    before=VALIDATION_ERRORS[]
    d=Native(match)
    println("DEVICE ",d.name," mode=",indirect ? "indirect" : "capacity")
    graphs=Graph[]
    try
        @testset "recorded native slots: $(d.name), indirect=$indirect" begin
            a=MantleArray(d,256);g=Graph(d);push!(graphs,g)
            dispatch!(g,a,Int32(1));dispatch!(g,a,Int32(2))
            compile!(g;indirect)
            run!(g);@test array(a)==fill(Int32(3),256)
            run!(g);@test array(a)==fill(Int32(6),256)
            run!(g);run!(g) # two accepted replays, no intervening host wait
            @test array(a)==fill(Int32(12),256)
            @test g.records==1
            # Recorded body uses the cell snapshot even when the GPU changes length.
            writer=Shader(d,lengthwriter!,Tuple{Ptr{UInt64},UInt64})
            expected=fill(Int32(12),256)
            for n in (0,1,129,256)
                waitvalue(d)
                args=arguments!(d,writer,(Ptr{UInt64}(d.address+a.state.cell),UInt64(n)))
                pre=command(d) do cb
                    launch(cb,writer,d.address+args,1);barrier(cb)
                end
                run!(g;prefix=pre)
                result=array(a)
                @test length(result)==n
                expected[1:n] .+= Int32(3)
                @test result==expected[1:n]
                V.free_command_buffers(d.device,d.commands,[pre])
            end
            @test g.records==1
            # A queued move changes only cell address, preserving the recording.
            waitvalue(d);old=a.state.storage;new=region!(d,4a.state.capacity)
            pre=command(d) do cb
                V.cmd_copy_buffer(cb,d.buffer,d.buffer,[V.BufferCopy(old,new,4a.state.capacity)])
                newaddr=Ref(d.address+new)
                GC.@preserve newaddr V.cmd_update_buffer(cb,d.buffer,a.state.cell,8,Ptr{Cvoid}(Base.unsafe_convert(Ptr{UInt64},newaddr)))
                barrier(cb)
            end
            previous=array(a)
            run!(g;prefix=pre);a.state.storage=new
            @test array(a)==previous .+ Int32(3)
            V.free_command_buffers(d.device,d.commands,[pre])
            @test g.records==1
            # Preparation failure must not consume a value or poison the queue.
            last=d.next
            @test_throws ErrorException run!(g;prepare=()->error("preclaim fault"))
            @test d.next==last && !d.poisoned
            # Graph hold outlives creator ownership and GC.
            state=a.state;free!(a);GC.gc()
            @test state.holds==1
            @test_throws ArgumentError array(a)
            run!(g);waitvalue(d)
            @test unsafe_load(Ptr{Int32}(d.mapped+state.storage))==previous[1]+6
            free!(g);@test state.holds==0
            @test VALIDATION_ERRORS[]==before
            println("records=",g.records," timeline=",d.next," pool_used=",d.cursor," validation_errors=",VALIDATION_ERRORS[]-before)
        end
    finally
        foreach(g->g.live && free!(g),graphs)
        cleanup(d)
    end
    nothing
end
end
