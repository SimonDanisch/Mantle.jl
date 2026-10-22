struct H264PictureOrder
    frame::Int
    offset::Int
end

"""Decoder state shared by the codecs: session, bitstream buffer, DPB slots and the
display-order queue. `Codec` is the codec's picture-order state, `Entry` a DPB entry."""
mutable struct CodecDecoder{Codec,Entry} <: VideoDecoder
    ctx::MANTLE.VkContext
    profile::Tuple
    sps::Any
    pps::Any
    chroma::Bool
    SESSION::Any
    PARAMS::Any
    PIN::Vector{Any}
    imgs::Vector{Any}
    nslots::Int
    outimgs::Vector{Any}
    mkoutimg::Any
    copyq::Vector{Any}
    bsalign::Int
    bbuf::Any
    bmem::Any
    bmap::Ptr{UInt8}
    bufsz::UInt64
    cbh::Any
    CW::Int
    CH::Int
    DW::Int
    DH::Int
    aus::Vector{Vector{Vector{UInt8}}}
    nextau::Int
    dpb::Vector{Entry}
    freeslots::Vector{Int}
    prevmsb::Int
    prevlsb::Int
    maxpoclsb::Int
    gop::Int
    decoded::Int
    isfirst::Bool
    pending::Vector{Tuple{Int,Int,Any,Any}}
    open::Bool
    codec::Codec
end

const H264Decoder = CodecDecoder{H264PictureOrder,NTuple{3,Int}}
const H265Decoder = CodecDecoder{Nothing,NTuple{2,Int}}

decodeprofile(dec::CodecDecoder) = dec.profile
memtype(ctx, bits, flags) = MANTLE.find_memory_type(ctx, bits, Vk.MemoryPropertyFlag(UInt32(flags)))
videoheader() = C.VkExtensionProperties(ntuple(_ -> Cchar(0), 256), UInt32(0))

function videocaps(ctx, profile, codec)
    hp, pr, pl = profile
    hc = Ref(codec)
    dc = Ref(C.VkVideoDecodeCapabilitiesKHR(C.VK_STRUCTURE_TYPE_VIDEO_DECODE_CAPABILITIES_KHR, pc(hc), UInt32(0)))
    e = C.VkExtent2D(0, 0)
    cp = Ref(C.VkVideoCapabilitiesKHR(C.VK_STRUCTURE_TYPE_VIDEO_CAPABILITIES_KHR, pc(dc), UInt32(0),
        UInt64(0), UInt64(0), e, e, e, UInt32(0), UInt32(0), videoheader()))
    GC.@preserve hp pr pl hc dc cp vkchk(ccall(Vk.function_pointer(ctx.instance,
        "vkGetPhysicalDeviceVideoCapabilitiesKHR"), Int32, (Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid}),
        ctx.physical_device.vks, pc(pr), pc(cp)), "vkGetPhysicalDeviceVideoCapabilitiesKHR")
    c = cp[]
    return (flags=dc[].flags, header=c.stdHeaderVersion,
            offset=Int(c.minBitstreamBufferOffsetAlignment), size=Int(c.minBitstreamBufferSizeAlignment),
            slots=Int(c.maxDpbSlots), refs=Int(c.maxActiveReferencePictures),
            mincoded=(Int(c.minCodedExtent.width), Int(c.minCodedExtent.height)),
            maxcoded=(Int(c.maxCodedExtent.width), Int(c.maxCodedExtent.height)))
end

function videosession(ctx, profile, caps, coded, slots, refs, pins)
    hdr = Ref(caps.header)
    fmt = C.VkFormat(1000156003) # NV12
    ci = Ref(C.VkVideoSessionCreateInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_SESSION_CREATE_INFO_KHR,
        C_NULL, UInt32(ctx.video_decode_queue_family_index), UInt32(0), rp(profile[2]), fmt,
        C.VkExtent2D(coded...), fmt, UInt32(slots), UInt32(refs), rp(hdr)))
    out = Ref{C.VkVideoSessionKHR}()
    GC.@preserve pins profile hdr ci vkchk(ccall(dfp(ctx,"vkCreateVideoSessionKHR"), Int32,
        (Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid}), ctx.device.vks, pc(ci), C_NULL, rp(out)),
        "vkCreateVideoSessionKHR")
    session = out[]
    try
        count = Ref(UInt32(0))
        vkchk(ccall(dfp(ctx,"vkGetVideoSessionMemoryRequirementsKHR"), Int32,
            (Ptr{Cvoid},C.VkVideoSessionKHR,Ptr{UInt32},Ptr{Cvoid}), ctx.device.vks, session,
            count, C_NULL), "vkGetVideoSessionMemoryRequirementsKHR")
        reqs = [C.VkVideoSessionMemoryRequirementsKHR(C.VK_STRUCTURE_TYPE_VIDEO_SESSION_MEMORY_REQUIREMENTS_KHR,
            C_NULL, UInt32(0), C.VkMemoryRequirements(UInt64(0),UInt64(0),UInt32(0))) for _ in 1:count[]]
        GC.@preserve reqs vkchk(ccall(dfp(ctx,"vkGetVideoSessionMemoryRequirementsKHR"), Int32,
            (Ptr{Cvoid},C.VkVideoSessionKHR,Ptr{UInt32},Ptr{Cvoid}), ctx.device.vks, session,
            count, pointer(reqs)), "vkGetVideoSessionMemoryRequirementsKHR")
        binds = C.VkBindVideoSessionMemoryInfoKHR[]
        for req in reqs
            mr = req.memoryRequirements
            # Session requirements may mandate host memory; local is a preference.
            type = MANTLE.find_memory_type(ctx, mr.memoryTypeBits, Vk.MemoryPropertyFlag(0);
                                          preferred_flags=Vk.MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
            mem = Vk.unwrap(Vk.allocate_memory(ctx.device, mr.size, type))
            push!(pins, mem)
            push!(binds, C.VkBindVideoSessionMemoryInfoKHR(C.VK_STRUCTURE_TYPE_BIND_VIDEO_SESSION_MEMORY_INFO_KHR,
                C_NULL, req.memoryBindIndex, mem.vks, UInt64(0), mr.size))
        end
        GC.@preserve binds pins vkchk(ccall(dfp(ctx,"vkBindVideoSessionMemoryKHR"), Int32,
            (Ptr{Cvoid},C.VkVideoSessionKHR,UInt32,Ptr{Cvoid}), ctx.device.vks, session,
            UInt32(length(binds)), pointer(binds)), "vkBindVideoSessionMemoryKHR")
        return session
    catch
        ccall(dfp(ctx,"vkDestroyVideoSessionKHR"), Cvoid,
              (Ptr{Cvoid},C.VkVideoSessionKHR,Ptr{Cvoid}), ctx.device.vks, session, C_NULL)
        rethrow()
    end
end

function videoparams(ctx, session, codecinfo, pins)
    ci = Ref(C.VkVideoSessionParametersCreateInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_SESSION_PARAMETERS_CREATE_INFO_KHR,
        pc(codecinfo), UInt32(0), C_NULL, session))
    out = Ref{C.VkVideoSessionParametersKHR}()
    GC.@preserve pins codecinfo ci vkchk(ccall(dfp(ctx,"vkCreateVideoSessionParametersKHR"), Int32,
        (Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid}), ctx.device.vks, pc(ci), C_NULL, rp(out)),
        "vkCreateVideoSessionParametersKHR")
    return out[]
end

function videoimage(ctx, pins, profile, coded, display, usage; families=UInt32[])
    fmt = Vk.Format(1000156003)
    image = GC.@preserve pins profile Vk.Image(ctx.device, Vk.IMAGE_TYPE_2D, fmt,
        Vk.Extent3D(coded..., 1), 1, 1, Vk.SAMPLE_COUNT_1_BIT, Vk.IMAGE_TILING_OPTIMAL, usage,
        isempty(families) ? Vk.SHARING_MODE_EXCLUSIVE : Vk.SHARING_MODE_CONCURRENT,
        families, Vk.IMAGE_LAYOUT_UNDEFINED; next=pc(profile[3]))
    mem = MANTLE.alloc_image_memory(ctx, image)
    view = Vk.ImageView(ctx.device, image, Vk.IMAGE_VIEW_TYPE_2D, fmt,
        Vk.ComponentMapping(Vk.COMPONENT_SWIZZLE_IDENTITY,Vk.COMPONENT_SWIZZLE_IDENTITY,
                            Vk.COMPONENT_SWIZZLE_IDENTITY,Vk.COMPONENT_SWIZZLE_IDENTITY),
        Vk.ImageSubresourceRange(Vk.IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1))
    push!(pins, image, mem, view)
    return VideoImage(image, mem, coded..., display..., fmt,
                      Ref(UInt32(C.VK_IMAGE_LAYOUT_UNDEFINED))), view
end

function videodecoder(::Type{CodecDecoder{S,E}}, ctx, data; chroma=false) where {S,E}
    ctx.video_decode_available || error("device has no video decode support")
    D = CodecDecoder{S,E}
    cfg = decoderconfig(D, data)
    profile = decoderprofile(D)
    caps = videocaps(ctx, profile, codeccapabilities(D))
    all(i -> caps.mincoded[i] <= cfg.coded[i] <= caps.maxcoded[i], 1:2) || throw(ArgumentError(
        "video coded extent $(cfg.coded) is outside $(caps.mincoded)–$(caps.maxcoded)"))
    distinct = distinctoutput(D, caps.flags)
    pins = Any[]
    slots = cfg.refs + 2
    dec = D(ctx, profile, cfg.sps, cfg.pps, chroma, nothing, nothing, pins, Any[], slots,
        Any[], nothing, Any[], bitstreamalignment(caps.offset,caps.size), nothing, nothing,
        Ptr{UInt8}(0), UInt64(0), nothing, cfg.coded..., cfg.display...,
        Vector{Vector{Vector{UInt8}}}(), 1, E[], collect(1:slots), 0, 0,
        1 << (cfg.sps.log2poc + 4), 0, 0, true, Tuple{Int,Int,Any,Any}[], true, cfg.state)
    try
        dec.SESSION = videosession(ctx, profile, caps, cfg.coded, slots, cfg.refs, pins)
        dec.PARAMS = videoparams(ctx, dec.SESSION, codecparameters(dec), pins)
        make(usage; families=UInt32[]) = videoimage(ctx, pins, profile, cfg.coded,
                                                  cfg.display, usage; families)
        usage = distinct ? Vk.IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR :
            Vk.IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR | Vk.IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR |
            Vk.IMAGE_USAGE_TRANSFER_SRC_BIT
        append!(dec.imgs, (make(usage) for _ in 1:slots))
        if distinct
            families = unique(UInt32[ctx.queue_family_index, ctx.video_decode_queue_family_index])
            dec.mkoutimg = () -> make(Vk.IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR | Vk.IMAGE_USAGE_TRANSFER_SRC_BIT;
                                     families=length(families)>1 ? families : UInt32[])
        end
        pool = Vk.unwrap(Vk.create_command_pool(ctx.device, ctx.video_decode_queue_family_index;
                         flags=Vk.COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT))
        dec.cbh = first(Vk.unwrap(Vk.allocate_command_buffers(ctx.device,
            Vk.CommandBufferAllocateInfo(pool, Vk.COMMAND_BUFFER_LEVEL_PRIMARY, 1))))
        push!(pins, dec.cbh)
        return dec
    catch original
        try
            close(dec)
        catch cleanup
            throw(CompositeException(Any[original, cleanup]))
        end
        rethrow()
    end
end

function feed!(dec::CodecDecoder, annexb::AbstractVector{UInt8})
    dec.open || throw(ArgumentError("decoder is closed"))
    aus = Vector{Vector{Vector{UInt8}}}()
    largest = current = 0
    for (first, nal) in videoslices(dec, annexb)
        if first
            push!(aus, Vector{UInt8}[])
            current = 0
        end
        push!(aus[end], nal)
        current += length(nal) + 3
        largest = max(largest, current)
    end
    ensurebitbuf!(dec, largest)
    dec.aus = aus
    dec.nextau = 1
    return dec
end

function decodevideo(D, ctx, data; maxframes::Int=typemax(Int), chroma::Bool=false)
    maxframes >= 0 || throw(ArgumentError("maxframes must be nonnegative"))
    dec = D(ctx, data; chroma)
    try
        feed!(dec, data)
        length(dec.aus) > maxframes && resize!(dec.aus, maxframes)
        out = decodemore!(dec, typemax(Int); holdback=typemax(Int))
        ys = LavaArray{UInt8,2}[y for (y,_) in out]
        uvs = chroma ? LavaArray{UInt8,2}[uv for (_,uv) in out] : LavaArray{UInt8,2}[]
        return dec.DW, dec.DH, ys, uvs
    finally
        close(dec)
    end
end

function picturemsb(dec::CodecDecoder, lsb::Integer)
    if lsb < dec.prevlsb && dec.prevlsb-lsb >= dec.maxpoclsb ÷ 2
        return dec.prevmsb + dec.maxpoclsb
    elseif lsb > dec.prevlsb && lsb-dec.prevlsb > dec.maxpoclsb ÷ 2
        return dec.prevmsb - dec.maxpoclsb
    end
    return dec.prevmsb
end

function videobarrier!(ctx, cmd; memory=nothing, image=nothing)
    stage = UInt32(C.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT)
    GC.@preserve memory image ccall(dfp(ctx,"vkCmdPipelineBarrier"), Cvoid,
        (C.VkCommandBuffer,UInt32,UInt32,UInt32,UInt32,Ptr{Cvoid},UInt32,Ptr{Cvoid},UInt32,Ptr{Cvoid}),
        cmd, stage, stage, UInt32(0), UInt32(memory!==nothing), memory===nothing ? C_NULL : pc(memory),
        UInt32(0), C_NULL, UInt32(image!==nothing), image===nothing ? C_NULL : pc(image))
    return nothing
end

function videolayoutbarrier(image, old, new)
    return Ref(C.VkImageMemoryBarrier(C.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, C_NULL,
        UInt32(C.VK_ACCESS_MEMORY_WRITE_BIT), UInt32(C.VK_ACCESS_MEMORY_READ_BIT)|UInt32(C.VK_ACCESS_MEMORY_WRITE_BIT),
        C.VkImageLayout(old), C.VkImageLayout(new), UInt32(0xffffffff), UInt32(0xffffffff), image,
        C.VkImageSubresourceRange(UInt32(C.VK_IMAGE_ASPECT_COLOR_BIT),UInt32(0),UInt32(1),UInt32(0),UInt32(1))))
end

picture_resource(view, width, height) = C.VkVideoPictureResourceInfoKHR(
    C.VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR, C_NULL, C.VkOffset2D(0,0),
    C.VkExtent2D(width,height), UInt32(0), view)

"""Write each slice of `au` with a 3-byte Annex-B start code to `dst`. Return the
slice offsets and the total size rounded up to `alignment`."""
function uploadbitstream!(dst::Ptr{UInt8}, au, alignment::Integer)
    offsets = Vector{UInt32}(undef, length(au))
    off = 0
    for (i, slice) in enumerate(au)
        offsets[i] = UInt32(off)
        unsafe_store!(dst + off, UInt8(0))
        unsafe_store!(dst + off + 1, UInt8(0))
        unsafe_store!(dst + off + 2, UInt8(1))
        GC.@preserve slice unsafe_copyto!(dst + off + 3, pointer(slice), length(slice))
        off += length(slice) + 3
    end
    return offsets, cld(off, alignment) * alignment
end

"""Decode targets per video submission under DPB_AND_OUTPUT_DISTINCT; bounds their memory."""
const MAX_DECODE_TARGETS = 16

"Frames not yet returned by [`decodemore!`](@ref) (undecoded AUs + pending reorder)."
remaining(dec::CodecDecoder) = (length(dec.aus) - dec.nextau + 1) + length(dec.pending)

"""Decode up to `maxframes` fed AUs and return `(Y, UV)` pairs in (GOP, POC) order,
holding back the last `holdback` frames until the feed is decoded. `UV` is `nothing`
without `chroma`. Each submission completes before its resources are reused."""
function decodemore!(dec::CodecDecoder, maxframes::Integer; holdback::Integer = dec.nslots)
    dec.open || throw(ArgumentError("decoder is closed"))
    hi = min(dec.nextau + Int(min(maxframes, typemax(Int) ÷ 2)) - 1, length(dec.aus))
    if dec.nextau <= hi
        batch = dec.nextau:hi
        ausize(au) = cld(sum(length(sl) + 3 for sl in au), dec.bsalign) * dec.bsalign
        w = dec.ctx; CB = dec.cbh.vks
        # COINCIDE copies each DPB slot in the decode submission. DISTINCT keeps one
        # target per AU until the main-queue copies finish, so it submits in sub-chunks.
        step = dec.mkoutimg === nothing ? length(batch) : min(length(batch), MAX_DECODE_TARGETS)
        for sub in Iterators.partition(batch, max(step, 1))
            ensurebitbuf!(dec, sum(ausize(dec.aus[i]) for i in sub))
            bi = Ref(C.VkCommandBufferBeginInfo(C.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, C_NULL,
                                                UInt32(C.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT), Ptr{Cvoid}(C_NULL)))
            GC.@preserve bi vkchk(ccall(dfp(w,"vkBeginCommandBuffer"),Int32,(C.VkCommandBuffer,Ptr{Cvoid}),CB,pc(bi)), "vkBeginCommandBuffer")
            base = 0
            for (j, i) in enumerate(sub)
                base += decodeau!(dec, dec.aus[i], base, j)
            end
            vkchk(ccall(dfp(w,"vkEndCommandBuffer"),Int32,(C.VkCommandBuffer,),CB), "vkEndCommandBuffer")
            cbref = Ref(CB)
            si = Ref(C.VkSubmitInfo(C.VK_STRUCTURE_TYPE_SUBMIT_INFO,C_NULL,UInt32(0),Ptr{C.VkSemaphore}(C_NULL),Ptr{UInt32}(C_NULL),UInt32(1),Base.unsafe_convert(Ptr{C.VkCommandBuffer},cbref),UInt32(0),Ptr{C.VkSemaphore}(C_NULL)))
            GC.@preserve si cbref vkchk(ccall(dfp(w,"vkQueueSubmit"),Int32,(Ptr{Cvoid},UInt32,Ptr{Cvoid},Ptr{Cvoid}),w.video_decode_queue.vks,UInt32(1),Base.unsafe_convert(Ptr{Cvoid},si),C_NULL), "vkQueueSubmit")
            vkchk(ccall(dfp(w,"vkQueueWaitIdle"),Int32,(Ptr{Cvoid},),w.video_decode_queue.vks), "vkQueueWaitIdle")
            # Waits for the copies, freeing the decode targets for the next sub-chunk.
            drain_copies!(dec)
        end
        dec.nextau = hi + 1
    end
    sort!(dec.pending, by = x -> (x[1], x[2]))   # display order = (GOP, POC)
    nkeep = dec.nextau <= length(dec.aus) ? min(holdback, typemax(Int)) : 0
    nemit = max(length(dec.pending) - nkeep, 0)
    out = [(y, uv) for (_, _, y, uv) in dec.pending[1:nemit]]
    deleteat!(dec.pending, 1:nemit)
    return out
end

"""Copy the queued DISTINCT decode targets on the main queue and wait. Call after
the video submission completed; the targets are shared across both queue families."""
function drain_copies!(dec::CodecDecoder)
    isempty(dec.copyq) && return nothing
    ctx = dec.ctx
    tok = oneshot!(ctx.default_bq; tag = :video) do e
        cb = e.cmd
        for (vimg, y, uv) in dec.copyq
            yb = y.buf[]
            record_luma_copy!(cb, vimg, yb.buffer, pool_offset(yb) + y.offset)
            if uv !== nothing
                ub = uv.buf[]
                record_chroma_copy!(cb, vimg, ub.buffer, pool_offset(ub) + uv.offset)
            end
        end
    end
    waitfor!(ctx.default_bq, tok)
    empty!(dec.copyq)
    return nothing
end

"Grow the mapped bitstream buffer to at least `need` bytes, rounded up to `bsalign`."
function ensurebitbuf!(dec::CodecDecoder, need::Integer)
    dec.open || throw(ArgumentError("decoder is closed"))
    size = UInt64(cld(max(need,1),dec.bsalign)*dec.bsalign)
    size <= dec.bufsz && return nothing
    ctx = dec.ctx
    profile = decodeprofile(dec)
    ci = Ref(C.VkBufferCreateInfo(C.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, pc(profile[3]), UInt32(0),
        size, UInt32(C.VK_BUFFER_USAGE_VIDEO_DECODE_SRC_BIT_KHR), C.VK_SHARING_MODE_EXCLUSIVE,
        UInt32(0), Ptr{UInt32}(C_NULL)))
    out = Ref{C.VkBuffer}()
    GC.@preserve ci profile vkchk(ccall(dfp(ctx,"vkCreateBuffer"), Int32,
        (Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid}), ctx.device.vks, pc(ci), C_NULL, rp(out)),
        "vkCreateBuffer")
    buffer = out[]
    memory = nothing
    mapped = Ref{Ptr{Cvoid}}()
    try
        req = Ref(C.VkMemoryRequirements(UInt64(0),UInt64(0),UInt32(0)))
        ccall(dfp(ctx,"vkGetBufferMemoryRequirements"), Cvoid,
              (Ptr{Cvoid},C.VkBuffer,Ptr{Cvoid}), ctx.device.vks, buffer, rp(req))
        flags = UInt32(C.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)|UInt32(C.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        memory = Vk.unwrap(Vk.allocate_memory(ctx.device,req[].size,memtype(ctx,req[].memoryTypeBits,flags)))
        vkchk(ccall(dfp(ctx,"vkBindBufferMemory"), Int32,
            (Ptr{Cvoid},C.VkBuffer,C.VkDeviceMemory,UInt64), ctx.device.vks, buffer, memory.vks,
            UInt64(0)), "vkBindBufferMemory")
        vkchk(ccall(dfp(ctx,"vkMapMemory"), Int32,
            (Ptr{Cvoid},C.VkDeviceMemory,UInt64,UInt64,UInt32,Ptr{Cvoid}), ctx.device.vks,
            memory.vks, UInt64(0), size, UInt32(0), pc(mapped)), "vkMapMemory")
    catch
        ccall(dfp(ctx,"vkDestroyBuffer"), Cvoid,
              (Ptr{Cvoid},C.VkBuffer,Ptr{Cvoid}), ctx.device.vks, buffer, C_NULL)
        rethrow()
    end
    # No submission reads the old buffer here. Replace it only once the new one is
    # mapped, so a failed allocation leaves the old one usable.
    if dec.bbuf !== nothing
        ccall(dfp(ctx,"vkUnmapMemory"), Cvoid, (Ptr{Cvoid},C.VkDeviceMemory),ctx.device.vks,dec.bmem.vks)
        ccall(dfp(ctx,"vkDestroyBuffer"), Cvoid,
              (Ptr{Cvoid},C.VkBuffer,Ptr{Cvoid}),ctx.device.vks,dec.bbuf,C_NULL)
    end
    dec.bbuf, dec.bmem, dec.bmap, dec.bufsz = buffer, memory, Ptr{UInt8}(mapped[]), size
    return nothing
end

function Base.close(dec::CodecDecoder)
    dec.open || return nothing
    ctx = dec.ctx
    vkchk(ccall(dfp(ctx,"vkQueueWaitIdle"),Int32,(Ptr{Cvoid},),ctx.video_decode_queue.vks),"vkQueueWaitIdle")
    if !isempty(dec.copyq)
        vkchk(ccall(dfp(ctx,"vkQueueWaitIdle"),Int32,(Ptr{Cvoid},),MANTLE.vkqueue(ctx.default_bq).vks),"vkQueueWaitIdle")
    end
    dec.open = false
    dec.PARAMS === nothing || ccall(dfp(ctx,"vkDestroyVideoSessionParametersKHR"),Cvoid,
        (Ptr{Cvoid},C.VkVideoSessionParametersKHR,Ptr{Cvoid}),ctx.device.vks,dec.PARAMS,C_NULL)
    dec.SESSION === nothing || ccall(dfp(ctx,"vkDestroyVideoSessionKHR"),Cvoid,
        (Ptr{Cvoid},C.VkVideoSessionKHR,Ptr{Cvoid}),ctx.device.vks,dec.SESSION,C_NULL)
    if dec.bbuf !== nothing
        ccall(dfp(ctx,"vkUnmapMemory"),Cvoid,(Ptr{Cvoid},C.VkDeviceMemory),ctx.device.vks,dec.bmem.vks)
        ccall(dfp(ctx,"vkDestroyBuffer"),Cvoid,(Ptr{Cvoid},C.VkBuffer,Ptr{Cvoid}),ctx.device.vks,dec.bbuf,C_NULL)
    end
    dec.PARAMS = dec.SESSION = dec.bbuf = dec.bmem = dec.cbh = dec.mkoutimg = nothing
    dec.bmap = Ptr{UInt8}(0)
    dec.bufsz = 0
    dec.nextau = 1
    for values in (dec.PIN,dec.imgs,dec.outimgs,dec.copyq,dec.pending,dec.aus)
        empty!(values)
    end
    return nothing
end

