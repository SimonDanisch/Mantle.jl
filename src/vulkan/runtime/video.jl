# Hardware H.264 decode on the GPU via VK_KHR_video_decode_queue.
# Input: an H.264 Annex-B elementary stream (SPS/PPS/slice NAL units).
# Output: luma (Y) planes in display order. Everything (session, DPB reference
# management, POC ordering) is driven through VulkanCore's raw video-decode FFI —
# VK.jl's high-level wrapper does not generate the *video codec* API. A
# minimal exp-Golomb parser reads the SPS/PPS and per-slice headers to build the
# StdVideoH264* structs and manage the decoded-picture buffer.
#
# The decoded *frames*, though, are ordinary Vulkan images: the decoder produces
# a [`VideoImage`] per frame (a real `VK.Image`), and moving a frame's luma
# onto a `LavaArray` is a normal `copyto!(dst, img)` through the high-level
# `cmd_copy_image_to_buffer` — no raw FFI on the resource/transfer side.

# ============================================================================
# VideoImage — a decoded video frame resident in a GPU image
# ============================================================================
"""
    VideoImage

One hardware-decoded video frame, resident in a device-local multi-planar
(NV12) `VK.Image`. It is a first-class Lava image handle:

  * `copyto!(dst::LavaArray{UInt8}, img)` copies the luma (Y) plane into a
    device-local array (image→buffer, entirely on the GPU — no host round-trip).
  * `Array(img)` downloads the luma plane to a host `Matrix{UInt8}`.
  * `size(img)` is the display (cropped) size; `eltype(img) == UInt8`.

Produced by the hardware H.264 decoder — see [`decode_h264_gpu`](@ref).
"""
mutable struct VideoImage
    image::VK.Image
    memory::VK.DeviceMemory
    codedwidth::Int
    codedheight::Int
    width::Int      # display (cropped) width
    height::Int     # display (cropped) height
    format::VK.Format
    # Current VkImageLayout as a raw UInt32 so the decode loop's in-place
    # barriers and the high-level copy path share one source of truth.
    layout::Base.RefValue{UInt32}
end
Base.size(img::VideoImage) = (img.width, img.height)
Base.eltype(::VideoImage) = UInt8
Base.show(io::IO, img::VideoImage) = print(io, "VideoImage($(img.width)×$(img.height), NV12)")

# Record a luma (Y-plane) image→buffer copy of `img` (display size, top-left,
# tightly packed) into `buffer` at byte `offset`, on high-level command buffer
# `cb`. `img` must already be TRANSFER_SRC_OPTIMAL.
function record_luma_copy!(cb, img::VideoImage, buffer::VK.Buffer, offset::Integer)
    VK.cmd_copy_image_to_buffer(cb, img.image,
        VK.IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer,
        [VK.BufferImageCopy(UInt64(offset), 0, 0,
            VK.ImageSubresourceLayers(VK.IMAGE_ASPECT_PLANE_0_BIT, 0, 0, 1),
            VK.Offset3D(0, 0, 0), VK.Extent3D(img.width, img.height, 1))])
    return cb
end

# Record the chroma (interleaved U/V) plane of an NV12 frame into `buffer`: a
# (width÷2 × height÷2) region of 2-byte (U,V) pixels, tightly packed, so the
# destination is a `width × (height÷2)` UInt8 buffer of interleaved U,V samples.
function record_chroma_copy!(cb, img::VideoImage, buffer::VK.Buffer, offset::Integer)
    VK.cmd_copy_image_to_buffer(cb, img.image,
        VK.IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer,
        [VK.BufferImageCopy(UInt64(offset), 0, 0,
            VK.ImageSubresourceLayers(VK.IMAGE_ASPECT_PLANE_1_BIT, 0, 0, 1),
            VK.Offset3D(0, 0, 0), VK.Extent3D(img.width ÷ 2, img.height ÷ 2, 1))])
    return cb
end

# Transition `img` to TRANSFER_SRC_OPTIMAL on `cb` if it isn't already there.
function transition_transfer_src!(cb, img::VideoImage)
    img.layout[] == UInt32(VK.IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) && return cb
    VK.cmd_pipeline_barrier(cb, [], [],
        [VK.ImageMemoryBarrier(VK.ACCESS_MEMORY_WRITE_BIT,
            VK.ACCESS_TRANSFER_READ_BIT, VK.ImageLayout(img.layout[]),
            VK.IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK.QUEUE_FAMILY_IGNORED, VK.QUEUE_FAMILY_IGNORED, img.image,
            VK.ImageSubresourceRange(VK.IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1))];
        src_stage_mask = VK.PIPELINE_STAGE_ALL_COMMANDS_BIT,
        dst_stage_mask = VK.PIPELINE_STAGE_TRANSFER_BIT)
    img.layout[] = UInt32(VK.IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
    return cb
end

"""
    copyto!(dst::LavaArray{UInt8}, img::VideoImage) -> dst

Copy the decoded frame's luma (Y) plane into the device-local array `dst`
(image→buffer, entirely on the GPU). `dst` must hold at least `prod(size(img))`
bytes and be allocated with a TRANSFER_DST usage flag. Runs one command buffer
on the video-decode queue and blocks until it lands.
"""
function Base.copyto!(dst::LavaArray{UInt8}, img::VideoImage)
    need = img.width * img.height
    length(dst) >= need || error("VideoImage copyto!: dst holds $(length(dst)) < $need bytes")
    mbuf = dst.buf[]
    ctx = mbuf.ctx::VkContext
    dev = ctx.device
    pool = @vk_checked "video_copy_pool" VK.create_command_pool(dev, ctx.video_decode_queue_family_index)
    cb = first(@vk_checked "video_copy_cb" VK.allocate_command_buffers(dev,
        VK.CommandBufferAllocateInfo(pool, VK.COMMAND_BUFFER_LEVEL_PRIMARY, 1)))
    @vk_checked "video_copy_begin" VK.begin_command_buffer(cb, VK.CommandBufferBeginInfo())
    transition_transfer_src!(cb, img)
    record_luma_copy!(cb, img, mbuf.buffer, pool_offset(mbuf) + dst.offset)
    @vk_checked "video_copy_end" VK.end_command_buffer(cb)
    @vk_checked "video_copy_submit" VK.queue_submit(ctx.video_decode_queue, [VK.SubmitInfo([], [], [cb], [])])
    @vk_checked "video_copy_wait" VK.queue_wait_idle(ctx.video_decode_queue)
    return dst
end

"""
    Array(img::VideoImage) -> Matrix{UInt8}

Download the decoded frame's luma (Y) plane to the host as a `width × height`
`Matrix{UInt8}` (grayscale = the NV12 Y plane).
"""
function Base.Array(img::VideoImage)
    # Video decode is single-context (one hardware video queue), and `VideoImage`
    # carries no device, so this download uses the process default. ambient-allocation-ok
    dst = LavaArray{UInt8,2}(undef, (img.width, img.height);
                             extra_usage = UInt32(VK.BUFFER_USAGE_TRANSFER_DST_BIT))
    copyto!(dst, img)
    return Array(dst)
end

module VideoDecode
# `Vulkan`, not `VK`, on the right-hand side: this is a submodule, so the outer
# `Mantle` bindings are not in scope and the alias has to be built from the
# package. Outside this module `Vulkan` is Mantle's backend MARKER and the
# package is `VK`, which is why the two lines disagree.
import Vulkan as VK
const Vk = VK
const C = Vk.VkCore

# What this module borrows from its parent. A submodule does NOT fall back to the
# enclosing module for unqualified names, so every one of these has to be said —
# and none of them were. `mkimage` threw `UndefVarError: alloc_image_memory not
# defined in Mantle.VideoDecode` on the first DPB image, which means
# `decode_h264_gpu` could not have worked at all; it went unnoticed because the
# only callers are the two tests that had themselves stopped running.
#
# `..` is `Mantle`. These are all defined before this point in the include
# order, so they can be imported outright.
import ..VideoImage, ..LavaArray, ..pool_offset,
       ..record_luma_copy!, ..record_chroma_copy!,
       ..oneshot!, ..waitfor!

# `alloc_image_memory` cannot be, and the reason is include order rather than
# anything about the name: it lives in `graphics/framebuffer.jl`, which comes
# AFTER `runtime/video.jl`, so at the moment this module body runs the binding
# does not exist yet and `import` warns and takes nothing. Reached through the
# parent at CALL time instead, which is when it does exist.
const MANTLE = parentmodule(@__MODULE__)

"""
    vkchk(result, what) -> result

Throw on a failing `VkResult` instead of carrying on.

Every `ccall` in this file returned an `Int32` that was thrown away. A decode
whose session, submit or memory binding failed then looks *exactly* like one that
worked and produced black: no exception, no validation message (a call that
returns an error is not a usage error, so the layers say nothing), and frames of
the right shape full of zeros. That is the failure this module has actually had —
every video at 854x480 and above decoded to zeros with nothing reported anywhere.
"""
@inline function vkchk(r::Int32, what::AbstractString)
    r == Int32(0) && return r
    error("video decode: $what returned VkResult $r")
end

# ---------- bitstream ----------
mutable struct BR; d::Vector{UInt8}; p::Int; end
rb1(b)=(byte=b.d[(b.p>>3)+1]; v=(byte>>(7-(b.p&7)))&0x01; b.p+=1; Int(v))
rbn(b,n)=(v=0; for _ in 1:n; v=(v<<1)|rb1(b); end; v)
ue(b)=(z=0; while b.p<8*length(b.d) && rb1(b)==0; z+=1; end; z==0 ? 0 : ((1<<z)-1+rbn(b,z)))
se(b)=(k=ue(b); iseven(k) ? -(k>>1) : (k+1)>>1)
function unescape(d)
    out=UInt8[]; z=0
    for b in d
        if z>=2 && b==0x03; z=0; continue; end
        push!(out,b); z = b==0 ? z+1 : 0
    end; out
end
function split_nals(d)
    out=Tuple{Int,Int,Vector{UInt8}}[]; starts=Int[]; i=1
    while i<=length(d)-3
        if d[i]==0 && d[i+1]==0 && d[i+2]==1; push!(starts,i+3); i+=3 else i+=1 end
    end
    for (k,st) in enumerate(starts)
        e = k<length(starts) ? starts[k+1]-4 : length(d)
        while e>st && d[e]==0; e-=1 end
        nal=d[st:e]; push!(out,(Int(nal[1]&0x1f), Int((nal[1]>>5)&3), nal))
    end
    out
end
pbytes(v::UInt32)=(UInt8(v&0xff),UInt8((v>>8)&0xff),UInt8((v>>16)&0xff),UInt8((v>>24)&0xff))
packflags(vals)=(v=UInt32(0); for (i,x) in enumerate(vals); v|=(UInt32(x&1)<<(i-1)); end; v)
const HIGHSET=Set([100,110,122,244,44,83,86,118,128,138,139,134,135])

function parse_sps(nal)
    b=BR(unescape(nal[2:end]),0)
    profile=rbn(b,8); cflags=rbn(b,8); level=rbn(b,8); sps_id=ue(b)
    chroma=1; sepcol=0; bdl=0; bdc=0; qpp=0; scal=0
    if profile in HIGHSET
        chroma=ue(b); chroma==3 && (sepcol=rb1(b)); bdl=ue(b); bdc=ue(b); qpp=rb1(b); scal=rb1(b)
        scal==1 && error("seq scaling matrix not handled")
    end
    log2fn=ue(b); poct=ue(b); log2poc=0; dpoaz=0
    if poct==0; log2poc=ue(b)
    elseif poct==1; dpoaz=rb1(b); se(b); se(b); nc=ue(b); for _ in 1:nc; se(b); end; end
    maxref=ue(b); gaps=rb1(b); wmbs=ue(b); hmap=ue(b); fmo=rb1(b)
    mbaff = fmo==0 ? rb1(b) : 0; direct8=rb1(b); crop=rb1(b); cl=cr=ct=cb=0
    crop==1 && (cl=ue(b);cr=ue(b);ct=ue(b);cb=ue(b)); vui=rb1(b)
    (;profile,cflags,level,sps_id,chroma,sepcol,bdl,bdc,qpp,log2fn,poct,log2poc,dpoaz,maxref,gaps,wmbs,hmap,fmo,mbaff,direct8,crop,cl,cr,ct,cb,vui)
end
function parse_pps(nal)
    b=BR(unescape(nal[2:end]),0)
    pps_id=ue(b); sps_id=ue(b); entropy=rb1(b); bottom=rb1(b); nsg=ue(b)
    nsg>0 && error("slice groups not handled")
    nl0=ue(b); nl1=ue(b); wp=rb1(b); wbi=rbn(b,2); qp=se(b); qs=se(b); cqp=se(b)
    dbf=rb1(b); cintra=rb1(b); redun=rb1(b); t8=0; cqp2=cqp
    if b.p < 8*length(b.d)-8; t8=rb1(b); ps=rb1(b); ps==1 && error("pic scaling not handled"); cqp2=se(b); end
    (;pps_id,sps_id,entropy,bottom,nl0,nl1,wp,wbi,qp,qs,cqp,dbf,cintra,redun,t8,cqp2)
end
# slice header (enough for POC/DPB): frame_mbs_only assumed
function parse_slice(nal, sps)
    nrid=Int((nal[1]>>5)&3); idr=(nal[1]&0x1f)==5
    b=BR(unescape(nal[2:end]),0)
    first_mb=ue(b); stype=ue(b)%5; ppsid=ue(b)
    sps.sepcol==1 && rbn(b,2)
    fnum=rbn(b, sps.log2fn+4)
    idrid = idr ? ue(b) : 0
    poclsb = sps.poct==0 ? rbn(b, sps.log2poc+4) : 0
    (;nrid,idr,first_mb,stype,fnum,idrid,poclsb)
end

# dec_ref_pic_marking (7.3.3.3): the memory-management control operations a
# reference picture carries. Returns (adaptive::Bool, ops::Vector{(op,arg,ltidx)}).
# Reaching this syntax means fully parsing the slice header up to it — num_ref_idx
# overrides, ref_pic_list_modification, and (skipping) pred_weight_table. Without
# this, hierarchical-B (B-pyramid) streams desync the DPB: the encoder uses MMCO
# op 1 to drop transient reference B-frames while keeping anchors, which a plain
# sliding window gets wrong. Only meaningful for reference slices (nal_ref_idc≠0).
function parse_mmco(nal, sps, pps)
    nrid=Int((nal[1]>>5)&3)
    nrid==0 && return (false, Tuple{Int,Int,Int}[])
    idr=(nal[1]&0x1f)==5
    b=BR(unescape(nal[2:end]),0)
    ue(b); stype=ue(b)%5; ue(b)                       # first_mb, slice_type, pps_id
    sps.sepcol==1 && rbn(b,2)
    rbn(b, sps.log2fn+4)                              # frame_num
    idr && ue(b)                                      # idr_pic_id
    if sps.poct==0
        rbn(b, sps.log2poc+4); pps.bottom==1 && se(b)
    elseif sps.poct==1 && sps.dpoaz==0
        se(b); pps.bottom==1 && se(b)
    end
    pps.redun==1 && ue(b)                             # redundant_pic_cnt
    stype==1 && rb1(b)                                # direct_spatial_mv_pred_flag
    nrl0=pps.nl0+1; nrl1=pps.nl1+1
    if stype==0 || stype==1                           # num_ref_idx_active_override
        if rb1(b)==1; nrl0=ue(b)+1; stype==1 && (nrl1=ue(b)+1); end
    end
    modlist()=(rb1(b)==1 && while true; idc=ue(b); idc==3 && break; (idc==0||idc==1||idc==2) && ue(b); end)
    stype!=2 && stype!=4 && modlist()                 # ref_pic_list_modification l0
    stype==1 && modlist()                             # ref_pic_list_modification l1
    if (pps.wp==1 && stype==0) || (pps.wbi==1 && stype==1)   # pred_weight_table (skip)
        ue(b); sps.chroma!=0 && ue(b)
        for cnt in (stype==1 ? (nrl0,nrl1) : (nrl0,))
            for _ in 1:cnt
                rb1(b)==1 && (se(b); se(b))
                sps.chroma!=0 && rb1(b)==1 && (se(b);se(b);se(b);se(b))
            end
        end
    end
    if idr; rb1(b); rb1(b); return (false, Tuple{Int,Int,Int}[]); end
    adaptive=rb1(b)==1
    ops=Tuple{Int,Int,Int}[]
    if adaptive
        while true
            op=ue(b); op==0 && break
            arg=0; lt=0
            (op==1||op==3) && (arg=ue(b))             # difference_of_pic_nums_minus1
            op==2 && (arg=ue(b))                      # long_term_pic_num
            (op==3||op==6) && (lt=ue(b))              # long_term_frame_idx
            op==4 && (arg=ue(b))                      # max_long_term_frame_idx_plus1
            push!(ops,(op,arg,lt))
            length(ops)>64 && break
        end
    end
    (adaptive, ops)
end

const LEVELMAP=Dict(10=>0,11=>1,12=>2,13=>3,20=>4,21=>5,22=>6,30=>7,31=>8,32=>9,40=>10,41=>11,42=>12,50=>13,51=>14,52=>15,60=>16,61=>17,62=>18)
function std_sps(s)
    csf(i)=(s.cflags>>(7-i))&1
    fl=packflags((csf(0),csf(1),csf(2),csf(3),csf(4),csf(5),s.direct8,s.mbaff,s.fmo,s.dpoaz,s.sepcol,s.gaps,s.qpp,s.crop,0,s.vui))
    C.StdVideoH264SequenceParameterSet(C.StdVideoH264SpsFlags(pbytes(fl)),
        C.StdVideoH264ProfileIdc(s.profile), C.StdVideoH264LevelIdc(LEVELMAP[s.level]),
        C.StdVideoH264ChromaFormatIdc(s.chroma), UInt8(s.sps_id), UInt8(s.bdl), UInt8(s.bdc),
        UInt8(s.log2fn), C.StdVideoH264PocType(s.poct), Int32(0),Int32(0), UInt8(s.log2poc), UInt8(0),
        UInt8(s.maxref), UInt8(0), UInt32(s.wmbs), UInt32(s.hmap), UInt32(s.cl),UInt32(s.cr),UInt32(s.ct),UInt32(s.cb),
        UInt32(0), Ptr{Int32}(C_NULL), Ptr{C.StdVideoH264ScalingLists}(C_NULL), Ptr{C.StdVideoH264SequenceParameterSetVui}(C_NULL))
end
function std_pps(p)
    fl=packflags((p.t8,p.redun,p.cintra,p.dbf,p.wp,p.bottom,p.entropy,0))
    C.StdVideoH264PictureParameterSet(C.StdVideoH264PpsFlags(pbytes(fl)), UInt8(p.sps_id),UInt8(p.pps_id),
        UInt8(p.nl0),UInt8(p.nl1), C.StdVideoH264WeightedBipredIdc(p.wbi),
        Int8(p.qp),Int8(p.qs),Int8(p.cqp),Int8(p.cqp2), Ptr{C.StdVideoH264ScalingLists}(C_NULL))
end

# ---------- Vulkan helpers ----------
struct Ctxwrap; ctx; dev; qf::UInt32; q; end
dfp(w,n)=Vk.function_pointer(w.dev,n)
rp(x)=Base.unsafe_convert(Ptr{eltype(x)},x); pc(x)=Ptr{Cvoid}(rp(x))
function memtype(w,bits,want)
    mp=w.ctx.memory_properties
    for i in 0:(Int(length(mp.memory_types))-1)
        (bits&(UInt32(1)<<i))!=0 && (UInt32(mp.memory_types[i+1].property_flags)&UInt32(want))==UInt32(want) && return i
    end
    for i in 0:(Int(length(mp.memory_types))-1); (bits&(UInt32(1)<<i))!=0 && return i; end
    error("no memory type")
end

# One decode core, several codecs: the session machinery (chunked submit,
# bitstream buffer, reorder, teardown) is shared across `VideoDecoder`s;
# codec-specific parsing/recording lives in `feed!`/`decodeau!` methods and
# the `decodeprofile` hook. H.265 lives in video_h265.jl.
abstract type VideoDecoder end

# returns (video-decode profile ref, profile-list ref, caps NamedTuple)
decodeprofile(dec::VideoDecoder) = video_profile(dec.w)   # h264 default; h265 overrides
"""
    decode_capability_flags(ctx) -> UInt32

`VkVideoDecodeCapabilityFlagsKHR` for the H.264 decode profile on this device:
`0x1` DPB_AND_OUTPUT_COINCIDE, `0x2` DPB_AND_OUTPUT_DISTINCT.

Split out so the decoder and its tests ask the same question. `H264Decoder`
supports both layouts and needs at least one of them; it prefers COINCIDE, which
costs one image and one layout transition per frame less. Picking the wrong one
fails silently (all-zero frames) without validation layers, because a
distinct-only device cannot create a `VIDEO_DECODE_DST|VIDEO_DECODE_DPB` image
at all.
"""
function decode_capability_flags(ctx)
    hp, pr, pl = video_profile(nothing)
    hc = Ref(C.VkVideoDecodeH264CapabilitiesKHR(C.VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_CAPABILITIES_KHR, C_NULL,
                                                C.StdVideoH264LevelIdc(0), C.VkOffset2D(0, 0)))
    e0 = C.VkExtent2D(0, 0)
    GC.@preserve hp pr pl hc begin
        dc = Ref(C.VkVideoDecodeCapabilitiesKHR(C.VK_STRUCTURE_TYPE_VIDEO_DECODE_CAPABILITIES_KHR, pc(hc), UInt32(0)))
        GC.@preserve dc begin
            cp = Ref(C.VkVideoCapabilitiesKHR(C.VK_STRUCTURE_TYPE_VIDEO_CAPABILITIES_KHR, pc(dc), UInt32(0),
                     UInt64(0), UInt64(0), e0, e0, e0, UInt32(0), UInt32(0),
                     C.VkExtensionProperties(ntuple(_ -> Cchar(0), 256), UInt32(0))))
            GC.@preserve cp begin
                vkchk(ccall(Vk.function_pointer(ctx.instance, "vkGetPhysicalDeviceVideoCapabilitiesKHR"),
                      Int32, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}),
                      ctx.physical_device.vks, pc(pr), pc(cp)), "vkGetPhysicalDeviceVideoCapabilitiesKHR")
            end
            return dc[].flags
        end
    end
end

"""
    video_capabilities(ctx) -> NamedTuple

The H.264 decode profile's `VkVideoCapabilitiesKHR` limits that callers have to
respect. `minBitstreamBufferOffsetAlignment` / `minBitstreamBufferSizeAlignment`
are the ones that bite: a `srcBufferOffset` or `srcBufferRange` that is not a
multiple of them is a decode that produces nothing, with no error reported.
"""
function video_capabilities(ctx)
    hp, pr, pl = video_profile(nothing)
    e0 = C.VkExtent2D(0, 0)
    # A decode profile REQUIRES both VkVideoDecodeCapabilitiesKHR and the
    # codec-specific VkVideoDecodeH264CapabilitiesKHR in the pNext chain
    # (VUID-vkGetPhysicalDeviceVideoCapabilitiesKHR-pVideoProfile-07183/-07185).
    # Passing C_NULL here segfaulted inside RADV, which writes the codec caps
    # unconditionally through a chain entry that was not there.
    hc = Ref(C.VkVideoDecodeH264CapabilitiesKHR(C.VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_CAPABILITIES_KHR, C_NULL,
                                                C.StdVideoH264LevelIdc(0), C.VkOffset2D(0, 0)))
    dc = Ref(C.VkVideoDecodeCapabilitiesKHR(C.VK_STRUCTURE_TYPE_VIDEO_DECODE_CAPABILITIES_KHR, pc(hc), UInt32(0)))
    cp = Ref(C.VkVideoCapabilitiesKHR(C.VK_STRUCTURE_TYPE_VIDEO_CAPABILITIES_KHR, pc(dc), UInt32(0),
             UInt64(0), UInt64(0), e0, e0, e0, UInt32(0), UInt32(0),
             C.VkExtensionProperties(ntuple(_ -> Cchar(0), 256), UInt32(0))))
    GC.@preserve hp pr pl hc dc cp begin
        vkchk(ccall(Vk.function_pointer(ctx.instance, "vkGetPhysicalDeviceVideoCapabilitiesKHR"),
              Int32, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}),
              ctx.physical_device.vks, pc(pr), pc(cp)), "vkGetPhysicalDeviceVideoCapabilitiesKHR")
    end
    c = cp[]
    return (minBitstreamBufferOffsetAlignment = Int(c.minBitstreamBufferOffsetAlignment),
            minBitstreamBufferSizeAlignment   = Int(c.minBitstreamBufferSizeAlignment),
            maxDpbSlots                       = Int(c.maxDpbSlots),
            maxActiveReferencePictures        = Int(c.maxActiveReferencePictures))
end

"""
    decode_coincide_supported(ctx) -> Bool

Whether one image may serve as both DPB and decode target. False just means the
decoder takes the separate-images path instead; see [`decode_supported`](@ref)
for whether it can run at all.
"""
decode_coincide_supported(ctx) = (decode_capability_flags(ctx) & UInt32(0x1)) != 0

"""
    decode_supported(ctx) -> Bool

Whether H.264 decode can run here: the device must offer COINCIDE or DISTINCT.
A device offering neither reports no way to allocate decode images.
"""
decode_supported(ctx) = (decode_capability_flags(ctx) & UInt32(0x3)) != 0

function video_profile(w)
    hp=Ref(C.VkVideoDecodeH264ProfileInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_PROFILE_INFO_KHR,C_NULL,C.STD_VIDEO_H264_PROFILE_IDC_HIGH,C.VK_VIDEO_DECODE_H264_PICTURE_LAYOUT_PROGRESSIVE_KHR))
    pr=Ref(C.VkVideoProfileInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_PROFILE_INFO_KHR, Ptr{Cvoid}(rp(hp)), C.VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR, UInt32(C.VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR), UInt32(C.VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR), UInt32(C.VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR)))
    pl=Ref(C.VkVideoProfileListInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_PROFILE_LIST_INFO_KHR, C_NULL, UInt32(1), rp(pr)))
    (hp,pr,pl)
end

"""
    H264Decoder(ctx, paramnals; chroma=false)

A PERSISTENT hardware H.264 decode session: the Vulkan video session, its parameter
objects, the DPB image pool and the bitstream upload buffer are created once and
reused across [`feed!`](@ref)/[`decodemore!`](@ref) calls, so a caller can decode a
long stream (or many GOPs of one stream) INCREMENTALLY — a few frames at a time,
interleaved with other GPU work on the same thread — instead of paying for a whole
GOP in one blocking call. `paramnals` is any Annex-B chunk that contains the
stream's SPS+PPS (the lead parameter sets); every later [`feed!`](@ref) must start
at an IDR so the DPB resets naturally. `close(dec)` frees the session.

The batch [`decode_h264`](@ref) is a thin wrapper over this type (feed everything,
drain, flush) — one decode core, two access patterns.
"""
mutable struct H264Decoder <: VideoDecoder
    w::Ctxwrap; dev::Any
    sps::Any; pps::Any
    chroma::Bool
    SESSION::Any; PARAMS::Any
    PIN::Vector{Any}
    imgs::Vector{Any}                       # DPB pool: (VideoImage, ImageView) per slot
    nslots::Int
    outimgs::Vector{Any}                    # DPB_AND_OUTPUT_DISTINCT: decode targets,
                                            # one per AU of the chunk; empty under COINCIDE
    mkoutimg::Any                           # grows `outimgs`; `nothing` under COINCIDE
    copyq::Vector{Any}                      # (image, Y, UV) copies deferred to the main queue
    bsalign::Int                            # minBitstreamBuffer{Offset,Size}Alignment
    bbuf::Any; bmem::Any; bmap::Ptr{UInt8}; bufsz::UInt64
    cbh::Any; CB::Any
    CW::Int; CH::Int; DW::Int; DH::Int
    # feed / decode state
    aus::Vector{Vector{Vector{UInt8}}}      # access units of the current feed
    nextau::Int
    dpb::Vector{Tuple{Int,Int,Int}}         # (slot, framenum, poc)
    freeslots::Vector{Int}
    prevmsb::Int; prevlsb::Int; maxpoclsb::Int; gop::Int
    # `pic_order_cnt_type = 2` carries no POC in the slice header: it is derived
    # from `frame_num`, so the wrap has to be tracked (8.2.1.3, prevFrameNum /
    # prevFrameNumOffset).
    prevfn::Int; prevfnoff::Int
    decoded::Int                            # AUs decoded ever
    isfirst::Bool
    pending::Vector{Tuple{Int,Int,Any,Any}} # decoded, not yet display-safe: (gop, poc, Y, UV)
    open::Bool
end

"""
    feed!(dec, annexb)

Queue an Annex-B chunk (typically ONE GOP, starting at an IDR) for incremental
decode. Undrained frames of a previous feed stay pending and drain first.
"""
function feed!(dec::H264Decoder, annexb::AbstractVector{UInt8})
    nalu = split_nals(annexb)
    aus = Vector{Vector{Vector{UInt8}}}()
    maxau = 0
    for n in nalu
        (n[1]==1 || n[1]==5) || continue
        sh = parse_slice(n[3], dec.sps)
        if sh.first_mb==0; push!(aus,[n[3]]) else push!(aus[end],n[3]) end
        maxau = max(maxau, sum(length(sl)+3 for sl in aus[end]))
    end
    ensurebitbuf!(dec, maxau)
    dec.aus = aus
    dec.nextau = 1
    return dec
end

"""
Decode targets allocated at once under DPB_AND_OUTPUT_DISTINCT. Bounds the memory
of an unbounded chunk; larger just means fewer video-queue submits per stream.
"""
const MAX_DECODE_TARGETS = 16

"Frames not yet returned by [`decodemore!`](@ref) (undedcoded AUs + pending reorder)."
remaining(dec::VideoDecoder) = (length(dec.aus) - dec.nextau + 1) + length(dec.pending)

"""
    decodemore!(dec, maxframes; holdback=dec.nslots) -> Vector{(Y, UV)}

Decode up to `maxframes` access units of the current feed and return every frame
that is DISPLAY-ORDER safe. B-frames decode before the frames they display after,
so up to `holdback` decoded frames are held back for reordering; once the feed is
exhausted everything flushes (`holdback = typemax(Int)` reproduces the batch
behavior exactly: emit nothing until the end, then the full sort). Each call costs
about `maxframes` × per-frame decode time — size it to the latency budget.
"""
function decodemore!(dec::VideoDecoder, maxframes::Integer; holdback::Integer = dec.nslots)
    hi = min(dec.nextau + Int(min(maxframes, typemax(Int) ÷ 2)) - 1, length(dec.aus))
    if dec.nextau <= hi
        batch = dec.nextau:hi
        ausize(au) = cld(sum(length(sl) + 3 for sl in au), dec.bsalign) * dec.bsalign
        w = dec.w; CB = dec.CB
        # COINCIDE copies out of the DPB slot inside this command buffer, so the
        # whole chunk is ONE command buffer + ONE queue wait — per-AU waits would
        # serialize the hardware decoder against ~2 ms round-trips per frame.
        #
        # DISTINCT needs a decode target per AU that survives until its copy runs
        # (slots are recycled mid-chunk), and `decode_h264` passes maxframes=typemax,
        # so an unbounded chunk would allocate one full-size image per frame of the
        # entire stream. Cap the targets and submit in sub-chunks of that size: the
        # copies drain and the images are reused every MAX_DECODE_TARGETS frames.
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
            GC.@preserve si cbref vkchk(ccall(dfp(w,"vkQueueSubmit"),Int32,(Ptr{Cvoid},UInt32,Ptr{Cvoid},Ptr{Cvoid}),w.q.vks,UInt32(1),Base.unsafe_convert(Ptr{Cvoid},si),C_NULL), "vkQueueSubmit")
            vkchk(ccall(dfp(w,"vkQueueWaitIdle"),Int32,(Ptr{Cvoid},),w.q.vks), "vkQueueWaitIdle")
            # Blocks until the copies complete (flush! waits on the timeline), so the
            # decode targets are free for the next sub-chunk.
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

"""
Record the chunk's deferred image→buffer copies on the MAIN queue and submit them.

Only DPB_AND_OUTPUT_DISTINCT defers: there the decode target is a separate image
and `vkCmdCopyImageToBuffer` cannot be recorded on the video-decode queue family,
which need not support transfer (AMD's reports VIDEO_DECODE only). Called after
`vkQueueWaitIdle` on the decode queue, so the decoded contents and the
TRANSFER_SRC layout transition recorded there have completed; the targets are
CONCURRENT across both families, so no ownership transfer is needed.
"""
function drain_copies!(dec::VideoDecoder)
    isempty(dec.copyq) && return nothing
    ctx = dec.w.ctx
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
    empty!(dec.copyq)
    waitfor!(ctx.default_bq, tok)
    return nothing
end

"Grow the (host-visible) bitstream upload buffer to hold an AU of `need` bytes."
function ensurebitbuf!(dec::VideoDecoder, need::Integer)
    sz = UInt64(cld(max(need, 1), dec.bsalign) * dec.bsalign)
    sz <= dec.bufsz && return nothing
    w = dec.w; dev = dec.dev
    dec.bbuf === nothing || ccall(dfp(w,"vkDestroyBuffer"),Cvoid,(Ptr{Cvoid},C.VkBuffer,Ptr{Cvoid}),dev.vks,dec.bbuf,C_NULL)
    hp,pr,pl = decodeprofile(dec); push!(dec.PIN, hp, pr, pl)
    ci=Ref(C.VkBufferCreateInfo(C.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,Ptr{Cvoid}(rp(pl)),UInt32(0),sz,UInt32(C.VK_BUFFER_USAGE_VIDEO_DECODE_SRC_BIT_KHR),C.VK_SHARING_MODE_EXCLUSIVE,UInt32(0),Ptr{UInt32}(C_NULL)))
    rb=Ref{C.VkBuffer}(); GC.@preserve ci pl vkchk(ccall(dfp(w,"vkCreateBuffer"),Int32,(Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid}),dev.vks,pc(ci),C_NULL,rp(rb)), "vkCreateBuffer"); bf=rb[]
    mr=Ref(C.VkMemoryRequirements(UInt64(0),UInt64(0),UInt32(0))); ccall(dfp(w,"vkGetBufferMemoryRequirements"),Cvoid,(Ptr{Cvoid},C.VkBuffer,Ptr{Cvoid}),dev.vks,bf,rp(mr))
    m=Vk.unwrap(Vk.allocate_memory(dev,mr[].size,memtype(w,mr[].memoryTypeBits,UInt32(C.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)|UInt32(C.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)))); push!(dec.PIN, m)
    vkchk(ccall(dfp(w,"vkBindBufferMemory"),Int32,(Ptr{Cvoid},C.VkBuffer,C.VkDeviceMemory,UInt64),dev.vks,bf,m.vks,UInt64(0)), "vkBindBufferMemory")
    pmap=Ref{Ptr{Cvoid}}(); vkchk(ccall(dfp(w,"vkMapMemory"),Int32,(Ptr{Cvoid},C.VkDeviceMemory,UInt64,UInt64,UInt32,Ptr{Cvoid}),dev.vks,m.vks,UInt64(0),sz,UInt32(0),Base.unsafe_convert(Ptr{Ptr{Cvoid}},pmap)), "vkMapMemory")
    dec.bbuf=bf; dec.bmem=m; dec.bmap=Ptr{UInt8}(pmap[]); dec.bufsz=sz
    return nothing
end

function Base.close(dec::VideoDecoder)
    dec.open || return nothing
    dec.open = false
    w = dec.w; dev = dec.dev
    vkchk(ccall(dfp(w,"vkQueueWaitIdle"),Int32,(Ptr{Cvoid},),w.q.vks), "vkQueueWaitIdle")
    dec.PARAMS === nothing || ccall(dfp(w,"vkDestroyVideoSessionParametersKHR"),Cvoid,(Ptr{Cvoid},C.VkVideoSessionParametersKHR,Ptr{Cvoid}),dev.vks,dec.PARAMS,C_NULL)
    dec.SESSION === nothing || ccall(dfp(w,"vkDestroyVideoSessionKHR"),Cvoid,(Ptr{Cvoid},C.VkVideoSessionKHR,Ptr{Cvoid}),dev.vks,dec.SESSION,C_NULL)
    dec.bbuf === nothing || ccall(dfp(w,"vkDestroyBuffer"),Cvoid,(Ptr{Cvoid},C.VkBuffer,Ptr{Cvoid}),dev.vks,dec.bbuf,C_NULL)
    empty!(dec.PIN); empty!(dec.imgs); empty!(dec.pending); empty!(dec.aus)
    return nothing
end

function H264Decoder(ctx, paramnals::AbstractVector{UInt8}; chroma::Bool=false)
    ctx.video_decode_available || error("device has no video decode support")
    Lava = parentmodule(@__MODULE__)
    w=Ctxwrap(ctx, ctx.device, UInt32(ctx.video_decode_queue_family_index), ctx.video_decode_queue)
    dev=w.dev
    nalu=split_nals(paramnals)
    sps = parse_sps(first(n[3] for n in nalu if n[1]==7))
    pps = parse_pps(first(n[3] for n in nalu if n[1]==8))
    # Only 4:2:0 (NV12) is supported: it's the one chroma format the H.264 video
    # decode engine handles, and the output/DPB images are hardcoded NV12. A
    # 4:2:2/4:4:4 stream would silently mis-decode, so reject it explicitly.
    sps.chroma == 1 ||
        error("decode_h264: only 4:2:0 chroma is supported (stream is chroma_format_idc=$(sps.chroma); " *
              "profile_idc=$(sps.profile)). Transcode to yuv420p first.")
    # Streams with max_num_ref_frames ≤ 1 (baseline single-reference / all-intra)
    # currently decode to garbage on this path; the common camera/delivery case
    # (multi-reference High profile) is pixel-exact. Reject rather than mis-decode.
    sps.maxref >= 2 ||
        error("decode_h264: streams with max_num_ref_frames=$(sps.maxref) (single-reference/all-intra) " *
              "are not yet supported; re-encode with -refs 2 or higher.")
    CW=(sps.wmbs+1)*16; CH=(sps.hmap+1)*16
    DW=CW-2*sps.cr; DH=CH-2*sps.cb   # display size (chroma 420 crop units = 2px)
    fmt=C.VkFormat(1000156003)       # G8_B8R8_2PLANE_420_UNORM (NV12)
    PIN=Any[]; pin(x)=(push!(PIN,x);x)
    hp,pr,pl=video_profile(w); pin(hp);pin(pr);pin(pl); pProf=rp(pr)
    # session
    rHdr=pin(Ref(C.VkExtensionProperties(ntuple(_->Cchar(0),256),UInt32(0))))
    # Set from the decode caps below: whether the DPB and the decode target must
    # be separate images. `let` assigns to this outer binding, it does not shadow it.
    DISTINCT = false
    BSALIGN = 256
    # get stdHeaderVersion from caps
    let hc=Ref(C.VkVideoDecodeH264CapabilitiesKHR(C.VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_CAPABILITIES_KHR,C_NULL,C.StdVideoH264LevelIdc(0),C.VkOffset2D(0,0))),
        dc=Ref(C.VkVideoDecodeCapabilitiesKHR(C.VK_STRUCTURE_TYPE_VIDEO_DECODE_CAPABILITIES_KHR,Ptr{Cvoid}(rp(hc)),UInt32(0))),
        e0=C.VkExtent2D(0,0), cp=Ref(C.VkVideoCapabilitiesKHR(C.VK_STRUCTURE_TYPE_VIDEO_CAPABILITIES_KHR,Ptr{Cvoid}(rp(dc)),UInt32(0),UInt64(0),UInt64(0),e0,e0,e0,UInt32(0),UInt32(0),C.VkExtensionProperties(ntuple(_->Cchar(0),256),UInt32(0))))
        GC.@preserve hc dc cp PIN vkchk(ccall(Vk.function_pointer(ctx.instance,"vkGetPhysicalDeviceVideoCapabilitiesKHR"),Int32,(Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid}),ctx.physical_device.vks,Ptr{Cvoid}(pProf),pc(cp)), "vkGetPhysicalDeviceVideoCapabilitiesKHR")
        rHdr[]=cp[].stdHeaderVersion
        # The bitstream buffer's offset AND range must each be a multiple of the
        # profile's alignment. Hardcoding 256 happened to satisfy devices asking for
        # <=256; this profile asks 4096 on AMD, and a misaligned range is a decode
        # that silently produces nothing. Both are powers of two, so the larger of
        # the two satisfies both.
        BSALIGN = Int(max(cp[].minBitstreamBufferOffsetAlignment, cp[].minBitstreamBufferSizeAlignment, 1))
        # `dc` was already being filled in and thrown away.  Its flags say
        # whether one image may serve as both DPB and decode target
        # (DPB_AND_OUTPUT_COINCIDE, 0x1) or whether they must be separate images
        # (DPB_AND_OUTPUT_DISTINCT, 0x2).  Both layouts are supported below; a
        # device must offer at least one.
        #
        # Getting this wrong fails SILENTLY: on a distinct-only device
        # `vkCreateImage` rejects `VIDEO_DECODE_DST|VIDEO_DECODE_DPB`, decode then
        # runs against nothing usable, every frame comes back all-zero, and the
        # only symptom is that the output does not match the reference. Without
        # validation layers there is nothing in the log at all. AMD Radeon 8060S
        # reports flags=0x2.
        #
        # COINCIDE is preferred where offered: it needs one image less and one
        # layout transition less per frame.
        let f = dc[].flags
            (f & UInt32(0x3)) != 0 || error(
                "H264Decoder: device reports video decode flags 0x", string(f, base=16),
                " — neither DPB_AND_OUTPUT_COINCIDE (0x1) nor DPB_AND_OUTPUT_DISTINCT (0x2) ",
                "is supported, so there is no way to allocate decode images.")
            DISTINCT = (f & UInt32(0x1)) == 0
        end
    end
    maxslots=UInt32(sps.maxref+2)
    sci=pin(Ref(C.VkVideoSessionCreateInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_SESSION_CREATE_INFO_KHR,C_NULL,w.qf,UInt32(0),pProf,fmt,C.VkExtent2D(CW,CH),fmt,maxslots,UInt32(sps.maxref),rp(rHdr))))
    rSess=Ref{C.VkVideoSessionKHR}(); GC.@preserve PIN vkchk(ccall(dfp(w,"vkCreateVideoSessionKHR"),Int32,(Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid}),dev.vks,pc(sci),C_NULL,rp(rSess)), "vkCreateVideoSessionKHR"); SESSION=rSess[]
    # bind session memory
    let mc=Ref(UInt32(0)); vkchk(ccall(dfp(w,"vkGetVideoSessionMemoryRequirementsKHR"),Int32,(Ptr{Cvoid},C.VkVideoSessionKHR,Ptr{UInt32},Ptr{Cvoid}),dev.vks,SESSION,mc,C_NULL), "vkGetVideoSessionMemoryRequirementsKHR")
        mreqs=[C.VkVideoSessionMemoryRequirementsKHR(C.VK_STRUCTURE_TYPE_VIDEO_SESSION_MEMORY_REQUIREMENTS_KHR,C_NULL,UInt32(0),C.VkMemoryRequirements(UInt64(0),UInt64(0),UInt32(0))) for _ in 1:Int(mc[])]
        GC.@preserve mreqs vkchk(ccall(dfp(w,"vkGetVideoSessionMemoryRequirementsKHR"),Int32,(Ptr{Cvoid},C.VkVideoSessionKHR,Ptr{UInt32},Ptr{Cvoid}),dev.vks,SESSION,mc,pointer(mreqs)), "vkGetVideoSessionMemoryRequirementsKHR")
        binds=C.VkBindVideoSessionMemoryInfoKHR[]
        for mr in mreqs
            m=Vk.unwrap(Vk.allocate_memory(dev,mr.memoryRequirements.size,memtype(w,mr.memoryRequirements.memoryTypeBits,C.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))); pin(m)
            push!(binds,C.VkBindVideoSessionMemoryInfoKHR(C.VK_STRUCTURE_TYPE_BIND_VIDEO_SESSION_MEMORY_INFO_KHR,C_NULL,mr.memoryBindIndex,m.vks,UInt64(0),mr.memoryRequirements.size))
        end
        GC.@preserve binds PIN vkchk(ccall(dfp(w,"vkBindVideoSessionMemoryKHR"),Int32,(Ptr{Cvoid},C.VkVideoSessionKHR,UInt32,Ptr{Cvoid}),dev.vks,SESSION,UInt32(length(binds)),pointer(binds)), "vkBindVideoSessionMemoryKHR")
    end
    # session params (SPS+PPS)
    rSPSs=pin(Ref(std_sps(sps))); rPPSs=pin(Ref(std_pps(pps)))
    add=pin(Ref(C.VkVideoDecodeH264SessionParametersAddInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_SESSION_PARAMETERS_ADD_INFO_KHR,C_NULL,UInt32(1),rp(rSPSs),UInt32(1),rp(rPPSs))))
    h264c=pin(Ref(C.VkVideoDecodeH264SessionParametersCreateInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_SESSION_PARAMETERS_CREATE_INFO_KHR,C_NULL,UInt32(1),UInt32(1),rp(add))))
    pci=pin(Ref(C.VkVideoSessionParametersCreateInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_SESSION_PARAMETERS_CREATE_INFO_KHR,Ptr{Cvoid}(rp(h264c)),UInt32(0),C_NULL,SESSION)))
    rParams=Ref{C.VkVideoSessionParametersKHR}(); GC.@preserve PIN vkchk(ccall(dfp(w,"vkCreateVideoSessionParametersKHR"),Int32,(Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid}),dev.vks,pc(pci),C_NULL,rp(rParams)), "vkCreateVideoSessionParametersKHR"); PARAMS=rParams[]
    # DPB image pool — real VK.Image handles wrapped as VideoImages. The
    # decode commands (raw FFI) take the raw `.vks` handles; the output copy uses
    # the high-level VideoImage path. High-level handles are refcounted, so the
    # whole pool auto-frees when this call returns.
    nslots=Int(maxslots)
    fmt_hl=Vk.Format(1000156003)   # G8_B8R8_2PLANE_420_UNORM (NV12)
    # DPB images MUST advertise the SAME video profile the session was created
    # with, or the driver silently declines to decode into them. Reuse the exact
    # raw `pl` (VkVideoProfileListInfoKHR) pointer the session uses — high-level
    # `create_image` accepts a raw pNext pointer, so no re-serialization can drift.
    # COINCIDE puts DST and DPB on one image and copies the display output straight
    # out of the DPB slot. DISTINCT forbids that combination, so the DPB images are
    # DPB-only and a single extra image is the decode target that gets copied from.
    dpbusage = DISTINCT ?
        Vk.IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR :
        Vk.IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR|Vk.IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR|Vk.IMAGE_USAGE_TRANSFER_SRC_BIT
    # Same profile pNext as the session, or the driver silently declines to decode.
    mkimage(usage; families=UInt32[]) = begin
        image=GC.@preserve PIN Vk.Image(dev, Vk.IMAGE_TYPE_2D, fmt_hl, Vk.Extent3D(CW,CH,1),
            1, 1, Vk.SAMPLE_COUNT_1_BIT, Vk.IMAGE_TILING_OPTIMAL, usage,
            isempty(families) ? Vk.SHARING_MODE_EXCLUSIVE : Vk.SHARING_MODE_CONCURRENT,
            families, Vk.IMAGE_LAYOUT_UNDEFINED; next=Ptr{Cvoid}(rp(pl)))
        mem=MANTLE.alloc_image_memory(ctx, image)
        view=Vk.ImageView(dev, image, Vk.IMAGE_VIEW_TYPE_2D, fmt_hl,
            Vk.ComponentMapping(Vk.COMPONENT_SWIZZLE_IDENTITY,Vk.COMPONENT_SWIZZLE_IDENTITY,Vk.COMPONENT_SWIZZLE_IDENTITY,Vk.COMPONENT_SWIZZLE_IDENTITY),
            Vk.ImageSubresourceRange(Vk.IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1))
        vimg=VideoImage(image, mem, CW, CH, DW, DH, fmt_hl, Ref(UInt32(C.VK_IMAGE_LAYOUT_UNDEFINED)))
        pin(image); pin(mem); pin(view)
        (vimg, view)
    end
    imgs=Vector{Any}(undef,nslots)  # (vimg::VideoImage, view::VK.ImageView)
    for k in 1:nslots
        imgs[k]=mkimage(dpbusage)
    end
    # One decode target PER AU of the chunk, not one shared target: a DPB slot is
    # recycled as soon as its frame stops being a reference, so within a single
    # chunk the same slot backs several frames. The output copies therefore cannot
    # all read one image at the end — each frame needs its own target that survives
    # until the copy runs. Grown on demand, reused across chunks.
    #
    # CONCURRENT across the video and main queue families because the copies run on
    # the main queue (a video-decode-only family cannot do vkCmdCopyImageToBuffer),
    # which would otherwise need an ownership transfer.
    QFAMS = unique(UInt32[UInt32(ctx.queue_family_index), UInt32(w.qf)])
    OUTIMGS = Any[]
    MKOUTIMG = DISTINCT ?
        (() -> mkimage(Vk.IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR|Vk.IMAGE_USAGE_TRANSFER_SRC_BIT;
                       families = length(QFAMS) > 1 ? QFAMS : UInt32[])) :
        nothing
    # command pool + buffer. The command buffer is a *high-level* Vulkan handle
    # so the output copy can use `cmd_copy_image_to_buffer`; the raw video-codec
    # commands take its `.vks` raw handle. (The bitstream upload buffer is created
    # lazily by `ensurebitbuf!` — its size depends on the fed stream's largest AU.)
    cbh=first(Vk.unwrap(Vk.allocate_command_buffers(dev,
        Vk.CommandBufferAllocateInfo(Vk.unwrap(Vk.create_command_pool(dev, w.qf; flags=Vk.COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT)),
            Vk.COMMAND_BUFFER_LEVEL_PRIMARY, 1)))); pin(cbh)
    CB=cbh.vks

    return H264Decoder(w, dev, sps, pps, chroma, SESSION, PARAMS, PIN, imgs, nslots, OUTIMGS, MKOUTIMG, Any[], BSALIGN,
                       nothing, nothing, Ptr{UInt8}(0), UInt64(0), cbh, CB,
                       Int(CW), Int(CH), Int(DW), Int(DH),
                       Vector{Vector{Vector{UInt8}}}(), 1,
                       Tuple{Int,Int,Int}[], collect(1:nslots),
                       0, 0, 1 << (sps.log2poc + 4), 0,
                       0, 0,                                    # prevfn, prevfnoff
                       0, true,
                       Tuple{Int,Int,Any,Any}[], true)
end

"""
Record ONE access unit's decode + output copy into the decoder's (already begun)
command buffer, uploading its bitstream at `bufbase`, and append the frame's output
arrays to `dec.pending`. Pure recording + CPU-side DPB bookkeeping — the caller
([`decodemore!`](@ref)) submits the whole chunk with ONE queue wait, which is what
makes chunked decode cheap (a per-frame `vkQueueWaitIdle` costs ~2 ms alone).
Returns the bitstream bytes consumed, aligned to the profile's requirement.
"""
function decodeau!(dec::H264Decoder, au, bufbase::Integer, slot::Integer=1)
    Lava = parentmodule(@__MODULE__)
    w=dec.w; dev=dec.dev; sps=dec.sps; pps=dec.pps
    SESSION=dec.SESSION; PARAMS=dec.PARAMS; PIN=dec.PIN; imgs=dec.imgs
    bbuf=dec.bbuf; bmap=dec.bmap; cbh=dec.cbh; CB=dec.CB
    CW=dec.CW; CH=dec.CH; DW=dec.DW; DH=dec.DH; chroma=dec.chroma
    dpb=dec.dpb; freeslots=dec.freeslots; maxpoclsb=dec.maxpoclsb
    allcol=C.VkImageSubresourceRange(UInt32(C.VK_IMAGE_ASPECT_COLOR_BIT),UInt32(0),UInt32(1),UInt32(0),UInt32(1))
    IGN=UInt32(0xffffffff); allst=UInt32(C.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT)
    barrier(img,old,new)=Ref(C.VkImageMemoryBarrier(C.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,C_NULL,UInt32(C.VK_ACCESS_MEMORY_WRITE_BIT),UInt32(C.VK_ACCESS_MEMORY_READ_BIT)|UInt32(C.VK_ACCESS_MEMORY_WRITE_BIT),C.VkImageLayout(old),C.VkImageLayout(new),IGN,IGN,img,allcol))
    emit(cb,b)=ccall(dfp(w,"vkCmdPipelineBarrier"),Cvoid,(C.VkCommandBuffer,UInt32,UInt32,UInt32,UInt32,Ptr{Cvoid},UInt32,Ptr{Cvoid},UInt32,Ptr{Cvoid}),cb,allst,allst,UInt32(0),UInt32(0),C_NULL,UInt32(0),C_NULL,UInt32(1),pc(b))
    emitmem(cb,b)=ccall(dfp(w,"vkCmdPipelineBarrier"),Cvoid,(C.VkCommandBuffer,UInt32,UInt32,UInt32,UInt32,Ptr{Cvoid},UInt32,Ptr{Cvoid},UInt32,Ptr{Cvoid}),cb,allst,allst,UInt32(0),UInt32(1),pc(b),UInt32(0),C_NULL,UInt32(0),C_NULL)
    # Consecutive decodes in one command buffer are NOT implicitly ordered: this AU
    # writes its reconstructed picture into a DPB slot that the next AU reads as a
    # reference. The per-image barriers below only fire when a LAYOUT changes, so
    # once every slot sits in DPB layout nothing separated one decode from the next.
    # The hardware decoder happens to serialize, which is why the output was right
    # anyway, but the dependency was genuinely missing.
    mb=Ref(C.VkMemoryBarrier(C.VK_STRUCTURE_TYPE_MEMORY_BARRIER,C_NULL,UInt32(C.VK_ACCESS_MEMORY_WRITE_BIT),UInt32(C.VK_ACCESS_MEMORY_READ_BIT)|UInt32(C.VK_ACCESS_MEMORY_WRITE_BIT)))
    picres(view)=C.VkVideoPictureResourceInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR,C_NULL,C.VkOffset2D(0,0),C.VkExtent2D(CW,CH),UInt32(0),view)
    DPBLAYOUT=UInt32(C.VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR); TSRC=UInt32(C.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
        DSTLAYOUT=UInt32(C.VK_IMAGE_LAYOUT_VIDEO_DECODE_DST_KHR)   # DISTINCT decode target
    begin
        sh=parse_slice(au[1],sps)
        if sh.idr
            for (slot,_,_) in dpb; push!(freeslots,slot); end; empty!(dpb)
            dec.prevmsb=0; dec.prevlsb=0; dec.gop+=1
            dec.prevfn=0; dec.prevfnoff=0
        end
        if sps.poct==0
            # 8.2.1.1
            if sh.idr; pmsb=0
            elseif sh.poclsb<dec.prevlsb && (dec.prevlsb-sh.poclsb)>=maxpoclsb÷2; pmsb=dec.prevmsb+maxpoclsb
            elseif sh.poclsb>dec.prevlsb && (sh.poclsb-dec.prevlsb)>maxpoclsb÷2; pmsb=dec.prevmsb-maxpoclsb
            else; pmsb=dec.prevmsb; end
            poc=pmsb+sh.poclsb
            if sh.nrid!=0; dec.prevmsb=pmsb; dec.prevlsb=sh.poclsb; end
        elseif sps.poct==2
            # 8.2.1.3. There is no POC in the slice header at all here: it is
            # `2*(FrameNumOffset + frame_num)`, one less for a non-reference
            # picture, and FrameNumOffset absorbs the frame_num wrap.
            #
            # This used to be `poc = dec.decoded`, a count of access units decoded
            # ever. That is not a picture order count and is not even in the same
            # units — a reference picture's POC advances by TWO — so the hardware
            # was handed a dense 0,1,2,… as `PicOrderCnt` for every picture and
            # every reference slot. x264 selects `pic_order_cnt_type = 2` whenever
            # it emits no B-frames, which is precisely what `-bf 0` asks for and
            # what VideoEditor's own mezzanine encodes with, so the editor's GPU
            # preview decoded solid black for every clip it ever transcoded while
            # the 128x96 B-pyramid fixture (type 0) stayed bit-exact.
            maxfn = 1 << (sps.log2fn + 4)
            fnoff = sh.idr ? 0 :
                    (dec.prevfn > sh.fnum ? dec.prevfnoff + maxfn : dec.prevfnoff)
            poc = sh.idr ? 0 :
                  sh.nrid == 0 ? 2 * (fnoff + sh.fnum) - 1 : 2 * (fnoff + sh.fnum)
            dec.prevfn = sh.fnum; dec.prevfnoff = fnoff
        else
            # Type 1 needs the SPS's offset_for_* cycle, which `parse_sps` reads
            # past without keeping. Refuse rather than invent a number: that is
            # the mistake type 2 was.
            error("decode_h264: pic_order_cnt_type $(sps.poct) is not implemented")
        end
        isref = sh.nrid!=0
        outslot=popfirst!(freeslots)
        outvimg,outviewhl=imgs[outslot]     # DPB slot — the setup reference picture
        outimg=outvimg.image.vks; outview=outviewhl.vks; outlay=outvimg.layout
        # Decode target. Under COINCIDE this IS the DPB slot, so every value derived
        # from it below stays bit-identical to the single-image path.
        distinct = dec.mkoutimg !== nothing
        if distinct
            while length(dec.outimgs) < slot; push!(dec.outimgs, dec.mkoutimg()); end
        end
        dstvimg,dstviewhl = distinct ? dec.outimgs[slot] : (outvimg,outviewhl)
        dstimg=dstvimg.image.vks; dstview=dstviewhl.vks; dstlay=dstvimg.layout
        # upload AU to its chunk-batched region of the bitstream buffer
        off=0; sliceoffs=UInt32[]
        for sl in au
            push!(sliceoffs,UInt32(off))
            # The start code and the slice, staged through a temporary that has to
            # be KEPT ALIVE across the copy. `pointer(vcat(...))` hands the address
            # of an array nothing references any more: the compiler is free to
            # treat it as dead the instant `pointer` returns, and a large slice is
            # exactly when the collector is most likely to take it. Small frames
            # survived, real ones did not.
            stage = vcat(UInt8[0,0,1], sl)
            GC.@preserve stage unsafe_copyto!(bmap+bufbase+off, pointer(stage), length(sl)+3)
            off+=length(sl)+3
        end
        # reference slots (active DPB)
        refpr=Ref{C.VkVideoPictureResourceInfoKHR}[]; refri=Ref{C.StdVideoDecodeH264ReferenceInfo}[]; refds=Ref{C.VkVideoDecodeH264DpbSlotInfoKHR}[]
        decref=C.VkVideoReferenceSlotInfoKHR[]; begref=C.VkVideoReferenceSlotInfoKHR[]
        for (slot,fn,pc0) in dpb
            rv=imgs[slot][2].vks
            pr=Ref(picres(rv)); ri=Ref(C.StdVideoDecodeH264ReferenceInfo(C.StdVideoDecodeH264ReferenceInfoFlags(pbytes(UInt32(0))),UInt16(fn),UInt16(0),(Int32(pc0),Int32(pc0))))
            ds=Ref(C.VkVideoDecodeH264DpbSlotInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_DPB_SLOT_INFO_KHR,C_NULL,rp(ri)))
            push!(refpr,pr);push!(refri,ri);push!(refds,ds)
            push!(decref,C.VkVideoReferenceSlotInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR,pc(ds),Int32(slot-1),rp(pr)))
            push!(begref,C.VkVideoReferenceSlotInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR,C_NULL,Int32(slot-1),rp(pr)))
        end
        # `outpr` names the DPB slot (setup reference + the slot activated in
        # BeginCoding); `dstpr` names the decode target. Equal under COINCIDE.
        outpr=Ref(picres(outview))
        dstpr=Ref(picres(dstview))
        outri=Ref(C.StdVideoDecodeH264ReferenceInfo(C.StdVideoDecodeH264ReferenceInfoFlags(pbytes(UInt32(0))),UInt16(sh.fnum),UInt16(0),(Int32(poc),Int32(poc))))
        outds=Ref(C.VkVideoDecodeH264DpbSlotInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_DPB_SLOT_INFO_KHR,C_NULL,rp(outri)))
        push!(begref,C.VkVideoReferenceSlotInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR,C_NULL,Int32(-1),rp(outpr)))
        setup=Ref(C.VkVideoReferenceSlotInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR,pc(outds),Int32(outslot-1),rp(outpr)))
        picflags=packflags((0, sh.stype==2 ? 1 : 0, sh.idr ? 1 : 0, 0, isref ? 1 : 0, 0))
        stdpic=Ref(C.StdVideoDecodeH264PictureInfo(C.StdVideoDecodeH264PictureInfoFlags(pbytes(picflags)),UInt8(0),UInt8(0),UInt8(0),UInt8(0),UInt16(sh.fnum),UInt16(sh.idrid),(Int32(poc),Int32(poc))))
        soff=copy(sliceoffs)
        h264pi=Ref(C.VkVideoDecodeH264PictureInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_PICTURE_INFO_KHR,C_NULL,rp(stdpic),UInt32(length(soff)),pointer(soff)))
        decinfo=Ref(C.VkVideoDecodeInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_DECODE_INFO_KHR,pc(h264pi),UInt32(0),bbuf,UInt64(bufbase),UInt64(cld(off,dec.bsalign)*dec.bsalign),dstpr[],rp(setup),UInt32(length(decref)),isempty(decref) ? Ptr{C.VkVideoReferenceSlotInfoKHR}(C_NULL) : pointer(decref)))
        beginfo=Ref(C.VkVideoBeginCodingInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_BEGIN_CODING_INFO_KHR,C_NULL,UInt32(0),SESSION,PARAMS,UInt32(length(begref)),pointer(begref)))
        ctl=Ref(C.VkVideoCodingControlInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_CODING_CONTROL_INFO_KHR,C_NULL,UInt32(C.VK_VIDEO_CODING_CONTROL_RESET_BIT_KHR)))
        endinfo=Ref(C.VkVideoEndCodingInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_END_CODING_INFO_KHR,C_NULL,UInt32(0)))
        isfirst=dec.isfirst
        # This frame's luma lands in its own device-local LavaArray (cropped to
        # the display size DW×DH), copied out through the high-level VideoImage
        # `record_luma_copy!` — a real image→buffer transfer, no raw copy FFI.
        dst=LavaArray{UInt8,2}(undef,(DW,DH); bq=dec.w.ctx.default_bq, extra_usage=UInt32(Vk.BUFFER_USAGE_TRANSFER_DST_BIT))
        dstbuf=dst.buf[]
        duv = chroma ? LavaArray{UInt8,2}(undef,(DW,DH÷2); bq=dec.w.ctx.default_bq, extra_usage=UInt32(Vk.BUFFER_USAGE_TRANSFER_DST_BIT)) : nothing
        GC.@preserve PIN dst duv cbh outvimg dstvimg mb refpr refri refds decref begref outpr dstpr outri outds setup stdpic soff h264pi decinfo beginfo ctl endinfo imgs begin
            for (slot,_,_) in dpb; lay=imgs[slot][1].layout; if lay[]!=DPBLAYOUT; emit(CB,barrier(imgs[slot][1].image.vks,lay[],DPBLAYOUT)); lay[]=DPBLAYOUT; end; end
            emit(CB,barrier(outimg,outlay[],DPBLAYOUT)); outlay[]=DPBLAYOUT
            # Under DISTINCT the decode target is its own image and needs the
            # decode-DST layout; it comes back round as TRANSFER_SRC from the
            # previous frame's copy. Under COINCIDE `dstlay === outlay`, already
            # DPBLAYOUT, and this is skipped.
            if distinct
                emit(CB,barrier(dstimg,dstlay[],DSTLAYOUT)); dstlay[]=DSTLAYOUT
            end
            ccall(dfp(w,"vkCmdBeginVideoCodingKHR"),Cvoid,(C.VkCommandBuffer,Ptr{Cvoid}),CB,pc(beginfo))
            isfirst && ccall(dfp(w,"vkCmdControlVideoCodingKHR"),Cvoid,(C.VkCommandBuffer,Ptr{Cvoid}),CB,pc(ctl))
            ccall(dfp(w,"vkCmdDecodeVideoKHR"),Cvoid,(C.VkCommandBuffer,Ptr{Cvoid}),CB,pc(decinfo))
            ccall(dfp(w,"vkCmdEndVideoCodingKHR"),Cvoid,(C.VkCommandBuffer,Ptr{Cvoid}),CB,pc(endinfo))
            emitmem(CB,mb)   # this decode's DPB writes → the next decode's reference reads
            emit(CB,barrier(dstimg,dstlay[],TSRC)); dstlay[]=TSRC   # decode target now TRANSFER_SRC
            if distinct
                # vkCmdCopyImageToBuffer needs GRAPHICS/COMPUTE/TRANSFER; a
                # video-decode-only queue family (AMD reports exactly that) cannot
                # record it. Defer to the main queue, drained by `decodemore!` once
                # the decode queue is idle.
                push!(dec.copyq, (dstvimg, dst, duv))
            else
                record_luma_copy!(cbh, dstvimg, dstbuf.buffer, pool_offset(dstbuf) + dst.offset)
                if chroma
                    uvbuf = duv.buf[]
                    record_chroma_copy!(cbh, dstvimg, uvbuf.buffer, pool_offset(uvbuf) + duv.offset)
                end
            end
        end
        push!(dec.pending,(dec.gop,poc,dst,duv))   # completes at the chunk's queue wait
        if isref
            adaptive, mmops = parse_mmco(au[1], sps, pps)
            if adaptive
                # Adaptive marking: MMCO fully controls the DPB (no sliding window).
                MaxFN = 1<<(sps.log2fn+4)
                picnum(fn) = fn > sh.fnum ? fn - MaxFN : fn   # PicNum = FrameNumWrap (frames)
                for (op,arg,_lt) in mmops
                    if op==1                              # unmark a short-term ref by PicNum
                        px = sh.fnum - (arg+1)
                        k = findfirst(e -> picnum(e[2]) == px, dpb)
                        k !== nothing && (push!(freeslots, dpb[k][1]); deleteat!(dpb, k))
                    elseif op==5                          # unmark all references
                        for e in dpb; push!(freeslots, e[1]); end; empty!(dpb)
                    end
                    # ops 2/3/4/6 (long-term refs) unused by the encoders we target
                end
                push!(dpb,(outslot,sh.fnum,poc))
            else
                push!(dpb,(outslot,sh.fnum,poc))
                while length(dpb)>sps.maxref; old=popfirst!(dpb); push!(freeslots,old[1]); end
            end
        else; push!(freeslots,outslot); end
        dec.isfirst=false
        dec.decoded+=1
    end
    return Int(cld(off, dec.bsalign) * dec.bsalign)
end

"""
    decode_h264(ctx, annexb; maxframes=typemax(Int)) -> (w, h, Vector{VideoImage-backed LavaArray})

Hardware-decode an H.264 Annex-B stream on the GPU. Each decoded frame's luma (Y)
plane is left GPU-resident in its own device-local `LavaArray{UInt8,2}` (cropped to
the display size, copied out through the high-level [`VideoImage`] `copyto!` path —
no host round-trip). Frames are returned in DISPLAY order — (GOP, POC):
pic_order_cnt resets to 0 at every IDR, so POC alone would interleave frames from
different coded video sequences. The video *codec* commands (session/decode/DPB) are
raw VulkanCore FFI — VK.jl generates no wrappers for the video codec API — but
the frame images and the transfer are ordinary VK.jl abstractions.

The batch form of [`H264Decoder`](@ref): feed everything, drain once (`holdback =
typemax` ⇒ emit only the final full display-order sort), free the session.
"""
function decode_h264(ctx, annexb::Vector{UInt8}; maxframes::Int=typemax(Int), chroma::Bool=false)
    Lava = parentmodule(@__MODULE__)
    dec = H264Decoder(ctx, annexb; chroma)
    try
        feed!(dec, annexb)
        length(dec.aus) > maxframes && resize!(dec.aus, maxframes)
        outs = decodemore!(dec, typemax(Int); holdback = typemax(Int))
        ys  = LavaArray{UInt8,2}[o[1] for o in outs]
        uvs = chroma ? LavaArray{UInt8,2}[o[2] for o in outs] : LavaArray{UInt8,2}[]
        return (dec.DW, dec.DH, ys, uvs)
    finally
        close(dec)
    end
end
include("video_h265.jl")
end # module
