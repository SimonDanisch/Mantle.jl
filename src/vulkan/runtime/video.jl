# Hardware H.264/H.265 decode from Annex-B to GPU-resident NV12. Session and
# stream state are shared (video_common.jl); each codec builds its own picture
# and reference structures and applies its own reference marking.

"""NV12 GPU image with coded/display extents. `size` is the cropped display size;
`copyto!(::LavaArray{UInt8}, img)` and `Array(img)` read its luma plane."""
mutable struct VideoImage
    image::VK.Image
    memory::VK.DeviceMemory
    codedwidth::Int
    codedheight::Int
    width::Int      # display (cropped) width
    height::Int     # display (cropped) height
    format::VK.Format
    # Current `VkImageLayout` as raw `UInt32`, shared by the raw decode barriers
    # and the VK.jl copy path.
    layout::Base.RefValue{UInt32}
end
Base.size(img::VideoImage) = (img.width, img.height)
Base.eltype(::VideoImage) = UInt8
Base.show(io::IO, img::VideoImage) = print(io, "VideoImage($(img.width)×$(img.height), NV12)")

# Record a copy of `img`'s luma plane (display size, tightly packed) to byte
# `offset` of `buffer`. `img` must be TRANSFER_SRC_OPTIMAL when `cb` executes.
function record_luma_copy!(cb, img::VideoImage, buffer::VK.Buffer, offset::Integer)
    VK.cmd_copy_image_to_buffer(cb, img.image,
        VK.IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer,
        [VK.BufferImageCopy(UInt64(offset), 0, 0,
            VK.ImageSubresourceLayers(VK.IMAGE_ASPECT_PLANE_0_BIT, 0, 0, 1),
            VK.Offset3D(0, 0, 0), VK.Extent3D(img.width, img.height, 1))])
    return cb
end

# Record a copy of `img`'s interleaved U/V plane to byte `offset` of `buffer`:
# `width × (height ÷ 2)` bytes, tightly packed. Same layout requirement as luma.
function record_chroma_copy!(cb, img::VideoImage, buffer::VK.Buffer, offset::Integer)
    VK.cmd_copy_image_to_buffer(cb, img.image,
        VK.IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer,
        [VK.BufferImageCopy(UInt64(offset), 0, 0,
            VK.ImageSubresourceLayers(VK.IMAGE_ASPECT_PLANE_1_BIT, 0, 0, 1),
            VK.Offset3D(0, 0, 0), VK.Extent3D(img.width ÷ 2, img.height ÷ 2, 1))])
    return cb
end

# `img.layout` is updated when the barrier is recorded, not when it executes.
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

Copy the frame's luma plane into `dst`, which must hold at least
`prod(size(img))` bytes and have TRANSFER_DST usage. Submits on the video-decode
queue and waits for it to go idle.
"""
function Base.copyto!(dst::LavaArray{UInt8}, img::VideoImage)
    need = img.width * img.height
    length(dst) >= need || error("VideoImage copyto!: dst holds $(length(dst)) < $need bytes")
    mbuf = dst.buf[]
    ctx = mbuf.ctx::VkContext
    dev = ctx.device
    pool = @vk_checked ctx "video_copy_pool" VK.create_command_pool(dev, ctx.video_decode_queue_family_index)
    cb = first(@vk_checked ctx "video_copy_cb" VK.allocate_command_buffers(dev,
        VK.CommandBufferAllocateInfo(pool, VK.COMMAND_BUFFER_LEVEL_PRIMARY, 1)))
    @vk_checked ctx "video_copy_begin" VK.begin_command_buffer(cb, VK.CommandBufferBeginInfo())
    transition_transfer_src!(cb, img)
    record_luma_copy!(cb, img, mbuf.buffer, pool_offset(mbuf) + dst.offset)
    @vk_checked ctx "video_copy_end" VK.end_command_buffer(cb)
    @vk_checked ctx "video_copy_submit" VK.queue_submit(ctx.video_decode_queue, [VK.SubmitInfo([], [], [cb], [])])
    @vk_checked ctx "video_copy_wait" VK.queue_wait_idle(ctx.video_decode_queue)
    return dst
end

"""
    Array(img::VideoImage) -> Matrix{UInt8}

Download the frame's luma plane as a `width × height` `Matrix{UInt8}`.
"""
function Base.Array(img::VideoImage)
    # `VideoImage` carries no device; assumes it was decoded on the default
    # context. ambient-allocation-ok
    dst = LavaArray{UInt8,2}(undef, (img.width, img.height);
                             extra_usage = UInt32(VK.BUFFER_USAGE_TRANSFER_DST_BIT))
    copyto!(dst, img)
    return Array(dst)
end

module VideoDecode
# Use the Vulkan package inside this submodule; Mantle.Vulkan names an API marker.
import Vulkan as VK
const Vk = VK
const C = Vk.VkCore

import ..VideoImage, ..LavaArray, ..pool_offset,
       ..record_luma_copy!, ..record_chroma_copy!,
       ..oneshot!, ..waitfor!

import ..feed!, ..decodemore!, ..remaining

# Parent names defined after this include, such as `alloc_image_memory`.
const MANTLE = parentmodule(@__MODULE__)

"""Throw unless `r` is `VK_SUCCESS`."""
@inline function vkchk(r::Int32, what::AbstractString)
    r == Int32(0) && return r
    error("video decode: $what returned VkResult $r")
end

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
split_nals(d) = split_nals(d, nal -> (Int(nal[1]&0x1f), Int((nal[1]>>5)&3), nal))

function split_nals(d, header)
    starts = Int[]
    i = 1
    while i <= length(d)-3
        if d[i]==0 && d[i+1]==0 && d[i+2]==1
            push!(starts,i+3)
            i += 3
        else
            i += 1
        end
    end
    return map(eachindex(starts)) do k
        first = starts[k]
        last = k < length(starts) ? starts[k+1]-4 : length(d)
        while last > first && d[last]==0
            last -= 1
        end
        header(d[first:last])
    end
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
# Slice header fields needed for POC and the DPB; assumes frame_mbs_only.
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

# H.264 7.3.3.3: parse MMCO reference marking after the optional slice fields.
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

dfp(ctx,n) = Vk.function_pointer(ctx.device,n)
rp(x) = Base.unsafe_convert(Ptr{eltype(x)},x)
pc(x) = Ptr{Cvoid}(rp(x))

abstract type VideoDecoder end
include("video_common.jl")

"""H.264 decode capability flags: `0x1` DPB_AND_OUTPUT_COINCIDE, `0x2` DPB_AND_OUTPUT_DISTINCT."""
decode_capability_flags(ctx) =
    videocaps(ctx, decoderprofile(H264Decoder), codeccapabilities(H264Decoder)).flags

"""H.264 bitstream alignment and DPB/reference limits reported by the device."""
function video_capabilities(ctx)
    c = videocaps(ctx, decoderprofile(H264Decoder), codeccapabilities(H264Decoder))
    return (minBitstreamBufferOffsetAlignment=c.offset, minBitstreamBufferSizeAlignment=c.size,
            maxDpbSlots=c.slots, maxActiveReferencePictures=c.refs)
end

"""Whether H.264 may use the DPB image as its decode target."""
decode_coincide_supported(ctx) = (decode_capability_flags(ctx) & UInt32(0x1)) != 0

"""Whether H.264 supports either the shared-image or separate-image output path."""
decode_supported(ctx) = (decode_capability_flags(ctx) & UInt32(0x3)) != 0

function video_profile(w)
    hp=Ref(C.VkVideoDecodeH264ProfileInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_PROFILE_INFO_KHR,C_NULL,C.STD_VIDEO_H264_PROFILE_IDC_HIGH,C.VK_VIDEO_DECODE_H264_PICTURE_LAYOUT_PROGRESSIVE_KHR))
    pr=Ref(C.VkVideoProfileInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_PROFILE_INFO_KHR, Ptr{Cvoid}(rp(hp)), C.VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR, UInt32(C.VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR), UInt32(C.VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR), UInt32(C.VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR)))
    pl=Ref(C.VkVideoProfileListInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_PROFILE_LIST_INFO_KHR, C_NULL, UInt32(1), rp(pr)))
    (hp,pr,pl)
end

"""Common power-of-two alignment for bitstream offsets and sizes; reject invalid limits."""
function bitstreamalignment(offset::Integer, size::Integer)
    a = Int(max(offset, size, 1))
    ispow2(a) && a % max(offset, 1) == 0 && a % max(size, 1) == 0 || error(
        "video decode: bitstream alignments $offset (offset) and $size (size) are not " *
        "powers of two, so no single alignment satisfies both")
    return a
end

decoderprofile(::Type{H264Decoder}) = video_profile(nothing)
codeccapabilities(::Type{H264Decoder}) = C.VkVideoDecodeH264CapabilitiesKHR(
    C.VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_CAPABILITIES_KHR, C_NULL,
    C.StdVideoH264LevelIdc(0), C.VkOffset2D(0,0))

function decoderconfig(::Type{H264Decoder}, data)
    nals = split_nals(data)
    sps = parse_sps(first(n[3] for n in nals if n[1]==7))
    pps = parse_pps(first(n[3] for n in nals if n[1]==8))
    sps.chroma == 1 || error("decode_h264: only 4:2:0 chroma is supported")
    sps.maxref >= 2 || error("decode_h264: single-reference/all-intra streams are not supported")
    coded = ((sps.wmbs+1)*16, (sps.hmap+1)*16)
    return (; sps, pps, coded, display=(coded[1]-2*sps.cr, coded[2]-2*sps.cb),
            refs=sps.maxref, state=H264PictureOrder(0,0))
end

function distinctoutput(::Type{H264Decoder}, flags)
    flags & UInt32(3) != 0 || error("H264Decoder: no supported DPB/output layout (flags=$flags)")
    return flags & UInt32(1) == 0
end

videoslices(dec::H264Decoder, data) =
    ((parse_slice(n[3],dec.sps).first_mb==0, n[3]) for n in split_nals(data) if n[1] in (1,5))

"""Open a persistent H.264 session from the SPS/PPS in `data`. Feed Annex-B chunks
beginning at an IDR; close when done."""
H264Decoder(ctx, data::AbstractVector{UInt8}; chroma::Bool=false) =
    videodecoder(H264Decoder, ctx, data; chroma)

function codecparameters(dec::H264Decoder)
    pin(x) = (push!(dec.PIN,x); x)
    sps = pin(Ref(std_sps(dec.sps)))
    pps = pin(Ref(std_pps(dec.pps)))
    add = pin(Ref(C.VkVideoDecodeH264SessionParametersAddInfoKHR(
        C.VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_SESSION_PARAMETERS_ADD_INFO_KHR,
        C_NULL, UInt32(1), rp(sps), UInt32(1), rp(pps))))
    return pin(Ref(C.VkVideoDecodeH264SessionParametersCreateInfoKHR(
        C.VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_SESSION_PARAMETERS_CREATE_INFO_KHR,
        C_NULL, UInt32(1), UInt32(1), rp(add))))
end

"""Record the decode of one AU at `bufbase` in the bitstream buffer and its output
copy (queued under DISTINCT), and append its display-order entry. Return the
aligned bytes used. The caller submits and waits."""
function decodeau!(dec::H264Decoder, au, bufbase::Integer, slot::Integer=1)
    w=dec.ctx; sps=dec.sps; pps=dec.pps
    SESSION=dec.SESSION; PARAMS=dec.PARAMS; PIN=dec.PIN; imgs=dec.imgs
    bbuf=dec.bbuf; bmap=dec.bmap; cbh=dec.cbh; CB=dec.cbh.vks
    CW=dec.CW; CH=dec.CH; DW=dec.DW; DH=dec.DH; chroma=dec.chroma
    dpb=dec.dpb; freeslots=dec.freeslots
    # A decode writes reference pixels the next decode reads, even when no layout changes.
    mb=Ref(C.VkMemoryBarrier(C.VK_STRUCTURE_TYPE_MEMORY_BARRIER,C_NULL,UInt32(C.VK_ACCESS_MEMORY_WRITE_BIT),UInt32(C.VK_ACCESS_MEMORY_READ_BIT)|UInt32(C.VK_ACCESS_MEMORY_WRITE_BIT)))
    DPBLAYOUT=UInt32(C.VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR); TSRC=UInt32(C.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
    DSTLAYOUT=UInt32(C.VK_IMAGE_LAYOUT_VIDEO_DECODE_DST_KHR)   # DISTINCT decode target
    begin
        sh=parse_slice(au[1],sps)
        if sh.idr
            for (slot,_,_) in dpb; push!(freeslots,slot); end; empty!(dpb)
            dec.prevmsb=0; dec.prevlsb=0; dec.gop+=1
            dec.codec = H264PictureOrder(0,0)
        end
        if sps.poct==0
            # 8.2.1.1
            pmsb = sh.idr ? 0 : picturemsb(dec,sh.poclsb)
            poc=pmsb+sh.poclsb
            if sh.nrid!=0; dec.prevmsb=pmsb; dec.prevlsb=sh.poclsb; end
        elseif sps.poct==2
            # H.264 8.2.1.3: POC from frame_num and its wrap offset.
            maxfn = 1 << (sps.log2fn + 4)
            fnoff = sh.idr ? 0 :
                    (dec.codec.frame > sh.fnum ? dec.codec.offset + maxfn : dec.codec.offset)
            poc = sh.idr ? 0 :
                  sh.nrid == 0 ? 2 * (fnoff + sh.fnum) - 1 : 2 * (fnoff + sh.fnum)
            dec.codec = H264PictureOrder(sh.fnum,fnoff)
        else
            # Type 1 requires SPS offsets this parser does not retain.
            error("decode_h264: pic_order_cnt_type $(sps.poct) is not implemented")
        end
        isref = sh.nrid!=0
        outslot=popfirst!(freeslots)
        outvimg,outviewhl=imgs[outslot]     # DPB slot of the setup reference picture
        outimg=outvimg.image.vks; outview=outviewhl.vks; outlay=outvimg.layout
        # Decode target; under COINCIDE the DPB slot itself.
        distinct = dec.mkoutimg !== nothing
        if distinct
            while length(dec.outimgs) < slot; push!(dec.outimgs, dec.mkoutimg()); end
        end
        dstvimg,dstviewhl = distinct ? dec.outimgs[slot] : (outvimg,outviewhl)
        dstimg=dstvimg.image.vks; dstview=dstviewhl.vks; dstlay=dstvimg.layout
        sliceoffs, bitrange = uploadbitstream!(bmap+bufbase, au, dec.bsalign)
        # Reference slots of the active DPB.
        refpr=Ref{C.VkVideoPictureResourceInfoKHR}[]; refri=Ref{C.StdVideoDecodeH264ReferenceInfo}[]; refds=Ref{C.VkVideoDecodeH264DpbSlotInfoKHR}[]
        decref=C.VkVideoReferenceSlotInfoKHR[]; begref=C.VkVideoReferenceSlotInfoKHR[]
        for (slot,fn,pc0) in dpb
            rv=imgs[slot][2].vks
            pr=Ref(picture_resource(rv,CW,CH)); ri=Ref(C.StdVideoDecodeH264ReferenceInfo(C.StdVideoDecodeH264ReferenceInfoFlags(pbytes(UInt32(0))),UInt16(fn),UInt16(0),(Int32(pc0),Int32(pc0))))
            ds=Ref(C.VkVideoDecodeH264DpbSlotInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_DPB_SLOT_INFO_KHR,C_NULL,rp(ri)))
            push!(refpr,pr);push!(refri,ri);push!(refds,ds)
            push!(decref,C.VkVideoReferenceSlotInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR,pc(ds),Int32(slot-1),rp(pr)))
            push!(begref,C.VkVideoReferenceSlotInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR,C_NULL,Int32(slot-1),rp(pr)))
        end
        # `outpr` names the DPB slot (setup reference + the slot activated in
        # BeginCoding); `dstpr` names the decode target. Equal under COINCIDE.
        outpr=Ref(picture_resource(outview,CW,CH))
        dstpr=Ref(picture_resource(dstview,CW,CH))
        outri=Ref(C.StdVideoDecodeH264ReferenceInfo(C.StdVideoDecodeH264ReferenceInfoFlags(pbytes(UInt32(0))),UInt16(sh.fnum),UInt16(0),(Int32(poc),Int32(poc))))
        outds=Ref(C.VkVideoDecodeH264DpbSlotInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_DPB_SLOT_INFO_KHR,C_NULL,rp(outri)))
        push!(begref,C.VkVideoReferenceSlotInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR,C_NULL,Int32(-1),rp(outpr)))
        setup=Ref(C.VkVideoReferenceSlotInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR,pc(outds),Int32(outslot-1),rp(outpr)))
        picflags=packflags((0, sh.stype==2 ? 1 : 0, sh.idr ? 1 : 0, 0, isref ? 1 : 0, 0))
        stdpic=Ref(C.StdVideoDecodeH264PictureInfo(C.StdVideoDecodeH264PictureInfoFlags(pbytes(picflags)),UInt8(0),UInt8(0),UInt8(0),UInt8(0),UInt16(sh.fnum),UInt16(sh.idrid),(Int32(poc),Int32(poc))))
        soff=sliceoffs
        h264pi=Ref(C.VkVideoDecodeH264PictureInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_PICTURE_INFO_KHR,C_NULL,rp(stdpic),UInt32(length(soff)),pointer(soff)))
        decinfo=Ref(C.VkVideoDecodeInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_DECODE_INFO_KHR,pc(h264pi),UInt32(0),bbuf,UInt64(bufbase),UInt64(bitrange),dstpr[],rp(setup),UInt32(length(decref)),isempty(decref) ? Ptr{C.VkVideoReferenceSlotInfoKHR}(C_NULL) : pointer(decref)))
        beginfo=Ref(C.VkVideoBeginCodingInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_BEGIN_CODING_INFO_KHR,C_NULL,UInt32(0),SESSION,PARAMS,UInt32(length(begref)),pointer(begref)))
        ctl=Ref(C.VkVideoCodingControlInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_CODING_CONTROL_INFO_KHR,C_NULL,UInt32(C.VK_VIDEO_CODING_CONTROL_RESET_BIT_KHR)))
        endinfo=Ref(C.VkVideoEndCodingInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_END_CODING_INFO_KHR,C_NULL,UInt32(0)))
        isfirst=dec.isfirst
        dst=LavaArray{UInt8,2}(undef,(DW,DH); bq=dec.ctx.default_bq, extra_usage=UInt32(Vk.BUFFER_USAGE_TRANSFER_DST_BIT))
        dstbuf=dst.buf[]
        duv = chroma ? LavaArray{UInt8,2}(undef,(DW,DH÷2); bq=dec.ctx.default_bq, extra_usage=UInt32(Vk.BUFFER_USAGE_TRANSFER_DST_BIT)) : nothing
        GC.@preserve PIN dst duv cbh outvimg dstvimg mb refpr refri refds decref begref outpr dstpr outri outds setup stdpic soff h264pi decinfo beginfo ctl endinfo imgs begin
            for (slot,_,_) in dpb; lay=imgs[slot][1].layout; if lay[]!=DPBLAYOUT; videobarrier!(w,CB;image=videolayoutbarrier(imgs[slot][1].image.vks,lay[],DPBLAYOUT)); lay[]=DPBLAYOUT; end; end
            videobarrier!(w,CB;image=videolayoutbarrier(outimg,outlay[],DPBLAYOUT)); outlay[]=DPBLAYOUT
            # DISTINCT has a separate target; COINCIDE already uses DPB layout.
            if distinct
                videobarrier!(w,CB;image=videolayoutbarrier(dstimg,dstlay[],DSTLAYOUT)); dstlay[]=DSTLAYOUT
            end
            ccall(dfp(w,"vkCmdBeginVideoCodingKHR"),Cvoid,(C.VkCommandBuffer,Ptr{Cvoid}),CB,pc(beginfo))
            isfirst && ccall(dfp(w,"vkCmdControlVideoCodingKHR"),Cvoid,(C.VkCommandBuffer,Ptr{Cvoid}),CB,pc(ctl))
            ccall(dfp(w,"vkCmdDecodeVideoKHR"),Cvoid,(C.VkCommandBuffer,Ptr{Cvoid}),CB,pc(decinfo))
            ccall(dfp(w,"vkCmdEndVideoCodingKHR"),Cvoid,(C.VkCommandBuffer,Ptr{Cvoid}),CB,pc(endinfo))
            videobarrier!(w,CB;memory=mb)   # DPB writes before the next decode's reads
            videobarrier!(w,CB;image=videolayoutbarrier(dstimg,dstlay[],TSRC)); dstlay[]=TSRC
            if distinct
                # A video-only queue cannot copy to buffers; DISTINCT copies run on the main queue.
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
                    # Long-term ops 2/3/4/6 are ignored; target encoders do not emit them.
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
    return bitrange
end

"""Decode H.264 in display order, returning `(width, height, ys, uvs)` of GPU arrays;
`uvs` is empty unless `chroma=true`. Closes the session on return."""
decode_h264(ctx, annexb::Vector{UInt8}; kw...) = decodevideo(H264Decoder,ctx,annexb;kw...)

include("video_h265.jl")
end # module
