# H.265 Main, 8-bit 4:2:0 frame pictures; feeds start at a closed IRAP GOP.
# Short-term reference sets and tiles are supported; scaling lists and long-term
# references are refused.

function std_strps(s)
    fl = packflags((s.pred ? 1 : 0, s.sign))
    d0 = zeros(UInt16, 16); d1 = zeros(UInt16, 16)
    prev = 0
    for (i, d) in enumerate(s.s0); d0[i] = UInt16(prev - d - 1); prev = d; end
    prev = 0
    for (i, d) in enumerate(s.s1); d1[i] = UInt16(d - prev - 1); prev = d; end
    u0 = UInt16(0); u1 = UInt16(0)
    for (i, u) in enumerate(s.u0); u == 1 && (u0 |= UInt16(1) << (i - 1)); end
    for (i, u) in enumerate(s.u1); u == 1 && (u1 |= UInt16(1) << (i - 1)); end
    C.StdVideoH265ShortTermRefPicSet(C.StdVideoH265ShortTermRefPicSetFlags(pbytes(fl)),
        UInt32(0), UInt16(0), UInt16(s.absd - 1), u0 | (u1 << length(s.u0)), u0, u1,
        UInt16(0), UInt8(0), UInt8(0), UInt8(length(s.s0)), UInt8(length(s.s1)),
        ntuple(i -> d0[i], 16), ntuple(i -> d1[i], 16))
end

function std_ptl(p)
    fl = packflags((p.tier, p.prog, p.inter, p.nonpacked, p.frameonly))
    C.StdVideoH265ProfileTierLevel(C.StdVideoH265ProfileTierLevelFlags(pbytes(fl)),
        C.StdVideoH265ProfileIdc(p.profile), C.StdVideoH265LevelIdc(levelidx265(p.level)))
end

# general_level_idc is 30 × the level number; the Std enum indexes this list.
const H265LEVELS = [30, 60, 63, 90, 93, 120, 123, 150, 153, 156, 180, 183, 186]
levelidx265(lv) = something(findfirst(==(lv), H265LEVELS), length(H265LEVELS)) - 1

function std_dpbm(sps)
    lat = zeros(UInt32, 7); buf = zeros(UInt8, 7); reo = zeros(UInt8, 7)
    i = sps.maxsub + 1
    lat[i] = UInt32(sps.latency); buf[i] = UInt8(sps.dpbsize); reo[i] = UInt8(sps.reorder)
    C.StdVideoH265DecPicBufMgr(ntuple(k -> lat[k], 7), ntuple(k -> buf[k], 7), ntuple(k -> reo[k], 7))
end

function std_sps265(sps, rPTL, rDPBM, rSTRPS)
    fl = packflags((sps.tnest, sps.sepcol, sps.confwin, sps.subord, sps.scal, 0,
                    sps.amp, sps.sao, sps.pcm, sps.pcmloop, sps.ltref, sps.tmvp,
                    sps.strongintra, sps.vui, 0))
    C.StdVideoH265SequenceParameterSet(C.StdVideoH265SpsFlags(pbytes(fl)),
        C.StdVideoH265ChromaFormatIdc(sps.chroma), UInt32(sps.w), UInt32(sps.h),
        UInt8(sps.vps_id), UInt8(sps.maxsub), UInt8(sps.sps_id),
        UInt8(sps.bdl), UInt8(sps.bdc), UInt8(sps.log2poc),
        UInt8(sps.log2mincb), UInt8(sps.log2diffcb), UInt8(sps.log2mintb), UInt8(sps.log2difftb),
        UInt8(sps.maxtdinter), UInt8(sps.maxtdintra),
        UInt8(sps.nsets), UInt8(0),
        UInt8(sps.pcm == 1 ? sps.pcmbl - 1 : 0), UInt8(sps.pcm == 1 ? sps.pcmbc - 1 : 0),
        UInt8(sps.pcm == 1 ? sps.pcmmin : 0), UInt8(sps.pcm == 1 ? sps.pcmdiff : 0),
        UInt8(0), UInt8(0), UInt8(0), UInt8(0), UInt8(0), UInt8(0),
        UInt32(sps.cl), UInt32(sps.cr), UInt32(sps.ct), UInt32(sps.cb),
        rp(rPTL), rp(rDPBM), Ptr{C.StdVideoH265ScalingLists}(C_NULL),
        length(rSTRPS[]) == 0 ? Ptr{C.StdVideoH265ShortTermRefPicSet}(C_NULL) : pointer(rSTRPS[]),
        Ptr{C.StdVideoH265LongTermRefPicsSps}(C_NULL),
        Ptr{C.StdVideoH265SequenceParameterSetVui}(C_NULL),
        Ptr{C.StdVideoH265PredictorPaletteEntries}(C_NULL))
end

function std_vps265(sps, rDPBM, rPTL)
    fl = packflags((sps.tnest, 1, 0, 0))
    C.StdVideoH265VideoParameterSet(C.StdVideoH265VpsFlags(pbytes(fl)),
        UInt8(sps.vps_id), UInt8(sps.maxsub), UInt8(0), UInt8(0),
        UInt32(0), UInt32(0), UInt32(0), UInt32(0),
        rp(rDPBM), Ptr{C.StdVideoH265HrdParameters}(C_NULL), rp(rPTL))
end

function std_pps265(pps, sps)
    fl = packflags((pps.depslice, pps.outflag, pps.signhide, pps.cabacinit, pps.cintra,
                    pps.tskip, pps.cuqp, pps.slqp, pps.wp, pps.wbp, pps.tqbypass,
                    pps.tiles, pps.entsync, pps.uniform, 1, pps.lfslices,
                    pps.dbctl, pps.dbovr, pps.dbdis, 0, pps.listmod, pps.shext, 0))
    C.StdVideoH265PictureParameterSet(C.StdVideoH265PpsFlags(pbytes(fl)),
        UInt8(pps.pps_id), UInt8(pps.sps_id), UInt8(sps.vps_id), UInt8(pps.extrabits),
        UInt8(pps.nl0), UInt8(pps.nl1), Int8(pps.initqp), UInt8(pps.cuqpdepth),
        Int8(pps.cbqp), Int8(pps.crqp), Int8(pps.beta), Int8(pps.tc),
        UInt8(pps.log2pml), UInt8(0), UInt8(0), UInt8(0),
        ntuple(_ -> Int8(0), 6), ntuple(_ -> Int8(0), 6),
        UInt8(0), UInt8(0), Int8(0), Int8(0), Int8(0), UInt8(0), UInt8(0), UInt8(0),
        UInt8(pps.ntilec), UInt8(pps.ntiler), UInt8(0), UInt8(0),
        # Zero under uniform spacing, where the driver derives the geometry from
        # the picture size and the tile counts; the explicit widths otherwise.
        ntuple(i -> i <= length(pps.colw) ? pps.colw[i] : UInt16(0), 19),
        ntuple(i -> i <= length(pps.rowh) ? pps.rowh[i] : UInt16(0), 21), UInt32(0),
        Ptr{C.StdVideoH265ScalingLists}(C_NULL),
        Ptr{C.StdVideoH265PredictorPaletteEntries}(C_NULL))
end

function video_profile_h265(w)
    hp = Ref(C.VkVideoDecodeH265ProfileInfoKHR(
        C.VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_PROFILE_INFO_KHR, C_NULL,
        C.STD_VIDEO_H265_PROFILE_IDC_MAIN))
    pr = Ref(C.VkVideoProfileInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_PROFILE_INFO_KHR,
        Ptr{Cvoid}(rp(hp)), C.VK_VIDEO_CODEC_OPERATION_DECODE_H265_BIT_KHR,
        UInt32(C.VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR),
        UInt32(C.VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR),
        UInt32(C.VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR)))
    pl = Ref(C.VkVideoProfileListInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_PROFILE_LIST_INFO_KHR,
        C_NULL, UInt32(1), rp(pr)))
    (hp, pr, pl)
end

decoderprofile(::Type{H265Decoder}) = video_profile_h265(nothing)
codeccapabilities(::Type{H265Decoder}) = C.VkVideoDecodeH265CapabilitiesKHR(
    C.VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_CAPABILITIES_KHR, C_NULL, C.StdVideoH265LevelIdc(0))

function decoderconfig(::Type{H265Decoder}, data)
    nals = split_nals_h265(data)
    sps = parse_sps_h265(first(n[2] for n in nals if n[1]==33))
    pps = parse_pps_h265(first(n[2] for n in nals if n[1]==34))
    sps.chroma == 1 || error("decode_h265: only 4:2:0 chroma is supported")
    sps.bdl == sps.bdc == 0 || error("decode_h265: only 8-bit is supported")
    return (; sps, pps, coded=(sps.w,sps.h),
            display=(sps.w-2*(sps.cl+sps.cr), sps.h-2*(sps.ct+sps.cb)),
            refs=max(sps.dpbsize,1), state=nothing)
end

function distinctoutput(::Type{H265Decoder}, flags)
    flags & UInt32(1) != 0 || error(
        "decode_h265 requires DPB_AND_OUTPUT_COINCIDE (device flags=$flags)")
    return false
end

"""Open a persistent H.265 Main 8-bit 4:2:0 session from the SPS/PPS in `data`. Feed
closed GOPs starting at an IRAP; close when done."""
H265Decoder(ctx, data::AbstractVector{UInt8}; chroma::Bool=false) =
    videodecoder(H265Decoder, ctx, data; chroma)

videoslices(::H265Decoder, data) =
    (((nal[3] >> 7) & 0x01 == 1, nal) for (t,nal) in split_nals_h265(data) if t <= H265_VCL_MAX)

function codecparameters(dec::H265Decoder)
    pin(x) = (push!(dec.PIN,x); x)
    sps, pps = dec.sps, dec.pps
    ptl = pin(Ref(std_ptl(sps.ptl)))
    dpbm = pin(Ref(std_dpbm(sps)))
    strps = pin(Ref([std_strps(s) for s in sps.sets]))
    vps = pin(Ref(std_vps265(sps,dpbm,ptl)))
    stdsps = pin(Ref(std_sps265(sps,ptl,dpbm,strps)))
    stdpps = pin(Ref(std_pps265(pps,sps)))
    add = pin(Ref(C.VkVideoDecodeH265SessionParametersAddInfoKHR(
        C.VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_SESSION_PARAMETERS_ADD_INFO_KHR, C_NULL,
        UInt32(1), rp(vps), UInt32(1), rp(stdsps), UInt32(1), rp(stdpps))))
    return pin(Ref(C.VkVideoDecodeH265SessionParametersCreateInfoKHR(
        C.VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_SESSION_PARAMETERS_CREATE_INFO_KHR,
        C_NULL, UInt32(1), UInt32(1), UInt32(1), rp(add))))
end

# The decode-target index is unused: H.265 requires COINCIDE, so the target is the DPB slot.
function decodeau!(dec::H265Decoder, au, bufbase::Integer, ::Integer = 1)
    sps = dec.sps; pps = dec.pps; imgs = dec.imgs
    bbuf = dec.bbuf; bmap = dec.bmap
    CW = dec.CW; CH = dec.CH
    dpb = dec.dpb; freeslots = dec.freeslots
    sh = parse_slice_h265(au[1], sps, pps)
    if sh.idr
        for (slot, _) in dpb; push!(freeslots, slot); end; empty!(dpb)
        dec.prevmsb = 0; dec.prevlsb = 0; dec.gop += 1
    end
    # POC (8.3.1): the lsb/msb wrap of H.264 POC type 0.
    if sh.idr
        poc = 0
    else
        pmsb = picturemsb(dec,sh.poclsb)
        poc = pmsb + sh.poclsb
        dec.prevmsb = pmsb; dec.prevlsb = sh.poclsb
    end
    # The RPS lists every reference that stays; free all other slots before decoding.
    rpspocs = Int[]
    if sh.rps !== nothing
        append!(rpspocs, poc + d for d in sh.rps.s0)
        append!(rpspocs, poc + d for d in sh.rps.s1)
    end
    keep = Set(rpspocs)
    filter!(dpb) do (slot, p)
        p in keep && return true
        push!(freeslots, slot); false
    end
    outslot = popfirst!(freeslots)
    outvimg, outviewhl = imgs[outslot]
    outview = outviewhl.vks
    sliceoffs, bitrange = uploadbitstream!(bmap+bufbase, au, dec.bsalign)
    # Reference slots in DPB order. The Std picture info's RPS lists hold DPB slot
    # indices, mapped from POC by `pocpos`.
    refpr = Ref{C.VkVideoPictureResourceInfoKHR}[]; refri = Ref{C.StdVideoDecodeH265ReferenceInfo}[]; refds = Ref{C.VkVideoDecodeH265DpbSlotInfoKHR}[]
    decref = C.VkVideoReferenceSlotInfoKHR[]; begref = C.VkVideoReferenceSlotInfoKHR[]
    pocpos = Dict{Int, Int}()
    for (k, (slot, p)) in enumerate(dpb)
        rv = imgs[slot][2].vks
        pr = Ref(picture_resource(rv,CW,CH))
        ri = Ref(C.StdVideoDecodeH265ReferenceInfo(C.StdVideoDecodeH265ReferenceInfoFlags(pbytes(UInt32(0))), Int32(p)))
        ds = Ref(C.VkVideoDecodeH265DpbSlotInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_DPB_SLOT_INFO_KHR, C_NULL, rp(ri)))
        push!(refpr, pr); push!(refri, ri); push!(refds, ds)
        push!(decref, C.VkVideoReferenceSlotInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR, pc(ds), Int32(slot - 1), rp(pr)))
        push!(begref, C.VkVideoReferenceSlotInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR, C_NULL, Int32(slot - 1), rp(pr)))
        pocpos[p] = slot - 1
    end
    fill8(pocs) = ntuple(i -> UInt8(i <= length(pocs) ? get(pocpos, pocs[i], 0xff) : 0xff), 8)
    before = sh.rps === nothing ? Int[] : [poc + d for (d, u) in zip(sh.rps.s0, sh.rps.u0) if u == 1]
    after  = sh.rps === nothing ? Int[] : [poc + d for (d, u) in zip(sh.rps.s1, sh.rps.u1) if u == 1]
    outpr = Ref(picture_resource(outview,CW,CH))
    outri = Ref(C.StdVideoDecodeH265ReferenceInfo(C.StdVideoDecodeH265ReferenceInfoFlags(pbytes(UInt32(0))), Int32(poc)))
    outds = Ref(C.VkVideoDecodeH265DpbSlotInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_DPB_SLOT_INFO_KHR, C_NULL, rp(outri)))
    push!(begref, C.VkVideoReferenceSlotInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR, C_NULL, Int32(-1), rp(outpr)))
    setup = Ref(C.VkVideoReferenceSlotInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR, pc(outds), Int32(outslot - 1), rp(outpr)))
    picflags = packflags((sh.irap ? 1 : 0, sh.idr ? 1 : 0, sh.isref ? 1 : 0, sh.spsflag))
    stdpic = Ref(C.StdVideoDecodeH265PictureInfo(C.StdVideoDecodeH265PictureInfoFlags(pbytes(picflags)),
        UInt8(sps.vps_id), UInt8(pps.sps_id), UInt8(pps.pps_id),
        UInt8(sh.refnd), Int32(poc), UInt16(sh.rpsbits), UInt16(0),
        fill8(before), fill8(after), ntuple(_ -> UInt8(0xff), 8)))
    soff = sliceoffs
    h265pi = Ref(C.VkVideoDecodeH265PictureInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_PICTURE_INFO_KHR, C_NULL, rp(stdpic), UInt32(length(soff)), pointer(soff)))
    decinfo = Ref(C.VkVideoDecodeInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_DECODE_INFO_KHR, pc(h265pi), UInt32(0), bbuf, UInt64(bufbase), UInt64(bitrange), outpr[], rp(setup), UInt32(length(decref)), isempty(decref) ? Ptr{C.VkVideoReferenceSlotInfoKHR}(C_NULL) : pointer(decref)))
    dst, duv = recorddecode!(dec, (slot for (slot, _) in dpb), outvimg, outvimg, decinfo, begref,
                             (refpr, refri, refds, decref, outpr, outri, outds, setup, stdpic, soff, h265pi))
    push!(dec.pending, (dec.gop, poc, dst, duv))
    if sh.isref
        push!(dpb, (outslot, poc))
    else
        push!(freeslots, outslot)
    end
    return bitrange
end

"""
    decode_h265(ctx, annexb; maxframes=typemax(Int), chroma=false)

Batch form of [`H265Decoder`](@ref), like [`decode_h264`](@ref): return
`(width, height, ys, uvs)` in display order and close the session.
"""
decode_h265(ctx, annexb::Vector{UInt8}; kw...) = decodevideo(H265Decoder,ctx,annexb;kw...)
