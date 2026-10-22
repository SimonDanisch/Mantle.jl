# H.265 Main, 8-bit 4:2:0 frame pictures; feeds start at a closed IRAP GOP.
# Short-term reference sets and tiles are supported; scaling lists and long-term
# references are refused.

h265naltype(nal) = Int((nal[1] >> 1) & 0x3f)
const H265_VCL_MAX = 21          # nal types 0..21 are slices
h265isirap(t) = 16 <= t <= 23
h265isidr(t) = t == 19 || t == 20
# Even types below 16 are sub-layer non-reference; with one temporal layer
# nothing references them.
h265isref(t) = t >= 16 || isodd(t)

split_nals_h265(d) = split_nals(d, nal -> (h265naltype(nal),nal))

# profile_tier_level(1, maxsub): general block, then per-sub-layer blocks.
function parse_ptl(b, maxsub)
    rbn(b, 2); tier = rb1(b); profile = rbn(b, 5)
    rbn(b, 32)                                    # compatibility flags
    prog = rb1(b); inter = rb1(b); nonpacked = rb1(b); frameonly = rb1(b)
    rbn(b, 32); rbn(b, 11)                        # 43 reserved bits
    rb1(b)                                        # inbld/reserved
    level = rbn(b, 8)
    subp = zeros(Int, maxsub); subl = zeros(Int, maxsub)
    for i in 1:maxsub
        subp[i] = rb1(b); subl[i] = rb1(b)
    end
    maxsub > 0 && for _ in (maxsub + 1):8; rbn(b, 2) end
    for i in 1:maxsub
        subp[i] == 1 && (rbn(b, 32); rbn(b, 32); rbn(b, 24))
        subl[i] == 1 && rbn(b, 8)
    end
    (; profile, tier, level, prog, inter, nonpacked, frameonly)
end

# st_ref_pic_set(idx), explicit or predicted from the already parsed `sets`.
# Returns cumulative negative POC deltas `s0` and positive `s1`, nearest first,
# their used flags `u0`/`u1`, and the prediction fields `pred, sign, absd, refidx`.
function parse_strps(b, idx, sets, nsets)
    pred = idx != 0 ? rb1(b) == 1 : false
    if pred
        didx = idx == nsets ? ue(b) + 1 : 1       # delta_idx only in slice headers
        refidx = idx - didx
        r = sets[refidx + 1]
        sign = rb1(b); absd = ue(b) + 1
        drps = sign == 1 ? -absd : absd
        nref = length(r.s0) + length(r.s1)
        used = Int[]; udelta = Int[]
        for _ in 1:(nref + 1)
            u = rb1(b); push!(used, u)
            push!(udelta, u == 1 ? 1 : rb1(b))
        end
        # 7.4.8 derivation: candidate dPocs from the reference set (+ deltaRps)
        s0 = Int[]; u0 = Int[]; s1 = Int[]; u1 = Int[]
        n0r = length(r.s0)
        for j in length(r.s1):-1:1                # ref S1, descending
            d = r.s1[j] + drps
            d < 0 && udelta[n0r + j] == 1 && (push!(s0, d); push!(u0, used[n0r + j]))
        end
        drps < 0 && udelta[nref + 1] == 1 && (push!(s0, drps); push!(u0, used[nref + 1]))
        for j in 1:n0r                            # ref S0, ascending
            d = r.s0[j] + drps
            d < 0 && udelta[j] == 1 && (push!(s0, d); push!(u0, used[j]))
        end
        for j in n0r:-1:1                         # ref S0, descending → new S1
            d = r.s0[j] + drps
            d > 0 && udelta[j] == 1 && (push!(s1, d); push!(u1, used[j]))
        end
        drps > 0 && udelta[nref + 1] == 1 && (push!(s1, drps); push!(u1, used[nref + 1]))
        for j in 1:length(r.s1)                   # ref S1, ascending
            d = r.s1[j] + drps
            d > 0 && udelta[j + n0r] == 1 && (push!(s1, d); push!(u1, used[j + n0r]))
        end
        return (; s0, u0, s1, u1, pred = true, sign, absd, refidx)
    end
    n0 = ue(b); n1 = ue(b)
    s0 = Int[]; u0 = Int[]; s1 = Int[]; u1 = Int[]
    d = 0
    for _ in 1:n0
        d -= ue(b) + 1; push!(s0, d); push!(u0, rb1(b))
    end
    d = 0
    for _ in 1:n1
        d += ue(b) + 1; push!(s1, d); push!(u1, rb1(b))
    end
    (; s0, u0, s1, u1, pred = false, sign = 0, absd = 1, refidx = 0)
end

function parse_sps_h265(nal)
    b = BR(unescape(nal[3:end]), 0)               # 2-byte NAL header
    vps_id = rbn(b, 4); maxsub = rbn(b, 3); tnest = rb1(b)
    ptl = parse_ptl(b, maxsub)
    sps_id = ue(b)
    chroma = ue(b); sepcol = chroma == 3 ? rb1(b) : 0
    w = ue(b); h = ue(b)
    confwin = rb1(b); cl = cr = ct = cb = 0
    confwin == 1 && (cl = ue(b); cr = ue(b); ct = ue(b); cb = ue(b))
    bdl = ue(b); bdc = ue(b)
    log2poc = ue(b)
    subord = rb1(b)
    dpbsize = 0; reorder = 0; latency = 0
    for _ in (subord == 1 ? 0 : maxsub):maxsub
        dpbsize = ue(b); reorder = ue(b); latency = ue(b)   # keep the top layer's
    end
    log2mincb = ue(b); log2diffcb = ue(b)
    log2mintb = ue(b); log2difftb = ue(b)
    maxtdinter = ue(b); maxtdintra = ue(b)
    scal = rb1(b)
    scal == 1 && rb1(b) == 1 && error("decode_h265: scaling lists not handled")
    amp = rb1(b); sao = rb1(b)
    pcm = rb1(b); pcmbl = pcmbc = pcmmin = pcmdiff = pcmloop = 0
    if pcm == 1
        pcmbl = rbn(b, 4); pcmbc = rbn(b, 4)
        pcmmin = ue(b); pcmdiff = ue(b); pcmloop = rb1(b)
    end
    nsets = ue(b)
    sets = NamedTuple[]
    for i in 0:(nsets - 1)
        push!(sets, parse_strps(b, i, sets, nsets))
    end
    ltref = rb1(b)
    ltref == 1 && ue(b) > 0 && error("decode_h265: long-term reference pictures not handled")
    tmvp = rb1(b); strongintra = rb1(b); vui = rb1(b)
    (; vps_id, maxsub, tnest, ptl, sps_id, chroma, sepcol, w, h, confwin, cl, cr, ct, cb,
       bdl, bdc, log2poc, subord, dpbsize, reorder, latency,
       log2mincb, log2diffcb, log2mintb, log2difftb, maxtdinter, maxtdintra,
       scal, amp, sao, pcm, pcmbl, pcmbc, pcmmin, pcmdiff, pcmloop,
       nsets, sets, ltref, tmvp, strongintra, vui)
end

function parse_pps_h265(nal)
    b = BR(unescape(nal[3:end]), 0)
    pps_id = ue(b); sps_id = ue(b)
    depslice = rb1(b); outflag = rb1(b); extrabits = rbn(b, 3)
    signhide = rb1(b); cabacinit = rb1(b)
    nl0 = ue(b); nl1 = ue(b); initqp = se(b)
    cintra = rb1(b); tskip = rb1(b)
    cuqp = rb1(b); cuqpdepth = cuqp == 1 ? ue(b) : 0
    cbqp = se(b); crqp = se(b)
    slqp = rb1(b); wp = rb1(b); wbp = rb1(b); tqbypass = rb1(b)
    tiles = rb1(b); entsync = rb1(b)
    ntilec = ntiler = 0; uniform = 1
    colw = UInt16[]; rowh = UInt16[]
    if tiles == 1
        ntilec = ue(b); ntiler = ue(b); uniform = rb1(b)
        if uniform == 0
            # 7.3.2.3.1: explicit tile sizes precede the loop-filter flag. Skipping
            # them misaligns the rest of the PPS (deblocking, parallel merge level).
            colw = UInt16[ue(b) for _ in 1:ntilec]
            rowh = UInt16[ue(b) for _ in 1:ntiler]
        end
        # The Std struct holds 19 column widths and 21 row heights, the profile
        # maximum; refuse more rather than drop tiles.
        (ntilec <= 19 && ntiler <= 21) || error(
            "decode_h265: $(ntilec + 1)x$(ntiler + 1) tiles exceeds the " *
            "19x21 the Vulkan video parameter set can carry")
        rb1(b)                                    # loop_filter_across_tiles
    end
    lfslices = rb1(b)
    dbctl = rb1(b); dbovr = 0; dbdis = 0; beta = 0; tc = 0
    if dbctl == 1
        dbovr = rb1(b); dbdis = rb1(b)
        dbdis == 0 && (beta = se(b); tc = se(b))
    end
    rb1(b) == 1 && error("decode_h265: pps scaling lists not handled")
    listmod = rb1(b); log2pml = ue(b); shext = rb1(b)
    (; pps_id, sps_id, depslice, outflag, extrabits, signhide, cabacinit,
       nl0, nl1, initqp, cintra, tskip, cuqp, cuqpdepth, cbqp, crqp, slqp,
       wp, wbp, tqbypass, tiles, entsync, ntilec, ntiler, uniform, colw, rowh, lfslices,
       dbctl, dbovr, dbdis, beta, tc, listmod, log2pml, shext)
end

# First slice segment of a picture: POC lsb, the RPS in effect, and the exact bit
# count of an inline st_ref_pic_set, which the driver needs.
function parse_slice_h265(nal, sps, pps)
    t = h265naltype(nal)
    b = BR(unescape(nal[3:end]), 0)
    first = rb1(b)
    h265isirap(t) && rb1(b)                       # no_output_of_prior_pics
    ue(b)                                         # slice_pic_parameter_set_id
    first == 1 || error("decode_h265: parse_slice_h265 expects the first slice segment")
    for _ in 1:pps.extrabits; rb1(b) end
    stype = ue(b)
    pps.outflag == 1 && rb1(b)
    sps.sepcol == 1 && rbn(b, 2)
    poclsb = 0; rps = nothing; spsflag = 0; rpsbits = 0; refnd = 0
    if !h265isidr(t)
        poclsb = rbn(b, sps.log2poc + 4)
        spsflag = rb1(b)
        if spsflag == 0
            p0 = b.p
            rps = parse_strps(b, sps.nsets, sps.sets, sps.nsets)
            rpsbits = b.p - p0
            rps.pred && (refnd = length(sps.sets[rps.refidx + 1].s0) +
                                 length(sps.sets[rps.refidx + 1].s1))
        else
            idx = sps.nsets > 1 ? rbn(b, cld(ndigits(sps.nsets - 1; base = 2), 1)) : 0
            idx = sps.nsets > 1 ? idx : 0
            rps = sps.sets[idx + 1]
        end
    end
    (; t, idr = h265isidr(t), irap = h265isirap(t), isref = h265isref(t),
       stype, poclsb, rps, spsflag, rpsbits, refnd)
end

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
