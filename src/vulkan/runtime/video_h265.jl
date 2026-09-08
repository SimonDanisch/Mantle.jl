# Hardware H.265/HEVC decode on the GPU via VK_KHR_video_decode_h265 — the
# second codec on the shared `VideoDecoder` core (see video.jl): same session
# machinery, chunked submit, bitstream buffer, display reorder and teardown;
# this file adds the HEVC bitstream parsing (VPS/SPS/PPS, slice segment
# headers, short-term reference picture sets), the Std/Vk h265 structs and the
# RPS-driven DPB bookkeeping. HEVC's RPS is DECLARATIVE — every picture lists
# the complete set of references that stay alive — which replaces H.264's
# stateful MMCO/sliding-window logic.
#
# Scope: Main profile, 8-bit 4:2:0, frame pictures, closed GOPs (feeds start
# at an IRAP that does not depend on earlier pictures — IDR, or CRA when the
# feed drops its leading RASL pictures). Scaling lists and long-term references
# error out rather than mis-decode.
#
# Tiles are handled, uniform spacing or not. They were on that refusal list, and
# the refusal was not the safe half of the trade it looked like: the parse gave
# up *after* `uniform_spacing_flag` and before the explicit widths, so anything
# that swallowed the error carried on with a bit reader pointing into the middle
# of the tile geometry, and every PPS field past it — deblocking, parallel merge
# level — was read from the wrong bits. A phone recording three columns of
# 24/24/12 CTBs across 1920 decoded to a flat green frame.

# ---------- H.265 bitstream parsing ----------
h265naltype(nal) = Int((nal[1] >> 1) & 0x3f)
const H265_VCL_MAX = 21          # nal types 0..21 are slices
h265isirap(t) = 16 <= t <= 23
h265isidr(t) = t == 19 || t == 20
# _N types (sub-layer non-reference); with one temporal layer they are never referenced
h265isref(t) = t >= 16 || isodd(t)

function split_nals_h265(d)
    out = Tuple{Int, Vector{UInt8}}[]
    starts = Int[]; i = 1
    while i <= length(d) - 3
        if d[i] == 0 && d[i+1] == 0 && d[i+2] == 1; push!(starts, i + 3); i += 3 else i += 1 end
    end
    for (k, st) in enumerate(starts)
        e = k < length(starts) ? starts[k+1] - 4 : length(d)
        while e > st && d[e] == 0; e -= 1 end
        nal = d[st:e]
        push!(out, (h265naltype(nal), nal))
    end
    out
end

# profile_tier_level(1, maxsub): general block + per-sub-layer blocks
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

# st_ref_pic_set(idx): explicit or inter-set-predicted; `sets` are the already
# parsed sets. Returns (s0 = cumulative NEGATIVE deltas in decoding order,
# u0 = used flags, s1 = positive, u1, pred, sign, absdelta, refidx).
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
            # 7.3.2.3.1: the explicit geometry sits between `uniform_spacing_flag`
            # and `loop_filter_across_tiles_enabled_flag`. Skipping it does not
            # merely lose the tile widths — it leaves the reader misaligned for
            # the whole rest of the PPS, so the deblocking parameters and
            # `log2_parallel_merge_level` are read out of the wrong bits and the
            # `StdVideoH265PictureParameterSet` handed to the driver is nonsense.
            # A phone's HEVC does this routinely: three columns of 24/24/12 CTBs
            # across 1920 is what surfaced it.
            colw = UInt16[ue(b) for _ in 1:ntilec]
            rowh = UInt16[ue(b) for _ in 1:ntiler]
        end
        # The Std struct carries 19 columns and 21 rows, which is what the
        # profiles allow; a stream past that would silently lose its last tiles.
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

# First slice segment of a picture: POC lsb + the RPS actually in effect, plus
# the exact bit count of an inline st_ref_pic_set (the driver re-parses it).
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

# ---------- Std struct builders ----------
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

# general_level_idc = 30 × level number; the Std enum counts known levels upward
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

# ---------- the decoder ----------
"""
    H265Decoder(ctx, paramnals; chroma=false)

A persistent hardware H.265 decode session on the shared [`VideoDecoder`](@ref)
core — the HEVC sibling of [`H264Decoder`](@ref): same incremental
[`feed!`](@ref)/[`decodemore!`](@ref) access pattern, RPS-driven DPB instead of
MMCO. `paramnals` is any Annex-B chunk containing the stream's VPS+SPS+PPS.
"""
mutable struct H265Decoder <: VideoDecoder
    w::Ctxwrap; dev::Any
    sps::Any; pps::Any
    chroma::Bool
    SESSION::Any; PARAMS::Any
    PIN::Vector{Any}
    imgs::Vector{Any}
    nslots::Int
    # The four fields the shared core in video.jl reads off a `VideoDecoder`.
    # H.264 grew them with `DPB_AND_OUTPUT_DISTINCT` and the core was changed to
    # read them; H.265 was not, so `feed!` and `decodemore!` died on a
    # `FieldError` the moment either touched one. A shared core means the state
    # it reads is part of the contract, not of one implementor.
    #
    # The first three are the COINCIDE defaults, which is what this decoder does:
    # it copies out of the DPB slot in the same command buffer, so there are no
    # separate decode targets and nothing is deferred. `mkoutimg === nothing` is
    # what tells the core that. DISTINCT is not implemented for H.265; a device
    # that only offers it would need this filled in the way video.jl does.
    outimgs::Vector{Any}
    mkoutimg::Any
    copyq::Vector{Any}
    bsalign::Int
    bbuf::Any; bmem::Any; bmap::Ptr{UInt8}; bufsz::UInt64
    cbh::Any; CB::Any
    CW::Int; CH::Int; DW::Int; DH::Int
    aus::Vector{Vector{Vector{UInt8}}}
    nextau::Int
    dpb::Vector{Tuple{Int, Int}}            # (slot, poc)
    freeslots::Vector{Int}
    prevmsb::Int; prevlsb::Int; maxpoclsb::Int; gop::Int
    decoded::Int
    isfirst::Bool
    pending::Vector{Tuple{Int, Int, Any, Any}}
    open::Bool
end

decodeprofile(dec::H265Decoder) = video_profile_h265(dec.w)

function H265Decoder(ctx, paramnals::AbstractVector{UInt8}; chroma::Bool = false)
    ctx.video_decode_available || error("device has no video decode support")
    Lava = parentmodule(@__MODULE__)
    w = Ctxwrap(ctx, ctx.device, UInt32(ctx.video_decode_queue_family_index), ctx.video_decode_queue)
    dev = w.dev
    nalu = split_nals_h265(paramnals)
    sps = parse_sps_h265(first(n[2] for n in nalu if n[1] == 33))
    pps = parse_pps_h265(first(n[2] for n in nalu if n[1] == 34))
    sps.chroma == 1 ||
        error("decode_h265: only 4:2:0 chroma is supported (chroma_format_idc=$(sps.chroma))")
    sps.bdl == 0 && sps.bdc == 0 ||
        error("decode_h265: only 8-bit is supported (bit depth $(8 + sps.bdl)); transcode first")
    CW = sps.w; CH = sps.h
    DW = CW - 2 * (sps.cl + sps.cr); DH = CH - 2 * (sps.ct + sps.cb)
    fmt = C.VkFormat(1000156003)              # NV12
    PIN = Any[]; pin(x) = (push!(PIN, x); x)
    hp, pr, pl = video_profile_h265(w); pin(hp); pin(pr); pin(pl); pProf = rp(pr)
    rHdr = pin(Ref(C.VkExtensionProperties(ntuple(_ -> Cchar(0), 256), UInt32(0))))
    BSALIGN = 1
    let hc = Ref(C.VkVideoDecodeH265CapabilitiesKHR(C.VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_CAPABILITIES_KHR, C_NULL, C.StdVideoH265LevelIdc(0))),
        dc = Ref(C.VkVideoDecodeCapabilitiesKHR(C.VK_STRUCTURE_TYPE_VIDEO_DECODE_CAPABILITIES_KHR, Ptr{Cvoid}(rp(hc)), UInt32(0))),
        e0 = C.VkExtent2D(0, 0), cp = Ref(C.VkVideoCapabilitiesKHR(C.VK_STRUCTURE_TYPE_VIDEO_CAPABILITIES_KHR, Ptr{Cvoid}(rp(dc)), UInt32(0), UInt64(0), UInt64(0), e0, e0, e0, UInt32(0), UInt32(0), C.VkExtensionProperties(ntuple(_ -> Cchar(0), 256), UInt32(0))))
        GC.@preserve hc dc cp PIN ccall(Vk.function_pointer(ctx.instance, "vkGetPhysicalDeviceVideoCapabilitiesKHR"), Int32, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}), ctx.physical_device.vks, Ptr{Cvoid}(pProf), pc(cp))
        rHdr[] = cp[].stdHeaderVersion
        BSALIGN = Int(max(cp[].minBitstreamBufferOffsetAlignment,
                          cp[].minBitstreamBufferSizeAlignment, 1))
        # `dc` was filled in and thrown away here, exactly as it once was on the
        # H.264 side. Its flags say whether one image may serve as both DPB and
        # decode target (DPB_AND_OUTPUT_COINCIDE, 0x1) or whether the two must be
        # separate images (DPB_AND_OUTPUT_DISTINCT, 0x2). This decoder copies the
        # frame straight out of its DPB slot, which is only legal under COINCIDE.
        #
        # On a DISTINCT-only device that copy reads an image the driver never
        # meant as a copy source, and RADV does not reject it — it **segfaults
        # inside `vkCmdCopyImageToBuffer`**, taking the process with it. AMD
        # Radeon 8060S reports 0x2, so every H.265 file killed the editor on this
        # card whatever its contents; the tiled phone recording that surfaced it
        # was a red herring twice over.
        #
        # Refusing here is what lets a caller fall back: `H265Decoder` is built
        # before any frame is fed, so this throws with nothing recorded and
        # nothing submitted.
        COINCIDE = (dc[].flags & 0x1) != 0
        COINCIDE || error("decode_h265: this device offers only " *
            "DPB_AND_OUTPUT_DISTINCT (flags=0x$(string(dc[].flags; base = 16))), and " *
            "the H.265 decoder copies out of its DPB slot, which needs " *
            "DPB_AND_OUTPUT_COINCIDE. H.264 implements DISTINCT (see video.jl); " *
            "H.265 does not yet. Decode this stream on the CPU, or transcode it.")
    end
    maxrefs = max(sps.dpbsize, 1)
    maxslots = UInt32(maxrefs + 2)
    sci = pin(Ref(C.VkVideoSessionCreateInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_SESSION_CREATE_INFO_KHR, C_NULL, w.qf, UInt32(0), pProf, fmt, C.VkExtent2D(CW, CH), fmt, maxslots, UInt32(maxrefs), rp(rHdr))))
    rSess = Ref{C.VkVideoSessionKHR}(); GC.@preserve PIN ccall(dfp(w, "vkCreateVideoSessionKHR"), Int32, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}), dev.vks, pc(sci), C_NULL, rp(rSess)); SESSION = rSess[]
    let mc = Ref(UInt32(0)); ccall(dfp(w, "vkGetVideoSessionMemoryRequirementsKHR"), Int32, (Ptr{Cvoid}, C.VkVideoSessionKHR, Ptr{UInt32}, Ptr{Cvoid}), dev.vks, SESSION, mc, C_NULL)
        mreqs = [C.VkVideoSessionMemoryRequirementsKHR(C.VK_STRUCTURE_TYPE_VIDEO_SESSION_MEMORY_REQUIREMENTS_KHR, C_NULL, UInt32(0), C.VkMemoryRequirements(UInt64(0), UInt64(0), UInt32(0))) for _ in 1:Int(mc[])]
        GC.@preserve mreqs ccall(dfp(w, "vkGetVideoSessionMemoryRequirementsKHR"), Int32, (Ptr{Cvoid}, C.VkVideoSessionKHR, Ptr{UInt32}, Ptr{Cvoid}), dev.vks, SESSION, mc, pointer(mreqs))
        binds = C.VkBindVideoSessionMemoryInfoKHR[]
        for mr in mreqs
            m = Vk.unwrap(Vk.allocate_memory(dev, mr.memoryRequirements.size, memtype(w, mr.memoryRequirements.memoryTypeBits, C.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))); pin(m)
            push!(binds, C.VkBindVideoSessionMemoryInfoKHR(C.VK_STRUCTURE_TYPE_BIND_VIDEO_SESSION_MEMORY_INFO_KHR, C_NULL, mr.memoryBindIndex, m.vks, UInt64(0), mr.memoryRequirements.size))
        end
        GC.@preserve binds PIN ccall(dfp(w, "vkBindVideoSessionMemoryKHR"), Int32, (Ptr{Cvoid}, C.VkVideoSessionKHR, UInt32, Ptr{Cvoid}), dev.vks, SESSION, UInt32(length(binds)), pointer(binds))
    end
    # session parameters: VPS + SPS + PPS with their pinned nested tables
    rPTL = pin(Ref(std_ptl(sps.ptl)))
    rDPBM = pin(Ref(std_dpbm(sps)))
    rSTRPS = pin(Ref([std_strps(s) for s in sps.sets]))
    rVPSs = pin(Ref(std_vps265(sps, rDPBM, rPTL)))
    rSPSs = pin(Ref(std_sps265(sps, rPTL, rDPBM, rSTRPS)))
    rPPSs = pin(Ref(std_pps265(pps, sps)))
    add = pin(Ref(C.VkVideoDecodeH265SessionParametersAddInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_SESSION_PARAMETERS_ADD_INFO_KHR, C_NULL,
        UInt32(1), rp(rVPSs), UInt32(1), rp(rSPSs), UInt32(1), rp(rPPSs))))
    h265c = pin(Ref(C.VkVideoDecodeH265SessionParametersCreateInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_SESSION_PARAMETERS_CREATE_INFO_KHR, C_NULL,
        UInt32(1), UInt32(1), UInt32(1), rp(add))))
    pci = pin(Ref(C.VkVideoSessionParametersCreateInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_SESSION_PARAMETERS_CREATE_INFO_KHR, Ptr{Cvoid}(rp(h265c)), UInt32(0), C_NULL, SESSION)))
    rParams = Ref{C.VkVideoSessionParametersKHR}(); GC.@preserve PIN ccall(dfp(w, "vkCreateVideoSessionParametersKHR"), Int32, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}), dev.vks, pc(pci), C_NULL, rp(rParams)); PARAMS = rParams[]
    # DPB pool + command buffer — identical to the h264 pool (NV12, decode+dpb usage)
    nslots = Int(maxslots)
    fmt_hl = Vk.Format(1000156003)
    dpbusage = Vk.IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR | Vk.IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR | Vk.IMAGE_USAGE_TRANSFER_SRC_BIT
    imgs = Vector{Any}(undef, nslots)
    for k in 1:nslots
        image = GC.@preserve PIN Vk.Image(dev, Vk.IMAGE_TYPE_2D, fmt_hl, Vk.Extent3D(CW, CH, 1),
            1, 1, Vk.SAMPLE_COUNT_1_BIT, Vk.IMAGE_TILING_OPTIMAL, dpbusage,
            Vk.SHARING_MODE_EXCLUSIVE, UInt32[], Vk.IMAGE_LAYOUT_UNDEFINED; next = Ptr{Cvoid}(rp(pl)))
        mem = MANTLE.alloc_image_memory(ctx, image)
        view = Vk.ImageView(dev, image, Vk.IMAGE_VIEW_TYPE_2D, fmt_hl,
            Vk.ComponentMapping(Vk.COMPONENT_SWIZZLE_IDENTITY, Vk.COMPONENT_SWIZZLE_IDENTITY, Vk.COMPONENT_SWIZZLE_IDENTITY, Vk.COMPONENT_SWIZZLE_IDENTITY),
            Vk.ImageSubresourceRange(Vk.IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1))
        vimg = VideoImage(image, mem, CW, CH, DW, DH, fmt_hl, Ref(UInt32(C.VK_IMAGE_LAYOUT_UNDEFINED)))
        pin(image); pin(mem); pin(view)
        imgs[k] = (vimg, view)
    end
    cbh = first(Vk.unwrap(Vk.allocate_command_buffers(dev,
        Vk.CommandBufferAllocateInfo(Vk.unwrap(Vk.create_command_pool(dev, w.qf; flags = Vk.COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT)),
            Vk.COMMAND_BUFFER_LEVEL_PRIMARY, 1)))); pin(cbh)
    return H265Decoder(w, dev, sps, pps, chroma, SESSION, PARAMS, PIN, imgs, nslots,
                       Any[], nothing, Any[], BSALIGN,
                       nothing, nothing, Ptr{UInt8}(0), UInt64(0), cbh, cbh.vks,
                       Int(CW), Int(CH), Int(DW), Int(DH),
                       Vector{Vector{Vector{UInt8}}}(), 1,
                       Tuple{Int, Int}[], collect(1:nslots),
                       0, 0, 1 << (sps.log2poc + 4), 0, 0, true,
                       Tuple{Int, Int, Any, Any}[], true)
end

function feed!(dec::H265Decoder, annexb::AbstractVector{UInt8})
    nalu = split_nals_h265(annexb)
    aus = Vector{Vector{Vector{UInt8}}}()
    maxau = 0
    for (t, nal) in nalu
        t <= H265_VCL_MAX || continue
        first = (nal[3] >> 7) & 0x01 == 1         # first_slice_segment_in_pic_flag
        if first; push!(aus, [nal]) else push!(aus[end], nal) end
        maxau = max(maxau, sum(length(sl) + 3 for sl in aus[end]))
    end
    ensurebitbuf!(dec, maxau)
    dec.aus = aus
    dec.nextau = 1
    return dec
end

# The fourth argument is the shared core's index into `dec.outimgs`, which only
# DISTINCT fills. Accepted and ignored here so the H.264 and H.265 `decodeau!` are
# the same call — `decodemore!` passes four to both, and a three-argument method
# here is a `MethodError` from inside the decode loop.
#
# NOT named `outslot`: that is this function's own DPB slot, taken from
# `freeslots` below and used for the reference-slot index and the DPB entry. A
# parameter of that name is live from the first line to that assignment, which is
# a decode into whatever slot the core's loop counter happened to name.
function decodeau!(dec::H265Decoder, au, bufbase::Integer, ::Integer = 1)
    Lava = parentmodule(@__MODULE__)
    w = dec.w; sps = dec.sps; pps = dec.pps
    SESSION = dec.SESSION; PARAMS = dec.PARAMS; PIN = dec.PIN; imgs = dec.imgs
    bbuf = dec.bbuf; bmap = dec.bmap; cbh = dec.cbh; CB = dec.CB
    CW = dec.CW; CH = dec.CH; DW = dec.DW; DH = dec.DH; chroma = dec.chroma
    dpb = dec.dpb; freeslots = dec.freeslots; maxpoclsb = dec.maxpoclsb
    allcol = C.VkImageSubresourceRange(UInt32(C.VK_IMAGE_ASPECT_COLOR_BIT), UInt32(0), UInt32(1), UInt32(0), UInt32(1))
    IGN = UInt32(0xffffffff); allst = UInt32(C.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT)
    barrier(img, old, new) = Ref(C.VkImageMemoryBarrier(C.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, C_NULL, UInt32(C.VK_ACCESS_MEMORY_WRITE_BIT), UInt32(C.VK_ACCESS_MEMORY_READ_BIT) | UInt32(C.VK_ACCESS_MEMORY_WRITE_BIT), C.VkImageLayout(old), C.VkImageLayout(new), IGN, IGN, img, allcol))
    emit(cb, b) = ccall(dfp(w, "vkCmdPipelineBarrier"), Cvoid, (C.VkCommandBuffer, UInt32, UInt32, UInt32, UInt32, Ptr{Cvoid}, UInt32, Ptr{Cvoid}, UInt32, Ptr{Cvoid}), cb, allst, allst, UInt32(0), UInt32(0), C_NULL, UInt32(0), C_NULL, UInt32(1), pc(b))
    picres(view) = C.VkVideoPictureResourceInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR, C_NULL, C.VkOffset2D(0, 0), C.VkExtent2D(CW, CH), UInt32(0), view)
    DPBLAYOUT = UInt32(C.VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR); TSRC = UInt32(C.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
    sh = parse_slice_h265(au[1], sps, pps)
    if sh.idr
        for (slot, _) in dpb; push!(freeslots, slot); end; empty!(dpb)
        dec.prevmsb = 0; dec.prevlsb = 0; dec.gop += 1
    end
    # POC (8.3.1): same lsb/msb wrap arithmetic as h264 poc type 0
    if sh.idr
        poc = 0
    else
        if sh.poclsb < dec.prevlsb && (dec.prevlsb - sh.poclsb) >= maxpoclsb ÷ 2
            pmsb = dec.prevmsb + maxpoclsb
        elseif sh.poclsb > dec.prevlsb && (sh.poclsb - dec.prevlsb) > maxpoclsb ÷ 2
            pmsb = dec.prevmsb - maxpoclsb
        else
            pmsb = dec.prevmsb
        end
        poc = pmsb + sh.poclsb
        dec.prevmsb = pmsb; dec.prevlsb = sh.poclsb
    end
    # RPS-driven DPB: the slice DECLARES every reference that stays alive —
    # anything else frees its slot before this decode
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
    outimg = outvimg.image.vks; outview = outviewhl.vks; outlay = outvimg.layout
    off = 0; sliceoffs = UInt32[]
    for sl in au
        push!(sliceoffs, UInt32(off))
        # Kept alive across the copy — see the note on the same staging in video.jl.
        stage = vcat(UInt8[0, 0, 1], sl)
        GC.@preserve stage unsafe_copyto!(bmap + bufbase + off, pointer(stage), length(sl) + 3)
        off += length(sl) + 3
    end
    # reference slots in DPB order; the Std picture info lists RPS entries as
    # positions into this array
    refpr = Ref{C.VkVideoPictureResourceInfoKHR}[]; refri = Ref{C.StdVideoDecodeH265ReferenceInfo}[]; refds = Ref{C.VkVideoDecodeH265DpbSlotInfoKHR}[]
    decref = C.VkVideoReferenceSlotInfoKHR[]; begref = C.VkVideoReferenceSlotInfoKHR[]
    pocpos = Dict{Int, Int}()
    for (k, (slot, p)) in enumerate(dpb)
        rv = imgs[slot][2].vks
        pr = Ref(picres(rv))
        ri = Ref(C.StdVideoDecodeH265ReferenceInfo(C.StdVideoDecodeH265ReferenceInfoFlags(pbytes(UInt32(0))), Int32(p)))
        ds = Ref(C.VkVideoDecodeH265DpbSlotInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_DPB_SLOT_INFO_KHR, C_NULL, rp(ri)))
        push!(refpr, pr); push!(refri, ri); push!(refds, ds)
        push!(decref, C.VkVideoReferenceSlotInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR, pc(ds), Int32(slot - 1), rp(pr)))
        push!(begref, C.VkVideoReferenceSlotInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR, C_NULL, Int32(slot - 1), rp(pr)))
        pocpos[p] = slot - 1              # RPS entries carry the DPB slotIndex
    end
    fill8(pocs) = ntuple(i -> UInt8(i <= length(pocs) ? get(pocpos, pocs[i], 0xff) : 0xff), 8)
    before = sh.rps === nothing ? Int[] : [poc + d for (d, u) in zip(sh.rps.s0, sh.rps.u0) if u == 1]
    after  = sh.rps === nothing ? Int[] : [poc + d for (d, u) in zip(sh.rps.s1, sh.rps.u1) if u == 1]
    outpr = Ref(picres(outview))
    outri = Ref(C.StdVideoDecodeH265ReferenceInfo(C.StdVideoDecodeH265ReferenceInfoFlags(pbytes(UInt32(0))), Int32(poc)))
    outds = Ref(C.VkVideoDecodeH265DpbSlotInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_DPB_SLOT_INFO_KHR, C_NULL, rp(outri)))
    push!(begref, C.VkVideoReferenceSlotInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR, C_NULL, Int32(-1), rp(outpr)))
    setup = Ref(C.VkVideoReferenceSlotInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR, pc(outds), Int32(outslot - 1), rp(outpr)))
    picflags = packflags((sh.irap ? 1 : 0, sh.idr ? 1 : 0, sh.isref ? 1 : 0, sh.spsflag))
    stdpic = Ref(C.StdVideoDecodeH265PictureInfo(C.StdVideoDecodeH265PictureInfoFlags(pbytes(picflags)),
        UInt8(sps.vps_id), UInt8(pps.sps_id), UInt8(pps.pps_id),
        UInt8(sh.refnd), Int32(poc), UInt16(sh.rpsbits), UInt16(0),
        fill8(before), fill8(after), ntuple(_ -> UInt8(0xff), 8)))
    soff = copy(sliceoffs)
    h265pi = Ref(C.VkVideoDecodeH265PictureInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_PICTURE_INFO_KHR, C_NULL, rp(stdpic), UInt32(length(soff)), pointer(soff)))
    decinfo = Ref(C.VkVideoDecodeInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_DECODE_INFO_KHR, pc(h265pi), UInt32(0), bbuf, UInt64(bufbase), UInt64(cld(off, 256) * 256), outpr[], rp(setup), UInt32(length(decref)), isempty(decref) ? Ptr{C.VkVideoReferenceSlotInfoKHR}(C_NULL) : pointer(decref)))
    beginfo = Ref(C.VkVideoBeginCodingInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_BEGIN_CODING_INFO_KHR, C_NULL, UInt32(0), SESSION, PARAMS, UInt32(length(begref)), pointer(begref)))
    ctl = Ref(C.VkVideoCodingControlInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_CODING_CONTROL_INFO_KHR, C_NULL, UInt32(C.VK_VIDEO_CODING_CONTROL_RESET_BIT_KHR)))
    endinfo = Ref(C.VkVideoEndCodingInfoKHR(C.VK_STRUCTURE_TYPE_VIDEO_END_CODING_INFO_KHR, C_NULL, UInt32(0)))
    isfirst = dec.isfirst
    dst = LavaArray{UInt8, 2}(undef, (DW, DH); bq = dec.w.ctx.default_bq, extra_usage = UInt32(Vk.BUFFER_USAGE_TRANSFER_DST_BIT))
    dstbuf = dst.buf[]
    duv = chroma ? LavaArray{UInt8, 2}(undef, (DW, DH ÷ 2); bq = dec.w.ctx.default_bq, extra_usage = UInt32(Vk.BUFFER_USAGE_TRANSFER_DST_BIT)) : nothing
    GC.@preserve PIN dst duv cbh outvimg refpr refri refds decref begref outpr outri outds setup stdpic soff h265pi decinfo beginfo ctl endinfo imgs begin
        for (slot, _) in dpb; lay = imgs[slot][1].layout; if lay[] != DPBLAYOUT; emit(CB, barrier(imgs[slot][1].image.vks, lay[], DPBLAYOUT)); lay[] = DPBLAYOUT; end; end
        emit(CB, barrier(outimg, outlay[], DPBLAYOUT)); outlay[] = DPBLAYOUT
        ccall(dfp(w, "vkCmdBeginVideoCodingKHR"), Cvoid, (C.VkCommandBuffer, Ptr{Cvoid}), CB, pc(beginfo))
        isfirst && ccall(dfp(w, "vkCmdControlVideoCodingKHR"), Cvoid, (C.VkCommandBuffer, Ptr{Cvoid}), CB, pc(ctl))
        ccall(dfp(w, "vkCmdDecodeVideoKHR"), Cvoid, (C.VkCommandBuffer, Ptr{Cvoid}), CB, pc(decinfo))
        ccall(dfp(w, "vkCmdEndVideoCodingKHR"), Cvoid, (C.VkCommandBuffer, Ptr{Cvoid}), CB, pc(endinfo))
        emit(CB, barrier(outimg, DPBLAYOUT, TSRC)); outlay[] = TSRC
        record_luma_copy!(cbh, outvimg, dstbuf.buffer, pool_offset(dstbuf) + dst.offset)
        if chroma
            uvbuf = duv.buf[]
            record_chroma_copy!(cbh, outvimg, uvbuf.buffer, pool_offset(uvbuf) + duv.offset)
        end
    end
    push!(dec.pending, (dec.gop, poc, dst, duv))
    if sh.isref
        push!(dpb, (outslot, poc))
    else
        push!(freeslots, outslot)
    end
    dec.isfirst = false
    dec.decoded += 1
    return Int(cld(off, 256) * 256)
end

"""
    decode_h265(ctx, annexb; maxframes=typemax(Int), chroma=false)

Hardware-decode an H.265 Annex-B stream on the GPU — the batch form of
[`H265Decoder`](@ref), mirroring [`decode_h264`](@ref): feed everything, drain
once in display order, free the session.
"""
function decode_h265(ctx, annexb::Vector{UInt8}; maxframes::Int = typemax(Int), chroma::Bool = false)
    dec = H265Decoder(ctx, annexb; chroma)
    try
        feed!(dec, annexb)
        out = decodemore!(dec, min(maxframes, length(dec.aus)); holdback = typemax(Int))
        return (dec.DW, dec.DH, [y for (y, _) in out], [uv for (_, uv) in out])
    finally
        close(dec)
    end
end
