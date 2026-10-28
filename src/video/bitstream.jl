# Annex-B H.264 and H.265 bitstreams on the host: NAL splitting, emulation-prevention
# removal, Exp-Golomb reads, the parameter-set and slice-header parsers a hardware
# decoder needs, and the bitstream packing it uploads. Every decoding backend reads
# its stream through these; none of it touches a device.
module Bitstream

export BR, rb1, rbn, ue, se, unescape, split_nals, split_nals_h265, uploadbitstream!, pocmsb,
       parse_sps, parse_pps, parse_slice, parse_mmco,
       h265naltype, H265_VCL_MAX, h265isirap, h265isidr, h265isref,
       parse_ptl, parse_strps, parse_sps_h265, parse_pps_h265, parse_slice_h265

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
        # 20 columns and 22 rows is the profile maximum (A.4.1), and what the
        # decoders' parameter sets carry; refuse more rather than drop tiles.
        (ntilec <= 19 && ntiler <= 21) || error(
            "decode_h265: $(ntilec + 1)x$(ntiler + 1) tiles exceeds the profile " *
            "maximum of 20x22")
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


"""
    pocmsb(prevmsb, prevlsb, lsb, maxlsb) -> Int

A picture order count's most significant part from its lsb and the previous
reference picture's (H.264 8.2.1.1, H.265 8.3.1): the lsb wrapped forward or back
when it moved by half the range or more.
"""
function pocmsb(prevmsb::Integer, prevlsb::Integer, lsb::Integer, maxlsb::Integer)
    lsb < prevlsb && prevlsb - lsb >= maxlsb ÷ 2 && return prevmsb + maxlsb
    lsb > prevlsb && lsb - prevlsb > maxlsb ÷ 2 && return prevmsb - maxlsb
    return prevmsb
end

end # module
