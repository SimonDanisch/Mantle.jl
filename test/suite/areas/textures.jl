# Textures: an Image a shader samples gets an index in the device-wide texture
# table on its first use (api.md 2.6, 2.6b; recording-plan.md, Graphics;
# resizing-and-raytracing.jl B.9). descriptorsets.adoc is the Vulkan
# specification's chapter.
#
# Image arguments are typed SampledImage values. Mantle.textureindex is the
# suite's internal device projection; actual sampled contents are checked by
# RayMakie's RM-15, not inferred from index equality.
#
using ColorTypes: RGBA
using FixedPointNumbers: N0f8

"""`f()` and the prefix steps applied while it ran (the step log restarted around it)."""
function stepsduring(f, dev = testdevice())
    Mantle.steplog!(dev, false)
    Mantle.steplog!(dev, true)
    v = f()
    log = Mantle.steplog(dev)
    Mantle.steplog!(dev, false)
    return v, log
end

"""The kinds of the steps in `log` that changed `x`, in emission order."""
stepkinds(log, x) = [s.kind for s in log if s.target === x]

"""An open gate: kernels read it, so a case can order them after a slowwrite! into a gate of its own."""
readygate(dev) = MantleArray(dev, Int32[0])

"""`w` × `h` deterministic texels."""
texels(w, h; seed = 1) = collect(reshape(reinterpret(RGBA{N0f8}, testdata(UInt32, w * h; seed)), w, h))

"""
out[1] = the table index of `tex` as the kernel reads it, by one work-item (the
ndrange of 1 is rounded up to a whole work-group); reading `gate` orders it
after a slowwrite!.
"""
function tx_index!(out, tex, gate)
    KI.get_global_id().x == 1 || return nothing
    out[1] = Mantle.textureindex(tex) + (gate[1] == Int32(-1) ? UInt32(1) : UInt32(0))
    return nothing
end

"""`tex`'s table index, read by one eager launch (which applies what is pending on `tex`)."""
function textureindex(dev, tex)
    out = MantleArray(dev, UInt32, 1)
    dispatch!(dev, tx_index!, (out, tex, readygate(dev)), 1)
    return Array(out)[1]
end

"""A graph that writes `tex`'s index into a new array each run; returns the graph and the array."""
function indexgraph(dev, tex; gate = readygate(dev), slow = false)
    out = MantleArray(dev, UInt32, 1)
    g = Graph(dev)
    slow && slowwrite!(g, gate, 7)
    dispatch!(g, tx_index!, (out, tex, gate), 1)
    return g, out
end

# ── Textures (TX) ──

# api.md 2.6; resizing-and-raytracing.jl B.9: a regrown texture gets a new, unused index; the old index is retired with the applying submission's token, after the runs that used it. resize! while a run using the texture is in flight; the next run.
@case "TX-01" 5 :graphics begin
    dev = testdevice()
    atlas = Image(dev, RGBA{N0f8}, 1024, 1024)
    old = textureindex(dev, atlas)
    gate = MantleArray(dev, Int32[0])
    ga, outa = indexgraph(dev, atlas; gate, slow = true)
    gb, outb = indexgraph(dev, atlas)
    run!(ga); run!(gb)                                     # compiled
    run!(ga)                                               # in flight for about 200 ms with the old index
    resize!(atlas, 2048, 2048)
    @test size(atlas) == (2048, 2048)
    run!(gb)                                               # applies the resize behind ga's use of the old index
    other = Image(dev, RGBA{N0f8}, 16, 16)             # created while the old index is still retiring
    @test textureindex(dev, other) != old
    @test Array(outa)[1] == old
    new = Array(outb)[1]
    @test new != old && new == textureindex(dev, atlas)
end

# api.md 2.6; resizing-and-raytracing.jl B.9: an upload of the same size and type writes the image the texture has: same index, no cell write. copyto! of 64 × 64 texels.
@case "TX-02" 5 :graphics begin
    dev = testdevice()
    tex = Image(dev, RGBA{N0f8}, 64, 64)
    index = textureindex(dev, tex)
    data = texels(64, 64)
    copyto!(tex, data)
    i, log = stepsduring(() -> textureindex(dev, tex), dev)
    @test i == index
    @test :CellWrite ∉ stepkinds(log, tex)
    @test Array(tex) == data
end

# api.md 2.6: an upload of a new size is a new image under a new index, as for resize!. copy! of 128 × 32 texels into a 64 × 64 texture.
@case "TX-03" 5 :graphics begin
    dev = testdevice()
    tex = Image(dev, RGBA{N0f8}, 64, 64)
    index = textureindex(dev, tex)
    data = texels(128, 32)
    copy!(tex, data)
    @test size(tex) == (128, 32)
    i, log = stepsduring(() -> textureindex(dev, tex), dev)
    @test i != index
    @test :CellWrite in stepkinds(log, tex)
    @test Array(tex) == data
end

# resizing-and-raytracing.jl B.9: two resizes before any submission leave no intermediate image or index behind. Repeated 40 times; indices and pool bytes stay bounded.
@case "TX-04" 5 :graphics begin
    dev = testdevice()
    tex = Image(dev, RGBA{N0f8}, 64, 64)
    textureindex(dev, tex)
    indices = UInt32[]
    reservedbytes = Int[]
    for k in 1:40
        side = isodd(k) ? 256 : 64
        resize!(tex, 128, 128)                             # the intermediate size, never submitted
        resize!(tex, side, side)
        @test size(tex) == (side, side)
        push!(indices, textureindex(dev, tex))
        push!(reservedbytes, memory(dev).reserved)
    end
    @test length(unique(indices)) <= 3                     # indices come back through the free list
    @test reservedbytes[end] == reservedbytes[10]          # no image kept per intermediate size
end

# api.md 2.6: a texture resize is never a recompile. 100 resizes across runs of a graph using the texture.
@case "TX-05" 5 :graphics begin
    dev = testdevice()
    tex = Image(dev, RGBA{N0f8}, 64, 64)
    g, out = indexgraph(dev, tex)
    run!(g)
    _, d = counted() do
        for k in 1:100
            resize!(tex, 64 + k, 64)
            run!(g)
        end
    end
    @test d.plancompiles == 0 && d.kernelcompiles == 0
    @test size(tex) == (164, 64) && Array(out)[1] == textureindex(dev, tex)
end

# resizing-and-raytracing.jl B.9, descriptorsets.adoc:793-807: a new index's descriptor is written while command buffers using other indices are pending. A run using texture a in flight; texture b resized and used.
@case "TX-06" 5 :graphics begin
    dev = testdevice()
    a = Image(dev, RGBA{N0f8}, 64, 64)
    b = Image(dev, RGBA{N0f8}, 64, 64)
    olda, oldb = textureindex(dev, a), textureindex(dev, b)
    gate = MantleArray(dev, Int32[0])
    ga, outa = indexgraph(dev, a; gate, slow = true)
    run!(ga)
    run!(ga)                                               # pending, using a's index
    resize!(b, 128, 128)
    newb = textureindex(dev, b)                            # writes b's new index while ga is pending
    @test newb != oldb && newb != olda
    @test Array(outa)[1] == olda
    # VAL: clean (UPDATE_UNUSED_WHILE_PENDING: the written index is not used by the pending buffer)
end

# api.md 2.2 (free!), 5 (the last hold); resizing-and-raytracing.jl B.9, B.10: at the texture's last hold its images and indices are released; a pending resize is dropped. free!(tex) with a resize pending.
@case "TX-07" 5 :graphics begin
    dev = testdevice()
    base = memory(dev)
    tex = Image(dev, RGBA{N0f8}, 64, 64)
    textureindex(dev, tex)
    resize!(tex, 128, 128)
    _, d = counted(() -> (free!(tex); memory(dev)))
    @test submitcount(d) == 0                              # the resize is never applied
    @test memory(dev).cells == base.cells
end

# api.md 2.6; resizing-and-raytracing.jl B.9: resize! keeps the texels where the old and new sizes overlap. resize! of a texture with contents.
@case "TX-08" 5 :graphics begin
    dev = testdevice()
    data = texels(64, 64)
    tex = Image(dev, RGBA{N0f8}, 64, 64)
    copyto!(tex, data)
    resize!(tex, 128, 96)
    @test Array(tex)[1:64, 1:64] == data
    resize!(tex, 32, 32)
    @test Array(tex) == data[1:32, 1:32]
end

# resizing-and-raytracing.jl B.9 (Metal residency; textures): on Metal the new resource ID is in the residency set when the regrown texture is used. A resize, an upload, a kernel and a host read.
@case "TX-09" 9 :metal begin
    dev = testdevice()
    tex = Image(dev, RGBA{N0f8}, 64, 64)
    old = textureindex(dev, tex)
    resize!(tex, 128, 128)
    data = texels(128, 128)
    copyto!(tex, data)
    @test textureindex(dev, tex) != old
    @test Array(tex) == data
    # The old resource ID leaving the residency set after its token is not visible through the suite's hooks.
end
