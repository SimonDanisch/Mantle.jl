# Consumer patterns of the video editor, written against Mantle alone: frame
# graphs sharing device arrays, per-frame values as one-element stores, a DNN
# beside the frame, resolution changes, double buffering through an array
# GPURef, readback, and the steady-state frame (recording-plan.md Ordering,
# Run, Memory; api.md 2.2-2.6b; examples.jl 24-33).
#
# Pixels are Int32, so every result compares exactly with a host model.
# Phase: the latest ledger phase among the contracts a case relies on: stores
# and resize! are phase 4, the array GPURef phase 6 (the suite runs nothing
# before APIPHASE).

using KernelAbstractions: @kernel, @index

const ED_W = 64
const ED_H = 48
const ED_FRAMES = 24

"""Host model of a decoded frame `f`."""
decodedframe(w, h, f) = [Int32((3x + 5y + 7f) % 256) for x in 1:w, y in 1:h]

"""Host model of the grade: every pixel times the frame's gain, plus 1."""
gradedframe(frame, gain) = frame .* Int32(gain) .+ Int32(1)

"""Host model of the preview: 2×2 box sums."""
previewof(img) = [sum(img[2x-1:2x, 2y-1:2y]) for x in 1:size(img, 1) ÷ 2, y in 1:size(img, 2) ÷ 2]

"""Host model of `ed_paint!` at time `t`."""
paintedcanvas(w, h, t) = [Int32(3x + 5y + t) for x in 1:w, y in 1:h]

"""The gain of frame `f`: a per-frame value the editor stores."""
framegain(f) = Int32(f % 3 + 1)

@kernel function ed_decode!(frame, findex)
    x, y = @index(Global, NTuple)
    @inbounds frame[x, y] = (Int32(3x + 5y) + Int32(7) * findex[1]) % Int32(256)
end

@kernel function ed_grade!(out, frame, gain)
    i = @index(Global)
    @inbounds out[i] = frame[i] * gain[1] + Int32(1)
end

@kernel function ed_preview!(preview, img)
    x, y = @index(Global, NTuple)
    @inbounds preview[x, y] = img[2x-1, 2y-1] + img[2x, 2y-1] + img[2x-1, 2y] + img[2x, 2y]
end

# One work item: the frame's checksum into its slot of `sums`.
@kernel function ed_checksum!(sums, img, findex)
    s = Int32(0)
    for i in 1:length(img)
        @inbounds s += img[i]
    end
    @inbounds sums[findex[1]] = s
end

# Over the canvas's current length; the width from the device-side dims, so
# it follows a resize! without a recompile.
@kernel function ed_paint!(canvas, t)
    i = @index(Global)
    w = size(canvas, 1)
    x = (i - 1) % w + 1
    y = (i - 1) ÷ w + 1
    @inbounds canvas[i] = Int32(3x + 5y) + t[1]
end

@kernel function ed_alpha!(alpha, canvas)
    i = @index(Global)
    @inbounds alpha[i] = canvas[i] % Int32(2)
end

@kernel function ed_copy!(dst, src)
    i = @index(Global)
    @inbounds dst[i] = src[i]
end

@kernel function ed_scale!(y, x, s)
    i = @index(Global)
    @inbounds y[i] = x[i] * s
end

"""
A matte network as one graph with a host node between its stages
(`output = 3 * (2 * input)`); `kw` goes to `Graph` (open decision 12:
`priority`, keyword and values open).
"""
function mattegraph(dev, input, output, host; kw...)
    g = Graph(dev; kw...)
    h = MantleArray(g, Int32, length(input))
    dispatch!(g, ed_scale!, (h, input, Int32(2)), length(h))
    dispatch!(g, host, (h,); writes = ())
    dispatch!(g, ed_scale!, (output, h, Int32(3)), length(output))
    return g
end

"""The editor's frame: paints `canvas` (an array or an array GPURef) at the time in `t`."""
function paintgraph(dev, canvas, t; kw...)
    g = Graph(dev; kw...)
    dispatch!(g, ed_paint!, (canvas, t), launchrange(canvas))
    return g
end

# ED-01: decode → effects → preview → export as four graphs connected by
# device arrays (plan, "One call, one graph": separate calls are separate
# graphs). Per frame the editor stores the frame index and the gain into
# one-element arrays and runs the four graphs; nothing waits on the host. Each
# later graph sees each earlier graph's write of the same frame: the export's
# and the preview's checksums of every frame match the host model.
@case "ED-01" 4 begin
    dev = testdevice()
    w, h, frames = ED_W, ED_H, ED_FRAMES
    findex, gain = MantleArray(dev, Int32[1]), MantleArray(dev, Int32[1])
    frame, grade = MantleArray(dev, Int32, w, h), MantleArray(dev, Int32, w, h)
    preview = MantleArray(dev, Int32, w ÷ 2, h ÷ 2)
    previewsums, exportsums = MantleArray(dev, zeros(Int32, frames)), MantleArray(dev, zeros(Int32, frames))
    decoder = Graph(dev)
    dispatch!(decoder, ed_decode!, (frame, findex), size(frame))
    effects = Graph(dev)
    dispatch!(effects, ed_grade!, (grade, frame, gain), length(grade))
    previewer = Graph(dev)
    dispatch!(previewer, ed_preview!, (preview, grade), size(preview))
    dispatch!(previewer, ed_checksum!, (previewsums, preview, findex), 1)
    exporter = Graph(dev)
    dispatch!(exporter, ed_checksum!, (exportsums, grade, findex), 1)
    graphs = (decoder, effects, previewer, exporter)
    function showframe(f)
        findex[1:1] = Int32[f]
        gain[1:1] = Int32[framegain(f)]
        foreach(run!, graphs)
    end
    showframe(1)                                     # compiles the four plans
    _, d = counted(() -> foreach(showframe, 2:frames))
    @test d.hostwaits == 0
    @test d.plancompiles == 0
    expected = [sum(gradedframe(decodedframe(w, h, f), framegain(f))) for f in 1:frames]
    @test Array(exportsums) == expected
    @test Array(previewsums) == expected             # 2×2 sums add up to the same total
    @test Array(preview) == previewof(gradedframe(decodedframe(w, h, frames), framegain(frames)))
    foreach(free!, graphs)
end

# ED-02: a DNN graph with a host node beside the frame graph, on separate
# arena parts (examples.jl 33; plan, Ordering "Arenas"; open decision 12: core
# decides parts from the graphs' priorities). While the DNN's run is held in
# its host function, holding its arena parts, frames still run and are
# correct, and wait for nothing on the host. The DNN's result is exact after
# its host function returns.
@case "ED-02" 4 begin
    dev = testdevice()
    input, output = MantleArray(dev, collect(Int32, 1:4096)), MantleArray(dev, Int32, 4096)
    gate = Gate()
    matte = mattegraph(dev, input, output, blockinghost(gate); priority = 0)   # open decision 12
    canvas, t = MantleArray(dev, Int32, ED_W, ED_H), MantleArray(dev, Int32[0])
    frame = paintgraph(dev, canvas, t; priority = 1)                          # open decision 12
    run!(frame)
    letgo!(gate)
    run!(matte)                                      # compiles with the gate open
    waitentered(gate)
    task = Threads.@spawn run!(matte)
    try
        waitentered(gate)                            # the DNN's run is between its stages
        for f in 1:8
            t[1:1] = Int32[f]
            _, d = counted(() -> withtimeout(() -> run!(frame), 30))
            @test d.hostwaits == 0
            @test Array(canvas) == paintedcanvas(ED_W, ED_H, f)
        end
        @test !istaskdone(task)
    finally
        letgo!(gate)
    end
    withtimeout(() -> fetch(task), 30)
    @test Array(output) == 6 .* collect(Int32, 1:4096)
    foreach(free!, (matte, frame))
end

# ED-03: the same pair with no priorities, so core picks the arena parts
# (either choice is correct, examples.jl 33): the DNN runs 50 times on one
# thread while the editor runs 50 frames on another. No deadlock, every frame
# and the DNN's result exact.
@case "ED-03" 4 begin
    dev = testdevice()
    input, output = MantleArray(dev, collect(Int32, 1:4096)), MantleArray(dev, Int32, 4096)
    matte = mattegraph(dev, input, output, HostCall(_ -> nothing))
    canvas, t = MantleArray(dev, Int32, ED_W, ED_H), MantleArray(dev, Int32[0])
    frame = paintgraph(dev, canvas, t)
    run!(matte)
    run!(frame)
    task = Threads.@spawn foreach(_ -> run!(matte), 1:50)
    wrong = 0
    for f in 1:50
        t[1:1] = Int32[f]
        withtimeout(() -> run!(frame), 60)
        wrong += Array(canvas) != paintedcanvas(ED_W, ED_H, f)
    end
    withtimeout(() -> fetch(task), 120)
    @test wrong == 0
    @test Array(output) == 6 .* collect(Int32, 1:4096)
    foreach(free!, (matte, frame))
end

# ED-04: a resolution change: resize! of the canvas arrays (api.md 2.2;
# B3.13). The call changes the host view only: nothing is allocated and the
# change stays pending until the next frame applies it. The next frame is
# correct at the new size. The first resize! recompiles once (the plan
# launched over the fixed arrays directly); later ones, smaller, larger and
# past the storage, recompile nothing.
@case "ED-04" 4 begin
    dev = testdevice()
    canvas, alpha = MantleArray(dev, Int32, ED_W, ED_H), MantleArray(dev, Int32, ED_W, ED_H)
    t = MantleArray(dev, Int32[0])
    g = Graph(dev)
    dispatch!(g, ed_paint!, (canvas, t), launchrange(canvas))
    dispatch!(g, ed_alpha!, (alpha, canvas), launchrange(alpha))
    run!(g)
    sizes = ((2ED_W, 2ED_H), (ED_W, ED_H), (ED_W ÷ 2, ED_H), (4ED_W, 3ED_H), (ED_W, 2ED_H))
    for (k, (w, h)) in enumerate(sizes)
        base = Mantle.reserved(dev)
        resize!(canvas, (w, h))
        resize!(alpha, (w, h))
        @test Mantle.reserved(dev) == base
        @test size(canvas) == (w, h)
        @test !isempty(Mantle.pendingof(canvas))
        t[1:1] = Int32[k]
        _, d = counted(() -> run!(g))
        @test d.plancompiles == (k == 1 ? 1 : 0)
        @test Array(canvas) == paintedcanvas(w, h, k)
        @test Array(alpha) == paintedcanvas(w, h, k) .% Int32(2)
    end
    free!(g)
end

# ED-05: double buffering through an array GPURef (api.md 2.6b): each frame
# rebinds the box to the buffer it draws into (a pending cell write, no copy)
# while the other buffer keeps the previous frame. No recompile, no host wait;
# both buffers exact. Then per frame with reads: the shown buffer holds the
# previous frame while the drawn one holds this frame.
@case "ED-05" 6 begin
    dev = testdevice()
    front, back = MantleArray(dev, Int32, ED_W, ED_H), MantleArray(dev, Int32, ED_W, ED_H)
    target = GPURef(dev, front)
    t = MantleArray(dev, Int32[0])
    g = paintgraph(dev, target, t)
    buffers(f) = isodd(f) ? (front, back) : (back, front)
    run!(g)
    _, d = counted() do
        for f in 1:ED_FRAMES
            target[] = first(buffers(f))
            t[1:1] = Int32[f]
            run!(g)
        end
    end
    @test d.plancompiles == 0
    @test d.hostwaits == 0
    @test Array(first(buffers(ED_FRAMES))) == paintedcanvas(ED_W, ED_H, ED_FRAMES)
    @test Array(last(buffers(ED_FRAMES))) == paintedcanvas(ED_W, ED_H, ED_FRAMES - 1)
    for f in ED_FRAMES+1:ED_FRAMES+8
        drawn, shown = buffers(f)
        target[] = drawn
        @test target[] === drawn
        t[1:1] = Int32[f]
        run!(g)
        @test Array(shown) == paintedcanvas(ED_W, ED_H, f - 1)
        @test Array(drawn) == paintedcanvas(ED_W, ED_H, f)
    end
    free!(g)
    free!(target)
end

# ED-06: the canvas read back every frame (api.md 2.2 Array and
# copyto!(host, a): its own submission, one wait, a memcpy), and a copy the
# frame graph writes into Readback memory (name open), whose read is a wait
# and a memcpy with no submission.
@case "ED-06" 4 begin
    dev = testdevice()
    canvas = MantleArray(dev, Int32, ED_W, ED_H)
    shown = MantleArray(dev, Int32, ED_W, ED_H; memory = Mantle.Readback())
    t = MantleArray(dev, Int32[0])
    host = Matrix{Int32}(undef, ED_W, ED_H)
    g = Graph(dev)
    dispatch!(g, ed_paint!, (canvas, t), length(canvas))
    dispatch!(g, ed_copy!, (shown, canvas), length(shown))
    for f in 1:8
        t[1:1] = Int32[f]
        run!(g)
        expected = paintedcanvas(ED_W, ED_H, f)
        a, d = counted(() -> Array(canvas))
        @test a == expected
        @test submitcount(d) == 1
        @test d.hostwaits == 1
        copyto!(host, canvas)
        @test host == expected
        r, d = counted(() -> Array(shown))
        @test r == expected
        @test submitcount(d) == 0
    end
    free!(g)
end

# ED-07: the steady-state frame (api.md 6, test_run_is_submit.jl): decode and
# grade in one frame graph, the frame's values as one-element stores before
# each run. Each run is one submission and compiles, records, allocates and
# waits for nothing (the stores took their upload regions at their calls and
# ride inline in the run's submission).
@case "ED-07" 4 begin
    dev = testdevice()
    findex, gain = MantleArray(dev, Int32[1]), MantleArray(dev, Int32[1])
    canvas = MantleArray(dev, Int32, ED_W, ED_H)
    g = Graph(dev)
    frame, grade = MantleArray(g, Int32, ED_W, ED_H), MantleArray(g, Int32, ED_W, ED_H)
    dispatch!(g, ed_decode!, (frame, findex), size(frame))
    dispatch!(g, ed_grade!, (grade, frame, gain), length(grade))
    copyto!(canvas, grade)                           # a copy node: the frame's output
    function storeframe(f)
        findex[1:1] = Int32[f]
        gain[1:1] = Int32[framegain(f)]
    end
    for f in 1:3
        storeframe(f)
        run!(g)
    end
    stats = map(4:103) do f
        storeframe(f)
        last(counted(() -> run!(g)))
    end
    @test all(d -> submitcount(d) == 1, stats)
    @test all(d -> d.plancompiles == 0 && d.pipelinecompiles == 0 && d.kernelcompiles == 0, stats)
    @test all(d -> d.records == 0, stats)
    @test all(d -> d.allocations == 0, stats)
    @test all(d -> d.hostwaits == 0, stats)
    @test Array(canvas) == gradedframe(decodedframe(ED_W, ED_H, 103), framegain(103))
    free!(g)
end
