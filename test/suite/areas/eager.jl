# Eager calls: operations on device arrays run now (catalogue-core.md, table
# 18: EAGER). Contracts: api.md:126, 451, 481; recording-plan.md "Eager work"
# (933-982); internals.jl section 11 (2181-2210). Line numbers of 2026-10-05.
#
# Assumed of the hooks: Mantle.steplog entries have `kind` (a Symbol),
# `target` (the root array) and `token`. Device side, lengthof(a) arrives as
# a one-element integer array (`len[1] = n`).

using KernelAbstractions: @kernel, @index
using LinearAlgebra: mul!

@kernel function ea_scale!(x, s)
    i = @index(Global)
    x[i] *= s
end

# a[i] = i for i <= n, and n as a's length; launched over the capacity.
@kernel function ea_writelength!(a, len, n)
    i = @index(Global)
    if i <= n
        a[i] = Int32(i)
    end
    if i == 1
        len[1] = n
    end
end

# marks[i] = 1 for each work item the launch has.
@kernel function ea_mark!(marks, a)
    i = @index(Global)
    marks[i] = Int32(1)
end

"""`f()` and the prefix steps applied while it ran (the step log restarted around it)."""
function stepsduring(f, dev = testdevice())
    Mantle.steplog!(dev, false)
    Mantle.steplog!(dev, true)
    v = f()
    log = Mantle.steplog(dev)
    Mantle.steplog!(dev, false)
    return v, log
end

"""The CUDA module; loaded wherever a CudaDevice exists."""
cudamodule() = Base.require(Base.PkgId(Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA"))

# ── 18. Eager calls (EAGER) ──

# api.md:451, recording-plan.md:959-971: a KA launch, dispatch!, broadcast, fill!, copyto!, mul!, fft and render!(dev) on device arrays: every launch goes through compilekernel and is one submission (trace!(dev) is left to the ray-tracing areas).
@case "EAGER-01" 6 begin
    d = FaultDevice(testdevice())
    x = testdata(Float32, 1024)
    a, b = MantleArray(d, x), MantleArray(d, Float32, 1024)
    singles = (() -> fill!(b, 1f0), () -> (b .= a .* 2f0), () -> copyto!(b, a),
               () -> ea_scale!(d)(b, 2f0; ndrange = 1024), () -> dispatch!(d, ea_scale!, (b, 2f0), 1024))
    for op in singles
        k = calls(d, :compilekernel)
        _, diag = counted(op, d)
        @test submitcount(diag) == 1
        @test calls(d, :compilekernel) == k + 1
    end
    @test Array(b) == x .* 4f0
    A = MantleArray(d, reshape(testdata(Float32, 4096; seed = 2), 64, 64))
    B = MantleArray(d, reshape(testdata(Float32, 4096; seed = 3), 64, 64))
    C = MantleArray(d, Float32, 64, 64)
    z = MantleArray(d, ComplexF32.(x))
    for op in (() -> mul!(C, A, B), () -> Mantle.fft(z))
        k = calls(d, :compilekernel)
        _, diag = counted(op, d)
        @test submitcount(diag) >= 1
        @test calls(d, :compilekernel) - k == submitcount(diag)   # one submission per launch
    end
    @test Array(C) ≈ Array(A) * Array(B)
    img = Image(d, Float32, 16, 16)
    _, diag = counted(() -> render!(p -> nothing, d, img => Clear(1f0)), d)
    @test submitcount(diag) == 1
    @test all(==(1f0), Array(img))
end

# recording-plan.md:964-966: 100 eager calls are 100 submissions; no plan is compiled or cached for them.
@case "EAGER-02" 3 begin
    dev = testdevice()
    a = MantleArray(dev, zeros(Int32, 1024))
    _, diag = counted(dev) do
        for _ in 1:100
            a .+= Int32(1)
        end
    end
    @test submitcount(diag) == 100
    @test diag.plancompiles == 0
    @test Array(a) == fill(Int32(100), 1024)
end

# internals.jl:2185-2195: a pending resize and store on an argument are applied in front of the launch, in its one submission.
@case "EAGER-03" 4 begin
    dev = testdevice()
    x, y = testdata(Int32, 1024), testdata(Int32, 1024; seed = 2)
    a = MantleArray(dev, x)
    resize!(a, 2048)
    a[1025:2048] = y
    (_, diag), log = stepsduring(() -> counted(() -> (a .+= Int32(1)), dev), dev)
    steps = [s for s in log if s.target === a]
    @test submitcount(diag) == 1
    @test !isempty(steps)
    @test all(s -> s.token == first(steps).token, steps)
    @test isempty(Mantle.pendingof(a))
    @test Array(a) == [x; y] .+ Int32(1)
end

# internals.jl:2190-2203: an eager call that throws under its locks (a freed argument) retires its packed argument region at once: 100 such calls leave memory at the baseline.
@case "EAGER-04" 3 begin
    dev = testdevice()
    a, b = MantleArray(dev, Int32, 1024), MantleArray(dev, Int32, 1024)
    g = Graph(dev)
    t = MantleArray(g, Int32, 1024)
    t .= a
    copyto!(b, t)                         # g holds a, so its storage stays after free!
    free!(a)
    base = memory(dev)
    for _ in 1:100
        @test_throws ArgumentError (b .= a .+ Int32(1))
    end
    @test memory(dev) == base
end

# api.md:481, recording-plan.md:962-963: an eager call on an array lent to withview throws; after the block it runs.
@case "EAGER-05" 10 :vendorview begin
    dev = testdevice()
    cuda = cudamodule()
    a = MantleArray(dev, zeros(Int32, 1024))
    withview(cuda.CuArray, a, cuda.stream()) do v
        @test_throws ArgumentError (a .+= Int32(1))
        @test_throws ArgumentError fill!(a, Int32(2))
        @test_throws ArgumentError dispatch!(dev, ea_mark!, (a, a), 1024)
    end
    a .+= Int32(1)
    @test Array(a) == fill(Int32(1), 1024)
end

# internals.jl:2196, resizing-and-raytracing.jl:804-805: an eager launch over a length a kernel wrote is indirect from the cell: 1000 work items, not the capacity of 4096.
@case "EAGER-06" 4 begin
    dev = testdevice()
    a = MantleArray(dev, Int32, 0; capacity = 4096)
    marks = MantleArray(dev, zeros(Int32, 4096))
    dispatch!(dev, ea_writelength!, (a, lengthof(a), Int32(1000)), 4096)
    dispatch!(dev, ea_mark!, (marks, a), launchrange(a))
    @test count(==(Int32(1)), Array(marks)) == 1000
    @test devicesize(a) == 1000
    @test Array(a) == Int32.(1:1000)
end

# api.md:126: an eager call returns once submitted, before the GPU finishes: a launch of about 1 s returns within 0.5 s, and a host read then sees its result.
@case "EAGER-07" 3 begin
    dev = testdevice()
    out = MantleArray(dev, Int32, 1)
    slowwrite!(dev, out, 1; ms = 1)       # compiled and calibrated
    Array(out)
    t = @elapsed slowwrite!(dev, out, 7; ms = 1000)
    @test t < 0.5
    @test Array(out) == Int32[7]
end
