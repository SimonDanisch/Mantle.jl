# Vendor views: withview (catalogue-core.md, table 20: VIEW). Contracts:
# api.md:200-207, 481; recording-plan.md "Arrays", vendor views (260-270);
# internals.jl section 14 (2311-2368). Line numbers of 2026-10-05.
#
# withview's name and form are open (open decision 9); written here as
# `withview(f, CuArray | MtlArray, a, stream)`. Vendor kernels are KA kernels
# launched on the view's backend, which uses the task's stream (CUDA) or the
# device's global queue (Metal): the stream passed to withview.

using KernelAbstractions: @kernel, @index, get_backend

const VW_LEN = 64 * 2^20          # Int32 elements of a large array: 256 MiB
const VW_BYTES = 4 * VW_LEN

# Spins `iters` steps, then writes `value` into x[1] (vendor work in flight).
@kernel function vw_spinwrite!(x, value, iters)
    acc = Int32(0)
    for i in Int32(1):iters
        acc = acc * Int32(1103515245) + i
    end
    x[1] = value + (acc == Int32(-7) ? Int32(1) : Int32(0))
end

"""The CUDA module; loaded wherever a CudaDevice exists."""
cudamodule() = Base.require(Base.PkgId(Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA"))

"""The Metal module; loaded wherever a MetalDevice exists."""
metalmodule() = Base.require(Base.PkgId(Base.UUID("dde4c033-4e86-420c-a63e-0dd931031962"), "Metal"))

"""Writes 7 into `v[1]` on `v`'s stream after about `ms` milliseconds."""
spinwrite!(v, dev; ms = 500) = vw_spinwrite!(get_backend(v))(v, Int32(7), Int32(calibratedspin(dev, ms)); ndrange = 1)

"""Waits for `t` (a watchdog fails the test after `timeout` s) and returns what it threw, or `nothing`."""
function thrownby(t::Task; timeout = 30)
    ok = timedwait(() -> istaskdone(t), timeout) === :ok
    @test ok
    ok || error("task did not finish after $timeout s: likely a deadlock")
    return istaskfailed(t) ? t.result : nothing
end

"""A FaultDevice over the suite's device whose pool and cell arena were used once."""
function faultdevice(dev = testdevice())
    d = FaultDevice(dev)
    free!(MantleArray(d, Int32, 1))
    return d
end

"""Caps `d`'s budget at what its pool holds now plus `headroom` bytes [FI-9]; returns the cap."""
function limit!(d, headroom)
    cap = Mantle.reserved(d) + headroom
    inject!(d, :budget, Answer(calls(d, :budget) + 1, cap))
    return cap
end

# ── 20. Vendor views (VIEW) ──

# api.md:202-203: withview on an array with a pending store: the stream sees the store.
@case "VIEW-01" 10 :vendorview begin
    dev = testdevice()
    cuda = cudamodule()
    x = testdata(Int32, 1024)
    a = MantleArray(dev, x)
    a[1:4] = Int32[1, 2, 3, 4]
    seen = Ref{Vector{Int32}}()
    withview(cuda.CuArray, a, cuda.stream()) do v
        seen[] = Array(v)
    end
    @test seen[] == [Int32[1, 2, 3, 4]; x[5:end]]
end

# api.md:203-204: a vendor kernel of about 500 ms writes a inside the block; a Mantle call on a after the block waits for the stream.
@case "VIEW-02" 10 :vendorview begin
    dev = testdevice()
    cuda = cudamodule()
    a = MantleArray(dev, zeros(Int32, 1024))
    withview(v -> spinwrite!(v, dev), cuda.CuArray, a, cuda.stream())
    a .+= Int32(1)
    @test Array(a) == [Int32(8); fill(Int32(1), 1023)]
end

# api.md:204: the block throws after starting a vendor write: the exception propagates, a later Mantle call still waits for the stream, and the view's hold is dropped (memory back after free!).
@case "VIEW-03" 10 :vendorview begin
    dev = testdevice()
    cuda = cudamodule()
    base = memory(dev)
    a = MantleArray(dev, zeros(Int32, 1024))
    @test_throws ErrorException withview(cuda.CuArray, a, cuda.stream()) do v
        spinwrite!(v, dev)
        error("thrown inside the block")
    end
    a .+= Int32(1)
    @test Array(a)[1] == 8
    free!(a)
    @test memory(dev) == base
end

# api.md:206: an eager call, a store, resize! and Array on a inside the block each throw; after the block they work.
@case "VIEW-04" 10 :vendorview begin
    dev = testdevice()
    cuda = cudamodule()
    a = MantleArray(dev, zeros(Int32, 1024))
    withview(cuda.CuArray, a, cuda.stream()) do v
        @test_throws ArgumentError (a .+= Int32(1))
        @test_throws ArgumentError (a[1:1] = Int32[1])
        @test_throws ArgumentError resize!(a, 2048)
        @test_throws ArgumentError Array(a)
    end
    a[1:1] = Int32[1]
    @test Array(a)[1] == 1
end

# api.md:206, internals.jl:2342-2344 (ambiguity 8, settled there): another task runs a graph that writes a during the block, nothing pending on a: the run throws; after the block it runs.
@case "VIEW-05" 10 :vendorview begin
    dev = testdevice()
    cuda = cudamodule()
    a = MantleArray(dev, zeros(Int32, 1024))
    g = Graph(dev)
    t = MantleArray(g, Int32, 1024)
    fill!(t, Int32(5))
    copyto!(a, t)
    run!(g)
    withview(cuda.CuArray, a, cuda.stream()) do v
        @test thrownby(Threads.@spawn run!(g)) isa ArgumentError
    end
    fill!(a, Int32(0))
    run!(g)
    @test Array(a) == fill(Int32(5), 1024)
end

# api.md:207: inside the block, Mantle calls on another array complete, on this task and on another: no Mantle lock is held while the block runs.
@case "VIEW-06" 10 :vendorview begin
    dev = testdevice()
    cuda = cudamodule()
    a, b = MantleArray(dev, zeros(Int32, 1024)), MantleArray(dev, zeros(Int32, 1024))
    withview(cuda.CuArray, a, cuda.stream()) do v
        b .+= Int32(1)
        @test Array(b) == fill(Int32(1), 1024)
        @test withtimeout(() -> (b .+= Int32(1); Array(b)), 30) == fill(Int32(2), 1024)
    end
end

# internals.jl:2315-2317: free!(a) from another task while a vendor kernel writes a inside the block: nothing is released before the stream's token (canaries allocated during and right after the block stay intact).
@case "VIEW-07" 10 :vendorview begin
    dev = testdevice()
    cuda = cudamodule()
    s = cuda.stream()
    base = memory(dev)
    a = MantleArray(dev, Int32, 1024)
    inside = Any[]
    withview(cuda.CuArray, a, s) do v
        spinwrite!(v, dev)
        withtimeout(() -> free!(a), 30)
        append!(inside, [canary(dev, 1024) for _ in 1:4])
    end
    after = [canary(dev, 1024) for _ in 1:4]
    cuda.synchronize(s)
    @test all(intact, inside)
    @test all(intact, after)
    foreach(free!, inside); foreach(free!, after)
    @test memory(dev) == base
end

# internals.jl:2313-2315: withview of an array already lent (nested, the same array) throws.
@case "VIEW-08" 10 :vendorview begin
    dev = testdevice()
    cuda = cudamodule()
    s = cuda.stream()
    a = MantleArray(dev, zeros(Int32, 1024))
    withview(cuda.CuArray, a, s) do v
        @test_throws ArgumentError withview(w -> nothing, cuda.CuArray, a, s)
    end
end

# api.md:200: withview(MtlArray, …) on Metal: the view sees a pending store, and a Mantle call after the block waits for the queue's work.
@case "VIEW-09" 9 :metal begin
    dev = testdevice()
    mtl = metalmodule()
    x = testdata(Int32, 1024)
    a = MantleArray(dev, x)
    a[1:4] = Int32[1, 2, 3, 4]
    seen = Ref{Vector{Int32}}()
    withview(mtl.MtlArray, a, mtl.global_queue(mtl.device())) do v
        seen[] = Array(v)
        spinwrite!(v, dev)
    end
    a .+= Int32(1)
    @test seen[] == [Int32[1, 2, 3, 4]; x[5:end]]
    @test Array(a) == [Int32[8, 3, 4, 5]; x[5:end] .+ Int32(1)]
end

# recording-plan.md:263-264, internals.jl:2333-2336: withview on an evicted array restores it first: the view holds its contents.
@case "VIEW-10" 10 :vendorview :separateheap begin
    # Needs faultdevice.jl to forward the device type's vendor-view verb once it has a name (G11).
    d = faultdevice()
    cuda = cudamodule()
    x = testdata(Int32, VW_LEN)
    a = MantleArray(d, x)
    limit!(d, VW_BYTES ÷ 2)
    free!(MantleArray(d, Int32, VW_LEN))  # evicts a
    seen = Ref{Vector{Int32}}()
    withview(cuda.CuArray, a, cuda.stream()) do v
        seen[] = Array(view(v, 1:8))
    end
    @test seen[] == x[1:8]
end

# api.md:203: a graph run writing a is on the GPU when withview starts: the stream waits for it and sees the write.
@case "VIEW-11" 10 :vendorview begin
    dev = testdevice()
    cuda = cudamodule()
    a = MantleArray(dev, zeros(Int32, 1024))
    g = Graph(dev)
    slowwrite!(g, a, 7; ms = 500)
    run!(g)
    fill!(a, Int32(0))
    run!(g)                               # on the GPU for about 500 ms
    seen = Ref{Vector{Int32}}()
    withview(cuda.CuArray, a, cuda.stream()) do v
        seen[] = Array(v)
    end
    @test seen[][1] == 7
end
