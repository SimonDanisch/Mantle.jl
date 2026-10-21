# Devices, the registry and caps (DEV). Contracts: api.md 2.1, 4.1 and 10;
# recording-plan.md "Devices"; internals.jl section 13. Process-level cases
# (the registry exists once per process) start a fresh Julia in this
# environment and carry the `:lavapipe` need, so they run once per suite, on
# the lavapipe pass.

import KernelAbstractions as KA
using LinearAlgebra: triu!

function dev_add!(dst, src, v)
    i = KI.get_global_id().x
    i <= length(dst) || return nothing    # ndrange is rounded up to whole work-groups
    @inbounds dst[i] = src[i] + v
    return nothing
end

function dev_fill!(a, v)
    i = KI.get_global_id().x
    i <= length(a) || return nothing
    @inbounds a[i] = v
    return nothing
end

"""Standard output of `code` run by a fresh Julia process in this environment, with `env` set."""
freshjulia(code; env = ()) = withenv(env...) do
    read(`$(Base.julia_cmd()) --project=$(Base.active_project()) --startup-file=no -e $code`, String)
end

islavapipe(name) = occursin("llvmpipe", lowercase(name))


# submissions(g) lists stages (cut by host nodes), each with its segments (cut
# by budget and queue). api.md 2.3 leaves the shape open; this is the one the
# suite assumes.
segmentcount(g) = sum(s -> length(s.segments), Mantle.submissions(g))

"""
A chain of kernels over two device arrays of `n` Float32 whose estimated cost
(8n bytes per kernel at `caps(dev).bandwidth`, bytes per second) is about
`seconds`. Returns the graph, the array the last kernel writes, and the
number of kernels (that array then holds it in every element).
"""
function costlychain(dev, seconds; n = 1 << 25)
    a, b = MantleArray(dev, Float32, n), MantleArray(dev, Float32, n)
    fill!(a, 0f0)
    g = Graph(dev)
    k = ceil(Int, seconds * caps(dev).bandwidth / (8n))
    for j in 1:k
        isodd(j) ? dispatch!(g, dev_add!, (b, a, 1f0), n) : dispatch!(g, dev_add!, (a, b, 1f0), n)
    end
    return g, isodd(k) ? b : a, k
end

"""The inline store limits caps(dev) reports: 65536 bytes at 4-byte offsets on Vulkan (vkCmdUpdateBuffer)."""
checkinline(c, ::Mantle.LavaDevice) = @test (c.inlinestore, c.inlinealign) == (65536, 4)
checkinline(c, ::Mantle.Device) = @test c.inlinestore >= 0

# ── Devices and caps (DEV) ──

# api.md 2.1 (Device() returns the same object on every call), plan, Devices (one object per hardware device): Device() twice, then from 8 threads at once: the same object every time.
@case "DEV-01" 8 :defaultgpu begin
    d = Device()
    @test Device() === d
    fromthreads = withtimeout(() -> fetch.([Threads.@spawn(Device()) for _ in 1:8]), 60)
    @test all(x -> x === d, fromthreads)
    @test count(x -> x === d, devices()) == 1       # one object, so one pool
end

# api.md 2.1 (devices(): driver objects are created on first use), internals.jl
# section 13: devices() in a fresh process lists every device once and
# opens none. The registry's driver-object creation is not a verb FaultDevice
# can count; the case reads the pool hook instead (no device has a block yet).
@case "DEV-02" 8 :lavapipe begin
    out = split(freshjulia("""
        using Mantle
        ds = devices()
        print(allunique(map(objectid, ds)), " ",
              all(((a, b),) -> a === b, zip(ds, devices())), " ",
              all(d -> Mantle.reserved(d) == 0, ds))
        """))
    @test out == ["true", "true", "true"]
end

# api.md 2.1 (Device(): GPU 0, never a CPU driver), plan, Devices: lavapipe
# enumerated first (Mesa's device-select layer, lavapipe's vendor id 0x10005);
# Device() is still a GPU.
@case "DEV-03" 8 :lavapipe :defaultgpu begin
    @test any(d -> !has(d, :lavapipe), devices())  # a GPU exists to be the default
    firstname, defaultname = split(freshjulia(
        """using Mantle; print(Mantle.name(first(devices())), "|", Mantle.name(Device()))""";
        env = ("MESA_VK_DEVICE_SELECT" => "10005:0",)), "|")
    @test islavapipe(firstname)                     # the order the case needs
    @test !islavapipe(defaultname)
end

# api.md 2.1 (`using Lavapipe_jll` must happen before the first Device()),
# plan, Devices: `using Lavapipe_jll` before and after the first Device() in
# fresh processes. Needs Lavapipe_jll in the main environment. Loaded late it
# changes nothing (the listing equals a process without it) and does not crash
# (a crash fails `read`).
@case "DEV-04" 8 :lavapipe begin
    listed = "print(any(d -> occursin(\"llvmpipe\", lowercase(Mantle.name(d))), devices()))"
    @test freshjulia("using Lavapipe_jll, Mantle; " * listed) == "true"
    late = freshjulia("using Mantle; devices(); using Lavapipe_jll; " * listed)
    @test late == freshjulia("using Mantle; " * listed)
end

# api.md 2.1 (one GPU through two drivers is two devices with two pools), 2.4
# (location): one GPU through LavaDevice and CudaDevice: two devices, two
# pools; mixing their arrays throws at location. The suite supplies a physically verified driver pair; display names are not identities.
@case "DEV-05" 10 :vendorview :vulkanpeer begin
    cu = testdevice()
    lv = first(vulkanpeers(cu))
    a, b = MantleArray(cu, Float32, 16), MantleArray(lv, Float32, 16)
    before = Mantle.reserved(lv)
    large = MantleArray(cu, UInt8, 256 << 20)
    @test Mantle.reserved(lv) == before             # the CUDA allocation is not in the Vulkan pool
    @test_throws ArgumentError location(a, b)
    @test_throws ArgumentError (a .= b)
    free!(large)
end

# api.md 2.1, 2.2, plan, Devices, internals.jl section 2: KA.get_backend on a device array is its device; on a graph array it is the graph's device, Mantle.Device(g).
@case "DEV-06" 6 begin
    dev = testdevice()
    @test KA.get_backend(MantleArray(dev, Float32, 16)) === dev
    g = Graph(dev)
    @test KA.get_backend(MantleArray(g, Float32, 16)) === Mantle.Device(g)
end

# api.md 2.2 (GPUArrays' generic code serves the rest, on devices and graphs
# alike), plan, Arrays (tril/triu as KA kernels on get_backend(dest)):
# GPUArrays' generic triu! on a graph array declares a kernel node into the
# graph and submits nothing; the run computes it.
@case "DEV-07" 6 begin
    dev = testdevice()
    g = Graph(dev)
    x = MantleArray(g, Float32, 8, 8)
    out = MantleArray(dev, Float32, 8, 8)
    fill!(x, 1f0)
    _, d = counted(() -> triu!(x))
    @test submitcount(d) == 0
    copyto!(out, x)
    run!(g)
    @test Array(out) == triu!(ones(Float32, 8, 8))
end

# api.md 2.4 (location: arguments from two devices throw), 7, examples.jl
# "Threads and devices": a broadcast across two devices, and a graph on one
# device given an array of the other: ArgumentError, the graph unchanged.
@case "DEV-08" 6 :twodevices begin
    dev = testdevice()
    other = first(d for d in SUITE.fixtures[].devices if d !== dev)
    a1, a2 = MantleArray(dev, fill(1f0, 16)), MantleArray(other, fill(5f0, 16))
    @test_throws ArgumentError (a2 .= a1)
    @test Array(a2) == fill(5f0, 16)
    g = Graph(dev)
    x = MantleArray(g, Float32, 16)
    out = MantleArray(dev, Float32, 16)
    fill!(x, 2f0)
    @test_throws ArgumentError (x .= a2)
    copyto!(out, x)
    run!(g)
    @test Array(out) == fill(2f0, 16)               # the refused broadcast left no node
end

# api.md 4.1 and execution-contracts.md §6 (fixed capability schema): caps(dev) is immutable and has
# the fields api.md 4.1 and helpers.jl name; Vulkan reports inlinestore 65536,
# inlinealign 4.
@case "DEV-09" 8 begin
    dev = testdevice()
    c = caps(dev)
    @test !ismutable(c)
    @test c == caps(dev)
    for f in (:api, :kernel, :scalars, :atomicfeatures, :software,
              :sparseresidency, :sparsepage, :sparseaddressspace, :maxalloc,
              :maxinstances, :accelbyaddress, :raytracing, :rayquery,
              :display, :displayattached, :throughput, :bandwidth, :transfer,
              :crossqueuecost, :inlinestore, :inlinealign)
        @test f in propertynames(c)
    end
    @test c.kernel isa KI.DeviceCaps
    @test KI.matrix_shapes(c) == KI.matrix_shapes(c.kernel)
    @test KI.wggranularity(c, 32) == KI.wggranularity(c.kernel, 32)
    @test KI.bestshape(c, Float16, Float32) == KI.bestshape(c.kernel, Float16, Float32)
    @test KI.supports_float64(dev) == (Float64 in c.scalars)
    for shape in KI.matrix_shapes(c)
        @test KI.supports(c, shape)
    end
    checkinline(c, dev)
end

# api.md 4 (a verb never throws to signal "not supported"; capabilities are
# answered by caps), 2.2 (resize!): the fault device disables sparse residency and native RT;
# arrays that would move into a reserve, arena transients and a TLAS work
# without a "not supported".
@case "DEV-10" 5 begin
    dev = FaultDevice(testdevice())
    inject!(dev, :caps, Answer(1, withcaps(caps(testdevice());
        sparseresidency = false, raytracing = false, rayquery = false)))
    @test !caps(dev).sparseresidency
    @test !caps(dev).raytracing
    a = MantleArray(dev, Int32, 1024)
    fill!(a, Int32(1))
    for n in (1 << 20, 1 << 24)                     # beyond the storage: a move (a reserve on a sparse device)
        resize!(a, n)
        @test Array(a)[1:1024] == fill(1, 1024)
    end
    g = Graph(dev)
    t = MantleArray(g, Int32, 1 << 24)              # a non-sparse arena part
    dispatch!(g, dev_fill!, (t, Int32(2)), 1 << 24)
    dispatch!(g, dev_add!, (a, t, Int32(1)), launchrange(a))
    run!(g)
    @test all(==(3), Array(a))
    @test TLAS(dev) isa TLAS                        # the software path
end

# api.md 4.1 (caps: whether the device drives a display), plan, "Submission
# split: estimated": the same estimated work with caps.displayattached true and false
# (FaultDevice): the display device gets the smaller budget, so more segments;
# the results are the same.
@case "DEV-11" 1 begin
    c = caps(testdevice())
    headless, shown = FaultDevice(testdevice()), FaultDevice(testdevice())
    inject!(headless, :caps, Answer(1, withcaps(c; displayattached = false)))
    inject!(shown, :caps, Answer(1, withcaps(c; displayattached = true)))
    gh, outh, k = costlychain(headless, 0.3)        # about 0.3 s of estimated work
    gs, outs, _ = costlychain(shown, 0.3)
    run!(gh); run!(gs)
    @test segmentcount(gs) > segmentcount(gh)
    @test all(==(Float32(k)), Array(outh))
    @test all(==(Float32(k)), Array(outs))
end

# A software Vulkan device remains explicitly usable on CPU-only hosts; the
# hardware-only default must give an actionable failure, never choose it silently.
@case "DEV-12" 8 :lavapipe :nodefaultgpu begin
    @test !isempty(devices())
    @test_throws Mantle.NoDefaultGPU Device()
    a = MantleArray(testdevice(), Int32[4])
    @test Array(a) == Int32[4]
    free!(a)
end
