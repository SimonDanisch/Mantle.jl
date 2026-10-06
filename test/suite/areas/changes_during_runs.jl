# Changes during runs: between stages, WaitFor, inrun (catalogue-resize-rt.md,
# table J: RN). A change made while a host function runs is applied in front
# of the run's next stage, unless the plan cannot follow it mid-run (a length
# it launches over directly, a temporary sized for fewer elements, a capacity
# it sized temporaries from): then the plan's own later stages leave it for
# after the run, any other submitter waits on the plan's runend (WaitFor), and
# a submitter inside a host function throws (api.md section 7;
# recording-plan.md, "A run with host nodes is one submission per stage";
# resizing-and-raytracing.jl B.4, takepending, waitforruns; internals.jl
# section 6).
#
# Doc keys as in the catalogue: PLAN recording-plan.md, RR
# resizing-and-raytracing.jl, API api.md, INT internals.jl, DEC decisions.md.
# A run is held between its stages by a host node that blocks on a Gate
# (helpers.jl, FI-H); stage 1 writes t, the host node reads and writes it, and
# stage 2 reads it, so compile cannot move either kernel across the cut.
# Eager writes are dispatch!(dev, …) over launchrange(a): resolved when the
# call is submitted, after any wait.

using KernelAbstractions: @kernel, @index
import GeometryBasics, LinearAlgebra, ColorTypes, FixedPointNumbers

@kernel function rn_copy!(dst, src)
    i = @index(Global)
    @inbounds dst[i] = src[i]
end

@kernel function rn_fill!(a, v)
    i = @index(Global)
    @inbounds a[i] = v
end

@kernel function rn_add!(dst, a, b)
    i = @index(Global)
    @inbounds dst[i] = a[i] + b[i]
end

# Stage 1: dst[i] = src[i], and t[1] = 0 for the host node.
@kernel function rn_copyfirst!(dst, src, t)
    i = @index(Global)
    @inbounds dst[i] = src[i]
    if i == 1
        @inbounds t[1] = Int32(0)
    end
end

# Stage 2: dst[i] = src[i] + t[1].
@kernel function rn_copythen!(dst, src, t)
    i = @index(Global)
    @inbounds dst[i] = src[i] + t[1]
end

@kernel function rn_writefirst!(dst, t, v)   # stage 1: dst[i] = v, t[1] = 0
    i = @index(Global)
    @inbounds dst[i] = v
    if i == 1
        @inbounds t[1] = Int32(0)
    end
end

@kernel function rn_writethen!(dst, t, v)    # stage 2: dst[i] = v + t[1]
    i = @index(Global)
    @inbounds dst[i] = v + t[1]
end

@kernel function rn_indices!(idx, t)         # stage 1: idx[i] = i - 1, t[1] = 0
    i = @index(Global)
    @inbounds idx[i] = UInt32(i - 1)
    if i == 1
        @inbounds t[1] = Int32(0)
    end
end

@kernel function rn_rescale!(m, t)           # m[i] *= t[1] + 1 (t[1] = 0: unchanged), a write of m after reading t
    i = @index(Global)
    @inbounds m[i] = m[i] * (Float32(t[1]) + 1f0)
end

# A point per vertex. The shader convention (vertex_index(), a NamedTuple with
# `position`, the fragment taking the draw's arguments) is the one of
# test/test_headless_rebindable_draw.jl, a guess for the new API (api.md G1).
# Both read t, so the draw comes after the host node that writes it.
rn_pointvertex(t) = (position = GeometryBasics.Vec4f(0f0, 0f0, 0f0, 1f0 + Float32(t[1])),)
rn_pointfragment(inputs, t) = GeometryBasics.Vec4f(1f0, 1f0, 1f0, 1f0 + Float32(t[1]))
rn_points() = Mantle.Rasterizer(; vertex = Mantle.VertexShader(rn_pointvertex),
                                  fragment = Mantle.FragmentShader(rn_pointfragment),
                                  topology = Mantle.PointList(), blend = Mantle.Opaque(),
                                  cull = Mantle.NoCull(), depth = Mantle.DepthOff())
rn_target(g) = Image(g, ColorTypes.BGRA{FixedPointNumbers.N0f8}, 16, 16)

rn_triangle() = [GeometryBasics.Point3f(0, 0, 0), GeometryBasics.Point3f(1, 0, 0), GeometryBasics.Point3f(0, 1, 0)]
rn_identity() = GeometryBasics.Mat4f(LinearAlgebra.I)

"""
A graph whose run stops between its stages until `letgo!(gate)`: stage 1
copies src into dst1, stage 2 src into dst2, both over launchrange(src).
"""
function rn_gated(dev, gate, src, dst1, dst2)
    g = Graph(dev)
    t = MantleArray(g, Int32, 1)
    dispatch!(g, rn_copyfirst!, (dst1, src, t), launchrange(src))
    dispatch!(g, blockinghost(gate), (t,); writes = (t,))
    dispatch!(g, rn_copythen!, (dst2, src, t), launchrange(src))
    return g
end

"""One whole run of a gated graph (the gate opened before, its entry taken after)."""
rn_through!(g, gate) = (letgo!(gate); run!(g); waitentered(gate))

"""A run of a gated graph on another task, returned once it is between its stages."""
rn_start(g, gate) = (task = Threads.@spawn(run!(g)); waitentered(gate); task)

"""Opens the gate and waits for the run to end."""
rn_finish(gate, task) = (letgo!(gate); withtimeout(() -> wait(task), 60))

"""An eager write of v into every element of a, over its length at submission."""
rn_eagerfill!(dev, a, v) = dispatch!(dev, rn_fill!, (a, Int32(v)), launchrange(a))

"""rn_fill!'s kernel compiled, so an eager fill never waits for a compile."""
rn_warm(dev) = (rn_eagerfill!(dev, MantleArray(dev, Int32, 4), 0); Mantle.waitidle(dev))

"""In g's host function: a resize g cannot follow mid-run, then an eager call that would apply it."""
function rn_fillinside(dev, a, n, waited)
    resize!(a, 2n)
    waited[] = @elapsed @test_throws ArgumentError rn_eagerfill!(dev, a, 1)
    return nothing
end

"""In a host function: a host read of an array whose change another run holds."""
rn_readinside(a) = (@test_throws ArgumentError Array(a); nothing)

"""In g's host function: a resize held for g, then a task that reads a, waited for at most 5 s."""
function rn_spawnread(a, n, reader, finished)
    resize!(a, 2n)
    reader[] = Threads.@spawn Array(a)
    finished[] = timedwait(() -> istaskdone(reader[]), 5) === :ok
    return nothing
end

"""Blocks on the gate the first time, then throws; later calls pass through."""
function rn_failonce(gate, fail)
    fail[] || return nothing
    fail[] = false
    put!(gate.entered, nothing); take!(gate.release)
    error("host function failed")
end

"""A host function: dst = src + 1."""
rn_hostaddone(src, dst) = (dst .= src .+ Int32(1); nothing)

"""x += 1 + w through a host node: stage 1 x + w into t, the host function adds one, stage 2 back into x."""
function rn_hostbump(dev, x, w, n)
    g = Graph(dev)
    t, u = MantleArray(g, Int32, n), MantleArray(g, Int32, n)
    dispatch!(g, rn_add!, (t, x, w), n)
    dispatch!(g, HostCall(rn_hostaddone), (t, u); writes = (u,))
    dispatch!(g, rn_copy!, (x, u), n)
    return g
end

# ── J. Changes during runs: between stages, WaitFor, inrun ──

# INT:1321-1324, PLAN:660-664: stage 2 launches indirectly over resizable a; between the stages a[1:4] = x: stage 1 saw the old data, stage 2's prefix applies x.
@case "RN-01" 4 begin   # pending decision 7: one state per stage
    dev = testdevice(); n = 64
    a = MantleArray(dev, Int32, 16); resize!(a, n); copyto!(a, Int32.(1:n))   # resizable: launches over it are indirect
    out1, out2 = MantleArray(dev, Int32, n), MantleArray(dev, Int32, n)
    gate = Gate()
    g = rn_gated(dev, gate, a, out1, out2)
    rn_through!(g, gate)
    running = rn_start(g, gate)
    a[1:4] = Int32[-1, -2, -3, -4]
    rn_finish(gate, running)
    @test Array(out1) == 1:n
    @test Array(out2) == [-1, -2, -3, -4, 5:n...]
end

# RR:343-347: stage 2 launches directly over fixed a; between the stages resize!(a, 2n): stage 2 uses n and the old storage, the Resize is still pending after the run, and the next run recompiles once and covers 2n.
@case "RN-02" 4 begin
    dev = testdevice(); n = 64
    a = MantleArray(dev, Int32.(1:n))
    out1, out2 = MantleArray(dev, fill(Int32(-1), 2n)), MantleArray(dev, fill(Int32(-1), 2n))
    gate = Gate()
    g = rn_gated(dev, gate, a, out1, out2)
    rn_through!(g, gate)
    running = rn_start(g, gate)
    resize!(a, 2n)
    rn_finish(gate, running)
    @test Array(out2) == [1:n; fill(-1, n)]
    @test only(Mantle.pendingof(a)) isa Mantle.Resize
    @test length(a) == 2n
    a[n+1:2n] = Int32.(n+1:2n)
    _, d = counted(() -> rn_through!(g, gate))
    @test d.plancompiles == 1
    @test Array(out2) == 1:2n
end

# RR:345-347: as RN-02, with a[3:4] = z pended before the resize and a[1:2] = y after it: stage 2 sees z, not y; the next run sees y.
@case "RN-03" 4 begin   # pending decision 7: one state per stage
    dev = testdevice(); n = 64
    a = MantleArray(dev, Int32.(1:n))
    out1, out2 = MantleArray(dev, fill(Int32(-1), 2n)), MantleArray(dev, fill(Int32(-1), 2n))
    gate = Gate()
    g = rn_gated(dev, gate, a, out1, out2)
    rn_through!(g, gate)
    running = rn_start(g, gate)
    a[3:4] = Int32[-3, -4]; resize!(a, 2n); a[1:2] = Int32[-1, -2]
    rn_finish(gate, running)
    @test Array(out2)[1:4] == [1, 2, -3, -4]
    a[n+1:2n] = Int32.(n+1:2n)
    rn_through!(g, gate)
    @test Array(out2)[1:4] == [-1, -2, -3, -4]
end

# RR:347-349, API:487: as RN-02, plus thread C's eager fill of a: C waits on g's runend until g submits its last stage, then the Resize and the fill are applied; the fill covers 2n.
@case "RN-04" 4 begin
    dev = testdevice(); n = 64
    a = MantleArray(dev, Int32.(1:n))
    out1, out2 = MantleArray(dev, fill(Int32(-1), 2n)), MantleArray(dev, fill(Int32(-1), 2n))
    gate = Gate()
    g = rn_gated(dev, gate, a, out1, out2)
    rn_warm(dev); rn_through!(g, gate)
    running = rn_start(g, gate)
    resize!(a, 2n)
    filler = Threads.@spawn rn_eagerfill!(dev, a, 1)
    sleep(0.3)
    @test !istaskdone(filler)
    rn_finish(gate, running)
    withtimeout(() -> wait(filler), 30)
    @test Array(out2) == [1:n; fill(-1, n)]
    @test Array(a) == fill(Int32(1), 2n)
end

# RR:349-350,449-451: as RN-04, but the eager call comes from g's own host function (after its resize): ArgumentError at once, no wait; the run completes on n elements; the Resize stays pending.
@case "RN-05" 4 begin
    dev = testdevice(); n = 64
    a = MantleArray(dev, Int32.(1:n))
    out1, out2 = MantleArray(dev, Int32, n), MantleArray(dev, Int32, n)
    waited = Ref(Inf)
    g = Graph(dev); t = MantleArray(g, Int32, 1)
    dispatch!(g, rn_copyfirst!, (out1, a, t), launchrange(a))
    dispatch!(g, HostCall(_ -> rn_fillinside(dev, a, n, waited)), (t,); writes = (t,))
    dispatch!(g, rn_copythen!, (out2, a, t), launchrange(a))
    rn_warm(dev)
    run!(g)
    @test waited[] < 0.1
    @test Array(out2) == 1:n
    @test only(Mantle.pendingof(a)) isa Mantle.Resize
end

# RR:442-451: g2's host function calls Array(a) while a has a change held for g (g between its stages): ArgumentError.
@case "RN-06" 4 begin
    dev = testdevice(); n = 64
    a = MantleArray(dev, Int32.(1:n))
    out1, out2, z = MantleArray(dev, Int32, n), MantleArray(dev, Int32, n), MantleArray(dev, Int32[0])
    gate = Gate()
    g = rn_gated(dev, gate, a, out1, out2)
    g2 = Graph(dev)                                   # no transient: no arena part shared with g
    dispatch!(g2, rn_fill!, (z, Int32(1)), 1)
    dispatch!(g2, HostCall(() -> rn_readinside(a)), (); writes = ())
    rn_through!(g, gate); run!(g2)
    running = rn_start(g, gate)
    resize!(a, 2n)                                    # held for g
    run!(g2)
    rn_finish(gate, running)
end

# INT:996-998,1151-1152, RR:450,460, API:487: thread C runs plain g2 (no host node), which needs a's Resize that g holds: C waits for g's last stage, then runs without a throw (ambiguity A2, pinned to api.md:487).
@case "RN-07" 4 begin
    dev = testdevice(); n = 64
    a = MantleArray(dev, Int32.(1:n))
    out1, out2, out = MantleArray(dev, Int32, n), MantleArray(dev, Int32, n), MantleArray(dev, Int32, 2n)
    gate = Gate()
    g = rn_gated(dev, gate, a, out1, out2)
    g2 = Graph(dev); dispatch!(g2, rn_copy!, (out, a), launchrange(a))   # no transient
    rn_through!(g, gate); run!(g2); Mantle.waitidle(dev)
    running = rn_start(g, gate)
    resize!(a, 2n); a[n+1:2n] = Int32.(n+1:2n)
    other = Threads.@spawn run!(g2)
    sleep(0.3)
    @test !istaskdone(other)
    rn_finish(gate, running)
    withtimeout(() -> wait(other), 30)
    @test Array(out) == 1:2n
end

# RR:444-446: g's host function resizes a, starts a task that calls Array(a) and waits for it: an undetected deadlock (the task waits for g's runend), bounded here by a 5 s wait in the host function.
@case "RN-08" 4 begin
    dev = testdevice(); n = 64
    a = MantleArray(dev, Int32.(1:n))
    out1, out2 = MantleArray(dev, Int32, n), MantleArray(dev, Int32, n)
    reader, finished = Ref{Task}(), Ref(true)
    g = Graph(dev); t = MantleArray(g, Int32, 1)
    dispatch!(g, rn_copyfirst!, (out1, a, t), launchrange(a))
    dispatch!(g, HostCall(_ -> rn_spawnread(a, n, reader, finished)), (t,); writes = (t,))
    dispatch!(g, rn_copythen!, (out2, a, t), launchrange(a))
    run!(g)
    @test_broken finished[]
    @test length(withtimeout(() -> fetch(reader[]), 30)) == 2n
end

# INT:1132-1134: a one-stage plan in flight (500 ms); thread C makes 1 000 resizes of a it launches over directly, each with an eager write: C never waits; the next run recompiles and covers the last length.
@case "RN-09" 4 begin
    dev = testdevice(); n = 64
    a, out, busy = MantleArray(dev, Int32.(1:n)), MantleArray(dev, Int32, n + 1000), MantleArray(dev, Int32[0])
    g = Graph(dev)
    dispatch!(g, rn_copy!, (out, a), launchrange(a))
    slowwrite!(g, busy, 1; ms = 500)
    rn_warm(dev); run!(g); Mantle.waitidle(dev)
    run!(g)                                           # one stage, in flight
    took, d = counted() do
        withtimeout(30) do
            @elapsed for k in 1:1000
                resize!(a, n + k); rn_eagerfill!(dev, a, k)
            end
        end
    end
    @test d.hostwaits == 0
    @test took < 0.25
    run!(g)
    @test Array(out) == fill(Int32(1000), n + 1000)
end

# INT:1132-1133: g's run waits in holdparts! (h's run holds the shared arena part between its stages); a resize stales g and an eager write applies it: the eager call does not wait; g then sees its stale flag and recompiles.
@case "RN-10" 4 begin
    dev = testdevice(); n = 64
    a, out = MantleArray(dev, Int32.(1:n)), MantleArray(dev, fill(Int32(-1), 2n))
    gate = Gate()
    h = Graph(dev); th = MantleArray(h, Int32, 1)
    dispatch!(h, rn_fill!, (th, Int32(0)), 1)
    dispatch!(h, blockinghost(gate), (th,); writes = (th,))
    g = Graph(dev); tg = MantleArray(g, Int32, 2n)    # a transient: the arena part h holds
    dispatch!(g, rn_copy!, (tg, a), launchrange(a))   # direct over fixed a
    dispatch!(g, rn_copy!, (out, tg), launchrange(a))
    rn_warm(dev); rn_through!(h, gate); run!(g); Mantle.waitidle(dev)
    hrun = rn_start(h, gate)
    Mantle.resetdiagnostics!(dev)
    grun = Threads.@spawn run!(g)
    sleep(0.3)
    resize!(a, 2n)
    @test (@elapsed rn_eagerfill!(dev, a, 5)) < 0.1
    @test !istaskdone(grun)
    rn_finish(gate, hrun)
    withtimeout(() -> wait(grun), 30)
    @test Mantle.diagnostics(dev).plancompiles == 1
    @test Array(out) == fill(Int32(5), 2n)
end

# INT:1157-1158: thread C waits (WaitFor) for g's runend; g's host function throws: endbetween! wakes C, C applies the Resize and its fill; g's next run recompiles and is exact.
@case "RN-11" 4 begin
    dev = testdevice(); n = 64
    a = MantleArray(dev, Int32.(1:n))
    out1, out2 = MantleArray(dev, fill(Int32(-1), 2n)), MantleArray(dev, fill(Int32(-1), 2n))
    gate, fail = Gate(), Ref(false)
    g = Graph(dev); t = MantleArray(g, Int32, 1)
    dispatch!(g, rn_copyfirst!, (out1, a, t), launchrange(a))
    dispatch!(g, HostCall(_ -> rn_failonce(gate, fail)), (t,); writes = (t,))
    dispatch!(g, rn_copythen!, (out2, a, t), launchrange(a))
    rn_warm(dev); run!(g); Mantle.waitidle(dev)
    fail[] = true
    failing = Threads.@spawn run!(g)
    waitentered(gate)
    resize!(a, 2n)
    filler = Threads.@spawn rn_eagerfill!(dev, a, 3)
    sleep(0.3)
    @test !istaskdone(filler)
    letgo!(gate)
    @test_throws TaskFailedException withtimeout(() -> wait(failing), 30)
    withtimeout(() -> wait(filler), 30)
    @test Array(a) == fill(Int32(3), 2n)
    _, d = counted(() -> run!(g))
    @test d.plancompiles == 1
    @test Array(out2) == fill(Int32(3), 2n)
end

# RR:340-342, PLAN:666-670: stage 2 draws with idx bound by handle; between the stages resize!(idx) past its storage: stage 2's prefix moves idx (keeping what stage 1 wrote) and re-records the parts that recorded its handle; no recompile.
@case "RN-12" 6 begin
    dev = testdevice(); n = 64
    idx = MantleArray(dev, UInt32, n)
    gate = Gate()
    g = Graph(dev); t = MantleArray(g, Int32, 1)
    dispatch!(g, rn_indices!, (idx, t), n)
    dispatch!(g, blockinghost(gate), (t,); writes = (t,))
    render!(g, rn_target(g) => Clear((0f0, 0f0, 0f0, 1f0))) do p
        draw!(p, rn_points(), (t,), n; indices = idx)
    end
    rn_through!(g, gate); Mantle.waitidle(dev)
    Mantle.resetdiagnostics!(dev)
    running = rn_start(g, gate)
    @test (@elapsed resize!(idx, 4n)) < 0.01
    rn_finish(gate, running)
    d = Mantle.diagnostics(dev)
    @test d.plancompiles == 0
    @test d.records >= 1
    @test Array(idx)[1:n] == UInt32.(0:n-1)
end

# RR:346, API:487: stage 2 copies into a temporary sized at compile for 2n (SizedBy, n = length(a)); between the stages resize!(a, 3n): held for g, another submitter (an eager fill) waits for g's last stage, the next run recompiles.
@case "RN-13" 4 begin
    dev = testdevice(); n = 64
    a = MantleArray(dev, Int32, 2n); resize!(a, n); copyto!(a, Int32.(1:n))   # resizable, n elements
    out1, out2 = MantleArray(dev, fill(Int32(-1), 3n)), MantleArray(dev, fill(Int32(-1), 3n))
    gate = Gate()
    g = Graph(dev); t = MantleArray(g, Int32, 1)
    tmp = MantleArray(g, Int32, Mantle.SizedBy(identity, a))     # api.md 2.2 (internal): 2 * length(a) at compile
    dispatch!(g, rn_copyfirst!, (out1, a, t), launchrange(a))
    dispatch!(g, blockinghost(gate), (t,); writes = (t,))
    dispatch!(g, rn_copythen!, (tmp, a, t), launchrange(a))
    dispatch!(g, rn_copy!, (out2, tmp), launchrange(a))
    rn_warm(dev); rn_through!(g, gate)
    running = rn_start(g, gate)
    resize!(a, 3n); a[n+1:3n] = Int32.(n+1:3n)
    filler = Threads.@spawn rn_eagerfill!(dev, a, 2)
    sleep(0.3)
    @test !istaskdone(filler)
    rn_finish(gate, running)
    withtimeout(() -> wait(filler), 30)
    @test Array(out2) == [1:n; fill(-1, 2n)]
    _, d = counted(() -> rn_through!(g, gate))
    @test d.plancompiles == 1
    @test Array(out2) == fill(Int32(2), 3n)
end

# API:487, RR:438: stage 2 builds a software TLAS (temporaries sized from its capacity, 64); between the stages a push! past the capacity: held for g, another graph refitting the TLAS waits, g's next run recompiles.
@case "RN-14" 5 begin
    dev = testdevice()
    tl = TLAS(dev; hardware = false)
    blas = BLAS(dev, MantleArray(dev, rn_triangle()), MantleArray(dev, [GeometryBasics.TriangleFace{UInt32}(1, 2, 3)]))
    transforms = MantleArray(dev, fill(rn_identity(), 64))
    push!(tl, blas, transforms)                       # 64 instances: the starting capacity
    gate = Gate()
    g = Graph(dev); t = MantleArray(g, Int32, 1)
    dispatch!(g, rn_fill!, (t, Int32(0)), 1)
    dispatch!(g, blockinghost(gate), (t,); writes = (t,))
    dispatch!(g, rn_rescale!, (transforms, t), 64)    # reads t, writes a member: stage 2
    build!(g, tl)                                     # after that write: stage 2
    g2 = Graph(dev); refit!(g2, tl)
    rn_through!(g, gate); run!(g2); Mantle.waitidle(dev)
    running = rn_start(g, gate)
    push!(tl, blas, rn_identity())                    # 65 instances: the capacity doubles
    other = Threads.@spawn run!(g2)
    sleep(0.3)
    @test !istaskdone(other)
    rn_finish(gate, running)
    withtimeout(() -> wait(other), 30)
    _, d = counted(() -> rn_through!(g, gate))
    @test d.plancompiles == 1
end

# RR:437-441,714-721: stage 2 draws with indices through array GPURef r (its target bound by handle); between the stages r[] = b: stage 2's parts are re-recorded with b; no wait, no recompile.
@case "RN-15" 6 begin
    dev = testdevice(); n = 64
    idx, other = MantleArray(dev, UInt32.(0:n-1)), MantleArray(dev, UInt32.(n-1:-1:0))
    r = GPURef(dev, idx)
    gate = Gate()
    g = Graph(dev); t = MantleArray(g, Int32, 1)
    dispatch!(g, rn_fill!, (t, Int32(0)), 1)
    dispatch!(g, blockinghost(gate), (t,); writes = (t,))
    render!(g, rn_target(g) => Clear((0f0, 0f0, 0f0, 1f0))) do p
        draw!(p, rn_points(), (t,), n; indices = r)
    end
    rn_through!(g, gate); Mantle.waitidle(dev)
    Mantle.resetdiagnostics!(dev)
    running = rn_start(g, gate)
    @test (@elapsed (r[] = other)) < 0.01
    rn_finish(gate, running)
    d = Mantle.diagnostics(dev)
    @test d.plancompiles == 0
    @test d.records >= 1
    @test r[] === other
end

# DEC:225-243, PLAN:594-601: plan Q reads a and is between its stages; plan P (thread C) writes a: P waits for Q's last run on the GPU, not on the host; Q's stage 2 then finds a in its inbox and sees P's write.
@case "RN-17" 3 begin   # pending decision 7: one state per stage
    dev = testdevice()
    a, out1, out2 = MantleArray(dev, Int32[1, 1]), MantleArray(dev, Int32, 2), MantleArray(dev, Int32, 2)
    gate = Gate()
    q = rn_gated(dev, gate, a, out1, out2)
    p = Graph(dev); dispatch!(p, rn_fill!, (a, Int32(7)), 1)   # no transient: no arena part shared with q
    rn_through!(q, gate); run!(p); Mantle.waitidle(dev)
    rn_eagerfill!(dev, a, 1)
    running = rn_start(q, gate)
    _, d = counted(() -> withtimeout(() -> fetch(Threads.@spawn run!(p)), 10))
    @test d.hostwaits == 0
    rn_finish(gate, running)
    @test Array(out1) == [1, 1]
    @test Array(out2) == [7, 1]
end

# API:452 vs INT:1321-1324: g writes through array GPURef r in both stages; between the stages r[] = b: stage 1 wrote a, stage 2 writes b (ambiguity A8).
@case "RN-18" 6 begin   # pending decision 7: one state per stage
    dev = testdevice(); n = 64
    a, b = MantleArray(dev, zeros(Int32, n)), MantleArray(dev, zeros(Int32, n))
    r = GPURef(dev, a)
    gate = Gate()
    g = Graph(dev); t = MantleArray(g, Int32, 1)
    dispatch!(g, rn_writefirst!, (r, t, Int32(1)), n)
    dispatch!(g, blockinghost(gate), (t,); writes = (t,))
    dispatch!(g, rn_writethen!, (r, t, Int32(2)), n)
    running = rn_start(g, gate)
    r[] = b
    rn_finish(gate, running)
    @test Array(a) == fill(Int32(1), n)
    @test Array(b) == fill(Int32(2), n)
end

# API:421-433: four tasks: two run graphs with host nodes that share w and an arena part, one stores zeros into w, one makes eager calls reading x and w; 10^4 rounds each, watchdog: no deadlock, exact.
@case "RN-19" 4 :long begin
    dev = testdevice(); n = 1024; rounds = 10^4
    w, x, y, v = (MantleArray(dev, zeros(Int32, n)) for _ in 1:4)
    g1, g2 = rn_hostbump(dev, x, w, n), rn_hostbump(dev, y, w, n)
    tasks = [Threads.@spawn(foreach(_ -> run!(g1), 1:rounds)),
             Threads.@spawn(foreach(_ -> run!(g2), 1:rounds)),
             Threads.@spawn(foreach(_ -> (w[1:n] = zeros(Int32, n)), 1:rounds)),
             Threads.@spawn(foreach(_ -> dispatch!(dev, rn_add!, (v, x, w), n), 1:rounds))]
    withtimeout(() -> foreach(wait, tasks), 3600)
    @test Array(x) == fill(Int32(rounds), n)
    @test Array(y) == fill(Int32(rounds), n)
    @test Array(w) == zeros(Int32, n)
end

# RR:449-455: thread C waits on g's runend; thread D pends a store and a resize on a and runs an unrelated graph: D never blocks; afterwards C's fill covers the last length.
@case "RN-20" 4 begin
    dev = testdevice(); n = 64
    a = MantleArray(dev, Int32.(1:n))
    out1, out2 = MantleArray(dev, Int32, n), MantleArray(dev, Int32, n)
    hsrc, hout = MantleArray(dev, Int32.(1:4)), MantleArray(dev, Int32, 4)
    gate = Gate()
    g = rn_gated(dev, gate, a, out1, out2)
    h = Graph(dev); dispatch!(h, rn_copy!, (hout, hsrc), 4)   # names nothing of a; no transient
    rn_warm(dev); rn_through!(g, gate); run!(h); Mantle.waitidle(dev)
    running = rn_start(g, gate)
    resize!(a, 2n)
    waiter = Threads.@spawn rn_eagerfill!(dev, a, 1)
    sleep(0.3)
    took = withtimeout(10) do
        @elapsed (a[1:2] = Int32[5, 6]; resize!(a, 3n); run!(h); Array(hout))
    end
    @test took < 1
    @test !istaskdone(waiter)
    rn_finish(gate, running)
    withtimeout(() -> wait(waiter), 30)
    @test Array(a) == fill(Int32(1), 3n)
    @test Array(hout) == 1:4
end
