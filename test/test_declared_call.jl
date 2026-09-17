# A library call, declared: `mul!` as a pass member.
#
# The question this answers is the one a graph runtime cannot avoid: a `mul!`
# hides an entire GEMM behind rocBLAS or behind this backend's own
# cooperative-matrix kernel, and Mantle still has to place the operands, order
# the call against everything that touches those bytes, and not serialise two
# calls that share a read-only weight. Doing that per library — a verb for
# `mul!`, another for `fft!`, another for a `cudnn` convolution — is the
# boilerplate the three-argument `dispatch!` exists to avoid: the caller names
# the function and says which argument is written, and `mul!` on a device array
# already routes to the right implementation in every ecosystem.
#
# So what is under test is NOT the GEMM. It is that the call runs at the right
# point in the order, with the arguments the placer chose, and that a dispatch
# reading the product sees the product.
#
# Portable, and the two backends answer differently — `runscalls(dev)` is the
# question. ROCm says yes because its recording is a stream capture: the rocBLAS
# kernels the call submits land in the capture, so the graph replays them without
# rocBLAS being involved again. The Vulkan backend says no, and that is not a
# caveat to work around: `run!` there submits a command buffer it built and a
# Lava plan has no walk path, so a host call would run once at record time,
# submit outside the buffer, and be missing from every replay. Refused at
# compile, naming `coopmat_gemm!` as what to declare instead.

using Test, Mantle
using KernelAbstractions
using Mantle: recordsplans
using LinearAlgebra: mul!

const M = Mantle
const BE = Main.MANTLE_TEST_BACKEND

# The consumer, so the ordering is exercised rather than asserted: if the call
# has not happened when this runs, the trace is of whatever the bytes held.
KernelAbstractions.@kernel function tracesum!(out, @Const(c), n::Int)
    i = @index(Global, Linear)
    if i == 1
        # `t`, and not the obvious `acc`: guard 0.7b asks whether a shared test
        # names a binding only one backend defines, by token, and `acc` is one
        # of them. Over-approximate on purpose, and cheaper to rename than to
        # weaken.
        t = zero(eltype(c))
        for k in 1:n
            t += c[k, k]
        end
        out[1] = t
    end
end

@testset "a declared call — $(nameof(typeof(BE)))" begin
    dev = M.Device(BE)
    n = 8
    ah = Float32[i + j for i in 1:n, j in 1:n]
    bh = Float32[i - 2j for i in 1:n, j in 1:n]
    want = ah * bh

    A = M.Buffer(dev, ah)
    B = M.Buffer(dev, bh)
    C = M.Buffer(dev, zeros(Float32, n, n))
    tr = M.Buffer(dev, zeros(Float32, 1))

    # `mul!` writes its first argument and reads the rest, declared once in
    # `graph/access.jl` because that is true of rocBLAS, of a cooperative-matrix
    # kernel and of Metal Performance Shaders alike. `Write`/`Read` wrappers at
    # this call site are `use` again: one fact, restated per
    # call site, wrong at the N+1st. The five-argument form is what makes it
    # concrete — `C = A*B*alpha + C*beta` reads `C` too.
    @test M.argument_usage(mul!, (C, A, B)) === (M.WRITE, M.READ, M.READ)
    @test M.argument_usage(mul!, (C, A, B, 1f0, 0f0))[1] === M.Touch(true, true, false)
    @test M.argument_usage(mul!, (C, A, B, 1f0, 0f0))[4] === M.NOTOUCH

    # A call nobody declared is refused, not guessed at. Widening to read+write
    # would be safe for the barrier phase and indistinguishable from an answer,
    # so a library entry point that grew an output would keep working and
    # silently serialise everything sharing its inputs.
    undeclared = try
        M.dispatch!(M.Graph(dev), foldl, (C, A))
        nothing
    catch e
        e
    end
    @test undeclared isa M.UndeclaredCall
    msg = sprint(showerror, undeclared)
    @test occursin("argument_usage", msg)
    @test occursin("foldl", msg)
    # The suggestion is per-argument, so it can be pasted rather than counted.
    @test occursin("2 arguments", msg)

    g = M.Graph(dev)
    # No ndrange: `mul!` decides its own launch, so there is nothing for Mantle
    # to divide.
    M.dispatch!(g, mul!, (C, A, B); name = "gemm")
    # A dispatch reading what the call wrote. `Barriers` derives the wait from
    # what each side touches; nothing here orders it by hand.
    M.dispatch!(g, tracesum!, (tr, C, n), 1; name = "trace")

    if !M.runscalls(dev)
        # The refusal is the behaviour under test here, and it has to happen at
        # COMPILE: `Plan` runs `Pipelines`, which is where a dispatch is turned
        # into something runnable and therefore the first moment the answer is
        # knowable. Failing later — at `record!`, or worse at `run!` — would
        # mean a graph that declares a call looks fine until it is submitted.
        err = try
            M.Plan(g)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("declared as a call", err.msg)
        @test occursin("mul!", err.msg)
        # And it names the alternative rather than only the problem.
        @test occursin("coopmat_gemm!", err.msg)
        return
    end

    pl = M.Plan(g)
    M.record!(pl)
    M.run!(pl)
    M.waitidle(dev)

    @test Array(M.storage(C)) ≈ want
    # The consumer pass read what the call wrote, which is the ordering: nothing
    # in the graph says "after", only `use`.
    @test Array(M.storage(tr))[1] ≈ sum(want[k, k] for k in 1:n)

    # The recording holds the library's work, not the library: a replay submits
    # the captured kernels and rocBLAS is not called again. So a second run must
    # land on the same answer from zeroed outputs.
    fill!(M.storage(C), 0.0f0)
    fill!(M.storage(tr), 0.0f0)
    M.run!(pl)
    M.waitidle(dev)
    @test Array(M.storage(C)) ≈ want
    @test Array(M.storage(tr))[1] ≈ sum(want[k, k] for k in 1:n)
    # …and on a backend that records, the second run really was a replay. The
    # host takes calls because it WALKS, which is the other way to say yes to
    # `runscalls`, so `recordsplans` is the question here and not the same one.
    recordsplans(dev) && @test pl.recording !== nothing

    M.free!(pl)
end
