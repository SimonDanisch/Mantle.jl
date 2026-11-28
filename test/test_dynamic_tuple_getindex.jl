# `ml[l][T]` with a run-time `l`: indexing a tuple of device arrays, the shape of
# BiotSavartBCs' FMM `interaction!` kernel (`ml` holds one array per multigrid
# level). Julia spills the tuple to the stack and steps it in bytes,
# `(l - 1) * sizeof(element)`, so the compiler has to map that offset back onto
# the tuple's type: a tuple element is `[1 x {ptr, [1 x i64]}]`, an array that
# wraps the device array's struct. Load, store and load + store are each
# checked, with `(l, T)` read from device memory as the FMM kernel does.
using Test, Mantle, KernelAbstractions, Random
include(joinpath(@__DIR__, "testbackend.jl"))
const KA = KernelAbstractions

@kernel cpu=false unsafe_indices=true function dyn_read!(out, @Const(idx), ml)
    i = @index(Global, Linear)
    (l, T) = idx[i]
    @inbounds out[i] = ml[l][T]
end

@kernel cpu=false unsafe_indices=true function dyn_store!(ml, @Const(idx), @Const(vals))
    i = @index(Global, Linear)
    (l, T) = idx[i]
    @inbounds ml[l][T] = vals[i]
end

@kernel cpu=false unsafe_indices=true function dyn_readwrite!(ml, @Const(idx))
    i = @index(Global, Linear)
    (l, T) = idx[i]
    @inbounds ml[l][T] = ml[l][T] + 1f0
end

# Without `@inbounds`: the checked tuple getindex, with its bounds branch.
@kernel cpu=false unsafe_indices=true function dyn_read_checked!(out, @Const(idx), ml)
    i = @index(Global, Linear)
    (l, T) = idx[i]
    out[i] = ml[l][T]
end

# Seven levels of different lengths, like the FMM tuple, and 64 distinct
# `(l, T)` targets: a repeated target would race the non-atomic store and
# read + write.
function dtg_setup(backend; sizes = (8, 16, 32, 4, 12, 24, 40), n = 64)
    host = [Float32.(10l .+ (1:sizes[l])) for l in eachindex(sizes)]
    ml = ntuple(l -> KA.allocate(backend, Float32, sizes[l]), length(sizes))
    foreach(l -> copyto!(ml[l], host[l]), eachindex(sizes))
    pairs = [(l, T) for l in eachindex(sizes) for T in 1:sizes[l]]
    idx_h = pairs[randperm(MersenneTwister(7), length(pairs))[1:n]]
    idx = KA.allocate(backend, Tuple{Int64,Int64}, n)
    copyto!(idx, idx_h)
    return (; host, ml, idx_h, idx, n)
end

@testset "dynamic tuple getindex ml[l][T]" begin
    backend = TESTBACKEND
    @testset "read, $(nameof(k))" for k in (dyn_read!, dyn_read_checked!)
        s = dtg_setup(backend)
        out = KA.allocate(backend, Float32, s.n); fill!(out, -1f0)
        k(backend, 64)(out, s.idx, s.ml; ndrange = s.n)
        KA.synchronize(backend)
        @test Array(out) == [s.host[l][T] for (l, T) in s.idx_h]
    end

    @testset "store" begin
        s = dtg_setup(backend)
        vals_h = Float32.(1000 .+ (1:s.n))
        vals = KA.allocate(backend, Float32, s.n); copyto!(vals, vals_h)
        dyn_store!(backend, 64)(s.ml, s.idx, vals; ndrange = s.n)
        KA.synchronize(backend)
        want = copy.(s.host)
        for (i, (l, T)) in enumerate(s.idx_h); want[l][T] = vals_h[i]; end
        @test [Array(a) for a in s.ml] == want
    end

    @testset "read + write through the same ml[l]" begin
        s = dtg_setup(backend)
        dyn_readwrite!(backend, 64)(s.ml, s.idx; ndrange = s.n)
        KA.synchronize(backend)
        want = copy.(s.host)
        for (l, T) in s.idx_h; want[l][T] += 1f0; end
        @test [Array(a) for a in s.ml] == want
    end
end
