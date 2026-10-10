# What `error()` in a kernel means, on every backend: the invocation stops where
# it threw, the device records it, and the next synchronize throws. Metal.jl
# always did this; on Vulkan a throw stopped the invocation and nothing else, so
# a failed bounds check or an `error` returned silently and the run reported
# success with that invocation's results missing.
#
# The barrier is the hard half, and why this file exists. An invocation that
# returns before a workgroup barrier the others reach leaves them waiting: the
# Vulkan spec requires every invocation to reach every control barrier, and
# lavapipe deadlocked on exactly this kernel. Lava reroutes such a path through
# the barrier (`fix_barrier_skipping_paths!`) — and the rerouted invocation then
# ran the rest of the kernel and wrote its result, so a "dead" invocation stored
# `A[64] = 64`. Now it reaches the barrier and writes nothing: `A[64]` stays 0.
#
# The compiler half — that the reroute finds a barrier behind a wrapper call, and
# that the rerouted store sits behind the flag — is Lava's own suite
# (`test_barrier_skipping_paths.jl`); this is what a user sees.

using Test, Mantle, KernelAbstractions
include(joinpath(@__DIR__, "testbackend.jl"))
const KA = KernelAbstractions

@kernel function kerr_after_barrier!(A, kill)
    i = @index(Global)
    @synchronize()
    if i == kill[1]
        error("dead")
    end
    @synchronize()
    A[i] = Int32(i)
end

@kernel function kerr_plain!(A, kill)
    i = @index(Global)
    if i == kill[1]
        error("dead")
    end
    A[i] = Int32(i)
end

"""Launch `k` over 128 invocations with invocation 64 throwing, and return what
the synchronize threw (or `nothing`) and what the kernel wrote."""
function throwat64(k)
    A = Mantle.devicearray(TESTBACKEND, zeros(Int32, 128))
    kill = Mantle.devicearray(TESTBACKEND, Int32[64])
    k(TESTBACKEND)(A, kill; ndrange = 128, workgroupsize = 128)
    err = try
        KA.synchronize(TESTBACKEND)
        nothing
    catch e
        e
    end
    return err, Array(A)
end

@testset "error() in a kernel" begin
    for (name, k) in (("no barrier", kerr_plain!), ("before a barrier", kerr_after_barrier!))
        @testset "$name" begin
            err, r = throwat64(k)
            # The next synchronize reports it…
            @test err !== nothing
            err === nothing || @test occursin("exception", lowercase(sprint(showerror, err)))
            # …the invocation that threw wrote nothing after the throw…
            @test r[64] == 0
            # …and every other one finished, with no deadlock and no corruption.
            @test all(r[i] == i for i in 1:128 if i != 64)
        end
    end

    @testset "reported once, and the device goes on" begin
        # The report is consumed: the next synchronize, with nothing thrown since,
        # is clean, and the device still computes.
        B = Mantle.devicearray(TESTBACKEND, zeros(Int32, 4))
        fill!(B, Int32(3))
        KA.synchronize(TESTBACKEND)
        @test Array(B) == fill(Int32(3), 4)
    end
end
