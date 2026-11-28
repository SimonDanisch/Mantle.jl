# Phase 3 — a device array's lifetime: freed once, whoever frees it, and never
# while work that reads it is still in flight.
#
# The Vulkan version asserted the buffer state machine behind this
# (`ALIVE -> DEFERRED | DEAD`, a compare-and-swap so a second free cannot move it
# back) and that the deferred destroy is core's list on the submit channel rather
# than the backend's. Both are one backend's implementation. What a user sees of
# them is asserted here: a second free is a no-op, a free right after a launch
# that reads the array does not pull its memory out from under the launch, and the
# device keeps working. The deferred-destroy ordering itself is core's, and
# `test_submit_channel.jl` pins it with no device.

using Test, Mantle, KernelAbstractions
include(joinpath(@__DIR__, "testbackend.jl"))
const KA = KernelAbstractions

@kernel function lifecycle_copy!(dst, @Const(src))
    i = @index(Global)
    @inbounds dst[i] = src[i]
end

@testset "Phase 3 — lifecycle" begin

@testset "a second free is a no-op" begin
    a = Mantle.devicearray(TESTBACKEND, Float32, 4)
    Mantle.unsafe_free!(a)
    # Idempotent: the first free decided the array's fate, and a second one —
    # an explicit call followed by the finalizer, say — must neither throw nor
    # free anything again.
    @test Mantle.unsafe_free!(a) === nothing
    GC.gc(); GC.gc()
    b = Mantle.devicearray(TESTBACKEND, fill(5f0, 4))
    @test Array(b) == fill(5f0, 4)
end

@testset "a free while a launch reads the array waits for it" begin
    n = 1 << 16
    src = Mantle.devicearray(TESTBACKEND, collect(Float32, 1:n))
    dst = KA.allocate(TESTBACKEND, Float32, n)
    lifecycle_copy!(TESTBACKEND, 256)(dst, src; ndrange = n)
    # No synchronize: the destroy is requested while the copy may be in flight,
    # and has to wait for it rather than hand the memory back now.
    Mantle.unsafe_free!(src)
    src = nothing
    GC.gc(); GC.gc()
    # Memory handed back early would be reused by these allocations and the copy
    # would read their bytes.
    junk = [Mantle.devicearray(TESTBACKEND, fill(-1f0, n)) for _ in 1:4]
    KA.synchronize(TESTBACKEND)
    @test Array(dst) == collect(Float32, 1:n)
    foreach(Mantle.unsafe_free!, junk)
end

end  # @testset
