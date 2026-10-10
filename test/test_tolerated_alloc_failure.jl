# A tolerated allocation failure answers `nothing`, and leaves nothing behind for
# whoever calls the device next — in particular, no validation message.
#
# `trydevicearray` exists to attempt an allocation that may legitimately fail.
# On Vulkan the refusal also produces validation-layer messages
# (`VkBufferCreateInfo-size-06409`, `vkAllocateMemory-pAllocateInfo-01713`), and
# those are the attempt's to absorb: the failure was expected and handled.
#
# It absorbed them incorrectly once. The layer's callback writes into a ring and
# a drain moves entries out; the tolerated path emptied the drained list WITHOUT
# draining first, so its own messages stayed in the ring and the next check —
# belonging to entirely unrelated code — found them and threw. Observed exactly
# that way: a test asked for 40 GB on purpose, and the failure surfaced 40 lines
# later as an error during a four-element upload. The test that reported it was
# innocent, which is the part worth pinning: a misattributed validation error
# sends whoever reads the log to the wrong file.
#
# Core now refuses a request larger than `maxalloc` without asking the driver at
# all, which is where every machine here refuses this one; the drain discipline
# is still the backend's for a size the driver is asked about.
#
# The validated half runs in a process started for it (`Mantle.debugenv`): Metal
# loads its validation layer when a process starts, and a device rebuilt with
# validation inside the suite would leave it on for every file after this.

using Test, Mantle, KernelAbstractions
include(joinpath(@__DIR__, "testbackend.jl"))
include(joinpath(@__DIR__, "freshprocess.jl"))
const KA = KernelAbstractions

# Larger than one allocation may be where the device says how large that is, and
# larger than its memory where it says there is no limit: refused, not unlucky.
toolarge(dev) = Mantle.maxalloc(dev) < typemax(Int) ? Mantle.maxalloc(dev) + 1 :
                                                       2 * Mantle.capacity(dev)

@testset "a refused allocation is nothing, and the device goes on" begin
    dev = Mantle.Device(TESTBACKEND)
    @test Mantle.trydevicearray(TESTBACKEND, UInt8, toolarge(dev)) === nothing
    # One that fits is an ordinary device array.
    a = Mantle.trydevicearray(TESTBACKEND, Float32, 4)
    @test a !== nothing
    copyto!(a, Float32[1, 2, 3, 4])
    b = a .+ 10f0
    KA.synchronize(TESTBACKEND)
    @test Array(b) == Float32[11, 12, 13, 14]
end

@testset "a tolerated allocation failure leaves no validation messages" begin
    status, out = infreshprocess(raw"""
        dev = Mantle.reset_device!(Mantle.Device(BACKEND); debug = DebugConfig(validation = true))
        Mantle.validating(dev) || exit(3)
        be = Mantle.backend(dev)
        toolarge = Mantle.maxalloc(dev) < typemax(Int) ? Mantle.maxalloc(dev) + 1 :
                                                         2 * Mantle.capacity(dev)
        @testset "refused, quietly" begin
            # Start from a known-clean queue, so what is observed is what was caused.
            Mantle.validationmessages!(dev)
            @test Mantle.trydevicearray(dev, UInt8, toolarge) === nothing
            # Nothing left in the layer's queue either: this is the assertion with
            # teeth, the one that fails if the messages wait to ambush someone else.
            msgs = Mantle.validationmessages!(dev)
            foreach(println, msgs)
            @test isempty(msgs)
            # And the next unrelated operation survives: the shape of the original
            # symptom, a small upload that has nothing to do with the refusal.
            a = Mantle.devicearray(be, Float32[1, 2, 3, 4])
            b = a .+ 10f0
            KA.synchronize(be)
            @test Array(b) == Float32[11, 12, 13, 14]
            @test isempty(Mantle.validationmessages!(dev))
        end
        """; debug = DebugConfig(validation = true))
    if status == 3
        @info "no validation layer for this device here; nothing can leak and this cannot assert"
        @test_skip status == 0
    else
        reportfresh(status, out)
        @test status == 0
    end
end
