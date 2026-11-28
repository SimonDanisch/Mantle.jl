"""
The shutdown hook retires EVERY device, not just the default one.

Julia runs `atexit` hooks before its final finalizer sweep, and Mantle uses that:
each backend's hook retires the devices the process built, so an array finalizer
running afterwards skips the driver instead of calling into one that has already
been torn down (`retire!`).

The Vulkan hook retired ONE device, the bound one. A second device — built to
compare against, or a lavapipe reference — stayed live, and on 2026-08-30 the
suite passed every test and then died on the way out with exit 139:

    pthread_mutex_lock                      libc
    …                                       libvulkan_lvp.so     <- lavapipe
    vkGetSemaphoreCounterValue
    query_timeline                          runtime/command.jl:944
    vk_free!                                runtime/memory.jl:742
    unsafe_free!(::LavaArray)               array/lavaarray.jl:251
    run_finalizer / ijl_atexit_hook

after `Test Summary` had printed — the worst shape a crash can take, because
every summary said the suite passed and the process still failed.

**What this asserts, and what it deliberately does not.** It asserts the
invariant: after the hook has run, every device the process built reports
`retired`, which is precisely what the finalizers consult. It does not try to
reproduce the segfault by allocating on two devices and exiting: whether that
crashes depends on the order of the finalizers in one sweep, and the small case
gets the safe order, so it passes with the bug in place.

The check is itself an `atexit` hook, registered before Mantle loads so that it
runs after Mantle's (hooks run last-registered first), and it decides the exit
code. Hence a process of its own. The second device is a new device on the same
hardware, found by name: what matters is two devices, not two drivers.
"""

using Test, Mantle, KernelAbstractions
include(joinpath(@__DIR__, "testbackend.jl"))
include(joinpath(@__DIR__, "freshprocess.jl"))

@testset "the shutdown hook covers every device" begin
    status, out = infreshprocess(raw"""
        dev1 = Mantle.Device(BACKEND)
        dev2 = Mantle.Device(Mantle.devicename(dev1))
        # Distinct devices, or the test is one device asserted twice.
        dev2 === dev1 && error("expected two devices, got one")
        (Mantle.retired(dev1) || Mantle.retired(dev2)) && error("a device was retired before exit")
        # Something on each, for its finalizers to release after the hook.
        a1 = Mantle.Buffer(dev1, ones(Float32, 256))
        a2 = Mantle.Buffer(dev2, ones(Float32, 256))
        Mantle.waitidle(dev1)
        Mantle.waitidle(dev2)
        push!(DEVICES, dev1, dev2)
        exit(0)
        """; before = raw"""
        # Runs after Mantle's own exit hook, and turns what it left into the exit
        # code: 2 if a device is still live, 3 if there was none to look at (the
        # script failed before it registered them, and printed why).
        const DEVICES = Any[]
        atexit() do
            isempty(DEVICES) && exit(3)
            all(d -> Main.Mantle.retired(d), DEVICES) || exit(2)
        end
        """)
    reportfresh(status, out)
    @test status == 0
end
