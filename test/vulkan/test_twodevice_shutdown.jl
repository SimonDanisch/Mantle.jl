"""
The shutdown hook marks EVERY device lost, not just the bound one.

`atexit` runs before Julia's final finalizer sweep, and `MantleVulkanExt.__init__`
uses that: it marks the device lost so a `LavaArray` finalizer running afterwards
takes the `device_lost(Mantle.ctxof(bq))` branch in `vk_free!` and skips `query_timeline`,
instead of calling `vkGetSemaphoreCounterValue` on a driver that has already been
torn down.

It marked ONE — `VK_CONTEXT_REF[]`. `twodevice_probe.jl` beside this file makes a
second, and on 2026-08-30 the suite passed every test and then died on the way
out with exit 139:

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
invariant: after `mark_all_devices_lost!`, every live context reports
`device_lost`, which is precisely what `vk_free!` consults and precisely what was
true for one device and false for the other.

It does NOT try to reproduce the segfault by making two devices, allocating on
both and exiting. That was written first and it passes with the bug still in
place: whether the crash happens depends on whether Vulkan.jl's handle finalizers
destroy the device before or after Mantle's array finalizers run in the same
sweep, and the small case gets the safe order. A regression test that passes
either way is worse than none, so this checks the property that actually
differs.

In a subprocess because it is destructive: marking devices lost turns every
later Vulkan call in the process into a no-op, which would take the rest of the
suite with it.
"""

using Test, Mantle, Lava

@testset "the shutdown hook covers every device" begin
    # Skipped rather than failed where the machine has no software rasterizer:
    # the assertion needs a second driver, and one is all some boxes offer.
    # Loud, because a quiet skip is how this went unnoticed for months.
    have_lvp = any(i -> i.kind == :cpu, Mantle.devices(Mantle.VulkanAPI()))
    if !have_lvp
        @info "no lavapipe device here; the two-device shutdown check needs a second driver"
    else
        code = """
        using Mantle, Lava, Test
        gpu = Mantle.vk_context()
        cpu = Mantle.VkContext(select = "llvmpipe")
        # Distinct devices, or the test is one device asserted twice.
        gpu === cpu && error("expected two contexts, got one")
        Mantle.device_lost(gpu) && error("gpu context was already marked lost")
        Mantle.device_lost(cpu) && error("lavapipe context was already marked lost")

        Mantle.mark_all_devices_lost!()            # what `atexit` calls

        # The bound one was always marked. The second is the regression: with the
        # old hook it stayed live, and its buffers' finalizers called into a dead
        # driver during the shutdown sweep.
        Mantle.device_lost(gpu) || error("bound context not marked lost")
        Mantle.device_lost(cpu) || error("SECOND context not marked lost — the bug")
        exit(0)
        """
        cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) -e $code`
        log = joinpath(mktempdir(), "twodevice.log")
        ok = success(pipeline(ignorestatus(cmd); stdout = log, stderr = log))
        ok || println(read(log, String))
        @test ok
    end
end
