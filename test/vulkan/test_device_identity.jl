"""
Two live devices, and every constructor lands its work on the one it was given.

Backend independence made a device an explicit argument. This is the check that
it took: with the real GPU bound as the default and lavapipe built beside it, a
graph, an array, a framebuffer and an `Adapt.adapt` asked of the SECOND device's
backend go to the second device, not to whichever is current. Before the fix
`Device(::LavaBackend)` dropped its argument, `backend(dev)` was unpinned,
`Framebuffer`/`Window` fell through to a `vk_context()` default, and `Adapt` and
`similar` allocated on the process default — so a renderer handed a second-device
backend built its transients on the first device while its accel was on the
second, and the first cross-device buffer use faulted the driver.

Needs a second driver, which every machine here has: the loader enumerates
lavapipe beside the real GPU. Skipped, loudly, where it does not.
"""

using Test, Mantle, Lava, KernelAbstractions, Adapt, LinearAlgebra
const KA = KernelAbstractions

@kernel function di_bump!(a, n)
    i = @index(Global)
    @inbounds i <= n[1] && (a[i] += Int32(1))
end

@testset "a device is an explicit argument" begin
    infos = Mantle.devices(Mantle.VulkanAPI())
    lvp = findfirst(i -> i.kind == :cpu, infos)
    if lvp === nothing
        @info "no software rasterizer here; the two-device identity check needs a second driver"
    else
        gpudev = Mantle.Device(Mantle.VulkanAPI())          # the process default
        gpu = gpudev.ctx
        cpudev = Mantle.Device(Mantle.VulkanAPI(); select = "llvmpipe")
        cpu = cpudev.ctx
        try
            @test gpu !== cpu
            @test MVE.vk_context() === gpu               # building the second did not install it

            # ── identity: the second device's backend resolves to the second device
            bcpu = MVE.LavaBackend(cpu)
            @test Mantle.Device(bcpu) === cpudev
            @test Mantle.Device(bcpu) !== gpudev
            @test MVE.vk_context(Mantle.backend(cpudev)) === cpu
            @test MVE.vk_context(MVE.LavaBackend()) === gpu             # the default, pinned
            @test MVE.vk_context(Mantle.defaultbackend()) === gpu

            # ── allocation lands on the named device, never the default
            @test MVE.vk_context(KA.allocate(bcpu, Float32, 8)) === cpu
            @test MVE.vk_context(Adapt.adapt(bcpu, rand(Float32, 4))) === cpu
            @test MVE.vk_context(similar(KA.allocate(bcpu, Float32, 8))) === cpu
            @test Mantle.Framebuffer(bcpu, 16, 16; depth = false).ctx === cpu

            # ── a backend's identity is its device, not its queue
            @test MVE.LavaBackend() == MVE.LavaBackend(gpu)
            @test MVE.LavaBackend(gpu) != bcpu
            q = Mantle.allocate_batch_queue!(cpu)
            try
                @test MVE.LavaBackend(q) == bcpu           # another queue, same device
            finally
                Mantle.release_batch_queue!(q)
            end

            # ── a recorded graph on the second device runs THERE, on its own timeline
            let n = 128
                out = Mantle.Buffer(cpudev, zeros(Int32, n))
                cnt = Mantle.Buffer(cpudev, Int32[n])
                g = Mantle.Graph(cpudev)
                Mantle.compute!(g, "bump") do p
                    Mantle.use(p, out; read = true, write = true)
                    Mantle.use(p, cnt; read = true)
                    Mantle.dispatch!(p, di_bump!, (out, cnt), Mantle.DeviceRange(cnt); group = 64)
                end
                pl = Mantle.record!(Base.invokelatest(Mantle.Plan, g))
                before_gpu = MVE.driver(gpu.default_bq).next_timeline
                Mantle.run!(pl); Mantle.waitfor!(pl)
                @test Array(Mantle.storage(out)) == fill(Int32(1), n)
                @test MVE.driver(gpu.default_bq).next_timeline == before_gpu   # nothing ran on the default
                Mantle.free!(pl)
            end

            # ── a cross-device copy stages through the host instead of faulting
            let
                a = KA.allocate(bcpu, Float32, 16); fill!(a, 3f0)
                b = KA.allocate(MVE.LavaBackend(gpu), Float32, 16); fill!(b, 0f0)
                copyto!(b, a)
                KA.synchronize(MVE.LavaBackend(gpu))
                @test all(Array(b) .== 3f0)
            end

            # ── the compiler feature record is per device, not a global the last
            #    bind left behind
            @test cpu.features.ray_query === cpu.ray_query_available
            @test gpu.features.ser === gpu.ser_available
        finally
            MVE.mark_device_lost!(cpu)      # retire the device this test built; nothing else can
        end
    end
end
