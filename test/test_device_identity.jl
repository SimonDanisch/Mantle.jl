"""
Two live devices, and every constructor lands its work on the one it was given.

Backend independence made a device an explicit argument. This is the check that
it took: with the device under test live and a second device built beside it, a
graph, an array, a framebuffer and an `Adapt.adapt` asked of the SECOND device's
backend go to the second device, not to whichever is current. Before the fix the
Vulkan backend's `Device(::LavaBackend)` dropped its argument, `backend(dev)` was
unpinned, `Framebuffer`/`Window` fell through to the process default, and `Adapt`
and `similar` allocated on the process default — so a renderer handed a
second-device backend built its transients on the first device while its accel
was on the second, and the first cross-device buffer use faulted the driver.

Needs a second device on the same API. Every Linux box here has one, because the
Vulkan loader enumerates lavapipe beside the real GPU; a Mac has one Metal
device. Skipped, loudly, where there is none.
"""

using Test, Mantle, KernelAbstractions, Adapt, LinearAlgebra
include(joinpath(@__DIR__, "testbackend.jl"))
const KA = KernelAbstractions

@kernel function di_bump!(a, n)
    i = @index(Global)
    @inbounds i <= n[1] && (a[i] += Int32(1))
end

di_fragment(_) = Mantle.Vec4f(0.25f0, 0.5f0, 0.75f0, 1f0)

"""
Another device on the API `dev` is on, built now and not installed as the
default, or `nothing` when this machine has no second one.

A CPU device first where there is one: lavapipe costs no GPU memory and is on
every Linux machine here. The API is the one `dev` reports (`syncbackend`),
because a second device of ANOTHER API — the Metal GPU beside a lavapipe run on a
Mac — has a backend that is not tied to one device, and this file is about
devices of one API telling each other apart.
"""
function otherdevice(dev)
    api = Mantle.syncbackend(dev)
    others = filter(i -> i.name != Mantle.devicename(dev), Mantle.devices(api))
    isempty(others) && return nothing
    pick = something(findfirst(i -> i.kind == :cpu, others), 1)
    return Mantle.Device(api; select = others[pick].index)
end

@testset "a device is an explicit argument" begin
    dev = Mantle.Device(TESTBACKEND)
    api = Mantle.syncbackend(dev)
    default = Mantle.Device(api)                     # the process default of this API
    other = otherdevice(dev)
    if other === nothing
        @info "only one device on this API here; the two-device identity check needs a second" device = Mantle.devicename(dev)
    else
        @test other !== dev
        # Building the second did not install it.
        @test Mantle.Device(api) === default
        @test Mantle.Device(TESTBACKEND) === dev

        # ── identity: the second device's backend resolves to the second device
        bo = Mantle.backend(other)
        @test Mantle.Device(bo) === other
        @test Mantle.Device(bo) !== dev
        @test Mantle.Device(Mantle.backend(dev)) === dev    # the first, still pinned

        # ── allocation lands on the named device, never the default
        devof(x) = Mantle.Device(KA.get_backend(x))
        @test devof(KA.allocate(bo, Float32, 8)) === other
        @test devof(Adapt.adapt(bo, rand(Float32, 4))) === other
        @test devof(similar(KA.allocate(bo, Float32, 8))) === other

        # ── a backend's identity is its device
        @test Mantle.backend(dev) == TESTBACKEND
        @test bo != TESTBACKEND

        # ── a framebuffer built from the second device's backend is the second
        #    device's: a draw on that device renders into it and reads back. Had
        #    it fallen through to the default, the draw would name an image of
        #    another device.
        if Mantle.supports_graphics(other)
            fb = Mantle.Framebuffer(bo, 16, 16; depth = false, color_format = RGBA{Float32})
            pipe = Mantle.GraphicsPipeline(; vertex = Mantle.VertexShader(Mantle.blit_vertex),
                                             fragment = Mantle.FragmentShader(di_fragment),
                                             blend = Mantle.Opaque(), cull = Mantle.NoCull(),
                                             depth = Mantle.DepthOff())
            Mantle.draw!(other, pipe, Mantle.OffscreenTarget(fb), 3;
                         clear_color = (0f0, 0f0, 0f0, 1f0))
            Mantle.flush!(other)
            px = Mantle.readback_framebuffer(fb)[8, 8]
            @test px.r ≈ 0.25f0 atol = 0.01
            @test px.g ≈ 0.5f0 atol = 0.01
            @test px.b ≈ 0.75f0 atol = 0.01
        end

        # ── a recorded graph on the second device runs THERE. The result is in
        #    the second device's memory, so a plan that ran on the default would
        #    not have produced it.
        let n = 128
            out = Mantle.Buffer(other, zeros(Int32, n))
            cnt = Mantle.Buffer(other, Int32[n])
            g = Mantle.Graph(other)
            Mantle.dispatch!(g, di_bump!, (out, cnt), Mantle.DeviceRange(cnt); group = 64, name = "bump")
            pl = Mantle.record!(Base.invokelatest(Mantle.Plan, g))
            Mantle.run!(pl); Mantle.waitfor!(pl)
            @test Array(Mantle.storage(out)) == fill(Int32(1), n)
            Mantle.free!(pl)
        end

        # ── a cross-device copy stages through the host instead of faulting
        let
            a = KA.allocate(bo, Float32, 16); fill!(a, 3f0)
            b = KA.allocate(TESTBACKEND, Float32, 16); fill!(b, 0f0)
            copyto!(b, a)
            KA.synchronize(TESTBACKEND)
            @test all(Array(b) .== 3f0)
        end
    end
end
