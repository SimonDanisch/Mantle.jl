"""
A graph arena's memory must be allocated for device addresses, and a graph that
runs on it must leave the validation layer silent.

On Vulkan, `rawalloc(dev, ::Buffers, …)` creates the arena buffer with
`BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT` — kernels reach a suballocated
transient as `address + offset`, which is the whole bridge — and the spec
requires the memory bound to such a buffer to have been allocated with
`VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT`. `device_memory` passed no
`VkMemoryAllocateFlagsInfo` at all, so every arena in every graph violated it.

Nothing failed on NVIDIA. That driver returns a usable address for memory
allocated without the flag, so the whole suite, every renderer and every
benchmark ran green on it for as long as the arena has existed. RADV does not:
it takes the address, and the process dies later and elsewhere — a SIGSEGV
inside `vkCreateRayTracingPipelinesKHR`, with nothing in the stack pointing at
an allocation. It reads as a driver bug, which is what it was first called.

So the assertion here is not "the render is correct" — it was correct on NVIDIA
throughout — but that the VALIDATION LAYER is silent: the layer is not the
driver, so it reports the fault on any GPU, including the one that tolerates it.
On Metal the same question goes to its API validation layer, which checks what a
graph's arena and its recorded commands do with buffers and residency.

In a process of its own, started with what validation needs (`Mantle.debugenv`):
Metal decides validation when a process starts, and a device rebuilt with
validation inside the suite would leave it on for every file after this one —
which the Vulkan version did once, and `test_pool_alloc_oom` then reported a
validation error where it expected "out of memory", 40 minutes downstream.

A transient buffer is what makes the arena exist; a graph of plain `Buffer`s
allocates through the pool's persistent blocks.
"""

using Test, Mantle, KernelAbstractions
include(joinpath(@__DIR__, "testbackend.jl"))
include(joinpath(@__DIR__, "freshprocess.jl"))

@testset "arena memory is allocated for device addresses" begin
    status, out = infreshprocess(raw"""
        @kernel function arena_copy!(dst, @Const(src))
            i = @index(Global)
            @inbounds dst[i] = src[i] + 1f0
        end

        # A graph with a transient, so an arena is allocated to back it.
        function arenaplan(dev, out, n)
            g = Mantle.Graph(dev)
            seed = Mantle.Buffer(dev, zeros(Float32, n))
            t = Mantle.Transient.Buffer(g, Float32, n)
            Mantle.dispatch!(g, arena_copy!, (t, seed), n; name = "in")
            Mantle.dispatch!(g, arena_copy!, (out, t), n; name = "out")
            pl = Mantle.Plan(g)
            Mantle.recordable(dev, pl) && Mantle.record!(pl)
            return pl, seed
        end

        dev = Mantle.reset_device!(Mantle.Device(BACKEND); debug = DebugConfig(validation = true))
        # A device whose layer did not load (no Vulkan validation layer installed)
        # cannot answer the question; that is reported, not passed.
        Mantle.validating(dev) || exit(3)
        Mantle.validationmessages!(dev)
        @testset "the validation layer is silent" begin
            n = 4096
            out = Mantle.Buffer(dev, zeros(Float32, n))
            pl, seed = arenaplan(dev, out, n)
            Mantle.run!(pl)
            Mantle.waitidle(dev)
            # Correct on a tolerant driver either way, so this is not the
            # assertion — it proves the arena was exercised.
            @test Array(Mantle.storage(out)) == fill(2f0, n)
            # THE assertion. On Vulkan, without the flag this holds
            # `VUID-vkBindBufferMemory-bufferDeviceAddress-03339`.
            msgs = Mantle.validationmessages!(dev)
            foreach(println, msgs)
            @test isempty(msgs)
            Mantle.free!(pl)
        end
        """; debug = DebugConfig(validation = true))
    if status == 3
        @info "no validation layer for this device here; the arena check needs one"
        @test_skip status == 0
    else
        reportfresh(status, out)
        @test status == 0
    end
end
