"""
A graph arena's memory must be allocated for device addresses.

`rawalloc(dev, ::Buffers, …)` creates the arena buffer with
`BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT` — kernels reach a suballocated transient
as `address + offset`, which is the whole bridge — and the spec requires the
memory bound to such a buffer to have been allocated with
`VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT`. `device_memory` passed no
`VkMemoryAllocateFlagsInfo` at all, so every arena in every graph violated it.

Nothing failed on NVIDIA. That driver returns a usable address for memory
allocated without the flag, so the whole suite, every renderer and every
benchmark ran green on it for as long as the arena has existed. RADV does not:
it takes the address, and the process dies later and elsewhere — a SIGSEGV
inside `vkCreateRayTracingPipelinesKHR`, with nothing in the stack pointing at
an allocation. It reads as a driver bug, which is what it was first called.

So the assertion here is not "the render is correct" — it was correct on NVIDIA
throughout — but that the VALIDATION LAYER is silent. The layer is part of the
loader rather than the driver, so this catches the fault on any GPU, including
the one that tolerates it.

A transient buffer is what makes the arena exist; a graph of plain `Buffer`s
allocates through the pool, which passes the flag and always did.
"""

using Test, Mantle, Lava, KernelAbstractions
const KA = KernelAbstractions

@kernel function arena_copy!(dst, @Const(src))
    i = @index(Global)
    @inbounds dst[i] = src[i] + 1f0
end

"""A graph with a transient, so an arena is allocated to back it."""
function _arenaplan(dev, out, n)
    g = Mantle.Graph(dev)
    seed = Mantle.Buffer(dev, zeros(Float32, n))
    t = Mantle.Transient.Buffer(g, Float32, n)
    Mantle.compute!(g, "in") do p
        s = Mantle.use(p, seed; read = true); d = Mantle.use(p, t; write = true)
        Mantle.dispatch!(p, arena_copy!, (d, s), n)
    end
    Mantle.compute!(g, "out") do p
        s = Mantle.use(p, t; read = true); d = Mantle.use(p, out; write = true)
        Mantle.dispatch!(p, arena_copy!, (d, s), n)
    end
    (Mantle.record!(Mantle.Plan(g)), seed)
end

@testset "arena memory is allocated for device addresses" begin
    # Validation on, which needs a device rebuild: the suite's device is built
    # without it. Captured and restored explicitly — `reset_device!()` with no
    # `debug` PRESERVES the current config (`old.debug`), so the obvious
    # `finally reset_device!()` leaves validation ON for every later test. It did:
    # `test_pool_alloc_oom` then reported a validation error where it expected
    # "out of memory", and `test_coopmat_perelement` got a spirv-val error, both
    # ~40 minutes downstream of here with nothing pointing back.
    prev = Mantle.vk_context().debug
    Mantle.reset_device!(debug = Mantle.DebugConfig(validation = true))
    try
        dev = Mantle.Device(Mantle.VulkanAPI())
        be = Mantle.defaultbackend()
        n = 4096
        out = Mantle.Buffer(dev, zeros(Float32, n))
        pl, seed = Base.invokelatest(_arenaplan, dev, out, n)

        Mantle.run!(pl)
        KA.synchronize(be)

        # Correct on the tolerant driver either way, so this is not the assertion
        # — it is here to prove the arena was actually exercised.
        @test Array(Mantle.storage(out)) == fill(2f0, n)

        # THE assertion. Without the flag this holds
        # `VUID-vkBindBufferMemory-bufferDeviceAddress-03339`.
        ctx = Mantle.vk_context()
        Mantle.drain_validation_messages!(ctx)
        @test isempty(ctx.validation.messages)

        Mantle.free!(pl)
    finally
        Mantle.reset_device!(debug = prev)
    end
end
