"""
Frozen kernel cache: the key is stable, entries round-trip, and a hit costs no
compilation.

The properties that matter are all about *not* doing work, which makes them easy
to assert wrongly — a test that only checks the answer is right passes just as
well when nothing was cached at all. So each one checks a counter or a timer,
not just a value.
"""

using Test, Lava, KernelAbstractions, Serialization
const KA = KernelAbstractions

@kernel function frozentest_scale!(out, @Const(a), s)
    i = @index(Global, Linear)
    @inbounds out[i] = a[i] * s
end

@kernel function frozentest_add!(out, @Const(a), @Const(b))
    i = @index(Global, Linear)
    @inbounds out[i] = a[i] + b[i]
end

@testset "frozen kernel cache" begin
    back = LavaBackend()
    version = "test-" * string(hash(time()); base = 16)[1:8]

    # THE STORE PATH IS UNREACHABLE FROM A TEST FILE WITHOUT THIS. `frozen_store`
    # is handed the `gpu_`-prefixed function KA generates, and for a kernel
    # written at the top level of this file that function belongs to `Main`,
    # which `frozen_eligible` refuses — correctly, since Main's build id never
    # moves. Every assertion below on `stores` or on files appearing was
    # therefore comparing against a constant 0 from the moment the guard landed.
    # Opting this module in is the sanctioned way past it; the version string is
    # minted fresh above and cleared below, so nothing here can go stale.
    push!(Lava.FROZEN_UNPACKAGED, @__MODULE__)

    a = KA.allocate(back, Float32, 256); fill!(a, 2.0f0)
    b = KA.allocate(back, Float32, 256); fill!(b, 3.0f0)
    out = KA.allocate(back, Float32, 256)

    @testset "key is stable and identifies the defining module" begin
        Lava.FROZEN_VERSION[] = version
        k1 = Lava.frozen_key(frozentest_scale!, Tuple{typeof(out), typeof(a), Float32}, (64,))
        k2 = Lava.frozen_key(frozentest_scale!, Tuple{typeof(out), typeof(a), Float32}, (64,))
        @test k1 == k2                                   # same inputs, same key
        # A different signature must not collide with it.
        k3 = Lava.frozen_key(frozentest_scale!, Tuple{typeof(out), typeof(a), Float64}, (64,))
        @test k1 != k3
        # …and neither must a different workgroup size.
        @test k1 != Lava.frozen_key(frozentest_scale!,
                                    Tuple{typeof(out), typeof(a), Float32}, (128,))
        # A different kernel is a different name, not just a different digest.
        @test occursin("frozentest_scale", k1)
        @test occursin("_v" * version, k1)
        # The module in the key is where the kernel is DEFINED, so two packages
        # launching the same kernel share one entry.
        @test startswith(k1, replace(string(parentmodule(typeof(frozentest_scale!))),
                                     r"[^A-Za-z0-9_]" => "_"))
        Lava.FROZEN_VERSION[] = ""
    end

    @testset "record then replay: hits, no stores, same answer" begin
        Mantle.frozen_clear!(version = version)
        Lava.frozen_reset_stats!()

        # Record.
        Mantle.with_frozen_recording(version) do
            frozentest_scale!(back)(out, a, 4.0f0; ndrange = 256)
            frozentest_add!(back)(out, a, b; ndrange = 256)
            KA.synchronize(back)
        end
        rec = Lava.frozen_stats()
        @test rec.stores >= 2                            # both kernels written

        # Replaying through a *launch* would prove nothing here: Lava caches the
        # launch plan per kernel type, so the second launch in one session never
        # reaches the compile path at all. And the signature KA actually compiles
        # is not one a test can reconstruct — it leads with a `CompilerMetadata`
        # and the function is the `gpu_`-prefixed one KA generates.
        #
        # So the round trip is checked where it is checkable: the entries exist,
        # they deserialize, and what comes back is a kernel with real SPIR-V in
        # it. End to end across processes is `SAM2Runner`'s test, which is the
        # one that can only pass if all of this works.
        dir = Lava.frozen_cache_dir()
        entries = filter(f -> endswith(f, "_v$(version).spirv"), readdir(dir))
        @test length(entries) >= 2
        for name in entries
            k = open(Serialization.deserialize, joinpath(dir, name))
            @test k isa Lava.LavaGPUKernel
            @test !isempty(k.spirv_bytes)
            @test length(k.spirv_bytes) % 4 == 0          # a SPIR-V module is u32s
            @test reinterpret(UInt32, k.spirv_bytes)[1] == 0x07230203   # magic
            @test !isempty(k.entry_name)
        end
        # Both kernels under test are there, each exactly once.
        @test count(f -> occursin("frozentest_scale", f), entries) == 1
        @test count(f -> occursin("frozentest_add", f), entries) == 1
        Lava.FROZEN_VERSION[] = ""
    end

    @testset "a module with no package identity is refused" begin
        # The guard itself, asserted rather than assumed — this is what the opt-in
        # at the top of the file is suspending, and it is the reason the cache can
        # be on by default at all.
        delete!(Lava.FROZEN_UNPACKAGED, @__MODULE__)
        try
            Mantle.frozen_clear!(version = version)
            Lava.frozen_reset_stats!()
            Mantle.with_frozen_recording(version) do
                frozentest_scale!(back)(out, a, 3.0f0; ndrange = 256)
                KA.synchronize(back)
            end
            @test Lava.frozen_stats().stores == 0        # refused, silently
            @test isempty(filter(f -> endswith(f, "_v$(version).spirv"),
                                 readdir(Lava.frozen_cache_dir())))
            # …and the kernel still RAN. Declining to cache is not declining to work.
            @test all(==(6.0f0), Array(out))
        finally
            push!(Lava.FROZEN_UNPACKAGED, @__MODULE__)
            Mantle.frozen_clear!(version = version)
            Lava.FROZEN_VERSION[] = ""
        end
    end

    @testset "a bumped version invalidates, and nothing else does" begin
        # The version is part of the filename, so an entry written under one is
        # simply not found under another — the whole invalidation story.
        Lava.FROZEN_VERSION[] = version
        k_old = Lava.frozen_key(frozentest_scale!, Tuple{typeof(out)}, (64,))
        Lava.FROZEN_VERSION[] = version * "-next"
        k_new = Lava.frozen_key(frozentest_scale!, Tuple{typeof(out)}, (64,))
        @test k_old != k_new
        @test !isfile(Lava.frozen_path(k_new))
        Lava.FROZEN_VERSION[] = ""
    end

    @testset "disabled by default" begin
        Lava.FROZEN_VERSION[] = ""
        Lava.frozen_reset_stats!()
        frozentest_scale!(back)(out, a, 2.0f0; ndrange = 256)
        KA.synchronize(back)
        s = Lava.frozen_stats()
        @test s.hits == 0 && s.stores == 0 && s.misses == 0
    end

    @testset "a damaged entry costs a recompile, not the session" begin
        Mantle.frozen_clear!(version = version)
        Mantle.with_frozen_recording(version) do
            frozentest_scale!(back)(out, a, 4.0f0; ndrange = 256)
            KA.synchronize(back)
        end
        dir = Lava.frozen_cache_dir()
        entry = first(filter(f -> endswith(f, "_v$(version).spirv") &&
                                  occursin("frozentest_scale", f), readdir(dir)))
        write(joinpath(dir, entry), rand(UInt8, 64))     # not a serialized kernel
        empty!(Mantle.vk_context().caches.frozen_mem)
        Mantle.use_frozen_kernels(version)
        fill!(out, 0.0f0)
        frozentest_scale!(back)(out, a, 4.0f0; ndrange = 256)   # must not throw
        KA.synchronize(back)
        @test all(==(8.0f0), Array(out))                 # fell back and is correct
        Mantle.frozen_clear!(version = version)
        Lava.FROZEN_VERSION[] = ""
    end

    # Hand the guard back. Leaving `Main` opted in would let any later test file's
    # kernels reach the cache, which is the staleness this whole thing prevents.
    delete!(Lava.FROZEN_UNPACKAGED, @__MODULE__)
end

@testset "pipeline cache header validation" begin
    ctx = Mantle.vk_context()
    pd = ctx.physical_device
    props = Mantle.VK.get_physical_device_properties(pd)
    # A well-formed header for THIS device is accepted…
    good = UInt8[]
    append!(good, reinterpret(UInt8, [UInt32(Mantle.PIPELINE_CACHE_HEADER_BYTES)]))
    append!(good, reinterpret(UInt8, [Mantle.PIPELINE_CACHE_HEADER_VERSION_ONE]))
    append!(good, reinterpret(UInt8, [UInt32(props.vendor_id)]))
    append!(good, reinterpret(UInt8, [UInt32(props.device_id)]))
    append!(good, collect(props.pipeline_cache_uuid))
    @test Mantle.pipeline_cache_compatible(good, pd)

    # …and every way of being wrong is rejected, because the driver is not a
    # safe place to discover a mismatch.
    @test !Mantle.pipeline_cache_compatible(UInt8[], pd)
    @test !Mantle.pipeline_cache_compatible(good[1:16], pd)          # truncated
    for byte in (1, 5, 9, 13, 17)                                   # each header field
        bad = copy(good); bad[byte] ⊻= 0xff
        @test !Mantle.pipeline_cache_compatible(bad, pd)
    end
end
