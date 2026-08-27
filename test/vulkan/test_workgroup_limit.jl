"""
Workgroups above 256 used to write part of their output, silently.

The recorded diagnosis was hardware: "above 256 this driver silently runs fewer
invocations than the shader declares". It was wrong, and the way it was wrong is
the reason these tests exist in this shape.

The cause was one line in `get_compute_pipeline`:

    cache_key = hash((spirv_bytes, ...))

`Base.hash` on a large `Vector` **samples** elements rather than reading all of
them. The 256- and 512-wide modules of one kernel differ at **exactly one byte**
— the `LocalSize` operand — and collide. The 512 launch then looked up the 256
pipeline, dispatched a 256-thread shader over a grid computed for 512, and wrote
exactly `256/wg` of its output. Everything that made it look like a hardware lane
cap follows from that: whichever size compiled first won ("order dependence"),
whether a given body's two modules happened to collide decided if it reproduced
("body dependence"), and adding any unrelated store changed enough bytes to miss
the collision (the "fix" that made no sense).

Three things are pinned here, so a regression is loud:

  * the hardware runs every lane it is given, at every size;
  * full coverage at every size on **both** launch spellings;
  * and directly, that two modules differing in one byte get different cache
    keys — the property the whole thing rests on.
"""

using Test, Lava, KernelAbstractions

const KA = KernelAbstractions
const Atomixwg = Lava.Atomix

# 64 simultaneously live values — the body whose two modules collided. 32 and 128
# did not, which is why the old cap could not be predicted from any statistic.
@kernel cpu=false function wglimit_probe!(out, ::Val{K}) where {K}
    I = @index(Global, Linear)
    acc = ntuple(k -> Int32(I) * Int32(k) + Int32(k * k), Val(K))
    s = zero(Int32)
    for k in 1:K
        s = s ⊻ acc[k]
    end
    @inbounds out[I] = Float16((s & Int32(3)) + Int32(1))   # never zero
end

# No KA index machinery: the raw builtin, an unconditional store, an atomic
# tally. This is what answered "does the hardware run the lanes at all".
@kernel cpu=false unsafe_indices=true function wglimit_raw!(who, ran)
    li = Lava.lava_local_invocation_index()
    Atomixwg.@atomic ran[1] += Int32(1)
    @inbounds who[1 + li] = Int32(1)
end

"Fraction of a single workgroup's output that actually got written."
function groupcoverage(backend, K, wg; static::Bool)
    out = KA.allocate(backend, Float16, wg)
    fill!(out, zero(Float16))
    if static
        wglimit_probe!(backend, wg)(out, Val(K); ndrange = wg)
    else
        wglimit_probe!(backend)(out, Val(K); ndrange = wg, workgroupsize = wg)
    end
    KA.synchronize(backend)
    count(!=(zero(Float16)), Array(out)) / wg
end

@testset "workgroup limit" begin
    backend = LavaBackend()

    @testset "a one-byte difference is a different pipeline" begin
        # The regression test for the actual bug, independent of any device
        # behaviour: `Base.hash` collides on these, `spirv_content_hash` must not.
        a = rand(UInt8, 9260)
        b = copy(a); b[230] = a[230] ⊻ 0x02
        @test Lava.spirv_content_hash(a) != Lava.spirv_content_hash(b)
        # Every byte must matter, not just one position.
        for i in (1, 4096, length(a))
            c = copy(a); c[i] = a[i] ⊻ 0xff
            @test Lava.spirv_content_hash(a) != Lava.spirv_content_hash(c)
        end
        # And it must be a pure function of the content.
        @test Lava.spirv_content_hash(a) == Lava.spirv_content_hash(copy(a))
    end

    @testset "the hardware runs every lane it is given" begin
        for wg in (256, 384, 512, 768, 1024)
            who = KA.zeros(backend, Int32, wg)
            ran = KA.zeros(backend, Int32, 1)
            wglimit_raw!(backend, wg)(who, ran; ndrange = wg)
            KA.synchronize(backend)
            @test Array(ran)[1] == wg
            @test sum(Array(who)) == wg
        end
    end

    @testset "full coverage at every size, both launch spellings" begin
        # 1024 is this device's `maxComputeWorkGroupInvocations`, now queried
        # rather than assumed — the sizes below have to be launchable for the
        # coverage assertions to mean anything.
        @test Mantle.workgroup_limit() == 1024
        for K in (32, 64, 128), wg in (64, 128, 256, 512, 1024)
            @test groupcoverage(backend, K, wg; static = true) == 1.0
            @test groupcoverage(backend, K, wg; static = false) == 1.0
        end
    end

    @testset "compile order does not change the answer" begin
        # 256 before 512 is the order that used to poison the cache.
        @test groupcoverage(backend, 96, 256; static = false) == 1.0
        @test groupcoverage(backend, 96, 512; static = false) == 1.0
        @test groupcoverage(backend, 97, 512; static = false) == 1.0
        @test groupcoverage(backend, 97, 256; static = false) == 1.0
    end

    @testset "past the device limit it still throws" begin
        out = KA.allocate(backend, Float16, 2048)
        @test_throws ArgumentError wglimit_probe!(backend)(out, Val(64);
                                                          ndrange = 2048, workgroupsize = 2048)
    end
end
