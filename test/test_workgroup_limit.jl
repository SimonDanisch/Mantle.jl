"""
Workgroups above 256 must not write part of their output, silently.

That failure reads as hardware — "above 256 this driver silently runs fewer
invocations than the shader declares" — and is not, which is the reason these
tests exist in this shape.

The cause was one line in the Vulkan backend's `get_compute_pipeline`:

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

Pinned here, so a regression is loud on any backend whose pipelines are keyed by
their compiled code:

  * the hardware runs every lane it is given, at every size it allows;
  * full coverage at every size on **both** launch spellings;
  * compiling one size first does not change what another size computes.

(The direct check of the key itself — that two modules differing in one byte get
different `spirv_content_hash`es — is a host-only test of Lava's compiler and
belongs in Lava's own suite, not in a file that runs on every backend.)
"""

using Test, KernelAbstractions, Atomix
import KernelInterface as KI
include(joinpath(@__DIR__, "testbackend.jl"))

const KA = KernelAbstractions

# 64 simultaneously live values — the body whose two modules collided. 32 and 128
# did not, which is why the apparent cap cannot be predicted from any statistic.
@kernel cpu=false function wglimit_probe!(out, ::Val{K}) where {K}
    I = @index(Global, Linear)
    acc = ntuple(k -> Int32(I) * Int32(k) + Int32(k * k), Val(K))
    s = zero(Int32)
    for k in 1:K
        s = s ⊻ acc[k]
    end
    @inbounds out[I] = Float16((s & Int32(3)) + Int32(1))   # never zero
end

# No KA index machinery: the backend's own local index, an unconditional store,
# an atomic tally. This is what answered "does the hardware run the lanes at all".
@kernel cpu=false unsafe_indices=true function wglimit_raw!(who, ran)
    li = KI.get_local_id().x
    Atomix.@atomic ran[1] += Int32(1)
    @inbounds who[li] = Int32(1)
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
    backend = TESTBACKEND
    # The device's own limit, queried rather than assumed: a size above it is not
    # launchable, and a coverage assertion at such a size would mean nothing.
    limit = Mantle.caps(backend).workgrouplimit
    fits(sizes) = filter(<=(limit), sizes)

    @testset "the hardware runs every lane it is given" begin
        for wg in fits((256, 384, 512, 768, 1024))
            who = KA.zeros(backend, Int32, wg)
            ran = KA.zeros(backend, Int32, 1)
            wglimit_raw!(backend, wg)(who, ran; ndrange = wg)
            KA.synchronize(backend)
            @test Array(ran)[1] == wg
            @test sum(Array(who)) == wg
        end
    end

    @testset "full coverage at every size, both launch spellings" begin
        for K in (32, 64, 128), wg in fits((64, 128, 256, 512, 1024))
            @test groupcoverage(backend, K, wg; static = true) == 1.0
            @test groupcoverage(backend, K, wg; static = false) == 1.0
        end
    end

    @testset "compile order does not change the answer" begin
        # 256 before 512 is the order that poisons a sampling-hashed cache.
        if 512 <= limit
            @test groupcoverage(backend, 96, 256; static = false) == 1.0
            @test groupcoverage(backend, 96, 512; static = false) == 1.0
            @test groupcoverage(backend, 97, 512; static = false) == 1.0
            @test groupcoverage(backend, 97, 256; static = false) == 1.0
        end
    end

    @testset "past the device limit it still throws" begin
        out = KA.allocate(backend, Float16, 2limit)
        @test_throws ArgumentError wglimit_probe!(backend)(out, Val(64);
                                                          ndrange = 2limit, workgroupsize = 2limit)
    end
end
