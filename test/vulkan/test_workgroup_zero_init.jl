# Workgroup (shared) memory must start zeroed.
#
# Vulkan leaves Workgroup storage undefined at the start of a dispatch, so a
# kernel that reads a slot it never wrote sees garbage. AcceleratedKernels'
# block-level merge does exactly that when `len < 2 * block_size`, which used to
# produce wrong results on Lava and forced an `AK.merge_sort_by_key!` override in
# array/mapreduce.jl. That override was circular — AK builds `sortperm` *on top
# of* `merge_sort_by_key!` — so the two recursed until either a
# StackOverflowError or 34 GB of pool growth and ERROR_OUT_OF_DEVICE_MEMORY.
#
# The fix is at the source: the Workgroup Block variable is emitted with an
# OpConstantNull initializer (shaderZeroInitializeWorkgroupMemory, Vulkan 1.3
# core), so every kernel starts with zeroed shared memory.

using Test, Lava, KernelAbstractions
import AcceleratedKernels as AK
const KA = KernelAbstractions

@kernel function _read_unwritten_shared!(out)
    i = @index(Local, Linear)
    scratch = @localmem Float32 (64,)
    # Write only the first half; the second half is never stored to.
    if i <= 32
        @inbounds scratch[i] = 1.0f0
    end
    @synchronize
    @inbounds out[i] = scratch[i]
end

@testset "workgroup memory is zero-initialized" begin
    be = LavaBackend()
    out = KA.allocate(be, Float32, 64)
    fill!(out, -1.0f0)
    _read_unwritten_shared!(be, 64)(out; ndrange = 64)
    KA.synchronize(be)
    got = Array(out)

    @test all(got[1:32] .== 1.0f0)     # written half
    @test all(got[33:64] .== 0.0f0)    # never written -> must read as zero
end

# The observable symptom: AK's block-level merge for len < 2*block_size (256).
@testset "merge_sort_by_key! is correct below 2*block_size" begin
    for n in (7, 100, 511, 1000)
        h = rand(UInt32, n)
        k = Mantle.LavaArray(copy(h))
        v = Mantle.LavaArray(collect(Int32(1):Int32(n)))
        AK.merge_sort_by_key!(k, v)
        Mantle.flush!(Mantle.Device())
        @test Array(k) == sort(h)
        @test h[Array(v)] == sort(h)   # values permuted consistently with keys
    end
end

# sortperm must terminate (no mutual recursion) and not blow up the pool.
@testset "sortperm terminates and does not balloon the pool" begin
    n = 100_000
    before = Mantle.gpu_live_bytes()
    h = rand(UInt32, n)
    v = Mantle.LavaArray(copy(h))
    ix = Mantle.LavaArray(collect(Int32(1):Int32(n)))
    AK.sortperm!(ix, v)
    Mantle.flush!(Mantle.Device())
    @test h[Array(ix)] == sort(h)
    # Recursion used to grow the pool by tens of GB before dying.
    @test Mantle.gpu_live_bytes() - before < 256_000_000
end
