using Test
using Raycore
using Raycore: MultiTypeSet, SetKey, TextureRef, with_index, deref
using Lava, Mantle
using Adapt
using KernelAbstractions
using GPUArraysCore: @allowscalar

# Regression suite for the MultiTypeSet surgical-per-mutation design
# (2026-04-20).  MultiTypeSet has no dirty flag; every mutator keeps `static`
# consistent with CPU state in one call via `resize!` + `@allowscalar setindex!`
# on the affected slot.  These tests pin down:
#
#   * Consistency invariant: after any `push!` / `update!` / `store_texture` /
#     `copyto_texture!`, `dhv.static.data[i]` / `.textures[i]` match what a
#     full re-adapt of the CPU state would produce, with NO intermediate
#     rebuild / sync call.
#   * Identity / capacity invariants: repeated pushes to an existing type slot
#     reuse the same LavaArray (same Julia identity, same VkBuffer) until the
#     buffer's pool-rounded capacity is exceeded.
#   * `copyto_texture!` same-size / grow-within-capacity / grow-beyond-capacity:
#     only the grow-beyond-capacity branch changes the device pointer, and
#     when it does, just one isbits slot is refreshed.
#   * `update!(mts, key, new)`: single-element GPU write leaves other slots
#     untouched.

const TINY_BACKEND = LavaBackend()

struct TinyMat
    x::Float32
    y::Float32
end

struct TinyOther
    v::Int32
end

@testset "MultiTypeSet surgical-per-mutation invariants" begin

@testset "no dirty flag, no rebuild_static! API" begin
    @test !(:dirty in fieldnames(MultiTypeSet))
    @test !(:texture_isbits in fieldnames(MultiTypeSet))
    @test !isdefined(Raycore, :rebuild_static!)
end

@testset "push! keeps static consistent with no intermediate call" begin
    mts = MultiTypeSet(TINY_BACKEND)
    @test length(mts.static.data) == 0

    k1 = push!(mts, TinyMat(1f0, 2f0))
    # Existing GPU slot must reflect the new element with no sync/rebuild call.
    @test length(mts.static.data) == 1
    @test length(mts.static.data[1]) == 1
    @test @allowscalar(mts.static.data[1][1]) === TinyMat(1f0, 2f0)
    @test k1 == SetKey(UInt32(1), UInt32(1))

    k2 = push!(mts, TinyMat(3f0, 4f0))
    # Same type → same slot; GPU array grew by one element in place.
    @test length(mts.static.data) == 1
    @test length(mts.static.data[1]) == 2
    @test @allowscalar(mts.static.data[1][2]) === TinyMat(3f0, 4f0)
    @test k2 == SetKey(UInt32(1), UInt32(2))

    k3 = push!(mts, TinyOther(Int32(7)))
    # New type → tuple shape grew by one slot.
    @test length(mts.static.data) == 2
    @test length(mts.static.data[2]) == 1
    @test @allowscalar(mts.static.data[2][1]) === TinyOther(Int32(7))
    @test k3 == SetKey(UInt32(2), UInt32(1))
end

@testset "push! reuses same LavaArray identity for an existing type slot" begin
    mts = MultiTypeSet(TINY_BACKEND)
    push!(mts, TinyMat(1f0, 2f0))
    arr_after_first = mts.static.data[1]

    # Push several more into the same type slot — Julia identity must not change.
    for i in 2:6
        push!(mts, TinyMat(Float32(i), Float32(i+1)))
        @test mts.static.data[1] === arr_after_first
    end
    @test length(mts.static.data[1]) == 6
end

@testset "push! is capacity-aware — VkBuffer reused until capacity exceeded" begin
    mts = MultiTypeSet(TINY_BACKEND)
    push!(mts, TinyMat(0f0, 0f0))
    arr = mts.static.data[1]
    initial_buf_address = arr.buf[].address
    initial_capacity = arr.buf[].size  # bytes; pool-rounded, usually ≥ 16

    # Push up to what we expect still fits in the pool's 16-byte minimum —
    # TinyMat is 8 bytes, so 1 push after the first (2 elements total) still fits.
    push!(mts, TinyMat(1f0, 1f0))
    @test arr.buf[].address == initial_buf_address  # no realloc within capacity
    @test arr.buf[].size == initial_capacity

    # Now force a grow far past initial capacity.
    for i in 1:256
        push!(mts, TinyMat(Float32(i), 0f0))
    end
    @test length(mts.static.data[1]) == 258
    # After growth the backing VkBuffer is a new allocation (address changed),
    # but the wrapping LavaArray Julia object is the same.
    @test mts.static.data[1] === arr
    @test arr.buf[].address != initial_buf_address

    # The surgical setindex in push! actually wrote the new elements — readback
    # confirms values (not uninitialised memory).
    hostcopy = Array(mts.static.data[1])
    @test hostcopy[1] === TinyMat(0f0, 0f0)
    @test hostcopy[2] === TinyMat(1f0, 1f0)
    @test hostcopy[3] === TinyMat(1f0, 0f0)   # i=1 from the grow loop
    @test hostcopy[end] === TinyMat(256f0, 0f0)
end

@testset "update!(mts, key, new) is a single-element write" begin
    mts = MultiTypeSet(TINY_BACKEND)
    k1 = push!(mts, TinyMat(1f0, 1f0))
    k2 = push!(mts, TinyMat(2f0, 2f0))
    k3 = push!(mts, TinyMat(3f0, 3f0))

    # Modify the middle element.
    Raycore.update!(mts, k2, TinyMat(99f0, 99f0))
    host = Array(mts.static.data[1])
    @test host[1] === TinyMat(1f0, 1f0)
    @test host[2] === TinyMat(99f0, 99f0)
    @test host[3] === TinyMat(3f0, 3f0)
    # CPU side updated too (authoritative data vector).
    @test mts.data_vectors[TinyMat][2] === TinyMat(99f0, 99f0)
end

@testset "store_texture grows the isbits slot surgically" begin
    mts = MultiTypeSet(TINY_BACKEND)
    ref1 = Raycore.store_texture(mts, Float32[10, 20, 30])
    @test length(mts.static.textures) == 1
    @test length(mts.static.textures[1]) == 1
    @test ref1.idx == 1

    # Same AT → same slot.
    ref2 = Raycore.store_texture(mts, Float32[100, 200])
    @test length(mts.static.textures) == 1
    @test length(mts.static.textures[1]) == 2
    @test ref2.idx == 2

    # Both isbits entries should dereference to the correct underlying data.
    # We do this via the @allowscalar scalar reads of the LavaDeviceArray handles.
    iptr1 = @allowscalar mts.static.textures[1][1]
    iptr2 = @allowscalar mts.static.textures[1][2]
    @test size(iptr1) == (3,)
    @test size(iptr2) == (2,)

    # Different AT (Float32[] 1-D vs 2-D) → new slot.
    ref3 = Raycore.store_texture(mts, Float32[1f0 2f0; 3f0 4f0])
    @test length(mts.static.textures) == 2
    @test length(mts.static.textures[2]) == 1
    @test ref3.idx == 1
end

@testset "copyto_texture! same-size: pointer unchanged, data updated" begin
    mts = MultiTypeSet(TINY_BACKEND)
    ref = Raycore.store_texture(mts, Float32[1, 2, 3, 4])
    slot = mts.static.textures[1]
    old_isbits = @allowscalar slot[ref.idx]
    inner_arr = mts.texture_gpu_arrays[1]
    old_buf_addr = inner_arr.buf[].address

    Raycore.copyto_texture!(mts, ref, Float32[5, 6, 7, 8])

    # Same size → no realloc → same VkBuffer address → same isbits handle.
    @test inner_arr.buf[].address == old_buf_addr
    new_isbits = @allowscalar slot[ref.idx]
    @test new_isbits === old_isbits
    # And the data was actually overwritten.
    @test Array(inner_arr) == Float32[5, 6, 7, 8]
end

@testset "copyto_texture! grow-beyond-capacity: ptr moves, only affected isbits slot refreshed" begin
    mts = MultiTypeSet(TINY_BACKEND)
    refA = Raycore.store_texture(mts, Float32[1])
    refB = Raycore.store_texture(mts, Float32[10, 11])
    # Give B's slot room by pushing more small textures around it.
    slot = mts.static.textures[1]
    @test length(slot) == 2

    isbitsA_before = @allowscalar slot[refA.idx]
    isbitsB_before = @allowscalar slot[refB.idx]

    # Force B to grow far past its pool-rounded capacity.
    big = collect(Float32, 1:10_000)
    Raycore.copyto_texture!(mts, refB, big)

    isbitsA_after = @allowscalar slot[refA.idx]
    isbitsB_after = @allowscalar slot[refB.idx]

    # A wasn't touched — its slot must be untouched.
    @test isbitsA_after === isbitsA_before
    # B's pointer moved — its slot was refreshed.
    @test isbitsB_after !== isbitsB_before
    # B's isbits handle now points to a 10_000-element array.
    @test size(isbitsB_after) == (10_000,)
    # And the content is correct.
    @test Array(mts.texture_gpu_arrays[2]) == big
end

@testset "Adapt.adapt_structure reads static directly — no sync needed" begin
    mts = MultiTypeSet(TINY_BACKEND)
    push!(mts, TinyMat(1f0, 2f0))
    push!(mts, TinyMat(3f0, 4f0))
    # static.data[1] is already the LavaArray — Adapt.adapt(LavaBackend, ...)
    # recurses via StaticMultiTypeSet's adapt_structure and hands each tuple
    # entry through the LavaBackend adaptor (LavaArray → LavaArray identity).
    adapted = Adapt.adapt(TINY_BACKEND, mts)
    @test length(adapted.data) == 1
    @test adapted.data[1] isa LavaArray{TinyMat,1}
    @test length(adapted.data[1]) == 2
end

end  # @testset
