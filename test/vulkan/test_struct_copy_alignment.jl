# Whole-struct copy must not claim an alignment the address does not have.
#
#     struct S3I16; a::Int16; b::Int16; c::Int16; end   # sizeof 6, offsets 0/2/4
#     dst .= src        # b == 0 for even elements
#     d[i] = s[i]       # same, in a hand-written kernel
#
# NOT broadcast-specific: any whole-struct copy went through the same path.
#
# Cause (fixed): `emit_memcpy!` in compiler/spirv/emit.jl copied 4 bytes at a
# time and stamped `Aligned 4` on every load and store. These copies are ARRAY
# ELEMENTS, so consecutive elements sit at multiples of the element size — for a
# 6-byte struct element i is at byte 6(i-1), 4-aligned only for odd i. Every even
# element therefore declared an alignment its address did not have, and the 32-bit
# access dropped its high half: field `a` (bytes 0..1) survived, field `b`
# (bytes 2..3) came back zero. The intrinsic being lowered declares `align 1`, so
# the emitter was contradicting its own input. It now derives the chunk width from
# the largest power of two dividing the copy size and emits `Aligned` to match —
# three `OpLoad %ushort ... Aligned 2` here instead of one over-aligned `%uint`.
#
# Two hypotheses were eliminated first, recorded so they are not re-tried:
#   * `lower_memcpy!` (compiler/compilation.jl) — the obvious suspect, and its
#     4-byte fallback IS over-aligned in the same way, but it never runs for this
#     copy: LLVM emits `llvm.memmove`, and re-applying the fix there produced
#     byte-identical SPIR-V.
#   * `try_decomposed_struct_store!` and `is_padding_gep` — both guard on the base
#     being an alloca, so neither applies to a store through a device pointer.
#
# The sizes below are the boundary: 3 (odd, already byte-copied), 4 and 8 (element
# size a multiple of 4, so genuinely aligned), 12 and 12-padded. Any change to the
# chunk derivation must keep all of them correct.

using Test, Lava, KernelAbstractions
const KA = KernelAbstractions

struct CopyS6;  a::Int16; b::Int16; c::Int16; end                    # 6  — the bug
struct CopyS2;  a::Int16; b::Int16; end                              # 4
struct CopyS4;  a::Int16; b::Int16; c::Int16; d::Int16; end          # 8
struct CopyS3B; a::Int8;  b::Int8;  c::Int8; end                     # 3
struct CopyS12; a::Float32; b::Float32; c::Float32; end              # 12
struct CopyS10; a::Int16; b::Int32; c::Int32; end                    # 12 (padded)

@kernel function aggregate_copy!(d, @Const(s))
    i = @index(Global)
    @inbounds d[i] = s[i]          # whole-struct load then store -> memcpy
end

function check_copy(::Type{S}, mk) where {S}
    n = 16
    src_data = [mk(i) for i in 1:n]
    src = MVE.LavaArray(src_data)

    bcast = MVE.LavaArray{S}(undef, n)
    bcast .= src

    kern = MVE.LavaArray{S}(undef, n)
    aggregate_copy!(LavaBackend(), 64)(kern, src; ndrange = n)
    KA.synchronize(LavaBackend())

    return Array(bcast) == src_data, Array(kern) == src_data
end

@testset "whole-struct copy preserves every field" begin
    # The size that broke: 6 bytes, so even elements are only 2-aligned.
    @test sizeof(CopyS6) == 6
    b, k = check_copy(CopyS6, i -> CopyS6(Int16(i), Int16(i * 3), Int16(i * 7)))
    @test b        # broadcast
    @test k        # hand-written kernel — same emit path

    # Sizes that ARE correct. They pin the boundary of the fault, and a fix that
    # changes how copies are chunked must not regress them.
    for (S, mk) in (
            (CopyS2,  i -> CopyS2(Int16(i), Int16(i * 3))),
            (CopyS4,  i -> CopyS4(Int16(i), Int16(i * 3), Int16(i * 7), Int16(i * 11))),
            (CopyS3B, i -> CopyS3B(Int8(i), Int8(i * 3), Int8(i * 7))),
            (CopyS12, i -> CopyS12(Float32(i), Float32(i * 3), Float32(i * 7))),
            (CopyS10, i -> CopyS10(Int16(i), Int32(i * 3), Int32(i * 7))),
        )
        b, k = check_copy(S, mk)
        @test b
        @test k
    end

    # Paths that never used the memcpy fallback; kept so a future change to it
    # cannot quietly start routing them through a broken one.
    n = 16
    sd = [CopyS6(Int16(i), Int16(i * 3), Int16(i * 7)) for i in 1:n]
    src = MVE.LavaArray(sd)
    d = MVE.LavaArray{CopyS6}(undef, n)
    copyto!(d, src)
    MVE.vk_flush!(MVE.vk_context())
    @test Array(d) == sd
end
