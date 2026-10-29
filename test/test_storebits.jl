using Test, Mantle

struct PackedPointerDescriptor
    ptr::Ptr{Float32}
    len::Int32
end
struct PackedPaddedValue
    weight::Float32
    index::Int64
end

function put_packed_bits!(bytes, value)
    GC.@preserve bytes Mantle.storebits!(pointer(bytes), value)
    return nothing
end

@testset "nested packed values preserve padding without allocation" begin
    descriptor = PackedPointerDescriptor(Ptr{Float32}(UInt(0x1234)), 23)
    padded = PackedPaddedValue(0.25f0, -4)
    # Specify the four occupied byte ranges independently of the generated walk.
    expected = fill(0xcd, 32)
    for (offset, value) in ((0, descriptor.ptr), (8, descriptor.len),
                            (16, padded.weight), (24, padded.index))
        copyto!(expected, offset + 1, reinterpret(UInt8, [value]), 1, sizeof(value))
    end
    for value in ((descriptor, padded), (Val(2), (descriptor, padded), ()))
        bytes = fill(0xcd, sizeof(value))
        put_packed_bits!(bytes, value)
        @test bytes == expected
        @test (@allocated put_packed_bits!(bytes, value)) == 0
        put_packed_bits!(bytes, value)
        @test bytes == expected
    end
    empty = UInt8[]
    @test put_packed_bits!(empty, ()) === nothing
    @test isempty(empty)
    compact = (0.25f0, Int32(23))
    bytes = zeros(UInt8, sizeof(compact))
    put_packed_bits!(bytes, compact)
    @test bytes == [reinterpret(UInt8, [compact[1]]); reinterpret(UInt8, [compact[2]])]
end
