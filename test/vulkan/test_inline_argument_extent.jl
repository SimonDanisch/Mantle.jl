using Test, Mantle

struct InlineExtentOwner <: Mantle.Closed end
struct InlineExtentPayload
    data::NTuple{3,Float32}
end

# Regression ported from AScanner's independently qualified extent fix. Packing
# starts after fixed argument slots, which need not end on an eight-byte edge.
@testset "inline argument extent includes fixed-block alignment" begin
    @test Mantle.compute_inline_extra_from_byval([16,12],4)==32
    @test Mantle.compute_inline_extra_from_byval([0,16,0,12,0],60)==32
    @test Mantle.compute_inline_extra_from_byval(Int[],4)==0
    for base in 0:64, sizes in (Int[],[0],[4],[12],[16,12],[0,24,0,312,16,12,0])
        cursor=base
        for size in sizes
            size>0 || continue
            while cursor%8!=0
                cursor+=1
            end
            cursor+=size
        end
        @test base+Mantle.compute_inline_extra_from_byval(sizes,base)==cursor
        @test Mantle.compute_inline_extra_from_byval(sizes)==Mantle.compute_inline_extra_from_byval(sizes,0)
    end
    # Real generated packer + canary, without a GPU device or GPU allocations.
    base=12;sizes=[12,0]
    allocated=base+Mantle.compute_inline_extra_from_byval(sizes,base)
    bytes=fill(UInt8(0xa5),allocated+16)
    GC.@preserve bytes begin
        ptr=pointer(bytes)
        Mantle.pack_args_direct!(InlineExtentOwner(),ptr,UInt64(ptr),[0,8],base,sizes,
            (InlineExtentPayload((1f0,2f0,3f0)),7f0))
        @test unsafe_load(Ptr{UInt64}(ptr))==UInt64(ptr)+16
        @test unsafe_load(Ptr{Float32}(ptr+8))==7f0
        @test unsafe_load(Ptr{Float32}(ptr+24))==3f0
    end
    @test all(==(0xa5),@view bytes[allocated+1:end])
end
