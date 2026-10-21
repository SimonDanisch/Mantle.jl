using Test, Mantle
const VD = Mantle.VideoDecode

@testset "shared Annex-B splitting" begin
    a, b = UInt8[0x65,0x88,0x80], UInt8[0x41,0xaa,0x80]
    bytes = vcat(UInt8[0,0,0,1], a, UInt8[0,0,1], b, UInt8[0,0])
    expected = [(5,3,a), (1,2,b)]
    @test VD.split_nals(bytes) == expected
    @test VD.split_nals(@view bytes[:]) == expected
    @test eltype(VD.split_nals(UInt8[])) === Tuple{Int,Int,Vector{UInt8}}
    hevc = UInt8[0x26,0x01,0x80]
    @test VD.split_nals_h265(vcat(UInt8[0,0,1],hevc)) == [(19,hevc)]
    @test eltype(VD.split_nals_h265(UInt8[])) === Tuple{Int,Vector{UInt8}}
end

@testset "direct bitstream packing preserves bytes and bounds" begin
    for n in 0:16
        slices = [UInt8[(i*j)%256 for j in 1:(i*17)] for i in 1:n]
        expected = reduce(vcat,(vcat(UInt8[0,0,1],s) for s in slices);init=UInt8[])
        guard = 8
        bytes = fill(UInt8(0xa5),length(expected)+2guard+32)
        GC.@preserve bytes slices begin
            offsets, range = VD.uploadbitstream!(pointer(bytes)+guard,slices,32)
            @test bytes[guard+1:guard+length(expected)] == expected
            @test all(==(0xa5),bytes[1:guard])
            @test all(==(0xa5),bytes[guard+length(expected)+1:end])
            @test offsets == UInt32[sum(length(s)+3 for s in slices[1:i-1];init=0) for i in 1:n]
            @test range == cld(length(expected),32)*32
        end
    end
end
