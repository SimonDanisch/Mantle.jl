using Test, Mantle, Metal

@testset "Metal: upload host array wrappers" begin
    dev = Mantle.Device(Mantle.MetalAPI())
    data = reshape(Float32.(1:24), 4, 6)
    bytes = collect(reinterpret(UInt8, vec(data)))
    wrapped = reshape(reinterpret(Float32, bytes), size(data))
    for source in (
            data, view(data, 1:2:4, 2:2:6), transpose(data),
            PermutedDimsArray(data, (2, 1)), wrapped,
            reshape(view(vec(data), 1:12), 3, 4), view(data, 1:0, :),
        )
        uploaded = Mantle.devicearray(dev, source)
        try
            @test size(uploaded) == size(source)
            @test Array(uploaded) == Array(source)
        finally
            Mantle.unsafe_free!(uploaded)
        end
    end

    indices = UInt32[1, 2, 3, 4, 5, 6]
    uploaded = Mantle.indexbuffer(dev, view(indices, 1:2:5))
    try
        @test Array(uploaded) == UInt32[1, 3, 5]
    finally
        Mantle.unsafe_free!(uploaded)
    end

    # Keep native device copies, including a source whose storage mode differs
    # from the pool's. Uploading it must produce an independent array.
    expected = Float32[2, 3, 5, 7]
    source = Metal.MtlArray{Float32,1,Metal.PrivateStorage}(expected)
    copied = nothing
    try
        copied = Mantle.devicearray(dev, source)
        fill!(source, 0.0f0)
        Metal.synchronize()
        @test Array(copied) == expected
        @test all(iszero, Array(source))
    finally
        copied === nothing || Mantle.unsafe_free!(copied)
        Mantle.unsafe_free!(source)
    end
end
