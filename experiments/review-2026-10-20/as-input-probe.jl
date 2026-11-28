# baseline(), then Base.invokelatest(ASInputProbe.measure, ctx).
module ASInputProbe
using Mantle, Test

function baseline()
    root = normpath(joinpath(@__DIR__, "../.."))
    source = read(`git -C $root show f62b33d:src/vulkan/raytracing/acceleration.jl`,String)
    for (old,new) in (("refit_blas!","probe_old_refit_blas!"),
                      ("pack_as_instance!","probe_old_pack_as_instance!"),
                      ("pack_geometry!","probe_old_pack_geometry!"))
        # Select the removed keyword geometry encoder, not its typed overloads.
        pattern = old == "pack_geometry!" ?
            r"function pack_geometry!\(buf::Vector\{UInt8\}, offset::Int;.*?\nend"s :
            Regex("function " * old * raw"\(.*?\nend", "s")
        method = match(pattern,source).match
        Base.include_string(Mantle,replace(method,"function $old("=>"function $new("))
    end
end

function bytes()
    @testset "native instance encodings preserve bytes" begin
        for offset in (0,16), addr in UInt64[0,0x123456789abcdef0,typemax(UInt64)], flags in UInt32[0,1,2]
            old = fill(UInt8(0xa5),128); new = copy(old)
            Mantle.probe_old_pack_geometry!(old,offset;geometry_type=:instances,instance_addr=addr,geo_flags=flags)
            Mantle.pack_geometry!(new,offset,Mantle.VulkanInstancesGeometry(addr);geo_flags=flags)
            @test old == new
        end
        transforms = (nothing,Tuple(Float32.(1:12)))
        for transform in transforms, id in UInt32[0,0x12345678], mask in UInt8[0,0x7f,0xff]
            old = zeros(UInt8,64)
            Mantle.probe_old_pack_as_instance!(old,0,UInt64(0x12345678);transform,custom_index=id,mask)
            record = Mantle.VulkanInstanceRecord(transform === nothing ? Mantle.identity_transform() :
                                                 Mantle.Mat3x4f(transform),UInt64(0x12345678);custom_index=id,mask)
            @test old == reinterpret(UInt8,[record])
        end
    end
end

function measure(ctx)
    bq = ctx.default_bq
    vertices = repeat([(0f0,0f0,0f0),(1f0,0f0,0f0),(0f0,1f0,0f0)],16384)
    indices = collect(UInt32,0:length(vertices)-1)
    blas = Mantle.build_accel!(bq) do build
        Mantle.build_blas(build,vertices,indices;allow_update=true)
    end
    run(f) = Mantle.build_accel!(bq) do build
        f(build,blas,vertices)
    end
    try
        old = Mantle.probe_old_refit_blas!
        new = Mantle.refit_blas!
        for _ in 1:5
            run(old); run(new)
        end
        before = minimum([@allocated(run(old)) for _ in 1:5])
        after = minimum([@allocated(run(new)) for _ in 1:5])
        @test before-after >= sizeof(vertices)
        @test Array(blas.vertex_arr) == reinterpret(UInt8,vertices)
        row=(;device=ctx.device_name,source_bytes=sizeof(vertices),before_host_bytes=before,after_host_bytes=after)
        println(row)
        return row
    finally
        Mantle.unsafe_free!(blas)
        Mantle.waitidle(Mantle.lavadevice(ctx)); Mantle.drain!(bq)
        @test isempty(Mantle.validationmessages!(Mantle.lavadevice(ctx)))
    end
end
end
