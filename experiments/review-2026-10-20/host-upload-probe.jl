# include, before=HostUploadProbe.baseline(), then
# Base.invokelatest(HostUploadProbe.measure, ctx, before; unified=false/true).
module HostUploadProbe
using Mantle, Test

function baseline()
    root = normpath(joinpath(@__DIR__, "../.."))
    source = read(`git -C $root show f80dc6f:src/vulkan/array/lavaarray.jl`,String)
    method = match(r"function upload!\(dst::LavaArray\{T\}.*?\nend"s,source).match
    method = replace(method,"function upload!("=>"function probe_before_array_upload!(")
    Base.include_string(Mantle,method)
    return Mantle.probe_before_array_upload!
end

function measure(ctx,before;unified=false)
    source = reshape(Float32.(1:1048576),1024,1024)
    dst = Mantle.LavaArray{Float32}(undef,(length(source),);bq=ctx.default_bq,unified)
    try
        for _ in 1:5
            before(dst,source); Mantle.upload!(dst,source)
        end
        oldbytes = @allocated before(dst,source)
        newbytes = @allocated Mantle.upload!(dst,source)
        @test oldbytes >= 2sizeof(source)
        @test newbytes < sizeof(source)÷100
        @test Array(dst) == vec(source)
        mapped = dst.buf[].mapped_ptr != C_NULL
        @test !unified || mapped
        row = (;device=ctx.device_name,unified,mapped,source_bytes=sizeof(source),
               before_host_bytes=oldbytes,after_host_bytes=newbytes)
        println(row)
        return row
    finally
        Mantle.unsafe_free!(dst)
        Mantle.waitidle(Mantle.lavadevice(ctx)); Mantle.drain!(ctx.default_bq)
        @test isempty(Mantle.validationmessages!(Mantle.lavadevice(ctx)))
    end
end
end
