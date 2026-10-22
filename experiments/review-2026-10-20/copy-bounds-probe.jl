# baseline(), then Base.invokelatest(CopyBoundsProbe.measure, ctx; unified).
module CopyBoundsProbe
using Mantle, Test

function baseline()
    root=normpath(joinpath(@__DIR__,"../.."))
    source=read(`git -C $root show 5cdedb7:src/vulkan/array/gpuarrays.jl`,String)
    patterns=(r"function Base.copyto!\(dest::LavaArray\{T\}, doffs::Integer,\n\s+src::Array\{T\}.*?\nend"s,
              r"function Base.copyto!\(dest::LavaArray\{T\},doffs::Integer,\n\s+src::HostReinterpretArray\{T\}.*?\nend"s)
    for pattern in patterns
        method=match(pattern,source).match
        Base.include_string(Mantle,replace(method,"function Base.copyto!("=>"function probe_old_host_copy!("))
    end
end

function measure(ctx;unified=false)
    bq=ctx.default_bq
    source=UInt32.(1:65536)
    words=reinterpret(UInt32,collect(reinterpret(UInt8,source)))
    dest=Mantle.LavaArray{UInt32}(undef,size(source);bq,unified)
    settle()=(Mantle.waitidle(Mantle.lavadevice(ctx));Mantle.drain!(bq))
    function allocated(f,src)
        settle()
        bytes=@allocated f(dest,1,src,1,length(src))
        settle()
        return bytes
    end
    rows=[]
    try
        for src in (source,words)
            old=Mantle.probe_old_host_copy!
            for _ in 1:5
                allocated(old,src);allocated(copyto!,src)
            end
            before=minimum(allocated(old,src) for _ in 1:10)
            after=minimum(allocated(copyto!,src) for _ in 1:10)
            @test after<=before
            @test Array(dest)==source
            row=(;device=ctx.device_name,unified,source=src isa Array ? "array" : "words",
                  source_bytes=sizeof(source),before_host_bytes=before,after_host_bytes=after)
            println(row);push!(rows,row)
        end
        return rows
    finally
        Mantle.unsafe_free!(dest);settle()
        @test isempty(Mantle.validationmessages!(Mantle.lavadevice(ctx)))
    end
end
end
