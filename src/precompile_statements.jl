# Infer backend types without constructing a device or probing hardware.
function _precompile_device_types()
    result = DataType[]
    for method in methods(makedevice)
        signature = Base.unwrap_unionall(method.sig)
        length(signature.parameters) == 4 || continue
        api = signature.parameters[2]
        api isa DataType && isconcretetype(api) && api <: Backend || continue
        device = Core.Compiler.return_type(makedevice, Tuple{api, Nothing, DebugConfig})
        device <: Device && isconcretetype(device) && push!(result, device)
    end
    return unique(result)
end

# Backends with an inference barrier around their native constructor supply its
# signature explicitly. The constructor is compiled, never called.
_precompile_constructor_signatures(::Type{<:Device}) = ()

function _precompile_statements()
    signatures = Any[]
    for device in _precompile_device_types()
        append!(signatures, _precompile_constructor_signatures(device))
        for phase in (Place, Pipelines, Barriers)
            push!(signatures, Tuple{typeof(run!), phase, Compile{device}})
        end
    end
    return Base.count(Base.precompile, signatures), length(signatures)
end

PrecompileTools.@setup_workload begin
    PrecompileTools.@compile_workload begin
        _precompile_statements()
    end
end
