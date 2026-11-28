# Infer backend types without constructing a device or probing hardware.
function precompile_device_types()
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
precompile_constructor_signatures(::Type{<:Device}) = ()

# Optional device-free kernel ABI descriptions for consumer precompile workloads.
# The backend owns its array wrappers, iteration context and interpreter.
kernel_precompile_abi(::Type{<:Device}, group::Int) = nothing

"""
    precompile_kernelinterpreter(D, kernel, tt, group) -> interpreter

The interpreter a device of type `D` infers `kernel` with, for argument types `tt` and a
`group`-thread workgroup: what a consumer's device-free precompile workload, built from
[`kernel_precompile_abi`](@ref), infers through. A device type without such an ABI
has no interpreter that works without a device.
"""
precompile_kernelinterpreter(::Type{D}, kernel, tt, group::Int) where {D<:Device} =
    throw(ArgumentError("$D describes no device-free kernel ABI (`kernel_precompile_abi` " *
                        "is `nothing`), so no kernel is inferred for it without a device"))

function precompile_statements()
    signatures = Any[]
    for device in precompile_device_types()
        append!(signatures, precompile_constructor_signatures(device))
        for phase in (Place, Pipelines, Barriers)
            push!(signatures, Tuple{typeof(run!), phase, Compile{device}})
        end
    end
    return Base.count(Base.precompile, signatures), length(signatures)
end

PrecompileTools.@setup_workload begin
    PrecompileTools.@compile_workload begin
        precompile_statements()
    end
end
