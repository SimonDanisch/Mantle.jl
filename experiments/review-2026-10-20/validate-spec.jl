# Syntax and macro/lowering validation only. Does not import Mantle or run cases.
module SpecValidation
using Test
import KernelInterface as KI
const root = normpath(joinpath(@__DIR__, "../.."))
function install_case(ex)
    ex isa Expr || return
    if ex.head === :macro && ex.args[1] isa Expr && ex.args[1].args[1] === :case
        Core.eval(@__MODULE__, ex)
    else
        foreach(install_case, ex.args)
    end
end
install_case(Meta.parseall(read(joinpath(root,"test/suite/helpers.jl"),String)))
function errors(ex)
    ex isa Expr || return false
    ex.head in (:error,:incomplete) && return true
    any(errors, ex.args)
end
function validate()
    suite = joinpath(root,"test/suite")
    areas = sort(filter(endswith(".jl"),readdir(joinpath(suite,"areas");join=true)))
    sources = vcat(areas,
        [joinpath(suite,f) for f in ("runsuite.jl","helpers.jl","faultdevice.jl")],
        [joinpath(root,"docs",f) for f in ("internals.jl","examples.jl","resizing-and-raytracing.jl")],
        filter(endswith(".jl"),readdir(joinpath(root,"test/design");join=true)))
    for path in sources
        ast = Meta.parseall(read(path,String);filename=path)
        errors(ast) && error("Parse failure: $path")
    end
    ids = String[]
    for path in areas
        source = read(path,String)
        append!(ids, [m.captures[1] for m in eachmatch(r"@case\s+\"([^\"]+)\"",source)])
        ast = Meta.parseall(source;filename=path)
        expanded = Base.invokelatest(() -> macroexpand(@__MODULE__,ast))
        lowered = Meta.lower(@__MODULE__,expanded)
        errors(lowered) && error("Lowering failure: $path")
    end
    length(unique(ids)) == length(ids) || error("Duplicate case IDs")
    println("Parsed ",length(sources)," suite/design/doc Julia files; expanded/lowered ",
        length(areas)," areas with ",length(ids)," unique cases. No GPU bodies executed.")
end
end
SpecValidation.validate()
