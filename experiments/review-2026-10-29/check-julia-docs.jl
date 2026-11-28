# Static Julia syntax/name check only. Run through the persistent Julia worker.
# Does not import Mantle or assert that target methods/native features exist.
using Test: @testset, @test, @test_throws
let root = normpath(joinpath(@__DIR__, "..", ".."))
    function validate_expr(ex)
        ex isa Expr || return nothing
        ex.head in (:error, :incomplete) && error("Invalid Julia pseudocode: $(repr(ex))")
        foreach(validate_expr, ex.args)
        nothing
    end

    function inventory!(calls, definitions, ex)
        ex isa Expr || return nothing
        if ex.head === :function && ex.args[1] isa Expr &&
           ex.args[1].head === :call && ex.args[1].args[1] isa Symbol
            push!(definitions, ex.args[1].args[1])
        end
        if ex.head === :call && ex.args[1] isa Symbol
            push!(calls, ex.args[1])
        end
        foreach(arg -> inventory!(calls, definitions, arg), ex.args)
        nothing
    end

    function bindings!(kinds, ex)
        ex isa Expr || return nothing
        if ex.head === :(=) && ex.args[1] isa Symbol && ex.args[2] isa Expr
            rhs = ex.args[2]
            if rhs.head === :call
                if rhs.args[1] in (:Device, :Graph, :Group)
                    kinds[ex.args[1]] = rhs.args[1]
                elseif rhs.args[1] === :call! && length(rhs.args) >= 3 && rhs.args[3] === :Array
                    kinds[ex.args[1]] = :RecordedFuture
                elseif rhs.args[1] === :Readback && length(rhs.args) == 2 &&
                       get(kinds, rhs.args[2], nothing) === :RecordedFuture
                    kinds[ex.args[1]] = :RecordedFuture
                end
            end
        end
        foreach(arg -> bindings!(kinds, arg), ex.args)
        nothing
    end

    function check_routing(ex, kinds)
        ex isa Expr || return nothing
        if ex.head === :call && length(ex.args) >= 2
            f, first_arg = ex.args[1], ex.args[2]
            first_kind = get(kinds, first_arg, nothing)
            if f in (:mul!, :broadcast!, :copyto!, :fill!, :map!, :compact!, :rand!)
                first_kind in (:Device, :Graph, :Group) &&
                    error("Public eager operation with an extra target: $(repr(ex))")
            elseif f === :adapt
                first_kind in (:Graph, :Group) && error("Graph/Group array adaptation: $(repr(ex))")
                if first_arg isa Expr && first_arg.head === :call && first_arg.args[1] === :Device
                    error("Device(target) adaptation ceremony in public examples: $(repr(ex))")
                end
            elseif f === :Readback
                length(ex.args) == 2 && first_kind === :RecordedFuture ||
                    error("Array readback must be call!(target,Array,a); Readback only re-requests a recorded future")
            elseif f === :call!
                first_kind === :Device && error("Standard eager methods do not need call!(dev,...)")
            end
        end
        foreach(arg -> check_routing(arg, kinds), ex.args)
        nothing
    end

    @testset "Static routing guard controls" begin
        kinds = Dict{Any,Symbol}(:g => :Graph, :step => :Group,
                                 :dev => :Device, :previous => :RecordedFuture)
        for text in ("adapt(step, a)", "adapt(Device(step), a)",
                     "mul!(dev, C, A, B)", "mul!(step, C, A, B)",
                     "Readback(a)", "Readback(g, a)", "call!(dev, mul!, C, A, B)")
            @test_throws ErrorException check_routing(Meta.parse(text), kinds)
        end
        for text in ("mul!(C, A, B)", "call!(step, mul!, C, A, B)",
                     "call!(g, Array, a)", "Readback(previous)")
            @test check_routing(Meta.parse(text), kinds) === nothing
        end
    end

    api = first(split(read(joinpath(root, "docs", "api.md"), String), "## 3. Core types"))
    documented = Set(Symbol(m.captures[1]) for m in
        eachmatch(r"\b([A-Za-z_][A-Za-z_0-9]*!?)\(", api))
    println("Julia ", VERSION)
    for name in ("internals.jl", "examples.jl", "resizing-and-raytracing.jl")
        path = joinpath(root, "docs", name)
        parsed = Meta.parseall(read(path, String); filename=path)
        validate_expr(parsed)
        println("Parsed ", name)
        if name == "examples.jl"
            calls, definitions = Set{Symbol}(), Set{Symbol}()
            inventory!(calls, definitions, parsed)
            unknown = sort!(filter(collect(calls)) do call
                base_name = Symbol(lstrip(string(call), '.'))
                !(isdefined(Base, base_name) || call in documented ||
                  call in definitions || call in (:mul!, :rand!) ||
                  startswith(string(call), "app_"))
            end; by=string)
            isempty(unknown) || error("Undocumented example call names: $unknown")
            println("Checked ", length(calls), " unqualified example call names.")
            kinds = Dict{Any,Symbol}()
            bindings!(kinds, parsed)
            check_routing(parsed, kinds)
            println("Checked explicit recorded-call routing; no graph/group adaptation or numerical target overloads.")
        end
    end
    println("Static syntax/name checks only; no Mantle import or GPU execution.")
end
