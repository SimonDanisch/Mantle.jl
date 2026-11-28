# Source probe of an existing compiler defect, not a passing target-API test.
# Run through a Julia REPL in the workspace environment; no Lava/GPU import.
module AtomicDemotionProbe
using LLVM
using SHA

function probe()
    path = get(ENV, "MANTLE_ATOMIC_BASELINE",
        normpath(joinpath(@__DIR__, "../../../Lava/src/compiler/target.jl")))
    source = read(path, String)
    ast = Meta.parseall(source)
    definitions = Expr[]
    function visit(ex)
        ex isa Expr || return
        if ex.head === :function && ex.args[1] isa Expr &&
                ex.args[1].args[1] === :demote_atomic_accesses!
            push!(definitions, ex)
        else
            foreach(visit, ex.args)
        end
    end
    visit(ast)
    isempty(definitions) && error("The demotion pass is absent from this source; use the fixed compiler regression suite, or set MANTLE_ATOMIC_BASELINE to replay the archived defect")
    definition = only(definitions)
    Core.eval(@__MODULE__, definition)
    ir = """
    define i32 @message(ptr %flag, ptr %payload) {
    entry:
      %value = load atomic i32, ptr %flag acquire, align 4
      store atomic i32 %value, ptr %payload release, align 4
      ret i32 %value
    }
    """
    return LLVM.Context() do _
        mod = parse(LLVM.Module, ir)
        try
            before = string(mod)
            changed = Base.invokelatest(() -> demote_atomic_accesses!(mod))
            after = string(mod)
            println("source: ", path)
            println("source SHA256: ", bytes2hex(sha256(source)))
            println("changed: ", changed)
            println("before:\n", before)
            println("after:\n", after)
            println("DEFECT REPRODUCED: ", occursin("load atomic", before) &&
                occursin("store atomic", before) && !occursin(" atomic ", after))
            return nothing
        finally
            LLVM.dispose(mod)
        end
    end
end
end
AtomicDemotionProbe.probe()
