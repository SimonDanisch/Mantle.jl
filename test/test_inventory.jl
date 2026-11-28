using Test

# Compare assertion call sites and multiplicities across the two CI runners.
# Test records passing assertions on Julia 1.13 when JULIA_TEST_RECORD_PASSES is set.
function write_test_inventory(io::IO, ts::Test.DefaultTestSet)
    passes = 0
    for result in ts.results
        if result isa Test.DefaultTestSet
            write_test_inventory(io, result)
        elseif result isa Test.Broken
            file = ts.file === nothing ? "" : basename(ts.file)
            println(io, result.test_type, '\t', file, '\t', repr(string(result.orig_expr)))
        elseif result isa Union{Test.Pass,Test.Fail,Test.Error}
            passes += result isa Test.Pass
            source = result.source
            file = source === nothing ? "" : basename(string(source.file))
            line = source === nothing ? 0 : source.line
            println(io, "ran\t", file, ':', line, '\t', repr(string(result.orig_expr)))
        end
    end
    passes == ts.n_passed || error("Test inventory requires JULIA_TEST_RECORD_PASSES=true")
    return nothing
end

function save_test_inventory(name = nothing)
    dir = get(ENV, "MANTLE_TEST_INVENTORY", "")
    isempty(dir) && return nothing
    mkpath(dir)
    path = name === nothing ? tempname(dir; cleanup = false) : joinpath(dir, name)
    open(path * ".tsv", "w") do io
        write_test_inventory(io, Test.get_testset())
    end
    return nothing
end
