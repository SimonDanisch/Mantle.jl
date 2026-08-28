"""
Every name Mantle's Vulkan backend takes from Lava is actually imported.

`src/vulkan/vulkan.jl` lists them one by one — `using Lava: <name>, <name>, …` —
and that list is hand-maintained. A name that is used but not listed is not a
load error: Julia resolves a global inside a function body when the body first
RUNS, so the file precompiles, the package loads, and the `UndefVarError` waits
for whichever path calls it. `kernel_source_name` sat unlisted in
`runtime/profiling.jl` that way, reachable only with pipeline stats enabled.

This has bitten the same list twice. Forty-three device intrinsics were missing
from it after the runtime moved out of Lava, and they were invisible to a
syntactic scan of Lava's source because they are generated in `@eval` loops —
they surfaced only when a kernel refused to compile.

So the check runs against the LOADED modules rather than against Lava's source:
`isdefined` sees a name however it was defined. Parse Mantle for what it
references, ask Lava whether it has it, ask Mantle whether it can see it.

Needs both packages loaded and no device. It is a fast source-and-bindings check
— run it before anything that can take a device down with it.
"""

using Test
using Mantle
using Lava

"""
Every identifier a file references, from its syntax.

`a.b` contributes `a` only — `b` is a field or a qualified name, and a qualified
`Lava.foo` needs no import, which is the whole distinction. Docstrings drop out
for free: a docstring is a string literal, not a symbol.
"""
function referenced_names(path::AbstractString)
    out = Set{Symbol}()
    function walk(ex)
        ex isa Symbol && return push!(out, ex)
        ex isa Expr || return
        ex.head === :quote && return
        if ex.head === :.
            walk(ex.args[1]); return
        elseif ex.head === :macrocall
            m = ex.args[1]
            m isa Symbol && push!(out, m)
            m isa Expr && m.head === :. && walk(m.args[1])
            foreach(walk, ex.args[2:end]); return
        elseif ex.head === :kw          # `f(; key = v)`: `key` is not a reference
            length(ex.args) > 1 && walk(ex.args[2]); return
        end
        foreach(walk, ex.args)
        return
    end
    walk(Meta.parseall(read(path, String)))
    return out
end

@testset "every Lava name Mantle uses is imported" begin
    src = joinpath(pkgdir(Mantle), "src")

    sites = Dict{Symbol, Vector{String}}()
    for (root, _, files) in walkdir(src), f in files
        endswith(f, ".jl") || continue
        path = joinpath(root, f)
        for s in referenced_names(path)
            push!(get!(Vector{String}, sites, s), relpath(path, src))
        end
    end

    # Base and Core are the other two ways a bare name resolves, and a name in
    # either is not evidence of anything about the import list.
    visible(s) = isdefined(Mantle, s) || isdefined(Base, s) || isdefined(Core, s)
    unimported = sort([s for s in keys(sites) if isdefined(Lava, s) && !visible(s)];
                      by = string)

    # Named, not counted: the failure has to say which name and which file, or
    # the next person re-derives what this test already knows.
    @test [(s, sort(unique(sites[s]))) for s in unimported] == Tuple{Symbol,Vector{String}}[]
end
