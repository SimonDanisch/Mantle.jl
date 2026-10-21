# Parse-only source size accounting. Run in the similarity tooling environment.
using JuliaSyntax, TOML

function source_inventory(root)
    rows = NamedTuple[]
    for sub in ("src", "ext"), (dir, _, fs) in walkdir(joinpath(root, sub)), f in fs
        endswith(f, ".jl") || continue
        path = joinpath(dir, f)
        raw = read(path, String)
        bytes = collect(codeunits(raw))
        tree = JuliaSyntax.parseall(JuliaSyntax.SyntaxNode, raw; filename=path)
        docs, comments = falses(length(bytes)), falses(length(bytes))
        function markdocs(node)
            if JuliaSyntax.kind(node) == JuliaSyntax.K"doc"
                doc = JuliaSyntax.children(node)[1]
                docs[JuliaSyntax.first_byte(doc):JuliaSyntax.last_byte(doc)] .= true
            end
            for child in something(JuliaSyntax.children(node), ())
                markdocs(child)
            end
        end
        markdocs(tree)
        for token in JuliaSyntax.tokenize(raw)
            JuliaSyntax.kind(token) == JuliaSyntax.K"Comment" &&
                (comments[token.range] .= true)
        end
        total = code = doc = comment = blank = 0
        pos = 1
        lines = split(raw, '\n'; keepempty=true)
        endswith(raw, "\n") && pop!(lines)
        for line in lines
            total += 1
            active = [pos + i - 1 for i in eachindex(line) if !isspace(line[i])]
            if isempty(active)
                blank += 1
            elseif any(i -> !docs[i] && !comments[i], active)
                code += 1
            elseif any(i -> docs[i], active)
                doc += 1
            else
                comment += 1
            end
            pos += ncodeunits(line) + 1
        end
        push!(rows, (; file=relpath(path, root), total, code, doc, comment, blank))
    end
    return sort!(rows; by=r -> r.code, rev=true)
end

function write_inventory(root, output)
    rows = source_inventory(root)
    totals = Dict(string(k) => sum(getproperty(r, k) for r in rows)
                  for k in (:total, :code, :doc, :comment, :blank))
    data = Dict("files" => length(rows), "totals" => totals,
                "sources" => [Dict(string(k) => getproperty(r, k) for k in keys(r)) for r in rows])
    open(io -> TOML.print(io, data), output, "w")
    return totals
end

if abspath(PROGRAM_FILE) == @__FILE__
    root = length(ARGS) >= 1 ? abspath(ARGS[1]) : normpath(joinpath(@__DIR__, "../.."))
    output = length(ARGS) >= 2 ? ARGS[2] : joinpath(@__DIR__, ".cache/inventory.toml")
    mkpath(dirname(output))
    println(write_inventory(root, output))
end
