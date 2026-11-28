# Parse only: this never loads Mantle or evaluates source macros.
using JuliaSyntax, TOML, SHA

function extract_chunks(root)
    chunks = Dict{String,Any}[]
    files = sort!([joinpath(dir, f) for sub in ("src", "ext")
                   for (dir, _, fs) in walkdir(joinpath(root, sub))
                   for f in fs if endswith(f, ".jl")])
    function visit(node, path)
        if JuliaSyntax.kind(node) == JuliaSyntax.K"function"
            raw = String(JuliaSyntax.sourcetext(node))
            # Preserve line locations while removing comments, using the lexer
            # so '#' inside shader strings and Julia strings stays intact.
            bytes = collect(codeunits(raw))
            for token in JuliaSyntax.tokenize(raw)
                if JuliaSyntax.kind(token) == JuliaSyntax.K"Comment"
                    for i in token.range
                        bytes[i] == 0x0a || (bytes[i] = 0x20)
                    end
                end
            end
            code = String(bytes)
            first = JuliaSyntax.source_line(node)
            lines = split(code, '\n')
            signature = String(lines[1])
            declaration = String(JuliaSyntax.sourcetext(JuliaSyntax.children(node)[1]))
            function add(code, lo, hi, kind)
                length(strip(code)) < 80 && return
                push!(chunks, Dict("file" => relpath(path, root), "start" => lo,
                    "end" => hi, "owner_start" => first, "owner_end" => first + length(lines) - 1,
                    "kind" => kind, "signature" => signature,
                    "declaration" => declaration,
                    "code" => code, "sha256" => bytes2hex(sha256(code))))
            end
            # Keep complete methods. Windows also expose repeated substeps in
            # large methods; never silently truncate the model's input.
            add(code, first, first + length(lines) - 1, "method")
            if length(lines) > 60
                for i in 1:24:length(lines)
                    j = min(i + 39, length(lines))
                    j - i < 9 && continue
                    add(join(lines[i:j], '\n'), first + i - 1, first + j - 1, "region")
                end
            end
        end
        for child in something(JuliaSyntax.children(node), ())
            visit(child, path)
        end
    end
    for path in files
        tree = JuliaSyntax.parseall(JuliaSyntax.SyntaxNode, read(path, String); filename=path)
        visit(tree, path)
    end
    return Dict("files" => length(files), "chunks" => chunks)
end

if abspath(PROGRAM_FILE) == @__FILE__
    root = length(ARGS) >= 1 ? abspath(ARGS[1]) : normpath(joinpath(@__DIR__, "../.."))
    out = length(ARGS) >= 2 ? ARGS[2] : joinpath(@__DIR__, ".cache/chunks.toml")
    mkpath(dirname(out))
    data = extract_chunks(root)
    open(io -> TOML.print(io, data), out, "w")
    println("Parsed ", data["files"], " files; extracted ", length(data["chunks"]), " method/region chunks to ", out)
end
