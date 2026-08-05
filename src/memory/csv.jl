"""
Reader for MiniMalloc's benchmark corpus, so our placer is measured against the
same instances its authors used.

Faithful to `converter.cc:122-237`, including the trap: a column named `end` is an
*inclusive* last time and adds 1 to every upper bound, lifespans and gap bounds
alike, while a column named `upper` is exclusive. The same file with the column
renamed is a different problem. We read `end`; we never write it.
"""
function parsegap(s::AbstractString, addend::Int)
    at = split(s, '@')
    span = split(at[1], '-')
    length(span) == 2 || error("improperly formed gap: $s")
    during = Span(parse(Int, span[1]), parse(Int, span[2]) + addend)
    window = nothing
    if length(at) > 1
        w = split(at[2], ':')
        length(w) == 2 || error("improperly formed gap: $s")
        window = OffsetWindow(parse(Int, w[1]), parse(Int, w[2]))
    end
    Gap(during, window)
end

function readproblem(path::AbstractString, capacity::Integer)
    lines = filter(!isempty, readlines(path))
    addend = 0
    cols = Dict{String,Int}()
    for (i, raw) in enumerate(strip.(split(lines[1], ',')))
        key = String(raw)
        key in ("buffer", "buffer_id") && (key = "id")
        key in ("begin", "start") && (key = "lower")
        key == "end" && (key = "upper"; addend = 1)
        haskey(cols, key) && error("duplicate column: $key")
        cols[key] = i
    end
    for required in ("id", "lower", "upper", "size")
        haskey(cols, required) || error("missing required column: $required")
    end

    items = Item[]
    for line in lines[2:end]
        f = split(line, ',')
        length(f) == length(cols) || error("row has $(length(f)) fields, header has $(length(cols))")
        at(k) = f[cols[k]]
        has(k) = haskey(cols, k)
        gaps = has("gaps") ?
            [parsegap(g, addend) for g in split(at("gaps"), ' '; keepempty = false)] : Gap[]
        push!(items, Item(at("id"),
                          Span(parse(Int, at("lower")), parse(Int, at("upper")) + addend),
                          parse(Int, at("size"));
                          alignment = has("alignment") ? parse(Int, at("alignment")) : 1,
                          gaps = gaps,
                          offset = has("offset") ? parse(Int, at("offset")) : nothing))
    end
    Problem(items, Int(capacity))
end

"""Capacity is not in the file; MiniMalloc puts it in the name as `NAME.CAPACITY.csv`."""
function readproblem(path::AbstractString)
    parts = split(basename(path), '.')
    length(parts) >= 3 || error("no capacity in filename: $path")
    readproblem(path, parse(Int, parts[end-1]))
end
