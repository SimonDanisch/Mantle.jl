"""
The Vulkan backend never allocates a device buffer without naming its device.

The bug behind the multi-GPU HWTLAS leak was structural, not a typo: a
device-less `LavaArray{T}(undef, n)` / `LavaArray(data)` falls back to the
PROCESS-GLOBAL context, so any backend code that forgets `bq` silently puts its
buffer on the wrong device, and the first cross-context use faults. Fixing the
five HWTLAS sites by hand does not stop the sixth from being written.

This holds the line the way `test_backend_vocabulary.jl` holds the core/backend
one: it scans the backend source and fails on any device-less allocation. A
buffer's device is either passed as `bq`, or carried by the `DataRef`/`copy` the
constructor wraps — never taken from the global. Two call sites are the ambient
default BY CONSTRUCTION (the type-based `adapt(LavaArray, x)`, which names a type
and not a device, and the single-context video download); each is tagged
`ambient-allocation-ok` on the line, and nothing else may be.

Consumers (Hikari, Raycore, RayMakie) are not scanned because they do not name
`LavaArray` at all — they allocate through `KA.allocate(backend, ...)`,
`Adapt.adapt(backend, ...)` and `Mantle.Buffer(dev, ...)`, which carry the device
already. That is the point of the portable API, and this test guards the one
layer allowed to reach past it.
"""

using Test, Mantle

# Per-line, index-stable: docstring lines and comment tails are blanked but the
# line stays, so a call's number is real. The `ambient-allocation-ok` tag is read
# from the comment-bearing raw line, since that is where a tag lives.
function scan_lines(src::AbstractString)
    raw = String.(split(src, '\n'))
    code = String[]
    indoc = false
    for l in raw
        # An odd number of `"""` on a line flips in/out of a docstring; blank the
        # bodies. A line-comment tail is dropped so a `LavaArray(` in prose is not
        # a call. Line count is preserved so a hit reports its real number.
        odd = isodd(count("\"\"\"", l))
        was = indoc
        odd && (indoc = !indoc)
        push!(code, (was || indoc) ? "" : replace(l, r"#.*" => ""))
    end
    return raw, code
end

# A `LavaArray` allocation names its device unless it is a `DataRef`/`copy(...)`
# wrapper (the ref carries the context), an empty `LavaArray{T,1}()`, a call that
# passes `bq`, or a line tagged `ambient-allocation-ok` in a neighbouring comment.
function ambient_allocations(path)
    root = dirname(dirname(pathof(Mantle)))
    raw, code = scan_lines(read(joinpath(root, path), String))
    hits = String[]
    for (i, line) in enumerate(code)
        occursin("LavaArray", line) || continue
        # `bq` may sit on the following line; the tag may be on the line above,
        # on this line, or below — so look at a small window of RAW lines.
        lo = max(1, i-2); hi = min(length(raw), i+1)
        rawwindow = join(raw[lo:hi], " ")
        occursin("ambient-allocation-ok", rawwindow) && continue
        codewindow = line * " " * (i < length(code) ? code[i+1] : "")
        isalloc = occursin(r"LavaArray(\{[^}]*\})?\(\s*undef", line) ||
                  (occursin(r"LavaArray(\{[^}]*\})?\(", line) &&
                   !occursin("copy(", line) && !occursin("::", line) &&
                   !occursin(r"LavaArray(\{[^}]*\})?\(\s*\)", line) &&
                   !occursin("DataRef", line) &&
                   occursin(r"LavaArray(\{[^}]*\})?\(\s*(collect|fill|reinterpret|zeros|ones|[a-z])", line))
        isalloc || continue
        occursin("bq", codewindow) && continue
        push!(hits, "  $path:$i  " * strip(line))
    end
    return hits
end

@testset "the Vulkan backend names a device for every allocation" begin
    root = dirname(dirname(pathof(Mantle)))
    files = String[]
    for (d, _, fs) in walkdir(joinpath(root, "src", "vulkan")), f in fs
        endswith(f, ".jl") || continue
        # `array/lavaarray.jl` is where the constructors are DEFINED; its own
        # `LavaArray(...)` are the definitions, not calls, and the one ambient
        # default (`LavaArray(data)`'s `bq = vk_context().default_bq`) lives here.
        endswith(f, "lavaarray.jl") && continue
        push!(files, relpath(joinpath(d, f), root))
    end
    @test !isempty(files)
    offenders = reduce(vcat, (ambient_allocations(f) for f in files); init = String[])
    isempty(offenders) ||
        @info "device-less LavaArray allocations in the backend (add `bq`, or tag `ambient-allocation-ok` if truly device-less):\n" * join(offenders, "\n")
    @test isempty(offenders)
end
