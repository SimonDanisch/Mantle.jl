"""
Every name a Mantle backend uses bare actually resolves in the module that owns it.

Two ways it can fail to, and both are silent until something runs:

  * the name is Lava's and is not in the import list, or
  * the name is exported by Mantle AND by the driver package, which binds it to
    NEITHER.

The first half is the original subject and is described below; the second is at
the bottom of the file.

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

A PARAMETER NAME is not a reference either, and that distinction only started
mattering when a second backend arrived: `mergeconstraints(::MetalDevice, kind,
a, b)` binds `kind`, it does not use Lava's. Counting it reported the Metal
backend as missing imports it never needed. A parameter's TYPE and DEFAULT are
still walked — those really are references.

Nor is a name that a parameter SHADOWS. `bestshape(b, ab, acc; scope = …) =
bestshape(caps(b), ab, acc; scope)` passes its own parameter along in keyword
shorthand; the `scope` at the call site is the local, whatever `scope` means
elsewhere. So a function's parameter names are subtracted from what its body
contributes, which is the smallest amount of scope tracking that makes the
answer right.
"""
function referenced_names(path::AbstractString)
    out = Set{Symbol}()

    # A parameter contributes its annotation and its default, never its name —
    # and the name it binds is collected so the body cannot report it.
    function param(ex, bound)
        ex isa Symbol && return push!(bound, ex)    # `x`
        ex isa Expr || return
        if ex.head === :(::)                        # `x::T` or `::T`
            length(ex.args) > 1 && ex.args[1] isa Symbol && push!(bound, ex.args[1])
            walk(ex.args[end]); return
        elseif ex.head === :kw                      # `x::T = default`
            param(ex.args[1], bound); length(ex.args) > 1 && walk(ex.args[2]); return
        elseif ex.head === :(...)                   # `xs...`
            param(ex.args[1], bound); return
        elseif ex.head === :parameters              # everything after the `;`
            foreach(a -> param(a, bound), ex.args); return
        end
        walk(ex)
    end

    # The callee is a reference; its parameters are not, and neither is anything
    # in the body that they shadow.
    function signature(ex, bound)
        while ex isa Expr && ex.head in (:where, :(::))
            ex.head === :where && foreach(walk, ex.args[2:end])
            ex = ex.args[1]
        end
        ex isa Expr && ex.head === :call || return walk(ex)
        walk(ex.args[1])
        foreach(a -> param(a, bound), ex.args[2:end])
        return
    end

    function walk(ex)
        ex isa Symbol && return push!(out, ex)
        ex isa Expr || return
        ex.head === :quote && return
        # A MODULE PATH in a `using`/`import` is not a reference to a name that
        # has to resolve in this module — it is what makes names resolve. Left
        # in, `import KernelInterface as KI` reported `KernelInterface` as an
        # unimported Lava name the moment Lava imported that module itself.
        # `test_ext_imports_are_declared.jl` was fixed for exactly this on
        # 2026-09-08; this scanner had the same blind spot.
        (ex.head === :using || ex.head === :import) && return
        if ex.head === :.
            walk(ex.args[1]); return
        elseif ex.head === :macrocall
            m = ex.args[1]
            m isa Symbol && push!(out, m)
            m isa Expr && m.head === :. && walk(m.args[1])
            foreach(walk, ex.args[2:end]); return
        elseif ex.head === :kw          # `f(; key = v)`: `key` is not a reference
            length(ex.args) > 1 && walk(ex.args[2]); return
        elseif ex.head === :function || (ex.head === :(=) && ex.args[1] isa Expr &&
                                         ex.args[1].head in (:call, :where))
            bound = Set{Symbol}()
            signature(ex.args[1], bound)
            if length(ex.args) > 1
                outer, out = out, Set{Symbol}()     # collect the body separately…
                walk(ex.args[2])
                union!(outer, setdiff(out, bound))  # …minus what the signature binds
                out = outer
            end
            return
        end
        foreach(walk, ex.args)
        return
    end
    walk(Meta.parseall(read(path, String)))
    return out
end

"""
Which module a source file's bare names have to resolve in.

`src/vulkan/` is no longer part of `Mantle`: it is included by
`ext/MantleVulkanExt.jl`, so a bare `release!` in it resolves against the
EXTENSION's bindings, not Mantle's. That is precisely why the extension carries
explicit `import Mantle: …` and `using Mantle: …` lists — and precisely why this
check has to follow the file into the module that now owns it, or it would
report every one of the backend's Lava names as unimported while the real
question, whether the extension's lists are complete, went unasked.

`nothing` means no module owns the file in this session: the extension is not
loaded, because this machine has no Vulkan loader.
"""
const BACKEND_EXTS = ("vulkan/" => :MantleVulkanExt, "metal/" => :MantleMetalExt)

function owner(relpath_)
    for (dir, ext) in BACKEND_EXTS
        startswith(relpath_, dir) && return Base.get_extension(Mantle, ext)
    end
    # `src/host/` is NOT in that list: the host backend is included straight into
    # `Mantle`, because its trigger would be KernelAbstractions — a hard
    # dependency — and an extension on one of those always fires anyway.
    return Mantle
end

@testset "every Lava name Mantle uses is imported" begin
    src = joinpath(pkgdir(Mantle), "src")

    # Grouped by owning module, since a bare name resolves against whichever
    # module included the file.
    sites = Dict{Module, Dict{Symbol, Vector{String}}}()
    skipped = String[]
    for (root, _, files) in walkdir(src), f in files
        endswith(f, ".jl") || continue
        path = joinpath(root, f)
        rel = relpath(path, src)
        m = owner(rel)
        if m === nothing
            push!(skipped, rel)
            continue
        end
        for s in referenced_names(path)
            push!(get!(Vector{String}, get!(Dict{Symbol,Vector{String}}, sites, m), s), rel)
        end
    end

    for (m, bymod) in sites
        # Base and Core are the other two ways a bare name resolves, and a name in
        # either is not evidence of anything about the import list.
        visible(s) = isdefined(m, s) || isdefined(Base, s) || isdefined(Core, s)
        unimported = sort([s for s in keys(bymod) if isdefined(Lava, s) && !visible(s)];
                          by = string)

        # Named, not counted: the failure has to say which name and which file, or
        # the next person re-derives what this test already knows.
        @test (nameof(m), [(s, sort(unique(bymod[s]))) for s in unimported]) ==
              (nameof(m), Tuple{Symbol,Vector{String}}[])
    end

    # ── Names two of the extension's `using`s both export ─────────────────────
    #
    # The same question from a direction the loop above cannot see. A module that
    # says `using Mantle` and `using Vulkan` gets every name they BOTH export
    # bound to NEITHER: Julia refuses to guess, so the name is `isdefined` ==
    # false and the first use is an `UndefVarError` whose message suggests
    # checking the spelling.
    #
    # Six collide today, and the backend means Mantle's every time. Three were
    # disambiguated for unrelated reasons; the others surfaced ONE AT A TIME, in
    # load order, over four reloads — `DrawIndirectCommand` in
    # `graphics/pipeline.jl`, `Buffer` in four `rename!` signatures in the last
    # file included, and `Backend` not until the sync lowering RAN, because that
    # one is inside a function body. That is the whole argument for checking it
    # statically instead of finding out.
    #
    # Against every module the extension bare-`using`s, not just the driver:
    # `Backend` collides with KernelAbstractions, not with Vulkan. And a
    # collision is created by EITHER side adding an export, so neither package's
    # own tests can see one coming.
    #
    # Checked against what the backend actually REFERENCES rather than against
    # the whole intersection — `Scalar` (Mantle's vs StaticArrays') collides and
    # appears in this backend only in comments, which is not a defect.
    """The modules a file bare-`using`s: `using Foo`, not `using Foo: bar`."""
    function bare_usings(path::AbstractString)
        out = Symbol[]
        walk(ex) = ex isa Expr &&
            (ex.head === :using && !(ex.args[1] isa Expr && ex.args[1].head === :(:)) ?
             foreach(a -> a isa Expr && a.head === :. && length(a.args) == 1 &&
                          push!(out, a.args[1]), ex.args) :
             foreach(walk, ex.args))
        walk(Meta.parseall(read(path, String)))
        return out
    end
    loadedmod(n) = (for (_, m) in Base.loaded_modules; nameof(m) === n && return m; end;
                    nothing)

    @testset "$extname" for (_, extname) in BACKEND_EXTS
        ext = Base.get_extension(Mantle, extname)
        if ext === nothing
            @info "$extname is not loaded here, so its collisions are unchecked."
            continue
        end
        extfile = joinpath(pkgdir(Mantle), "ext", string(extname) * ".jl")
        others = filter(!isnothing, loadedmod.(setdiff(bare_usings(extfile), [:Mantle])))
        both = union(Set{Symbol}(), (intersect(Set(names(Mantle)), Set(names(m)))
                                     for m in others)...)
        used = get(sites, ext, Dict{Symbol,Vector{String}}())
        # Named with their files, so the failure says where to add the import.
        clashing = sort([(s, sort(unique(used[s]))) for s in both
                         if haskey(used, s) && !isdefined(ext, s)]; by = first)
        @test (extname, clashing) == (extname, Tuple{Symbol,Vector{String}}[])
    end

    # Loudly, not silently. A machine with no Vulkan loader cannot load the
    # extension, so the backend's half of this check does not run there — and a
    # test that quietly covers less than it says is how the import lists rotted
    # the first time.
    if !isempty(skipped)
        missing_exts = [String(e) for (_, e) in BACKEND_EXTS
                        if Base.get_extension(Mantle, e) === nothing]
        @info """$(join(missing_exts, " and ")) not loaded, so $(length(skipped)) backend \
                 files were not checked. Run this where those backends load to cover them."""
    end
    @test isempty(skipped) ||
          any(e -> Base.get_extension(Mantle, e) === nothing, last.(BACKEND_EXTS))
end
