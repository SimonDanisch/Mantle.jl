"""
A backend extension may only import names its parent actually declares.

`ext/MantleVulkanExt.jl` and `ext/MantleMetalExt.jl` each open with a long
`import Mantle: …` list, and the header of the first one explains why they have
to: `src/vulkan/` was written inside `module Mantle`, where every name resolved
for free, and in an extension a method on an un-imported `release!` silently
defines `MantleVulkanExt.release!` instead of extending Mantle's.

The failure this file pins is the same mistake from the other end — importing a
name Mantle does NOT have. Julia allows it: it creates the binding in the parent
and prints

    WARNING: Imported binding Mantle.emit_draw! was undeclared at import time
             during import to MantleVulkanExt.

which is a warning inside a precompile log nobody reads. Two things then go
wrong. The first method definition on such a name is a hard `invalid method
definition in MantleVulkanExt: exported function Mantle.emit_draw! does not
exist`, hundreds of lines later and pointing at the definition rather than the
import. And if the extension gets that far, seven of Vulkan's private recording
helpers have become Mantle API that neither core nor another backend can reach.

Found on 2026-08-30, when the Vulkan half of the split was loaded for the first
time: `emit_draw!`, `packdispatch!`, `record_updates!`, `recordlaunch!`,
`recycle!`, `takehost!` and `write_update!` were all in the list, all defined and
used only in `src/vulkan/graph.jl`, and all of them made it through the Mac's
Metal-only test run because that machine never loads this extension.

Source-level and device-free on purpose. It reads the extension SOURCE, so it
checks the Metal extension on a Linux box and the Vulkan one on a Mac — the two
cases neither machine can check by loading. `isdefined` is the right question
even after an extension has loaded: an undeclared binding created by `import`
stays undefined.
"""

using Test, Mantle

"""
Every `import M: a, b` / `using M: a, b` in `path`, as `module name => names`.

Only the `M: names` form. A bare `using Vulkan` binds whatever that package
exports and asserts nothing about the names, and `import Vulkan as VK` binds one
name that is the module itself.
"""
function selective_imports(path::AbstractString)
    out = Dict{Symbol, Set{Symbol}}()
    # `Expr(:., :A, :B)` is `A.B`; the module of `import A.B: x` is the last
    # component, which is also the name a test can look up in `loaded_modules`.
    modname(ex) = ex isa Expr && ex.head === :. && !isempty(ex.args) ?
                  last(ex.args) : nothing
    leaf(ex) = ex isa Expr && ex.head === :. && length(ex.args) == 1 ?
               ex.args[1] : nothing

    function walk(ex)
        ex isa Expr || return
        if ex.head in (:import, :using) && length(ex.args) == 1 &&
           ex.args[1] isa Expr && ex.args[1].head === :(:)
            spec = ex.args[1]
            m = modname(spec.args[1])
            if m isa Symbol
                names = get!(Set{Symbol}, out, m)
                for a in spec.args[2:end]
                    n = leaf(a)
                    n isa Symbol && push!(names, n)
                end
            end
            return
        end
        foreach(walk, ex.args)
        return
    end
    walk(Meta.parseall(read(path, String)))
    return out
end

"""The loaded module called `name`, or `nothing` if this session has no such package."""
function loaded_module(name::Symbol)
    name === :Mantle && return Mantle
    for (_, m) in Base.loaded_modules
        nameof(m) === name && return m
    end
    return nothing
end

@testset "extension imports name something the parent declares" begin
    extdir = joinpath(pkgdir(Mantle), "ext")
    files = sort(filter(f -> endswith(f, ".jl"), readdir(extdir)))
    # Both of them, and this is the assertion rather than the setup: a rename
    # that leaves an extension unlisted would make this file pass by checking
    # nothing.
    @test files == ["MantleMetalExt.jl", "MantleVulkanExt.jl"]

    unchecked = Tuple{String,Symbol}[]
    for f in files
        # `by = first`: a `Pair` compares its values when the keys tie, and a
        # `Set` has no ordering, so a bare `sort` is a `MethodError` here.
        @testset "$f" for (mod, names) in
                sort(collect(selective_imports(joinpath(extdir, f))); by = first)
            m = loaded_module(mod)
            if m === nothing
                # Metal on a Vulkan box, Vulkan on a Mac. Recorded and reported
                # below, never silently dropped.
                push!(unchecked, (f, mod))
                continue
            end
            # Named, not counted: the failure says which names and where they
            # would have to be declared.
            undeclared = sort([n for n in names if !isdefined(m, n)]; by = string)
            @test (f, mod, undeclared) == (f, mod, Symbol[])
        end
    end

    isempty(unchecked) ||
        @info """Not checked here, because these modules are not loaded in this \
                 session: $(join(["$f: $m" for (f, m) in unchecked], ", ")). Run \
                 this on a machine that loads them."""
end

# ── The same boundary, crossed the other way ─────────────────────────────────
#
# Above: the extension naming something the parent does not have. Here: CORE
# naming something only the extension has. Both are silent — Julia resolves a
# global in a function body when the body first RUNS — and the second is worse,
# because the file it breaks is portable code that reads as if it were.
#
# Two of these came out of the runtime move and neither failed at load:
#
#   * `INDIRECT_CEILING` stayed in `src/vulkan/graph.jl` while the
#     `dispatchrange` methods that use it moved to `src/graph/build.jl`. It
#     throws on the first dispatch over a DEVICE-side count, which is every
#     Hikari frame and nothing at all before that.
#   * `barrierspan(::BufferRange, st)` moved to core with a body that calls
#     `pool_offset(st.buf[])` — both the backend's. It throws on the first
#     barrier scoped to a slice.
#
# Only the loaded backend can be checked, so this covers Vulkan here and Metal on
# a Mac. It is still worth running on both: the two backends do not have the same
# names, so each machine checks a different half of core.

"""
Bare names a file references, minus everything it binds itself.

The subtraction is what makes the answer usable: without it a struct's field
names, a `function f end` declaration and every local variable read as
references, and the signal drowns. Measured on Mantle's core: 6 hits, of which 2
were real.
"""
function free_names(path::AbstractString)
    used, bound = Set{Symbol}(), Set{Symbol}()
    bind(ex) = ex isa Symbol ? push!(bound, ex) :
               ex isa Expr && ex.head in (:(::), :(=), :tuple, :(...)) ?
               foreach(bind, ex.args[1:1]) : nothing
    function walk(ex)
        ex isa Symbol && return push!(used, ex)
        ex isa Expr || return
        ex.head === :quote && return
        ex.head === :. && return walk(ex.args[1])
        if ex.head === :struct                      # field names are not uses
            for f in ex.args[3].args
                f isa Expr && f.head === :(::) ? walk(f.args[end]) : nothing
            end
            return
        elseif ex.head === :function && length(ex.args) == 1
            return bind(ex.args[1])                 # `function f end` declares f
        elseif ex.head === :(=) || ex.head === :(::)
            bind(ex.args[1])
        elseif ex.head === :for || ex.head === :generator
            for a in ex.args
                a isa Expr && a.head === :(=) && bind(a.args[1])
            end
        elseif ex.head === :macrocall
            m = ex.args[1]; m isa Symbol && push!(used, m)
            m isa Expr && m.head === :. && walk(m.args[1])
            return foreach(walk, ex.args[2:end])
        elseif ex.head === :kw
            return length(ex.args) > 1 && walk(ex.args[2])
        end
        foreach(walk, ex.args)
    end
    walk(Meta.parseall(read(path, String)))
    return setdiff(used, bound)
end

# ── A method that lands on the wrong function ────────────────────────────────
#
# The third way this boundary fails, and the quietest. `function makeargmemory(
# dev::LavaDevice, passes)` inside an extension that never imported the name
# defines `MantleVulkanExt.makeargmemory` — a NEW function. Nothing errors:
# core keeps its own `makeargmemory(::Device, passes) = nothing`, the backend's
# method is simply never reached, and the default is a legal answer.
#
# So the failure surfaces wherever the default is wrong. `makeargmemory` was
# four found this way on 2026-08-30:
#
#   * `makeargmemory` — every `Plan` built with no argument memory, so the first
#     `run!` was `MethodError: no method matching nextslot!(::Nothing, …)`. That
#     is every render on this backend.
#   * `makeprofiler`  — `profile = true` silently gets no profiler.
#   * `submit!`       — core's `submit!(::Any) = nothing`, so "a command buffer
#     must not stay open across a dispatch" went unenforced.
#   * `supports_batch_queue` — core's `false`, on the backend that has one.
#
# Detected by IDENTITY, which is exact: two modules holding different function
# objects under one name is precisely what "defined instead of extended" means.
#
# `eval` and `include` are excluded because every module has its own — that is
# Julia giving each module a copy, not a shadowing bug. `take!` and
# `unsafe_free!` are excluded and are NOT clean: the backend's are separate from
# `Base.take!` and `GPUArrays.unsafe_free!`, so `GPUArrays.unsafe_free!(::LavaArray)`
# does not exist and nothing generic will free one. That predates the split —
# `Mantle` had `using GPUArrays` and defined the name outright, before and after
# — so it is pinned here as known rather than fixed in passing.
const SHADOW_ALLOWED = Set([:eval, :include, :take!, :unsafe_free!])

@testset "a backend extends Mantle's functions, it does not shadow them" begin
    exts = filter(!isnothing, [Base.get_extension(Mantle, e)
                               for e in (:MantleVulkanExt, :MantleMetalExt)])
    isempty(exts) && @info "No backend extension loaded; nothing to check here."
    @testset "$(nameof(ext))" for ext in exts
        shadowed = Symbol[]
        for n in names(ext; all = true, imported = false)
            n in SHADOW_ALLOWED && continue
            (isdefined(ext, n) && isdefined(Mantle, n)) || continue
            a, b = getglobal(ext, n), getglobal(Mantle, n)
            a === b && continue
            a isa Function && b isa Function && push!(shadowed, n)
        end
        # Named: the fix is always to add the name to the extension's
        # `import Mantle:` list, and the failure has to say which name.
        @test (nameof(ext), sort(shadowed; by = string)) == (nameof(ext), Symbol[])
    end
end

@testset "core names nothing that only a backend defines" begin
    src = joinpath(pkgdir(Mantle), "src")
    isbackend(p) = occursin(joinpath("src", "vulkan"), p) ||
                   occursin(joinpath("src", "metal"), p)
    core = [joinpath(r, f) for (r, _, fs) in walkdir(src) for f in fs
            if endswith(f, ".jl") && !isbackend(joinpath(r, f))]
    @test !isempty(core)

    loaded = filter(!isnothing, [Base.get_extension(Mantle, e)
                                 for e in (:MantleVulkanExt, :MantleMetalExt)])
    if isempty(loaded)
        @info "No backend extension is loaded, so core cannot be checked against one."
    end
    @testset "$(nameof(ext))" for ext in loaded
        offenders = Tuple{Symbol,String}[]
        for p in core, s in free_names(p)
            (isdefined(Mantle, s) || isdefined(Base, s) || isdefined(Core, s)) && continue
            isdefined(ext, s) && push!(offenders, (s, relpath(p, src)))
        end
        # Named with their files: the fix is always "move it to whichever side
        # actually owns it", and which side that is depends on the name.
        @test (nameof(ext), sort(unique(offenders))) ==
              (nameof(ext), Tuple{Symbol,String}[])
    end
end
