"""
Core names nothing that only a backend defines.

`mantle-owns-it.md` in one direction: a file outside `src/vulkan/` and
`src/metal/` may not reach for a name the backend owns. Core calling into a
backend is the violation that a green single-device suite cannot see, because
on that machine the name resolves.

This used to be one of four testsets policing the `import Mantle: …` list at the
top of `ext/MantleVulkanExt.jl`. The backend is `@static include`d into `Mantle`
now, so that list is gone and with it three whole bug classes: an import naming
something Mantle does not declare, a method landing on a new local function
instead of extending core's, and a backend name shadowing a core one. None of
them can happen inside one module. This one can, so it is what is left — and it
got stricter in the move: "only a backend defines it" was module membership and
is now a question about where the METHODS are, which is the thing the rule was
always about.

Source-level and device-free on purpose, so it runs everywhere.
"""

using Test, Mantle

const SRC = joinpath(pkgdir(Mantle), "src")

isbackendfile(p::AbstractString) = occursin(joinpath("src", "vulkan"), p) ||
                                   occursin(joinpath("src", "metal"), p)
iscorefile(p::AbstractString) = startswith(p, SRC) && !isbackendfile(p)

function free_names(path::AbstractString)
    used, bound = Set{Symbol}(), Set{Symbol}()
    bind(ex) = ex isa Symbol ? push!(bound, ex) :
               ex isa Expr && ex.head in (:(::), :(=), :tuple, :(...)) ?
               foreach(bind, ex.args[1:1]) : nothing
    # A signature BINDS its parameters. Without this the extractor reported
    # three whole syntactic classes as free references to backend names:
    # `Item(id, live, size)`'s positional `id` (the backend has ObjectiveC's),
    # `RayTracingPipeline(; raygen, closest_hit, …)`'s bare keywords (the
    # backend has Raycore's `closest_hit` and `any_hit`), and the module path in
    # `using KernelInterface: …`. All four offenders on 2026-09-08 were these.
    # A call in expression position is untouched, which is the case that matters.
    bindparam(a) = a isa Expr && a.head === :parameters ? foreach(bindparam, a.args) :
                   a isa Expr && a.head === :kw ? (bind(a.args[1]); walk(a.args[2])) :
                   bind(a)
    function bindsig(ex)
        ex isa Expr || return bind(ex)
        ex.head === :where && return bindsig(ex.args[1])
        ex.head === :call || return bind(ex)
        foreach(bindparam, ex.args[2:end])
        return
    end
    function walk(ex)
        ex isa Symbol && return push!(used, ex)
        ex isa Expr || return
        ex.head === :quote && return
        ex.head in (:using, :import, :export) && return  # neither names a value
        ex.head === :. && return walk(ex.args[1])
        if ex.head === :struct                      # field names are not uses
            for f in ex.args[3].args
                f isa Expr && f.head === :(::) ? walk(f.args[end]) : nothing
            end
            return
        elseif ex.head === :function && length(ex.args) == 1
            return bind(ex.args[1])                 # `function f end` declares f
        elseif ex.head === :function
            bindsig(ex.args[1])
        elseif ex.head === :(=) || ex.head === :(::)
            bindsig(ex.args[1])
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

# A name whose every method comes from a backend tree is that backend's, whatever
# module it now lives in. Scoped to functions and types: a plain `const` has no
# definition site to ask about, and the rule is about behaviour, not constants.
#
# EXCEPT the vocabulary. `Window` and `trace_rays!` have no core method at all —
# core declares them and only a backend answers — and core naming those is the
# arrangement working, not a violation of it.
function backendonly(s::Symbol)
    s in Mantle.BACKEND_VOCABULARY && return false
    isdefined(Mantle, s) || return false
    v = getglobal(Mantle, s)
    (v isa Function || v isa Type) || return false
    # A name Mantle merely IMPORTED belongs to whoever declared it, and core is
    # free to call it: `get_backend` is KernelAbstractions' vocabulary, and that
    # only the backend answers it here says nothing about the core/backend line.
    parentmodule(v) === Mantle || return false
    # A method from a third package is neither side's: `LavaArray` inherits
    # GPUArrays' `AnyGPUArray(::UniformScaling, dims)` constructors, and counting
    # those as core's would hide the backend's own array type from this.
    ms = collect(methods(v))
    return any(m -> isbackendfile(string(m.file)), ms) &&
          !any(m -> iscorefile(string(m.file)), ms)
end

@testset "core names nothing that only a backend defines" begin
    core = [joinpath(r, f) for (r, _, fs) in walkdir(SRC) for f in fs
            if endswith(f, ".jl") && !isbackendfile(joinpath(r, f))]
    @test !isempty(core)

    offenders = Tuple{Symbol,String}[]
    for p in core, s in free_names(p)
        isdefined(Base, s) || isdefined(Core, s) ? continue : nothing
        backendonly(s) && push!(offenders, (s, relpath(p, SRC)))
    end
    # Named with their files: the fix is always "move it to whichever side
    # actually owns it", and which side that is depends on the name.
    @test sort(unique(offenders)) == Tuple{Symbol,String}[]
end
