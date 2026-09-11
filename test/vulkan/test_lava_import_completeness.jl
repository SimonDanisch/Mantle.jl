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
        # `test_core_names_no_backend.jl` was fixed for exactly this on
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

# Uppercase, and the same value the other two guards bind: these files are
# `include`d into `Main` side by side, and a const re-bound to what it already
# holds is not a redefinition.
const SRC = joinpath(pkgdir(Mantle), "src")

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

@testset "every Lava name Mantle uses is imported" begin

    # One dict, because one module owns every file: `src/vulkan/` is `@static
    # include`d into `Mantle` again, so a bare `release!` in it resolves against
    # Mantle's bindings the way it did before the extension split.
    sites = Dict{Symbol, Vector{String}}()
    for (root, _, files) in walkdir(SRC), f in files
        endswith(f, ".jl") || continue
        path = joinpath(root, f)
        rel = relpath(path, SRC)
        for s in referenced_names(path)
            push!(get!(Vector{String}, sites, s), rel)
        end
    end

    # Base and Core are the other two ways a bare name resolves, and a name in
    # either is not evidence of anything about the import list.
    visible(s) = isdefined(Mantle, s) || isdefined(Base, s) || isdefined(Core, s)
    unimported = sort([s for s in keys(sites) if isdefined(Lava, s) && !visible(s)];
                      by = string)

    # Named, not counted: the failure has to say which name and which file, or
    # the next person re-derives what this test already knows.
    @test [(s, sort(unique(sites[s]))) for s in unimported] ==
          Tuple{Symbol,Vector{String}}[]

    # ── Names two of Mantle's `using`s both export ────────────────────────────
    #
    # The same question from a direction the loop above cannot see. A module that
    # says `using Vulkan` alongside its own definitions gets every name they BOTH
    # export bound to NEITHER: Julia refuses to guess, so the name is `isdefined`
    # == false and the first use is an `UndefVarError` whose message suggests
    # checking the spelling.
    #
    # Five collide with Vulkan today — `Buffer`, `Device`, `DrawIndirectCommand`,
    # `Framebuffer` and `Sampler` — and the backend means Mantle's every time.
    # They surfaced ONE AT A TIME, in load order, over four reloads:
    # `DrawIndirectCommand` in `graphics/pipeline.jl`, `Buffer` in four `rename!`
    # signatures in the last file included, and `Backend` not until the sync
    # lowering RAN, because that one is inside a function body. That is the whole
    # argument for checking it statically instead of finding out.
    #
    # `src/Mantle.jl` says `import Vulkan as VK` for exactly this reason, so the
    # five are not live — and this is what fails if anyone makes it a bare
    # `using` again.
    #
    # Against every module bare-`using`d anywhere in `src/`, not just the driver:
    # `Backend` collides with KernelAbstractions, not with Vulkan. And a
    # collision is created by EITHER side adding an export, so neither package's
    # own tests can see one coming.
    #
    # Checked against what Mantle actually REFERENCES rather than against the
    # whole intersection — `Scalar` (Mantle's vs StaticArrays') collides and
    # appears only in comments, which is not a defect.
    loadedmod(n) = (for (_, m) in Base.loaded_modules; nameof(m) === n && return m; end;
                    nothing)

    # Every file, because a bare `using` anywhere in the tree binds for the whole
    # module — which is the difference the extension split used to hide.
    usings = Set{Symbol}()
    for (root, _, files) in walkdir(SRC), f in files
        endswith(f, ".jl") && union!(usings, bare_usings(joinpath(root, f)))
    end
    others = filter(!isnothing, loadedmod.(setdiff(collect(usings), [:Mantle])))
    both = union(Set{Symbol}(), (intersect(Set(names(Mantle)), Set(names(m)))
                                 for m in others)...)
    # Named with their files, so the failure says where to add the import.
    clashing = sort([(s, sort(unique(sites[s]))) for s in both
                     if haskey(sites, s) && !isdefined(Mantle, s)]; by = first)
    @test clashing == Tuple{Symbol,Vector{String}}[]
end

# ── Names Mantle DEFINES that a dependency also defines ──────────────────────
#
# The third way a bare name goes to the wrong function, and the one that cost
# the most. `vulkan/raytracing/acceleration.jl` says
#
#     function unsafe_free!(as::Union{LavaBLAS, LavaTLAS})
#
# unqualified. While `src/vulkan/` was an extension that extended whatever
# `import GPUArrays: unsafe_free!` had brought in; back inside `module Mantle`
# with no such import it DEFINES `Mantle.unsafe_free!`, a new function. The
# `finalizer(unsafe_free!, xs)` in `vulkan/array/lavaarray.jl` then registers
# that one, which has no `LavaArray` method, and every array finalization prints
# `error in running finalizer: MethodError` from the GC. 125 in one suite run,
# and not one test failed — a finalizer error is not an exception anyone catches.
#
# GPUArrays does not EXPORT `unsafe_free!`, so the collision check above cannot
# see it: the name never collides, it is simply absent and gets defined. What is
# checkable is the SET of names in this position, pinned. A new entry means
# someone wrote a definition that may have been meant to extend a dependency —
# and if it was meant, it goes in the list with the others.
@testset "no new name shadows a dependency's" begin
    # `eval`, `include` and `__init__` are in every module by construction.
    universal = (:eval, :include, :__init__)
    loadedmod(n) = (for (_, m) in Base.loaded_modules; nameof(m) === n && return m; end;
                    nothing)
    usings = Set{Symbol}()
    for (root, _, files) in walkdir(SRC), f in files
        endswith(f, ".jl") && union!(usings, bare_usings(joinpath(root, f)))
    end
    deps = filter(!isnothing, loadedmod.(setdiff(collect(usings), [:Mantle])))

    # `binding_module`, not `parentmodule`: it answers for an imported binding
    # as well as a defined one, and it does not throw on a union alias the way
    # `parentmodule(Mantle.AnyLavaArray)` does.
    defines(m, n) = isdefined(m, n) && Base.binding_module(m, n) === m &&
                    (v = getglobal(m, n); v isa Function || v isa Type)
    ours = [n for n in names(Mantle; all = true)
            if !startswith(String(n), "#") && !(n in universal) && defines(Mantle, n)]
    shadows = sort([(n, nameof(m)) for n in ours for m in deps if defines(m, n)]; by = first)

    # Every one of these is Mantle's own vocabulary that happens to spell a name
    # a dependency also uses — `Backend` and `Window` are the documented ones,
    # `fence` is the timeline counter and not UnsafeAtomics' barrier. Reviewed
    # on 2026-09-11; `unsafe_free!` is NOT here, because it is imported now.
    known = [(:Attribute, :LLVM), (:Backend, :KernelAbstractions),
             (:Mat4f, :GeometryBasics), (:Pass, :LLVM), (:Window, :GLFW),
             (:alignment, :LLVM), (:allocate, :KernelAbstractions),
             (:backend, :KernelAbstractions), (:count, :AcceleratedKernels),
             (:device, :KernelAbstractions), (:fence, :UnsafeAtomics),
             (:free!, :LLVM), (:gemv!, :LinearAlgebra), (:offset, :LLVM),
             (:overlaps, :GeometryBasics), (:register!, :LLVM), (:run!, :LLVM),
             (:storage, :GPUArrays), (:workgroupsize, :KernelAbstractions)]
    @test setdiff(shadows, known) == Tuple{Symbol,Symbol}[]
end
