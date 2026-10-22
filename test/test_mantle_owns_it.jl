# Ownership guards for core/backend boundaries (docs/recording-plan.md).
# Known debt is explicit; backend source checks run even when that backend is not loaded.

using Test, Mantle

const ROOT = dirname(dirname(pathof(Mantle)))
const BACKEND_DIRS = ["vulkan", "metal"]

"""Code with docstrings, string literals and comments removed."""
function codeonly(src::AbstractString)
    s = replace(src, r"\"\"\"[\s\S]*?\"\"\"" => "\"\"")
    s = replace(s, r"\"(?:\\.|[^\"\\])*\"" => "\"\"")
    return join((replace(l, r"#.*" => "") for l in split(s, '\n')), '\n')
end

"""Every `.jl` under `src/`, split into the backend trees and everything else."""
function sourcefiles()
    core, backend = String[], String[]
    for (d, _, fs) in walkdir(joinpath(ROOT, "src")), f in fs
        endswith(f, ".jl") || continue
        p = joinpath(d, f)
        isb = any(occursin(joinpath("src", b), p) for b in BACKEND_DIRS)
        push!(isb ? backend : core, p)
    end
    return core, backend
end

"""Names with top-level method/type definitions in source files or trees. Handles
Mantle-qualified extension methods and leading macros. The regex may mistake a
column-zero call for a definition; implementation completeness also checks defaults."""
function definednames(paths)
    mac = raw"(?:@[\w.]+(?:\s*\([^\n]*\))?\s+)*"
    pat = Regex("^" * mac * raw"(?:function\s+)?(?:Mantle\.)?([A-Za-z_][\w!]*)\s*\(", "m")
    tpat = Regex("^" * mac * raw"(?:mutable\s+struct|struct|abstract\s+type|primitive\s+type)\s+([A-Za-z_]\w*)", "m")
    out = Set{Symbol}()
    for p in paths
        fs = isdir(p) ? [joinpath(d, f) for (d, _, ns) in walkdir(p) for f in ns
                         if endswith(f, ".jl")] : [p]
        for f in fs
            code = codeonly(read(f, String))
            for r in (pat, tpat), m in eachmatch(r, code)
                push!(out, Symbol(m.captures[1]))
            end
        end
    end
    return out
end

"""Source trees for every backend, plus extensions defining caps. Read source rather
than loaded methods so completeness checks do not depend on the current host."""
function backendsources()
    d = Dict{String,Vector{String}}(b => [joinpath(ROOT, "src", b)] for b in BACKEND_DIRS)
    extdir = joinpath(ROOT, "ext")
    for f in (isdir(extdir) ? readdir(extdir) : String[])
        endswith(f, ".jl") || continue
        p = joinpath(extdir, f)
        :caps in definednames([p]) && (d[f] = [p])
    end
    return d
end

# 0.1: portable vocabulary names contain no vendor prefixes.

@testset "0.1 the vocabulary names no vendor" begin
    bad = filter(n -> occursin(r"^(vk|mtl|lava|VK_|MTL)"i, String(n)),
                 collect(Mantle.BACKEND_VOCABULARY))
    isempty(bad) || @info "0.1 vendor-named vocabulary entries" bad
    # A ratchet: the set is empty and stays empty.
    @test isempty(bad)
end

# 0.2: every vocabulary verb has a core default or the required native implementations.
# Optional resource-specific hooks are distinguished from missing implementations below.
const BACKEND_SPECIFIC_FUNCS = Set{Symbol}([
    :deviceslice,                         # native array views; some backends keep blocks instead
    :hostspan,                            # directly host-addressable device memory only
    :feed!, :decodemore!, :remaining,      # hardware decoder objects only
    :buildgeometry!, :instancearray, :updatebatch!,  # BatchedTLAS native primitives
    :mipmapped,                           # graphics textures only
])

# Known missing implementations. New omissions and resolved entries both fail the guard.
const UNIMPLEMENTED_BACKEND_FUNCS = Set{Symbol}([
    :access, :acquire_next_image!, :allocate_batch_queue!, :batchqueue,
    :begin_pass!, :begin_render_pass!, :beginframe!, :bind_textures,
    # These native graphics/RT operations are not implemented by the ROCm backend.
    :build_blas_aabb, :build_blas, :copytexels!,
    :build_accel!, :colorimage, :compile_draw, :currentimage,
    :depthimage, :destroyrecording!, :deviceof, :draw_in_pass!,
    :draw_indexed_in_pass!, :draw_indirect_in_pass!, :emitkernel!, :end_pass!,
    :end_render_pass!, :imageusage, :indirectslot, :initbackend!, :layout,
    :makeimage, :makerecording, :present_frame!, :readback_framebuffer,
    :readback_window, :record_draw!, :recorder, :refit_tlas!,
    :release_batch_queue!, :remakeimage!, :reset_device!, :resetrecording!,
    :screenshot, :set_viewport!, :setviewport!,
    :stages, :storebytes!, :submit!,
    :use_bindings!,
    :vertextouches, :fragmenttouches, :shadertouches,
    # Joined 2026-10-10 with the portable lifecycle (`runtime/lifecycle.jl`):
    # answered by Vulkan and Metal, not by the ROCm extension, which no worker
    # here can load to test an answer against.
    :submissions, :tryallocate,
])

@testset "0.2 a vocabulary name has a core default or every backend" begin
    bysrc = backendsources()
    defs = Dict(b => definednames(ps) for (b, ps) in bysrc)
    core, _ = sourcefiles()
    coredefs = definednames(core)
    backendfiles = reduce(vcat, values(bysrc))
    isbackendfile(f) = any(b -> occursin(b, string(f)), backendfiles)

    # Read implementation presence from all backend source trees. Inspect defaults in
    # method tables, excluding extensions and test doubles: neither supplies a core default.
    function hasdefault(n)
        n in coredefs && return true
        isdefined(Mantle, n) || return false
        v = getglobal(Mantle, n)
        (v isa Function || v isa Type) || return false
        # Test doubles do not supply core defaults.
        return any(m -> !isbackendfile(m.file) && !startswith(string(m.file), @__DIR__),
                   methods(v))
    end

    lonely = Symbol[]
    for n in Mantle.BACKEND_VOCABULARY
        n in BACKEND_SPECIFIC_FUNCS && continue
        hasdefault(n) && continue
        all(d -> n in d, values(defs)) || push!(lonely, n)
    end
    backends = sort(collect(keys(defs)))
    new = sort(collect(setdiff(Set(lonely), UNIMPLEMENTED_BACKEND_FUNCS)); by = string)
    fixed = sort(collect(setdiff(UNIMPLEMENTED_BACKEND_FUNCS, Set(lonely))); by = string)
    isempty(new) || @info "0.2 NEW vocabulary names with no default and not every \
        backend. Three places this can go: implement it on the backends that \
        lack it; or `UNIMPLEMENTED_BACKEND_FUNCS` if it SHOULD be answered \
        everywhere and is not yet; or `BACKEND_SPECIFIC_FUNCS`, with the reason, \
        if asking every backend is the wrong requirement — a build-global hook \
        taking no device cannot be answered by the always-loaded host backend" backends new
    isempty(fixed) || @info "0.2 names in UNIMPLEMENTED_BACKEND_FUNCS that are now answered \
        everywhere — delete them from the list" backends fixed
    @test isempty(new)
    @test isempty(fixed)
end

# 0.2b: identical nullary core/backend methods would overwrite during precompilation.
# Compiler device overlays are separate method tables and do not count.
@testset "0.2b a no-argument method in core is not answered again by a backend" begin
    # Indented too, unlike `definednames`: the one that clashed sat inside an
    # `@static if`. So a definition is spelled out, `function NAME()` or
    # `NAME() =`, and a bare `NAME()` on its own line is a call.
    mac = raw"(?:@[\w.]+(?:\s*\([^\n]*\))?\s+)*"
    pat = Regex("^\\s*(" * mac * raw")(?:function\s+(?:Mantle\.)?([A-Za-z_][\w!]*)\(\)|(?:Mantle\.)?([A-Za-z_][\w!]*)\(\)\s*=(?!=))", "m")
    function nullary(paths)
        out = Set{Symbol}()
        for p in paths, m in eachmatch(pat, codeonly(read(p, String)))
            occursin("device_override", m.captures[1]) && continue
            push!(out, Symbol(something(m.captures[2], m.captures[3])))
        end
        return out
    end
    core, backend = sourcefiles()
    twice = sort(collect(intersect(nullary(core), nullary(backend))); by = string)
    isempty(twice) || @info "0.2b no-argument methods defined in core and in a backend tree" twice
    @test isempty(twice)
    @test :staged_gemm_tile in nullary(core)
end

# 0.3: triage backend vocabulary methods with neither driver operations nor delegation.
# A pure result can be a native fact rather than misplaced policy; review each candidate.
# This known debt remains @test_broken.

const DRIVER_TOKENS = r"\b(VK\.|vk[A-Z_]|VK_|MTL\.|MTLm\.|Metal\.|@objc)\b"
const PURE_BOOKKEEPING_ALLOWED = Set{Symbol}()

@testset "0.3 a backend method names its driver" begin
    _, backend = sourcefiles()
    pure = Tuple{Symbol,String}[]
    voc = Set(Mantle.BACKEND_VOCABULARY)
    # Every name the backend trees define, so a call to one counts as reaching
    # the driver through it.
    local_names = Set{String}()
    for p in backend, m in eachmatch(r"^(?:@inline\s+)?function\s+(?:Mantle\.)?([A-Za-z_][\w!]*)|^([A-Za-z_][\w!]*)\s*\([^)]*\)\s*="m,
                                     codeonly(read(p, String)))
        push!(local_names, something(m.captures[1], m.captures[2]))
    end
    for p in backend
        src = codeonly(read(p, String))
        # Top-level definitions, crudely: `function NAME(` up to a line that is
        # exactly `end` at column 1. Enough to see whether a body names a driver.
        for m in eachmatch(r"^function\s+(?:Mantle\.)?([A-Za-z_][\w!]*)\s*\([\s\S]*?^end$"m, src)
            name = Symbol(m.captures[1])
            name in voc || continue
            name in PURE_BOOKKEEPING_ALLOWED && continue
            occursin(DRIVER_TOKENS, m.match) && continue
            # Delegation to another function of this backend reaches the driver.
            body = m.match
            any(n -> n != String(name) && occursin(Regex("\\b" * n * "\\("), body), local_names) && continue
            push!(pure, (name, relpath(p, ROOT)))
        end
    end
    isempty(pure) || @info "0.3 vocabulary methods in a backend with no driver call" pure
    @test_broken isempty(pure)
end

# 0.4: core names no driver in code, except the exact backend-loader imports.

@testset "0.4 core names no backend" begin
    core, _ = sourcefiles()
    # `VulkanAPI`, `MetalAPI`, `WebGPUAPI` are core's own marker types and the
    # suffix is deliberate — `sync/backend.jl` explains that a marker called
    # `Vulkan` shadows the package inside this module. They are not violations.
    markers = r"^(Vulkan|Metal|WebGPU)API$"
    hits = Tuple{String,String}[]
    for p in core
        for m in eachmatch(r"\b(Lava\w*|lava_\w+|Vulkan\w*|vk_\w+|VK_\w+|MTL\w*|mtl[a-z]\w*)\b",
                           codeonly(read(p, String)))
            occursin(markers, m.match) && continue
            # `/` on every OS, so the pinned exceptions below match on Windows too.
            push!(hits, (replace(relpath(p, ROOT), '\\' => '/'), m.match))
        end
    end
    # Only Mantle.jl's backend loader may import Vulkan; keep the exception exact.
    allowed = [("src/Mantle.jl", "Vulkan"), ("src/Mantle.jl", "Vulkan")]
    extra = sort(hits) != sort(allowed) ? hits : Tuple{String,String}[]
    isempty(extra) || @info "0.4 backend names in core" count = length(hits) first10 = first(hits, 10)
    @test sort(hits) == sort(allowed)
end

# 0.6: all Metal capability/device entry points must share one device and pool.
# Consumer boundary checks run in consumer suites.

@testset "0.6 one device per process" begin
    hasmetal = any(m -> occursin(joinpath("src", "metal"), string(m.file)),
                   methods(Mantle.caps))
    if !hasmetal
        @info "0.6 Metal not loaded; nothing to check"
    else
        d = Mantle.Device(Mantle.MetalAPI())
        Mantle.caps(Mantle.backend(d))          # the second entry point
        # A second cache not existing at all is the passing state.
        second = isdefined(Mantle, :_DEVICE) ? getglobal(Mantle, :_DEVICE)[] : nothing
        second === nothing && (second = d)
        same = d === second
        same || @info "0.6 a second device exists" pools_differ = Mantle.pool(d) !== Mantle.pool(second)
        # A ratchet: one device, one cache.
        @test same
    end
end

# 0.7: shared test files select their backend through TESTBACKEND.
# Only the harness and this explicit ownership guard may name a backend directly.

const BACKEND_NAMED_ALLOWED = Set([
    "runtests.jl",                # the harness names the sections it guards
    "test_mantle_owns_it.jl",     # 0.6 asks a Metal-specific question
])

@testset "0.7 the shared test layer names no backend" begin
    dir = joinpath(ROOT, "test")
    hits = Tuple{String,Int}[]
    for f in readdir(dir)
        endswith(f, ".jl") && !(f in BACKEND_NAMED_ALLOWED) || continue
        n = count(m -> true, eachmatch(r"\b(VulkanAPI|MetalAPI|WebGPUAPI)\(\)|\bimport +\w*, *Lava\b|\busing +\w*, *Lava\b",
                                       codeonly(read(joinpath(dir, f), String))))
        n == 0 || push!(hits, (f, n))
    end
    isempty(hits) || @info "0.7 backend named in the shared test layer" hits
    @test isempty(hits)
end

# 0.7b: shared tests may use core/vocabulary bindings, not backend-private ones.
# Method definition sites identify ownership; the conservative token scan may report
# local names that collide with a backend binding, requiring review.
@testset "0.7b the shared test layer names no backend-only binding" begin
    backendsrc(f) = any(d -> occursin(joinpath("src", d), string(f)), BACKEND_DIRS)
    coresrc(f) = startswith(string(f), joinpath(ROOT, "src")) && !backendsrc(f)

    function backendonly(s::Symbol)
        s in Mantle.BACKEND_VOCABULARY && return false
        (isdefined(Mantle, s) && Base.binding_module(Mantle, s) === Mantle) || return false
        v = getglobal(Mantle, s)
        (v isa Function || v isa Type) || return false
        ms = collect(methods(v))
        return any(m -> backendsrc(m.file), ms) && !any(m -> coresrc(m.file), ms)
    end

    dir = joinpath(ROOT, "test")
    hits = Tuple{String,Vector{Symbol}}[]
    for f in readdir(dir)
        endswith(f, ".jl") && !(f in BACKEND_NAMED_ALLOWED) || continue
        code = codeonly(read(joinpath(dir, f), String))
        # Include trailing ! and type annotations; quoted symbols are not binding references.
        named = Set(Symbol(m.match) for m in eachmatch(r"(?<![A-Za-z0-9_!])(?<![^:]:)[A-Za-z_][A-Za-z0-9_]*!?", code))
        bad = sort(collect(filter(backendonly, named)); by = string)
        isempty(bad) || push!(hits, (f, bad))
    end
    isempty(hits) || @info "0.7b backend-only bindings named in the shared test layer" hits
    @test isempty(hits)
end

# 0.8: monitor permissive defaults implemented by only one native backend tree.
# Some are valid capability/default answers; others need explicit refusal or implementation.
# Keep the legacy count bound until those contracts have individual regressions.
@testset "0.8 hooks that default to silence are a known set" begin
    corefile(f) = startswith(string(f), joinpath(ROOT, "src")) &&
                  !any(d -> occursin(joinpath("src", d), string(f)), BACKEND_DIRS)

    # Unwrap parametric method signatures, but treat only Any, Device and Type as
    # catch-alls. Bare parametric types such as CompiledDispatch are specific dispatches.
    permissive(m) = corefile(m.file) &&
        all(t -> t === Any || t === Mantle.Device || t === Type,
            collect(Base.unwrap_unionall(m.sig).parameters)[2:end])

    # Read both backend trees; loaded method tables depend on the host platform.
    defines(dir, nm) = any(readdir(joinpath(ROOT, "src", dir); join = true)) do f
        isdir(f) && return any(g -> endswith(g, ".jl") &&
                                    occursin(Regex("(^|\n)\\s*(function\\s+)?(Mantle\\.)?\\Q$(nm)\\E\\("),
                                             read(g, String)),
                               [joinpath(r, x) for (r, _, xs) in walkdir(f) for x in xs])
        endswith(f, ".jl") && occursin(Regex("(^|\n)\\s*(function\\s+)?(Mantle\\.)?\\Q$(nm)\\E\\("),
                                       read(f, String))
    end

    lonely = Symbol[]
    for nm in Mantle.BACKEND_VOCABULARY
        (isdefined(Mantle, nm) && getglobal(Mantle, nm) isa Function) || continue
        ms = collect(methods(getglobal(Mantle, nm)))
        (isempty(ms) || !any(permissive, ms)) && continue
        answered = [d for d in BACKEND_DIRS if defines(d, nm)]
        length(answered) == 1 && push!(lonely, nm)
    end
    sort!(lonely; by = string)

    @info "0.8 vocabulary verbs with a permissive core default and ONE backend tree implementing" count=length(lonely) lonely
    # Legacy ceiling for the reviewed default set; this is not proof that every default is safe.
    @test length(lonely) <= 31
end

# 0.9: extension-defined Mantle methods must belong to the declared vocabulary.
# Read source so unloaded extensions are checked too.
@testset "0.9 an extension defines only vocabulary names" begin
    voc = Set(Mantle.BACKEND_VOCABULARY)
    dir = joinpath(ROOT, "ext")
    hits = Tuple{String,Vector{Symbol}}[]
    for f in (isdir(dir) ? readdir(dir) : String[])
        endswith(f, ".jl") || continue
        src = codeonly(read(joinpath(dir, f), String))
        named = Symbol[Symbol(m.captures[1]) for m in
                       eachmatch(r"^(?:@inline\s+)?(?:function\s+)?Mantle\.([A-Za-z_][\w!]*)\s*\("m, src)]
        bad = sort(unique(filter(n -> !(n in voc), named)); by = string)
        isempty(bad) || push!(hits, (f, bad))
    end
    isempty(hits) || @info "0.9 extension methods outside the vocabulary" hits
    @test isempty(hits)
end
