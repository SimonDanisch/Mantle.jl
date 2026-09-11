# The guards for `docs/mantle-owns-it.md`, written BEFORE the refactor deletes
# anything.
#
# Every one of them fails on the code as it stands. That failure is the point:
# a guard that passes when it is written has not been shown to detect anything.
# They are `@test_broken` so the suite stays usable while the work runs — Julia
# turns a `@test_broken` that starts passing into a FAILURE ("Unexpectedly
# Pass"), which is exactly the signal that a phase is done and the guard should
# be promoted to a plain `@test`.
#
# Each guard names the finding it pins and the phase of the plan that clears it.
# The violations are `@info`-ed rather than only counted, because the list is
# the work list.
#
# This file lives at the top level and not under `test/vulkan/`, unlike
# `test_backend_vocabulary.jl`, which asks a closely related question and runs
# only in the Vulkan section — so it has never run on a machine with no Vulkan
# driver, which is every machine where the Metal backend is the one under test.

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

# ── 0.1 No vendor name in the portable vocabulary ────────────────────────────
#
# `BACKEND_VOCABULARY` is the list of core functions a backend may extend, so a
# vendor-prefixed entry in it is a portable hook that only one vendor can mean.
# `:vkformat` is one; Metal's counterpart is `mtlformat`. CLAUDE.md states the
# rule for code paths and this is a code path.
# Cleared by: phase 1.7.

@testset "0.1 the vocabulary names no vendor" begin
    bad = filter(n -> occursin(r"^(vk|mtl|lava|VK_|MTL)"i, String(n)),
                 collect(Mantle.BACKEND_VOCABULARY))
    isempty(bad) || @info "0.1 vendor-named vocabulary entries" bad
    # PROMOTED after phase 1.7 removed `:vkformat`. A ratchet from here on.
    @test isempty(bad)
end

# ── 0.2 Every vocabulary name is answered by core, or by every backend ───────
#
# A name only one backend answers is either not portable (it belongs in that
# backend) or not implemented (it belongs on a list). Both are decisions; being
# in the vocabulary with one implementation is the absence of one.
#
# `NOT_PORTABLE` is where a deliberate answer goes, with its reason. It is empty
# on purpose: filling it is part of the work, not a way to make this pass.
# Cleared by: phase 2, as each name is either implemented, moved, or listed.

const NOT_PORTABLE = Set{Symbol}()

@testset "0.2 a vocabulary name has a core default or every backend" begin
    # A backend is `@static include`d into Mantle now, not an extension, so
    # "is one here" is a question about files rather than about modules.
    hasbackend = any(methods(Mantle.caps)) do m
        any(b -> occursin(joinpath("src", b), string(m.file)), BACKEND_DIRS)
    end
    if !hasbackend
        @info "0.2 no backend loaded; nothing to check"
    else
        lonely = Symbol[]
        for n in Mantle.BACKEND_VOCABULARY
            n in NOT_PORTABLE && continue
            isdefined(Mantle, n) || continue
            f = getglobal(Mantle, n)
            f isa Function || continue
            ms = collect(methods(f))
            infile(pat) = any(m -> occursin(pat, string(m.file)), ms)
            # A method outside both backend trees is core's answer for everyone.
            hascore = any(m -> !occursin(joinpath("src", "vulkan"), string(m.file)) &&
                               !occursin(joinpath("src", "metal"), string(m.file)), ms)
            hascore && continue
            answered = [b for b in BACKEND_DIRS if infile(joinpath("src", b))]
            length(answered) == length(BACKEND_DIRS) || push!(lonely, n)
        end
        isempty(lonely) ||
            @info "0.2 vocabulary names with no core default and not every backend" count = length(lonely) lonely
        @test_broken isempty(lonely)
    end
end

# ── 0.3 A method in a backend touches the driver ─────────────────────────────
#
# The mechanical form of the rule: a function in a backend that contains no
# driver call is portable logic in the wrong package. `recycle!` is what
# motivates it — one implementation, no `vkDestroy`, no `vkFree`, no driver
# call at all, just pin lists, region releases through Mantle's own `release!`,
# and two free lists.
#
# Scoped to methods of VOCABULARY names, which is where the rule bites and
# what a reader can act on; a private helper with no driver call is usually a
# few lines of arithmetic. Phase 2 may widen this.
#
# DELEGATION COUNTS. `upload!(dst::VkManagedBuffer, …)` is one line handing off
# to `copy_buffer!`, which is where the driver call is; reporting it would train
# a reader to skim the list. So a body that calls any function defined in the
# same backend tree has reached the driver as far as this guard is concerned.
# What is left is a leaf of pure bookkeeping, which is the thing being hunted.
#
# This is a TRIAGE list, not a violation list: an entry is either moved to core
# or added to `PURE_BOOKKEEPING_ALLOWED` with the reason it is a leaf that
# legitimately touches nothing (`fence` returning a counter, say — except that
# one is finding 2 and does not get a pass).
# Cleared by: phases 1.1, 1.2, 2.2, 2.3.

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

# ── 0.4 Core names no backend ────────────────────────────────────────────────
#
# `src/` outside the two backend trees must not name a driver or a driver
# package in code position. Comments and docstrings are stripped, as in
# `test_core_names_no_backend.jl`.
# ALREADY TRUE, so this is a plain `@test` and not a `@test_broken`: with the
# three deliberate `…API` markers excluded, core names no backend in ONE place.
# It is a ratchet over a property that already holds, and the one guard here
# that must never go red rather than one that starts red.

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
            push!(hits, (relpath(p, ROOT), m.match))
        end
    end
    # The one exception, pinned exactly rather than pattern-matched away.
    # `src/Mantle.jl` ends in an `@static if Sys.isapple()` that picks the
    # backend at PARSE time, and picking it means naming the package: `import
    # Vulkan as VK` and `using Vulkan: unwrap, iserror, unwrap_error`. That is
    # the whole point of the block — it is what replaced `ext/MantleVulkanExt.jl`
    # — and it is the only spelling of a backend core gets. A third mention,
    # here or anywhere else, fails. (`Metal` and `include("vulkan/vulkan.jl")`
    # do not match the pattern, so the Metal arm of the same block needs no
    # entry; if the pattern ever widens, they belong here beside these two.)
    allowed = [("src/Mantle.jl", "Vulkan"), ("src/Mantle.jl", "Vulkan")]
    extra = sort(hits) != sort(allowed) ? hits : Tuple{String,String}[]
    isempty(extra) || @info "0.4 backend names in core" count = length(hits) first10 = first(hits, 10)
    @test sort(hits) == sort(allowed)
end

# ── 0.6 One device per process ───────────────────────────────────────────────
#
# `caps(::MetalBackend)` builds a second `MetalDevice` through a second cache
# (`_DEVICE` in `metal/caps.jl`) beside `METAL_DEVICE` in `metal/device.jl`:
# a second `Pool`, a second `MTLCommandQueue`, a second `MTLSharedEvent`.
# `metal/device.jl:65` forbids exactly that, and `metal/graphics.jl` records two
# Metal queues as the cause of a hang, because they have no order between them.
# Cleared by: phase 1.6.
#
# 0.5 (downstream names no backend) lives in the consumers' own suites; it
# cannot run from here, because Mantle does not depend on Hikari or RayMakie.

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
        # PROMOTED after phase 1.6 removed the second cache. A ratchet from here.
        @test same
    end
end

# ── 0.7 The shared test layer names no backend ───────────────────────────────
#
# `test/*.jl` looks shared and was not: it hardcoded `VulkanAPI()` 57 times
# against `MetalAPI()` once — `test_window.jl` 35 times in 2,021 lines,
# `test_arena_recording.jl` 10, `test_compile_golden.jl` 4,
# `test_devicerange.jl` 3. Windows, surfaces, resize, presentation, arena
# recording and device ranges are portable behaviour, and they were checked
# against exactly one backend. Both bugs a person found by looking at a window
# were in that blind spot, and `test_window.jl` is the file that should have
# caught them.
#
# Those four now take their device from `TESTBACKEND`, which `runtests.jl` sets
# once per `Mantle.eachbackend()`.
#
# The exceptions are named, not waved through:
#   * `runtests.jl` — the harness itself, which has to name the sections it
#     guards on.
#   * `test_host.jl` — its "Host and Vulkan devices coexist" testset is about
#     two named backends being usable at once, which cannot be asked of one.
#   * this file — 0.6 asks a Metal-specific question about a second device.

const BACKEND_NAMED_ALLOWED = Set([
    "runtests.jl",                # the harness names the sections it guards
    "test_host.jl",               # "Host and Vulkan devices coexist" needs two
    "test_mantle_owns_it.jl",     # 0.6 asks a Metal-specific question
    # Two of its assertions are Vulkan's: a `pool_offset` inside a VkBuffer and
    # `plan.recording isa Recording`. The second has a portable spelling
    # (`recordsplans`); the first belongs in `test/vulkan/`. A third was
    # `vk_flush!`, now core's `flush!(device)`. Splitting the rest is what is
    # left of 0.7, and this line goes then.
    "test_arena_recording.jl",
    # 2,000 lines with 26 backend sites, 16 of them raw `VK.` enums for image
    # layouts, load ops and aspects — genuinely that backend's, and they belong
    # under `test/vulkan/`. The portable half is already split out into
    # `test_window_portable.jl`, which runs per backend; splitting the rest is
    # the remainder of 2.8 and this line goes then.
    "test_window.jl",
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

# ── 0.7b The shared test layer names no backend-only BINDING ─────────────────
#
# 0.7 above asks whether a shared test hardcodes an API marker. It never asked
# what those tests NAME, and for as long as the backend was an extension it did
# not have to: a reach was spelled `MVE.LavaArray`, the module prefix announced
# itself, and `grep MVE\.` found all 1,960 of them. The extension is gone and
# the prefix with it — `Mantle.LavaArray` reads exactly like portable API.
#
# So ask the definition SITE, which is what the rule was always about and needs
# no vendor name in the pattern: is this a thing only `src/vulkan/` or
# `src/metal/` defines? That covers more than the prefix ever did, because it
# also catches a shared test that names a backend-private binding unqualified.
#
# 1,913 of the 1,960 reaches were in `test/vulkan/`, which is the backend's own
# test directory and exactly where they belong. The 47 in the shared layer are
# in the four files below, all of them already named by `BACKEND_NAMED_ALLOWED`
# with the split that retires them. This starts green and is a ratchet: it locks
# in that no OTHER shared test reaches into a backend.
#
# Functions and types only: those are the things with a definition site to ask
# about. A plain `const` (`GEMM_TILE`) and a module alias (`VK`, from `import
# Vulkan as VK`) have none, so three of the seventeen names the shared layer
# reaches for are not covered here — all three are in files on the allow-list.
# Asking about modules by any other means does not work: every `using`d package
# and every core submodule (`Transient`) is a module bound in `Mantle` too, and
# separating a driver from a dependency would mean naming the vendor.
#
# The token scan OVER-approximates: it does not know a binding from a reference,
# so a local named `acc` or `stage` counts as naming the backend function that
# happens to share the spelling. That is the right direction for a ratchet which
# starts green — a new entry is read by a person either way — and the precise
# extractor lives in `test_core_names_no_backend.jl`, which is a separate file
# included separately, so reaching for it here would be an ordering dependency.
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
        named = Set(Symbol(m.match) for m in eachmatch(r"\b[A-Za-z_][A-Za-z0-9_]*!?\b", code))
        bad = sort(collect(filter(backendonly, named)); by = string)
        isempty(bad) || push!(hits, (f, bad))
    end
    isempty(hits) || @info "0.7b backend-only bindings named in the shared test layer" hits
    @test isempty(hits)
end
