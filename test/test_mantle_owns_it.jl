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

"""
Every name DEFINED in a set of files or directories: a method, or a type.

Crude on the same terms as 0.8, and for the same reason: a definition is a
top-level `NAME(` or a top-level `struct`/`abstract type`/`primitive type NAME`,
at column zero. In-tree backends spell a method without a module prefix because
they are included INTO `Mantle`; an extension spells it `Mantle.NAME`. Leading
macros are skipped, which is not cosmetic: `supports_tessellation`'s core
default is `@doc (@doc supports_geometry_stage) supports_tessellation(b) = false`
and a pattern anchored on `function` or the bare name misses it.

Types are here because five vocabulary entries ARE types (`Framebuffer`,
`Sampler`, `Texture2D`, `Window`, `Profiler`), declared abstract in core and
constructed per backend. A method-table view could not see that: an abstract
type is not a `Function`, so the loop that asked `f isa Function` skipped all
five, and `methods(Framebuffer)` shows only the one backend's outer constructor.

A call at column zero would be read as a definition, which over-approximates
towards saying a name IS answered, so it can hide a hole but never invent one.
"""
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

"""
Each backend's files, keyed by backend: the trees under `src/`, plus every
extension that answers `caps`.

`caps` is the marker because it is the one verb a backend must answer to be a
device at all, so asking for it needs no vendor name and admits a future
extension that is not a backend without breaking this guard.

Files rather than method tables, which is the correction: this question is about
the REPO, and a method table only shows the backends this process loaded. Metal
never loads on the machines the Vulkan backend is tested on and the ROCm
extension loads on neither, so a method-table answer changes per machine and
`UNIMPLEMENTED_BACKEND_FUNCS` could not be a ratchet against it. It also read a method in
`ext/` as core's answer for everyone, which is how `caps`, `closerecording!`,
`defaultdevice!`, `devicename` and `devices` came to look portable.
"""
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

# ── 0.1 No vendor name in the portable vocabulary ────────────────────────────
#
# `BACKEND_VOCABULARY` is the list of core functions a backend may extend, so a
# vendor-prefixed entry in it is a portable hook that only one vendor can mean.
# `:vkformat` is one; Metal's counterpart is `mtlformat`. CLAUDE.md states the
# rule for code paths and this is a code path.

@testset "0.1 the vocabulary names no vendor" begin
    bad = filter(n -> occursin(r"^(vk|mtl|lava|VK_|MTL)"i, String(n)),
                 collect(Mantle.BACKEND_VOCABULARY))
    isempty(bad) || @info "0.1 vendor-named vocabulary entries" bad
    # A ratchet: the set is empty and stays empty.
    @test isempty(bad)
end

# ── 0.2 Every vocabulary name is answered by core, or by every backend ───────
#
# A name only one backend answers is either not portable (it belongs in that
# backend) or not implemented (it belongs on a list). Both are decisions; being
# in the vocabulary with one implementation is the absence of one.
#
# `BACKEND_SPECIFIC_FUNCS` is where a deliberate answer goes, with its reason. Not
# an escape hatch: a name belongs here only when "every backend answers it" is the
# WRONG requirement, not when it is merely unmet. An unmet one is
# `UNIMPLEMENTED_BACKEND_FUNCS` below.
#
# Both entries are BUILD-GLOBAL hooks, and that is the distinction. They take no
# device or backend argument -- `staged_gemm_tile()` takes nothing,
# `use_frozen_kernels(version)` takes a version -- so they are answered once per
# build, not once per device. `src/vulkan/` and `src/metal/` are `@static include`d
# on `Sys.isapple()` and are therefore mutually exclusive, which is the mechanism
# that makes a no-argument hook well defined. `src/host/` is ALWAYS included
# alongside whichever of those is compiled in, so a host definition is not a second
# method, it is the SAME signature: defining it there is `ERROR: Method overwriting
# is not permitted during Module precompilation`, and Mantle stops precompiling.
#
# So "every backend answers it" cannot be satisfied here, and is not the right ask.
#
# `staged_gemm_tile` could in principle dispatch on a backend. It must not: its
# consumers are `DNNKernels`' plan functions, which take a `DeviceCaps` and no
# backend precisely so they can be asked about hardware this machine is not -- the
# suite builds synthetic ones and asks the planner about them. Those vary the
# DEVICE; the BUILD is fixed, and which tile Mantle emits kernels at is a property
# of the build.
const BACKEND_SPECIFIC_FUNCS = Set{Symbol}([
    :staged_gemm_tile, :use_frozen_kernels
])

# The 53 that are lonely TODAY, so the guard can fail on a 54th.
#
# Five names left this list when the guard started reading FILES: `caps`,
# `closerecording!`, `defaultdevice!`, `devicename` and `devices` are answered
# by all three backends and always were. They were on it because the guard
# asked method tables, and neither Metal nor the ROCm extension loads on the
# machine this suite runs on, so both looked silent.
#
# `@test_broken isempty(lonely)` could never do that: a list of 58 passes as
# "still broken" whether it holds 58 or 580, so the guard reported the number and
# detected nothing. `emitkernel!` is the entry that proves the cost — a verb
# `graph/backend.jl` declares and only Lava answers, which was noted in two
# docstrings of the ROCm extension and then worked around instead of
# implemented, because a HIP capture made a different verb produce the right
# behaviour. A guard that named it as new would have been read; one naming it
# among 57 others was not.
#
# So: a name here is a known hole, and the test fails on anything NOT here —
# and also on anything here that has been fixed without being removed, so the
# list can only shrink.
const UNIMPLEMENTED_BACKEND_FUNCS = Set{Symbol}([
    :access, :acquire_next_image!, :allocate_batch_queue!, :batchqueue,
    :begin_pass!, :begin_render_pass!, :beginframe!, :bind_textures,
    :blittarget, :build_accel!, :colorimage, :compile_draw, :currentimage,
    :depthimage, :destroyrecording!, :deviceof, :draw_in_pass!,
    :draw_indexed_in_pass!, :draw_indirect_in_pass!, :emitkernel!, :end_pass!,
    :end_render_pass!, :imageusage, :indirectslot, :initbackend!, :layout,
    :makeimage, :makerecording, :present_frame!, :readback_framebuffer,
    :readback_window, :record_draw!, :recorder, :refit_tlas!,
    :release_batch_queue!, :remakeimage!, :reset_device!, :resetrecording!,
    :screenshot, :set_anyhit_pipeline!, :set_viewport!, :setviewport!,
    :stages, :storebytes!, :submit!, :trace_closest_hits!,
    :trace_closest_hits_anyhit!, :trace_closest_hits_anyhit_indirect!,
    :trace_closest_hits_indirect!, :trace_rays!, :trace_rays_indirect!,
    :transition_image!, :use_bindings!
])

@testset "0.2 a vocabulary name has a core default or every backend" begin
    bysrc = backendsources()
    defs = Dict(b => definednames(ps) for (b, ps) in bysrc)
    core, _ = sourcefiles()
    coredefs = definednames(core)
    backendfiles = reduce(vcat, values(bysrc))
    isbackendfile(f) = any(b -> occursin(b, string(f)), backendfiles)

    # Two questions, each asked of the source that can answer it.
    #
    # "Does every backend answer this name" is about the REPO, so it reads
    # files: Metal does not load on the machines the Vulkan backend is tested
    # on, and a method-table answer would therefore call every graphics verb
    # unanswered on Linux and answered on a Mac. `UNIMPLEMENTED_BACKEND_FUNCS` cannot ratchet
    # against a number that moves per machine.
    #
    # "Is there a default for everyone" is NOT only about this repo: `caps`,
    # `supports`, `bestshape`, `matrix_shapes` and `wggranularity` are
    # `import`ed from KernelInterface, so whatever fallback they have is in a
    # package that loads whenever Mantle does. That one reads the method table,
    # and reads it safely — an unloaded backend contributes no methods, so it
    # can never make a name look defaulted when it is not.
    function hasdefault(n)
        n in coredefs && return true
        isdefined(Mantle, n) || return false
        v = getglobal(Mantle, n)
        (v isa Function || v isa Type) || return false
        return any(m -> !isbackendfile(m.file), methods(v))
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
        # A ratchet: one device, one cache.
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
# what those tests NAME. With the backend in an extension a reach is spelled
# `MVE.LavaArray` and the module prefix announces itself; compiled in, there is
# no prefix and `Mantle.LavaArray` reads exactly like portable API.
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

# ── 0.8 a hook whose default is a silent no-op ───────────────────────────────
#
# The shape that produced the worst bug this backend has had. `resource_moved!`
# was a vocabulary verb with a PERMISSIVE CORE DEFAULT — one accepting any
# argument, returning `nothing` — that Vulkan implemented and Metal did not. Core
# announced every buffer move through it; on Metal the announcement went nowhere,
# so a recorded plan kept reading a `resize!`d buffer's old storage. Silently: a
# retired region is not freed, so it returned well-formed stale numbers, and the
# whole suite was green throughout. A passing suite is NOT evidence against this
# class, which is exactly why it wants a ratchet.
#
# What this pins is the SET, not zero. Some of these are legitimate — a capability
# one backend has, or machinery only one submission model uses — and the honest
# state is a list that does not grow without someone looking. Audited 2026-09-14,
# all by design:
#
#   emitpreparebarrier!, emitkernel!, workgroupsize
#       the device-sized-dispatch prepare. Metal records a `DeviceRange` at its
#       CEILING and bounds-checks in the kernel, so there is no indirect command
#       to write and no prepare to order — see `recordedrange`.
#   hold!, holdleaves!, stampof
#       `SubmitChannel`, which is Vulkan's submission model. Metal holds through
#       its batch's roots instead (`retire!(::Pool, ::MTLBuffer)`).
#   supports_*  capability questions, where the answer legitimately differs.
#
# A NEW name here is the thing to look at: ask whether the backend that does not
# implement it reaches the call, and whether the default is right when it does.
@testset "0.8 hooks that default to silence are a known set" begin
    corefile(f) = startswith(string(f), joinpath(ROOT, "src")) &&
                  !any(d -> occursin(joinpath("src", d), string(f)), BACKEND_DIRS)

    # `unwrap_unionall`: a parametric method's `sig` IS a `UnionAll` and has no
    # `.parameters` at all.
    # The catch-alls, ENUMERATED. `t isa UnionAll` was the test here and it is a
    # false positive on every bare parametric type: `::Launch`, `::Call`,
    # `::CompiledDispatch` and `::CompiledDraw` are all `UnionAll`s, so twelve
    # perfectly specific dispatches counted as permissive defaults —
    # `argsize`, `argument_usage`, `callgroup`, `collect!`, `devicesized`,
    # `extrausage`, `finishrecording!`, `hold!`, `holdleaves!`, `indirectindex`,
    # `initial_usage`, `workgroupsize`. That is also why this number drifted from
    # 26 without anyone being able to act on it: core grew parametric types, and
    # each one inflated the count without adding any silent-default risk.
    #
    # `Type` stays, and is why this is a list rather than `t === Any`: it is a
    # `UnionAll` AND a genuine catch-all — `argument_usage(F::Type, A::Type) =
    # nothing` answers for every function there is.
    permissive(m) = corefile(m.file) &&
        all(t -> t === Any || t === Mantle.Device || t === Type,
            collect(Base.unwrap_unionall(m.sig).parameters)[2:end])

    # Read the backend TREES, not the loaded methods. Only one backend is compiled
    # into any build (chosen at parse time), so `methods()` can never see the other
    # one — and "exactly one backend answers" would then be trivially true for
    # everything. Both source trees are on disk either way, so the answer here is
    # the same on a Metal build and a Vulkan one, which is what a ratchet needs.
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
    # 26 on both builds, 2026-09-14. Reading the trees rather than the loaded
    # methods is what makes that number the same in each; if it ever differs by
    # build, this test has drifted back to asking about the compiled backend.
    #
    # 28 after the recording verbs were declared, 2026-09-15: 25 of the 26, plus
    # the three below. `Surface` left the set in the same merge — the portable
    # `Window(width, height)` and `color_format` work made both trees answer it.
    #
    # The three, each looked at as the note above this testset asks:
    #
    #   submitrecording!, callgroup
    #       ARTIFACTS of `permissive`, not holes. Their core methods take
    #       `RecordingParts` and `KernelAbstractions.Kernel` — parametric types
    #       named without their parameters, so each is a `UnionAll` and the
    #       predicate reads it as "accepts anything". Neither accepts anything:
    #       a `Call` or a `Launch` matches neither. Tightening the predicate
    #       would mean deciding that a bare `::Plan` is specific while `::Any`
    #       is not, which is a judgement this ratchet should not make silently,
    #       so they are recorded here instead.
    #   abandonrecording!
    #       REAL, and narrow. `abandonrecording!(::Device, e) = nothing` is a
    #       no-op default; Vulkan and the ROCm extension release the recording,
    #       Metal does not implement it. `recordparts!` calls it when a
    #       partition throws part way through, so on Metal that path leaks the
    #       open indirect command buffer. An error path, and a leak rather than
    #       a wrong answer — but it is the `resource_moved!` shape and the fix
    #       is one method on `MetalRecorder`, on a machine that can run it.
    #
    # 31 after the access walk was declared, 2026-09-16: the 28 above plus
    # `argtype`, `isdevicearray` and `accesscache`. Looked at together, because
    # the same fact answers all three — **Metal does not implement
    # `kerneltouches`**, so no part of the walk runs there and none of these
    # three is reached on that build:
    #
    #   isdevicearray
    #       The one with the `resource_moved!` shape if it were reachable.
    #       `isdevicearray(x) = false` makes `resourceleaves!` descend into a
    #       value's FIELDS instead of recording it, so a Metal array answering
    #       the default would drop a declaration rather than raise anything.
    #   argtype
    #       `typeof(y)` is not a permissive default but the RIGHT answer for a
    #       backend that hands a kernel what it resolved, which is what the host
    #       does. Only a backend that adapts on the way to the shader has to
    #       answer, and one that adapts has a `kerneltouches` to answer it from.
    #   accesscache
    #       `nothing` means "no cache", which costs the analysis again and
    #       cannot be wrong. The host answers it; Metal need not.
    #
    # So the note for whoever brings the walk to Metal: `kerneltouches` is not
    # one method, it is four, and `isdevicearray` is the one that fails quietly.
    #
    # 30 after merging the Metal frame-allocation work in the same session: it
    # answers `devicearray` in that tree, so the name left the set. Which is the
    # direction this number is supposed to move.
    #
    # 31 for `intrinsic_usage`, and this one is not a permissive default at all:
    # `nothing` means the symbol an `llvmcall` calls is UNDECLARED, which the
    # walk treats as "assume the worst" and is on its way to refusing outright.
    # A backend that spells its intrinsics as `llvmcall`s and does not answer
    # gets conservative declarations, not silence -- barriers it does not need,
    # never a missing one. Metal's intrinsics are `@device_override`s on real
    # instructions rather than external symbols, so it may never need a method;
    # if it does, the shape is Lava's in `src/vulkan/access.jl`.
    @test length(lonely) <= 31
end

# ── 0.9 An extension extends the vocabulary and nothing else ─────────────────
#
# `BACKEND_VOCABULARY` is the list of core functions a backend may extend, so a
# backend method on a name outside it is a decision taken in the wrong package:
# either core has no opinion where it should have one, or the backend is
# answering a question core never asked. The in-tree backends are checked by
# 0.2 and 0.3 from their method tables; a backend that ships as a package
# EXTENSION has no method table here, because the package it weakly depends on
# is not installed on most machines this suite runs on.
#
# So this reads the files. Crude on purpose: a top-level `Mantle.NAME(` or
# `function Mantle.NAME(` is a definition, and every other mention is a call.
# That is enough, because the thing being hunted is a `Mantle.foo(...) = ...`
# for a `foo` that no other backend answers and core never declared — which is
# how an extension grows its own private API on Mantle's namespace and how the
# pair of `resource_moved!`/`arena_moved!` methods that were DEFINING two dead
# functions survived as long as they did.
#
# No extension is named here and none is special-cased: the directory is the
# subject. Starts green and is a ratchet.
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
