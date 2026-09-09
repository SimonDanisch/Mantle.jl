# Which backend to use when the caller did not say.
#
# Mantle is the portable API, so "the default device" cannot be a name from one
# driver. It used to be, in every downstream caller: RayMakie's `ScreenConfig`
# and its precompile workload both wrote `Mantle.LavaBackend()`, which is a
# Vulkan type living in an extension — so the default was simultaneously the
# wrong choice on a Mac and a name that did not resolve there.
#
# A backend registers a probe when its extension loads. The probe returns the
# backend object if this machine can actually use it and `nothing` if it cannot,
# and it must not throw: "is there a GPU" is a question, not an error. Highest
# priority that answers wins.

struct BackendProbe
    priority::Int
    name::Symbol
    probe::Any
end

# ── Which device ──────────────────────────────────────────────────────────────
#
# `Device(api)` is the process default and the ONE ambient thing in Mantle. It is
# chosen once, lazily, by the `MANTLE_DEVICE` environment variable when that is
# set and by the backend's ranking otherwise, and changed with `defaultdevice!`.
# Everything else names its device: `Device(api; select)` builds one the caller
# holds, `Device(backend)` is the device a backend already belongs to, and every
# constructor takes one of those. The selector vocabulary below is shared by the
# backends so that "NVIDIA", `2` and `info -> info.kind == :cpu` mean the same
# thing on each of them.

"""
    DeviceInfo

One physical device a backend can build a [`Device`](@ref) on: its `index` in
[`devices`](@ref)' answer, its `name`, its `kind` (`:discrete`, `:integrated`,
`:virtual`, `:cpu` or `:other`) and the `driver` behind it.
"""
struct DeviceInfo
    index::Int
    name::String
    kind::Symbol
    driver::String
end

"""
    devices(api) -> Vector{DeviceInfo}

The physical devices `api` can build a device on, in enumeration order. What
`Device(api; select)` and the `MANTLE_DEVICE` variable select from.
"""
function devices end

"""
    defaultdevice!(dev) -> dev

Make `dev` the device its API answers `Device(api)` with from now on, and the
one `defaultbackend()` reports. The previous default is not torn down: it stays
a device of its own, and everything built on it keeps working.

    dev = Device(VulkanAPI(); select = "NVIDIA")
    defaultdevice!(dev)
"""
function defaultdevice! end

"""
    selectdevice(select, infos) -> Int

The index in `infos` that `select` names. `nothing` admits every device, a
string admits those whose name contains it (case-insensitive), an integer
admits the device at that index, and anything else is called as a predicate over
`DeviceInfo`. Among the admitted, the ranking is `:discrete` before
`:integrated` before `:virtual` before `:cpu` before `:other`, enumeration order
within a rank, and the first wins; so `"AMD"` on a box with a Radeon and an APU
is the Radeon, and `nothing` is the best GPU there is. Throws naming every
device when nothing is admitted.
"""
function selectdevice(select, infos::AbstractVector{DeviceInfo})
    admitted = filter(i -> admits(select, i), infos)
    if isempty(admitted)
        listing = join(("  $(i.index): $(i.name) ($(i.kind), $(i.driver))" for i in infos), "\n")
        throw(ArgumentError("Mantle: no device matches $(repr(select)). The devices are:\n" * listing))
    end
    rank = Dict(:discrete => 0, :integrated => 1, :virtual => 2, :cpu => 3, :other => 4)
    return first(sort(admitted; by = i -> (rank[i.kind], i.index))).index
end
admits(::Nothing, ::DeviceInfo) = true
admits(s::AbstractString, i::DeviceInfo) = occursin(lowercase(s), lowercase(i.name))
admits(n::Integer, i::DeviceInfo) = i.index == n
admits(f, i::DeviceInfo) = f(i)::Bool

const BACKEND_PROBES = BackendProbe[]

"""
    register_backend!(probe; name, priority)

Register `probe` as a source of a default device.

`probe()` returns a backend object, or `nothing` when this machine cannot use
it — a missing driver, no discrete GPU, a simulator. It must not throw, because
[`defaultbackend`](@ref) asks every registered backend in turn and one that
raises would hide the ones behind it.

`priority` orders the candidates, highest first.
"""
function register_backend!(probe; name::Symbol, priority::Int)
    filter!(e -> e.name !== name, BACKEND_PROBES)
    push!(BACKEND_PROBES, BackendProbe(priority, name, probe))
    sort!(BACKEND_PROBES; by = e -> -e.priority)
    return nothing
end

"""
    defaultbackend() -> backend

The best GPU backend loaded in this session.

Throws when none is: an explicit failure naming what to load beats silently
falling back to `KA.CPU()` and leaving someone to wonder why their GPU renderer
takes four minutes a frame.
"""
function defaultbackend()
    for e in BACKEND_PROBES
        b = e.probe()
        b === nothing || return b
    end
    throw(ArgumentError(
        "Mantle: no GPU backend is available. Load one — `using Lava` (with a " *
        "Vulkan loader) or `using Metal` on an Apple GPU — or pass the backend " *
        "explicitly instead of relying on the default."))
end

"""
    availablebackends() -> Vector{Symbol}

The registered backends that report themselves usable here, best first.
"""
availablebackends() = Symbol[e.name for e in BACKEND_PROBES if e.probe() !== nothing]

"""
    eachbackend() -> Vector

Every backend usable here, as backend OBJECTS, best first — what
[`availablebackends`](@ref) names, ready to hand to `Device`.

For iterating: a test or a benchmark that checks portable behaviour runs the
same body against each of them and names none.

    for be in eachbackend()
        dev = Device(be)
        …
    end

That is not a nicety. Mantle's own suite hardcoded `VulkanAPI()` 57 times in
the files that look backend-neutral — `test_window.jl` 35 times in 2,021 lines
— so windows, surfaces, resize and presentation were only ever checked on one
backend, and the clip-space flip and a wrong drawable extent both reached a
person looking at a window instead of a failing test.

Empty when nothing is usable. `defaultbackend()` throws in that case because a
caller asking for THE device wants one; a caller iterating wants to run zero
times.
"""
eachbackend() = [b for b in (e.probe() for e in BACKEND_PROBES) if b !== nothing]
