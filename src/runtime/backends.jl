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
