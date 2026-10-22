# Platform backend probes and shared physical-device selection.

struct BackendProbe
    priority::Int
    name::Symbol
    probe::Any
end

"""Physical-device index, name, kind and driver, as returned by devices(api)."""
struct DeviceInfo
    index::Int
    name::String
    kind::Symbol
    driver::String
end

"""
    devices(api) -> Vector{DeviceInfo}

The physical devices `api` can build a device on, in enumeration order. What
`Device(select)`, `Device(api; select)` and `defaultdevice!(select)` select from.
"""
function devices end

"""defaultdevice!(dev) -> dev

Install dev as its API's Device(api) default. The previous device and its resources
remain usable. This does not change backend priority: defaultbackend() still
queries registered probes in order. Use backend(dev) to select that exact backend."""
function defaultdevice! end

"""selectdevice(select, infos) -> Int

Filter by nothing (all), index, case-insensitive name/driver substring, or a
DeviceInfo predicate. Choose discrete, integrated, virtual, CPU, then other;
enumeration index breaks ties. Throw with the device list if nothing matches."""
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
function admits(s::AbstractString, i::DeviceInfo)
    needle = lowercase(s)
    return occursin(needle, lowercase(i.name)) || occursin(needle, lowercase(i.driver))
end
admits(n::Integer, i::DeviceInfo) = i.index == n
admits(f, i::DeviceInfo) = f(i)::Bool

const BACKEND_PROBES = BackendProbe[]

"""register_backend!(probe; name, priority)

Register or replace a default-backend probe. It returns a usable backend or
nothing, must not throw, and is queried in descending priority order."""
function register_backend!(probe; name::Symbol, priority::Int)
    filter!(e -> e.name !== name, BACKEND_PROBES)
    push!(BACKEND_PROBES, BackendProbe(priority, name, probe))
    sort!(BACKEND_PROBES; by = e -> -e.priority)
    return nothing
end

"""defaultbackend() -> backend

Return the first usable registered backend by priority. Throw if none is available;
there is no implicit CPU fallback. Use backend(dev) for an explicitly chosen device."""
function defaultbackend()
    for e in BACKEND_PROBES
        b = e.probe()
        b === nothing || return b
    end
    throw(ArgumentError(
        "Mantle: no GPU backend is available. `using Mantle` compiles in the " *
        "one for this platform, so reaching here means the driver is missing " *
        "or exposes no usable device — not that a package was left unloaded."))
end

"""
    availablebackends() -> Vector{Symbol}

The registered backends that report themselves usable here, best first.
"""
availablebackends() = Symbol[e.name for e in BACKEND_PROBES if e.probe() !== nothing]

"""eachbackend() -> Vector

Return every usable registered backend, in priority order, for portable tests
and workloads. Empty when none is usable."""
eachbackend() = [b for b in (e.probe() for e in BACKEND_PROBES) if b !== nothing]
