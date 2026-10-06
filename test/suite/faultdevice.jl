# A device type for fault injection (catalogue FI-1..FI-10): it forwards
# every device-type verb (docs/api.md section 4) to a real device and, on a
# schedule the test sets, throws, blocks, or answers differently instead.
# No globals and no probes in Mantle: the schedule is a field of the device
# object, and core cannot tell this device from any other.

"""What a scheduled verb call does instead of (or before) forwarding."""
abstract type Fault end
struct Throw <: Fault          # throw `err` on the `at`-th call
    at::Int
    err::Exception
end
struct Block <: Fault          # block on the `at`-th call until the channel is fed
    at::Int
    wait::Channel{Nothing}
end
struct Answer <: Fault         # return `value` (from the `at`-th call on) instead of forwarding
    at::Int
    value::Any
end

struct FaultDevice{D<:Mantle.Device} <: Mantle.Device
    inner::D
    schedule::Dict{Symbol,Vector{Fault}}
    calls::Dict{Symbol,Int}
    lock::ReentrantLock
end
FaultDevice(inner::Mantle.Device) = FaultDevice(inner, Dict{Symbol,Vector{Fault}}(), Dict{Symbol,Int}(), ReentrantLock())

"""Schedule `fault` for the device-type verb `verb` (e.g. `:compilekernel`)."""
inject!(d::FaultDevice, verb::Symbol, fault::Fault) =
    lock(() -> push!(get!(Vector{Fault}, d.schedule, verb), fault), d.lock)
calls(d::FaultDevice, verb::Symbol) = lock(() -> get(d.calls, verb, 0), d.lock)
"""Every verb call the FaultDevice has forwarded or answered so far."""
totalcalls(d::FaultDevice) = lock(() -> sum(values(d.calls); init = 0), d.lock)

# Count the call, then act on the first fault due at this count.
function due(d::FaultDevice, verb::Symbol)
    n, fs = lock(d.lock) do
        n = d.calls[verb] = get(d.calls, verb, 0) + 1
        n, copy(get(d.schedule, verb, Fault[]))
    end
    for f in fs
        r = act(f, n)
        r === nothing || return r
    end
    return nothing
end
struct AnswerOnce <: Fault      # return `value` on the `at`-th call only
    at::Int
    value::Any
end
act(f::Throw, n) = n == f.at ? throw(f.err) : nothing
act(f::AnswerOnce, n) = n == f.at ? Some(f.value) : nothing
act(f::Block, n) = n == f.at ? (take!(f.wait); nothing) : nothing
act(f::Answer, n) = n >= f.at ? Some(f.value) : nothing

# Forward a verb whose first argument is the device. The verbs whose first
# argument is a builder or a channel reach the inner device through the
# builder and channel core made for this device (Builder(d, ch) and the
# channels are the inner device's), so faults are injected at the verbs that
# take the device: compile, allocate, acquire, surface, budget, caps.
macro forward(verb)
    quote
        function Mantle.$verb(d::FaultDevice, args...; kw...)
            r = due(d, $(QuoteNode(verb)))
            r === nothing ? Mantle.$verb(d.inner, args...; kw...) : something(r)
        end
    end
end
@forward compilekernel
@forward kernelinfo
@forward compiledraw
@forward compiletrace
@forward compilebindings
@forward recordpiece
@forward rawalloc
@forward rawfree
@forward createreserve
@forward placeimage
@forward placeaccel
@forward accelsizes
@forward acquirenext
@forward createswapchain
@forward surfacesize
@forward budget
@forward caps
@forward queuekinds
@forward candidates
@forward hardwarekeys
@forward imageformat
@forward instancerecordtype
@forward countsource
@forward config
@forward acceladdress
@forward tableentry!
@forward samplerentry!

# Window verbs take the window or its swapchain first (api.md 4.7). Both are
# parameterized by their device's type (a suite hook, P0.A8), so a window made
# on a FaultDevice reaches these methods.
macro forwardwindow(verb, T)
    quote
        function Mantle.$verb(x::$T{<:FaultDevice}, args...; kw...)
            d = Mantle.device(x)
            r = due(d, $(QuoteNode(verb)))
            r === nothing ? Mantle.$verb(Mantle.inner(x), args...; kw...) : something(r)
        end
    end
end
@forwardwindow acquirenext Mantle.Swapchain
@forwardwindow createswapchain Mantle.Window
@forwardwindow surfacesize Mantle.Window
# Mantle.inner(x): the same window or swapchain seen through the inner device
# (a suite hook, P0.A8).

# Channels and builders belong to the inner device; submitnative! is reached
# through the channel, so FI-6 (a submission that throws) is injected with
# `failsubmit!`, which marks the channel core made for this device.
Mantle.Builder(d::FaultDevice, ch) = Mantle.Builder(d.inner, ch)
failsubmit!(d::FaultDevice, at::Integer, err::Exception) = inject!(d, :submitnative!, Throw(at, err))
