# A device type for fault injection: it forwards
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

mutable struct FaultDevice{D<:Mantle.Device} <: Mantle.Device
    inner::D
    runtime::Any                 # fresh core Runtime; test instrumentation only
    isolated::Bool               # native Throw requires a disposable fake domain
    schedule::Dict{Symbol,Vector{Fault}}
    calls::Dict{Symbol,Int}
    lock::ReentrantLock
end
function FaultDevice(inner::Mantle.Device; isolated = false)
    d = FaultDevice(inner, nothing, isolated, Dict{Symbol,Vector{Fault}}(),
                    Dict{Symbol,Int}(), ReentrantLock())
    d.runtime = Mantle.Runtime(d)    # no second native device or queue ownership
    return d
end
Mantle.runtime(d::FaultDevice) = d.runtime

# A fixed native headroom answer is not a fixed allocation ceiling: native
# free bytes exclude this process's existing allocations. Pressure fixtures
# therefore recompute headroom as pool reservations change.
struct BudgetLimit{D} <: Fault
    at::Int
    owner::D
    ceiling::Int
end
act(f::BudgetLimit, n) = n >= f.at ? Some(max(0, f.ceiling - Mantle.reserved(f.owner))) : nothing

"""Schedule `fault` for the device-type verb `verb` (e.g. `:compilekernel`)."""
function inject!(d::FaultDevice, verb::Symbol, fault::Fault)
    verb in (:submitnative!, :bindpages!, :unbindpages!, :presentnative!, :streamwaits!, :signal!) &&
        fault isa Throw && !d.isolated &&
        throw(ArgumentError("native failure requires an isolated fake device; use :preparesubmit for a recoverable fault"))
    return lock(d.lock) do
        # Copy on schedule mutation, never on each intercepted hot-path call.
        d.schedule[verb] = [get(d.schedule, verb, Fault[]); fault]
    end
end
calls(d::FaultDevice, verb::Symbol) = lock(() -> get(d.calls, verb, 0), d.lock)
"""Every verb call the FaultDevice has forwarded or answered so far."""
totalcalls(d::FaultDevice) = lock(() -> sum(values(d.calls); init = 0), d.lock)

# Count the call, then act on the first fault due at this count.
function due(d::FaultDevice, verb::Symbol)
    lock(d.lock)
    local n, fs
    try
        n = d.calls[verb] = get(d.calls, verb, 0) + 1
        fs = get(d.schedule, verb, nothing) # immutable after publication
    finally
        unlock(d.lock)
    end
    fs === nothing && return nothing
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

# Dispatch always keeps the wrapper owner. Native methods forward to the inner
# driver, while core allocation/accounting uses runtime(d), not runtime(d.inner).
macro forward(verb)
    quote
        function Mantle.$verb(d::FaultDevice, args...; kw...)
            r = due(d, $(QuoteNode(verb)))
            r === nothing ? Mantle.$verb(d.inner, args...; kw...) : something(r)
        end
    end
end
@forward timestampinfo
@forward preparetimestamps
@forward readtimestamps!
@forward compilekernel
@forward kernelinfo
@forward compiledraw
@forward sbtproperties
@forward shaderhandles
@forward compiletrace
@forward compilebindings
@forward rawalloc
@forward objectbytes
@forward destroy!
@forward imagerequirements
@forward vendorview
@forward videorequirements
@forward createdecoder
@forward openwindow
@forward devicearg
@forward flushhost!
@forward invalidatehost!
@forward externalcaps
@forward exportmemory
@forward importmemory
@forward exportsemaphore
@forward importsemaphore
@forward rawfree
@forward createreserve
@forward placeimage
@forward placeaccel
@forward accelsizes
@forward acquirenext
@forward createswapchain
@forward surfacesize
@forward memoryclasses
@forward budget
@forward caps
@forward queuekinds
@forward candidates
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
@forwardwindow surfacecaps Mantle.Window
@forwardwindow releaseacquired! Mantle.Swapchain
# Mantle.inner(x): the same window or swapchain seen through the inner device
# (a suite hook, P0.A8).

# A channel view owns no queue/timeline/lock/counter. Tokens and core locks use
# channelstate, so the view and the base endpoint cannot submit out of order.
struct FaultChannel{D,C} <: Mantle.SubmitChannel
    owner::D
    inner::C
end
Mantle.device(ch::FaultChannel) = ch.owner
Mantle.channelstate(ch::FaultChannel) = Mantle.channelstate(ch.inner)
function Mantle.openchannels(d::FaultDevice, kinds)
    r = due(d, :openchannels)
    inner = r === nothing ? Mantle.openchannels(d.inner, kinds) : something(r)
    return map(ch -> FaultChannel(d, ch), inner)
end
function Mantle.foreignchannel(d::FaultDevice, stream)
    r = due(d, :foreignchannel)
    ch = r === nothing ? Mantle.foreignchannel(d.inner, stream) : something(r)
    return FaultChannel(d, ch)
end
macro forwardchannel(verb)
    quote
        function Mantle.$verb(ch::FaultChannel, args...; kw...)
            r = due(ch.owner, $(QuoteNode(verb)))
            r === nothing ? Mantle.$verb(ch.inner, args...; kw...) : something(r)
        end
    end
end
@forwardchannel preparesubmit
@forwardchannel submitnative!
@forwardchannel passed
@forwardchannel waitfor
@forwardchannel waitnativeidle!
@forwardchannel recycle!
@forwardchannel bindpages!
@forwardchannel unbindpages!
@forwardchannel presentnative!
@forwardchannel streamwaits!
@forwardchannel signal!

struct FaultBuilder{D,B}
    owner::D
    inner::B
end
Mantle.device(b::FaultBuilder) = b.owner
function Mantle.Builder(d::FaultDevice, ch::FaultChannel)
    r = due(d, :Builder)
    b = r === nothing ? Mantle.Builder(d.inner, ch.inner) : something(r)
    return FaultBuilder(d, b)
end
macro forwardbuilder(verb)
    quote
        function Mantle.$verb(b::FaultBuilder, args...; kw...)
            r = due(b.owner, $(QuoteNode(verb)))
            r === nothing ? Mantle.$verb(b.inner, args...; kw...) : something(r)
        end
    end
end
@forwardbuilder emittimestamp!
@forwardbuilder emitdispatch!
@forwardbuilder emittrace!
@forwardbuilder emitcopy!
@forwardbuilder emitstore!
@forwardbuilder emitbuild!
@forwardbuilder beginrender!
@forwardbuilder emitpieces!
@forwardbuilder endrender!
@forwardbuilder bindindices!
@forwardbuilder emitdraw!
@forwardbuilder emitlibrary!
@forwardbuilder emitdecode!
@forwardbuilder finish
function Mantle.recordpiece(d::FaultDevice, node, region, emit)
    r = due(d, :recordpiece)
    r === nothing || return something(r)
    return Mantle.recordpiece(d.inner, node, region,
                             b -> emit(FaultBuilder(d, b)))
end
# Completion records also retain the instrument owner, rather than bypassing
# presentpassed when the release engine polls them.
struct FaultPresent{D,R}
    owner::D
    inner::R
end
function Mantle.presentpassed(record::FaultPresent)
    r = due(record.owner, :presentpassed)
    return r === nothing ? Mantle.presentpassed(record.inner) : something(r)
end
# Core wraps outcome.retirement with nativeowner(ch, record); this generic
# identity hook is used for completion objects, never for queue state.
Mantle.nativeowner(ch::FaultChannel, record) = FaultPresent(ch.owner, record)
failsubmit!(d::FaultDevice, at::Integer, err::Exception) =
    inject!(d, :preparesubmit, Throw(at, err))

# Completion wrappers are destroyed/accounted through their native record.
function Mantle.destroy!(d::FaultDevice, record::FaultPresent)
    r = due(d, :destroy!)
    r === nothing ? Mantle.destroy!(d.inner, record.inner) : something(r)
end
