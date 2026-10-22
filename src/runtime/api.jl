# Portable runtime vocabulary; graphs and native backends provide concrete methods.

"""Device(api)
    Device(api; select)
    Device(backend)

A device owns its graphs, resources and submit channels. Device(api) returns its
cached default; select creates an independent device without installing it.
Device(backend) returns the cached device of that backend's context.
See devices, selectdevice and defaultdevice!."""
abstract type Device end
abstract type Resource end
# Graph and Plan are shared concrete types in graph/types.jl.

# DeviceCaps and caps are imported from KernelInterface; all device handles share them.

"""todevice(x) -> Device

Return a Device unchanged, or resolve a KernelAbstractions.Backend to its cached
device. Other argument types are rejected by dispatch."""
todevice(d::Device) = d
todevice(b::KernelAbstractions.Backend) = Device(b)

"""Window(width, height; title="", vsync=false)
    Window(backend, width, height; title="", vsync=false, color_format=BGRA{N0f8}, srgb=false)

A presentation window; the short form uses defaultbackend(). isopen only queries
state. run! pumps events, submits and presents; frame slots provide pacing.
By default shaders write display-referred values; srgb=true encodes linear values."""
abstract type Window end

Window(width::Integer, height::Integer; kw...) =
    Window(defaultbackend(), width, height; kw...)

"""backend(device)

The KernelAbstractions backend for launches outside a graph. A backend returns
itself. Within a graph, dispatch! declares accesses and derives ordering."""
function backend end

backend(b::KernelAbstractions.Backend) = b

"""screenshot(window) -> Matrix

Read window pixels in its format's element type, synchronizing internally and
returning the acquired image to presentation. See readback_window."""
function screenshot end

"""Attachment load policy: Keep loads, Clear(value) overwrites, and Discard skips
loading because the pass promises to write every pixel. Clear and Discard allow
prior contents to be discarded during layout transitions."""
abstract type LoadOp end

struct KeepOp <: LoadOp end
struct DiscardOp <: LoadOp end
struct Clear{T} <: LoadOp
    value::T
end

"""Load what is there. What a bare target means."""
const Keep = KeepOp()

"""Do not load. The pass is asserting it writes every pixel."""
const Discard = DiscardOp()

"""Does this load op throw away the previous contents?"""
discards(::KeepOp) = false
discards(::DiscardOp) = true
discards(::Clear) = true

function Surface end
function Attribute end
"""draw!(pass, pipeline, args, count; frag_args=())

Declare a draw with vertex arguments or fragment arguments. A fragment receives
varyings before frag_args. The pipeline has one argument range: use the stage
that reads the arguments, or the same tuple for both. Fragment buffers use use,
not the vertex-binding Attribute wrapper."""
function draw! end

function render! end

"""newpass(graph, name, kind) -> pass
    handle(graph, pass) -> declaration handle
    dispatches(pass) -> recorded operations
    passes(graph) -> passes in declaration order

Shared pass-declaration accessors; concrete Graph/Pass methods live in graph/build.jl."""
function newpass end
function handle end
function dispatches end

"""checklive(plan, regions, ntransients)

Reject a freed plan: a plan declaring transients requires live pool regions
before it can record or run."""
function checklive(plan, regions, ntransients::Integer)
    (ntransients > 0 && isempty(regions)) && throw(ArgumentError(
        "this plan has been freed — `free!` gave its arenas back, and running it " *
        "now would record against memory the pool may have handed to another " *
        "plan. Build a new plan from the graph."))
    return plan
end

"""Identity-based resource IDs for scheduling, with a reverse id-to-resource table.
IDs are assigned in first-use order and remain stable within the graph."""
struct IdTable
    ids::IdDict{Any,Int}
    by_id::Dict{Int,Any}
end
IdTable() = IdTable(IdDict{Any,Int}(), Dict{Int,Any}())

"""The id `r` is known by, assigning one if this is the first time it is named."""
resourceid(t::IdTable, r) = get!(t.ids, r) do
    id = length(t.ids) + 1
    t.by_id[id] = r
    id
end

"""What an id names. The inverse of [`resourceid`](@ref)."""
byid(t::IdTable, id::Integer) = t.by_id[id]
Base.haskey(t::IdTable, r) = haskey(t.ids, r)

"""Look up an existing resource ID. Unlike resourceid, this never registers a resource."""
Base.getindex(t::IdTable, r) = t.ids[r]
Base.get(t::IdTable, r, default) = get(t.ids, r, default)
Base.length(t::IdTable) = length(t.ids)

"""
    copy!(graph, name, dst, src)

A copy as a graph pass, so its layouts and ordering are derived rather than
stated.
"""
function copy! end
# `dispatch!` is a graph verb and lives with the graph, in `graph/build.jl`.

function run! end
function npipelines end

"""free!(plan)

Release a plan's pool regions and deregister its arena tenancy. Explicit cleanup
is required; dropping a plan does not release its regions through a finalizer."""
function free! end

"""Per-pass host recording time and GPU execution time. These overlap across
passes and frames; their sum is not frame latency."""
struct PassTiming
    name::String
    kind::Symbol
    host_ms::Float64
    gpu_ms::Float64
    samples::Int
end

Base.show(io::IO, t::PassTiming) =
    print(io, rpad(t.name, 16), lpad(round(t.host_ms; digits = 3), 8), " ms host",
          lpad(round(t.gpu_ms; digits = 3), 8), " ms gpu")

function Base.show(io::IO, ::MIME"text/plain", ts::Vector{PassTiming})
    println(io, rpad("pass", 16), rpad("kind", 9), lpad("host ms", 9), lpad("gpu ms", 9),
            lpad("n", 6))
    for t in ts
        println(io, rpad(t.name, 16), rpad(t.kind, 9),
                lpad(round(t.host_ms; digits = 3), 9),
                lpad(isnan(t.gpu_ms) ? "-" : round(t.gpu_ms; digits = 3), 9),
                lpad(t.samples, 6))
    end
end

"""timings(plan) -> Vector{PassTiming}

Available for Plan(g; profile=true). Report medians over up to NSAMPLES completed
samples, in scheduled pass order. Timestamp results are polled without waiting;
not-ready frames are skipped. Inspect PassTiming.samples and warm the workload
before comparing configurations."""
function timings end

"""importhost(backend, A::DenseArray) -> device array

Import A's host allocation for direct device access without a device copy.
Keep A alive and unmoved while used. Query canimporthost first; unsupported
backends throw. Vulkan requires imported-pointer alignment at both ends; the RADV
import path was qualified with private anonymous memory. Use importablehost there.
Discrete devices read imported memory across the bus."""
importhost(backend, A::DenseArray) = throw(ArgumentError(
    "importhost: $(nameof(typeof(backend))) cannot read host memory in place " *
    "(canimporthost(backend) is false); upload it instead"))

"""importablehost(T, n) -> Vector{T}

Allocate private anonymous, page-aligned host storage suitable for importhost.
On Linux request transparent huge pages to reduce GPU address-translation cost.
Suitability and bandwidth still depend on the backend and working set."""
function importablehost(::Type{T}, n::Integer) where {T}
    a = Mmap.mmap(Vector{T}, n; shared = false)
    @static if Sys.islinux()
        MADV_HUGEPAGE = Cint(14)
        r = ccall(:madvise, Cint, (Ptr{Cvoid}, Csize_t, Cint), a, sizeof(a), MADV_HUGEPAGE)
        r == 0 || @warn("madvise(MADV_HUGEPAGE) failed (errno $(Libc.errno())): imported memory " *
                        "is read in 4 KiB pages, at about two thirds of the bandwidth", maxlog = 1)
    end
    return a
end

"""
    canimporthost(backend) -> Bool

Whether [`importhost`](@ref) works on `backend`.
"""
canimporthost(backend) = false

"""Maximum completed samples retained per profiled pass."""
const NSAMPLES = 120

"""stride(x) -> Int

Elements per logical index. Zero broadcasts one value: shaders read
1 + stride * (i - 1), sharing the same pipeline for scalar and array operands."""
function stride end

"""Number of elements a draw covers, taken from the binding so it cannot disagree."""
function count end

"""
Transient resources. Constructing one against a graph or a pass means the same
thing, because liveness derives the interval from use either way.
"""
module Transient
function Buffer end
function Image end
end
