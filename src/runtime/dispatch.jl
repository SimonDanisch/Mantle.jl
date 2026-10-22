# Shared dispatch records, device counts and portable copy/fill kernels.

"""
    Dispatch(kernel, args, ndrange, group)

One kernel launch inside a pass. Backends read the fields and record; nothing
here is backend-specific, so the record is shared.
"""
struct Dispatch
    kernel::Any
    args::Tuple
    ndrange::Any
    group::Any
end

"""DeviceRange(count; max=nothing)

A device-resident element count for an indirect dispatch. Core registers an
Indirect read and backends prepare native workgroup counts and ordering.
max is an optional static compilation ceiling. Kernels must guard i <= count[1]:
whole workgroups can launch padded tail invocations."""
struct DeviceRange{C}
    count::C
    max::Union{Nothing,Int}
end
DeviceRange(count; max = nothing) = DeviceRange(count, max)

"""The resource a `DeviceRange` reads its count from, or `nothing` for a host-side
ndrange. Backends use this to decide which recording path a dispatch takes."""
countresource(r::DeviceRange) = r.count
countresource(::Any) = nothing

"""refuserefs(args)

Reject Base.RefValue in tuple, NamedTuple or immutable-struct arguments. Recorded
arguments freeze at record!; use GPURef for values updated between runs. Stop at
resources, arrays and other mutable handles, whose internals have their own owners."""
refuserefs(::Base.RefValue) = throw(ArgumentError(
    "a Ref is not a dispatch argument. It would be read once, at record!, and never " *
    "again, which is not what a Ref is for: a value that changes between runs is a " *
    "GPURef (`r = GPURef(dev, x)`, then `r[] = x′` before a run), and a value that " *
    "does not is passed as itself."))
refuserefs(::Resource) = nothing
refuserefs(::AbstractArray) = nothing
@generated function refuserefs(x::T) where {T}
    isbitstype(T) && return :(nothing)
    (T <: Tuple || T <: NamedTuple) && return Expr(
        :block, Expr[:(refuserefs(x[$i])) for i in 1:fieldcount(T)]..., :(nothing))
    (T <: Type || T <: Symbol || T <: AbstractString || T <: Module ||
     T === Nothing || T <: Ptr || ismutabletype(T)) && return :(nothing)
    n = fieldcount(T)
    n == 0 && return :(nothing)
    Expr(:block, Expr[:(refuserefs(getfield(x, $i))) for i in 1:n]..., :(nothing))
end

"""passof(handle) -> pass
    graphof(handle) -> graph

Declaration-handle accessors; the defaults read .pass and .graph."""
passof(h) = h.pass
graphof(h) = h.graph

# `resourceid(::Graph, r)` is in `graph/build.jl`: it names the concrete
# `Graph`, which is defined after this file.

"""touch!(graph, x) -> x

Register x with the graph. Transient specializations also extend its live interval."""
touch!(g, x) = (resourceid(g, x); x)

# Graphs declare work before placement and scheduling.

# Declared copies/fills use these kernels; eager array operations choose native transfers.

"""
    d2dcopy_kernel!(dst, src, doff, soff, n)

`n` elements from `src` at `soff` into `dst` at `doff`, as a dispatch. Offsets
are ZERO-based, so a caller passes `first - 1`.
"""
function d2dcopy_kernel!(dst, src, doff::Int64, soff::Int64, n::Int64)
    i = KI.get_global_id().x
    if i <= n
        @inbounds dst[doff + i] = src[soff + i]
    end
    return nothing
end

"""    fill_kernel!(a, v)

`v` into every element of `a`, as a dispatch."""
function fill_kernel!(a, v)
    i = KI.get_global_id().x
    # Guard padded workgroup tails.
    i <= length(a) || return nothing
    @inbounds a[i] = v
    return nothing
end

"""dispatch!(pass, kernel, args, ndrange; group=nothing)

Append a kernel launch to the pass and register any DeviceRange count as an
Indirect read. group selects workgroup shape; use a suitable 2D shape for images."""
function dispatch!(p, kernel, args, ndrange; group = nothing)
    refuserefs(args)
    n = countresource(ndrange)
    n === nothing || indirectcount!(p, n)
    push!(dispatches(passof(p)), Dispatch(kernel, args, ndrange, group))
end

"""dispatch!(graph, call, args; name=nothing)
    dispatch!(pass, call, args)

Declare a host/library call that submits its own work, with accesses inferred
from its body or argument_usage. Reject on backends where runscalls is false:
a call executed only while recording would disappear from replay. Walking plans
and native stream capture can support this form; Vulkan command recording cannot."""
function dispatch!(p, call, args)
    refuserefs(args)
    push!(dispatches(passof(p)), Dispatch(call, args, nothing, nothing))
end

"""True for a call record: ndrange=nothing means the callable submits its own work."""
iscall(d::Dispatch) = d.ndrange === nothing

"""indirectcount!(pass, count)

Register an Indirect read and touch the count resource. Its command-processor
stage/access differs from an ordinary shader storage read."""
function indirectcount!(p, n)
    push!(passof(p).usages, resourceid(graphof(p), n) => Indirect)
    touch!(graphof(p), n)
    return n
end
