# Array routines select kernels and allocate independent default workspace here.
# Eager submission and graph declaration consume the same launch descriptions.

"""
    ArrayLaunch(kern, args, ndrange, group = 0)

One kernel launch, chosen but not yet submitted. `group` is the workgroup size,
or `0` to let the backend pick one from `ndrange` (a count, a tuple of counts, or
a `DeviceRange`). A plain KernelInterface function carries no workgroup size in
its type, so `group` is where one comes from.
"""
struct ArrayLaunch{K,A}
    kern::K
    args::A
    ndrange::Any
    group::Int
end

ArrayLaunch(kern, args, ndrange) = ArrayLaunch(kern, args, ndrange, 0)


"""
    runlaunches!(backend, launches)

Submit now, in order. Through `invokelatest` because some consumers generate
kernels at runtime (DNNKernels flash kernels).
"""
function runlaunches!(backend, launches)
    for l in launches
        group = l.group == 0 ? nothing : l.group
        # A `@kernel` answers `kern(backend)` with the object to call; a plain
        # KernelInterface function goes through `kilaunch!`. `buildskernel`, not
        # `applicable(l.kern, backend)`: a one-argument kernel such as
        # `rngadvance_kernel!(state)` is applicable to a backend and would run on the host.
        if buildskernel(l.kern, backend)
            k = Base.invokelatest(kernelfor, l.kern, group, backend)
            Base.invokelatest(k, l.args...; ndrange = l.ndrange)
        else
            Base.invokelatest(kilaunch!, backend, l.kern, l.args...; ndrange = l.ndrange,
                              workgroupsize = group)
        end
    end
    return nothing
end

"""Launch a plain KernelInterface function through the device's retained kikernel,
passing original arguments so native emitters can retain their owners."""
function kilaunch!(backend, f, args...; ndrange, workgroupsize = nothing)
    k = kikernel(f, todevice(backend), args)
    if workgroupsize === nothing
        k(args...; ndrange)
    else
        k(args...; ndrange, workgroupsize)
    end
    return nothing
end

"""
    dispatchlaunches!(g, launches; name) -> nothing

Declare the same launches into a graph, one pass each, named `name`, or `name.1`,
`name.2`, … when there are several, so a profile tells a split-K GEMM's
accumulate from its reduce.
"""
function dispatchlaunches!(g, launches; name::AbstractString)
    n = length(launches)
    for (i, l) in enumerate(launches)
        # `dispatch!` calls the kernel constructor, which for a generated kernel
        # family may be newer than this method.
        Base.invokelatest(dispatch!, g, l.kern, l.args, l.ndrange;
                          group = l.group == 0 ? nothing : l.group,
                          name = n == 1 ? name : "$name.$i")
    end
    return nothing
end

# ── Dispatch grids ──────────────────────────────────────────────────────────
#
# A dispatch is at most 3-D and each axis has a device limit (`maxdims`, Vulkan's
# `maxComputeWorkGroupCount`, WebGPU's `maxComputeWorkgroupsPerDimension`).
# `pad_to_3d` maps a block grid onto it, splitting large dimensions across Y and Z;
# a kernel built through KernelAbstractions recovers its block from the dispatched
# grid (Lava's `linear_block_index`). Extra workgroups write out of bounds in
# kernels that index auxiliary arrays by workgroup id without checks (AK's
# `_accumulate_block!`), so splits use exact factors where they exist.

"""
A divisor of `n` within 3 of `max_dim`, else the smallest divisor in
`cld(n, max_dim):max_dim`, else `max_dim`, which over-dispatches.
"""
function find_split_factor(n::Int, max_dim::Int)
    for d in (max_dim, max_dim-1, max_dim-2, max_dim-3)
        d > 0 && n % d == 0 && return d
    end
    lo = cld(n, max_dim)
    for x in lo:min(n, max_dim)
        n % x == 0 && return x
    end
    return max_dim
end

"""
    pad_to_3d(maxdims, blocks) -> NTuple{3,Int}

The dispatch grid for a grid of `blocks` (any rank) on a device with at most
`maxdims` workgroups per axis.
"""
function pad_to_3d(maxdims::NTuple{3,Int}, t::NTuple{1,<:Integer})
    n = Int(t[1])
    max_x, max_y, _ = maxdims
    n <= max_x && return (n, 1, 1)
    x = find_split_factor(n, max_x)
    y = min(cld(n, x), max_y)
    z = cld(n, x * y)
    # Unsafe for kernels that index by workgroup id without bounds checks.
    x * y * z == n || @warn "pad_to_3d: over-dispatching $n as ($x, $y, $z) = $(x*y*z) workgroups" maxlog=1
    return (x, y, z)
end
function pad_to_3d(maxdims::NTuple{3,Int}, t::NTuple{2,<:Integer})
    x, y = Int(t[1]), Int(t[2])
    max_x, max_y, _ = maxdims
    if x <= max_x && y <= max_y
        return (x, y, 1)
    end
    return pad_to_3d(maxdims, (x * y,))
end
pad_to_3d(::NTuple{3,Int}, t::NTuple{3,<:Integer}) = (Int(t[1]), Int(t[2]), Int(t[3]))
# N > 3: drop trailing unit axes and flatten only if a non-unit axis remains past
# the third. This keeps the common `(W, H, C, 1)` grid 1:1 with the dispatch,
# which `directdispatch` needs to index without dividing.
function pad_to_3d(maxdims::NTuple{3,Int}, t::NTuple{N,<:Integer}) where N
    n = N
    while n > 3 && Int(t[n]) == 1
        n -= 1
    end
    n > 3 && return pad_to_3d(maxdims, (Int(prod(t)),))
    return pad_to_3d(maxdims, ntuple(i -> Int(t[i]), n))
end
