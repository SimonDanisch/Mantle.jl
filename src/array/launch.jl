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
