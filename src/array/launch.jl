# Array routines select kernels and allocate independent default workspace here.
# Eager submission and graph declaration consume the same launch descriptions.

"""
One kernel launch, chosen but not yet submitted.

`group` is the workgroup size, or `0` to let the backend pick one from the
`ndrange`. `ndrange` is whatever the launching side accepts: a count, a tuple of
them, or a `DeviceRange`.

A kernel here is a plain function over `KernelInterface`'s intrinsics, so it is
launched through [`kilaunch!`](@ref) and declared by handing the same function to
`dispatch!`. There is no kernel object carrying a workgroup size in its type,
which is why `group` is the only place one can come from.
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

Submit now, in order.

Use invokelatest for consumers that still generate kernels at runtime
(DNNKernels flash kernels); core GEMM/GEMV/FFT callables are already typed.
"""
function runlaunches!(backend, launches)
    for l in launches
        group = l.group == 0 ? nothing : l.group
        # MIGRATION SHIM. A plain function over `KernelInterface`'s intrinsics is
        # launched through `kilaunch!`; a `@kernel` is its own constructor
        # and answers `kern(backend)` with the object to call. The branch goes
        # away with the last `@kernel`.
        #
        # `buildskernel` and NOT `applicable(l.kern, backend)`: a one-argument
        # kernel — `rngadvance_kernel!(state)` — is applicable to a backend, so
        # that test sent it down the constructor path and ran the kernel body on
        # the host. Core's predicate is the one place that question is answered.
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

Declare the same launches into a graph: one pass each.

The passes are named `name` when there is one and `name.1`, `name.2`, … when a
routine takes several, so a profile can tell a split-K GEMM's accumulate from its
reduce.
"""
function dispatchlaunches!(g, launches; name::AbstractString)
    n = length(launches)
    for (i, l) in enumerate(launches)
        # `invokelatest` for the reason above: `dispatch!` calls the kernel
        # constructor, and for a generated kernel family that constructor may be
        # newer than this method.
        Base.invokelatest(dispatch!, g, l.kern, l.args, l.ndrange;
                          group = l.group == 0 ? nothing : l.group,
                          name = n == 1 ? name : "$name.$i")
    end
    return nothing
end
