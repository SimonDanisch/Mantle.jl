# What a library routine IS for a given shape, separately from running it.
#
# Mantle's array library (GEMM, GEMV, FFT) picks a kernel and a tiling from the
# operands, and it has to do that in two contexts: an immediate call submits the
# launches now, and a graph DECLARES them as passes. Those two must not be able
# to choose differently, so the choice is a function that returns launches and
# has no side effects, and the two contexts are two consumers of it.
#
# That shape was written three times in `vulkan/array/gemm.jl` before it was
# written once here -- `coopmat_gemm_launches`/`coopmat_gemm!`/
# `coopmat_gemm_dispatch!` and the scalar trio -- and each copy repeated the
# `group == 0` sentinel and the `kern(backend)` / `kern(backend, group)` split.
#
# Portable, and in `src/array/` rather than under a backend, because `gemv.jl`
# needs it and that file is one every backend loads.

"""
One kernel launch, chosen but not yet submitted.

`group` is the workgroup size, or `0` to let the backend pick one from the
`ndrange`. `ndrange` is whatever the launching side accepts: a count, a tuple of
them, or a `DeviceRange`.

A kernel here is a plain function over `KernelInterface`'s intrinsics, so it is
launched as `KI.Kernel(backend, kern)` and declared by handing the same function
to `dispatch!`. There is no kernel object carrying a workgroup size in its type,
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

`Base.invokelatest` because some of these kernels are `@eval`-generated on
demand (see `gemv_ncontig_kernel`), so the constructor can be newer than the
method calling it. It costs a dynamic dispatch per launch against a kernel that
runs for tens of microseconds.
"""
function runlaunches!(backend, launches)
    for l in launches
        # MIGRATION SHIM. A plain function over `KernelInterface`'s intrinsics is
        # launched as `KI.Kernel(backend, f)`; a `@kernel` is its own constructor
        # and answers `kern(backend)` with the object to call. The branch goes
        # away with the last `@kernel`.
        #
        # `buildskernel` and NOT `applicable(l.kern, backend)`: a one-argument
        # kernel — `rngadvance_kernel!(state)` — is applicable to a backend, so
        # that test sent it down the constructor path and ran the kernel body on
        # the host. Core's predicate is the one place that question is answered.
        if buildskernel(l.kern, backend)
            k = l.group == 0 ? Base.invokelatest(l.kern, backend) :
                               Base.invokelatest(l.kern, backend, l.group)
            Base.invokelatest(k, l.args...; ndrange = l.ndrange)
        else
            k = KI.Kernel(backend, l.kern)
            if l.group == 0
                Base.invokelatest(k, l.args...; ndrange = l.ndrange)
            else
                Base.invokelatest(k, l.args...; ndrange = l.ndrange,
                                  workgroupsize = l.group)
            end
        end
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
