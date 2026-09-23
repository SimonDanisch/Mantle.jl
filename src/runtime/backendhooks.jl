# What core asks a backend, outside the graph.
#
# `graph/backend.jl` holds the graph's own list and says why it is short. These
# two are not graph questions — one is the compiled-kernel cache, the other is
# which matrix kernels this build has — and they are here for the same reason
# that list exists: `src/vulkan/` and `src/metal/` are `@static include`d, so a
# name defined under one of them does not EXIST under the other. It is not an
# empty table or a disabled feature; there is no binding.
#
# Same shape as `initbackend!`, deliberately: **declared here, answered in
# `src/vulkan/` or `src/metal/`.** A core file that instead wrote the answer
# would have to name the backend's types to write it, which is what
# `test_core_names_no_backend.jl` catches.
#
# ── The alternative, and why it is not this ───────────────────────────────────
#
# A caller can reach into this module by name instead — `isdefined(Mantle,
# :GEMM_TILINGS)` — and that is what `DNNKernels.coopmatkernels` and fifteen
# runners' `__init__` did. It works. It also keeps working through a rename, a
# split of the table, or a second backend that implements the same feature at a
# different width and is then silently admitted, because the consumer is
# spelling the provider's internal names and nobody agreed to that contract.
#
# A backend that forgets one of these gets a `MethodError` naming the function
# it did not answer, which is the failure worth having: the `isdefined` version
# fails instead at `Mantle.GEMM_TILINGS`, two frames into a kernel, as an
# `UndefVarError` on a name the caller was never supposed to know.

"""
    use_frozen_kernels(version)

Read the kernel entries frozen under `version`, recording none.

What a *using* package calls at load time; `__init__` is the natural place. The
version has to match the one the workload froze under, which is why it belongs
in a `const` both refer to.

A backend that compiles kernels fresh each session answers by doing nothing —
nothing to read is not an error, it is no cache. Callers need no guard, and
before this was declared they all carried one, because an `UndefVarError` in an
`__init__` is an `InitError` that stops `using` the package at all.
"""
function use_frozen_kernels end

"""
    staged_gemm_tile() -> Union{Int,Nothing}

The square cooperative-matrix extent this BUILD's staged GEMM is emitted at, or
`nothing` if this build has no staged GEMM.

A caller that wants those kernels needs two answers and this is the second. The
first is about the machine and `KernelInterface.DeviceCaps` gives it: `coopmat`
says the driver has cooperative matrices, `tile` says how wide its square is.
Neither says whether MANTLE has kernels for that width, and neither can — a
`DeviceCaps` is a record of a device and carries no backend identity, which is
what makes it portable and what stops this being a dispatch. So both halves, and
the tile makes them one comparison:

    dev.coopmat && (t = Mantle.staged_gemm_tile()) !== nothing && dev.tile == t

Answering with the tile rather than with a `Bool` is the point of the `Int`: the
caller had grown its own copy of `16` to compare `dev.tile` against, and a
restated constant is a second place for the two to disagree.
"""
function staged_gemm_tile end
