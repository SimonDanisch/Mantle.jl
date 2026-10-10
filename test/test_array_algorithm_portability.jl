"""
How much of `src/vulkan/array/` is actually Vulkan's — measured, and ratcheted.

Step 5 of the split asks where the array algorithms should live. They sit in the
Vulkan backend, and if they stay there a second backend reimplements GEMM, FFT
and GEMV rather than inheriting them. So the question is how much of each file is
really Vulkan, and the answer turns out to be: almost none of it.

`gemm.jl` is 2335 lines with 67 KernelAbstractions references and 6 Vulkan ones.
`fft.jl` is 880 with 29 and (as of 2026-08-27) **0**. Every coupling that was
removed was the same thing — a capability query routed through a `VkContext`:

    workgroup_limit(vk_context(A))   ->   workgrouplimit(A)
    max_shared_memory(vk_context(A)) ->   sharedbudget(A)

Array to context to caps, versus array to backend to `KernelInterface.caps`. Same
three fields, and the middle step is the one a Metal backend does not have.
`test_array_capability_queries.jl` checks on every backend that the array's
queries answer for its device; this file needs no device.

**GEMM moved by splitting.** `mul!(C::LavaArray{T,2}, …)` is `LinearAlgebra`'s
function, and widening it to `::AbstractGPUArray` would make Mantle claim it for
every GPU array type in the session, CUDA's included. So the entry points stay in
each backend and everything under them is core; see "What moved" below.

This file is the ratchet: the count may fall and may not rise. A new
`vk_context(` in `fft.jl` is a regression whether or not anything fails.
"""

using Test, Mantle
import KernelInterface as KI

# ── What moved ────────────────────────────────────────────────────────────────
#
# `gemv.jl`, `fft.jl`, `gemm.jl` and `gemm_cm2.jl` are in `src/array/` now — core,
# not the backend — and a second backend gets them for free. GEMV and FFT widened
# their dispatch from `::LavaArray` to `::AbstractGPUArray`, which is safe because
# `gemv!`, `fft!`, `rfft!` and `stft` are Mantle's own functions.
#
# GEMM split instead (2026-10-09): `LinearAlgebra.mul!` is Base's, and a method on
# `::AbstractGPUArray` would have Mantle answering for every GPU array package in
# the session, so the `mul!` entry points stay in each backend and the kernels,
# tilings and launch planning moved. What a backend adds is vocabulary:
# `gemmstrides` for its dense arrays and `splitscratch` for split-k scratch. The
# cooperative-matrix kernels are written against KernelInterface's `CoopMatrix`
# and its operations, and `gemm_cm2.jl` against KI's tensor layouts, gated on
# `KI.supports_tensor_addressing`.
#
# `coopmat_gemm_available` stays in the Vulkan tree: it is the probe that fills
# `DeviceCaps.coopmat`, and consumers ask the portable `coopmatgemm(x)`.

"""How many lines of `f` name something only the Vulkan backend has."""
function vulkan_lines(path::AbstractString)
    pat = r"\bVK\.|VkContext|VulkanBatchQueue|vk_context|\bvk_[a-z_]+"
    count(l -> occursin(pat, l), readlines(path))
end

# Where each algorithm stood on 2026-08-27, after the capability queries moved to
# `KernelInterface.caps`. Zero means the file is portable as far as this measure
# goes.
const VULKAN_BUDGET = Dict(
    # What is left of `gemm.jl` in the Vulkan tree after the kernels moved: the
    # `mul!` entry points, `coopmat_gemm_available` (the probe that fills
    # `DeviceCaps.coopmat`), and `splitscratch`, which caches a scratch buffer on
    # `ctx.caches` and is genuine runtime state. 4 lines, 2 of them comments.
    "gemm.jl" => 4,
)

# The moved files are not in this table: a file that has left the backend cannot
# regress a Vulkan-line count. What keeps THEM honest is that they must not name a
# backend array type again — checked below.

@testset "the portable array algorithms stay portable" begin
    dir = joinpath(dirname(pathof(Mantle)), "vulkan", "array")

    @testset "Mantle exposes KernelInterface's complete cooperative-matrix surface" begin
        @test Mantle.CoopMatrix === KI.CoopMatrix
        @test Mantle.AcceleratedMatrix === KI.AcceleratedMatrix
        @test Mantle.WorkgroupMatrix === KI.WorkgroupMatrix
        for name in (:coopmat_load, :coopmat_store, :coopmat_muladd,
                     :coopmat_mul, :coopmat_add, :coopmat_zero, :coopmat_undef,
                     :coopmat_convert, :coopmat_length, :coopmat_getcomp,
                     :coopmat_setcomp)
            @test getfield(Mantle, name) === getfield(KI, name)
        end
    end

    @testset "$f" for (f, budget) in sort(collect(VULKAN_BUDGET))
        n = vulkan_lines(joinpath(dir, f))
        # Named, not counted: the failure says which file and by how much.
        @test (f, n <= budget) == (f, true)
        n < budget &&
            @info "$f is down to $n Vulkan lines from $budget — lower VULKAN_BUDGET"
    end

    # The ones that made it out. A backend array type reappearing in one is the
    # regression — it is how they got stuck in the backend the first time, and it
    # would not fail anything until a second backend existed to be excluded by it.
    @testset "the moved algorithms name no backend array type" begin
        core = joinpath(dirname(pathof(Mantle)), "array")
        @test isdir(core)
        for f in ("gemv.jl", "fft.jl", "gemm.jl", "gemm_cm2.jl")
            src = read(joinpath(core, f), String)
            # In CODE. The headers discuss these names, so the word itself is
            # expected in prose.
            offenders = [l for l in split(src, '\n')
                         if occursin(r"\bLavaArray\b|\bLavaBackend\b|\bVkContext\b", l) &&
                            !startswith(strip(l), "#")]
            @test (f, offenders) == (f, String[])
        end
        # …and they really are loaded from core, not still shadowed by a copy in
        # the backend.
        @test occursin(joinpath("src", "array"), String(first(methods(Mantle.gemv!)).file))
        @test occursin(joinpath("src", "array"), String(first(methods(Mantle.fft!)).file))
        @test occursin(joinpath("src", "array"), String(first(methods(Mantle.coopmat_gemm!)).file))
    end

    # The point of the exercise — the portable queries answer for the device an
    # array lives on — needs a device, so it is `test_array_capability_queries.jl`,
    # which runs on every backend. This file reads source and module bindings
    # only.
end
