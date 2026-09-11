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
three fields, same values (asserted below), and the middle step is the one a
Metal backend does not have.

**What is NOT done, and why it is not a shortcut.** These files dispatch on
`::LavaArray` — `mul!(C::LavaArray{T,2}, …)`, `fft!(dst::LavaArray, …)`. Moving
them to core means widening that to `::AbstractGPUArray`, and doing that today
would make Mantle claim `LinearAlgebra.mul!` for every GPU array type in the
session, CUDA's included. That widening needs a second backend to be designed
against — which is exactly what step 5 pairs the decision with ("prove a vendor
override works"). The capability queries were the half that could be done blind.

This file is the ratchet: the count may fall and may not rise. A new
`vk_context(` in `fft.jl` is a regression whether or not anything fails.
"""

using Test, Mantle, KernelAbstractions
const KA = KernelAbstractions

# ── What moved, and what is blocked on what ───────────────────────────────────
#
# `gemv.jl` and `fft.jl` are in `src/array/` now — core, not the backend — and a
# second backend gets them for free. Their dispatch widened from `::LavaArray` to
# `::AbstractGPUArray` on the way, which is safe because `gemv!`, `fft!`, `rfft!`
# and `stft` are Mantle's own functions: widening them claims nothing that
# belongs to anyone else.
#
# **`gemm.jl` is blocked, and not on what it looked like.** The obvious blocker
# was `LinearAlgebra.mul!` — Base's function, so a method on `::AbstractGPUArray`
# would have Mantle answering for every GPU array package in the session. That
# one is solvable: the entry points stay in the backend and the kernels move,
# which is the shape a Metal backend would repeat.
#
# The real blocker WAS `AcceleratedMatrix`, and it is gone as of 2026-08-28.
#
# The cooperative-matrix half of GEMM is written against it, and it used to be
# **Lava's type**: the coopmat merge (step 2) took the vocabulary (`MatrixA`,
# `Accumulator`, `MatrixShape`, `DeviceCaps`) and left the matrix type itself
# behind, so core would have had to name a Lava type to hold these kernels —
# the dependency the whole split exists to remove. A first attempt died exactly
# there: `UndefVarError: AcceleratedMatrix not defined in Mantle`.
#
# `CoopMatrix{T,M,N,Use,Scope}` and its `AcceleratedMatrix`/`WorkgroupMatrix`
# aliases are `KernelInterface`'s now, beside the vocabulary that went ahead of
# them. The type was always portable — one `Int32` SSA anchor, no element
# storage — and what stayed in Lava is the 766 lines of `llvmcall` that lower
# KI's nine `coopmat_*` operations to `OpCooperativeMatrix*`. `Mantle` names the
# type through `using KernelInterface` and no longer needs Lava for it.
#
# So the prerequisite is met and what remains for GEMM is the OTHER blocker, the
# one this file already called solvable: `LinearAlgebra.mul!` is Base's, so the
# entry points stay in the backend and the kernels move. That split is the shape
# a Metal backend repeats, and it is now the only thing in the way.
#
# `coopmat_gemm_available` is NOT in the way and should not move: it is the
# Vulkan probe that fills `DeviceCaps.coopmat`, and `coopmatgemm(x)` in
# `src/memory/array.jl` is already the portable question consumers ask.

"""How many lines of `f` name something only the Vulkan backend has."""
function vulkan_lines(path::AbstractString)
    pat = r"\bVK\.|VkContext|VulkanBatchQueue|vk_context|\bvk_[a-z_]+"
    count(l -> occursin(pat, l), readlines(path))
end

# Where each algorithm stood on 2026-08-27, after the capability queries moved to
# `KernelInterface.caps`. Zero means the file is portable as far as this measure
# goes.
const VULKAN_BUDGET = Dict(
    # 5, and 3 of them are comments. The two real ones are the floor without
    # moving the file: `coopmat_gemm_available`, which is the Vulkan PROBE that
    # fills `DeviceCaps.coopmat` (consumers read the field, which is portable),
    # and `splitscratch`, which caches a scratch buffer on `ctx.caches` and is
    # genuine runtime state.
    "gemm.jl"     => 5,
    # `coopmat_shape(vk_context(), Float16, …)`: shape support, which
    # `KernelInterface.supports(caps, MatrixShape(…))` covers. Not converted
    # because it wants the shape vocabulary rather than a single field, and that
    # is worth doing beside the dispatch widening rather than on its own.
    "gemm_cm2.jl" => 1,
)

# `fft.jl` and `gemv.jl` are not in this table any more: they are in `src/array/`,
# and a file that has left the backend cannot regress a Vulkan-line count. What
# keeps THEM honest is that they must not name a backend array type again —
# checked below.

@testset "the portable array algorithms stay portable" begin
    dir = joinpath(dirname(pathof(Mantle)), "vulkan", "array")

    @testset "$f" for (f, budget) in sort(collect(VULKAN_BUDGET))
        n = vulkan_lines(joinpath(dir, f))
        # Named, not counted: the failure says which file and by how much.
        @test (f, n <= budget) == (f, true)
        n < budget &&
            @info "$f is down to $n Vulkan lines from $budget — lower VULKAN_BUDGET"
    end

    # The two that made it out. A backend array type reappearing in either is the
    # regression — it is how they got stuck in the backend the first time, and it
    # would not fail anything until a second backend existed to be excluded by it.
    @testset "the moved algorithms name no backend array type" begin
        core = joinpath(dirname(pathof(Mantle)), "array")
        @test isdir(core)
        for f in ("gemv.jl", "fft.jl")
            src = read(joinpath(core, f), String)
            # In CODE. The headers explain what they used to say, so the word
            # itself is expected in prose.
            offenders = [l for l in split(src, '\n')
                         if occursin(r"\bLavaArray\b|\bLavaBackend\b|\bVkContext\b", l) &&
                            !startswith(strip(l), "#")]
            @test (f, offenders) == (f, String[])
        end
        # …and they really are loaded from core, not still shadowed by a copy in
        # the backend.
        @test occursin(joinpath("src", "array"), String(first(methods(Mantle.gemv!)).file))
        @test occursin(joinpath("src", "array"), String(first(methods(Mantle.fft!)).file))
    end

    # The point of the exercise: the portable route answers what the Vulkan one
    # did. If these ever disagree, a kernel sized through `caps` is being launched
    # against a different device's limits — which is silent, and wrong.
    @testset "the portable queries agree with the context ones" begin
        be = Mantle.LavaBackend()
        a = KA.allocate(be, Float32, 16)
        ctx = Mantle.vk_context()
        @test Mantle.workgrouplimit(a) == Mantle.workgroup_limit(ctx)
        @test Mantle.sharedbudget(a) == Mantle.max_shared_memory(ctx)
        @test Mantle.coopmatgemm(a) == Mantle.coopmat_gemm_available(ctx)
        # …and they are real numbers, so "they agree" is not two zeros agreeing.
        @test Mantle.workgrouplimit(a) > 0
        @test Mantle.sharedbudget(a) > 0
    end
end
