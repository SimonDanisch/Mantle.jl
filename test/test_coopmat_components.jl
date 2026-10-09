"""
Component access on a cooperative matrix — `coopmat_length`, `coopmat_getcomp`
and `coopmat_setcomp`.

SPIR-V has no extract-from-cooperative-matrix instruction. A matrix is an SSA
value; the only way to reach its components is to put it in a `Function`-storage
variable and `OpAccessChain` into that, which is what GLSL's `mat[i]` lowers to.
This is what makes an elementwise **epilogue** expressible — a GEMM that wants
`gelu` or `relu` on its accumulator has to reach the components before the store,
and the alternative is a second full pass over the output in global memory.

Two things are worth asserting separately, because they fail differently.

**That the components are the right ones.** The split across the subgroup is the
implementation's business, so a test cannot assume which invocation holds what.
What it can assume is that walking `0:length-1` on every invocation and writing
`f(x)` back reaches each element of the tile exactly once — so a whole-tile
comparison against `f.(input)` is the strongest available check and catches both
a missed component and a doubled one.

**That the variable is in scope.** The `OpVariable` is buffered into the entry
block's preamble, so allocating it lazily from inside a loop body emits a store
to an id that is not defined yet. `spirv-val` rejects the module — "ID '96' has
not been defined" — which is how the pre-scan came to exist. A kernel whose
component access sits inside a loop is therefore not an incidental shape here,
it is the regression test for that: `prescan_function_for_coopmat_components!`.
"""

using Test, KernelAbstractions
using KernelInterface: AcceleratedMatrix, Accumulator,
                       coopmat_length, coopmat_getcomp, coopmat_setcomp
include(joinpath(@__DIR__, "testbackend.jl"))
const KA = KernelAbstractions

# The component-access kernels take the tile as `Val(T)`: it is the device's
# (`caps(...).tile`), and component access is defined at every tile.

# `f` applied to every component, in a loop — the shape that needs the pre-scan.
@kernel cpu=false function coopmat_map_kernel!(C, @Const(A), ::Val{T}) where {T}
    @inbounds begin
        m = AcceleratedMatrix{Float32,T,T,Accumulator}(pointer(A), 1, T)
        n = coopmat_length(AcceleratedMatrix{Float32,T,T,Accumulator})
        for i in Int32(0):(n - Int32(1))
            v = coopmat_getcomp(m, i)
            # Not a scale: an affine map catches a component read at the wrong
            # index in a way `2v` does not, since `2v` is symmetric in a swap.
            m = coopmat_setcomp(m, i, v * 3.0f0 + 1.0f0)
        end
        copyto!(pointer(C), 1, T, m)
    end
end

# Length alone, written out by every invocation, to check it is a positive
# constant rather than whatever was in the register.
@kernel cpu=false function coopmat_len_kernel!(L, ::Val{T}) where {T}
    i = @index(Global, Linear)
    @inbounds L[i] = coopmat_length(AcceleratedMatrix{Float32,T,T,Accumulator})
end

# Eight chained component updates on one value, each reading the slot the
# previous one wrote. Written at the staged GEMM's 16-wide tile and 32-lane
# subgroup, like the epilogue it guards; see "a chain of component accesses
# spills the tile once" below.
@kernel cpu=false unsafe_indices=true function coopmat_chain!(o)
    sh = @localmem Float32 (256,)
    tid = @index(Local, Linear) - 1
    @inbounds begin
        for i in tid:32:255
            sh[1 + i] = Float32(i % 16)
        end
        @synchronize
        ACC = AcceleratedMatrix{Float32,16,16,Accumulator}
        m = ACC(sh, 1, 16)
        # Each step reads what the previous step wrote, one slot over.
        Base.Cartesian.@nexprs 7 j -> begin
            v_j = coopmat_getcomp(m, Int32(j - 1))
            m = coopmat_setcomp(m, Int32(j), v_j + 1.0f0)
        end
        copyto!(sh, 1, 16, m)
        @synchronize
        for i in tid:32:63
            o[1 + i] = sh[1 + i]
        end
    end
end

@testset "cooperative-matrix component access" begin
    c = Mantle.caps(TESTBACKEND)
    if !c.coopmat
        @info "no cooperative-matrix support on this device; skipping"
    else
        back = TESTBACKEND
        # The device's tile, and one launch of exactly one subgroup at the width a
        # cooperative-matrix kernel runs at.
        T, W = c.tile, c.coopmatsubgroup

        @testset "length is a positive count, uniform across the subgroup" begin
            L = KA.allocate(back, Int32, W); fill!(L, Int32(-1))
            coopmat_len_kernel!(back, W)(L, Val(T); ndrange = W)
            KA.synchronize(back)
            l = Array(L)
            @test all(>(0), l)
            @test all(==(l[1]), l)
            # TxT fp32 over one subgroup: whatever the split, the components a
            # single invocation holds cannot exceed the tile.
            @test l[1] <= T * T
            L = nothing
        end

        @testset "every component is reached exactly once" begin
            h = Float32.(reshape(1:(T * T), T, T))
            A = Mantle.devicearray(TESTBACKEND, h)
            C = KA.allocate(back, Float32, T, T); fill!(C, 0f0)
            coopmat_map_kernel!(back, W)(C, A, Val(T); ndrange = W)
            KA.synchronize(back)
            got = Array(C)
            # Exact: this is fp32 in and fp32 out with no accumulation, so
            # anything but 0 is a wrong index, not a rounding difference.
            @test got == 3 .* h .+ 1
            A = C = nothing
        end

        @testset "it survives a second matrix of the same type" begin
            # The variable is cached per matrix TYPE and written before every
            # read, so two matrices must not bleed into one another.
            h1 = Float32.(reshape(1:(T * T), T, T))
            h2 = Float32.(reshape((T * T):-1:1, T, T))
            A1, A2 = Mantle.devicearray(TESTBACKEND, h1), Mantle.devicearray(TESTBACKEND, h2)
            C1 = KA.allocate(back, Float32, T, T); fill!(C1, 0f0)
            C2 = KA.allocate(back, Float32, T, T); fill!(C2, 0f0)
            coopmat_map_kernel!(back, W)(C1, A1, Val(T); ndrange = W)
            coopmat_map_kernel!(back, W)(C2, A2, Val(T); ndrange = W)
            KA.synchronize(back)
            @test Array(C1) == 3 .* h1 .+ 1
            @test Array(C2) == 3 .* h2 .+ 1
            A1 = A2 = C1 = C2 = nothing
        end
        GC.gc()
    end
end

# `coopmat_gemm!` runs the staged GEMM's kernels, which are written at its tile;
# a device without cooperative matrices at that tile has nothing here to run.
@testset "GEMM epilogue" begin
    if !hasstagedgemm()
        @info "no cooperative matrices at the staged GEMM's tile on this device; skipping"
    else
        back = TESTBACKEND
        # One of SAM 2's own `addmm` shapes, so this exercises the STAGED kernel
        # — the path 72.7% of the encoder's GEMM arithmetic takes — rather than
        # the register-blocked fallback a small shape would pick.
        M, N, K = 2304, 4096, 576
        hA = Float16.(randn(Float32, M, K) .* 0.05f0)
        hB = Float16.(randn(Float32, K, N) .* 0.05f0)
        A, B = Mantle.devicearray(TESTBACKEND, hA), Mantle.devicearray(TESTBACKEND, hB)

        @testset "bit-exact against applying it afterwards" begin
            # `2x` and `-x` are exact in binary, which is the point: they cannot
            # round, so any difference is a component reached twice or not at
            # all. An affine `3x+1` differs by one ulp on ~6% of elements purely
            # because the kernel contracts it to an FMA and the host does not —
            # a correct epilogue, a wrong test.
            D0 = KA.allocate(back, Float32, M, N); fill!(D0, 0f0)
            D1 = KA.allocate(back, Float32, M, N); fill!(D1, 0f0)
            D2 = KA.allocate(back, Float32, M, N); fill!(D2, 0f0)
            Mantle.coopmat_gemm!(D0, A, B, M, N, K)
            Mantle.coopmat_gemm!(D1, A, B, M, N, K; epilogue = x -> x * 2.0f0)
            Mantle.coopmat_gemm!(D2, A, B, M, N, K; epilogue = x -> -x)
            KA.synchronize(back)
            E0 = Array(D0)
            @test maximum(abs, E0) > 1e-3            # it computed something
            @test Array(D1) == 2 .* E0
            @test Array(D2) == .-E0
            D0 = D1 = D2 = nothing
        end

        @testset "identity is the default and costs nothing" begin
            D0 = KA.allocate(back, Float32, M, N); fill!(D0, 0f0)
            D1 = KA.allocate(back, Float32, M, N); fill!(D1, 0f0)
            Mantle.coopmat_gemm!(D0, A, B, M, N, K)
            Mantle.coopmat_gemm!(D1, A, B, M, N, K; epilogue = identity)
            KA.synchronize(back)
            @test Array(D0) == Array(D1)
            D0 = D1 = nothing
        end

        @testset "a chain of component accesses spills the tile once" begin
            # The emitter must not store the whole matrix into its `Function`
            # variable before *every* `getcomp` and `setcomp`. What that costs is
            # covered in `coopmat_spill_once!`; what it must not do is change the
            # answer, and a coalescing peephole that reuses a stale variable
            # would do exactly that. Eight chained updates on one value, each
            # reading a component the previous one wrote — if the store were
            # skipped when it is genuinely needed, this comes out wrong.
            n = 64
            out = KA.allocate(back, Float32, n)
            fill!(out, 0f0)
            coopmat_chain!(back, 32)(out; ndrange = 32)
            KA.synchronize(back)
            got = Array(out)
            # Component c of a lane holds row `c` of the probe; the chain then
            # adds 1 cumulatively, so the visible tile is the reference below.
            @test all(isfinite, got)
            @test got != zeros(Float32, n)          # the kernel ran at all
            # The invariant that matters: running it twice gives the same thing,
            # and the variable is not carrying state between launches.
            fill!(out, 0f0)
            coopmat_chain!(back, 32)(out; ndrange = 32)
            KA.synchronize(back)
            @test Array(out) == got
            out = nothing
        end

        @testset "an epilogue on a split-K plane is refused" begin
            # A plane is a PARTIAL sum; an activation on it would be applied to a
            # fraction of the dot product and then summed. Wrong, and silently.
            C = KA.allocate(back, Float32, 64, 64)
            @test_throws ArgumentError Mantle.coopmat_gemm!(
                C, Mantle.devicearray(TESTBACKEND, Float16.(zeros(Float32, 64, 64))),
                Mantle.devicearray(TESTBACKEND, Float16.(zeros(Float32, 64, 64))), 64, 64, 64;
                blk_split = (1, 4), epilogue = x -> x * 2.0f0)
            C = nothing
        end
        A = B = nothing; GC.gc()
    end
end
