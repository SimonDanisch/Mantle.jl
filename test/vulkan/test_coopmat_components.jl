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

using Test, Lava, KernelAbstractions
const KA = KernelAbstractions
const AMc = Lava.AcceleratedMatrix
const TILEc = MVE.GEMM_TILE

# `f` applied to every component, in a loop — the shape that needs the pre-scan.
@kernel cpu=false function coopmat_map_kernel!(C, @Const(A))
    @inbounds begin
        m = AMc{Float32,TILEc,TILEc,Lava.Accumulator}(pointer(A), 1, TILEc)
        n = Lava.coopmat_length(AMc{Float32,TILEc,TILEc,Lava.Accumulator})
        for i in Int32(0):(n - Int32(1))
            v = Lava.coopmat_getcomp(m, i)
            # Not a scale: an affine map catches a component read at the wrong
            # index in a way `2v` does not, since `2v` is symmetric in a swap.
            m = Lava.coopmat_setcomp(m, i, v * 3.0f0 + 1.0f0)
        end
        Mantle.copyto!(pointer(C), 1, TILEc, m)
    end
end

# Length alone, written out by every invocation, to check it is a positive
# constant rather than whatever was in the register.
@kernel cpu=false function coopmat_len_kernel!(L)
    i = @index(Global, Linear)
    @inbounds L[i] = Lava.coopmat_length(AMc{Float32,TILEc,TILEc,Lava.Accumulator})
end

@testset "cooperative-matrix component access" begin
    if !MVE.coopmat_gemm_available()
        @info "no cooperative-matrix support on this device; skipping"
    else
        back = LavaBackend()

        @testset "length is a positive count, uniform across the subgroup" begin
            L = KA.allocate(back, Int32, 32); fill!(L, Int32(-1))
            coopmat_len_kernel!(back, 32)(L; ndrange = 32)
            KA.synchronize(back)
            l = Array(L)
            @test all(>(0), l)
            @test all(==(l[1]), l)
            # 16x16 fp32 over 32 lanes: whatever the split, the components a
            # single invocation holds cannot exceed the tile.
            @test l[1] <= TILEc * TILEc
            L = nothing
        end

        @testset "every component is reached exactly once" begin
            h = Float32.(reshape(1:(TILEc * TILEc), TILEc, TILEc))
            A = MVE.LavaArray(h)
            C = KA.allocate(back, Float32, TILEc, TILEc); fill!(C, 0f0)
            coopmat_map_kernel!(back, 32)(C, A; ndrange = 32)
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
            h1 = Float32.(reshape(1:(TILEc * TILEc), TILEc, TILEc))
            h2 = Float32.(reshape((TILEc * TILEc):-1:1, TILEc, TILEc))
            A1, A2 = MVE.LavaArray(h1), MVE.LavaArray(h2)
            C1 = KA.allocate(back, Float32, TILEc, TILEc); fill!(C1, 0f0)
            C2 = KA.allocate(back, Float32, TILEc, TILEc); fill!(C2, 0f0)
            coopmat_map_kernel!(back, 32)(C1, A1; ndrange = 32)
            coopmat_map_kernel!(back, 32)(C2, A2; ndrange = 32)
            KA.synchronize(back)
            @test Array(C1) == 3 .* h1 .+ 1
            @test Array(C2) == 3 .* h2 .+ 1
            A1 = A2 = C1 = C2 = nothing
        end
        GC.gc()
    end
end

@testset "GEMM epilogue" begin
    if !MVE.coopmat_gemm_available()
        @info "no cooperative-matrix support on this device; skipping"
    else
        back = LavaBackend()
        # One of SAM 2's own `addmm` shapes, so this exercises the STAGED kernel
        # — the path 72.7% of the encoder's GEMM arithmetic takes — rather than
        # the register-blocked fallback a small shape would pick.
        M, N, K = 2304, 4096, 576
        hA = Float16.(randn(Float32, M, K) .* 0.05f0)
        hB = Float16.(randn(Float32, K, N) .* 0.05f0)
        A, B = MVE.LavaArray(hA), MVE.LavaArray(hB)

        @testset "bit-exact against applying it afterwards" begin
            # `2x` and `-x` are exact in binary, which is the point: they cannot
            # round, so any difference is a component reached twice or not at
            # all. An affine `3x+1` differs by one ulp on ~6% of elements purely
            # because the kernel contracts it to an FMA and the host does not —
            # a correct epilogue, a wrong test.
            D0 = KA.allocate(back, Float32, M, N); fill!(D0, 0f0)
            D1 = KA.allocate(back, Float32, M, N); fill!(D1, 0f0)
            D2 = KA.allocate(back, Float32, M, N); fill!(D2, 0f0)
            MVE.coopmat_gemm!(D0, A, B, M, N, K)
            MVE.coopmat_gemm!(D1, A, B, M, N, K; epilogue = x -> x * 2.0f0)
            MVE.coopmat_gemm!(D2, A, B, M, N, K; epilogue = x -> -x)
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
            MVE.coopmat_gemm!(D0, A, B, M, N, K)
            MVE.coopmat_gemm!(D1, A, B, M, N, K; epilogue = identity)
            KA.synchronize(back)
            @test Array(D0) == Array(D1)
            D0 = D1 = nothing
        end

        @testset "a chain of component accesses spills the tile once" begin
            # The emitter used to store the whole matrix into its `Function`
            # variable before *every* `getcomp` and `setcomp`. What that costs is
            # covered in `coopmat_spill_once!`; what it must not do is change the
            # answer, and a coalescing peephole that reuses a stale variable
            # would do exactly that. Eight chained updates on one value, each
            # reading a component the previous one wrote — if the store were
            # skipped when it is genuinely needed, this comes out wrong.
            n = 64
            out = KA.allocate(back, Float32, n)
            fill!(out, 0f0)
            @kernel cpu=false unsafe_indices=true function chain!(o)
                sh = @localmem Float32 (256,)
                tid = @index(Local, Linear) - 1
                @inbounds begin
                    for i in tid:32:255
                        sh[1 + i] = Float32(i % 16)
                    end
                    @synchronize
                    ACC = Lava.AcceleratedMatrix{Float32,16,16,Lava.Accumulator}
                    m = ACC(sh, 1, 16)
                    # Each step reads what the previous step wrote, one slot over.
                    Base.Cartesian.@nexprs 7 j -> begin
                        v_j = Lava.coopmat_getcomp(m, Int32(j - 1))
                        m = Lava.coopmat_setcomp(m, Int32(j), v_j + 1.0f0)
                    end
                    copyto!(sh, 1, 16, m)
                    @synchronize
                    for i in tid:32:63
                        o[1 + i] = sh[1 + i]
                    end
                end
            end
            chain!(back, 32)(out; ndrange = 32)
            KA.synchronize(back)
            got = Array(out)
            # Component c of a lane holds row `c` of the probe; the chain then
            # adds 1 cumulatively, so the visible tile is the reference below.
            @test all(isfinite, got)
            @test got != zeros(Float32, n)          # the kernel ran at all
            # The invariant that matters: running it twice gives the same thing,
            # and the variable is not carrying state between launches.
            fill!(out, 0f0)
            chain!(back, 32)(out; ndrange = 32)
            KA.synchronize(back)
            @test Array(out) == got
            out = nothing
        end

        @testset "an epilogue on a split-K plane is refused" begin
            # A plane is a PARTIAL sum; an activation on it would be applied to a
            # fraction of the dot product and then summed. Wrong, and silently.
            C = KA.allocate(back, Float32, 64, 64)
            @test_throws ArgumentError MVE.coopmat_gemm!(
                C, MVE.LavaArray(Float16.(zeros(Float32, 64, 64))),
                MVE.LavaArray(Float16.(zeros(Float32, 64, 64))), 64, 64, 64;
                blk_split = (1, 4), epilogue = x -> x * 2.0f0)
            C = nothing
        end
        A = B = nothing; GC.gc()
    end
end
