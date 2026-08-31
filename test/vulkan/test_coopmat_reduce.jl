"""
`OpCooperativeMatrixReduceNV` — combine along an axis with a binary function.

`spirv-intrinsics.md` calls this the cheapest of the five missing coopmat2 items,
because it takes a function operand exactly like
`OpCooperativeMatrixPerElementOpNV` and so reuses the thunk / `keepparam` /
marker machinery already built and tested for that one. This is the instruction
`kernels-to-port.md` item 17 (and K3, the flash softmax's row max and sum) needs.

**Everything asserted here about the semantics came from
`flash_attn_cm2.comp`, not from the extension spec**, which is not on this
machine. Two of them are easy to guess wrong and are the reason this file exists:

  * the callback is a **binary combiner** `(x, y) -> z` — a row maximum is
    `max` — and *not* the `(row, col, element)` of the per-element op;
  * the result is **not a vector**. It has the destination's shape with the
    reduced value repeated along the reduced axis, which is why the reference
    carries a `smearReduce(x, y) = x` that reduces nothing and exists purely to
    broadcast.

The mask is a bitfield: `Row = 1`, and the reference's
`gl_CooperativeMatrixReduceRowAndColumnNV` is the union, which is how
`Column = 2` is known without the spec.

**The open question this file answered is scope, and the answer is good.** Every
use in the reference is `gl_ScopeWorkgroup`, a coopmat2 feature, while every
matrix Lava emits is `Subgroup` — so a subgroup-scoped reduce might have been
refused, which would have meant item 17 needed workgroup-scope matrices first and
was not cheap at all. It is accepted, and computes correctly.

**What it also found: the combine order is unspecified per output element**, so
`f` must be associative and commutative. See the third case below — this is the
thing that makes a naive port of `smearReduce` wrong outside the reference's own
usage.
"""

using Test, Lava, KernelAbstractions
using Lava: AcceleratedMatrix, Accumulator, CoopMatReduce, coopmat_reduce
const KA = KernelAbstractions
const RT = MVE.GEMM_TILE

# Top-level, not closures, and NOT `@noinline` — `coopmat_reduce_thunk` is the
# function the instruction names and `f` should melt into it.
rmax(x::Float32, y::Float32) = max(x, y)
rsum(x::Float32, y::Float32) = x + y
rfirst(x::Float32, y::Float32) = x          # the reference's `smearReduce`

@kernel cpu=false unsafe_indices=true function reducerow!(out, @Const(x), ::Val{OP}) where {OP}
    m = AcceleratedMatrix{Float32,RT,RT,Accumulator}(pointer(x), 1, RT)
    f = OP === :max ? rmax : OP === :sum ? rsum : rfirst
    r = coopmat_reduce(f, AcceleratedMatrix{Float32,RT,RT,Accumulator}, m,
                       Val(CoopMatReduce.Row))
    copyto!(pointer(out), 1, RT, r)
end

@testset "coopmat_reduce: OpCooperativeMatrixReduceNV" begin
    ctx = MVE.vk_context()
    backend = LavaBackend()
    if !ctx.coopmat2.available || !ctx.coopmat2.reductions
        @info "no coopmat2 reductions on this device; skipping" ctx.device_name
    else
        # Column-major, so element (r, c) is at r + c*RT. Distinct per row so a
        # row reduction cannot pass by accident.
        xh = Float32[(r - 1) * 100 + (c - 1) for r in 1:RT, c in 1:RT]
        x = MVE.LavaArray(xh)
        out = KA.allocate(backend, Float32, RT, RT)

        # ── Does a SUBGROUP-scoped reduce compile and run at all?
        ok = true
        try
            fill!(out, -1.0f0)
            reducerow!(backend, 32)(out, x, Val(:max); ndrange = 32)
            KA.synchronize(backend)
        catch e
            ok = false
            @info """subgroup-scoped OpCooperativeMatrixReduceNV was REFUSED.
                     Item 17 then needs workgroup-scope matrices first, which is
                     not the cheap change it is filed as.""" exception = e
        end
        @test ok

        if ok
            got = Array(out)
            # Row reduce with `max`: every element of a row holds that row's max,
            # smeared — the destination is a matrix, not a vector.
            want = maximum(xh; dims = 2)
            @test all(got[r, c] == want[r] for r in 1:RT, c in 1:RT)

            # Sum, to show the combiner is genuinely applied pairwise and it is
            # not just broadcasting one element.
            fill!(out, -1.0f0)
            reducerow!(backend, 32)(out, x, Val(:sum); ndrange = 32)
            KA.synchronize(backend)
            s = Array(out)
            wsum = sum(xh; dims = 2)
            @test all(isapprox(s[r, c], wsum[r]; rtol = 1f-5) for r in 1:RT, c in 1:RT)

            # ── The combine ORDER is unspecified per output element.
            #
            # `f` must be associative and commutative. `max` and `+` are, which is
            # why the two assertions above hold; `smearReduce(x, y) = x` is not,
            # and it shows: on a row of 200..215 the result is
            # `[200..207, 200..207]` — the whole row IS reduced (the `sum` above
            # returns the exact full-row total), but different output elements
            # combine it in different orders, so a combiner that discards one
            # argument returns a different survivor per position.
            #
            # That is not a bug and it does not contradict the reference, which
            # only ever hands `smearReduce` a matrix that is ALREADY uniform along
            # the row — its `eM` comes from a row max. There it reduces nothing and
            # exists purely to RESIZE, and any order gives the same answer.
            fill!(out, -1.0f0)
            reducerow!(backend, 32)(out, x, Val(:first); ndrange = 32)
            KA.synchronize(backend)
            nonassoc = Array(out)
            @test !all(nonassoc[3, c] == nonassoc[3, 1] for c in 1:RT)

            # And the same combiner on a row-uniform input — the reference's
            # actual usage — is exact.
            uh = Float32[(r - 1) * 10 for r in 1:RT, c in 1:RT]
            u = MVE.LavaArray(uh)
            fill!(out, -1.0f0)
            reducerow!(backend, 32)(out, u, Val(:first); ndrange = 32)
            KA.synchronize(backend)
            sm = Array(out)
            @test all(sm[r, c] == uh[r, 1] for r in 1:RT, c in 1:RT)
        end
    end
end
