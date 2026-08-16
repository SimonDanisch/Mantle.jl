"""
Vocabulary that more than one GPU backend has to name.

This package exists because `Mantle` and `Lava` must agree on what a device's
matrix hardware can do, and neither depends on the other — `Mantle` weak-depends
on `Lava` for its extension, so an edge back would close a cycle. Before this,
the shared parts were spelled twice and bridged by a positional conversion in
`MantleLavaExt`.

**It is a vocabulary, not an interface.** There are deliberately no operations
here — no load, no store, no multiply-accumulate. Those need a representation
(SPIR-V hands back an opaque handle, CUDA a register tuple, Metal a builtin) and
a dispatch story, and neither can be settled honestly against a single backend.
The line is: settled semantics go in, open design decisions stay out. Everything
here is either three singleton types whose meaning the hardware fixes, or a
record of what a driver reported.

If a general `KernelInterfaces.jl` lands in the ecosystem, this is meant to be
absorbed by it — which is the other reason to keep it this small.
"""
module KernelInterfaces

export MatrixUse, MatrixA, MatrixB, Accumulator
export MatrixScope, SubgroupScope, WorkgroupScope
export MatrixShape, supports, bestshape

"""
Which operand position a matrix occupies in `A*B + C`.

Part of a matrix's type rather than a property of the value, because the
hardware distributes the three differently across a subgroup's lanes: `A` by row,
`B` by column, the accumulator differently again. It is not an annotation any one
API chose — SPIR-V, CUDA (`wmma::matrix_a`) and AMD's WMMA all expose it, because
all three must. Metal's `simdgroup_matrix` is the exception, having one uniform
type.

The names are inherited and imperfect: they label a position in `C = A*B + C`
rather than the property they stand for, which is really which axis carries the
contraction. A matrix that is `B` in one product and `A` in the next needs a
conversion despite holding the same values.
"""
abstract type MatrixUse end
struct MatrixA <: MatrixUse end
struct MatrixB <: MatrixUse end
struct Accumulator <: MatrixUse end

"""
Whose registers hold a matrix: one subgroup's, or the whole workgroup's.

Subgroup scope is the portable one. Workgroup scope exists on far less hardware
and buys register pressure — the same accumulator costs a fraction of the
components per lane when it is spread over every invocation instead of one
subgroup — so a kernel wanting it has to be selected against a device query
rather than reaching for it by default.
"""
abstract type MatrixScope end
struct SubgroupScope <: MatrixScope end
struct WorkgroupScope <: MatrixScope end

"""
    MatrixShape(ab, acc, M, N, K, scope)

One `(M, N, K)` matrix-multiply shape the hardware implements, for operands of
type `ab` accumulating into `acc`.

**Deliberately not parameterised on `ab`/`acc`.** A device reports a whole table
of these and the entries disagree on type — this card lists `16x16x16` for
`Float16 -> Float32`, `Float16 -> Float16`, `UInt8 -> Int32` and `Int8 -> Int32`.
As type parameters those are four distinct types, so the table's element type
becomes abstract and every entry boxes. Nothing is bought by it either: the shape
never reaches the device. It is queried once, cached on the context, and the only
part that crosses into a kernel is an `Int` extracted from it and passed as a
`Val`.

`ab` and `acc` are separate because the pair is what distinguishes shapes in
practice; extents alone do not. Matching on `(M, N, K)` and ignoring the types
says "yes" for `Float16` on hardware that implements only the integer forms at
those extents, and the kernel then emits instructions the device cannot run.
"""
struct MatrixShape
    ab::DataType
    acc::DataType
    M::Int
    N::Int
    K::Int
    scope::MatrixScope
end

# Spelled out rather than left to the default. `==` falls back to `===`, which
# happens to work here (every field is egal-comparable), but a table of these is
# searched and de-duplicated, and a shape's identity is its values — not which
# call to the driver produced it.
Base.:(==)(a::MatrixShape, b::MatrixShape) =
    a.ab === b.ab && a.acc === b.acc &&
    a.M == b.M && a.N == b.N && a.K == b.K && a.scope === b.scope

Base.hash(s::MatrixShape, h::UInt) =
    hash(s.ab, hash(s.acc, hash(s.M, hash(s.N, hash(s.K, hash(typeof(s.scope), h))))))

function Base.show(io::IO, s::MatrixShape)
    print(io, "MatrixShape(", s.ab, "->", s.acc, " ",
          s.M, "x", s.N, "x", s.K, " ", nameof(typeof(s.scope)), ")")
end

"""
    supports(shapes, s::MatrixShape) -> Bool

Whether `s` is in the table.

Takes the table rather than a capability record because this package cannot name
one: `Mantle` and `Lava` each have their own `DeviceCaps`. Both forward to this.

A shape being present means the device declares it **legal**, which is not the
same as fast and not the same as implemented in silicon — a software rasterizer
reports cooperative-matrix shapes it emulates. Ranking legal shapes is the
caller's job.
"""
supports(shapes, s::MatrixShape) = any(==(s), shapes)

"""
    bestshape(shapes, ab, acc; scope = SubgroupScope()) -> MatrixShape | nothing

The shape to use for `ab -> acc` at `scope`, or `nothing` if the device has none.

`nothing` rather than a default: a fabricated tile is a divisor, and a wrong one
divides silently. Every caller here has to decide what to do without matrix
hardware, so it is better that the type system make them.

Preference is the squarest shape, then the largest. Square first because a
kernel tiling a square accumulator with a non-square instruction has to carry two
different blocking factors; largest second because a bigger tile is fewer
instructions for the same work. Both are tie-breaks among shapes the device
already declared legal — this chooses a *default*, and a kernel that has measured
something better on its own shapes should carry that in its plan instead.

This lives beside the table for the reason the workgroup-granularity lookup does:
otherwise two callers write the same `any(s -> ...)` loop with different argument
orders, and they drift.
"""
function bestshape(shapes, ab, acc; scope::MatrixScope = SubgroupScope())
    best = nothing
    for s in shapes
        (s.ab === ab && s.acc === acc && typeof(s.scope) === typeof(scope)) || continue
        if best === nothing || isbetter(s, best)
            best = s
        end
    end
    return best
end

# Squarest, then largest. Split out so the ordering is one testable thing rather
# than a comparison buried in a loop.
function isbetter(a::MatrixShape, b::MatrixShape)
    sa = abs(a.M - a.N) + abs(a.N - a.K)
    sb = abs(b.M - b.N) + abs(b.N - b.K)
    sa != sb && return sa < sb
    return a.M * a.N * a.K > b.M * b.N * b.K
end

end # module
