# Compilation is a list of phases, not a function.
#
# RPS splits the same work six ways (`src/runtime/common/phases/`), and the
# reason is not tidiness: each analysis becomes testable on its own. Ours grew
# four interacting analyses inside one function, and fuzzing found three bugs
# that all lived in the interaction rather than in any one of them.
#
# A phase reads and writes named fields on a shared context. The backend adds the
# concrete phases; the core defines the protocol and the order.

abstract type Phase end

"""
    run!(phase, ctx)

Advance the compilation by one phase. Every phase is expected to be callable on
its own against a context carrying only what earlier phases produced, which is
what makes each one testable without running the others.
"""
function run!(phase::Phase, ctx)
    throw(MethodError(run!, (phase, ctx)))
end

# ── what a backend's compilation context has to answer ────────────────────────
#
# Five of the seven phases are graph analyses and contain nothing backend-shaped;
# only Barriers and Pipelines do. They used to live in the Lava extension anyway,
# which meant a second backend had to copy a scheduler — and two copies of a
# scheduler drift in exactly the places fuzzing found bugs. They live here now,
# and a backend supplies the five methods below.

"""
Everything the backend-independent phases compute.

One struct rather than a field-per-phase interface: a context holds one of these
and answers [`analysis`](@ref) with it, so adding a phase output does not add an
accessor pair to every backend.
"""
mutable struct Analysis
    deps::Vector{Vector{Int}}                   # Dag
    order::Vector{Int}                          # Schedule
    items::Vector{Item}                         # Liveness
    peak::Int                                   # Place
    naive::Int
    offsets::Vector{Int}
    # Aliasing: pass index => the (new, old) transient pairs whose bytes change
    # hands there. The old index is kept because the barrier is derived from what
    # that transient was last doing, not assumed to be everything.
    alias_begins::Dict{Int,Vector{Tuple{Int,Int}}}
end
Analysis() = Analysis(Vector{Int}[], Int[], Item[], 0, 0, Int[],
                      Dict{Int,Vector{Tuple{Int,Int}}}())

"""The [`Analysis`](@ref) a compilation context carries. One method per backend."""
function analysis end

"""The passes a context is compiling, in DECLARATION order."""
function passes end

"""A pass's declared resource usages, as `id => Usage` pairs."""
function usages end

"""
    overlapping(ctx, a::Int, b::Int) -> Bool

Whether two resource ids can name the same bytes.

Equal ids do, and so does any slicing relationship a backend supports — a slice
against its own parent, two intersecting slices of one parent. A scheduler told
that two such ids are unrelated is free to reorder two passes that write the same
memory, so a backend with no slices still has to answer `a == b`.
"""
function overlapping end

"""The scheduling [`Policy`](@ref) a context was built with."""
function policy end

"""
Which pass depends on which, from the resources they share.

Pass j depends on pass i when they touch a common resource and at least one of
them writes it. Read-after-read is not an edge, which is what leaves independent
work free to be reordered.
"""
struct Dag <: Phase end

function run!(::Dag, c)
    ps = passes(c)
    a = analysis(c)
    a.deps = [Int[] for _ in ps]
    for j in eachindex(ps), i in 1:(j - 1)
        shared = false
        for (idj, Uj) in usages(ps[j]), (idi, Ui) in usages(ps[i])
            overlapping(c, idi, idj) || continue
            (writes(Ui) || writes(Uj)) || continue
            shared = true
            break
        end
        shared && push!(a.deps[j], i)
    end
    return c
end

"""
Choose an execution order consistent with the DAG.

RPS spends its largest phase here (1171 lines) and encodes the whole policy as
bit *positions* in one 32-bit score, so competing objectives compare with a
single integer and retuning means moving a bit rather than rewriting a
comparator. The policy types below pick those positions.
"""
struct Schedule <: Phase end

abstract type Policy end

"""Prefer passes that free more bytes than they allocate: smaller peak, more
ordering between passes, which is RPS's `RPS_SCHEDULE_PREFER_MEMORY_SAVING_BIT`."""
struct Compact <: Policy end

"""Prefer declaration order and leave lifetimes long: larger peak, fewer
dependencies forced by shared memory. RPS's default."""
struct Overlap <: Policy end

"""Which bit range each objective occupies. Moving these is how the policy is
tuned, exactly as in `rps_dag_schedule.hpp:180-208`."""
memory_shift(::Compact) = 16
memory_shift(::Overlap) = 0
order_shift(::Compact) = 0
order_shift(::Overlap) = 16

"""Which passes a resource is live across. Inputs to placement."""
struct Liveness <: Phase end

"""Offsets for every transient, from the liveness intervals.

Named `Place` because `Placement` is already the *result* of `place`."""
struct Place <: Phase end

"""
Which passes begin using memory another transient just finished with.

Separate from placement because it is a *consequence* of placement that the
barrier phase needs, and because it is the one hazard per-resource tracking
cannot see: two aliased transients look unrelated to it.
"""
struct Aliasing <: Phase end

"""Transitions per pass, and which of them still need a barrier emitted."""
struct Barriers <: Phase end

"""Resolve and build every pipeline a draw needs, once."""
struct Pipelines <: Phase end

"""
The order phases run in.

Schedule must precede Liveness, because reordering passes changes the intervals.
Place must precede Aliasing, which produces the offsets it needs. Aliasing must
precede Barriers, which has to emit for the hazard it finds.
"""
const PHASES = (Dag(), Schedule(), Liveness(), Place(), Aliasing(), Barriers(), Pipelines())

"""Run the whole pipeline, or a prefix of it when testing one phase in isolation."""
function compile!(ctx, phases = PHASES)
    for p in phases
        run!(p, ctx)
    end
    ctx
end
