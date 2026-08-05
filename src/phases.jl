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

"""
Which pass depends on which, from the resources they share.

Pass j depends on pass i when they touch a common resource and at least one of
them writes it. Read-after-read is not an edge, which is what leaves independent
work free to be reordered.
"""
struct Dag <: Phase end

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
