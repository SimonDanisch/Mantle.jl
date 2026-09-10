# What a submission holds, what has to outlive it, and what gets reused —
# phases 2.2 and 2.3 of `docs/mantle-owns-it.md`.
#
# `graph/submission.jl` already had the model: `Outstanding(token, payload, tag)`,
# `submitted!`, `sweep!`, `passed`, `recycle!`. What was missing is that anything
# used it. Backend-side arrangements did the same job instead, and every one of
# them was a `Vector` on a backend queue:
#
#   `free_submissions`, `free_oneshots`   objects to reuse            (2.3)
#   `deferred_frees`, `deferred_as_frees` objects to destroy later    (2.2)
#     plus `deferred_frees_lock`, `pin!` at 35 call sites, and
#   `buf.last_write_bq` / `last_write_val` cross-queue ordering       (2.2)
#
# Five lists of three facts. This file is the three facts, once, in core.
#
# ── 2.2, and why it is not `pin!` ────────────────────────────────────────────
#
# `pin!(e, obj)` said "this recording must not let `obj` die". It lived in the
# backend, which meant the backend decided when a lifetime ended — and a backend
# deciding lifetimes is the thing this refactor is about. It also could not
# answer the question it was asked: a pinned object was released by draining a
# separate deferred list against a separate counter, so "has the submission that
# pinned this passed" had no single reader.
#
# `hold!(ch, obj)` is the same intent with the ownership the other way round: the
# CHANNEL accumulates what the recording being built must outlive, the submission
# takes that list as its payload, and `sweep!` drops it when `passed` says the
# device is done. Nothing is destroyed there — dropping the last reference is
# what frees, and a GC-owned object frees itself. What core owns is WHEN the
# reference goes, which is the whole of the question.
#
# Two things a reference cannot answer, so core answers them here as well:
#
#   * an EXPLICIT destroy (`unsafe_free!`) of a resource work in flight still
#     names. The request is the caller's; WHEN the driver's destructor may run
#     is not. `retire!` records it, `reclaim!` runs it once the device is past
#     the work that named it, and the backend's only part is `rawfree`.
#   * ORDER between two channels. A buffer written on one queue and read on
#     another needs a wait, and which wait is a fact about submissions, not
#     about buffers. Core stamps every resource a submission names with the
#     (channel, token) that took it, and derives the waits for the next one.
#
# ── 2.3, and why the free list is here ───────────────────────────────────────
#
# A recording is a pooled object: acquire one, record into it, submit it, get it
# back when the device is finished. That discipline — which one, when to make a
# new one, when to let go — is the same discipline `Pool` applies to memory, and
# it is not a driver's. The backend supplies four primitives on its own object
# and nothing else: make one, reset one, destroy one, and say what becomes of one
# whose submission has passed.

"""
    Stamp{T}

Which channel last submitted work naming a resource, under which token, and how
many recordings that can still submit are holding it.

The core-owned replacement for `VkManagedBuffer.last_write_bq` /
`last_write_val` / `@atomic pins`, and it is here rather than on the resource
for the reason the whole file is: the three questions those fields answered —
does this submission need a cross-channel wait, may this destructor run yet, has
everything that named this finished — are all questions about SUBMISSIONS. The
backend supplies the storage (a field on its own type, answered by
[`stampof`](@ref)); every value in it is written by core.

`channel === nothing` means no submission has ever named the resource.

`holders` is what `pins` was, and it is the one fact a reference count cannot
give: a closed-but-unsubmitted recording holds a resource that nothing has
stamped yet, so a destroy requested in that window looks safe by every timeline
and is not. `@atomic` because two channels on two threads can hold one resource.
"""
mutable struct Stamp{T}
    channel::Any
    token::T
    @atomic holders::Int
end
Stamp{T}() where {T} = Stamp{T}(nothing, zero(T), 0)

"""
    stampof(obj) -> Stamp or nothing

The [`Stamp`](@ref) a resource carries, or `nothing` for an object whose
lifetime is nobody's business but the GC's — a pipeline, a descriptor set, a
host array a command buffer names.

The one hook the tracking needs. A backend answers it for the types that have
device-visible bytes; everything else takes this default and costs one dynamic
call at the moments a hold list is built and dropped.
"""
stampof(@nospecialize(obj)) = nothing

"""
    SubmitChannel{Q,R,T,P}

An independent submission channel: what it submits through, what it has in
flight, what those submissions hold, what is waiting to be destroyed, and the
recordings it reuses.

This was `BatchQueue`, whose 41 fields were a Vulkan command pool, a timeline
semaphore, a family index, a back-reference to `VkContext` and five `Vector`s —
a driver arrangement in a shared struct, with type parameters standing in for
the names it could not write. What every backend actually needs is here; what
only one needs is behind [`channelof`](@ref).

`R` is the backend's recording type, `T` its submission token, `P` the payload a
sweep hands back. Parameters rather than `Any` because an abstract `Outstanding`
eltype boxes every token a sweep reads.
"""
mutable struct SubmitChannel{Q,R,T,P}
    # Whatever the backend submits through: a `VkQueue` with its pool and
    # timeline semaphore, an `MTLCommandQueue`. Core never looks inside it.
    channel::Q
    outstanding::Vector{Outstanding{T,P}}
    # 2.3: recordings the device has finished with, ready to be recorded into
    # again. Core's list, because when a recording may be reused is a question
    # about a submission having passed and nothing else.
    free::Vector{R}
    # 2.2: what each recording currently being built must outlive — one list
    # per OPEN recording, innermost last. A stack and not one list, because a
    # recording is routinely opened while another is being built: an upload
    # inside a frame, a build inside a run. With one list the inner submission
    # took the outer's holds with it and released them when IT passed, which is
    # before the outer had even been submitted. `acquire!` pushes a frame,
    # `takeholds!` and `release!` pop it.
    holds::Vector{Vector{Any}}
    # Emptied hold lists, kept so a submission that holds nothing allocates
    # nothing: `takeholds!` hands an empty frame back here instead of into the
    # payload, and `unhold!` returns a list once its claims are dropped.
    spare::Vector{Vector{Any}}
    # Destroys that have been REQUESTED and may not have run yet. `pending` is
    # written from any thread (a finalizer's, whichever the GC picked) and reads
    # nothing; `retiring` is the owning thread's, each entry stamped with the
    # token of the last submission on THIS channel that named it. Two phases for
    # the reason `Pool.reclaim!` has two: the thread that drops a resource is in
    # no position to ask a queue anything.
    pending::Vector{Any}
    # `ReentrantLock`, and it is not a preference: a destroy is REQUESTED from
    # finalizers, and a finalizer runs at whatever allocation safepoint the GC
    # picks — including one INSIDE this critical section, on this very thread,
    # since `push!` and `append!` allocate. A `SpinLock` re-entered that way
    # never returns: measured, `test_devicerange.jl` spun at 100% of one core
    # with a byte-identical RSS and zero voluntary context switches. A
    # `ReentrantLock` is per-TASK reentrant, so the finalizer's `retire!`
    # acquires it again and proceeds. `Pool.lock` is the same lock for the same
    # reason, taken from the same finalizers.
    pendinglock::ReentrantLock
    retiring::Vector{Tuple{Any,T}}
    # `reclaim!`'s working copy of `pending`, so the lock is held for an
    # `append!` and nothing else: a `rawfree` runs driver destructors and a
    # hand-off takes ANOTHER channel's lock, and neither may happen while a
    # finalizer thread is blocked on ours.
    taken::Vector{Any}
    # Single-writer: only this thread may record into or submit from this channel.
    thread::Int
end

# Fully parameterised — a backend spells its channel type once, as an alias, and
# calls it with the driver bundle. A partial application would bind the
# parameters in the wrong order (`SubmitChannel{R,T,P}` sets Q, R and T).
SubmitChannel{Q,R,T,P}(channel::Q) where {Q,R,T,P} =
    SubmitChannel{Q,R,T,P}(channel, Outstanding{T,P}[], R[],
                           Vector{Any}[], Vector{Any}[],
                           Any[], ReentrantLock(), Tuple{Any,T}[], Any[],
                           Threads.threadid())

"""
    payloadtype(ch) -> Type

What this channel records per submission: `Submission{…}` over whatever the
backend's recording type is. `oneshot!` builds its payload through this rather
than through `Submission(r, holds)` so a backend whose payload is wider than one
recording — Vulkan submits a run's front one-shot beside a plan's recording, or
a recording alone — keeps one concrete `Outstanding` element type.
"""
payloadtype(::SubmitChannel{Q,R,T,P}) where {Q,R,T,P} = P

"""
    channelof(ch) -> the backend's submission channel

What the backend submits through. The one field core hands back without looking
inside it.
"""
channelof(ch::SubmitChannel) = ch.channel

"""
    deviceof(ch) -> device

The Mantle device this channel submits to — what `rawfree` and `flush!` are
asked of.

A backend verb rather than a field: a device owns its channels and is built
around them, so a channel that held its device could not be constructed until
the device existed and vice versa. The backend already has the link.
"""
function deviceof end

outstanding(ch::SubmitChannel) = ch.outstanding

# The single-writer invariant, asserted where it can be broken rather than
# documented and hoped for. Recording into one channel from two threads
# interleaves two command streams into one buffer, which is not a race the driver
# reports — it is a corrupted recording.
@inline function ownthread(ch::SubmitChannel)
    Threads.threadid() == ch.thread || error(
        "SubmitChannel is single-writer: it was created on thread $(ch.thread) " *
        "and this is thread $(Threads.threadid()). Take a channel per thread " *
        "with `allocate_batch_queue!`.")
    return nothing
end

# ── 2.2: what a submission holds ─────────────────────────────────────────────

"""
    hold!(ch, obj) -> obj

The submission being built on `ch` must outlive `obj`: keep a reference until the
device says it is finished, and count `obj` as held by a recording that can still
submit until then.

This is what `pin!` meant, with the ownership the other way round. Nothing is
destroyed when the hold is dropped — losing the last reference is what frees.
What is owned here is WHEN.

Returns `obj`, so a call reads as an annotation on the value being used:

    draw!(e, pipeline, hold!(ch, vertices), n)
"""
function hold!(ch::SubmitChannel, obj)
    ownthread(ch)
    isempty(ch.holds) && throw(ArgumentError(
        "hold!: no recording is open on this channel, so there is no submission " *
        "for `obj` to outlive. Hold it from inside the recording that names it."))
    push!(last(ch.holds), obj)
    claim!(obj)
    return obj
end

"""
    holdleaves!(holder, x) -> nothing

Hold every array `x` reaches: through tuples, NamedTuples and immutable structs,
which is the value tree a packer flattens.

A launch is written into a command buffer the device reads after the call
returns, so an array passed to a kernel — or captured by a closure a ray-tracing
shader is compiled from — has to be held before the submission goes. Without it
the array is collected between record and completion, its finalizer runs, and
the device reads released memory.

`holder` is whatever answers [`hold!`](@ref): a channel, or a backend's closed
command buffer, which also records the buffer for ordering. Specialised per
concrete type by `@generated`, so it compiles to straight-line code with no
runtime dispatch and no allocation — this is on the dispatch path.

This was `pin_leaves!`, in the Vulkan backend, over `owner.pinned`. The walk is
a plain Julia value tree and names no driver; only the leaf action ever did.
"""
function holdleaves! end

@inline holdleaves!(holder, a::AbstractArray) = (hold!(holder, a); nothing)

@generated function holdleaves!(holder, x::Tuple)
    Expr(:block, Expr[:(holdleaves!(holder, x[$i])) for i in 1:fieldcount(x)]..., :(nothing))
end
@generated function holdleaves!(holder, x::NamedTuple)
    Expr(:block, Expr[:(holdleaves!(holder, x[$i])) for i in 1:fieldcount(x)]..., :(nothing))
end

# The generic walker, and the sole fallback — there is no `::Any` method for a
# `@generated` to collide with during precompile. The short-circuits are decided
# at codegen time, so the fast path is a literal `nothing`.
@generated function holdleaves!(holder, x::T) where {T}
    # An isbits type cannot transitively contain an array, which is mutable.
    isbitstype(T) && return :(nothing)
    (T <: Ptr || T <: Type || T === Nothing || T <: Symbol || T <: AbstractChar ||
     T <: AbstractString || T <: Module) && return :(nothing)
    n = fieldcount(T)
    n == 0 && return :(nothing)
    Expr(:block, Expr[:(holdleaves!(holder, getfield(x, $i))) for i in 1:n]..., :(nothing))
end

# The two halves of `holders`, spelled once. A resource with no stamp is not
# tracked and costs one dynamic call that returns `nothing`.
@inline function claim!(@nospecialize(obj))
    s = stampof(obj)
    s === nothing || @atomic s.holders += 1
    return nothing
end
@inline function unclaim!(@nospecialize(obj))
    s = stampof(obj)
    s === nothing || @atomic s.holders -= 1
    return nothing
end

"""
    takeholds!(ch) -> Vector{Any}

The holds of the recording being submitted — the innermost open one — handed
over and cleared.

Called by whatever builds a payload, once per submission, and by a backend that
gives the list to a recording instead: a plan's recording is submitted many
times and must outlive every one of them, so its holds belong to the RECORDING
and each submission holds only the recording. Pops the frame `acquire!` pushed,
so a submission's holds cannot be added to after it has gone. An empty frame goes
back to the spare list and the shared empty vector is returned instead, so a
submission that holds nothing allocates nothing.
"""
function takeholds!(ch::SubmitChannel)
    ownthread(ch)
    isempty(ch.holds) && return emptyframe!(ch)
    return pop!(ch.holds)
end

# A hold list to fill or to hand over: a spare one if there is one, a fresh one
# otherwise. Every list this channel has ever made comes back through
# `unhold!`, so in steady state there is always a spare and a submission that
# holds nothing allocates nothing — which is what `test_run_allocates_nothing`
# measures on a plan with no stores.
emptyframe!(ch::SubmitChannel) = isempty(ch.spare) ? Any[] : pop!(ch.spare)

"""
    unhold!(ch, holds)

Drop a hold list: every claim in it goes, the references go, and the list itself
goes back to the spare pool.

The other end of [`hold!`](@ref), called when a submission has passed
(`recycle!`) or when whatever owned the list is released. Not `empty!`: the
claims have to come off, or a resource stays "held by something that can still
submit" for ever and its destroy is never run.
"""
function unhold!(ch::SubmitChannel, h::Vector{Any})
    for o in h
        unclaim!(o)
    end
    empty!(h)
    push!(ch.spare, h)
    return nothing
end

# Open a hold frame for a recording that is about to be built.
pushholds!(ch::SubmitChannel) = (push!(ch.holds, emptyframe!(ch)); nothing)

# Drop the innermost frame without submitting: the recording it belonged to
# never reached the device, so nothing had to outlive it.
function dropholds!(ch::SubmitChannel)
    isempty(ch.holds) && return nothing
    unhold!(ch, pop!(ch.holds))
    return nothing
end

# ── 2.2: order between channels, and destroys that have to wait ──────────────

"""
    crosswaits!(ch, resources, waits)

Collect what a submission on `ch` naming `resources` has to wait for: for each
resource last taken by ANOTHER channel and not yet finished there, the pair
`(channel, token)` that says so, appended to `waits`.

The decision, in core, from the stamps core writes. A backend lowers a pair to
whatever a wait is on its driver — a timeline semaphore and a value on Vulkan —
and knows nothing about why it is waiting.

Any resource a submission NAMES is stamped, not only the ones it writes, so a
read on one channel of something read on another gets a wait it does not
strictly need. That is the safe direction and it is what makes a single stamp
slot correct: because a submission that does not wait is one whose predecessor
had already passed, moving the stamp can never lose an ordering.
"""
function crosswaits!(ch::SubmitChannel, resources, waits)
    for r in resources
        s = stampof(r)
        s === nothing && continue
        c = s.channel
        (c === nothing || c === ch) && continue
        passed(c, s.token) && continue
        push!(waits, (c, s.token))
    end
    return waits
end

"""
    stamp!(ch, token, resources)

Record that `token` on `ch` is now the work that last named each of `resources`.

Called after a submission has actually gone to the device — a submission that
failed never ran, and stamping it would make the next one wait for a token
nothing will signal.
"""
function stamp!(ch::SubmitChannel{Q,R,T,P}, token::T, resources) where {Q,R,T,P}
    for r in resources
        s = stampof(r)
        s === nothing && continue
        s.channel = ch
        s.token = token
    end
    return nothing
end

"""
    waitfor!(s::Stamp)

Block until the device has finished the last submission that named the resource
`s` belongs to. What a host readback waits for; a no-op for a resource nothing
has submitted.
"""
function waitfor!(s::Stamp)
    c = s.channel
    c === nothing && return nothing
    waitfor!(c, s.token)
    return nothing
end

"""
    retire!(ch, obj)

Destroy `obj` — once the device is finished with it.

The explicit half of 2.2. A reference keeps a resource alive until nobody wants
it; this is for the caller who says "I am done with this now" while work that
names it may still be running, which no reference count can see. Callable from
any thread, including a finalizer's: it appends and reads nothing, and the
owning thread decides the rest in [`reclaim!`](@ref).

`retire!(::Pool, ::Region)` is the same word for the same shape one layer down,
and deliberately so: say what is no longer wanted, from wherever you are, and
let the thread that may ask the device decide when it is safe.
"""
function retire!(ch::SubmitChannel, obj)
    lock(() -> push!(ch.pending, obj), ch.pendinglock)
    return nothing
end

"""
    reclaim!(ch) -> Int

Run the destroys that have become safe, and say how many.

Two phases, and the delay between them is the point. A newly retired object is
stamped with the token of the last submission that named it; a later call runs
`rawfree` once [`passed`](@ref) says the device is through with that submission.
An object still held by a recording that can still submit stays pending — it has
no meaningful stamp yet, which is exactly the window `pins` existed for. An
object last named on ANOTHER channel is handed to that channel, because the
token is a value on its timeline and only it can ask.

Owning thread only. Cheap and idempotent, so a submit path can call it every
time; it is a no-op when nothing has been dropped.
"""
function reclaim!(ch::SubmitChannel)
    ownthread(ch)
    # Nothing to destroy is the common case, and it costs one `isempty` each:
    # `deviceof` is asked only when there is something to ask it about.
    (isempty(ch.pending) && isempty(ch.retiring)) && return 0
    dev = deviceof(ch)
    if !isempty(ch.pending)
        taken = ch.taken
        lock(ch.pendinglock)
        try
            append!(taken, ch.pending)
            empty!(ch.pending)
        finally
            unlock(ch.pendinglock)
        end
        keep = 0
        for obj in taken
            s = stampof(obj)
            if s !== nothing && (@atomic s.holders) > 0
                # Named by a recording nobody has submitted yet. Nothing about
                # it reads as a timeline, so it goes back and is asked again.
                keep += 1
                taken[keep] = obj
                continue
            end
            c = s === nothing ? nothing : s.channel
            if c === nothing
                rawfree(dev, obj)          # nothing ever named it
            elseif c === ch
                push!(ch.retiring, (obj, s.token))
            else
                retire!(c::SubmitChannel, obj)
            end
        end
        resize!(taken, keep)
        if keep > 0
            lock(ch.pendinglock)
            try
                append!(ch.pending, taken)
            finally
                unlock(ch.pendinglock)
            end
        end
        empty!(taken)
    end
    isempty(ch.retiring) && return 0
    freed = keep = 0
    for (obj, tok) in ch.retiring
        if passed(ch, tok)
            rawfree(dev, obj)
            freed += 1
        else
            keep += 1
            ch.retiring[keep] = (obj, tok)
        end
    end
    resize!(ch.retiring, keep)
    return freed
end

"""
    drain!(ch)

Give back everything the device has finished with: the submissions that have
passed ([`sweep!`](@ref)) and the destroys that were waiting for them
([`reclaim!`](@ref)).

The one call a backend makes wherever it is already doing bookkeeping — opening
a recording, submitting, flushing. Both halves ask the same question of the same
tokens, so they belong in one place; they were `sweep_retired_batches!`,
`drain_deferred_frees!` and `drain_deferred_as_frees!`, three functions over
three lists on a backend queue.
"""
function drain!(ch::SubmitChannel)
    sweep!(ch)
    reclaim!(ch)
    return nothing
end

# ── 2.3: recordings, reused ──────────────────────────────────────────────────

"""
    makerecording(ch) -> R

A fresh recording object on this channel — a command buffer, an encoder.

One of four primitives a backend supplies. Called only when [`acquire!`](@ref)
finds the free list empty.
"""
function makerecording end

"""
    resetrecording!(ch, r) -> r

Put `r` back into the state a recording starts in. Called before it is handed out
again, not when it is given back, so a recording that is never reused is never
reset.
"""
function resetrecording! end

"""
    destroyrecording!(ch, r)

Destroy `r`. Called when the channel is released, and never while a submission
that carries `r` is outstanding.
"""
function destroyrecording! end

"""
    finishrecording!(ch, r)

A submission carrying `r` has passed: decide what becomes of the recording.

The default pools it, which is what a resettable recording wants. A backend whose
recordings are SINGLE USE overrides this to drop it — a Metal command buffer
cannot be reset and its queue pools for you, so pooling here would accumulate
dead buffers one per submission.

The fourth primitive rather than a `reusable(ch)::Bool`, because the two answers
are two different things to do and not one condition on the same thing.
"""
finishrecording!(ch::SubmitChannel, r) = (push!(ch.free, r); nothing)
finishrecording!(::SubmitChannel, ::Nothing) = nothing

"""
    acquire!(ch) -> R

A recording to build into: one the device has finished with, or a new one.

The reuse discipline is core's, which is the whole of 2.3. It was two `Vector`s
on the backend's queue — `free_submissions` and `free_oneshots` — swept against a
counter the backend also owned.

Drains first, so a channel in steady state finds something and allocates nothing,
and the destroys waiting on the submissions it just gave back run at the same
moment.
"""
function acquire!(ch::SubmitChannel)
    ownthread(ch)
    drain!(ch)
    r = isempty(ch.free) ? makerecording(ch) : resetrecording!(ch, pop!(ch.free))
    pushholds!(ch)
    return r
end

"""
    release!(ch, r)

Give a recording back without submitting it — the path a recording takes when
building it threw.

Distinct from the sweep, which gives back recordings that HAVE been submitted and
have passed. This one never reached the device, so there is nothing to wait for
and its holds are dropped outright.

Through [`finishrecording!`](@ref), because what becomes of a recording is the
same question either way: a backend that gives borrowed scratch back when a
submission passes has to give it back here too, and a backend that cannot reuse
a recording must drop this one as well.
"""
release!(ch::SubmitChannel, r) = (dropholds!(ch); finishrecording!(ch, r); nothing)

"""
    Submission(recording, holds)

What a submission carries: the recording the device is executing, and what that
recording must outlive.

The payload `Outstanding` holds, and what `recycle!` is handed once `passed` says
the device is done: the recording goes back on the free list, the holds are
dropped.
"""
struct Submission{R}
    recording::R
    holds::Vector{Any}
end

# The payload half of the two facts. Nothing is destroyed here — dropping the
# holds is what releases, and what becomes of the recording is
# `finishrecording!`'s.
function recycle!(ch::SubmitChannel, s::Submission)
    finishrecording!(ch, s.recording)
    unhold!(ch, s.holds)
    return nothing
end

"""
    oneshot!(f, ch; tag = nothing) -> token

Record `f` into a recording of its own and submit it.

    oneshot!(ch; tag = :overlays) do e
        draw!(e, pipeline, args, count)
    end

Core's, and it is the shape every ad hoc piece of work takes: a blit, an upload,
the overlay compositor's pass. It was the Vulkan backend's, which is why
RayMakie reached through `Base.get_extension` to call it.

`f` gets whatever the backend's recording hands to a recorder — the same `e` a
plan's emitter is. Anything the recording must outlive goes through
[`hold!`](@ref) on the channel.

On a throw the recording goes back unsubmitted, so a failed build costs nothing
and leaks nothing.
"""
function oneshot!(f, ch::SubmitChannel; tag = nothing)
    ownthread(ch)
    r = acquire!(ch)
    local token
    try
        f(recorder(ch, r))
        token = submit!(ch, r)
    catch
        release!(ch, r)
        rethrow()
    end
    return handover!(ch, token, r; tag)
end

"""
    handover!(ch, token, recording; tag = nothing) -> token

Record that `token` covers `recording` and everything the recording being built
holds — the two lines every submitter writes after [`submit!`](@ref) returns.

Separate from `submit!` because only the caller knows what it handed over: a
pooled one-shot it is giving away, a plan's recording that is submitted again
next run and stays the plan's, or both at once. `submit!` hands command buffers
to the driver; this says what becomes of them.
"""
handover!(ch::SubmitChannel, token, recording; tag = nothing) =
    submitted!(ch, token, payloadtype(ch)(recording, takeholds!(ch)); tag)

"""
    recorder(ch, r) -> emitter

What a caller records through, given a recording. The backend's — a Vulkan
`Emitter` wrapping a command buffer, a Metal encoder — and the one place the
recording type meets the emitter type.
"""
function recorder end
