# What a submission holds, and what gets reused — phases 2.2 and 2.3 of
# `docs/mantle-owns-it.md`.
#
# `graph/submission.jl` already had the model: `Outstanding(token, payload, tag)`,
# `submitted!`, `sweep!`, `passed`, `recycle!`. What was missing is that anything
# used it. Two backend-side arrangements did the same job instead, and both were
# `Vector`s on a backend queue:
#
#   `free_submissions`, `free_oneshots`   objects to reuse       (2.3)
#   `deferred_frees`, `deferred_as_frees` objects to outlive it  (2.2)
#     plus `deferred_frees_lock`, and `pin!` at 35 call sites
#
# Four lists of two facts. This file is the two facts, once, in core.
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
# device is done. Nothing is destroyed here — dropping the last reference is
# what frees, and a backend object with a finalizer frees itself. What core owns
# is WHEN the reference goes, which is the whole of the question.
#
# ── 2.3, and why the free list is here ───────────────────────────────────────
#
# A recording is a pooled object: acquire one, record into it, submit it, get it
# back when the device is finished. That discipline — which one, when to make a
# new one, when to let go — is the same discipline `Pool` applies to memory, and
# it is not a driver's. The backend supplies three primitives on its own object
# and nothing else: make one, reset one, destroy one, and say what becomes of one
# whose submission has passed.

"""
    SubmitChannel{D,Q,R,T,P}

An independent submission channel: what it submits through, what it has in
flight, what those submissions hold, and the recordings it reuses.

This was `BatchQueue`, whose fields were a Vulkan command pool, a timeline
semaphore, a family index, a back-reference to `VkContext` and four `Vector`s —
a driver arrangement in a shared struct, with type parameters standing in for the
names it could not write. What every backend actually needs is here; what only
one needs is behind [`channelof`](@ref).

`R` is the backend's recording type, `T` its submission token, `P` the payload a
sweep hands back. Parameters rather than `Any` because an abstract `Outstanding`
eltype boxes every token a sweep reads.
"""
mutable struct SubmitChannel{D,Q,R,T,P}
    device::D
    # Whatever the backend submits through: a `VkQueue` and its pool, an
    # `MTLCommandQueue`. Core never looks inside it.
    channel::Q
    outstanding::Vector{Outstanding{T,P}}
    # 2.3: recordings the device has finished with, ready to be recorded into
    # again. Core's list, because when a recording may be reused is a question
    # about a submission having passed and nothing else.
    free::Vector{R}
    # 2.2: what the recording currently being built must outlive. Emptied into
    # the submission's payload when it is submitted.
    holds::Vector{Any}
    # Single-writer: only this thread may record into or submit from this channel.
    thread::Int
end

SubmitChannel{R,T,P}(device::D, channel::Q) where {D,Q,R,T,P} =
    SubmitChannel{D,Q,R,T,P}(device, channel, Outstanding{T,P}[], R[], Any[],
                             Threads.threadid())

"""
    channelof(ch) -> the backend's submission channel

What the backend submits through. The one field core hands back without looking
inside it.
"""
channelof(ch::SubmitChannel) = ch.channel

deviceof(ch::SubmitChannel) = ch.device
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
device says it is finished.

This is what `pin!` meant, with the ownership the other way round. Nothing is
destroyed when the hold is dropped — losing the last reference is what frees, and
a driver object with a finalizer frees itself. What is owned here is WHEN.

Returns `obj`, so a call reads as an annotation on the value being used:

    draw!(e, pipeline, hold!(ch, vertices), n)
"""
function hold!(ch::SubmitChannel, obj)
    ownthread(ch)
    push!(ch.holds, obj)
    return obj
end

"""
    takeholds!(ch) -> Vector{Any}

The holds accumulated since the last submission, handed over and cleared.

Called by whatever builds a payload. Moving the vector rather than copying it:
the channel starts a fresh list, so a submission's holds cannot be added to
after it has gone.
"""
function takeholds!(ch::SubmitChannel)
    ownthread(ch)
    isempty(ch.holds) && return EMPTY_HOLDS
    h = ch.holds
    ch.holds = Any[]
    return h
end

# One shared empty vector for the common case, because a submission that holds
# nothing is most of them and allocating a fresh `Any[]` per submission was the
# kind of per-frame garbage the pool exists to avoid. Never mutated: `takeholds!`
# returns it only when there is nothing to take.
const EMPTY_HOLDS = Any[]

# ── 2.3: recordings, reused ──────────────────────────────────────────────────

"""
    makerecording(ch) -> R

A fresh recording object on this channel — a command buffer, an encoder.

One of three primitives a backend supplies. Called only when
[`acquire!`](@ref) finds the free list empty.
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
    retire!(ch, r)

A submission carrying `r` has passed: decide what becomes of the recording.

The default pools it, which is what a resettable recording wants. A backend whose
recordings are SINGLE USE overrides this to drop it — a Metal command buffer
cannot be reset and its queue pools for you, so pooling here would accumulate
dead buffers one per submission.

The fourth primitive rather than a `reusable(ch)::Bool`, because the two answers
are two different things to do and not one condition on the same thing.
"""
retire!(ch::SubmitChannel, r) = (push!(ch.free, r); nothing)

"""
    acquire!(ch) -> R

A recording to build into: one the device has finished with, or a new one.

The reuse discipline is core's, which is the whole of 2.3. It was two `Vector`s
on the backend's queue — `free_submissions` and `free_oneshots` — swept against a
counter the backend also owned.

Sweeps first, so a channel in steady state finds something and allocates nothing.
"""
function acquire!(ch::SubmitChannel)
    ownthread(ch)
    sweep!(ch)
    isempty(ch.free) && return makerecording(ch)
    return resetrecording!(ch, pop!(ch.free))
end

"""
    release!(ch, r)

Give a recording back without submitting it — the path a recording takes when
building it threw.

Distinct from the sweep, which gives back recordings that HAVE been submitted and
have passed. This one never reached the device, so there is nothing to wait for.
"""
release!(ch::SubmitChannel, r) = (push!(ch.free, r); nothing)

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

# The payload half of the two facts. Nothing is destroyed here — dropping `holds`
# is what releases, and what becomes of the recording is `retire!`'s.
#
# Not `empty!(s.holds)`: the vector may be `EMPTY_HOLDS`, which is shared.
# Dropping the reference to the `Submission` is the release.
recycle!(ch::SubmitChannel, s::Submission) = retire!(ch, s.recording)

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
    return submitted!(ch, token, Submission(r, takeholds!(ch)); tag)
end

"""
    recorder(ch, r) -> emitter

What a caller records through, given a recording. The backend's — a Vulkan
`Emitter` wrapping a command buffer, a Metal encoder — and the one place the
recording type meets the emitter type.
"""
function recorder end
