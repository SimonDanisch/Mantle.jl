# One model of "work handed to the device", and one way to ask whether it is done.
#
# There were five, all meaning the same thing, each consulted by different code:
#
#   `in_flight` batches + `signal_value`   flush!, sweep_retired_batches!
#   `replay_watermark`                     flush!, replay!
#   `am.signal[slot]`                      the argument ring
#   `buf.last_write_bq`, `buf.last_write_val`  vk_free!, sync_access!
#   `arg_pool_frontier`                    the slab rewind — the slabs are gone;
#                                          a region's owner says when it is free
#
# Five records of one fact is five chances to read the wrong one, and the
# comments in tree record two occasions when that happened: `flush!` returning
# before a replay had run, because a replay puts no batch in `in_flight`; and
# the argument ring waiting on a timeline value nothing would ever signal,
# because the frame that reserved it had not been submitted. Both were patched where they
# surfaced. Neither could be fixed where it was caused, because there was no
# single place that knew what was outstanding.
#
# This is that place. Core keeps the list; the backend answers three questions
# about a token, which is exactly the contract `Pool` already asks of it —
# `fence`/`passed`/`waitfor` in `memory/pool.jl`. The queue path was the odd one
# out, not the memory path.
#
# A token is opaque here on purpose. Vulkan's is a timeline value, Metal's is a
# committed `MTLCommandBuffer`, a synchronous backend's is `nothing`. Core never
# compares two tokens or does arithmetic on one; it asks `passed` and `waitfor`.
# That is what stopped `UInt64` timeline values from being a portable concept the
# moment a second backend arrived.

"""
    Outstanding(token, payload, tag)

One submission the device has been given and may not have finished.

`token` is what the backend answers `passed` and `waitfor` about. `payload` is
what the backend wants back once it has passed — the closed command buffers the
submission carried, with their pins and scratch — and the sweep hands it to
[`recycle!`](@ref). A backend whose submissions own nothing passes `nothing`.

`tag` is for diagnostics only — a name, a frame number, whatever the submitter
wants in an error message. Nothing dispatches on it.

The payload used to be a second list on the queue (`in_flight`, holding the
same submissions in the same order), swept by a second function against the
same counter. Two lists of one fact, each consulted by different code, is what
this file's opening paragraph is about.
"""
struct Outstanding{T,P}
    token::T
    payload::P
    tag::Any
end

"""
    submitted!(queue, token, payload; tag = nothing) -> token

Record that `token` covers work now on its way to the device, and that
`payload` is what to give back once it has passed.

Every path that hands work over calls this and nothing else records it: a
recorded batch, a run of a recorded plan, a one-off upload. That is the whole
point — before this, a replay was invisible to `flush!` because it created no
batch, and the fix was a second field rather than a second caller here.
"""
function submitted!(q, token, payload; tag = nothing)
    push!(outstanding(q), Outstanding(token, payload, tag))
    return token
end

"""
    recycle!(queue, payload)

Give a passed submission's payload back: its command buffers to the queue's
pool, and everything they pinned or borrowed released. [`sweep!`](@ref) calls
it, oldest first, once `passed` says so; a backend calls it directly for a
buffer that never reached the device. A backend whose submissions own nothing
records `nothing` and needs no method.
"""
function recycle! end
recycle!(q, ::Nothing) = nothing

"""
    newest(queue) -> token or nothing

The most recent submission, or `nothing` when the queue is idle.

What [`flush!`](@ref) waits for. Completion is monotone on a queue — a backend's
tokens are ordered by submission — so waiting for the newest waits for all of
them, and no caller has to fold over the list to find a maximum. `flush!` used
to do exactly that fold, over two different lists, and get it wrong for replays.
"""
newest(q) = isempty(outstanding(q)) ? nothing : last(outstanding(q)).token

"""
    sweep!(queue) -> Int

Give back the submissions the device has finished on THIS queue, and say how
many.

The question is asked of the queue — `passed(queue, token)` — and not of the
device, because a token is a value on the queue's own timeline. A device with
several queues (Vulkan: RayMakie's graphics queue beside the default one)
answers `passed(device, token)` from its default queue's counter, which runs
far ahead of a queue that carries three submissions a frame; asked that way,
this swept the second queue's submissions the moment they were made, gave
their command buffers back to the pool while the GPU was still executing
them, and the next one-shot on that queue began the same buffer again
mid-flight. `test_sweep_per_queue.jl` pins it.

`sweep!` and not `retire!`: `retire!` is already `Pool`'s, for giving a region
back, and one name meaning two things is how the five records got here.

Called wherever the queue is already doing bookkeeping — opening a batch,
flushing — rather than on a timer. Cheap: it stops at the first token that has
not passed, since completion is in order.
"""
function sweep!(q)
    out = outstanding(q)
    n = 0
    for o in out
        passed(q, o.token) || break
        recycle!(q, o.payload)
        n += 1
    end
    n == 0 || deleteat!(out, 1:n)
    return n
end

"""
    idle(queue) -> Bool

Has everything submitted to this queue finished?
"""
idle(q) = (sweep!(q); isempty(outstanding(q)))

"""
    outstanding(queue) -> Vector{Outstanding}

The submissions in flight, oldest first. A backend's queue type answers this.
"""
function outstanding end
