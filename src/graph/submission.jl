# One outstanding list per queue. Tokens are opaque; ask passed/waitfor on their
# originating queue rather than comparing or doing arithmetic on them.

"""
    Outstanding(token, payload, tag)

One submitted token and its retained payload. Completion hands the payload to recycle!.
Tag is diagnostic only. This is the single outstanding list for the queue.
"""
struct Outstanding{T,P}
    token::T
    payload::P
    tag::Any
end

"""
    submitted!(queue, token, payload; tag = nothing) -> token

Track successfully submitted work and retain payload until completion. All submission
paths, including plan replays and one-shot uploads, must register here.
"""
function submitted!(q, token, payload; tag = nothing)
    push!(outstanding(q), Outstanding(token, payload, tag))
    return token
end

"""
    recycle!(queue, payload)

Return a completed payload's recordings/scratch and release its holds. Sweep calls this
oldest first; never-submitted recordings use their release path.
"""
function recycle! end
recycle!(q, ::Nothing) = nothing

"""
    newest(queue) -> token or nothing

Newest outstanding token, or nothing. Queue completion follows submission order,
so waiting for this token covers every older submission on the same queue.
"""
newest(q) = isempty(outstanding(q)) ? nothing : last(outstanding(q)).token

"""
    sweep!(queue) -> Int

Recycle the completed prefix and return its length. Query passed(queue,token), never
the default device timeline: tokens from separate queues are not interchangeable.
Stop at the first unfinished submission; see test_sweep_per_queue.jl.
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

Sweep completed work and report whether the queue has outstanding submissions.
"""
idle(q) = (sweep!(q); isempty(outstanding(q)))

"""
    outstanding(queue) -> Vector{Outstanding}

Return this queue's outstanding submissions, oldest first.
"""
function outstanding end
