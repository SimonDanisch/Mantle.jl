"""
    Outstanding(token, payload, tag)

One entry of a queue's outstanding list: a submitted token and the payload
retained until completion hands it to `recycle!`. `tag` is diagnostic only.
Tokens are opaque: query `passed`/`waitfor` on their own queue, never compare
them.
"""
struct Outstanding{T,P}
    token::T
    payload::P
    tag::Any
end

"""
    submitted!(queue, token, payload; tag = nothing) -> token

Track submitted work and retain `payload` until completion. Every submission path,
including plan replays and one-shot uploads, must register here.
"""
function submitted!(q, token, payload; tag = nothing)
    push!(outstanding(q), Outstanding(token, payload, tag))
    return token
end

"""
    recycle!(queue, payload)

Return a completed payload's recordings and scratch and release its holds; `sweep!`
calls it oldest first. Never-submitted recordings use their release path instead.
"""
function recycle! end
recycle!(q, ::Nothing) = nothing

"""
    newest(queue) -> token or nothing

The newest outstanding token. A queue completes in submission order, so waiting
for it covers every older submission on that queue.
"""
newest(q) = isempty(outstanding(q)) ? nothing : last(outstanding(q)).token

"""
    sweep!(queue) -> Int

Recycle the completed prefix of the outstanding list, up to the first unfinished
submission, and return its length. Completion is `passed(queue, token)`, never the
device's default timeline: tokens of different queues are not interchangeable.
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

Sweep completed work and return whether nothing is outstanding.
"""
idle(q) = (sweep!(q); isempty(outstanding(q)))

"""
    outstanding(queue) -> Vector{Outstanding}

This queue's outstanding submissions, oldest first.
"""
function outstanding end
