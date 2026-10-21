# Legacy channel ownership: retain referenced resources through completion, defer
# explicit destruction, order cross-channel uses and recycle completed recordings.
# Target token retirement replaces this with one release engine (docs/api.md §4.3).

"""
    Stamp{T}

Last submitting channel/token and atomic recording-holder count. A channel of nothing
means never submitted. Holders protect closed, unsubmitted recordings even before a token
exists; core updates the stamp and backends provide its storage through stampof.
"""
mutable struct Stamp{T}
    channel::Any
    token::T
    @atomic holders::Int
end
Stamp{T}() where {T} = Stamp{T}(nothing, zero(T), 0)

"""
    stampof(obj) -> Stamp or nothing

Return the resource's submission stamp, or nothing for objects managed only by retained
references (for example pipelines or host arrays).
"""
stampof(@nospecialize(obj)) = nothing

"""
    SubmitChannel{Q,R,T,P}

Core submission, hold, retirement and recording-pool state; Q is the native channel,
R its recording type, T its token and P its concrete outstanding payload type. Recording
and submission are single-writer; finalizers may enqueue retirement from any thread.
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
    # One hold frame per open recording, innermost last. Nested one-shots must not
    # consume their outer recording's references.
    holds::Vector{Vector{Any}}
    # Emptied hold vectors returned by unhold!, reused by emptyframe!.
    spare::Vector{Vector{Any}}
    # Any thread may enqueue pending destroys; only the owner stamps and reclaims
    # retiring entries against this channel's timeline.
    pending::Vector{Any}
    # Finalizers can re-enter retire! at allocation safepoints inside this lock.
    # Use a ReentrantLock; a SpinLock would deadlock on same-task re-entry.
    pendinglock::ReentrantLock
    retiring::Vector{Tuple{Any,T}}
    # Scratch for draining pending requests. Driver destructors and cross-channel
    # handoff run outside pendinglock.
    taken::Vector{Any}
    # Single-writer: only this thread may record into or submit from this channel.
    thread::Int
end

# What it submits through and how much is in flight, not every outstanding
# submission and every recording it holds.
Base.show(io::IO, ch::SubmitChannel) =
    print(io, "SubmitChannel(", ch.channel, ", ", length(ch.outstanding), " outstanding)")

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

Concrete outstanding payload type used by oneshot!/handover!, including backends whose
payload can combine a per-run recording with reusable plan recordings.
"""
payloadtype(::SubmitChannel{Q,R,T,P}) where {Q,R,T,P} = P

"""
    channelof(ch) -> the backend's submission channel

Return the opaque native submission channel; core does not inspect it.
"""
channelof(ch::SubmitChannel) = ch.channel

"""
    deviceof(ch) -> device

Return the owning device used for rawfree and flush!. Backend implementations resolve
the channel/device construction cycle.
"""
function deviceof end

# SubmitChannel is defined after runtime/api.jl, so its todevice method lives here.
todevice(ch::SubmitChannel) = todevice(deviceof(ch))

outstanding(ch::SubmitChannel) = ch.outstanding

"""
    WrongThread(owner, caller)

A recording or submission touched a channel outside its owning thread. Callers may catch
this exception; all ownership checks use ownthread.
"""
struct WrongThread <: Exception
    owner::Int
    caller::Int
end

Base.showerror(io::IO, e::WrongThread) = print(io,
    "SubmitChannel is single-writer: it belongs to thread $(e.owner) and this ",
    "is thread $(e.caller). Take a channel per thread with `allocate_batch_queue!`.")

"""
    ownthread(ch) -> nothing

Enforce the single-writer invariant; throw WrongThread before recording or submission.
"""
@inline function ownthread(ch::SubmitChannel)
    Threads.threadid() == ch.thread || throw(WrongThread(ch.thread, Threads.threadid()))
    return nothing
end

# ── 2.2: what a submission holds ─────────────────────────────────────────────

"""
    hold!(ch, obj) -> obj

Keep obj alive until this recording/submission is released and increment its stamp
holder count. Requires an open hold frame. Returns obj; dropping the hold releases the
reference/claim and does not itself invoke a driver destructor.
"""
function hold!(ch::SubmitChannel, obj)
    ownthread(ch)
    isempty(ch.holds) && throw(ArgumentError(
        "hold!: no recording is open on this channel (ch=$(objectid(ch)), " *
        "thread=$(ch.thread)), so there is no submission for `obj` to outlive. " *
        "Hold it from inside the recording that names it."))
    push!(last(ch.holds), obj)
    claim!(obj)
    return obj
end

"""
    holdleaves!(holder, x) -> nothing

Hold array leaves in tuples, NamedTuples and struct fields, including captured shader
arrays. DeviceArray is a leaf: descending into driver ownership can encounter cycles.
Generated walks specialize the traversal; holder supplies hold!.
"""
function holdleaves! end

@inline holdleaves!(holder, a::AbstractArray) = (hold!(holder, a); nothing)

# DeviceArray is not an AbstractArray, but remains a leaf: descending through its
# region into driver ownership can encounter cycles.
@inline holdleaves!(holder, a::DeviceArray) = (hold!(holder, a); nothing)

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
    takeholds!(ch) -> Vector{Any} or nothing

Pop the innermost open recording's hold vector, or return nothing if no frame is open.
One-shot submissions own the vector; reusable recordings retain it until released.
"""
function takeholds!(ch::SubmitChannel)
    ownthread(ch)
    isempty(ch.holds) && return nothing
    return pop!(ch.holds)
end

# Reuse emptied hold vectors to avoid allocation after warmup.
emptyframe!(ch::SubmitChannel) = isempty(ch.spare) ? Any[] : pop!(ch.spare)

"""
    unhold!(ch, holds)

Drop every claim/reference and return the emptied vector to the channel's spare pool.
Nothing means the submission owned no hold frame.
"""
function unhold!(ch::SubmitChannel, h::Vector{Any})
    for o in h
        unclaim!(o)
    end
    empty!(h)
    push!(ch.spare, h)
    return nothing
end

# No submission-owned frame uses nothing; allocating an empty vector per replay
# would defeat the zero-allocation render path.
unhold!(::SubmitChannel, ::Nothing) = nothing

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

Append (channel,token) waits for resources last named by another unfinished channel.
All named resources are stamped, including reads, so this legacy single-stamp model
conservatively serializes read/read uses too; transferring the stamp cannot lose an ordering.
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

Update last-use channel/token only after successful submission. Stamping a failed submit
would make later work wait on a token that will never signal.
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

Wait for the last submission naming the stamped resource; no-op before its first submit.
"""
function waitfor!(s::Stamp)
    c = s.channel
    c === nothing && return nothing
    waitfor!(c, s.token)
    return nothing
end

"""
    retire!(ch, obj)

Enqueue an explicit destroy request from any thread, including finalizers. The owning
thread later checks recording holds and completion in reclaim! before invoking rawfree.
"""
function retire!(ch::SubmitChannel, obj)
    lock(() -> push!(ch.pending, obj), ch.pendinglock)
    return nothing
end

"""
    holdsretired(ch) -> Bool

Whether pending or token-waiting destroys make a channel drain useful to automatic trim.
"""
holdsretired(ch::SubmitChannel) =
    lock(() -> !isempty(ch.pending), ch.pendinglock) || !isempty(ch.retiring)
holdsretired(ch) = false

"""
    reclaim!(ch) -> Int

On the owning thread, defer recording-held objects, hand foreign-channel objects to their
last channel, and destroy safe objects. Driver calls and cross-channel handoff run outside
pendinglock. Return the count freed from the completed-token retiring list.
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

Recycle completed submissions, then reclaim safe explicit destroys.
"""
function drain!(ch::SubmitChannel)
    sweep!(ch)
    reclaim!(ch)
    return nothing
end

# ── 2.3: recordings, reused ──────────────────────────────────────────────────

"""
    makerecording(ch) -> R

Create a native recording when the channel's free list is empty.
"""
function makerecording end

"""
    resetrecording!(ch, r) -> r

Reset a completed pooled recording before reuse, rather than at return time.
"""
function resetrecording! end

"""
    destroyrecording!(ch, r)

Destroy a recording at channel teardown only after all submissions carrying it have passed.
"""
function destroyrecording! end

"""
    finishrecording!(ch, r)

Handle a completed or never-submitted recording. Default: pool it. Single-use backends
must override this to drop it; pooling completed Metal command buffers would retain them.
"""
finishrecording!(ch::SubmitChannel, r) = (push!(ch.free, r); nothing)
finishrecording!(::SubmitChannel, ::Nothing) = nothing

"""
    acquire!(ch) -> R

Drain completed work, take/reset a free recording or create one, and open its hold frame.
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

Release a recording that never submitted: drop its hold frame and call finishrecording!.
"""
release!(ch::SubmitChannel, r) = (dropholds!(ch); finishrecording!(ch, r); nothing)

"""
    Submission(recording, holds)

Outstanding recording and its hold frame. Completion recycles the recording and drops
claims/references; nothing means this submission owns no hold frame.
"""
struct Submission{R}
    recording::R
    # `nothing` when the submission held nothing — see `unhold!`.
    holds::Union{Nothing,Vector{Any}}
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

Acquire a recording, invoke f(recorder), submit and transfer its holds to the completion
payload. If recording/submission throws, release it unsubmitted and rethrow.
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

# Emission receives the plan/device/channel explicitly; do not use task-global
# open-batch state to route dispatches.

"""
    handover!(ch, token, recording; tag = nothing) -> token

Track the submitted recording and its submission-owned holds under token. Kept separate
from native submit! because reusable recordings retain their own holds across replays.
"""
handover!(ch::SubmitChannel, token, recording; tag = nothing) =
    submitted!(ch, token, payloadtype(ch)(recording, submissionholds!(ch, recording)); tag)

"""
    submissionholds!(ch, recording) -> holds or `nothing`

Return the hold frame owned by this submission. Default pops the one-shot frame.
Reusable recordings take their frame when sealed and override this to nothing; replay
must not pop an unrelated outer recording's frame.
"""
submissionholds!(ch::SubmitChannel, recording) = takeholds!(ch)

"""
    recorder(ch, r) -> emitter

Return the backend emitter used to record into r.
"""
function recorder end
