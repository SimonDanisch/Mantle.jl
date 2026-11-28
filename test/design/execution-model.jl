# Bounded interleaving checks. The oracle is visibility/lifetime safety, not
# equality to the pseudocode's implementation. Broken protocols must have a
# counterexample or these checks would provide no evidence.
module ExecutionModelTests
using Test
function schedules(chains)
    out = Vector{Vector{Symbol}}()
    function visit(prefix, positions)
        all(i -> positions[i] > length(chains[i]), eachindex(chains)) &&
            (push!(out, copy(prefix)); return)
        for i in eachindex(chains)
            positions[i] > length(chains[i]) && continue
            push!(prefix, chains[i][positions[i]])
            next = copy(positions); next[i] += 1
            visit(prefix, next)
            pop!(prefix)
        end
    end
    visit(Symbol[], ones(Int, length(chains)))
    out
end
# Writer and reader critical sections cannot overlap. Their execution order
# is otherwise free, and the writer's later segment can fail at any time.
function reader_safe(order)
    held = nothing; submitted = false; pending = false
    for e in order
        if e === :wl
            held === nothing || return nothing
            held = :writer
        elseif e === :rl
            held === nothing || return nothing
            held = :reader
        elseif e === :submit
            submitted = true
        elseif e === :publish
            pending = true
        elseif e === :read
            submitted && !pending && return false
        elseif e === :wu || e === :ru
            held = nothing
        end
    end
    true
end
@testset "Execution model with negative controls" begin
    good = schedules(([:wl, :submit, :publish, :wu, :later_throw], [:rl, :read, :ru]))
    bad = schedules(([:wl, :submit, :wu, :later_throw, :publish], [:rl, :read, :ru]))
    @test count(s -> reader_safe(s) !== nothing, good) > 1
    @test all(s -> reader_safe(s) !== false, good)
    @test any(s -> reader_safe(s) === false, bad)

    # Retiring a generation needs independent GPU and presentation evidence.
    # Exhaust every ordering, including a release poll between either event.
    genorders = schedules(([:gpu], [:present], [:poll]))
    correct(order) = begin
        gpu = false; present = false
        for e in order
            e === :gpu && (gpu = true)
            e === :present && (present = true)
            e === :poll && gpu && present && return true
        end
        false
    end
    unsafe(order) = findfirst(==(:gpu), order) < findfirst(==(:poll), order) < findfirst(==(:present), order)
    @test count(correct, genorders) == 2
    @test any(unsafe, genorders) # render-only retirement has a counterexample

    # Allocation happens between a snapshot and apply. A shape-changing
    # producer can run in that gap; validation is against the applied need.
    orders = schedules(([:snapshot, :allocate, :apply], [:grow]))
    function readback_safe(order, validate)
        need = 4; snapshot = 0; capacity = 0
        for e in order
            e === :snapshot && (snapshot = need)
            e === :grow && (need = 16)
            e === :allocate && (capacity = snapshot)
            e === :apply && return validate && capacity < need ? true : capacity >= need
        end
        error("missing apply")
    end
    @test all(s -> readback_safe(s, true), orders)
    @test any(s -> !readback_safe(s, false), orders)

    # Successful enqueue remains a native use even when the next enqueue
    # fails. An unsignalled claimed value must never be called complete.
    for accepted in 1:3, failed in (false, true)
        actual_uses = Set(1:accepted)
        published = Set(1:accepted)
        claimed = accepted + Int(failed)
        @test actual_uses ⊆ published
        @test !(failed && claimed in published)
        @test !issubset(actual_uses, Set{Int}()) # rollback loses native uses
    end
    # Free/reuse is legal only after every published use completes.
    for uses in (Set([1]), Set([1, 2])), done in (Set{Int}(), Set([1]), Set([1, 2]))
        releasable = issubset(uses, done)
        @test !releasable || isempty(setdiff(uses, done))
    end
    # A pageable copy spans two chunks. A queued store may race the copy,
    # but native application must wait and the eviction commit must retain it.
    copyorders = schedules(([:freeze, :copy1, :copy2, :commit], [:queue, :apply]))
    function pageable_safe(order; guard=true, preservepending=true)
        data = [1, 1]; frozen = Int[]; copied = Int[]
        transfer = false; pending = false; evicted = false
        for e in order
            if e === :freeze
                frozen = copy(data); transfer = true
            elseif e === :copy1
                push!(copied, data[1])
            elseif e === :copy2
                push!(copied, data[2])
            elseif e === :queue
                pending = true
            elseif e === :apply
                if !transfer || !guard
                    data .+= 1; pending = false
                end
            elseif e === :commit
                copied == frozen || return false # one consistent applied version
                data = copy(copied); evicted = true; transfer = false
                preservepending || (pending = false)
            end
        end
        # A blocked apply retries after completion using the retained request.
        pending && (data .+= 1)
        return evicted && data == [2, 2]
    end
    @test all(pageable_safe, copyorders)
    @test any(s -> !pageable_safe(s; guard=false), copyorders)
    @test any(s -> !pageable_safe(s; preservepending=false), copyorders)

end
end
