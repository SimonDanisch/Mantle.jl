# A stack walk over a tree whose loop `continue`s past the nodes it rejects.
#
# StructurizeCFG gives such a loop header phis that look distinct but carry the
# same value (the hit count, the stack pointer), and RADV's compiler once stored
# the copy that had not been incremented: the count came back short. Lava turned
# spirv-opt on for every kernel to merge them (April 2026), and merges them itself
# since (`merge_equivalent_loop_phis!`). This pins the pattern on every device,
# whether or not spirv-opt runs: the count and the order the walk visited the
# accepted nodes in, against the same walk on the host.
#
# The second kernel indexes the stack dynamically in both branches of the
# `continue`, the variant the April commit still listed as miscompiled.

using Test, Mantle, KernelAbstractions, StaticArrays
include(joinpath(@__DIR__, "testbackend.jl"))
const KA = KernelAbstractions

"""A complete binary tree of `n` nodes in heap order; node `k`'s children are `2k` and `2k + 1`."""
wc_children(k, n) = (2k <= n ? Int32(2k) : Int32(0), 2k + 1 <= n ? Int32(2k + 1) : Int32(0))

# Visits every node reachable through accepted nodes; a rejected node is skipped
# with `continue` before its children are pushed. Returns the accepted count and a
# checksum of the visit order.
@inline function wc_walk(key, n, threshold, salt)
    stack = MVector{32, Int32}(undef)
    sp = Int32(1)
    @inbounds stack[sp] = Int32(1)
    hits = Int32(0)
    order = UInt32(0)
    while sp > Int32(0)
        @inbounds node = stack[sp]
        sp -= Int32(1)
        @inbounds k = key[node]
        if k < threshold
            continue
        end
        hits += Int32(1)
        order = order * UInt32(31) + UInt32(node) + salt
        l, r = wc_children(node, n)
        if l != Int32(0)
            sp += Int32(1)
            @inbounds stack[sp] = l
        end
        if r != Int32(0)
            sp += Int32(1)
            @inbounds stack[sp] = r
        end
    end
    return hits, order
end

# Both branches of the `continue` index the stack with a value the walk computes.
@inline function wc_dyn_both(key, n, threshold)
    stack = MVector{32, Int32}(undef)
    sp = Int32(1)
    @inbounds stack[sp] = Int32(1)
    acc = Int32(0)
    while sp > Int32(0)
        @inbounds node = stack[sp]
        sp -= Int32(1)
        @inbounds k = key[node]
        l, r = wc_children(node, n)
        if k < threshold
            # Rejected: still descend into the right child, through the stack.
            if r != Int32(0)
                sp += Int32(1)
                @inbounds stack[sp] = r
                @inbounds acc += stack[sp] & Int32(1)
            end
            continue
        end
        acc += k
        if l != Int32(0)
            sp += Int32(1)
            @inbounds stack[sp] = l
            @inbounds acc += stack[sp] & Int32(3)
        end
    end
    return acc
end

@kernel function wc_walk_kernel!(hits, orders, @Const(key), n::Int32)
    i = @index(Global)
    h, o = wc_walk(key, n, Int32(i), UInt32(i))
    @inbounds hits[i] = h
    @inbounds orders[i] = o
end

@kernel function wc_dyn_both_kernel!(out, @Const(key), n::Int32)
    i = @index(Global)
    @inbounds out[i] = wc_dyn_both(key, n, Int32(i))
end

@testset "a tree walk that continues past rejected nodes" begin
    backend = TESTBACKEND
    n = Int32(63)
    lanes = 128
    key = Int32[(37 * k) % 101 for k in 1:n]
    dkey = KA.allocate(backend, Int32, n)
    copyto!(dkey, key)

    hits = KA.allocate(backend, Int32, lanes)
    orders = KA.allocate(backend, UInt32, lanes)
    wc_walk_kernel!(backend, 64)(hits, orders, dkey, n; ndrange = lanes)
    acc = KA.allocate(backend, Int32, lanes)
    wc_dyn_both_kernel!(backend, 64)(acc, dkey, n; ndrange = lanes)
    KA.synchronize(backend)

    want = [wc_walk(key, n, Int32(i), UInt32(i)) for i in 1:lanes]
    @test Array(hits) == first.(want)
    @test Array(orders) == last.(want)
    # Not vacuous: some lanes reject nodes and still accept others.
    @test any(h -> 0 < h < n, first.(want))
    @test Array(acc) == [wc_dyn_both(key, n, Int32(i)) for i in 1:lanes]
end
