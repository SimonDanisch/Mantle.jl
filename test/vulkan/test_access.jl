# What a kernel does to its arguments, read off the kernel.
#
# This is the analysis every declaration now comes from, so what it has to be is
# not "usually right": EXACT where it can be, and CONSERVATIVE where it cannot —
# claiming an access that does not happen costs a barrier, and missing one costs
# a race. Both halves are pinned here, on kernels small enough that the answer is
# obvious by reading them.
#
# The cases are the ones a walk over typed IR gets wrong:
#
#   * a scalar argument came back read+write, because an argument was tainted
#     before its type was asked whether it could reach memory at all;
#   * an argument whose LENGTH a kernel reads came back read, for the same
#     reason one step later;
#   * a kernel written `f(queue, args...)` — which is every wavefront stage —
#     gave all twenty arguments whatever any one of them got, because the
#     inferred IR collapses a vararg tail into one argument;
#   * an atomic came back as an ordinary read-write, because Lava's atomics are
#     `llvmcall` and the module is named by a `GlobalRef` rather than inline.
# Here rather than in `test/` because every testset below builds a device and
# runs the walk through the interpreter THIS backend compiles with — which is
# the whole point of the analysis (see `accessof`), and also a backend name in
# every line. The half that is a question about types and not about a device is
# `test/test_access.jl`.
using Test, Mantle, KernelAbstractions, Lava
import Atomix, Adapt
import KernelInterface as KI
const M = Mantle
const TESTBACKEND = M.VulkanAPI()

@kernel function acc_mixed!(dst, @Const(src), @Const(unused), scale::Float32)
    i = @index(Global)
    @inbounds dst[i] = src[i] * scale
end

@kernel function acc_atomic!(acc, @Const(src))
    i = @index(Global)
    @inbounds Atomix.@atomic acc[i] += src[i]
end

@kernel function acc_lengthonly!(dst, @Const(other))
    i = @index(Global)
    @inbounds dst[i] = Float32(length(other))
end

# Not inlined, so the walk has to follow the call to find the store.
@noinline function acc_fill_one!(v, i)
    @inbounds v[i] = 1f0
    return nothing
end
@kernel function acc_helper!(dst)
    i = @index(Global)
    acc_fill_one!(dst, i)
end

struct AccPair{A}
    a::A
    b::A
end
Adapt.@adapt_structure AccPair

@kernel function acc_struct!(q::AccPair, @Const(src))
    i = @index(Global)
    @inbounds q.a[i] = src[i]
end

# A work queue's append, which is the shape the wavefront stages have: claim a
# slot with an atomic, then store the payload at it. The payload store is an
# ORDINARY store and no two invocations can collide, which is what makes the
# whole queue unordered against another pass doing the same.
struct AccQueue{I,S}
    items::I
    size::S
end
Adapt.@adapt_structure AccQueue

@kernel function acc_append!(q::AccQueue, @Const(src))
    i = @index(Global)
    v = @inbounds src[i]
    slot = Atomix.@atomic q.size[1] += Int32(1)
    @inbounds q.items[slot] = v
end

# The higher-order shape: the kernel takes the real kernel and a tail of
# arguments, and splats them. Every stage of Hikari's integrator is this.
@kernel function acc_map!(f, n::Int32, args...)
    i = @index(Global)
    i <= n && f(i, args...)
end
acc_inner!(i, dst, src, spare) = (@inbounds dst[i] = src[i]; nothing)

"""`Touch` as a string: `R`, `W`, and `~` for a write nothing can collide with."""
show_t(t) = (t.read ? "R" : "-") * (t.write ? "W" : "-") * (t.write && t.atomic ? "~" : "")
touches(dev, k, args, nd; group = nothing) =
    show_t.(M.kerneltouches(dev, k, args, nd, group))

@testset "inferred access: exact where it can be" begin
    dev = M.Device(TESTBACKEND)
    n = 256
    dst = M.devicearray(dev, Float32, n)
    src = M.devicearray(dev, fill(1f0, n))
    other = M.devicearray(dev, Float32, n)

    # Written, read, untouched, and a scalar that has no bytes to touch.
    @test touches(dev, acc_mixed!, (dst, src, other, 2f0), n) == ["-W", "R-", "--", "--"]

    # An argument a kernel only asks the LENGTH of is not read: the size is a
    # by-value field of the array handle, not a load from its storage.
    @test touches(dev, acc_lengthonly!, (dst, other), n) == ["-W", "--"]

    # Through a call the optimiser left standing.
    @test touches(dev, acc_helper!, (dst,), n) == ["-W"]

    # Through a struct of arrays: `q.a` is written, and `q` is what the graph
    # declares, so the whole container is written and `src` is only read.
    q = AccPair(M.devicearray(dev, Float32, n), M.devicearray(dev, Float32, n))
    @test touches(dev, acc_struct!, (q, src), n) == ["-W", "R-"]
end

@testset "inferred access: an atomic is a write nothing can collide with" begin
    dev = M.Device(TESTBACKEND)
    n = 256
    acc = M.devicearray(dev, Float32, n)
    src = M.devicearray(dev, fill(1f0, n))
    @test touches(dev, acc_atomic!, (acc, src), n) == ["RW~", "R-"]

    # …and so is a plain store at an index claimed from one. The counter and the
    # payload are fields of ONE argument, which is what makes the disjointness
    # provable: a slot claimed from this queue's counter is a slot in this
    # queue's payload, and no two invocations — or two passes appending to it —
    # can be handed the same one.
    q = AccQueue(M.devicearray(dev, Float32, n), M.devicearray(dev, Int32, 1))
    @test touches(dev, acc_append!, (q, src), n) == ["RW~", "R-"]
end

@testset "inferred access: a vararg tail is not one argument" begin
    dev = M.Device(TESTBACKEND)
    n = 64
    dst = M.devicearray(dev, Float32, n)
    src = M.devicearray(dev, fill(2f0, n))
    spare = M.devicearray(dev, Float32, n)
    # `acc_map!` has FIVE arguments in the inferred IR — the kernel, the
    # iteration context, `f`, `n`, and one tuple holding the other three — while
    # the signature has seven. Without the element index, `dst`, `src` and
    # `spare` would all answer whatever any one of them got.
    @test touches(dev, acc_map!, (acc_inner!, Int32(n), dst, src, spare), n) ==
          ["--", "--", "-W", "R-", "--"]
end

# A kernel calling an intrinsic this backend does not implement — which is what
# `KI.localmemory` WAS, before its signature grew a call-site id. Every path
# through the body throws, so it infers to `Union{}` and its IR is unreachable
# from the first statement on.
function acc_missing_intrinsic end

function acc_uninferable!(dst, n::Int)
    i = KI.get_global_id().x
    v = acc_missing_intrinsic(Float32(i))
    @inbounds dst[i] = v
    return
end

@testset "inferred access: a kernel that does not infer is refused" begin
    dev = M.Device(TESTBACKEND)
    n = 64
    dst = M.devicearray(dev, Float32, n)
    # "No store" is the one answer that must never be a guess — it is the answer
    # that removes a barrier. Widened to read+write it is safe for the barrier
    # phase and indistinguishable from a proof; the kernel does
    # not compile either, so the refusal is the same error arriving earlier with
    # the argument named.
    err = try
        touches(dev, acc_uninferable!, (dst, n), n)
        nothing
    catch e
        e
    end
    @test err isa M.UnanalysableAccess
    msg = sprint(showerror, err)
    @test occursin("cannot infer", msg)
    @test occursin("acc_uninferable!", msg)
    @test occursin("will not guess", msg)
end

@testset "inferred access: a device array carries" begin
    # The one `carries` answer that belongs to a backend. The rest of them, and
    # everything else the walk decides from types alone, is in the portable
    # `test/test_access.jl`.
    @test M.carries(M.LavaDeviceArray{Float32,1}) === true
end

@testset "inferred access: what a dispatch declares" begin
    dev = M.Device(TESTBACKEND)
    n = 64
    g = M.Graph(dev)
    dst = M.Buffer(dev, zeros(Float32, n))
    src = M.Buffer(dev, fill(1f0, n))
    p = M.dispatch!(g, acc_mixed!, (dst, src, src, 2f0), n)

    byid = Dict(id => U for (id, U) in p.usages)
    @test byid[M.resourceid(g, dst)] === Storage{BufferKind, WriteOnly}
    @test byid[M.resourceid(g, src)] === Storage{BufferKind, ReadOnly}
    @test p.name == "acc_mixed!"          # the kernel names its own pass

    # A container is declared by its LEAVES, which is what a barrier is scoped to.
    #
    # Built from `devicearray` and not from `Buffer`, and that is a real
    # constraint rather than a preference: `resolve`/`storage` descends into a
    # `Tuple` but not into a struct, so an `AccPair{Buffer}` reaches the walk
    # with graph handles still inside it, `setindex!` has no method for one, and
    # the kernel infers to nothing but its throw path. It cannot RUN either --
    # the packer would be handed a non-isbits `AccPair{Buffer}` and GPUCompiler
    # would refuse it -- so a walk that widens that throw path to read+write
    # makes this assertion pass on a dispatch that can never execute.
    #
    # Resolving a struct of resources elementwise is the remaining half of what
    # `storage(::Tuple)` did; until then a container argument holds device
    # arrays, which is what every consumer of this shape already passes.
    qa = M.devicearray(dev, zeros(Float32, n))
    qb = M.devicearray(dev, zeros(Float32, n))
    q = AccPair(qa, qb)
    p2 = M.dispatch!(g, acc_struct!, (q, src), n; name = "struct")
    @test M.resourceid(g, qa) in first.(p2.usages)
    @test M.resourceid(g, qb) in first.(p2.usages)
    # And `src` is only read, which is the half that the widening hid: with the
    # container's write coming from a throw path, everything in the pass looked
    # read+write.
    byid2 = Dict(id => U for (id, U) in p2.usages)
    @test byid2[M.resourceid(g, src)] === Storage{BufferKind, ReadOnly}
    @test byid2[M.resourceid(g, qa)] === Storage{BufferKind, WriteOnly}

    # A slice is an argument, and it is the slice the pass is recorded touching.
    sl = M.slice(g, dst, 1:32)
    p3 = M.dispatch!(g, acc_mixed!, (sl, src, src, 3f0), 32; name = "sliced")
    @test M.resourceid(g, sl) in first.(p3.usages)
    @test !(M.resourceid(g, dst) in first.(p3.usages))
end

@testset "inferred access: the device answers from a cache" begin
    dev = M.Device(TESTBACKEND)
    n = 64
    dst = M.devicearray(dev, Float32, n)
    src = M.devicearray(dev, fill(1f0, n))
    M.kerneltouches(dev, acc_mixed!, (dst, src, src, 1f0), n, nothing)
    c = M.accesscache(dev)
    @test c isa M.AccessCache
    # The second ask is a lookup, not an inference pass — the difference is
    # seconds per kernel on a wavefront stage, so it is worth pinning that it
    # happens at all. Measured rather than counted, because the cache is dropped
    # whole when the world age moves and a test file moves it constantly.
    t = @elapsed M.kerneltouches(dev, acc_mixed!, (dst, src, src, 1f0), n, nothing)
    @test t < 0.1
end

@testset "inferred access: a device intrinsic declares what it does" begin
    # `_lava_coopmat_load_f16_16x16_a` is not a `load` instruction, and the
    # module it lives in has none: an `llvmcall` for a backend intrinsic is a
    # `declare` plus a `call`, so there is nothing in the IR that says a load is
    # a read. Classified by `occursin("load ", src)`, which that name misses by
    # a space, it falls through to read+write.
    #
    # The op is what both of Lava's naming schemes put first and what Lava's own
    # emitter parses them back out as, so a prefix covers every load and store
    # exactly. An op that takes a pointer and is not named for what it does gets
    # `nothing`, which is a refusal rather than a guess.
    @test M.intrinsic_usage(:_lava_coopmat_load_f16_16x16_a) === M.READ
    @test M.intrinsic_usage(:_lava_coopmat_loadw2_f16_16x16_b) === M.READ
    @test M.intrinsic_usage(:_lava_coopmat_store_f32_16x16_acc) === M.WRITE
    @test M.intrinsic_usage(:_lava_coopmat_storew_f16_16x16_acc) === M.WRITE
    @test M.intrinsic_usage(:_lava_tensor_load_2_1_f16_16x16_a) === M.READ
    @test M.intrinsic_usage(:_lava_tensor_loadv_2_1_f16_16x16_b) === M.READ
    @test M.intrinsic_usage(:_lava_tensor_store_2_0_f32_16x16_acc) === M.WRITE
    @test M.intrinsic_usage(:_lava_tensor_storev_2_0_f32_16x16_acc) === M.WRITE
    # Not memory, so never asked — the walk short-circuits on an intrinsic with
    # no tainted operand — and `nothing` if it ever is.
    @test M.intrinsic_usage(:_lava_tensor_setstride_2_1) === nothing
    @test M.intrinsic_usage(:_lava_coopmat_muladd_f16_16x16) === nothing
    @test M.intrinsic_usage(:llvm_pow_f32) === nothing

    # The classification reads INSTRUCTIONS, anchored, and not substrings
    # anywhere in the text. The first of these is the shape every Lava intrinsic
    # has and the name is the only thing in it that mentions loading.
    ext = """
        declare i32 @_lava_coopmat_load_f16_16x16_a(i64, i32) #0
        define i32 @entry(i64 %p, i32 %o) #0 {
            %r = call i32 @_lava_coopmat_load_f16_16x16_a(i64 %p, i32 %o)
            ret i32 %r
        }
        """
    @test M.llvmcallee(ext) === :_lava_coopmat_load_f16_16x16_a
    @test M.llvmcallusage(ext) === M.READ
    @test M.llvmcallusage("""
        define i32 @entry(ptr %p) {
            %r = load i32, ptr %p
            ret i32 %r
        }
        """) === M.READ
    @test M.llvmcallusage("""
        define void @entry(ptr %p, i32 %v) {
            store i32 %v, ptr %p
            ret void
        }
        """) === M.WRITE
    @test M.llvmcallusage("""
        define i32 @entry(ptr %p, i32 %v) {
            %r = atomicrmw add ptr %p, i32 %v acq_rel
            ret i32 %r
        }
        """) === M.ATOMIC
    # An intrinsic nobody declared: `nothing`, and the walk refuses on it rather
    # than widening to read+write, because read+write is indistinguishable from
    # a proof.
    mystery = """
        declare i32 @_lava_mystery(i64) #0
        define i32 @entry(i64 %p) #0 {
            %r = call i32 @_lava_mystery(i64 %p)
            ret i32 %r
        }
        """
    @test M.llvmcallee(mystery) === :_lava_mystery
    @test M.llvmcallusage(mystery) === nothing

    # And the whole of it on the kernel it matters for: `gemm_cm2!` takes
    # `C, @Const(A), @Const(B)`, so the walk must agree with the `@Const` the
    # author already wrote. It did not -- all three came out read+write, and
    # every pass sharing a weight matrix with another got a barrier for it.
    dev = M.Device(TESTBACKEND)
    n = 64
    C = M.devicearray(dev, zeros(Float16, n, n))
    A = M.devicearray(dev, ones(Float16, n, n))
    B = M.devicearray(dev, ones(Float16, n, n))
    t = M.kerneltouches(dev, M.gemm_cm2!,
                        (C, A, B, Int32(n), Int32(n), Int32(n),
                         Val(32), Val(32), Val(16), Val(false), Val(0)),
                        (32 * 2, 2), (32, 1))
    @test t[1] === M.WRITE          # C
    @test t[2] === M.READ           # A, @Const
    @test t[3] === M.READ           # B, @Const
    # The extents and the `Val`s are not accesses, which is the property the
    # scalar cases above pin and which a coopmat kernel must not break.
    @test all(u -> u === M.NOTOUCH, t[4:end])
end
