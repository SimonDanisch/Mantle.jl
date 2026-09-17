# What the access walk decides from a TYPE, with no device in the room.
#
# The walk itself needs the interpreter a backend compiles through — a device
# array's `getindex` lives on that method table and nowhere else — so the
# kernels, and what right looks like for them, are in `vulkan/test_access.jl`.
# What is here is the part that has no backend in it: whether a type can reach
# memory at all, and what a `Touch` means as a usage.
using Test, Mantle
const M = Mantle

@testset "inferred access: the unknown is refused, not guessed at" begin
    # `accessof` on a signature nothing can be inferred for must not answer
    # read+write for every argument: safe for the barrier phase, and
    # indistinguishable from a proof: a kernel that stopped being analysable kept
    # compiling and paid a barrier per pass for ever with nothing to say so.
    err = try
        M.accessof(nothing, sin, (Any,))
        nothing
    catch e
        e
    end
    @test err isa M.UnanalysableAccess
    msg = sprint(showerror, err)
    @test occursin("cannot say what", msg)
    @test occursin("will not guess", msg)
    # `OPAQUE` is still the usage a read+write argument has; what changed is that
    # nothing produces it by giving up.
    @test M.usagetype(M.OPAQUE, M.BufferKind()) === Storage{BufferKind, ReadWrite}

    # And the pieces the walk decides with.
    @test M.carries(Float32) === false
    @test M.carries(Tuple{Int,Int}) === false
    @test M.carries(Ptr{Float32}) === true
    @test M.carries(Any) === true

    # A `Touch` with an ordinary write is ordered; one whose writes are all
    # atomic is not.
    @test M.usagetype(M.WRITE, M.BufferKind()) === Storage{BufferKind, WriteOnly}
    @test M.usagetype(M.ATOMIC, M.BufferKind()) ===
          Unordered{Storage{BufferKind, ReadWrite}}
    @test M.usagetype(M.READ, M.BufferKind()) === Storage{BufferKind, ReadOnly}
end

# A device address laundered through an integer is still an address.
#
# `carries` asks what a TYPE can hold and answers nothing for `UInt64`, which is
# right for a length and wrong for an address. The walk had one hop of an
# exception for it -- taint survived a statement with a `Ptr`-typed operand -- and
# Lava's cooperative-matrix intrinsics need two, because they take the address as
# a `UInt64` with the element offset already added in:
#
#     %18 = getfield(C, :ptr)::Ptr{Float16}   # tainted by C
#     %19 = bitcast(UInt64, %18)              # kept: an operand is a `Ptr`
#     %31 = add_int(%19, %28)                 # DROPPED: both operands are UInt64
#           llvmcall(@_lava_coopmat_store_..., %31, ...)
#
# `llvmcalltaint!` then found no tainted operand and recorded nothing, so every
# argument of `coopmat_gemm_kernel_4!` came back untouched -- INCLUDING its
# destination -- and `Place` saw its output transient with no interval at all.
# `gemm_cm2!` reads correctly because its intrinsics take a `Ptr` directly, which
# is why the failure was confined to the block kernels.
#
# Stated here on `unsafe_load`/`unsafe_store!` rather than on a kernel, because
# the laundering is the bug and neither the intrinsics nor a device are part of
# it: written this way it reproduces on the host walk, with no backend loaded.
function storelaundered(out::Ptr{Float32}, src::Ptr{Float32}, n::Int)
    a = Base.bitcast(UInt64, out)
    b = Base.bitcast(UInt64, src)
    for i in 1:n
        v = unsafe_load(Base.bitcast(Ptr{Float32}, b + UInt64(4 * (i - 1))))
        unsafe_store!(Base.bitcast(Ptr{Float32}, a + UInt64(4 * (i - 1))), v)
    end
    return
end

@testset "an address laundered through an integer" begin
    t = M.accessof(nothing, storelaundered, (Ptr{Float32}, Ptr{Float32}, Int))
    @test t[2] === M.WRITE
    @test t[3] === M.READ
    # And the reason the type test exists in the first place: `n` reaches the
    # same arithmetic and is NOT an address, so following every integer is not
    # the fix. Before `carries` was consulted at all, an argument whose length a
    # kernel read came back read+write.
    @test t[4] === M.NOTOUCH
end
