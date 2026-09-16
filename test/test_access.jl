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
    # `accessof` on a signature nothing can be inferred for used to answer
    # read+write for every argument. Safe for the barrier phase, and
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
