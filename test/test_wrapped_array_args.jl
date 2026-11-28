"""
A recorded dispatch reads an array that reaches it inside another array.

An argument can be an `AbstractArray` that only wraps storage: Raycore's
`StaticMultiTypeSet` is a vector over one array per element type, and Hikari
hands its lights and materials to every kernel that way. A backend that declares
what a recorded plan reaches has to find the storage inside such a wrapper. On
Metal it did not: the walk held the wrapper whole, so a slot Metal.jl had
allocated (`resize!` when a second mesh added lights of a type the scene had)
was in no residency set and read zeros. The scene's constant-colour emitters
rendered black while every other material was right.
"""

using Test, Mantle, Adapt
using KernelAbstractions: @kernel, @index
include(joinpath(@__DIR__, "testbackend.jl"))
const M = Mantle

# A vector over one storage array, as Raycore's sets are.
struct WrappedSlot{T,A<:AbstractVector{T}} <: AbstractVector{T}
    data::A
end
Base.size(w::WrappedSlot) = size(w.data)
Base.@propagate_inbounds Base.getindex(w::WrappedSlot, i::Int) = w.data[i]
Adapt.adapt_structure(to, w::WrappedSlot) = WrappedSlot(Adapt.adapt(to, w.data))

@kernel function wrapped_read!(out, w)
    i = @index(Global)
    @inbounds out[i] = w.data[i]
end

@testset "a recorded dispatch reads storage wrapped in an array argument" begin
    dev = M.Device(TESTBACKEND)
    n = 4096
    data = Float32.(1:n)
    # Grown by `resize!`: new storage from the array library, not from Mantle.
    slot = M.devicearray(dev, data[1:n ÷ 2])
    resize!(slot, n)
    copyto!(slot, n ÷ 2 + 1, data, n ÷ 2 + 1, n - n ÷ 2)
    out = M.Buffer(dev, zeros(Float32, n))
    g = M.Graph(dev)
    M.dispatch!(g, wrapped_read!, (out, WrappedSlot(slot)), n; name = "read")
    plan = M.record!(M.Plan(g))
    for _ in 1:3
        M.run!(plan)
        M.waitfor!(plan)
        @test Array(M.storage(out)) == data
    end
    M.free!(plan)
end
