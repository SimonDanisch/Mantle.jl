# Whether a ray query through an accel is a call already (`tracescall`).
#
# Hikari outlines its shadow trace around an inline ray query, to give the query a
# register frame of its own, and inlines it where the traversal is a linked call
# already: there an outline is a second call boundary, 1.6x slower on an M1 Pro.
# Hikari decided that by reading `AdaptedAccel.scene`, Metal's in-band binding; the
# question is the accel's, so Mantle answers it.

using Test, Mantle

@testset "a ray query is a call where the accel binds its scene in band" begin
    inline_query = Mantle.AdaptedAccel(nothing, Int32[], Int32[], nothing, nothing, nothing, nothing)
    in_band = Mantle.AdaptedAccel(nothing, Int32[], Int32[], nothing, UInt8[], nothing, nothing)
    @test !Mantle.tracescall(inline_query)
    @test Mantle.tracescall(in_band)
    # a software BVH, or no accel at all
    @test !Mantle.tracescall(nothing)
end
