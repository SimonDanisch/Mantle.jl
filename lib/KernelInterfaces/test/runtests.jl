using KernelInterfaces, Test

# This card's actual table: four entries share 16x16x16 and differ only in type,
# which is why `MatrixShape` carries `ab`/`acc` and why matching on extents alone
# is wrong.
const THISCARD = [
    MatrixShape(Float16, Float32, 16, 16, 16, SubgroupScope()),
    MatrixShape(Float16, Float16, 16, 16, 16, SubgroupScope()),
    MatrixShape(UInt8,   Int32,   16, 16, 16, SubgroupScope()),
    MatrixShape(Int8,    Int32,   16, 16, 16, SubgroupScope()),
]

@testset "equality and hashing are by value" begin
    a = MatrixShape(Float16, Float32, 16, 16, 16, SubgroupScope())
    b = MatrixShape(Float16, Float32, 16, 16, 16, SubgroupScope())
    @test a == b
    @test hash(a) == hash(b)
    @test length(Set([a, b])) == 1
    @test a != MatrixShape(Float16, Float16, 16, 16, 16, SubgroupScope())   # acc differs
    @test a != MatrixShape(Float16, Float32, 16, 16, 16, WorkgroupScope())  # scope differs
end

@testset "supports matches on type, not just extents" begin
    @test supports(THISCARD, MatrixShape(Float16, Float32, 16, 16, 16, SubgroupScope()))
    # Same extents, a type combination the card does not list. Matching on
    # (M,N,K) alone would say yes and emit instructions the device cannot run.
    @test !supports(THISCARD, MatrixShape(Float32, Float32, 16, 16, 16, SubgroupScope()))
    @test !supports(MatrixShape[], MatrixShape(Float16, Float32, 16, 16, 16, SubgroupScope()))
end

@testset "bestshape" begin
    @test bestshape(THISCARD, Float16, Float32).M == 16
    @test bestshape(THISCARD, Int8, Int32).acc === Int32
    # No matrix hardware, or no shape for this type pair: `nothing`, never a
    # fabricated tile — the value is used as a divisor.
    @test bestshape(MatrixShape[], Float16, Float32) === nothing
    @test bestshape(THISCARD, Float64, Float64) === nothing
    @test bestshape(THISCARD, Float16, Float32; scope = WorkgroupScope()) === nothing
end

@testset "bestshape prefers square, then large" begin
    mixed = [MatrixShape(Float16, Float32, 16, 8, 32, SubgroupScope()),   # skewed
             MatrixShape(Float16, Float32,  8,  8,  8, SubgroupScope()),  # square, small
             MatrixShape(Float16, Float32, 16, 16, 16, SubgroupScope())]  # square, large
    @test bestshape(mixed, Float16, Float32) == mixed[3]
    # With no square option it still returns the least skewed rather than nothing.
    skewed = [MatrixShape(Float16, Float32, 16, 8, 32, SubgroupScope()),
              MatrixShape(Float16, Float32, 16, 8, 16, SubgroupScope())]
    @test bestshape(skewed, Float16, Float32) == skewed[2]
end
