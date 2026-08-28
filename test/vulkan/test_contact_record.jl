using Test, Lava, Mantle
using Mantle: ContactRecord
using GeometryBasics: Vec3f

@testset "ContactRecord layout" begin

    @testset "isbitstype (precondition for GPU dispatch)" begin
        @test isbitstype(ContactRecord)
    end

    @testset "fields match spec (i, j, n_hat, p, depth)" begin
        @test fieldnames(ContactRecord) == (:i, :j, :n_hat, :p, :depth)
        @test fieldtype(ContactRecord, :i)     === UInt32
        @test fieldtype(ContactRecord, :j)     === UInt32
        @test fieldtype(ContactRecord, :n_hat) === Vec3f
        @test fieldtype(ContactRecord, :p)     === Vec3f
        @test fieldtype(ContactRecord, :depth) === Float32
    end

    @testset "size is tight (no internal padding)" begin
        # i(4) + j(4) + n_hat(12) + p(12) + depth(4) = 36 bytes
        @test sizeof(ContactRecord) == 36
    end

    @testset "constructor + field access roundtrip" begin
        c = ContactRecord(UInt32(7), UInt32(13),
                          Vec3f(1f0, 0f0, 0f0),
                          Vec3f(0.5f0, 0.5f0, 0f0),
                          0.1f0)
        @test c.i == UInt32(7)
        @test c.j == UInt32(13)
        @test c.n_hat == Vec3f(1f0, 0f0, 0f0)
        @test c.p == Vec3f(0.5f0, 0.5f0, 0f0)
        @test c.depth == 0.1f0
    end
end

