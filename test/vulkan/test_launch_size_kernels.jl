using Test, Mantle, KernelAbstractions, LinearAlgebra

# Lava compiles a kernel's workgroup size into its SPIR-V. A launch that left the size to
# the backend got one clamped to its range, so every short length was a kernel and a
# pipeline of its own: a scene with one more instance compiled its TLAS kernels again, and
# each `fill!` of a new short length did the same. A launch's size must not pick its kernel.

@kernel function launchsize_index!(out)
    i = @index(Global)
    @inbounds out[i] = Int32(i)
end

function check_launch_size_kernels(ctx)
    lengths = (3, 5, 9, 17, 33)
    @testset "fill! of short arrays links one kernel for every length" begin
        arrays = [Mantle.LavaArray{Int32}(undef, (n,); bq = ctx.default_bq) for n in lengths]
        fill!(arrays[1], Int32(7))
        linked = length(ctx.caches.linked)
        for a in arrays[2:end]
            fill!(a, Int32(7))
        end
        @test length(ctx.caches.linked) == linked
        @test all(a -> Array(a) == fill(Int32(7), length(a)), arrays)
    end
    @testset "an identity matrix of any shape has only its diagonal" begin
        # A lane past a wide matrix's diagonal still lands inside it; launched in a
        # full group, the kernel wrote row 1, column 4 of a 2x4 one.
        for dims in ((2, 4), (4, 2), (3, 3), (1, 5))
            x = Mantle.LavaArray{Int32, 2}(Int32(3) * I, dims)
            @test Array(x) == Matrix{Int32}(Int32(3) * I, dims...)
        end
    end
    @testset "a dynamic KA kernel over short ranges links one kernel" begin
        outs = [Mantle.LavaArray{Int32}(undef, (n + 1,); bq = ctx.default_bq) for n in lengths]
        foreach(o -> fill!(o, Int32(0)), outs)
        launchsize_index!(ctx)(outs[1]; ndrange = lengths[1])
        linked = length(ctx.caches.linked)
        for (o, n) in zip(outs[2:end], lengths[2:end])
            launchsize_index!(ctx)(o; ndrange = n)
        end
        KernelAbstractions.synchronize(ctx)
        @test length(ctx.caches.linked) == linked
        # The lanes past the range, which the shorter launches now run, write nothing.
        @test all(((o, n),) -> Array(o) == [Int32.(1:n); Int32(0)], zip(outs, lengths))
    end
end
