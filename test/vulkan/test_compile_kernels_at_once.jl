using Test, Mantle, KernelAbstractions, Lava

# A plan's kernels compile and link at once, on Julia's threads, before its dispatch
# loop (`compilekernels!`): a scene with a dozen material kernels waits for the
# slowest instead of for all of them in turn, and the loop finds each compiled and
# its pipeline made. Lava counts one lookup per compile request, a hit (this session
# or the object cache) or a miss (compiled), so the counter says which of the two
# compiled the kernels.

for k in 1:4
    @eval KernelAbstractions.@kernel function $(Symbol(:atonce_scale, k, :!))(y, @Const(x))
        i = @index(Global, Linear)
        @inbounds y[i] = x[i] * $(Float32(k))
    end
end

function check_compile_kernels_at_once(dev)
    M = Mantle
    n = 64
    kernels = (atonce_scale1!, atonce_scale2!, atonce_scale3!, atonce_scale4!)
    x = M.Buffer(dev, Float32.(1:n))
    ys = [M.Buffer(dev, zeros(Float32, n)) for _ in kernels]
    g = M.Graph(dev)
    for (kernel, y) in zip(kernels, ys)
        M.dispatch!(g, kernel, (y, x), n)
    end
    @testset "a plan's kernels compile before its dispatch loop" begin
        c = M.Compile(g)
        M.compile!(c, (M.Dag(), M.Schedule(), M.Liveness(), M.Place(), M.Aliasing(), M.Barriers()))
        Lava.reset_compile_stats!()
        M.compilekernels!(c)
        ahead = Lava.compile_stats()
        @test ahead.hits + ahead.misses == length(kernels)
        linked = length(dev.caches.linked)
        # The phase looks each kernel up twice, in its own `compilekernels!` and in its
        # dispatch loop, and finds it compiled both times.
        M.run!(M.Pipelines(), c)
        @test Lava.compile_stats() == (; hits = ahead.hits + 2 * length(kernels), misses = ahead.misses)
        # And the pipelines: made ahead too, unless the device validates, which makes
        # them in the loop so each error names its pipeline.
        @test M.validating(dev) || length(dev.caches.linked) == linked
    end
    @testset "and run as they would compiled one at a time" begin
        pl = M.Plan(g)
        M.record!(pl)
        M.run!(pl)
        M.waitidle(dev)
        @test all(k -> Array(M.storage(ys[k])) == Float32.(1:n) .* k, eachindex(ys))
        M.free!(pl)
    end
end
