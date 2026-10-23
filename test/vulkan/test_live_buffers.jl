using Test, Mantle

# Finalizers free buffers on any thread while the owning thread allocates.
function check_live_buffers(ctx; n = 2000)
    @testset "freeing from another thread keeps dedicated-buffer accounting intact" begin
        bq = ctx.default_bq
        pol = Mantle.mempolicy(ctx)
        freed = Mantle.VkManagedBuffer[]
        ch = Channel{Mantle.VkManagedBuffer}(256)
        freer = Threads.@spawn :default for buf in ch
            Mantle.vk_free!(buf)
        end
        for _ in 1:n
            buf = Mantle.vk_alloc(bq, 256)
            push!(freed, buf)
            put!(ch, buf)
        end
        close(ch)
        wait(freer)
        Mantle.flush!(bq, ctx)
        Mantle.reclaim!(bq)
        # Every writer has finished, so the set is read without its lock.
        entries = collect(pol.live_buffers)
        @test all(i -> isassigned(entries, i), eachindex(entries))
        @test all(b -> (@atomic b.state) == Mantle.BUF_STATE_DEAD, freed)
        @test !any(b -> b in Set(entries), freed)
        @test Mantle.live_buffer_count(ctx) == length(entries)
    end
end
