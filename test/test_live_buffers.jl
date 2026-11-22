"""
The buffers a device holds outside its pool are counted alike on every device.

`live_buffer_count` and `gpu_live_bytes` are what a consumer's leak check reads (Hikari's
and RayMakie's suites), so they answer on every device, from one ledger per device. A
release can run on another thread, a finalizer's included, while the owning thread
allocates: the ledger stays exact, and the count returns to where it was once every
array is freed.
"""

using Test, Mantle
import KernelAbstractions as KA
include(joinpath(@__DIR__, "testbackend.jl"))
const M = Mantle

@testset "a ledger stays exact when another thread frees" begin
    l = M.BufferLedger()
    ch = Channel{Int}(256)
    freer = Threads.@spawn :default for bytes in ch
        M.freed!(l, bytes)
    end
    for i in 1:2000
        M.allocated!(l, 16i)
        put!(ch, 16i)
    end
    close(ch)
    wait(freer)
    @test l.count[] == 0
    @test l.bytes[] == 0
end

@testset "arrays freed from another thread give their buffers back" begin
    dev = M.Device(TESTBACKEND)
    GC.gc(true)
    M.waitidle(dev)
    baseline = M.live_buffer_count(dev)
    usage = M.gpu_memory_usage(dev)
    @test usage.live_buffers == baseline
    @test usage.live_bytes == M.gpu_live_bytes(dev) >= usage.pool_bytes

    arrays = [KA.allocate(dev, UInt8, 256) for _ in 1:200]
    @test M.live_buffer_count(dev) >= baseline
    ch = Channel{Any}(64)
    # `finalize`: the release a collection would run, on this other thread.
    freer = Threads.@spawn :default for a in ch
        finalize(a)
    end
    foreach(a -> put!(ch, a), arrays)
    close(ch)
    wait(freer)
    empty!(arrays)
    M.waitidle(dev)
    M.drain!(M.batchqueue(dev))
    @test M.live_buffer_count(dev) == baseline
end

@testset "every device reports its memory heaps" begin
    dev = M.Device(TESTBACKEND)
    heaps = M.probe_device_memory_budget(dev)
    @test !isempty(heaps)
    @test any(h -> h.device_local, heaps)
    @test all(h -> h.size > 0 && h.usage >= 0 && 0 <= h.budget, heaps)
    # Trimming gives back what nothing holds, and never more than the pool had.
    held = M.gpu_memory_usage(dev).pool_bytes
    blocks, bytes = M.trim_gpu_pool!(dev)
    @test blocks >= 0 && 0 <= bytes <= held
    @test !M.retired(dev)
end
