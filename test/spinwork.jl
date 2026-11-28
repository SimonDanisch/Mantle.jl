# GPU work of a chosen length, and a task that counts while another waits for it: how
# `test_hostwait.jl` and the window tests tell a wait that yields its thread from one
# that holds it. Included after `testbackend.jl` and `const M = Mantle`.

using KernelAbstractions: @kernel, @index

# One lane, `n` dependent steps: as long on a wide GPU as on a narrow one.
@kernel function hw_spin!(out, n)
    i = @index(Global)
    x = UInt32(i)
    for _ in 1:n
        x = x ⊻ (x << UInt32(13))
        x = x ⊻ (x >> UInt32(17))
        x = x ⊻ (x << UInt32(5))
    end
    @inbounds out[i] = x
end

function spinplan(dev, n)
    out = M.Buffer(dev, zeros(UInt32, 1))
    g = M.Graph(dev)
    M.dispatch!(g, hw_spin!, (out, n), 1; name = "spin")
    return M.record!(M.Plan(g))
end

"""
    spinsteps(dev, seconds) -> Int

How many `hw_spin!` steps take about `seconds` on `dev` under load. Measured, because one
lane's speed differs between devices, and measured under load because it differs as much
between clock states: on an RTX 4000, 2^16 steps took 11 ms as a lone run and a tenth of
that back to back, once the GPU had clocked up. A loop of frames runs back to back, so a
length calibrated idle came out ten times too short there and nothing waited. Only the
fastest of three runs counts, so a run's fixed cost (the first timed run after compiling
took 30 ms) does not.
"""
function spinsteps(dev, seconds)
    n = 1 << 16
    while true
        pl = spinplan(dev, n)
        M.run!(pl); M.waitfor!(pl)              # compiled
        t0 = time()
        while time() - t0 < 0.3                 # clocked up
            M.run!(pl); M.waitfor!(pl)
        end
        t = minimum(@elapsed((M.run!(pl); M.waitfor!(pl))) for _ in 1:3)
        M.free!(pl)
        t >= seconds / 4 && return ceil(Int, n * seconds / t)
        n *= 4
    end
end

# Counts on the calling thread (`@async` is sticky) until `stop`.
function ticker(stop::Ref{Bool})
    ticks = Ref(0)
    task = @async while !stop[]
        ticks[] += 1
        sleep(0.001)
    end
    return ticks, task
end

# How far the ticker got while `f()` ran, and how long that was.
function progress_during(f::F, ticks) where {F}
    t0, k0 = time(), ticks[]
    f()
    return ticks[] - k0, time() - t0
end
