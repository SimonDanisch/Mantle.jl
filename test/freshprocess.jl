# Part of a per-backend test in a process of its own: for what changes the process
# itself — a device reset, the exit hook, a validation layer that a process has to
# be started with — and so must not change the suite's, or must not take the
# suite down with it when it regresses into a crash.
#
# Included after `testbackend.jl`, by the files that need it.

"""
    infreshprocess(code; before = "", debug = nothing, timeout = 900) -> (status, output)

Run `code` in a new Julia process on this project, after
`using Mantle, KernelAbstractions, Test`, `const KA = KernelAbstractions` and
`const BACKEND`, the backend under test: the same one, found by its place in
`Mantle.eachbackend()`. `before` runs ahead of the `using`, for what has to
precede Mantle's own `__init__` — an `atexit` hook that must run after Mantle's.
`debug` is a `DebugConfig` the process has to be able to build a device with, so
it is started with `Mantle.debugenv`'s environment.

`status` is the exit code, or `nothing` when the process did not finish within
`timeout` seconds and was killed; `output` is what it printed. A failed `@test`
inside a `@testset` in `code` exits with 1.
"""
function infreshprocess(code::AbstractString; before::AbstractString = "", debug = nothing,
                        timeout = 900)
    i = findfirst(==(TESTBACKEND), Mantle.eachbackend())
    i === nothing && error("the backend under test is not among `Mantle.eachbackend()`")
    script = string(before, "\n",
                    "using Mantle, KernelAbstractions, Test\n",
                    "const KA = KernelAbstractions\n",
                    "const BACKEND = Mantle.eachbackend()[", i, "]\n",
                    code)
    env = debug === nothing ? Pair{String,String}[] : Mantle.debugenv(TESTBACKEND, debug)
    cmd = addenv(`$(Base.julia_cmd()) --project=$(Base.active_project()) -e $script`, env...)
    out = IOBuffer()
    p = run(pipeline(ignorestatus(cmd); stdout = out, stderr = out); wait = false)
    t0 = time()
    while process_running(p) && time() - t0 < timeout
        sleep(0.5)
    end
    if process_running(p)
        kill(p)
        wait(p)
        return (nothing, String(take!(out)))
    end
    return (p.exitcode, String(take!(out)))
end

"""
    reportfresh(status, output)

Print what a fresh process said when it did not exit cleanly, so a failure
inside it is read in the suite's log.
"""
function reportfresh(status, output)
    status == 0 && return nothing
    println(status === nothing ? "── fresh process TIMED OUT ──" :
                                 "── fresh process exited with $status ──")
    println(output)
    return nothing
end
