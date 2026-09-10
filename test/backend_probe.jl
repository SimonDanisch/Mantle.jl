# Is a backend actually usable here, as opposed to merely installed?
#
# Both GPU backends are optional and for opposite reasons: `Vulkan` has no
# loader on a Mac, `Metal` has no device in a CI container or a VM. In either
# case the suite must SKIP rather than error.
#
# It used to error. A bare `import Vulkan` at file scope took the whole run down
# on a machine with no `libvulkan`, before it reached the Metal tests or the
# includes below it — the same failure mode the `HAVE_BENCH` note in
# `runtests.jl` describes, where "the suite failed" hides "N files never ran".
#
# Loadability is the honest question, not the OS: both package names resolve on
# any platform, so `Sys.isapple()` would answer a different question than the
# one being asked. Returns the module, or `nothing` if it cannot be loaded.
#
# ── Asked ONCE, and quietly ──────────────────────────────────────────────────
#
# Julia does not cache a FAILED `require`, so every call re-ran the whole
# attempt: on a Mac the four call sites in this suite each triggered a fresh
# precompilation of Vulkan, each ending in the same `libvulkan.dylib` error, and
# each printing it live — Pkg announces "output will be shown live" for a package
# it was explicitly asked for, and the `exception =` on the log line printed a
# sixty-line backtrace on top. Four times, before a single test ran. A reader
# looking for a Metal failure had four Vulkan stack traces to scroll past first,
# which is how the noise stops being harmless.
#
# So: memoized, and the attempt's own output goes to a temp file. What survives
# is one line naming the package and the reason. The backtrace is not
# information here — the reason a Vulkan-less machine cannot load Vulkan is the
# first line of it.
#
# Guarded, because this file is included TWICE: `runtests.jl` needs it before
# anything else, and `test_host.jl` includes it again so that it also runs
# standalone. A second `const BACKEND_PROBED = Dict()` binds a fresh empty one and
# the memo above it is lost — which is exactly what happened, and why the Vulkan
# notice appeared twice in a suite that asks four times.
if !@isdefined(BACKEND_PROBED)
    const BACKEND_PROBED = Dict{String, Any}()
end

"""
The one line of a load failure worth printing.

A `PkgPrecompileError` opens with "The following 1 direct dependency failed to
precompile:" and says what actually went wrong sixty lines down, behind the failed
package's own `ERROR:`. That line is the answer — "Failed to retrieve a valid
Vulkan library called libvulkan.dylib" — so it is the one taken, and the first
line is the fallback for an error shaped differently.
"""
function rootcause(err)
    # The colours come off first: Pkg writes its `ERROR:` with ANSI escapes around
    # it, so a plain `startswith` on the raw line never matches and the header wins.
    plain = replace(sprint(showerror, err), r"\e\[[0-9;]*m" => "")
    lines = split(plain, '\n')
    for l in lines
        i = findfirst("ERROR: ", l)
        i === nothing || return String(strip(l[(last(i) + 1):end]))
    end
    return String(strip(first(filter(!isempty, strip.(lines)); init = "")))
end

function backend_loadable(name::AbstractString)
    haskey(BACKEND_PROBED, name) && return BACKEND_PROBED[name]
    id = Base.identify_package(name)
    id === nothing && return (BACKEND_PROBED[name] = nothing)
    why = Ref{Any}(nothing)
    mod = mktemp() do _, io
        try
            # Both streams: the loader's error goes to stderr and Pkg's live
            # precompile log to stdout, and the point is that neither reaches a
            # reader who asked for test results.
            redirect_stdout(io) do
                redirect_stderr(io) do
                    Base.require(id)
                end
            end
        catch err
            why[] = err
            nothing
        end
    end
    if mod === nothing
        @info "Mantle tests: $name is installed but not loadable here; skipping it. " *
              rootcause(why[])
    end
    return (BACKEND_PROBED[name] = mod)
end
