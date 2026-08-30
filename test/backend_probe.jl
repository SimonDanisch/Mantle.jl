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
function backend_loadable(name::AbstractString)
    id = Base.identify_package(name)
    id === nothing && return nothing
    try
        return Base.require(id)
    catch err
        @info "Mantle tests: $name is installed but not loadable here; skipping it" exception = err
        return nothing
    end
end
