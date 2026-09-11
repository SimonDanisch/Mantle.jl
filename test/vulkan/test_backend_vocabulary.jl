# The line between core and a backend, held by a test rather than by memory.
#
# What went wrong in the submission refactor: the Vulkan backend grew its own
# `record!`, `emitplan!` and `execute!` — a backend implementing a sequence.
# That is not visible to a green suite that only exercises one device.
#
# The backend used to be an extension module, and the boundary was the module.
# It is `@static include`d into `Mantle` now, so the boundary is the SOURCE TREE:
# a method defined under `src/vulkan/` or `src/metal/` is a backend's, whatever
# module it lands in. That is the stricter reading anyway, and it is the one that
# survives the merge — a backend name colliding with a core one no longer makes a
# separate binding, it silently adds a method to core's function, and a method on
# a core name outside the vocabulary is exactly what this asks about.
using Test, Mantle

# Both trees, unconditionally: only the one this build included can produce hits.
const SRC = joinpath(dirname(dirname(pathof(Mantle))), "src")
const BACKENDDIRS = [joinpath(SRC, "vulkan"), joinpath(SRC, "metal")]

isbackendfile(file) = any(d -> startswith(String(file), d), BACKENDDIRS)

# THREE buckets, not two. A method from a third package is neither core's nor the
# backend's and says nothing about the line between them: `LavaArray` inherits
# GPUArrays' `AnyGPUArray(::UniformScaling, dims)` constructors, and reading those
# as "core also defines it" reported the backend's own array type as a violation.
iscorefile(file) = startswith(String(file), SRC) && !isbackendfile(file)

# "A core name" cannot be "a name visible in `Mantle`" any more — the backend's
# own `vkformat` and `AllocFailure` are that too now. It is a name core DEFINES:
# one with at least one method outside the backend trees. A vocabulary name that
# only the backend answers has no core method and is not one of these, which is
# right — `BACKEND_VOCABULARY` is the list of names core declares for exactly
# that, and it is checked separately by 0.2 in `test_mantle_owns_it.jl`.
names_with = Dict{Symbol, Tuple{Bool, Bool}}()   # name => (has core method, has backend method)
for n in names(Mantle; all = true)
    startswith(String(n), "#") && continue
    isdefined(Mantle, n) || continue
    v = getglobal(Mantle, n)
    (v isa Function || v isa Type) || continue
    ms = collect(methods(v))
    isempty(ms) && continue
    names_with[n] = (any(m -> iscorefile(m.file), ms),
                     any(m -> isbackendfile(m.file), ms))
end

@testset "the backend speaks the vocabulary" begin
    # A backend adds methods to a core function only where the vocabulary says so.
    extended = Set(n for (n, (hascore, hasbackend)) in names_with if hascore && hasbackend)

    # Nothing under src/vulkan means this build has no backend tree to police,
    # and a vacuous pass would be indistinguishable from a green one.
    @test any(last, values(names_with))

    extra = sort(collect(setdiff(extended, Set(Mantle.BACKEND_VOCABULARY))))
    @test isempty(extra)
    isempty(extra) || @info "the backend extends core names outside BACKEND_VOCABULARY" extra
end
