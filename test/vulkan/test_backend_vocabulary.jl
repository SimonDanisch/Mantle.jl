# The line between core and a backend, held by a test rather than by memory.
#
# Three things went wrong in the submission refactor and each is checked here:
# the Vulkan backend grew its own `record!`, `emitplan!` and `execute!` (a backend
# implementing a sequence); an `import Mantle:` line was missed, so the backend
# defined a NEW local function instead of extending core's; and a core name the
# backend calls unqualified (`region_bda`, `recordable`) was not imported at all
# and threw the first time that path ran. None of these are visible to a green
# suite that only exercises one device.
using Test, Mantle

# Every loaded backend extension, by module.
backends = filter(!isnothing, [Base.get_extension(Mantle, :MantleVulkanExt),
                               Base.get_extension(Mantle, :MantleMetalExt)])

# Source of a backend: the extension file and the src/<name>/ tree it includes.
function backendsources(ext::Module)
    root = dirname(dirname(pathof(Mantle)))
    name = ext === Base.get_extension(Mantle, :MantleVulkanExt) ? "vulkan" : "metal"
    files = [joinpath(root, "ext", string(nameof(ext), ".jl"))]
    for (d, _, fs) in walkdir(joinpath(root, "src", name)), f in fs
        endswith(f, ".jl") && push!(files, joinpath(d, f))
    end
    return files
end

# Code only: docstrings and comments stripped, crudely but enough for names.
function codeonly(src::AbstractString)
    s = replace(src, r"\"\"\"[\s\S]*?\"\"\"" => "")
    return join((replace(l, r"#.*" => "") for l in split(s, '\n')), '\n')
end

corenames = Set(n for n in names(Mantle; all = true)
                if isdefined(Mantle, n) && !startswith(String(n), "#") &&
                   (getfield(Mantle, n) isa Function || getfield(Mantle, n) isa Type))

for ext in backends
    @testset "$(nameof(ext)) speaks the vocabulary" begin
        # 1. A backend adds methods only to the functions in the vocabulary.
        extended = Set{Symbol}()
        for n in corenames
            f = getfield(Mantle, n)
            any(m -> m.module === ext, methods(f)) && push!(extended, n)
        end
        extra = sort(collect(setdiff(extended, Set(Mantle.BACKEND_VOCABULARY))))
        @test isempty(extra)
        isempty(extra) || @info "$(nameof(ext)) extends core names outside BACKEND_VOCABULARY" extra

        # 2. No name is bound in both modules to different functions: that is
        #    a backend function core cannot reach, born of a missing import.
        shadows = Symbol[]
        for n in names(ext; all = true)
            startswith(String(n), "#") && continue
            n in (:eval, :include) && continue
            (isdefined(ext, n) && n in corenames) || continue
            a = getfield(ext, n)
            (a isa Function || a isa Type) && a !== getfield(Mantle, n) && push!(shadows, n)
        end
        @test isempty(shadows)
        isempty(shadows) || @info "$(nameof(ext)) shadows core names" shadows

        # 3. Every core function the backend CALLS unqualified resolves to
        #    core's binding — the `region_bda` class, which only a call reveals.
        called = Set{Symbol}()
        for f in backendsources(ext)
            src = codeonly(read(f, String))
            for n in corenames
                s = String(n)
                (length(s) >= 4 && isletter(s[1])) || continue
                occursin(Regex("(?<![\\w.])" * s * "\\("), src) && push!(called, n)
            end
        end
        unresolved = sort([n for n in called
                           if !(isdefined(ext, n) && getfield(ext, n) === getfield(Mantle, n))])
        # Names a backend legitimately has its own local function for shadow the
        # core one on purpose; those are the `shadows` above and already caught.
        filter!(n -> !isdefined(ext, n), unresolved)
        @test isempty(unresolved)
        isempty(unresolved) || @info "$(nameof(ext)) calls core names it never imported" unresolved
    end
end
