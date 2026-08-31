"""
An `export` list has to describe the module it is in.

Both halves of this are failures the 2026-08-27 runtime move produced, and
neither of them broke anything at load — which is why they need a test rather
than a code review.

**Exporting a name you do not define.** Legal Julia, and silent. `Lava.jl` kept
`export dump_state, gpu_memory_usage` after both functions went to Mantle, and
the effect is worse than nothing: `using Lava` brings an undefined binding into
scope that SHADOWS Mantle's real export, so a file that had loaded both got
`UndefVarError: dump_state` while a working definition sat one module away. It
also hides deletions — `CoopMat`, `Scalar` and `matriximpl` were exported and
defined nowhere for months, and only turned up when someone checked which of 208
exports still resolved.

**Defining a public name and not exporting it.** 79 names moved here with the
runtime and their `export` lines stayed behind in `Lava.jl`. Again nothing
failed at load; it surfaced one `UndefVarError` at a time, at whatever line first
said `Rasterizer` or `LavaArray` or `trim_gpu_pool!` — 43 test files, and
`test_window.jl` had not run at all since the move.

The second half cannot be checked mechanically in general: "should this be
public" is a judgement. What CAN be checked is the specific claim this split
makes, which is that a name is exported by exactly one of the two packages —
never both, because that is ambiguous to anything loading them together, and
never neither if the pre-split package exported it.
"""

using Test
using Mantle
using Lava

@testset "no module exports a name it does not define" begin
    for m in (Lava, Mantle)
        stale = sort([String(n) for n in names(m) if !isdefined(m, n)])
        # Named, not counted: the failure has to say which name.
        @test (nameof(m), stale) == (nameof(m), String[])
    end
end

@testset "a name exported by both means the same thing in both" begin
    # `using Lava, Mantle` errors on a name the two resolve DIFFERENTLY; a name
    # both re-export from a common source is fine, and there are five of those —
    # `MatrixA`, `MatrixB`, `Accumulator`, `DeviceCaps`, `caps` are
    # `KernelInterface`'s, re-exported by each so a kernel library needs one
    # import. So the check is on identity, not on the name.
    #
    # The SHADER BUILTINS are the deliberate exception, and they arrived after
    # this testset did. `vertex_index`, `instance_index` and the `frag_coord`
    # family are DECLARED by Mantle (`function vertex_index end`, no methods) so
    # that a shader can write `Mantle.vertex_index()` and have the compiler that
    # is running decide what it means — Lava's version on Vulkan, Metal's on
    # Metal, through `@lava_device_override` / `@device_override`. Lava exports
    # its own implementation under the same name, on purpose. So here the two
    # SHOULD differ, and requiring identity would be requiring the portable
    # builtin not to exist.
    shared = [n for n in intersect(Set(names(Lava)), Set(names(Mantle)))
              if n !== :Lava && n !== :Mantle && !(n in Mantle.SHADER_BUILTINS)]
    conflicting = sort([String(n) for n in shared
                        if getproperty(Lava, n) !== getproperty(Mantle, n)])
    @test conflicting == String[]
    # The exemption is not a hole: each builtin Mantle declares must have a Lava
    # counterpart to be overridden WITH, or the override above binds nothing.
    @test all(n -> isdefined(Lava, n), Mantle.SHADER_BUILTINS)
    # …and the benign overlap is exactly the vocabulary, not something that drifted
    # into being defined twice.
    @test all(n -> parentmodule(getproperty(Mantle, n)) !== Mantle ||
                   getproperty(Lava, n) === getproperty(Mantle, n), shared)
    # `frag_coord` is why the builtins are exempt AND why they are still checked:
    # `graphics/api.jl` wrote the override's right-hand side unqualified, which
    # resolved to Mantle's own declaration and made the method call itself. This
    # asserts the direction the bridge has to run in.
    @test parentmodule(Mantle.frag_coord) === Mantle
    @test parentmodule(Lava.frag_coord) === Lava
end

@testset "every name a test imports from Lava is still Lava's" begin
    # `using Lava: X` for an X that moved binds nothing and shadows the real
    # export, which is the same trap as a stale `export` seen from the other end.
    # 22 test files were doing it after the move.
    function lavaimports(path)
        out = Symbol[]
        function walk(e)
            e isa Expr || return
            if (e.head === :using || e.head === :import) && length(e.args) == 1 &&
                    e.args[1] isa Expr && e.args[1].head === :(:)
                spec = e.args[1]
                if spec.args[1] isa Expr && spec.args[1].head === :. &&
                        spec.args[1].args == [:Lava]
                    for nm in spec.args[2:end]
                        nm isa Expr && nm.head === :. && push!(out, nm.args[1])
                    end
                end
            end
            foreach(walk, e.args)
        end
        walk(Meta.parseall(read(path, String)))
        return out
    end

    offenders = Tuple{String,Vector{String}}[]
    testdir = joinpath(pkgdir(Mantle), "test")
    for (root, _, files) in walkdir(testdir), f in files
        endswith(f, ".jl") || continue
        path = joinpath(root, f)
        missing = sort(unique([String(n) for n in lavaimports(path) if !isdefined(Lava, n)]))
        isempty(missing) || push!(offenders, (relpath(path, testdir), missing))
    end
    @test sort(offenders) == Tuple{String,Vector{String}}[]
end
