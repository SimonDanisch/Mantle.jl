"""
`@setup_workload` / `@compile_workload`: freeze every kernel a body reaches.

Shaped after PrecompileTools deliberately, because it is the same job done twice
over. PrecompileTools makes Julia keep the *native* code it inferred while
running a workload; this makes Lava keep the *SPIR-V* it compiled while running
the same one. A package that wants no first-call latency needs both, so they are
written to be used together:

    module SAM2Runner
        using Lava, PrecompileTools
        const KERNELS_VERSION = "1"

        @setup_workload begin
            model = load(...)
            @compile_workload KERNELS_VERSION begin
                run_sam2(model, image, point, label)
            end
        end
    end

`@compile_workload` sets `Lava.FROZEN_VERSION[]` to the version for the duration
of the block and turns recording on, so every kernel compiled inside it lands in
the frozen cache under that version. It also wraps the body in
PrecompileTools' `@compile_workload` when PrecompileTools is loaded, so the
Julia side is specialised by the same block that froze the kernels — one
workload, both caches.

The version is a value, not a literal, so it can live in a `const` next to the
kernels it describes and be bumped in one place.

Nothing here inspects the body. Whatever kernels it happens to launch are the
ones that get frozen, which is why the *runtime* API and the workload have to
agree: a code path the workload does not take is a kernel that is not cached,
and there is no way to notice that from here. Cover it with a test that asserts
`Lava.frozen_stats().misses == 0` over the paths that matter.
"""

"""
    @compile_workload version begin ... end

Run the body with the frozen kernel cache recording under `version`, *and* with
PrecompileTools tracking the Julia code it infers.

Every kernel compiled inside lands on disk as
`<module>_<kernel>_<digest>_v<version>.spirv` and is read back by any later
session asking for the same signature under the same version — no inference, no
hashing, no staleness check. Everything Julia inferred along the way is kept in
the package image by PrecompileTools.

Both, because either alone leaves most of the latency in place: with the kernels
frozen but Julia cold, SAM 2's encoder still spends ~70 s in inference; with
Julia warm but the kernels cold, it pays the SPIR-V compile instead.

`@setup_workload` is PrecompileTools', re-exported — this only adds the kernel
half, and there is no reason to have a second name for the setup block.
"""
macro compile_workload(version, ex)
    return esc(quote
        $PrecompileTools.@compile_workload begin
            $with_frozen_recording($version) do
                $ex
            end
        end
    end)
end

"""
    with_frozen_recording(f, version)

Run `f` with `FROZEN_VERSION` set to `version` and recording enabled, restoring
both afterwards. The building block `@compile_workload` expands to, and usable
directly when a macro is awkward — a test that wants to freeze one call, say.
"""
function with_frozen_recording(f, version)
    oldv, oldr = FROZEN_VERSION[], FROZEN_RECORDING[]
    FROZEN_VERSION[] = string(version)
    FROZEN_RECORDING[] = true
    try
        return f()
    finally
        FROZEN_VERSION[] = oldv
        FROZEN_RECORDING[] = oldr
    end
end

"""
    use_frozen_kernels(version)

Read frozen entries written under `version`, without recording new ones.

What a *using* package calls at load time — `__init__` is the natural place. The
version has to match the one the workload froze under, which is why it belongs
in a `const` both refer to.
"""
function use_frozen_kernels(version)
    FROZEN_VERSION[] = string(version)
    FROZEN_RECORDING[] = false
    return nothing
end
