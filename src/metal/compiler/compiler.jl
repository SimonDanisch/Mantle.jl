"""
Compiles Julia render stages (vertex, fragment, mesh) to Metal functions, with a
compiler job of Mantle's own parameters on Metal.jl's target.

A stage job is a `CompilerJob{MetalCompilerTarget,MetalStageParams}`: GPUCompiler's
Metal target does the codegen, the address spaces and the argument metadata exactly
as for a kernel, Metal.jl's `compile_to_metallib` packs the result, and the methods
here rewrite the kernel into the stage between the two (`stage_passes!`).
"""
module MetalStageCompiler

using GPUCompiler
using GPUCompiler: AbstractCompilerParams, CompilerConfig, CompilerJob, MetalCompilerTarget
using LLVM, LLVM.IR, LLVM.Build
import GPUToolbox
import Metal
using Metal: MTL
using ..MetalIntrinsics: VertexID, InstanceID, FragCoord, Varying, Texture2DPtr, SamplerPtr,
    MeshObject, VERTEX_ID_GLOBAL, INSTANCE_ID_GLOBAL, FRAG_COORD_GLOBAL, POINT_SIZE_GLOBAL,
    TEXTURE_PLACEHOLDER, SAMPLER_PLACEHOLDER

"""
Overlays that apply only inside render stages: the shader vocabulary (Lava's
`vertex_index`, `frag_coord`, `sample_texture_2d`, the mesh setters, ...) mapped
onto Metal's stage intrinsics. Stage jobs search it first, then the tables a
Metal.jl kernel uses; compute kernels never see it. Methods are added with
`Base.Experimental.@overlay` (`metal/graphics.jl`, `metal/mesh.jl`).
"""
Base.Experimental.@MethodTable(STAGE_METHODS)

"""
    MetalStageParams(stage)

The compiler parameters of a render stage job: which AIR program it becomes,
`:vertex`, `:fragment` or `:mesh`.
"""
struct MetalStageParams <: AbstractCompilerParams
    stage::Symbol
    function MetalStageParams(stage::Symbol)
        stage in (:vertex, :fragment, :mesh) ||
            throw(ArgumentError("a stage is :vertex, :fragment or :mesh, got :$stage"))
        return new(stage)
    end
end

const MetalStageJob = CompilerJob{MetalCompilerTarget,MetalStageParams}

include("stages.jl")    # vertex and fragment: outputs, inputs, builtins, metadata
include("mesh.jl")      # the mesh object's metadata
include("texture.jl")   # sampled textures and samplers as entry parameters

# ── GPUCompiler interface ────────────────────────────────────────────────────

# Metal.jl's device runtime (`report_oom`, `malloc`, ...). It has no
# `signal_exception`: see `stage_drop_exception_signal!`.
GPUCompiler.runtime_module(::MetalStageJob) = Metal

# The stage overlays first, then exactly the stack a Metal.jl kernel compiles
# against, so device code behaves the same in a stage as in a kernel.
GPUCompiler.method_tables(::MetalStageJob) =
    (STAGE_METHODS, Metal.method_table, GPUToolbox.Overlays.float64_overrides,
     GPUToolbox.Overlays.int128_overrides, GPUToolbox.Overlays.throw_overrides)

# Exception sites report through the `KernelState`, and a graphics stage has none:
# nothing binds one, and a buffer for it would take slot 0 and shift every buffer
# the caller does bind. GPUCompiler's default `kernel_state_type` (`Nothing`) keeps
# `kernel_state_to_reference!` from prepending the parameter in the first place.
GPUCompiler.uses_exception_sites(::MetalStageJob) = false

function GPUCompiler.finish_module!(@nospecialize(job::MetalStageJob), mod::LLVM.Module,
                                    entry::LLVM.Function)
    # Materialize `Metal.apple_family()` before GPUCompiler optimizes the linked
    # module, as Metal.jl's own jobs do: device code gates features on it (64-bit
    # atomic min/max), and an unresolved external global would reach the driver.
    gv = get(mod.globals, "apple_family", nothing)
    if gv !== nothing
        gv.initializer = ConstantInt(LLVM.Int32Type(),
                                     something(job.config.target.apple_family, 0))
        gv.linkage = LLVM.Linkage.Private
    end
    return invoke(GPUCompiler.finish_module!,
                  Tuple{CompilerJob{MetalCompilerTarget}, LLVM.Module, LLVM.Function},
                  job, mod, entry)
end

function GPUCompiler.finish_ir!(@nospecialize(job::MetalStageJob), mod::LLVM.Module,
                                entry::LLVM.Function)
    entry = invoke(GPUCompiler.finish_ir!,
                   Tuple{CompilerJob{MetalCompilerTarget}, LLVM.Module, LLVM.Function},
                   job, mod, entry)
    entry = stage_passes!(job, mod, entry)
    # Metal.jl's device functions emit intrinsics in their newest form; rewrite them
    # for the AIR and Metal versions this job targets.
    Metal.legalize_intrinsics!(job, mod)
    return entry
end

"""
    stage_job(dev, f, tt, stage, name) -> CompilerJob

The job compiling `f(::tt...)` into the `stage` function named `name` in the
metallib, for `dev`. Compile it with `Metal.compile_to_metallib`.

`tt` ends with the stage's output pointer for a vertex or fragment stage, and starts
with its `MeshPtr` for a mesh stage.
"""
function stage_job(dev::MTL.MTLDevice, f, tt::Type, stage::Symbol, name::String)
    # `kernel = true` even for a graphics stage: that flag is what turns on the
    # address-space and argument-metadata work a stage also needs.
    #
    # A graphics stage reports no exceptions, so it is compiled at debug level 0
    # whatever the session's `-g` is. Reporting means writing the `KernelState`
    # exception mailbox, and a stage HAS no kernel state (see `uses_exception_sites`
    # above). At level >= 1 a throw emits `report_exception`, which no runtime
    # defines for a stage, and the module then fails validation over a call no
    # shader author wrote, from a bounds check they did not know they had. What a
    # stage keeps is the `llvm.trap` GPUCompiler emits at every throw site, so an
    # out-of-bounds vertex fetch still ends the lane rather than reading whatever is
    # at that address. Only the reporting goes, and there was no channel to report
    # on.
    #
    # `always_inline`: textures and samplers are entry parameters
    # (`lower_texture_bindings!`), so every sampling call must be inlined into the
    # entry; the optimiser alone does not guarantee it.
    config = Metal.compiler_config(dev; kernel = true, name, always_inline = true,
                                   debug_level = 0)
    params = MetalStageParams(stage)
    return CompilerJob(methodinstance(typeof(f), tt), CompilerConfig(config; params))
end

end
