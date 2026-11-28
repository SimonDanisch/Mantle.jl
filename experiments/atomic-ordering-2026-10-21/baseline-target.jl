# GPUCompiler target definition for Lava.jl
#
# Uses GPUCompiler.SPIRVCompilerTarget with backend=:llvm to get the
# spirv64-unknown-unknown-unknown triple. We only use GPUCompiler for
# LLVM IR generation (compile(:llvm, ...)), never for SPIR-V codegen —
# that's handled by our custom emitter.

# ── Compiler parameters ──

struct LavaCompilerParams <: GPUCompiler.AbstractCompilerParams
    workgroup_size::NTuple{3,Int}
    enable_ray_query::Bool
    # What the target device lets the module declare. In the params so it is part
    # of the job GPUCompiler caches on and of every frozen key derived from it:
    # two devices with different feature sets compile the same kernel twice, as
    # they must.
    features::TargetFeatures
end
LavaCompilerParams() = LavaCompilerParams((64, 1, 1), false, TargetFeatures())
LavaCompilerParams(workgroup_size::NTuple{3,Int}) =
    LavaCompilerParams(workgroup_size, false, TargetFeatures())

const LavaCompilerConfig = GPUCompiler.CompilerConfig{GPUCompiler.SPIRVCompilerTarget, LavaCompilerParams}
const LavaCompilerJob = GPUCompiler.CompilerJob{GPUCompiler.SPIRVCompilerTarget, LavaCompilerParams}

# ── Device runtime stubs ──
#
# GPUCompiler's lower_throw! pass converts `throw(...)` into calls to these
# runtime functions, then inserts `llvm.trap` + `unreachable` after them, which
# `finish_ir!` turns into a return: the invocation stops there.
#
# `signal_exception` is the one that does something. It raises a word in device
# memory, the device's exception flag, whose address a compute stage reads from
# its push constants (`wrap_entry_for_vulkan!`); the host finds the word raised
# when it next waits for the device, and throws. That is Metal.jl's contract for a
# device-side exception, and CUDA's: the invocation stops, the device records it,
# the next synchronize reports it. The rest stay no-ops: there is no mailbox to
# record a name or a frame in.

module LavaRuntime
    """
    Raise the device's exception flag: one store, of 1, to the word whose address
    a compute stage reads from its push constants. A call to an intrinsic the
    entry wrapper lowers (`wrap_entry_for_vulkan!`), and not that store spelled
    here, so the throwing block stays straight-line: a branch around the store
    would hide the throw from `fix_barrier_skipping_paths!`, which recognises an
    error path by its shape. Where nothing collects the flag — a graphics or
    ray-tracing stage — the call is deleted and a throw only stops the invocation.
    """
    signal_exception() = Base.llvmcall(("""
        declare void @_lava_signal_exception()
        define void @entry() #0 {
            call void @_lava_signal_exception()
            ret void
        }
        attributes #0 = { alwaysinline }
        """, "entry"), Cvoid, Tuple{})
    report_exception(ex) = return
    report_oom(sz) = return
    report_exception_name(ex) = return
    report_exception_frame(idx, func, file, line) = return

    # GPUCompiler's runtime expects malloc/free for dynamic allocation.
    # On GPU these are no-ops that return null — dynamic allocation is not
    # supported in Vulkan compute shaders. The compiler should eliminate
    # these via dead code elimination in practice.
    malloc(sz) = reinterpret(Ptr{Nothing}, UInt(0))
    free(ptr) = return
end

# ── Method table for device overrides ──
#
# @lava_device_override puts methods on this table, which GPUCompiler uses
# instead of the regular method table during compilation. This lets us
# replace CPU functions with GPU-safe versions (e.g., intrinsics, math).

Base.Experimental.@MethodTable lava_method_table

macro lava_device_override(ex)
    esc(quote
        Base.Experimental.@overlay($lava_method_table, $ex)
    end)
end

# ── GPUCompiler hooks ──

GPUCompiler.runtime_module(::LavaCompilerJob) = LavaRuntime

GPUCompiler.method_table(::LavaCompilerJob) = lava_method_table

"""
    kernelinterpreter(f, tt) -> AbstractInterpreter

The interpreter `f(tt...)` will be inferred through when it is compiled, so that
anything reasoning about a kernel ahead of time reasons about the DEVICE code.

Without it, `code_typed` on a kernel is the host's: `LavaDeviceArray`'s indexing
lives on `lava_method_table` and nowhere else, so plain inference walks into
`error_if_canonical_getindex` and concludes the kernel does nothing at all.
Mantle's dispatch-access analysis is the caller; it is exported for anyone else
reading a kernel before it runs.
"""
function kernelinterpreter(@nospecialize(f), @nospecialize(tt);
                           workgroup_size::NTuple{3,Int} = (64, 1, 1))
    config = lava_compiler_config(; workgroup_size)
    source = GPUCompiler.methodinstance(typeof(f), tt)
    return GPUCompiler.get_interpreter(GPUCompiler.CompilerJob(source, config))
end

# Vulkan doesn't use GPUCompiler's kernel state mechanism
GPUCompiler.kernel_state_type(::LavaCompilerJob) = Nothing

# ── Finishing the IR for Lava's emitter ──
#
# Since 2.13, GPUCompiler's SPIR-V target selects `__spirv_Atomic*` builtins for
# LLVM's atomics in `finish_ir!`, and its `validate_ir` rejects atomics outside
# address spaces 1, 3 and 4. Both are for the Khronos translator and LLVM's SPIR-V
# back-end. Lava emits SPIR-V itself: it lowers `atomicrmw` and `cmpxchg` to the
# atomic instructions, on PhysicalStorageBuffer pointers, which are address space 0.
# So these two keep the target's other steps and leave the atomics to the emitter.
function GPUCompiler.validate_ir(job::LavaCompilerJob, mod::LLVM.Module)
    errors = GPUCompiler.IRError[]
    target = job.config.target
    target.supports_fp16 || append!(errors, GPUCompiler.check_ir_values(mod, LLVM.HalfType()))
    target.supports_fp64 || append!(errors, GPUCompiler.check_ir_values(mod, LLVM.DoubleType()))
    target.supports_bfloat16 ||
        append!(errors, GPUCompiler.check_ir_values(mod, LLVM.BFloatType()))
    return errors
end

function GPUCompiler.finish_ir!(job::LavaCompilerJob, mod::LLVM.Module, entry::LLVM.Function)
    # SPIR-V cannot abort a compute kernel: `trap` goes, `unreachable` becomes `ret`
    GPUCompiler.lower_unreachable_control_flow!(job, mod)
    demote_atomic_accesses!(mod)
    # LLVM's minimum/maximum order signed zeros and propagate NaNs
    GPUCompiler.lower_minimum_maximum!(mod)
    # byval parameters wrapped in a struct, which the entry wrapper expects
    job.config.kernel && (entry = GPUCompiler.wrap_byval(job, mod, entry))
    # SPIR-V has no i128: alloca arrays of it become vectors
    GPUCompiler.convert_i128_allocas!(mod)
    push!(get!(mod.metadata, "opencl.ocl.version").operands,
          LLVM.MDNode([LLVM.ConstantInt(Int32(2)), LLVM.ConstantInt(Int32(0))]))
    push!(get!(mod.metadata, "opencl.spirv.version").operands,
          LLVM.MDNode([LLVM.ConstantInt(Int32(1)), LLVM.ConstantInt(Int32(5))]))
    return entry
end

"""
    demote_atomic_accesses!(mod) -> Bool

Make every atomic load and store a plain one, leaving `atomicrmw` and `cmpxchg`.
The emitter has no atomic load or store, and on the device the orderings of the
loads and stores Julia emits (type tags, references, `unordered` heap accesses)
carry no meaning. GPUCompiler did this for every SPIR-V job until 2.10.
"""
function demote_atomic_accesses!(mod::LLVM.Module)
    changed = false
    for f in mod.functions, bb in f.blocks, inst in bb.instructions
        (inst isa LLVM.LoadInst || inst isa LLVM.StoreInst) || continue
        LLVM.isatomic(inst) || continue
        inst.ordering = LLVM.AtomicOrdering.NotAtomic
        changed = true
    end
    return changed
end

# Relocatable code, so compiled SPIR-V can go into a package image with its
# `CodeInstance` (`compiler/cache.jl`): GPUCompiler keeps results across sessions
# only for the `:patch` and `:table` lowerings. `:table` is the one Vulkan can
# take — code cannot be patched after `vkCreateShaderModule` — and no Lava kernel
# needs it: SPIR-V cannot reach host memory, so a kernel that names a host
# object (a Julia value's address, a global binding) is refused here.
GPUCompiler.relocation_lowering(::LavaCompilerJob) = :table
GPUCompiler.relocation_table_pointer(job::LavaCompilerJob, builder, fun) =
    error("$(job.source) refers to a host object (a Julia value's address or a global " *
          "binding). A Vulkan kernel cannot reach host memory: pass the value as an " *
          "argument, or make it a constant the kernel can hold.")

# ── Disable loop unswitching ──
#
# GPUCompiler's optimizer runs `SimpleLoopUnswitchPass` (twice, at opt_level≥2).
# When a loop contains a branch on a loop-invariant condition, unswitching
# duplicates the loop into two bodies dispatched on that condition. After our
# StructurizeCFG pipeline this becomes two sequential guarded loops joined by
# `Flow` selection-merge blocks. That SPIR-V is valid and semantically correct
# (spirv-val passes, and it matches the LLVM IR), but **NVIDIA's shader compiler
# miscompiles it** — a comparison inside one of the unswitched loop bodies
# silently returns the wrong result. This corrupted, with no error, anything
# with a loop-invariant branch in a loop: e.g. every `Base.isless`-based merge
# sort binary search (AcceleratedKernels.sort!), where the loop-invariant
# operand's NaN check is the branch that gets unswitched.
#
# Unswitching is a pure optimization (never required for correctness) and brings
# little benefit on a GPU — a branch on a uniform/loop-invariant condition is
# cheap and doesn't diverge, while the duplicated loop body just enlarges the
# shader. So we drop ONLY `SimpleLoopUnswitchPass` and keep every other loop
# optimization (LICM, rotation, indvar simplify, unroll, …). This is the general
# root-cause fix for the whole "loop-invariant branch miscompiles on NVIDIA"
# class; pinned by `test_loop_unswitch_miscompile.jl`.
#
# This mirrors GPUCompiler's own `buildLoopOptimizerPipeline` verbatim except for
# the two omitted `SimpleLoopUnswitchPass` lines. Passes live in LLVM.jl; the
# `instcombine_pass`/`*Callbacks`/`BasicSimplifyCFGOptions` helpers are
# GPUCompiler-internal. Dispatching on our own `LavaCompilerJob` makes this a
# normal (precompile-safe) method extension, not piracy. If GPUCompiler's
# pipeline changes upstream, re-sync this body (the regression test will catch a
# silent unswitch reintroduction).
function GPUCompiler.buildLoopOptimizerPipeline(fpm, job::LavaCompilerJob, opt_level)
    # All pass/helper symbols are qualified `GPUCompiler.` — that's the namespace
    # the original pipeline uses them from (some, e.g. LowerSIMDLoopPass /
    # JuliaLICMPass, live in LLVM.Interop and are not bare `LLVM` bindings).
    GPUCompiler.add!(fpm, GPUCompiler.LoopPassManager(; use_memory_ssa=true)) do lpm
        GPUCompiler.add!(lpm, GPUCompiler.LowerSIMDLoopPass())
        if opt_level >= 2
            GPUCompiler.add!(lpm, GPUCompiler.LoopInstSimplifyPass())
            GPUCompiler.add!(lpm, GPUCompiler.LoopSimplifyCFGPass())
            GPUCompiler.add!(lpm, GPUCompiler.LICMPass(; allowspeculation=false))
            GPUCompiler.add!(lpm, GPUCompiler.JuliaLICMPass())
            GPUCompiler.add!(lpm, GPUCompiler.LoopRotatePass())
            GPUCompiler.add!(lpm, GPUCompiler.LICMPass())
            GPUCompiler.add!(lpm, GPUCompiler.JuliaLICMPass())
            # SimpleLoopUnswitchPass intentionally omitted — see comment above.
        end
        if LLVM.version() >= v"17"
            GPUCompiler.add!(lpm, GPUCompiler.LateLoopOptimizationsCallbacks(; opt_level))
        end
    end
    if opt_level >= 2
        GPUCompiler.add!(fpm, GPUCompiler.IRCEPass())
    end
    GPUCompiler.add!(fpm, GPUCompiler.SimplifyCFGPass(; GPUCompiler.BasicSimplifyCFGOptions...))
    GPUCompiler.add!(fpm, GPUCompiler.instcombine_pass(job))
    GPUCompiler.add!(fpm, GPUCompiler.LoopPassManager()) do lpm
        if opt_level >= 2
            GPUCompiler.add!(lpm, GPUCompiler.LoopIdiomRecognizePass())
            GPUCompiler.add!(lpm, GPUCompiler.IndVarSimplifyPass())
            # SimpleLoopUnswitchPass intentionally omitted — see comment above.
            GPUCompiler.add!(lpm, GPUCompiler.LoopDeletionPass())
            GPUCompiler.add!(lpm, GPUCompiler.LoopFullUnrollPass())
        end
        if LLVM.version() >= v"17"
            GPUCompiler.add!(lpm, GPUCompiler.LoopOptimizerEndCallbacks(; opt_level))
        end
    end
end

# Allow SPIR-V intrinsic function calls to pass IR validation.
# llvm.spv.* intrinsics are used by the LLVM SPIR-V backend.
# OpenCL-mangled names (_Z*) are registered dynamically by our intrinsics
# module when @builtin_ccall is used.
const KNOWN_INTRINSICS = String[]

function GPUCompiler.isintrinsic(::LavaCompilerJob, fn::String)
    # LLVM SPIR-V intrinsics (used by the SPIR-V backend for workgroup/subgroup ops)
    startswith(fn, "llvm.spv.") && return true
    # Custom _lava_glsl_* externals → SPIR-V emitter maps to GLSL.std.450
    startswith(fn, "_lava_glsl_") && return true
    # RT intrinsics → SPIR-V emitter maps to OpTraceRayKHR, payload load/store
    startswith(fn, "_lava_rt_") && return true
    # @lava_printf externals → SPIR-V emitter maps to NonSemantic.DebugPrintf
    startswith(fn, "_lava_debug_printf_") && return true
    # Cooperative-matrix and tensor-addressing intrinsics carry their shape in
    # the NAME (`_lava_coopmat_<op>_<dtype>_<M>x<N>_<use>`), which the emitter
    # parses back out. Enumerating the shapes here instead — as this did — makes
    # the gate, not the hardware, decide which shapes exist: coopmat2 FLEXIBLE
    # DIMENSIONS allows any M/N the device's max-dimension limit permits, and a
    # 64x16 matrix was rejected before reaching the emitter that would have
    # handled it fine. A prefix rule defers the decision to the emitter, which
    # errors precisely ("bad shape in …", "not a cooperative-matrix intrinsic",
    # "unsupported tensor-addressing op") rather than silently accepting.
    startswith(fn, "_lava_coopmat_") && return true
    startswith(fn, "_lava_tensor_") && return true
    # A throw's flag store, which `wrap_entry_for_vulkan!` lowers (or deletes)
    # before anything is emitted.
    fn == "_lava_signal_exception" && return true
    # OpenCL C++ mangled builtins (thread indices, barriers, math)
    fn in KNOWN_INTRINSICS && return true
    return false
end

# Override check_invocation to allow types with zero-sized non-isbits fields.
# Julia's broadcast wraps functions in RefValue{typeof(f)} which is not isbits
# but has sizeof == 0 and gets elided by LLVM. GPUCompiler's default check
# rejects these, but they're safe to pass to GPU since they carry no data.
function GPUCompiler.check_invocation(job::LavaCompilerJob)
    sig = job.source.specTypes
    for (arg_i, dt) in enumerate(sig.parameters)
        GPUCompiler.isghosttype(dt) && continue
        Core.Compiler.isconstType(dt) && continue
        fieldcount(dt) == 0 && continue
        if !isbitstype(dt) && !is_gpu_compatible(dt)
            throw(GPUCompiler.KernelError(job, "passing non-bitstype argument",
                """Argument $arg_i to your kernel function is of type $dt, which is not a bitstype:
                   $(GPUCompiler.explain_nonisbits(dt))

                   Only bitstypes can be used in GPU kernels."""))
        end
    end
    return
end

# Check if a non-isbits type is still GPU-compatible by verifying all
# non-isbits fields are zero-sized (will be elided by LLVM).
#
# `Base.allocatedinline(t)` is the predicate that captures both:
#   - non-concrete/UnionAll types (`Vector`, `Tuple`, `Tuple{Vararg{Int}}`)
#   - concrete-but-flexible-size builtins (`Memory{T}`, `Symbol`, `String`),
#     which are heap-referenced and would trip `sizeof` if we called it.
# Anything not allocated inline carries a heap reference and is therefore
# not GPU-compatible.  Once `allocatedinline` is true, `sizeof` is safe.
function is_gpu_compatible(@nospecialize(dt::Type))
    isbitstype(dt) && return true
    Base.allocatedinline(dt) || return false
    sizeof(dt) == 0 && return true
    for i in 1:fieldcount(dt)
        ft = fieldtype(dt, i)
        isbitstype(ft) && continue
        Base.allocatedinline(ft) || return false
        sizeof(ft) == 0 && continue
        is_gpu_compatible(ft) || return false
    end
    return true
end

# ── Compiler configuration cache ──

const COMPILER_CONFIGS = Dict{UInt, LavaCompilerConfig}()

"""
    lava_compiler_config(; workgroup_size=(64,1,1)) -> LavaCompilerConfig

Get or create a cached compiler configuration. The configuration sets up
GPUCompiler's SPIRVCompilerTarget with:
- backend=:llvm for the spirv64 triple (required for LLVM IR generation)
- validate=false (we validate SPIR-V ourselves after our custom emitter)
- supports_fp64=true (AMD RX 7900 XTX supports Float64)
- always_inline=true (SPIR-V requires all functions inlined for compute)
- kernel=true (entry point is a kernel, not a regular function)
"""
function lava_compiler_config(; workgroup_size::NTuple{3,Int} = (64, 1, 1),
                               enable_ray_query::Bool = false,
                               features::TargetFeatures = TargetFeatures(),
                               kwargs...)
    h = hash((workgroup_size, enable_ray_query, features, kwargs))
    config = get(COMPILER_CONFIGS, h, nothing)
    if config !== nothing
        return config
    end
    config = lava_full_compiler_config(; workgroup_size, enable_ray_query, features, kwargs...)
    COMPILER_CONFIGS[h] = config
    return config
end

@noinline function lava_full_compiler_config(;
        kernel = true,
        name = nothing,
        # GPUCompiler's `always_inline=true` raises Julia's inline_cost_threshold
        # to MAX_INLINE_COST, which forces every callee into the entry function
        # at Julia IR level — overriding even `@noinline`. That was needed when
        # Lava's SPIR-V emitter could only handle single-OpFunction kernels, but
        # now that Lava's force_inline_all! supports multi-OpFunction output
        # (force_inline_all=false default), we want Julia/GPUCompiler to respect
        # `@noinline` so callees like `trace_shadow_transmittance` (which has a
        # self-contained ray-query lifecycle) can survive as a separate
        # OpFunction and keep their state out of the caller's register frame.
        always_inline = false,
        workgroup_size::NTuple{3,Int} = (64, 1, 1),
        enable_ray_query::Bool = false,
        features::TargetFeatures = TargetFeatures(),
        kwargs...)
    target = GPUCompiler.SPIRVCompilerTarget(;
        backend = :llvm,       # gives spirv64-unknown-unknown-unknown triple
        validate = false,      # we validate after our custom SPIR-V emitter
        supports_fp64 = true,  # AMD RX 7900 XTX supports Float64
    )
    params = LavaCompilerParams(workgroup_size, enable_ray_query, features)
    GPUCompiler.CompilerConfig(target, params; kernel, name, always_inline)
end
