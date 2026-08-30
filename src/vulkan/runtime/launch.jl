# High-level kernel launch API for Lava.jl
#
# Handles: compile → pipeline cache → arg packing → dispatch → sync.

# ── GPU kernel cache ──
#
# Lookup is delegated to `GPUCompiler.cached_compilation`, which is the same
# primitive AMDGPU.jl and CUDA.jl use. It keys by Julia's `MethodInstance`
# (type-based), so two different closure instances of the same type share
# one compiled kernel — important for KA `@kernel` expansions inside outer
# functions (e.g. WaterLily's `measure!(...) function fill!(...) @loop fill!`
# pattern creates a fresh closure instance on every call, but all of them
# hit the same MethodInstance). `cached_compilation` also carries the
# current world age in the key, so Revise edits to kernel bodies correctly
# invalidate the cache.
#
# Our `compiler` function wraps `lava_compile_gpu_from_job` with Lava's
# own disk cache (KA-generated funcs don't get GPUCompiler's build_id, so
# its built-in disk cache doesn't trigger for them). The `linker` function
# creates the session-dependent `VkPipeline` via `link_kernel(ctx, ...)`.

# Cache shape matches `GPUCompiler.cached_compilation`'s expectation:
# `Dict{Any, LavaLinkedKernel}` with keys like `(objectid(ci), world, cfg)`.
# Per device, because a `LavaLinkedKernel` owns a `VkPipeline` (GUARDRAILS §8).
#
# `Dict{Any, LavaLinkedKernel}` because the dict is handed to
# `GPUCompiler.cached_compilation`, which derives the key from `(source, config)`
# itself — there is no device to add to it from here. One dict per device is the
# guarantee at the level we do control, and it is a field on the device.

"""
    linked_kernel_cache(ctx) -> Dict

This device's compiled-kernel cache, created on first use.
"""
@inline linked_kernel_cache(ctx) = ctx.caches.linked

# No reset callback: arg slabs are per-BQ and die with the old ctx, and the
# kernel cache is now a field on it, so a reset produces a fresh one.

# ── Type signature helper ──
#
# When a kernel arg is itself a Type (e.g. WaterLily's `measure_sdf!` kernels
# capture `T::Type{Float32}` as a value), `typeof(a) === DataType`, and
# `Tuple{..., DataType, ...}` fails `Base.isdispatchtuple` — so GPUCompiler's
# `methodinstance` assert trips. Julia's actual specialization binds that arg
# to `Type{Float32}`, which IS a dispatchtuple leaf. Matching that here.
@inline arg_sigtype(@nospecialize(a)) = a isa Type ? Type{a} : typeof(a)

# ── Launch argument validation ──

"""
    validate_launch_args(bq.ctx::VkContext, args)

Check that buffer arguments are valid (not freed, not poisoned).
Runs by default; disable with `ctx.diag.launch_arg_validation = false`.
"""

@generated function validate_launch_args(ctx::VkContext, args::T) where T <: Tuple
    exprs = Expr[]
    # The toggle is read at RUNTIME inside the generated body, so it can come off
    # the context the caller already has — it does not need to be a global to be
    # cheap. `@generated` constrains what is known at *compile* time, and this is
    # not one of those things.
    push!(exprs, :((ctx.diag.launch_arg_validation || return)))
    for i in 1:fieldcount(T)
        Ti = fieldtype(T, i)
        if Ti <: LavaArray
            push!(exprs, quote
                let arg = args[$i]
                    local buf
                    try
                        buf = arg.buf[]
                    catch
                        throw(LavaError("kernel launch",
                            "Argument $($i): LavaArray has been freed (DataRef released)",
                            "Don't pass freed arrays to GPU kernels. Check array lifetime."))
                    end
                    # Single source of truth: the atomic state machine.  A
                    # buffer is dead iff its state is BUF_STATE_DEAD.  The
                    # BDA_POISON marker is kept as a GPU-side use-after-free
                    # trap (shader reads see 0xDEADDEAD...) but is not the
                    # CPU-side gate any more.
                    if (@atomic :acquire buf.state) == BUF_STATE_DEAD
                        throw(LavaError("kernel launch",
                            "Argument $($i): LavaArray backing buffer is dead (state=BUF_STATE_DEAD)",
                            "This array was freed. Reallocate before use."))
                    end
                end
            end)
        end
    end
    push!(exprs, :(return nothing))
    return Expr(:block, exprs...)
end

"""
    lava_launch!(bq, f, args...; ndrange, workgroup_size=(64,1,1))

Compile and dispatch a Julia function as a Vulkan compute kernel on `bq`.

Arguments flow through `LavaAdaptor(batch)`, which strips every `LavaArray`
reached by `Adapt.adapt` (top level or nested inside closures / wrapper
structs) to a `LavaDeviceArray{T,N}` AND pins its backing `VkManagedBuffer`
into the batch.  Kernels therefore see exactly the post-adapt types — the
same ABI as KernelAbstractions — never `Ptr{T}` for what was originally a
LavaArray.

Example:
    a = LavaArray{Float32,1}(undef, (n,))
    lava_launch!(bq, my_kernel, a, b, Int32(n); ndrange=n, workgroup_size=(256,1,1))
    # kernel signature: my_kernel(a::LavaDeviceArray{Float32,1}, ...)
"""
function lava_launch!(bq::VulkanBatchQueue, @nospecialize(f), args...;
                       ndrange::Union{Integer, NTuple{3,<:Integer}},
                       workgroup_size::NTuple{3,Int} = (64, 1, 1),
                       tlas=nothing)  # Union{Nothing, VulkanTLAS} — declared later in raytracing/hwtlas.jl
    validate_launch_args(bq.ctx::VkContext, args)
    if ndrange isa Integer
        ndrange_3d = (Int(ndrange), 1, 1)
    else
        ndrange_3d = (Int(ndrange[1]), Int(ndrange[2]), Int(ndrange[3]))
    end
    groups = (
        cld(ndrange_3d[1], workgroup_size[1]),
        cld(ndrange_3d[2], workgroup_size[2]),
        cld(ndrange_3d[3], workgroup_size[3]),
    )

    # Pin pass (side effects): walk the closure + args and pin every LavaArray
    # leaf into `batch.pinned` so the backing VkManagedBuffers outlive submit.
    # Strip pass (pure): Adapt.jl rewrites LavaArray → LavaDeviceArray via the
    # side-effect-free `adapt_storage(::LavaAdaptor, ::LavaArray)`.
    batch = ensure_active_batch!(bq)
    pin_leaves!(batch, f)
    pin_leaves!(batch, args)
    adaptor = LavaAdaptor(batch)
    converted_f = Adapt.adapt(adaptor, f)
    converted_args = map(a -> Adapt.adapt(adaptor, a), args)

    # Kernel ABI: kernel signature types are POST-adapt (LavaDeviceArray, not Ptr{T}).
    tt = Tuple{map(arg_sigtype, converted_args)...}

    enable_ray_query = tlas !== nothing
    compiled, pipeline, offsets, byval_sizes =
        get_compiled_kernel_and_pipeline(bq.ctx::VkContext, converted_f, tt, workgroup_size;
                                         enable_ray_query)

    # Loud error: kernel needs HWTLAS but none was provided at launch.
    if pipeline.needs_tlas_descriptor && tlas === nothing
        error("kernel was compiled with enable_ray_query=true but launch_kernel was " *
              "called without a tlas keyword. Pass tlas=<VulkanTLAS> to bind the " *
              "acceleration structure to descriptor set 0, binding 0.")
    end

    # GPUCompiler prepends typeof(f) as the LLVM entry's first param; the entry
    # wrapper allocates a BDA slot for it unless it's a ghost type.
    all_args = (converted_f, converted_args...)
    inline_extra = compute_inline_extra_from_byval(byval_sizes)
    total_size = compiled.push_info.arg_buffer_size + inline_extra
    arg_buf = get_arg_buffer(bq, total_size)

    pack_args_direct!(bq, arg_buf.mapped_ptr, arg_buf.address, offsets,
                       compiled.push_info.arg_buffer_size, byval_sizes, all_args)

    if (bq.ctx::VkContext).diag.dispatch_logging
        bq.last_dispatch_info = "compute f=$(nameof(typeof(converted_f))) groups=$groups"
    end
    vk_dispatch!(bq, pipeline, arg_buf.address, groups, tlas)
    return nothing
end

# ── Zero-allocation argument packing ──
#
# Kernel args are laid out in a host-mapped arg buffer (one sub-allocation per
# dispatch).  The push constant is the BDA of that sub-allocation; the entry
# wrapper generated by the SPIR-V compiler loads each arg from
# `arg_buf[layout[i]]` at kernel entry.
#
# For reference/BDA-pointer args (VkManagedBuffer, LavaDeviceArray, Ptr{T}):
# the slot holds a 64-bit BDA address.  For isbits struct args: the bytes are
# inlined right after the fixed layout and the slot holds a self-referencing
# BDA that points back at the inlined bytes.
#
# Every leaf that is a GPU-visible buffer was already stripped + pinned by
# `LavaAdaptor` at the kernel call boundary, so by the time `pack_args_direct!`
# runs we're looking at `LavaDeviceArray{T,N}` (a plain isbits struct) — no
# special case needed for it here.  `VkManagedBuffer` is kept as an escape
# hatch for a few internal low-level callers.

"""
    compute_inline_extra_from_byval(byval_sizes::Vector{Int})

Total bytes needed after the fixed arg layout to hold inlined isbits-struct
data.  Uses LLVM byval sizes (which can exceed Julia's sizeof for types with
zero-sized fields like Nothing).
"""
function compute_inline_extra_from_byval(byval_sizes::Vector{Int})
    extra = 0
    for sz in byval_sizes
        sz > 0 || continue
        extra = (extra + 7) & ~7
        extra += sz
    end
    return extra
end

# Per-type arg packer.  Dispatched (no big if/elseif ladder):
#   * default: isbits struct → inline + self-ref BDA; primitive → direct store
#   * UInt64: direct 64-bit store
#   * Ptr: direct 64-bit store (the pointer value *is* the BDA)
#   * VkManagedBuffer: pin for batch lifetime + write its BDA

@inline function pack_arg!(x::T,
                           mapped_ptr::Ptr{UInt8}, arg_buf_bda::UInt64,
                           offset::Int, byval_size::Int, inline_offset::Int,
                           batch::CommandBatch) where T
    if isbitstype(T) && !isprimitivetype(T)
        inline_offset = (inline_offset + 7) & ~7
        ccall(:memset, Ptr{Cvoid}, (Ptr{Cvoid}, Cint, Csize_t),
              mapped_ptr + inline_offset, 0, byval_size)
        unsafe_store!(Ptr{T}(mapped_ptr + inline_offset), x)
        unsafe_store!(Ptr{UInt64}(mapped_ptr + offset),
                      arg_buf_bda + UInt64(inline_offset))
        return inline_offset + byval_size
    else
        unsafe_store!(Ptr{T}(mapped_ptr + offset), x)
        return inline_offset
    end
end

@inline function pack_arg!(x::UInt64,
                           mapped_ptr::Ptr{UInt8}, arg_buf_bda::UInt64,
                           offset::Int, byval_size::Int, inline_offset::Int,
                           batch::CommandBatch)
    unsafe_store!(Ptr{UInt64}(mapped_ptr + offset), x)
    return inline_offset
end

@inline function pack_arg!(p::Ptr,
                           mapped_ptr::Ptr{UInt8}, arg_buf_bda::UInt64,
                           offset::Int, byval_size::Int, inline_offset::Int,
                           batch::CommandBatch)
    unsafe_store!(Ptr{UInt64}(mapped_ptr + offset), UInt64(p))
    return inline_offset
end

@inline function pack_arg!(buf::VkManagedBuffer,
                           mapped_ptr::Ptr{UInt8}, arg_buf_bda::UInt64,
                           offset::Int, byval_size::Int, inline_offset::Int,
                           batch::CommandBatch)
    pin!(batch, buf)
    if (buf.ctx::VkContext).diag.pack_arg_assert_live
        st = @atomic :acquire buf.state
        if st != BUF_STATE_ALIVE
            throw(LavaError("pack_arg!",
                "packing buffer with state=$st (not ALIVE) at addr=0x$(string(buf.address, base=16, pad=16))",
                "buffer was freed but its BDA is being packed into an arg slab — use-after-free"))
        end
    end
    unsafe_store!(Ptr{UInt64}(mapped_ptr + offset), buf.address)
    return inline_offset
end

"""
    pack_args_direct!(bq, mapped_ptr, arg_buf_bda, offsets, base_size, byval_sizes, all_args)

Write kernel arguments directly into mapped GPU memory via per-type
`pack_arg!` dispatch.  Zero heap allocations for the arg packing itself.
`bq.active_batch` must already exist (callers call `ensure_active_batch!`
before us).
"""
@generated function pack_args_direct!(bq::VulkanBatchQueue,
                                        mapped_ptr::Ptr{UInt8}, arg_buf_bda::UInt64,
                                        offsets::Vector{Int}, base_size::Int,
                                        byval_sizes::Vector{Int},
                                        all_args::T) where {T <: Tuple}
    types = T.parameters
    non_ghost = Int[]
    for (i, Ti) in enumerate(types)
        sizeof(Ti) == 0 && continue
        # Type-valued args (e.g. `T=Float32` captured by @kernel) are ghost
        # in GPUCompiler's view — push_info allocates no bytes for them —
        # even though `typeof(Float32) === DataType` has nonzero sizeof.
        # Skip or we'll pack into a slot that doesn't exist → segfault.
        Ti <: Type && continue
        push!(non_ghost, i)
    end
    exprs = Expr[]
    for (layout_i, arg_i) in enumerate(non_ghost)
        push!(exprs, :(inline_offset = pack_arg!(
            all_args[$arg_i], mapped_ptr, arg_buf_bda,
            @inbounds(offsets[$layout_i]),
            @inbounds(byval_sizes[$layout_i]),
            inline_offset, batch)))
    end
    quote
        batch = bq.active_batch::CommandBatch
        inline_offset = base_size
        $(exprs...)
        return nothing
    end
end


# ── Lava disk cache ──
# GPUCompiler's disk cache only works for precompiled package code (needs build_id).
# KA @kernel macros generate functions at expansion time without build_id, so we
# implement our own disk cache keyed by (specTypes hash, workgroup_size).
# The specTypes hash is stable across sessions for the same kernel+argtypes.

const LAVA_DISK_CACHE_DIR = Ref("")

function lava_disk_cache_dir()
    dir = LAVA_DISK_CACHE_DIR[]
    if isempty(dir)
        dir = joinpath(first(Base.DEPOT_PATH), "scratchspaces", "lava_spirv_cache")
        LAVA_DISK_CACHE_DIR[] = dir
    end
    return dir
end

function lava_disk_cache_key(source::Core.MethodInstance, workgroup_size)
    # Hash the type signature as a STRING for stability across sessions.
    # Julia's hash(Type) uses object identity which changes per session.
    # String representation is stable for the same source code.
    h = hash(string(source.specTypes))
    h = hash(workgroup_size, h)
    return string(h, base=16) * ".jls"
end

"""
    lava_disk_cache_load(source, workgroup_size) -> Union{Nothing, LavaGPUKernel}

Try to load cached SPIR-V from disk. Returns nothing on miss.
"""
function lava_disk_cache_load(source::Core.MethodInstance, workgroup_size)
    dir = lava_disk_cache_dir()
    isdir(dir) || return nothing
    path = joinpath(dir, lava_disk_cache_key(source, workgroup_size))
    isfile(path) || return nothing
    local entry
    try
        entry = open(Serialization.deserialize, path)
    catch ex
        # Narrowed like the frozen cache's readers: a truncated or
        # version-mismatched entry is a recompile, anything else is a bug here
        # and must not be absorbed by a cache miss.
        cache_io_error(ex) || rethrow()
        @warn "Lava: disk cache load failed" path exception=(ex, catch_backtrace()) maxlog = 1
        return nothing
    end
    if string(entry.spec_types) == string(source.specTypes) && entry.workgroup_size == workgroup_size
        return entry.kernel::LavaGPUKernel
    end
    return nothing
end

"""
    lava_disk_cache_store(source, workgroup_size, kernel::LavaGPUKernel)

Store compiled SPIR-V to disk for future sessions.
"""
function lava_disk_cache_store(source::Core.MethodInstance, workgroup_size, kernel::LavaGPUKernel)
    dir = lava_disk_cache_dir()
    mkpath(dir)
    path = joinpath(dir, lava_disk_cache_key(source, workgroup_size))
    entry = (
        spec_types = source.specTypes,
        workgroup_size = workgroup_size,
        kernel = LavaGPUKernel(
            kernel.spirv_bytes, kernel.entry_name, kernel.workgroup_size,
            kernel.push_info, "",  # don't cache the LLVM IR string (large, session-specific)
            kernel.enable_ray_query,
            kernel.source_name     # …but DO keep it: small, portable, and the profiler needs it
        ),
    )
    try
        tmppath, io = mktemp(dir; cleanup=false)
        Serialization.serialize(io, entry)
        close(io)
        mv(tmppath, path; force=true)
    catch ex
        # A cache is an optimisation, so a failed WRITE may not take the session
        # down — but it must be visible, or a permanently unwritable cache looks
        # exactly like a working one. IO faults only; anything else is a bug here.
        ex isa Union{SystemError, IOError, ArgumentError} || rethrow()
        @warn "Lava: disk cache store failed; kernels will recompile next session" path exception = ex maxlog = 1
    end
end

"""Clear Lava's SPIR-V disk cache."""
function clear_spirv_disk_cache!()
    dir = lava_disk_cache_dir()
    isdir(dir) && rm(dir; recursive=true, force=true)
end

"""
    clear_kernel_cache!()

Evict this device's in-session kernel + pipeline caches so the next dispatch of
each kernel recompiles from Julia source.

Use this after editing a Julia kernel under Revise — Revise invalidates the
Julia method, but Lava's hash-keyed kernel cache stays populated with the old
SPIR-V because `hash(f, tt, workgroup_size)` doesn't change when the method
body changes. Unlike `reset_device!()`, this keeps all existing
`LavaArray`s and the Vulkan context alive.

**Both** caches have to go. `caches.launchplans` holds its own `VkPipeline`
and is consulted *before* `caches.linked` on every dispatch, so emptying
only the latter left the old pipeline running with no symptom — the function
silently did nothing. That is not hypothetical: it made a SPIR-V A/B harness
report "no difference" for six variants on 2026-08-02, including one that had
its `OpStore` deleted. The Revise path happened to work anyway, because a
method redefinition moves the world counter and `launch_plan` rejects plans
from a superseded world; a caller who only clears the cache had no such luck.
"""
function clear_kernel_cache!(ctx::VkContext = vk_context())
    empty!(ctx.caches.linked)
    empty!(ctx.caches.launchplans)
    return nothing
end

"""
    link_kernel(compiled::LavaGPUKernel) -> LavaLinkedKernel

Create session-dependent Vulkan objects (VkPipeline) from cached SPIR-V bytes.
"""
function link_kernel(ctx::VkContext, compiled::LavaGPUKernel; pipeline_cache=nothing)
    pipeline = get_compute_pipeline(ctx, compiled.spirv_bytes, compiled.entry_name;
                                    push_constant_size=compiled.push_info.push_size,
                                    needs_tlas_descriptor=compiled.enable_ray_query,
                                    pipeline_cache)
    offsets = compiled.push_info.arg_offsets
    byval_sizes = compiled.push_info.byval_llvm_sizes
    return LavaLinkedKernel(compiled, pipeline, offsets, byval_sizes)
end

"""
    lava_kernel_compile(job::GPUCompiler.CompilerJob) -> LavaGPUKernel

`compiler` function passed to `GPUCompiler.cached_compilation`. Must call
`GPUCompiler.compile` (which `lava_compile_gpu_from_job` does) so that a
`CodeInstance` is registered in GPUCompiler's ci_cache — `cached_compilation`
looks it up after the linker runs.  Still writes to Lava's disk cache on
compile so subsequent sessions can short-circuit via `lava_disk_cache_load`
in `link_kernel` (see below).
"""
function lava_kernel_compile(job::GPUCompiler.CompilerJob)
    # enable_ray_query is read from the job's LavaCompilerParams,
    # where it was set by lava_compiler_config(; enable_ray_query).
    # Try the disk cache first — same (specTypes, workgroup_size) yields the
    # same SPIR-V bytes across sessions, which lets the driver's persistent
    # VkPipelineCache match by bit-identical SPIR-V hash.
    #
    # Opt-in via env var: empirically AMDVLK Windows crashes when fed
    # previously-serialized SPIR-V (even byte-identical to a fresh compile).
    # Other drivers are fine. We always WRITE to disk so the cache is ready
    # if/when the load gets enabled; only the LOAD path is gated.
    if get(ENV, "LAVA_LOAD_SPIRV_DISK_CACHE", "0") == "1"
        cached = lava_disk_cache_load(job.source, job.config.params.workgroup_size)
        cached === nothing || return cached
    end
    # Run the whole pipeline in the world frozen at `__init__`. CUDACore does
    # this around `GPUCompiler.compile` and cuTile around its entire
    # `cufunction_compile`; the latter is the right analogue here because Lava's
    # own SPIR-V emitter is roughly half of compile time and is just as
    # invalidatable as GPUCompiler's codegen.
    #
    # `invoke_in_world` is not inferable, hence the return-type annotation.
    compiled = invoke_frozen(lava_compile_gpu_from_job, job)::LavaGPUKernel
    lava_disk_cache_store(job.source, job.config.params.workgroup_size, compiled)
    return compiled
end

"""
    LavaLinker(ctx)

`linker` for `GPUCompiler.cached_compilation`.  A callable struct with one
field, NOT a closure — closures get a fresh anonymous type per call site,
which forces Julia to re-infer `cached_compilation` (and its 5+ generic
parameters) on every dispatch.  That cost showed up as massive
`typeinf_ext_toplevel` time in the profile during WaterLily steady-state.
With a struct, `typeof(linker)` is stable, so cached_compilation hits its
own MethodInstance cache and skips inference.
"""
# `<: Function` lets it satisfy `cached_compilation`'s `linker::Function` arg.
struct LavaLinker <: Function
    ctx::VkContext
end
# While recording, each kernel is linked through a pipeline cache holding only
# itself, so `frozen_store` can snapshot that kernel's ISA rather than whatever
# the device-wide cache has accumulated. Outside recording this is the plain
# path and the device-wide cache is used as before.
@inline function (l::LavaLinker)(::GPUCompiler.CompilerJob, compiled::LavaGPUKernel)
    if FROZEN_RECORDING[] && !isempty(FROZEN_VERSION[])
        # No try. Creating an EMPTY pipeline cache cannot fail for any reason
        # this code can handle: it takes no input to be malformed. The bare
        # `catch nothing` here meant a failure produced `pc = nothing`, which
        # `frozen_store` reads as "no ISA to snapshot" — so the frozen cache
        # would silently degrade to level 1 forever, invisibly.
        pc = VK.PipelineCache(l.ctx.device,
                                  VK.PipelineCacheCreateInfo(Ptr{Cvoid}(C_NULL);
                                                                 initial_data_size=UInt64(0)))
        l.ctx.caches.frozen_last_pcache = pc
        return link_kernel(l.ctx, compiled; pipeline_cache=pc)
    end
    l.ctx.caches.frozen_last_pcache = nothing
    return link_kernel(l.ctx, compiled)
end

"""
    get_compiled_kernel_and_pipeline(ctx, f, tt, workgroup_size)
      -> (compiled, pipeline, offsets, byval_sizes)

Look up (or compile + cache) the SPIR-V kernel + VkPipeline for `(f, tt,
workgroup_size)`.  Delegates to `GPUCompiler.cached_compilation`, which
hashes by `(objectid(MethodInstance), world, cfg)` — type-based, so
different closure instances of the same type share one compiled kernel,
and world-age tracking means Revise edits invalidate correctly.
"""
#
# `@noinline` is load-bearing, not a code-size preference. `@nospecialize` on `f`
# and `tt` widens this method's OWN signature, but inference will still infer a
# specialised copy in order to inline it into a caller that is itself specialised
# per kernel — which is every launch site. Measured on SAM 2's encoder: 1 002
# specialisations of this function and of `frozen_load`/`frozen_store` beneath
# it, one per kernel, for bodies that do not vary with the kernel. The barrier in
# `build_launch_plan!` supplies the type-erased caller; this keeps the compiler
# from undoing it.
@noinline function get_compiled_kernel_and_pipeline(ctx::VkContext, @nospecialize(f), @nospecialize(tt),
                                          workgroup_size;
                                          enable_ray_query::Bool=false)
    # The frozen cache first, and deliberately before anything that needs a
    # `MethodInstance`. `GPUCompiler.methodinstance` infers the kernel, so
    # reaching `cached_compilation` at all costs the inference the cache exists
    # to avoid — SAM 2's encoder spends 70 s there with every kernel already on
    # disk. `frozen_load` keys on types the caller already holds.
    let hit = frozen_load(ctx, f, tt, workgroup_size)
        if hit !== nothing
            hit = hit::LavaLinkedKernel
            return hit.compiled, hit.pipeline, hit.offsets, hit.byval_sizes
        end
    end
    isempty(FROZEN_VERSION[]) || (FROZEN_MISSES[] += 1)

    # `invokelatest` around the compile, for the same reason `vk_context` uses one
    # (see `device.jl`): a direct call puts GPUCompiler, the SPIR-V emitter and
    # the file-path handling in `run_spirv_opt`/`validate_spirv`/
    # `dump_spirv_to_disk` into the inference chain of the LAUNCH path. That is a
    # lot of foreign surface to depend on — `FilePathsBase` pirates `Base.arg_gen`
    # and `Base.*`, `Unitful` pirates `Base.Colon` — so loading anything that
    # pulls them in invalidates `get_or_build_iter_plan` and every launch above
    # it. This runs once per kernel; the dispatch is free next to compiling one.
    config = lava_compiler_config(; workgroup_size, enable_ray_query)
    source = GPUCompiler.methodinstance(typeof(f), tt)
    linked = Base.invokelatest(GPUCompiler.cached_compilation, linked_kernel_cache(ctx),
                               source, config, lava_kernel_compile, LavaLinker(ctx))::LavaLinkedKernel
    frozen_store(ctx, f, tt, workgroup_size, linked.compiled)

    # Dump SPIR-V if dump dir is set
    dd = ctx.diag.spirv_dump_dir
    if dd !== nothing && !isempty(dd)
        ctx.diag.spirv_dump_counter += 1
        fname = string(nameof(typeof(f)))
        path = joinpath(dd, "$(lpad(ctx.diag.spirv_dump_counter, 3, '0'))_$(fname).spv")
        write(path, linked.compiled.spirv_bytes)
    end

    return linked.compiled, linked.pipeline, linked.offsets, linked.byval_sizes
end

# ── Arg buffer + indirect dispatch slab allocators ──
#
# Each GPU dispatch needs an arg buffer region (for push-constant BDA pointer
# target) and optionally an indirect dispatch buffer (3 UInt32s).  Both kinds
# of slab are plain `LavaArray`s with BAR-mapped memory (unified=true) —
# there is no raw "VkMappedBuffer" type any more.  Slabs are bump-allocated
# with 256-byte alignment and recycled when `bq.in_flight` drains.

const ARG_SLAB_SIZE = 4 * 1024 * 1024     # 4 MiB per arg slab (~16K dispatches)
const ARG_SLAB_ALIGN = 256                # BDA alignment for sub-allocations
const INDIRECT_SLAB_ALIGN = 256           # 12 bytes needed, 256-aligned
const INDIRECT_SLAB_ELEMS = INDIRECT_SLAB_SIZE ÷ sizeof(UInt32)

"""A sub-allocation within an arg buffer slab (CPU-mapped write target)."""
struct ArgBufferAlloc
    address::UInt64           # BDA of this sub-allocation
    mapped_ptr::Ptr{UInt8}    # CPU-writable pointer
    size::Int
end

"""Lazy-grow `bq.arg_slabs` so `bq.arg_slab_idx` indexes a live slab big
enough for `min_size`.  Advances `arg_slab_idx` if the current slab is full."""
function ensure_arg_slab!(bq::VulkanBatchQueue, min_size::Int)
    while length(bq.arg_slabs) < bq.arg_slab_idx
        push!(bq.arg_slabs,
              LavaArray{UInt8,1}(undef, (max(ARG_SLAB_SIZE, min_size),);
                                 bq=bq, unified=true))
    end
    slab = bq.arg_slabs[bq.arg_slab_idx]::LavaArray{UInt8,1}
    if bq.arg_slab_offset + min_size > slab.buf[].size
        bq.arg_slab_idx += 1
        bq.arg_slab_offset = 0
        while length(bq.arg_slabs) < bq.arg_slab_idx
            push!(bq.arg_slabs,
                  LavaArray{UInt8,1}(undef, (max(ARG_SLAB_SIZE, min_size),);
                                     bq=bq, unified=true))
        end
    end
end

function get_arg_buffer(bq::VulkanBatchQueue, nbytes::Integer)
    @assert Threads.threadid() == bq.owning_thread  "VulkanBatchQueue is single-writer; cross-thread arg-buf alloc forbidden"
    aligned_size = (max(Int(nbytes), 16) + ARG_SLAB_ALIGN - 1) & ~(ARG_SLAB_ALIGN - 1)
    # A capture allocates from its OWN slabs. The address is baked into the
    # command buffer as a push constant, so a replay reads whatever those bytes
    # hold at replay time — which means nothing else may ever be given them. That
    # used to be arranged by pushing the shared pool's high-water mark past them
    # for good; owning them says the same thing and can also be undone. See
    # `CapturedSequence`.
    let cap = bq.capturing
        cap === nothing || return capture_arg_buffer!(cap, bq, aligned_size)
    end
    # Rewind here rather than at end-of-frame: safe exactly when the GPU has
    # passed every batch that allocated from the pool (see `arg_pool_in_use!`).
    reclaim_arg_buffer_pool!(bq)
    ensure_arg_slab!(bq, aligned_size)
    slab = bq.arg_slabs[bq.arg_slab_idx]::LavaArray{UInt8,1}
    mb = slab.buf[]::VkManagedBuffer
    offset = bq.arg_slab_offset
    bq.arg_slab_offset = offset + aligned_size
    bq.arg_alloc_count += 1
    return ArgBufferAlloc(
        mb.address + UInt64(offset),
        mb.mapped_ptr + offset,
        aligned_size
    )
end

"""Bump-allocate `aligned_size` bytes from a capture's own slabs, growing them
as needed. No reclaim and no frontier: a capture's slabs are never rewound,
which is the whole point of them being its own."""
function capture_arg_buffer!(cap, bq::VulkanBatchQueue, aligned_size::Int)
    while length(cap.slabs) < cap.slab_idx
        push!(cap.slabs, LavaArray{UInt8,1}(undef, (max(ARG_SLAB_SIZE, aligned_size),);
                                            bq = bq, unified = true))
    end
    slab = cap.slabs[cap.slab_idx]::LavaArray{UInt8,1}
    if cap.slab_offset + aligned_size > slab.buf[].size
        cap.slab_idx += 1
        cap.slab_offset = 0
        while length(cap.slabs) < cap.slab_idx
            push!(cap.slabs, LavaArray{UInt8,1}(undef, (max(ARG_SLAB_SIZE, aligned_size),);
                                                bq = bq, unified = true))
        end
        slab = cap.slabs[cap.slab_idx]::LavaArray{UInt8,1}
    end
    mb = slab.buf[]::VkManagedBuffer
    offset = cap.slab_offset
    cap.slab_offset = offset + aligned_size
    return ArgBufferAlloc(mb.address + UInt64(offset), mb.mapped_ptr + offset, aligned_size)
end

# `reserve_arg_slabs!` used to live here: it pushed a high-water mark past the
# slabs a capture had filled so `reset_arg_buffer_pool!` could not hand them out
# again, because a replay reads whatever the baked push-constant address points
# at. The mark only went up and nothing ever lowered it, so every capture cost
# the pool a few megabytes for the life of the process. A capture owns its
# argument slabs now, which says the same thing about who may write them and can
# also be given back — see `CapturedSequence` and `release!`.

"""
    arg_pool_in_use!(bq, signal_value)

Record that everything allocated from `bq`'s arg pool is read by batches up to
`signal_value`; the pool may be rewound once the timeline passes it.

The pool is a bump allocator whose addresses are baked into command buffers as
push constants, so rewinding it while a batch that allocated from it is still
executing hands the next caller memory an in-flight shader is still reading.
Rewinding at end-of-frame did exactly that: geometry corruption over a static
scene, intermittent, hidden by any full sync, and invisible to validation —
overwriting your own host-mapped memory is perfectly legal.
"""
function arg_pool_in_use!(bq::VulkanBatchQueue, signal_value::Integer)
    bq.arg_pool_frontier = UInt64(signal_value)
    # Everything handed out so far now belongs to a submitted batch, and the
    # frontier covers it. The recording that starts next holds nothing yet, which
    # is what `arg_alloc_count` means from here on.
    bq.arg_alloc_count = 0
    nothing
end

"""
Rewind the arg pool if — and only if — the GPU has finished with everything
allocated from it *and* the recording in progress holds none of it. Cheap: one
non-blocking timeline query, and only when there is a frontier to clear.

The second condition is not redundant, and leaving it out is a GPU crash. The
timeline can cross the frontier *while a frame is being recorded*: draws 1..k have
already had their arg buffer addresses baked into push constants, the GPU then
finishes the previous frame, and the next `get_arg_buffer` rewinds to offset zero
and hands the same bytes to draw k+1. The earlier draws are left reading whatever
the later ones wrote — a null buffer device address, and a GPUVM fault at 0x0.

It needs many draws in one frame for a reclaim to land mid-recording, and frames
in flight for the timeline to move during it, which is why it appeared the day the
per-frame flush went away and only with a hundred plots. `arg_alloc_count` is the
count since the last submit, so it is exactly "this recording holds handouts".
"""
function reclaim_arg_buffer_pool!(bq::VulkanBatchQueue)
    bq.arg_pool_frontier == UInt64(0) && return false
    bq.arg_alloc_count == 0 || return false
    query_timeline(bq) >= bq.arg_pool_frontier || return false
    reset_arg_buffer_pool!(bq)
    bq.arg_pool_frontier = UInt64(0)
    return true
end

"""Reset arg buffer slab allocator for `bq` after its in_flight batches drained.

All the way to the first slab: nothing is reserved here any more, because a
capture allocates its arguments from slabs it owns rather than from this pool."""
function reset_arg_buffer_pool!(bq::VulkanBatchQueue)
    bq.arg_slab_idx = 1
    bq.arg_slab_offset = 0
    bq.arg_alloc_count = 0
    bq.arg_pool_frontier = UInt64(0)
end

"""Lazy-grow `bq.indirect_slabs` with a LavaArray{UInt32,1} backed by
host-mapped BAR memory + INDIRECT_BUFFER usage.  Same sub-allocation shape
as arg slabs; every sub-allocation is a LavaArray view over the slab so
callers never see a raw Vulkan buffer."""
function ensure_indirect_slab!(bq::VulkanBatchQueue)
    while length(bq.indirect_slabs) < bq.indirect_slab_idx
        push!(bq.indirect_slabs,
              LavaArray{UInt32,1}(undef, (INDIRECT_SLAB_ELEMS,);
                  bq=bq, unified=true,
                  extra_usage=UInt32(VK.BUFFER_USAGE_INDIRECT_BUFFER_BIT)))
    end
end

"""
    get_indirect_buffer(bq) -> LavaArray{UInt32,1}

Sub-allocate a 3-element LavaArray view from `bq`'s indirect-dispatch slab
(enough for one `VkDispatchIndirectCommand`).  The view shares the slab's
DataRef, so the slab stays alive while any view is pinned.
"""
function get_indirect_buffer(bq::VulkanBatchQueue)
    @assert Threads.threadid() == bq.owning_thread  "VulkanBatchQueue is single-writer; cross-thread indirect-buf alloc forbidden"
    alloc_bytes = INDIRECT_SLAB_ALIGN
    ensure_indirect_slab!(bq)
    slab = bq.indirect_slabs[bq.indirect_slab_idx]::LavaArray{UInt32,1}
    if bq.indirect_slab_offset + alloc_bytes > slab.buf[].size
        bq.indirect_slab_idx += 1
        bq.indirect_slab_offset = 0
        ensure_indirect_slab!(bq)
        slab = bq.indirect_slabs[bq.indirect_slab_idx]::LavaArray{UInt32,1}
    end
    byte_offset = bq.indirect_slab_offset
    bq.indirect_slab_offset = byte_offset + alloc_bytes
    ref = copy(slab.buf)
    return LavaArray{UInt32,1}(ref, (3,); offset=byte_offset)
end

function reset_indirect_buffer_pool!(bq::VulkanBatchQueue)
    bq.indirect_slab_idx = 1
    bq.indirect_slab_offset = 0
end
