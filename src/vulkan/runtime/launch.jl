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
    validate_launch_args(ctxof(bq), args)

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
    oneshot!(bq; tag = :launch) do e
        emitkernel!(e, f, args...; ndrange, workgroup_size, tlas)
    end
    return nothing
end

"""
    emitkernel!(emitter, f, args...; ndrange, workgroup_size, tlas = nothing)

The same launch, written where the emitter is writing.

`lava_launch!` is the unmodelled path: nothing has declared what the kernel
touches, so it goes into a one-shot of its own, which opens with the global
barrier and is submitted on the way out. This one writes into whatever the
emitter holds — a plan's recording, whose barriers were derived from declared
usage and already emitted, or a one-shot a caller is composing.

Everything up to the dispatch is shared: [`preparekernel`](@ref) compiles,
adapts, pins and packs identically for both.
"""
function emitkernel!(e::Emitter, @nospecialize(f), args...;
                     ndrange::Union{Integer, NTuple{3,<:Integer}},
                     workgroup_size::NTuple{3,Int} = (64, 1, 1),
                     tlas = nothing)
    pipeline, argaddr, groups, name =
        preparekernel(e.owner, f, args, ndrange, workgroup_size, tlas)
    if e.ctx.diag.dispatch_logging
        queueof(e).last_dispatch_info = name
    end
    emit_dispatch!(e, pipeline, argaddr, groups, tlas, name)
    return nothing
end

"""
Compile `f` for `args`, pin and pack them where `owner` holds memory, and answer
what a dispatch needs: the pipeline, the argument address, the workgroup counts
and a name for the log.

The whole of a launch except the launch. `lava_launch!` and [`emitkernel!`](@ref)
differ in one line each — which command they then write — so this is where the
kernel cache lookup, the `Adapt` walk, the HWTLAS check and the argument packing
live once.
"""
function preparekernel(owner::O, @nospecialize(f), args::Tuple,
                       ndrange::Union{Integer, NTuple{3,<:Integer}},
                       workgroup_size::NTuple{3,Int}, tlas) where {O<:Closed}
    bq = queueof(owner)
    ctx = ctxof(bq)
    validate_launch_args(ctx, args)
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
    # leaf into the owner's pinned set so the backing VkManagedBuffers outlive
    # the last submission that names them.
    # Strip pass (pure): Adapt.jl rewrites LavaArray → LavaDeviceArray via the
    # side-effect-free `adapt_storage(::LavaAdaptor, ::LavaArray)`.
    holdleaves!(owner, f)
    holdleaves!(owner, args)
    adaptor = LavaAdaptor(owner)
    converted_f = Adapt.adapt(adaptor, f)
    converted_args = map(a -> Adapt.adapt(adaptor, a), args)

    # Kernel ABI: kernel signature types are POST-adapt (LavaDeviceArray, not Ptr{T}).
    tt = Tuple{map(arg_sigtype, converted_args)...}

    enable_ray_query = tlas !== nothing
    compiled, pipeline, offsets, byval_sizes =
        get_compiled_kernel_and_pipeline(ctx, converted_f, tt, workgroup_size;
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
    arg_buf = get_arg_buffer(owner, total_size)

    pack_args_direct!(owner, arg_buf.mapped_ptr, arg_buf.address, offsets,
                       compiled.push_info.arg_buffer_size, byval_sizes, all_args)

    name = ctx.diag.dispatch_logging || ctx.diag.dispatch_timing ?
           "compute f=$(nameof(typeof(converted_f))) groups=$groups" : ""
    return pipeline, arg_buf.address, groups, name
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
                           batch::O) where {T,O<:Closed}
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
                           batch::O) where {O<:Closed}
    unsafe_store!(Ptr{UInt64}(mapped_ptr + offset), x)
    return inline_offset
end

@inline function pack_arg!(p::Ptr,
                           mapped_ptr::Ptr{UInt8}, arg_buf_bda::UInt64,
                           offset::Int, byval_size::Int, inline_offset::Int,
                           batch::O) where {O<:Closed}
    unsafe_store!(Ptr{UInt64}(mapped_ptr + offset), UInt64(p))
    return inline_offset
end

@inline function pack_arg!(buf::VkManagedBuffer,
                           mapped_ptr::Ptr{UInt8}, arg_buf_bda::UInt64,
                           offset::Int, byval_size::Int, inline_offset::Int,
                           batch::O) where {O<:Closed}
    # The kernel reads through this address, so the submission must outlive the
    # buffer and be ordered against whoever wrote it last.
    hold!(batch, buf)
    if (buf.ctx::VkContext).diag.pack_arg_assert_live
        st = @atomic :acquire buf.state
        if st != BUF_STATE_ALIVE
            throw(LavaError("pack_arg!",
                "packing buffer with state=$st (not ALIVE) at addr=0x$(string(buf.address, base=16, pad=16))",
                "buffer was freed but its BDA is being packed into an arg slab — use-after-free"))
        end
    end
    unsafe_store!(Ptr{UInt64}(mapped_ptr + offset), buf.address)
    # A single store at the layout slot, so a move can rewrite exactly there.
    recpatch!(batch, buf.address, mapped_ptr + offset)
    return inline_offset
end

# A device array is the one by-value aggregate whose bytes a move must be able
# to rewrite: the struct is inlined past `base_size`, and its first field IS
# the resource's device address. Same stores as the generic branch, plus noting
# where the pointer landed — see `recpatch!`. (An aggregate that CONTAINS a
# device array several fields deep is not here on purpose: those are scene
# structures, whose moves invalidate the plan rather than patch it.)
@inline function pack_arg!(x::LavaDeviceArray{T,N},
                           mapped_ptr::Ptr{UInt8}, arg_buf_bda::UInt64,
                           offset::Int, byval_size::Int, inline_offset::Int,
                           batch::O) where {T,N,O<:Closed}
    inline_offset = (inline_offset + 7) & ~7
    ccall(:memset, Ptr{Cvoid}, (Ptr{Cvoid}, Cint, Csize_t),
          mapped_ptr + inline_offset, 0, byval_size)
    unsafe_store!(Ptr{LavaDeviceArray{T,N}}(mapped_ptr + inline_offset), x)
    unsafe_store!(Ptr{UInt64}(mapped_ptr + offset),
                  arg_buf_bda + UInt64(inline_offset))
    recpatch!(batch, UInt64(x.ptr), mapped_ptr + inline_offset)
    return inline_offset + byval_size
end

"""
Which positions in an argument tuple occupy a push-constant slot, in order.

The one statement of the rule. Zero-size arguments take no bytes, and
type-valued ones
(`T = Float32` captured by an `@kernel`) are ghost in GPUCompiler's view even
though `typeof(Float32) === DataType` has nonzero `sizeof` — packing into a slot
that does not exist is a segfault.

Returns `all_args` indices; the position in the returned vector is the LAYOUT
index, which is what `offsets` and `byval_sizes` are indexed by.
"""
function slotpositions(types)
    non_ghost = Int[]
    for (i, Ti) in enumerate(types)
        sizeof(Ti) == 0 && continue
        Ti <: Type && continue
        push!(non_ghost, i)
    end
    return non_ghost
end

"""
Whether `pack_arg!` writes this type at its own offset and nowhere else.

A by-value aggregate goes into the inline area past `base_size`, at a position
that depends on every by-value argument before it, so it cannot be rewritten on
its own. Everything else — primitives, pointers, buffer addresses — is a single
store at `offsets[layout_i]`, which is what makes a per-argument update possible
at all. Read off the branch in `pack_arg!` rather than restated: that method
tests exactly `isbitstype(T) && !isprimitivetype(T)`.
"""
standalone_slot(::Type{T}) where {T} = !(isbitstype(T) && !isprimitivetype(T))

"""
    pack_args_direct!(owner, mapped_ptr, arg_buf_bda, offsets, base_size, byval_sizes, all_args)

Write kernel arguments directly into mapped GPU memory via per-type
`pack_arg!` dispatch.  Zero heap allocations for the arg packing itself.

`owner` is what the buffer-typed leaves are pinned into — the [`OneShot`](@ref)
for work submitted once, the [`Recording`](@ref) for work a plan submits every
run. It was `bq`, and the body read the queue's open batch: a recording's
arguments were therefore pinned into whatever batch happened to be open while it
was being written, and that batch's completion released them.
"""
@generated function pack_args_direct!(batch::Closed,
                                        mapped_ptr::Ptr{UInt8}, arg_buf_bda::UInt64,
                                        offsets::Vector{Int}, base_size::Int,
                                        byval_sizes::Vector{Int},
                                        all_args::T) where {T <: Tuple}
    non_ghost = slotpositions(T.parameters)
    exprs = Expr[]
    for (layout_i, arg_i) in enumerate(non_ghost)
        push!(exprs, :(inline_offset = pack_arg!(
            all_args[$arg_i], mapped_ptr, arg_buf_bda,
            @inbounds(offsets[$layout_i]),
            @inbounds(byval_sizes[$layout_i]),
            inline_offset, batch)))
    end
    quote
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
    config = lava_compiler_config(; workgroup_size, enable_ray_query, features = ctx.features)
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

# ── What a recording reads while it runs ──
#
# A dispatch reaches its arguments through a device address baked into the
# command buffer as a push constant, so those bytes have to be host-writable,
# device-addressable, and untouchable by anyone else for as long as the command
# that names them can still run. That is a `Region` of the [`Unified`](@ref)
# arena and an OWNER, and both already exist.
#
# What was here instead: two bump allocators on the queue and a third inside
# every capture, with a high-water mark, a handout counter and a rewind test
# between them, all working out when bytes were safe to reuse. The fence already
# knew. A modelled plan does not come through here at all — it lays its arguments
# and its indirect commands out at compile and owns the region they live in.

const ARG_ALIGN = 256                     # BDA alignment for sub-allocations
"""A block of the [`Unified`](@ref) arena. Small, because BAR memory is scarce on
a device without resizable BAR and a recording's arguments are kilobytes."""
const UNIFIED_BLOCK_SIZE = 4 * 1024 * 1024
"""A block of the [`Readback`](@ref) arena. A download of a full frame is tens of
megabytes and `acquire!` sizes a block to the larger of the request and this, so
this only decides how many SMALL downloads share one allocation."""
const READBACK_BLOCK_SIZE = 16 * 1024 * 1024

"""A slice of the unified arena a recording writes its arguments into."""
struct ArgBufferAlloc
    address::UInt64           # BDA of this sub-allocation
    mapped_ptr::Ptr{UInt8}    # CPU-writable pointer
    size::Int
end

"""
    scratch!(owner, nbytes) -> Region

`nbytes` of the unified arena for the commands `owner` holds, owned BY it —
which is what decides when the bytes go back. A [`OneShot`](@ref) gives them up
when the submission that carried it has passed; a [`Recording`](@ref) when it
is released, because until then it may be submitted again and read them again.

`own!(bq, r)` stood in front of this and asked `bq.capturing` which of the two
was the owner. The owner is now the argument, so there is nothing to ask.

One `acquire!` per launch, where the two allocators this replaces did a
bump-pointer add. That is deliberate and it is the cheap half of the trade: the
only caller is the unmodelled launch path, which looks up a compiled kernel,
adapts an argument tree and packs it on every call — and what the bump pointer
bought was five fields of state that could rewind under a recording still holding
the address. A modelled plan does not come through here at all.
"""
@inline function scratch!(owner::O, nbytes::Integer) where {O<:Closed}
    bq = queueof(owner)
    @assert Threads.threadid() == bq.thread  "VulkanBatchQueue is single-writer; cross-thread scratch alloc forbidden"
    dev = lavadevice(ctxof(bq))
    r = acquire!(pool(dev), dev, Unified(), nothing, max(Int(nbytes), 16);
                 align = ARG_ALIGN, blocksize = UNIFIED_BLOCK_SIZE)
    push!(owner.regions, r)
    return r
end

"""Bytes for one launch's arguments, as the address the push constant carries and
the pointer the packer writes through."""
@inline function get_arg_buffer(owner::O, nbytes::Integer) where {O<:Closed}
    r = scratch!(owner, nbytes)
    blk = memoryof(r)::BufferBlock
    off = offset(r)
    return ArgBufferAlloc(blk.address + UInt64(off),
                          (blk.ref[]::VkManagedBuffer).mapped_ptr + off,
                          length(r))
end

"""
    indirect_command!(owner) -> LavaArray{UInt32,1}

One `VkDispatchIndirectCommand` for an unmodelled indirect launch: three
`UInt32`s a prepare kernel writes and the command processor reads.

`get_indirect_buffer` was this, from a slab ring on the queue that was rewound
whenever the queue drained — while a recording holds the address of its
command for as long as it can be replayed. A modelled plan has no need of either:
its commands are laid out at compile, in its own slot, beside its arguments.
"""
function indirect_command!(owner::O) where {O<:Closed}
    r = scratch!(owner, INDIRECT_STRIDE)
    blk = memoryof(r)::BufferBlock
    return LavaArray{UInt32,1}(copy(blk.ref), (3,); offset = offset(r))
end
