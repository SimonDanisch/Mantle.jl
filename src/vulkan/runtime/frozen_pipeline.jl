# The frozen cache, DEVICE half.
#
# Split from `compiler/frozen_spirv.jl`, which holds the keys, the paths, the
# eligibility rule and the ray-tracing entries. Everything here needs a
# `VkContext`, and needs it for a real reason rather than for a log line:
#
#   * `frozen_load`/`frozen_store` memoise into `ctx.caches.frozen_mem`, whose
#     values are `LavaLinkedKernel`s — they own a `VkPipeline`, so the memo is
#     per device and cannot be module-level. (`FROZEN_RT_MEM` on the other side
#     holds SPIR-V and no handle, which is why that one is shared.)
#   * `frozen_pipeline_cache`/`frozen_store_bin` are level 2 of the cache: SPIR-V
#     is portable, the ISA a driver derives from it is not. The blob is validated
#     against vendor ID, device ID and cache UUID before `vkCreatePipelineCache`
#     sees a byte of it.
#   * `frozen_prune!` empties the per-device memo along with the directory.
#
# The compiler calls none of these. It calls `frozen_rt_load`/`frozen_rt_store`,
# which take no context at all — that asymmetry is the split.

"""
    frozen_prune_once!(ctx)

Bound the frozen cache to `frozen_max_bytes()`, at most once per session.

Budgeted in BYTES, not entries: entries here average ~2 MB (SPIR-V plus the
per-kernel pipeline blob) but RT stages are far larger than compute kernels, so
an entry count is a poor proxy for disk. Measured: 817 entries occupying 1.6 GB.

Deletes oldest-first down to 75 % of the budget, so it does not re-trigger every
session — the same high/low water shape cuTile's `DiskCache` uses.
"""
function frozen_prune_once!(ctx::VkContext)
    FROZEN_PRUNED[] && return nothing
    FROZEN_PRUNED[] = true
    dir = frozen_cache_dir()
    isdir(dir) || return nothing
    budget = frozen_max_bytes()
    budget <= 0 && return nothing                    # explicit opt-out
    entries = filter(f -> endswith(f, ".spirv"), readdir(dir))
    isempty(entries) && return nothing
    # Size of an entry is its .spirv plus the .bin beside it, if any.
    entry_bytes(f) = filesize(joinpath(dir, f)) +
                     filesize(joinpath(dir, replace(f, r"\.spirv$" => ".bin")))
    total = sum(entry_bytes, entries)
    total <= budget && return nothing
    # Newest first; keep as many as fit in 75 % of the budget.
    sort!(entries; by = f -> mtime(joinpath(dir, f)), rev = true)
    target = (budget * 3) ÷ 4
    acc = 0
    keep = 0
    for f in entries
        acc += entry_bytes(f)
        acc > target && break
        keep += 1
    end
    removed = frozen_prune!(; keep, ctx)
    @debug "Lava: pruned frozen kernel cache" bytes_before=total budget keep removed
    return nothing
end

"""
    frozen_pipeline_cache(ctx, key) -> VkPipelineCache | nothing

A `VkPipelineCache` seeded with this kernel's own `<key>.bin`, or `nothing` when
there is none for this device.

This is level 2, per kernel: SPIR-V is portable, the ISA the driver derives from
it is not. The blob is validated against vendor ID, device ID and cache UUID
before `vkCreatePipelineCache` sees a byte of it — a foreign or truncated blob
is a documented way to crash *inside* the driver, and a crash there takes the
process with it before any Julia `try` can run. After a driver update the UUID
changes, every `.bin` stops matching, and the kernel is rebuilt from its
`.spirv`, which is exactly the fallback the two levels exist for.
"""
function frozen_pipeline_cache(ctx::VkContext, key::AbstractString)
    path = frozen_binpath(key)
    isfile(path) || return nothing
    bytes = try
        read(path)
    catch ex
        cache_io_error(ex) || rethrow()
        @warn "Lava: frozen pipeline blob unreadable; the driver will recompile its ISA" path exception = ex maxlog = 1
        return nothing
    end
    pipeline_cache_compatible(bytes, ctx.physical_device) || return nothing
    try
        return GC.@preserve bytes begin
            ci = VK.PipelineCacheCreateInfo(Ptr{Cvoid}(pointer(bytes));
                                                initial_data_size = UInt64(length(bytes)))
            VK.PipelineCache(ctx.device, ci)
        end
    catch ex
        # Passed the header check and the driver still refused it: that is a
        # failure of the cache MEDIUM, so keep the session and drop the blob so
        # the next run does not retry it.
        #
        # Only a driver refusal. This used to be a bare `catch`, which is the
        # very thing `cache_io_error`'s docstring warns about applied to the
        # Vulkan call instead of the deserializer: an OOM or a bug building the
        # CreateInfo would be laundered into "blob rejected", the blob deleted,
        # and every later run would silently pay a rebuild.
        ex isa VK.VulkanError || rethrow()
        @warn "Lava: frozen pipeline blob rejected by the driver; rebuilding from SPIR-V" path exception = ex maxlog = 1
        rm(path; force = true)
        return nothing
    end
end

"""Snapshot `pcache` to `<key>.bin`, atomically. Best effort."""
function frozen_store_bin(ctx::VkContext, key::AbstractString, pcache)
    try
        size, ptr = unwrap(VK.get_pipeline_cache_data(ctx.device, pcache))
        try
            size == 0 && return nothing
            bytes = unsafe_wrap(Array, Ptr{UInt8}(ptr), Int(size); own = false)
            dir = frozen_cache_dir(); mkpath(dir)
            tmppath, io = mktemp(dir; cleanup = false)
            write(io, bytes); close(io)
            mv(tmppath, frozen_binpath(key); force = true)
        finally
            Libc.free(ptr)
        end
    catch ex
        cache_io_error(ex) || rethrow()
        @warn "Lava: frozen pipeline blob store failed; level-2 cache will stay cold" key exception = ex maxlog = 1
    end
    return nothing
end

"""
    frozen_load(ctx, f, tt, workgroup_size) -> LavaLinkedKernel | nothing

The linked kernel for this signature, from memory or from disk, without ever
asking Julia to infer anything.

`@noinline` for the reason `get_compiled_kernel_and_pipeline` carries it: the
`@nospecialize` alone does not stop inference from making a per-kernel copy of
this to inline into a per-kernel caller, and on SAM 2's encoder that was 1 002
copies of a dictionary lookup.
"""
@noinline function frozen_load(ctx::VkContext, @nospecialize(f), @nospecialize(tt), workgroup_size)
    isempty(FROZEN_VERSION[]) && return nothing
    frozen_eligible(f) || return nothing
    memkey = (typeof(f), tt, workgroup_size)
    hit = get(ctx.caches.frozen_mem, memkey, nothing)
    hit === nothing || return hit
    key = frozen_key(f, tt, workgroup_size)
    path = frozen_path(key)
    if !isfile(path)
        frozen_logging() && println("frozen MISS: ", key, " || ", typestring(tt))
        return nothing
    end
    compiled = try
        open(Serialization.deserialize, path)::LavaGPUKernel
    catch ex
        # A damaged entry costs a recompile, never the session — but it says so.
        cache_io_error(ex) || rethrow()
        @warn "Lava: frozen cache entry unreadable; recompiling this kernel" path exception = ex maxlog = 1
        return nothing
    end
    # Level 2: hand the pipeline this kernel's own driver blob when there is a
    # matching one, so the driver reuses its ISA instead of recompiling.
    linked = link_kernel(ctx, compiled; pipeline_cache = frozen_pipeline_cache(ctx, key))
    ctx.caches.frozen_mem[memkey] = linked
    FROZEN_HITS[] += 1
    return linked
end

"""
    frozen_store(ctx, f, tt, workgroup_size, compiled)

Write a compiled kernel under its frozen key. Only while recording.

`@noinline` for the same reason as [`frozen_load`](@ref) — and this one is the
starker case, since outside a recording its whole body is one early return that
was being inferred a thousand times.
"""
@noinline function frozen_store(ctx::VkContext, @nospecialize(f), @nospecialize(tt), workgroup_size,
                      compiled::LavaGPUKernel)
    (FROZEN_RECORDING[] && !isempty(FROZEN_VERSION[])) || return nothing
    frozen_eligible(f) || return nothing
    frozen_prune_once!(ctx)
    dir = frozen_cache_dir()
    mkpath(dir)
    key = frozen_key(f, tt, workgroup_size)
    path = frozen_path(key)
    # The LLVM IR string is session-specific and large; the frozen entry is the
    # SPIR-V and what it takes to build a pipeline from it, nothing else.
    entry = LavaGPUKernel(compiled.spirv_bytes, compiled.entry_name,
                          compiled.workgroup_size, compiled.push_info, "",
                          compiled.enable_ray_query, compiled.source_name)
    try
        tmppath, io = mktemp(dir; cleanup = false)
        Serialization.serialize(io, entry)
        close(io)
        mv(tmppath, path; force = true)          # atomic: no half-written entry
        FROZEN_STORES[] += 1
        # …and the driver's ISA for it, from a cache holding only this kernel.
        # Recording builds the pipeline through `frozen_link_recording`, which
        # leaves the per-kernel cache in `ctx.caches.frozen_last_pcache`.
        let pc = ctx.caches.frozen_last_pcache
            pc === nothing || frozen_store_bin(ctx, key, pc)
        end
        frozen_logging() &&
            println("frozen STORE: ", basename(path)[1:end-6], " || ", typestring(tt))
    catch ex
        cache_io_error(ex) || rethrow()
        @warn "Lava: frozen cache store failed; this kernel will recompile next session" path exception = ex maxlog = 1
    end
    return nothing
end

"""
    frozen_prune!(; keep = 2000, ctx = vk_context()) -> Int

Delete all but the `keep` most recently modified frozen entries. Returns how many
were removed.

**Orphans are the normal outcome of editing kernels, not a malfunction.** An
entry is keyed by signature, workgroup, version and the build id of the defining
module (see [`frozen_key`](@ref)); recompiling that module makes every one of its
entries unreachable but does not delete them. A development session that edits
kernels a few times leaves hundreds behind — one measured session accumulated
**6066**.

They are harmless except for disk, because an unreachable entry is never loaded.
This exists so the directory can be bounded when that matters, and is deliberately
**not** called automatically: deleting a cache another package is about to hit
turns a hit into a recompile, and that choice belongs to whoever knows the
machine.

`frozen_clear!(; version)` is the other tool — it removes one generation
outright, which is what a deliberate `KERNELS_VERSION` bump obsoletes.
"""
function frozen_prune!(; keep::Int = 2000, ctx::VkContext = vk_context())
    empty!(ctx.caches.frozen_mem)
    empty!(FROZEN_RT_MEM)
    dir = frozen_cache_dir()
    isdir(dir) || return 0
    spirv = filter(f -> endswith(f, ".spirv"), readdir(dir))
    length(spirv) <= keep && return 0
    # Oldest first, by the entry's own mtime.
    sort!(spirv; by = f -> mtime(joinpath(dir, f)))
    n = 0
    for f in spirv[1:(length(spirv) - keep)]
        rm(joinpath(dir, f); force = true)
        rm(joinpath(dir, replace(f, r"\.spirv$" => ".bin")); force = true)
        n += 1
    end
    # Sweep `.bin` blobs whose `.spirv` is already gone. `frozen_clear!` used to
    # leave these behind, and nothing else names them.
    live = Set(readdir(dir))
    for f in readdir(dir)
        endswith(f, ".bin") || continue
        replace(f, r"\.bin$" => ".spirv") in live && continue
        rm(joinpath(dir, f); force = true)
        n += 1
    end
    return n
end

"""
    frozen_clear!(; version = FROZEN_VERSION[])

Delete every on-disk entry for `version`, and the session's memory of them.
The one supported way to invalidate, short of bumping the version.
"""
function frozen_clear!(; version::AbstractString = FROZEN_VERSION[],
                        ctx::VkContext = vk_context())
    # The per-device memo is the only part that needs a context: its values are
    # `LavaLinkedKernel`s and they own a `VkPipeline`. Everything else — the RT
    # memo, the files — is the compiler's, and `Lava.frozen_rt_clear!` does it.
    empty!(ctx.caches.frozen_mem)
    return Lava.frozen_rt_clear!(; version)
end
