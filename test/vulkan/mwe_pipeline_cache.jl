# mwe_pipeline_cache.jl — focused reproducer for the VkPipelineCache load/save path.
#
# Exercises just the pieces I added in commit 3b0759c, without any RayMakie /
# Hikari / Crown moving parts:
#
#   1. Initialize Lava with no prior cache file on disk.
#   2. Compile + dispatch a trivial compute kernel — populates the in-memory
#      VkPipelineCache.
#   3. save_pipeline_cache! to disk. Verify the file exists, has plausible
#      header (Vulkan VkPipelineCacheHeaderVersion = 1, vendorID, deviceID,
#      driverVersion).
#   4. reset_device! — tears down the device, save fires, then a fresh
#      device gets created which loads from the on-disk blob.
#   5. Compile + dispatch the same kernel against the fresh device. If
#      anything in the load/save path is unsafe (GC lifetime, double-free,
#      pointer-to-freed-buffer, racing atexit), this is where it will crash
#      reliably without any of the rest of the stack to confuse the picture.
#   6. Loop a few times to catch state corruption that only shows up on the
#      Nth restart.
#
# Run:  julia --project=. dev/Lava/test/mwe_pipeline_cache.jl

using Lava, Test
import Vulkan

# ── Setup ─────────────────────────────────────────────────────────────────
# Start from a clean cache file each run so we don't carry poison across
# unrelated sessions.
function clean_pipeline_cache_file!()
    ctx = Mantle.vk_context()
    path = Mantle.lava_pipeline_cache_path(ctx.device_name, ctx.driver_version)
    isfile(path) && rm(path; force=true)
    return path
end

# Tiny compute kernel — guaranteed to produce ONE compute pipeline, no RT,
# no ray-query, no Hikari/Makie. Touches just the pipeline_cache parameter
# threading we added.
function trivial_kernel!(buf::LavaDeviceArray{Float32, 1})
    i = Int(Lava.lava_global_invocation_id_x()) + 1
    if i <= length(buf)
        @inbounds buf[i] = Float32(i)
    end
    return nothing
end

# ── Step 1: clean start ───────────────────────────────────────────────────
path = clean_pipeline_cache_file!()
println("\n[step 1] cleaned cache file: $path")
ctx = Mantle.vk_context()
println("         pipeline_cache handle: $(typeof(ctx.pipeline_cache))")
println("         file exists post-init: $(isfile(path))  (expected: false)")

# ── Step 2: compile + dispatch ────────────────────────────────────────────
println("\n[step 2] compile + dispatch trivial kernel")
N = 64
buf = Mantle.LavaArray(zeros(Float32, N))
bq = Mantle.defaultbackend().dispatch_bq
Mantle.lava_launch!(bq, trivial_kernel!, buf; ndrange=N, workgroup_size=(64, 1, 1))
Mantle.flush!(bq)
result = Array(buf)
@assert result == Float32.(1:N) "kernel produced wrong values: $(result[1:8])"
println("         kernel produced expected values ✓")

# ── Step 3: save ──────────────────────────────────────────────────────────
println("\n[step 3] save pipeline cache to disk")
Mantle.save_pipeline_cache!(ctx)
@assert isfile(path) "save_pipeline_cache! did not write a file"
sz = filesize(path)
println("         file size: $sz bytes")
@assert sz >= 32 "cache file too small to contain even a Vulkan header"

# Validate the on-disk header.  VkPipelineCacheHeaderVersionOne:
#   uint32 headerSize       = 32
#   uint32 headerVersion    = 1
#   uint32 vendorID
#   uint32 deviceID
#   uint8[16] pipelineCacheUUID
let bytes = read(path)
    @assert length(bytes) >= 32 "truncated cache header"
    header_size    = reinterpret(UInt32, bytes[1:4])[1]
    header_version = reinterpret(UInt32, bytes[5:8])[1]
    vendor_id      = reinterpret(UInt32, bytes[9:12])[1]
    device_id      = reinterpret(UInt32, bytes[13:16])[1]
    println("         header_size=$header_size  version=$header_version  vendorID=0x$(string(vendor_id, base=16))  deviceID=0x$(string(device_id, base=16))")
    @assert header_size == 32 "unexpected header size: $header_size"
    @assert header_version == 1 "unexpected pipeline cache header version: $header_version"
end

# ── Step 4: reset device → loads from disk ────────────────────────────────
println("\n[step 4] reset_device! — should save again then re-load from disk")
Mantle.reset_device!()
ctx = Mantle.vk_context()
@assert isfile(path) "reset deleted the cache file"
sz2 = filesize(path)
println("         post-reset file size: $sz2 bytes  (was: $sz)")

# ── Step 5: compile + dispatch on the reloaded device ─────────────────────
println("\n[step 5] re-dispatch trivial kernel on fresh device w/ loaded cache")
buf2 = Mantle.LavaArray(zeros(Float32, N))
bq2 = Mantle.defaultbackend().dispatch_bq
Mantle.lava_launch!(bq2, trivial_kernel!, buf2; ndrange=N, workgroup_size=(64, 1, 1))
Mantle.flush!(bq2)
result2 = Array(buf2)
@assert result2 == Float32.(1:N) "kernel after reload produced wrong values: $(result2[1:8])"
println("         kernel still produces correct values ✓")

# ── Step 6: loop ──────────────────────────────────────────────────────────
println("\n[step 6] reset+dispatch loop (5 iterations) — exercise restart path")
for iter in 1:5
    Mantle.reset_device!()
    ctx = Mantle.vk_context()
    buf_i = Mantle.LavaArray(zeros(Float32, N))
    bq_i = Mantle.defaultbackend().dispatch_bq
    Mantle.lava_launch!(bq_i, trivial_kernel!, buf_i; ndrange=N, workgroup_size=(64, 1, 1))
    Mantle.flush!(bq_i)
    r = Array(buf_i)
    @assert r == Float32.(1:N) "iter $iter produced wrong values"
    sz_i = filesize(path)
    println("         iter $iter: ok  cache_size=$sz_i")
end

println("\nAll cache load/save checks passed ✓")
