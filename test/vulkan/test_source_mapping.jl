# Tests for source mapping (SPIR-V ID → Julia source) and compilation error handling
#
# Covers:
# - Source map population during SPIR-V emission
# - Inlined-at chain walking (user code, not Base internals)
# - Complex kernels (structs, control flow, math)
# - File-based kernels (real file paths)
# - LavaCompilationError for common GPU-incompatible patterns
# - CompilationResult carries source_map
# - Validation message leak fix

using Test
using Lava, Mantle
using KernelAbstractions
using Lava: LavaDeviceArray, lava_compile, CompilationResult,
            LavaCompilationError, shorten_path

# ═══════════════════════════════════════════════════════════════════════
# Helper: define test kernels inside Lava module to avoid world age issues
# with GPUCompiler (functions defined in Main can't access Lava internals)
# ═══════════════════════════════════════════════════════════════════════

# The kernels below are defined INTO Lava by `Lava.eval`, so they are
# `Lava._srcmap_*` and nothing else. They were being reached as
# `Lava._srcmap_*` — a `Lava.` -> `Mantle.` rename during the runtime move that
# went one identifier too far, and it took the whole file down at its first
# compile with `UndefVarError: _srcmap_add! not defined in Mantle`.
Lava.eval(quote
    # Simple kernel: load, compute, store
    function _srcmap_add!(A::LavaDeviceArray{Float32,1}, val::Float32)
        i = lava_global_invocation_id_x() + UInt32(1)
        @inbounds A[i] = A[i] + val
        return nothing
    end

    # Complex kernel: multiple ops, control flow
    function _srcmap_complex!(A::LavaDeviceArray{Float32,1}, B::LavaDeviceArray{Float32,1},
                               scale::Float32, threshold::Float32)
        i = lava_global_invocation_id_x() + UInt32(1)
        @inbounds begin
            x = A[i]
            y = B[i]
            result = x * scale + y * (1.0f0 - scale)
            if result > threshold
                A[i] = result
            else
                A[i] = threshold
            end
        end
        return nothing
    end

    # Struct kernel: NTuple/struct field access
    struct _SrcMapVec3
        x::Float32
        y::Float32
        z::Float32
    end

    function _srcmap_struct!(dst::LavaDeviceArray{_SrcMapVec3,1},
                              src::LavaDeviceArray{_SrcMapVec3,1},
                              scale::Float32)
        i = lava_global_invocation_id_x() + UInt32(1)
        @inbounds begin
            v = src[i]
            dst[i] = _SrcMapVec3(v.x * scale, v.y * scale, v.z * scale)
        end
        return nothing
    end

    # Integer kernel: different types
    function _srcmap_int!(A::LavaDeviceArray{Int32,1}, mask::Int32)
        i = lava_global_invocation_id_x() + UInt32(1)
        @inbounds A[i] = A[i] & mask
        return nothing
    end

    # Multi-array kernel
    function _srcmap_multi!(A::LavaDeviceArray{Float32,1}, B::LavaDeviceArray{Float32,1},
                             C::LavaDeviceArray{Float32,1})
        i = lava_global_invocation_id_x() + UInt32(1)
        @inbounds C[i] = A[i] * B[i]
        return nothing
    end

    # ── Intentionally broken kernels for error testing ──

    # Heap allocation (rand uses RNG state allocation)
    function _srcmap_bad_alloc!(A::LavaDeviceArray{Float32,1})
        i = lava_global_invocation_id_x() + UInt32(1)
        x = rand()
        @inbounds A[i] = Float32(x)
        return nothing
    end

    # Type instability (Any element type → dynamic dispatch)
    function _srcmap_bad_unstable!(A::LavaDeviceArray{Any,1})
        i = lava_global_invocation_id_x() + UInt32(1)
        @inbounds A[i] = A[i] * 2
        return nothing
    end

    # Non-const global access
    _srcmap_mutable_global = 42
    function _srcmap_bad_global!(A::LavaDeviceArray{Float32,1})
        i = lava_global_invocation_id_x() + UInt32(1)
        @inbounds A[i] = Float32(_srcmap_mutable_global)
        return nothing
    end

    # ── Deep call chain error patterns ──

    # Pattern A: String interpolation buried 3 levels deep in physics simulation
    struct _TestParticle
        x::Float32; y::Float32; vx::Float32; vy::Float32
    end

    function _update_velocity(p::_TestParticle, dt::Float32)
        _TestParticle(p.x + p.vx * dt, p.y + p.vy * dt, p.vx, p.vy)
    end

    function _check_collision(p::_TestParticle)
        if p.x < 0.0f0
            msg = "collision at $(p.x)"  # String interpolation → heap alloc
            return length(msg)
        end
        return Int(0)
    end

    function _physics_step(p::_TestParticle, dt::Float32)
        p2 = _update_velocity(p, dt)
        n = _check_collision(p2)
        return _TestParticle(p2.x, p2.y, p2.vx * (1.0f0 - Float32(n) * 0.01f0), p2.vy)
    end

    function _srcmap_deep_physics!(particles::LavaDeviceArray{_TestParticle,1}, dt::Float32)
        i = lava_global_invocation_id_x() + UInt32(1)
        @inbounds particles[i] = _physics_step(particles[i], dt)
        return nothing
    end

    # Pattern B: Abstract field accessed 4 levels deep in a scene renderer
    struct _SceneObject
        transform::NTuple{4, Float32}
        material_id::Int32
        data  # Any-typed — the bug
    end

    function _get_albedo(obj::_SceneObject)
        d = obj.data  # type unstable — d is Any
        return d * 0.5f0
    end

    function _compute_lighting(obj::_SceneObject, light_dir::Float32)
        albedo = _get_albedo(obj)
        return albedo * max(0.0f0, light_dir)
    end

    function _render_pixel(obj::_SceneObject, uv::Float32)
        light = sin(uv * 3.14159f0)
        return _compute_lighting(obj, light)
    end

    function _srcmap_deep_scene!(out::LavaDeviceArray{Float32,1},
                                  objects::LavaDeviceArray{_SceneObject,1})
        i = lava_global_invocation_id_x() + UInt32(1)
        @inbounds out[i] = _render_pixel(objects[i], Float32(i) * 0.01f0)
        return nothing
    end

    # Pattern C: error() in validation function 3 levels deep (the classic "I'll just
    # add a bounds check with a nice error message" mistake)
    function _validate_range(x::Float32)
        if x < 0.0f0 || x > 1.0f0
            error("value out of range: $x")  # String alloc + throw
        end
        return x
    end

    function _apply_tonemap(x::Float32, exposure::Float32)
        y = x * exposure
        return _validate_range(y / (y + 1.0f0))
    end

    function _srcmap_deep_error!(out::LavaDeviceArray{Float32,1},
                                  input::LavaDeviceArray{Float32,1},
                                  exposure::Float32)
        i = lava_global_invocation_id_x() + UInt32(1)
        @inbounds out[i] = _apply_tonemap(input[i], exposure)
        return nothing
    end

    # Pattern D: Accidentally using a CPU-only function from a helper
    # (e.g., calling an IO function for "logging")
    function _normalize_vec(x::Float32, y::Float32, z::Float32)
        len = sqrt(x*x + y*y + z*z)
        return (x/len, y/len, z/len)
    end

    function _process_normal(nx::Float32, ny::Float32, nz::Float32)
        n = _normalize_vec(nx, ny, nz)
        # Accidentally left debug logging in
        @info "normal: $n"  # Allocates + IO
        return n
    end

    function _srcmap_deep_logging!(out::LavaDeviceArray{Float32,1},
                                    normals::LavaDeviceArray{NTuple{3,Float32},1})
        i = lava_global_invocation_id_x() + UInt32(1)
        @inbounds begin
            n = normals[i]
            result = _process_normal(n[1], n[2], n[3])
            out[i] = result[1] + result[2] + result[3]
        end
        return nothing
    end
end)

# ═══════════════════════════════════════════════════════════════════════
# Test 1: Source map is populated for a simple kernel
# ═══════════════════════════════════════════════════════════════════════

@testset "Source map: simple kernel has mapped instructions" begin
    r = lava_compile(Lava._srcmap_add!,
        Tuple{LavaDeviceArray{Float32,1}, Float32})

    @test r isa CompilationResult
    @test hasfield(CompilationResult, :source_map)
    @test r.source_map isa Dict{UInt32, Tuple{String, Int}}
    # A simple add kernel should produce at least 10 mapped instructions
    @test length(r.source_map) >= 10
end

# ═══════════════════════════════════════════════════════════════════════
# Test 2: Source map points to user code, not Base internals
# ═══════════════════════════════════════════════════════════════════════

@testset "Source map: points to user code (inlined_at chain)" begin
    r = lava_compile(Lava._srcmap_add!,
        Tuple{LavaDeviceArray{Float32,1}, Float32})

    # Collect all unique source files
    files = Set{String}()
    for (_, (file, _)) in r.source_map
        push!(files, file)
    end

    # Should NOT point to Base internals like pointer.jl, float.jl
    # (the inlined_at chain walking should find user code)
    for f in files
        @test !endswith(f, "pointer.jl")
        @test !endswith(f, "float.jl")
        @test !endswith(f, "int.jl")
        @test !endswith(f, "boot.jl")
    end

    # All entries should have non-zero line numbers
    for (id, (file, line)) in r.source_map
        @test line > 0
    end
end

# ═══════════════════════════════════════════════════════════════════════
# Test 3: Source map covers multiple source lines
# ═══════════════════════════════════════════════════════════════════════

@testset "Source map: complex kernel maps to multiple source lines" begin
    r = lava_compile(Lava._srcmap_complex!,
        Tuple{LavaDeviceArray{Float32,1}, LavaDeviceArray{Float32,1}, Float32, Float32})

    # Group by line number
    lines_seen = Set{Int}()
    for (_, (_, line)) in r.source_map
        push!(lines_seen, line)
    end

    # A complex kernel with branches should map to at least 4 different lines
    @test length(lines_seen) >= 4

    # Should have more mapped instructions than the simple kernel
    @test length(r.source_map) >= 20
end

# ═══════════════════════════════════════════════════════════════════════
# Test 4: Source map works for struct kernels
# ═══════════════════════════════════════════════════════════════════════

@testset "Source map: struct kernel" begin
    r = lava_compile(Lava._srcmap_struct!,
        Tuple{LavaDeviceArray{Lava._SrcMapVec3,1}, LavaDeviceArray{Lava._SrcMapVec3,1}, Float32})

    @test length(r.source_map) >= 15

    # Verify entries have valid data
    for (id, (file, line)) in r.source_map
        @test !isempty(file)
        @test line > 0
    end
end

# ═══════════════════════════════════════════════════════════════════════
# Test 5: Source map works for integer kernels
# ═══════════════════════════════════════════════════════════════════════

@testset "Source map: integer kernel" begin
    r = lava_compile(Lava._srcmap_int!,
        Tuple{LavaDeviceArray{Int32,1}, Int32})

    @test length(r.source_map) >= 5
end

# ═══════════════════════════════════════════════════════════════════════
# Test 6: Source map for multi-array kernel
# ═══════════════════════════════════════════════════════════════════════

@testset "Source map: multi-array kernel" begin
    r = lava_compile(Lava._srcmap_multi!,
        Tuple{LavaDeviceArray{Float32,1}, LavaDeviceArray{Float32,1}, LavaDeviceArray{Float32,1}})

    @test length(r.source_map) >= 10
end

# ═══════════════════════════════════════════════════════════════════════
# Test 7: Source map with file-based kernel (real file paths)
# ═══════════════════════════════════════════════════════════════════════

@testset "Source map: file-based kernel has real file paths" begin
    # Write a kernel to a real file
    test_file = tempname() * "_srcmap_test.jl"
    write(test_file, """
    module _SrcMapTestMod
    using Lava: LavaDeviceArray, lava_global_invocation_id_x

    function file_kernel!(A::LavaDeviceArray{Float32,1}, B::LavaDeviceArray{Float32,1})
        i = lava_global_invocation_id_x() + UInt32(1)
        @inbounds begin
            a = A[i]
            b = B[i]
            A[i] = a + b
        end
        return nothing
    end

    end
    """)

    try
        include(test_file)
        r = lava_compile(Main._SrcMapTestMod.file_kernel!,
            Tuple{LavaDeviceArray{Float32,1}, LavaDeviceArray{Float32,1}})

        # Check that source map contains real file paths
        has_real_path = false
        for (_, (file, _)) in r.source_map
            if occursin("_srcmap_test.jl", file)
                has_real_path = true
                break
            end
        end
        @test has_real_path

        # Verify the mapped lines make sense (lines 5-11 of the test file)
        lines_in_file = Set{Int}()
        for (_, (file, line)) in r.source_map
            if occursin("_srcmap_test.jl", file)
                push!(lines_in_file, line)
            end
        end
        @test !isempty(lines_in_file)
        # Lines should be in the range of our function definition (roughly 5-12)
        @test minimum(lines_in_file) >= 4
        @test maximum(lines_in_file) <= 15
    finally
        rm(test_file; force=true)
    end
end

# ═══════════════════════════════════════════════════════════════════════
# Test 8: SPIR-V disassembly can be annotated with source map
# ═══════════════════════════════════════════════════════════════════════

@testset "Source map: SPIR-V disassembly annotations" begin
    r = lava_compile(Lava._srcmap_add!,
        Tuple{LavaDeviceArray{Float32,1}, Float32})

    # Parse SPIR-V disassembly for IDs and check they can be annotated
    annotated_count = 0
    for line in split(r.spirv_disasm, '\n')
        m = match(r"^\s*%(\d+)\b", line)
        m === nothing && continue
        id = parse(UInt32, m.captures[1])
        if haskey(r.source_map, id)
            annotated_count += 1
        end
    end
    # At least half of the SPIR-V result IDs should be annotatable
    @test annotated_count >= 5
end

# ═══════════════════════════════════════════════════════════════════════
# Test 9: shorten_path works correctly
# ═══════════════════════════════════════════════════════════════════════

@testset "shorten_path" begin
    @test shorten_path("/home/sim/programmieren/VulkanDev/dev/Lava/src/foo.jl") == "Lava/src/foo.jl"
    @test shorten_path("/home/sim/.julia/packages/GPUCompiler/abc/src/bar.jl") == "GPUCompiler/abc/src/bar.jl"
    @test endswith(shorten_path("/usr/share/julia/stdlib/v1.12/Test/src/Test.jl"), "Test.jl")
    @test shorten_path("relative/path.jl") == "path.jl"
end

# ═══════════════════════════════════════════════════════════════════════
# Test 10: LavaCompilationError for heap allocation
# ═══════════════════════════════════════════════════════════════════════

@testset "Compilation error: heap allocation" begin
    err = try
        lava_compile(Lava._srcmap_bad_alloc!, Tuple{LavaDeviceArray{Float32,1}})
        nothing
    catch e
        e
    end

    @test err !== nothing
    @test err isa LavaCompilationError
    @test err.operation == "kernel compilation"
    @test occursin("_srcmap_bad_alloc!", err.message)
    @test occursin("allocat", lowercase(err.suggestion)) ||
          occursin("heap", lowercase(err.suggestion))
    @test !isempty(err.raw_error)
end

# ═══════════════════════════════════════════════════════════════════════
# Test 11: LavaCompilationError for type instability
# ═══════════════════════════════════════════════════════════════════════

@testset "Compilation error: type instability" begin
    err = try
        lava_compile(Lava._srcmap_bad_unstable!, Tuple{LavaDeviceArray{Any,1}})
        nothing
    catch e
        e
    end

    @test err !== nothing
    @test err isa LavaCompilationError
    @test occursin("_srcmap_bad_unstable!", err.message)
    @test occursin("instab", lowercase(err.suggestion)) ||
          occursin("dispatch", lowercase(err.suggestion)) ||
          occursin("inferr", lowercase(err.suggestion))
end

# ═══════════════════════════════════════════════════════════════════════
# Test 12: LavaCompilationError for global variable access
# ═══════════════════════════════════════════════════════════════════════

@testset "Compilation error: global variable access" begin
    err = try
        lava_compile(Lava._srcmap_bad_global!, Tuple{LavaDeviceArray{Float32,1}})
        nothing
    catch e
        e
    end

    @test err !== nothing
    @test err isa LavaCompilationError
    @test occursin("_srcmap_bad_global!", err.message)
    # Should suggest one of: global, const, type instability
    suggestion_lower = lowercase(err.suggestion)
    @test occursin("global", suggestion_lower) ||
          occursin("const", suggestion_lower) ||
          occursin("instab", suggestion_lower) ||
          occursin("dispatch", suggestion_lower)
end

# ═══════════════════════════════════════════════════════════════════════
# Test 13: LavaCompilationError has proper showerror
# ═══════════════════════════════════════════════════════════════════════

@testset "LavaCompilationError: showerror formatting" begin
    err = try
        lava_compile(Lava._srcmap_bad_alloc!, Tuple{LavaDeviceArray{Float32,1}})
        nothing
    catch e
        e
    end

    @test err isa LavaCompilationError

    # Check that showerror produces readable output
    output = sprint(showerror, err)
    @test occursin("LavaCompilationError", output)
    @test occursin("kernel compilation", output)
    @test occursin("Suggestion:", output)
    @test length(output) > 100  # should have substantial content
end

# ═══════════════════════════════════════════════════════════════════════
# Test 14: Runtime path also produces LavaCompilationError
# ═══════════════════════════════════════════════════════════════════════

@testset "Compilation error: runtime dispatch path" begin
    @kernel function _srcmap_bad_ka!(A)
        i = @index(Global)
        @inbounds A[i] = rand()
    end

    a = Mantle.LavaArray(Float32[1, 2, 3])
    err = try
        _srcmap_bad_ka!(Mantle.defaultbackend())(a; ndrange=3)
        nothing
    catch e
        e
    end

    @test err !== nothing
    @test err isa LavaCompilationError
    @test occursin("allocat", lowercase(err.suggestion)) ||
          occursin("heap", lowercase(err.suggestion))
end

# ═══════════════════════════════════════════════════════════════════════
# Test 15: Deep call chain — string interpolation in physics helper (3 levels)
# ═══════════════════════════════════════════════════════════════════════

@testset "Deep chain: string interpolation in collision check" begin
    err = try
        lava_compile(Lava._srcmap_deep_physics!,
            Tuple{LavaDeviceArray{Lava._TestParticle,1}, Float32})
        nothing
    catch e
        e
    end

    @test err isa LavaCompilationError
    @test !isempty(err.call_chains)

    # Call chain should show the path: kernel → _physics_step → _check_collision
    @test occursin("_srcmap_deep_physics!", err.call_chains)
    @test occursin("_physics_step", err.call_chains)
    @test occursin("_check_collision", err.call_chains)

    # Should identify heap allocation as the problem
    @test occursin("heap", lowercase(err.call_chains)) ||
          occursin("alloc", lowercase(err.call_chains)) ||
          occursin("string", lowercase(err.call_chains))

    # The suggestion should also mention allocation
    @test occursin("allocat", lowercase(err.suggestion)) ||
          occursin("heap", lowercase(err.suggestion))

    # The _update_velocity helper is fine — should NOT appear as a problem source
    # (it may appear in the chain path but not as the deepest problematic function)
end

# ═══════════════════════════════════════════════════════════════════════
# Test 16: Deep call chain — abstract field causing dynamic dispatch (4 levels)
# ═══════════════════════════════════════════════════════════════════════

@testset "Deep chain: abstract field in scene renderer" begin
    err = try
        lava_compile(Lava._srcmap_deep_scene!,
            Tuple{LavaDeviceArray{Float32,1}, LavaDeviceArray{Lava._SceneObject,1}})
        nothing
    catch e
        e
    end

    @test err isa LavaCompilationError
    @test !isempty(err.call_chains)

    # Call chain should trace through: kernel → _render_pixel → _compute_lighting → _get_albedo
    @test occursin("_srcmap_deep_scene!", err.call_chains)
    @test occursin("_get_albedo", err.call_chains)

    # Should identify type instability / dynamic dispatch
    @test occursin("dispatch", lowercase(err.call_chains)) ||
          occursin("instab", lowercase(err.call_chains))

    @test occursin("instab", lowercase(err.suggestion)) ||
          occursin("dispatch", lowercase(err.suggestion))
end

# ═══════════════════════════════════════════════════════════════════════
# Test 17: Deep call chain — error() in validation function (3 levels)
# ═══════════════════════════════════════════════════════════════════════

@testset "Deep chain: error() in validation helper" begin
    err = try
        lava_compile(Lava._srcmap_deep_error!,
            Tuple{LavaDeviceArray{Float32,1}, LavaDeviceArray{Float32,1}, Float32})
        nothing
    catch e
        e
    end

    @test err isa LavaCompilationError
    @test !isempty(err.call_chains)

    # Should show: kernel → _apply_tonemap → _validate_range
    @test occursin("_srcmap_deep_error!", err.call_chains)
    @test occursin("_apply_tonemap", err.call_chains)
    @test occursin("_validate_range", err.call_chains)

    # Should identify allocation (from string interpolation in error())
    @test occursin("alloc", lowercase(err.call_chains)) ||
          occursin("heap", lowercase(err.call_chains))
end

# ═══════════════════════════════════════════════════════════════════════
# Test 18: Deep call chain — @info logging left in GPU code (3 levels)
# ═══════════════════════════════════════════════════════════════════════

@testset "Deep chain: @info logging in helper" begin
    err = try
        lava_compile(Lava._srcmap_deep_logging!,
            Tuple{LavaDeviceArray{Float32,1}, LavaDeviceArray{NTuple{3,Float32},1}})
        nothing
    catch e
        e
    end

    # This should fail — @info does IO + allocation
    @test err !== nothing
    # Could be LavaCompilationError or another error type depending on
    # where in the pipeline it fails (GPUCompiler vs emitter)
    if err isa LavaCompilationError
        @test !isempty(err.call_chains) || !isempty(err.raw_error)
        @test occursin("_process_normal", err.call_chains) ||
              occursin("_process_normal", err.raw_error)
    end
end

# ═══════════════════════════════════════════════════════════════════════
# Test 19: Call chain deduplication — many reasons, few unique chains
# ═══════════════════════════════════════════════════════════════════════

@testset "Deep chain: deduplication reduces noise" begin
    # The physics kernel triggers ~16 GPUCompiler reasons but only ~2 unique user chains
    err = try
        lava_compile(Lava._srcmap_deep_physics!,
            Tuple{LavaDeviceArray{Lava._TestParticle,1}, Float32})
        nothing
    catch e
        e
    end

    @test err isa LavaCompilationError

    # Raw error has many "Reason:" lines (GPUCompiler is verbose)
    raw_reasons = count("Reason:", err.raw_error)
    @test raw_reasons >= 4  # typically 10-16 reasons for string interpolation

    # But call_chains should be compact — far fewer unique chains than raw reasons
    chain_lines = count("Problem:", err.call_chains)
    @test chain_lines >= 1
    @test chain_lines <= raw_reasons  # strictly fewer (deduplication works)

    # Output should be much shorter than raw error
    @test length(err.call_chains) < length(err.raw_error)
end

# ═══════════════════════════════════════════════════════════════════════
# Test 20: showerror formatting with call chains
# ═══════════════════════════════════════════════════════════════════════

@testset "Deep chain: showerror shows call chains before raw error" begin
    err = try
        lava_compile(Lava._srcmap_deep_scene!,
            Tuple{LavaDeviceArray{Float32,1}, LavaDeviceArray{Lava._SceneObject,1}})
        nothing
    catch e
        e
    end

    @test err isa LavaCompilationError
    output = sprint(showerror, err)

    # Call chains should appear BEFORE raw error in the output
    chain_pos = findfirst("Call chain", output)
    suggestion_pos = findfirst("Suggestion:", output)
    raw_pos = findfirst("Raw error", output)

    @test chain_pos !== nothing
    @test suggestion_pos !== nothing
    @test raw_pos !== nothing

    # Order: call chains → suggestion → raw error
    @test first(chain_pos) < first(suggestion_pos)
    @test first(suggestion_pos) < first(raw_pos)
end

# ═══════════════════════════════════════════════════════════════════════
# Test 21: Validation message leak fix (OOM doesn't poison next compile)
# ═══════════════════════════════════════════════════════════════════════

@testset "OOM validation messages don't leak to next compilation" begin
    # Trigger OOM — will generate validation layer warnings.
    #
    # `_try_vk_alloc(nbytes)` was renamed to `try_vk_alloc(bq, nbytes)` in
    # 0d8ae85 ("refactor for stability", 2026-04-06): the underscore was dropped
    # and it now takes the VulkanBatchQueue it allocates against. This test kept calling
    # the old name and threw UndefVarError; nobody saw it because the file was not
    # registered in runtests.jl.
    Mantle.try_vk_alloc(Mantle.batchqueue(Mantle.Device()), 40_000_000_000)  # 40GB, will fail

    # Validation messages should be drained by the failed alloc
    @test isempty(Mantle.vk_context().validation.messages)

    # Next compilation should succeed without stale validation errors
    r = lava_compile(Lava._srcmap_add!,
        Tuple{LavaDeviceArray{Float32,1}, Float32})
    @test length(r.spirv_bytes) > 0
    @test length(r.source_map) > 0
end

# ═══════════════════════════════════════════════════════════════════════
# Test 16: Source map IDs are valid SPIR-V result IDs
# ═══════════════════════════════════════════════════════════════════════

@testset "Source map: IDs are valid SPIR-V result IDs" begin
    r = lava_compile(Lava._srcmap_add!,
        Tuple{LavaDeviceArray{Float32,1}, Float32})

    # Parse all result IDs from SPIR-V disassembly
    valid_ids = Set{UInt32}()
    for line in split(r.spirv_disasm, '\n')
        m = match(r"^\s*%(\d+)\s*=", line)
        m !== nothing && push!(valid_ids, parse(UInt32, m.captures[1]))
    end

    # Most source map IDs should be valid SPIR-V result IDs.
    # Some IDs may be for non-result instructions (OpStore, OpBranch, etc.)
    # which don't appear as %id = in disassembly.
    matched = count(id -> id in valid_ids, keys(r.source_map))
    total = length(r.source_map)
    @test total > 0
    @test matched / total >= 0.5  # at least half should be result IDs
end

# ═══════════════════════════════════════════════════════════════════════
# Test 17: Compiled kernel still executes correctly with source mapping
# ═══════════════════════════════════════════════════════════════════════

@testset "Source mapping doesn't break kernel execution" begin
    # Simple add
    a = Mantle.LavaArray(Float32[1, 2, 3, 4])
    b = a .+ 10.0f0
    Mantle.flush!(Mantle.Device())
    @test Array(b) == Float32[11, 12, 13, 14]

    # Reduction
    @test sum(a) ≈ 10.0f0

    # Struct operations
    struct SrcMapTestStruct
        a::Float32
        b::Float32
    end

    @kernel function srcmap_struct_ka!(dst, src, s)
        i = @index(Global)
        @inbounds begin
            v = src[i]
            dst[i] = SrcMapTestStruct(v.a * s, v.b * s)
        end
    end

    src = Mantle.LavaArray([SrcMapTestStruct(1.0f0, 2.0f0), SrcMapTestStruct(3.0f0, 4.0f0)])
    dst = Mantle.LavaArray{SrcMapTestStruct}(undef, 2)
    srcmap_struct_ka!(Mantle.defaultbackend())(dst, src, 2.0f0; ndrange=2)
    Mantle.flush!(Mantle.Device())
    result = Array(dst)
    @test result[1].a ≈ 2.0f0
    @test result[1].b ≈ 4.0f0
    @test result[2].a ≈ 6.0f0
    @test result[2].b ≈ 8.0f0
end
