# Tier 1: Real kernel compilation tests using actual Hikari/Raycore types
#
# These test that our SPIR-V emitter handles production-grade nested structs,
# complex control flow, and atomic patterns. No GPU dispatch — compilation only.
#
# Requires Hikari + Raycore — skipped if not available (e.g. CI without [sources]).

using Test

# Guard: skip entire file if Hikari/Raycore aren't loadable. `return` at
# toplevel of an `include`d file is NOT an early exit in Julia, so wrap the
# rest of the file in an `if` instead.
const _can_load_hikari_raycore = try
    Base.require(Main, :Hikari)
    Base.require(Main, :Raycore)
    true
catch
    false
end

if !_can_load_hikari_raycore
    @testset "Real Kernel Compilation" begin
        @test_broken false  # mark as known-skipped
    end
else

if !@isdefined(SPIRVTestUtils)
    include(joinpath(pkgdir(Lava), "test", "spirv_test_utils.jl"))
end
import .SPIRVTestUtils: check, check_not, check_dag, check_sequence, check_count, check_regex, normalize_spirv, compare_golden, compile_and_disasm, spirv_opt_roundtrip, check_vendor_safety, compile_with_llc
using Hikari
using Raycore
using GeometryBasics
using StructArrays
using StaticArrays

@testset "Real Kernel Compilation" begin

    # ── Test 1: VPRayWorkItem kernel (14 fields) ──
    @testset "VPRayWorkItem — deep struct read/write" begin
        function ray_work_kernel(items, output)
            i = Lava.lava_global_invocation_id_x()
            @inbounds begin
                item = items[i]
                # Access nested fields: ray, wavelengths, spectral radiance
                ox = item.ray.o[1]
                dx = item.ray.d[1]
                w1 = item.lambda.lambda[1]
                b1 = item.beta.data[1]
                r1 = item.r_u.data[1]
                result = ox * dx + w1 * b1 + r1 + item.eta_scale
                output[i] = result
            end
            return nothing
        end

        WI = Hikari.VPRayWorkItem
        r = Lava.lava_compile(ray_work_kernel,
                              Tuple{Lava.LavaDeviceArray{WI, 1},
                                    Lava.LavaDeviceArray{Float32, 1}})
        @test !isempty(r.spirv_bytes)
        @testset "vendor safety" begin
            check_vendor_safety(r.spirv_disasm)
        end
        @testset "spirv-opt roundtrip" begin
            opt = spirv_opt_roundtrip(r.spirv_bytes)
            @test !isempty(opt)
        end
    end

    # ── Test 2: VPMediumSampleWorkItem kernel (29 fields — largest work item) ──
    @testset "VPMediumSampleWorkItem — 29-field struct" begin
        function medium_sample_kernel(items, output)
            i = Lava.lava_global_invocation_id_x()
            @inbounds begin
                item = items[i]
                result = if item.has_surface_hit
                    # Access deep nested hit geometry
                    item.hit_pi[1] + item.hit_n[2] + item.hit_dpdu[3] +
                    item.hit_uv[1] + item.hit_triangle_area
                else
                    item.ray.o[1] + item.t_max + item.eta_scale
                end
                output[i] = result
            end
            return nothing
        end

        WI = Hikari.VPMediumSampleWorkItem
        r = Lava.lava_compile(medium_sample_kernel,
                              Tuple{Lava.LavaDeviceArray{WI, 1},
                                    Lava.LavaDeviceArray{Float32, 1}})
        @test !isempty(r.spirv_bytes)
        @testset "vendor safety" begin
            check_vendor_safety(r.spirv_disasm)
        end
        @testset "spirv-opt roundtrip" begin
            opt = spirv_opt_roundtrip(r.spirv_bytes)
            @test !isempty(opt)
        end
    end

    # ── Test 3: VPMaterialEvalWorkItem kernel (27 fields with SVector) ──
    @testset "VPMaterialEvalWorkItem — SVector + SetKey" begin
        function material_eval_kernel(items, output)
            i = Lava.lava_global_invocation_id_x()
            @inbounds begin
                item = items[i]
                # Dot product of shading normal and outgoing direction
                dot_val = item.ns[1] * item.wo[1] +
                          item.ns[2] * item.wo[2] +
                          item.ns[3] * item.wo[3]
                # Access SVector barycentrics
                bary_sum = item.bary[1] + item.bary[2] + item.bary[3]
                output[i] = dot_val + bary_sum + item.eta_scale
            end
            return nothing
        end

        WI = Hikari.VPMaterialEvalWorkItem
        r = Lava.lava_compile(material_eval_kernel,
                              Tuple{Lava.LavaDeviceArray{WI, 1},
                                    Lava.LavaDeviceArray{Float32, 1}})
        @test !isempty(r.spirv_bytes)
        @testset "vendor safety" begin
            check_vendor_safety(r.spirv_disasm)
        end
        @testset "spirv-opt roundtrip" begin
            opt = spirv_opt_roundtrip(r.spirv_bytes)
            @test !isempty(opt)
        end
    end

    # ── Test 4: Atomic WorkQueue push pattern ──
    @testset "atomic WorkQueue push — Device scope" begin
        function atomic_push_kernel(counter, items, values)
            i = Lava.lava_global_invocation_id_x()
            @inbounds begin
                val = values[i]
                if val > 0.0f0
                    idx = Lava.Atomix.@atomic counter[1] += Int32(1)
                    items[idx + Int32(1)] = val
                end
            end
            return nothing
        end

        r = Lava.lava_compile(atomic_push_kernel,
                              Tuple{Lava.LavaDeviceArray{Int32, 1},
                                    Lava.LavaDeviceArray{Float32, 1},
                                    Lava.LavaDeviceArray{Float32, 1}})
        @test !isempty(r.spirv_bytes)
        # Verify Device scope atomics (not CrossDevice)
        check(r.spirv_disasm, "OpAtomicIAdd")
        check_not(r.spirv_disasm, "CrossDevice")
        @testset "vendor safety" begin
            check_vendor_safety(r.spirv_disasm)
        end
        @testset "spirv-opt roundtrip" begin
            opt = spirv_opt_roundtrip(r.spirv_bytes)
            @test !isempty(opt)
        end
    end

    # ── Test 5: BVH traversal-like kernel (while-loop with break, Mat4f) ──
    @testset "BVH traversal — structured CF stress" begin
        function bvh_traverse_kernel(nodes, rays, hits)
            i = Lava.lava_global_invocation_id_x()
            @inbounds begin
                ray_ox = rays[i]
                # Simulate BVH stack walk with while loop + break
                stack_top = Int32(0)
                best_t = Float32(1.0e10)
                node_idx = Int32(1)
                while node_idx > Int32(0)
                    node_val = nodes[node_idx]
                    # AABB test: simple min/max
                    t_enter = max(node_val - ray_ox, 0.0f0)
                    t_exit = min(node_val + ray_ox, best_t)
                    if t_enter < t_exit
                        if node_val < 0.0f0
                            # Leaf: update best hit
                            best_t = t_enter
                            break
                        else
                            # Internal: descend
                            node_idx = unsafe_trunc(Int32, node_val)
                        end
                    else
                        # Pop stack (simplified)
                        stack_top -= Int32(1)
                        node_idx = stack_top
                    end
                end
                hits[i] = best_t
            end
            return nothing
        end

        r = Lava.lava_compile(bvh_traverse_kernel,
                              Tuple{Lava.LavaDeviceArray{Float32, 1},
                                    Lava.LavaDeviceArray{Float32, 1},
                                    Lava.LavaDeviceArray{Float32, 1}})
        @test !isempty(r.spirv_bytes)
        check(r.spirv_disasm, "OpLoopMerge")
        check(r.spirv_disasm, "OpSelectionMerge")
        @testset "vendor safety" begin
            check_vendor_safety(r.spirv_disasm)
        end
        @testset "spirv-opt roundtrip" begin
            opt = spirv_opt_roundtrip(r.spirv_bytes)
            @test !isempty(opt)
        end
    end

    # ── Test 6: Multi-field output kernel (write large structs through BDA) ──
    @testset "large struct write — PSB alignment" begin
        function struct_write_kernel(input, output)
            i = Lava.lava_global_invocation_id_x()
            @inbounds begin
                val = input[i]
                # Create a VPShadowRayWorkItem and write it
                ray = Raycore.Ray(Point3f(val, 0.0f0, 0.0f0),
                                  Vec3f(0.0f0, 0.0f0, 1.0f0),
                                  0.0f0, 1000.0f0, 0.0f0)
                lambda = Hikari.SampledWavelengths{4}(
                    NTuple{4, Float32}((400.0f0, 500.0f0, 600.0f0, 700.0f0)),
                    NTuple{4, Float32}((0.25f0, 0.25f0, 0.25f0, 0.25f0)))
                ld = Hikari.SampledSpectrum{4}(NTuple{4, Float32}((val, val, val, val)))
                item = Hikari.VPShadowRayWorkItem(
                    ray, 100.0f0, lambda, ld, ld, ld,
                    Int32(i), Raycore.SetKey(UInt32(0), UInt32(0)))
                output[i] = item
            end
            return nothing
        end

        WI = Hikari.VPShadowRayWorkItem
        r = Lava.lava_compile(struct_write_kernel,
                              Tuple{Lava.LavaDeviceArray{Float32, 1},
                                    Lava.LavaDeviceArray{WI, 1}})
        @test !isempty(r.spirv_bytes)
        @testset "vendor safety" begin
            check_vendor_safety(r.spirv_disasm)
        end
        @testset "spirv-opt roundtrip" begin
            opt = spirv_opt_roundtrip(r.spirv_bytes)
            @test !isempty(opt)
        end
    end

    # ── Test 7: vp_shade_material_kernel! with TypedHitRef{Conductor{PiecewiseLinearSpectrum{56}}} ──
    # Regression test for retype_allocas bug: LLVM emits a float access (f32/f64) wider than the
    # alloca's chosen integer element type (e.g. T=i8), hitting the decomposition branch that
    # only handled int→int.  Fix: pick_uniform_type bails when decomp would involve non-integer types.
    #
    # Originally framed against `vp_sample_surface_direct_lighting_kernel!` which Hikari fused
    # into `vp_shade_surface_hits_kernel!` and then split per-material into `vp_shade_material_
    # kernel!{T}` once per concrete material type. The retype_allocas path is exercised the
    # same way — heavy struct access through Conductor{PiecewiseLinearSpectrum{56}}.
    @testset "vp_shade_material_kernel! — Conductor PiecewiseLinearSpectrum" begin
        using KernelAbstractions

        T_conductor = Hikari.Conductor{Hikari.PiecewiseLinearSpectrum{56},
                                       Hikari.PiecewiseLinearSpectrum{56},
                                       Float32, Hikari.RGBSpectrum}
        # The per-material split (2026-06-03) replaced the full-payload
        # `TypedHit{T}` queue with a 4-byte `TypedHitRef{T}` index queue plus a
        # shared `hit_surface_queue::WorkQueue{VPHitSurfaceWorkItem}`; the kernel
        # signature gained `hit_surface_queue` as its 2nd argument.
        T_typed_hit = Hikari.TypedHitRef{T_conductor}
        T_size      = Lava.LavaDeviceArray{Int32, 1}
        # Per-material typed queue: 4-byte TypedHitRef{T} indices (AOS).
        T_in_q      = Hikari.WorkQueue{T_typed_hit,
                                       Lava.LavaDeviceArray{T_typed_hit, 1},
                                       T_size}
        # Shared hit queue (SOA): one device component array per top-level field
        # of VPHitSurfaceWorkItem, generated from the struct so this type tracks
        # field changes automatically instead of drifting like the old spelling.
        _hs_item = Hikari.VPHitSurfaceWorkItem
        _hs_cols = NamedTuple{fieldnames(_hs_item),
                              Tuple{map(F -> Lava.LavaDeviceArray{F, 1}, fieldtypes(_hs_item))...}}
        T_hit_surface_q = Hikari.WorkQueue{_hs_item,
                                           StructArrays.StructVector{_hs_item, _hs_cols, Int64},
                                           T_size}
        T_next_ray_q = Hikari.WorkQueue{Hikari.VPRayWorkItem,
                                        StructArrays.StructVector{Hikari.VPRayWorkItem, @NamedTuple{
                                            ray::Lava.LavaDeviceArray{Raycore.Ray, 1},
                                            depth::Lava.LavaDeviceArray{Int32, 1},
                                            lambda::Lava.LavaDeviceArray{Hikari.SampledWavelengths{4}, 1},
                                            pixel_index::Lava.LavaDeviceArray{Int32, 1},
                                            beta::Lava.LavaDeviceArray{Hikari.SampledSpectrum{4}, 1},
                                            r_u::Lava.LavaDeviceArray{Hikari.SampledSpectrum{4}, 1},
                                            r_l::Lava.LavaDeviceArray{Hikari.SampledSpectrum{4}, 1},
                                            prev_intr_p::Lava.LavaDeviceArray{Point{3, Float32}, 1},
                                            prev_intr_n::Lava.LavaDeviceArray{Vec{3, Float32}, 1},
                                            eta_scale::Lava.LavaDeviceArray{Float32, 1},
                                            specular_bounce::Lava.LavaDeviceArray{Bool, 1},
                                            any_non_specular_bounces::Lava.LavaDeviceArray{Bool, 1},
                                            medium_idx::Lava.LavaDeviceArray{Raycore.SetKey, 1},
                                        }, Int64},
                                        T_size}
        T_mats = Raycore.StaticMultiTypeSet{Tuple{
            Lava.LavaDeviceArray{T_conductor, 1},
        }, Tuple{}}
        T_lights = Raycore.StaticMultiTypeSet{Tuple{
            Lava.LavaDeviceArray{Hikari.DiffuseAreaLight{Hikari.RGBSpectrum}, 1},
        }, Tuple{}}
        T_rgb_table = Hikari.RGBToSpectrumTable{
            Lava.LavaDeviceArray{Float32, 1},
            Lava.LavaDeviceArray{Float32, 5},
            Lava.LavaDeviceArray{Float32, 1}}
        T_accel = Mantle.AdaptedAccel{Nothing,
                                       Lava.LavaDeviceArray{Raycore.Triangle{Hikari.TriangleMeta}, 1},
                                       Lava.LavaDeviceArray{UInt32, 1},
                                       Raycore.Triangle{Hikari.TriangleMeta}}

        ws = 256
        kernel_obj = Hikari.workqueue_map_kernel!(Mantle.LavaBackend(), ws)
        iterspace, _ = KernelAbstractions.partition(kernel_obj, (1024 * 1024 * ws,), (ws,))
        ctx = KernelAbstractions.mkcontext(kernel_obj, (1024 * 1024 * ws,), iterspace)

        tt = Tuple{
            typeof(ctx),
            typeof(Hikari.vp_shade_material_kernel!),
            T_in_q,                                      # workqueue-map iterates TypedHitRef{Conductor}
            T_hit_surface_q,                             # shared hit_surface_queue (VPHitSurfaceWorkItem)
            T_next_ray_q,                                # next_ray_queue
            Lava.LavaDeviceArray{Float32, 1},            # pixel_L
            T_accel,                                     # accel
            Lava.LavaDeviceArray{Hikari.MediumInterfaceIdx, 1},  # media_interfaces
            Raycore.StaticMultiTypeSet{Tuple{}, Tuple{}},        # media
            T_mats,                                      # materials
            T_lights,                                    # lights
            T_rgb_table,                                 # rgb2spec_table
            Lava.LavaDeviceArray{Hikari.LightBVHNode, 1},        # bvh_nodes
            Lava.LavaDeviceArray{Int32, 1},                      # infinite_light_indices
            Lava.LavaDeviceArray{UInt32, 1},                     # light_to_bit_trail
            Int32, Int32, Int32,                                 # num_inf / num_bvh / num_lights
            Int32, Bool,                                         # max_depth, do_regularize
            Hikari.SobolRNG{Lava.LavaDeviceArray{UInt32, 1}},    # sobol_rng
            Int32,                                               # sample_idx
            Hikari.PerspectiveCamera,                            # camera
            Int32, Int32,                                        # samples_per_pixel, rr_depth
        }

        d, bytes = compile_and_disasm(Hikari.gpu_workqueue_map_kernel!, tt;
                                       workgroup_size=(ws, 1, 1),
                                       enable_ray_query=true)  # inline shadow trace via rayQuery
        @test !isempty(bytes)
        @testset "vendor safety" begin
            check_vendor_safety(d)
        end
        @testset "spirv-opt roundtrip" begin
            opt = spirv_opt_roundtrip(bytes)
            @test !isempty(opt)
        end
    end

end

end  # if _can_load_hikari_raycore
