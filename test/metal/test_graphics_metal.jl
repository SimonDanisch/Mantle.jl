# Julia → AIR vertex, fragment and mesh programs (`src/metal/compiler/`), and the
# bindings they are drawn through (`src/metal/bindings/`).
#
# A graphics stage differs from a compute kernel in three ways that all have to hold
# at once (it returns a packed struct, its inputs are values, it is registered under
# its own named metadata), so the first testsets assert the shape of the emitted
# module rather than any one of them alone. The reference for every expectation is
# a shipping metallib (PencilKit's `default.metallib`), read with Metal.jl's
# `MetalLib` parser and LLVM.jl; `src/metal/compiler/stages.jl` records what was
# found and how to reproduce it. The draws after that grade the pixels against the
# same shaders written in MSL and compiled by the system compiler.
#
# Shader arguments are bound with `set_*_bytes!`, as Mantle's own draws bind them:
# a vertex list or an extent travels by value, and a buffer the shader writes is an
# `MtlDeviceArray` whose memory is made resident with `use!`.

module TestGraphicsMetal

using Test, Mantle, Metal, LLVM, GPUCompiler
import Lava
using Metal: MTL
using Mantle: StageOut
using Mantle.MetalStageCompiler: stage_job, mangle_varying, air_stage_name
using Mantle.MetalIntrinsics: VertexID, FragCoord, Varying, MeshPtr, Texture2DPtr,
    SamplerPtr, air_sample_texture_2d, set_position_mesh, set_vertex_data_mesh,
    set_primitive_data_mesh, set_index_mesh, set_primitive_count_mesh
import Mantle.MetalBindings as MB

# The Mantle device first: creating it initialises the backend, which loads
# QuartzCore for the layer testset.
const GFX_MDEV = Mantle.Device(Mantle.MetalAPI())
const GFX_DEV = GFX_MDEV.dev
# Held at file scope, not per call: a command queue released while its command
# buffer is still in flight drops the work silently, which shows up as an empty
# target rather than as an error.
const GFX_QUEUE = MTL.MTLCommandQueue(GFX_DEV)
# Every compiled library stays reachable: an `MTLFunction` does not keep its library
# alive, and a released library makes the pipeline draw nothing.
const GFX_LIBS = Any[]

struct GfxTri
    v::NTuple{3,NTuple{4,Float32}}
end

const GFX_VERTS = GfxTri(((-0.9f0, -0.9f0, 0f0, 1f0), (0.9f0, -0.9f0, 0f0, 1f0),
                          (0f0, 0.9f0, 0f0, 1f0)))

struct GfxVOut
    position::NTuple{4,Float32}
    color::NTuple{4,Float32}
end

struct GfxOnlyPos
    position::NTuple{4,Float32}
end

struct GfxFOut
    color::NTuple{4,Float32}
end

# The dynamic tuple index is bounds-checked, so the stage has a throw site, which
# `stage_drop_exception_signal!` has to make valid without a kernel state.
function gfx_vertex(verts::GfxTri, vid::VertexID, out::StageOut{GfxVOut})
    Base.unsafe_store!(out, GfxVOut(verts.v[Int(vid.value) + 1], (1f0, 0f0, 0f0, 1f0)))
    return nothing
end

function gfx_fragment(pos::FragCoord, out::StageOut{GfxFOut})
    Base.unsafe_store!(out, GfxFOut((pos.value[1], pos.value[2], 0f0, 1f0)))
    return nothing
end

function gfx_tri_vertex(verts::GfxTri, vid::VertexID, out::StageOut{GfxOnlyPos})
    Base.unsafe_store!(out, GfxOnlyPos(verts.v[Int(vid.value) + 1]))
    return nothing
end

gfx_green(out::StageOut{GfxFOut}) =
    (Base.unsafe_store!(out, GfxFOut((0f0, 1f0, 0f0, 1f0))); nothing)

const GFX_VERTEX_TT = Tuple{GfxTri, VertexID, StageOut{GfxVOut}}
const GFX_TRI_TT    = Tuple{GfxTri, VertexID, StageOut{GfxOnlyPos}}

"""
Compile `f` as `stage` and call `check` on the LLVM module, inside the context it
was compiled in: the module belongs to the caller of `compile` and is freed with
that context, so it cannot be returned.
"""
function compile_stage(check, f, tt, stage::Symbol; name = string(nameof(f)))
    job = stage_job(GFX_DEV, f, tt, stage, name)
    GPUCompiler.JuliaContext() do _
        mod = first(GPUCompiler.compile(:llvm, job))
        try
            check(mod)
        finally
            LLVM.dispose(mod)
        end
    end
    return
end

nodes(mod, key) = haskey(mod.metadata, key) ?
                  collect(mod.metadata[key].operands) : LLVM.MDNode[]

"""Compile `f` as `stage` through Mantle's cached path and hand back the `MTLFunction`."""
function stage_function(f, tt, stage::Symbol, name::String)
    fn, lib = Mantle.compile_stage_function(GFX_DEV, f, tt, stage, name)
    push!(GFX_LIBS, lib)
    return fn
end

"""A `W`x`H` RGBA8 render target, readable from the host."""
function render_target(W, H)
    td = MTL.MTLTextureDescriptor(MTL.MTLPixelFormatRGBA8Unorm, W, H, false)
    td.usage = MTL.MTLTextureUsageRenderTarget | MTL.MTLTextureUsageShaderRead
    td.storageMode = MTL.MTLStorageModeShared
    return MTL.MTLTexture(GFX_DEV, td)
end

"""A pass clearing `tex` to opaque black."""
function clearing_pass(tex)
    rp = MB.render_pass_descriptor()
    ca = MB.colorattachment(rp, 1)
    ca.texture = tex
    ca.loadAction  = MTL.MTLLoadActionClear
    ca.storeAction = MTL.MTLStoreActionStore
    ca.clearColor  = MTL.MTLClearColor(0.0, 0.0, 0.0, 1.0)
    return rp
end

"""The texels of a `W`x`H` RGBA8 texture, read with `getbytes!`."""
function texels(tex, W, H)
    px = Vector{UInt8}(undef, W * H * 4)
    GC.@preserve px MB.getbytes!(pointer(px), tex, W * 4,
                                 MTL.MTLRegion(MTL.MTLOrigin(0, 0, 0), MTL.MTLSize(W, H, 1)))
    return px
end

greens(px) = count(i -> px[4i + 2] > 0x80, 0:(length(px) ÷ 4 - 1))

"""
Draw `GFX_VERTS` with `vsf`/`fsf` into a 64x64 target and return the target's
texels. `bind!(enc)` binds what the fragment stage needs.
"""
function draw_triangle(vsf, fsf; bind! = enc -> nothing)
    W = H = 64
    tex = render_target(W, H)
    pd = MB.render_pipeline_descriptor()
    pd.vertexFunction = vsf
    pd.fragmentFunction = fsf
    MB.colorattachment(pd, 1).pixelFormat = MTL.MTLPixelFormatRGBA8Unorm
    pipe = MB.render_pipeline_state(GFX_DEV, pd)

    cb  = MTL.MTLCommandBuffer(GFX_QUEUE)
    enc = MB.render_command_encoder(cb, clearing_pass(tex))
    MB.set_pipeline!(enc, pipe)
    verts = Ref(GFX_VERTS)
    GC.@preserve verts MB.set_vertex_bytes!(enc, Base.unsafe_convert(Ptr{GfxTri}, verts),
                                            sizeof(GfxTri), 1)
    bind!(enc)
    MB.draw_primitives!(enc, MTL.MTLPrimitiveTypeTriangle, 0, 3)
    MTL.endEncoding!(enc)
    MTL.commit!(cb)
    MTL.wait_completed(cb)
    return tex, texels(tex, W, H)
end

const GFX_MSL_TRIANGLE = """
    #include <metal_stdlib>
    using namespace metal;
    struct VO { float4 position [[position]]; };
    vertex VO ref_vs(constant float4 *v [[buffer(0)]], uint vid [[vertex_id]]) {
        VO o; o.position = v[vid]; return o;
    }
    fragment float4 ref_fs() { return float4(0, 1, 0, 1); }
    """

@testset "the varying linkage tag" begin
    # A varying is matched between stages by THIS STRING and nothing else: not by
    # position, not by name, not by type. These four are what the reference metallib
    # carries; getting the length prefix or the type encoding wrong silently
    # disconnects the stages instead of failing to build.
    @test mangle_varying("snapshotTexCoord", NTuple{2,Float32}) ==
          "generated(16snapshotTexCoordDv2_f)"
    @test mangle_varying("texCoord",   NTuple{2,Float32}) == "generated(8texCoordDv2_f)"
    @test mangle_varying("pixelCoord", NTuple{2,Float32}) == "generated(10pixelCoordDv2_f)"
    @test mangle_varying("color",      NTuple{4,Float32}) == "generated(5colorDv4_f)"

    # `air.arg_type_name` uses a different spelling for the same type, and the
    # reference carries both for every varying.
    @test air_stage_name(NTuple{4,Float32}) == "float4"
    @test air_stage_name(NTuple{2,Float32}) == "float2"
    @test air_stage_name(NTuple{4,Float16}) == "half4"
    @test air_stage_name(UInt32) == "uint"
end

@testset "a Julia function compiles to an AIR vertex program" begin
    compile_stage(gfx_vertex, GFX_VERTEX_TT, :vertex) do mod
        @test length(nodes(mod, "air.vertex")) == 1
        # …and it must STOP being a kernel: leaving the old node would leave
        # `air.kernel` naming a function that now returns a struct, and the metallib
        # writer would pack it as a compute function.
        @test isempty(nodes(mod, "air.kernel"))

        entry = only(f for f in mod.functions
                     if !LLVM.isdeclaration(f) && f.name == "gfx_vertex")
        ft = entry.function_type

        # The return type is the whole point. `convert(LLVMType, NTuple{4,Float32})`
        # is `[4 x float]`, an ARRAY; AIR wants `<4 x float>` in a PACKED struct, and
        # the loader rejects the other one.
        rt = ft.return_type
        @test rt isa LLVM.StructType
        @test LLVM.ispacked(rt)
        @test string(rt) == "<{ <4 x float>, <4 x float> }>"

        # The trailing output pointer is gone, and the vertex id is a VALUE: the
        # kernel ABI would have passed it as another buffer pointer.
        @test string(last(collect(ft.parameters))) == "i32"

        ops = only(nodes(mod, "air.vertex")).operands
        outs = [string(o) for o in ops[2].operands]
        @test occursin("air.position", outs[1])
        @test occursin("\"position\"", outs[1])
        @test occursin("air.vertex_output", outs[2])
        @test occursin("generated(5colorDv4_f)", outs[2])

        ins = [string(o) for o in ops[3].operands]
        @test any(s -> occursin("air.vertex_id", s), ins)
        # The by-value argument keeps GPUCompiler's buffer description verbatim.
        @test any(s -> occursin("air.buffer", s) && occursin("air.address_space", s), ins)

        # The bounds check's throw site signals through a function a stage can have:
        # defined (empty), not a declaration IR validation would reject.
        sig = get(mod.functions, "gpu_signal_exception", nothing)
        @test sig === nothing || !LLVM.isdeclaration(sig)
    end
end

@testset "a Julia function compiles to an AIR fragment program" begin
    compile_stage(gfx_fragment, Tuple{FragCoord, StageOut{GfxFOut}}, :fragment) do mod
        @test length(nodes(mod, "air.fragment")) == 1
        @test isempty(nodes(mod, "air.kernel"))

        entry = only(f for f in mod.functions
                     if !LLVM.isdeclaration(f) && f.name == "gfx_fragment")
        @test string(entry.function_type.return_type) == "<{ <4 x float> }>"

        ops = only(nodes(mod, "air.fragment")).operands
        # A fragment's outputs are render targets, indexed from zero.
        outs = [string(o) for o in ops[2].operands]
        @test occursin("air.render_target", outs[1])
        @test occursin("i32 0", outs[1])

        # The interpolated position arrives as a value, and declares how it is
        # sampled: the reference pairs `air.position` with `air.center` and
        # `air.no_perspective`.
        ins = [string(o) for o in ops[3].operands]
        pos = only(filter(s -> occursin("air.position", s), ins))
        @test occursin("air.center", pos)
        @test occursin("air.no_perspective", pos)
    end
end

function gfx_pointsize_vertex(verts::GfxTri, vid::VertexID, out::StageOut{GfxOnlyPos})
    Lava.set_point_size!(4f0)
    Base.unsafe_store!(out, GfxOnlyPos(verts.v[Int(vid.value) + 1]))
    return nothing
end

@testset "set_point_size! adds a [[point_size]] output" begin
    compile_stage(gfx_pointsize_vertex, GFX_TRI_TT, :vertex) do mod
        entry = only(f for f in mod.functions
                     if !LLVM.isdeclaration(f) && f.name == "gfx_pointsize_vertex")
        # The position, then the point size as a trailing float.
        @test string(entry.function_type.return_type) == "<{ <4 x float>, float }>"
        outs = [string(o) for o in only(nodes(mod, "air.vertex")).operands[2].operands]
        @test length(outs) == 2
        @test occursin("air.point_size", outs[2])
        # The global the intrinsic stored into is gone.
        @test !haskey(mod.globals, "__air_stage_point_size")
    end
end

@testset "Metal itself accepts the stage" begin
    # Pack the AIR into a metallib, hand it to the driver, and ask what it thinks it
    # got. Everything above checks our own emission against a specification; this
    # checks it against Metal. Metal.jl's writer derives the program type from the
    # `air.vertex`/`air.fragment` metadata; a stage tagged as a kernel is refused.
    job = stage_job(GFX_DEV, gfx_vertex, GFX_VERTEX_TT, :vertex, "gfx_vertex")
    lib = MTL.MTLLibraryFromData(GFX_DEV, Metal.compile_to_metallib(job).metallib)
    @test String.(lib.functionNames) == ["gfx_vertex"]
    @test MTL.MTLFunction(lib, "gfx_vertex").functionType == MTL.MTLFunctionTypeVertex

    fjob = stage_job(GFX_DEV, gfx_fragment, Tuple{FragCoord, StageOut{GfxFOut}},
                     :fragment, "gfx_fragment")
    flib = MTL.MTLLibraryFromData(GFX_DEV, Metal.compile_to_metallib(fjob).metallib)
    @test MTL.MTLFunction(flib, "gfx_fragment").functionType == MTL.MTLFunctionTypeFragment
end

@testset "a triangle drawn by Julia shaders" begin
    # Graded against the same triangle drawn by hand-written MSL rather than against a
    # number somebody wrote down.
    #
    # It is also the test that catches what the rest cannot. `stage_return!` and
    # `stage_inputs!` hand the cloned body a stack `alloca` cast into the parameter's
    # address space, `thread` to `device`, which is not a legal pointer on Metal. A
    # stage that kept it still compiled, still linked, still built a render pipeline,
    # and drew NOTHING. `stage_cleanup!` runs SROA and the slot goes away; without it
    # this testset reports zero green pixels and every assertion above still passes.
    msl = MTL.MTLLibrary(GFX_DEV, GFX_MSL_TRIANGLE)
    ref_vs = MTL.MTLFunction(msl, "ref_vs")
    ref_fs = MTL.MTLFunction(msl, "ref_fs")
    jl_vs = stage_function(gfx_tri_vertex, GFX_TRI_TT, :vertex, "gfx_tri_vertex")
    jl_fs = stage_function(gfx_green, Tuple{StageOut{GfxFOut}}, :fragment, "gfx_green")

    reftex, refpx = draw_triangle(ref_vs, ref_fs)
    # The control really drew a triangle: about 40 % of the frame, which is the area
    # of this one in NDC. A blank or fully-covered target fails here.
    reference = greens(refpx)
    @test 1000 < reference < 3000

    # Each stage on its own against the reference's other half, then both, so a
    # failure names which stage broke. The whole buffer is compared: the two shaders
    # do exactly the same arithmetic, so anything but equality is a bug.
    for (label, v, f) in (("Julia vertex", jl_vs, ref_fs),
                          ("Julia fragment", ref_vs, jl_fs),
                          ("both", jl_vs, jl_fs))
        _, px = draw_triangle(v, f)
        @test (label, greens(px)) == (label, reference)
        @test (label, px == refpx) == (label, true)
    end

    # The texture-to-buffer blit reads the same texels `getbytes!` does.
    W = H = 64
    buf = MTL.MTLBuffer(GFX_DEV, W * H * 4; storage = Metal.SharedStorage)
    cb = MTL.MTLCommandBuffer(GFX_QUEUE)
    MTL.MTLBlitCommandEncoder(cb) do enc
        MB.append_copy!(enc, buf, 0, W * 4, 0, reftex,
                        MTL.MTLOrigin(0, 0, 0), MTL.MTLSize(W, H, 1))
    end
    MTL.commit!(cb)
    MTL.wait_completed(cb)
    blitted = GC.@preserve buf copy(unsafe_wrap(Array, convert(Ptr{UInt8}, MTL.contents(buf)),
                                                W * H * 4))
    @test blitted == refpx
end

# ── the shared shader vocabulary ─────────────────────────────────────────────

# Written the way Mantle's shaders are: Lava's `vertex_index()` as a free call, not
# as a declared parameter, reaching Metal through the stage method table.
function gfx_builtin_vertex(verts::GfxTri, out::StageOut{GfxOnlyPos})
    Base.unsafe_store!(out, GfxOnlyPos(verts.v[Lava.vertex_index()]))
    return nothing
end

function gfx_builtin_fragment(out::StageOut{GfxFOut})
    # Reading the interpolated position is what makes this a stage input rather than
    # a constant-folded shader.
    x = Lava.frag_coord_x()
    Base.unsafe_store!(out, GfxFOut((0f0, x > -1f0 ? 1f0 : 0f0, 0f0, 1f0)))
    return nothing
end

@testset "shaders written with the vocabulary's free-function builtins" begin
    # `vertex_index()` is a zero-argument call in Julia and a PARAMETER in AIR: there
    # is no ambient vertex id. `stage_builtins!` bridges that: the intrinsic loads an
    # undefined global, and the pass turns each referenced one into a trailing entry
    # parameter with the matching `air.*` tag. The builtin must be APPENDED and the
    # output pointer rotated back to last, because `stage_return!` takes the last
    # parameter to be the output.
    jl_vs = stage_function(gfx_builtin_vertex, Tuple{GfxTri, StageOut{GfxOnlyPos}},
                           :vertex, "gfx_builtin_vertex")
    jl_fs = stage_function(gfx_builtin_fragment, Tuple{StageOut{GfxFOut}},
                           :fragment, "gfx_builtin_fragment")
    @test jl_vs.functionType == MTL.MTLFunctionTypeVertex
    @test jl_fs.functionType == MTL.MTLFunctionTypeFragment

    green = greens(last(draw_triangle(jl_vs, jl_fs)))
    @test 1000 < green < 3000

    # One-based, like Lava's: if this were zero-based the triangle would read one
    # vertex past its tuple and come out degenerate.
    tri_vs = stage_function(gfx_tri_vertex, GFX_TRI_TT, :vertex, "gfx_tri_vertex")
    @test green == greens(last(draw_triangle(tri_vs, jl_fs)))
end

# ── mesh stages ──────────────────────────────────────────────────────────────
#
# A mesh stage writes its primitives through an object handed to it in address
# space 7, so nothing about it goes through `stage_return!`. Graded against the same
# triangle emitted by hand-written MSL, because "some pixels came out" is not the
# assertion; "the same pixels the system compiler produces" is.

const MeshV = @NamedTuple{position::NTuple{4,Float32}, uv::NTuple{2,Float32}}
const MeshP = @NamedTuple{colour::NTuple{4,Float32}}
const MeshOut = MeshPtr{MeshV, MeshP, 4, 2, :triangle}
const MeshFragTT = Tuple{Varying{:uv, NTuple{2,Float32}}, Varying{:colour, NTuple{4,Float32}},
                         StageOut{GfxFOut}}

"""The fullscreen triangle the MSL reference emits, with its corners scaled by `s`."""
@inline function emit_triangle!(out, s::Float32)
    set_position_mesh(out, Int32(0), (-s, -s, 0f0, 1f0))
    set_position_mesh(out, Int32(1), (3f0 * s, -s, 0f0, 1f0))
    set_position_mesh(out, Int32(2), (-s, 3f0 * s, 0f0, 1f0))
    set_vertex_data_mesh(out, Int32(0), Int32(0), (0f0, 0f0))
    set_vertex_data_mesh(out, Int32(0), Int32(1), (2f0, 0f0))
    set_vertex_data_mesh(out, Int32(0), Int32(2), (0f0, 2f0))
    set_index_mesh(out, Int32(0), UInt8(0))
    set_index_mesh(out, Int32(1), UInt8(1))
    set_index_mesh(out, Int32(2), UInt8(2))
    set_primitive_data_mesh(out, Int32(0), Int32(0), (0f0, 1f0, 0f0, 1f0))
    set_primitive_count_mesh(out, Int32(1))
    return nothing
end

gfx_mesh_tri(out::MeshOut) = emit_triangle!(out, 1f0)

# The same, with the triangle's extent read from an argument of its own, which is
# what every real mesh stage does, and the case the slot rule governs.
gfx_mesh_scaled(out::MeshOut, extent::Float32) = emit_triangle!(out, extent)

# Reads BOTH planes: `uv` is per-vertex and interpolated, `colour` is per-primitive.
# A fragment stage that ignored them would pass even if only one plane linked, and
# the two are matched by separate metadata.
function gfx_mesh_fragment(uv::Varying{:uv, NTuple{2,Float32}},
                           colour::Varying{:colour, NTuple{4,Float32}},
                           out::StageOut{GfxFOut})
    c = colour.value
    u = uv.value
    Base.unsafe_store!(out, GfxFOut((u[1] * 0f0, c[2], u[2] * 0f0, 1f0)))
    return nothing
end

"""
Draw `groups` mesh threadgroups of `threads` threads into a 64x64 target and return
its texels. `bind!(enc)` binds the mesh stage's arguments.
"""
function draw_mesh(meshfn, fragfn; bind! = enc -> nothing, threads::Int = 1, groups::Int = 1)
    W = H = 64
    pd = MB.mesh_render_pipeline_descriptor()
    pd.meshFunction = meshfn
    pd.fragmentFunction = fragfn
    MB.colorattachment(pd, 1).pixelFormat = MTL.MTLPixelFormatRGBA8Unorm
    pd.maxTotalThreadsPerMeshThreadgroup = threads
    pipe = MB.render_pipeline_state(GFX_DEV, pd)

    tex = render_target(W, H)
    cb  = MTL.MTLCommandBuffer(GFX_QUEUE)
    enc = MB.render_command_encoder(cb, clearing_pass(tex))
    MB.set_pipeline!(enc, pipe)
    MB.set_viewport!(enc, MTL.MTLViewport(0, 0, W, H, 0, 1))
    bind!(enc)
    MB.draw_mesh_threadgroups!(enc, MTL.MTLSize(groups, 1, 1), MTL.MTLSize(1, 1, 1),
                               MTL.MTLSize(threads, 1, 1))
    MTL.endEncoding!(enc)
    MTL.commit!(cb)
    MTL.wait_completed(cb)
    return texels(tex, W, H)
end

"""Bind `extent` as the mesh stage's argument at 1-based `slot`."""
bind_extent(extent::Float32, slot::Int) = enc -> begin
    r = Ref(extent)
    GC.@preserve r MB.set_mesh_bytes!(enc, Base.unsafe_convert(Ptr{Float32}, r),
                                      sizeof(Float32), slot)
end

@testset "a mesh stage drawn by Julia shaders" begin
    # A mesh pipeline's fragment stage takes ONE `stage_in` struct nesting both planes.
    # Spelled as two parameters, `(VO v [[stage_in]], PO p)`, it still compiles and
    # still builds a pipeline, and `p` is then an unbound buffer that reads as zeros:
    # a black frame that looks exactly like a mesh stage which emitted nothing.
    msl = MTL.MTLLibrary(GFX_DEV, """
    #include <metal_stdlib>
    using namespace metal;
    struct VO { float4 position [[position]]; float2 uv; };
    struct PO { float4 colour; };
    using MeshT = metal::mesh<VO, PO, 4, 2, metal::topology::triangle>;
    [[mesh]] void ref_mesh(MeshT out) {
        VO v;
        v.position = float4(-1, -1, 0, 1); v.uv = float2(0, 0); out.set_vertex(0, v);
        v.position = float4( 3, -1, 0, 1); v.uv = float2(2, 0); out.set_vertex(1, v);
        v.position = float4(-1,  3, 0, 1); v.uv = float2(0, 2); out.set_vertex(2, v);
        out.set_index(0, 0); out.set_index(1, 1); out.set_index(2, 2);
        PO p; p.colour = float4(0, 1, 0, 1); out.set_primitive(0, p);
        out.set_primitive_count(1);
    }
    struct FSIn { VO v; PO p; };
    fragment float4 ref_mesh_fs(FSIn in [[stage_in]]) {
        return float4(in.v.uv.x * 0.0, in.p.colour.g, in.v.uv.y * 0.0, 1.0);
    }
    """)
    ref_ms = MTL.MTLFunction(msl, "ref_mesh")
    ref_fs = MTL.MTLFunction(msl, "ref_mesh_fs")
    @test ref_ms.functionType == MTL.MTLFunctionTypeMesh

    jl_ms = stage_function(gfx_mesh_tri, Tuple{MeshOut}, :mesh, "gfx_mesh_tri")
    jl_fs = stage_function(gfx_mesh_fragment, MeshFragTT, :fragment, "gfx_mesh_fragment")
    # The driver's own word on what it loaded: a mesh program, not a kernel that
    # happens to be tagged as one.
    @test jl_ms.functionType == MTL.MTLFunctionTypeMesh

    refpx = draw_mesh(ref_ms, ref_fs)
    # A fullscreen triangle covers the frame. Blank fails here, and so does a target
    # that only partly covers because a position went somewhere else.
    @test greens(refpx) == 64 * 64

    for (label, m, f) in (("Julia mesh", jl_ms, ref_fs),
                          ("Julia fragment", ref_ms, jl_fs),
                          ("both", jl_ms, jl_fs))
        px = draw_mesh(m, f)
        @test (label, greens(px)) == (label, 64 * 64)
        @test (label, px == refpx) == (label, true)
    end
end

@testset "a mesh stage's buffers are numbered from zero" begin
    # The object is NOT a buffer and consumes no binding, so a mesh stage's own
    # arguments start at Metal slot 0 exactly like a vertex stage's. MSL agrees:
    # `[[mesh]] void f(MeshT out, device uint *p [[buffer(0)]])` binds at 0.
    jl_ms = stage_function(gfx_mesh_scaled, Tuple{MeshOut, Float32}, :mesh, "gfx_mesh_scaled")
    jl_fs = stage_function(gfx_mesh_fragment, MeshFragTT, :fragment, "gfx_mesh_fragment")

    # Slot 1 here is Metal slot 0, and it is where the stage's first argument goes.
    @test greens(draw_mesh(jl_ms, jl_fs; bind! = bind_extent(1f0, 1))) == 64 * 64
    # One past it binds nothing, so the extent reads as nothing and the triangle
    # collapses. Pinned because it is the failure the numbering causes.
    @test greens(draw_mesh(jl_ms, jl_fs; bind! = bind_extent(1f0, 2))) == 0

    # And the argument is genuinely read rather than constants folded in: a smaller
    # extent covers strictly fewer pixels.
    full    = greens(draw_mesh(jl_ms, jl_fs; bind! = bind_extent(1f0, 1)))
    half    = greens(draw_mesh(jl_ms, jl_fs; bind! = bind_extent(0.5f0, 1)))
    quarter = greens(draw_mesh(jl_ms, jl_fs; bind! = bind_extent(0.25f0, 1)))
    @test full > half > quarter > 0
end

# ── the data intrinsics take the FIELD before the SLOT ───────────────────────
#
# Four vertices with four different `uv`s and two triangles with two different
# per-primitive colours, graded against the same thing in MSL. With the operand
# order swapped the interpolated `uv` collapsed to a ramp falling away from vertex 0
# and the second triangle came out black (`intrinsics/mesh.jl`).

function gfx_mesh_quad(out::MeshOut)
    set_position_mesh(out, Int32(0), (-1f0, -1f0, 0f0, 1f0))
    set_position_mesh(out, Int32(1), ( 1f0, -1f0, 0f0, 1f0))
    set_position_mesh(out, Int32(2), (-1f0,  1f0, 0f0, 1f0))
    set_position_mesh(out, Int32(3), ( 1f0,  1f0, 0f0, 1f0))
    set_vertex_data_mesh(out, Int32(0), Int32(0), (0f0, 0f0))
    set_vertex_data_mesh(out, Int32(0), Int32(1), (1f0, 0f0))
    set_vertex_data_mesh(out, Int32(0), Int32(2), (0f0, 1f0))
    set_vertex_data_mesh(out, Int32(0), Int32(3), (1f0, 1f0))
    set_index_mesh(out, Int32(0), UInt8(0))
    set_index_mesh(out, Int32(1), UInt8(1))
    set_index_mesh(out, Int32(2), UInt8(2))
    set_index_mesh(out, Int32(3), UInt8(2))
    set_index_mesh(out, Int32(4), UInt8(1))
    set_index_mesh(out, Int32(5), UInt8(3))
    set_primitive_data_mesh(out, Int32(0), Int32(0), (0f0, 1f0, 0f0, 1f0))
    set_primitive_data_mesh(out, Int32(0), Int32(1), (0f0, 1f0, 1f0, 1f0))
    set_primitive_count_mesh(out, Int32(2))
    return nothing
end

# `uv` in red and green, the per-primitive blue in blue: one fragment stage that
# shows both planes at once, and WHICH vertex and which triangle each value came from.
function gfx_mesh_quad_fragment(uv::Varying{:uv, NTuple{2,Float32}},
                                colour::Varying{:colour, NTuple{4,Float32}},
                                out::StageOut{GfxFOut})
    u = uv.value
    c = colour.value
    Base.unsafe_store!(out, GfxFOut((u[1], u[2], c[3], 1f0)))
    return nothing
end

@testset "a mesh stage's data planes are addressed per field and per slot" begin
    msl = MTL.MTLLibrary(GFX_DEV, """
    #include <metal_stdlib>
    using namespace metal;
    struct VO { float4 position [[position]]; float2 uv; };
    struct PO { float4 colour; };
    using MeshT = metal::mesh<VO, PO, 4, 2, metal::topology::triangle>;
    [[mesh]] void ref_quad(MeshT out) {
        VO v;
        v.position = float4(-1, -1, 0, 1); v.uv = float2(0, 0); out.set_vertex(0, v);
        v.position = float4( 1, -1, 0, 1); v.uv = float2(1, 0); out.set_vertex(1, v);
        v.position = float4(-1,  1, 0, 1); v.uv = float2(0, 1); out.set_vertex(2, v);
        v.position = float4( 1,  1, 0, 1); v.uv = float2(1, 1); out.set_vertex(3, v);
        out.set_index(0, 0); out.set_index(1, 1); out.set_index(2, 2);
        out.set_index(3, 2); out.set_index(4, 1); out.set_index(5, 3);
        PO p;
        p.colour = float4(0, 1, 0, 1); out.set_primitive(0, p);
        p.colour = float4(0, 1, 1, 1); out.set_primitive(1, p);
        out.set_primitive_count(2);
    }
    struct FSIn { VO v; PO p; };
    fragment float4 ref_quad_fs(FSIn in [[stage_in]]) {
        return float4(in.v.uv.x, in.v.uv.y, in.p.colour.b, 1.0);
    }
    """)
    ref_ms = MTL.MTLFunction(msl, "ref_quad")
    ref_fs = MTL.MTLFunction(msl, "ref_quad_fs")
    jl_ms = stage_function(gfx_mesh_quad, Tuple{MeshOut}, :mesh, "gfx_mesh_quad")
    jl_fs = stage_function(gfx_mesh_quad_fragment, MeshFragTT, :fragment,
                           "gfx_mesh_quad_fragment")

    refpx = draw_mesh(ref_ms, ref_fs)
    jlpx  = draw_mesh(jl_ms, jl_fs)

    # What the reference itself has to show, so this testset says what it means even
    # when the two agree on something wrong: `uv` sweeps nearly the whole range in
    # both directions, and the two triangles differ in blue. Nearly, because a pixel
    # is sampled at its CENTRE and the outermost centre is half a pixel inside the
    # quad.
    reds   = [refpx[4i + 1] for i in 0:(64 * 64 - 1)]
    gs     = [refpx[4i + 2] for i in 0:(64 * 64 - 1)]
    blues  = [refpx[4i + 3] for i in 0:(64 * 64 - 1)]
    @test minimum(reds) < 0x08 && maximum(reds) > 0xf0
    @test minimum(gs)   < 0x08 && maximum(gs)   > 0xf0
    @test Set(unique(blues)) == Set([0x00, 0xff])
    # About half the quad each: the diagonal splits it, and which side the pixels ON
    # the diagonal land in is a fill rule.
    @test count(==(0xff), blues) ≈ count(==(0x00), blues) rtol = 0.05

    # …and then the whole frame, byte for byte.
    @test jlpx == refpx
end

# ── a mesh stage is a compute stage that rasterises ──────────────────────────

"""
Every thread records its own indices, so a collapsed index cannot hide. It also
emits a triangle, because a mesh stage that emits NOTHING is dropped whole once the
fragment stage reads its planes, side effects and all.
"""
function gfx_mesh_indices(out::MeshOut, probe::Metal.MtlDeviceVector{UInt32,1})
    tid = Metal.thread_index_in_threadgroup()          # 1-based
    gid = Metal.threadgroup_position_in_grid().x       # 1-based
    slot = (Int(gid) - 1) * 8 + Int(tid)
    @inbounds probe[slot] = UInt32(gid) * UInt32(100) + UInt32(tid)
    @inbounds probe[33] = Metal.threads_per_threadgroup().x
    tid == UInt32(1) && emit_triangle!(out, 1f0)
    return nothing
end

"""
Four threads meet in threadgroup memory: three write a corner, one reads all three
back and emits the triangle. A barrier that does not hold, or threadgroup memory a
mesh stage cannot have, gives a degenerate triangle and no coverage.
"""
function gfx_mesh_cooperative(out::MeshOut)
    corners = Metal.MtlThreadGroupArray(NTuple{4,Float32}, 4)
    tid = Metal.thread_index_in_threadgroup()
    x = tid == UInt32(1) ? -1f0 : (tid == UInt32(2) ?  3f0 : -1f0)
    y = tid == UInt32(1) ? -1f0 : (tid == UInt32(2) ? -1f0 :  3f0)
    tid <= UInt32(3) && (@inbounds corners[tid] = (x, y, 0f0, 1f0))
    Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)
    if tid == UInt32(1)
        @inbounds set_position_mesh(out, Int32(0), corners[1])
        @inbounds set_position_mesh(out, Int32(1), corners[2])
        @inbounds set_position_mesh(out, Int32(2), corners[3])
        set_vertex_data_mesh(out, Int32(0), Int32(0), (0f0, 0f0))
        set_vertex_data_mesh(out, Int32(0), Int32(1), (2f0, 0f0))
        set_vertex_data_mesh(out, Int32(0), Int32(2), (0f0, 2f0))
        set_index_mesh(out, Int32(0), UInt8(0))
        set_index_mesh(out, Int32(1), UInt8(1))
        set_index_mesh(out, Int32(2), UInt8(2))
        set_primitive_data_mesh(out, Int32(0), Int32(0), (0f0, 1f0, 0f0, 1f0))
        set_primitive_count_mesh(out, Int32(1))
    end
    return nothing
end

@testset "a mesh stage gets the compute builtins" begin
    # When the index builtins are missing they do not fail: they read zero, every
    # thread does the first thread's work, and the stage behaves like a
    # single-threaded one that happens to write the same slots repeatedly.
    probe = Metal.MtlArray{UInt32,1,Metal.SharedStorage}(fill(typemax(UInt32), 40))
    # The draw runs on a queue of its own, which nothing orders after the upload.
    Metal.synchronize()
    dprobe = Metal.mtlconvert(probe)
    jl_ms = stage_function(gfx_mesh_indices, Tuple{MeshOut, typeof(dprobe)},
                           :mesh, "gfx_mesh_indices")
    jl_fs = stage_function(gfx_mesh_fragment, MeshFragTT, :fragment, "gfx_mesh_fragment")
    bind! = enc -> begin
        # The array is reached through the address in its device-side struct, so its
        # buffer has to be made resident; a non-resident one reads and writes nothing.
        MB.use!(enc, Mantle.metal_buffer(probe), MTL.ReadWriteUsage, MTL.MTLRenderStageMesh)
        r = Ref(dprobe)
        GC.@preserve r MB.set_mesh_bytes!(enc, Base.unsafe_convert(Ptr{typeof(dprobe)}, r),
                                          sizeof(dprobe), 1)
    end
    draw_mesh(jl_ms, jl_fs; bind!, threads = 4, groups = 3)
    r = Array(probe)

    # Twelve invocations, each with its own (group, thread), and gid*100+tid is unique
    # per pair, so a collapsed index shows up as a missing slot.
    seen = [(g, t, r[(g - 1) * 8 + t]) for g in 1:3, t in 1:4]
    @test all(x -> x[3] == UInt32(x[1] * 100 + x[2]), seen)
    @test length(unique(x[3] for x in seen)) == 12
    @test r[33] == UInt32(4)      # threads_per_threadgroup, from the descriptor
end

@testset "a mesh stage gets threadgroup memory and a barrier" begin
    jl_ms = stage_function(gfx_mesh_cooperative, Tuple{MeshOut}, :mesh, "gfx_mesh_cooperative")
    jl_fs = stage_function(gfx_mesh_fragment, MeshFragTT, :fragment, "gfx_mesh_fragment")
    @test greens(draw_mesh(jl_ms, jl_fs; threads = 4)) == 64 * 64
end

# ── sampling a bound texture ─────────────────────────────────────────────────
#
# The AGX compiler recognises a texture argument by its POINTEE's struct name, and
# LLVM 22 has no pointee to give it. `compiler/texture.jl` records how it is put back
# and what the alternative was: handed `{} addrspace(1)*`, the compiler service
# SEGFAULTS, and `MTLRenderPipelineState` reports it as
# `XPC_ERROR_CONNECTION_INTERRUPTED` while naming nothing. Both halves are pinned
# here: the shape of the AIR, and a pipeline actually built from it.

function gfx_tex_fragment(pos::FragCoord, tex::Texture2DPtr{Float32}, samp::SamplerPtr,
                          out::StageOut{GfxFOut})
    s = air_sample_texture_2d(tex, samp, 0.25f0, 0.25f0)
    Base.unsafe_store!(out, GfxFOut((s.value[1].value, s.value[2].value,
                                     s.value[3].value, 1f0)))
    return nothing
end

const GFX_TEX_TT = Tuple{FragCoord, Texture2DPtr{Float32}, SamplerPtr, StageOut{GfxFOut}}

"""The AIR a stage is handed to the driver as, parsed back out of its metallib."""
function stage_air_text(f, tt, stage::Symbol, name::String)
    job = stage_job(GFX_DEV, f, tt, stage, name)
    bytes = Metal.compile_to_metallib(job).metallib
    fn = only(read(IOBuffer(bytes), Metal.MetalLib).functions)
    LLVM.@dispose ctx = LLVM.Context() begin
        return string(parse(LLVM.Module, fn.air_module))
    end
end

"""
A 2x2 sampled texture with a distinct colour per texel, uploaded through Mantle's
`copytexels!`, and a nearest-filtering sampler. Texel (0, 0) is green, and
(0.25, 0.25) samples it.
"""
function sampled_green_texture()
    tex = Mantle.maketexture(GFX_MDEV, UInt32, MTL.MTLPixelFormatRGBA8Unorm, 2, 2, 1)
    # Little-endian RGBA8; one Julia column is one image row.
    green, red, blue, yellow = 0xff00ff00, 0xff0000ff, 0xffff0000, 0xff00ffff
    Mantle.copytexels!(tex, UInt32[green blue; red yellow])
    return tex, Mantle.makesampler(GFX_MDEV, :nearest, :clamp, 1f0)
end

@testset "a texture argument reaches AIR as an opaque texture pointer" begin
    air = stage_air_text(gfx_tex_fragment, GFX_TEX_TT, :fragment, "gfx_tex_fragment")

    # The two handle types, BODYLESS. A struct with a body (`type {}` or
    # `type { i8 }`) puts the crash straight back; only `opaque` builds.
    @test occursin("%struct._texture_2d_t = type opaque", air)
    @test occursin("%struct._sampler_t = type opaque", air)

    entry = only(filter(l -> startswith(l, "define"), split(air, '\n')))
    # The parameter is a pointer to the texture handle type, in the texture address
    # space, and the pointee is still named.
    @test occursin("ptr addrspace(1) byref(%struct._texture_2d_t)", entry)
    @test occursin("ptr addrspace(2) byref(%struct._sampler_t)", entry)
    # …and NOT a pointee it could not work out, which is what the parameters degrade
    # to the moment the `byref` is lost.
    @test !occursin("byref({})", entry)

    # The argument metadata names them as a texture and a sampler rather than as the
    # buffers the kernel ABI made them, each with its own location namespace.
    @test occursin("!\"air.texture\"", air)
    @test occursin("!\"air.sampler\"", air)
    @test occursin("texture2d<float, sample>", air)
end

@testset "a fragment stage samples a bound texture" begin
    tex, samp = sampled_green_texture()
    vsf = stage_function(gfx_vertex, GFX_VERTEX_TT, :vertex, "gfx_vertex")
    # Building the pipeline state is the step that used to kill the compiler service,
    # so reaching the draw at all is half the assertion.
    fsf = stage_function(gfx_tex_fragment, GFX_TEX_TT, :fragment, "gfx_tex_fragment")
    _, px = draw_triangle(vsf, fsf; bind! = enc -> begin
        MB.set_fragment_texture!(enc, tex.tex, 1)
        MB.set_fragment_sampler!(enc, samp.state, 1)
    end)
    # The texel at (0, 0) is green, and the triangle covers the middle of the target,
    # so a covered pixel carries the SAMPLE and not the clear colour.
    W = H = 64
    mid = ((H ÷ 2) * W + (W ÷ 2)) * 4
    @test px[mid + 1] == 0x00 && px[mid + 2] == 0xff && px[mid + 3] == 0x00
    @test greens(px) > 500
end

# ── sampling next to a by-reference argument ────────────────────────────────
#
# The same `XPC_ERROR_CONNECTION_INTERRUPTED`, from another cause: a stage that
# samples AND loads a by-reference argument whose load carries Julia's alias scopes
# (`drop_alias_scopes!` in `compiler/texture.jl`). RayMakie's image fragment, which
# samples and reads its packed arguments, failed exactly so.

struct GfxTexArgs
    args::Tuple{NTuple{2,Float32},Float32,Int32}
end

function gfx_tex_args_fragment(a::GfxTexArgs, pos::FragCoord, tex::Texture2DPtr{Float32},
                               samp::SamplerPtr, out::StageOut{GfxFOut})
    s = air_sample_texture_2d(tex, samp, 0.25f0, 0.25f0)
    # Blue from the argument, so the draw shows the load ran and read the right field.
    Base.unsafe_store!(out, GfxFOut((s.value[1].value, s.value[2].value, a.args[2], 1f0)))
    return nothing
end

const GFX_TEX_ARGS_TT = Tuple{GfxTexArgs, FragCoord, Texture2DPtr{Float32}, SamplerPtr,
                              StageOut{GfxFOut}}

@testset "a stage that samples and reads a by-reference argument" begin
    air = stage_air_text(gfx_tex_args_fragment, GFX_TEX_ARGS_TT, :fragment,
                         "gfx_tex_args_fragment")
    # The argument is loaded, and no load carries a scope.
    @test any(l -> occursin("= load ", l), split(air, '\n'))
    @test !occursin("!alias.scope", air)
    @test !occursin("!noalias", air)

    tex, samp = sampled_green_texture()
    vsf = stage_function(gfx_vertex, GFX_VERTEX_TT, :vertex, "gfx_vertex")
    # The step that crashed the compiler service.
    fsf = stage_function(gfx_tex_args_fragment, GFX_TEX_ARGS_TT, :fragment,
                         "gfx_tex_args_fragment")
    args = Ref(GfxTexArgs(((0f0, 0f0), 1f0, Int32(7))))
    _, px = draw_triangle(vsf, fsf; bind! = enc -> begin
        GC.@preserve args MB.set_fragment_bytes!(
            enc, Base.unsafe_convert(Ptr{GfxTexArgs}, args), sizeof(GfxTexArgs), 1)
        MB.set_fragment_texture!(enc, tex.tex, 1)
        MB.set_fragment_sampler!(enc, samp.state, 1)
    end)
    # Green from the texel, blue from the argument.
    W = H = 64
    mid = ((H ÷ 2) * W + (W ÷ 2)) * 4
    @test px[mid + 1] == 0x00 && px[mid + 2] == 0xff && px[mid + 3] == 0xff
end

# ── what the wrappers own ────────────────────────────────────────────────────

@testset "heap-placed textures are released" begin
    # `newTextureWithDescriptor:offset:` returns a +1 reference, and a placed texture
    # retains its heap: a texture that was never released kept the heap alive too.
    texdesc = MTL.MTLTextureDescriptor(MTL.MTLPixelFormatRGBA8Unorm, 64, 64)
    texdesc.storageMode = MTL.MTLStorageModePrivate
    placement = MTL.MTLHeapDescriptor()
    placement.type = MTL.MTLHeapTypePlacement
    placement.storageMode = MTL.MTLStorageModePrivate
    placement.size = MB.heap_texture_size_and_align(GFX_DEV, texdesc).size
    let heap = MTL.MTLHeap(GFX_DEV, placement)
        place_and_drop(n) = (for _ in 1:n; MB.heap_texture(heap, texdesc, 0); end)
        place_and_drop(3)
        GC.gc(true)
        @test heap.retainCount == 1
    end
end

@testset "layers and their drawables are released" begin
    # `[CAMetalLayer layer]` and `nextDrawable` return autoreleased objects; wrapped
    # as raw `id`s they were never released, and every drawable kept the layer alive.
    let layer = MB.CAMetalLayer(GFX_DEV, 64, 64)
        take_and_drop(n) = (for _ in 1:n; MB.next_drawable(layer); end)
        take_and_drop(3)
        GC.gc(true)
        @test layer.retainCount == 1
    end
end

end # module TestGraphicsMetal
