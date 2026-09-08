# The mesh pipeline's description, and the vocabulary a shader body reaches it
# through.
#
# No GPU: everything here is a description or a name, and that it needs no
# device is part of what is being asserted. The emitter's arithmetic is tested
# where it lives, in `KernelInterface`'s suite, over `HostMeshOutput`.

const M = Mantle
# Not `import KernelInterface`: the assertion is that these names come from ONE
# module below Mantle, so the reference is taken from where a name actually
# lives rather than from a second import that could resolve elsewhere.
const MESH_KI = parentmodule(M.MeshEmitter)

@testset "MeshPipeline" begin
    mcfg = M.MeshConfig(max_vertices = 4, max_primitives = 2,
                        topology = M.TriangleStrip(), threads = 32)

    @testset "description" begin
        p = M.MeshPipeline(mesh = (identity, mcfg), fragment = identity,
                           varyings = (uv = NTuple{2,Float32}, color = NTuple{4,Float32}))
        @test M.meshconfig(p) === mcfg
        @test M.meshconfig(p).topology === M.TriangleStrip()
        @test M.objectconfig(p) === nothing
        @test p.varyings === (uv = NTuple{2,Float32}, color = NTuple{4,Float32})
        # The fixed-function state is the same vocabulary `GraphicsPipeline` takes.
        @test p.blend === M.Opaque()
        @test p.cull === M.CullBack()
        @test p.depth === M.DepthLess()

        ocfg = M.ObjectConfig(threads = 8)
        q = M.MeshPipeline(mesh = (identity, mcfg), fragment = identity,
                           object = (identity, ocfg))
        @test M.objectconfig(q) === ocfg
    end

    # A stage given in the wrong shape is caught where the pipeline is BUILT. The
    # alternative is a MethodError from inside a shader compile, which is the
    # wrong place and the wrong time to learn it.
    @testset "the shape is checked at construction" begin
        @test_throws ArgumentError M.MeshPipeline(mesh = identity, fragment = identity)
        @test_throws ArgumentError M.MeshPipeline(mesh = (identity, 4), fragment = identity)
        @test_throws ArgumentError M.MeshPipeline(mesh = (identity, mcfg),
                                                  fragment = identity, object = identity)
        @test_throws ArgumentError M.MeshPipeline(mesh = (identity, mcfg),
                                                  fragment = identity,
                                                  object = (identity, mcfg))
    end

    # There is deliberately no `topology` field: a `GraphicsPipeline`'s describes
    # the INPUT stream the assembler groups, and a mesh stage has no input stream.
    # What it emits is `MeshConfig`'s.
    @testset "no second topology" begin
        @test !hasfield(M.MeshPipeline, :topology)
        @test hasfield(M.GraphicsPipeline, :topology)
    end

    @testset "capability" begin
        # `false` by default, so a backend opts in, and a caller with a geometry
        # body can ask both questions before building anything.
        @test M.supports_mesh_pipeline(nothing) === false
        @test M.supports_geometry_stage(nothing) === false
        # In the vocabulary list, because a backend extends it.
        @test :supports_mesh_pipeline in M.BACKEND_VOCABULARY
    end

    # The names a shader body uses have to come from `using Mantle` alone, or the
    # rule that a downstream package names no backend cannot hold.
    @testset "vocabulary is re-exported" begin
        for n in (:MeshConfig, :ObjectConfig, :PrimitiveEmitter, :NativeEmitter,
                  :MeshEmitter, :emit!, :endprimitive!, :set_mesh_vertex!,
                  :set_mesh_triangle!, :set_mesh_line!, :set_mesh_point!,
                  :set_mesh_outputs!, :set_mesh_groups!, :mesh_thread_index,
                  :mesh_group_index, :MeshPipeline, :meshconfig, :objectconfig,
                  :supports_mesh_pipeline)
            @test n in names(Mantle)
        end
        # And they are KernelInterface's, not copies: a second declaration here
        # is what the graphics builtins used to be, bridged and drifting.
        for n in (:MeshConfig, :NativeEmitter, :MeshEmitter, :emit!, :endprimitive!,
                  :set_mesh_vertex!, :mesh_thread_index)
            @test getproperty(Mantle, n) === getproperty(MESH_KI, n)
        end
    end

    # `emit!` was Mantle's graph walk over a plan's passes, which is now
    # `emitplan!`. Two unrelated meanings of one name in one module is one too
    # many, and the shader-facing one has to win because it is exported and a
    # body writes it. If a graph method comes back under this name, a geometry
    # body silently stops dispatching.
    @testset "emit! is the emitter's, not the graph's" begin
        @test isdefined(Mantle, :emitplan!)
        @test any(m -> m.name === :emitplan!, methods(Mantle.emitplan!))
        for m in methods(Mantle.emit!)
            @test parentmodule(m) === MESH_KI
            @test m.sig.parameters[2] <: M.PrimitiveEmitter
        end
    end
end
