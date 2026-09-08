# The stages a pipeline is made of, and the interfaces between them.
#
# No GPU: everything here is a description or a name, and that it needs no
# device is part of what is being asserted. The mesh emitter's arithmetic is
# tested where it lives, in `KernelInterface`'s suite, over `HostMeshOutput`.

const M = Mantle
# Not `import KernelInterface`: the assertion is that these names come from ONE
# module below Mantle, so the reference is taken from where a name actually
# lives rather than from a second import that could resolve elsewhere.
const STAGE_KI = parentmodule(M.MeshEmitter)

@testset "pipeline stages" begin
    V4 = NTuple{4,Float32}
    V2 = NTuple{2,Float32}

    @testset "Flat partitions an interface list" begin
        iface = (uv = V2, halfwidth = M.Flat{Float32}, color = M.Flat{V4})
        @test M.smoothnames(iface) == (:uv,)
        @test M.flatnames(iface) == (:halfwidth, :color)
        # The order within each side is the DECLARATION order, because that is
        # what fixes the layout a backend assigns.
        @test (M.smoothnames(iface)..., M.flatnames(iface)...) ==
              (:uv, :halfwidth, :color)
        # Flatness decides WHERE a value goes, never what it is.
        @test M.valuetypes(iface) == (uv = V2, halfwidth = Float32, color = V4)
        @test M.unflat(M.Flat{Float32}) === Float32
        @test M.unflat(Float32) === Float32
        @test M.isflat(M.Flat{Float32})
        @test !M.isflat(Float32)
    end

    @testset "a stage declares only its outputs" begin
        vs = M.VertexShader(identity; outputs = (uv = V2, tint = M.Flat{V4}))
        @test M.stagefunction(vs) === identity
        @test M.stageoutputs(vs) == (uv = V2, tint = M.Flat{V4})
        @test M.smoothoutputs(vs) == (:uv,)
        @test M.flatoutputs(vs) == (:tint,)
        # `position` is prepended by `outputtype` and never declared, and it is a
        # `Vec4f` because a clip position IS one.
        @test M.outputtype(vs) === @NamedTuple{position::Vec4f, uv::V2, tint::V4}
        # The consumer's inputs ARE the producer's outputs, so there is one
        # declaration and nothing to disagree with.
        @test M.stageinputs(vs) === M.stageoutputs(vs)

        # A stage with no outputs is a real stage: a shadow pass writes only its
        # clip position.
        @test M.outputtype(M.VertexShader(identity)) === @NamedTuple{position::Vec4f}
        @test M.stageoutputs(M.FragmentShader(identity)) == (;)
    end

    @testset "the configs are absorbed" begin
        gs = M.GeometryShader(identity; outputs = (uv = V2,),
                              input = M.LineStripAdjacency(),
                              output = M.TriangleStrip(), max_vertices = 4)
        c = M.stageconfig(gs)
        @test c isa M.GeometryConfig
        @test c.input_topology === M.LineStripAdjacency()
        @test c.output_topology === M.TriangleStrip()
        @test c.max_vertices == 4
        @test c.invocations == 1
        # The topology is a type parameter because a compiler dispatches on it.
        @test c isa M.GeometryConfig{M.LineStripAdjacency, M.TriangleStrip}

        ms = M.MeshShader(identity; outputs = (uv = V2,), max_vertices = 4,
                          max_primitives = 2, topology = M.TriangleStrip(),
                          threads = 32)
        @test M.stageconfig(ms) isa M.MeshConfig{M.TriangleStrip}
        @test M.stageconfig(ms).threads == 32

        @test M.stageconfig(M.ObjectShader(identity; threads = 8)).threads == 8
        # The two stages that have nothing to configure say so.
        @test M.stageconfig(M.VertexShader(identity)) === nothing
        @test M.stageconfig(M.FragmentShader(identity)) === nothing
    end

    # A mesh output object addresses its vertices with 8 bits, which is a
    # correctness bound and not a style choice.
    @testset "the mesh vertex ceiling is enforced" begin
        @test M.MeshShader(identity; max_vertices = 256, max_primitives = 1) isa M.MeshShader
        @test_throws ArgumentError M.MeshShader(identity; max_vertices = 257,
                                                          max_primitives = 1)
    end

    # The reason `varyings` had to go: it named ONE interface, and a pipeline
    # with a geometry stage has two.
    @testset "two interfaces, each declared once" begin
        vs = M.VertexShader(identity; outputs = (colour = V4, thickness = Float32))
        gs = M.GeometryShader(identity; outputs = (uv = V2, halfwidth = M.Flat{Float32}),
                              input = M.LineStripAdjacency(),
                              output = M.TriangleStrip(), max_vertices = 4)
        fs = M.FragmentShader(identity)

        flat = M.GraphicsPipeline(; vertex = vs, fragment = fs)
        @test M.lastgeometrystage(flat) === vs
        @test M.fragmentinputs(flat) == (colour = V4, thickness = Float32)

        withgeom = M.GraphicsPipeline(; vertex = vs, geometry = gs, fragment = fs)
        # The vertex stage feeds the GEOMETRY stage, and the fragment stage reads
        # the geometry stage. Two different sets, and a single list could not
        # have said both.
        @test M.lastgeometrystage(withgeom) === gs
        @test M.fragmentinputs(withgeom) == (uv = V2, halfwidth = M.Flat{Float32})
        @test M.outputtype(withgeom.vertex) ===
              @NamedTuple{position::Vec4f, colour::V4, thickness::Float32}
        # `position` is NOT a fragment input: a fragment stage reads its position
        # through `frag_coord`, which the rasteriser supplies.
        @test M.fragmentinputtype(withgeom) === @NamedTuple{uv::V2, halfwidth::Float32}
        @test !(:position in keys(M.fragmentinputs(withgeom)))
    end

    @testset "the stages are type-checked at construction" begin
        vs = M.VertexShader(identity)
        fs = M.FragmentShader(identity)
        # A bare function is not a stage. Catching it here beats a MethodError
        # from inside a shader compile.
        @test_throws TypeError M.GraphicsPipeline(; vertex = identity, fragment = fs)
        @test_throws TypeError M.GraphicsPipeline(; vertex = vs, fragment = identity)
        @test_throws TypeError M.GraphicsPipeline(; vertex = vs, fragment = fs,
                                                   geometry = identity)
    end

    @testset "MeshPipeline" begin
        ms = M.MeshShader(identity; outputs = (uv = V2, tint = M.Flat{V4}),
                          max_vertices = 4, max_primitives = 2,
                          topology = M.TriangleStrip(), threads = 32)
        p = M.MeshPipeline(; mesh = ms, fragment = M.FragmentShader(identity))
        @test M.meshconfig(p) === M.stageconfig(ms)
        @test M.meshconfig(p).topology === M.TriangleStrip()
        @test M.objectconfig(p) === nothing
        @test M.fragmentinputs(p) == (uv = V2, tint = M.Flat{V4})
        @test M.fragmentinputtype(p) === @NamedTuple{uv::V2, tint::V4}
        @test p.blend === M.Opaque()

        q = M.MeshPipeline(; mesh = ms, fragment = M.FragmentShader(identity),
                             object = M.ObjectShader(identity; threads = 8))
        @test M.objectconfig(q).threads == 8

        @test_throws TypeError M.MeshPipeline(; mesh = identity,
                                               fragment = M.FragmentShader(identity))
    end

    # There is deliberately no `topology` on a MeshPipeline: a GraphicsPipeline's
    # describes the INPUT stream an assembler groups, and a mesh stage has none.
    # What it emits is its MeshShader's.
    @testset "no second topology, and no varyings anywhere" begin
        @test !hasfield(M.MeshPipeline, :topology)
        @test hasfield(M.GraphicsPipeline, :topology)
        @test !hasfield(M.GraphicsPipeline, :varyings)
        @test !hasfield(M.MeshPipeline, :varyings)
    end

    @testset "capability" begin
        # `false` by default, so a backend opts in, and a caller with a geometry
        # body can ask both questions before building anything.
        @test M.supports_mesh_pipeline(nothing) === false
        @test M.supports_geometry_stage(nothing) === false
        @test :supports_mesh_pipeline in M.BACKEND_VOCABULARY
    end

    # The names a shader body and a pipeline description use have to come from
    # `using Mantle` alone, or the rule that a downstream package names no
    # backend cannot hold.
    @testset "vocabulary is re-exported" begin
        for n in (:ShaderStage, :VertexShader, :FragmentShader, :GeometryShader,
                  :MeshShader, :ObjectShader, :stagefunction, :stageoutputs,
                  :stageinputs, :stageconfig, :flatoutputs, :smoothoutputs,
                  :outputtype, :lastgeometrystage, :fragmentinputs,
                  :fragmentinputtype, :Flat, :isflat, :unflat, :GeometryConfig,
                  :MeshConfig, :ObjectConfig, :MeshPipeline, :GraphicsPipeline,
                  :NativeEmitter, :MeshEmitter, :emit!, :endprimitive!,
                  :set_mesh_vertex!, :mesh_thread_index)
            @test n in names(Mantle)
        end
        # And the device-side half is KernelInterface's, not a copy: a second
        # declaration here is what the graphics builtins used to be, bridged and
        # drifting.
        for n in (:Flat, :MeshConfig, :GeometryConfig, :NativeEmitter, :MeshEmitter,
                  :emit!, :endprimitive!, :set_mesh_vertex!, :mesh_thread_index)
            @test getproperty(Mantle, n) === getproperty(STAGE_KI, n)
        end
    end

    # `emit!` was Mantle's graph walk over a plan's passes, which is now
    # `emitplan!`. Two unrelated meanings of one name in one module is one too
    # many, and the shader-facing one has to win because it is exported and a
    # body writes it. If a graph method comes back under this name, a geometry
    # body silently stops dispatching.
    @testset "emit! is the emitter's, not the graph's" begin
        @test isdefined(Mantle, :emitplan!)
        for m in methods(Mantle.emit!)
            @test parentmodule(m) === STAGE_KI
            @test m.sig.parameters[2] <: M.PrimitiveEmitter
        end
    end
end
