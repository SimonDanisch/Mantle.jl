using Test, Mantle
using ColorTypes: BGRA
using ColorTypes.FixedPointNumbers: N0f8

function software_mesh_test_shader(out, selection)
    lane = mesh_thread_index()
    if mesh_group_index() == @inbounds selection[1]
        if lane == 1
            set_mesh_vertex!(out, 1, (position = (-1f0, -1f0, 0f0, 1f0), weight = 1f0))
            set_mesh_vertex!(out, 3, (position = (-1f0,  3f0, 0f0, 1f0), weight = 1f0))
            set_mesh_triangle!(out, 1, 1, 2, 3)
            set_mesh_primitive_data!(out, 1, (tint = (0f0, 1f0, 0f0, 1f0),))
            set_mesh_outputs!(out, 3, 1)
        else
            set_mesh_vertex!(out, 2, (position = (3f0, -1f0, 0f0, 1f0), weight = 1f0))
        end
    elseif lane == 1
        set_mesh_outputs!(out, 0, 0)
    end
    return nothing
end

software_mesh_test_fragment(inputs, gain) =
    (inputs.tint[1], inputs.tint[2] * inputs.weight * gain, inputs.tint[3], inputs.tint[4])

@testset "software mesh shading" begin
    dev = Mantle.defaultbackend()
    selection = Mantle.Buffer(dev, Int32[2])
    mesh = Mantle.MeshShader(software_mesh_test_shader;
        outputs = (weight = Float32, tint = Mantle.Flat{NTuple{4,Float32}}),
        max_vertices = 3, max_primitives = 2, threads = 2)
    pipeline = Mantle.MeshPipeline(; mesh, fragment = Mantle.FragmentShader(software_mesh_test_fragment),
                                   cull = Mantle.NoCull(), depth = Mantle.DepthOff())
    fallback = Mantle.uses_mesh_fallback(dev, pipeline)
    get(ENV, "MANTLE_TEST_SOFTWARE_MESH", "false") == "true" &&
        get(ENV, "MANTLE_TEST_METAL", "false") == "true" &&
        !fallback && error("Hosted Metal CI must exercise capability dispatch to the mesh fallback")
    clear = (1f0, 0f0, 0f0, 1f0)
    green, red = BGRA{N0f8}(0, 1, 0, 1), BGRA{N0f8}(1, 0, 0, 1)
    g = Mantle.Graph(dev)
    img = Mantle.Transient.Image(g, BGRA{N0f8}, (16, 16))
    pixels = Mantle.Transient.Buffer(g, BGRA{N0f8}, 256)

    Mantle.render!(g, "mesh", img => Mantle.Clear(clear)) do pass
        if fallback
            Mantle.draw!(pass, pipeline, (selection,), 2; frag_args = (1f0,))
        else
            Mantle.draw_software_mesh!(pass, pipeline, (selection,), 2;
                frag_args = (1f0,), viewport = nothing, scissor = nothing,
                bindings = nothing, indices = nothing, instances = 1)
        end
    end
    Mantle.copy!(g, "read", pixels, img)
    @test [p.kind for p in g.passes] == [:compute, :render, :copy]
    plan = Mantle.record!(Mantle.Plan(g))
    try
        Mantle.run!(plan)
        Mantle.waitfor!(plan)
        @test all(==(green), Array(Mantle.storage(pixels)))
        # A group that was active last frame must emit nothing after it is culled.
        Mantle.update!(selection, Int32[0])
        Mantle.run!(plan)
        Mantle.waitfor!(plan)
        @test all(==(red), Array(Mantle.storage(pixels)))
    finally
        Mantle.free!(plan)
    end

    Mantle.update!(selection, Int32[1])
    fb = Mantle.Framebuffer(dev, 16, 16; depth = false, color_format = BGRA{N0f8})
    target = Mantle.OffscreenTarget(fb)
    if fallback
        Mantle.draw!(dev, pipeline, target, 2;
            args = (selection,), frag_args = (1f0,), clear_color = clear)
    else
        Mantle.draw_software_mesh!(dev, pipeline, target, 2;
            args = (selection,), frag_args = (1f0,), instances = 1,
            clear_color = clear, depth_clear = 1f0, indices = nothing, bindings = nothing)
    end
    @test all(==(green), Mantle.readback_framebuffer(fb))
    Mantle.free!(selection)
end
