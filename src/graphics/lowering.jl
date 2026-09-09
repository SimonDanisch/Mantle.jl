# Turning a geometry stage into a mesh pipeline.
#
# NOT IMPLEMENTED. This file exists so the gap has a NAME and a loud error rather
# than being a capability question a caller has to know to ask, and so the design
# that has to be built is written down where the work happens instead of in a
# commit message.
#
# ── Why it is needed ─────────────────────────────────────────────────────────
#
# A `GraphicsPipeline` with a `GeometryShader` runs natively on Vulkan and cannot
# run at all on Metal: Apple removed the geometry stage and replaced it with the
# mesh pipeline (object + mesh stages). Everything a geometry shader does is
# expressible on a mesh stage — one invocation reads a primitive's vertices and
# emits new ones — so the translation is mechanical, and doing it once here means
# a shader author never writes it twice.
#
# What is affected today, in tree: RayMakie's `scatter` and `lines` overlays, the
# only two `GeometryShader` construction sites. Scatter is also the sprite path,
# so text goes through it. Without this, a Metal figure has no axis grid, no
# ticks, no labels, and no scatter or line plot.
#
# ── The design, decided ──────────────────────────────────────────────────────
#
# ONE MESH THREADGROUP PER INPUT PRIMITIVE, ONE THREAD.
#
# That single thread evaluates the vertex body once per input vertex of its
# primitive — `inputvertices(topology)` times, in a plain loop — gathers the
# results into the `prim` NamedTuple the geometry body already reads
# (`prim.world_pos[1]` and so on: one field per declared output, each an NTuple
# over the primitive's vertices), builds a `MeshEmitter` over the output object,
# and calls the geometry body unchanged.
#
# An earlier reading of this had one thread PER INPUT VERTEX, with the vertex
# outputs meeting in threadgroup memory behind a barrier. That was forced by a
# constraint that does not exist: it is the only arrangement in which
# `vertex_index()` — a zero-argument builtin — can answer correctly for each
# vertex, because the answer has to come from the thread's own identity. Pass the
# index to the vertex body INSTEAD and one thread can evaluate it at as many
# indices as it likes. Then there is no threadgroup memory, no barrier, and no
# special handling for the overlapping primitives of a strip or adjacency
# topology — `MeshEmitter` already turns strip emission into explicit indices.
#
# So the contract is `KernelInterface.VertexIndex`: a vertex body may declare a
# leading `::VertexIndex` parameter, and one that does is passed the index rather
# than calling `vertex_index()`. Opt-in per shader, detected from the method table
# by `wantsvertexindex`, so a body that does not declare it is untouched on either
# backend.
#
# ── What is left to write ────────────────────────────────────────────────────
#
#  1. `GeometryAsMesh{VF,GF,IT,OT,NP,FN}`, a callable whose body is the loop
#     above. Its `MeshShader` outputs are the geometry stage's outputs verbatim,
#     its `max_vertices`/`max_primitives` the geometry stage's `max_vertices` and
#     what the output topology makes of it, its `threads` 1.
#  2. The primitive gather: `ntuple(k -> vertexbody(VertexIndex(base + k), args...),
#     Val(inputvertices(IT)))` reshaped from a tuple-of-NamedTuples into a
#     NamedTuple-of-tuples. Pure type arithmetic, so `@generated`.
#  3. `firstvertex(IT, group)` — the stride table. A LIST topology advances by its
#     arity, a STRIP by one, `TriangleStripAdjacency` by two.
#  4. THE INDEXED CASE, which `lines` is. On a real vertex stage `vertex_index()`
#     is the value FETCHED from the index buffer, so the lowered stage has to read
#     the index buffer itself: it becomes one of the mesh stage's own buffers and
#     the gather reads `indices[base + k]`. This is the one part that is more than
#     a rearrangement, and it is why `lines` is not simply `scatter` again.
#  5. The vertex-stage wrapper on BOTH backends passing `VertexIndex(vertex_index())`
#     to a body that declares it, so the same shader still runs natively on Vulkan.
#  6. `draw!` for the lowered pipeline dispatching one threadgroup per input
#     primitive, which is the input vertex count divided by the topology's stride.

"""
    lower_geometry_to_mesh(p::GraphicsPipeline) -> MeshPipeline

Rewrite a pipeline's vertex + geometry stages as a mesh stage.

**Not implemented.** Calling this raises a descriptive error; the design and the
remaining work are written out at the top of `src/graphics/lowering.jl`.

Once it exists, a backend with no geometry stage can run a geometry pipeline by
lowering it, and `supports_geometry_stage` stops being a question a renderer has
to branch on.
"""
function lower_geometry_to_mesh(p::GraphicsPipeline)
    p.geometry === nothing && throw(ArgumentError(
        "lower_geometry_to_mesh: this pipeline has no geometry stage, so there is " *
        "nothing to lower. Draw it as it is."))
    error("""
        lower_geometry_to_mesh IS NOT IMPLEMENTED — implement it, do not work around it.

        The shader that needs it: $(stagefunction(p.geometry))
          input topology  $(stageconfig(p.geometry).input_topology)
          output topology $(stageconfig(p.geometry).output_topology)
          max_vertices    $(stageconfig(p.geometry).max_vertices)

        Every piece this depends on is built and tested: Metal emits AIR mesh
        programs, a mesh stage has the compute builtins and threadgroup memory,
        KernelInterface's portable mesh vocabulary lowers onto it, and
        `Mantle.MeshPipeline` compiles and draws. What is missing is only the
        translation, and the design is decided — see the comment block at the top
        of src/graphics/lowering.jl:

          one mesh threadgroup per input primitive, one thread; that thread
          evaluates the vertex body once per input vertex via a declared
          `KernelInterface.VertexIndex` parameter, gathers `prim`, and runs the
          geometry body over a `MeshEmitter` unchanged.

        Do NOT reintroduce a skip-with-a-warning path. A renderer that quietly
        drops an overlay produces an image that is wrong in a way nobody reading
        it can see — no axis grid, no ticks, no labels, no scatter, no lines —
        and reports success.""")
end
