# The shader builtins, as Mantle's names.
#
# `vertex_index()` and `frag_coord_x()` were the LAST thing in a shader that
# still named a backend. Every other word `bench/showcase.jl`'s shaders use —
# `Rasterizer`, `TriangleList`, `CullBack`, `DrawIndirectCommand`, the varyings
# declaration, the returned output tuple — is already Mantle's, so a shader
# written against them compiles anywhere. These six were not: they were Lava's
# and, separately, Metal's, and a file that used one had to say which backend it
# was for at the top.
#
# They are declared here and OVERRIDDEN per backend, rather than defined here
# and dispatched. A builtin takes no arguments, so there is nothing to dispatch
# ON — what decides the answer is which compiler is running, and the mechanism
# for that is GPUCompiler's method-table overlay (`@device_override`, and Lava's
# `@lava_device_override` around the same thing). The host method below is the
# error a caller gets for running a shader on the CPU, which is the honest
# answer rather than a plausible number.
#
# Indices are ONE-BASED on both backends. A shader that had to remember which
# convention a builtin follows will eventually get it wrong.

"""
    vertex_index() -> Int32

Which vertex this invocation is for, counting from one.

`gl_VertexIndex` in GLSL, `[[vertex_id]]` in MSL. Only callable from a vertex
stage.
"""
function vertex_index end

"""
    instance_index() -> Int32

Which instance this invocation is for, counting from one.
"""
function instance_index end

"""
    frag_coord(dim = 1) -> Float32

One component of the interpolated fragment position, counting from one: 1 is x,
4 is w.

`gl_FragCoord` in GLSL, `[[position]]` in MSL. Only callable from a fragment
stage.
"""
function frag_coord end

"""x of [`frag_coord`](@ref)."""
function frag_coord_x end
"""y of [`frag_coord`](@ref)."""
function frag_coord_y end
"""z of [`frag_coord`](@ref), the depth this fragment writes."""
function frag_coord_z end
"""w of [`frag_coord`](@ref), the reciprocal of the clip w."""
function frag_coord_w end
"""x and y of [`frag_coord`](@ref) together."""
function frag_coord_xy end

"""
    clip_y(y) -> Float32

Bring a clip-space y into THIS backend's convention, or read one back out of it.

**Mantle's clip space is Vulkan's: +y is DOWN the screen.** Metal's is the
other one, so on that backend this negates and everywhere else it is the
identity. It is its own involution — applying it twice gets you back — which is
what lets the same function serve both directions.

Every vertex stage's `position` goes through it automatically. **A shader that
REPROJECTS has to call it itself**, and that is the whole reason it is public:
a shadow lookup that computes `sunvp * world` and turns the clip y into a texel
row is doing the viewport transform by hand, and it has to do the same one the
rasteriser did.

Getting that wrong is quiet. The shadow map is still full of real depths, the
lookup still lands inside it, and the frame still looks like a frame — it just
samples the mirror image of the row it wanted. What surfaced it here was the
demo's own bias knob having no effect at all, because every sample was reading
the same wrong place.

"""
function clip_y end

# One list, so a backend adding the overrides cannot quietly miss one.
const SHADER_BUILTINS = (:vertex_index, :instance_index, :frag_coord,
                         :frag_coord_x, :frag_coord_y, :frag_coord_z,
                         :frag_coord_w, :frag_coord_xy)

# `clip_y` is NOT in that list: it takes an argument, it is meaningful on the
# host (a backend may need it there too), and its default is the identity rather
# than an error. Only a backend whose clip space differs from Mantle's overrides
# it.
clip_y(y::Float32) = y

# The host methods. They exist to SAY something rather than to work: a bare
# `MethodError` on a zero-argument function tells a caller the method is
# missing, not that the function is a GPU builtin they have called from the CPU
# — which is what actually happened, usually by testing a shader as an ordinary
# Julia function. Same shape as `KernelInterface.barrier`'s host method, and the
# reason the backends must use `@device_override` rather than a plain
# definition: a plain one would shadow these and let a host call reach a GPU
# instruction.
for f in SHADER_BUILTINS
    @eval $f(args...) = error(
        $(string(f)) * " is a shader builtin: it reads a value the rasteriser " *
        "supplies, and only a compiled vertex or fragment stage has one. " *
        "Calling it on the host cannot mean anything.")
end
