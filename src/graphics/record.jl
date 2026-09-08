# Recording a render pass by hand, once, in core.
#
# There were TWO families of pass verbs, and only one backend had the second:
#
#   the graph's      begin_render_pass!  record_draw!  end_render_pass!
#   imperative       begin_pass!  end_pass!  set_viewport!  draw_in_pass!
#                    draw_indexed_in_pass!  blit!  oneshot!
#
# The first is implemented by every backend. The second was Vulkan's alone, and
# RayMakie's overlay compositor — a package that must name no backend — was
# written against it. On Metal every one of those seven calls was a `MethodError`,
# which is why compositing an overlay could not work there at all.
#
# The second family is not a different capability. It is the same three
# primitives with the ORDER decided by a caller instead of by the graph. So this
# file is that ordering, in core, over the primitives that already exist: nothing
# new is asked of a backend except the two verbs a pass needs mid-flight, and one
# of those it already had.
#
# ── The DSL ──────────────────────────────────────────────────────────────────
#
#     pass!(dev, target; clear = nothing) do p
#         viewport!(p, x, y, w, h)
#         draw!(p, compiled, args, count; instances = 1)
#     end
#
# Deliberately shaped like the graph's, which is
#
#     render!(g, "name", target => Clear(c)) do p
#         draw!(p, pipeline, args, count)
#     end
#
# so that the difference between "schedule this" and "record this now" is the
# verb and nothing else. A caller that learns one has learned both.
#
# `do` block and not begin/end verbs, because the pass has to be closed on the
# way out of a throw as well. `begin_pass!`/`end_pass!` left that to every caller
# and the overlay compositor did not do it.

"""
    PassRecorder

A render pass open right now, and what a caller records draws into.

Holds the backend's handle — whatever [`begin_render_pass!`](@ref) answered — and
nothing else. `handle` is opaque here, exactly as it is to the graph.
"""
struct PassRecorder{H}
    handle::H
end

"""
    viewport!(p::PassRecorder, x, y, width, height)

Which rectangle of the attachment the following draws land in, in pixels.

A negative `height` puts clip-space +y at the TOP, which is what a caller
matching a pixel convention with the origin at the top-left wants; the scissor is
derived from the same numbers, so there is one place to get it wrong instead of
two.
"""
viewport!(p::PassRecorder, x::Real, y::Real, w::Real, h::Real) =
    setviewport!(p.handle, x, y, w, h)

"""
    setviewport!(handle, x, y, width, height)

The backend half of [`viewport!`](@ref): put a viewport and its scissor on an
open pass.

One of the two verbs a pass needs mid-flight that the graph never asks for — the
graph's passes each cover one whole render area, so it sets the viewport once
when the pass opens and never again.
"""
function setviewport! end

"""
    draw!(p::PassRecorder, compiled, args, count; instances = 1, indices = nothing)

Record one draw into the open pass.

`compiled` is what [`compile_draw`](@ref) answered. `indices` makes it an indexed
draw and `count` is then the index count; without it `count` is the vertex count.
"""
function draw!(p::PassRecorder, compiled, args, count::Integer;
               instances::Integer = 1, indices = nothing)
    record_draw!(p.handle, compiled, args, count; instances, indices)
    return nothing
end

"""
    bindings!(p::PassRecorder, compiled, bindings)

Make `bindings` the texture table the following draws sample from. Built by
[`bind_textures`](@ref).
"""
bindings!(p::PassRecorder, compiled, b) = use_bindings!(p.handle, compiled, b)

"""
    passtargets(target) -> (targets, loads, depth, depth_load)

Resolve a [`RenderTarget`](@ref) into what [`begin_render_pass!`](@ref) takes.

The graph resolves attachments during its place phase and hands the backend the
results; a hand-recorded pass has one target and resolves it here, so the backend
hook has ONE shape whichever side calls it. That is the whole reason
`begin_render_pass!` was given resolved attachments rather than a `Pass`.
"""
function passtargets(t::OffscreenTarget, clear)
    load = clear === nothing ? Keep : Clear(clear)
    fb = t.fb
    d = depthimage(fb)
    return ((colorimage(fb),), (load,), d, d === nothing ? nothing : Clear(1.0f0))
end

function passtargets(t::WindowTarget, clear)
    load = clear === nothing ? Keep : Clear(clear)
    return ((currentimage(t.window),), (load,), nothing, nothing)
end

"""
    colorimage(fb) -> the attachment `begin_render_pass!` draws into
    depthimage(fb) -> the depth attachment, or `nothing`

What a [`Framebuffer`](@ref) hands a pass. A backend's framebuffer answers both;
they are separate verbs because a depth-only pass has no colour and a colour-only
one has no depth, and a single accessor returning a tuple made the second case
read as an error.
"""
function colorimage end
@doc (@doc colorimage) function depthimage end

"""
    currentimage(window) -> the image this frame draws into

The acquired swapchain image. A [`Window`](@ref)'s, because which one is current
is the window's bookkeeping and changes every frame.
"""
function currentimage end

"""
    pass!(f, dev, target; clear = nothing) -> whatever `f` returned

Open a render pass over `target`, record `f` into it, and close it.

    pass!(dev, target) do p
        viewport!(p, 0, h, w, -h)
        draw!(p, compiled, args, 6)
    end

`clear` is a colour to clear the attachment to, or `nothing` to draw on top of
what is there — which is what compositing an overlay wants.

The pass is closed on the way out of a throw as well as a return. The verb pair
this replaces left that to the caller, and the one caller in tree did not do it:
a shader that failed to compile mid-pass left the pass open, and the next frame's
`begin_pass!` opened a second one on the same buffer.
"""
function pass!(f, dev, target::RenderTarget; clear = nothing)
    targets, loads, depth, depth_load = passtargets(target, clear)
    h = begin_render_pass!(dev, targets, loads, depth, depth_load)
    p = PassRecorder(h)
    result = try
        f(p)
    finally
        end_render_pass!(h)
    end
    return result
end
