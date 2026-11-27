# Render to a regular attachment, then copy into the page's drawable while
# encoding the submission. The retained attachment also serves screenshots.

mutable struct WebGPUWindow
    dev::WebGPUDevice
    id::Int
    image::WebGPUAttachment
    events::Bonito.Observable{Any}
    wanted::Dims{2}
    scale::Float32
    listener::Any
    acquirer::Union{Nothing,Task}
    open::Bool
end

Mantle.canvasdensity(d::WebGPUDevice) = Float32(call(d, "canvasScale", d.id))
Mantle.canvasdensity(w::WebGPUWindow) = w.scale
Mantle.canvasevents(w::WebGPUWindow) = w.events

function Mantle.Window(d::WebGPUDevice, w::Integer, h::Integer; title::AbstractString = "",
                        vsync::Bool = false, color_format::Type = BGRA{N0f8}, srgb::Bool = false)
    format = Mantle.pixelformat(WebGPUAPI(), color_format; srgb)
    format in ("bgra8unorm", "rgba8unorm", "bgra8unorm-srgb", "rgba8unorm-srgb") ||
        throw(ArgumentError("WebGPU canvases require an RGBA8 or BGRA8 color format"))
    scale = Mantle.canvasdensity(d)
    events = Bonito.Observable{Any}(nothing)
    id = lock(() -> (d.nbuffers += 1), d.lock)
    call(d, "createCanvas", d.id, id, Int(w), Int(h), scale, format, String(title), events)
    win = WebGPUWindow(d, id, attachment(d, Int(w), Int(h), format), events, (Int(w), Int(h)),
                        scale, nothing, nothing, true)
    win.listener = Bonito.on(events; update = true) do msg
        (msg === nothing || msg[1] != "resize") && return
        win.scale = Float32(msg[4])
        w, h = round(Int, msg[2] * win.scale), round(Int, msg[3] * win.scale)
        w > 0 && h > 0 && (win.wanted = (w, h))
    end
    return win
end

Base.isopen(w::WebGPUWindow) = w.open && isopen(w.dev.session)
Base.size(w::WebGPUWindow) = w.image.dims
Base.eltype(w::WebGPUWindow) = Mantle.eltypeof(w.image.format)
Mantle.target_extent(w::WebGPUWindow) = size(w)
Mantle.target_format(w::WebGPUWindow) = w.image.format
Mantle.target_view(w::WebGPUWindow) = w.image
Mantle.target_image(w::WebGPUWindow) = w.image
Mantle.currentimage(w::WebGPUWindow) = w.image

function Mantle.pollwindow!(w::WebGPUWindow)
    isopen(w) || return nothing
    w.wanted == size(w) && return nothing
    Mantle.checkframeowner(w.acquirer, "resize!")
    d = w.dev
    Mantle.flush!(d)
    next = attachment(d, w.wanted..., w.image.format)
    post(d, "resizeCanvas", d.id, w.id, next.dims...)
    w.image = next
    return nothing
end

Mantle.beginframe!(w::WebGPUWindow) =
    (isopen(w) || throw(EOFError()); nothing)

function Mantle.acquire_next_image!(w::WebGPUWindow)
    Mantle.beginframe!(w)
    Mantle.checkacquirable(w.acquirer)
    w.acquirer = current_task()
    return nothing
end

Mantle.presented!(w::WebGPUWindow) = (w.acquirer = nothing; nothing)

function Mantle.abandonframe!(d::WebGPUDevice, pl::Mantle.Plan)
    for s in pl.graph.surfaces
        s.win.acquirer === current_task() && Mantle.presented!(s.win)
    end
    return nothing
end

function queue_canvas_present!(d::WebGPUDevice, w::WebGPUWindow)
    enqueue!(d, Dict{String,Any}("op" => "present", "canvas" => w.id, "src" => w.image.id))
    return nothing
end

function Mantle.present_frame!(d::WebGPUDevice, w::WebGPUWindow)
    queue_canvas_present!(d, w)
    Mantle.flush!(d)
    Mantle.presented!(w)
    return nothing
end

function Mantle.readback_window(w::WebGPUWindow)
    Mantle.beginframe!(w)
    Mantle.checkreadable(w.acquirer)
    return readattachment(w.image, eltype(w))
end

Mantle.screenshot(w::WebGPUWindow) = Mantle.readback_window(w)

function Mantle.pacewindow!(w::WebGPUWindow)
    isopen(w) && call(w.dev, "canvasFrame")
    return nothing
end

function Base.close(w::WebGPUWindow)
    w.open || return nothing
    Mantle.checkframeowner(w.acquirer, "close")
    w.open = false
    Bonito.off(w.listener)
    if isopen(w.dev.session)
        Mantle.flush!(w.dev)
        post(w.dev, "destroyCanvas", w.dev.id, w.id)
    end
    Mantle.presented!(w)
    return nothing
end
