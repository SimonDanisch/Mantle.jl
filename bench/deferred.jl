# Deferred shading with GPU culling, twice: boxes against the view frustum, and
# lights against each screen tile.
#
#   clear count ─> cull ─> draw count ─┐
#   lights ────────────────────────────┴─> gbuffer ─> read x3 ─> tiles ─> light
#
# Compute decides how many boxes are visible and writes the draw command itself,
# so the host never learns the number. Compute decides which lights reach each
# tile and the fragment shader reads that. Nothing below states a barrier or a
# layout — every one of them falls out of what each pass says it does to each
# resource, and `run!` is the whole frame.

import Mantle
using Lava, GeometryBasics, LinearAlgebra, KernelAbstractions, Atomix, Random
using ColorTypes: RGBA
using ColorTypes.FixedPointNumbers: N0f8
M = Mantle

W, H = 1280, 800
NPX = W * H
NBOX = 4000
NLIGHT = 96
VPB = 36     # vertices per box
TILE = 16    # screen tile edge for the light lists
NTX, NTY = cld(W, TILE), cld(H, TILE)
NTILE = NTX * NTY
PERTILE = 32 # lights a tile keeps

# unit cube, 12 triangles, per-face normals, wound CCW seen from outside
function cube_geometry()
    pos, nrm = Vec4f[], Vec4f[]
    for axis in 1:3, s in (1f0, -1f0)
        n = Vec3f(ntuple(k -> k == axis ? s : 0f0, 3))
        a = Vec3f(ntuple(k -> k == mod1(axis + 1, 3) ? 1f0 : 0f0, 3))
        b = Vec3f(ntuple(k -> k == mod1(axis + 2, 3) ? 1f0 : 0f0, 3))
        u, v = s > 0 ? (a, b) : (b, a)
        q = (n - u - v, n + u - v, n + u + v, n - u + v)
        for k in (1, 2, 3, 1, 3, 4)
            push!(pos, Vec4f(q[k]..., 1))
            push!(nrm, Vec4f(n..., 0))
        end
    end
    pos, nrm
end

# a ground slab plus a tower per grid cell; w of a centre is its bounding radius
function city(n)
    side = ceil(Int, sqrt(n - 1))
    pitch = 5f0
    span = side * pitch
    centers = [Vec4f(0, -0.5, 0, span)]
    sizes = [Vec4f(span, 0.5, span, 0)]
    colors = [Vec4f(0.06, 0.06, 0.07, 1)]
    for i in 1:(n - 1)
        gx, gz = (i - 1) % side, (i - 1) ÷ side
        x = (gx - side / 2) * pitch + 0.6f0 * randn(Float32)
        z = (gz - side / 2) * pitch + 0.6f0 * randn(Float32)
        h = 1f0 + 7f0 * rand(Float32)^2
        s = Vec4f(0.8f0 + 0.7f0 * rand(Float32), h, 0.8f0 + 0.7f0 * rand(Float32), 0)
        push!(centers, Vec4f(x, h, z, sqrt(s[1]^2 + s[2]^2 + s[3]^2)))
        push!(sizes, s)
        t = rand(Float32)
        push!(colors, Vec4f(0.10 + 0.16t, 0.11 + 0.13t, 0.14 + 0.11t, 1))
    end
    centers, sizes, colors
end

# a random hue at the given saturation, so a surface reached by two lights reads as two
function tint(sat)
    h = 6rand(Float32); i = floor(Int, h); f = h - i
    c = (Vec3f(1, f, 0), Vec3f(1 - f, 1, 0), Vec3f(0, 1, f),
         Vec3f(0, 1 - f, 1), Vec3f(f, 0, 1), Vec3f(1, 0, 1 - f))[i + 1]
    Vec3f(1 - sat) + sat * c
end

function camera(t, radius, height)
    eye = Vec3f(radius * cos(0.13f0 * t), height + 5f0 * sin(0.21f0 * t), radius * sin(0.13f0 * t))
    at = Vec3f(0, 3, 0)
    f = normalize(at - eye); r = normalize(cross(f, Vec3f(0, 1, 0))); u = cross(r, f)
    view = Mat4f(r[1], u[1], -f[1], 0, r[2], u[2], -f[2], 0, r[3], u[3], -f[3], 0,
                 -dot(r, eye), -dot(u, eye), dot(f, eye), 1)
    tn = tan(55f0 * Float32(pi) / 360f0)
    near, far = 0.5f0, 260f0
    proj = Mat4f(1 / (Float32(W / H) * tn), 0, 0, 0, 0, -1 / tn, 0, 0,
                 0, 0, far / (near - far), -1, 0, 0, (near * far) / (near - far), 0)
    proj * view, eye
end

@kernel function reset_counter!(counter)
    counter[1] = UInt32(0)
end

# Gribb-Hartmann: the six clip-space tests read straight off the rows of vp
@inline function frustum_plane(vp::Mat4f, k::Int32)
    row(i) = Vec4f(vp[i, 1], vp[i, 2], vp[i, 3], vp[i, 4])
    w = row(4)
    p = k == Int32(1) ? w + row(1) : k == Int32(2) ? w - row(1) :
        k == Int32(3) ? w + row(2) : k == Int32(4) ? w - row(2) :
        k == Int32(5) ? row(3) : w - row(3)
    p / sqrt(p[1] * p[1] + p[2] * p[2] + p[3] * p[3])
end

@kernel function cull!(visible, counter, @Const(centers), vp::AbstractVector{Mat4f})
    i = @index(Global)
    @inbounds begin
        c = centers[i]
        m = vp[1]
        keep = true
        for k in Int32(1):Int32(6)
            p = frustum_plane(m, k)
            keep &= (p[1] * c[1] + p[2] * c[2] + p[3] * c[3] + p[4]) >= -c[4]
        end
        if keep
            slot = Atomix.@atomic counter[1] += UInt32(1)
            visible[slot] = UInt32(i)
        end
    end
end

# the whole point of culling on the GPU: the draw's size is decided here
@kernel function write_draw!(cmds, @Const(counter), vpb::UInt32)
    @inbounds cmds[1] = DrawIndirectCommand(counter[1] * vpb, UInt32(1), UInt32(0), UInt32(0))
end

@kernel function move_lights!(pos, @Const(home), t::AbstractVector{Float32})
    i = @index(Global)
    @inbounds begin
        h = home[i]
        tv = t[1]
        ph = 0.7f0 * Float32(i)
        pos[i] = Vec4f(h[1] + 6f0 * sin(0.5f0 * tv + ph), h[2] + 1.5f0 * sin(0.9f0 * tv + ph),
                       h[3] + 6f0 * cos(0.4f0 * tv + ph), h[4])
    end
end

# Every invocation belongs to a box that survived: the draw covers the visible
# count and nothing else, so there is no culled instance to collapse.
function gbuffer_vertex(cubepos, cubenrm, visible, centers, sizes, colors, vp::AbstractVector{Mat4f})
    v = vertex_index() - Int32(1)
    inst = v ÷ Int32(36)
    k = v - inst * Int32(36)
    @inbounds begin
        o = Int32(visible[inst + Int32(1)])
        c = centers[o]; s = sizes[o]
        lp = cubepos[k + Int32(1)]; n = cubenrm[k + Int32(1)]
        p = Vec4f(c[1] + lp[1] * s[1], c[2] + lp[2] * s[2], c[3] + lp[3] * s[3], 1f0)
        m = vp[1]
        (position = m * p, albedo = colors[o], normal = Vec3f(n[1], n[2], n[3]))
    end
end

gbuffer_fragment(inputs) = (inputs.albedo,
                            Vec4f(0.5f0 * inputs.normal[1] + 0.5f0,
                                  0.5f0 * inputs.normal[2] + 0.5f0,
                                  0.5f0 * inputs.normal[3] + 0.5f0, 1f0))

GBUFFER = Rasterizer(vertex = gbuffer_vertex, fragment = gbuffer_fragment,
                     varyings = (albedo = Vec4f, normal = Vec3f),
                     topology = TriangleList(), blend = Opaque(),
                     cull = CullBack(), depth = DepthLess())

@inline unpack8(px::UInt32, shift::UInt32) = Float32((px >> shift) & 0x000000ff) * (1f0 / 255f0)

@inline function unproject(invvp::Mat4f, fx::Float32, fy::Float32, d::Float32, w::Int32, h::Int32)
    q = invvp * Vec4f(2f0 * fx / Float32(w) - 1f0, 2f0 * fy / Float32(h) - 1f0, d, 1f0)
    Vec3f(q[1] / q[4], q[2] / q[4], q[3] / q[4])
end

# one work item per screen tile: the depth range it covers becomes a world box,
# and a light whose sphere misses that box cannot reach any pixel in the tile
@kernel function tile_lights!(tilelights, tilecount, @Const(dep), @Const(lpos),
                              invvp::AbstractVector{Mat4f}, w::Int32, h::Int32, tile::Int32,
                              ntx::Int32, nlight::AbstractVector{Int32}, pertile::Int32)
    tx0, ty0 = @index(Global, NTuple)
    @inbounds begin
        ti = (Int32(ty0) - Int32(1)) * ntx + Int32(tx0)
        x0 = (Int32(tx0) - Int32(1)) * tile; y0 = (Int32(ty0) - Int32(1)) * tile
        x1 = min(x0 + tile, w); y1 = min(y0 + tile, h)
        dmin = 1f0; dmax = 0f0
        for py in y0:(y1 - Int32(1)), px in x0:(x1 - Int32(1))
            d = dep[py * w + px + Int32(1)]
            if d < 1f0                      # sky would stretch the box to the far plane
                dmin = min(dmin, d); dmax = max(dmax, d)
            end
        end
        n = UInt32(0)
        if dmax > 0f0                       # a tile that is all sky needs no light
            lo = Vec3f(1f30, 1f30, 1f30)
            hi = Vec3f(-1f30, -1f30, -1f30)
            m = invvp[1]
            for c in Int32(0):Int32(7)
                p = unproject(m, Float32((c & Int32(1)) == 0 ? x0 : x1),
                              Float32((c & Int32(2)) == 0 ? y0 : y1),
                              (c & Int32(4)) == 0 ? dmin : dmax, w, h)
                lo = Vec3f(min(lo[1], p[1]), min(lo[2], p[2]), min(lo[3], p[3]))
                hi = Vec3f(max(hi[1], p[1]), max(hi[2], p[2]), max(hi[3], p[3]))
            end
            for l in Int32(1):nlight[1]
                lp = lpos[l]
                dx = max(0f0, max(lo[1] - lp[1], lp[1] - hi[1]))
                dy = max(0f0, max(lo[2] - lp[2], lp[2] - hi[2]))
                dz = max(0f0, max(lo[3] - lp[3], lp[3] - hi[3]))
                if dx * dx + dy * dy + dz * dz <= lp[4] * lp[4] && n < UInt32(pertile)
                    n += UInt32(1)
                    tilelights[(ti - Int32(1)) * pertile + Int32(n)] = UInt32(l)
                end
            end
        end
        tilecount[ti] = n
    end
end

# a fullscreen triangle out of the vertex index, so the vertex stage takes nothing
function light_vertex()
    v = vertex_index()
    p = v == Int32(1) ? Vec4f(-1, -1, 0, 1) : v == Int32(2) ? Vec4f(3, -1, 0, 1) : Vec4f(-1, 3, 0, 1)
    (position = p,)
end

# and the fragment stage is the deferred pass: one pixel, its tile's lights
function light_fragment(inputs, alb, nrm, dep, lpos, lcol, tilelights, tilecount,
                        invvp::AbstractVector{Mat4f}, eye::AbstractVector{Vec4f}, w::Int32, h::Int32,
                        tile::Int32, ntx::Int32, pertile::Int32, exposure::AbstractVector{Float32})
    px = unsafe_trunc(Int32, frag_coord_x())
    py = unsafe_trunc(Int32, frag_coord_y())
    @inbounds begin
        i = py * w + px + Int32(1)
        d = dep[i]
        out = Vec3f(0.004f0, 0.006f0, 0.014f0) +
              Vec3f(0.010f0, 0.011f0, 0.026f0) * (1f0 - Float32(py) / Float32(h))
        if d < 1f0
            a = alb[i]; nn = nrm[i]
            albedo = Vec3f(unpack8(a, UInt32(0)), unpack8(a, UInt32(8)), unpack8(a, UInt32(16)))
            n = Vec3f(2f0 * unpack8(nn, UInt32(0)) - 1f0, 2f0 * unpack8(nn, UInt32(8)) - 1f0,
                      2f0 * unpack8(nn, UInt32(16)) - 1f0)
            world = unproject(invvp[1], Float32(px) + 0.5f0, Float32(py) + 0.5f0, d, w, h)
            ev = eye[1]
            view = normalize(Vec3f(ev[1] - world[1], ev[2] - world[2], ev[3] - world[3]))
            acc = albedo * Vec3f(0.05f0, 0.06f0, 0.10f0)
            ti = (py ÷ tile) * ntx + (px ÷ tile) + Int32(1)
            for k in Int32(1):Int32(tilecount[ti])
                l = Int32(tilelights[(ti - Int32(1)) * pertile + k])
                lp = lpos[l]
                dv = Vec3f(lp[1] - world[1], lp[2] - world[2], lp[3] - world[3])
                dist = sqrt(dv[1] * dv[1] + dv[2] * dv[2] + dv[3] * dv[3]) + 1f-4
                fall = max(0f0, 1f0 - dist / lp[4])
                if fall > 0f0
                    ld = dv / dist
                    ndl = max(0f0, ld[1] * n[1] + ld[2] * n[2] + ld[3] * n[3])
                    hv = normalize(ld + view)
                    spec = max(0f0, hv[1] * n[1] + hv[2] * n[2] + hv[3] * n[3])^48f0
                    c = lcol[l]
                    acc += Vec3f(c[1], c[2], c[3]) * c[4] * fall * fall * (albedo * ndl + Vec3f(spec))
                end
            end
            out = acc
        end
        # linear out: the swapchain is an sRGB format and the hardware encodes on write
        e = out * exposure[1]
        Vec4f(e[1] / (1f0 + e[1]), e[2] / (1f0 + e[2]), e[3] / (1f0 + e[3]), 1f0)
    end
end

LIGHTING = Rasterizer(vertex = light_vertex, fragment = light_fragment,
                      varyings = NamedTuple(), topology = TriangleList(),
                      blend = Opaque(), cull = NoCull(), depth = DepthOff())

Random.seed!(7)
cubes, normals = cube_geometry()
boxc, boxs, boxcol = city(NBOX)

dev = M.Device(M.VulkanAPI())
win = M.Window(W, H; title = "mantle: deferred + gpu culling", vsync = false)

cubepos = M.Buffer(dev, cubes)
cubenrm = M.Buffer(dev, normals)
centers = M.Buffer(dev, boxc)
sizes   = M.Buffer(dev, boxs)
colors  = M.Buffer(dev, boxcol)
visible = M.Buffer(dev, UInt32, NBOX)
counter = M.Buffer(dev, UInt32[0])
# The element type is what makes the draw indirect, and what gets the buffer
# allocated so the command processor may read it.
drawcmd = M.Buffer(dev, DrawIndirectCommand, 1)

# w is the radius the falloff reaches zero at, so it is also the culling radius
lighthome = M.Buffer(dev, [Vec4f(150f0 * (rand(Float32) - 0.5f0), 3f0 + 11f0 * rand(Float32),
                                 150f0 * (rand(Float32) - 0.5f0), 11f0 + 9f0 * rand(Float32))
                           for _ in 1:NLIGHT])
lightpos = M.Buffer(dev, zeros(Vec4f, NLIGHT))
lightcol = M.Buffer(dev, [Vec4f(tint(0.85f0)..., 2.5f0) for _ in 1:NLIGHT])

# every per-frame value the passes read is a GPURef: the dispatches hold its
# device address, and `frame!` below writes the pending value with `ref[] = x`
vp0 = first(camera(0f0, 46f0, 13f0))
vpref = M.GPURef(dev, vp0)
cullref = M.GPURef(dev, vp0)
invref = M.GPURef(dev, inv(vp0))
eyeref = M.GPURef(dev, Vec4f(0, 0, 0, 1))
timeref = M.GPURef(dev, 0f0)
lightref = M.GPURef(dev, Int32(NLIGHT))
exposure = M.GPURef(dev, 1f0)

graph = M.Graph(dev)
screen = M.Surface(graph, win)
albedo = M.Transient.Image(graph, RGBA{N0f8}, (W, H))
normal = M.Transient.Image(graph, RGBA{N0f8}, (W, H))
zbuf   = M.Transient.Image(graph, Float32, (W, H))
albedo_px = M.Transient.Buffer(graph, UInt32, NPX)
normal_px = M.Transient.Buffer(graph, UInt32, NPX)
depth_px  = M.Transient.Buffer(graph, Float32, NPX)
tilelights = M.Transient.Buffer(graph, UInt32, NTILE * PERTILE)
tilecount  = M.Transient.Buffer(graph, UInt32, NTILE)

M.compute!(graph, "clear count") do p
    M.dispatch!(p, reset_counter!, (M.use(p, counter; write = true),), 1)
end
M.compute!(graph, "cull") do p
    M.use(p, cullref; read = true)
    M.dispatch!(p, cull!, (M.use(p, visible; write = true),
                           M.use(p, counter; read = true, write = true),
                           M.use(p, centers; read = true), cullref), NBOX)
end
M.compute!(graph, "draw count") do p
    M.dispatch!(p, write_draw!, (M.use(p, drawcmd; write = true),
                                 M.use(p, counter; read = true), UInt32(VPB)), 1)
end
M.compute!(graph, "lights") do p
    M.use(p, timeref; read = true)
    M.dispatch!(p, move_lights!, (M.use(p, lightpos; write = true),
                                  M.use(p, lighthome; read = true), timeref), NLIGHT)
end
M.render!(graph, "gbuffer", albedo => M.Clear((0f0, 0f0, 0f0, 1f0)),
                            normal => M.Discard, zbuf => M.Clear(1f0)) do p
    M.use(p, vpref; read = true)
    args = (M.Attribute(p, cubepos), M.Attribute(p, cubenrm), M.Attribute(p, visible),
            M.Attribute(p, centers), M.Attribute(p, sizes), M.Attribute(p, colors), vpref)
    M.draw!(p, GBUFFER, args, drawcmd)
end
M.copy!(graph, "read albedo", albedo_px, albedo)
M.copy!(graph, "read normal", normal_px, normal)
M.copy!(graph, "read depth", depth_px, zbuf)
M.compute!(graph, "tiles") do p
    M.use(p, invref; read = true)
    M.use(p, lightref; read = true)
    M.dispatch!(p, tile_lights!, (M.use(p, tilelights; write = true),
                                  M.use(p, tilecount; write = true),
                                  M.use(p, depth_px; read = true),
                                  M.use(p, lightpos; read = true), invref,
                                  Int32(W), Int32(H), Int32(TILE), Int32(NTX),
                                  lightref, Int32(PERTILE)), (NTX, NTY); group = (8, 8))
end
# Discard, not Clear: the triangle covers every pixel, so loading the old
# contents is work with nothing to show for it.
M.render!(graph, "light", screen => M.Discard) do p
    M.use(p, invref; read = true)
    M.use(p, eyeref; read = true)
    M.use(p, exposure; read = true)
    M.draw!(p, LIGHTING, (), 3;
            frag_args = (M.use(p, albedo_px; read = true), M.use(p, normal_px; read = true),
                         M.use(p, depth_px; read = true), M.use(p, lightpos; read = true),
                         M.use(p, lightcol; read = true), M.use(p, tilelights; read = true),
                         M.use(p, tilecount; read = true), invref, eyeref,
                         Int32(W), Int32(H), Int32(TILE), Int32(NTX), Int32(PERTILE), exposure))
end
plan = M.Plan(graph; profile = true)

function frame!(t, radius, height, freeze)
    vp, eye = camera(t, radius, height)
    vpref[] = vp
    freeze || (cullref[] = vp)
    invref[] = inv(vp)
    eyeref[] = Vec4f(eye[1], eye[2], eye[3], 1f0)
    timeref[] = t
    M.run!(plan)
end

begin # ── the render loop, async so the REPL keeps working ─────────────────────
    orbit = 46f0
    height = 13f0
    speed = 1f0
    freeze_cull = false
    stop = Threads.Atomic{Bool}(false)
    frames = Ref(0)
    nvisible = Ref(0)
    pertile = Ref(0f0)
    # @async, not @spawn: Vulkan recording is single threaded and a sticky task
    # keeps it on the thread that owns the queue.
    render = errormonitor(@async begin
        t, last = 0f0, time()
        while isopen(win) && !stop[]
            now = time(); t += speed * Float32(now - last); last = now
            frame!(t, orbit, height, freeze_cull)
            frames[] += 1
            # A readback waits for the device, so it happens here rather than from
            # the REPL, where it would race the recording, and rarely enough that
            # the stall does not set the frame rate.
            if frames[] % 30 == 0
                nvisible[] = Int(Array(counter)[1])
                pertile[] = sum(Array(M.storage(tilecount))) / NTILE
            end
            sleep(0.001)   # the yield that lets the REPL run between frames
        end
    end)
end

# ── eval any of these while the window is up ─────────────────────────────────

nvisible[]               # boxes that survived the frustum test, of NBOX

pertile[]                # lights the average screen tile kept, of NLIGHT

freeze_cull = true       # the frustum stops following the camera; culled geometry pops out

freeze_cull = false

lightref[] = Int32(8)    # what the lighting costs, one light count against another

lightref[] = Int32(96)

exposure[] = 3f0

exposure[] = 1f0

orbit = 90f0             # further out, so more of the city is inside the frustum

height = 40f0

speed = 0f0              # hold still

M.timings(plan)          # per pass, host and gpu milliseconds

round(M.peakbytes(plan) / 2^20, digits = 2)   # what the transients cost together

begin # ── stop ──────────────────────────────────────────────────────────────
    stop[] = true
    wait(render)
    close(win)
end
