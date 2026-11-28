# Adaptive tessellation of curved FEM cells: the port of `reference.jl` onto Mantle.
# Every literal is Float32 (`1f0`), because Apple GPUs have no Float64 at all.

using Mantle, GeometryBasics, LinearAlgebra
import Hikari
using KernelAbstractions: @kernel, @index, get_backend
using Hikari: monomials, eval_poly, surfacepoint, surfacenormal

@inline eval_position(cx, cy, cell, a, b) =
    (eval_poly(cx, cell, a, b), eval_poly(cy, cell, a, b))

function basetriangles(ncells::Integer = 2)
    rim = ((-1f0, -1f0), (1f0, -1f0), (1f0, 1f0), (-1f0, 1f0))
    corners = Float32[]
    cells = Int32[]
    for c in 1:ncells, i in 1:4
        a, b = rim[i], rim[mod1(i + 1, 4)]
        # (b, centre, a): the split edge (v1, v3) must be the element edge.
        append!(corners, (b[1], b[2], 0f0, 0f0, a[1], a[2]))
        push!(cells, Int32(c))
    end
    return corners, cells
end

@inline function basecorner(basecorners, t::Integer, j::Integer)
    o = (t - 1) * 6 + (j - 1) * 2
    return (basecorners[o + 1], basecorners[o + 2])
end

@inline key_base(k::UInt64) = Int(k >> 32)
@inline key_leb(k::UInt64) = k % UInt32
@inline make_key(b::Integer, leb::UInt32) = (UInt64(b) << 32) | UInt64(leb)
@inline root_key(b::Integer) = make_key(b, UInt32(1))
@inline key_depth(k::UInt64) = 31 - leading_zeros(key_leb(k))
@inline key_parent(k::UInt64) = make_key(key_base(k), key_leb(k) >> 1)
@inline key_child(k::UInt64, i) = make_key(key_base(k), key_leb(k) << 1 | UInt32(i))
@inline is_child0(k::UInt64) = iseven(key_leb(k))

@inline function key_corners(k::UInt64, basecorners)
    b = key_base(k)
    c1 = basecorner(basecorners, b, 1)
    c2 = basecorner(basecorners, b, 2)
    c3 = basecorner(basecorners, b, 3)
    w1 = (1f0, 0f0, 0f0); w2 = (0f0, 1f0, 0f0); w3 = (0f0, 0f0, 1f0)
    leb = key_leb(k)
    for i in (key_depth(k) - 1):-1:0
        m = ((w1[1] + w3[1]) / 2, (w1[2] + w3[2]) / 2, (w1[3] + w3[3]) / 2)
        if ((leb >> i) & 1) == UInt32(0)
            w1, w2, w3 = w1, m, w2
        else
            w1, w2, w3 = w2, m, w3
        end
    end
    mix(w) = (w[1] * c1[1] + w[2] * c2[1] + w[3] * c3[1],
              w[1] * c1[2] + w[2] * c2[2] + w[3] * c3[2])
    return mix(w1), mix(w2), mix(w3)
end

# Three flat tuples read at a literal index: a GPU cannot index a nested tuple at runtime.
@inline sample_a() = (0.5f0, 0.5f0, 0f0,   1f0/3f0)
@inline sample_b() = (0f0,   0.5f0, 0.5f0, 1f0/3f0)
@inline sample_c() = (0.5f0, 0f0,   0.5f0, 1f0/3f0)

@inline function excess_levels(k::UInt64, cx, cy, cf, basecorners, basecell, geo_tol, fld_tol)
    cell = Int(basecell[key_base(k)])
    p1, p2, p3 = key_corners(k, basecorners)
    x1 = eval_position(cx, cy, cell, p1...); f1 = eval_poly(cf, cell, p1...)
    x2 = eval_position(cx, cy, cell, p2...); f2 = eval_poly(cf, cell, p2...)
    x3 = eval_position(cx, cy, cell, p3...); f3 = eval_poly(cf, cell, p3...)
    per = ntuple(Val(4)) do si
        a = sample_a()[si]; b = sample_b()[si]; c = sample_c()[si]
        q1 = a * p1[1] + b * p2[1] + c * p3[1]
        q2 = a * p1[2] + b * p2[2] + c * p3[2]
        ex, ey = eval_position(cx, cy, cell, q1, q2)
        lx = a * x1[1] + b * x2[1] + c * x3[1]
        ly = a * x1[2] + b * x2[2] + c * x3[2]
        return (sqrt((ex - lx)^2 + (ey - ly)^2),
                abs(eval_poly(cf, cell, q1, q2) - (a * f1 + b * f2 + c * f3)))
    end
    geo_err = maximum(map(first, per))
    fld_err = maximum(map(last, per))
    return max(log2(max(geo_err, 1f-16) / geo_tol), log2(max(fld_err, 1f-16) / fld_tol))
end

q2nodes() = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0),
             (0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, 0.0))

function q2basis()
    nmono = Hikari.FEM_NMONO
    V = zeros(nmono, nmono)
    for (k, (a, b)) in enumerate(q2nodes())
        V[k, :] .= collect(monomials(a, b))
    end
    return inv(V)
end

function annulus(; nθ = 20, nr = 3, r0 = 0.40, r1 = 1.15, lobes = 3, amp = 0.46)
    Vinv = q2basis()
    nmono = Hikari.FEM_NMONO
    ncells = nθ * nr
    cx = zeros(Float32, nmono, ncells); cy = zeros(Float32, nmono, ncells)
    cf = zeros(Float32, nmono, ncells)
    xs = zeros(nmono); ys = zeros(nmono); fs = zeros(nmono)
    c = 0
    for j in 1:nr, i in 1:nθ
        c += 1
        for (k, (a, b)) in enumerate(q2nodes())
            u = (i - 1 + 0.5 * (a + 1)) / nθ
            v = (j - 1 + 0.5 * (b + 1)) / nr
            θ = -2π * u  # clockwise, so the normals point up
            r = r0 + (r1 - r0) * v
            xs[k] = r * cos(θ)
            ys[k] = r * sin(θ)
            fs[k] = 0.5 + amp * cos(lobes * θ) * v
        end
        cx[:, c] = Vinv * xs; cy[:, c] = Vinv * ys; cf[:, c] = Vinv * fs
    end
    return cx, cy, cf
end

@kernel function classify_kernel!(counts, @Const(keys), @Const(cx), @Const(cy), @Const(cf),
                                  @Const(basecorners), @Const(basecell),
                                  geo_tol, fld_tol, max_depth)
    i = @index(Global)
    @inbounds begin
        k = keys[i]
        d = key_depth(k)
        if d < max_depth && excess_levels(k, cx, cy, cf, basecorners, basecell, geo_tol, fld_tol) > 0
            counts[i] = Int32(2)
        elseif d > 0 && excess_levels(key_parent(k), cx, cy, cf, basecorners, basecell, geo_tol, fld_tol) <= 0
            counts[i] = is_child0(k) ? Int32(1) : Int32(0)
        else
            counts[i] = Int32(1)
        end
    end
end

@kernel function scatter_kernel!(out, @Const(keys), @Const(counts), @Const(offsets),
                                 @Const(cx), @Const(cy), @Const(cf),
                                 @Const(basecorners), @Const(basecell),
                                 geo_tol, fld_tol, max_depth)
    i = @index(Global)
    @inbounds begin
        k = keys[i]
        n = counts[i]
        o = offsets[i]
        if n == Int32(2)
            out[o + 1] = key_child(k, 0)
            out[o + 2] = key_child(k, 1)
        elseif n == Int32(1)
            d = key_depth(k)
            merging = d > 0 &&
                      excess_levels(key_parent(k), cx, cy, cf, basecorners, basecell, geo_tol, fld_tol) <= 0 &&
                      !(d < max_depth && excess_levels(k, cx, cy, cf, basecorners, basecell, geo_tol, fld_tol) > 0)
            out[o + 1] = merging ? key_parent(k) : k
        end
    end
end

function refine!(keys, cx, cy, cf, basecorners, basecell; geo_tol, fld_tol, max_depth)
    n = length(keys)
    counts = similar(keys, Int32, n)
    backend = get_backend(keys)
    # The coefficients' precision: a Float64 tolerance would promote the kernel to double.
    geo_tol = eltype(cf)(geo_tol)
    fld_tol = eltype(cf)(fld_tol)
    classify_kernel!(backend, 64)(counts, keys, cx, cy, cf, basecorners, basecell,
                                  geo_tol, fld_tol, max_depth; ndrange = n)
    offsets = accumulate(+, counts) .- counts
    total = Int(scalarat(offsets, n)) + Int(scalarat(counts, n))
    out = similar(keys, total)
    scatter_kernel!(backend, 64)(out, keys, counts, offsets, cx, cy, cf,
                                 basecorners, basecell,
                                 geo_tol, fld_tol, max_depth; ndrange = n)
    return out
end

scalarat(a::AbstractArray, i::Integer) =
    Mantle.isdevicearray(a) ? only(Array(view(a, i:i))) : a[i]

function refine(keys0, cx, cy, cf, basecorners, basecell; geo_tol, fld_tol, max_depth = 12)
    keys = keys0
    for _ in 1:(max_depth + 1)
        new = refine!(keys, cx, cy, cf, basecorners, basecell; geo_tol, fld_tol, max_depth)
        length(new) == length(keys) && Array(new) == Array(keys) && break
        keys = new
    end
    return keys
end

rootkeys(nbase::Integer = 8) = UInt64[root_key(b) for b in 1:nbase]

# A function, not a global: GPU stages read it.
@inline meshthreads() = 32

sundirection(; azimuth = deg2rad(140.0), elevation = deg2rad(38.0)) =
    Vec3f(cos(elevation) * cos(azimuth), cos(elevation) * sin(azimuth), sin(elevation))

function demoenv(; sun = sundirection(), intensity = 1.0f0, turbidity = 2.2f0,
                 groundtile = 0.34f0, sunangle = 2.5f0, sunradiance = 830.0f0)
    envlight, _ = Hikari.sunsky_to_envlight(direction = sun,
                                                   intensity = intensity,
                                                   turbidity = turbidity,
                                                   ground_enabled = false,
                                                   resolution = 1024)
    sq = copy(envlight.env_map.data)
    n = size(sq, 1)
    acc = Hikari.RGBSpectrum(0f0); cnt = 0
    for j in 1:n, i in 1:n
        d = Hikari.equal_area_square_to_sphere(Point2f((i - 0.5f0)/n, (j - 0.5f0)/n))
        if abs(d[3]) < 0.02f0
            acc = Hikari.RGBSpectrum((acc.c .+ sq[j, i].c)...); cnt += 1
        end
    end
    hz = Hikari.RGBSpectrum((acc.c ./ max(cnt, 1))...)
    for j in 1:n, i in 1:n
        d = Hikari.equal_area_square_to_sphere(Point2f((i - 0.5f0)/n, (j - 0.5f0)/n))
        d[3] >= 0f0 && continue
        t = (-d[3])^2
        f = 1f0 - 0.72f0 * t
        gx = d[1] / max(-d[3], 1f-3)
        gy = d[2] / max(-d[3], 1f-3)
        r = sqrt(gx * gx + gy * gy)
        fade = clamp(1f0 - r / 9f0, 0f0, 1f0)
        sx = sinpi(gx / groundtile); sy = sinpi(gy / groundtile)
        w = clamp(0.05f0 * (1f0 + 0.35f0 * r), 0.05f0, 1f0)
        tile = 0.55f0 + 0.45f0 * (0.5f0 * (1f0 + clamp(sx * sy / w, -1f0, 1f0)))
        g = 1f0 - fade * (1f0 - tile)
        sq[j, i] = Hikari.RGBSpectrum(hz.c[1]*f*g*0.96f0, hz.c[2]*f*g*0.94f0, hz.c[3]*f*g*0.88f0)
    end
    sd = normalize(sun)
    cosr = cos(deg2rad(sunangle))
    for j in 1:n, i in 1:n
        d = Hikari.equal_area_square_to_sphere(Point2f((i - 0.5f0)/n, (j - 0.5f0)/n))
        c = d[1]*sd[1] + d[2]*sd[2] + d[3]*sd[3]
        c <= cosr && continue
        t = clamp((c - cosr) / ((1f0 - cosr) * 0.34f0), 0f0, 1f0)
        w = sunradiance * (t * t * (3f0 - 2f0 * t))
        sq[j, i] = Hikari.RGBSpectrum(sq[j, i].c[1] + w,
                                      sq[j, i].c[2] + w * 0.95f0,
                                      sq[j, i].c[3] + w * 0.86f0)
    end
    return sq
end
