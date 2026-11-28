# GPURef and drawn attributes.
#
# A GPURef holds a value: one slot in device memory, and `r[] = v` is a
# scheduled store into it (api.md 2.5, internals.jl section 7). An array
# argument is the array itself: a drawn attribute is a plain MantleArray,
# mutated in place, of length 1 for one value or n for one per element, read
# on the device as attribute(a, i) = a[min(i, length(a))] (api.md 2.6b,
# resizing-and-raytracing.jl A and D.3). The cases here check the attribute.
#
# What this file assumes where the docs name nothing:
#   Mantle.attribute(a, i)  the device function (name open in api.md 2.6b).
#   Mantle.pendingof(x)  Change objects, in sequence order.
# Validation-layer and GPU-AV expectations are comments marked VAL.

# ── This file's helpers ──

"""The kinds of the changes pending on `x`, in sequence order."""
pendingkinds(x) = [nameof(typeof(c)) for c in Mantle.pendingof(x)]

"""out[i] = attribute(a, i): one value for every i, or one per element."""
function ab_attribute!(out, a)
    i = KI.get_global_id().x
    i <= length(out) || return nothing      # ndrange is rounded up to whole work-groups
    out[i] = Mantle.attribute(a, i)
    return nothing
end

# ── Drawn attributes ──

# api.md 2.6b, resizing-and-raytracing.jl A and D.3: attribute(a, i) = a[min(i, length(a))]: one argument type for one value and one per element; switching compiles nothing. Length 1, then n, then 1.
@case "AB-21" 6 begin
    dev = testdevice()
    n = 64
    color = MantleArray(dev, [2f0])
    out = MantleArray(dev, Float32, n)
    g = Graph(dev)
    dispatch!(g, ab_attribute!, (out, color), n)
    run!(g)
    @test all(==(2f0), Array(out))
    data = testdata(Float32, n)
    _, d = counted(() -> (copy!(color, data); run!(g)))
    @test d.plancompiles == 0 && d.kernelcompiles == 0 && d.pipelinecompiles == 0
    @test Array(out) == data
    copy!(color, [3f0]); run!(g)
    @test all(==(3f0), Array(out))
end

# api.md 2.2, resizing-and-raytracing.jl A and B.5: copy!(color, data) with a new length pends a Resize and a Store on the attribute itself, which is mutated in place. One value, then one per element.
@case "AB-22" 6 begin
    dev = testdevice()
    n = 64
    color = MantleArray(dev, [2f0])
    out = MantleArray(dev, Float32, n)
    dispatch!(dev, ab_attribute!, (out, color), n)
    data = testdata(Float32, n)
    copy!(color, data)
    @test pendingkinds(color) == [:Resize, :Store]
    dispatch!(dev, ab_attribute!, (out, color), n)
    @test Array(out) == data
end

# api.md 2.6b (contract open: the docs do not decide attribute(a, i) of an empty a): a colour attribute emptied while the launch keeps n elements. The run completes and writes every output.
@case "AB-23" 6 begin
    dev = testdevice()
    n = 64
    color = MantleArray(dev, [2f0])
    out = MantleArray(dev, Float32, n)
    g = Graph(dev)
    dispatch!(g, ab_attribute!, (out, color), n)
    run!(g)
    resize!(color, 0)
    fill!(out, -1f0)
    run!(g)
    @test length(Array(out)) == n
    # VAL: GPU-AV clean: attribute(color, i) reads no element of the empty array (the value is open)
end
