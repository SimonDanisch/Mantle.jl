# Which arguments of a compute dispatch Lava may load as memory nothing writes.
#
# Lava loads a `Lava.Const` with `!invariant.load` and declares that memory
# NonWritable, so a driver may read a uniform address through its scalar cache
# (RADV only does for memory it knows is not written). The access walk already says
# what a kernel does to each argument. It cannot say whether two arguments reach the
# same buffer, and NonWritable is a promise about the memory, not the argument:
# `x .= x .+ 1` reads `x` through the `Broadcasted` and writes it as the destination.
# That half is decided per dispatch, from the byte ranges the arguments reach.
#
# `packdispatch!` packs the unwrapped arguments: `Lava.Const` has the layout of the
# array it wraps, so the bytes are the ones the pipeline was compiled for, the patch
# points a move rewrites are the array's, and recording stays free of the analysis.

"""
    readonlyargs(g, touches, raw, adapted) -> Tuple

`adapted`, a dispatch's arguments after `LavaAdaptor`, with every argument the kernel
reads and does not write, and whose bytes no written argument reaches, adapted
through `KA.ConstAdaptor`: each device array in it becomes a `Lava.Const`. `raw` are
the same arguments before adapting, `touches` what the kernel does to each, and `g`
the graph slices are interned in (`nothing` for an eager launch).
"""
function readonlyargs(g, touches::AbstractVector{Touch}, raw::Tuple, adapted::Tuple)
    any(isreadonly, touches) || return adapted
    return constargs(g, touches, raw, raw, adapted, 1)
end

# A graph computes these aliases once, while compiling its plan. The tuple
# recursion below is for eager launches; specializing it on every dispatch's
# argument tuple cost eight seconds of first-render compilation. Collect written
# byte ranges, then adapt through one kernel-independent walk. Distinct arrays
# allocated from one pool slab can share an owner without sharing any bytes.
# Comparing these summaries avoids compiling a method for every pair of array
# element types reached by a large dispatch.
struct ReadonlyRegion
    owner::Any
    first::UInt128
    past::UInt128
    bounded::Bool
end

readonlyregion(x) = ReadonlyRegion(bufferof(x), 0, 0, false)
function readonlyregion(x::LavaArray)
    base, offset = GPUArrays.memory_location(x)
    start = UInt128(base) + offset
    return ReadonlyRegion(bufferof(x), start, start + sizeof(x), true)
end
# A slice of a native array's owner (`nativeslice`), in bytes from the owner's start.
function readonlyregion(r::BufferRange)
    isdevicearray(r.parent) || return ReadonlyRegion(bufferof(r), 0, 0, false)
    base, _ = GPUArrays.memory_location(r.parent)
    start = UInt128(base) + first(r.range) - 1
    return ReadonlyRegion(bufferof(r), start, start + length(r.range), true)
end
readonlyoverlap(a::ReadonlyRegion, b::ReadonlyRegion) = a.bounded && b.bounded ?
    a.first < a.past && b.first < b.past && a.first < b.past && b.first < a.past :
    a.owner === b.owner

@noinline Base.@nospecializeinfer function readonlyargs(g::Graph, touches::AbstractVector{Touch},
        @nospecialize(raw::Tuple), @nospecialize(adapted::Tuple))
    any(isreadonly, touches) || return adapted
    written_regions = ReadonlyRegion[]
    for i in eachindex(touches)
        touches[i].write || continue
        for resource in resourceleaves!(Any[], g, raw[i])
            push!(written_regions, readonlyregion(resource))
        end
    end
    result = Vector{Any}(undef, length(adapted))
    for i in eachindex(adapted)
        wrap = isreadonly(touches[i])
        if wrap
            for resource in resourceleaves!(Any[], g, raw[i])
                region = readonlyregion(resource)
                for written in written_regions
                    if readonlyoverlap(region, written)
                        wrap = false
                        break
                    end
                end
                wrap || break
            end
        end
        result[i] = wrap ? Adapt.adapt(KA.ConstAdaptor(), adapted[i]) : adapted[i]
    end
    return Tuple(result)
end

isreadonly(t::Touch) = t.read & !t.write

# Walked as a tuple, so every argument keeps its type: an eager launch takes this path
# on each call, and an argument that is a device array allocates nothing here.
constargs(g, touches, all::Tuple, ::Tuple{}, ::Tuple{}, i::Int) = ()
function constargs(g, touches, all::Tuple, raw::Tuple, adapted::Tuple, i::Int)
    a = first(adapted)
    wrap = isreadonly(touches[i]) &&
           !anyresource(b -> written(g, touches, all, b, 1), g, first(raw))
    return (wrap ? Adapt.adapt(KA.ConstAdaptor(), a) : a,
            constargs(g, touches, all, Base.tail(raw), Base.tail(adapted), i + 1)...)
end

"""Whether one of `args`, the `j`th argument onwards, is written and reaches buffer `b`."""
written(g, touches, ::Tuple{}, b, j::Int) = false
written(g, touches, args::Tuple, b, j::Int) =
    (touches[j].write && anyresource(c -> readonlyoverlap(c, b), g, first(args))) ||
    written(g, touches, Base.tail(args), b, j + 1)

"""
Whether `p` holds for a resource `x` reaches, following slices and views.
"""
anyresource(p, g, x) = isdevicearray(x) ? p(x) :
                       any(p, resourceleaves!(Any[], g, x))

readonlyoverlap(a, b) = bufferof(a) === bufferof(b)
readonlyoverlap(a::LavaArray, b::LavaArray) = Base.mightalias(a, b)

bufferof(r) = resource_identity(rootresource(r))
