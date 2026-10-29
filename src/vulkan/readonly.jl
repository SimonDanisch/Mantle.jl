# Which arguments of a compute dispatch Lava may load as memory nothing writes.
#
# Lava loads a `Lava.Const` with `!invariant.load` and declares that memory
# NonWritable, so a driver may read a uniform address through its scalar cache
# (RADV only does for memory it knows is not written). The access walk already says
# what a kernel does to each argument. It cannot say whether two arguments reach the
# same buffer, and NonWritable is a promise about the memory, not the argument:
# `x .= x .+ 1` reads `x` through the `Broadcasted` and writes it as the destination.
# That half is decided per dispatch, from the buffers the arguments reach.
#
# `packdispatch!` packs the unwrapped arguments: `Lava.Const` has the layout of the
# array it wraps, so the bytes are the ones the pipeline was compiled for, the patch
# points a move rewrites are the array's, and recording stays free of the analysis.

"""
    readonlyargs(g, touches, raw, adapted) -> Tuple

`adapted`, a dispatch's arguments after `LavaAdaptor`, with every argument the kernel
reads and does not write, and whose buffers no written argument reaches, adapted
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
# storage owners by identity, then adapt through one kernel-independent walk.
@noinline Base.@nospecializeinfer function readonlyargs(g::Graph, touches::AbstractVector{Touch},
        @nospecialize(raw::Tuple), @nospecialize(adapted::Tuple))
    any(isreadonly, touches) || return adapted
    written_buffers = Base.IdSet{Any}()
    for i in eachindex(touches)
        touches[i].write || continue
        for resource in resourceleaves!(Any[], g, raw[i])
            push!(written_buffers, bufferof(resource))
        end
    end
    result = Vector{Any}(undef, length(adapted))
    for i in eachindex(adapted)
        wrap = isreadonly(touches[i])
        if wrap
            for resource in resourceleaves!(Any[], g, raw[i])
                if bufferof(resource) in written_buffers
                    wrap = false
                    break
                end
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
           !anybuffer(b -> written(g, touches, all, b, 1), g, first(raw))
    return (wrap ? Adapt.adapt(KA.ConstAdaptor(), a) : a,
            constargs(g, touches, all, Base.tail(raw), Base.tail(adapted), i + 1)...)
end

"""Whether one of `args`, the `j`th argument onwards, is written and reaches buffer `b`."""
written(g, touches, ::Tuple{}, b, j::Int) = false
written(g, touches, args::Tuple, b, j::Int) =
    (touches[j].write && anybuffer(c -> c === b, g, first(args))) ||
    written(g, touches, Base.tail(args), b, j + 1)

"""
Whether `p` holds for a buffer `x` reaches, by identity (`===`, since
`resource_identity` may answer an array, which `==` compares element by element),
following slices and views to their parent.
"""
anybuffer(p, g, x) = isdevicearray(x) ? p(bufferof(x)) :
                     any(r -> p(bufferof(r)), resourceleaves!(Any[], g, x))

bufferof(r) = resource_identity(rootresource(r))
