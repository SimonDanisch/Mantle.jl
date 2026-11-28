# Which arguments of a compute dispatch nothing writes while it runs.
#
# Lava loads the memory such an argument reaches with `!invariant.load` and declares it
# NonWritable, so a driver may read a uniform address through its scalar cache (RADV only
# does for memory it knows is not written). The access walk already says what a kernel
# does to each argument. It cannot say whether two arguments reach the same buffer, and
# NonWritable is a promise about the memory, not the argument: `x .= x .+ 1` reads `x`
# through the `Broadcasted` and writes it as the destination. That half is decided per
# dispatch, from the byte ranges the arguments reach.
#
# The answer is a mask, one bit per argument, which the compile job carries
# (`Lava.LavaCompilerParams`): the argument types stay the ones the walk inferred, so the
# compile finds the walk's inference cached. Wrapping the arguments in `Lava.Const`
# instead gave the compile a signature of its own, and Julia inferred every kernel twice.
# A mask, not a tuple of flags, so no launch path is compiled per argument count, and an
# eager launch compares it with its cached plans without allocating. An argument past
# the 64th is never read-only: it loses NonWritable, nothing else.

"""
    readonlymask(g, touches, raw) -> UInt64

Bit `i - 1` set for each of a dispatch's arguments `raw` that the kernel reads and does
not write, and that no argument it writes reaches the same bytes of. `touches` is what
the kernel does to each, `g` the graph slices are interned in (`nothing` for an eager
launch).
"""
function readonlymask(g, touches::AbstractVector{Touch}, raw::Tuple)
    any(isreadonly, touches) || return UInt64(0)
    return flagged(g, touches, raw, raw, 1)
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

@noinline Base.@nospecializeinfer function readonlymask(g::Graph, touches::AbstractVector{Touch},
                                                        @nospecialize(raw::Tuple))
    mask = UInt64(0)
    any(isreadonly, touches) || return mask
    written_regions = ReadonlyRegion[]
    for i in eachindex(touches)
        touches[i].write || continue
        for resource in resourceleaves!(Any[], g, raw[i])
            push!(written_regions, readonlyregion(resource))
        end
    end
    for i in 1:min(nfields(raw), 64)
        readonly = isreadonly(touches[i])
        if readonly
            for resource in resourceleaves!(Any[], g, raw[i])
                region = readonlyregion(resource)
                for written in written_regions
                    if readonlyoverlap(region, written)
                        readonly = false
                        break
                    end
                end
                readonly || break
            end
        end
        readonly && (mask |= UInt64(1) << (i - 1))
    end
    return mask
end

isreadonly(t::Touch) = t.read & !t.write

# Walked as a tuple, so an eager launch, which takes this path on each call, allocates
# nothing here for an argument that is a device array. The recursion is the only part
# compiled per argument tuple, and it is small: whether a written argument reaches
# `first(raw)` is `writtenalias`, compiled once, comparing pairs through `argsoverlap`,
# compiled per pair of argument types, which kernels share. Closures over the whole tuple
# there compiled anew for every kernel, about 70 ms each for every argument.
flagged(g, touches, all::Tuple, ::Tuple{}, i::Int) = UInt64(0)
flagged(g, touches, all::Tuple, raw::Tuple, i::Int) =
    (i <= 64 && isreadonly(touches[i]) && !writtenalias(g, touches, all, first(raw)) ?
     UInt64(1) << (i - 1) : UInt64(0)) | flagged(g, touches, all, Base.tail(raw), i + 1)


"""Whether an argument of `all` that the kernel writes reaches a buffer `x` reaches."""
@noinline Base.@nospecializeinfer function writtenalias(g, touches, @nospecialize(all::Tuple),
                                                        @nospecialize(x))
    # `nfields`/`getfield` are builtins: iterating the abstract tuple would dispatch on,
    # and compile for, its concrete type.
    for j in 1:nfields(all)
        touches[j].write || continue
        argsoverlap(g, getfield(all, j), x) && return true
    end
    return false
end

"""Whether a resource `written` reaches overlaps one `x` reaches."""
argsoverlap(g, written, x) = anyresource(b -> anyresource(c -> readonlyoverlap(c, b), g, written), g, x)

"""
Whether `p` holds for a resource `x` reaches, following slices and views.
"""
anyresource(p, g, x) = isdevicearray(x) ? p(x) :
                       any(p, resourceleaves!(Any[], g, x))

readonlyoverlap(a, b) = bufferof(a) === bufferof(b)
readonlyoverlap(a::LavaArray, b::LavaArray) = Base.mightalias(a, b)

bufferof(r) = resource_identity(rootresource(r))
