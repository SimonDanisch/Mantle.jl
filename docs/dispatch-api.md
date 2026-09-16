# One launch, one pass, and no declarations

What a consumer writes to put work in a graph, after the refactor that deleted
`compute!` and `use`. Sibling of `submission-refactor.md` and
`backend-independence.md`; like those, it describes what IS, and the "was"
paragraphs are there because the reason is the interesting part.

## The whole surface

```julia
dispatch!(g, shade!, (radiance, queue, materials), n; group = 256, name = "shade")
trace!(g, rt_pipeline, accel, (queue, film), DeviceRange(queue.size))
render!(g, "gbuffer", colour => Clear(black), depth) do p
    draw!(p, MESH, (Attribute(p, positions), mvp), positions)
end
repeat!(g, bump!, (queue, film), DeviceRange(queue.size); max = 8,
        while_nonzero = queue.size)
```

A dispatch is a pass. A trace is a pass. A render pass groups draws, because
they share attachments and that is what a render pass IS on every backend. There
is no fourth thing, and nothing anywhere says what it reads or writes.

## What was deleted

```julia
compute!(g, "shade") do p                      # gone
    use(p, radiance; read = true, write = true, unordered = true)   # gone
    use(p, queue; read = true)                                      # gone
    dispatch!(p, shade!, (radiance, queue, materials), n)
end
```

`use` was a SECOND source for something the kernel already stated. The call site
said `write = true`; the body did the storing; and when one changed the other did
not. Every usage is now derived from the kernel itself — `dispatch!` from its
body, `draw!` from its two stages, `trace!` from every shader in the pipeline —
by the taint walk in `src/graph/access.jl`.

`compute!` existed for two reasons and neither survived it. It handed a body a
pass to declare into, which nothing needs any more. And it grouped dispatches so
that no barrier fell between them — an assertion the author made and the graph
could not check, "these two write the same buffer and do not collide", which is
now derived like everything else.

## How the access is known

`kerneltouches(device, kernel, args, ndrange, group)` answers one `Touch` per
argument: read, write, and whether every write is one nothing can collide with.
The backend supplies the SIGNATURE — which interpreter, what the device-side type
of each argument is, what leading arguments the kernel has of its own — and core
walks the IR.

It is the same typed IR the backend is about to compile, which is the only IR
worth walking: a device array's `getindex` lives on the backend's method table
and nowhere else, so inference run without it reaches
`error_if_canonical_getindex` and concludes the kernel does nothing at all.

Four things it gets right that a simpler walk does not:

* **A scalar is not an access.** Taint follows types that can reach memory. A
  `Float32` handed to a call the walk cannot see through comes back untouched,
  not read-write.
* **A length is not a read.** `length(other)` is a by-value field of the array
  handle, so an argument a kernel only measures is untouched.
* **A vararg tail is not one argument.** `f(queue, args...)` — every wavefront
  stage — is FOUR arguments in the inferred IR and twenty-four in the signature,
  the tail collapsed into a tuple. The walk tracks which element of that tuple a
  value came from, or the stage that appends to one ray queue is recorded as
  writing all of them.
* **A claimed slot cannot collide.** A store at an index handed back by an
  atomic read-modify-write on the same argument is disjoint from every other
  invocation's, which is exactly what a work queue's append is. That is
  `Unordered`, and it is what `accumulates!` used to assert by hand.

Everything it cannot see widens to read+write. That is the only safe direction —
claiming an access that does not happen costs a barrier, missing one costs a
race — and `test/vulkan/test_access.jl` pins both halves: that known kernels come
out exact, and that an opaque call comes out conservative.

The answers are cached on the DEVICE, keyed by signature, and dropped whole when
the world age moves. A wavefront stage is ten thousand statements after inlining
and takes seconds to walk; a bounce loop asks for the same one every round.

## What a call site still says

Three things, because none of them is an access:

* **`name`** — what the pass is called in a profile. Defaults to the kernel's
  own name, or for a higher-order kernel to the inner one, so twenty stages are
  not all called `workqueue_map_kernel!`.
* **`group`** — the workgroup size. Worth giving for anything image-shaped: the
  default partitions an ndrange along its first axis, which for a 2-D range makes
  a workgroup one long row.
* **`slice(g, buffer, range)`** — which PART of a buffer a pass touches. This was
  `use`'s `range` keyword and it is not an access at all, so it became an
  argument: the slice is what the dispatch is handed and what the pass is
  recorded touching.

## Consequences a reader should know about

**More passes.** One dispatch is one pass, so a body that held four dispatches is
four passes. Nothing serialises that did not before: passes with no hazard
between them get no barrier, which is what the grouping was asserting.

**Declarations can now be wider than they were.** Inference finds reads of scene
data that call sites never declared — the BVH, the material arrays, the
RGB-to-spectrum table. Nothing in those graphs writes them, so no barrier
appears; what changes is that the graph now knows they are read.

**And narrower.** An emitters stage in a scene with no area lights touches
nothing at all, because the loop over area lights specialises away. The
declaration follows the kernel that was actually compiled for this scene.

## Where it is implemented

| file | what |
|---|---|
| `src/graph/access.jl` | `Touch`, the taint walk, `accessof`, the device cache |
| `src/graph/build.jl` | `dispatch!`, `draw!`, `declare!`, `resourceleaves!`, `slice` |
| `src/raytracing/api.jl` | `trace!`, and the shaders a trace declares from |
| `src/vulkan/access.jl` | the signatures: compute, ray tracing, and the two draw stages |
| `src/host/host.jl` | the same through Julia's own method table |
| `test/vulkan/test_access.jl` | what right looks like, on kernels |
| `test/test_access.jl` | and what the walk decides from a type alone |

## Not built

**Field-sensitive taint past the first level.** An argument's `Touch` applies to
all of its leaves, so a kernel handed a multi-type queue container declares every
queue in it. That is exactly the granularity the hand-written declarations used,
so nothing regressed — but a container passed whole and used in part is coarser
than it could be.

**`unordered` for anything but a claimed index.** Two passes writing provably
disjoint halves of a buffer by construction — one takes the even lanes, one the
odd — are ordered against each other. `slice` is the way to say that today.
