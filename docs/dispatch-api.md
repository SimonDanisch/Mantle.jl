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

`argument_usage(device, kernel, args, ndrange, group)` answers one `Touch` per
argument: read, write, and whether every write is one nothing can collide with.
It has two sources and no third — a DECLARATION for a callable with no body on
this side of the boundary, and INFERENCE for a kernel — and raises if neither
applies.

Inference is `kerneltouches`, where the backend supplies the SIGNATURE (which
interpreter, what the device-side type of each argument is, what leading
arguments the kernel has of its own) and core walks the IR. The signature is
built at `Core.Typeof` and not `typeof`, because a type-valued argument is
`Type{Float32}` in the signature the kernel is compiled at and `DataType` under
`typeof`, which is not a dispatch-tuple element.

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

**Everything it cannot see it REFUSES.** It used to widen to read+write, on the
grounds that claiming an access that does not happen only costs a barrier while
missing one costs a race. True, and beside the point: read+write is
indistinguishable from a proof, so a kernel that stopped being analysable kept
compiling and paid that barrier for ever with nothing to say so. Measured cost
of that, on the hottest kernel in the tree: `gemm_cm2!` takes
`C, @Const(A), @Const(B)` and declared all three read+write, because
`@_lava_coopmat_load_f16_16x16_a` is not a `load` instruction and the classifier
was looking for the substring `"load "`.

Four of the five constructs that used to widen cannot reach a compiled kernel at
all -- a `ccall` is C, SPIR-V forbids recursion, GPUCompiler rejects a dynamic
call, and a signature that does not infer does not compile -- so refusing them
moves an error that was going to happen anyway from `Pipelines` to `dispatch!`,
where the message can name the argument. The fifth is an `llvmcall` whose symbol
nothing declares, and `intrinsic_usage` is what answers that.

One construct had to be learned rather than refused, and it is most kernels: a
call that CANNOT RETURN. `throw_boundserror` and `throw_methoderror` infer to
`Union{}` and are reached from the bounds check and the `setindex!` fallback of
ordinary code. An abort path contributes nothing, and soundly — a store on the
way to a throw cannot be observed by a later pass, because the invocation does
not complete.

`test/vulkan/test_access.jl` pins both halves: that known kernels come out
exact, and that an undeclared intrinsic and an uninferable kernel are refused by
name.

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

Not at the call site, and this is the one that changed twice: what a CALL does to
its arguments. It was `use`, then `Read`/`Write`/`ReadWrite` wrappers one per
argument, and both are the same mistake — one fact restated at N call sites, with
the N+1st getting it wrong. `mul!(C, A, B, α, β)` is `C = A*B*α + C*β`, so `C` is
read as well as written, and a call site that copied `Write(C)` from the
three-argument form is a missing barrier rather than a compile error. So the
direction belongs to the function: `argument_usage(f, args)`, declared once next
to it, and `mul!` is declared in core because writing its first argument is true
of rocBLAS, of a cooperative-matrix kernel and of MPS alike.

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
| `src/graph/access.jl` | `Touch`, `argument_usage`, `intrinsic_usage`, the taint walk, `accessof`, the device cache |
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
