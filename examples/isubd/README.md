# Adaptive tessellation of curved FEM cells, on Mantle

Two renderers for the same two quadratic elements, behind one button:

* **raster** — a **mesh shader** expands subdivision keys into triangles. No
  vertex buffer and no index buffer: the geometry is reconstructed from a
  `UInt64` key inside the shader and never exists in memory.
* **path traced** — a Makie `mesh!` whose `material` is `Hikari.FEMMaterial`,
  rendered by RayMakie. **Two AABBs, no triangles**: the material carries the
  coefficients, so the ray is solved against the isoparametric map itself. Exact
  silhouette at any zoom, real shadows, real Monte Carlo noise — and a real
  BSDF, because `FEMMaterial` wraps ANY Hikari material and merges the field
  into whatever that one uses as its colour:

  ```julia
  FEMMaterial(cx, cy, cf)                                 # matte
  FEMMaterial(CoatedDiffuse(roughness = 0.04), cx, cy, cf) # under a clear coat
  FEMMaterial(Conductor(roughness = 0.06), cx, cy, cf)     # metal
  FEMMaterial(Dielectric(index = 1.5), cx, cy, cf)         # glass
  ```

Both call the same `eval_poly`, and the CPU reference the tests compare against
calls it too. The sky is a Hosek-Wilkie atmosphere computed from
`SUN_DIRECTION`, baked once and handed to each renderer in its own convention.

```
julia> include("isubd_gui.jl");  isubd_gui(backend, cx, cy, cf)   # the window
julia> include("record_demo.jl")                                  # the video
```

## The demo

`record_demo.jl` writes `isubd_demo.mp4` into the working directory. Nothing is
checked in: a recording and the comparison renders are artifacts of RUNNING this,
not part of it, and an `examples/` directory that ships them puts megabytes into
every clone forever.

The same two elements render as matte, clear-coated, gold and glass — which is
`FEMMaterial` wrapping four different Hikari materials, and one line each.

The arc: an adaptive mesh with the triangles drawn, refined until there are
hundreds of them and they are visibly smaller where the field curves; zoom onto
the most curved element edge until individual triangles and the straight
segments of the boundary are plain; switch to the traced path, where the same
edge is a curve; switch back and refine to **10665 triangles** to agree with
what two boxes did; and end zoomed onto one ridge, where 44 triangles give a
hard crease with a visible fan of facets and the traced path gives a curve.

One button flips the two in place — same camera, same sky — so the only thing
that changes is the geometry.

## What is in here

| file | |
|---|---|
| `reference.jl` | the CPU reference, **vendored unedited** from [koehlerson/mantle_mwe](https://github.com/koehlerson/mantle_mwe) @ `d32c990` |
| `REFERENCE_README.md` | its own README — read this for the FEM side |
| `mantle_isubd.jl` | the port: passes A (compute + scan), B (mesh shader), C (fragment), D (ray traced) |
| `isubd_gui.jl` | the Makie GUI, on RayMakie |
| `record_demo.jl` | drives the GUI with a visible cursor and records it |

The reference is not edited, deliberately: a reference that drifts toward the
implementation it checks has stopped being one. It earned that on the first run
by catching a swapped pair of base corners — eight plausible triangles, the
split edge on a fan diagonal instead of an element edge, and a triangle *count*
would never have shown it.

Tested by `Mantle/test/vulkan/test_isubd_mesh.jl` (Tier 3h): key sets compared
exactly, images compared per pixel, and refinement AND coarsening driven through
all their intermediate states.

## Things that will bite the next person

* **A tuple indexed at a runtime position** gives
  `Unsupported ConstantExpr opcode: LLVMICmp` from the SPIR-V emitter, with no
  source line. It happened four times here — the base corners, `mono[m]`, the
  estimator's `SAMPLES` loop, and the Newton seeds. Fix is `ntuple(…, Val(N))`
  or `Base.Cartesian.@nexprs`; the base triangles became buffers, which is the
  right shape anyway.
* **Newton has to check CONVERGENCE, not just that ξ landed in `[-1,1]²`.** An
  iterate that has not converged but happens to fall in range reports a hit and
  the image fills with speckle along every grazing edge.
* **One Newton seed is not enough** — a ray can cross a warped element twice, so
  which root a single solve finds depends on where it started. Nine seeds, take
  the smallest `t`.
* **Inline ray query carries no hit attributes for a generated intersection.**
  The shader keeps its own best hit; only ever generating an *improving* `t` is
  what makes that local best the one traversal commits.
* **Setup does not belong in the per-frame path.** Building the pipeline and the
  acceleration structure per frame cost 1061 ms; the work itself is 0.1 ms.
* **The environment was the picture.** At 1024×512 a 46° field of view covers
  131 source pixels stretched over 560 on screen, and the frame read as mush no
  matter what the material did. It is a 1:1 copy of the 4096×2048 file now —
  100 MB on the device, and the single largest change to how this looks.
* **The background went through four versions**, and each failure looked like a
  different problem. A measured outdoor HDRI is sharp and photographic and takes
  the eye to the gravel. A sky dome is calm and has NOTHING below the horizon,
  so half the frame is black. Turning Hosek's ground off fills the lower
  hemisphere with one clamped horizon colour — a flat grey slab, which reads as
  a wall. Grading it downward is what finally makes it ground.
* **Equirectangular and equal-area octahedral look interchangeable and are
  not.** Hikari samples the second; every HDRI you can download is the first.
  Read one as the other and you get a plausible sky that is wrong in every
  direction — the ground ends up overhead. `equirect_to_equalarea` and its
  inverse convert at the boundary.
* **A specular lobe has to START above where the surface sits.** At 0.65 the
  highlight was measurably present — 4% of the visible surface inside the lobe —
  and invisible: a 65% boost on a floor near 1.0, which `x/(1+x)` flattens to
  about 10% on screen. 3.5 reads.
* **One ray per pixel is a binary hit test**, and a curved surface against a sky
  is all silhouette. `SS = 2` in the GUI renders BOTH paths at 2× and box-filters
  down — per-path supersampling would flatter whichever one got it.
