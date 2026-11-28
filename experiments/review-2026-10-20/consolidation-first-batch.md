# First implementation consolidation — 2026-10-21

Compared with Mantle `e03a365` (implementation unchanged from `45021c6`).
Source changes consolidate six duplicated implementation steps; the broader
memory/lifetime, cache, execution and TLAS replacement remains open.

| Shared implementation | Result |
|---|---|
| Vulkan memory-type search | The throwing selector delegates to the optional selector, retaining its existing error on no match and its zero-based UInt32 result. |
| Fragment input type construction | One derived implementation uses the existing raster/geometry/mesh `fragmentinputs` dispatch. |
| Borrowed Metal arrays | `deviceview` delegates to `deviceslice`, retaining the non-owning DataRef, shared managed buffer, dimensions, offset and remaining capacity. |
| Metal raster/mesh attachments and depth state | Shared attachment and depth-state helpers; native descriptor types, stage compilation and pipeline keys remain distinct. The raster path retains its depth-target validation. |
| TLAS mesh preparation | One core `meshtriangles` converts meshes, filters degenerate faces and preserves face ids, typed metadata, normals and UVs. Both backend callers use it. |
| TLAS vertex expansion | One core `trianglevertices` supplies contiguous Float32 triples. Vulkan consumes them as vertices; Metal copies the same packed triples into its native buffer, with no second host payload allocation. |

## Qualification executed

| Check | Result |
|---|---|
| Existing `test_pipeline_stages.jl` and `test_lowering.jl` | 165 host assertions pass. |
| New `test_raytracing_geometry.jl`, included in the regular portable suite | 15 host assertions pass: degeneration, retained face numbering, typed face metadata, normals, explicit/default UVs, tuple packing, exact bytes/capacity and no-triangle error. |
| Native dedicated and host-visible allocation through Mantle `try_vk_alloc`, plus optional/throwing no-match behavior | 16 assertions pass across AMD RX 7900 XTX and NVIDIA RTX 4000 Ada. |
| Existing `test_hwtlas_batch_triangles.jl` | 13 assertions per GPU, 26 total: actual BLAS/TLAS construction and tracing, including a batch with no triangles and offsets into subsequent batches. |
| Validation/synchronization-validation message checks after native runs | 2 assertions pass, no captured messages on either GPU. |
| Source parsing | All 112 current Julia files in `src/` and `ext/` parse with JuliaSyntax; 1,990 methods/regions extracted. |

Total: 224 assertions pass. This is targeted qualification, not a full legacy
suite run or execution of the unimplemented refactor specification.

Metal cannot run on the local Linux host. Its modified sources parse and the
shared host data layout is exercised, but borrowed-view ownership and native
raster/mesh/BLAS execution still need Metal qualification. This limitation
does not justify introducing another implementation or changing the API.

During development, a host packing check caught that `sizeof` of a reinterpret
wrapper measures its header rather than its payload. The final Metal caller
uses the concrete packed vertex vector directly for both buffer capacity and
copy length; no reinterpret-wrapper sizing or additional host copy remains.

## Size and remaining obligations

Physical source inventory (`src/` + `ext/`, including comments and blanks):
111 files / 55,115 lines before; 112 files / 55,073 lines after. Net reduction:
42 lines, including the new shared geometry file. Tests, tooling and reports
are outside that count. No broad code-size or performance claim follows from
this small batch.

The [ownership/deletion gates](consolidation-order.md#active-paths-and-replacement-gates)
record the larger active paths. In particular, two GC policies can still run
on one Vulkan allocation path; native caches still have multiple ownership
layers; the legacy immediate emitter still serves Metal's recorded front work;
and backend TLAS mutation timing still differs.

Raycore's software TLAS also contains its own mesh-to-triangle preparation
(`build_and_append_blas!` and mesh replacement). This batch removes the two
Mantle backend copies; it does not claim to remove that cross-package
duplication or introduce a dependency from Raycore back to Mantle.
