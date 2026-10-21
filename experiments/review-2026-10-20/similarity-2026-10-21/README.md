# Semantic similarity experiment — 2026-10-21

Recommendation: use local semantic retrieval as the main discovery channel,
retain lexical matching as a complementary channel, then review candidate
contracts. Start with Qwen3-Embedding-0.6B. The code-specific Jina model did not
outperform it in this small repository check. This is a practical selection,
not a claim of the best model on every Julia retrieval task.

## Executed check

Source: Mantle `56f8d79`, whose implementation is unchanged from `45021c6`.
The [tool](../../../tools/similarity/README.md) parsed all 111 Julia files in
`src/` and `ext/`, producing 1,686 methods and 301 overlapping regions.
Complete method bodies and partial regions are indexed separately. This does
not cover all top-level code or prove that every abstraction opportunity was
found.

Models: pinned Qwen3-Embedding-0.6B and jina-code-embeddings-0.5b revisions
recorded in the JSON reports. Execution used Python 3.12.12, PyTorch 2.8.0+cu128,
Transformers 4.57.1, BF16, SDPA, and the NVIDIA RTX 4000 Ada. Corpus embedding
took 12.8 seconds for Qwen and 7.8 seconds for Jina, excluding loading,
tokenization and downloads. These are one-run tool timings, not benchmarks
of Mantle or repeated inference measurements.

Each query uses a complete known implementation and searches all 1,987
chunks. Self-matches and regions within the query method are excluded from
ranks. The lexical baseline uses TF-IDF over identifiers and punctuation.
The six pairs were selected from existing review findings before model runs;
they are a calibration set, not an independent discovery benchmark.

| Expected related implementation | Lexical rank | Qwen rank | Jina rank |
|---|---:|---:|---:|
| Optional/throwing Vulkan memory-type selector | 1 | 1 | 1 |
| Metal borrowed array constructors | 1 | 1 | 6 |
| Raster/mesh pipeline compilation with shared attachment setup | 1 | 2 | 1 |
| Metal/Vulkan TLAS batch transform update | 1 | 1 | 2 |
| Metal/Vulkan TLAS single transform update | 1 | 2 | 2 |
| Core GC rate policy versus Vulkan collection policy | 20 | 1 | 1 |

The GC result demonstrates useful retrieval beyond close textual matches.
It does not establish identical algorithms: the core uses elapsed cost and
configured minimum intervals; Vulkan also gates on memory pressure and its
own collection statistics. The shared collection controller now owns coordination; native pressure facts
remain separate (consolidation-memory.md).

Natural-language queries retrieved both memory selectors, both Metal view
constructors, both collection-policy implementations and both TLAS batch
update implementations among their leading results. The reports preserve
the exact queries, ranks and scores.

An explicit counterexample: both models returned depth mappings among the
leading answers to a Float32 sampled-color-texture query. Float32 depth and
R32 color mappings are not interchangeable. Retrieval cannot approve merges
or replace source/caller inspection. A score is not an equivalence probability.

## Candidate spot checks from the all-pairs results

These candidates were inspected on the original snapshot. Current dispositions
are below; historical file:line locations are not current source coordinates.

| Candidate locations | Review implication |
|---|---|
| `src/graphics/mesh.jl:76`, `src/graphics/pipeline.jl:89` | `fragmentinputtype` repeats the same type construction. Consolidated through the distinct `fragmentinputs` dispatch; see consolidation-first-batch.md. |
| `src/graph/build.jl:1012`, `src/runtime/dispatch.jl:315` | `indirectcount!` repeats usage registration and resource touching. Its wrappers return different values (`Commands(n)` versus `n`); preserve that distinction if sharing registration. |
| `src/vulkan/runtime/video.jl:500`, `src/vulkan/runtime/video_h265.jl:495` | `feed!` shares access-unit accumulation, maximum-size accounting, buffer reservation and decoder reset. Consolidated in video_common.jl; codec NAL/slice parsing remains distinct. |
| `src/metal/graphics.jl:1164`, `src/metal/mesh.jl:503` | `compile_draw` repeats argument normalization, format conversion and compilation plumbing. Stage-specific argument and compiled-result types remain distinct. |
| `src/metal/hwtlas.jl:358`, `src/vulkan/raytracing/hwtlas.jl:262` | Highly similar `AdaptedAccel` field walks differ on adapting `scene`. Determine its representation contract before sharing the projection. |
| `ext/MantleROCmExt.jl:1022`, `src/metal/ka.jl:193` | Batched-GEMM dispatch wrappers are related, but native eligibility and launch configuration differ. Similarity alone does not establish a duplicated algorithm. |

Full outputs: [Qwen report](qwen-report.json), [Jina report](jina-report.json).
Each contains the 100 highest retained cross-file candidate pairs. Overlapping
regions and nested functions can create multiple hits for one implementation;
the reports are discovery aids, not counts of independent duplication groups.

## Why this approach

[Semantic embeddings](https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html)
support meaning-based retrieval, including descriptions of responsibilities.
[MinHash](https://ekzhu.com/datasketch/minhash.html) instead estimates set
overlap: making a matcher probabilistic does not by itself make it semantic.
At this corpus size a normalized embedding matrix and direct dot products are
sufficient. Approximate nearest-neighbor infrastructure is unnecessary.

[Qwen](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) supplies both general
semantic and code retrieval under Apache-2.0;
[Jina](https://huggingface.co/jinaai/jina-code-embeddings-0.5b) has an explicit
code-to-code task and a CC-BY-NC-4.0 model license. Neither model card proves
Julia abstraction-discovery quality. Keep the tested model and lexical
baseline, broaden checks as additional independently reviewed groups emerge,
and inspect ownership, synchronization and errors before implementing merges.

## Refreshed discovery after consolidation

[Source a021c67, 1,985 chunks](refresh-a021c67.json): all-pairs ranking now
includes separate methods in one file, groups each method's overlapping windows,
and supplies substantial/diverse views. Previously, cross-file-only nearest-12
selection could hide candidates behind self-windows and thin wrappers. Six host
regressions cover those discovery failures. Scores still require contract review;
GEMM variants lead the same-file list and are not interchangeable implementations.

Refreshing at `32d387d` yields 1,964 chunks. Same-file discovery led to shared
GEMM generators; cross-file discovery also found GJK/EPA support-pair duplication.
The implemented deletions and native checks are in consolidation-gemm-generator.md
and consolidation-geometry.md. Rankings remain candidate evidence, not equivalence proofs.
