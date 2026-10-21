# Shared geometry and stride calculations — 2026-10-22

Baseline: `32d387d`. GJK and EPA share transformed support pairs; triangle
regions reuse the line-simplex implementation; tetrahedron reductions return
triangle results directly. EPA's three contact interpolation paths share
face_result. Uniform transform construction delegates to per-axis construction.
Transpose, swapped permutation and real-adjoint GEMM strides share one method;
complex adjoints still require conjugation and return nothing.

Both contact-list insertions use append_contact!. Their default Atomix ordering
failed compilation after F33, exposing a missed caller migration. Explicit
monotonic ordering suffices for unique slot reservation: no invocation polls
these counters for publication; readers require dispatch completion and normal
producer-to-consumer synchronization. No compiler ordering rule was weakened.

629 targeted assertions pass: GJK/EPA, native narrow-phase/contact records and
canaries (367), plus unchanged native transform writing, scalar GEMM/wrappers
and fp16 widening (262). Native checks run on RX 7900 XTX and RTX 4000 Ada with
validation/synchronization validation enabled and empty message drains. New
coverage checks complex-adjoint refusal and 257 contenders across five workgroups.

Comments now state actual simplex order, affine layouts, EPA normal orientation,
unused seed argument and bounded failure returns. JuliaSyntax confirms the prose
edits preserve definitions; the existing unattached EPA documentation is attached.
All 115 source files parse. Source decreases 51,636 → 51,251 physical and
23,011 → 22,943 classified implementation lines: 49 from code consolidation,
19 from attaching previously executable prose. Metal/ROCm execution is unqualified.
