# Open decisions of 2026-10-05 (asked, not yet answered)

Until Simon answers, the suite follows the recommendation and marks each case
that depends on one with `# pending decision N`.

1. `persistent = true` graph arrays contradict B3.16 (only device arrays
   survive recompiles). Recommended: drop `persistent`; state that outlives a
   run is a device array.
2. `when!` flag values other than 0 and 1. Recommended: nonzero means "runs
   once"; the GPU computes count × (flag ≠ 0).
3. Eager calls (all arguments device arrays) inside `when!` / `repeat!`
   blocks. Recommended: they throw at declaration.
4. Windows. Recommended: only the presenting graph writes the window
   (`render!` or a kernel); `render!(dev, win)` throws; the binding ends with
   `free!(g)`; `Array(win)` throws when minimized or from a host function of
   the presenting graph; `run!` returns `Submitted()` or `Skipped()`
   publicly.
5. Read-only inputs (B3.9). Recommended: name `readonly`; writes throw at
   declaration or at the store (needs per-argument access, G2);
   `MantleArray(dev, path; mmap = true)`; a file changed meanwhile is
   undefined (documented).
6. Kernel-written lengths. Recommended: `length(a)` throws for such an array
   (use `devicesize(a)`, `capacity(a)`); `resize!` and `copy!` on it throw; a
   kernel writing more than the capacity is clamped to it by the prepare node.
7. Consistency within a run. Recommended: "old or new, never a mix" holds per
   stage (host functions may change things between stages).
8. Inserting a plot of a new kind compiles a pipeline from the cache.
   Recommended: the "zero compiles" gate counts plan compiles; pipeline
   compiles are counted separately.
9. `resize!` of a texture keeps the overlapping texels (the glyph atlas needs
   it). Recommended: yes.
10. Value `GPURef`s and hostwritten arrays in pieces. Recommended: each piece
    has its own entry slots in its argument region and is marked per piece.
11. Piece order. Recommended: `reorder!(p, handles)` (name open), a list
    rewrite.
12. `push!` past the device's `maxInstanceCount`. Recommended: throws at the
    call (the requested total is known there).
