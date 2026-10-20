# Mantle API test catalogue: graphs, ordering, windows, eviction, holds and eager work

I left resize!, stores, TLAS/BLAS, array GPURefs and textures to the other agent. Cases from that area appear here only where my scope depends on them, for example eviction of a store's target. The catalogue has 236 cases, and 29 contracts are ambiguous or contradictory (listed at the end). I edited no files and ran nothing.

**Doc keys:** A = `dev/Mantle/docs/api.md`, P = `recording-plan.md`, I = `internals.jl`, R = `resizing-and-raytracing.jl`, E = `examples.jl`, L = `refactor-ledger.md`, D = `experiments/plan-review/decisions.md`. A reference reads doc:line.

**Device keys:** any = every joined type, lavapipe included. Vulkan, lavapipe, Metal and CUDA mean that device type. mq = more than one queue. sparse / non-sparse = whether the device has sparse residency. sep-heap = Readback memory is a separate heap (a discrete GPU). win = needs a surface.

**Kinds:** F = functional, C = concurrency, E = error-path, P = perf-gate, L = leak.

**Fault-injection points.** Proposal: a test-only `FaultDevice{D}` in `test/`. It forwards every verb in A section 4 to a real device and fails or pauses on a schedule held by the device object, so no globals and no probes are needed.
- **FI-1:** `compilekernel` / `compiledraw` / `recordpiece` throw.
- **FI-2:** `rawalloc` fails.
- **FI-3:** `acquirenext` returns OUT_OF_DATE or SUBOPTIMAL, or blocks.
- **FI-4:** `surfacesize` returns (0,0), i.e. the window is minimized.
- **FI-5:** `createswapchain` throws.
- **FI-6:** `submitnative!` throws (device loss).
- **FI-7:** pause points. `compilekernel` blocks on a `Channel`, which holds compile and register! open. `acquirenext` blocks, which holds a run between applywindows! and the acquire.
- **FI-8:** lock tracker. A debug build of `locked` / `lockplans` / `waitfor` that records each task's held locks, asserts the lock order and asserts no sync-state, channel, pool or window lock is held during `waitfor`.
- **FI-9:** `budget(dev)` override, to force pressure.
- **FI-10:** `caps` override (`sparseresidency`, `separatehostheap`, `inlinestore`, `display`).
- **Counters needed:** per-device compile counter (A:310); record, submission and allocation counters on Builder / `submit!` / `acquire!`.

## 1. Devices and caps (DEV)
| id | contract | scenario | expected | kind | device |
|---|---|---|---|---|---|
| DEV-01 | A:55, P:152-156 | `Device()` twice, then from 8 threads at once | same object (`===`) every time, one pool | F/C | any |
| DEV-02 | A:56, I:2232-2236 | call `devices()`; FaultDevice counts driver-object creation | every physical device listed once; no device or queue objects until first use | F | Vulkan+lavapipe |
| DEV-03 | A:55, P:147 | ICD order puts lavapipe first | `Device()` is a GPU, never lavapipe | F | Vulkan+lavapipe |
| DEV-04 | A:59, P:160-166 | fresh process: `using Lavapipe_jll` before vs after the first `Device()` | before: listed; after: not listed, no crash | F | Vulkan |
| DEV-05 | A:57 | one GPU through LavaDevice and CudaDevice; mix their arrays | two devices, two pools; mixing throws at `location` | E | Vulkan+CUDA |
| DEV-06 | A:58, I:206 | `KA.get_backend` on a device array, then on a graph array | `device(a)`; then throws | F/E | any |
| DEV-07 | P:169-171 | operation with only a device method, called on a graph array | MethodError; submission counter unchanged | E | any |
| DEV-08 | A:450, E:510-513 | `a2 .= a1` across devices; graph on d1 given a d2 array | ArgumentError "two devices"; graph unchanged | E | GPU+lavapipe |
| DEV-09 | A:262, R:1262-1266 | read `caps(dev)` on each type | immutable; the listed fields exist; Vulkan `inlinestore`=65536, `inlinealign`=4 | F | any, Metal |
| DEV-10 | A:253-254 | lavapipe without sparse/RT; declare graphs that would use them | no verb throws "not supported"; core picks non-sparse arena parts | F | lavapipe |
| DEV-11 | A:269-275, P:540-545 | same graph with `caps.display` true vs false [FI-10] | smaller submission budget when display is true, so more segments | F | any |

## 2. Graph lifecycle (GR)
| id | contract | scenario | expected | kind | device |
|---|---|---|---|---|---|
| GR-01 | A:101, P:433-434 | 3 kernels; `run!` ×20 | compiled and recorded once; later runs only submit | F/P | any |
| GR-02 | P:888-889 | `run!` an empty graph, and a graph whose head is empty | no throw; empty head not recorded (return value: ambiguity 18) | F | any |
| GR-03 | A:101, I:1010 | `free!(g); run!(g)` | ArgumentError | E | any |
| GR-04 | A:102, I:2088-2090 | `free!(g)` twice, and from 2 threads at once | released exactly once (reserved drops once; no double retire under the debug pool) | C/L | any |
| GR-05 | A:102, I:243 | after `free!(g)`: `fill!` on its array, `MantleArray(g,…)`, `dispatch!(g,…)` | each throws "declaring into a freed graph"; no hold added to argument device arrays | E | any |
| GR-06 | P:1911-1913 | drop every reference to g and its arrays while a long run is in flight; `GC.gc()` | run completes correctly; memory returns after the token (`reserved(dev)` back to baseline) | L | any |
| GR-07 | A:102, I:2088 | T1 `run!(g)` whose host function sleeps; T2 `free!(g)` | T2 blocks until T1's last stage is submitted; GPU results correct; a later `run!` throws | C | any |
| GR-08 | A:474, I:2084-2087 | host function of g calls `free!(g)` | throws before `freed` is set; error propagates from `run!`; g still runs; GC finalizer later releases the plan | E/L | any |
| GR-09 | A:103 | `submissions(g)` on a graph with a host cut and a budget cut | both cuts listed | F | any |
| GR-10 | A:105 | `naivebytes`/`peakbytes` on a chain of transients | peak < naive; peak = placement's maxload | F | any |
| GR-11 | I:58-61 | keep a graph array, drop g, GC, then `fill!` the array | clean error, no segfault (error type unspecified) | E | any |
| GR-12 | A:100, E:105-117 | `mul!`+bias+gelu with default rewrites vs `FuseEpilogues()` | default: 2 nodes; fused: 1; results identical | F | Vulkan |
| GR-13 | P:546-548 | `Graph(dev; profile=true)`, 20 runs | recorded once; profile only calibrates the next compile | P | any |
| GR-14 | A:217, P:729-731 | compile under the graph lock hits pressure that needs finalizers [FI-9] | finalizers run (the lock is a semaphore); allocation succeeds | F | any |
| GR-15 | A:101 | `run!(g)` from 8 threads ×1000 | all complete; serialized; no overlap | C | any |

## 3. Graph mutation, B3.16 (MUT)
| id | contract | scenario | expected | kind | device |
|---|---|---|---|---|---|
| MUT-01 | A:107-115, D:244-254 | `run!`; declare a kernel; `run!` | same graph object; one recompile; device arrays keep contents; window and its swapchain kept | F | any |
| MUT-02 | A:108-110 | 5 declarations between two runs | exactly one recompile | P | any |
| MUT-03 | A:119-120 | `h=insert!(g) do K1;K2 end`; run; `delete!(g,h)`; run | K1/K2 run, then stop; one recompile per change | F | any |
| MUT-04 | A:120, I:279-296 | A named only by block h; `free!(A)`; `delete!(g,h)` while the old plan's last run is in flight | A's storage released only after that run's tokens (realloc canary does not alias) | L/C | any |
| MUT-05 | I:2068-2074 | A named by block h and by a node outside it; `delete!(g,h)` | hold kept (counted per node); A still used | F | any |
| MUT-06 | A:119, I:268-274 | `insert!(g) do K1; error() end` | throws; K1 gone; A's hold count restored | E/L | any |
| MUT-07 | A:119, I:263,265 | `insert!(g)` nested, inside `repeat!`, inside `when!` | each throws; graph unchanged | E | any |
| MUT-08 | A:120 | `delete!(g,h)` twice | second is a no-op; no hold underflow (ambiguity 11) | E | any |
| MUT-09 | A:120 | `delete!(g2,h)` with h from g1 | throws; no hold change (ambiguity 11) | E | any |
| MUT-10 | A:120 | nodes outside blocks, then 50 insert/delete cycles | still run every time | F | any |
| MUT-11 | A:114 | T1 runs g, host function sleeps; T2 declares into g | T2 waits for T1's last stage; next run recompiles | C | any |
| MUT-12 | A:114-115, I:238 | host function of g declares into g | throws (owner); g unchanged | E | any |
| MUT-13 | A:112-113 | device-array film accumulates over 100 runs with insert/delete in between | accumulated sum exact | F | any |
| MUT-14 | A:71 vs A:112 | `persistent=true` counter, recompile in between | survival unspecified; record the outcome (ambiguity 4) | F | any |
| MUT-15 | I:1028-1058 | `insert!` a kernel that fails to compile [FI-1]; `run!` twice; `delete!(g,h)`; `run!` | both runs throw; recompile mark kept; after delete runs with old structure; no leaked namers or tenancy | E/L | any |
| MUT-16 | I:1038-1041, R:672-692 | T2 makes a first `resize!` of a directly launched array between T1's compile and register! [FI-7] | register! → Again → recompile; discarded plan leaves no namers or tenancy; output correct | C | any |
| MUT-17 | I:1059-1065 | recompile while the old plan is in flight | old argument block and recordings released after its last run; tenancy = new peak | L | any |
| MUT-18 | P:72-73, I:1216,1348 | RMW counter in a device array; `insert!` between runs, no host waits, ×1000 | every run counted once: the new plan's first run is ordered after the old plan's last run (ambiguity 5) | C | any, mq |
| MUT-19 | D:244-254 | 10 000 insert/delete cycles | `reserved`, holds, namers, `g.named`, `r.plans` back to baseline | L | any |
| MUT-20 | I:244-248 vs P:1440 | two tasks run `insert!` blocks into one graph at once | serialized; each handle holds only its own nodes (ambiguity 19) | C | any |

## 4. Declaration rules (DECL)
| id | contract | scenario | expected | kind | device |
|---|---|---|---|---|---|
| DECL-01 | A:127, I:196-202 | arguments from two graphs | ArgumentError; both graphs unchanged | E | any |
| DECL-02 | A:127, P:342-345 | graph array + device array of the same device | declared into the graph; device array gets one hold | F | any |
| DECL-03 | I:198 | graph on dev1 + `GPURef(dev2,v)` | throws "two devices" | E | 2 devices |
| DECL-04 | A:92, A:477 | `getindex`, iterate, `Array`, `collect`, `show` on a graph array | each throws; hostwritten `x[:]=d` allowed | E | any |
| DECL-05 | I:213, I:2060 | declare using a freed device array | throws; no node, no hold | E | any |
| DECL-06 | A:76, A:476 | `lengthof(a)` without a capacity | refused at declaration | E | any |
| DECL-07 | P:1304-1307 | one device array as Broadcasted leaf, kwarg, view parent and `when!` flag of one node | one hold per graph, counted per node | F/L | any |
| DECL-08 | A:134-136, I:413-426 | `mul!` in `repeat!` / `when!` where a foreign candidate is best | kernel candidate inside; foreign allowed outside | F | CUDA (or FaultDevice candidates) |
| DECL-09 | A:132, P:974-976 | `copyto!(out::device, x::graph)`; run; `Array(out)` | copy node recorded once; host sees this run's result | F | any |
| DECL-10 | A:77, I:2141-2142 | `MantleArray(l,T,n) do t … end` on a device and on a graph | device: region retired with the call's token; graph: placed transient | F/L | any |
| DECL-11 | A:127 + A:130 | inside `when!(g,flag)` or a `repeat!` body, a call whose arguments are all device arrays | runs eagerly once, unconditionally (ambiguity 7) | F | any |

## 5. repeat! (REP)
| id | contract | scenario | expected | kind | device |
|---|---|---|---|---|---|
| REP-01 | A:129, P:365-367 | `repeat!(g,50) do i` kernel writes `out[i]=i` | out == 1:50; body compiled and recorded once | F | any |
| REP-02 | A:129, P:356-361 | `while_nonzero`: run 1 stops at 7; reset flag; run 2 stops at 3 | 7, then 3 iterations; loop start restores the counts | F | any |
| REP-03 | A:129, I:359-361 | body holds `render!`, `build!`, HostCall, image copy, `insert!(g)`, nested `repeat!` | throws at declaration; nodes and holds unchanged | E | any |
| REP-04 | I:348-362 vs I:873-879 | `render!` + `draw!(device array)` in a body | throws, and the array's hold count is restored (suspected leak, ambiguity 12) | L | any |
| REP-05 | A:131 | `trace!` in a body ended by the flag | counts zeroed after the end; matches a host reference | F | Vulkan RT |
| REP-06 | A:130 | `when!` in a body, flag written per iteration | prepare node per iteration; launches only where flag ≠ 0 | F | any |
| REP-07 | I:546 | last body node writes X, first body node reads X | each iteration sees the previous write | F | any |
| REP-08 | P:2311-2315 | loop cost > budget, one iteration fits | cut between iterations; exact | F/P | any |
| REP-09 | I:604-605 | per-iteration transient; hostwritten array read in the body | results exact (bytes shared across iterations) | F | any |
| REP-10 | A:129 | `repeat!(g,0)`, `repeat!(g,-1)` | unspecified (ambiguity 18) | E | any |
| REP-11 | P:348-355 | body calls an operation with no graph method | MethodError; graph unchanged | E | any |
| REP-12 | A:132 | `copyto!(a,b)` between graph arrays in a body | allowed (copy kernel); exact | F | any |

## 6. when! (WHEN)
| id | contract | scenario | expected | kind | device |
|---|---|---|---|---|---|
| WHEN-01 | A:130, D:133-136 | `when!(g, GPURef(dev,Int32(1)))`; run; `r[]=0`; run | runs, then skipped; compile and record counters unchanged | F/P | any |
| WHEN-02 | P:2288-2295 | flag written by a kernel of the graph, in Readback memory | prepare node after the writer; `Array(flag)` needs no GPU copy | F | any |
| WHEN-03 | A:130 | flag is a one-element device array changed by a store | effective at the next run; inline store, no host wait | F | any |
| WHEN-04 | A:130 vs P:2287 | flag = 2, -1, typemax, 0.5f0, NaN | contract: K runs once per run; mechanism multiplies the count (ambiguity 6) | F | any |
| WHEN-05 | I:368 | nested `when!` | throws; graph unchanged | E | any |
| WHEN-06 | A:478 | `when!` containing `render!` / `build!` / HostCall / image copy / `repeat!` | throws; nodes and holds unchanged | E/L | any |
| WHEN-07 | P:103-108, I:628-637 | K1 writes A; `when!`(flag 0 or 1) K2 reads and writes A; K3 reads A | K3 always sees the latest write; the barrier is never dropped | F | any |
| WHEN-08 | I:503 | WHEN-07 as a CUDA/HIP graph | edge K1→K3 kept in the transitive reduction | F | CUDA/HIP |
| WHEN-09 | P:2291-2292 | flag array moved by `resize!` between runs | no recompile, no re-record | F | any |
| WHEN-10 | A:130 | `when!(dev, flag)` (eager) | error, or the call runs (unspecified) | E | any |
| WHEN-11 | A:131 | `trace!` in `when!` | count × flag | F | Vulkan RT |
| WHEN-12 | I:369 | freed flag | throws at declaration | E | any |
| WHEN-13 | P:1518, P:1556-1558 | 10 000 tiny conditional launches vs plain ones | overhead within the measured indirect cost | P | Vulkan |

## 7. Host nodes and stages (HOST)
| id | contract | scenario | expected | kind | device |
|---|---|---|---|---|---|
| HOST-01 | A:128, P:914-919 | K1 → `HostCall(f)` reads K1's output, writes idx → K2 reads idx | exact; f runs on `run!`'s thread; 2 stages in `submissions(g)` | F | any |
| HOST-02 | P:500, I:584-588 | two adjacent host nodes plus an independent kernel | merged into one cut; the kernel moved across | F | any |
| HOST-03 | I:1328-1331 | f reads data stage 1 wrote | f sees finished data | F | any |
| HOST-04 | A:128, P:920-922 | f throws | propagates; later stages not submitted; no part held; next run exact; a part-sharing graph can run | E | any |
| HOST-05 | A:473 | f calls `run!(g)` | throws, no deadlock | E | any |
| HOST-06 | A:473, I:994 | f runs g2, which shares an arena part | throws | E | any |
| HOST-07 | P:694-695 | f runs g3 with disjoint parts | allowed; both exact | F | any |
| HOST-08 | A:437, I:1268-1289 | f holds part 5 and runs g4 whose part 3 is held by T2's run | `holdparts!` throws; f's parts not leaked (wording: ambiguity 10) | E/C | any |
| HOST-09 | P:662-664 | f stores into a device array stage 2 reads | stage 2 sees the store | F | any |
| HOST-10 | P:671-675, R:343-351 | f calls `resize!` on an array the plan launches over directly | call returns at once; stage 2 uses old size; next run recompiles | F | any |
| HOST-11 | D:500-505, R:449-451 | f makes an eager call needing a change that stales its own plan | throws (inrun) | E | any |
| HOST-12 | P:1921-1928 | T2 runs part-sharing g2 while f sleeps; T3 resizes an array g uses | g2 waits for g's last stage; all exact | C | any |
| HOST-13 | P:1044-1045, I:660-667 | host node between the window write and the present | compile error at the first `run!` | E | Vulkan win |
| HOST-14 | A:128 | HostCall inside `repeat!` / `when!` | throws | E | any |
| HOST-15 | I:648 | graph starting with a host node; graph ending with one | both run; heads correct | F | any |
| HOST-16 | I:589-591 | f reads a device array (normal memory, and Readback) | copy before the cut; none for Readback | F | any |
| HOST-17 | P:1925-1928 | f runs part-sharing g2 while T2 waits on g2's lock | throws, no deadlock | C/E | any |
| HOST-18 | P:925-926, I:986-990 | f spawns a task that runs g and waits for it | known undetected deadlock; document with a timeout | C | any |
| HOST-19 | P:1353-1355 | host-visible part on a sparse device | f gets contiguous views | F | sparse |
| HOST-20 | A:128 | f writes an array not listed in `writes` | undefined, not detected (ambiguity 25) | F | any |
| HOST-21 | P:926-927 | submissions and GPU→host round trips per run | = number of stages | P | any |

## 8. Arena parts (ARENA)
| id | contract | scenario | expected | kind | device |
|---|---|---|---|---|---|
| ARENA-01 | P:201-206 | two non-overlapping graphs with peaks P1, P2 | arena = max, not sum | P | any |
| ARENA-02 | P:1919-1920 | compile a big and a small graph; `free!(big)` | reserved drops to the small one's need | L | any |
| ARENA-03 | P:1359-1373, I:1930-1942 | pressure [FI-9] unmaps an idle graph's pages | plan marked; its next run binds before submitting; exact | F | sparse |
| ARENA-04 | I:1933 | pressure while a graph's host function runs | running plan's pages untouched | C | sparse |
| ARENA-05 | I:1990-2004 | another tenant widens the gap between page acquire and bind | Again; retry; no stray pages | C | sparse |
| ARENA-06 | P:1356-1358 | non-sparse device [FI-10]; later graph needs more | new part; existing part addresses unchanged | F | non-sparse |
| ARENA-07 | I:1059-1065 | recompile changes the plan's parts | tenancy dropped in unused parts | L | sparse |
| ARENA-08 | A:382 | two part-sharing graphs on 2 threads ×10 000 | no deadlock; transients never clobbered | C | any |

## 9. Run semantics (RUN)
| id | contract | scenario | expected | kind | device |
|---|---|---|---|---|---|
| RUN-01 | A:453, P:1895-1899 | RMW persistent array, 2 threads, 2 queues, ×10 000, no host waits; negative control without the previous-run wait | count exact; control fails | C | mq |
| RUN-02 | A:455, P:1891-1894 | steady state | 0 allocations, 0 compiles, 0 records; 1 submit per segment | P | any |
| RUN-03 | P:863-868 | `run!` with nothing marked | returns before GPU completion; no host wait | P | any |
| RUN-04 | R:705-712 | T2 makes a first `resize!` of a directly launched array just before `run!` | stale → Again → recompile inside `run!`; exact | F/C | any |
| RUN-05 | I:1015-1021 | window changed + stale + unmapped at once | `run!` converges; one submitted run | F | win, sparse |
| RUN-06 | P:712-715 | `rawalloc` fails in `acquireunmapped!` / `prepare!` [FI-2] | OOM before any token claimed; timeline values have no gap; usable after freeing memory | E | any |
| RUN-07 | P:712-715, A:281 | `submitnative!` throws [FI-6] | propagates; all locks released (another task can lock them); parts released | E | any |
| RUN-08 | P:928-930 | stale plan / out-of-date swapchain | retried inside `run!`, never thrown | F | any, win |
| RUN-09 | A:101 | `run!(g)` from a task that holds a part of g's plan | throws | E | any |
| RUN-10 | P:1884-1885 | SAM 2 encoder and decoder, 20 runs | each plan recorded exactly once | P | Vulkan |
| RUN-11 | D:498-499 | stage of 2 segments on 2 queues; segment 2 touches a record whose change segment 1's prefix applied | segment 2 waits for segment 1's token | C | mq |
| RUN-12 | P:645-651 | runs on 2 queues with validation on | timeline values strictly rising per queue; validation clean | F | mq |
| RUN-13 | P:676-685 | T2's eager call applies a change that stales plan P while P is between stages | T2 waits until P's last stage, then applies; P's later stages use the old state; next run recompiles | C | any |
| RUN-14 | P:679-680 | RUN-13 with a single-stage P | T2 never waits | C | any |
| RUN-15 | P:684-685 | RUN-13 while P runs in a tight loop | T2 eventually proceeds (no fairness promised); document the bound | C | any |
| RUN-16 | R:460 vs A:487 | eager call inside an `insert!(g)` block that would wait for another graph's run | internals throws (inrun); api.md says only inside a run (ambiguity 9) | E | any |
| RUN-17 | P:712-715 | allocation tracking inside `submitnative!` / `bindpages!` | 0 Julia allocations | P | any |

## 10. Entry, value GPURefs, hostwritten (ENTRY)
| id | contract | scenario | expected | kind | device |
|---|---|---|---|---|---|
| ENTRY-01 | A:142-143, I:1339-1347 | `r=GPURef(dev,1f0)`; `r[]=2f0`; run | run uses 2f0; host waited for the previous run | F | any |
| ENTRY-02 | A:143 | r unchanged; `run!` repeatedly | no host wait; entry not written | P | any |
| ENTRY-03 | P:896-899 | T1 writes a Mat4f GPURef in a loop; T2 runs ×10 000; kernel checks all words agree | never torn | C | any |
| ENTRY-04 | I:1333-1337 | `r[]=v` during another thread's `writeentry!` | that run uses old or new; next run uses newest | C | any |
| ENTRY-05 | D:507-508 | r used by g1 and g2; g3 does not use it | g1, g2 marked; g3 neither waits nor writes | F | any |
| ENTRY-06 | A:406 | first run of a new plan | entry written with current values | F | any |
| ENTRY-07 | P:960-961 | GPURef in an eager call; `r[]` changed afterwards | call used the value at call time | F | any |
| ENTRY-08 | I:1364 | 1000 recompiles/frees of plans using r | `r.plans` does not grow | L | any |
| ENTRY-09 | A:144, P:1932-1933 | hostwritten `x[:]=d` before the first run | first run uses d | F | any |
| ENTRY-10 | A:144, I:1384 | `x[:]=d` while a run is in flight | the call waits, `run!` does not | F/C | any |
| ENTRY-11 | I:1381-1382 | `x[:]=d` with a recompile pending | kept on the host; in the new entry | F | any |
| ENTRY-12 | I:1045 | `x[:]=d1`; run; `insert!`; run with no new write | new plan uses d1 | F | any |
| ENTRY-13 | P:881-886 | no `x[:]` between runs | each run re-applies the last data | F | any |
| ENTRY-14 | I:1378 | `x[:]` on a non-hostwritten array; `x[1:3]=`; wrong length | errors (exact rules unspecified, ambiguity 18) | E | any |
| ENTRY-15 | I:1380 | `x[:]=d` from g's host function | throws | E | any |
| ENTRY-16 | A:379, A:428 | `r[]=v` while a finalizer-heavy GC and runs are in progress | no deadlock (leaf lock) | C | any |
| ENTRY-17 | I:1361-1372 | `r[]=` a value of another type | unspecified (ambiguity 18) | E | any |
| ENTRY-18 | I:826-836 vs R:686 | value GPURef or hostwritten array used in a piece's draw | unspecified (ambiguity 21) | F | any |
| ENTRY-19 | A:130 | GPURef as `when!` flag, toggled every run ×1000 | correct; host waits only on toggling runs | F/P | any |

## 11. Ordering between graphs (ORD)
| id | contract | scenario | expected | kind | device |
|---|---|---|---|---|---|
| ORD-01 | P:587-606, D:225-243 | P writes A; C reads A; run them back to back ×10 000, no host waits | C always sees P's latest write | C | any, mq |
| ORD-02 | P:595-597 | C reads A, then P writes A | P waits for C's complete last run | C | any |
| ORD-03 | I:1545-1548 | P1 and P2 alternately write A | write-after-write ordered | C | any |
| ORD-04 | P:599-601 | R1 and R2 only read A, on 2 queues | no mutual wait; GPU timestamps overlap | P | mq |
| ORD-05 | R:657-662 | C compiled while P is in flight | C's first run waits for P | C | any |
| ORD-06 | D:240-241 | 10 graphs naming A register and free while P exists | P's compile counter stays 0 | P | any |
| ORD-07 | R:667-671 | P freed/recompiled in flight; C reads A next | C waits for the retired plan's tokens | C | any |
| ORD-08 | A:458, P:1914-1918 | `test_crossplan_order`, 1 and 2 channels, read-then-write; control `Unordered()` | exact; control fails | C | any, mq |
| ORD-09 | I:1556-1561 | stage writes nothing shared, inbox empty | host time flat | P | any |
| ORD-10 | A:466, P:1934-1937 | plans naming 100 / 2 500 / 25 000 / 50 000 resources, nothing changed | flat host time per run; within 3× the model | P | any |
| ORD-11 | I:1557-1559 | A from stage 1's inbox used by stage 3 | stage 3 covered by the kept waits | C | any |
| ORD-12 | I:1186-1187 | a plan registers during another stage's locking | Retry, not recompile; waits correct | C | any |
| ORD-13 | P:647-651 | 2-queue plans sharing resources ×10 000 | no GPU deadlock; validation clean | C | mq |
| ORD-14 | D:244-254 + D:225-243 | `delete!` drops g's only read of A; P writes A | P waits for g's old plan until it retires, then not | C | any |
| ORD-15 | I:1589-1595 | inbox item the stage does not use | waited with the plan's use; kept for later stages | F | any |
| ORD-16 | P:572-573, P:2316-2317 | `Graph(dev; priority)` frame graph beside a background DNN | frame not delayed beyond its own dependencies (open decision 12) | P | mq |
| ORD-17 | I:1577-1579 | `free!(g)` with g naming A that P also names | A goes into P's inbox; P's next run waits for g's last run | C | any |

## 12. Inbox, ordering with eager and host work (INBOX)
| id | contract | scenario | expected | kind | device |
|---|---|---|---|---|---|
| INBOX-01 | P:622-631 | `run!(C)` reads A, then eager `fill!(A)` | eager waits for C's last run | C | any |
| INBOX-02 | P:610-614 | eager write to A, then `run!(C)` | C waits for the eager token | C | any |
| INBOX-03 | P:617-618 | pending store into A, then `run!(C)` | applied in front of C's first segment; no extra submission | F | any |
| INBOX-04 | P:622-631 | eager read of A while reader C is in flight | no wait | P | any |
| INBOX-05 | I:1639-1646 | eager segment reads then writes A | recorded as a writer; later readers wait | F | any |
| INBOX-06 | P:607-610 | sparse bind of A's reserve, then C reads A | C waits for the bind token | F | sparse |
| INBOX-07 | P:619-621 | 10 000 eager writes to A between two runs of C | C's host time bounded (inbox dedupe unspecified) | P | any |
| INBOX-08 | A:466 | frame naming 5 000 resources, one changed | host time follows the inbox only | P | any |

## 13. Locks and threads (THR)
| id | contract | scenario | expected | kind | device |
|---|---|---|---|---|---|
| THR-01 | A:459, P:1910 | compile on A, record on B, run on C, free on D | works | C | any |
| THR-02 | P:1876-1878 | `test_interleaved_recording`, 1 and 4 threads, yield between emits, ×10 000 | exact | C | any |
| THR-03 | P:1879-1880 | two builders alternating | no device, channel or pool field references either | F | any |
| THR-04 | A:454, P:1881-1883 | `record!` with checking wrappers | no allocation, compile, submit or mutable-field read | F | any |
| THR-05 | P:738-740 | T1 waits on a long GPU job; T2 calls `GC.gc()` | T2 finishes promptly (GC-safe wait) | C | any |
| THR-06 | A:439, D:476-477 | lock tracker over the whole suite [FI-8] | no sync-state, channel, pool or window lock held in `waitfor` | C | any |
| THR-07 | A:421-433 | fuzz: 8 threads, 6 graphs with crossing reads and writes, eager calls, stores, frees, window resizes | no deadlock (watchdog); lock order holds [FI-8] | C | any |
| THR-08 | P:980-981 | 2 threads make eager calls on one channel | each thread's program order kept | C | any |
| THR-09 | I:991-999 | f throws inside `lockgraph` | owner cleared; next `run!` on the same task works | E | any |
| THR-10 | I:244, I:237 | nested declaration (`fill!` inside a `repeat!` body) | no self-deadlock | F | any |
| THR-11 | P:1301-1302, I:2100-2111 | finalizer fires while another task holds a sync-state lock | finalizer only pushes to the inbox; no deadlock | C | any |

## 14. Render pass pieces, B3.17 (PIECE)
| id | contract | scenario | expected | kind | device |
|---|---|---|---|---|---|
| PIECE-01 | A:176, D:255-268 | run; `insert!(p) do draw! end`; run | piece drawn; plan compile counter 0; recorded = 1 piece + list part | F/P | any |
| PIECE-02 | A:176 | `delete!(p,h)`; run | gone; no recompile; no other piece re-recorded | F | any |
| PIECE-03 | I:881-883 | `insert!` then read the image before `run!` | unchanged until the run | F | any |
| PIECE-04 | A:176, I:918-919 | piece pipeline fails to compile [FI-1] | `run!` throws; piece removed; holds restored; next run fine | E/L | any |
| PIECE-05 | A:176 | `insert!(p)` block with `dispatch!` / HostCall / `render!` / `build!` | throws; nothing added | E | any |
| PIECE-06 | I:887 | nested `insert!(p)` | throws | E | any |
| PIECE-07 | A:178 | visibility store into a piece's `when!` flag | count × flag; nothing recorded | P | any |
| PIECE-08 | A:465, P:1255-1257 | 100 / 1 000 / 5 000 plots, insert+delete one per frame | flat insert time; visibility change records 0 | P | Vulkan, Metal |
| PIECE-09 | L:603 | 4M-triangle mesh in a 5 000-piece pass vs alone | equal draw time within noise | P | Vulkan, Metal |
| PIECE-10 | I:1031, I:955 | `insert!(p)` then `insert!(g)` before the run | one recompile; piece compiled once; registered once | F/L | any |
| PIECE-11 | I:1031 | `insert!(p)` then delete the g-block containing p | piece holds dropped | L | any |
| PIECE-12 | I:902-906 | `delete!(p,h)` twice | no hold underflow (ambiguity 11) | E | any |
| PIECE-13 | — | `insert!(p)` after p's block was deleted | should throw (unspecified) | E | any |
| PIECE-14 | I:826-830 | 10 000 insert/delete cycles | cell-copy table bounded; regions back; reserved at baseline | L | any |
| PIECE-15 | I:835-836 | inserted piece reads X that sim graph S writes; `run!(S)`; `run!(frame)` | frame waits for S from its first run after the insert | C | any |
| PIECE-16 | I:842-844 | window pass with 100 pieces; resize | each piece recorded once for the new generation; no recompile | F | Vulkan win |
| PIECE-17 | A:327 | kernel in the graph writes X before the pass; piece reads X | correct (fixed pass barrier) | F | any |
| PIECE-18 | I:913-916 | delete a piece whose last run is in flight | region and recording retired after it; validation clean | C | any |
| PIECE-19 | I:898 | `insert!(p)` before the first compile | in the first plan | F | any |
| PIECE-20 | A:175 | `insert!` on an eager `render!(dev,…)` pass | unspecified | E | any |
| PIECE-21 | A:491 | zero-count draw in a piece | draws nothing; piece kept | F | any |
| PIECE-22 | A:328 | interleaved pieces with different pipelines and viewports | correct images (no inherited state) | F | Vulkan, Metal |

## 15. Windows and present (WIN)
| id | contract | scenario | expected | kind | device |
|---|---|---|---|---|---|
| WIN-01 | A:190, D:185-192 | g1 and g2 both `render!` into win | g2 throws at declaration; g2 unchanged | E | win |
| WIN-02 | P:1022 | one graph `render!`s into 2 windows | throws (when unspecified, ambiguity 13) | E | win |
| WIN-03 | E:490 vs D:185 | a second graph's kernel writes the window | unspecified (ambiguity 13) | E | win |
| WIN-04 | A:191, P:1886-1890 | T-event `resize!(win)` while T-run has a run in flight; then run | correct image at the new size; no device-wide wait; no recompile; old swapchain released after its last frame | C | Vulkan win |
| WIN-05 | A:191 | `resize!(win)` | 0 driver calls on the event thread (FaultDevice count) | F | win |
| WIN-06 | A:191, D:442-444 | resize beyond capacity | one recompile at the next run; never throws | F | win |
| WIN-07 | A:483 | minimize [FI-4]; store pending; `run!` | returns; 0 submissions; store still pending; applied after restore | F | win |
| WIN-08 | A:483 | OUT_OF_DATE [FI-3] | resize to `surfacesize`; retried within the same `run!` | E | win |
| WIN-09 | A:483 | SUBOPTIMAL [FI-3] | presented; window marked; next run makes a new generation | F | win |
| WIN-10 | I:1475-1486 | resize between applywindows! and acquire [FI-7] | WindowChanged → Again → applied | C | win |
| WIN-11 | A:192, A:485 | host function throws after the acquire, before the present segment | image kept; next run presents it with no new acquire | E | win |
| WIN-12 | I:1461 | image kept, then a resize | kept image dropped with its generation | E | win |
| WIN-13 | P:1023-1029 | 1 000 runs with forced Again in between | acquires = presents; validation clean | L | win |
| WIN-14 | A:184, P:1272-1277 | `Array(win)` from another thread during frames | image of the next presented frame | F/C | win |
| WIN-15 | A:184 | `Array(win)` while minimized, or after `free!(g)` | hang or error unspecified (ambiguity 14) | E | win |
| WIN-16 | A:184 | `Array(win)` from the presenting graph's own host function | deadlock risk (ambiguity 14) | E | win |
| WIN-17 | A:75 | `MantleArray(g,T,win)` across resizes | size follows the window; no recompile within capacity | F | win |
| WIN-18 | P:1050-1053 | small window on a sparse device | window part maps pages for the size, not the capacity | P | sparse win |
| WIN-19 | P:1053-1054 | non-sparse device | capacity mapped; works | F | non-sparse win |
| WIN-20 | P:993-1000 | steady state | per-image segments recorded once per generation; no extra copy | P | Vulkan win |
| WIN-21 | I:1453-1455 | `createswapchain` throws [FI-5] | mark restored; next run retries | E | win |
| WIN-22 | P:1019 | g1 renders Image I; presenting g2 reads it | g2 waits for g1 through namers | C | win |
| WIN-23 | P:1060 | window opened on the main thread, runs on a worker | works | C | win |
| WIN-24 | I:1407-1413 | presenting graph freed or GC'd; another graph declares into win | unspecified (ambiguity 13) | E | win |
| WIN-25 | P:1046-1049, P:2309-2310 | retired swapchain destroyed | no validation error (open decision 10) | L | Vulkan win |
| WIN-26 | I:1409-1414 | 100 resize events between two runs | one new generation | P | win |
| WIN-27 | A:190 | default capacity vs explicit `capacity` | placed at capacity | F | win |
| WIN-28 | P:1001-1002 | Metal: new drawable per frame | correct frames | F | Metal win |
| WIN-29 | int:688-698 | build node in the present span, dependent and independent | dependent: compile error; independent: moved before | E/F | win |

## 16. Eviction, pressure, read-only modes (EVICT)
| id | contract | scenario | expected | kind | device |
|---|---|---|---|---|---|
| EVICT-01 | A:464, D:193-198, P:1938-1941 | dnn1 and dnn2 at 90% each, run alternately ×50 with no host wait | exact; 0 recompiles; evictions and restores happen | F | sep-heap (FI-9 elsewhere) |
| EVICT-02 | I:1799-1803 | eviction while the array's last write is in flight | copy waits for the write; restored data = latest | C | sep-heap |
| EVICT-03 | A:490 | array named by a running plan (host function sleeping) | not evicted | C | sep-heap |
| EVICT-04 | A:490 | array with a pending store | not evicted | F | sep-heap |
| EVICT-05 | A:490 | array lent to `withview` | not evicted | F | CUDA |
| EVICT-06 | A:490 | array a composite or array GPURef holds | never evicted | F | sep-heap |
| EVICT-07 | A:490 | index array bound by handle | never evicted | F | sep-heap |
| EVICT-08 | A:490, I:1821 | single heap (M5, or FI-10 `separatehostheap=false`) | no eviction; OutOfDeviceMemory after the other steps | F | Metal / FI-10 |
| EVICT-09 | I:1883-1884 | `Array(a)` of an evicted array | read from the host copy; 0 submissions | F | sep-heap |
| EVICT-10 | A:490 | `free!` of an evicted array (last hold) | host copy released | L | sep-heap |
| EVICT-11 | P:1409-1413 | store into an evicted array | restore then store, one submission | F | sep-heap |
| EVICT-12 | I:1881-1882 | `resize!` of an evicted array | restored into new-size storage | F | sep-heap |
| EVICT-13 | I:2117-2118 | eager call on an evicted array | restored in the same submission | F | sep-heap |
| EVICT-14 | P:1401-1407 | evicted after the run looked | run starts over; restores; exact | C | sep-heap |
| EVICT-15 | P:1404-1406 | a new plan's own compile evicts what it names | restored at the next attempt | F | sep-heap |
| EVICT-16 | P:1398-1399 | restore evicts the other graph's weights, never the running plan's | ping-pong exact | C | sep-heap |
| EVICT-17 | A:486, I:1757-1783 | instrumented pressure [FI-9] | order: unmap idle → trim → GC → wait retiring oldest-first → evict LRU → throw | F | any |
| EVICT-18 | P:1426-1428 | pressure submissions | only unmaps and eviction copies; no queue flush | P | any |
| EVICT-19 | I:1892-1900 | 2 threads allocate near the budget | used never exceeds budget | C | any |
| EVICT-20 | A:486 | single allocation > budget | OutOfDeviceMemory; existing arrays intact; nothing prepared leaks | E | any |
| EVICT-21 | P:1406-1407 | run host time vs eviction-list size | flat (an unmarked run walks nothing) | P | any |
| EVICT-22 | I:1824 | LRU order | least recently used evicted first | F | sep-heap |
| EVICT-23 | D:193-198 | `MantleArray(path; mmap=true)` evicted, then restored | no copy-out; RSS not kept; restore reads the file; exact | F/P | sep-heap |
| EVICT-24 | D:196-197 | read-only array from host data evicted | VRAM freed; no copy-out (host copy semantics: ambiguity 16) | F | sep-heap |
| EVICT-25 | D:197-198 | store, `resize!`, eager or graph kernel write into a read-only array | must fail (where and how open, ambiguity 16) | E | any |
| EVICT-26 | — | mmap'd file changed on disk, then evict and restore | unspecified (ambiguity 16) | E | any |
| EVICT-27 | P:1388-1390 | eviction while dnn1's run is on the GPU | `run!(dnn2)` works | C | sep-heap |
| EVICT-28 | I:1855-1861 | plan marked for restore | stage 1 restores; exact | F | sep-heap |
| EVICT-29 | A:486, D:471-472 | pressure on an array with a view past its size | pages beyond the size never unmapped | F | sparse |
| EVICT-30 | P:1359-1363 | idle graph's transients under pressure | arena pages unmapped, transients never evicted | F | sparse |

## 17. Holds, free!, finalizers (HOLD)
| id | contract | scenario | expected | kind | device |
|---|---|---|---|---|---|
| HOLD-01 | A:79 | `free!(a)` from 16 threads | creator hold dropped once | C | any |
| HOLD-02 | A:79, A:482 | `free!(a)` while g1 and g2 hold it | graphs keep running; eager calls, declarations and stores through `a` throw | F/E | any |
| HOLD-03 | E:333-336 | `free!(feat)`, `free!(e1)`, `free!(d1)` while their runs are in flight | released only after both last runs (canary) | L/C | any |
| HOLD-04 | A:471 | `free!(view)`, `resize!(view)` | throws | E | any |
| HOLD-05 | I:2016-2023 | declaration adding a hold races the last hold drop | never 0→1: throws "released resource" or wins first | C | any |
| HOLD-06 | A:375 | last hold with a pending store and resize | dropped; staging retired; no write into released storage (validation) | L | any |
| HOLD-07 | P:1301-1302 | drop references without `free!`; GC | finalizers only push to the inbox; `sweep!` releases | L | any |
| HOLD-08 | E:319-322 | `a3.+=1; a3=nothing; GC; a4` | a4 does not reuse the region before token 7 passes | C | any |
| HOLD-09 | A:91 | view of a freed parent | use throws | E | any |
| HOLD-10 | A:374 | random declare/insert/delete/free/GC ×10 000 | reserved at baseline; no double free (debug pool) | L | any |
| HOLD-11 | I:2092-2093 | `free!(g)` with `g.unnamed` non-empty | those holds dropped | L | any |
| HOLD-12 | A:79 | `free!(a)` while lent | memory released after the stream's token | C | CUDA |
| HOLD-13 | A:79 | `free!(a)` with a pending store while a graph holds it | applied by the graph's next run | F | any |
| HOLD-14 | A:79 vs I:2148-2172 | `Array(a)` after `free!(a)` while a graph holds it | throw or read unspecified (ambiguity 22) | E | any |

## 18. Eager calls (EAGER)
| id | contract | scenario | expected | kind | device |
|---|---|---|---|---|---|
| EAGER-01 | A:451, P:1929-1931 | KA launch, broadcast, `fill!`, `copyto!`, `render!(dev)`, `trace!(dev)`, `fft`, `mul!` | each reaches `compilekernel`; 1 submission per launch | F/P | any |
| EAGER-02 | P:964-965 | N eager calls | N submissions; no cached plans | F | any |
| EAGER-03 | I:2119-2132 | pending store and resize on an argument | applied in front, same submission | F | any |
| EAGER-04 | I:2133-2135 | throw under the locks (freed argument) | packed region retired immediately | L | any |
| EAGER-05 | A:481 | eager call on a lent array | throws | E | CUDA |
| EAGER-06 | I:2128 | eager launch over a kernel-written length | indirect from the cell | F | any |
| EAGER-07 | A:126 | eager call | returns before the GPU finishes | P | any |

## 19. Host reads (READ)
| id | contract | scenario | expected | kind | device |
|---|---|---|---|---|---|
| READ-01 | A:85 | `Array(a)` with a pending store | includes the store; own submission | F | any |
| READ-02 | I:2149-2156 | Readback output of a graph | no GPU copy when nothing is pending; waits for the run's token | F/P | any |
| READ-03 | A:85 | `Array` of a graph array | throws | E | any |
| READ-04 | I:2145-2147 | kernel-written length while the next run is in flight | length and data from the same point | C | any |
| READ-05 | A:87 | `devicesize(a)` | correct; waits | F | any |
| READ-06 | A:85 | `Array(a)` while lent | throws | E | CUDA |
| READ-07 | A:452 | `Array(a)` races a run writing `a` | whole run's result or none, never a mix | C | any |

## 20. Vendor views (VIEW)
| id | contract | scenario | expected | kind | device |
|---|---|---|---|---|---|
| VIEW-01 | A:200-203 | `withview` on an array with a pending store | the stream sees the store | F | CUDA |
| VIEW-02 | A:203-204 | long vendor kernel, then a Mantle op | the op waits for the stream | C | CUDA |
| VIEW-03 | A:204 | block throws | later Mantle uses still wait; hold dropped after | E | CUDA |
| VIEW-04 | A:206 | eager call, store, `resize!`, `Array` on `a` inside the block | each throws | E | CUDA |
| VIEW-05 | A:206 vs I:1179-1192 | T2 runs a graph writing `a` during the block (nothing pending) | contract: throws; design: neither throws nor waits (ambiguity 8) | C | CUDA |
| VIEW-06 | A:207 | Mantle op on another array inside the block | no deadlock | C | CUDA |
| VIEW-07 | I:2252 | `free!(a)` from T2 during the block | nothing released before the stream's token | C | CUDA |
| VIEW-08 | I:2269 | nested `withview` on the same array | throws | E | CUDA |
| VIEW-09 | A:200 | `withview(MtlArray, …)` | same contract | F | Metal |
| VIEW-10 | P:708-711 | `withview` on an evicted array | restored first | F | CUDA sep-heap |
| VIEW-11 | A:203 | a graph run writing `a` is in flight when `withview` starts | vendor sees the completed write | C | CUDA |

## 21. Creation from data (DATA)
| id | contract | scenario | expected | kind | device |
|---|---|---|---|---|---|
| DATA-01 | A:69, D:269-277 | `MantleArray(dev,data)` with 1 KB, 1 GB, 8 GB | exact; one submission per chunk, each within the budget | F | any |
| DATA-02 | R:746-748 | 8 GB creation | peak RSS bounded by a few chunks | P | any |
| DATA-03 | A:69 | caller mutates `data` after return | array unaffected | F | any |
| DATA-04 | R:748-749 | `run!` reading the new array right after creation | waits for the chunks; exact | C | any |
| DATA-05 | R:753 | host-visible storage (lavapipe, Unified) | direct write; 0 submissions | F | lavapipe |
| DATA-06 | A:69 | data larger than the budget | OutOfDeviceMemory after pressure; no chunks leaked | E | any |
| DATA-07 | A:69 | zero length; strided host view; non-isbits element type | unspecified | E | any |
| DATA-08 | A:69 | mmap of a missing file, a short file, or a file modified while streaming | errors (unspecified) | E | any |
| DATA-09 | D:275-277 | creation | no pending changes; no inbox pushes | F | any |
| DATA-10 | A:69 | which device the mmap form uses | unspecified (ambiguity 16) | F | any |

## 22. Compile, split, queues (SPLIT)
| id | contract | scenario | expected | kind | device |
|---|---|---|---|---|---|
| SPLIT-01 | P:537-548 | estimated cost 3× the budget | ≥3 segments; never re-recorded | F | any |
| SPLIT-02 | P:567-568 | independent host↔device copy | goes to the transfer queue where one exists | F | mq |
| SPLIT-03 | I:578 | independent branches with gain > `crossqueuecost` | second compute queue; exact | F | mq |
| SPLIT-04 | I:1218-1243 | change prefix applied only in front of the first segment | later segments correct | F | any |
| SPLIT-05 | P:2321-2323 | waits between segments on different queues | no waits added inside a queue | P | mq |

## Ambiguous or contradictory contracts

1. **Declaring into a compiled graph: throws, or only marks the plan?** A:472 says "Declaring into a compiled graph throws". A:107-115 and D:244-254 (B3.16) say it only marks the plan. P:928 ("the graph stays fixed") and D:72 ("a graph is fixed after its first run") are left over from before B3.16.
2. **Is a plot an `insert!(g)` block or a piece?** A:119 says `insert!(g)` returns "a plot's draws". A:176, D:255-268 and P:1233-1251 say plots are pieces, and pieces do not recompile. L:916 states both.
3. **What does "recompile" mean for a plot change?** A:492 and A:159 say a pipeline-key change is a recompile, but under B3.17 removing and adding a piece recompiles no plan. The RayMakie gate in A:465 asks for "zero compiles" with plot inserts, yet a new pipeline key on insert increments the device compile counter (I:554-557). Which counter the gate reads is not defined.
4. **Persistent graph arrays across recompiles.** A:71 says `persistent = true` is kept between runs. A:112-113 says graph-array contents live only during a run, and only device arrays survive a recompile.
5. **Runs never overlap across a recompile.** `lastrun` is per plan (I:1216, I:1348), so a new plan's first run waits for the old plan only through shared resources and namers. That ordering is implied, not stated.
6. **`when!` with flag values other than 0 and 1.** The contract is "runs only where nonzero" (A:130), but the mechanism multiplies the count by the flag (P:2287-2289). Flags of 2, negatives and floats are undefined, and the flag's element type is unspecified.
7. **Eager calls inside `when!` / `repeat!` blocks.** Calls whose arguments are all device arrays run eagerly at declaration, once and unconditionally (A:127; A:130 "an eager call runs"). Nothing says whether this should throw.
8. **Lent arrays and runs.** A:206 and P:707-711 say runs throw on a lent array. But `lockstage` (I:1179-1192) locks only inbox resources, and `withview` with nothing pending makes no submission (I:2263-2266), so the array is never put into an inbox. A run writing the lent array would then neither throw nor wait for the vendor stream.
9. **Scope of the "inside a run" check.** `inrun` (R:460) is true when the task owns any graph lock. That includes declarations and compiles, not only "inside a run (any host function)" as A:487 and P:681-683 say.
10. **`holdparts!` and equal part ids.** A:437 and D:455-457 say it throws when the task holds a part with a "higher id". I:1263-1266 and I:1289 throw for a "lower or equal" waited-for id (`>=`).
11. **Unguarded deletes.** `delete!(p,h)` (I:902-906) has no guard, so calling it twice queues `Deleted` twice and unnames twice. `delete!(g,h)` with another graph's handle (I:281-296) is also unspecified.
12. **Suspected hold leak.** `draw!` names resources unconditionally (I:878). If a `render!` sits inside a `repeat!` body, `repeat!`'s throw path (I:352-360) drops the nodes without unnaming the draws.
13. **Window ownership edges.**
    - P:1022 says a graph presents to at most one window, but not where the error is raised.
    - E:490 has a kernel (`blurpass!`) writing the window, while B3.8 defines the owner as the graph that "declares `render!`".
    - Eager `render!(dev, win=>…)` is unspecified.
    - Whether the window frees its binding after `free!(g)`, GC, or deleting the render block is unspecified (the reference is weak, I:1407-1413).
14. **`Array(win)` with no presenting run to follow** (A:184): a minimized window, a freed graph, or a call from the presenting graph's own host function. Hang, error or deadlock is not specified.
15. **Skipped runs are invisible.** A user cannot tell a skipped (minimized) `run!` from a real one (A:483). Its outputs are stale and no return value is specified.
16. **Read-only input modes** (D:193-198):
    - the name and how read-only is enforced are open;
    - the mmap form in A:69 has no device argument, which conflicts with rule 3 (P:50-53);
    - "leaves only VRAM" for host data implies Mantle keeps a host copy or aliases caller memory, which conflicts with B3.18's streaming (D:269-277);
    - a file changed on disk is not covered.
17. **Eviction status.** B3.9 rules eviction automatic, but A:464, A:486, A:490, P:1374 and I:1786 still call it a proposal.
18. **Small undefined cases:**
    - `run!` of an empty graph (P:888-889);
    - `repeat!(g,0)`;
    - nested `repeat!`, which is only implied (I:346);
    - hostwritten length mismatch or partial ranges (A:144, I:1377-1388);
    - a value GPURef set to a value of another type (I:1367);
    - `submissions(g)` before the first compile.
19. **Concurrent declaring.** P:1440 and I:53 say "declaring into one graph is done by one task", while A:114 and I:244 serialize declarations with the graph lock. Whether `MantleArray(g, …)` takes the lock is not shown.
20. **Window capacity after growth.** The capacity after a recompile for a beyond-capacity resize is unspecified (A:191, D:442-444).
21. **Value GPURefs and hostwritten arrays in pieces.** A piece has its own argument region (I:826-828), but value GPURefs reach entries through `register!` (R:686). `registerpiece!` is "for its uses only", so pieces may not get entry slots or marks.
22. **Host read after `free!`.** A:79 and R:1249-1251 list eager calls, declarations and stores as throwing after `free!(a)`. Host reads are not listed, and `Array` (I:2148-2172) has no `usable` check.
23. **No call to reorder pieces.** A:176 says "a changed order is a list rewrite", but no call is named (open decision 15).
24. **Stale comments from the multi-presenting design.** I:21-22 (waiting for "acquires in progress before a new swapchain generation") and I:1121-1122 ("made by another graph") predate B3.8.
25. **What a host function may write.** HostCall `writes` is not enforced (A:128). A:473 and P:692-693 forbid a host function to "write into" its own graph, but P:662-664 allows stores into the device arrays that graph names. The boundary is not stated.
26. **Insert cost vs pipeline compile.** The B3.17 rule "an insert in the ms range is fine" is not tied to whether a pipeline compile on insert counts against the RayMakie compile-counter gate (A:465). The same unclear compile-counter scope as item 3.
27. **When the present-span compile error appears.** "A host node between window write and present is a compile error" (P:1044-1045). It surfaces at the first `run!`, not at declaration. Unstated; tests HOST-13 and WIN-29 assume `run!`.
28. **Waiting for the previous run.** "A run's first segment waits on the GPU for the previous run of g" (A:101) is implemented per plan (I:1216). It holds across recompiles only through item 5.
29. **Hostwritten write while recompiling.** A:144 keeps data on the host only "while the graph has no current plan". I:1381 also keeps it while a recompile is pending, and I:1045 copies the old entry's ranges into the new one. A:144 should say so.
