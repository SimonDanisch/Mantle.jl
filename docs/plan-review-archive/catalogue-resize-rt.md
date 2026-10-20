# Mantle resizing, stores and ray-tracing API: test catalogue

This catalogue has 246 test cases in 18 tables, followed by 27 contracts that are ambiguous or contradict each other. I read the docs only; nothing was run. Doc references use `file:line`.

**File abbreviations**
- `RR` = `dev/Mantle/docs/resizing-and-raytracing.jl`
- `API` = `dev/Mantle/docs/api.md`
- `PLAN` = `dev/Mantle/docs/recording-plan.md`
- `INT` = `dev/Mantle/docs/internals.jl`
- `DEC` = `experiments/plan-review/decisions.md`
- `VK-AS` = `vkdocs/chapters/accelstructures.adoc`
- `VK-BC` = `vkdocs/chapters/commonvalidity/build_acceleration_structure_common.adoc`
- `VK-DC` = `vkdocs/chapters/commonvalidity/build_acceleration_structure_device_common.adoc`

**Instrumentation the rows assume.** These are test hooks, not API:
- **Counters:** a per-device compile counter, a record counter (recordings plus `AccelCommand` re-records), and tokens per channel.
- **Step log:** each submission's prefix steps, with kind, target and seq.
- **Memory:** `reserved`/`mapped(dev)`, live upload regions and bytes, the cell-arena count.
- **Per resource:** `holds`, `composites`, the readers snapshot, `pending(x)`, `between(plan)`.

**Fault-injection (FI) points**
- **FI-G:** a gate kernel that spins on a host-visible flag, to keep a submission in flight.
- **FI-H:** a `HostCall` that blocks on a `Channel`, to hold a run between stage 1 and stage 2.
- **FI-P:** a hook between `prepare!` and taking the locks (`submitapplying`, the run's stage loop).
- **FI-R:** a hook between `record!` and `register!`.
- **FI-A:** make `acquire!` or `acquireupload!` throw `OutOfDeviceMemory` at the n-th call.
- **FI-B:** throw inside `recordown`/`current!` after `bindnow!`.
- **FI-C:** test-only overrides of `caps(dev)`: `sparseresidency`, `sparsepage`, `inlinestore`, budget, `maxInstanceCount`.
- **FI-E:** force eviction (open decision 16).
- **FI-W:** remove one `mark!` or wait, for negative controls.

**Device codes**
- `any`: any device, lavapipe included.
- `SP`: sparse residency (RADV, NVIDIA, AMD Windows, lavapipe).
- `RT`: `VK_KHR_acceleration_structure` (RADV, NVIDIA, AMD Windows).
- `M5`: Metal.
- `2Q`: a second queue family.
- `VAL`: validation layers plus GPU-AV.
- `WIN`: a window.
- `VND`: a vendor view (CUDA, HIP or M5).
- `SW`: the software TLAS.
- `HEAP2`: Readback memory is a separate heap.

## A. Scheduling: a call changes nothing on the device (Rule 8)

| id | contract | scenario | expected | kind | device |
|---|---|---|---|---|---|
| SC-01 | PLAN:59-71; RR:238-247; DEC:208-216 | Fixed 1000-element `a`. Snapshot reserved, mapped and per-channel tokens. Call `resize!(a, 10^8)`. | `length(a)==10^8`. Reserved, mapped, tokens and `a.state.storage` are unchanged. `pending(a)==[Resize]`. | functional | any |
| SC-02 | RR:244-251; DEC:269-277 | `a[1:1024] = x` (4 KiB). | Exactly one 4 KiB upload region is live. Device reserved is unchanged. No token is claimed. `pending(a)==[Store]`. | functional | any |
| SC-03 | RR:18-20; PLAN:69-70 | FI-G: kernel K reads `a` and is submitted gated. Then `a[1:4]=y`. Then a run of `g2` reads `a`. Open the gate. | K sees the old values. g2 sees `y`, applied in g2's stage-1 prefix. | concurrency | any |
| SC-04 | PLAN:69-70; RR:544-548 | FI-G: run1 reads `a` (n elements), in flight. Thread B calls `resize!(a,2n)` (a move) and stores. Then run2. Open the gate. | run1 used n and the old address. run2 uses 2n and the new data. The old region is retired with run2's prefix token and freed only after run1's token. | concurrency | any |
| SC-05 | API:452 | Thread A calls `run!(g)` 10^4 times; g writes `length(a)` and a checksum to Readback. Thread B alternates `copy!(a, p_i)`. | Every (length, checksum) pair matches one fully applied `p_i`. | concurrency | any |
| SC-06 | RR:705-712; DEC:213-214 | FI-H holds g between stages; g launches directly over fixed `a`. Thread B calls `resize!(a,2n)`, `a[1:2]=x`, `copy!(b,y)`. | Every call returns without waiting on `runend` or on tokens. | perf-gate | any |
| SC-07 | API:80 | `resize!(a,…)`. Then run graph g2, which does not name `a`. Then `Array(b)`. | `a` is still pending, its storage unchanged, and no step on `a` appears in either submission. | functional | any |
| SC-08 | PLAN:66-67; INT:2117-2131 | Three stores pending on `a`. Then eager `dispatch!(dev,k,(a,out))`. | One submission: three InlineStores, then the dispatch. `out` reflects the stores. | perf-gate | any |
| SC-09 | API:483; INT:1003-1004 | Window minimized. Graph g names `a`, which has a pending store. `run!(g)` returns Skipped. Then `Array(a)`. | The store is still pending after `run!`. `Array` applies it in its own submission and returns the new data. | functional | WIN |
| SC-10 | PLAN:66-68; RR:291-292,352-353 | `a[1:1]=x`, then `b[1:1]=y`, then `resize!(a,…)`. One eager kernel over both. | Steps are emitted in seq order: a.Store, b.Store, a.Resize. | functional | any |

## B. `resize!` and storage growth

| id | contract | scenario | expected | kind | device |
|---|---|---|---|---|---|
| RZ-01 | RR:180-182,190,535 | Fixed region of 1000 elements; `resize!(a,500)`; run. | Steps: Becomes(Committed) and CellWrite, no MoveStorage. Address unchanged; `a[1:500]` intact. | functional | any |
| RZ-02 | RR:180-182,625-627 | A recorded plan launches directly over fixed `a`. Call `resize!(a, length(a))`. | Storage untouched (Adopt); `resizable` becomes true. Compile count is open (A14); per the code, one recompile. | functional | any |
| RZ-03 | RR:197,540-541 | Small fixed `a` (n elements, below the sparse threshold); `resize!(a,3n)`. | MoveStorage into Committed of 2·need bytes. New address in the cell; `a[1:n]` kept. Old region retired with the applying token. | functional | any |
| RZ-04 | RR:196,203,537-539 | Fixed `a` of at least 4 sparse pages; `resize!(a,2n)`. | New Reserve of `min(nextpow(2,4need), reservelimit)`. BindPages on the sparse queue before MoveStorage; the submission waits on the bind token; contents kept. | functional | SP |
| RZ-05 | RR:192 | Reserve-backed array grows within `mapped`. | CellWrite only, no BindPages, same address. | functional | SP |
| RZ-06 | RR:186-189,193 | Reserve-backed: `resize!(a,length(a)+1); run!(g)`, 10^5 times. | BindPages count ≤ ⌈log2(final/initial)⌉+1. Same device address every run; no recompiles. | perf-gate | SP |
| RZ-07 | RR:194 | Reserve grows past `reserve.bytes`. | One move into a new reserve: the address changes once, contents kept, old reserve retired after the token. | functional | SP |
| RZ-08 | RR:198 | FI-C `sparseresidency=false`; grow small and large arrays repeatedly. | Every growth past storage moves to 2·need; growth within storage is a CellWrite only; no BindPages. | functional | any |
| RZ-09 | RR:202 | Needs of `4·sparsepage-1` and `4·sparsepage` bytes. | The first becomes Committed, the second a Reserve. | functional | SP |
| RZ-10 | RR:203,50-51 | lavapipe (2 GiB sparse space): grow `a` past `reservelimit`. | Never a reserve smaller than the need (A11): either Committed or a defined error, and no device fault. | error-path | lavapipe |
| RZ-11 | RR:231-234; API:80 | Reserve with 64 MiB mapped. View `v=view(a,1:10^6)` taken beforehand. `resize!(a,1)`; a kernel reads `v`. | `mapped` unchanged; `v` reads the old values; GPU-AV clean. | functional | SP+VAL |
| RZ-12 | API:81; RR:100 | After RZ-11: `trim!(a)`; run; then `resize!(a,10^6)`; run. | `trim!` stays pending until the next submission. Then pages past the size are unbound and retired with that token. Regrowth binds again; `a[1]` kept. | functional | SP |
| RZ-13 | API:486; DEC:471-472 | FI-C small budget; pages exist past the size; allocate elsewhere. | Pressure never unbinds those pages. `OutOfDeviceMemory` if nothing else frees memory. | error-path | SP |
| RZ-14 | RR:506; API:80 | A 4×5 array; `resize!(a,7)`. | Size becomes (4,7); the first 20 elements are kept as linear memory. | functional | any |
| RZ-15 | RR:504-505; API:471 | `resize!(view(a,1:10),20)`; `resize!(reshape(a,…),n)`. | ArgumentError; nothing pending. | error-path | any |
| RZ-16 | RR:518 | `free!(a); resize!(a,10)`. | ArgumentError thrown under the lock; nothing pending; length unchanged. | error-path | any |
| RZ-17 | API:481; PLAN:707-711 | Inside `withview(CuArray,a,s)`: `resize!(a)` and `a[1:1]=x`. | Both throw; both succeed after the block. | error-path | VND |
| RZ-18 | RR:527-528,771-772; API:80 | Graph array placed for n. Resize to n÷2 and to n, then to n+1, across runs. | Within n: pending CellWrites, no recompile. n+1: exactly one recompile at the next `run!`, never a throw. | functional | any |
| RZ-19 | API:80; RR:767 | `resize!(a,0)`. Dispatch and draw over `launchrange(a)`. Then `resize!(a,n)`, a store, a run. | Zero groups or vertices, validation clean. Then correct at n. | functional | any+VAL |
| RZ-20 | RR:530-532 (`storageat`) | Committed `a` of n: `resize!(a,2n)` (move), `a[1:1]=x`, `resize!(a,8n)` (move), one run. | One prefix: Move, Store, Move; the second move reads the first's new storage after a barrier. `a[1]==x`, `a[2:n]` old. Both replaced storages retired with the one token. | functional | any |
| RZ-21 | RR:284-285,538 | 100-element `a`: `resize!(a,10); resize!(a,1000)`. | One Resize step (dims 1000, keep 10 elements); `a[1:10]` old. | functional | any |
| RZ-22 | RR:404-406,544-545 | FI-G: graph A reads `a` on queue X, in flight. Eager `resize!` plus `fill!` on queue Y moves `a`. | The move waits for A's token; the old storage is not reused before it. | concurrency | 2Q |
| RZ-23 | RR:542-543; INT:1881-1882 | FI-E evicts `a`; `resize!(a,2n)`; run. | Restore into storage of 2n copying min(host copy, requested), then CellWrite; the Resize applies in place. | functional | HEAP2 |

## C. Stores

| id | contract | scenario | expected | kind | device |
|---|---|---|---|---|---|
| ST-01 | RR:563; API:480 | `a[1:3]=[1,2]`. | DimensionMismatch before any upload region is acquired. | error-path | any |
| ST-02 | RR:572 | Length 5; `resize!(a,10)`; `a[6:10]=x`; run. | Accepted, since bounds are checked against the requested size; `a[6:10]==x`. | functional | any |
| ST-03 | RR:572,577 | Length 5; `resize!(a,3)`; `a[4:5]=x`. | BoundsError; the upload region is retired with no tokens; nothing pending. | error-path | any |
| ST-04 | RR:566-569 | `a::MantleArray{Int32}`; `a[1:1]=[1.5]`. | InexactError and the live upload count back to its earlier value. Per the code it leaks (A5). | leak | any |
| ST-05 | RR:249-251; DEC:269-273 | `a[1:n]=x; x .= 0`; run; `Array(a)`. | The original `x`. | functional | any |
| ST-06 | RR:553-555; API:82 | Store of exactly 65536 bytes at offset 0. Repeat on M5 and CUDA. | Vulkan: InlineStore via `emitstore!`. Other devices: written from the upload region. Same results everywhere. | functional | Vulkan/M5/CUDA |
| ST-07 | RR:553-555 | Store of 65540 bytes. | StagedStore copying from the upload region; the region is retired with the applying token. | functional | Vulkan |
| ST-08 | RR:584 | `UInt8` array: `a[2:4]=3 bytes`. | Staged (not 4-byte aligned); `a[1]` and `a[5]` untouched. | functional | Vulkan |
| ST-09 | RR:255-259; API:82 | `a[1:3]=x`, then `resize!(a,10^6)` (move); run. | Prefix order is Store then Move; `a[1:3]==x`. | functional | any |
| ST-10 | RR:558-559 | `v=view(m,1:2:9)`; `v[1:5]=x`. | Five strided elements written, the four in between untouched. When it runs is open (A6). | functional | any |
| ST-11 | RR:255 | `v=view(a,101:200)`; `v[1:10]=x`. | The Store's span is the parent bytes of elements 101:110. | functional | any |
| ST-12 | RR:555-557 | Store into a TLAS range source. | Pends TLAS Update (a refit) in the same call; step log shows Store then Update. | functional | RT |
| ST-13 | RR:556-557; RR:929-931 | Store into a BLAS index array. | Pends a BLAS Build and a TLAS Update. | functional | RT |
| ST-14 | RR:571,577 | `free!(a); a[1:1]=x`, directly and through a view. | ArgumentError; upload retired; nothing pending. | error-path | any |
| ST-15 | RR:562,588; API:93-94 | `a[1:n]=b` and `copyto!(a,b)` with `b` a MantleArray; `a[5]=1f0`; `a[:]=x`. | A device copy ordered after `b`'s pending changes, never a scalar or host round trip. Scalar and colon forms have defined behaviour (A15). | functional | any |
| ST-16 | RR:252-254 | 10^4 × `a[1:16]=x_i`, with `run!` every 100. | At most one live upload between runs; none after the last run; upload pool back at baseline. | leak | any |
| ST-17 | PLAN:1210-1213; RR:1455-1456 | One-element camera array; `cam[1:1]=[m]` every frame. FI-G keeps the GPU three frames behind. | No host wait in the store or in `run!`; InlineStore every frame; no staging allocation. | perf-gate | any |
| ST-18 | RR:560-561 | FI-A exhausts the upload budget while regions are retiring. Thread A stores into `a`; thread B calls `resize!(a,…)`. | A waits only inside `acquireupload!`; B's `change!` completes meanwhile; A then pends. | concurrency | any |
| ST-19 | RR:562-579 | `a[1:0]=T[]`. | Defined behaviour (A24): no Store pending and no 0-byte region. | error-path | any |
| ST-20 | API:85; INT:2157-2172 | A store is pending; `Array(a)`. | One submission: Store then copy; returns the new data. | functional | any |

## D. `copy!`, `copyto!` and views

| id | contract | scenario | expected | kind | device |
|---|---|---|---|---|---|
| CP-01 | RR:588; API:83 | `copyto!(a,x)` with `length(x) < length(a)`. | Prefix written, tail unchanged, no Resize. | functional | any |
| CP-02 | RR:588 | `copyto!(a,x)` with `length(x) > length(a)`. | BoundsError; nothing pending; upload retired. | error-path | any |
| CP-03 | RR:99,589-606; API:84,489; DEC:461 | `copy!(a,x)`, growing past storage. | `length(a)==length(x)` at return. Resize (keep 0) and Store have consecutive seqs. MoveStorage copies 0 bytes. After the run `a==x`. | functional | any |
| CP-04 | RR:589-591 | Thread A loops `copy!(a,x_i)`; thread B loops `a[1:1]=y`; runs check. | No run ever sees the new length with old contents; `a[2:end]==x_i[2:end]`. | concurrency | any |
| CP-05 | RR:607 | `copy!(view(a,1:10),x)` with `length(x)==10`. | A Store into parent 1:10, no Resize. | functional | any |
| CP-06 | RR:607-608 | `copy!(view(a,1:10),x)` with `length(x)==5`. | ArgumentError; nothing pending. | error-path | any |
| CP-07 | RR:595; DEC:269-273 | `copy!(a,x); x .= 0`; run. | `a` holds the original `x`, carried by an upload region (A4). | functional | any |
| CP-08 | RR:593-606,625-627 | Fixed `a` launched directly; `copy!(a,x)` with the same length. | Pins the ruling A14: a pure store (no recompile) or a Resize (one recompile). | perf-gate | any |
| CP-09 | API:91; PLAN:213-218 | `v=view(a,1:10)`; `free!(a)`; then `v[1:1]=x` and a dispatch over `v`. | Both throw: the view shares the freed mark. | error-path | any |
| CP-10 | API:91 | Live view `v`; `resize!(a,…)` moves `a`; a kernel reads `v`. | `v` reads the parent's new storage at its static offset. | functional | any |

## E. Creation from data (B3.18)

| id | contract | scenario | expected | kind | device |
|---|---|---|---|---|---|
| CR-01 | RR:741-749; API:69 | `a=MantleArray(dev,x)` on thread A; immediately `dispatch!(dev,k,(a,out))` on thread B. | `out` is computed from `x`: the dispatch waits on the upload's recorded write. | concurrency | any |
| CR-02 | RR:745-748 | `x` four times the submission budget. | One submission per chunk, each within the budget; peak live upload bytes bounded by a few chunks. | perf-gate | any |
| CR-03 | RR:746 | Storage is host-visible. | Direct copy, no submissions. | functional | M5 |
| CR-04 | DEC:193-198; API:69 | `MantleArray(path; mmap=true)` over a 2 GiB file. | Contents equal the file; RSS growth bounded by the chunks in flight. | perf-gate | any |
| CR-05 | RR:741-743 | `x .= 0` after the constructor returns. | The array holds the original `x`. | functional | any |
| CR-06 | RR:741-756 | FI-A throws at chunk k. | The constructor throws; storage and the k-1 regions are released after their tokens; reserved back at baseline. | leak | any |
| CR-07 | RR:748-749,660-664 | Create from `x`; then compile and run a new graph reading `a`, with no other access. | The first run waits on the upload and reads `x`. | functional | any |
| CR-08 | RR:660-664,695; PLAN:64-68 | `a=MantleArray(dev,T,n)` with no data; `a[1:3]=x`; then compile and run a graph naming `a`. | The first run applies the Store. Currently a likely miss (A1). | functional | any |

## F. Fixed vs resizable arrays, staleness, `register!`

| id | contract | scenario | expected | kind | device |
|---|---|---|---|---|---|
| FX-01 | RR:732; API:89 | Fresh or created-from-data `a`; `dispatch!(g,k,(a,),launchrange(a))`. | Compiled as a Direct launch. | functional | any |
| FX-02 | RR:633-636 | `resize!` called but not yet applied; compile a new graph. | Indirect launch; the first run applies the Resize in its prefix. | functional | any |
| FX-03 | RR:625-627; API:80 | A recorded plan launches directly over `a`; `resize!(a,n+1)`. | At the call: `plan.stale` and `g.recompile` set. The next `run!` recompiles once and launches over n+1. | functional | any |
| FX-04 | RR:620,736-739,757-758 | A plan with a `SizedBy(f,a)` temporary sized for 2n; then `resize!(a,2n)`, then shrink to n÷2. | Neither is stale; no recompiles. | perf-gate | any |
| FX-05 | RR:620,757 | Same plan; `resize!(a,2n+1)`. | Stale at the call; one recompile. | functional | any |
| FX-06 | RR:628-630,636-637 | A plan binds `a` by handle; a move is applied. | No stale flag. The parts that recorded the handle are marked and re-recorded; record counter goes up, compile counter does not. | perf-gate | any |
| FX-07 | RR:644-651,681 | FI-R: an eager call moves `a` between `record!` and `register!`. | `register!` returns Again; recompile; the run is correct. | concurrency | any |
| FX-08 | RR:705-712 | FI-R: a stale-making `resize!(a)` lands between compile and `register!`. | The plan recompiles before its first submission. | concurrency | any |
| FX-09 | RR:649-651 | 1000 recompiles of a graph naming `a`. | `length(a.state.dependents)==1`; the namers snapshot has one entry. | leak | any |
| FX-10 | RR:1222-1225 | Launch over `launchrange(r)`, where `r` is an array GPURef with a fixed target. | Indirect launch. | functional | any |
| FX-11 | RR:764-767; API:178 | A draw over a fixed array inside `when!`. | Indirect, count multiplied by the flag; toggling the flag is a store; no recompiles. | functional | any |
| FX-12 | RR:212 | Resize back to the original length; compile a new graph. | Still resizable, so the launch is Indirect. | functional | any |

## G. `lengthof` and kernel-written lengths

| id | contract | scenario | expected | kind | device |
|---|---|---|---|---|---|
| KL-01 | API:76,476 | `lengthof(x)` where `x` has no `capacity`. | Throws at declaration; graph unchanged. | error-path | any |
| KL-02 | RR:113-116; API:90 | A kernel writes 500 into `lengthof(alive)` (capacity 10^6); the next node launches over `launchrange(alive)`. | Exactly 500 work items: the prepare node wrote the command. | functional | any |
| KL-03 | API:85,87; INT:2145-2147 | After KL-02: `devicesize(alive)` and `Array(alive)`. | 500. `Array` returns 500 elements, with length and data from one submission. | functional | any |
| KL-04 | RR:767 | The kernel writes 0. | Downstream dispatch, draw and trace do no work; validation clean. | functional | any+VAL |
| KL-05 | RR:213-214 | A buggy kernel writes capacity+1. | Defined behaviour (A13): no launch past the capacity, no fault. | error-path | any+VAL |
| KL-06 | RR:737-739,758-759 | A temporary `SizedBy(alive)`; lengths vary from 0 to capacity over 100 runs. | No recompiles. | perf-gate | any |
| KL-07 | RR:761-762 | Eager dispatch over `launchrange(alive)`. | Indirect from the cell, with the kernel's count. | functional | any |
| KL-08 | RR:101,213 | `resize!(alive,7)` pending; the run's kernel then writes 300. | Defined behaviour (A13). In sequence: prefix CellWrite(7), then the kernel, so 300. | functional | any |
| KL-09 | INT:2145-2147 | Thread A runs g (writes `alive`) 1000 times; thread B calls `Array(alive)`. | Every result is a consistent (length, data) pair. | concurrency | any |

## H. Coalescing and sequence numbers

| id | contract | scenario | expected | kind | device |
|---|---|---|---|---|---|
| CO-01 | RR:283-284; API:82 | `a[1:4]=x; a[1:8]=y`; run. | One Store (y). x's region is retired at the second call with no tokens. | perf-gate | any |
| CO-02 | RR:283 | `a[1:4]=x; a[3:6]=y`. | Two Stores in order: `[x1,x2,y1..y4]`. | functional | any |
| CO-03 | RR:255,283 | `view(a,3:6)[1:4]=y; a[1:8]=z`. | One Store: spans compare in parent bytes. | perf-gate | any |
| CO-04 | RR:284-285 | `resize!(a,10); resize!(a,5); resize!(a,20)`. | One Resize: dims 20, keep 5 elements. | functional | any |
| CO-05 | RR:284 | `resize!(a,10); a[1:3]=x; resize!(a,2)`. | Three steps, no merge across the Store; final `[x1,x2]`. | functional | any |
| CO-06 | RR:285-288; PLAN:1100-1105 | Before any submission: `push!` (Build), `tlas[h]=m` (Store and Update), `visible!(r,false)` (Store and Update). | One Build, emitted after both stores; the build sees `m` and mask 0. | functional | RT |
| CO-07 | RR:285-286 | `tlas[h]=m1; tlas[h]=m2`. | One source Store (m2) and one Update. | perf-gate | RT |
| CO-08 | RR:286,1196-1197 | `r[]=b; r[]=c`; run; free the creators. | One CellWrite (c). The holds and reader entries of `a` and `b` drop after that token; both released. | leak | any |
| CO-09 | RR:286 | `r[]=b; r[]=a`, where `a` is the original target. | One Rebind (a). `a.composites==1`; `r` appears once in `a`'s readers; `b` is released after `free!(b)`. | leak | any |
| CO-10 | RR:286-288 | Store into src1 (Update), `resize!(src2)` (Build), store into src1 again (Update). | The merged Build takes the last seq, so it is emitted after the second src1 store. | functional | RT |
| CO-11 | RR:288-290 | `delete!(t,r1)`, `push!`, `delete!(t,r2)` before any submission. | Both deferred drops are kept; holds and readers of r1 and r2 drop after the applying token. | leak | RT |
| CO-12 | RR:292 | 8 threads × 10^4 stores into distinct arrays; one eager kernel over all of them. | Seqs are unique; steps are emitted sorted by seq. | concurrency | any |

## I. Applying changes: `prepare!`, `submitwith!`, Again

| id | contract | scenario | expected | kind | device |
|---|---|---|---|---|---|
| AP-01 | RR:317-328; API:432-433 | An eager call with a Resize past storage pending; assert in `acquire!` that no lock is held. | Storage is allocated in `prepare!` with no sync-state, channel or pool lock held. | functional | any |
| AP-02 | RR:329-334 | FI-P: another thread calls `resize!(a, larger)` in the gap. | Again before anything is emitted; prepared storage released; the retry applies both changes. | concurrency | any |
| AP-03 | RR:329-334 | FI-P: another submitter applies the change in the gap. | Again; prepared storage released; nothing applied twice; reserved as expected. | leak | any |
| AP-04 | RR:327 | FI-A: `acquire!` throws in `prepare!`. | The call throws; `pending` unchanged; once memory is freed, the next submitter applies it. | error-path | any |
| AP-05 | RR:393-397,415-420; PLAN:712-715 | FI-B: throw after `bindnow!`. | Pages bound once; no rebind on retry; no orphaned pages (A18). | error-path | SP+RT |
| AP-06 | INT:1241; DEC:498-499 | Two segments on two queues; segment 2 reads a Store applied in segment 1's prefix. FI-W removes the wait as a negative control. | Segment 2 reads the new data; the control reads stale data. | concurrency | 2Q |
| AP-07 | RR:404; PLAN:626-631 | Another graph reads `a` on queue X, in flight; an eager store into `a` on queue Y. | The prefix waits for that run (no write-after-read hazard). | concurrency | 2Q |
| AP-08 | RR:406-407 | Replaced storage while FI-G holds a reader. | Not reused before the reader's token; GPU-AV clean. | leak | any+VAL |
| AP-09 | PLAN:619-621; DEC:217-224 | 1000 arrays each with a pending store; one run names all of them. | One submission per segment; 1000 prefix steps; host time linear in the number of changes. | perf-gate | any |
| AP-10 | API:202-206 | A resize and a store pending; `withview(CuArray,a,s)`. | The view has the new length and data; the stream waited on the applying token. | functional | VND |
| AP-11 | RR:462-475,491-499 | FI-P: a TLAS member snapshot changes between expansion and locking. | `lockexpanded` starts over; no deadlock; the build sees the new member. | concurrency | RT |

## J. Changes during runs: between stages, WaitFor, `inrun`

FI-H holds g between stages unless noted.

| id | contract | scenario | expected | kind | device |
|---|---|---|---|---|---|
| RN-01 | INT:1321-1324; PLAN:660-664 | Stage 2 launches indirectly over resizable `a`; between stages `a[1:4]=x`. | Stage 1 saw the old data; stage 2's prefix applies `x`. | concurrency | any |
| RN-02 | RR:343-347 | Stage 2 launches directly over fixed `a`; between stages `resize!(a,2n)`. | Stage 2 uses n and the old storage (the Resize is held); it is still pending after the run; the next run recompiles once and uses 2n. | concurrency | any |
| RN-03 | RR:345-347 | RN-02, plus `a[3:4]=z` before the resize and `a[1:2]=y` after it. | Stage 2 sees `z` but not `y`; the next run sees `y`. | concurrency | any |
| RN-04 | RR:347-349; API:487 | RN-02, plus thread C runs an eager `fill!(a,1)`. | C waits on `runend` until g submits its last stage; then Resize and fill are applied; the fill covers 2n. | concurrency | any |
| RN-05 | RR:349-350,449-451 | RN-04, but the eager call comes from g's own host function. | ArgumentError (`inrun`) with no wait; the run completes; the change stays pending. | error-path | any |
| RN-06 | RR:442-451 | g2's host function calls `Array(a)` while `a` has a change held for g. | ArgumentError. | error-path | any |
| RN-07 | INT:996-998,1151-1152; RR:450,460; API:487 | Thread C calls plain `run!(g2)` (no host node), which needs `a`'s held change. | Pins A2: the api says it waits; the code throws. | concurrency | any |
| RN-08 | RR:444-446 | A host function starts a task that calls `Array(a)` and waits for it. | Documented as undetected deadlock: `@test_broken` with a watchdog. | concurrency | any |
| RN-09 | INT:1093-1094 | One-stage plan in flight; 1000 stale-making resizes plus eager calls on thread C. | C never waits. | concurrency | any |
| RN-10 | INT:1092-1093 | g's run is blocked in `holdparts!` (graph h holds the shared arena part); a stale-making resize; an eager call. | The eager call does not wait; g later sees the stale flag and recompiles. | concurrency | any |
| RN-11 | INT:1157-1158 | C is waiting (WaitFor); g's host function throws. | `endbetween!` wakes C; C applies the change; g's next run recompiles and is exact. | error-path | any |
| RN-12 | RR:340-342; PLAN:666-670 | Stage 2 binds indices by handle; between stages `resize!(indices)` past storage. | Stage 2's prefix moves the array (copying what stage 1 wrote) and re-records the parts; no wait; no recompile. | concurrency | any |
| RN-13 | RR:346; API:487 | Stage 2 has a temporary sized for 2n; between stages `resize!(a,3n)`. | Held for g; other submitters get WaitFor; the next run recompiles. | concurrency | any |
| RN-14 | API:487; RR:438 | Stage 2 runs a software TLAS build sized from capacity; between stages a `push!` past capacity. | Held for g; other submitters touching `t` wait; recompile next run. | concurrency | SW |
| RN-15 | RR:437-441,714-721 | Stage 2 binds `r`'s target by handle; between stages `r[]=b`. | Stage 2's parts are re-recorded with `b`; no wait; no recompile. | concurrency | any |
| RN-16 | RR:839-842 | Stage 2 traces `t`, whose range is over fixed `a` launched directly in stage 1; between stages `resize!(a,n+10)`. | `a`'s Resize is held. `t`'s Build applies with the host total n+10; the extra records are mask 0; rays hit only n instances; validation clean. | concurrency | RT+VAL |
| RN-17 | DEC:225-243; PLAN:594-601 | Plan P writes `a`; plan Q reads `a` and is between stages; P runs on thread C. | P waits for Q's whole last run on the GPU, not on the host. | concurrency | any |
| RN-18 | API:452 vs INT:1321-1324 | g writes through `r` in both stages; between stages `r[]=b`. | Stage 1 writes `a`, stage 2 writes `b`. Pins A8. | concurrency | any |
| RN-19 | API:421-433 | 4 threads, 2 graphs with host nodes sharing arrays and an arena part, plus changes and eager calls, 10^4 iterations, with a watchdog. | No deadlock; results exact. | concurrency | any |
| RN-20 | RR:449-455 | C waits on `runend`; thread D pends changes on `a` and runs unrelated graphs. | D never blocks. | concurrency | any |

## K. Free and release with pending changes

| id | contract | scenario | expected | kind | device |
|---|---|---|---|---|---|
| FR-01 | RR:1252-1256; API:79,375 | `a` has a pending Resize past storage and a Store; only the creator holds it; `free!(a)`. | No submission ever; upload retired with no tokens; no storage allocated; cell released; memory and uploads at baseline. | leak | any |
| FR-02 | API:482 | A graph holds `a`; a store is pending; `free!(a)`. | `a`'s handle throws; the next run applies the store; memory comes back after `free!(g)`. | functional | any |
| FR-03 | API:79 | `free!(a)` from 8 threads at once. | The creator's hold drops once; `release!` runs once. | concurrency | any |
| FR-04 | RR:1253-1254 | FI-P: the last hold drops between an eager call's `prepare!` and its locks. | The eager call throws under the lock; prepared storage released. | leak | any |
| FR-05 | RR:1208-1219 | `r[]=b`, then `free!(r)` before any submission. | Rebind dropped. The deferred drop of `a` runs with `r`'s last tokens; `b`'s composite hold drops. | leak | any |
| FR-06 | RR:1257-1258; INT:2100-2111 | Drop every reference to an array with pending changes; `GC.gc()`; sweep. | Released through the inbox; no submission. | leak | any |
| FR-07 | RR:1082-1084 | `t` has 3 ranges and a pending Build; `free!(t)`. | The Build is never applied; storage, scratch and records retired with `t`'s last tokens; members' holds drop. | leak | RT |
| FR-08 | API:156 | `free!(blas)` while a range holds it; trace; then `delete!` and a run. | Traces stay correct; the BLAS is released after the Build's token. | leak | RT |
| FR-09 | API:482 | `free!(src)` while a range holds it. | The handle throws; `src`'s instances are still traced. | functional | RT |
| FR-10 | INT:1009,1197-1200 | g is between stages and C waits (WaitFor); thread D calls `free!(g)`. | D waits for the graph lock; C wakes when the run ends; no deadlock. | concurrency | any |
| FR-11 | INT:1888 | FI-E: an evicted array with a pending resize and prepared restore storage drops its last hold. | The host copy and the prepared storage are released. | leak | HEAP2 |
| FR-12 | API:79; INT:2148-2172 | `free!(a)`, then `Array(a)` while a graph still holds `a`. | Defined behaviour (A25). | error-path | any |

## L. TLAS

| id | contract | scenario | expected | kind | device |
|---|---|---|---|---|---|
| TL-01 | RR:124-125; API:151 | `t=TLAS(dev)`; eager trace and ray-query dispatch over `t`. | Prefix Build with count 0; every ray misses; validation clean. | functional | RT+VAL |
| TL-02 | RR:124-125,660-664,695 | `t=TLAS(dev)`; compile and run a graph that traces `t`, with nothing else touching `t`. | The first run applies the zero-instance Build before its trace (A1). | functional | RT+VAL |
| TL-03 | RR:124,1056 | `push!` a range of 64 instances, then one more instance. | At 64: no capacity step. At 65: capacity steps (records Resize, CellWrite, MoveAccel) before the Build; capacity 128; old AS and scratch retired with the token; AccelCommands marked; no recompile. | functional | RT |
| TL-04 | RR:1056-1058 | Grow to 10^5 instances, then `delete!` every range. | Capacity stays 131072; Build count 0; storage kept until `free!(t)`. | functional | RT |
| TL-05 | RR:22,914; DEC:137-143 | Log `emitbuild!`. | `primitiveCount` equals the host total, never the capacity. | functional | RT |
| TL-06 | RR:909-911 | Same on Metal. | `countcell` equals the total; `builtcount` equals the host total. | functional | M5 |
| TL-07 | RR:131; API:155 | `tlas[h]=m`. | A source Store and an Update carrying `builtcount`; no Build; hits move. | functional | RT |
| TL-08 | RR:132,830; API:166 | `visible!(t,r2,false)`, later `true`. | Refit only. r2 is never hit; InstanceId of other instances unchanged; restored by `true`. | functional | RT |
| TL-09 | RR:1029-1046; API:156 | Three ranges; `delete!(t,r1)`. | r1's entry inactive; a Build; r1 never hit. r1's holds and reader entries drop after the applying token, not before. | functional | RT |
| TL-10 | RR:149-151,789,1013 | After TL-09, `push!` a new range. | It reuses r1's slot: `instanceCustomIndex` equals the slot. Other ranges' slots unchanged; `InstanceId-offsets[slot]` correct for every range. | functional | RT |
| TL-11 | RR:137,987 | `build!(g,t)`, 100 runs with a constant count. | One Build per run; no AccelCommand re-records after the first. | perf-gate | RT |
| TL-12 | RR:138,989-993; PLAN:1090-1093 | `refit!(g,t)`; `push!` between runs. | Prefix Build, then the marked refit re-recorded after the prefix with the new `builtcount`. VK-BC VUIDs 03758, 03769, 03663 and 03664 clean. | functional | RT+VAL |
| TL-13 | TL-12 with FI-W skipping the `mark!` in `buildcount!` | Negative control. | A validation error on the Update. | error-path | RT+VAL |
| TL-14 | RR:889,971 | `refit!(g,t)` compiled while `builtcount` is −1. | The command records a Build; after `built!` it is marked and records an Update. | functional | RT |
| TL-15 | RR:985-987 | Graph A has `build!(g,t)` with a changing count; graph B has `refit!(g,t)`. | B re-records once per count change, never otherwise. | perf-gate | RT |
| TL-16 | VK-BC:113-120 (VUID-03667) | Builds from the prefix, an AccelCommand and a capacity growth. | ALLOW_UPDATE is set on each, so any later refit is valid. | functional | RT+VAL |
| TL-17 | VK-DC:7-27 (VUID-12258/12259) | Build and Update at full capacity. | Scratch ≥ max(buildScratchSize, updateScratchSize); validation clean. | functional | RT+VAL |
| TL-18 | VK-BC:278-283; resources.adoc:8983-8986 | FI-C `maxInstanceCount=1000`; `push!` 1500 instances. | A defined error (A12); never a build or sizes query over the limit. | error-path | RT |
| TL-19 | RR:38-39,204,1061-1062 | `t` grows past the sparse threshold. | AS storage is on an Accel pool region, never a reserve. | functional | SP+RT |
| TL-20 | RR:797-798,912-924; API:479 | Range over `alive` (capacity C); the kernel writes 10, then 50, then 3; `refit!` only. | Build count is `t.capacity`. Records past the device total have mask 0 and reference a valid BLAS. Refits valid; only active instances hit. | functional | RT+VAL |
| TL-21 | RR:797-798 | `delete!` the kernel-written range. | Back to HostTotal; Build with the exact count. | functional | RT |
| TL-22 | RR:1048-1054 | `resize!(src,n')`. | A Build on `t` pending from the same call; capacity grown by the applier. | functional | RT |
| TL-23 | RR:1003,1048-1051 | `push!` with an array GPURef as source; then `r[]=b` with the same length, then a different length. | Update, then Build. | functional | RT |
| TL-24 | RR:1002,1022-1025 | `push!` with a freed BLAS, freed source or freed GPURef. | ArgumentError; holds taken so far are dropped; no slot used. | error-path | RT |
| TL-25 | RR:1042-1043 | `push!(t,blas,pos; scale=pos)`, then `delete!`. | Holds and reader counts back at baseline. | leak | RT |
| TL-26 | RR:818,821-823 | `push!` with a view as source. | Defined behaviour (A22). | error-path | RT |
| TL-27 | RR:1029-1031 | `delete!` the same range twice; `delete!` a range of another TLAS. | A defined error (A23); `t` unchanged. | error-path | RT |
| TL-28 | RR:1036-1043; API:156 | FI-G: a trace of `t` is in flight. `delete!(t,r)`, `free!(blas)`, `free!(src)`; the next frame; open the gate. | The in-flight trace is exact. `blas` and `src` are released only after the Build's token; GPU-AV clean. | concurrency | RT+VAL |
| TL-29 | RR:1061-1063 | FI-G: a trace on another queue is in flight; `push!` past capacity; the next run. | The old AS is retired after the trace's token. | concurrency | RT+2Q |
| TL-30 | RR:302-309,995-1027 | Thread 1 `push!(t,blas,a)`, thread 2 `resize!(a,n')`, 10^4 times with runs. | Build count equals the sum of current lengths; no deadlock. | concurrency | RT |
| TL-31 | RR:409-414 | A BLAS-only eager submission applies the BLAS Build; a later trace. | `t`'s Update waits for the BLAS write. | concurrency | RT |
| TL-32 | RR:999-1001,809-814 | `push!` 1000 one-instance ranges between runs. | The range table grows by Resize steps; no recompile; no WaitFor. | perf-gate | RT |
| TL-33 | RR:834-843 | A range source is moved by a resize past its storage. | The records kernel reads the new address through the cell. | functional | RT |
| TL-34 | RR:1339-1341 | A frame with nothing pending. | No AS steps and no AccelCommand re-records. | perf-gate | RT |
| TL-35 | INT:1296-1311; DEC:199-201 | Graph S writes the source every run; graph F traces `t`. | After each S run an Update is pending on `t`. F applies it after S's write; F's transforms match S's latest. | concurrency | RT |
| TL-36 | INT:1305-1306 | A graph writes the source and also has `refit!(g,t)`. | `pending(t)` is empty after the run (no duplicate Update). | perf-gate | RT |
| TL-37 | RR:1317-1319 | Software TLAS with no instances; trace. | Miss, no out-of-bounds read. | functional | SW+VAL |
| TL-38 | RR:1093-1096 | Two TLASes over one BLAS; the BLAS is rebuilt. | Both TLASes get an Update. | functional | RT |
| TL-39 | RR:1056-1058 | `push!` 300 instances before the first submission. | One capacity step, 64 to 512, and one Build. | perf-gate | RT |
| TL-40 | RR:934-945,1072 | MoveAccel in the same submission as a plan's refit. | The refit is recorded again after the prefix with the new handle. | functional | RT+VAL |
| TL-41 | RR:845-851 | FI-P: a `push!` past capacity lands in the gap. | Again; the retry prepares the larger capacity. | concurrency | RT |
| TL-42 | RR:1229-1233 | A BLAS reachable only through `t`. | Resident; it is hit; no GPU fault. | functional | M5 |

## M. BLAS

| id | contract | scenario | expected | kind | device |
|---|---|---|---|---|---|
| BL-01 | RR:1090-1091; API:150 | Store into the vertices, same count. | BLAS Update and TLAS Update; no BLAS Build. | functional | RT |
| BL-02 | RR:1091-1092; VK-BC:259-268 | Store into the faces. | BLAS Build into its own storage and a TLAS Update; VUID-03768 clean. | functional | RT+VAL |
| BL-03 | RR:1089; VK-BC:164-171 | Resize vertices and faces within the BLAS capacity. | Build in place, no MoveAccel; TLAS Update; VUID-03764 clean. | functional | RT+VAL |
| BL-04 | RR:1089-1095; VK-AS:159-162 | Resize past the BLAS capacity. | MoveAccel; the new address goes through the cell; the TLAS Update (not a Build) re-references it; clean. | functional | RT+VAL |
| BL-05 | RR:1089-1095 | BL-04 with a trace in flight (FI-G). | The old BLAS is retired after both the trace's and the TLAS Update's tokens. | concurrency | RT+VAL |
| BL-06 | RR:1093-1095 | The BLAS move is applied by a submission that does not touch the TLAS; then a trace. | The trace's prefix applies the TLAS Update first; no stale BLAS address is traced. | concurrency | RT+VAL |
| BL-07 | RR:924-931; VK-AS:261-262 | BLAS over a kernel-written length (with capacity). | Built over the capacity, with zero-area (not NaN) primitives past the length. A length change is a valid Update (`maxVertex` kept). | functional | RT+VAL |
| BL-08 | RR:929-931; INT:1298 | A run writes the vertices; a run writes the faces. | Pends a BLAS Update; pends a BLAS Build. | functional | RT |
| BL-09 | RR:1095-1096,1437 | `blas!(scene,mesh)` twice. | The same BLAS object. | functional | RT |
| BL-10 | RR:1252-1256 | `free!(blas)` with a pending Build and no holders. | The Build is dropped; no submission. | leak | RT |
| BL-11 | RR:888-889 | Refit of a BLAS before any Build. | Applied as a Build. | functional | RT |
| BL-12 | VK-DC:28-35 (VUID-03673) | Geometry arrays used as build input after moves into Committed or Reserve storage. | They keep the build-input usage; validation clean. | functional | SP+RT+VAL |

## N. GPURef of an array

| id | contract | scenario | expected | kind | device |
|---|---|---|---|---|---|
| AB-01 | RR:1137-1157 | `r=GPURef(dev,a)`; a kernel over `r`. | The first submission touching `r` applies the Rebind; the kernel reads `a`. | functional | any |
| AB-02 | RR:1160-1164 | `GPURef(dev, view)`. | ArgumentError; the cell is retired; no finalizer registered. | error-path | any |
| AB-03 | API:180,488; RR:1139,1159 | `GPURef(dev, MantleArray(g,T,n))`. | Throws, rather than creating a ValueBox (A16). | error-path | any |
| AB-04 | RR:1134,1155 | `GPURef` of a freed array. | Throws; cell retired. | error-path | any |
| AB-05 | API:180; RR:1166-1171 | FI-G: a run reading `r` is in flight. `r[]=b` (`r[] === b` at once); next run; `free!(a)`. | The in-flight run reads `a`, the next run reads `b`. `a` is released only after the applying token. | concurrency | any |
| AB-06 | RR:1175-1178 | `r[]=r[]`. | No Rebind pended and no step. | perf-gate | any |
| AB-07 | RR:1166-1167,1184,1201-1202; API:488 | `r[]=b` with another element type or dimension count, a view, a transient, a freed or lent array; also after `free!(r)`. | Each throws; nothing pending; no hold on `b` leaked. | error-path | any |
| AB-08 | RR:1175-1176 | Threads call `r[]=b` and `r[]=c`, 10^4 times, with runs. | `r[]` is one of `b`, `c`. After the change is applied exactly one of them holds `r`; no leaks. | concurrency | any |
| AB-09 | RR:1222-1225 | Launches over `launchrange(r)` with targets of length 10 and 10^6. | Launch sizes follow the target; no recompile. | functional | any |
| AB-10 | RR:1188-1189; API:180 | A TLAS range over `r`; rebind with the same and with another length. | Update; then Build. | functional | RT |
| AB-11 | RR:225-226; API:389 | Store into, or resize, `r[]`'s target. | Update or Build on `t`, forwarded through `r`. | functional | RT |
| AB-12 | RR:1193-1197 | After the rebind has passed, store into the old target. | No TLAS work. | perf-gate | RT |
| AB-13 | RR:1112-1117 | A kernel writes through `r`; `r[]=b` between runs. | The run writes `b`; reader updates go to TLASes over `b`, not over `a`. | functional | RT |
| AB-14 | RR:714-721; DEC:202-204 | Indices drawn through `r` by handle; a rebind is applied. | The parts are re-recorded; no recompile; the draw uses `b`. | functional | any |
| AB-15 | RR:716-719 | FI-R: a rebind between compile and `register!`. | `register!` returns Again; recompile. | concurrency | any |
| AB-16 | RR:721-722 | A plan binds `a` directly; some other `r` also points at `a`; `r[]=b`. | The plan is not marked. | perf-gate | any |
| AB-17 | RR:716-718,723 | A plan binds through `r` by handle; `r[]=b`; `free!(a)`. | `a` stays allocated until the plan retires (A17); measure the bytes retained. | leak | any |
| AB-18 | RR:720-721 | A chain a→b→c with frees under a handle-bound plan. | GPU-AV clean. | concurrency | any+VAL |
| AB-19 | RR:1212-1219 | `free!(r)` with a Rebind pending. | The CellWrite is dropped; both targets' holds drop after `r`'s last tokens. | leak | any |
| AB-20 | RR:1151,1155 | FI: throw inside the GPURef constructor's `change!`. | The box is released once; no finalizer. | error-path | any |
| AB-21 | API:179; RR:147 | Attribute arrays of length 1 and of length n under one pipeline. | Uniform and per-element output; switching compiles nothing. | functional | any |
| AB-22 | RR:145 | `copy!(color[], fill(c,n))`. | The target is resized and stored; no Rebind; per-element colours. | functional | any |
| AB-23 | API:179,491 | Colour target emptied while positions has n > 0. | Defined behaviour (A21); no out-of-bounds read. | error-path | any+VAL |

## O. Textures

| id | contract | scenario | expected | kind | device |
|---|---|---|---|---|---|
| TX-01 | RR:1238-1244; API:160 | `resize!(atlas,2048,2048)` while a sampling run is in flight (FI-G); then the next run. | The next run samples a new index at 2048. The in-flight run reads the old index, which returns to the free list only after the applying token. | concurrency | any |
| TX-02 | RR:1239; API:160 | Upload of the same size and type. | Same index; written in place; no table write. | functional | any |
| TX-03 | API:160 | Upload of a new size. | A new index, as for `resize!`. | functional | any |
| TX-04 | RR:1239-1244 | Two resizes before any submission. | Final size correct; the intermediate image and index are released (A19). | leak | any |
| TX-05 | API:160 | 100 resizes across runs. | No recompiles. | perf-gate | any |
| TX-06 | RR:1240-1241 | Table write to a new index while command buffers using other indices are pending. | Validation clean. | functional | Vulkan+VAL |
| TX-07 | RR:1243-1244 | `free!(tex)` with a resize pending. | Both images and both indices released. | leak | any |
| TX-08 | RR:1422-1426 | `resize!` of a texture with contents. | Contents defined (A19). | functional | any |
| TX-09 | RR:1235-1237 | Metal argument buffer and residency set. | The new resource ID is resident; the old one is removed after its token. | functional | M5 |

## P. Hikari (D.2)

| id | contract | scenario | expected | kind | device |
|---|---|---|---|---|---|
| HK-01 | RR:1329-1334; DEC:244-254 | Between frames: `push!` a mesh, `delete!` an object, store a material colour. | The integrator plan compiles nothing; the changes are visible next frame. | perf-gate | RT |
| HK-02 | RR:1339-1341 | 100 frames with no change. | No AS steps. | perf-gate | RT |
| HK-03 | RR:1334-1336 | A scene change that recompiles (a new material type). | Film and sample-count contents preserved. | functional | RT |
| HK-04 | RR:1336-1338 | Window resize within capacity. | Window segments re-recorded; nothing declared; no compile (film size: A20). | functional | RT+WIN |
| HK-05 | RR:1341-1343 | A graph kernel writes transforms and the graph declares `refit!(g, scene.tlas)`. | Exactly one Update per frame. | perf-gate | RT |
| HK-06 | RR:1342-1343 | Particle instance count written by a kernel. | Built over the capacity once; then refits only. | perf-gate | RT |
| HK-07 | RR:1330-1331; INT:1296-1302 | A simulation graph writes positions; the frame graph traces. | The frame applies the Update after the simulation's write; the simulation waits for the frame. | concurrency | RT |
| HK-08 | RR:1346-1350 | `repeat!` around a trace whose queue length a kernel writes, until it reaches 0. | The loop ends; zero-count traces are valid. | functional | RT+VAL |
| HK-09 | RR:1317-1319,124-125 | Render an empty scene. | Every ray misses; no errors. | functional | RT, SW |

## Q. RayMakie (D.3)

| id | contract | scenario | expected | kind | device |
|---|---|---|---|---|---|
| RM-01 | RR:1376-1377; DEC:157-170 | `update!(plot; positions=host)`, same length. | Store into the owned array; no Rebind, compile, record or host wait. | perf-gate | any |
| RM-02 | RR:1376-1377 | Same with a new length. | Resize and Store; the draw count comes from the cell; no compile. | perf-gate | any |
| RM-03 | RR:1380-1381 | Colour per element, then one value, then per element again. | Lengths n, 1, n; no compile. | perf-gate | any |
| RM-04 | RR:1376-1379 | `positions = device array`, then `positions = host data`. | First a Rebind only; then a store into the owned array and a Rebind back to it. | functional | any |
| RM-05 | RR:1382; DEC:255-268 | `visible` false, then true. | A flag store; no re-record; count 0, then n. | perf-gate | any |
| RM-06 | RR:1384-1400; PLAN:1254-1257 | Insert and delete one plot in scenes of 100, 1000 and 5000 plots. | No plan compile; one piece compiled and recorded; insert time does not grow with plot count. | perf-gate | any |
| RM-07 | RR:1413-1420 | A mesh from the simulation's arrays; `run!(sim)` every frame. | Frame and simulation wait for each other; the screen is marked for redraw; no host work. | concurrency | any |
| RM-08 | RR:1419-1420; API:493 | The simulation resizes the faces past capacity. | The piece is re-recorded; no compile. | perf-gate | any |
| RM-09 | RR:1443-1444 | Ray-traced meshscatter: particles move, same count. | TLAS refit only; the BLAS is untouched. | perf-gate | RT |
| RM-10 | RR:1444-1446 | The count changes. | TLAS Build with the exact count; the same BLAS, not rebuilt. | functional | RT |
| RM-11 | RR:1447-1449 | Rebind with the same and with another length; hide the plot. | Update; Build; refit with mask 0. | functional | RT |
| RM-12 | RR:1433-1435 | Two meshscatters share a marker; delete one. | The BLAS survives; it is released after the last range is deleted and built. | leak | RT |
| RM-13 | RR:1371-1372; API:181 | Layout change. | One store moves every plot in the scene; no re-record. | perf-gate | any |
| RM-14 | API:491 | A plot with zero elements that then grows. | The render object is kept; draw count 0, then n; no compile. | functional | any |
| RM-15 | RR:1422-1426 | The glyph atlas grows. | No text or scatter piece re-recorded or recompiled; glyph pixels identical. | functional | any |
| RM-16 | DEC:205-207 | Toggle transparency. | A flag store; no compile. | perf-gate | any |
| RM-17 | RR:1364-1365 | Delete a plot, drop references, run GC. | Holds drop through the finalizer; memory back at baseline. | leak | any |

## R. Fuzz and steady-state gates

| id | contract | scenario | expected | kind | device |
|---|---|---|---|---|---|
| GT-01 | API:463; PLAN:1942-1945 | 4 threads run `push!`, `delete!`, `resize!`, rebinds, stores and plot updates against 3 graphs' runs, 10^5 operations. | Validation clean; reserved, mapped, uploads and cells back at baseline after `free!`; no deadlock. | leak | RT+VAL |
| GT-02 | API:455; PLAN:1891-1894 | Steady state with per-frame stores and a TLAS refit. | `run!` allocates nothing, compiles nothing and records nothing; each segment is submitted once. | perf-gate | RT |
| GT-03 | API:466 | Plans naming 100 to 50 000 resources, including through array GPURefs; nothing changes. | Host time per run does not grow with the count. | perf-gate | any |

## Contracts that are ambiguous or contradictory

These are ordered roughly by how much they matter.

**A1. Changes pended before a plan exists may never reach its first run.** `register!` adds to a new plan's inbox only resources that another plan names or that have eager accesses (RR:660-664, `shared` at RR:695). A change pended at the call reaches only the plans that name the resource at that moment (INT:1600-1602, 1610). A store on a fresh `MantleArray(dev,T,n)`, or a new TLAS's zero-instance Build (RR:124-125), therefore reaches no plan registered later. That breaks Rule 8 (PLAN:64-68). Tests: CR-08, TL-02.

**A2. Every run counts as "in a run", so runs throw instead of waiting.** `run!(g)` holds the graph lock with `g.owner = current_task()` (INT:996-998, 1009). A run that meets a change another plan holds calls `waitforruns` (INT:1151-1152), and `waitforruns` throws whenever `inrun(current_task())` is true (RR:449-451, 460). So any run throws, not only host functions. API:487 and PLAN:676-684 say other submitters wait and only code "from inside any run (any host function)" throws. Test: RN-07.

**A3. Upload region versus staging.** The B3.18 ruling (DEC:269-277; RR:249-254, 326-328) says the store's upload region is acquired at the call and the applying submission only copies from it. Several places still describe a staging region allocated by the applier:
- API:82: "a staging region that submission allocated".
- API:234: Store holds "inline bytes … or a staging region".
- PLAN:455: "a copy from staging memory".
- INT:1097-1098 and INT:1736: `prepare!` allocates "staging".

**A4. `copy!` does not follow B3.18.** RR:595 builds `bytes = copy(reinterpret(UInt8, vec(data)))` and passes it as `Store.upload`, which is typed `Region` (RR:261). There is no `acquireupload!` and no retire on a throw. Tests: CP-07, CP-03.

**A5. Upload leak in `setindex!`.** RR:566-567 acquire the region and convert the data with `copyto!` before the `try` at RR:569. A conversion that throws (InexactError, MethodError) leaks the region. Test: ST-04.

**A6. Work that happens at the call, against Rule 8 (PLAN:59-64) and B3.13 "never allocates" (DEC:213-214).**
- A store through a strided view is "a copy kernel (an eager call)" that runs at the call (RR:558-559).
- A texture regrowth writes the new index's descriptor "at once" (RR:1239-1242), which needs the new image to exist at the call.

**A7. When handle dependents are marked.** PLAN:456 says plans that recorded "a handle" are marked stale at the call. RR:625-630 and API:493 say handle dependents are re-recorded when the move is applied, with no stale flag.

**A8. Can one run see a mix of old and new state?** API:452: "every run sees the old or the new state, never a mix". INT:1321-1324 and PLAN:660-664: a change made while a host function runs is applied before the same run's next stage. Test: RN-18.

**A9. Status of readers and eviction.** RR:26-27 and API:150, 180, 389, 404 and 464 still call them "proposal, open decision 16". DEC:193-198 (B3.9, eviction automatic) and DEC:199-201 (B3.10, readers), and PLAN:1162, record them as ruled.

**A10. What a deleted range's slot contains.** RR:831 says "active false … inactive records" and API:156 agrees. But builds use the exact count (DEC:137-143), so it is unclear whether a deleted range's records still count, and with what length. The entry's source cell refers to an array whose hold drops after the build (RR:1036-1043), yet the offsets and records kernels read the source cell of every entry (RR:834-843). They could read a released or reused cell. `inactiveentry` is never defined. Tests: TL-09, TL-10.

**A11. A reserve can be smaller than the need.** `reservesize = min(nextpow(2,4need), reservelimit)` (RR:203). `MoveTo(ReservePlan(...))` at RR:194 and RR:196 has no fallback, so on lavapipe's 2 GiB space (RR:50-51) the reserve can be smaller than the need. Test: RZ-10.

**A12. `maxInstanceCount` has no contract.** VK-BC:278-283 (VUID-03801) and resources.adoc:8983-8986 (VUID-03785) bound the build count and the sizes query. Capacity doubling (RR:1056-1058) and builds over the capacity for kernel-written totals (RR:912-915) can exceed the limit. Nothing says what `push!` past it does. Test: TL-18.

**A13. Semantics of kernel-written lengths.** Undefined so far:
- what `length(a)` reports for such an array (RR:101-102, 213-214; API:86-88);
- what `resize!` or `copy!` does on it, given the constant capacity (RR:214);
- what happens when a kernel writes more than the capacity;
- which total the TLAS capacity is sized from for such a source (RR:845-851).

Tests: KL-05, KL-08.

**A14. `copy!` with the same length, and `resize!` to the current length.** `copy!` always pends a Resize and sets `resizable` (RR:593-606), so it stales direct plans (RR:625-627) even though Base treats it as a pure store. RayMakie's `setattribute!` calls `copy!` on every host update (RR:1376-1377). Tests: CP-08, RZ-02.

**A15. Device-to-device copies, scalar stores and colon stores.** The store signatures at RR:562 and RR:588 accept any `AbstractVector`, including a MantleArray. The body then calls `copyto!(hostview(up,T), vec(data))`, which would scalar-index a device array. API:93-94 lists `copyto!` as a launched method. `a[i]=v` and `a[:]=x` on device arrays have no method. Test: ST-15.

**A16. `GPURef` of a transient becomes a value box.** API:180 and API:488 say `GPURef(dev, transient)` throws. But the array method is restricted to `<:Device` (RR:1139), so a graph array falls to `GPURef(dev::Device, v) = ValueBox(dev, v)` (RR:1159). Test: AB-03.

**A17. A plan keeps its first handle target indefinitely.** A plan holds the target it resolved at compile until it retires (RR:716-718, 723), even after a rebind and re-record. RayMakie plans are not recompiled (DEC:255-268), so the first target stays allocated, against "no waste" (DEC:163-165). Test: AB-17.

**A18. Something can fail after pages are bound.** `bindnow!` binds pages (RR:393-397). Then `recordown` calls `current!` (RR:967-982), which takes command objects from the free list and can fail. PLAN:712-715 says nothing can fail between claiming tokens and submitting. `commit!` (RR:406) never ran, so `Reserve.mapped` is stale and the retry maps the same pages again (RR:417-420). Test: AP-05.

**A19. Texture resize details.** Not specified:
- whether `resize!` keeps the texels, which the glyph atlas needs (RR:1239-1244, 1422-1426);
- what happens to an intermediate index when two resizes come before any submission.

API:607-610 (G14) also still lists index retirement as open, while RR:1242-1244 decides it. Tests: TX-04, TX-08.

**A20. Hikari film size on window resize.** The film is a device array the screen holds (RR:1334-1338), and a resize within capacity "declares nothing". But the first `resize!` of a fixed array recompiles direct plans (RR:625-627). Test: HK-04.

**A21. Zero-length attributes.** `attribute(a, i) = a[min(i, length(a))]` (API:179; RR:147) reads index 0 when `length(a)==0`. Test: AB-23.

**A22. Views as TLAS range sources.** `push!` accepts any MantleArray as source (RR:818). Range entries name cells, which carry no view offset (RR:821-823). GPURef rejects views (RR:1160-1164); `push!` does not. Test: TL-26.

**A23. Deleting an unknown range.** `delete!` of a range that is not in the TLAS, or a second `delete!` of the same range, has no error contract (RR:1029-1031). Test: TL-27.

**A24. Zero-length stores.** `a[1:0] = T[]` acquires a 0-byte region and pends a Store (RR:562-579); this is unspecified. Test: ST-19.

**A25. Host reads through a freed handle.** API:79 lists eager calls, declarations and stores as throwing after `free!`, but not host reads. `Array(a)` has no `usable` check (INT:2148-2172). Test: FR-12.

**A26. A covering store can cross a held Resize.** A later store covering an earlier one replaces it, with no "consecutive" requirement (RR:283-284). A store pended before a Resize that a running plan holds can thus be replaced by one pended after it. The replacement inherits the hold (RR:345-347), so the earlier data never reaches stage 2. Compare RN-03.

**A27. Ray-traced meshes and Raycore temporaries.**
- A BLAS over an array GPURef is unspecified: `BLAS(dev, vertices::MantleArray, …)` (RR:1086-1096) versus RayMakie attributes being GPURefs (RR:1360-1365, 1413-1420).
- Raycore's temporaries depend on `SizedBy(t.records)` (RR:1304-1305), while `t.dependents` is described as "plans sized from its capacity" (RR:802). The records are also said never to stale plans (RR:812-814). Which change causes a WaitFor is unclear (RR:438).
- Building kernel-written totals over the capacity is still "listed for confirmation" (RR:932; PLAN:1117-1118) against B3.2's "no padding".
