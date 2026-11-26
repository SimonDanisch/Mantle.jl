// Mantle's WebGPU executor. The Julia backend decides everything: which buffers
// exist, which pipelines, what each submission contains. This module only turns
// those decisions into WebGPU calls in the page, and reports what the page saw
// (validation errors, shader compile messages, read back bytes).
//
// Binding model of every compute pipeline (see the Lava WebGPU target):
//   @group(0) @binding(0)       args: array<u32>, read-only storage, one dynamic
//                               offset per dispatch, the packed kernel arguments
//   @group(0) @binding(1 + k)   heap k: array<u32>, read_write storage, device memory

const devices = new Map();   // device id -> MantleDevice

class MantleDevice {
    constructor(adapter, device) {
        this.adapter = adapter;
        this.device = device;
        this.buffers = new Map();     // buffer id -> GPUBuffer
        this.pipelines = new Map();   // pipeline id -> GPUComputePipeline
        this.errors = [];
        this.lost = null;
        device.addEventListener("uncapturederror", (e) => this.errors.push(e.error.message));
        device.lost.then((info) => { this.lost = info.message || info.reason || "device lost"; });
        this.heaps = [];              // the heaps' GPUBuffers, heap 0 first (`createHeaps`)
        this.computeLayout = null;
        this.pipelineLayout = null;
        this.bindGroups = new Map();  // argument buffer id -> GPUBindGroup
        // The heap's exception word as the last submission left it, copied out at
        // its end, so `finish` reads it without a submission of its own; `null`
        // once read.
        this.status = null;
        this.recordings = new Map();  // recording id -> its commands
    }

    check() {
        if (this.lost !== null) throw new Error("WebGPU device lost: " + this.lost);
        if (this.errors.length > 0) {
            const msg = this.errors.join("\n");
            this.errors = [];
            throw new Error("WebGPU validation: " + msg);
        }
    }

    bindGroup(args) {
        let bg = this.bindGroups.get(args);
        if (bg === undefined) {
            // One argument block per dispatch: the binding covers the largest.
            const entries = [{ binding: 0, resource: { buffer: this.buffers.get(args), size: ARGS_WINDOW } }];
            this.heaps.forEach((h, k) => entries.push({ binding: 1 + k, resource: { buffer: h } }));
            bg = this.device.createBindGroup({ layout: this.computeLayout, entries });
            this.bindGroups.set(args, bg);
        }
        return bg;
    }
}

// The bytes a dispatch's argument binding covers; argument blocks are packed at
// 256-byte (minStorageBufferOffsetAlignment) steps and none is larger.
export const ARGS_WINDOW = 4096;

const LIMITS = ["maxStorageBufferBindingSize", "maxBufferSize", "maxStorageBuffersPerShaderStage",
    "maxComputeWorkgroupStorageSize", "maxComputeInvocationsPerWorkgroup",
    "maxComputeWorkgroupSizeX", "maxComputeWorkgroupSizeY", "maxComputeWorkgroupSizeZ",
    "maxComputeWorkgroupsPerDimension", "maxDynamicStorageBuffersPerPipelineLayout",
    "minStorageBufferOffsetAlignment", "maxBindGroups"];

// Apple adapters can advertise a maximum of 64 even when compute shaders run
// with 32 lanes. Probe a typical workgroup for kernels that specialize on the
// device's subgroup width.
async function probeSubgroupSize(device) {
    const n = Math.min(256, device.limits.maxComputeWorkgroupSizeX,
                            device.limits.maxComputeInvocationsPerWorkgroup);
    const code = `enable subgroups;
        @group(0) @binding(0) var<storage, read_write> result: u32;
        @compute @workgroup_size(${n})
        fn main(@builtin(local_invocation_index) i: u32, @builtin(subgroup_size) size: u32) {
            if (i == 0u) { result = size; }
        }`;
    const module = device.createShaderModule({ code });
    const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
    const result = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readback = device.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    try {
        const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
            entries: [{ binding: 0, resource: { buffer: result } }] });
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, group);
        pass.dispatchWorkgroups(1);
        pass.end();
        enc.copyBufferToBuffer(result, 0, readback, 0, 4);
        device.queue.submit([enc.finish()]);
        await readback.mapAsync(GPUMapMode.READ);
        const width = new Uint32Array(readback.getMappedRange())[0];
        if (width === 0) throw new Error("WebGPU subgroup size probe returned zero");
        return width;
    } finally {
        readback.destroy();
        result.destroy();
    }
}

// Open a device on the adapter matching `powerPreference`, with every feature
// in `wanted` the adapter has and the adapter's own limits. Returns what the
// Julia side needs to fill the device's capabilities.
export async function open(id, powerPreference, wanted) {
    if (!navigator.gpu) throw new Error("navigator.gpu is missing (no WebGPU, or not a secure context)");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference });
    if (!adapter) throw new Error("no WebGPU adapter for powerPreference " + powerPreference);
    const features = wanted.filter((f) => adapter.features.has(f));
    const requiredLimits = {};
    for (const k of LIMITS) requiredLimits[k] = adapter.limits[k];
    const device = await adapter.requestDevice({ requiredFeatures: features, requiredLimits });
    const info = adapter.info || {};
    const d = new MantleDevice(adapter, device);
    let subgroupSize = info.subgroupMaxSize || 0;
    if (info.vendor === "apple" && features.includes("subgroups")) {
        try {
            subgroupSize = await probeSubgroupSize(device);
            d.check();
        } catch (error) {
            device.destroy();
            throw error;
        }
    }
    devices.set(id, d);
    const limits = {};
    for (const k of LIMITS) limits[k] = device.limits[k];
    return {
        vendor: info.vendor || "", architecture: info.architecture || "",
        description: info.description || "",
        subgroupMinSize: info.subgroupMinSize || 0, subgroupMaxSize: info.subgroupMaxSize || 0, subgroupSize,
        features, limits,
    };
}

function dev(id) {
    const d = devices.get(id);
    if (d === undefined) throw new Error("no Mantle WebGPU device " + id);
    return d;
}

// The calls Julia does not wait on (buffer writes, submissions) report a failure
// at the next `finish` or `readBuffer`, which do wait.
function guarded(id, f) {
    try {
        f(dev(id));
    } catch (e) {
        const d = devices.get(id);
        if (d === undefined) throw e;
        d.errors.push(e.message);
    }
}

export function close(id) {
    const d = devices.get(id);
    if (d === undefined) return;
    for (const b of d.buffers.values()) b.destroy();
    d.device.destroy();
    devices.delete(id);
}

export function createBuffer(id, bufid, size) {
    guarded(id, (d) => {
        const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
        d.buffers.set(bufid, d.device.createBuffer({ size, usage }));
    });
}

// A buffer the host maps to read (`readMapped`), which copies write into.
export function createReadback(id, bufid, size) {
    guarded(id, (d) => d.buffers.set(bufid, d.device.createBuffer(
        { size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })));
}

// `size` bytes of a mappable buffer at `offset` (a multiple of 8): no submission.
// Resolves once the submissions writing it are done.
export async function readMapped(id, bufid, offset, size) {
    const d = dev(id);
    const b = d.buffers.get(bufid);
    await b.mapAsync(GPUMapMode.READ, offset, size);
    const bytes = new Uint8Array(b.getMappedRange(offset, size).slice(0));
    b.unmap();
    d.check();
    return bytes;
}

// The device's heaps, buffers `ids` of `size` bytes, and the pipeline layout that
// binds them all after the arguments. Before any pipeline.
export function createHeaps(id, ids, size) {
    guarded(id, (d) => {
        const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
        d.heaps = ids.map((b) => {
            const buf = d.device.createBuffer({ size, usage });
            d.buffers.set(b, buf);
            return buf;
        });
        const entries = [{ binding: 0, visibility: GPUShaderStage.COMPUTE,
                           buffer: { type: "read-only-storage", hasDynamicOffset: true } }];
        d.heaps.forEach((_, k) => entries.push({ binding: 1 + k, visibility: GPUShaderStage.COMPUTE,
                                                 buffer: { type: "storage" } }));
        d.computeLayout = d.device.createBindGroupLayout({ entries });
        d.pipelineLayout = d.device.createPipelineLayout({ bindGroupLayouts: [d.computeLayout] });
    });
}

export function destroyBuffer(id, bufid) {
    const d = dev(id);
    d.buffers.get(bufid).destroy();
    d.buffers.delete(bufid);
    d.bindGroups.delete(bufid);
}

// `data` arrives as a Uint8Array; WebGPU wants a multiple of 4 bytes.
export function writeBuffer(id, bufid, offset, data) {
    guarded(id, (d) => d.device.queue.writeBuffer(d.buffers.get(bufid), offset, data));
}

// `size` bytes of `bufid` at `offset`, after `commands` (the open batch, see
// `submit`), in one submission.
export async function readBuffer(id, bufid, offset, size, commands, argbuffer, argdata, synchronize = false) {
    const d = dev(id);
    const padded = (size + 3) & ~3;
    const out = d.device.createBuffer({ size: padded + (synchronize ? 4 : 0),
                                       usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = begin(d, argbuffer, argdata);
    const staging = [];
    try {
        encode(d, enc, commands, staging);
        enc.copyBufferToBuffer(d.buffers.get(bufid), offset, out, 0, padded);
        if (synchronize) {
            // Consume exceptions in queue order, even if later work is submitted
            // while the host is waiting for this readback to map.
            enc.copyBufferToBuffer(d.heaps[0], 0, out, padded, 4);
            enc.clearBuffer(d.heaps[0], 0, 4);
        }
        end(d, enc, staging, !synchronize);
        staging.length = 0;
        await out.mapAsync(GPUMapMode.READ);
        d.check();
        const mapped = out.getMappedRange();
        const bytes = new Uint8Array(mapped.slice(0, size));
        return synchronize ? { bytes, word: new Uint32Array(mapped, padded, 1)[0] } : bytes;
    } finally {
        for (const s of staging) s.destroy();
        out.destroy();
    }
}

// Compile `wgsl` and build its compute pipeline. Shader errors come back as the
// compiler's messages, with line and column, so Julia can show them next to the
// source it printed.
export async function createPipeline(id, pid, wgsl, entry) {
    const d = dev(id);
    d.device.pushErrorScope("validation");
    const module = d.device.createShaderModule({ code: wgsl });
    const info = await module.getCompilationInfo();
    const messages = info.messages.map((m) => ({ type: m.type, line: m.lineNum, column: m.linePos, message: m.message }));
    if (messages.some((m) => m.type === "error")) {
        await d.device.popErrorScope();
        return { ok: false, messages };
    }
    const pipeline = await d.device.createComputePipelineAsync({
        layout: d.pipelineLayout, compute: { module, entryPoint: entry },
    }).catch((e) => e);
    const scope = await d.device.popErrorScope();
    if (pipeline instanceof Error || scope !== null) {
        messages.push({ type: "error", line: 0, column: 0,
                        message: pipeline instanceof Error ? pipeline.message : scope.message });
        return { ok: false, messages };
    }
    d.pipelines.set(pid, pipeline);
    return { ok: true, messages };
}

// One submission of `commands`, a list of
//   { op: "dispatch", pipeline, groups: [x, y, z], args, argoffset }
//   { op: "copy", src, srcoffset, dst, dstoffset, size }
//   { op: "write", dst, offset, data }   (a Uint8Array of whole words)
// in order. `argdata` (a Uint8Array) is written to `argbuffer` first: the
// dispatches' argument blocks.
export function submit(id, commands, argbuffer, argdata) {
    guarded(id, (d) => {
        const enc = begin(d, argbuffer, argdata);
        const staging = [];
        encode(d, enc, commands, staging);
        end(d, enc, staging);
    });
}

function begin(d, argbuffer, argdata) {
    if (argdata.length > 0) d.device.queue.writeBuffer(d.buffers.get(argbuffer), 0, argdata);
    return d.device.createCommandEncoder();
}

// A write is a copy from a staging buffer, so it lands in command order: a
// `queue.writeBuffer` would land when called, ahead of the commands before it.
function encode(d, enc, commands, staging, profile = null) {
    let pass = null;
    for (const c of commands) {
        if (c.op === "dispatch") {
            if (pass === null) pass = enc.beginComputePass(profile?.descriptor(c));
            pass.setPipeline(d.pipelines.get(c.pipeline));
            pass.setBindGroup(0, d.bindGroup(c.args), [c.argoffset]);
            pass.dispatchWorkgroups(c.groups[0], c.groups[1], c.groups[2]);
            if (profile !== null) { pass.end(); pass = null; }
            continue;
        }
        if (pass !== null) { pass.end(); pass = null; }
        if (c.op === "copy") {
            enc.copyBufferToBuffer(d.buffers.get(c.src), c.srcoffset, d.buffers.get(c.dst), c.dstoffset, c.size);
        } else if (c.op === "write") {
            const n = c.data.length;
            const s = d.device.createBuffer({ size: n, usage: GPUBufferUsage.COPY_SRC, mappedAtCreation: true });
            new Uint8Array(s.getMappedRange()).set(c.data);
            s.unmap();
            enc.copyBufferToBuffer(s, 0, d.buffers.get(c.dst), c.offset, n);
            staging.push(s);
        } else {
            throw new Error("unknown command " + c.op);
        }
    }
    if (pass !== null) pass.end();
}

// Submit, copying the exception word (heap 0, word 0) out for `finish` at the end.
function end(d, enc, staging, captureStatus = true) {
    const status = captureStatus ? d.device.createBuffer(
        { size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }) : null;
    if (status !== null) enc.copyBufferToBuffer(d.heaps[0], 0, status, 0, 4);
    d.device.queue.submit([enc.finish()]);
    // Destroyed buffers live until the submissions using them are done.
    for (const s of staging) s.destroy();
    // An unread earlier status is covered: the word stays raised until `finish` clears it.
    if (d.status !== null) d.status.destroy();
    d.status = status;
}

// A plan's recording: commands `replay` encodes into a command buffer each run.
export function record(id, recid, commands) {
    guarded(id, (d) => d.recordings.set(recid, commands));
}

export function forget(id, recid) {
    const d = devices.get(id);
    if (d !== undefined) d.recordings.delete(recid);
}


// One submission: `commands` (the open batch, see `submit`), then recordings in
// order, whose arguments are in the buffers their commands name.
export function replay(id, recids, commands, argbuffer, argdata) {
    guarded(id, (d) => {
        const enc = begin(d, argbuffer, argdata);
        const staging = [];
        encode(d, enc, commands, staging);
        for (const r of recids) {
            const c = d.recordings.get(r);
            if (c === undefined) throw new Error("no recording " + r);
            encode(d, enc, c, staging);
        }
        end(d, enc, staging);
    });
}

// The open batch is submitted as usual. Give each recorded dispatch its own
// compute pass so WebGPU can write a pair of timestamps around it.
export async function profileReplay(id, recids, commands, argbuffer, argdata) {
    const d = dev(id);
    const recordings = recids.map((r) => {
        const c = d.recordings.get(r);
        if (c === undefined) throw new Error("no recording " + r);
        return c;
    });
    const count = recordings.reduce((n, recording) =>
        n + recording.filter((c) => c.op === "dispatch").length, 0);
    const slotsPerSet = 4096;
    const total = 2 * count;
    const staging = [], sets = [], dispatches = [];
    let resolve, out;
    try {
        const enc = begin(d, argbuffer, argdata);
        encode(d, enc, commands, staging);
        for (let slot = 0; slot < total; slot += slotsPerSet) {
            sets.push(d.device.createQuerySet({ type: "timestamp", count: Math.min(total - slot, slotsPerSet) }));
        }
        const profile = {
            descriptor(c) {
                const slot = 2 * dispatches.length;
                dispatches.push(c);
                const begin = slot % slotsPerSet;
                return { timestampWrites: { querySet: sets[Math.floor(slot / slotsPerSet)],
                                            beginningOfPassWriteIndex: begin, endOfPassWriteIndex: begin + 1 } };
            },
        };
        for (const c of recordings) encode(d, enc, c, staging, profile);
        if (total > 0) {
            const size = total * 8;
            resolve = d.device.createBuffer({ size, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
            out = d.device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
            sets.forEach((set, i) => enc.resolveQuerySet(set, 0, set.count, resolve, i * slotsPerSet * 8));
            enc.copyBufferToBuffer(resolve, 0, out, 0, size);
        }
        end(d, enc, staging);
        staging.length = 0;
        if (total === 0) return [];
        await out.mapAsync(GPUMapMode.READ);
        d.check();
        const ticks = new BigUint64Array(out.getMappedRange());
        return dispatches.map((c, i) => ({ name: c.name, pipeline: c.pipeline,
            ns: ticks[2 * i + 1] >= ticks[2 * i] ? Number(ticks[2 * i + 1] - ticks[2 * i]) : null }));
    } finally {
        for (const s of staging) s.destroy();
        for (const set of sets) set.destroy();
        resolve?.destroy();
        out?.destroy();
    }
}

// Resolves when everything submitted so far has finished on the GPU, to the
// heap's exception word (word 0, raised by a kernel that threw) as the last
// submission left it, which it clears. Submits nothing: the word was copied out
// by that submission.
export async function finish(id) {
    const d = dev(id);
    await d.device.queue.onSubmittedWorkDone();
    d.check();
    const status = d.status;
    if (status === null) return 0;
    d.status = null;
    await status.mapAsync(GPUMapMode.READ);
    const word = new Uint32Array(status.getMappedRange())[0];
    status.unmap();
    status.destroy();
    if (word !== 0) d.device.queue.writeBuffer(d.heaps[0], 0, new Uint32Array([0]));
    return word;
}
