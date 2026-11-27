// deno-fmt-ignore-file
// deno-lint-ignore-file
// This code was bundled using `deno bundle` and it's not recommended to edit it manually

const devices = new Map();
let nextDeviceId = 0;
let idleDevice = null;
const MAX_PIPELINE_CACHE_BYTES = 32 * 1024 * 1024;
class BrowserDevice {
    constructor(key, adapter, device){
        this.key = key;
        this.adapter = adapter;
        this.device = device;
        this.layouts = new Map();
        this.pipelines = new Map();
        this.cacheBytes = 0;
        this.lost = null;
        this.subgroupSize = undefined;
        this.probedSubgroups = false;
        device.lost.then((info)=>{
            this.lost = info.message || info.reason || "device lost";
            if (idleDevice === this) idleDevice = null;
        });
    }
    remember(key, entry) {
        if (entry.bytes > MAX_PIPELINE_CACHE_BYTES) return;
        this.pipelines.set(key, entry);
        this.cacheBytes += entry.bytes;
        while(this.cacheBytes > MAX_PIPELINE_CACHE_BYTES || this.pipelines.size > 256){
            const oldest = this.pipelines.keys().next().value;
            this.cacheBytes -= this.pipelines.get(oldest).bytes;
            this.pipelines.delete(oldest);
        }
    }
}
class MantleDevice {
    constructor(browserDevice){
        this.browserDevice = browserDevice;
        const device = this.device = browserDevice.device;
        this.buffers = new Map();
        this.pipelines = new Map();
        this.renderPipelines = new Map();
        this.textures = new Map();
        this.samplers = new Map();
        this.mipPipelines = new Map();
        this.canvases = new Map();
        this.renderLayout = null;
        this.errors = [];
        this.onError = (e)=>this.errors.push(e.error.message);
        device.addEventListener("uncapturederror", this.onError);
        this.heaps = [];
        this.computeLayout = null;
        this.pipelineLayout = null;
        this.bindGroups = new Map();
        this.gatePipeline = null;
        this.countGatePipeline = null;
        this.gateGrid = null;
        this.gates = new Map();
        this.status = null;
        this.recordings = new Map();
        this.pendingPipelines = 0;
    }
    check() {
        if (this.browserDevice.lost !== null) throw new Error("WebGPU device lost: " + this.browserDevice.lost);
        if (this.errors.length > 0) {
            const msg = this.errors.join("\n");
            this.errors = [];
            throw new Error("WebGPU validation: " + msg);
        }
    }
    bindGroup(args) {
        let bg = this.bindGroups.get(args);
        if (bg === undefined) {
            const entries = [
                {
                    binding: 0,
                    resource: {
                        buffer: this.buffers.get(args),
                        size: ARGS_WINDOW
                    }
                }
            ];
            this.heaps.forEach((h, k)=>entries.push({
                    binding: 1 + k,
                    resource: {
                        buffer: h
                    }
                }));
            bg = this.device.createBindGroup({
                layout: this.computeLayout,
                entries
            });
            this.bindGroups.set(args, bg);
        }
        return bg;
    }
}
const ARGS_WINDOW = 4096;
const LIMITS = [
    "maxStorageBufferBindingSize",
    "maxBufferSize",
    "maxStorageBuffersPerShaderStage",
    "maxComputeWorkgroupStorageSize",
    "maxComputeInvocationsPerWorkgroup",
    "maxComputeWorkgroupSizeX",
    "maxComputeWorkgroupSizeY",
    "maxComputeWorkgroupSizeZ",
    "maxComputeWorkgroupsPerDimension",
    "maxDynamicStorageBuffersPerPipelineLayout",
    "minStorageBufferOffsetAlignment",
    "maxBindGroups"
];
async function probeSubgroupSize(device) {
    const n = Math.min(256, device.limits.maxComputeWorkgroupSizeX, device.limits.maxComputeInvocationsPerWorkgroup);
    const code = `enable subgroups;
        @group(0) @binding(0) var<storage, read_write> result: u32;
        @compute @workgroup_size(${n})
        fn main(@builtin(local_invocation_index) i: u32, @builtin(subgroup_size) size: u32) {
            if (i == 0u) { result = size; }
        }`;
    const module = device.createShaderModule({
        code
    });
    const pipeline = await device.createComputePipelineAsync({
        layout: "auto",
        compute: {
            module,
            entryPoint: "main"
        }
    });
    const result = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });
    const readback = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });
    try {
        const group = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: result
                    }
                }
            ]
        });
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, group);
        pass.dispatchWorkgroups(1);
        pass.end();
        enc.copyBufferToBuffer(result, 0, readback, 0, 4);
        device.queue.submit([
            enc.finish()
        ]);
        await readback.mapAsync(GPUMapMode.READ);
        const width = new Uint32Array(readback.getMappedRange())[0];
        if (width === 0) throw new Error("WebGPU subgroup size probe returned zero");
        return width;
    } finally{
        readback.destroy();
        result.destroy();
    }
}
async function open(powerPreference, wanted, maxCanvasScale) {
    if (!navigator.gpu) {
        if (!globalThis.isSecureContext) {
            throw new Error("WebGPU requires a secure page context. Open the result over HTTPS.");
        }
        throw new Error("This browser does not expose WebGPU on this secure page (navigator.gpu is missing).");
    }
    const key = JSON.stringify([
        powerPreference,
        wanted
    ]);
    let browserDevice = idleDevice;
    idleDevice = null;
    if (browserDevice?.key !== key || browserDevice.lost !== null) {
        browserDevice?.device.destroy();
        const adapter = await navigator.gpu.requestAdapter({
            powerPreference
        });
        if (!adapter) throw new Error("no WebGPU adapter for powerPreference " + powerPreference);
        const requiredFeatures = wanted.filter((f)=>adapter.features.has(f));
        const requiredLimits = {};
        for (const k of LIMITS)requiredLimits[k] = adapter.limits[k];
        const device = await adapter.requestDevice({
            requiredFeatures,
            requiredLimits
        });
        browserDevice = new BrowserDevice(key, adapter, device);
    }
    const { adapter , device  } = browserDevice;
    const features = Array.from(device.features);
    const info = adapter.info || {};
    const d = new MantleDevice(browserDevice);
    d.maxCanvasScale = maxCanvasScale ?? Infinity;
    if (browserDevice.subgroupSize === undefined) {
        browserDevice.subgroupSize = info.subgroupMaxSize || 0;
    }
    if (info.vendor === "apple" && features.includes("subgroups") && !browserDevice.probedSubgroups) {
        try {
            browserDevice.subgroupSize = await probeSubgroupSize(device);
            d.check();
            browserDevice.probedSubgroups = true;
        } catch (error) {
            device.destroy();
            throw error;
        }
    }
    const id = ++nextDeviceId;
    devices.set(id, d);
    const limits = {};
    for (const k of LIMITS)limits[k] = device.limits[k];
    return {
        id,
        vendor: info.vendor || "",
        architecture: info.architecture || "",
        description: info.description || "",
        subgroupMinSize: info.subgroupMinSize || 0,
        subgroupMaxSize: info.subgroupMaxSize || 0,
        subgroupSize: browserDevice.subgroupSize,
        features,
        limits,
        pipelineKeys: [
            ...browserDevice.pipelines.keys()
        ],
        countedGates: true
    };
}
function dev(id) {
    const d = devices.get(id);
    if (d === undefined) throw new Error("no Mantle WebGPU device " + id);
    return d;
}
function guarded(id, f) {
    try {
        f(dev(id));
    } catch (e) {
        const d = devices.get(id);
        if (d === undefined) throw e;
        d.errors.push(e.message);
    }
}
function close(id) {
    const d = devices.get(id);
    if (d === undefined) return;
    for (const b of d.buffers.values())b.destroy();
    for (const t of d.textures.values())t.destroy();
    d.gateGrid?.destroy();
    for (const gate of d.gates.values())gate.params.destroy();
    for (const surface of d.canvases.values())closeCanvas(surface);
    d.device.removeEventListener("uncapturederror", d.onError);
    devices.delete(id);
    const browserDevice = d.browserDevice;
    if (d.pendingPipelines === 0 && browserDevice.lost === null && browserDevice.pipelines.size > 0) {
        idleDevice?.device.destroy();
        idleDevice = browserDevice;
    } else {
        d.device.destroy();
    }
}
function createBuffer(id, bufid, size) {
    guarded(id, (d)=>{
        const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
        d.buffers.set(bufid, d.device.createBuffer({
            size,
            usage
        }));
    });
}
function createReadback(id, bufid, size) {
    guarded(id, (d)=>d.buffers.set(bufid, d.device.createBuffer({
            size,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        })));
}
async function readMapped(id, bufid, offset, size) {
    const d = dev(id);
    const b = d.buffers.get(bufid);
    await b.mapAsync(GPUMapMode.READ, offset, size);
    const bytes = new Uint8Array(b.getMappedRange(offset, size).slice(0));
    b.unmap();
    d.check();
    return bytes;
}
function createHeaps(id, ids, size) {
    guarded(id, (d)=>{
        const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.INDEX | GPUBufferUsage.INDIRECT;
        d.heaps = ids.map((b)=>{
            const buf = d.device.createBuffer({
                size,
                usage
            });
            d.buffers.set(b, buf);
            return buf;
        });
        const layouts = d.browserDevice.layouts;
        let layout = layouts.get(ids.length);
        if (!layout) {
            const entries = [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "read-only-storage",
                        hasDynamicOffset: true
                    }
                }
            ];
            d.heaps.forEach((_, k)=>entries.push({
                    binding: 1 + k,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage"
                    }
                }));
            const bindGroup = d.device.createBindGroupLayout({
                entries
            });
            const pipeline = d.device.createPipelineLayout({
                bindGroupLayouts: [
                    bindGroup
                ]
            });
            layout = {
                bindGroup,
                pipeline
            };
            layouts.set(ids.length, layout);
        }
        d.computeLayout = layout.bindGroup;
        d.pipelineLayout = layout.pipeline;
    });
}
function destroyBuffer(id, bufid) {
    const d = dev(id);
    d.buffers.get(bufid).destroy();
    d.buffers.delete(bufid);
    d.bindGroups.delete(bufid);
}
function writeBuffer(id, bufid, offset, data) {
    guarded(id, (d)=>d.device.queue.writeBuffer(d.buffers.get(bufid), offset, data));
}
async function readBuffer(id, bufid, offset, size, commands, argbuffer, argdata, synchronize = false) {
    const d = dev(id);
    const padded = size + 3 & ~3;
    const out = d.device.createBuffer({
        size: padded + (synchronize ? 4 : 0),
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });
    const enc = begin(d, argbuffer, argdata);
    const staging = [];
    try {
        encode(d, enc, commands, staging);
        enc.copyBufferToBuffer(d.buffers.get(bufid), offset, out, 0, padded);
        if (synchronize) {
            enc.copyBufferToBuffer(d.heaps[0], 0, out, padded, 4);
            enc.clearBuffer(d.heaps[0], 0, 4);
        }
        end(d, enc, staging, !synchronize);
        staging.length = 0;
        await out.mapAsync(GPUMapMode.READ);
        d.check();
        const mapped = out.getMappedRange();
        const bytes = new Uint8Array(mapped.slice(0, size));
        return synchronize ? {
            bytes,
            word: new Uint32Array(mapped, padded, 1)[0]
        } : bytes;
    } finally{
        for (const s of staging)s.destroy();
        out.destroy();
    }
}
async function withPendingPipeline(d, build) {
    d.pendingPipelines++;
    try {
        return await build();
    } finally{
        d.pendingPipelines--;
    }
}
function createPipeline(id, pid, source, entry, digest = null) {
    const d = dev(id);
    return withPendingPipeline(d, ()=>buildComputePipeline(d, pid, source, entry, digest));
}
function createComputePipelines(id, requests) {
    const d = dev(id);
    return withPendingPipeline(d, async ()=>{
        const results = [];
        const build = ([pid, digest, source])=>buildComputePipeline(d, pid, source, "main", null, digest);
        for(let i = 0; i < requests.length;){
            const first = requests[i++];
            const second = requests[i];
            if (first[2]?.length < 65536 && second?.[2]?.length < 65536) {
                i++;
                results.push(...await Promise.all([
                    build(first),
                    build(second)
                ]));
            } else {
                results.push(await build(first));
            }
        }
        return results;
    });
}
async function shaderDigest(bytes) {
    const hash = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
    return Array.from(hash, (b)=>b.toString(16).padStart(2, "0")).join("");
}
async function buildComputePipeline(d, pid, source, entry, digest, inlineDigest = null) {
    const signature = digest ?? inlineDigest ?? await shaderDigest(new TextEncoder().encode(source));
    const key = `${d.heaps.length}:${entry}:${signature}`;
    const cache = d.browserDevice.pipelines;
    const hit = cache.get(key);
    if (hit) {
        cache.delete(key);
        cache.set(key, hit);
        d.pipelines.set(pid, hit.pipeline);
        return {
            ok: true,
            messages: [],
            cached: true
        };
    }
    if (source === null) return {
        missing: true
    };
    let wgsl = source;
    let fetchMs = 0;
    if (digest !== null) {
        const started = performance.now();
        try {
            const response = await fetch(source, {
                cache: "force-cache"
            });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const bytes = await response.arrayBuffer();
            if (await shaderDigest(bytes) !== digest) throw new Error("response did not match the shader");
            wgsl = new TextDecoder("utf-8", {
                fatal: true
            }).decode(bytes);
        } catch (e) {
            return {
                sourceError: String(e.message || e)
            };
        }
        fetchMs = performance.now() - started;
    }
    const started = performance.now();
    d.device.pushErrorScope("validation");
    let module, pending, pipeline, failure;
    let moduleMs;
    try {
        module = d.device.createShaderModule({
            code: wgsl
        });
        moduleMs = performance.now() - started;
        pending = d.device.createComputePipelineAsync({
            layout: d.pipelineLayout,
            compute: {
                module,
                entryPoint: entry
            }
        });
    } catch (e) {
        failure = e;
    }
    const validation = d.device.popErrorScope();
    try {
        pipeline = await pending;
    } catch (e) {
        failure = e;
    }
    const scope = await validation;
    if (!pipeline || scope) {
        const info = module && await module.getCompilationInfo();
        const messages = info ? info.messages.map((m)=>({
                type: m.type,
                line: m.lineNum,
                column: m.linePos,
                message: m.message
            })) : [];
        for (const error of [
            failure,
            scope
        ]){
            if (error) messages.push({
                type: "error",
                line: 0,
                column: 0,
                message: String(error.message || error)
            });
        }
        return {
            ok: false,
            messages
        };
    }
    d.pipelines.set(pid, pipeline);
    if (!cache.has(key)) d.browserDevice.remember(key, {
        pipeline,
        bytes: wgsl.length
    });
    return {
        ok: true,
        messages: [],
        cached: false,
        fetchMs,
        moduleMs,
        pipelineMs: performance.now() - started - moduleMs
    };
}
function createRenderPipeline(id, ...args) {
    const d = dev(id);
    return withPendingPipeline(d, ()=>buildRenderPipeline(d, ...args));
}
async function buildRenderPipeline(d, pid, vertex, fragment, targets, depth, primitive, fragBinding, textures) {
    if (d.renderLayout === null) {
        const buffer = {
            type: "read-only-storage",
            hasDynamicOffset: true
        };
        const entries = [
            {
                binding: 0,
                visibility: GPUShaderStage.VERTEX,
                buffer
            },
            {
                binding: fragBinding,
                visibility: GPUShaderStage.FRAGMENT,
                buffer
            }
        ];
        const stages = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT;
        d.heaps.forEach((_, k)=>entries.push({
                binding: 1 + k,
                visibility: stages,
                buffer: {
                    type: "read-only-storage"
                }
            }));
        d.renderLayout = d.device.createBindGroupLayout({
            entries
        });
    }
    const layouts = [
        d.renderLayout
    ];
    if (textures.length > 0) {
        const entries = textures.flatMap((sampleType, i)=>[
                {
                    binding: 2 * i,
                    visibility: GPUShaderStage.FRAGMENT,
                    texture: {
                        sampleType
                    }
                },
                {
                    binding: 2 * i + 1,
                    visibility: GPUShaderStage.FRAGMENT,
                    sampler: {
                        type: sampleType === "float" ? "filtering" : "non-filtering"
                    }
                }
            ]);
        layouts.push(d.device.createBindGroupLayout({
            entries
        }));
    }
    const layout = d.device.createPipelineLayout({
        bindGroupLayouts: layouts
    });
    const modules = [
        d.device.createShaderModule({
            code: vertex
        })
    ];
    if (fragment !== null) modules.push(d.device.createShaderModule({
        code: fragment
    }));
    const info = await Promise.all(modules.map((m)=>m.getCompilationInfo()));
    const messages = info.flatMap((stage, i)=>stage.messages.map((m)=>({
                stage: i === 0 ? "vertex" : "fragment",
                type: m.type,
                line: m.lineNum,
                column: m.linePos,
                message: m.message
            })));
    if (messages.some((m)=>m.type === "error")) return {
        ok: false,
        messages
    };
    const descriptor = {
        layout,
        vertex: {
            module: modules[0],
            entryPoint: "main"
        },
        primitive
    };
    if (fragment !== null) descriptor.fragment = {
        module: modules[1],
        entryPoint: "main",
        targets
    };
    if (depth !== null) descriptor.depthStencil = depth;
    try {
        d.renderPipelines.set(pid, await d.device.createRenderPipelineAsync(descriptor));
    } catch (e) {
        messages.push({
            stage: "pipeline",
            type: "error",
            line: 0,
            column: 0,
            message: e.message
        });
        return {
            ok: false,
            messages
        };
    }
    return {
        ok: true,
        messages
    };
}
function createAttachment(id, tid, size, format) {
    guarded(id, (d)=>{
        const usage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC;
        d.textures.set(tid, d.device.createTexture({
            size,
            format,
            usage
        }));
    });
}
function canvasScale(id) {
    return Math.min(window.devicePixelRatio || 1, dev(id).maxCanvasScale);
}
function createCanvas(id, cid, width, height, scale, format, title, events) {
    const d = dev(id);
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    canvas.tabIndex = 0;
    canvas.setAttribute("aria-label", title || "RayMakie");
    Object.assign(canvas.style, {
        display: "block",
        width: "100%",
        height: "100%",
        touchAction: "none",
        outline: "none"
    });
    const owner = document.createElement("div");
    Object.assign(owner.style, {
        position: "relative",
        width: `min(${width / scale}px, 100vw)`,
        height: `min(${height / scale}px, 100vh)`
    });
    owner.append(canvas);
    document.body.append(owner);
    const context = canvas.getContext("webgpu");
    try {
        context.configure({
            device: d.device,
            format: format.replace("-srgb", ""),
            usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
            alphaMode: "premultiplied"
        });
        const reportSize = ()=>events.notify([
                "resize",
                canvas.clientWidth,
                canvas.clientHeight,
                canvasScale(id)
            ]);
        const observer = new ResizeObserver(reportSize);
        observer.observe(owner);
        const listeners = new AbortController();
        window.addEventListener("resize", reportSize, {
            signal: listeners.signal
        });
        const on = (event, fn)=>canvas.addEventListener(event, fn, {
                signal: listeners.signal
            });
        const mouse = (e)=>{
            const rect = canvas.getBoundingClientRect();
            events.notify([
                "pointer",
                e.clientX - rect.left,
                rect.bottom - e.clientY
            ]);
        };
        on("pointermove", mouse);
        on("pointerdown", (e)=>{
            canvas.focus();
            canvas.setPointerCapture(e.pointerId);
            mouse(e);
            events.notify([
                "button",
                e.button,
                true
            ]);
        });
        on("pointerup", (e)=>{
            mouse(e);
            events.notify([
                "button",
                e.button,
                false
            ]);
        });
        on("pointerenter", ()=>events.notify([
                "entered",
                true
            ]));
        on("pointerleave", ()=>events.notify([
                "entered",
                false
            ]));
        on("focus", ()=>events.notify([
                "focus",
                true
            ]));
        on("blur", ()=>events.notify([
                "focus",
                false
            ]));
        on("wheel", (e)=>{
            e.preventDefault();
            const factor = e.deltaMode === 0 ? 1 / 100 : e.deltaMode === 1 ? 1 : 8;
            events.notify([
                "scroll",
                -e.deltaX * factor,
                -e.deltaY * factor
            ]);
        });
        on("contextmenu", (e)=>e.preventDefault());
        on("keydown", (e)=>{
            if (e.key === "Tab") return;
            e.preventDefault();
            events.notify([
                "key",
                e.code,
                e.key,
                e.repeat ? 2 : 1,
                e.metaKey || e.ctrlKey || e.altKey
            ]);
        });
        on("keyup", (e)=>events.notify([
                "key",
                e.code,
                e.key,
                0,
                true
            ]));
        d.canvases.set(cid, {
            canvas,
            owner,
            context,
            observer,
            listeners
        });
        return true;
    } catch (e) {
        owner.remove();
        throw e;
    }
}
function resizeCanvas(id, cid, width, height) {
    guarded(id, (d)=>{
        const canvas = d.canvases.get(cid).canvas;
        canvas.width = width;
        canvas.height = height;
    });
}
function closeCanvas(surface) {
    surface.observer.disconnect();
    surface.listeners.abort();
    surface.context.unconfigure();
    surface.owner.remove();
}
function destroyCanvas(id, cid) {
    guarded(id, (d)=>{
        const surface = d.canvases.get(cid);
        if (surface === undefined) return;
        closeCanvas(surface);
        d.canvases.delete(cid);
    });
}
async function canvasFrame() {
    await Promise.race([
        new Promise((resolve)=>requestAnimationFrame(resolve)),
        new Promise((resolve)=>setTimeout(resolve, 50))
    ]);
    return true;
}
function createTexture(id, tid, size, format, mipLevelCount) {
    guarded(id, (d)=>{
        const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | (mipLevelCount > 1 ? GPUTextureUsage.RENDER_ATTACHMENT : 0);
        d.textures.set(tid, d.device.createTexture({
            size,
            format,
            mipLevelCount,
            usage
        }));
    });
}
function uploadTexture(id, tid, data, rowBytes) {
    guarded(id, (d)=>{
        const texture = d.textures.get(tid);
        d.device.queue.writeTexture({
            texture
        }, data, {
            bytesPerRow: rowBytes
        }, [
            texture.width,
            texture.height
        ]);
        if (texture.mipLevelCount > 1) buildMipmaps(d, texture);
    });
}
function buildMipmaps(d, texture) {
    let pipeline = d.mipPipelines.get(texture.format);
    if (pipeline === undefined) {
        const code = `
            @group(0) @binding(0) var src: texture_2d<f32>;
            @vertex fn vertex(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
                return vec4f(f32(i & 1u) * 4.0 - 1.0, f32(i >> 1u) * 4.0 - 1.0, 0.0, 1.0);
            }
            @fragment fn fragment(@builtin(position) p: vec4f) -> @location(0) vec4f {
                let extent = vec2i(textureDimensions(src)) - 1;
                let a = vec2i(p.xy) * 2;
                return 0.25 * (textureLoad(src, a, 0) +
                    textureLoad(src, min(a + vec2i(1, 0), extent), 0) +
                    textureLoad(src, min(a + vec2i(0, 1), extent), 0) +
                    textureLoad(src, min(a + vec2i(1, 1), extent), 0));
            }`;
        const module = d.device.createShaderModule({
            code
        });
        const group = d.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.FRAGMENT,
                    texture: {
                        sampleType: "unfilterable-float"
                    }
                }
            ]
        });
        const layout = d.device.createPipelineLayout({
            bindGroupLayouts: [
                group
            ]
        });
        pipeline = d.device.createRenderPipeline({
            layout,
            vertex: {
                module,
                entryPoint: "vertex"
            },
            fragment: {
                module,
                entryPoint: "fragment",
                targets: [
                    {
                        format: texture.format
                    }
                ]
            }
        });
        d.mipPipelines.set(texture.format, pipeline);
    }
    const enc = d.device.createCommandEncoder();
    for(let level = 1; level < texture.mipLevelCount; level++){
        const src = texture.createView({
            baseMipLevel: level - 1,
            mipLevelCount: 1
        });
        const dst = texture.createView({
            baseMipLevel: level,
            mipLevelCount: 1
        });
        const bg = d.device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: src
                }
            ]
        });
        const pass = enc.beginRenderPass({
            colorAttachments: [
                {
                    view: dst,
                    loadOp: "clear",
                    storeOp: "store"
                }
            ]
        });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bg);
        pass.draw(3);
        pass.end();
    }
    d.device.queue.submit([
        enc.finish()
    ]);
}
function getSampler(d, descriptor) {
    const key = JSON.stringify(descriptor);
    if (!d.samplers.has(key)) d.samplers.set(key, d.device.createSampler(descriptor));
    return d.samplers.get(key);
}
async function readAttachment(id, tid, rowBytes, commands, argbuffer, argdata) {
    const d = dev(id);
    const texture = d.textures.get(tid);
    const stride = Math.ceil(rowBytes / 256) * 256;
    const height = texture.height;
    const out = d.device.createBuffer({
        size: stride * height,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });
    const staging = [];
    try {
        const enc = begin(d, argbuffer, argdata);
        encode(d, enc, commands, staging);
        const aspect = texture.format === "depth32float" ? "depth-only" : "all";
        enc.copyTextureToBuffer({
            texture,
            aspect
        }, {
            buffer: out,
            bytesPerRow: stride
        }, [
            texture.width,
            height
        ]);
        end(d, enc, staging);
        staging.length = 0;
        await out.mapAsync(GPUMapMode.READ);
        d.check();
        const data = new Uint8Array(out.getMappedRange());
        const packed = new Uint8Array(rowBytes * height);
        for(let y = 0; y < height; y++){
            packed.set(data.subarray(y * stride, y * stride + rowBytes), y * rowBytes);
        }
        return packed;
    } finally{
        for (const b of staging)b.destroy();
        out.destroy();
    }
}
function encodeImageCopy(d, enc, command, staging) {
    const texture = d.textures.get(command.src);
    const [x, y, w, h] = command.region;
    const row = command.rowBytes;
    const out = d.buffers.get(command.dst);
    const aspect = texture.format === "depth32float" ? "depth-only" : "all";
    const source = {
        texture,
        aspect,
        origin: [
            x,
            y,
            0
        ]
    };
    if (row % 256 === 0) {
        enc.copyTextureToBuffer(source, {
            buffer: out,
            offset: command.offset,
            bytesPerRow: row
        }, [
            w,
            h
        ]);
        return;
    }
    const stride = Math.ceil(row / 256) * 256;
    const b = d.device.createBuffer({
        size: stride * h,
        usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });
    staging.push(b);
    enc.copyTextureToBuffer(source, {
        buffer: b,
        bytesPerRow: stride
    }, [
        w,
        h
    ]);
    for(let y = 0; y < h; y++){
        enc.copyBufferToBuffer(b, y * stride, out, command.offset + y * row, row);
    }
}
function encodeRender(d, enc, command, staging) {
    const colorAttachments = command.targets.map((a)=>({
            view: d.textures.get(a.id).createView(),
            loadOp: a.clear === null ? "load" : "clear",
            clearValue: a.clear ?? undefined,
            storeOp: "store"
        }));
    const descriptor = {
        colorAttachments
    };
    if (command.depth !== null) {
        const a = command.depth;
        descriptor.depthStencilAttachment = {
            view: d.textures.get(a.id).createView(),
            depthLoadOp: a.clear === null ? "load" : "clear",
            depthClearValue: a.clear ?? undefined,
            depthStoreOp: "store"
        };
    }
    let args, group;
    if (command.draws.length > 0) {
        args = d.device.createBuffer({
            size: command.args.length + ARGS_WINDOW,
            usage: GPUBufferUsage.STORAGE,
            mappedAtCreation: true
        });
        staging.push(args);
        new Uint8Array(args.getMappedRange()).set(command.args);
        args.unmap();
        const entries = [
            {
                binding: 0,
                resource: {
                    buffer: args,
                    size: 4096
                }
            },
            {
                binding: command.fragBinding,
                resource: {
                    buffer: args,
                    size: 4096
                }
            }
        ];
        d.heaps.forEach((buffer, k)=>entries.push({
                binding: k + 1,
                resource: {
                    buffer
                }
            }));
        group = d.device.createBindGroup({
            layout: d.renderLayout,
            entries
        });
    }
    const pass = enc.beginRenderPass(descriptor);
    for (const draw of command.draws){
        const pipeline = d.renderPipelines.get(draw.pipeline);
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, group, draw.offsets);
        if (draw.textures.length > 0) {
            const entries = draw.textures.flatMap((t, i)=>[
                    {
                        binding: 2 * i,
                        resource: d.textures.get(t.id).createView()
                    },
                    {
                        binding: 2 * i + 1,
                        resource: getSampler(d, t.sampler)
                    }
                ]);
            pass.setBindGroup(1, d.device.createBindGroup({
                layout: pipeline.getBindGroupLayout(1),
                entries
            }));
        }
        pass.setViewport(...draw.viewport, 0, 1);
        pass.setScissorRect(...draw.scissor);
        if (draw.index !== null) {
            const [heap, offset, bytes] = draw.index;
            pass.setIndexBuffer(d.heaps[heap], "uint32", offset, bytes);
            pass.drawIndexed(draw.count, draw.instances);
        } else if (draw.indirect !== null) {
            const [heap, offset] = draw.indirect;
            pass.drawIndirect(d.heaps[heap], offset);
        } else {
            pass.draw(draw.count, draw.instances);
        }
    }
    pass.end();
}
function submit(id, commands, argbuffer, argdata) {
    guarded(id, (d)=>{
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
function encodeGate(d, enc, c, staging) {
    const counted = c.counter !== undefined;
    if (d.gateGrid === null) {
        d.gateGrid = d.device.createBuffer({
            size: 12,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT
        });
    }
    if (counted && d.countGatePipeline === null) {
        const bindings = d.heaps.map((_, heap)=>`@group(0) @binding(${heap + 3}) var<storage, read> heap${heap}: array<u32>;`).join("\n");
        const cases = d.heaps.map((_, heap)=>`case ${heap}u: { return heap${heap}[word]; }`).join("\n");
        const code = `
            @group(0) @binding(0) var<storage, read> args: array<u32>;
            @group(0) @binding(1) var<uniform> config: array<vec4u, 2>;
            @group(0) @binding(2) var<storage, read_write> grid: array<u32>;
            ${bindings}
            fn loadWord(heap: u32, word: u32) -> u32 {
                switch heap { ${cases} default: { return 0u; } }
            }
            @compute @workgroup_size(1) fn main() {
                let flags = config[0];
                var n = 0u;
                if (loadWord(flags.x, flags.y) != 0u) {
                    let raw = loadWord(args[flags.z + 1u], args[flags.z] / 4u);
                    n = min(select(raw, 0u, config[1].y != 0u && bitcast<i32>(raw) < 0i), config[1].x);
                }
                grid[0] = n / flags.w + select(0u, 1u, n % flags.w != 0u);
                grid[1] = 1u;
                grid[2] = 1u;
            }`;
        const module = d.device.createShaderModule({
            code
        });
        d.countGatePipeline = d.device.createComputePipeline({
            layout: "auto",
            compute: {
                module,
                entryPoint: "main"
            }
        });
    } else if (!counted && d.gatePipeline === null) {
        const code = `
            @group(0) @binding(0) var<storage, read> flags: array<u32>;
            @group(0) @binding(1) var<uniform> config: vec4u;
            @group(0) @binding(2) var<storage, read_write> grid: array<u32>;
            @compute @workgroup_size(1) fn main() {
                grid[0] = select(0u, config.y, flags[config.x] != 0u);
                grid[1] = config.z;
                grid[2] = config.w;
            }`;
        const module = d.device.createShaderModule({
            code
        });
        d.gatePipeline = d.device.createComputePipeline({
            layout: "auto",
            compute: {
                module,
                entryPoint: "main"
            }
        });
    }
    const [heap, offset] = c.gate;
    const pipeline = counted ? d.countGatePipeline : d.gatePipeline;
    const key = counted ? `count:${c.args}:${heap}:${offset}:${c.counter.join(":")}` : `${heap}:${offset}:${c.groups.join(":")}`;
    let gate = d.gates.get(key);
    if (!gate) {
        const params = d.device.createBuffer({
            size: counted ? 32 : 16,
            usage: GPUBufferUsage.UNIFORM,
            mappedAtCreation: true
        });
        const words = new Uint32Array(params.getMappedRange());
        if (counted) {
            const [addressOffset, threads, ceiling, signed] = c.counter;
            words.set([
                heap,
                offset / 4,
                addressOffset / 4,
                threads,
                ceiling,
                signed,
                0,
                0
            ]);
        } else {
            words.set([
                offset / 4,
                ...c.groups
            ]);
        }
        params.unmap();
        const entries = [
            {
                binding: 0,
                resource: {
                    buffer: counted ? d.buffers.get(c.args) : d.heaps[heap]
                }
            },
            {
                binding: 1,
                resource: {
                    buffer: params
                }
            },
            {
                binding: 2,
                resource: {
                    buffer: d.gateGrid
                }
            }
        ];
        if (counted) d.heaps.forEach((buffer, i)=>entries.push({
                binding: i + 3,
                resource: {
                    buffer
                }
            }));
        const bind = d.device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries
        });
        gate = {
            params,
            bind
        };
        if (d.gates.size === 256) {
            const oldest = d.gates.keys().next().value;
            staging.push(d.gates.get(oldest).params);
            d.gates.delete(oldest);
        }
        d.gates.set(key, gate);
    }
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, gate.bind);
    pass.dispatchWorkgroups(1);
    pass.end();
}
function encode(d, enc, commands, staging, profile = null) {
    let pass = null;
    for (const c of commands){
        if (c.op === "dispatch") {
            if (c.gate !== undefined) {
                if (pass !== null) {
                    pass.end();
                    pass = null;
                }
                encodeGate(d, enc, c, staging);
            }
            if (pass === null) pass = enc.beginComputePass(profile?.descriptor(c));
            pass.setPipeline(d.pipelines.get(c.pipeline));
            pass.setBindGroup(0, d.bindGroup(c.args), [
                c.argoffset
            ]);
            if (c.gate !== undefined) {
                pass.dispatchWorkgroupsIndirect(d.gateGrid, 0);
            } else {
                pass.dispatchWorkgroups(c.groups[0], c.groups[1], c.groups[2]);
            }
            if (profile !== null) {
                pass.end();
                pass = null;
            }
            continue;
        }
        if (pass !== null) {
            pass.end();
            pass = null;
        }
        if (c.op === "render") {
            encodeRender(d, enc, c, staging);
        } else if (c.op === "copyImage") {
            encodeImageCopy(d, enc, c, staging);
        } else if (c.op === "present") {
            const source = d.textures.get(c.src);
            const target = d.canvases.get(c.canvas).context.getCurrentTexture();
            enc.copyTextureToTexture({
                texture: source
            }, {
                texture: target
            }, [
                source.width,
                source.height
            ]);
        } else if (c.op === "retireTexture") {
            staging.push({
                destroy: ()=>dropTexture(d, c.id)
            });
        } else if (c.op === "copy") {
            enc.copyBufferToBuffer(d.buffers.get(c.src), c.srcoffset, d.buffers.get(c.dst), c.dstoffset, c.size);
        } else if (c.op === "write") {
            const n = c.data.length;
            const s = d.device.createBuffer({
                size: n,
                usage: GPUBufferUsage.COPY_SRC,
                mappedAtCreation: true
            });
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
function dropTexture(d, id) {
    d.textures.get(id)?.destroy();
    d.textures.delete(id);
}
function retireTextures(id, textures) {
    guarded(id, (d)=>{
        d.device.queue.onSubmittedWorkDone().then(()=>{
            for (const tid of textures)dropTexture(d, tid);
        });
    });
}
function end(d, enc, staging, captureStatus = true) {
    const status = captureStatus ? d.device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    }) : null;
    if (status !== null) enc.copyBufferToBuffer(d.heaps[0], 0, status, 0, 4);
    d.device.queue.submit([
        enc.finish()
    ]);
    for (const s of staging)s.destroy();
    if (d.status !== null) d.status.destroy();
    d.status = status;
}
function record(id, recid, commands) {
    guarded(id, (d)=>d.recordings.set(recid, commands));
}
function forget(id, recid) {
    const d = devices.get(id);
    if (d !== undefined) d.recordings.delete(recid);
}
function replay(id, recids, commands, argbuffer, argdata) {
    guarded(id, (d)=>{
        const recordings = [
            commands
        ];
        for (const r of recids){
            const c = d.recordings.get(r);
            if (c === undefined) throw new Error("no recording " + r);
            recordings.push(c);
        }
        let enc = begin(d, argbuffer, argdata);
        let staging = [];
        let count = 0;
        for (const recording of recordings){
            for(let offset = 0; offset < recording.length;){
                if (count === 16) {
                    end(d, enc, staging, false);
                    enc = d.device.createCommandEncoder();
                    staging = [];
                    count = 0;
                }
                const length = Math.min(recording.length - offset, 16 - count);
                const chunk = offset === 0 && length === recording.length ? recording : recording.slice(offset, offset + length);
                encode(d, enc, chunk, staging);
                count += length;
                offset += length;
            }
        }
        end(d, enc, staging);
    });
}
async function profileReplay(id, recids, commands, argbuffer, argdata) {
    const d = dev(id);
    const recordings = recids.map((r)=>{
        const c = d.recordings.get(r);
        if (c === undefined) throw new Error("no recording " + r);
        return c;
    });
    const count = recordings.reduce((n, recording)=>n + recording.filter((c)=>c.op === "dispatch").length, 0);
    const slotsPerSet = 4096;
    const total = 2 * count;
    const staging = [], sets = [], dispatches = [];
    let resolve, out;
    try {
        const enc = begin(d, argbuffer, argdata);
        encode(d, enc, commands, staging);
        for(let slot = 0; slot < total; slot += slotsPerSet){
            sets.push(d.device.createQuerySet({
                type: "timestamp",
                count: Math.min(total - slot, 4096)
            }));
        }
        const profile = {
            descriptor (c) {
                const slot = 2 * dispatches.length;
                dispatches.push(c);
                const begin = slot % 4096;
                return {
                    timestampWrites: {
                        querySet: sets[Math.floor(slot / 4096)],
                        beginningOfPassWriteIndex: begin,
                        endOfPassWriteIndex: begin + 1
                    }
                };
            }
        };
        for (const c of recordings)encode(d, enc, c, staging, profile);
        if (total > 0) {
            const size = total * 8;
            resolve = d.device.createBuffer({
                size,
                usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC
            });
            out = d.device.createBuffer({
                size,
                usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
            });
            sets.forEach((set, i)=>enc.resolveQuerySet(set, 0, set.count, resolve, i * 4096 * 8));
            enc.copyBufferToBuffer(resolve, 0, out, 0, size);
        }
        end(d, enc, staging);
        staging.length = 0;
        if (total === 0) return [];
        await out.mapAsync(GPUMapMode.READ);
        d.check();
        const ticks = new BigUint64Array(out.getMappedRange());
        return dispatches.map((c, i)=>({
                name: c.name,
                pipeline: c.pipeline,
                ns: ticks[2 * i + 1] >= ticks[2 * i] ? Number(ticks[2 * i + 1] - ticks[2 * i]) : null
            }));
    } finally{
        for (const s of staging)s.destroy();
        for (const set of sets)set.destroy();
        resolve?.destroy();
        out?.destroy();
    }
}
async function finish(id) {
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
    if (word !== 0) d.device.queue.writeBuffer(d.heaps[0], 0, new Uint32Array([
        0
    ]));
    return word;
}
export { ARGS_WINDOW as ARGS_WINDOW };
export { open as open };
export { close as close };
export { createBuffer as createBuffer };
export { createReadback as createReadback };
export { readMapped as readMapped };
export { createHeaps as createHeaps };
export { destroyBuffer as destroyBuffer };
export { writeBuffer as writeBuffer };
export { readBuffer as readBuffer };
export { createPipeline as createPipeline };
export { createComputePipelines as createComputePipelines };
export { createRenderPipeline as createRenderPipeline };
export { createAttachment as createAttachment };
export { canvasScale as canvasScale };
export { createCanvas as createCanvas };
export { resizeCanvas as resizeCanvas };
export { destroyCanvas as destroyCanvas };
export { canvasFrame as canvasFrame };
export { createTexture as createTexture };
export { uploadTexture as uploadTexture };
export { readAttachment as readAttachment };
export { submit as submit };
export { retireTextures as retireTextures };
export { record as record };
export { forget as forget };
export { replay as replay };
export { profileReplay as profileReplay };
export { finish as finish };
