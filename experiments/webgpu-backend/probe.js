// What the page's WebGPU offers: adapter info, features, the limits the Mantle
// backend plans around, subgroup matrix configurations and WGSL language features.
(async () => {
  if (!navigator.gpu) return {error: "no navigator.gpu", ua: navigator.userAgent};
  const out = {ua: navigator.userAgent, adapters: []};
  const limits = ["maxStorageBufferBindingSize", "maxBufferSize", "maxStorageBuffersPerShaderStage",
    "maxComputeWorkgroupStorageSize", "maxComputeInvocationsPerWorkgroup",
    "maxDynamicStorageBuffersPerPipelineLayout", "minStorageBufferOffsetAlignment", "maxBindGroups",
    "maxImmediateSize", "maxSubgroupSize", "minSubgroupSize"];
  for (const pref of ["high-performance", "low-power"]) {
    const a = await navigator.gpu.requestAdapter({powerPreference: pref});
    if (!a) { out.adapters.push({pref, adapter: null}); continue; }
    const info = a.info || {};
    const lim = {};
    for (const k of limits) lim[k] = a.limits[k];
    const configs = info.subgroupMatrixConfigs ? [...info.subgroupMatrixConfigs].map(c =>
      ({componentType: c.componentType, resultComponentType: c.resultComponentType, M: c.M, N: c.N, K: c.K})) : null;
    out.adapters.push({pref, vendor: info.vendor, architecture: info.architecture, device: info.device,
      description: info.description, subgroupMinSize: info.subgroupMinSize, subgroupMaxSize: info.subgroupMaxSize,
      features: [...a.features].sort(), limits: lim, subgroupMatrixConfigs: configs});
  }
  out.wgslFeatures = navigator.gpu.wgslLanguageFeatures ? [...navigator.gpu.wgslLanguageFeatures].sort() : null;
  return out;
})()
