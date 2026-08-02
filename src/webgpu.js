// tiny WebGPU helpers
export async function initGPU() {
  if (!navigator.gpu) throw new Error('WebGPU not supported in this browser');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter');
  const device = await adapter.requestDevice();
  return device;
}

export function buf(device, byteLen, extra = 0) {
  return device.createBuffer({
    size: Math.max(byteLen, 16),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | extra,
  });
}

// compute pipeline + bind group over [uniform, ...storage buffers] at bindings 0..n
export function pipe(device, code, buffers, constants) {
  const module = device.createShaderModule({ code });
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module, entryPoint: 'main', ...(constants ? { constants } : {}) },
  });
  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: buffers.map((b, i) => ({ binding: i, resource: { buffer: b } })),
  });
  return { pipeline, bg };
}

export function dispatch(pass, po, wx, wy) {
  pass.setPipeline(po.pipeline);
  pass.setBindGroup(0, po.bg);
  pass.dispatchWorkgroups(Math.ceil(wx / 8), Math.ceil(wy / 8));
}

export async function readBuffer(device, src, byteLen) {
  const staging = device.createBuffer({
    size: byteLen, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(src, 0, staging, 0, byteLen);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(staging.getMappedRange().slice(0));
  staging.destroy();
  return out;
}
