import { initGPU } from './webgpu.js';
import { Sim, presets } from './solver.js';
import { RENDER } from './shaders.js';

const $ = id => document.getElementById(id);
const canvas = $('view');

let device, sim, preset, presetKey, renderer;
let paused = false, frames = 0, lastFPS = performance.now(), fpsN = 0;

const PRESETS = presets();

const EXPLAIN = {
  tunnel: `<b>Kármán vortex street.</b> Uniform inflow (left) meets a cylinder. Above a
    Reynolds number of ~47 the wake becomes unstable and sheds counter-rotating vortices
    alternately from each side — the same phenomenon that makes power lines "sing" and
    forced the Tacoma Narrows redesign. Here Re = UD/ν ≈ 200. Switch to the
    <i>vorticity</i> view to see the street as red/blue (counter-clockwise/clockwise) filaments.
    Drag/lift are integrated from pressure on the cylinder surface. Right-drag to add obstacles.`,
  rt: `<b>Rayleigh–Taylor instability.</b> Heavy fluid (bright) sits on top of light fluid —
    an unstable equilibrium. Any ripple in the interface grows: heavy fingers sink, light
    bubbles rise, and shear along the finger edges triggers secondary Kelvin–Helmholtz
    rolls, giving the classic mushroom shapes. This drives supernova remnant mixing and
    limits inertial-confinement fusion.`,
  plume: `<b>Buoyant plume.</b> A continuous source of hot "smoke" rises under buoyancy
    (Boussinesq approximation: force ∝ dye concentration). The rising column entrains
    surrounding fluid, goes unstable, and rolls into vortices. Vorticity confinement
    re-injects the fine swirl the grid would otherwise smear out.`,
  shear: `<b>Double shear layer.</b> Two bands of fluid slide past each other. The
    Kelvin–Helmholtz instability amplifies a tiny sinusoidal perturbation until the
    interface rolls up into a chain of vortices — the ur-instability behind cloud
    billows and Jupiter's bands. Watch it in the vorticity view.`,
  cavity: `<b>Lid-driven cavity, Re = 1000.</b> The top wall slides right at constant
    speed, dragging the fluid into a large central vortex with counter-rotating eddies
    in the corners. The standard CFD benchmark: this exact setup is validated against
    Ghia et al. (1982) in this project's <a href="validation.html">validation suite</a>.`,
};

const NS_EQ = `<div class="eq">∂<b>u</b>/∂t + (<b>u</b>·∇)<b>u</b> = −∇p/ρ + ν∇²<b>u</b> + <b>f</b>,&nbsp;&nbsp; ∇·<b>u</b> = 0</div>
  <p>Solved with Chorin projection: MacCormack (BFECC) advection → forces → viscous
  diffusion → pressure Poisson solve (red–black Gauss–Seidel + SOR) → projection to a
  divergence-free field. Staggered MAC grid, all stages WebGPU compute shaders.</p>`;

function makeRenderer() {
  const ctx = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format, alphaMode: 'opaque' });
  const module = device.createShaderModule({ code: RENDER });
  const pipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module, entryPoint: 'vs' },
    fragment: { module, entryPoint: 'fs', targets: [{ format }] },
    primitive: { topology: 'triangle-list' },
  });
  const rp = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const B = sim.b;
  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [B.params, rp, B.u, B.v, B.dye, B.p, B.vort, B.flags]
      .map((b, i) => ({ binding: i, resource: { buffer: b } })),
  });
  return { ctx, pipeline, rp, bg };
}

function draw() {
  const mode = +$('mode').value;
  const inten = +$('intensity').value;
  if (mode === 2) sim.computeVorticity();
  const s = preset.render;
  const a = new ArrayBuffer(32);
  new Uint32Array(a)[0] = mode;
  const f = new Float32Array(a);
  f[2] = canvas.width; f[3] = canvas.height;
  f[4] = s.sDye * inten; f[5] = s.sVel * inten; f[6] = s.sVort * inten; f[7] = s.sP * inten;
  device.queue.writeBuffer(renderer.rp, 0, a);
  const enc = device.createCommandEncoder();
  const pass = enc.beginRenderPass({
    colorAttachments: [{
      view: renderer.ctx.getCurrentTexture().createView(),
      loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 1 }, storeOp: 'store',
    }],
  });
  pass.setPipeline(renderer.pipeline);
  pass.setBindGroup(0, renderer.bg);
  pass.draw(3);
  pass.end();
  device.queue.submit([enc.finish()]);
}

function loadPreset(key) {
  presetKey = key;
  if (sim) sim.destroy();
  preset = PRESETS[key];
  sim = new Sim(device, { ...preset.cfg });
  preset.init(sim);
  canvas.width = sim.nx * 2; canvas.height = sim.ny * 2;
  renderer = makeRenderer();
  $('epsVC').value = preset.cfg.epsVC;
  $('iters').value = preset.cfg.iters;
  $('speed').value = preset.stepsPerFrame;
  $('explain-body').innerHTML = EXPLAIN[key] + NS_EQ;
  $('readout').textContent = '';
}

// ---- mouse ----
let dragging = 0; // 1 left (force+dye), 2 right (obstacle)
let lastPos = null;
function gridPos(e) {
  const r = canvas.getBoundingClientRect();
  return {
    x: (e.clientX - r.left) / r.width * sim.nx,
    y: (1 - (e.clientY - r.top) / r.height) * sim.ny,
  };
}
canvas.addEventListener('contextmenu', e => e.preventDefault());
canvas.addEventListener('pointerdown', e => {
  canvas.setPointerCapture(e.pointerId);
  dragging = e.button === 2 ? 2 : 1;
  lastPos = gridPos(e);
  if (dragging === 2) sim.addCircleObstacle(lastPos.x, lastPos.y, +$('brush').value / 2);
});
canvas.addEventListener('pointerup', () => { dragging = 0; sim.mouse.on = 0; });
canvas.addEventListener('pointermove', e => {
  if (!dragging) return;
  const p = gridPos(e);
  if (dragging === 2) { sim.addCircleObstacle(p.x, p.y, +$('brush').value / 2); lastPos = p; return; }
  const spf = Math.max(1, +$('speed').value);
  const force = +$('force').value / spf;
  sim.mouse = {
    x: p.x, y: p.y,
    vx: (p.x - lastPos.x) * force, vy: (p.y - lastPos.y) * force,
    r: +$('brush').value, amt: +$('dyeAmt').value / spf, on: 1,
  };
  lastPos = p;
});

// ---- drag/lift (pressure integral over solid faces), tunnel only ----
async function updateForces() {
  if (presetKey !== 'tunnel' || paused) return;
  const key = presetKey;
  const [p, flags] = [await sim.read('p'), sim.flagsCPU];
  if (presetKey !== key) return; // preset switched while reading
  const { nx, ny } = sim;
  let fx = 0, fy = 0;
  for (let j = 1; j < ny - 1; j++)
    for (let i = 1; i < nx - 1; i++) {
      if (flags[j * nx + i] !== 1) continue;
      if (flags[j * nx + i - 1] === 0) fx += p[j * nx + i - 1]; // fluid left pushes +x
      if (flags[j * nx + i + 1] === 0) fx -= p[j * nx + i + 1];
      if (flags[(j - 1) * nx + i] === 0) fy += p[(j - 1) * nx + i];
      if (flags[(j + 1) * nx + i] === 0) fy -= p[(j + 1) * nx + i];
    }
  const D = 32, U = sim.cfg.inflow || 1;
  const cd = 2 * fx / (U * U * D), cl = 2 * fy / (U * U * D);
  $('readout').textContent = `Cd ≈ ${cd.toFixed(2)}   Cl ≈ ${cl.toFixed(2)}  (pressure only)`;
}

// ---- main loop ----
function frame() {
  if (!paused) {
    sim.cfg.epsVC = +$('epsVC').value;
    sim.cfg.iters = +$('iters').value;
    if (sim.cfg.perturb && sim.stepIdx > 3000) sim.cfg.perturb = 0;
    sim.step(+$('speed').value);
    sim.mouse.on = dragging === 1 ? 1 : 0;
    if (dragging !== 1) sim.mouse.vx = sim.mouse.vy = sim.mouse.amt = 0;
  }
  draw();
  frames++; fpsN++;
  const now = performance.now();
  if (now - lastFPS > 1000) {
    $('fps').textContent = `${fpsN} fps · ${sim.nx}×${sim.ny}`;
    fpsN = 0; lastFPS = now;
  }
  if (frames % 45 === 0) updateForces();
  requestAnimationFrame(frame);
}

// ---- UI ----
$('preset').addEventListener('change', e => loadPreset(e.target.value));
$('restart').addEventListener('click', () => loadPreset(presetKey));
$('pause').addEventListener('click', () => {
  paused = !paused;
  $('pause').textContent = paused ? 'Resume' : 'Pause';
});
$('clearObs').addEventListener('click', () => {
  if (presetKey === 'tunnel') { sim.setObstacles(null); sim.addCircleObstacle(110, 131, 16); }
  else sim.setObstacles(null);
});
$('shot').addEventListener('click', () => {
  draw();
  canvas.toBlob(b => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(b);
    a.download = `fluid-${presetKey}-${Date.now()}.png`;
    a.click();
  });
});
$('help').addEventListener('click', () => $('explain').classList.toggle('open'));
$('explain-close').addEventListener('click', () => $('explain').classList.remove('open'));

// ---- boot ----
(async () => {
  try {
    device = await initGPU();
  } catch (err) {
    $('nogpu').style.display = 'block';
    $('nogpu').textContent = 'WebGPU unavailable: ' + err.message +
      ' — use Chrome/Edge 113+ (or enable chrome://flags/#enable-unsafe-webgpu).';
    return;
  }
  device.addEventListener?.('uncapturederror', e => console.error('WebGPU error:', e.error.message));
  const sel = $('preset');
  for (const [k, p] of Object.entries(PRESETS)) {
    const o = document.createElement('option');
    o.value = k; o.textContent = p.label;
    sel.appendChild(o);
  }
  const q = new URLSearchParams(location.search);
  loadPreset(PRESETS[q.get('preset')] ? q.get('preset') : 'tunnel');
  if (['0', '1', '2', '3'].includes(q.get('view'))) $('mode').value = q.get('view');
  window.__sim = () => sim; // for headless checks
  requestAnimationFrame(frame);
})();
