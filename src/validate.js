// Validation suite: Ghia et al. (1982) lid-driven cavity + Strouhal vortex shedding
// + projection sanity. Runs headless (no rendering), prints JSON for CI scraping.
import { initGPU } from './webgpu.js?v=2';
import { Sim, WALL, INFLOW, OUTFLOW } from './solver.js?v=2';
import { GHIA } from './ghia.js?v=2';

const logEl = document.getElementById('log');
const log = m => { logEl.textContent += m + '\n'; console.log(m); };
const fast = new URLSearchParams(location.search).has('fast');

function lerpAt(arr, idx) {
  const i0 = Math.max(0, Math.min(arr.length - 2, Math.floor(idx)));
  const f = Math.min(1, Math.max(0, idx - i0));
  return arr[i0] * (1 - f) + arr[i0 + 1] * f;
}

async function runSteps(sim, total, batch, onBatch) {
  for (let done = 0; done < total; done += batch) {
    sim.step(Math.min(batch, total - done));
    await sim.device.queue.onSubmittedWorkDone();
    if (onBatch && await onBatch(done + batch) === 'stop') return done + batch;
  }
  return total;
}

// ---------- lid-driven cavity ----------
async function cavity(device, Re, N, maxSteps, iters) {
  const dt = Re === 100 ? 0.15 : 0.5;
  const sim = new Sim(device, {
    nx: N, ny: N, dt, nu: N / Re, epsVC: 0, omega: 2 / (1 + Math.sin(Math.PI / N)),
    iters, lidVel: 1, edges: [WALL, WALL, WALL, WALL], slip: 0, dyeMode: 0,
  });
  log(`cavity Re=${Re}: ${N}x${N}, dt=${dt}, nu=${(N / Re).toFixed(4)}, target ${maxSteps} steps max`);
  let prev = null, steps = 0;
  const t0 = performance.now();
  steps = await runSteps(sim, maxSteps, 100, async done => {
    if (done % 2000 !== 0) return;
    const u = await sim.read('u');
    if (!u.every(Number.isFinite)) { log('  NaN detected!'); return 'stop'; }
    if (prev) {
      let d = 0;
      for (let i = 0; i < u.length; i++) d = Math.max(d, Math.abs(u[i] - prev[i]));
      if (done % 10000 === 0) log(`  step ${done}: max|du| over 2k steps = ${d.toExponential(2)}`);
      if (d < 5e-5) { log(`  steady at step ${done} (max|du|=${d.toExponential(2)})`); prev = u; return 'stop'; }
    }
    prev = u;
  });
  const secs = ((performance.now() - t0) / 1000).toFixed(0);
  const u = await sim.read('u'), v = await sim.read('v');
  const nan = !u.every(Number.isFinite);

  // u along x=0.5: u-face column i=N/2, values at y=(j+0.5)/N
  const uProf = GHIA.y.map(y => lerpAt(
    Array.from({ length: N }, (_, j) => u[j * (N + 1) + N / 2]), y * N - 0.5));
  // v along y=0.5: v-face row j=N/2, values at x=(i+0.5)/N
  const vProf = GHIA.x.map(x => lerpAt(
    Array.from({ length: N }, (_, i) => v[(N / 2) * N + i]), x * N - 0.5));

  const ref = Re === 100 ? { u: GHIA.u100, v: GHIA.v100 } : { u: GHIA.u1000, v: GHIA.v1000 };
  const rms = (a, b) => { // skip wall endpoints (trivially imposed, not interpolable)
    let s = 0, n = 0;
    for (let i = 1; i < a.length - 1; i++) { s += (a[i] - b[i]) ** 2; n++; }
    return Math.sqrt(s / n);
  };
  const rmsU = rms(uProf, ref.u), rmsV = rms(vProf, ref.v);
  const maxDiv = await sim.maxDivergence();
  sim.destroy();
  log(`  done in ${secs}s / ${steps} steps: RMS(u)=${rmsU.toFixed(4)} RMS(v)=${rmsV.toFixed(4)} max|div|=${maxDiv.toExponential(2)}`);
  return { Re, N, steps, secs: +secs, rmsU, rmsV, maxDiv, nan, uProf, vProf };
}

// ---------- vortex shedding / Strouhal ----------
async function strouhal(device) {
  const nx = 768, ny = 384, D = 32, U = 1, Re = 150;
  const cx = 192, cy = ny / 2 + 0.5; // half-cell offset seeds asymmetry
  const totalSteps = fast ? 4000 : 16000, dt = 0.5;
  const sim = new Sim(device, {
    nx, ny, dt, nu: U * D / Re, epsVC: 0, omega: 1.9, iters: 80,
    inflow: U, edges: [INFLOW, OUTFLOW, WALL, WALL], slip: 1, dyeMode: 0,
    perturb: 0.004, probe: [cx + 3 * D + 0.5, cy], probeLen: 65536,
  });
  sim.addCircleObstacle(cx, cy, D / 2);
  sim.setFields({ u: new Float32Array((nx + 1) * ny).fill(U) });
  log(`strouhal: Re=${Re}, ${nx}x${ny}, D=${D}, dt=${dt}, ${totalSteps} steps`);
  const t0 = performance.now();
  await runSteps(sim, totalSteps, 50, async done => {
    if (sim.stepIdx > 2500) sim.cfg.perturb = 0;
    if (done % 10000 === 0) log(`  step ${done}...`);
  });
  const secs = ((performance.now() - t0) / 1000).toFixed(0);
  const series = await sim.readProbe();
  sim.destroy();
  // frequency from mean upward-zero-crossing interval, transient dropped
  const tail = Array.from(series.slice(Math.floor(series.length / 2)));
  const mean = tail.reduce((a, b) => a + b, 0) / tail.length;
  let amp = 0;
  for (const s of tail) amp = Math.max(amp, Math.abs(s - mean));
  const crossings = [];
  for (let i = 1; i < tail.length; i++)
    if (tail[i - 1] - mean < 0 && tail[i] - mean >= 0) crossings.push(i);
  let St = NaN, periods = crossings.length - 1;
  if (periods >= 3) {
    const period = (crossings[crossings.length - 1] - crossings[0]) / periods; // steps
    St = (1 / (period * dt)) * D / U;
  }
  const StRef = 0.198 * (1 - 19.7 / Re);
  log(`  done in ${secs}s: amp=${amp.toFixed(3)} periods=${periods} St=${St.toFixed(4)} ref=${StRef.toFixed(4)} err=${(100 * Math.abs(St - StRef) / StRef).toFixed(1)}%`);
  return { Re, St, StRef, relErr: Math.abs(St - StRef) / StRef, amp, periods, secs: +secs };
}

// ---------- projection sanity ----------
async function projection(device) {
  const N = 128;
  const sim = new Sim(device, {
    nx: N, ny: N, dt: 1, nu: 0, epsVC: 0, omega: 2 / (1 + Math.sin(Math.PI / N)),
    iters: 200, edges: [WALL, WALL, WALL, WALL], slip: 1, dyeMode: 0,
  });
  const u = new Float32Array((N + 1) * N), v = new Float32Array(N * (N + 1));
  let seed = 42;
  const rnd = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff - 0.5;
  for (let i = 0; i < u.length; i++) u[i] = rnd();
  for (let i = 0; i < v.length; i++) v[i] = rnd();
  sim.setFields({ u, v });
  const before = await sim.maxDivergence();
  sim.step(1);
  await device.queue.onSubmittedWorkDone();
  const after = await sim.maxDivergence();
  sim.destroy();
  log(`projection: max|div| ${before.toFixed(3)} -> ${after.toExponential(2)} (x${(before / after).toFixed(0)} reduction)`);
  return { before, after, ratio: before / after };
}

// ---------- main ----------
(async () => {
  const results = { fast, gpu: null, projection: null, cavity100: null, cavity1000: null, strouhal: null, pass: {} };
  try {
    const device = await initGPU();
    results.gpu = 'ok';
    device.addEventListener?.('uncapturederror', e => log('WebGPU error: ' + e.error.message));

    results.projection = await projection(device);
    results.pass.projection = results.projection.ratio > 100;

    results.cavity100 = await cavity(device, 100, fast ? 96 : 128, fast ? 6000 : 50000, fast ? 150 : 110);
    results.pass.cavity100 = !results.cavity100.nan && results.cavity100.rmsU < 0.015 && results.cavity100.rmsV < 0.015;

    if (!fast) {
      results.cavity1000 = await cavity(device, 1000, 224, 40000, 110);
      results.pass.cavity1000 = !results.cavity1000.nan && results.cavity1000.rmsU < 0.02 && results.cavity1000.rmsV < 0.02;
    }

    results.strouhal = await strouhal(device);
    results.pass.strouhal = results.strouhal.periods >= 10 && results.strouhal.relErr < 0.15;

    results.allPass = Object.values(results.pass).every(Boolean);
  } catch (err) {
    results.error = err.message + '\n' + err.stack;
    results.allPass = false;
    log('FATAL: ' + err.message);
  }
  const summary = Object.entries(results.pass).map(([k, v]) => `${k}: ${v ? 'PASS' : 'FAIL'}`).join('  ·  ');
  document.getElementById('status').textContent =
    (results.allPass ? '✅ ALL PASS — ' : '❌ FAILURES — ') + summary;
  document.getElementById('status').className = results.allPass ? 'pass' : 'fail';
  console.log('VALIDATION_JSON ' + JSON.stringify(results));
  window.__validation = results;
})();
