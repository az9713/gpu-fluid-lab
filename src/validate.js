// Validation suite: Ghia et al. (1982) lid-driven cavity + Strouhal vortex shedding
// + projection sanity. Runs headless (no rendering), prints JSON for CI scraping.
import { initGPU } from './webgpu.js?v=2';
import { Sim, WALL, INFLOW, OUTFLOW } from './solver.js?v=2';
import { GHIA } from './ghia.js?v=2';

const logEl = document.getElementById('log');
const log = m => { logEl.textContent += m + '\n'; console.log(m); };
const fast = new URLSearchParams(location.search).has('fast');

// ---------- per-test progress cards ----------
const CARDS = {};
function makeCard(id, title, blurb, est) {
  const d = document.createElement('div');
  d.className = 'card';
  d.innerHTML = `<div class="card-head"><b>${title}</b><span class="stat">queued</span></div>
    <p>${blurb}</p><div class="bar"><div class="fill"></div></div><div class="detail">estimated ${est}</div>`;
  document.getElementById('tests').appendChild(d);
  CARDS[id] = { el: d, stat: d.querySelector('.stat'), fill: d.querySelector('.fill'),
    detail: d.querySelector('.detail'), t0: 0 };
}
function cardBegin(id) {
  const c = CARDS[id]; c.el.classList.add('running');
  c.stat.textContent = 'running'; c.t0 = performance.now();
}
function cardProg(id, frac, msg) {
  const c = CARDS[id];
  c.fill.style.width = (100 * Math.min(1, frac)).toFixed(1) + '%';
  const secs = ((performance.now() - c.t0) / 1000) | 0;
  c.detail.textContent = `${msg} · ${Math.floor(secs / 60)}m${(secs % 60 + '').padStart(2, '0')}s elapsed`;
}
function cardDone(id, pass, msg) {
  const c = CARDS[id]; c.el.classList.remove('running');
  c.el.classList.add(pass ? 'pass' : 'fail');
  c.stat.textContent = pass ? 'PASS ✓' : 'FAIL ✗';
  c.fill.style.width = '100%';
  c.detail.textContent = msg;
}
makeCard('projection', 'Pressure projection sanity',
  `Incompressible flow must have zero divergence — no cell may act as a source or sink of
   fluid. This test fills the box with random velocities and applies one pressure solve;
   the maximum divergence must drop by at least 100×. It verifies the heart of the solver
   (the Poisson solve + gradient subtraction) in isolation.`, 'a few seconds');
makeCard('cavity100', `Lid-driven cavity, Re=100 — vs Ghia et al. (1982)`,
  `The most-cited benchmark in CFD: a square box of fluid whose top wall slides sideways,
   spinning up a steady vortex. The velocity profiles along the two centerlines are compared
   point-by-point against the reference tables of Ghia, Ghia &amp; Shin (1982). Runs headless
   to steady state (up to ~16k timesteps); pass requires RMS error &lt; 0.015 on a lid speed of 1.`,
  fast ? '~30 s (fast mode)' : '~1–2 min');
if (!fast) makeCard('cavity1000', 'Lid-driven cavity, Re=1000 — vs Ghia et al. (1982)',
  `Same experiment at 10× the Reynolds number: boundary layers are thinner, corner eddies
   stronger, and cheap solvers visibly fail here. Uses a finer grid (224²) and ~32k timesteps
   to reach steady state. Pass requires RMS error &lt; 0.02.`, '~5–7 min');
makeCard('strouhal', 'Vortex-shedding frequency — vs the Strouhal relation',
  `A cylinder in a stream sheds vortices at a rate nature fixes precisely; the dimensionless
   frequency (Strouhal number) at Re=150 is an empirical law: St = 0.198(1 − 19.7/Re) ≈ 0.172.
   The test runs a wind tunnel, records the crossflow velocity behind the cylinder for ~22
   shedding cycles, and measures the frequency from the oscillation. Pass requires ±15%
   (wall blockage biases it slightly high). The longest test — this is the one to wait for.`,
  fast ? '~3–4 min (fast mode)' : '~10–12 min');

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
  const cid = Re === 100 ? 'cavity100' : 'cavity1000';
  cardBegin(cid);
  let prev = null, steps = 0;
  const t0 = performance.now();
  steps = await runSteps(sim, maxSteps, 100, async done => {
    cardProg(cid, done / maxSteps, `step ${done}/${maxSteps} (exits early once steady)`);
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
  cardBegin('strouhal');
  const t0 = performance.now();
  await runSteps(sim, totalSteps, 50, async done => {
    if (sim.stepIdx > 2500) sim.cfg.perturb = 0;
    const phase = done < totalSteps / 2 ? 'developing the vortex street' : 'recording shedding cycles';
    if (done % 500 === 0) cardProg('strouhal', done / totalSteps, `step ${done}/${totalSteps} — ${phase}`);
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
  cardBegin('projection');
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
    document.getElementById('status').textContent = 'running benchmarks… (tests execute in sequence below)';

    results.projection = await projection(device);
    results.pass.projection = results.projection.ratio > 100;
    cardDone('projection', results.pass.projection,
      `divergence reduced ${results.projection.ratio.toFixed(0)}× in one solve (needs >100×)`);

    results.cavity100 = await cavity(device, 100, fast ? 96 : 128, fast ? 6000 : 50000, fast ? 150 : 110);
    results.pass.cavity100 = !results.cavity100.nan && results.cavity100.rmsU < 0.015 && results.cavity100.rmsV < 0.015;
    cardDone('cavity100', results.pass.cavity100,
      `RMS vs Ghia: u=${results.cavity100.rmsU.toFixed(4)}, v=${results.cavity100.rmsV.toFixed(4)} (tolerance 0.015) · ${results.cavity100.steps} steps, ${results.cavity100.secs}s`);

    if (!fast) {
      results.cavity1000 = await cavity(device, 1000, 224, 40000, 110);
      results.pass.cavity1000 = !results.cavity1000.nan && results.cavity1000.rmsU < 0.02 && results.cavity1000.rmsV < 0.02;
      cardDone('cavity1000', results.pass.cavity1000,
        `RMS vs Ghia: u=${results.cavity1000.rmsU.toFixed(4)}, v=${results.cavity1000.rmsV.toFixed(4)} (tolerance 0.02) · ${results.cavity1000.steps} steps, ${results.cavity1000.secs}s`);
    }

    results.strouhal = await strouhal(device);
    results.pass.strouhal = results.strouhal.periods >= 10 && results.strouhal.relErr < 0.15;
    cardDone('strouhal', results.pass.strouhal,
      `St = ${results.strouhal.St.toFixed(4)} vs ${results.strouhal.StRef.toFixed(4)} reference — ${(100 * results.strouhal.relErr).toFixed(1)}% off (tolerance 15%) over ${results.strouhal.periods} cycles, ${results.strouhal.secs}s`);

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
