import * as S from './shaders.js?v=2';
import { buf, pipe, dispatch, readBuffer } from './webgpu.js?v=2';

export const WALL = 0, INFLOW = 1, OUTFLOW = 2;

const N_PARAMS = 28; // f32/u32 slots in Params

export class Sim {
  // cfg: {nx, ny, dt, nu, epsVC, omega, iters, inflow, lidVel, buoy, slip,
  //       edges:[L,R,B,T], dyeMode, dyeDecay, perturb, probe:[x,y]|null}
  constructor(device, cfg) {
    this.device = device;
    this.cfg = Object.assign({
      dt: 1, nu: 0, epsVC: 0, omega: 1.9, iters: 50, inflow: 0, lidVel: 0,
      buoy: 0, slip: 1, edges: [WALL, WALL, WALL, WALL], dyeMode: 0,
      dyeDecay: 1, perturb: 0, probe: null,
    }, cfg);
    const { nx, ny } = this.cfg;
    this.nx = nx; this.ny = ny;
    this.t = 0; this.stepIdx = 0;
    this.mouse = { x: 0, y: 0, vx: 0, vy: 0, r: 10, amt: 0, on: 0 };

    const nu_ = (nx + 1) * ny, nv = nx * (ny + 1), nc = nx * ny;
    const d = device;
    this.b = {
      u: buf(d, nu_ * 4), uF: buf(d, nu_ * 4), uB: buf(d, nu_ * 4), uN: buf(d, nu_ * 4),
      v: buf(d, nv * 4), vF: buf(d, nv * 4), vB: buf(d, nv * 4), vN: buf(d, nv * 4),
      dye: buf(d, nc * 4), dF: buf(d, nc * 4), dB: buf(d, nc * 4), dN: buf(d, nc * 4),
      p: buf(d, nc * 4), div: buf(d, nc * 4), vort: buf(d, nc * 4),
      flags: buf(d, nc * 4),
      params: d.createBuffer({ size: N_PARAMS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
    };
    if (this.cfg.probe) {
      this.probeLen = this.cfg.probeLen || 65536;
      this.b.probeCnt = buf(d, 4);
      this.b.probeOut = buf(d, this.probeLen * 4);
    }
    this.flagsCPU = new Uint32Array(nc);

    const B = this.b;
    this.k = {
      advUF: pipe(d, S.advect('U'), [B.params, B.u, B.v, B.u, B.uF], { DIR: 1 }),
      advUB: pipe(d, S.advect('U'), [B.params, B.u, B.v, B.uF, B.uB], { DIR: -1 }),
      corU:  pipe(d, S.correct('U'), [B.params, B.u, B.v, B.u, B.uF, B.uB, B.uN]),
      advVF: pipe(d, S.advect('V'), [B.params, B.u, B.v, B.v, B.vF], { DIR: 1 }),
      advVB: pipe(d, S.advect('V'), [B.params, B.u, B.v, B.vF, B.vB], { DIR: -1 }),
      corV:  pipe(d, S.correct('V'), [B.params, B.u, B.v, B.v, B.vF, B.vB, B.vN]),
      advDF: pipe(d, S.advect('C'), [B.params, B.u, B.v, B.dye, B.dF], { DIR: 1 }),
      advDB: pipe(d, S.advect('C'), [B.params, B.u, B.v, B.dF, B.dB], { DIR: -1 }),
      corD:  pipe(d, S.correct('C'), [B.params, B.u, B.v, B.dye, B.dF, B.dB, B.dN]),
      forces: pipe(d, S.FORCES, [B.params, B.u, B.v, B.dye, B.flags]),
      diffU: pipe(d, S.DIFF_U, [B.params, B.u, B.uN]),
      diffV: pipe(d, S.DIFF_V, [B.params, B.v, B.vN]),
      vort: pipe(d, S.VORT, [B.params, B.u, B.v, B.vort]),
      confine: pipe(d, S.CONFINE, [B.params, B.vort, B.u, B.v]),
      div: pipe(d, S.DIVERGENCE, [B.params, B.u, B.v, B.flags, B.div]),
      gsR: pipe(d, S.RBGS, [B.params, B.div, B.flags, B.p], { PASS: 0 }),
      gsB: pipe(d, S.RBGS, [B.params, B.div, B.flags, B.p], { PASS: 1 }),
      subgrad: pipe(d, S.SUBGRAD, [B.params, B.p, B.flags, B.u, B.v]),
      bc: pipe(d, S.BC, [B.params, B.flags, B.u, B.v]),
    };
    if (this.cfg.probe) {
      this.k.probe = pipe(d, S.PROBE, [B.params, B.v, B.probeCnt, B.probeOut]);
    }
    this.writeFlags();
  }

  writeParams() {
    const c = this.cfg, m = this.mouse;
    const a = new ArrayBuffer(N_PARAMS * 4);
    const u32 = new Uint32Array(a), f32 = new Float32Array(a);
    u32[0] = this.nx; u32[1] = this.ny;
    u32[2] = c.edges[0]; u32[3] = c.edges[1]; u32[4] = c.edges[2]; u32[5] = c.edges[3];
    u32[6] = c.dyeMode; u32[7] = this.stepIdx;
    f32[8] = c.dt; f32[9] = c.nu; f32[10] = c.epsVC; f32[11] = c.omega;
    f32[12] = c.inflow; f32[13] = c.lidVel; f32[14] = c.buoy; f32[15] = c.slip;
    f32[16] = m.x; f32[17] = m.y; f32[18] = m.vx; f32[19] = m.vy;
    f32[20] = m.r; f32[21] = m.amt; f32[22] = m.on; f32[23] = this.t;
    f32[24] = c.perturb; f32[25] = c.dyeDecay;
    f32[26] = c.probe ? c.probe[0] : 0; f32[27] = c.probe ? c.probe[1] : 0;
    this.device.queue.writeBuffer(this.b.params, 0, a);
  }

  // paint: (i,j) => true if solid, or null to clear
  setObstacles(paint) {
    const { nx, ny } = this;
    this.flagsCPU.fill(0);
    if (paint) {
      for (let j = 0; j < ny; j++)
        for (let i = 0; i < nx; i++)
          if (paint(i, j)) this.flagsCPU[j * nx + i] = 1;
    }
    this.writeFlags();
  }

  addCircleObstacle(cx, cy, r, solid = true) {
    const { nx, ny } = this;
    for (let j = Math.max(0, cy - r - 1 | 0); j < Math.min(ny, cy + r + 2); j++)
      for (let i = Math.max(0, cx - r - 1 | 0); i < Math.min(nx, cx + r + 2); i++) {
        const dx = i + 0.5 - cx, dy = j + 0.5 - cy;
        if (dx * dx + dy * dy < r * r) this.flagsCPU[j * nx + i] = solid ? 1 : 0;
      }
    this.writeFlags();
  }

  writeFlags() { this.device.queue.writeBuffer(this.b.flags, 0, this.flagsCPU); }

  // fields: {u,v,dye} Float32Arrays (optional each)
  setFields(f = {}) {
    const q = this.device.queue, B = this.b;
    if (f.u) q.writeBuffer(B.u, 0, f.u);
    if (f.v) q.writeBuffer(B.v, 0, f.v);
    if (f.dye) q.writeBuffer(B.dye, 0, f.dye);
    if (f.clearP !== false) q.writeBuffer(B.p, 0, new Float32Array(this.nx * this.ny));
  }

  encodeStep(enc) {
    const { nx, ny } = this, c = this.cfg, k = this.k, B = this.b;
    const nuBytes = (nx + 1) * ny * 4, nvBytes = nx * (ny + 1) * 4, ncBytes = nx * ny * 4;
    const pass = enc.beginComputePass();
    dispatch(pass, k.forces, nx + 1, ny + 1);
    if (c.epsVC > 0) {
      dispatch(pass, k.vort, nx, ny);
      dispatch(pass, k.confine, nx + 1, ny + 1);
    }
    pass.end();
    if (c.nu > 0) {
      const p2 = enc.beginComputePass();
      dispatch(p2, k.diffU, nx + 1, ny);
      dispatch(p2, k.diffV, nx, ny + 1);
      p2.end();
      enc.copyBufferToBuffer(B.uN, 0, B.u, 0, nuBytes);
      enc.copyBufferToBuffer(B.vN, 0, B.v, 0, nvBytes);
    }
    {
      const p3 = enc.beginComputePass();
      dispatch(p3, k.advUF, nx + 1, ny); dispatch(p3, k.advVF, nx, ny + 1);
      dispatch(p3, k.advDF, nx, ny);
      dispatch(p3, k.advUB, nx + 1, ny); dispatch(p3, k.advVB, nx, ny + 1);
      dispatch(p3, k.advDB, nx, ny);
      dispatch(p3, k.corU, nx + 1, ny); dispatch(p3, k.corV, nx, ny + 1);
      dispatch(p3, k.corD, nx, ny);
      p3.end();
      enc.copyBufferToBuffer(B.uN, 0, B.u, 0, nuBytes);
      enc.copyBufferToBuffer(B.vN, 0, B.v, 0, nvBytes);
      enc.copyBufferToBuffer(B.dN, 0, B.dye, 0, ncBytes);
    }
    const p4 = enc.beginComputePass();
    dispatch(p4, k.bc, nx + 1, ny + 1);
    dispatch(p4, k.div, nx, ny);
    for (let it = 0; it < c.iters; it++) {
      dispatch(p4, k.gsR, Math.ceil(nx / 2), ny);
      dispatch(p4, k.gsB, Math.ceil(nx / 2), ny);
    }
    dispatch(p4, k.subgrad, nx + 1, ny + 1);
    dispatch(p4, k.bc, nx + 1, ny + 1);
    if (this.k.probe) {
      p4.setPipeline(this.k.probe.pipeline);
      p4.setBindGroup(0, this.k.probe.bg);
      p4.dispatchWorkgroups(1);
    }
    p4.end();
    this.t += c.dt; this.stepIdx++;
  }

  // run n steps in one submit; params written once (t coarse within batch)
  step(n = 1) {
    this.writeParams();
    const enc = this.device.createCommandEncoder();
    for (let s = 0; s < n; s++) this.encodeStep(enc);
    this.device.queue.submit([enc.finish()]);
  }

  computeVorticity() {
    this.writeParams();
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    dispatch(pass, this.k.vort, this.nx, this.ny);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  async read(name) {
    const { nx, ny } = this;
    const len = { u: (nx + 1) * ny, v: nx * (ny + 1), dye: nx * ny, p: nx * ny, div: nx * ny, vort: nx * ny };
    return readBuffer(this.device, this.b[name], len[name] * 4);
  }

  async maxDivergence() {
    this.writeParams();
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    dispatch(pass, this.k.div, this.nx, this.ny);
    pass.end();
    this.device.queue.submit([enc.finish()]);
    const d = await this.read('div');
    let m = 0;
    for (let i = 0; i < d.length; i++) m = Math.max(m, Math.abs(d[i]));
    return m;
  }

  async readProbe() {
    const cnt = await readBuffer(this.device, this.b.probeCnt, 4);
    const n = new Uint32Array(cnt.buffer)[0];
    const all = await readBuffer(this.device, this.b.probeOut, this.probeLen * 4);
    return all.slice(0, Math.min(n, this.probeLen));
  }

  destroy() {
    for (const b of Object.values(this.b)) b.destroy();
  }
}

// ---------- presets ----------

export function presets() {
  return {
    tunnel: {
      label: 'Wind Tunnel (Kármán vortex street)',
      cfg: {
        nx: 512, ny: 256, dt: 1.0, nu: 0.16, epsVC: 0, omega: 1.9, iters: 50,
        inflow: 1, edges: [INFLOW, OUTFLOW, WALL, WALL], slip: 1,
        dyeMode: 1, dyeDecay: 1, perturb: 0.004,
      },
      stepsPerFrame: 2,
      render: { sDye: 1, sVel: 0.6, sVort: 4, sP: 1.5 },
      init(sim) {
        sim.addCircleObstacle(110, 131, 16);
        const u = new Float32Array((sim.nx + 1) * sim.ny).fill(1);
        sim.setFields({ u });
      },
    },
    rt: {
      label: 'Rayleigh–Taylor instability',
      cfg: {
        nx: 384, ny: 384, dt: 1.0, nu: 0, epsVC: 0.05, omega: 1.9, iters: 50,
        buoy: -0.012, edges: [WALL, WALL, WALL, WALL], slip: 1,
        dyeMode: 0, dyeDecay: 1,
      },
      stepsPerFrame: 2,
      render: { sDye: 1, sVel: 0.8, sVort: 8, sP: 0.5 },
      init(sim) {
        const { nx, ny } = sim;
        const dye = new Float32Array(nx * ny);
        for (let j = 0; j < ny; j++)
          for (let i = 0; i < nx; i++) {
            const yInt = ny / 2 + 5 * Math.sin(2 * Math.PI * 3 * i / nx);
            dye[j * nx + i] = j > yInt ? 1 : 0;   // heavy fluid on top
          }
        sim.setFields({ dye });
      },
    },
    plume: {
      label: 'Buoyant smoke plume',
      cfg: {
        nx: 384, ny: 384, dt: 1.0, nu: 0, epsVC: 0.15, omega: 1.9, iters: 50,
        buoy: 0.08, edges: [WALL, WALL, WALL, WALL], slip: 1,
        dyeMode: 2, dyeDecay: 0.9995,
      },
      stepsPerFrame: 2,
      render: { sDye: 1.6, sVel: 0.6, sVort: 6, sP: 0.5 },
      init() {},
    },
    shear: {
      label: 'Double shear layer',
      cfg: {
        nx: 384, ny: 384, dt: 0.8, nu: 0, epsVC: 0, omega: 1.9, iters: 50,
        edges: [WALL, WALL, WALL, WALL], slip: 1, dyeMode: 0, dyeDecay: 1,
      },
      stepsPerFrame: 2,
      render: { sDye: 1, sVel: 1.2, sVort: 10, sP: 1 },
      init(sim) {
        const { nx, ny } = sim;
        const u = new Float32Array((nx + 1) * ny);
        const v = new Float32Array(nx * (ny + 1));
        const dye = new Float32Array(nx * ny);
        const k = 60;
        // window the jets to zero near the side walls (box BC needs u=0 there;
        // without this the bands slam into the walls and spray noise)
        const win = x => Math.min(1, Math.min(x, nx - x) / 40);
        for (let j = 0; j < ny; j++) {
          const yn = (j + 0.5) / ny;
          const prof = yn <= 0.5 ? Math.tanh(k * (yn - 0.25)) : Math.tanh(k * (0.75 - yn));
          for (let i = 0; i <= nx; i++) u[j * (nx + 1) + i] = 0.5 * prof * win(i);
          for (let i = 0; i < nx; i++) dye[j * nx + i] = 0.5 + 0.5 * prof;
        }
        for (let j = 0; j <= ny; j++)
          for (let i = 0; i < nx; i++)
            v[j * nx + i] = 0.03 * Math.sin(2 * Math.PI * 4 * (i + 0.5) / nx);
        sim.setFields({ u, v, dye });
      },
    },
    cavity: {
      label: 'Lid-driven cavity (Re=1000)',
      cfg: {
        nx: 256, ny: 256, dt: 0.5, nu: 256 / 1000, epsVC: 0, omega: 1.9, iters: 60,
        lidVel: 1, edges: [WALL, WALL, WALL, WALL], slip: 0, dyeMode: 0, dyeDecay: 1,
      },
      stepsPerFrame: 4,
      render: { sDye: 1, sVel: 1.2, sVort: 15, sP: 3 },
      init(sim) {
        const { nx, ny } = sim;
        const dye = new Float32Array(nx * ny);
        for (let j = 0; j < ny; j++)
          for (let i = 0; i < nx; i++)
            dye[j * nx + i] = ((i / 24 | 0) + (j / 24 | 0)) % 2 ? 0.8 : 0.1;
        sim.setFields({ dye });
      },
    },
  };
}
