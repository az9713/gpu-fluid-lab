// WGSL kernels. MAC staggered grid, h = 1 grid unit.
// u: (nx+1) x ny at (i, j+.5) | v: nx x (ny+1) at (i+.5, j) | scalars: nx x ny at (i+.5, j+.5)
// flags: 0 fluid, 1 solid. edge modes: 0 wall, 1 inflow, 2 outflow.

export const PARAMS = /*wgsl*/`
struct Params {
  nx:u32, ny:u32, edgeL:u32, edgeR:u32,
  edgeB:u32, edgeT:u32, dyeMode:u32, stepIdx:u32,
  dt:f32, nu:f32, epsVC:f32, omega:f32,
  inflow:f32, lidVel:f32, buoy:f32, slip:f32,
  mouseX:f32, mouseY:f32, mouseVX:f32, mouseVY:f32,
  mouseR:f32, mouseAmt:f32, mouseOn:f32, t:f32,
  perturb:f32, dyeDecay:f32, probeX:f32, probeY:f32,
}
@group(0) @binding(0) var<uniform> P: Params;
fn clampf(x:f32,a:f32,b:f32)->f32{ return min(max(x,a),b); }
`;

// bilinear sampler over buffer `buf` whose value (i,j) sits at world pos (i+ox, j+oy)
const bilin = (name, buf, ox, oy, W, H) => /*wgsl*/`
fn ${name}(x:f32, y:f32)->f32 {
  let W = ${W}; let H = ${H};
  let gx = clampf(x - ${ox}, 0.0, f32(W - 1u));
  let gy = clampf(y - ${oy}, 0.0, f32(H - 1u));
  let i0 = min(u32(gx), W - 2u);
  let j0 = min(u32(gy), H - 2u);
  let fx = gx - f32(i0); let fy = gy - f32(j0);
  let a = ${buf}[j0*W+i0];      let b = ${buf}[j0*W+i0+1u];
  let c = ${buf}[(j0+1u)*W+i0]; let d = ${buf}[(j0+1u)*W+i0+1u];
  return mix(mix(a,b,fx), mix(c,d,fx), fy);
}`;

const gather4 = (name, buf, ox, oy, W, H) => /*wgsl*/`
fn ${name}(x:f32, y:f32)->vec4f {
  let W = ${W}; let H = ${H};
  let gx = clampf(x - ${ox}, 0.0, f32(W - 1u));
  let gy = clampf(y - ${oy}, 0.0, f32(H - 1u));
  let i0 = min(u32(gx), W - 2u);
  let j0 = min(u32(gy), H - 2u);
  return vec4f(${buf}[j0*W+i0], ${buf}[j0*W+i0+1u], ${buf}[(j0+1u)*W+i0], ${buf}[(j0+1u)*W+i0+1u]);
}`;

// geometry per field kind
const GEO = {
  U: { ox: '0.0', oy: '0.5', W: '(P.nx+1u)', H: 'P.ny',
       guard: 'if (gid.x > P.nx || gid.y >= P.ny) { return; }',
       pos: 'vec2f(f32(gid.x), f32(gid.y)+0.5)',
       idx: 'gid.y*(P.nx+1u)+gid.x' },
  V: { ox: '0.5', oy: '0.0', W: 'P.nx', H: '(P.ny+1u)',
       guard: 'if (gid.x >= P.nx || gid.y > P.ny) { return; }',
       pos: 'vec2f(f32(gid.x)+0.5, f32(gid.y))',
       idx: 'gid.y*P.nx+gid.x' },
  C: { ox: '0.5', oy: '0.5', W: 'P.nx', H: 'P.ny',
       guard: 'if (gid.x >= P.nx || gid.y >= P.ny) { return; }',
       pos: 'vec2f(f32(gid.x)+0.5, f32(gid.y)+0.5)',
       idx: 'gid.y*P.nx+gid.x' },
};

const VEL_SAMPLERS = /*wgsl*/`
${''}`;

function velSamplers() {
  return bilin('sVelU', 'velU', GEO.U.ox, GEO.U.oy, GEO.U.W, GEO.U.H) +
         bilin('sVelV', 'velV', GEO.V.ox, GEO.V.oy, GEO.V.W, GEO.V.H);
}

// semi-Lagrangian advection (RK2 traceback), DIR=+1 forward / -1 backward
export function advect(kind) {
  const g = GEO[kind];
  return PARAMS + /*wgsl*/`
@group(0) @binding(1) var<storage, read> velU: array<f32>;
@group(0) @binding(2) var<storage, read> velV: array<f32>;
@group(0) @binding(3) var<storage, read> src: array<f32>;
@group(0) @binding(4) var<storage, read_write> dst: array<f32>;
override DIR: f32 = 1.0;
${velSamplers()}
${bilin('sSrc', 'src', g.ox, g.oy, g.W, g.H)}
@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  ${g.guard}
  let pos = ${g.pos};
  let v1 = vec2f(sVelU(pos.x,pos.y), sVelV(pos.x,pos.y));
  let mid = pos - DIR*0.5*P.dt*v1;
  let v2 = vec2f(sVelU(mid.x,mid.y), sVelV(mid.x,mid.y));
  let p2 = pos - DIR*P.dt*v2;
  dst[${g.idx}] = sSrc(p2.x, p2.y);
}`;
}

// MacCormack correction: dst = fwd + 0.5*(orig - back), clamped to bilinear
// neighborhood of orig at the traceback point (prevents overshoot).
export function correct(kind) {
  const g = GEO[kind];
  return PARAMS + /*wgsl*/`
@group(0) @binding(1) var<storage, read> velU: array<f32>;
@group(0) @binding(2) var<storage, read> velV: array<f32>;
@group(0) @binding(3) var<storage, read> orig: array<f32>;
@group(0) @binding(4) var<storage, read> fwd: array<f32>;
@group(0) @binding(5) var<storage, read> back: array<f32>;
@group(0) @binding(6) var<storage, read_write> dst: array<f32>;
${velSamplers()}
${gather4('gOrig', 'orig', g.ox, g.oy, g.W, g.H)}
@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  ${g.guard}
  let pos = ${g.pos};
  let v1 = vec2f(sVelU(pos.x,pos.y), sVelV(pos.x,pos.y));
  let mid = pos - 0.5*P.dt*v1;
  let v2 = vec2f(sVelU(mid.x,mid.y), sVelV(mid.x,mid.y));
  let p2 = pos - P.dt*v2;
  let q = gOrig(p2.x, p2.y);
  let mn = min(min(q.x,q.y), min(q.z,q.w));
  let mx = max(max(q.x,q.y), max(q.z,q.w));
  let i = ${g.idx};
  dst[i] = clampf(fwd[i] + 0.5*(orig[i] - back[i]), mn, mx);
}`;
}

// forces: mouse splat, dye sources, buoyancy, shedding perturbation. In place.
export const FORCES = PARAMS + /*wgsl*/`
@group(0) @binding(1) var<storage, read_write> u: array<f32>;
@group(0) @binding(2) var<storage, read_write> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> dye: array<f32>;
@group(0) @binding(4) var<storage, read> flags: array<u32>;
@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x; let j = gid.y;
  if (i > P.nx || j > P.ny) { return; }
  // u faces
  if (i <= P.nx && j < P.ny && P.mouseOn > 0.5) {
    let dx = f32(i) - P.mouseX; let dy = f32(j)+0.5 - P.mouseY;
    let g = exp(-(dx*dx+dy*dy)/(P.mouseR*P.mouseR));
    u[j*(P.nx+1u)+i] += P.mouseVX * g;
  }
  // v faces
  if (i < P.nx && j <= P.ny) {
    var dv = 0.0;
    if (P.mouseOn > 0.5) {
      let dx = f32(i)+0.5 - P.mouseX; let dy = f32(j) - P.mouseY;
      let g = exp(-(dx*dx+dy*dy)/(P.mouseR*P.mouseR));
      dv += P.mouseVY * g;
    }
    if (P.buoy != 0.0) {
      let jm = max(j,1u)-1u; let jc = min(j, P.ny-1u);
      let davg = 0.5*(dye[jm*P.nx+i] + dye[jc*P.nx+i]);
      dv += P.buoy * P.dt * davg;
    }
    if (P.perturb != 0.0 && f32(i) < 0.3*f32(P.nx)) {
      dv += P.perturb * sin(6.2831853*P.t/600.0) * P.dt;
    }
    v[j*P.nx+i] += dv;
  }
  // dye cells
  if (i < P.nx && j < P.ny) {
    let ci = j*P.nx+i;
    if (flags[ci] == 1u) { dye[ci] = 0.0; return; }
    var d = dye[ci] * P.dyeDecay;
    if (P.mouseOn > 0.5 && P.mouseAmt > 0.0) {
      let dx = f32(i)+0.5 - P.mouseX; let dy = f32(j)+0.5 - P.mouseY;
      let g = exp(-(dx*dx+dy*dy)/(P.mouseR*P.mouseR));
      d += P.mouseAmt * g;
    }
    if (P.dyeMode == 1u && i < 2u) {          // wind-tunnel stripes
      d = select(0.0, 1.0, (j / 10u) % 2u == 0u);
    }
    if (P.dyeMode == 2u) {                    // plume source
      let dx = f32(i)+0.5 - 0.5*f32(P.nx); let dy = f32(j)+0.5 - 6.0;
      if (dx*dx+dy*dy < 49.0) { d = 1.0; }
    }
    dye[ci] = min(d, 2.0);
  }
}`;

// explicit viscous diffusion with no-slip/free-slip wall ghosts (src -> dst)
export const DIFF_U = PARAMS + /*wgsl*/`
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x > P.nx || gid.y >= P.ny) { return; }
  let i = gid.x; let j = gid.y; let W = P.nx+1u;
  let c = src[j*W+i];
  var l = c; if (i > 0u) { l = src[j*W+i-1u]; }
  var r = c; if (i < P.nx) { r = src[j*W+i+1u]; }
  var b = c; var t = c;
  if (j > 0u) { b = src[(j-1u)*W+i]; } else if (P.slip < 0.5) { b = -c; }
  if (j < P.ny-1u) { t = src[(j+1u)*W+i]; } else if (P.slip < 0.5) { t = 2.0*P.lidVel - c; }
  dst[j*W+i] = c + P.nu*P.dt*(l+r+b+t-4.0*c);
}`;

export const DIFF_V = PARAMS + /*wgsl*/`
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= P.nx || gid.y > P.ny) { return; }
  let i = gid.x; let j = gid.y; let W = P.nx;
  let c = src[j*W+i];
  var b = c; if (j > 0u) { b = src[(j-1u)*W+i]; }
  var t = c; if (j < P.ny) { t = src[(j+1u)*W+i]; }
  var l = c; var r = c;
  if (i > 0u) { l = src[j*W+i-1u]; } else if (P.slip < 0.5) { l = -c; }
  if (i < P.nx-1u) { r = src[j*W+i+1u]; } else if (P.slip < 0.5) { r = -c; }
  dst[j*W+i] = c + P.nu*P.dt*(l+r+b+t-4.0*c);
}`;

// cell-centered vorticity  w = dv/dx - du/dy
export const VORT = PARAMS + /*wgsl*/`
@group(0) @binding(1) var<storage, read> u: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> w: array<f32>;
fn uc(i:u32, j:u32)->f32 { return 0.5*(u[j*(P.nx+1u)+i] + u[j*(P.nx+1u)+i+1u]); }
fn vc(i:u32, j:u32)->f32 { return 0.5*(v[j*P.nx+i] + v[(j+1u)*P.nx+i]); }
@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= P.nx || gid.y >= P.ny) { return; }
  let i = gid.x; let j = gid.y;
  let im = max(i,1u)-1u; let ip = min(i+1u, P.nx-1u);
  let jm = max(j,1u)-1u; let jp = min(j+1u, P.ny-1u);
  let dvdx = (vc(ip,j) - vc(im,j)) / f32(ip-im);
  let dudy = (uc(i,jp) - uc(i,jm)) / f32(jp-jm);
  w[j*P.nx+i] = dvdx - dudy;
}`;

// vorticity confinement: f = eps * (N x w z), N = grad|w|/|grad|w||
export const CONFINE = PARAMS + /*wgsl*/`
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> u: array<f32>;
@group(0) @binding(3) var<storage, read_write> v: array<f32>;
${bilin('sW', 'w', '0.5', '0.5', 'P.nx', 'P.ny')}
fn nforce(px:f32, py:f32)->vec2f {
  let wl = abs(sW(px-1.0,py)); let wr = abs(sW(px+1.0,py));
  let wb = abs(sW(px,py-1.0)); let wt = abs(sW(px,py+1.0));
  var N = vec2f(wr-wl, wt-wb)*0.5;
  N = N / (length(N) + 1e-5);
  let w0 = sW(px,py);
  return P.epsVC * vec2f(N.y*w0, -N.x*w0);
}
@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x; let j = gid.y;
  if (i > P.nx || j > P.ny) { return; }
  if (i <= P.nx && j < P.ny && i > 0u && i < P.nx) {
    u[j*(P.nx+1u)+i] += P.dt * nforce(f32(i), f32(j)+0.5).x;
  }
  if (i < P.nx && j <= P.ny && j > 0u && j < P.ny) {
    v[j*P.nx+i] += P.dt * nforce(f32(i)+0.5, f32(j)).y;
  }
}`;

export const DIVERGENCE = PARAMS + /*wgsl*/`
@group(0) @binding(1) var<storage, read> u: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read> flags: array<u32>;
@group(0) @binding(4) var<storage, read_write> d: array<f32>;
@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= P.nx || gid.y >= P.ny) { return; }
  let i = gid.x; let j = gid.y; let ci = j*P.nx+i;
  if (flags[ci] == 1u) { d[ci] = 0.0; return; }
  d[ci] = u[j*(P.nx+1u)+i+1u] - u[j*(P.nx+1u)+i] + v[(j+1u)*P.nx+i] - v[j*P.nx+i];
}`;

// red-black Gauss-Seidel with SOR, solving lap(q) = div. In place on p.
// Neumann at walls/inflow/solids (drop term), Dirichlet 0 at outflow.
export const RBGS = PARAMS + /*wgsl*/`
@group(0) @binding(1) var<storage, read> div: array<f32>;
@group(0) @binding(2) var<storage, read> flags: array<u32>;
@group(0) @binding(3) var<storage, read_write> p: array<f32>;
override PASS: u32 = 0u;
@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let j = gid.y;
  let i = 2u*gid.x + ((j + PASS) & 1u);
  if (i >= P.nx || j >= P.ny) { return; }
  let ci = j*P.nx+i;
  if (flags[ci] == 1u) { return; }
  var s = 0.0; var diag = 0.0;
  if (i == 0u) { if (P.edgeL == 2u) { diag += 1.0; } }
  else if (flags[ci-1u] != 1u) { s += p[ci-1u]; diag += 1.0; }
  if (i == P.nx-1u) { if (P.edgeR == 2u) { diag += 1.0; } }
  else if (flags[ci+1u] != 1u) { s += p[ci+1u]; diag += 1.0; }
  if (j == 0u) { if (P.edgeB == 2u) { diag += 1.0; } }
  else if (flags[ci-P.nx] != 1u) { s += p[ci-P.nx]; diag += 1.0; }
  if (j == P.ny-1u) { if (P.edgeT == 2u) { diag += 1.0; } }
  else if (flags[ci+P.nx] != 1u) { s += p[ci+P.nx]; diag += 1.0; }
  if (diag > 0.0) {
    let pn = (s - div[ci]) / diag;
    p[ci] += P.omega * (pn - p[ci]);
  }
}`;

// subtract pressure gradient on interior faces
export const SUBGRAD = PARAMS + /*wgsl*/`
@group(0) @binding(1) var<storage, read> p: array<f32>;
@group(0) @binding(2) var<storage, read> flags: array<u32>;
@group(0) @binding(3) var<storage, read_write> u: array<f32>;
@group(0) @binding(4) var<storage, read_write> v: array<f32>;
@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x; let j = gid.y;
  if (i > P.nx || j > P.ny) { return; }
  if (i >= 1u && i <= P.nx-1u && j < P.ny) {
    let ui = j*(P.nx+1u)+i;
    if (flags[j*P.nx+i-1u] == 1u || flags[j*P.nx+i] == 1u) { u[ui] = 0.0; }
    else { u[ui] -= p[j*P.nx+i] - p[j*P.nx+i-1u]; }
  }
  if (j >= 1u && j <= P.ny-1u && i < P.nx) {
    let vi = j*P.nx+i;
    if (flags[(j-1u)*P.nx+i] == 1u || flags[j*P.nx+i] == 1u) { v[vi] = 0.0; }
    else { v[vi] -= p[j*P.nx+i] - p[(j-1u)*P.nx+i]; }
  }
}`;

// boundary conditions on domain-edge faces + solid faces
export const BC = PARAMS + /*wgsl*/`
@group(0) @binding(1) var<storage, read> flags: array<u32>;
@group(0) @binding(2) var<storage, read_write> u: array<f32>;
@group(0) @binding(3) var<storage, read_write> v: array<f32>;
@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x; let j = gid.y;
  if (i > P.nx || j > P.ny) { return; }
  let W = P.nx+1u;
  if (i <= P.nx && j < P.ny) {
    let ui = j*W+i;
    if (i == 0u) {
      if (P.edgeL == 0u) { u[ui] = 0.0; }
      else if (P.edgeL == 1u) { u[ui] = P.inflow; }
      else { u[ui] = u[j*W+1u]; }
    } else if (i == P.nx) {
      if (P.edgeR == 0u) { u[ui] = 0.0; }
      else if (P.edgeR == 1u) { u[ui] = P.inflow; }
      else { u[ui] = u[j*W+P.nx-1u]; }
    } else if (flags[j*P.nx+i-1u] == 1u || flags[j*P.nx+i] == 1u) { u[ui] = 0.0; }
  }
  if (i < P.nx && j <= P.ny) {
    let vi = j*P.nx+i;
    if (j == 0u) {
      if (P.edgeB == 0u) { v[vi] = 0.0; }
      else if (P.edgeB == 1u) { v[vi] = P.inflow; }
      else { v[vi] = v[P.nx+i]; }
    } else if (j == P.ny) {
      if (P.edgeT == 0u) { v[vi] = 0.0; }
      else if (P.edgeT == 1u) { v[vi] = P.inflow; }
      else { v[vi] = v[(P.ny-1u)*P.nx+i]; }
    } else if (flags[(j-1u)*P.nx+i] == 1u || flags[j*P.nx+i] == 1u) { v[vi] = 0.0; }
  }
}`;

// validation probe: append v(probeX,probeY) to a time series buffer
export const PROBE = PARAMS + /*wgsl*/`
@group(0) @binding(1) var<storage, read> v: array<f32>;
@group(0) @binding(2) var<storage, read_write> cnt: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
${bilin('sV', 'v', '0.5', '0.0', 'P.nx', '(P.ny+1u)')}
@compute @workgroup_size(1)
fn main() {
  let idx = atomicAdd(&cnt[0], 1u);
  if (idx < arrayLength(&out)) { out[idx] = sV(P.probeX, P.probeY); }
}`;

// fullscreen render: 0 dye, 1 speed, 2 vorticity, 3 pressure
export const RENDER = PARAMS + /*wgsl*/`
struct RP { mode:u32, pad:u32, w:f32, h:f32, sDye:f32, sVel:f32, sVort:f32, sP:f32 }
@group(0) @binding(1) var<uniform> R: RP;
@group(0) @binding(2) var<storage, read> u: array<f32>;
@group(0) @binding(3) var<storage, read> v: array<f32>;
@group(0) @binding(4) var<storage, read> dye: array<f32>;
@group(0) @binding(5) var<storage, read> p: array<f32>;
@group(0) @binding(6) var<storage, read> w: array<f32>;
@group(0) @binding(7) var<storage, read> flags: array<u32>;
${bilin('sU', 'u', '0.0', '0.5', '(P.nx+1u)', 'P.ny')}
${bilin('sV', 'v', '0.5', '0.0', 'P.nx', '(P.ny+1u)')}
${bilin('sC', 'dye', '0.5', '0.5', 'P.nx', 'P.ny')}
${bilin('sPr', 'p', '0.5', '0.5', 'P.nx', 'P.ny')}
${bilin('sW', 'w', '0.5', '0.5', 'P.nx', 'P.ny')}
fn ramp5(t:f32, c0:vec3f, c1:vec3f, c2:vec3f, c3:vec3f, c4:vec3f)->vec3f {
  let x = clampf(t, 0.0, 1.0) * 4.0;
  if (x < 1.0) { return mix(c0, c1, x); }
  if (x < 2.0) { return mix(c1, c2, x-1.0); }
  if (x < 3.0) { return mix(c2, c3, x-2.0); }
  return mix(c3, c4, x-3.0);
}
fn inferno(t:f32)->vec3f {
  return ramp5(t, vec3f(0.001,0.0,0.014), vec3f(0.28,0.04,0.42),
    vec3f(0.73,0.21,0.33), vec3f(0.98,0.55,0.04), vec3f(0.99,1.0,0.64));
}
fn viridis(t:f32)->vec3f {
  return ramp5(t, vec3f(0.27,0.005,0.33), vec3f(0.23,0.32,0.55),
    vec3f(0.13,0.57,0.55), vec3f(0.37,0.79,0.38), vec3f(0.99,0.91,0.14));
}
fn coolwarm(t:f32)->vec3f {
  return ramp5(t, vec3f(0.23,0.30,0.75), vec3f(0.55,0.69,0.99),
    vec3f(0.87,0.87,0.87), vec3f(0.96,0.60,0.49), vec3f(0.71,0.02,0.15));
}
@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  var pts = array<vec2f,3>(vec2f(-1.0,-3.0), vec2f(-1.0,1.0), vec2f(3.0,1.0));
  return vec4f(pts[vi], 0.0, 1.0);
}
@fragment
fn fs(@builtin(position) fc: vec4f) -> @location(0) vec4f {
  let x = fc.x / R.w * f32(P.nx);
  let y = (1.0 - fc.y / R.h) * f32(P.ny);
  let ci = min(u32(clampf(x,0.0,f32(P.nx)-1.0)), P.nx-1u);
  let cj = min(u32(clampf(y,0.0,f32(P.ny)-1.0)), P.ny-1u);
  if (flags[cj*P.nx+ci] == 1u) { return vec4f(0.13, 0.15, 0.18, 1.0); }
  var col = vec3f(0.0);
  if (R.mode == 0u) {
    col = inferno(sC(x,y) * R.sDye);
  } else if (R.mode == 1u) {
    let sp = length(vec2f(sU(x,y), sV(x,y)));
    col = viridis(sp * R.sVel);
  } else if (R.mode == 2u) {
    col = coolwarm(0.5 + 0.5*clampf(sW(x,y) * R.sVort, -1.0, 1.0));
  } else {
    col = coolwarm(0.5 + 0.5*clampf(sPr(x,y) * R.sP, -1.0, 1.0));
  }
  return vec4f(col, 1.0);
}`;
