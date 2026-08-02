# GPU Fluid Dynamics Lab

Real-time 2D incompressible Navier–Stokes, solved entirely in WebGPU compute shaders and
validated against published CFD benchmarks. Runs in Chrome/Edge 113+, deployable as a
static site (GitHub Pages friendly) — no build step, no dependencies.

**▶ Live: [az9713.github.io/gpu-fluid-lab](https://az9713.github.io/gpu-fluid-lab/)** ·
[equations tutorial](https://az9713.github.io/gpu-fluid-lab/navier-stokes.html) ·
[validation suite](https://az9713.github.io/gpu-fluid-lab/validation.html)
(locally: serve the folder over HTTP, e.g. `python -m http.server` — ES modules don't load from `file://`).

## Gallery — click any image to run it live

| | |
|---|---|
| [![Kármán vortex street](screenshots/karman-vortex-street.png)](https://az9713.github.io/gpu-fluid-lab/?preset=tunnel) <br>*Kármán vortex street — dye stripes, Re≈200* | [![Vorticity view](screenshots/vorticity-view.png)](https://az9713.github.io/gpu-fluid-lab/?preset=tunnel&view=2) <br>*Same wake in the vorticity view* |
| [![Rayleigh–Taylor](screenshots/rayleigh-taylor.png)](https://az9713.github.io/gpu-fluid-lab/?preset=rt) <br>*Rayleigh–Taylor mushroom fingers* | [![Kelvin–Helmholtz](screenshots/kelvin-helmholtz.png)](https://az9713.github.io/gpu-fluid-lab/?preset=shear&view=2) <br>*Kelvin–Helmholtz vortex chains* |
| [![Buoyant plume](screenshots/buoyant-plume.png)](https://az9713.github.io/gpu-fluid-lab/?preset=plume) <br>*Buoyant smoke plume* | [![Validation suite](screenshots/validation-suite.png)](https://az9713.github.io/gpu-fluid-lab/validation.html) <br>*Validation suite — all benchmarks green* |

## What you get

- **Wind tunnel** — Kármán vortex street behind a cylinder at Re≈200, with approximate
  pressure drag/lift readouts. Right-drag to add obstacles.
- **Rayleigh–Taylor instability** — heavy-over-light overturn with mushroom fingers and
  secondary Kelvin–Helmholtz roll-ups.
- **Buoyant plume** — continuous hot source, turbulent rise (Boussinesq buoyancy).
- **Double shear layer** — Kelvin–Helmholtz vortex chains.
- **Lid-driven cavity (Re=1000)** — the classic CFD benchmark, live.
- Render modes: dye, speed, vorticity, pressure. Mouse force/dye injection, obstacle
  painting, screenshot export, physics explainer overlay.

## Numerics

- **Staggered MAC grid** (pressure at cell centers, velocity on faces) — no
  checkerboard null mode.
- **MacCormack/BFECC advection**: semi-Lagrangian RK2 traceback forward + backward,
  error-corrected, min/max-limited. Second-order, far crisper than plain Stam advection.
- **Chorin projection**: pressure Poisson equation solved with red–black Gauss–Seidel +
  SOR (ω ≈ 2/(1+sin πh)), warm-started across steps; gradient subtracted on faces.
- **Explicit viscous diffusion** with no-slip/free-slip wall ghosts (2u_wall − u ghost
  values; the moving lid enters here).
- **Vorticity confinement** (ε · N̂ × ω) to re-inject subgrid swirl in smoke presets.
- Boundary types per edge: wall, inflow, outflow (Dirichlet p=0); solid obstacles via
  cell flags.

## Validation (`validation.html`)

Failing any benchmark turns the banner red — this is the CI gate.

| Test | Reference | Tolerance | Measured (RTX-class GPU) |
|---|---|---|---|
| Projection sanity | divergence of random field must drop >100× in one step | ratio > 100 | 2531× |
| Lid-driven cavity Re=100 (128²) | Ghia, Ghia & Shin (1982) centerline u/v profiles | RMS < 0.015 | 0.0020 / 0.0043 |
| Lid-driven cavity Re=1000 (224²) | same | RMS < 0.02 | 0.0042 / 0.0043 |
| Vortex shedding Re=150 (768×384) | Strouhal relation St = 0.198(1 − 19.7/Re) | ±15% | St 0.189 vs 0.172 (9.6%; wall-blockage bias raises St) |

The full suite takes ~20 min on a mid-range GPU (`?fast` for a smoke run). Results are
printed to the page, exposed as `window.__validation`, and logged as a
`VALIDATION_JSON {...}` console line for headless scraping.

## Layout

```
index.html            UI shell
navier-stokes.html    self-contained tutorial: the equations term by term,
                      derived + specialized per demo (MathJax + inline SVG)
tools/gen_ns_figs.py  regenerates the tutorial's computed figures
validation.html       benchmark runner
src/shaders.js        all WGSL kernels (advection, projection, forces, render)
src/solver.js         Sim class (buffers, pipelines, step encoding) + presets
src/webgpu.js         device/buffer/pipeline helpers
src/validate.js       benchmark implementations
src/ghia.js           Ghia et al. (1982) reference tables
src/main.js           UI wiring, mouse, render loop
```

## Notes

- All quantities are in grid units (h = 1, lid/inflow speed = 1); Reynolds numbers are
  set via ν = U·L/Re.
- fp32 throughout; conserved-quantity accuracy is what the validation suite measures.
- Video export was deliberately skipped (screenshot button exists; canvas
  `captureStream()` is the upgrade path if ever needed).
