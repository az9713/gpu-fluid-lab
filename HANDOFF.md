# HANDOFF — resume point for GPU Fluid Dynamics Lab

**Read this first each new session.** This file is the live "what's done / what's next".
The README covers architecture and validation details — don't re-derive them.

## Current state (as of `8bc777f`, 2026-08-02 — everything pushed, DONE)

The project is **complete and deployed**. All work from projects-overview.html §3 shipped:

- **Repo**: https://github.com/az9713/gpu-fluid-lab (branch `main`)
- **Live app**: https://az9713.github.io/gpu-fluid-lab/ (GitHub Pages, branch main /root)
- **Validation**: all 4 benchmarks PASS (projection 2531×, Ghia Re=100 RMS 0.0020/0.0043,
  Re=1000 RMS 0.0042/0.0043, Strouhal 9.6% err vs ±15% tol). ~19 min full, `?fast` ~4 min.
- **Tutorial**: `navier-stokes.html` — term-by-term NS explainer (483 eqs, 530 SVGs),
  linked from header + README, built via rigorous-explainer skill, hardened.
- Feature timeline (all in git log): core solver+UI (`91726e1`) → README gallery
  (`34c09b1`) → `?preset=&view=` URL params (`19f0156`, `7283cb2`) → cache-busting `?v=N`
  (`a433d50`) → validation progress cards (`9bc2afd`) → per-preset equation explainers
  (`657cc68`) → view-aware explainer sections (`0270d40`) → tutorial (`8bc777f`).
- Working tree clean except untracked `.ignore/` (intentional noise, leave it).

## Next task

- **None pending — awaiting user direction.** Likely candidates the user has touched on:
  1. **Build project 2 of projects-overview.html** (Market Microstructure Observatory /
     live order book). User asked whether trading signals + brief explanations can be
     embedded — answer given: yes (OFI, microprice, spread-regime, tape momentum, each
     with one-line rationale + live hit-rate scoring vs simulated future). If asked to
     build: matching engine + agents first, validate stylized facts (fat tails, vol
     clustering, sqrt impact), then signal layer. Start fresh session, new folder/repo.
  2. Fluid lab polish (more presets, multigrid, video export) — all deliberately skipped
     as YAGNI; only on request.

## Gotchas (project-specific)

- **GitHub Pages caches 10 min** (`max-age=600`). After any deploy, bump `?v=N` on
  `src/main.js` in index.html AND the import specifiers inside main/solver/validate
  (they cascade; currently v=4 for main.js, v=3 validate.js, v=2 inner imports —
  mismatched numbers are fine, they just need to CHANGE when the file changes).
- Serve locally with `python -m http.server` (ES modules break on file://). A server may
  still be running on port 8791 from the build session.
- Validation must run in Chrome/Edge (WebGPU). Headless check: page logs
  `VALIDATION_JSON {...}` to console and sets `window.__validation`.
- `tools/gen_ns_figs.py` regenerates the tutorial's computed SVG figures, splicing
  between HTML markers (idempotent). The committed `navier-stokes.html` is the durable
  record — edit it directly for text changes.

## Where to read things

- `README.md` — architecture, numerics, validation table with measured values, layout.
- `projects-overview.html` — the 5-project spec this came from (only §3 built).
- Memory `fluid-lab-status.md` — cross-session facts (validation runtimes, perf).
