# Generates the computed figures for navier-stokes.html and splices them
# between <!--FIG:key--> ... <!--/FIG:key--> markers. Re-run after any tweak:
#   python tools/gen_ns_figs.py
# All geometry, plot polylines and SMIL keyframes are computed here, never
# eyeballed. Prints the punchline numbers each figure must show.
import math
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HTML = os.path.join(HERE, "..", "navier-stokes.html")

FIGS = {}


def pts(seq):
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in seq)


def arrow(x0, y0, x1, y1, color, w=2.4, marker=None, dash=None, cls=None):
    m = marker or {"#2a7d2a": "a_grn", "#7a1f1f": "a_red",
                   "#1f4e8c": "a_blu", "#666": "a_gry",
                   "#c0392b": "a_vred", "#2e6da4": "a_vblu"}[color]
    d = f' stroke-dasharray="{dash}"' if dash else ""
    c = f' class="{cls}"' if cls else ""
    return (f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}"'
            f' stroke="{color}" stroke-width="{w}"{d}{c}'
            f' marker-end="url(#{m})"/>')


def text(x, y, s, size=12.0, color="#333", cls=None, anchor=None, extra=""):
    c = f' class="{cls}"' if cls else ""
    a = f' text-anchor="{anchor}"' if anchor else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}"'
            f' fill="{color}"{c}{a}{extra}>{s}</text>')


# ---------------------------------------------------------------- gradient
def fig_grad():
    cx, cy, sig = 130.0, 118.0, 48.0
    out = []
    # contours of T = exp(-r^2 / 2 sig^2): circles r = sig*sqrt(2 ln(1/T))
    levels = [0.8, 0.6, 0.4, 0.2]
    radii = [sig * math.sqrt(2.0 * math.log(1.0 / T)) for T in levels]
    for r, T in zip(radii, levels):
        out.append(f'<circle cx="{cx}" cy="{cy}" r="{r:.1f}" fill="none"'
                   f' stroke="#b8860b" stroke-width="1.3" opacity="0.75"/>')
    # gradient arrows: sample points on rays, arrows point INWARD (uphill),
    # length proportional to |grad T| = (r/sig^2) exp(-r^2/2 sig^2)
    def gmag(r):
        return (r / sig**2) * math.exp(-r * r / (2 * sig * sig))
    gmax = gmag(sig)
    samples = [(radii[3], 10), (radii[2], 10), (radii[1], 10),
               (radii[0], 190), (radii[2], 190), (radii[3], 190),
               (radii[2], 300), (radii[3], 300), (radii[1], 300),
               (radii[3], 80), (radii[2], 80)]
    for r, ang_deg in samples:
        a = math.radians(ang_deg)
        px = cx + r * math.cos(a)
        py = cy + r * math.sin(a)
        L = 30.0 * gmag(r) / gmax
        ex = px - L * math.cos(a)
        ey = py - L * math.sin(a)
        out.append(arrow(px, py, ex, ey, "#2a7d2a", w=2.2))
    out.append(text(cx, cy + 4, "hot", 11, "#7a1f1f", anchor="middle"))
    out.append(text(cx + 92, cy - 68, "cold", 11, "#555"))
    out.append(text(cx, cy + radii[3] + 18, "gold circles: T = const"
                    " contours", 11, "#b8860b", anchor="middle"))
    gx = cx + radii[2] * math.cos(math.radians(10))
    gy = cy + radii[2] * math.sin(math.radians(10))
    out.append(text(gx + 8, gy - 12, "&#8711;T", 13, "#2a7d2a", cls="v"))
    out.append(text(288, 70, "green arrows: &#8711;T,", 11, "#555"))
    out.append(text(288, 86, "always uphill, longest where", 11, "#555"))
    out.append(text(288, 102, "contours crowd (the flank,", 11, "#555"))
    out.append(text(288, 118, "not the flat peak or far field)", 11,
                    "#555"))
    print(f"[grad] contour radii px: "
          f"{', '.join(f'{r:.1f}' for r in radii)}; "
          f"|grad| max at r = sigma = {sig:.0f}px")
    FIGS["grad"] = svg(out, 460, 235, "temperature hill: contours and "
                       "gradient arrows pointing uphill")


# -------------------------------------------------------------- divergence
def fig_div():
    out = []
    for (cx, cy, kind) in [(115.0, 96.0, "src"), (345.0, 96.0, "rot")]:
        for r in (26.0, 54.0):
            for k in range(8):
                a = 2 * math.pi * (k + (0.5 if r > 30 else 0.0)) / 8
                px = cx + r * math.cos(a)
                py = cy + r * math.sin(a)
                if kind == "src":         # u = c*(x-cx, y-cy), div = 2c > 0
                    c = 0.36
                    ex, ey = px + c * (px - cx), py + c * (py - cy)
                    col = "#7a1f1f"
                else:                     # u = Omega x r, div = 0
                    Om = 0.36
                    ex = px - Om * (py - cy)
                    ey = py + Om * (px - cx)
                    col = "#1f4e8c"
                out.append(arrow(px, py, ex, ey, col, w=2.0))
        out.append(f'<circle cx="{cx}" cy="{cy}" r="40" fill="none"'
                   f' stroke="#666" stroke-width="1.3"'
                   f' stroke-dasharray="5 4"/>')
        out.append(f'<circle cx="{cx}" cy="{cy}" r="2.6" fill="#333"/>')
    out.append(text(115, 178, "&#8711;&#183;u &gt; 0 &#8212; net outflow",
                    12, "#7a1f1f", anchor="middle"))
    out.append(text(345, 178, "&#8711;&#183;u = 0 &#8212; spins, "
                    "nothing accumulates", 12, "#1f4e8c", anchor="middle"))
    print("[div] source: div = 2c = 0.72 per unit; rotation: div = 0 exactly")
    FIGS["div"] = svg(out, 460, 195, "a source field with positive "
                      "divergence beside a rotating field with zero "
                      "divergence")


# -------------------------------------------------------------------- curl
def fig_curl():
    out = []
    # left: shear u = k*(y_up - y0): horizontal arrows, top row fast right
    k = 0.62
    y0 = 100.0
    for ys in (40.0, 70.0, 100.0, 130.0, 160.0):
        yup = y0 - ys                      # screen y down -> flip
        u = k * yup
        if abs(u) < 1e-9:
            out.append(f'<circle cx="90" cy="{ys}" r="2.4" fill="#666"/>')
            continue
        out.append(arrow(90.0, ys, 90.0 + u, ys, "#1f4e8c", w=2.2))
    # paddle wheel at (215, 100): faster flow above drags top spoke right
    px, py, pr = 215.0, 100.0, 20.0
    out.append(f'<circle cx="{px}" cy="{py}" r="{pr}" fill="#f6f4ef"'
               f' stroke="#333" stroke-width="1.6"/>')
    for adeg in (0, 45, 90, 135):
        a = math.radians(adeg)
        out.append(f'<line x1="{px - pr * math.cos(a):.1f}"'
                   f' y1="{py - pr * math.sin(a):.1f}"'
                   f' x2="{px + pr * math.cos(a):.1f}"'
                   f' y2="{py + pr * math.sin(a):.1f}"'
                   f' stroke="#333" stroke-width="1.4"/>')
    # clockwise rotation arc (on screen, y-up convention: omega < 0)
    r2 = pr + 9
    out.append(f'<path d="M {px - r2:.1f},{py:.1f} A {r2} {r2} 0 0 1'
               f' {px:.1f},{py - r2:.1f}" fill="none" stroke="#2e6da4"'
               f' stroke-width="2.2" marker-end="url(#a_vblu)"/>')
    out.append(text(60, 26, "shear flow: &#969; = &#8722;&#8706;u/&#8706;y"
                    " &lt; 0, wheel spins clockwise (blue in the lab)",
                    11.5, "#2e6da4"))
    # right: rigid rotation, counterclockwise, omega = 2*Omega > 0
    cx, cy = 375.0, 100.0
    Om = 0.42
    for r in (22.0, 46.0):
        for kk in range(8):
            a = 2 * math.pi * (kk + (0.5 if r > 30 else 0.0)) / 8
            qx = cx + r * math.cos(a)
            qy = cy + r * math.sin(a)
            # CCW with y up == CW in screen coords? No: flip sign of screen
            # tangent. u = Omega z-hat x r; screen y is -y_up, so tangent
            # (-(qy-cy), (qx-cx)) in y-up becomes (+(qy-cy), ... ) flipped:
            ex = qx + Om * (qy - cy)
            ey = qy - Om * (qx - cx)
            out.append(arrow(qx, qy, ex, ey, "#c0392b", w=2.0))
    out.append(text(cx, 178, "rigid rotation: &#969; = 2&#937; &gt; 0"
                    " (red)", 11.5, "#c0392b", anchor="middle"))
    print("[curl] shear vorticity = -k =", -k,
          "per px; rigid-body vorticity = 2*Omega =", 2 * Om)
    FIGS["curl"] = svg(out, 460, 195, "a paddle wheel spun clockwise by "
                       "shear beside a counter-clockwise rigid rotation")


# --------------------------------------------------------------- laplacian
def fig_lap():
    x0, x1, ybase, ytop = 40.0, 430.0, 175.0, 30.0

    def f(x):
        return (0.42 + 0.34 * math.exp(-((x - 0.30) / 0.13) ** 2)
                - 0.26 * math.exp(-((x - 0.74) / 0.11) ** 2))

    def X(x):
        return x0 + (x1 - x0) * x

    def Y(v):
        return ybase - (ybase - ytop) * v

    curve = [(X(x), Y(f(x))) for x in np.linspace(0, 1, 80)]
    out = [f'<polyline points="{pts(curve)}" fill="none" stroke="#7a1f1f"'
           f' stroke-width="2.4"/>']
    out.append(f'<line x1="{x0}" y1="{ybase}" x2="{x1}" y2="{ybase}"'
               f' stroke="#999" stroke-width="1.2"/>')
    dx = 0.085
    for xc, lab, col in [(0.30, "peak", "#1f4e8c"), (0.74, "dip", "#2a7d2a")]:
        fc = f(xc)
        avg = 0.5 * (f(xc - dx) + f(xc + dx))
        for xs in (xc - dx, xc + dx):
            out.append(f'<circle cx="{X(xs):.1f}" cy="{Y(f(xs)):.1f}"'
                       f' r="3.2" fill="#b8860b"/>')
        out.append(f'<circle cx="{X(xc):.1f}" cy="{Y(fc):.1f}" r="3.6"'
                   f' fill="{col}"/>')
        out.append(f'<circle cx="{X(xc):.1f}" cy="{Y(avg):.1f}" r="3.6"'
                   f' fill="none" stroke="{col}" stroke-width="1.8"/>')
        out.append(arrow(X(xc), Y(fc), X(xc), Y(avg) - (6 if fc > avg
                         else -6), col, w=2.2))
        print(f"[lap] {lab}: f = {fc:.3f}, neighbour avg = {avg:.3f}, "
              f"nabla2 sign = {'-' if fc > avg else '+'}")
    out.append(text(X(0.30) + 12, Y(f(0.30)) - 8,
                    "above its neighbours &#8594; &#8711;&#178;f &lt; 0,"
                    " pulled down", 11, "#1f4e8c"))
    out.append(text(350.0, 104.0,
                    "below &#8594; &#8711;&#178;f &gt; 0, filled in",
                    11, "#2a7d2a", anchor="middle"))
    out.append(text(X(0.06), Y(f(0.06)) - 10, "f(x)", 12, "#7a1f1f",
                    cls="v"))
    out.append(text(46, ybase + 16, "solid dot: f at the point &#183; "
                    "hollow dot: average of the gold neighbours",
                    10.5, "#555"))
    FIGS["lap"] = svg(out, 460, 205, "a curve with a peak pulled toward "
                      "and a dip pushed toward the neighbour average",
                      vb="20 40 439 160")


# ------------------------------------------------- nozzle + moving parcel
def fig_nozzle():
    xa, xb = 40.0, 420.0     # channel span on screen
    ycl = 88.0               # centerline
    w0, w1 = 50.0, 23.0      # half-widths -> speed ratio w0/w1
    tx0, tx1 = 140.0, 300.0  # taper span

    def w(x):
        if x < tx0:
            return w0
        if x > tx1:
            return w1
        s = (x - tx0) / (tx1 - tx0)
        return w0 + (w1 - w0) * 0.5 * (1 - math.cos(math.pi * s))

    def u(x):
        return w0 / w(x)     # u_in = 1 in channel units

    top = [(x, ycl - w(x)) for x in np.linspace(xa, xb, 60)]
    bot = [(x, ycl + w(x)) for x in np.linspace(xa, xb, 60)]
    out = [f'<polyline points="{pts(top)}" fill="none" stroke="#333"'
           f' stroke-width="2.6"/>',
           f'<polyline points="{pts(bot)}" fill="none" stroke="#333"'
           f' stroke-width="2.6"/>']
    # speed arrows on the centerline (length ∝ u)
    for x in (70.0, 170.0, 250.0, 330.0, 400.0):
        L = 16.0 * u(x)
        out.append(arrow(x - L / 2, ycl, x + L / 2, ycl, "#1f4e8c", w=2.2))
    # parcel trajectory: integrate dx/dt = s*u(x), fine Euler
    speed = 55.0             # px/s at inlet
    T, dt = 0.0, 1e-3
    x = xa + 6.0
    times, xs = [0.0], [x]
    while x < xb - 8.0:
        x += speed * u(x) * dt
        T += dt
        times.append(T)
        xs.append(x)
    n = 40
    tt = np.linspace(0, T, n)
    xi = np.interp(tt, times, xs)
    vals = "; ".join(f"{xv - xs[0]:.1f} 0" for xv in xi)
    kt = ";".join(f"{t / T:.3f}" for t in tt)
    out.append(
        f'<g><circle cx="{xs[0]:.1f}" cy="{ycl}" r="7" fill="#7a1f1f"/>'
        f'<animateTransform attributeName="transform" type="translate" '
        f'values="{vals}" keyTimes="{kt}" calcMode="linear" '
        f'dur="{T:.2f}s" repeatCount="indefinite"/></g>')
    out.append(text(60, 24, "same parcel, steady channel:"
                    " &#8706;u/&#8706;t = 0 everywhere, yet the parcel"
                    " accelerates", 11.5, "#555"))
    out.append(text(392, ycl - w1 - 10, f"u &#215; {u(xb):.1f}", 11,
                    "#1f4e8c", cls="v", anchor="end"))
    out.append(text(52, ycl - 22, "u = U", 11, "#1f4e8c", cls="v"))
    # chart: u(x) below
    cy0, cy1 = 232.0, 172.0
    xg = np.linspace(xa, xb, 60)
    ug = [u(x) for x in xg]
    umax = max(ug)
    crv = [(x, cy0 - (cy0 - cy1) * (uu / umax)) for x, uu in zip(xg, ug)]
    out.append(f'<line x1="{xa}" y1="{cy0}" x2="{xb}" y2="{cy0}"'
               f' stroke="#999" stroke-width="1.2"/>')
    out.append(f'<polyline points="{pts(crv)}" fill="none"'
               f' stroke="#2a7d2a" stroke-width="2.2"/>')
    out.append(text(xa, cy1 - 6, "u(x) along the centerline", 11,
                    "#2a7d2a"))
    out.append(text(xb + 4, cy0 + 4, "x", 12, "#555", cls="v"))
    print(f"[nozzle] u_out/u_in = {u(xb):.2f} (width {w0:.0f} -> {w1:.0f});"
          f" traversal {T:.2f}s, {n} SMIL keyframes")
    FIGS["nozzle"] = svg(out, 460, 250, "a parcel accelerating through a "
                         "narrowing channel while the field stays steady")


# ------------------------------------------- projection before/after (FFT)
def fig_proj():
    n = 48
    k1 = 2 * np.pi * np.fft.fftfreq(n)
    kx, ky = k1[None, :], k1[:, None]
    k2 = kx**2 + ky**2
    rng = np.random.default_rng(7)
    lp = np.exp(-14.0 * k2)          # keep only long wavelengths
    u = np.real(np.fft.ifft2(lp * np.fft.fft2(rng.standard_normal((n, n)))))
    v = np.real(np.fft.ifft2(lp * np.fft.fft2(rng.standard_normal((n, n)))))
    sc = 1.0 / np.hypot(u, v).max()
    u, v = u * sc, v * sc

    def div(u, v):
        return np.real(np.fft.ifft2(1j * kx * np.fft.fft2(u)
                                    + 1j * ky * np.fft.fft2(v)))

    d0 = div(u, v)
    k2s = k2.copy()
    k2s[0, 0] = 1.0
    ph = np.fft.fft2(d0) / (-k2s)
    ph[0, 0] = 0.0
    u2 = u - np.real(np.fft.ifft2(1j * kx * ph))
    v2 = v - np.real(np.fft.ifft2(1j * ky * ph))
    d1 = div(u2, v2)
    print(f"[proj] max|div| before = {np.abs(d0).max():.3f}, "
          f"after = {np.abs(d1).max():.2e} "
          f"(drop x{np.abs(d0).max() / np.abs(d1).max():.1e})")
    out = []
    dmax = np.abs(d0).max()

    def panel(px0, py0, uu, vv, dd, title, tcol):
        cell = 16.0
        stp = n // 12
        for j in range(6):
            for i in range(12):
                dv = dd[j * stp + stp // 2, i * stp + stp // 2] / dmax
                if dv >= 0:
                    col = f"rgb(255,{int(255 - 130 * dv)}," \
                          f"{int(255 - 150 * dv)})"
                else:
                    col = f"rgb({int(255 + 150 * dv)}," \
                          f"{int(255 + 110 * dv)},255)"
                out.append(f'<rect x="{px0 + i * cell:.1f}"'
                           f' y="{py0 + j * cell:.1f}" width="{cell}"'
                           f' height="{cell}" fill="{col}"/>')
        for j in range(6):
            for i in range(12):
                cx = px0 + (i + 0.5) * cell
                cy = py0 + (j + 0.5) * cell
                uu_ = uu[j * stp + stp // 2, i * stp + stp // 2]
                vv_ = vv[j * stp + stp // 2, i * stp + stp // 2]
                L = 9.5
                out.append(f'<line x1="{cx - L * uu_:.1f}"'
                           f' y1="{cy + L * vv_:.1f}"'
                           f' x2="{cx + L * uu_:.1f}"'
                           f' y2="{cy - L * vv_:.1f}" stroke="#222"'
                           f' stroke-width="1.5"'
                           f' marker-end="url(#a_sm)"/>')
        out.append(f'<rect x="{px0}" y="{py0}" width="192" height="96"'
                   f' fill="none" stroke="#666" stroke-width="1.2"/>')
        out.append(text(px0 + 96, py0 - 8, title, 12, tcol,
                        anchor="middle"))

    panel(25, 42, u, v, d0, "before: &#8711;&#183;u* &#8800; 0", "#7a1f1f")
    panel(243, 42, u2, v2, d1, "after: &#8711;&#183;u = 0", "#2a7d2a")
    out.append(text(230, 168, "red cells: fluid piling up &#183; blue:"
                    " draining &#183; white: balanced &#8212; the arrows"
                    " barely change", 11, "#555", anchor="middle"))
    FIGS["proj"] = svg(out, 460, 182, "the same velocity field before and "
                       "after pressure projection removes its divergence")
    return np.abs(d0).max(), np.abs(d1).max()


# --------------------------------------------------------- demo schematics
def fig_tunnel():
    out = []
    X0, X1, Y0, Y1 = 20.0, 440.0, 40.0, 200.0
    out.append(f'<rect x="{X0}" y="{Y0}" width="{X1 - X0}"'
               f' height="{Y1 - Y0}" fill="#fbfaf7" stroke="#333"'
               f' stroke-width="1.6"/>')
    for ys in (70.0, 120.0, 170.0):
        out.append(arrow(X0 - 14, ys, X0 + 22, ys, "#1f4e8c", w=2.2))
    cx, cy, r = 106.0, 120.0, 13.0    # scale: 512x256 grid -> 420x160 px
    out.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#4a4f57"/>')
    lam = 70.0
    # the street is a repeating <pattern> filling a wake rect: it tiles
    # seamlessly, its bbox is just the rect (nothing spills the viewBox),
    # and animating patternTransform drifts it one wavelength per loop.
    wx0, wy0 = cx + 2 * r + 8.0, Y0 + 2      # wake rect (starts past cyl)
    ww, wh = X1 - wx0 - 2, Y1 - Y0 - 4
    tile = []
    for (vx, vy, col, cw) in [(lam * 0.25, 100.0 - wy0, "#2e6da4", 1),
                              (lam * 0.75, 142.0 - wy0, "#c0392b", 0)]:
        sweep = 1 if cw else 0
        tile.append(f'<circle cx="{vx}" cy="{vy}" r="13" fill="{col}"'
                    f' fill-opacity="0.16"/>')
        tile.append(f'<path d="M {vx - 9},{vy} A 9 9 0 1 {sweep}'
                    f' {vx + 9},{vy}" fill="none" stroke="{col}"'
                    f' stroke-width="2"'
                    f' marker-end="url(#{"a_vblu" if cw else "a_vred"})"/>')
    out.append(
        f'<pattern id="t9pat" x="{wx0}" y="{wy0}" width="{lam}"'
        f' height="{wh}" patternUnits="userSpaceOnUse">'
        + "".join(tile) +
        f'<animateTransform attributeName="patternTransform"'
        f' type="translate" from="0 0" to="-{lam} 0" dur="3s"'
        f' repeatCount="indefinite"/></pattern>')
    out.append(f'<rect x="{wx0}" y="{wy0}" width="{ww}" height="{wh}"'
               f' fill="url(#t9pat)"/>')
    out.append(text(X0 + 4, Y0 - 8, "free-slip wall (fluid slides, no"
                    " drag)", 10.5, "#555"))
    out.append(text(X0 - 14, Y1 + 16, "inflow u = (U, 0)", 10.5,
                    "#1f4e8c"))
    out.append(text(X1, Y1 + 16, "outflow p = 0", 10.5, "#7a1f1f",
                    anchor="end"))
    out.append(text(235, Y1 + 16, "no-slip cylinder, D = 32 cells", 10.5,
                    "#4a4f57", anchor="middle"))
    out.append(text(300, 62, "blue: clockwise &#969; &lt; 0", 10.5,
                    "#2e6da4"))
    out.append(text(300, 188, "red: counter-clockwise &#969; &gt; 0",
                    10.5, "#c0392b"))
    print("[tunnel] Re = U*D/nu = 1*32/0.16 =", 1 * 32 / 0.16)
    FIGS["tunnel"] = svg(out, 460, 226, "wind tunnel: inflow, no-slip "
                         "cylinder and the alternating vortex street "
                         "drifting downstream")


def fig_rt():
    out = []
    X0, X1, Y0, Y1 = 90.0, 370.0, 30.0, 205.0
    # interface: heavy dye above y_int; 3 sine periods as in the preset
    amp = 14.0     # exaggerated for legibility (5/384 of the box in code)
    ymid = 0.5 * (Y0 + Y1)
    iface = [(x, ymid + amp * math.sin(2 * math.pi * 3
                                       * (x - X0) / (X1 - X0)))
             for x in np.linspace(X0, X1, 80)]
    top_poly = pts([(X0, Y0)] + [(X1, Y0)])
    out.append(f'<path d="M {X0},{Y0} L {X1},{Y0} L {X1},{iface[-1][1]:.1f}'
               + " ".join(f" L {x:.1f},{y:.1f}" for x, y in reversed(iface))
               + ' Z" fill="#7a1f1f" fill-opacity="0.30"/>')
    out.append(f'<rect x="{X0}" y="{Y0}" width="{X1 - X0}"'
               f' height="{Y1 - Y0}" fill="none" stroke="#333"'
               f' stroke-width="1.6"/>')
    out.append(f'<polyline points="{pts(iface)}" fill="none"'
               f' stroke="#7a1f1f" stroke-width="2"/>')
    # arrows: heavy troughs sink, light crests rise. Crest of sin at 1/12
    # and trough at 3/12 of each period.
    per = (X1 - X0) / 3
    for k in range(3):
        xd = X0 + per * (k + 0.25)      # sin = +1 -> interface pushed DOWN
        xu = X0 + per * (k + 0.75)      # sin = -1 -> pushed UP
        out.append(arrow(xd, ymid + amp + 4, xd, ymid + amp + 30,
                         "#7a1f1f", w=2.4))
        out.append(arrow(xu, ymid - amp - 4, xu, ymid - amp - 30,
                         "#1f4e8c", w=2.4))
    out.append(text((X0 + X1) / 2, Y0 + 16, "heavy fluid  s = 1", 11.5,
                    "#7a1f1f", anchor="middle"))
    out.append(text((X0 + X1) / 2, Y1 - 8, "light fluid  s = 0", 11.5,
                    "#1f4e8c", anchor="middle"))
    out.append(arrow(408.0, 80.0, 408.0, 130.0, "#666", w=2.4))
    out.append(text(416, 108, "f = Bs&#375;,", 11, "#666", cls="v"))
    out.append(text(416, 122, "B &lt; 0", 11, "#666", cls="v"))
    out.append(text(30, 108, "walls on", 10.5, "#555"))
    out.append(text(30, 122, "all sides", 10.5, "#555"))
    print("[rt] interface modes = 3, code amplitude 5 cells "
          "(drawn exaggerated)")
    FIGS["rt"] = svg(out, 460, 226, "heavy fluid resting on light fluid "
                     "with a rippled interface; troughs sink, crests rise")


def fig_shear():
    out = []
    X0, X1, Y0, Y1 = 40.0, 300.0, 30.0, 205.0
    out.append(f'<rect x="{X0}" y="{Y0}" width="{X1 - X0}"'
               f' height="{Y1 - Y0}" fill="#fbfaf7" stroke="#333"'
               f' stroke-width="1.6"/>')
    # bands: outer quarters move left, middle half moves right
    H = Y1 - Y0

    def prof(yn):
        if yn <= 0.5:
            return math.tanh(60.0 * (yn - 0.25))
        return math.tanh(60.0 * (0.75 - yn))

    for yn in (0.10, 0.375, 0.5, 0.625, 0.90):
        u = 0.5 * prof(yn)
        ys = Y1 - H * yn
        if abs(u) < 0.05:
            continue
        L = 70.0 * u
        xm = (X0 + X1) / 2
        out.append(arrow(xm - L, ys, xm + L, ys,
                         "#1f4e8c" if u > 0 else "#7a1f1f", w=2.6))
    # the two interfaces with a mode-4 ripple (perturbation in the code)
    for yn in (0.25, 0.75):
        ys = Y1 - H * yn
        rip = [(x, ys + 5.0 * math.sin(2 * math.pi * 4
                                       * (x - X0) / (X1 - X0)))
               for x in np.linspace(X0, X1, 70)]
        out.append(f'<polyline points="{pts(rip)}" fill="none"'
                   f' stroke="#2a7d2a" stroke-width="1.8"'
                   f' stroke-dasharray="4 3"/>')
    # side chart: u(y) profile
    px0 = 330.0
    prof_pts = [(px0 + 55.0 + 55.0 * 0.5 * prof(yn) * 2,
                 Y1 - H * yn) for yn in np.linspace(0.02, 0.98, 60)]
    out.append(f'<line x1="{px0 + 55}" y1="{Y0}" x2="{px0 + 55}"'
               f' y2="{Y1}" stroke="#999" stroke-width="1.1"/>')
    out.append(f'<polyline points="{pts(prof_pts)}" fill="none"'
               f' stroke="#333" stroke-width="2"/>')
    out.append(text(px0 + 55, Y0 - 8, "u(y)", 11.5, "#333", cls="v",
                    anchor="middle"))
    out.append(text(px0 + 55, Y1 + 16, "tanh jumps", 10.5, "#555",
                    anchor="middle"))
    out.append(text(X0 + 6, Y0 + 16, "&#8592; band moves left", 10.5,
                    "#7a1f1f"))
    out.append(text(X0 + 6, (Y0 + Y1) / 2 - 6, "band moves right"
                    " &#8594;", 10.5, "#1f4e8c"))
    out.append(text(X0 + 6, Y1 - 8, "&#8592; band moves left", 10.5,
                    "#7a1f1f"))
    out.append(text(X0 + 4, Y1 + 16, "green dashes: the two shear"
                    " interfaces, seeded with a mode-4 ripple", 10.5,
                    "#2a7d2a"))
    print("[shear] profile tanh(60(yn-0.25)); seed mode 4, amplitude 0.03")
    FIGS["shear"] = svg(out, 460, 226, "double shear layer: counter-moving "
                        "bands, tanh velocity profile and rippled "
                        "interfaces")


# ------------------------------------------------------------------ write
def svg(children, w, h, label, vb=None):
    body = "\n  ".join(children)
    vbox = vb or f"0 0 {w} {h}"
    return (f'<svg class="setupfig" viewBox="{vbox}" width="100%"'
            f' role="img" aria-label="{label}">\n  {body}\n</svg>')


def main():
    fig_grad()
    fig_div()
    fig_curl()
    fig_lap()
    fig_nozzle()
    fig_proj()
    fig_tunnel()
    fig_rt()
    fig_shear()
    with open(HTML, encoding="utf-8") as fh:
        s = fh.read()
    for key, frag in FIGS.items():
        beg, end = f"<!--FIG:{key}-->", f"<!--/FIG:{key}-->"
        assert s.count(beg) == 1, f"marker {beg} count != 1"
        assert s.count(end) == 1, f"marker {end} count != 1"
        pat = re.compile(re.escape(beg) + r".*?" + re.escape(end), re.S)
        s = pat.sub(beg + "\n" + frag + "\n" + end, s)
    with open(HTML, "w", encoding="utf-8") as fh:
        fh.write(s)
    print(f"spliced {len(FIGS)} figures into {os.path.normpath(HTML)}")


if __name__ == "__main__":
    main()
