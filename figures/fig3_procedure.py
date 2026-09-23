"""Figure 3 - the trust layer as ONE decision procedure (BMEMat revision).

Pure-vector flow chart (matplotlib patches, arrows and text only; no icons, no
raster images).  Drawn at the FINAL print size: 17.5 cm (double column) x 12.4 cm.
Every text element is >= 7 pt at that size; panel letters are 10 pt bold Arial,
lowercase, no parentheses, top-left.  Colours are Okabe-Ito with the semantic
roles of style.ROLE, and every colour is backed by a second, non-colour cue
(marker shape, icon shape, outline shape, line style, text label).

Vocabulary is bound to the manuscript: the two states a candidate can be in are
named "source-calibrated" and "target-recalibrated" here exactly as in Figures
4-7 and in the text, and the flagged region is the "design region".

Run:  python fig3_procedure.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon
from matplotlib.path import Path

import style

style.apply()
plt.rcParams.update({
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "mathtext.default": "it",
})

# ----------------------------------------------------------------------------- canvas
# The figure is DRAWN AT FINAL PRINT SIZE: 17.5 cm (BMEMat double column) x 12.4 cm,
# and is never scaled afterwards.  style.save() writes with bbox_inches="tight", which
# adds matplotlib's default 0.1 in (0.254 cm) pad on each side, so the drawing canvas is
# made exactly that much smaller and the data limits are inset by the same amount:
# 1 data unit stays 1 cm and the saved file measures 17.5 x 12.4 cm.
W, H = 17.5, 12.4                      # cm; 1 data unit == 1 cm
PAD = 0.254                            # cm, matplotlib's savefig pad_inches=0.1
FIG = plt.figure(figsize=((W - 2 * PAD) * style.CM, (H - 2 * PAD) * style.CM))
ax = FIG.add_axes([0, 0, 1, 1])
ax.set_xlim(PAD, W - PAD)
ax.set_ylim(PAD, H - PAD)
ax.axis("off")

# ----------------------------------------------------------------------------- style
INK = "#1a1a1a"
EDGE = "#3d3d3d"
SOFT = "#8c8c8c"
PANEL = "#f2f2f2"
BLUE = style.ROLE["source"]        # source-calibrated stratum
GREEN = style.ROLE["recal"]        # target-recalibrated stratum
VERM = style.ROLE["fail"]          # violation / not guaranteed

FS_H = 8.0      # box headers (bold)
FS_B = 7.5      # body text
FS_S = 7.0      # small labels
FS_P = 10.0     # panel letters
LH = FS_B * 1.30 / 72 * 2.54       # body line height (cm)

TEXTS = []      # (artist, max width in cm, string) -> EVERY string goes here


def tint(hex_colour, frac):
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    r, g, b = (int(round(c * frac + 255 * (1 - frac))) for c in (r, g, b))
    return "#%02x%02x%02x" % (r, g, b)


def T(x, y, s, fs=FS_B, color=INK, weight="normal", ha="left", va="top",
      maxw=None, italic=False, z=6):
    """Every text element in this figure is created here, so the width guard at
    the bottom of the file measures all of them (no ax.text() calls elsewhere)."""
    t = ax.text(x, y, s, fontsize=fs, color=color, fontweight=weight, ha=ha, va=va,
                zorder=z, fontstyle="italic" if italic else "normal")
    TEXTS.append((t, maxw, s))
    return t


def block(x, y_top, lines, fs=FS_B, color=INK, maxw=None, lh=None):
    lh = lh or (fs * 1.30 / 72 * 2.54)
    y = y_top
    for s in lines:
        T(x, y, s, fs=fs, color=color, maxw=maxw)
        y -= lh
    return y


def box(x0, y0, x1, y1, ec=EDGE, fc="white", lw=0.8, ls="-", r=0.14, z=2):
    p = FancyBboxPatch((x0 + r, y0 + r), (x1 - x0) - 2 * r, (y1 - y0) - 2 * r,
                       boxstyle="round,pad=%.3f,rounding_size=%.3f" % (r, r),
                       linewidth=lw, edgecolor=ec, facecolor=fc, linestyle=ls, zorder=z)
    ax.add_patch(p)
    return p


def sqbox(x0, y0, x1, y1, ec=EDGE, fc="white", lw=0.8, ls="-", z=2):
    p = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="square,pad=0",
                       linewidth=lw, edgecolor=ec, facecolor=fc, linestyle=ls, zorder=z)
    ax.add_patch(p)
    return p


def arrow(p0, p1, color=EDGE, lw=0.9, ls="-", ms=6.5, z=4):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, linewidth=lw,
                        linestyle=ls, edgecolor=color, facecolor=color, shrinkA=0,
                        shrinkB=0, zorder=z, joinstyle="miter", capstyle="butt")
    ax.add_patch(a)
    return a


def poly_arrow(pts, color=EDGE, lw=0.9, ls="-", ms=6.5, z=4):
    path = Path(pts, [Path.MOVETO] + [Path.LINETO] * (len(pts) - 1))
    a = FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=ms, linewidth=lw,
                        linestyle=ls, edgecolor=color, facecolor=color, shrinkA=0,
                        shrinkB=0, zorder=z, joinstyle="miter", capstyle="butt")
    ax.add_patch(a)
    return a


def gbadge(xr, yt, tag, tw=0.62, th=0.32):
    """Small grey tag (G1/G2/G3) with its right edge at xr and top edge at yt."""
    box(xr - tw, yt - th, xr, yt, ec=EDGE, fc="#ececec", lw=0.6, r=0.07, z=3)
    T(xr - tw / 2, yt - th / 2, tag, fs=FS_S, weight="bold", ha="center",
      va="center_baseline", maxw=tw - 0.12)


def head(x0, y1, x1, n, title, gtag=None, tw=0.62):
    """Step header: number badge (top left), bold title, optional guarantee tag."""
    ax.add_patch(Circle((x0 + 0.26, y1 - 0.26), 0.19, facecolor=INK, edgecolor="none",
                        zorder=5))
    T(x0 + 0.26, y1 - 0.26, n, fs=FS_S, color="white", weight="bold", ha="center",
      va="center_baseline", maxw=0.30)
    T(x0 + 0.50, y1 - 0.10, title, fs=FS_H, weight="bold",
      maxw=(x1 - x0) - 0.62 - (tw + 0.12 if gtag else 0.10))
    if gtag:
        gbadge(x1 - 0.14, y1 - 0.14, gtag, tw=tw)


# ------------------------------------------------------------------- decision chips
# The four decisions use style.DECISION, the canonical glyphs shared with Figure 1:
# accept = bluish-green star, reject = vermillion down-triangle, measure = reddish-purple
# plus, not certified = open grey cross.  Colour, marker shape, outline shape and the
# word label are four independent cues, so nothing depends on colour alone.
def chip(x0, y, w, label, shape="round", ls="-", h=0.50):
    d = style.DECISION[label]
    colour = d["color"]
    y0, y1 = y - h / 2, y + h / 2
    if shape == "round":
        box(x0, y0, x0 + w, y1, ec=colour, fc=tint(colour, 0.16), lw=1.0, r=h / 2, z=4)
    elif shape == "square":
        sqbox(x0, y0, x0 + w, y1, ec=colour, fc=tint(colour, 0.16), lw=1.0, ls=ls, z=4)
    else:
        e = 0.17
        ax.add_patch(Polygon([(x0 + e, y0), (x0 + w - e, y0), (x0 + w, y), (x0 + w - e, y1),
                              (x0 + e, y1), (x0, y)], closed=True, facecolor=tint(colour, 0.16),
                             edgecolor=colour, lw=1.0, zorder=4))
    ax.add_line(Line2D([x0 + 0.31], [y], marker=d["marker"], ms=d["ms"] * 1.25,
                       markerfacecolor=colour if d["filled"] else "white",
                       markeredgecolor=colour, markeredgewidth=1.0, color="none", zorder=7))
    T(x0 + 0.56, y, label, fs=FS_B, weight="bold", va="center_baseline", maxw=w - 0.56 - 0.24)
    return colour


# ============================================================================= panel a
if not style.panel_data(ax, "a", 0.26, 12.14):
    T(0.26, 12.14, "a", fs=FS_P, weight="bold", va="top")

IX0, IX1 = 0.26, 3.15          # inputs
AX0, AX1 = 3.72, 7.62          # left step column
BX0, BX1 = 8.10, 12.40         # right step column
DX0, DX1 = 12.90, 17.24        # decide panel
PADX = 0.16

# --- inputs ---------------------------------------------------------------------------
box(IX0, 4.00, IX1, 11.72, ec="none", fc=PANEL, r=0.16, z=1)
T(IX0 + 0.14, 11.63, "Inputs", fs=FS_H, weight="bold", maxw=2.6)
for (y0, y1, l1, l2) in [
    (10.30, 11.25, "Trained predictor $f$", "any model class"),
    (9.07, 10.02, "Source calibration", "records ($x$, $y$)"),
    (7.84, 8.79, "Candidates $x$", "to be judged"),
    (6.61, 7.56, "Specification", "property $y$ ≥ $\\tau$"),
    (5.38, 6.33, "Level $\\alpha$ = 0.10", "nominal 0.90"),
    (4.15, 5.10, "Target budget $m$", "measurements"),
]:
    box(IX0 + 0.10, y0, IX1 - 0.10, y1, ec=SOFT, fc="white", lw=0.7, r=0.10, z=2)
    T(IX0 + 0.26, y1 - 0.13, l1, fs=FS_B, maxw=2.30)
    T(IX0 + 0.26, y1 - 0.13 - LH, l2, fs=FS_S, color="#4d4d4d", maxw=2.30)

# --- step 1: calibrate -----------------------------------------------------------------
box(AX0, 9.26, AX1, 12.05)
head(AX0, 12.05, AX1, "1", "Calibrate", gtag="G1")
block(AX0 + PADX, 12.05 - 0.54, [
    "Nonconformity scores",
    "$|y - f(x)|$ on the source",
    "calibration records",
    "→ conformal quantile $q$",
    "Interval $f(x)$ ± $q$: coverage",
    "≥ 1 − $\\alpha$ if exchangeable",
], maxw=3.58)
# step 1 consumes BOTH the trained predictor and the source calibration records,
# so each of those two input boxes has its own connector.
arrow((IX1 + 0.02, 10.75), (AX0 - 0.03, 10.75))       # trained predictor f
arrow((IX1 + 0.02, 9.55), (AX0 - 0.03, 9.55))         # source calibration records

# --- step 2: shift check ----------------------------------------------------------------
box(BX0, 10.30, BX1, 12.05)
head(BX0, 12.05, BX1, "2", "Shift check", gtag="G3")
block(BX0 + PADX, 12.05 - 0.54, [
    "Novelty score $s(x)$: distance to",
    "the training inputs",
    "→ conformal $p$-value $p(x)$",
], maxw=3.98)
arrow((AX1 + 0.03, 11.20), (BX0 - 0.03, 11.20))

# --- shift decision ----------------------------------------------------------------------
dcx, dcy, dhw, dhh = 10.25, 8.58, 1.50, 0.46
arrow((dcx, 10.30), (dcx, dcy + dhh + 0.03))
ax.add_patch(Polygon([(dcx, dcy + dhh), (dcx + dhw, dcy), (dcx, dcy - dhh), (dcx - dhw, dcy)],
                     closed=True, facecolor="white", edgecolor=EDGE, lw=0.8, zorder=3))
T(dcx, dcy, "$p(x)$ ≥ 0.05?", fs=FS_B, ha="center", va="center_baseline", maxw=2.10)
arrow((dcx - dhw, dcy), (AX1 + 0.03, dcy))                       # no -> target-recalibrated
T((AX1 + dcx - dhw) / 2, dcy + 0.34, "no", fs=FS_S, color="#4d4d4d", ha="center", maxw=0.8)
arrow((dcx, dcy - dhh), (dcx, 7.87))                             # yes -> source-calibrated
T(dcx + 0.26, 7.97, "yes", fs=FS_S, color="#4d4d4d", va="center_baseline", maxw=0.9)

# --- source-calibrated stratum --------------------------------------------------------------
SBY0, SBY1 = 6.64, 7.84
box(BX0, SBY0, BX1, SBY1, ec=BLUE, fc=tint(BLUE, 0.09), lw=0.9)
ax.add_line(Line2D([BX0 + PADX + 0.08], [SBY1 - 0.26], marker=style.MARK["source"], ms=3.4,
                   color=BLUE, markerfacecolor=BLUE, zorder=6))
T(BX0 + PADX + 0.26, SBY1 - 0.13, "Source stratum", fs=FS_B, weight="bold", color=BLUE,
  maxw=3.72)
block(BX0 + PADX, SBY1 - 0.13 - LH, [
    "source-calibrated: keeps the",
    "quantile $q$ of step 1",
], maxw=3.98)

# --- step 3: recalibrate ---------------------------------------------------------------------
box(AX0, 5.75, AX1, 8.90, ec=GREEN, fc=tint(GREEN, 0.07), lw=0.9)
head(AX0, 8.90, AX1, "3", "Recalibrate")
ax.add_line(Line2D([AX0 + PADX + 0.08], [8.90 - 0.54 - 0.13], marker=style.MARK["recal"],
                   ms=3.2, color=GREEN, markerfacecolor=GREEN, zorder=6))
T(AX0 + PADX + 0.26, 8.90 - 0.54, "Target stratum", fs=FS_B, weight="bold", color=GREEN,
  maxw=3.3)
block(AX0 + PADX, 8.90 - 0.54 - LH, [
    "target-recalibrated design",
    "region. Quantile from $k$ ≤ $m$",
    "target measurements from a",
    "recalibration pool, never from",
    "the candidates judged",
    "No target data → not certified",
], maxw=3.58)

# --- step 4: stratify by measurements -----------------------------------------------------------
box(BX0, 3.60, BX1, 6.40)
head(BX0, 6.40, BX1, "4", "Stratify by measurements")
block(BX0 + PADX, 6.40 - 0.54, [
    "Separate (Mondrian) strata, one",
    "quantile per pattern of available",
    "measurements, e.g.",
], maxw=3.98)
# no subscripts anywhere: mathtext subscripts would print below the 7 pt floor
for i, lab in enumerate(["composition only", "+ density", "+ expansion"]):
    yy = 4.78 - i * 0.36
    ax.add_line(Line2D([BX0 + PADX + 0.10], [yy], marker="s", ms=2.6, color=SOFT,
                       markerfacecolor=SOFT, zorder=6))
    T(BX0 + PADX + 0.30, yy + 0.13, lab, fs=FS_B, maxw=3.6)

arrow((dcx, SBY0), (dcx, 6.43))                                   # source stratum -> step 4
arrow((AX1 + 0.03, 6.05), (BX0 - 0.03, 6.05))                     # step 3 -> step 4
poly_arrow([(IX1 + 0.02, 4.63), (4.10, 4.63), (4.10, 5.72)], color=SOFT, lw=0.7)  # budget m

# ================================================================================ decide
box(DX0, 5.75, DX1, 12.05, ec="#c9c9c9", fc=PANEL, lw=0.7, r=0.16, z=1)
T(DX0 + PADX, 11.92, "Decide", fs=FS_H, weight="bold", maxw=1.7)
T(DX0 + PADX, 11.56, "interval vs specification $y$ ≥ $\\tau$", fs=FS_B, maxw=4.0)

xs0, xs1, xtau = 13.06, 14.95, 14.10
ax.add_line(Line2D([xtau, xtau], [8.25, 10.92], color=INK, lw=0.7, ls=(0, (3, 2)), zorder=3))
T(xtau, 10.94, "$\\tau$", fs=FS_B, ha="center", va="bottom", maxw=0.30)
ax.add_line(Line2D([xs0, xs1], [8.25, 8.25], color=EDGE, lw=0.7, zorder=3))
T((xs0 + xs1) / 2, 8.16, "predicted property $y$", fs=FS_S, color="#4d4d4d", ha="center",
  maxw=2.3)
for y, (lo, hi, mu), lab, shape in [
    (10.70, (14.25, 14.85, 14.55), "accept", "round"),
    (9.80, (13.20, 13.90, 13.55), "reject", "square"),
    (8.90, (13.80, 14.60, 14.20), "measure", "tag"),
]:
    ax.add_line(Line2D([lo, hi], [y, y], color=INK, lw=1.0, zorder=5))
    for xx in (lo, hi):
        ax.add_line(Line2D([xx, xx], [y - 0.12, y + 0.12], color=INK, lw=1.0, zorder=5))
    ax.add_line(Line2D([mu], [y], marker="o", ms=2.6, color=INK, markerfacecolor=INK, zorder=6))
    chip(15.05, y, 2.10, lab, shape=shape)

T(DX0 + PADX, 7.60, "no valid interval:", fs=FS_S, color="#4d4d4d", maxw=2.0)
# widest label of the four: its chip is widened so the word keeps the same optical
# padding inside the outline as accept / reject / measure.
chip(DX0 + PADX, 6.95, 2.50, "not certified", shape="square", ls=(0, (2.4, 1.6)))
arrow((BX1 + 0.03, 6.05), (DX0 - 0.03, 6.05))                     # step 4 -> decide

# --- optional FDR branch --------------------------------------------------------------------
box(DX0, 2.85, 16.05, 4.95, ec=EDGE, fc="white", lw=0.8, ls=(0, (3, 2)))
T(DX0 + PADX, 4.83, "Optional branch", fs=FS_B, weight="bold", maxw=2.1)
gbadge(16.05 - 0.14, 4.83 - 0.04, "G2")
block(DX0 + PADX, 4.83 - 0.42, [
    "accept with FDR",
    "control (conformal",
    "selection +",
    "Benjamini–Hochberg)",
], maxw=2.83)
arrow((13.70, 5.72), (13.70, 4.98), ls=(0, (2.5, 1.6)), lw=0.8)

# --- feedback loop: measure -> target stratum ---------------------------------------------------
poly_arrow([(16.30, 8.65), (16.30, 2.40), (4.95, 2.40), (4.95, 5.72)],
           color=style.DECISION["measure"]["color"], lw=1.0)
T(5.15, 4.30, "measured candidates", fs=FS_B, color=INK, italic=True, maxw=2.7)
T(5.15, 4.30 - LH, "join the target stratum", fs=FS_B, color=INK, italic=True, maxw=2.7)

# ============================================================================= panel b
if not style.panel_data(ax, "b", 0.26, 2.30):
    T(0.26, 2.30, "b", fs=FS_P, weight="bold", va="top")
gx0, gy0, gy1, gap = 0.26, 0.26, 1.84, 0.28
gw = (17.24 - gx0 - 3 * gap) / 4
# "q" is the conformal quantile of step 1 everywhere in this figure, so the FDR
# statement names the selection level in words instead of reusing the symbol.
for i, (hd, lines, col, ls) in enumerate([
    ("G1", ["coverage ≥ 1 − $\\alpha$ within a", "stratum (marginal, over",
            "exchangeable draws)"], EDGE, "-"),
    ("G2", ["FDR ≤ the nominal selection", "level among accepted",
            "candidates (exchangeable)"], EDGE, "-"),
    ("G3", ["false-flag rate ≤ 0.05; the", "shift check has no",
            "guaranteed power"], EDGE, "-"),
    ("Not guaranteed", ["conditional coverage,", "clinical safety"], VERM, (0, (3, 2))),
]):
    x0 = gx0 + i * (gw + gap)
    box(x0, gy0, x0 + gw, gy1, ec=col, fc="white" if i == 3 else "#fafafa", lw=0.9, ls=ls)
    T(x0 + 0.20, gy1 - 0.13, hd, fs=FS_B, weight="bold", color=VERM if i == 3 else INK,
      maxw=gw - 0.40)
    block(x0 + 0.20, gy1 - 0.13 - 0.33, lines, maxw=gw - 0.40, lh=0.33)

# ============================================================================== output
FIG.canvas.draw()
bad = []
for t, maxw, s in TEXTS:
    if maxw is None:
        continue
    wcm = t.get_window_extent().width / FIG.dpi * 2.54
    if wcm > maxw + 0.01:
        bad.append((s, wcm, maxw))
for s, wcm, maxw in bad:
    print("OVERFLOW  %-42s %.2f cm > %.2f cm" % (s[:42], wcm, maxw))
untracked = sum(1 for t in ax.texts if all(t is not a for a, _, _ in TEXTS))
print("texts: %d total, %d width-checked, %d untracked"
      % (len(ax.texts), len([1 for _, m, _ in TEXTS if m]), untracked))
if not bad:
    print("text fits: no overflow")

png = style.save(FIG, "fig3_procedure", 3)
print("saved:", png)
