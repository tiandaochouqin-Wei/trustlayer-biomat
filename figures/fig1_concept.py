# -*- coding: utf-8 -*-
"""Figure 1 - conceptual framework of the model-agnostic trust layer.

Pure-vector schematic: every element is a matplotlib patch, line or text object.
No icons, clip-art, photographs or embedded raster images (the PowerPoint version
of this figure, which embedded ~200 third-party raster icons, is replaced).

Drawn at the final print size (17.5 cm double-column width); never downscaled.
All text >= 7 pt Arial - including the subscript of T_g, which is composed from
separate 7 pt text objects rather than mathtext (mathtext would print it at
0.7 x the base size).  Panel letters 10 pt bold lowercase, no parentheses;
Okabe-Ito colours with semantic roles plus redundant marker shapes.
Decision glyphs come from style.DECISION so that they are identical everywhere.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style  # noqa: E402

style.apply()
plt.rcParams.update({
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "mathtext.sf": "Arial",
})

CM = style.CM
PAD = 0.254                      # cm; savefig(bbox_inches='tight') adds 0.1 in back
CW = style.WIDTH_2COL / CM - 2 * PAD    # content width  = 16.99 cm
CH = 9.46                               # content height (figure = CH + 2*PAD)

# ---------------------------------------------------------------- colours ----
GREEN = style.ROLE["recal"]        # bluish green - target-recalibrated / trust layer
BLUE = style.ROLE["source"]        # blue         - source-calibrated
VERM = style.ROLE["fail"]          # vermillion   - failure / violation
GREY = style.OI["grey"]
INK = "#1A1A1A"
BODY = "#3C3C3C"
KEY_TXT = "#4D4D4D"                # explanatory key - must stay legible
GREY_TXT = "#595959"               # "reused unchanged" boxes: >= 6:1 on GREY_FILL
GREY_EDGE = "#B4B4B4"
GREY_FILL = "#F2F2F2"
GREEN_FILL = "#E2F3EE"
GREEN_DARK = "#00654A"
BOX_EDGE = "#4D4D4D"

F_BODY, F_HEAD, F_PANEL, F_SUB = 7.0, 8.0, 10.0, 8.5
F_TG = 8.5                         # illustrative T_g value (subscript then 7 pt)


def Y(d):
    """Convert a top-down distance (cm) into axes y (cm, origin bottom-left)."""
    return CH - d


def box(ax, x, ytop, w, h, fc="white", ec=BOX_EDGE, lw=0.8, ls="-", r=0.10, z=2):
    p = FancyBboxPatch((x, Y(ytop) - h), w, h,
                       boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, ls=ls, zorder=z,
                       mutation_aspect=1.0)
    ax.add_patch(p)
    return p


def arrow(ax, x0, d0, x1, d1, color=BOX_EDGE, lw=0.9, ls="-", ms=6.5, z=4):
    a = FancyArrowPatch((x0, Y(d0)), (x1, Y(d1)), arrowstyle="-|>",
                        mutation_scale=ms, lw=lw, ls=ls, color=color,
                        shrinkA=0, shrinkB=0, joinstyle="miter", zorder=z)
    ax.add_patch(a)
    return a


def txt(ax, x, d, s, size=F_BODY, weight="normal", color=INK, ha="center",
        va="top", z=5, ls_=1.32, style_="normal"):
    return ax.text(x, Y(d), s, fontsize=size, fontweight=weight, color=color,
                   ha=ha, va=va, zorder=z, linespacing=ls_, style=style_)


def flowbox(ax, x, ytop, w, h, head, lines, grey=False):
    """Flow-chart box: bold heading, then a stack of 7 pt items."""
    fc = GREY_FILL if grey else "white"
    ec = GREY_EDGE if grey else BOX_EDGE
    tc = GREY_TXT if grey else INK
    box(ax, x, ytop, w, h, fc=fc, ec=ec)
    txt(ax, x + w / 2, ytop + 0.30, head, size=F_HEAD, weight="bold", color=tc)
    txt(ax, x + w / 2, ytop + 0.68, "\n".join(lines), size=F_BODY,
        color=GREY_TXT if grey else BODY)


# ------------------------------------------------------------- geometry ------
X_DATA, W_DATA = 0.00, 2.60
X_MODL, W_MODL = 3.02, 2.25
X_BAND = 5.72                      # left edge of prediction box / trust-layer band
W_BAND = CW - X_BAND

DATA_LINES = ["composition", "microstructure images", "spectra",
              "mechanical tests", "device signals"]
MODEL_LINES = ["random forest", "GNN · CNN", "transformer"]

fig = plt.figure(figsize=(style.WIDTH_2COL, (CH + 2 * PAD) * CM))
ax = fig.add_axes([PAD / (CW + 2 * PAD), PAD / (CH + 2 * PAD),
                   CW / (CW + 2 * PAD), CH / (CH + 2 * PAD)])
ax.set_xlim(0, CW)
ax.set_ylim(0, CH)
ax.set_axis_off()
ax.set_aspect("equal")

# =============================================================== panel a =====
A_HEAD = 0.28                       # baseline of panel letter / heading
txt(ax, 0.0, A_HEAD, "a", size=F_PANEL, weight="bold", ha="left", va="baseline")
txt(ax, 0.52, A_HEAD, "Conventional pipeline", size=F_SUB, weight="bold",
    ha="left", va="baseline")

A_TOP, A_H = 0.62, 2.42
A_MID = A_TOP + A_H / 2

flowbox(ax, X_DATA, A_TOP, W_DATA, A_H, "Material data", DATA_LINES)
arrow(ax, X_DATA + W_DATA + 0.05, A_MID, X_MODL - 0.05, A_MID)
flowbox(ax, X_MODL, A_TOP, W_MODL, A_H, "Any ML model", MODEL_LINES)
arrow(ax, X_MODL + W_MODL + 0.05, A_MID, X_BAND - 0.05, A_MID)

# ---- point prediction -------------------------------------------------------
W_PRED = 2.52
box(ax, X_BAND, A_TOP, W_PRED, A_H)
txt(ax, X_BAND + W_PRED / 2, A_TOP + 0.30, "Point prediction",
    size=F_HEAD, weight="bold")

# "e.g. Tg = 812 K" composed from separate text objects: the subscript is a real
# 7 pt glyph with a manual baseline offset, NOT mathtext (which would set it at
# 0.7 x the base size and so break the 7 pt floor).
TG_D = A_TOP + 1.00                                     # baseline, top-down cm
TG_SUB_DROP = 0.060                                     # ~0.20 em at 8.5 pt
TG_SEGS = [("e.g. ", F_TG, "normal", 0.0),
           ("T", F_TG, "italic", 0.0),
           ("g", F_BODY, "normal", TG_SUB_DROP),
           (" = 812 K", F_TG, "normal", 0.0)]
tg_objs = [ax.text(0.0, Y(TG_D + dy), s, fontsize=fs, style=st, color=BODY,
                   ha="left", va="baseline", zorder=5)
           for s, fs, st, dy in TG_SEGS]

# tiny glyph: a single point on a value axis - deliberately no whiskers
gx, gd = X_BAND + W_PRED / 2, A_TOP + 1.58
ax.add_line(Line2D([gx - 0.72, gx + 0.72], [Y(gd), Y(gd)], color="#AAAAAA",
                   lw=0.6, zorder=3))
ax.plot([gx], [Y(gd)], marker="o", ms=4.0, color=INK, zorder=5, clip_on=False)
txt(ax, gx, A_TOP + 1.80, "no interval", size=F_BODY, color=KEY_TXT)

# ---- trust gap (an annotation on the flow, not a stage: no fill) ------------
W_DEC = 1.85
X_DEC = CW - W_DEC
G_CONN = 0.70                       # long enough that the dash pattern reads
X_GAP = X_BAND + W_PRED + G_CONN
W_GAP = X_DEC - G_CONN - X_GAP
arrow(ax, X_BAND + W_PRED + 0.05, A_MID, X_GAP - 0.05, A_MID,
      color=VERM, ls=(0, (2.2, 1.5)), ms=5.6)
box(ax, X_GAP, A_TOP, W_GAP, A_H, fc="none", ec=VERM, lw=1.0,
    ls=(0, (3.2, 2.0)), z=1)
txt(ax, X_GAP + W_GAP / 2, A_TOP + 0.32, "trust gap", size=F_HEAD,
    weight="bold", color=VERM)
GAP_LINES = ["no calibrated uncertainty",
             "distribution shift (design → bench → body)",
             "sparse, selectively reported data"]
for i, s in enumerate(GAP_LINES):
    d = A_TOP + 0.98 + i * 0.46
    # neutral dash bullet: the down-triangle is reserved for the reject decision
    ax.add_line(Line2D([X_GAP + 0.26, X_GAP + 0.44], [Y(d) - 0.10] * 2,
                       color=VERM, lw=1.3, solid_capstyle="butt", zorder=5))
    txt(ax, X_GAP + 0.56, d, s, size=F_BODY, ha="left", color=BODY)
arrow(ax, X_GAP + W_GAP + 0.05, A_MID, X_DEC - 0.05, A_MID,
      color=VERM, ls=(0, (2.2, 1.5)), ms=5.6)

flowbox(ax, X_DEC, A_TOP, W_DEC, A_H, "Decision",
        ["fabricate", "test", "submit"])

# =============================================================== panel b =====
B_HEAD = 3.58
txt(ax, 0.0, B_HEAD, "b", size=F_PANEL, weight="bold", ha="left", va="baseline")
txt(ax, 0.52, B_HEAD, "With the trust layer", size=F_SUB, weight="bold",
    ha="left", va="baseline")

BAND_TOP, BAND_PAD, STEP_H = 3.72, 0.56, 1.66
STEP_TOP = BAND_TOP + BAND_PAD          # band title sits above the step row
BAND_H = BAND_PAD + STEP_H + 0.36
B_MID = STEP_TOP + STEP_H / 2

# greyed data + model boxes (unchanged, not retrained)
GB_H = 2.42
GB_TOP = B_MID - GB_H / 2
flowbox(ax, X_DATA, GB_TOP, W_DATA, GB_H, "Material data", DATA_LINES, grey=True)
arrow(ax, X_DATA + W_DATA + 0.05, B_MID, X_MODL - 0.05, B_MID, color=GREY_EDGE)
flowbox(ax, X_MODL, GB_TOP, W_MODL, GB_H, "Any ML model", MODEL_LINES, grey=True)
arrow(ax, X_MODL + W_MODL + 0.05, B_MID, X_BAND - 0.05, B_MID, color=BOX_EDGE)

# trust-layer band
box(ax, X_BAND, BAND_TOP, W_BAND, BAND_H, fc=GREEN_FILL, ec=GREEN, lw=1.1,
    r=0.14, z=1)
t1 = txt(ax, X_BAND + 0.22, BAND_TOP + 0.42, "Trust layer", size=F_SUB,
         weight="bold", color=GREEN_DARK, ha="left", va="baseline")

# NB: the recalibration budget is *m* throughout the manuscript (Box 1, Methods,
# Figure 3); *k* is reserved for the number of exchangeable calibration records.
STEPS = [
    ("Calibrate", ["conformal interval,", "coverage $\\geq 1-\\alpha$"]),
    ("Shift check", ["is the candidate", "exchangeable with",
                     "the calibration data?"]),
    ("Recalibrate /\nmeasure", ["spend $m$ target", "measurements"]),
    ("Stratify", ["by available", "measurements"]),
]
S_PAD, S_GAP = 0.16, 0.32
W_STEP = (W_BAND - 2 * S_PAD - 3 * S_GAP) / 4
HEAD_LH = 0.34
for i, (head, lines) in enumerate(STEPS):
    x = X_BAND + S_PAD + i * (W_STEP + S_GAP)
    box(ax, x, STEP_TOP, W_STEP, STEP_H, fc="white", ec=GREEN, lw=0.8, z=2)
    ax.add_patch(Circle((x + 0.30, Y(STEP_TOP + 0.30)), 0.185, fc=GREEN,
                        ec="none", zorder=4))
    ax.text(x + 0.30, Y(STEP_TOP + 0.30), str(i + 1), fontsize=F_BODY,
            fontweight="bold", color="white", ha="center", va="center_baseline",
            zorder=5)
    txt(ax, x + 0.58, STEP_TOP + 0.13, head, size=7.5, weight="bold",
        ha="left", va="top", color=INK, ls_=1.28)
    nhead = head.count("\n") + 1
    # headings and body blocks align across the four steps; STEP_H is sized for
    # the longest step (1 heading line + 3 body lines) so no box looks empty
    txt(ax, x + 0.11, STEP_TOP + 0.21 + nhead * HEAD_LH, "\n".join(lines),
        size=F_BODY, ha="left", va="top", color=BODY)
    if i < 3:
        arrow(ax, x + W_STEP + 0.045, B_MID, x + W_STEP + S_GAP - 0.045, B_MID,
              color=GREEN, lw=1.0)

# ---- fan-out from the band to the four decision chips ----
CHIP_TOP, CHIP_H, CHIP_GAP = BAND_TOP + BAND_H + 0.66, 1.66, 0.26
W_CHIP = (W_BAND - 3 * CHIP_GAP) / 4
CHIP_X = [X_BAND + i * (W_CHIP + CHIP_GAP) for i in range(4)]
CHIP_CX = [x + W_CHIP / 2 for x in CHIP_X]
bus_d = BAND_TOP + BAND_H + 0.32
ax.add_line(Line2D([X_BAND + W_BAND / 2] * 2,
                   [Y(BAND_TOP + BAND_H), Y(bus_d)], color="#555555", lw=0.8,
                   zorder=3))
ax.add_line(Line2D([CHIP_CX[0], CHIP_CX[-1]], [Y(bus_d), Y(bus_d)],
                   color="#555555", lw=0.8, zorder=3))
for cx in CHIP_CX:
    arrow(ax, cx, bus_d, cx, CHIP_TOP - 0.03, color="#555555", lw=0.8, ms=6.0)

X_KEY = X_MODL + W_MODL
txt(ax, X_KEY, CHIP_TOP + 0.52, "Decision", size=F_HEAD,
    weight="bold", ha="right", va="baseline")
txt(ax, X_KEY, CHIP_TOP + 0.90, "one per candidate", size=F_BODY,
    ha="right", va="baseline", color=KEY_TXT)
txt(ax, X_KEY, CHIP_TOP + 1.42, "bars: prediction interval\n"
    "dashed line: specification limit", size=F_BODY, ha="right", va="baseline",
    color=KEY_TXT, ls_=1.34)

# interval span of the schematic prediction interval, relative to the spec limit
CHIPS = [
    ("accept", (0.18, 0.82), ["interval", "meets spec"]),
    ("reject", (-0.82, -0.18), ["interval", "violates spec"]),
    ("measure", (-0.34, 0.80), ["interval", "straddles spec"]),
    ("not certified", None, ["no valid interval", "without target data"]),
]
for (label, span, desc), x, cx in zip(CHIPS, CHIP_X, CHIP_CX):
    d = style.DECISION[label]
    col, mk, filled, ms = d["color"], d["marker"], d["filled"], d["ms"]
    box(ax, x, CHIP_TOP, W_CHIP, CHIP_H, fc="white", ec=col, lw=1.0, r=0.12)
    txt(ax, cx, CHIP_TOP + 0.16, label, size=F_HEAD, weight="bold", color=INK)
    gd = CHIP_TOP + 0.78
    ax.add_line(Line2D([cx - 0.92, cx + 0.92], [Y(gd), Y(gd)], color="#C8C8C8",
                       lw=0.6, zorder=3))
    ax.add_line(Line2D([cx, cx], [Y(gd) - 0.20, Y(gd) + 0.20], color=INK,
                       lw=0.7, ls=(0, (1.8, 1.4)), zorder=4))
    if span is not None:
        lo, hi = cx + span[0], cx + span[1]
        ax.add_line(Line2D([lo, hi], [Y(gd), Y(gd)], color=col, lw=1.2, zorder=5))
        for e in (lo, hi):
            ax.add_line(Line2D([e, e], [Y(gd) - 0.11, Y(gd) + 0.11], color=col,
                               lw=1.2, zorder=5))
        mx = (lo + hi) / 2
    else:
        mx = cx + 0.45
    ax.plot([mx], [Y(gd)], marker=mk, ms=ms, mfc=col if filled else "white",
            mec=col, mew=1.0, color=col, zorder=6, clip_on=False)
    txt(ax, cx, CHIP_TOP + 1.07, "\n".join(desc), size=F_BODY, color=BODY)

txt(ax, X_BAND, CHIP_TOP + CHIP_H + 0.32,
    "Guarantees are statistical (coverage, false-discovery rate) and conditional on\n"
    "exchangeability — not clinical-safety guarantees.",
    size=F_BODY, ha="left", va="top", color=KEY_TXT, ls_=1.30)

# ---- second pass: position text that needs measured widths ------------------
fig.canvas.draw()
r = fig.canvas.get_renderer()
inv = ax.transData.inverted()


def w_cm(obj):
    bb = obj.get_window_extent(renderer=r)
    return inv.transform((bb.x1, bb.y0))[0] - inv.transform((bb.x0, bb.y0))[0]


# centre the composed "e.g. Tg = 812 K" run inside the Point prediction box
widths = [w_cm(o) for o in tg_objs]
xc = X_BAND + W_PRED / 2 - sum(widths) / 2
for o, w in zip(tg_objs, widths):
    o.set_x(xc)
    xc += w

# band subtitle placed after the measured width of the bold band title
bb = t1.get_window_extent(renderer=r)
x_after = inv.transform((bb.x1, bb.y0))[0]
txt(ax, x_after + 0.14, BAND_TOP + 0.42, "wraps the model, no retraining",
    size=7.5, ha="left", va="baseline", color=GREEN_DARK)

png = style.save(fig, "fig1_concept", 1)
print(png)
print("figure size (cm):", fig.get_size_inches() / CM)
