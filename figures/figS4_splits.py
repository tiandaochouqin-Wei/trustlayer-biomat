"""Figure S4 - the design-region splits actually used in the analyses.

One panel per task: every record of the task is drawn on its design (shift) axis
(x) and on a second informative composition / descriptor axis (y), coloured by
the pool it belongs to (source, excluded, target).  The per-split record counts
of the five disjoint index sets (train, source-calibration, source-test,
recalibration pool, target-test) and the threshold that defines the design
region are printed in the panel header band, which is kept free of data.

Pools and splits come from revision/code/common.py (design_shift_datasets(),
design_split(ds, seed=0)); every printed count is asserted against
revision/results/R1_core.json ["<task>"]["sizes_seed0"], i.e. against the counts
the reported analyses were run on.  Colour roles are those of Figure 4
(source = ROLE['source'], target = ROLE['recal'], excluded = grey).
"""
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REV, "code"))

import style  # noqa: E402
from common import design_shift_datasets, design_split  # noqa: E402

style.apply()
R = json.load(open(os.path.join(REV, "results", "R1_core.json")))
GREY = style.OI["grey"]
C_SRC, C_TGT = style.ROLE["source"], style.ROLE["recal"]

# key, panel letter, title, x label, second axis column, y label, x tick fmt, threshold fmt
PANELS = [
    ("glass_Tg", "a", "Glass Tg", "P (at. frac.)", "Si", "Si (at. frac.)", "%.2f", "{:.3f}"),
    ("glass_E", "b", "Glass Young's modulus", "P (at. frac.)", "Si", "Si (at. frac.)", "%.2f", "{:.3f}"),
    ("glass_HV", "c", "Glass microhardness", "P (at. frac.)", "Si", "Si (at. frac.)", "%.2f", "{:.3f}"),
    ("glass_Tliq", "d", "Glass liquidus", "P (at. frac.)", "Si", "Si (at. frac.)", "%.2f", "{:.3f}"),
    ("steel_yield", "e", "Steel yield strength", "Ni (wt%)", "cr", "Cr (wt%)", "%.0f", "{:.1f}"),
    ("polymer_Tg", "f", "Polymer Tg", "aromatic-atom fraction", "TPSA",
     u"TPSA (Å²)", "%.2f", "{:.2f}"),
]

# (source quantile, target quantile) of the shift-axis rules defined in common.py
RULES = {"glass_Tg": (0.70, 0.75), "glass_E": (0.70, 0.75), "glass_HV": (0.70, 0.75),
         "glass_Tliq": (0.70, 0.75), "steel_yield": (0.60, 0.85), "polymer_Tg": (0.55, 0.85)}

W, H = 17.5, 8.0           # final print size in cm; nothing is ever downscaled
fig = plt.figure(figsize=(W * style.CM, H * style.CM))


def box(x0, y0, x1, y1):
    """axes rectangle given in centimetres of the printed figure."""
    return fig.add_axes([x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H])


COLX = [(1.54, 5.87), (7.23, 11.56), (12.92, 17.25)]
ROWY = [(5.15, 7.20), (1.84, 3.89)]
AXES = [box(COLX[i % 3][0], ROWY[i // 3][0], COLX[i % 3][1], ROWY[i // 3][1]) for i in range(6)]
AXH = ROWY[0][1] - ROWY[0][0]                       # axes height in cm
LINE = 8.8 / (72 * AXH * style.CM)                  # 8.8 pt line pitch in axes fraction

DS = {d.name: d for d in design_shift_datasets()}
SHOWN = {}                 # every number printed in the figure, for the caption


def view(v0, v1, frac, pad=0.05):
    """Limits that put every record between `pad` and `frac` of the axis, so the
    header band that carries the count text never sits on top of a record."""
    span = max(v1 - v0, 1e-9) / (frac - pad)
    lo = v0 - pad * span
    return lo, lo + span


def ticks_in(lo, v1, nb=4):
    """Ticks only where there are data: none in the reserved header band."""
    t = MaxNLocator(nb, steps=[1, 2, 2.5, 5, 10]).tick_values(lo, v1)
    return [v for v in t if lo - 1e-12 <= v <= v1 + 1e-12]


for ax, (key, letter, title, xlab, ycol, ylab, xfmt, tfmt) in zip(AXES, PANELS):
    ds = DS[key]
    sp = design_split(ds, 0)
    n = {k: int(len(v)) for k, v in sp.items()}
    assert n == R[key]["sizes_seed0"], (key, n, R[key]["sizes_seed0"])
    assert n["train"] + n["cal"] + n["stest"] == len(ds.src) == R[key]["dataset"]["n_source"]
    assert n["rpool"] + n["ttest"] == len(ds.tgt) == R[key]["dataset"]["n_target"]

    x = ds.shift
    y = ds.X[:, ds.feat.index(ycol)]
    thr_src, thr_tgt = float(x[ds.src].max()), float(x[ds.tgt].min())

    # the two quantile rules of common.py must reproduce exactly these two pools
    qs, qt = RULES[key]
    q_src = float(np.quantile(x, qs))
    q_tgt = float(np.quantile(x[x > x.min()], qt))
    assert np.array_equal(np.where(x <= q_src)[0], ds.src)
    assert np.array_equal(np.where(x >= q_tgt)[0], ds.tgt)
    SHOWN[key] = dict(n, n_excluded=len(ds.excluded), design_threshold=tfmt.format(thr_tgt),
                      rule_quantile_threshold=round(q_tgt, 5), source_threshold=round(q_src, 5))

    xlo, xhi = view(float(x.min()), float(x.max()), 0.965, pad=0.035)
    ylo, yhi = view(float(y.min()), float(y.max()), 0.68)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)

    # design region, and the excluded band that separates the two rules; the
    # boundary line stops below the header band so that it crosses no text
    ax.axvspan(thr_tgt, xhi, color=C_TGT, alpha=0.07, lw=0, zorder=0)
    ax.axvspan(thr_src, thr_tgt, color=GREY, alpha=0.10, lw=0, zorder=0)
    ax.plot([thr_tgt, thr_tgt], [ylo, ylo + 0.70 * (yhi - ylo)], color=C_TGT, lw=0.7,
            ls=(0, (2.4, 2.0)), zorder=1)

    for idx, col, mk, z in [(ds.excluded, GREY, "x", 2), (ds.src, C_SRC, "o", 3),
                            (ds.tgt, C_TGT, "s", 4)]:
        s = float(np.clip(700.0 / max(len(idx), 1), 1.0, 4.5))
        al = 0.35 if len(idx) > 800 else 0.75          # dense pools drawn more transparently
        ax.scatter(x[idx], y[idx], s=s, marker=mk, c=col, alpha=al,
                   linewidths=0.3 if mk == "x" else 0.0, zorder=z)

    # header band: the five per-split counts and the design-region rule
    ax.text(0.015, 0.985, f"train {n['train']} · cal {n['cal']} · s-test {n['stest']}",
            transform=ax.transAxes, fontsize=7, color=C_SRC, ha="left", va="top")
    ax.text(0.015, 0.985 - LINE, f"recal {n['rpool']} · t-test {n['ttest']}",
            transform=ax.transAxes, fontsize=7, color=C_TGT, ha="left", va="top")
    ax.text(0.985, 0.985 - LINE, f"design ≥ {tfmt.format(thr_tgt)}",
            transform=ax.transAxes, fontsize=7, color=C_TGT, ha="right", va="top")

    ax.set_xlabel(xlab, labelpad=2)
    ax.set_ylabel(ylab, labelpad=2)
    ax.set_title(title, fontsize=8, pad=2)
    ax.tick_params(pad=1.6)
    ax.set_xticks(ticks_in(xlo, float(x.max()), 5))
    ax.set_yticks(ticks_in(ylo, float(y.max()), 4))
    ax.xaxis.set_major_formatter(FormatStrFormatter(xfmt))

# ------------------------------------------------------------------ legend + key
# Both shaded bands are keyed, at the same alpha as each other, so that the grey
# interval between the two quantile rules is never an unexplained grey block.
handles = [
    Line2D([], [], ls="none", marker="o", ms=3.2, mfc=C_SRC, mec=C_SRC, label="source pool"),
    Line2D([], [], ls="none", marker="x", ms=3.2, mec=GREY, mew=0.9, label="excluded"),
    Patch(facecolor=GREY, alpha=0.22, lw=0, label="excluded region"),
    Line2D([], [], ls="none", marker="s", ms=3.2, mfc=C_TGT, mec=C_TGT, label="target pool"),
    Patch(facecolor=C_TGT, alpha=0.22, lw=0, label="design region"),
]
fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.30 / W, 0.52 / H),
           ncol=5, handlelength=1.0, handletextpad=0.4, columnspacing=1.2, borderpad=0.0)
FOOT = fig.text(0.30 / W, 0.10 / H, "cal, source-calibration; s-test, source-test; "
                "recal, recalibration pool; t-test, target-test; counts are for "
                "resampling seed 0", fontsize=7, ha="left", va="bottom")

fig.canvas.draw()


def letter(ax, L, x_cm):
    """style.panel with the letter pinned to an absolute x (cm) of the page, so
    that the six letters line up in three columns whatever the y-label width."""
    bb = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    style.panel(ax, L, x=x_cm / W - bb.x0)


LETTERX = [0.32, 6.01, 11.70]           # COLX starts - 1.22 cm
_nt = len(fig.texts)
for i, (ax, (_, L, *_rest)) in enumerate(zip(AXES, PANELS)):
    letter(ax, L, LETTERX[i % 3])
LETTERS = fig.texts[_nt:]

# guard: in-panel text must stay inside its panel and the two header cells must not touch
ren = fig.canvas.get_renderer()
for ax, P in zip(AXES, PANELS):
    p = ax.get_position()
    aw, ax0, ax1 = p.width * W, p.x0 * W, p.x1 * W
    ext = [(t, t.get_window_extent(ren)) for t in ax.texts]
    for t, b in ext:
        x0, x1 = b.x0 / fig.dpi * 2.54, b.x1 / fig.dpi * 2.54
        if x0 < ax0 - 0.02 or x1 > ax1 + 0.02:
            print(f"OVERFLOW {P[0]} {t.get_text()!r} {x0:.2f}-{x1:.2f} vs {ax0:.2f}-{ax1:.2f}")
    gap = ext[2][1].x0 / fig.dpi * 2.54 - ext[1][1].x1 / fig.dpi * 2.54
    print(f"{P[0]:11s} panel {aw:.2f} cm | line1 {(ext[0][1].width / fig.dpi * 2.54):.2f}"
          f" | header gap {gap:.2f} cm" + ("   <== TIGHT" if gap < 0.15 else ""))

bb = fig.get_tightbbox(ren)
print("content cm: x %.2f-%.2f  y %.2f-%.2f   (canvas %.2f x %.2f)" %
      (bb.x0 * 2.54, bb.x1 * 2.54, bb.y0 * 2.54, bb.y1 * 2.54, W, H))

# bottom band: the legend row and the abbreviation row must clear the row-2 x
# labels above them and each other
cm = lambda v: v / fig.dpi * 2.54          # noqa: E731
leg = fig.legends[0].get_window_extent(ren)
foot = FOOT.get_window_extent(ren)
lab_bot = min(cm(AXES[i].get_tightbbox(ren).y0) for i in (3, 4, 5))
print("legend x %.2f-%.2f y %.2f-%.2f | footnote x %.2f-%.2f y %.2f-%.2f | "
      "row-2 labels bottom %.2f" % (cm(leg.x0), cm(leg.x1), cm(leg.y0), cm(leg.y1),
                                    cm(foot.x0), cm(foot.x1), cm(foot.y0), cm(foot.y1),
                                    lab_bot))
assert cm(leg.x1) < 17.25 and cm(foot.x1) < 17.25
assert cm(foot.y1) < cm(leg.y0) < cm(leg.y1) < lab_bot

# panel letters: same x in each column, and clear of the row above
lx = [round(cm(t.get_window_extent(ren).x0), 3) for t in LETTERS]
ly = [(round(cm(t.get_window_extent(ren).y0), 2), round(cm(t.get_window_extent(ren).y1), 2))
      for t in LETTERS]
print("letter x:", lx, " y:", ly)
assert lx[:3] == lx[3:], lx
row1_bot = min(cm(AXES[i].get_tightbbox(ren).y0) for i in (0, 1, 2))
row2_top = max(cm(AXES[i].get_tightbbox(ren).y1) for i in (3, 4, 5))
print("row-1 labels bottom %.2f  vs  row-2 titles top %.2f" % (row1_bot, row2_top))
assert row2_top < row1_bot
# the row-2 letters sit left of the row-1 axes boxes, so they may rise past the
# row-1 x labels; check the real 2-D overlap rather than the y extent alone
for t in LETTERS[3:]:
    tb = t.get_window_extent(ren)
    for i in (0, 1, 2):
        ab = AXES[i].get_tightbbox(ren)
        assert tb.x1 < ab.x0 or tb.x0 > ab.x1 or tb.y1 < ab.y0, (t.get_text(), i)
png = style.save(fig, "figS4_splits", "S4", tight=False)
print(png)
print(json.dumps(SHOWN))
