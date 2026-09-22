"""Figure 4 - calibration and design-region shift (data).

a  phosphorus design axis of the glass Tg task (source / excluded / target);
b  in-distribution calibration curve, split conformal vs tree-spread heuristic;
c  source-calibrated -> target-recalibrated coverage over the six tasks;
d  target-test coverage versus the number k of target recalibration records.

All numbers come from revision/results/R1_core.json (20 resampling splits per
task, protocol in revision/code/common.py); panel a reads the glass Tg dataset
definition from revision/code/common.py.

Panel d keeps the ROLE colours free: the six task curves are drawn in neutral
black/greys plus the two unreserved Okabe-Ito hues, each with its own marker and
line style, so that the reader never carries the panel b/c meaning of blue,
green, orange and vermillion into panel d.  No mathtext is used anywhere, so the
only fonts embedded in the PDF are ArialMT and Arial-BoldMT.
"""
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Patch

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REV, "code"))

import style  # noqa: E402
from common import glass_dataset  # noqa: E402

style.apply()
R = json.load(open(os.path.join(REV, "results", "R1_core.json")))
LEVELS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
GREY = style.OI["grey"]

TASKS = [("glass_Tg", "glass Tg"), ("glass_E", "glass Young's modulus"),
         ("glass_HV", "glass microhardness"), ("glass_Tliq", "glass liquidus"),
         ("steel_yield", "steel yield strength"), ("polymer_Tg", "polymer Tg")]

# Panel d: task identity only.  Colour here is NOT a calibration state, so the
# four reserved role colours (blue, bluish green, vermillion, orange) are not
# used; marker and line style carry the same information as the colour.  The
# definition lives in style.TASKSTYLE so that Fig. 9a,c, which plot the same six
# design regions, use exactly the same identities.
DSTYLE = style.TASKSTYLE
# x slot of each task inside the k = 0 strip: chosen so that the two pairs with
# nearly equal coverage (glass Tg 0.403 / glass liquidus 0.406 and glass HV
# 0.210 / glass E 0.230) are never neighbours.
DSLOT = {"glass_Tg": 1, "glass_E": 5, "glass_HV": 2,
         "glass_Tliq": 3, "steel_yield": 0, "polymer_Tg": 4}

W, H = 17.5, 12.0          # final print size in cm; nothing is ever downscaled
fig = plt.figure(figsize=(W * style.CM, H * style.CM))


def box(x0, y0, x1, y1):
    """axes rectangle given in centimetres of the printed figure."""
    return fig.add_axes([x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H])


axA = box(1.80, 7.80, 8.90, 11.15)
axB = box(10.95, 7.80, 17.15, 11.15)
axC = box(3.40, 3.35, 8.00, 6.35)
axD0 = box(10.20, 3.35, 11.20, 6.35)      # k = 0 strip, full 0-1 coverage scale
axD = box(11.60, 3.35, 16.55, 6.35)       # k >= 9, magnifies 0.78-1.00

# ------------------------------------------------------------------ a
ds = glass_dataset("Tg", "glass_Tg", "K")
P = ds.shift
rng = np.random.default_rng(0)
rows = [(0, ds.src, style.ROLE["source"], "o", 1.0, 0.30),
        (1, ds.excluded, GREY, "x", 2.2, 0.55),
        (2, ds.tgt, style.ROLE["recal"], "s", 3.0, 0.75)]
thr = float(P[ds.tgt].min())
axA.axvspan(thr, 0.27, color=style.ROLE["recal"], alpha=0.07, lw=0)
axA.axvline(thr, color=GREY, lw=0.6, ls=(0, (2, 2)), zorder=1)
for y0, idx, col, mk, ms, al in rows:
    yy = y0 + rng.uniform(-0.32, 0.32, len(idx))
    axA.scatter(P[idx], yy, s=ms, marker=mk, c=col, alpha=al, linewidths=0.35,
                zorder=2)
axA.set_yticks([0, 1, 2])
axA.set_yticklabels(["source\nn = 6592", "excluded\nn = 773", "target\nn = 258"])
axA.set_ylim(-0.7, 3.5)
axA.set_xlim(-0.012, 0.27)
axA.set_xticks([0, 0.05, 0.10, 0.15, 0.20, 0.25])
axA.set_xlabel("P atomic fraction")
axA.set_title("glass Tg design axis", fontsize=8)
axA.annotate("", xy=(0.135, 2.85), xytext=(0.004, 2.85),
             arrowprops=dict(arrowstyle="-|>", lw=0.9, color="black",
                             mutation_scale=7, shrinkA=0, shrinkB=0))
axA.text(0.069, 2.95, "design shift", ha="center", va="bottom", fontsize=7.5)
axA.text(0.266, 2.30, "design region", ha="right", va="bottom", fontsize=7.5,
         color=style.ROLE["recal"])
axA.spines["left"].set_visible(False)
axA.tick_params(axis="y", length=0)

# ------------------------------------------------------------------ b
S = R["glass_Tg"]
axB.plot([0.45, 1.0], [0.45, 1.0], color="black", lw=0.7, ls=(0, (1, 1.6)), zorder=1)
axB.plot(LEVELS, S["id_curve_conformal"], color=style.ROLE["source"],
         marker=style.MARK["source"], ls="-", ms=3.5, zorder=3)
axB.plot(LEVELS, S["id_curve_heuristic"], color=style.ROLE["heuristic"],
         marker=style.MARK["heuristic"], ls=(0, (4, 1.6)), ms=3.5, zorder=3)
ce_c = S["id_ce"]["conformal_mean"]["mean"]
ce_h = S["id_ce"]["heuristic_mean"]["mean"]
axB.text(0.468, 0.995, f"tree-spread heuristic\ncalibration error {ce_h:.3f}",
         color=style.ROLE["heuristic"], fontsize=7, ha="left", va="top", linespacing=1.3)
axB.text(0.695, 0.625, f"split conformal\ncalibration error {ce_c:.3f}",
         color=style.ROLE["source"], fontsize=7, ha="left", va="top", linespacing=1.3)
axB.text(0.575, 0.468, "y = x", fontsize=7, ha="left", va="bottom", color="black")
axB.set_xlim(0.45, 1.0)
axB.set_ylim(0.45, 1.0)
axB.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
axB.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
axB.set_xlabel("nominal coverage")
axB.set_ylabel("empirical coverage")
axB.set_title("In-distribution calibration", fontsize=8)

# ------------------------------------------------------------------ c
ypos = np.arange(len(TASKS))[::-1]
for y, (key, lab) in zip(ypos, TASKS):
    S = R[key]
    s, r = S["shift_source"]["coverage"], S["shift_recal"]["coverage"]
    axC.plot([s["mean"], r["mean"]], [y, y], color=GREY, lw=0.8, zorder=1)
    axC.errorbar(s["mean"], y, xerr=[[s["mean"] - s["lo"]], [s["hi"] - s["mean"]]],
                 fmt=style.MARK["source"], ms=4, mfc=style.ROLE["source"],
                 mec=style.ROLE["source"], ecolor=style.ROLE["source"], elinewidth=0.8,
                 capsize=1.6, capthick=0.8, zorder=3,
                 label="source-calibrated" if y == ypos[0] else None)
    axC.errorbar(r["mean"], y, xerr=[[r["mean"] - r["lo"]], [r["hi"] - r["mean"]]],
                 fmt=style.MARK["recal"], ms=4, mfc=style.ROLE["recal"],
                 mec=style.ROLE["recal"], ecolor=style.ROLE["recal"], elinewidth=0.8,
                 capsize=1.6, capthick=0.8, zorder=3,
                 label="target-recalibrated" if y == ypos[0] else None)
    # in-distribution coverage: lifted clear of the nominal line so that it reads
    # as a per-task datum sitting on 0.90, not as part of the reference line.
    axC.plot([S["id_conformal"]["coverage"]["mean"]], [y + 0.30], marker="v", ms=3.0,
             mfc="none", mec=GREY, mew=0.8, ls="none", zorder=4,
             label="in-distribution" if y == ypos[0] else None)
    axC.text(1.035, y, f"m = {S['m']}", fontsize=7, ha="left", va="center")
axC.axvline(0.90, color="black", lw=0.7, ls=(0, (4, 3)), zorder=0)
axC.text(0.885, 5.85, "nominal 0.90", fontsize=7, ha="right", va="center")
axC.set_yticks(ypos)
axC.set_yticklabels([lab for _, lab in TASKS])
axC.set_ylim(-0.6, 7.4)
axC.set_xlim(0.0, 1.02)
axC.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
axC.set_xlabel("coverage")
axC.set_title("Coverage on the target-test half", fontsize=8)
axC.tick_params(axis="y", length=0)
axC.legend(loc="upper left", bbox_to_anchor=(-0.01, 1.02), ncol=1, handlelength=1.2,
           borderpad=0.2, labelspacing=0.25, handletextpad=0.4)

# ------------------------------------------------------------------ d
YLO, YHI = 0.780, 1.005                   # zoom of the main panel-d axes
kb = sorted({int(k) for S in R.values() for k in S["ksweep"] if int(k) >= 9})
bt = R["polymer_Tg"]["ksweep"]
blo = [max(bt[str(k)]["beta_theory_5_50_95"][0], YLO - 0.006) for k in kb]
bhi = [bt[str(k)]["beta_theory_5_50_95"][2] for k in kb]
axD.fill_between(kb, blo, bhi, color="#E9E9E9", lw=0, zorder=0)

for key, lab in TASKS:
    col, mk, ls, mfc = DSTYLE[key]
    ks = sorted(int(k) for k in R[key]["ksweep"] if int(k) >= 9)
    cov = [R[key]["ksweep"][str(k)]["coverage"]["mean"] for k in ks]
    axD.plot(ks, cov, color=col, marker=mk, ls=ls, ms=3.2, lw=1.0, mfc=mfc,
             mew=0.8, zorder=3)
    # k = 0 (source-calibrated) on its own full-scale strip, one x slot per task
    x0 = 0.10 + 0.16 * DSLOT[key]
    axD0.plot([x0], [R[key]["ksweep"]["0"]["coverage"]["mean"]], color=col,
              marker=mk, ms=3.2, mfc=mfc, mew=0.8, ls="none", zorder=3)

axD.axhline(0.90, color="black", lw=0.7, ls=(0, (4, 3)), zorder=2)
axD.set_xscale("log")
axD.set_xlim(8.0, 380)
axD.set_ylim(YLO, YHI)
axD.set_xticks([10, 20, 50, 100, 200])
axD.set_xticklabels(["10", "20", "50", "100", "200"])
axD.set_xticks([], minor=True)
axD.set_yticks([0.80, 0.85, 0.90, 0.95, 1.00])
axD.set_yticklabels(["0.80", "0.85", "0.90", "0.95", "1.00"])
axD.yaxis.tick_right()                    # keeps the strip-to-panel gap clear
axD.spines["right"].set_visible(True)
axD.spines["left"].set_visible(False)
# inline key for the nominal line, in the empty lower-right corner
axD.plot([62, 86], [0.800, 0.800], color="black", lw=0.7, ls=(0, (4, 3)), zorder=4)
axD.text(94, 0.800, "nominal 0.90", fontsize=7, ha="left", va="center")

# k = 0 strip: same quantity, full 0-1 scale; the two dashed connectors open from
# the strip's 0.78-1.00 slice into the main axes, which magnifies exactly that slice.
axD0.axhline(0.90, color="black", lw=0.7, ls=(0, (4, 3)), zorder=2)
axD0.set_xlim(0, 1)
axD0.set_ylim(-0.02, 1.02)
axD0.set_xticks([0.5])
axD0.set_xticklabels(["0"])
axD0.set_yticks([0, 0.5, 1.0])
axD0.set_yticklabels(["0", "0.5", "1.0"])
axD0.set_ylabel("coverage")
for xy0, xy1 in [((1.0, YHI), (0.0, 1.0)), ((1.0, YLO), (0.0, 0.0))]:
    fig.add_artist(ConnectionPatch(xyA=xy0, coordsA=axD0.transData,
                                   xyB=xy1, coordsB=axD.transAxes,
                                   color=GREY, lw=0.5, ls=(0, (2, 2)), zorder=0))

# shared title, x label and the two short notes, centred on the whole of panel d
xmid = (10.20 + 16.55) / 2 / W
fig.text(xmid, 6.52 / H, "Recalibration budget", fontsize=8, ha="center", va="bottom")
fig.text(xmid, 2.60 / H, "target recalibration records k", fontsize=8,
         ha="center", va="top")
fig.text(xmid, 2.22 / H, "0 = source-calibrated;  k < 9: infinite interval",
         fontsize=7, ha="center", va="top")
fig.text(xmid, 1.90 / H, "colour / line style = task, not calibration state",
         fontsize=7, ha="center", va="top")

handles = [Patch(facecolor="#E9E9E9", edgecolor="none", label="Beta theory 5–95%")]
handles += [Line2D([], [], color=DSTYLE[k][0], marker=DSTYLE[k][1], ls=DSTYLE[k][2],
                   mfc=DSTYLE[k][3], mew=0.8, ms=3.2, lw=1.0, label=lab)
            for k, lab in TASKS]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(xmid, 1.62 / H),
           bbox_transform=fig.transFigure, ncol=2, handlelength=2.0, borderpad=0.2,
           labelspacing=0.28, handletextpad=0.45, columnspacing=1.0)

fig.canvas.draw()


def letter(ax, L, x_cm):
    """style.panel with the letter pinned to an absolute x (cm) of the page."""
    bb = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    style.panel(ax, L, x=x_cm / W - bb.x0)


for ax, L, xc in [(axA, "a", 0.32), (axB, "b", 9.32),
                  (axC, "c", 0.32), (axD0, "d", 9.32)]:
    letter(ax, L, xc)
png = style.save(fig, "fig4_design_shift", 4)
print(png)
