"""Figure 9 - the cost of target recalibration and the retrospective replay.

a  realised target-test coverage versus the number k of target recalibration
   records: median over 20 splits for each of the six design regions, with the
   exact exchangeable Beta-theory 5-95% band (one split) and, for the design
   region with the widest k range, the empirical 5-95% spread across splits;
   1 <= k < 9 gives no finite interval at alpha = 0.10;
b  budget rules: records required so that coverage is within eps of nominal
   with probability 1 - delta = 0.90 - the Chebyshev rule of the submitted
   manuscript against the exact Beta rule;
c  retrospective replay: mean coverage on the not-yet-measured remainder
   versus the number of measurements, random order against novelty order.

Colour conventions.  Panels a and c draw DESIGN REGIONS, not calibration
states, so they use style.TASKSTYLE - the same neutral black/grey plus sky blue
and reddish purple identities, with the same markers and line styles, that
Fig. 4d uses for the same six design regions.  The four reserved ROLE hues stay
free there, and orange = heuristic / bluish green = recalibrated appear only in
panel b, where they really do compare the manuscript's heuristic rule with the
exact one.  Greek letters are set through the custom Arial mathtext fontset
configured in style.apply(), so the only fonts embedded in the PDF are Arial.

Every number is read from revision/results/R8_budget.json
(script revision/code/exp_R8_budget.py, report revision/results/R8_budget.md).
Reviewer 2 comments 7 and 9; Reviewer 1 comments 1 and 3.
"""
import json
import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Patch

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import style  # noqa: E402

style.apply()
mpl.rcParams["hatch.linewidth"] = 0.5
R = json.load(open(os.path.join(REV, "results", "R8_budget.json")))
DAT = R["datasets"]
GREY = style.OI["grey"]
MID = style.TASKSTYLE["polymer_Tg"][0]     # the spread band is polymer Tg's

TASKS = style.TASKS                        # six design regions, shared with Fig. 4
DSTYLE = style.TASKSTYLE                   # colour / marker / line style / face
# x slot of each design region inside the k = 0 strip, identical to Fig. 4: the
# two nearly equal pairs (glass Tg 0.403 / glass liquidus 0.408 and glass
# microhardness 0.200 / glass Young's modulus 0.242) are never neighbours.
DSLOT = {"glass_Tg": 1, "glass_E": 5, "glass_HV": 2,
         "glass_Tliq": 3, "steel_yield": 0, "polymer_Tg": 4}

W, H = 17.5, 9.25         # final print size in cm; nothing is ever downscaled
fig = plt.figure(figsize=(W * style.CM, H * style.CM))


def box(x0, y0, x1, y1):
    """axes rectangle given in centimetres of the printed figure."""
    return fig.add_axes([x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H])


YB, YT = 3.30, 8.30
# x labels / panel notes / legend foot, in cm.  Y_XLAB matches where matplotlib
# puts the b and c x labels, so panel a's shared x label sits on the same line.
Y_XLAB, Y_NOTE, Y_LEG = 2.71, 2.32, 0.32
axA0 = box(1.70, YB, 2.60, YT)     # k = 0 strip, full 0-1 coverage scale
axA = box(3.05, YB, 6.85, YT)      # k >= 1, magnifies 0.69-1.00
axB = box(8.85, YB, 12.05, YT)
axC = box(13.65, YB, 17.20, YT)
XA_MID = 4.35                      # centre of the whole of panel a, in cm

# =================================================================== a
YLO, YHI = 0.690, 1.005
KMIN, KMAX = 3.6, 620              # left edge is a crop of the 1 <= k < 9 zone
KFIN = 9                           # ceil(1/alpha) - 1 = 9 at alpha = 0.10

# exact exchangeable Beta band; it depends on k alone, which the assert checks
band = {}
for key, _ in TASKS:
    for v in DAT[key]["ksweep"]["alpha_0.10"].values():
        bt = v.get("theory_beta_5_50_95")
        if bt is None or any(x is None for x in bt):
            continue
        k = int(v["k_range"][0])
        if k in band:
            assert max(abs(a - b) for a, b in zip(band[k], bt)) < 1e-9, (key, k)
        band[k] = bt
kb = sorted(band)
axA.fill_between(kb, [band[k][0] for k in kb], [band[k][2] for k in kb],
                 color=style.OI["lightgrey"], lw=0, zorder=0)

# Empirical 5-95% ACROSS THE 20 SPLITS, for the design region that reaches the
# largest k.  Drawn as whiskers, not as a second band, because it is a different
# summary level from the grey band (which is the spread of ONE split) and from
# the plotted markers (the median of twenty), and a reader must not confuse the
# three.  It very nearly coincides with the theory band, which is the point.
SP = DAT["polymer_Tg"]["ksweep"]["alpha_0.10"]
spread = {int(v["k_range"][0]): v for v in SP.values()}
WK = [9, 30, 120, 482]
for k in WK:
    v = spread[k]
    axA.errorbar([k], [v["p50"]], yerr=[[v["p50"] - v["p5"]], [v["p95"] - v["p50"]]],
                 fmt="none", ecolor=MID, elinewidth=0.7, capsize=1.6, capthick=0.7,
                 zorder=2)

# 1 <= k < 9: the conformal interval is infinite, so no finite interval exists
# (minimum size ceil(1/alpha) - 1 = 9).  White with a light hatch and a thin
# border, so it never reads as part of the grey theory band it abuts.
axA.axvspan(KMIN, KFIN, facecolor="white", edgecolor="#BEBEBE", hatch="///",
            lw=0.5, zorder=0.5)
axA.text(np.sqrt(KMIN * KFIN), (YLO + YHI) / 2, "1 ≤ k < 9: no finite interval",
         fontsize=7, rotation=90, ha="center", va="center", color="#333333",
         zorder=2, bbox=dict(facecolor="white", edgecolor="none", pad=0.8))

for key, lab in TASKS:
    col, mk, ls, mfc = DSTYLE[key]
    S = DAT[key]["ksweep"]["alpha_0.10"]
    # dict keyed on k: steel yield reaches its pool at k = 20, so the "20" and
    # "full_pool" sweep entries are the same k and must not be drawn twice
    pts = sorted({int(v["k_range"][0]): v["p50"] for v in S.values()
                  if int(v["k_range"][0]) >= KFIN}.items())
    assert all(YLO < p[1] < YHI for p in pts), key
    axA.plot([p[0] for p in pts], [p[1] for p in pts], color=col, marker=mk,
             ls=ls, ms=3.0, lw=0.9, mfc=mfc, mew=0.8, zorder=3)
    # k = 0 (source-calibrated) on its own full-scale strip, one x slot per region
    axA0.plot([0.10 + 0.16 * DSLOT[key]], [S["0"]["p50"]], color=col, marker=mk,
              ms=3.0, mfc=mfc, mew=0.8, ls="none", zorder=3)

axA.axhline(0.90, color=style.ROLE["ideal"], lw=0.7, ls=(0, (4, 3)), zorder=2)
axA.set_xscale("log")
axA.set_xlim(KMIN, KMAX)
axA.set_ylim(YLO, YHI)
axA.set_xticks([10, 20, 50, 100, 200, 500])
axA.set_xticklabels(["10", "20", "50", "100", "200", "500"])
axA.set_xticks([], minor=True)
axA.set_yticks([0.70, 0.80, 0.90, 1.00])
axA.set_yticklabels(["0.70", "0.80", "0.90", "1.00"])
axA.yaxis.tick_right()                    # keeps the strip-to-panel gap clear
axA.spines["right"].set_visible(True)
axA.spines["left"].set_visible(False)

axA0.axhline(0.90, color=style.ROLE["ideal"], lw=0.7, ls=(0, (4, 3)), zorder=2)
axA0.set_xlim(0, 1)
axA0.set_ylim(-0.02, 1.02)
axA0.set_xticks([0.5])
axA0.set_xticklabels(["0"])
axA0.set_yticks([0, 0.5, 1.0])
axA0.set_yticklabels(["0", "0.5", "1.0"])
axA0.set_ylabel("target-test coverage")
# the two dashed connectors open from the strip's 0.69-1.00 slice into the main
# axes, which magnifies exactly that slice
for xy0, xy1 in [((1.0, YHI), (0.0, 1.0)), ((1.0, YLO), (0.0, 0.0))]:
    fig.add_artist(ConnectionPatch(xyA=xy0, coordsA=axA0.transData,
                                   xyB=xy1, coordsB=axA.transAxes,
                                   color=GREY, lw=0.5, ls=(0, (2, 2)), zorder=0))

fig.text(XA_MID / W, 8.44 / H, "Six design regions, median of 20 splits",
         fontsize=8, ha="center", va="bottom")
fig.text(XA_MID / W, Y_XLAB / H, "target recalibration records k", fontsize=8,
         ha="center", va="top")
fig.text(XA_MID / W, Y_NOTE / H, "k = 0: source-calibrated;  k ≥ 9: target-recalibrated",
         fontsize=7, ha="center", va="top")
fig.text(XA_MID / W, (Y_NOTE - 0.32) / H, "each curve ends at its recalibration pool",
         fontsize=7, ha="center", va="top")

# =================================================================== b
TH = {(e["eps"], e["delta"]): e for e in R["theory"]["alpha_0.10"]}
EPS = [0.05, 0.08, 0.10]
DELTA = 0.10
xs = np.arange(len(EPS))
wd = 0.33
cheb = [TH[(e, DELTA)]["chebyshev_m"] for e in EPS]
bst = [TH[(e, DELTA)]["exact_beta_stable"] for e in EPS]
bfi = [TH[(e, DELTA)]["exact_beta_first"] for e in EPS]
BASE = 5.0
assert cheb == [359, 140, 89] and bst == [60, 21, 11] and bfi == [15, 12, 11]
RATIO = TH[(0.08, DELTA)]["chebyshev_over_exact"]

axB.bar(xs - wd / 2, np.array(cheb) - BASE, wd, bottom=BASE,
        color=style.ROLE["heuristic"], edgecolor="black", lw=0.4, zorder=3)
axB.bar(xs + wd / 2, np.array(bst) - BASE, wd, bottom=BASE,
        color=style.ROLE["recal"], edgecolor="black", lw=0.4, hatch="///", zorder=3)
# value labels in black: at 7 pt, orange numerals on white are the least legible
# text in the figure, and the fill plus the hatch already name the series
for x, v in list(zip(xs - wd / 2, cheb)) + list(zip(xs + wd / 2, bst)):
    axB.text(x, v * 1.14, str(v), fontsize=7, ha="center", va="bottom", color="black")
axB.plot(xs + wd / 2, bfi, ls="none", marker="o", ms=3.6, mfc="white",
         mec=style.ROLE["ideal"], mew=0.7, zorder=5)

axB.text(xs[1], 330, "140 vs 21\n(%.1f×)" % RATIO, fontsize=7, ha="center",
         va="bottom", linespacing=1.25)

axB.set_yscale("log")
axB.set_xticks(xs)
axB.set_xticklabels(["0.05", "0.08", "0.10"])
axB.set_xlim(-0.62, 2.62)
axB.set_ylim(BASE, 3000)
axB.set_yticks([10, 30, 100, 300, 1000])
axB.set_yticklabels(["10", "30", "100", "300", "1000"])
axB.set_yticks([], minor=True)
axB.set_xlabel("tolerance " + r"$\epsilon$" + " (coverage)")
axB.set_ylabel("required records m")
axB.set_title("Budget rule, " + r"$\delta$" + " = 0.10", fontsize=8)
axB.legend(handles=[Patch(facecolor=style.ROLE["heuristic"], edgecolor="black",
                          lw=0.4, label="Chebyshev"),
                    Patch(facecolor=style.ROLE["recal"], edgecolor="black", lw=0.4,
                          hatch="///", label="exact Beta (stable)"),
                    Line2D([], [], ls="none", marker="o", ms=3.6, mfc="white",
                           mec="black", mew=0.7, label="exact Beta (first)")],
           loc="upper center", bbox_to_anchor=(0.5, 1.02), handlelength=1.1,
           borderpad=0.15, labelspacing=0.2, handletextpad=0.4)

# =================================================================== c
REPLAY = [("glass_Tg", "glass Tg"), ("polymer_Tg", "polymer Tg")]
# In THIS panel the line style carries the measurement order, not the design
# region (the inline key says so); colour and marker still carry the region.
ORDER = [("random", "-"), ("novelty", (0, (3.2, 1.4)))]
for key, _ in REPLAY:
    col, mk = DSTYLE[key][0], DSTYLE[key][1]
    # curves stop at the documented horizon (remainder >= max(20, 20% of |T|)),
    # beyond which a single miss moves remainder coverage by more than 0.05
    hor = DAT[key]["replay"]["stopping_horizon_n"]
    for order, ls in ORDER:
        c = DAT[key]["replay"][order]["curve"]
        n = np.array(c["n"], float)
        m = np.array([np.nan if v is None else v for v in c["mean"]], float)
        sel = (n >= 9) & (n <= hor)
        nn, mm = n[sel], m[sel]
        mev = [int(np.argmin(np.abs(nn - t)))
               for t in (11, 18, 30, 55, 100, 180, 330, 620) if t <= nn.max()]
        axC.plot(nn, mm, color=col, ls=ls, lw=0.9, marker=mk, ms=3.0,
                 markevery=sorted(set(mev)), zorder=3)

axC.axhline(0.90, color=style.ROLE["ideal"], lw=0.7, ls=(0, (4, 3)), zorder=2)
axC.set_xscale("log")
axC.set_xlim(8.0, 920)
axC.set_ylim(0.775, 1.015)
axC.set_xticks([10, 20, 50, 100, 200, 500])
axC.set_xticklabels(["10", "20", "50", "100", "200", "500"])
axC.set_xticks([], minor=True)
axC.set_yticks([0.80, 0.85, 0.90, 0.95, 1.00])
axC.set_yticklabels(["0.80", "0.85", "0.90", "0.95", "1.00"])
axC.set_xlabel("measurements n")
axC.set_ylabel("remainder coverage")
axC.set_title("Retrospective replay", fontsize=8)
axC.legend(handles=[Line2D([], [], color="black", ls="-", lw=0.9, label="random order"),
                    Line2D([], [], color="black", ls=(0, (3.2, 1.4)), lw=0.9,
                           label="novelty order")],
           loc="lower left", bbox_to_anchor=(0.0, 0.0), handlelength=2.0,
           borderpad=0.15, labelspacing=0.25, handletextpad=0.45)
xC_MID = (13.65 + 17.20) / 2
fig.text(xC_MID / W, Y_NOTE / H, "unmeasured candidates receive",
         fontsize=7, ha="center", va="top")
fig.text(xC_MID / W, (Y_NOTE - 0.32) / H, "intervals, not measured values",
         fontsize=7, ha="center", va="top")

# ------------------------------------------------------- shared figure legend
handles = [Patch(facecolor=style.OI["lightgrey"], edgecolor="none",
                 label="Beta theory 5–95%, one split"),
           Line2D([], [], color=MID, ls="-", lw=0.7, marker="|", ms=4, mew=0.7,
                  label="polymer Tg 5–95%, 20 splits"),
           Line2D([], [], color="black", ls=(0, (4, 3)), lw=0.7, label="nominal 0.90")]
handles += [Line2D([], [], ls="none", marker=DSTYLE[k][1], ms=3.4,
                   color=DSTYLE[k][0], mfc=DSTYLE[k][3], mew=0.8, label=lab)
            for k, lab in TASKS]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, Y_LEG / H),
           ncol=3, handlelength=2.0, columnspacing=1.6, borderpad=0.2,
           labelspacing=0.3, handletextpad=0.5)

fig.canvas.draw()


def letter(ax, L, x_cm):
    """style.panel with the letter pinned to an absolute x (cm) of the page."""
    bb = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    style.panel(ax, L, x=x_cm / W - bb.x0)


for ax, L, xc in [(axA0, "a", 0.30), (axB, "b", 7.75), (axC, "c", 12.55)]:
    letter(ax, L, xc)
png = style.save(fig, "fig9_budget", 9, tight=False)

# ---------------------------------------------------------- printed numbers
print(png)
print("a  k = 0 medians :", {k: round(DAT[k]["ksweep"]["alpha_0.10"]["0"]["p50"], 4)
                             for k, _ in TASKS})
allp = [v["p50"] for k, _ in TASKS for v in DAT[k]["ksweep"]["alpha_0.10"].values()
        if int(v["k_range"][0]) >= KFIN]
print("a  k >= 9 medians span : %.4f - %.4f" % (min(allp), max(allp)))
print("a  Beta band k=9/30/120 :", [[round(x, 4) for x in band[k][::2]] for k in (9, 30, 120)])
print("a  polymer Tg whiskers (p5, p50, p95) :",
      {k: tuple(round(spread[k][q], 4) for q in ("p5", "p50", "p95")) for k in WK})
print("a  largest k plotted :", max(kb))
print("b  chebyshev / stable / first :", cheb, bst, bfi, "ratio %.4f" % RATIO)
for k, lab in REPLAY:
    rp = DAT[k]["replay"]
    for o, _ in ORDER:
        c = rp[o]["curve"]
        n = np.array(c["n"], float)
        m = np.array([np.nan if v is None else v for v in c["mean"]], float)
        s = (n >= 9) & (n <= rp["stopping_horizon_n"])
        print("c  %-11s %-7s n %3d->%3d  cov %.4f->%.4f  min %.4f max %.4f"
              % (k, o, n[s][0], n[s][-1], m[s][0], m[s][-1],
                 np.nanmin(m[s]), np.nanmax(m[s])))
