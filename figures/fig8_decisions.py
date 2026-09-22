"""Figure 8 - the layer as one decision procedure, and incomplete measurements.

a  false-accept proportion among accepted design-region candidates, per dataset
   and per arm (A0 point prediction, A1 conformal without shift check, A3 full
   trust layer, A4 layer + conformal selection), against the selection level
   q = 0.10 and against the pooled rate over all accepted candidates for A1;
b  what those decisions cost: accept / reject / measure / not certified for the
   representative glass Tg candidate list, plus the total measured share
   (measure + not certified + recalibration budget m) across all six datasets;
c  per-pattern coverage of the eight natural missing-measurement patterns of
   glass Tg, pooled versus Mondrian-by-pattern calibration;
d  per-pattern Mondrian interval width of the fused model against the same
   glasses under a composition-only model.

Panels a-b read revision/results/R7_pipeline.json (model RF, headline tau =
recalibration-pool median, 20 splits; report revision/results/R7_pipeline.md).
Panels c-d read revision/results/R6_fusion.json (design B1_Tg, composition-
grouped random split, fused HistGB, 20 splits; report R6_fusion.md).

Colour discipline.  The reserved ROLE hues carry ONE meaning per figure:
blue = source-calibrated / un-adapted calibration (A1 in a, the single pooled
quantile in c), green = target-recalibrated / pattern-adapted calibration (A3
in a, the per-pattern Mondrian quantile in c).  Panel d contrasts two MODELS,
not two calibration states, so it spends no reserved hue at all: both series
are drawn in neutral dark grey and separated by filled (fused) versus open
(composition-only) squares, the square keeping its panel-c meaning "Mondrian,
one quantile per pattern".  The circle therefore never means anything but
"pooled calibration" anywhere in the figure.
"""
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import style  # noqa: E402

style.apply()
R7 = json.load(open(os.path.join(REV, "results", "R7_pipeline.json")))["results"]
R6 = json.load(open(os.path.join(REV, "results", "R6_fusion.json")))

GREY = style.OI["grey"]
MGREY = "#999999"
DGREY = "#4D4D4D"
RHO = "ρ"
MIN_ACC = 20               # < 1 accepted design-region candidate per split

# ---------------------------------------------------------------- data, a & b
MODEL, TAU = "RF", "rpool_median"
DS = [("glass_Tg", "glass\nTg"), ("glass_E", "glass\nE"), ("glass_HV", "glass\nHV"),
      ("glass_Tliq", "glass\nTliq"), ("steel_yield", "steel\nyield"),
      ("polymer_Tg", "polymer\nTg")]
# arm -> (label, colour, marker); colours are the fixed semantic roles of style.ROLE
ARMS = [("A0_point", "A0 point prediction", style.ROLE["fail"], "v"),
        ("A1_conformal", "A1 conformal, source-calibrated", style.ROLE["source"], "o"),
        ("A3_layer", "A3 layer, target-recalibrated", style.ROLE["recal"], "s"),
        ("A4_layer_fdr", "A4 layer + conformal selection", style.ROLE["other"], "^")]


def head(ds):
    return R7[ds]["models"][MODEL]["tau"][TAU]


W, H = 17.5, 12.0          # final print size in cm; nothing is ever downscaled
fig = plt.figure(figsize=(W * style.CM, H * style.CM))


def box(x0, y0, x1, y1):
    return fig.add_axes([x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H])


axA = box(1.98, 7.75, 9.10, 11.05)
axB = box(11.55, 7.75, 16.95, 11.05)
axC = box(2.32, 1.45, 8.55, 5.20)
axD = box(12.05, 1.45, 16.95, 5.20)

# ------------------------------------------------------------------------- a
# Rates that rest on fewer than MIN_ACC accepted design-region candidates over
# the 20 splits are not reportable.  They are drawn as grey crosses on their own
# baseline BELOW the value axis (the left spine stops at 0), so that a
# "not reportable" mark can never be read as a plotted proportion of 0.
NOREP = -0.037
OFF = [-0.255, -0.085, 0.085, 0.255]
for (arm, lab, col, mk), dx in zip(ARMS, OFF):
    for i, (ds, _) in enumerate(DS):
        p = head(ds)["pooled"][arm]["true_target"]
        tot, fa = p["n_accept_total"], p["false_accepts_total"]
        if tot < MIN_ACC:                       # rate not reportable
            axA.plot([i + dx], [NOREP], marker="x", ms=3.4, mew=0.9, color=GREY,
                     ls="none", zorder=4)
            continue
        axA.plot([i + dx], [fa / tot], marker=mk, ms=4.2, mfc=col, mec=col, mew=0.9,
                 ls="none", zorder=5)
    axA.plot([], [], marker=mk, ms=4.2, mfc=col, mec=col, ls="none", label=lab)

# grey reference: A1 pooled over ALL accepted candidates of that dataset.  The
# same reportability rule applies here, so steel yield (5 accepts over 20
# splits) carries no grey bar either.
for i, (ds, _) in enumerate(DS):
    p = head(ds)["pooled"]["A1_conformal"]["all"]
    if p["n_accept_total"] < MIN_ACC:
        continue
    v = p["false_accepts_total"] / p["n_accept_total"]
    axA.plot([i - 0.36, i + 0.36], [v, v], color=GREY, lw=1.6, solid_capstyle="butt",
             zorder=2)
axA.plot([], [], color=GREY, lw=1.6, label="A1, all accepted candidates")

axA.axhline(0.10, color="black", lw=0.7, ls=(0, (4, 3)), zorder=1)
axA.text(4.40, 0.118, "selection level q = 0.10", fontsize=7, ha="right", va="bottom")
axA.text(-0.50, NOREP, "×  not reportable (< 1 accept per split)", fontsize=7,
         color=GREY, ha="left", va="center")
axA.set_xlim(-0.55, len(DS) - 0.45)
axA.set_ylim(-0.092, 0.86)         # headroom so the key clears every plotted rate
axA.set_xticks(range(len(DS)))
axA.set_xticklabels([l for _, l in DS])
axA.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
axA.spines["left"].set_bounds(0, 0.6)
axA.set_ylabel("false-accept proportion")
axA.set_title("Accepted design-region candidates", fontsize=8)
axA.tick_params(axis="x", length=0)
axA.legend(loc="upper right", bbox_to_anchor=(1.02, 1.03), ncol=1, handlelength=1.1,
           borderpad=0.15, labelspacing=0.22, handletextpad=0.4)

# ------------------------------------------------------------------------- b
BARMS = ["A0_point", "A1_conformal", "A2_layer_no_target", "A3_layer", "A4_layer_fdr"]
# bottom -> top of the stack.  Segments are drawn without internal edges and the
# whole stack is stroked once, so that the 0.012-0.015 "rejected" sliver prints
# as a visible fill rather than as a doubled hairline.
SEG = [("n_accept", "accepted", "white"), ("n_reject", "rejected", MGREY),
       ("n_measure", "sent to measurement", style.OI["skyblue"]),
       ("n_not_certified", "not certified", DGREY)]
REP = "glass_Tg"
A = head(REP)["arms"]
npool = A["A0_point"]["all"]["n"]["mean"]
for j, arm in enumerate(BARMS):
    bottom = 0.0
    for key, _, fc in SEG:
        v = A[arm]["all"][key]["mean"] / npool
        if v <= 0:
            continue
        axB.bar(j - 0.17, v, width=0.44, bottom=bottom, color=fc, edgecolor="none",
                zorder=3)
        bottom += v
    axB.add_patch(Rectangle((j - 0.39, 0.0), 0.44, bottom, fill=False,
                            edgecolor="black", lw=0.5, zorder=4))
    # total measured = sent to measurement + not certified + recalibration
    # budget m; m is spent by A3 and A4 only (code/exp_R7_pipeline.py BUDGET_ARMS)
    axB.plot([j - 0.17], [A[arm]["all"]["measurements"]["mean"] / npool], marker="D",
             ms=3.4, color="black", ls="none", zorder=5)
    # the same quantity across all six datasets
    shares = [head(d)["arms"][arm]["all"]["measurements"]["mean"]
              / head(d)["arms"][arm]["all"]["n"]["mean"] for d, _ in DS]
    axB.plot([j + 0.25, j + 0.25], [min(shares), max(shares)], color=GREY, lw=1.0,
             solid_capstyle="round", zorder=3)
    axB.plot([j + 0.25], [float(np.median(shares))], marker="_", ms=5, mew=1.2,
             color=GREY, ls="none", zorder=4)

axB.set_xlim(-0.62, len(BARMS) - 0.38)
axB.set_ylim(-0.035, 1.47)                 # A0 sits at 0.000; keep it off the spine
axB.set_xticks(range(len(BARMS)))
axB.set_xticklabels(["A0", "A1", "A2", "A3", "A4"])
axB.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
axB.spines["left"].set_bounds(0, 1.0)
axB.set_ylabel("share of candidate list")
axB.set_xlabel("arm (A2 = layer without target records)", fontsize=7)
axB.set_title("Cost of the decisions (glass Tg, n = %d)" % round(npool), fontsize=8)
axB.tick_params(axis="x", length=0)
hb = [Patch(fc=fc, ec="black", lw=0.5, label=lab) for _, lab, fc in SEG]
hb += [Line2D([], [], marker="D", ms=3.4, color="black", ls="none",
              label="total measured"),
       Line2D([], [], color=GREY, lw=1.0, marker="_", ms=5, mew=1.2,
              label="range, 6 tasks")]
order = [hb[0], hb[1], hb[2], hb[3], hb[4], hb[5]]
axB.legend(handles=order, loc="upper left", bbox_to_anchor=(-0.035, 1.025), ncol=2,
           handlelength=1.0, borderpad=0.15, labelspacing=0.26, handletextpad=0.4,
           columnspacing=0.9)
xb = (11.55 + 16.95) / 2 / W
fig.text(xb, 6.92 / H, "total measured = sent to measurement",
         fontsize=7, ha="center", va="top")
fig.text(xb, 6.63 / H, "+ not certified + budget m (A3, A4 only)",
         fontsize=7, ha="center", va="top")

# ---------------------------------------------------------------- data, c & d
B1 = R6["designs"]["B1_Tg"]
SP = B1["splits"]["comp"]
IQR = B1["describe"]["y_iqr"]
PD = B1["describe"]["patterns"]
PATS = sorted(PD, key=lambda p: -PD[p]["n"])          # ordered by n
PLAB = [p.replace("rho", RHO).replace("comp-only", "composition only") for p in PATS]
FUS = SP["models"]["hgb_fused"]
CMP = SP["models"]["hgb_comp"]
ypos = np.arange(len(PATS))[::-1]
OFFY = 0.20                                   # vertical split of the two series
# Head room above the top row, so that the key, the reference-line label and the
# first row of data each get their own band: no annotation is ever set over the
# data (and none is pushed under the last row, where it would collide with the
# CTE+nD interval, which runs out to 1.000).
YTOP = len(PATS) + 1.60
REFTOP, REFBOT = len(PATS) + 0.10, -0.45      # reference lines stop clear of row 0
LABY = len(PATS) + 0.35                       # both reference lines label at the top

# ------------------------------------------------------------------------- c
for ytk, p in zip(ypos, PATS):
    for src, col, mk, lab in [("pooled", style.ROLE["source"], style.MARK["source"],
                               "pooled calibration"),
                              ("mondrian", style.ROLE["recal"], style.MARK["recal"],
                               "Mondrian by pattern")]:
        c = FUS[src]["per_pattern"][p]["coverage"]
        axC.errorbar(c["mean"], ytk + (OFFY if src == "pooled" else -OFFY),
                     xerr=[[c["mean"] - c["lo"]], [c["hi"] - c["mean"]]], fmt=mk,
                     ms=3.8, mfc=col, mec=col, ecolor=col, elinewidth=0.8, capsize=1.6,
                     capthick=0.8, zorder=3, label=lab if ytk == ypos[0] else None)
    axC.text(0.6885, ytk, "%d" % PD[p]["n"], fontsize=7, ha="right", va="center")
axC.plot([0.90, 0.90], [REFBOT, REFTOP], color="black", lw=0.7, ls=(0, (4, 3)),
         zorder=0)
axC.text(0.905, LABY, "nominal 0.90", fontsize=7, ha="left", va="center")
axC.text(0.6885, len(PATS) - 0.10, "n", fontsize=7, ha="right", va="center",
         style="italic")
axC.set_yticks(ypos)
axC.set_yticklabels(PLAB)
axC.set_ylim(-0.75, YTOP)
axC.set_xlim(0.655, 1.005)
axC.set_xticks([0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
axC.spines["left"].set_bounds(-0.5, len(PATS) - 0.5)
axC.spines["bottom"].set_bounds(0.70, 1.00)
axC.set_xlabel("coverage")
axC.set_title("Coverage per missing-measurement pattern", fontsize=8)
axC.tick_params(axis="y", length=0)
axC.legend(loc="upper left", bbox_to_anchor=(-0.015, 1.03), ncol=2, handlelength=1.1,
           borderpad=0.15, labelspacing=0.24, handletextpad=0.4, columnspacing=1.6)
fig.text((2.32 + 8.55) / 2 / W, 0.62 / H,
         "%s = density, CTE = dilatometric expansion, nD = refractive index" % RHO,
         fontsize=7, ha="center", va="top")

# ------------------------------------------------------------------------- d
# Two MODELS, not two calibration states: no reserved ROLE hue is spent here.
# Filled square = fused, open square = composition-only; the square keeps its
# panel-c meaning (Mondrian, one quantile per pattern).
for ytk, p in zip(ypos, PATS):
    wf = FUS["mondrian"]["per_pattern"][p]["norm_width"]
    wc = CMP["mondrian"]["per_pattern"][p]["norm_width"]
    axD.plot([wf["mean"], wc["mean"]], [ytk + OFFY, ytk - OFFY], color=GREY, lw=0.7,
             zorder=1)
    axD.plot([wc["mean"]], [ytk - OFFY], marker="s", ms=3.9, mfc="white", mec=DGREY,
             mew=0.9, ls="none", zorder=3,
             label="composition-only model" if ytk == ypos[0] else None)
    axD.plot([wf["mean"]], [ytk + OFFY], marker="s", ms=3.7, mfc=DGREY, mec=DGREY,
             ls="none", zorder=4,
             label="fused model" if ytk == ypos[0] else None)
pooled_nw = FUS["pooled"]["width"]["mean"] / IQR
axD.plot([pooled_nw, pooled_nw], [REFBOT, REFTOP], color=GREY, lw=0.7,
         ls=(0, (1.5, 1.5)), zorder=0)
axD.text(pooled_nw + 0.012, LABY, "pooled %.2f" % pooled_nw, fontsize=7, ha="left",
         va="center", color=DGREY)
axD.set_yticks(ypos)
axD.set_yticklabels(PLAB)
axD.set_ylim(-0.75, YTOP)
axD.set_xlim(0.50, 1.02)
axD.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
axD.spines["left"].set_bounds(-0.5, len(PATS) - 0.5)
axD.spines["bottom"].set_bounds(0.5, 1.0)
axD.set_xlabel("normalised width (× target IQR)")
axD.set_title("Mondrian width per pattern", fontsize=8)
axD.tick_params(axis="y", length=0)
axD.legend(loc="upper left", bbox_to_anchor=(-0.015, 1.03), ncol=2, handlelength=1.0,
           borderpad=0.15, labelspacing=0.24, handletextpad=0.4, columnspacing=1.0)
fig.text((12.05 + 16.95) / 2 / W, 0.62 / H, "target IQR = %.0f K" % IQR, fontsize=7,
         ha="center", va="top")

fig.canvas.draw()


def _x0(ax):
    r = fig.canvas.get_renderer()
    return ax.get_tightbbox(r).transformed(fig.transFigure.inverted()).x0


# One left edge per COLUMN, as in fig4_design_shift: the letters of a and c (and
# of b and d) share an x even though panel c's y-tick labels are much wider.
for column in [[(axA, "a"), (axC, "c")], [(axB, "b"), (axD, "d")]]:
    xt = min(_x0(ax) for ax, _ in column) - 0.018
    for ax, L in column:
        style.panel(ax, L, x=xt - _x0(ax))

png = style.save(fig, "fig8_decisions", 8)
print(png)

# ---------------------------------------------------------------- printed numbers
print("\n[a] pooled false-accept proportion among accepted design-region candidates")
for arm, lab, _, _ in ARMS:
    row = []
    for ds, _ in DS:
        p = head(ds)["pooled"][arm]["true_target"]
        t, f = p["n_accept_total"], p["false_accepts_total"]
        row.append("%s %s" % (ds, "x (%d acc.)" % t if t < MIN_ACC
                              else "%.3f (%d/%d)" % (f / t, f, t)))
    print("  %-38s %s" % (lab, "; ".join(row)))
print("  %-38s %s" % ("A1, all accepted (grey)", "; ".join(
    ("%s %.3f (%d/%d)" % (ds, p["false_accepts_total"] / p["n_accept_total"],
                          p["false_accepts_total"], p["n_accept_total"]))
    if (p := head(ds)["pooled"]["A1_conformal"]["all"])["n_accept_total"] >= MIN_ACC
    else "%s x (%d acc.)" % (ds, p["n_accept_total"]) for ds, _ in DS)))
print("\n[b] glass Tg decision shares (n = %d)" % round(npool))
for arm in BARMS:
    a = A[arm]["all"]
    sh = [head(d)["arms"][arm]["all"]["measurements"]["mean"]
          / head(d)["arms"][arm]["all"]["n"]["mean"] for d, _ in DS]
    print("  %-19s acc %.3f rej %.3f meas %.3f not-cert %.3f | total measured %.3f "
          "| 6 datasets %.3f / %.3f / %.3f"
          % (arm, a["n_accept"]["mean"] / npool, a["n_reject"]["mean"] / npool,
             a["n_measure"]["mean"] / npool, a["n_not_certified"]["mean"] / npool,
             a["measurements"]["mean"] / npool, min(sh), float(np.median(sh)), max(sh)))
print("\n[c,d] B1_Tg, composition-grouped split, IQR = %.0f K, pooled width %.1f K "
      "(%.2f x IQR)" % (IQR, FUS["pooled"]["width"]["mean"], pooled_nw))
for p in PATS:
    cp = FUS["pooled"]["per_pattern"][p]["coverage"]
    cm = FUS["mondrian"]["per_pattern"][p]["coverage"]
    wf = FUS["mondrian"]["per_pattern"][p]["norm_width"]
    wc = CMP["mondrian"]["per_pattern"][p]["norm_width"]
    print("  %-11s n %4d  pooled %.3f [%.3f, %.3f]  Mondrian %.3f [%.3f, %.3f]"
          "  width %.2f vs %.2f x IQR (%.0f vs %.0f K)"
          % (p, PD[p]["n"], cp["mean"], cp["lo"], cp["hi"], cm["mean"], cm["lo"],
             cm["hi"], wf["mean"], wc["mean"], wf["mean"] * IQR, wc["mean"] * IQR))
