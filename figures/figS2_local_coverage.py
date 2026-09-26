"""Figure S2 - local coverage of source-calibrated intervals in glass design space.

Corrects the submitted Figure 12.

a  kernel-smoothed local coverage of the source-calibrated intervals over the
   first two principal components of a centred-log-ratio (CLR) embedding of the
   7,623 glass compositions, with the source glasses (density contours enclosing
   50 % and 90 % OF THE SOURCE GLASSES, not of the kernel mass) and all 258
   design-region (high-phosphorus) glasses marked, and a contour at the nominal
   0.90;
b  direct coverage (marginal, source-test, intermediate, design region), the
   submitted marginal before and after removing the calibration records, and the
   kernel-smoothed local value at the design-region locations for five
   bandwidths, as a dot plot.

Every number is read from revision/results/R9_legacy.json key 'local'
(producing script revision/code/exp_R9_local.py, report revision/results/R9_legacy.md
section 2): 20 protocol design splits, alpha = 0.10, mean [2.5, 97.5 percentile].
Nothing is hard-coded.

Design notes (revision QA):
* The axes of a cover the whole design-region point cloud, which runs to
  PC2 = -9.04, below the stored 90 x 90 field grid. Cells with no estimate
  (kernel mass < 3) and the strip below the grid share ONE light-grey, so the
  "n = 258" printed in the panel is the number of markers actually drawn.
* The density contour levels are the 50th and 10th percentiles of the kernel
  density EVALUATED AT THE 6,592 SOURCE GLASSES, so they enclose 50 % and 90 %
  of those glasses by construction (levels taken from the cumulative grid mass
  enclose 62 % and 94 %, which is not what the label says).
* Colour encodes the GROUP OF GLASSES, identically in both panels and as in
  Fig. 4 / Fig. S4: source = ROLE['source'] circle, design region =
  ROLE['recal'] square, intermediate (excluded) = grey, pooled = black,
  superseded value = grey open pentagon (a shape used by no MARK or DECISION).
  No group changes colour between the two panels. ROLE['fail'] is not used:
  nothing here is a failing method, and the under-coverage is shown by
  position against the nominal rule, which every row is worded for.
* Kernel-smoothed values are ROLE['heuristic'] orange diamonds - a smoothed
  estimate, not a measured coverage - matching MARK['heuristic'].
"""
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import style  # noqa: E402

style.apply()
L = json.load(open(os.path.join(REV, "results", "R9_legacy.json")))["local"]

EPS = "0.0001"                       # main pseudo-count of the CLR embedding
NW = L["eps"][EPS]["nw"]
GREY = style.OI["grey"]
BLANK = style.OI["lightgrey"]        # "no local estimate" - one colour, one meaning
TGT = style.ROLE["recal"]            # design-region glasses, as in Fig. 4 / Fig. S4
HEUR = style.ROLE["heuristic"]       # kernel-smoothed estimate
H_DENS = 1.15                        # bandwidth of the source-density contours in a

# ---------------------------------------------------------------- data
F = L["field"]
gx, gy = np.array(F["gx"]), np.array(F["gy"])
FLD = np.ma.masked_invalid(
    np.array([[np.nan if v is None else v for v in row] for row in F["scott_x1"]], float))
H_FIELD = NW["scott_x1"]["h"]["mean"]

pc1 = np.array(L["points"]["pc1"])
pc2 = np.array(L["points"]["pc2"])
role = np.array(L["points"]["role"])
evr = L["pca_explained_variance"][EPS]

NROLE = L["n_eval_by_role_seed0"]
N_EVAL = sum(NROLE.values())
LEG = L["legacy"]

msrc, mtgt = role == "source", role == "target"


def kde_at(qx, qy, px, py, h):
    """Unnormalised Gaussian kernel density of (px, py) evaluated at (qx, qy)."""
    D = np.zeros(len(qx))
    for i in range(0, len(px), 400):
        dx = qx[:, None] - px[i:i + 400]
        dy = qy[:, None] - py[i:i + 400]
        D += np.exp(-(dx * dx + dy * dy) / (2 * h * h)).sum(1)
    return D


# ---------------------------------------------------------------- panel-a window
# half a cell beyond the stored field grid horizontally and at the top, and far
# enough down to hold EVERY design-region glass (min PC2 = -9.04).
sx = (gx[-1] - gx[0]) / (len(gx) - 1)
sy = (gy[-1] - gy[0]) / (len(gy) - 1)
XLIM = (gx[0] - sx / 2, gx[-1] + sx / 2)
YLIM = (float(pc2[mtgt].min()) - 0.16, gy[-1] + sy / 2)
N_IN = int(((pc1[mtgt] >= XLIM[0]) & (pc1[mtgt] <= XLIM[1]) &
            (pc2[mtgt] >= YLIM[0]) & (pc2[mtgt] <= YLIM[1])).sum())
assert N_IN == int(mtgt.sum()), f"{int(mtgt.sum()) - N_IN} design-region glasses clipped"

# density contours: levels fixed by the density AT THE SOURCE GLASSES, so that
# they enclose 50 % and 90 % of them (not 50 %/90 % of the grid's kernel mass).
DP = kde_at(pc1[msrc], pc2[msrc], pc1[msrc], pc2[msrc], H_DENS)
LEV = list(np.quantile(DP, [0.10, 0.50]))            # ascending: 90 % then 50 %
ENC = [float((DP >= lv).mean()) for lv in LEV]
dgx = np.linspace(XLIM[0], XLIM[1], 180)
dgy = np.linspace(YLIM[0], YLIM[1], 200)
DGX, DGY = np.meshgrid(dgx, dgy)
DENS = kde_at(DGX.ravel(), DGY.ravel(), pc1[msrc], pc2[msrc], H_DENS).reshape(DGX.shape)

# ---------------------------------------------------------------- panel-b rows
# colour/marker per GROUP OF GLASSES, identical in both panels:
#   source = ROLE['source'] circle, design region = ROLE['recal'] square,
#   intermediate = grey, pooled = black, superseded = grey open pentagon.
# last field: side on which the numeral is printed ("all evaluated" goes left, so
# that its label does not run into the nominal-0.90 rule).
DIRECT = [
    (f"all evaluated ({N_EVAL:,})", L["marginal"], style.OI["black"], "o", False, "left"),
    (f"source-test ({NROLE['source_test']:,})", L["by_role"]["source_test"],
     style.ROLE["source"], style.MARK["source"], True, "right"),
    (f"intermediate ({NROLE['intermediate']:,})", L["by_role"]["intermediate"],
     GREY, style.MARK["other"], True, "right"),
    (f"design region ({NROLE['target']:,})", L["by_role"]["target"],
     TGT, style.MARK["recal"], True, "right"),
]
HKEYS = ["scott_x0.25", "scott_x0.5", "abs_0.55", "scott_x1", "scott_x2"]
HNOTE = {"abs_0.55": " (submitted h)", "scott_x1": " (Scott)"}

# ---------------------------------------------------------------- canvas
W, H = 17.5, 8.0           # final print size in cm; saved without bbox cropping
fig = plt.figure(figsize=(W * style.CM, H * style.CM))
fig.patch.set_facecolor("white")


def box(x0, y0, x1, y1):
    return fig.add_axes([x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H])


AX, AY0, AY1 = 1.45, 1.32, 6.66                       # panel a: equal aspect
AH = AY1 - AY0
AW = AH * (XLIM[1] - XLIM[0]) / (YLIM[1] - YLIM[0])
axA = box(AX, AY0, AX + AW, AY1)
cax = box(AX + AW + 0.20, AY0, AX + AW + 0.42, AY1)
axB = box(11.05, AY0, 17.15, AY1)

# ---------------------------------------------------------------- a
cmap = plt.get_cmap("cividis").copy()
cmap.set_bad(BLANK)
axA.set_facecolor(BLANK)                 # the strip below the grid reads the same
mesh = axA.pcolormesh(gx, gy, FLD, cmap=cmap, vmin=0.55, vmax=1.0,
                      shading="nearest", zorder=0)

# source glasses: contours enclosing 50 % (solid) and 90 % (dashed) of them
CSRC = axA.contour(dgx, dgy, DENS, levels=LEV, colors=[style.ROLE["source"]] * 2,
                   linewidths=[0.9, 0.9], linestyles=["--", "-"], zorder=2)
CSRC.set(path_effects=[pe.Stroke(linewidth=1.9, foreground="white"), pe.Normal()])
# every design-region glass, individually
axA.scatter(pc1[mtgt], pc2[mtgt], s=5.0, marker=style.MARK["recal"],
            facecolor=TGT, edgecolor="white", linewidths=0.3, zorder=3)
# nominal level
axA.contour(gx, gy, FLD, levels=[0.90], colors="white", linewidths=1.3, zorder=4)

WSTROKE = [pe.Stroke(linewidth=1.7, foreground="white"), pe.Normal()]
DSTROKE = [pe.Stroke(linewidth=2.0, foreground="0.15"), pe.Normal()]
# the two keys sit in opposite bottom corners, on the "no estimate" band, where
# there is neither a contour nor a design-region glass: no key covers data.
axA.text(XLIM[0] + 0.22, YLIM[0] + 0.12,
         "source glasses,\n50 % / 90 %", fontsize=7,
         color=style.ROLE["source"], ha="left", va="bottom", linespacing=1.25,
         path_effects=WSTROKE, zorder=5)
axA.text(XLIM[1] - 0.25, -8.30, "design region\n(high P), n = %s" % f"{int(mtgt.sum()):,}",
         fontsize=7, color=TGT, ha="right", va="bottom", linespacing=1.25,
         path_effects=WSTROKE, zorder=5)
axA.text(6.9, 1.35, "nominal 0.90", fontsize=7, color="white", ha="right", va="bottom",
         path_effects=DSTROKE, zorder=5)

axA.set_xlim(*XLIM)
axA.set_ylim(*YLIM)
axA.set_aspect("equal", adjustable="box")
axA.set_xticks([-4, -2, 0, 2, 4, 6, 8])
axA.set_yticks([-8, -6, -4, -2, 0, 2, 4])
axA.set_xlabel(f"PC1 ({100 * evr[0]:.1f} % of CLR variance)")
axA.set_ylabel(f"PC2 ({100 * evr[1]:.1f} %)")
axA.set_title("Source-calibrated local coverage", fontsize=8)
for s in ("top", "right"):
    axA.spines[s].set_visible(True)
    axA.spines[s].set_linewidth(0.6)

fig.text((AX + AW / 2) / W, 0.17 / H, "grey: fewer than 3 effective records",
         fontsize=7, color="0.35", ha="center", va="bottom")

cb = fig.colorbar(mesh, cax=cax)
cb.set_label(f"local coverage (h = {H_FIELD:.2f})", fontsize=8, labelpad=3)
cb.set_ticks([0.6, 0.7, 0.8, 0.9, 1.0])
cb.ax.tick_params(labelsize=7, width=0.6, length=2.5)
cb.ax.plot([0, 1], [0.90, 0.90], color="white", lw=1.3,
           transform=cb.ax.get_yaxis_transform(), clip_on=False)
cb.outline.set_linewidth(0.6)

# ---------------------------------------------------------------- b
Y_LEG = 9.90                                    # superseded value, set apart
YD = [7.60, 6.75, 5.90, 5.05]                   # direct coverage, corrected set
YH = [3.45, 2.70, 1.95, 1.20, 0.45]             # kernel-smoothed, design region

axB.plot([0.90, 0.90], [-0.55, 10.45], color=style.ROLE["ideal"], lw=0.7,
         ls=(0, (4, 3)), zorder=0)
tgt = L["by_role"]["target"]["mean"]
axB.plot([tgt, tgt], [0.15, 5.05], color=TGT, lw=0.7, ls=(0, (1.5, 1.5)), zorder=1)

# the correction the panel exists to make: 0.81 -> 0.64 on removing the
# 1,978 calibration records, both values plotted, not only the arrow's label.
axB.annotate("", xy=(LEG["marginal_excl_cal_rows"], Y_LEG),
             xytext=(LEG["marginal_incl_cal_rows"], Y_LEG),
             arrowprops=dict(arrowstyle="-|>", color=GREY, lw=0.7,
                             shrinkA=4.5, shrinkB=4.5, mutation_scale=6), zorder=2)
for xv, ha, dx in [(LEG["marginal_incl_cal_rows"], "left", 0.012),
                   (LEG["marginal_excl_cal_rows"], "right", -0.012)]:
    axB.plot([xv], [Y_LEG], marker="p", ms=4.2, mfc="white", mec=GREY, mew=0.9,
             ls="none", zorder=3)
    axB.text(xv + dx, Y_LEG, f"{xv:.3f}", fontsize=7, color=GREY, ha=ha, va="center")
axB.text(0.5 * (LEG["marginal_incl_cal_rows"] + LEG["marginal_excl_cal_rows"]), 9.25,
         "without those records", fontsize=7, color=GREY, ha="center", va="center")

for y, (lab, v, col, mk, fl, side) in zip(YD, DIRECT):
    axB.errorbar(v["mean"], y, xerr=[[v["mean"] - v["lo"]], [v["hi"] - v["mean"]]],
                 fmt=mk, ms=4.0, mfc=col if fl else "white", mec=col, mew=0.9,
                 ecolor=col, elinewidth=0.8, capsize=1.6, capthick=0.8, zorder=3)
    xt, ha = ((v["hi"] + 0.013, "left") if side == "right" else (v["lo"] - 0.013, "right"))
    axB.text(xt, y, f"{v['mean']:.3f}", fontsize=7, color=col, ha=ha, va="center")
for y, hk in zip(YH, HKEYS):
    v = NW[hk]["local_at_target_loo"]       # leave-one-out: the in-sample form is biased (R9_legacy.md)
    axB.errorbar(v["mean"], y, xerr=[[v["mean"] - v["lo"]], [v["hi"] - v["mean"]]],
                 fmt=style.MARK["heuristic"], ms=3.6, mfc=HEUR, mec=HEUR,
                 ecolor=HEUR, elinewidth=0.8, capsize=1.6, capthick=0.8, zorder=3)

axB.set_yticks([Y_LEG] + YD + YH)
axB.set_yticklabels(["submitted (Fig. 12)\n1 split, old eval. set"]
                    + [lab for lab, *_ in DIRECT]
                    + [f"h = {NW[hk]['h']['mean']:.2f}{HNOTE.get(hk, '')}" for hk in HKEYS])
TICKCOL = [GREY] + [r[2] for r in DIRECT] + ["black"] * len(YH)
for t, c in zip(axB.get_yticklabels(), TICKCOL):
    t.set_color(c)

axB.text(0.345, 8.30, "direct coverage, 20 splits", fontsize=7.5, style="italic",
         ha="left", va="center")
axB.text(0.345, 4.25, "kernel-smoothed (leave-one-out), design region", fontsize=7.5,
         style="italic", color=HEUR, ha="left", va="center",
         path_effects=WSTROKE)
axB.text(0.888, -0.15, "nominal 0.90", fontsize=7, ha="right", va="center")
axB.text(0.335, 13.00,
         "Submitted 0.81 counted the %s calibration\n"
         "records as evaluation data; submitted 0.64\n"
         "was smoothed, the direct value is 0.40."
         % f"{LEG['n_cal_rows_in_eval']:,}",
         fontsize=7, ha="left", va="top", linespacing=1.35)

axB.set_xlim(0.33, 1.02)
axB.set_ylim(-0.60, 13.05)
axB.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
axB.set_xlabel("coverage")
axB.set_title("Direct and kernel-smoothed coverage", fontsize=8)
axB.tick_params(axis="y", length=0)
axB.spines["left"].set_visible(False)

fig.canvas.draw()
for ax, letter in [(axA, "a"), (axB, "b")]:
    style.panel(ax, letter)
png = style.save(fig, "figS2_local_coverage", "S2", tight=False)

# ---------------------------------------------------------------- verification
print(png)
print("panel a window x %.3f..%.3f  y %.3f..%.3f ; design-region glasses drawn %d/%d"
      % (XLIM[0], XLIM[1], YLIM[0], YLIM[1], N_IN, int(mtgt.sum())))
print("density contour h = %.2f ; levels %.1f / %.1f enclose %.3f / %.3f of the "
      "%d source glasses" % (H_DENS, LEV[1], LEV[0], ENC[1], ENC[0], int(msrc.sum())))
print("field %.3f..%.3f, masked %.3f of cells"
      % (FLD.min(), FLD.max(), float(FLD.mask.mean())))
S1 = NW["scott_x1"]
print("supported cells %.3f, of which below 0.90: %.3f (all cells %.3f)"
      % (S1["grid_frac_supported"]["mean"], S1["grid_frac_supported_below_0.9"]["mean"],
         S1["grid_frac_all_cells_below_0.9"]["mean"]))
print("marginal %.3f [%.3f, %.3f]" % (L["marginal"]["mean"], L["marginal"]["lo"],
                                      L["marginal"]["hi"]))
for k in ("source_test", "intermediate", "target"):
    v = L["by_role"][k]
    print("%-12s %.3f [%.3f, %.3f]" % (k, v["mean"], v["lo"], v["hi"]))
for hk in HKEYS:
    v = NW[hk]["local_at_target_loo"]
    print("  h=%.2f  %.3f [%.3f, %.3f]" % (NW[hk]["h"]["mean"], v["mean"], v["lo"], v["hi"]))
print("submitted %.4f  excl-cal %.4f  n_cal %d  legacy local h0.55 %.4f"
      % (LEG["marginal_incl_cal_rows"], LEG["marginal_excl_cal_rows"],
         LEG["n_cal_rows_in_eval"], LEG["local_at_target_h0.55"]))
KNN = L["eps"][EPS]["knn_target"]
print("kNN at design region: " + ", ".join("k=%s %.3f" % (k, KNN[k]["mean"])
                                           for k in ("25", "50", "100")))
