"""Figure S3 - histology detail (Supporting Information; supports Section 5.3).

Panels
  a  coverage by domain, source-calibrated (k = 0) and target-recalibrated (k = 30)
  b  mean prediction-set size for the same conditions (validity on T2 is bought
     with near-complete sets)
  c  class-conditional coverage (minimum over the classes PRESENT in that domain,
     i.e. over 9 PathMNIST classes for source-test / T1 / stress test but over the
     5 unambiguous Kather classes on T2 -- the class count is tagged on every row
     group because the minima are therefore NOT comparable across rows)
  d  slide-grouped recalibration sensitivity: per-split coverage at k = 200 when
     the recalibration tiles and the test tiles come from disjoint slides

Domains: S source-test (in-distribution half of PathMNIST val), T1 = PathMNIST
test cohort CRC-VAL-HE-7K (genuine shift, independent patients), T2 = Kather
et al. 2016 external cohort (genuine external shift) and the SECONDARY synthetic
colour stress test of the submitted manuscript.

Score: LAC; alpha = 0.10; mean and [2.5, 97.5] percentile over 20 splits x 3
training seeds (60 values; 20 for the single cached submitted model).

Data: revision/results/R5_imaging.json (exp_R5_imaging.py; report
revision/results/R5_imaging.md).  Drawn at the final print size (17.5 cm wide);
never scale this figure.
"""
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import style  # noqa: E402

style.apply()

with open(os.path.join(REV, "results", "R5_imaging.json"), "r", encoding="utf-8") as fh:
    R = json.load(fh)
ARCH = R["architectures"]
META = R["meta"]
K0, KR, KD = "0", str(META["k_headline_protocol"]), str(META["k_submitted"])
NCLS = len(META["pathmnist_label_order"])

MODELS = [("SmallCNN", "small CNN"), ("ResNet18", "ResNet-18"),
          ("SmallCNN-submitted", "submitted model")]
# (json key, label line 1, label line 2, is the row a genuine shift?)
DOMAINS = [("S_source-test", "source-test", "in-distribution", None),
           ("T1_CRC-VAL-HE-7K", "T1 PathMNIST test", "genuine shift", True),
           ("T2_Kather2016_5class", "T2 Kather 2016", "genuine external", True),
           ("STRESS_T1_synthetic_colour", "colour stress test", "synthetic", False)]

# number of classes the panel-c minimum is taken over, per domain (read from the
# JSON itself so the tag can never drift from the data)
NC = {d[0]: len(ARCH["SmallCNN"]["results"][d[0]]["LAC"][K0]["class_cov"]) for d in DOMAINS}

C_SRC, C_REC, C_FAIL = style.ROLE["source"], style.ROLE["recal"], style.ROLE["fail"]
M_SRC, M_REC, M_FAIL = style.MARK["source"], style.MARK["recal"], style.MARK["fail"]
GREY, LGREY = style.OI["grey"], style.OI["lightgrey"]
NOTE = "#444444"

# ---------------------------------------------------------------- layout (cm)
W, H = 17.40, 8.50
AX_TOP, AX_H = 1.05, 5.62          # measured from the top of the canvas
A_L, A_W = 4.05, 2.75
B_L, B_W = 7.15, 2.25
C_L, C_W = 9.80, 2.70              # widened: the right margin carries the class-count tag
D_L, D_W = 13.40, 3.35
LAB_X = 0.10                       # left edge of the domain labels


def box(x_cm, y_top_cm, w_cm, h_cm):
    return [x_cm / W, 1.0 - (y_top_cm + h_cm) / H, w_cm / W, h_cm / H]


fig = plt.figure(figsize=(W * style.CM, H * style.CM))
axA = fig.add_axes(box(A_L, AX_TOP, A_W, AX_H))
axB = fig.add_axes(box(B_L, AX_TOP, B_W, AX_H))
axC = fig.add_axes(box(C_L, AX_TOP, C_W, AX_H))
axD = fig.add_axes(box(D_L, AX_TOP, D_W, AX_H))

# --------------------------------------------------------------- row geometry
STEP, GAP = 1.0, 0.95              # rows inside a group; extra gap between groups
YPOS = {}
for g, (dkey, _, _, _) in enumerate(DOMAINS):
    for i, (mkey, _) in enumerate(MODELS):
        YPOS[(dkey, mkey)] = -(g * (3 * STEP + GAP) + i * STEP)
Y_LO = min(YPOS.values()) - 1.45   # footer strip for the "all 9" tag in b
Y_HI = 1.15                        # header strip for the "empty sets" note in b


def stat(mkey, dkey, k, field):
    """mean / lo / hi of one LAC field; returns None when the condition does not exist."""
    node = ARCH[mkey]["results"][dkey]["LAC"]
    if k not in node:
        return None
    s = node[k][field]
    return s["mean"], s["lo"], s["hi"]


def dotpanel(ax, field, xlim, xticks, nominal_line, data_hi=None):
    """One metric column: k = 0 and k = 30 for every model x domain row.

    ``data_hi`` is the x value where the data region ends; backgrounds, the rule
    between domain groups and the axis spine stop there, so that any margin to
    its right stays clean for in-axes tags.
    """
    hi = xlim[1] if data_hi is None else data_hi
    frac = (hi - xlim[0]) / (xlim[1] - xlim[0])
    for g, (dkey, _, _, genuine) in enumerate(DOMAINS):
        ys = [YPOS[(dkey, m)] for m, _ in MODELS]
        if genuine is False:                      # synthetic stress test band
            ax.axhspan(min(ys) - 0.55, max(ys) + 0.55, xmax=frac, color=LGREY,
                       alpha=0.55, lw=0, zorder=0)
        if g:                                     # rule between domain groups
            ax.axhline(max(ys) + 0.98, xmax=frac, color=LGREY, lw=0.6, zorder=0)
    if nominal_line:
        ax.axvline(0.90, color=style.ROLE["ideal"], lw=0.7, ls=(0, (4, 3)), zorder=1)
    for (dkey, _, _, _) in DOMAINS:
        for (mkey, _) in MODELS:
            y = YPOS[(dkey, mkey)]
            a, b = stat(mkey, dkey, K0, field), stat(mkey, dkey, KR, field)
            if b is not None:
                ax.plot([a[0], b[0]], [y, y], color=GREY, lw=0.7, zorder=2)
            # the circle is drawn larger than the square so that an unchanged
            # condition (source == recalibrated) still shows both markers
            for s, col, mk, ms, z in ((a, C_SRC, M_SRC, 4.8, 4), (b, C_REC, M_REC, 3.0, 5)):
                if s is None:
                    continue
                ax.errorbar(s[0], y, xerr=[[s[0] - s[1]], [s[2] - s[0]]], fmt=mk,
                            ms=ms, mfc=col, mec="white", mew=0.4, ecolor=col,
                            elinewidth=0.7, capsize=1.3, capthick=0.7, zorder=z)
    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    ax.spines["bottom"].set_bounds(xticks[0], xticks[-1])
    ax.set_ylim(Y_LO, Y_HI)
    ax.set_yticks([YPOS[(d[0], m[0])] for d in DOMAINS for m in MODELS])
    ax.tick_params(axis="y", length=0, pad=2)
    ax.tick_params(axis="x", pad=2)


# ------------------------------------------------------------------ a, b, c
dotpanel(axA, "coverage", (-0.055, 1.03), [0, 0.25, 0.5, 0.75, 1.0], True)
dotpanel(axB, "set_size", (-0.3, NCLS + 0.3), [0, 3, 6, 9], False)
dotpanel(axC, "class_min_cov", (-0.055, 1.355), [0, 0.25, 0.5, 0.75, 1.0], True,
         data_hi=1.03)

axA.set_yticklabels([lab for _ in DOMAINS for _, lab in MODELS])
for ax in (axB, axC):
    ax.set_yticklabels([])

# b: the two things a reader has to be told inside the axes (kept on separate
# strips so that they never run into each other in this narrow panel)
axB.axvline(NCLS, color=GREY, lw=0.6, ls=(0, (1.5, 1.5)), zorder=1)
axB.text(0.2, Y_HI - 0.05, "< 1: empty sets", fontsize=7, color=NOTE, ha="left", va="top")
axB.text(NCLS - 0.3, Y_LO + 0.15, "all 9", fontsize=7, color=NOTE, ha="right", va="bottom")

# c: the minimum is taken over the classes present in that domain -- 9 everywhere
# except on the 5-class Kather cohort, so the rows are NOT directly comparable
for (dkey, _, _, _) in DOMAINS:
    ys = [YPOS[(dkey, m)] for m, _ in MODELS]
    axC.text(1.09, 0.5 * (min(ys) + max(ys)), f"of {NC[dkey]}", fontsize=7,
             color=NOTE, ha="left", va="center")

axA.set_xticklabels(["0", "0.25", "0.50", "0.75", "1"])
axC.set_xticklabels(["0", "", "0.5", "", "1"])   # narrower data region: label every 2nd tick
axA.set_title("Marginal coverage", pad=4)
axB.set_title("Mean set size", pad=4)
axC.set_title("Worst-class coverage", pad=4)

# ------------------------------------------------ domain labels (left margin)
xfrac = (LAB_X - A_L) / A_W
for (dkey, l1, l2, _) in DOMAINS:
    ys = [YPOS[(dkey, m)] for m, _ in MODELS]
    yc = 0.5 * (min(ys) + max(ys))
    axA.text(xfrac, yc + 0.30, l1, transform=axA.get_yaxis_transform(), fontsize=7.5,
             fontweight="bold", ha="left", va="center", clip_on=False)
    axA.text(xfrac, yc - 0.42, l2, transform=axA.get_yaxis_transform(), fontsize=7,
             ha="left", va="center", color=NOTE, clip_on=False)

# ------------------------------------- shared legend for a-c (never over d)
handles = [Line2D([], [], color=C_SRC, marker=M_SRC, ms=4.8, mec="white", mew=0.4,
                  ls="none", label="source-calibrated (k = 0)"),
           Line2D([], [], color=C_REC, marker=M_REC, ms=3.0, mec="white", mew=0.4,
                  ls="none", label="target-recalibrated (k = 30)"),
           Line2D([], [], color=style.ROLE["ideal"], lw=0.7, ls=(0, (4, 3)),
                  label="nominal 0.90")]
LEG_L = 3.55                       # left-anchored so the key can never reach panel d
leg = fig.legend(handles=handles, loc="upper left",
                 bbox_to_anchor=(LEG_L / W, 1.0 - 0.06 / H), ncol=3,
                 handlelength=1.3, handletextpad=0.35, columnspacing=0.9, borderpad=0.0,
                 borderaxespad=0.0)

# ------------------------------------------------------------------------- d
GRP = [("T2_Kather2016_5class", f"tile-level split (k = {KD})", C_REC, M_REC),
       ("T2sens_Kather2016_5class_slidegrouped", f"slide-grouped split (k = {KD})",
        C_FAIL, M_FAIL)]
rng = np.random.default_rng(3)
axD.axhline(0.90, color=style.ROLE["ideal"], lw=0.7, ls=(0, (4, 3)), zorder=1)
for gi, (mkey, mlab) in enumerate(MODELS):
    per = ARCH[mkey]["per_split_LAC"]
    for si, (dkey, _, col, mk) in enumerate(GRP):
        x = gi + (-0.23 if si == 0 else 0.23)
        v = np.array([r[f"{dkey}_k{KD}"] for r in per])
        axD.plot(x + rng.uniform(-0.085, 0.085, v.size), v, ls="none", marker="o",
                 ms=1.6, mfc=col, mec="none", alpha=0.35, zorder=2)
        s = stat(mkey, dkey, KD, "coverage")
        axD.plot([x, x], [s[1], s[2]], color=col, lw=0.8, solid_capstyle="butt",
                 zorder=3)
        axD.plot([x], [s[0]], marker=mk, ms=4.0, mfc=col, mec="white", mew=0.6,
                 ls="none", zorder=4)
axD.set_xlim(-0.62, 2.62)
axD.set_ylim(0.44, 1.16)           # data span 0.48-1.00; the strip above 1.00 holds the key
axD.set_xticks(range(len(MODELS)))
axD.set_xticklabels([lab for _, lab in MODELS], rotation=22, ha="right",
                    rotation_mode="anchor")
axD.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
axD.set_yticklabels(["0.5", "", "0.7", "", "0.9", "1"])
axD.spines["left"].set_bounds(0.5, 1.0)
axD.set_ylabel("coverage", labelpad=2)
axD.set_xlabel("T2 Kather 2016\n(external cohort)", labelpad=2, linespacing=1.25)
axD.set_title("Recalibration split", pad=4)
axD.tick_params(pad=2)
hD = [Line2D([], [], color=c, marker=mk, ms=3.6, ls="none", mec="white", mew=0.5,
             label=lab) for _, lab, c, mk in GRP]
axD.legend(handles=hD, loc="upper left", bbox_to_anchor=(-0.04, 1.005), ncol=1,
           handletextpad=0.35, labelspacing=0.28, borderpad=0.0, borderaxespad=0.0)

# ------------------------------------------------------------- panel letters
fig.canvas.draw()


def letter(ax, lab, x_cm):
    bb = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    style.panel(ax, lab, x=x_cm / W - bb.x0)


letter(axA, "a", 0.05)
letter(axB, "b", B_L - 0.30)
letter(axC, "c", C_L - 0.30)
letter(axD, "d", D_L - 1.05)

# ---- geometry check: the shared a-c key must not reach into panel d's column
fig.canvas.draw()
lb = leg.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
print(f"a-c legend spans {lb.x0 * 2.54:.2f}-{lb.x1 * 2.54:.2f} cm "
      f"(axC right edge {C_L + C_W:.2f} cm, d letter at {D_L - 1.05:.2f} cm)")
assert lb.x1 * 2.54 <= C_L + C_W + 0.02, "legend overhangs panel d"

png = style.save(fig, "figS3_imaging_detail", "S3")
print(png)
bb = fig.get_tightbbox(fig.canvas.get_renderer())
print("tight bbox (cm): x", round(bb.x0*2.54,2), round(bb.x1*2.54,2), " y", round(bb.y0*2.54,2), round(bb.y1*2.54,2), " size", np.round(np.array(bb.size)*2.54+0.508,2))

# ------------------------------------------------------- numbers for the caption
print("\n--- values drawn (LAC, mean [2.5, 97.5] over splits x seeds) ---")
for mkey, mlab in MODELS:
    for dkey, l1, _, _ in DOMAINS:
        row = [mlab, f"{l1} (min over {NC[dkey]} cl.)"]
        for k in (K0, KR):
            for f in ("coverage", "set_size", "class_min_cov"):
                s = stat(mkey, dkey, k, f)
                row.append("-" if s is None else f"{s[0]:.3f}[{s[1]:.3f},{s[2]:.3f}]")
        print(" | ".join(f"{c:<24}" for c in row))
print("\n--- panel d, k = 200 ---")
for mkey, mlab in MODELS:
    for dkey, lab, _, _ in GRP:
        s = stat(mkey, dkey, KD, "coverage")
        v = np.array([r[f"{dkey}_k{KD}"] for r in ARCH[mkey]["per_split_LAC"]])
        print(f"{mlab:<16} {lab:<30} {s[0]:.3f} [{s[1]:.3f}, {s[2]:.3f}]  "
              f"n={v.size} min={v.min():.3f} max={v.max():.3f} "
              f"below 0.90: {np.mean(v < 0.90):.2f}")
print("\n--- panel c: which class is the 9-class minimum? ---")
for mkey, mlab in MODELS:
    for dkey, l1, _, _ in DOMAINS:
        cc = ARCH[mkey]["results"][dkey]["LAC"][K0]["class_cov"]
        am = min(cc, key=lambda c: cc[c]["mean"])
        print(f"{mlab:<16} {l1:<20} n_classes={len(cc)} argmin(mean per-class cov)={am}")
