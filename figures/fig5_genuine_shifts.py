"""Figure 5 - genuine, non-constructed distribution shifts.

a  SciGlass: coverage on the target-test half for three genuine shift families
   (held-out laboratories, future publications, chemically analysed compositions)
   x four glass properties x five calibration schemes.
b  the conditional failure and its fix: distribution of per-laboratory coverage
   under row-calibrated (i.i.d.) intervals versus laboratory-specific
   recalibration on 10 of that laboratory's own glasses; 45S5 Bioglass example.
c  computed versus measured labels (DFT -> calorimetry): marginal coverage sits
   on nominal at every recalibration budget, but hides chemistry-class spread.
d  external histology cohort: coverage and prediction-set size for two CNNs on
   the source-test half, the PathMNIST test cohort (T1) and Kather-2016 (T2).

Colour convention (same as Figures 3-4).  In panels a and b colour + marker
encode the CALIBRATION STATE: blue circle = row-calibrated (i.i.d.), bluish
green square = target-recalibrated, sky blue triangle = laboratory-calibrated,
purple plus = weighted conformal, open grey triangle = in-distribution.  In
panels c and d colour carries no calibration meaning - the two models are drawn
in neutral black / dark grey with different markers and line styles - and the
calibration state is the open versus filled marker face.  No mathtext is used,
so the only fonts embedded in the PDF are ArialMT and Arial-BoldMT.

Readability rules applied throughout (revision QA):
* every key sits OUTSIDE the data rectangle of the axes it explains, so that no
  legend glyph can be read as a datum;
* every series carries line style or marker AND a word label, never colour
  alone;
* x positions are never allowed to imply a quantity they do not carry: panel c
  puts the k = 0 computed-label reference in its own shaded block and spaces the
  recalibration budgets on equal slots, panel d separates the two calibration
  states of each model horizontally so that neither marker can hide the other.

All numbers are read from revision/results/R4_realshift.json,
revision/results/R4b_dft_expt.json and revision/results/R5_imaging.json at plot
time and aggregated here (mean and 2.5-97.5 percentiles over the evaluations);
nothing is hard-coded.  Reviewer 1 comment 2; Reviewer 2 comments 3 and 8.
"""
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import style  # noqa: E402

style.apply()
RES = os.path.join(REV, "results")
MIRROR = os.path.join(REV, "repo", "trustlayer-biomat", "results")
PROP_KEYS = ["Tg", "YoungModulus", "Microhardness", "Tliquidus"]
FAM_KEYS = ["A_labout", "B_temporal", "C_analysed"]


def load_R4():
    """The full (quick = False) R4 run, from results/ or the packaged mirror.

    revision/results/R4_realshift.json is the live working file and is rewritten
    in place while a run is going; a partially written file, or a file left
    behind by a ``quick`` smoke run, must never reach the figure.  The loader
    therefore validates the run before accepting it and says which file it used.
    """
    reasons = []
    for path in (os.path.join(RES, "R4_realshift.json"),
                 os.path.join(MIRROR, "R4_realshift.json")):
        if not os.path.exists(path):
            reasons.append(f"{path}: missing")
            continue
        try:
            d = json.load(open(path))
        except json.JSONDecodeError as exc:            # being written right now
            reasons.append(f"{path}: not readable JSON ({exc})")
            continue
        if d.get("config", {}).get("quick"):
            reasons.append(f"{path}: quick = True (smoke run)")
            continue
        missing = [f"{f}/{p}" for f in FAM_KEYS for p in PROP_KEYS
                   if p not in d.get(f, {})] + (
                       [] if "E_45S5" in d else ["E_45S5"])
        if missing:
            reasons.append(f"{path}: incomplete, missing {len(missing)} blocks "
                           f"(e.g. {missing[0]})")
            continue
        print(f"R4 source: {path}")
        return d
    raise SystemExit("no complete R4_realshift.json:\n  " + "\n  ".join(reasons))


R4 = load_R4()
R4B = json.load(open(os.path.join(RES, "R4b_dft_expt.json")))
R5 = json.load(open(os.path.join(RES, "R5_imaging.json")))

OI, ROLE = style.OI, style.ROLE
GREY = OI["grey"]
DARK, MID = "#000000", "#4D4D4D"          # model identity in panels c and d
LS_SRC = (0, (3.2, 1.3))                  # row-calibrated / i.i.d. line style
REPORT = {}                               # every number drawn, for the caption


def agg(vals):
    """mean and 2.5-97.5 percentiles, exactly as the R4 summaries are built."""
    v = np.asarray(vals, float)
    return float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


W, H = 17.5, 12.0          # final print size in cm; nothing is ever downscaled
fig = plt.figure(figsize=(W * style.CM, H * style.CM))


def box(x0, y0, x1, y1):
    """axes rectangle given in centimetres of the printed figure."""
    return fig.add_axes([x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H])


# ===================================================================== a
PROPS = [("Tg", "Tg"), ("YoungModulus", "Young's mod."),
         ("Microhardness", "microhard."), ("Tliquidus", "liquidus")]
FAM = [("A_labout", "Held-out laboratories"), ("B_temporal", "Later publications"),
       ("C_analysed", "Analysed compositions")]
# key, label, colour, marker, marker face  (colour AND marker AND row are redundant)
METH = [("id.row.coverage", "in-distribution", GREY, "v", "none"),
        ("tgt_test_a.row.coverage", "source-calibrated (i.i.d.)", ROLE["source"], "o", None),
        ("tgt_test_a.lab.coverage", "laboratory-calibrated", OI["skyblue"], "^", None),
        ("weighted.test_a.coverage", "weighted conformal", OI["purple"], "P", None),
        ("recal_a.m.coverage", "target-recalibrated, m = 30", ROLE["recal"], "s", None)]

axA = [box(2.62 + i * 3.00, 7.30, 2.62 + i * 3.00 + 2.74, 10.80) for i in range(3)]
step = 0.185
REPORT["a"] = {}
for ia, (fkey, ftitle) in enumerate(FAM):
    ax = axA[ia]
    REPORT["a"][fkey] = {}
    for ip, (pkey, plab) in enumerate(PROPS):
        pe = R4[fkey][pkey]["per_eval"]
        y0 = len(PROPS) - 1 - ip
        if ip:
            ax.axhline(y0 + 0.5, color=OI["lightgrey"], lw=0.5, zorder=0)
        for im, (mk, mlab, col, mrk, mfc) in enumerate(METH):
            m, lo, hi = agg([e[mk] for e in pe])
            REPORT["a"][fkey].setdefault(plab, {})[mlab] = (m, lo, hi)
            y = y0 + (2 - im) * step
            ax.plot([lo, hi], [y, y], color=col, lw=0.6, alpha=0.5,
                    solid_capstyle="butt", zorder=2)
            ax.plot([m], [y], marker=mrk, ms=3.3, mfc=mfc or col, mec=col, mew=0.7,
                    color=col, ls="none", zorder=4)
    ax.axvline(0.90, color=ROLE["ideal"], lw=0.7, ls=(0, (4, 3)), zorder=1)
    ax.set_xlim(0.44, 1.02)
    ax.set_ylim(-0.55, len(PROPS) - 0.45)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xticklabels(["0.5", "0.6", "0.7", "0.8", "0.9", "1.0"])
    ax.set_title(ftitle, fontsize=8)
    ax.set_yticks(np.arange(len(PROPS))[::-1])
    ax.tick_params(axis="y", length=0)
    if ia == 0:
        ax.set_yticklabels([p for _, p in PROPS])
    else:
        ax.set_yticklabels([])
        ax.spines["left"].set_visible(False)
xmidA = (2.62 + 2.62 + 2 * 3.00 + 2.74) / 2 / W
fig.text(xmidA, 6.80 / H, "coverage on the target-test half", fontsize=8,
         ha="center", va="top")

hA = [Line2D([], [], color=c, marker=mk, ls="none", ms=3.5, mfc=mf or c, mec=c,
             mew=0.7, label=lab) for _, lab, c, mk, mf in METH]
hA.append(Line2D([], [], color=ROLE["ideal"], lw=0.7, ls=(0, (4, 3)),
                 label="nominal 0.90"))
fig.legend(handles=[hA[i] for i in (0, 1, 2, 5, 3, 4)], loc="upper center",
           bbox_to_anchor=(xmidA, 6.50 / H), ncol=3, handlelength=1.4,
           handletextpad=0.4, columnspacing=1.1, labelspacing=0.32, borderpad=0.0)

# ===================================================================== b
axB = box(13.05, 8.90, 17.05, 10.80)      # per-laboratory coverage histogram
axBi = box(12.95, 5.65, 17.10, 7.45)      # 45S5 Bioglass example

row, lab10 = [], []
for pkey, _ in PROPS:
    for pl in R4["A_labout"][pkey]["per_lab_lists"]:
        for r in pl["lab_specific"]:
            row.append(r["row"])
            lab10.append(r["lab10"])
row, lab10 = np.array(row), np.array(lab10)
edges = np.arange(0.0, 1.0001, 0.05)
axB.axvspan(0.0, 0.80, color=OI["lightgrey"], alpha=0.45, lw=0, zorder=0)
for v, col, ls in [(row, ROLE["source"], LS_SRC), (lab10, ROLE["recal"], "-")]:
    h = np.histogram(v, bins=edges)[0] / len(v) * 100
    axB.stairs(h, edges, color=col, lw=1.1, ls=ls, zorder=3)
axB.plot([0.90, 0.90], [0, 58], color=ROLE["ideal"], lw=0.7, ls=(0, (4, 3)), zorder=2)
axB.set_xlim(0, 1.0)
axB.set_ylim(0, 66)
axB.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
axB.set_yticks([0, 20, 40, 60])
sh_row, sh_l10 = float((row < 0.8).mean()), float((lab10 < 0.8).mean())
med_row, med_l10 = float(np.median(row)), float(np.median(lab10))
axB.set_xlabel("per-laboratory coverage")
axB.set_ylabel("laboratory\nevaluations (%)", linespacing=1.2)
axB.set_title("Coverage inside one laboratory", fontsize=8)

# key: line style carries the series identity, so it is legible without colour,
# and it sits in the empty upper-left corner, outside the two histograms.
hB = [Line2D([], [], color=ROLE["source"], lw=1.1, ls=LS_SRC,
             label="source-calibrated (i.i.d.)"),
      Line2D([], [], color=ROLE["recal"], lw=1.1, ls="-",
             label="laboratory-recalibrated,\n10 of its own glasses")]
axB.legend(handles=hB, loc="upper left", bbox_to_anchor=(0.055, 1.03),   # clear of sub-panel label i
           handlelength=1.5, handletextpad=0.4, labelspacing=0.22, borderpad=0.0)
# the two shares, named in words so that nothing here depends on colour
axB.text(0.03, 30.0, f"share below 0.80 (n = {len(row):,})", fontsize=7, ha="left", va="top")
axB.text(0.03, 19.5, f"{sh_row * 100:.0f}% i.i.d.,  {sh_l10 * 100:.1f}% recalibrated",
         fontsize=7, ha="left", va="top")

E = R4["E_45S5"]
rec = np.array([r["Tg_K"] for r in E["description"]["records"]], float)
cen = E["model"]["row.pred_records_mean"]["mean"]
q_row, q_lab = E["model"]["row.q"]["mean"], E["model"]["lab.q"]["mean"]
c_row = E["model"]["row.coverage_records"]["mean"]
c_lab = E["model"]["lab.coverage_records"]["mean"]
rng = np.random.default_rng(3)
axBi.scatter(rec, 2.55 + rng.uniform(-0.15, 0.15, len(rec)), s=2.4, marker="o",
             c=ROLE["ideal"], alpha=0.6, linewidths=0, zorder=3)
# each interval is named in place, so the inset does not borrow panel a's key
for y, q, col, mrk, cv, nm in [(1.35, q_lab, OI["skyblue"], "^", c_lab,
                                "laboratory-calibrated"),
                               (0.15, q_row, ROLE["source"], "o", c_row,
                                "source-calibrated")]:
    axBi.plot([cen - q, cen + q], [y, y], color=col, lw=1.6, solid_capstyle="butt",
              zorder=3)
    axBi.plot([cen], [y], marker=mrk, ms=3.2, color=col, zorder=4)
    axBi.text(cen - q, y + 0.30, f"{nm}, ±{q:.0f} K", fontsize=7, color=col,
              ha="left", va="bottom")
    axBi.text(cen + q + 7, y, f"covers {cv:.2f}", fontsize=7, color=col,
              ha="left", va="center")
axBi.set_xlim(698, 966)
axBi.set_ylim(-0.55, 3.55)
axBi.set_yticks([])
axBi.set_xticks([725, 775, 825, 875])
axBi.spines["left"].set_visible(False)
axBi.spines["bottom"].set_bounds(710, 895)
axBi.set_title("45S5 Bioglass Tg (K)", fontsize=7.5)
axBi.text(963, 3.50, f"{E['description']['n_records']} records, "
                     f"{E['description']['n_labs']} laboratories", fontsize=7,
          ha="right", va="top")
REPORT["b"] = {"n_lab_evals": len(row), "median_row": med_row, "median_lab10": med_l10,
               "share_below_0.8_row": sh_row, "share_below_0.8_lab10": sh_l10,
               "45S5": {"n": int(E["description"]["n_records"]),
                        "labs": int(E["description"]["n_labs"]),
                        "sd_K": E["description"]["Tg_sd_K"], "centre_K": cen,
                        "q_row_K": q_row, "cov_row": c_row,
                        "q_lab_K": q_lab, "cov_lab": c_lab}}

# ===================================================================== c
axC1 = box(2.25, 1.80, 6.35, 4.72)
axC2 = box(8.35, 1.80, 11.05, 4.72)
MODc = [("RF", "random forest", DARK, "o", "-"),
        ("HistGB", "gradient boosting", MID, "D", LS_SRC)]
KS = [0, 9, 10, 20, 30, 50]
# The recalibration budgets get one equal slot each, so that the k = 9 and k = 10
# points can no longer overlap, and the k = 0 coverage of the COMPUTED labels -
# a reference, not a budget - sits in its own shaded block on the left instead of
# on the k axis, where it used to read as part of a downward trend.
XREF, DXM = -1.75, 0.18                   # computed-label block, model offset
REPORT["c"] = {"ksweep_measured_labels": {}, "by_class": {}, "dft_label_cov_k0": {}}
axC1.axvspan(-2.75, -0.65, color=OI["lightgrey"], alpha=0.40, lw=0, zorder=0)
axC1.axvline(-0.65, color=GREY, lw=0.6, ls=(0, (1.5, 1.5)), zorder=1)
for im, (mkey, mlab, col, mrk, ls) in enumerate(MODc):
    S = R4B["MP"][mkey]
    dx = -DXM if im == 0 else DXM
    cov = [S["insilico_vs_expt"]["coverage"]["mean"] if k == 0 else
           S["ksweep"][str(k)]["plain"]["coverage"]["mean"] for k in KS]
    axC1.plot([i + dx for i in range(len(KS))], cov, color=col, marker=mrk, ls=ls,
              ms=3.2, lw=0.9, zorder=3)
    d0 = S["insilico_vs_dft"]["coverage"]["mean"]
    axC1.plot([XREF + dx], [d0], color=col, marker=mrk, ms=3.2, ls="none",
              mfc="white", mew=0.8, zorder=3)
    REPORT["c"]["ksweep_measured_labels"][mlab] = {str(k): c for k, c in zip(KS, cov)}
    REPORT["c"]["dft_label_cov_k0"][mlab] = d0
axC1.axhline(0.90, color=ROLE["ideal"], lw=0.7, ls=(0, (4, 3)), zorder=1)
axC1.set_xlim(-2.75, 5.50)
axC1.set_xticks([XREF] + list(range(len(KS))))
axC1.set_xticklabels(["computed\n(DFT) labels"] + [str(k) for k in KS])
axC1.set_xlabel("measurements k")
axC1.set_ylabel("coverage")
axC1.set_title("Computed vs measured labels", fontsize=8)
axC1.set_ylim(0.58, 1.02)
axC1.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
# inline key for the nominal line, in the empty lower half of the axes
axC1.plot([0.05, 0.85], [0.655, 0.655], color=ROLE["ideal"], lw=0.7, ls=(0, (4, 3)),
          zorder=3)
axC1.text(1.00, 0.655, "nominal 0.90", fontsize=7, ha="left", va="center")

CLS = [("f-block (Ln/An) compounds", "f-block"), ("B/C/Si/Ge compounds", "B/C/Si/Ge"),
       ("sp-metal intermetallics", "sp-metal"),
       ("transition-metal-only intermetallics", "TM-only")]
for mkey, mlab, col, mrk, _ in MODc:
    S = R4B["MP"][mkey]["insilico_vs_expt_by_class"]
    off = 0.15 if mkey == "RF" else -0.15
    for ic, (ckey, clab) in enumerate(CLS):
        d = S[ckey]["coverage"]
        y = len(CLS) - 1 - ic + off
        axC2.plot([d["lo"], d["hi"]], [y, y], color=col, lw=0.6, alpha=0.5,
                  solid_capstyle="butt", zorder=2)
        axC2.plot([d["mean"]], [y], color=col, marker=mrk, ms=3.2, ls="none", zorder=4)
        REPORT["c"]["by_class"].setdefault(clab, {})[mlab] = (
            d["mean"], d["lo"], d["hi"], S[ckey]["n_mean"])
mar = [R4B["MP"][m]["insilico_vs_expt"]["coverage"]["mean"] for m, _, _, _, _ in MODc]
mlo, mhi = min(mar), max(mar)
axC2.axvspan(mlo, mhi, color=GREY, alpha=0.55, lw=0, zorder=0)
axC2.axvline(0.90, color=ROLE["ideal"], lw=0.7, ls=(0, (4, 3)), zorder=1)
axC2.set_xlim(0.58, 1.02)
axC2.set_ylim(-0.55, 4.35)
axC2.set_xticks([0.6, 0.7, 0.8, 0.9, 1.0])
axC2.set_yticks(np.arange(len(CLS))[::-1])
axC2.set_yticklabels([c for _, c in CLS])
axC2.tick_params(axis="y", length=0)
axC2.set_xlabel("coverage")
axC2.set_title("By chemistry class, k = 0", fontsize=8)
# the marginal value is named with its numbers and its leader ends on the LEFT
# edge of the grey band, so that it cannot be read as pointing at nominal 0.90
axC2.annotate(f"marginal\n{mlo:.2f}-{mhi:.2f}", xy=(mlo, 3.30), xytext=(0.745, 4.00),
              fontsize=7, color="#3A3A3A", ha="center", va="center", linespacing=1.25,
              arrowprops=dict(arrowstyle="-", lw=0.6, color="#3A3A3A",
                              shrinkA=2, shrinkB=1))
REPORT["c"]["marginal_band"] = (mlo, mhi)

# ===================================================================== d
axD1 = box(12.70, 3.55, 17.20, 4.72)
axD2 = box(12.70, 1.80, 17.20, 3.25)
DOM = [("S_source-test", "source\ntest"), ("T1_CRC-VAL-HE-7K", "T1\nPathMNIST"),
       ("T2_Kather2016_5class", "T2\nKather-2016")]
MODd = [("SmallCNN", "small CNN", DARK, "o"), ("ResNet18", "ResNet-18", MID, "D")]
KR = "200"
NCLASS = len(R5["meta"]["pathmnist_label_order"])
DXMOD, DXCAL = 0.21, 0.085                # model slot, calibration-state slot
REPORT["d"] = {"k_recal": int(KR), "n_classes": NCLASS}
for ax, fld in [(axD1, "coverage"), (axD2, "set_size")]:
    for idom, (dkey, dlab) in enumerate(DOM):
        for imo, (mkey, mlab, col, mrk) in enumerate(MODd):
            L = R5["architectures"][mkey]["results"][dkey]["LAC"]
            xm = idom + (DXMOD if imo else -DXMOD)
            has_r = KR in L
            # the two calibration states of one model are separated in x, so the
            # open marker can never hide the filled one when the two values are
            # nearly equal (source-test and T1)
            x0 = xm - DXCAL if has_r else xm
            v0 = L["0"][fld]["mean"]
            if has_r:
                x1 = xm + DXCAL
                v1, lo1, hi1 = (L[KR][fld][q] for q in ("mean", "lo", "hi"))
                ax.plot([x0, x1], [v0, v1], color=GREY, lw=0.7, zorder=1)
                ax.plot([x1, x1], [lo1, hi1], color=col, lw=0.7, alpha=0.55, zorder=2)
                ax.plot([x1], [v1], color=col, marker=mrk, ms=3.6, ls="none", zorder=4)
            ax.plot([x0, x0], [L["0"][fld]["lo"], L["0"][fld]["hi"]], color=col,
                    lw=0.7, alpha=0.55, zorder=2)
            ax.plot([x0], [v0], color=col, marker=mrk, ms=3.6, ls="none", mfc="white",
                    mew=0.8, zorder=4)
            REPORT["d"].setdefault(fld, {}).setdefault(
                dlab.replace("\n", " "), {})[mlab] = (v0, v1 if has_r else None)
    ax.set_xlim(-0.55, 2.55)
    ax.set_xticks(range(len(DOM)))
axD1.axhline(0.90, color=ROLE["ideal"], lw=0.7, ls=(0, (4, 3)), zorder=1)
axD1.set_ylim(-0.10, 1.16)
axD1.set_yticks([0, 0.5, 0.9])
axD1.set_yticklabels(["0", "0.5", "0.9"])
axD1.set_ylabel("coverage")
axD1.set_xticklabels([])
axD1.tick_params(axis="x", length=0)
axD1.set_title("External histology cohort", fontsize=8)
axD1.text(-0.33, 0.955, "nominal 0.90", fontsize=7, ha="left", va="bottom",
          color=ROLE["ideal"])
T2 = {m: R5["architectures"][m]["results"]["T2_Kather2016_5class"]["LAC"]
      for m, _, _, _ in MODd}
# the source-calibrated collapse on T2 is the panel's headline, so both values
# are written next to their own open marker
axD1.text(2 - DXMOD - DXCAL - 0.09, T2["SmallCNN"]["0"]["coverage"]["mean"],
          f"{T2['SmallCNN']['0']['coverage']['mean']:.2f}", fontsize=7, ha="right",
          va="center", color=DARK)
axD1.annotate(f"{T2['ResNet18']['0']['coverage']['mean']:.3f}",
              xy=(2 + DXMOD - DXCAL, T2["ResNet18"]["0"]["coverage"]["mean"]),
              xytext=(2 - DXMOD - DXCAL - 0.09, 0.0), fontsize=7, ha="right",
              va="center", color=MID,
              arrowprops=dict(arrowstyle="-", lw=0.5, color=MID, shrinkA=2, shrinkB=2))
axD2.axhline(NCLASS, color=ROLE["ideal"], lw=0.7, ls=(0, (1.4, 1.4)), zorder=1)
axD2.set_ylim(-0.55, 10.6)
axD2.set_yticks([0, 3, 6, 9])
axD2.set_ylabel("prediction-set\nsize", linespacing=1.2)
axD2.set_xticklabels([d for _, d in DOM])
axD2.text(-0.30, 9.25, f"all {NCLASS} classes", fontsize=7, ha="left", va="bottom",
          color=ROLE["ideal"])
axD2.text(2 - DXMOD + DXCAL - 0.09, T2["SmallCNN"][KR]["set_size"]["mean"],
          f"{T2['SmallCNN'][KR]['set_size']['mean']:.1f}", fontsize=7, ha="right",
          va="center", color=DARK)
axD2.text(2 + DXMOD + DXCAL - 0.09, T2["ResNet18"][KR]["set_size"]["mean"],
          f"{T2['ResNet18'][KR]['set_size']['mean']:.1f}", fontsize=7, ha="right",
          va="top", color=MID)

# ------------------------------------------------- keys, outside every axes
hC = [Line2D([], [], color=c, marker=m, ls=l, ms=3.2, lw=0.9, label=lab)
      for _, lab, c, m, l in MODc]
# as in panel d, the label-type entries carry BOTH model glyphs, so that neither
# of them repeats the "random forest" / "gradient boosting" handle
hC += [tuple(Line2D([], [], color=c, marker=m, ls="none", ms=3.2, mfc="white",
                    mew=0.8) for _, _, c, m, _ in MODc),
       tuple(Line2D([], [], color=c, marker=m, ls="none", ms=3.2)
             for _, _, c, m, _ in MODc)]
fig.legend(handles=hC, labels=[MODc[0][1], MODc[1][1], "computed (DFT) labels",
                               "measured labels"], loc="upper center",
           bbox_to_anchor=((2.25 + 11.05) / 2 / W, 0.78 / H), ncol=2,
           handlelength=1.6, handletextpad=0.4, labelspacing=0.3,
           columnspacing=1.0, borderpad=0.0,
           handler_map={tuple: HandlerTuple(ndivide=None, pad=0.35)})

hD = [Line2D([], [], color=c, marker=m, ls="none", ms=3.6, label=lab)
      for _, lab, c, m in MODd]
# the calibration-state entries show BOTH model glyphs, open then filled, so that
# neither of them can be confused with the "small CNN" / "ResNet-18" entries
hD += [tuple(Line2D([], [], color=c, marker=m, ls="none", ms=3.6, mfc="white",
                    mew=0.8) for _, _, c, m in MODd),
       tuple(Line2D([], [], color=c, marker=m, ls="none", ms=3.6)
             for _, _, c, m in MODd)]
fig.legend(handles=hD, labels=[MODd[0][1], MODd[1][1], "source-calibrated",
                               f"target-recalibrated, k = {KR}"],
           loc="upper center", bbox_to_anchor=(14.52 / W, 0.78 / H),
           ncol=2, handlelength=1.5, handletextpad=0.4, labelspacing=0.3,
           columnspacing=1.0, borderpad=0.0,
           handler_map={tuple: HandlerTuple(ndivide=None, pad=0.35)})

# ============================================== sub-panel labels (editor: i ii iii, bold, top left)
def sub(ax, roman, x=0.025, y=0.975):
    ax.text(x, y, roman, transform=ax.transAxes, fontsize=8, fontweight="bold",
            ha="left", va="top", zorder=9,
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.8))


for ax, r in [(axA[0], "i"), (axA[1], "ii"), (axA[2], "iii"), (axB, "i"), (axBi, "ii"),
              (axC1, "i"), (axC2, "ii"), (axD1, "i"), (axD2, "ii")]:
    sub(ax, r)

# ===================================================================== save
fig.canvas.draw()


def letter(ax, L, x_cm):
    """style.panel with the letter pinned to an absolute x (cm) of the page."""
    bb = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    style.panel(ax, L, x=x_cm / W - bb.x0)


for ax, L, xc in [(axA[0], "a", 0.26), (axB, "b", 11.95),
                  (axC1, "c", 0.26), (axD1, "d", 11.95)]:
    letter(ax, L, xc)

bb = fig.get_tightbbox(fig.canvas.get_renderer())
print("content bbox (cm): x %.2f-%.2f  y %.2f-%.2f" %
      (bb.x0 * 2.54, bb.x1 * 2.54, bb.y0 * 2.54, bb.y1 * 2.54))
png = style.save(fig, "fig5_genuine_shifts", 5)
print(png)
print(json.dumps(REPORT, indent=1, default=float))
