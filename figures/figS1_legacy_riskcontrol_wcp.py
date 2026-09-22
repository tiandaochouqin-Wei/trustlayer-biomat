"""Supporting Information Figure S1 - two re-run legacy analyses.

a  conformal risk control (CRC) of the false-negative rate on glass-forming-ability
   classification, candidates exchangeable with the calibration set (30 splits);
b  the same procedure when whole chemical systems are held out;
c  weighted (likelihood-ratio) conformal prediction on the design region (20 splits).

Every number comes from revision/results/R9_legacy.json
(keys 'classif' -> grouped_composition / grouped_chemical_system, and 'wcp'),
produced by revision/code/exp_R9_classif.py and exp_R9_wcp.py; the report is
revision/results/R9_legacy.md.

Design notes (QA round 2)
-------------------------
* Panel c plots six DESIGN REGIONS, not calibration states, so the task identities
  come from style.TASKSTYLE (the same colour / marker / line style as Fig. 4d and
  Fig. 9a,c).  The four reserved ROLE hues stay free there, and every task is
  labelled once at its own cluster, so the per-task numbers quoted in the caption
  can be located on the figure.  Markers carry a white halo so that overlapping
  points separate at print size.
* The open/filled "> 50% infinite" cue is gone.  Certification is carried by
  panel c ii alone - the two sub-panels share the x axis, so each point sits
  directly above its own share of infinite intervals - with the not-certified
  zone shaded.  Nothing in c i can therefore contradict it.
* Every legend / annotation sits in a region that holds no data: the key block of
  c i occupies the empty lower-right corner, the band and nominal-line labels sit
  clear of both the band and the markers, and the split-share note for a and b is
  set below the axes.
* Labels in a and b carry a drawn line sample in their own line style, so the two
  series are never separated by colour alone (vermillion / orange in b).
* No mathtext: the Greek alpha is the Arial glyph, so only ArialMT and
  Arial-BoldMT are embedded in the PDF.
"""
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import style  # noqa: E402

style.apply()
R = json.load(open(os.path.join(REV, "results", "R9_legacy.json")))
C, W = R["classif"], R["wcp"]

GREY = style.OI["grey"]
SRC, RECAL, FAIL, HEUR = (style.ROLE["source"], style.ROLE["recal"],
                          style.ROLE["fail"], style.ROLE["heuristic"])
EN = chr(0x2013)          # en dash
ALPHA = chr(0x3b1)       # Arial glyph, not mathtext

ALPHAS = [float(a) for a in C["alphas"]]
DS = ["glass_Tg", "glass_E", "glass_HV", "glass_Tliq", "steel_yield", "polymer_Tg"]
DGRID = W["protocol"]["D_grid"]                      # [5, 20, 50, 100, 250, 500]


def crc(fam, field):
    """mean / 2.5 / 97.5 percentile of `field` over the alpha grid for a split family."""
    b = C[fam]["crc"]
    g = lambda k: np.array([b[str(a)][field][k] for a in ALPHAS])  # noqa: E731
    return g("mean"), g("lo"), g("hi")


def msize(D):
    """marker radius encodes the random-feature dimension D (floor 3.0 pt)."""
    return 3.0 + 1.35 * np.log10(D / DGRID[0])


# ----------------------------------------------------------------- canvas
FW, FH = 17.5, 7.0          # final print size in cm; nothing is ever downscaled
fig = plt.figure(figsize=(FW * style.CM, FH * style.CM))


def box(x0, y0, x1, y1):
    return fig.add_axes([x0 / FW, y0 / FH, (x1 - x0) / FW, (y1 - y0) / FH])


COLS = [(1.00, 4.62), (6.05, 9.67), (11.14, 16.92)]
YM, YS = (3.00, 5.72), (1.45, 2.72)      # main row / strip row, in cm
axAm, axBm, axCm = [box(x0, YM[0], x1, YM[1]) for x0, x1 in COLS]
axAs, axBs, axCs = [box(x0, YS[0], x1, YS[1]) for x0, x1 in COLS]


def sub(ax, roman, x=0.025, y=0.975):
    ax.text(x, y, roman, transform=ax.transAxes, fontsize=8, fontweight="bold",
            ha="left", va="top", zorder=9,
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.8))


# ============================================================ a, b  (CRC)
FNR_LIM, FLAG_LIM = (0.0, 0.58), (0.45, 1.02)
GX0, GX1, TX = 0.016, 0.042, 0.049       # inline line sample, then the text
KY = [0.548, 0.481, 0.414]               # the three label rows of panel i

for ax, axs, fam, col, mk, ls, title, qual in [
        (axAm, axAs, "grouped_composition", SRC, style.MARK["source"], "-",
         "Exchangeable candidates", "(exchangeable)"),
        (axBm, axBs, "grouped_chemical_system", FAIL, style.MARK["fail"], (0, (4, 1.6)),
         "Held-out chemical systems", "(systems held out)")]:
    NAIVE_LS = (0, (5, 1.4, 1, 1.4))
    m, lo, hi = crc(fam, "fnr")
    nf = C[fam]["naive_0.5"]["fnr"]["mean"]
    ax.plot([0, 0.215], [0, 0.215], color="black", lw=0.7, ls=(0, (1, 1.6)), zorder=1)
    ax.fill_between(ALPHAS, lo, hi, color=col, alpha=0.16, lw=0, zorder=2)
    ax.plot(ALPHAS, m, color=col, marker=mk, ls=ls, ms=3.2, zorder=4)
    ax.axhline(nf, color=HEUR, lw=0.9, ls=NAIVE_LS, zorder=3)
    ax.text(0.211, 0.168, "y = x", fontsize=7, ha="right", va="top", zorder=6)
    ax.set_xlim(0, 0.215)
    ax.set_ylim(*FNR_LIM)
    ax.set_xticks([0, 0.05, 0.10, 0.15, 0.20])
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    ax.set_xticklabels([])
    ax.spines["left"].set_bounds(0, 0.4)
    ax.spines["bottom"].set_bounds(0, 0.20)
    ax.set_ylabel("false-negative rate")
    ax.set_title(title, fontsize=8)

    # in-panel key: a drawn line sample in the series' own style precedes each
    # label, so the two series are never told apart by colour alone.
    ax.plot([GX0, GX1], [KY[0]] * 2, color=col, ls=ls, lw=1.1, zorder=6)
    ax.plot([(GX0 + GX1) / 2], [KY[0]], color=col, marker=mk, ms=3.2, ls="none",
            zorder=7)
    ax.text(TX, KY[0], "conformal risk control", color=col, fontsize=7,
            ha="left", va="center")
    ax.text(TX, KY[1], qual, color=col, fontsize=7, ha="left", va="center")
    ax.plot([GX0, GX1], [KY[2]] * 2, color=HEUR, ls=NAIVE_LS, lw=1.1, zorder=6)
    ax.text(TX, KY[2], f"naive 0.5 threshold {nf:.3f}", color=HEUR, fontsize=7,
            ha="left", va="center")

    fm, flo, fhi = crc(fam, "flagged")
    nflag = C[fam]["naive_0.5"]["flagged"]["mean"]
    axs.fill_between(ALPHAS, flo, fhi, color=col, alpha=0.16, lw=0, zorder=2)
    axs.plot(ALPHAS, fm, color=col, marker=mk, ls=ls, ms=3.2, zorder=4)
    axs.axhline(nflag, color=HEUR, lw=0.9, ls=NAIVE_LS, zorder=3)
    axs.plot([0.100, 0.126], [0.962] * 2, color=HEUR, ls=NAIVE_LS, lw=1.1, zorder=6)
    axs.text(0.133, 0.962, f"naive {nflag:.2f}", color=HEUR, fontsize=7,
             ha="left", va="center", zorder=6)
    axs.text(0.004, 0.468, "30 splits", fontsize=7, ha="left", va="bottom",
             color=GREY)
    axs.set_xlim(0, 0.215)
    axs.set_ylim(*FLAG_LIM)
    axs.set_xticks([0, 0.05, 0.10, 0.15, 0.20])
    axs.set_yticks([0.5, 0.7, 0.9])
    axs.spines["left"].set_bounds(0.5, 0.9)
    axs.spines["bottom"].set_bounds(0, 0.20)
    axs.set_xlabel("target false-negative rate")
    axs.set_ylabel("share flagged")

# the split-share note is set below the axes so that neither panel i carries it
shares = {}
for fam in ("grouped_composition", "grouped_chemical_system"):
    sh = [C[fam]["crc"][str(a)]["share_splits_fnr_above_alpha"] for a in ALPHAS]
    shares[fam] = (min(sh), max(sh))
sa, sb = shares["grouped_composition"], shares["grouped_chemical_system"]
xmid = (COLS[0][0] + COLS[1][1]) / 2 / FW
fig.text(xmid, 0.40 / FH, "the guarantee is in expectation only:", fontsize=7,
         ha="center", va="baseline")
fig.text(xmid, 0.11 / FH,
         "the target is exceeded in %d%s%d%% of splits in a, %d%s%d%% in b"
         % (round(sa[0] * 100), EN, round(sa[1] * 100),
            round(sb[0] * 100), EN, round(sb[1] * 100)),
         fontsize=7, ha="center", va="baseline")

# ============================================================ c  (WCP)
XLIM = (-0.02, 0.97)
for ax in (axCm, axCs):
    ax.set_xlim(*XLIM)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.spines["bottom"].set_bounds(0, 0.88)

rec = [W[d]["recal_m"]["coverage"]["mean"] for d in DS]
axCm.axhspan(min(rec), max(rec), color=RECAL, alpha=0.18, lw=0, zorder=0)
axCm.axhline(0.90, color="black", lw=0.7, ls=(0, (4, 3)), zorder=1)
axCs.axhspan(0.5, 1.20, color=GREY, alpha=0.13, lw=0, zorder=0)
axCs.axhline(0.5, color=GREY, lw=0.6, ls=(0, (2, 2)), zorder=1)


def pt(ax, x, y, D, col, mk, mfc):
    """one (task, D) marker with a white halo so overlapping points separate."""
    ax.plot([x], [y], marker=mk, ms=msize(D) + 1.6, mfc="white", mec="white",
            mew=0.0, ls="none", zorder=3)
    ax.plot([x], [y], marker=mk, ms=msize(D), mfc=mfc, mec=col, mew=0.7,
            ls="none", zorder=4)


def cross(ax, x, y, z=5):
    """raw-feature weight model: one glyph for all six tasks (see the key)."""
    ax.plot([x], [y], marker="x", ms=4.6, mec="white", mew=2.0, ls="none",
            zorder=z)
    ax.plot([x], [y], marker="x", ms=3.8, mec="black", mew=1.0, ls="none",
            zorder=z + 1)


for d in DS:                                    # random-feature weight models
    col, mk, lsty, mfc = style.TASKSTYLE[d]
    wd = W[d]["weighted"]
    xs = [wd[f"rff_{D}"]["ess_frac"]["mean"] for D in DGRID]
    ys = [wd[f"rff_{D}"]["coverage"]["mean"] for D in DGRID]
    fs = [wd[f"rff_{D}"]["frac_infinite"]["mean"] for D in DGRID]
    axCm.plot(xs, ys, color=col, lw=0.7, ls=lsty, zorder=2, alpha=0.75)
    axCs.plot(xs, fs, color=col, lw=0.7, ls=lsty, zorder=2, alpha=0.75)
    for x, y, f, D in zip(xs, ys, fs, DGRID):
        pt(axCm, x, y, D, col, mk, mfc)
        pt(axCs, x, f, D, col, mk, mfc)
    r = wd["raw_LR"]                            # standard low-dimensional weights
    cross(axCm, r["ess_frac"]["mean"], r["coverage"]["mean"])
    cross(axCs, r["ess_frac"]["mean"], r["frac_infinite"]["mean"])

axCm.set_ylim(0.10, 1.10)
axCm.set_yticks([0.4, 0.6, 0.8, 1.0])
axCm.set_xticklabels([])
axCm.spines["left"].set_bounds(0.4, 1.0)
axCm.set_ylabel("coverage")
axCm.set_title("Weighted conformal, design region", fontsize=8)

# band label: above the band, on the only stretch that holds no marker
axCm.text(0.965, 1.040, f"target-recalibrated {min(rec):.3f}{EN}{max(rec):.3f}",
          color=RECAL, fontsize=7, ha="right", va="center", zorder=6)

# one task label per cluster, each on an empty patch next to its own connector
TASKLAB = [("glass_Tg", "glass Tg", 0.140, 0.645, "left"),
           ("glass_E", "glass E", 0.375, 0.380, "left"),
           ("glass_Tliq", "glass Tliq", 0.470, 0.500, "left"),
           ("glass_HV", "glass HV", 0.440, 0.735, "left"),
           ("steel_yield", "steel yield", 0.817, 0.690, "center"),
           ("polymer_Tg", "polymer Tg", 0.168, 0.775, "center")]
for key, lab, x, y, ha in TASKLAB:
    axCm.text(x, y, lab, color=style.TASKSTYLE[key][0], fontsize=7, ha=ha,
              va="center", zorder=6)

# key: two rows along the empty floor of c i, clear of every marker
KR1, KR2 = 0.255, 0.160
axCm.text(0.335, KR1, "D =", fontsize=7, ha="left", va="center")
for x, D in [(0.420, 5), (0.492, 50), (0.590, 500)]:
    axCm.plot([x], [KR1], marker="o", ms=msize(D), mfc=GREY, mec=GREY, mew=0.7,
              ls="none", zorder=6)
    axCm.text(x + 0.020, KR1, str(D), fontsize=7, ha="left", va="center")
cross(axCm, 0.345, KR2, z=6)
axCm.text(0.372, KR2, "raw features", fontsize=7, ha="left", va="center")
axCm.plot([0.630, 0.685], [KR2] * 2, color="black", lw=0.7, ls=(0, (4, 3)),
          zorder=6)
axCm.text(0.705, KR2, "nominal 0.90", fontsize=7, ha="left", va="center")

axCs.set_ylim(-0.06, 1.18)
axCs.set_yticks([0, 0.5, 1.0])
axCs.spines["left"].set_bounds(0, 1.0)
axCs.set_xlabel("normalised ESS of weights")
axCs.set_ylabel("share infinite")
axCs.text(0.96, 1.06, "shaded: > 50% infinite (not certified)", fontsize=7,
          ha="right", va="center", zorder=6)
axCs.text(0.96, 0.02, f"20 splits, {ALPHA} = 0.10", fontsize=7, ha="right",
          va="bottom", color=GREY, zorder=6)

# ----------------------------------------------------------------- labels
for ax, r in [(axAm, "i"), (axBm, "i"), (axCm, "i"),
              (axAs, "ii"), (axBs, "ii"), (axCs, "ii")]:
    sub(ax, r)
fig.canvas.draw()


def letter(ax, L, x_cm):
    bb = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    style.panel(ax, L, x=x_cm / FW - bb.x0)


for ax, L, xc in [(axAm, "a", 0.02), (axBm, "b", 4.88), (axCm, "c", 9.93)]:
    letter(ax, L, xc)

png = style.save(fig, "figS1_legacy_riskcontrol_wcp", "S1")
print(png)

# ------------------------------------------------- numbers printed on the figure
print("naive FNR  comp %.4f  chem %.4f" % (C["grouped_composition"]["naive_0.5"]["fnr"]["mean"],
                                           C["grouped_chemical_system"]["naive_0.5"]["fnr"]["mean"]))
print("naive flagged comp %.4f chem %.4f" % (C["grouped_composition"]["naive_0.5"]["flagged"]["mean"],
                                             C["grouped_chemical_system"]["naive_0.5"]["flagged"]["mean"]))
print("share splits above alpha: comp %.2f-%.2f  chem %.2f-%.2f" % (sa + sb))
print("recal coverage %.4f-%.4f" % (min(rec), max(rec)))
raw_inf = [W[d]["weighted"]["raw_LR"]["frac_infinite"]["mean"] for d in DS]
raw_cov = [W[d]["weighted"]["raw_LR"]["coverage"]["mean"] for d in DS]
print("raw_LR frac_infinite %.3f-%.3f  coverage %.3f-%.3f"
      % (min(raw_inf), max(raw_inf), min(raw_cov), max(raw_cov)))
lowinf = [(d, D, W[d]["weighted"][f"rff_{D}"]["coverage"]["mean"])
          for d in DS for D in DGRID
          if W[d]["weighted"][f"rff_{D}"]["frac_infinite"]["mean"] < 0.10]
print("cov where <10%% infinite: %.3f-%.3f over %d settings"
      % (min(c for _, _, c in lowinf), max(c for _, _, c in lowinf), len(lowinf)))
