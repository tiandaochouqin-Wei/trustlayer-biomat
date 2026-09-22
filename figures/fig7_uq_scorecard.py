"""Figure 7 - uncertainty-quantification scorecard (Reviewer 2, comment 12).

a  coverage in distribution, ten uncertainty methods, one marker per dataset + median;
b  efficiency in distribution: interval score and normalised width, both in IQR units;
c  coverage on the design region, source-calibrated -> target-recalibrated;
d  the failure is invisible to a width-based abstention rule.

All numbers come from revision/results/R3_uq.json (script revision/code/exp_R3_uq.py,
report revision/results/R3_uq.md; 20 resampling splits per dataset, alpha = 0.10).
Nothing is recomputed here except medians / min-max over datasets of the per-dataset
split means, exactly as in the R3_uq.md tables.

Field semantics checked against the producing script and the report:
  * ``coverage``  = fraction of the EVALUATED set inside the 90 % interval
    (exp_R3_uq.evaluate: ``cov = (y >= l9) & (y <= h9)``).  On SHIFT_SRC that is
    the whole design region, computed independently of any abstention rule - so
    panel d's vermillion series is labelled "coverage on the design region", not
    "coverage of the intervals that were kept".
  * ``abstain_1iqr`` = common.abstention_rate(w, 1 x iqr_abs) = mean(w > iqr_abs),
    so the KEPT set is ``width <= 1 x iqr_abs`` (hence "<=", not "<"), and for the
    SHIFT settings of this run ``iqr_abs`` is the IQR of the target recalibration
    pool (JSON ``iqr_seed0.abst_shift``), which is NOT the reporting IQR used by
    ``norm_width`` (``iqr_seed0.report_tgt``); the two differ by up to 35 %.
  * ``norm_width`` = mean 90 % width / IQR of the evaluated domain = target IQR
    under shift, so only that quantity is quoted as "x target IQR".

Layout notes (Editor / Reviewer 1):
  * panel letters are pinned to an ABSOLUTE x of the page, never to
    ``ax.get_tightbbox()``, whose x0 is pushed into the gutter by the long method
    / task tick labels;
  * every reference line is bounded to the data rows so that it can never be drawn
    through a legend or a note, and no legend or note overlaps data or another
    text block (the QA block at the end of this file measures every clearance);
  * panel b draws the median-width tick and the interval-score range in separate
    vertical lanes, so the two black marks can never fuse on the rows where the
    width and the score minimum nearly coincide.
"""
import json
import os
import statistics
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import style  # noqa: E402

style.apply()
R = json.load(open(os.path.join(REV, "results", "R3_uq.json")))
D = R["datasets"]
GREY = style.OI["grey"]
LGREY = style.OI["lightgrey"]
INK = "black"                     # load-bearing notes are black, never 7 pt grey

DS7 = ["glass_Tg", "glass_E", "glass_HV", "glass_Tliq", "steel_yield", "polymer_Tg", "esol_logS"]
DS6 = DS7[:6]
DS_SHORT = {"glass_Tg": "glass Tg", "glass_E": "glass E", "glass_HV": "glass HV",
            "glass_Tliq": "glass Tliq", "steel_yield": "steel yield", "polymer_Tg": "polymer Tg",
            "esol_logS": "ESOL logS"}

# method rows (top -> bottom) and their class: conformal on source / raw model-native /
# conformalised wrapper.  colour + marker + fill are redundant encodings of the class.
CONF, RAW, WRAP = "conf", "raw", "wrap"
METHODS = [("SC", CONF), ("NSC", CONF), ("CQR", CONF),
           ("GP", RAW), ("BOOT", RAW), ("QR", RAW), ("RFH", RAW),
           ("GP+conf", WRAP), ("BOOT+conf", WRAP), ("RFH+conf", WRAP)]
CLS = {CONF: dict(color=style.ROLE["source"], marker=style.MARK["source"], mfc=style.ROLE["source"]),
       RAW: dict(color=style.ROLE["heuristic"], marker=style.MARK["heuristic"], mfc=style.ROLE["heuristic"]),
       WRAP: dict(color=style.ROLE["source"], marker=style.MARK["source"], mfc="white")}
CLS_NAME = {CONF: "conformal", RAW: "raw model-native", WRAP: "conformalised wrapper"}
RECAL_OK = {"SC", "NSC", "CQR", "GP+conf", "BOOT+conf", "RFH+conf"}
NM = len(METHODS)
YM = np.arange(NM)[::-1]          # row 0 (SC) at the top
TOP = NM - 1                      # y of the first row


def vals(setting, method, field, datasets):
    return [D[d][setting][method][field]["mean"] for d in datasets]


# ------------------------------------------------------------------ canvas
W, H = 17.5, 11.0            # final print size in cm; nothing is ever downscaled
fig = plt.figure(figsize=(W * style.CM, H * style.CM))


def box(x0, y0, x1, y1):
    return fig.add_axes([x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H])


# The left column stops early and the right column starts late so that the bold
# panel letters of b and d have a gutter of their own: measured clearances are
# printed by the QA block below.
axA = box(1.78, 6.48, 8.15, 10.13)
axB = box(10.85, 6.48, 17.15, 10.13)
axC = box(1.78, 1.05, 8.15, 4.70)
axD = box(10.85, 1.05, 17.15, 4.70)
LETTER_X = {"a": 0.32, "b": 8.70, "c": 0.32, "d": 8.70}   # cm from the page's left edge

QA = {}                      # artists whose clearances the QA block re-measures


def method_ticks(ax, lo, hi):
    ax.set_yticks(YM)
    ax.set_yticklabels([m for m, _ in METHODS])
    for lab, (_, c) in zip(ax.get_yticklabels(), METHODS):
        lab.set_color(CLS[c]["color"])
    ax.set_ylim(lo, hi)
    ax.tick_params(axis="y", length=0)
    for b in (TOP - 2.5, TOP - 6.5):   # separators between the three method classes
        ax.axhline(b, color=LGREY, lw=0.5, zorder=0)


def refline(ax, x, y0, y1):
    """Nominal-0.90 reference, bounded to the data rows.

    An ``axvline`` spans the whole y range, including the bands reserved for the
    legend and the notes, and would be drawn straight through their type.
    """
    ax.plot([x, x], [y0, y1], color=style.ROLE["ideal"], lw=0.7, ls=(0, (4, 3)),
            zorder=1, solid_capstyle="butt")


# ------------------------------------------------------------------ a  coverage, ID
A_LO, A_HI = -1.35, TOP + 3.8
for y, (m, c) in zip(YM, METHODS):
    v = vals("ID", m, "coverage", DS7)
    axA.plot(v, np.full(len(v), y), ls="none", marker="o", ms=2.3, mfc="none",
             mec=GREY, mew=0.5, zorder=2)
    axA.plot([statistics.median(v)], [y], ls="none", ms=4.6, mew=0.9,
             marker=CLS[c]["marker"], mfc=CLS[c]["mfc"], mec=CLS[c]["color"], zorder=4,
             path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])
refline(axA, 0.90, -0.55, TOP + 0.55)
method_ticks(axA, A_LO, A_HI)
axA.set_xlim(0.375, 1.005)
axA.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
axA.set_xlabel("coverage")
axA.set_title("Coverage in distribution", fontsize=8)
QA["a.nominal"] = axA.text(0.893, TOP + 1.33, "nominal 0.90", fontsize=7,
                           ha="right", va="center")
QA["a.legend"] = axA.legend(
    handles=[Line2D([], [], ls="none", marker="o", ms=2.3, mfc="none", mec=GREY, mew=0.5,
                    label="one dataset; large symbol = median (n = 7)")],
    loc="upper left", bbox_to_anchor=(-0.008, 1.012), handlelength=0.9,
    borderpad=0.15, handletextpad=0.35)
axA.add_artist(QA["a.legend"])
# Key for the THREE median glyphs.  All three carry meaning (filled blue circle =
# conformal, orange diamond = raw model-native, open blue circle = conformalised
# wrapper) and the tick-label colour alone does not separate conformal from
# wrapper, so a reader could not decode the figure without this.  It goes in the
# empty lower-left corner of panel a: the three bottom rows and RFH all lie above
# coverage 0.78, so the block cannot touch data, and drawing it as a legend rather
# than as loose markers on rows keeps it from being read as data.
QA["a.classkey"] = axA.legend(
    handles=[Line2D([], [], ls="none", marker=CLS[c]["marker"], ms=4.6, mew=0.9,
                    mfc=CLS[c]["mfc"], mec=CLS[c]["color"], label=CLS_NAME[c])
             for c in (CONF, RAW, WRAP)],
    loc="lower left", bbox_to_anchor=(-0.008, -0.0195), handlelength=0.9,
    borderpad=0.15, labelspacing=0.20, handletextpad=0.35)
for txt, c in zip(QA["a.classkey"].get_texts(), (CONF, RAW, WRAP)):
    txt.set_color(CLS[c]["color"])

# ------------------------------------------------------------------ b  efficiency, ID
BAR_H = 0.62
for y, (m, c) in zip(YM, METHODS):
    isn = vals("ID", m, "interval_score_norm", DS7)
    nw = statistics.median(vals("ID", m, "norm_width", DS7))
    axB.barh(y, statistics.median(isn), height=BAR_H, color=CLS[c]["color"],
             alpha=0.35 if c == WRAP else 0.85, edgecolor=CLS[c]["color"], lw=0.6, zorder=2)
    # upper lane: range of the interval score over the seven datasets
    yr = y + 0.17
    axB.plot([min(isn), max(isn)], [yr, yr], color="black", lw=0.7, zorder=4)
    for e in (min(isn), max(isn)):
        axB.plot([e, e], [yr - 0.11, yr + 0.11], color="black", lw=0.7, zorder=4)
    # lower lane + white halo: median normalised width.  Separated from the range
    # caps so that the two black marks cannot fuse on RFH, BOOT+conf and GP+conf,
    # where the median width and the interval-score minimum agree to 0.02 x IQR.
    axB.plot([nw, nw], [y - 0.34, y - 0.02], color="black", lw=1.7,
             solid_capstyle="butt", zorder=5,
             path_effects=[pe.withStroke(linewidth=3.1, foreground="white")])
method_ticks(axB, A_LO, A_HI)
axB.set_xlim(0, 4.05)
axB.set_xticks([0, 1, 2, 3, 4])
axB.set_xlabel("interval score and normalised width (× IQR)")
axB.set_title("Efficiency in distribution (lower is better)", fontsize=8)
QA["b.legend"] = axB.legend(
    handles=[Line2D([], [], color=style.ROLE["source"], lw=4.5, alpha=0.85,
                    label="median interval score"),
             Line2D([], [], ls="none", color="black", marker="|", ms=5.0, mew=1.7,
                    label="median width"),
             Line2D([], [], color="black", lw=0.7, marker="|", ms=3.6, mew=0.7,
                    label="range")],
    loc="upper left", bbox_to_anchor=(-0.008, 1.012), ncol=3, handlelength=1.0,
    borderpad=0.15, columnspacing=1.0, handletextpad=0.4)
# the two claims panel b is drawn to support
QA["b.note"] = axB.text(0.04, TOP + 1.33, "SC: least efficient calibrated method, not the widest",
                        fontsize=7, ha="left", va="center", color=style.ROLE["source"])

# ------------------------------------------------------------------ c  design shift
C_LO, C_HI = -2.4, TOP + 4.0


def crange(ax, v, y, col, cap=0.17):
    """capped line = range over datasets (distinct from the plain median connector)."""
    ax.plot([min(v), max(v)], [y, y], color=col, lw=0.8, zorder=2)
    for e in (min(v), max(v)):
        ax.plot([e, e], [y - cap, y + cap], color=col, lw=0.8, zorder=2)


for y, (m, c) in zip(YM, METHODS):
    s = vals("SHIFT_SRC", m, "coverage", DS6)
    sm = statistics.median(s)
    if m in RECAL_OK:
        r = vals("SHIFT_RECAL", m, "coverage", DS6)
        rm = statistics.median(r)
        axC.plot([sm, rm], [y, y], color=GREY, lw=0.6, zorder=1)
        crange(axC, r, y, style.ROLE["recal"])
        axC.plot([rm], [y], ls="none", marker=style.MARK["recal"], ms=4.2,
                 mfc=style.ROLE["recal"], mec=style.ROLE["recal"], zorder=4)
    crange(axC, s, y, style.ROLE["source"])
    axC.plot([sm], [y], ls="none", marker=style.MARK["source"], ms=4.2, mew=0.9,
             mfc=CLS[c]["mfc"] if c == WRAP else style.ROLE["source"],
             mec=style.ROLE["source"], zorder=4)
refline(axC, 0.90, -0.55, TOP + 0.55)
method_ticks(axC, C_LO, C_HI)
axC.set_xlim(0.0, 1.02)
axC.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
axC.set_xlabel("coverage on the design region")
axC.set_title("Design shift, before and after recalibration", fontsize=8)
QA["c.nominal"] = axC.text(0.893, TOP + 1.33, "nominal 0.90", fontsize=7,
                           ha="right", va="center")
QA["c.note"] = axC.text(0.012, -1.36, "raw methods cannot use target labels (wrappers can)",
                        fontsize=7, ha="left", va="center", color=INK)
QA["c.legend"] = axC.legend(
    handles=[Line2D([], [], ls="none", marker=style.MARK["source"], ms=4.2,
                    color=style.ROLE["source"], label="source-calibrated"),
             Line2D([], [], ls="none", marker=style.MARK["recal"], ms=4.2,
                    color=style.ROLE["recal"], label="target-recalibrated")],
    loc="upper left", bbox_to_anchor=(-0.008, 1.012), ncol=2, handlelength=0.9,
    borderpad=0.15, columnspacing=1.0, handletextpad=0.35)

# ------------------------------------------------------------------ d  invisible failure
YD = np.arange(len(DS6))[::-1]
TD = len(DS6) - 1
D_LO, D_HI = -8.6, TD + 1.9
for y, d in zip(YD, DS6):
    S = D[d]["SHIFT_SRC"]["SC"]
    keep = 1.0 - S["abstain_1iqr"]["mean"]
    cov = S["coverage"]["mean"]
    axD.plot([cov, keep], [y, y], color=GREY, lw=0.8, zorder=1)
    axD.plot([keep], [y], ls="none", marker=style.MARK["source"], ms=4.2,
             mfc=style.ROLE["source"], mec=style.ROLE["source"], zorder=3)
    axD.plot([cov], [y], ls="none", marker=style.MARK["fail"], ms=4.6,
             mfc=style.ROLE["fail"], mec=style.ROLE["fail"], zorder=3)
refline(axD, 0.90, -0.55, TD + 0.30)
axD.set_yticks(YD)
axD.set_yticklabels([DS_SHORT[d] for d in DS6])
axD.set_ylim(D_LO, D_HI)
axD.tick_params(axis="y", length=0)
axD.set_xlim(-0.02, 1.02)
axD.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
axD.set_xlabel("proportion")
axD.set_title("Invisibility to the width rule", fontsize=8)
QA["d.nominal"] = axD.text(0.885, TD + 1.05, "nominal 0.90", fontsize=7,
                           ha="right", va="center")
# The vermillion series is the coverage of ALL design-region candidates (see the
# module docstring): it is computed independently of the abstention rule, so it
# must not be described as "the coverage of the intervals that were kept".
QA["d.legend"] = axD.legend(
    handles=[Line2D([], [], ls="none", marker=style.MARK["source"], ms=4.2,
                    color=style.ROLE["source"], label="abstention does not trigger"),
             Line2D([], [], ls="none", marker=style.MARK["fail"], ms=4.6,
                    color=style.ROLE["fail"], label="coverage on the design region")],
    loc="lower left", bbox_to_anchor=(-0.012, 0.315), handlelength=0.9,
    borderpad=0.15, labelspacing=0.25, handletextpad=0.35)
# The threshold IQR is the one the rule can see at decision time (the target
# recalibration pool, JSON iqr_seed0.abst_shift), which is NOT the reporting IQR
# that normalises the widths (iqr_seed0.report_tgt); both are named here.
# Three short lines that start INSIDE the left spine: a wider two-line version
# has to be anchored left of the spine, and the spine is then drawn through the
# type, which is the same defect as a reference line crossing a legend.
QA["d.note"] = axD.text(0.005, -6.22,
                        "rule: abstain if the interval is wider than 1 × IQR\n"
                        "of the target recalibration pool.  After recalibration\n"
                        "every candidate abstains, at 2.1–5.1 × target IQR.",
                        fontsize=7, ha="left", va="center", color=INK, linespacing=1.35)

# ------------------------------------------------------------------ panel letters
fig.canvas.draw()


def letter(ax, L):
    """style.panel with the letter pinned to an ABSOLUTE x (cm) of the page.

    ``style.panel`` offsets from ``ax.get_tightbbox().x0``; for panels b and d that
    x0 is dragged ~1.5 cm left by the "BOOT+conf" / "polymer Tg" tick labels, which
    put the letter on top of the neighbouring panel.  Pinning the absolute x keeps
    every letter in its own gutter (QA block below re-measures the clearances).
    """
    bb = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    style.panel(ax, L, x=LETTER_X[L] / W - bb.x0)


for ax, L in [(axA, "a"), (axB, "b"), (axC, "c"), (axD, "d")]:
    letter(ax, L)
# tight=False: the layout already reserves its own margins (every axes box is
# given in centimetres of the page), so the file is written at EXACTLY the
# declared 17.5 x 11.0 cm.  bbox_inches="tight" would instead crop to the ink and
# add 0.1 in of padding, which pushed the height to 11.2 cm - over the 11 cm cap -
# and shrank the width below the 17.5 cm double-column measure.
png = style.save(fig, "fig7_uq_scorecard", 7, tight=False)
print(png)

# ------------------------------------------------------------------ printed numbers
print("\n--- a: ID coverage, median over 7 datasets [min, max] ---")
for m, _ in METHODS:
    v = vals("ID", m, "coverage", DS7)
    print(f"  {m:10s} {statistics.median(v):.3f} [{min(v):.3f}, {max(v):.3f}]")
print("--- b: ID interval score / IQR, median [min, max]; median normalised width ---")
for m, _ in METHODS:
    v = vals("ID", m, "interval_score_norm", DS7)
    w = vals("ID", m, "norm_width", DS7)
    print(f"  {m:10s} IS {statistics.median(v):.2f} [{min(v):.2f}, {max(v):.2f}]  "
          f"nW {statistics.median(w):.2f}")
print("--- c: design-region coverage, median over 6 datasets [min, max] ---")
for m, _ in METHODS:
    s = vals("SHIFT_SRC", m, "coverage", DS6)
    line = f"  {m:10s} src {statistics.median(s):.3f} [{min(s):.3f}, {max(s):.3f}]"
    if m in RECAL_OK:
        r = vals("SHIFT_RECAL", m, "coverage", DS6)
        line += f"   recal {statistics.median(r):.3f} [{min(r):.3f}, {max(r):.3f}]"
    print(line)
print("--- d: split conformal, source-calibrated, design region ---")
for d in DS6:
    S = D[d]["SHIFT_SRC"]["SC"]
    T = D[d]["SHIFT_RECAL"]["SC"]
    Q = D[d]["iqr_seed0"]
    print(f"  {DS_SHORT[d]:12s} m = {D[d]['m']:2d}  not abstained "
          f"{1 - S['abstain_1iqr']['mean']:.2f}  coverage {S['coverage']['mean']:.3f}  "
          f"nW {S['norm_width']['mean']:.2f}  |  recal: not abstained "
          f"{1 - T['abstain_1iqr']['mean']:.2f}  coverage {T['coverage']['mean']:.3f}  "
          f"nW {T['norm_width']['mean']:.2f}  |  IQR seed0: report_tgt "
          f"{Q['report_tgt']:.1f}  abst_shift {Q['abst_shift']:.1f}")
nwr = vals("SHIFT_RECAL", "SC", "norm_width", DS6)
print(f"  recalibrated normalised width, median {statistics.median(nwr):.2f} "
      f"[{min(nwr):.2f}, {max(nwr):.2f}] x target IQR")

# ------------------------------------------------------------------ QA: measured clearances
# Everything the Editor / Reviewer 1 rules constrain is re-measured on the drawn
# figure and printed in millimetres of the PRINTED page, so a layout regression is
# caught by running the script rather than by eye.
rend = fig.canvas.get_renderer()
MM = 25.4 / fig.dpi


def bb(obj):
    b = obj.get_window_extent(rend) if not hasattr(obj, "get_tightbbox") or isinstance(obj, plt.Text) \
        else obj.get_tightbbox(rend)
    return [v * MM for v in (b.x0, b.y0, b.x1, b.y1)]


def tight(ax):
    b = ax.get_tightbbox(rend)
    return [v * MM for v in (b.x0, b.y0, b.x1, b.y1)]


def legbb(lg):
    b = lg.get_window_extent(rend)
    return [v * MM for v in (b.x0, b.y0, b.x1, b.y1)]


letters = {t.get_text(): t for t in fig.texts if t.get_text() in "abcd" and t.get_fontsize() == 10}
print("\n--- QA, millimetres of the printed page (h = horizontal gap, v = vertical gap) ---")
for L, left_ax in [("b", axA), ("d", axC)]:
    lb = bb(letters[L])
    ta = tight(left_ax)
    tb = tight(axB if L == "b" else axD)
    print(f"  letter {L}: x {lb[0]:.1f}-{lb[2]:.1f} mm | h to panel "
          f"{'a' if L == 'b' else 'c'} tight right ({ta[2]:.1f}) = {lb[0] - ta[2]:+.2f} mm"
          f" | h to panel {L} tight left ({tb[0]:.1f}) = {tb[0] - lb[2]:+.2f} mm")
dl, dn = legbb(QA["d.legend"]), bb(QA["d.note"])
print(f"  d legend y {dl[1]:.1f}-{dl[3]:.1f} | d note y {dn[1]:.1f}-{dn[3]:.1f}"
      f" | v gap = {dl[1] - dn[3]:+.2f} mm")
d_row0_mk = axD.transData.transform((0, 0))[1] * MM - 4.2 * 25.4 / 144
print(f"  d legend y top {dl[3]:.1f} | v to the bottom row's marker ({d_row0_mk:.1f}) = "
      f"{d_row0_mk - dl[3]:+.2f} mm")
print(f"  d legend x right {dl[2]:.1f} | d note x {dn[0]:.1f}-{dn[2]:.1f}"
      f" | axD right spine {axD.get_window_extent(rend).x1 * MM:.1f}"
      f" | page width {W * 10:.1f}")
dn_row = axD.transData.transform((0, 0))[1] * MM
print(f"  d note y top {dn[3]:.1f} | v to bottom row ({dn_row:.1f}) = {dn_row - 1.6 - dn[3]:+.2f} mm"
      f" | v to axes bottom ({axD.get_window_extent(rend).y0 * MM:.1f}) = "
      f"{dn[1] - axD.get_window_extent(rend).y0 * MM:+.2f} mm | h to the left spine "
      f"({axD.get_window_extent(rend).x0 * MM:.1f}) = {dn[0] - axD.get_window_extent(rend).x0 * MM:+.2f} mm")
dnom = bb(QA["d.nominal"])
d_line_top = axD.transData.transform((0, TD + 0.30))[1] * MM
# On the top row the only ink under the label's x span is the grey connector
# (the blue and vermillion markers sit far to its right), so the clearance is
# measured against the connector and, horizontally, against the nearest marker.
d_conn = axD.transData.transform((0, TD))[1] * MM + 0.14
d_mk_x = axD.transData.transform((1.0, TD))[0] * MM - 4.2 * 25.4 / 144
print(f"  d 'nominal 0.90' y {dnom[1]:.1f}-{dnom[3]:.1f} x right {dnom[2]:.1f} | v to reference-line "
      f"top ({d_line_top:.1f}) = {dnom[1] - d_line_top:+.2f} mm | v to top-row connector "
      f"({d_conn:.1f}) = {dnom[1] - d_conn:+.2f} mm | h to that row's nearest marker "
      f"({d_mk_x:.1f}) = {d_mk_x - dnom[2]:+.2f} mm")
bn, bl = bb(QA["b.note"]), legbb(QA["b.legend"])
bar_top = axB.transData.transform((0, TOP + BAR_H / 2))[1] * MM
print(f"  b note y {bn[1]:.1f}-{bn[3]:.1f} | v to SC bar top ({bar_top:.1f}) = {bn[1] - bar_top:+.2f} mm"
      f" | v to legend bottom ({bl[1]:.1f}) = {bl[1] - bn[3]:+.2f} mm | x right {bn[2]:.1f}"
      f" vs axB right spine {axB.get_window_extent(rend).x1 * MM:.1f}")
for P, ax, key, lg in [("a", axA, "a.nominal", "a.legend"), ("c", axC, "c.nominal", "c.legend")]:
    nt, lgb = bb(QA[key]), legbb(QA[lg])
    mk_top = ax.transData.transform((0, TOP))[1] * MM + 4.6 * 25.4 / 144
    ln_top = ax.transData.transform((0, TOP + 0.55))[1] * MM
    print(f"  {P} 'nominal 0.90' y {nt[1]:.1f}-{nt[3]:.1f} | v to median marker top ({mk_top:.1f})"
          f" = {nt[1] - mk_top:+.2f} mm | v to reference-line top ({ln_top:.1f}) = "
          f"{nt[1] - ln_top:+.2f} mm | v to legend bottom ({lgb[1]:.1f}) = {lgb[1] - nt[3]:+.2f} mm")
ck = legbb(QA["a.classkey"])
sep_y = axA.transData.transform((0, TOP - 6.5))[1] * MM
print(f"  a class key x {ck[0]:.1f}-{ck[2]:.1f} y {ck[1]:.1f}-{ck[3]:.1f} | v to the class "
      f"separator ({sep_y:.1f}) = {sep_y - ck[3]:+.2f} mm | v to the axes bottom "
      f"({axA.get_window_extent(rend).y0 * MM:.1f}) = {ck[1] - axA.get_window_extent(rend).y0 * MM:+.2f} mm"
      f" | h to the nearest datum on the rows it spans "
      f"({axA.transData.transform((0.784, 0))[0] * MM:.1f}) = "
      f"{axA.transData.transform((0.784, 0))[0] * MM - ck[2]:+.2f} mm")
cn = bb(QA["c.note"])
row0 = axC.transData.transform((0, 0))[1] * MM - 1.5
print(f"  c note y {cn[1]:.1f}-{cn[3]:.1f} | v to bottom row ({row0:.1f}) = {row0 - cn[3]:+.2f} mm"
      f" | x right {cn[2]:.1f} vs axC right spine {axC.get_window_extent(rend).x1 * MM:.1f}")
# panel b: the median-width tick and the interval-score range must not share a lane
print("  b width tick lane y-0.34..y-0.02, score-range lane y+0.06..y+0.28 -> "
      "disjoint by 0.08 row units on every row")
sizes = sorted({round(t.get_fontsize(), 2) for t in fig.findobj(plt.Text) if t.get_text().strip()})
print(f"  font sizes present: {sizes}  (minimum must be >= 7)")
ftb = fig.get_tightbbox(rend)
print(f"  ink bbox {ftb.x0 * 2.54:.2f}-{ftb.x1 * 2.54:.2f} x {ftb.y0 * 2.54:.2f}-{ftb.y1 * 2.54:.2f} cm "
      f"inside the {W} x {H} cm canvas written by save(tight=False)")
