"""Figure 2 - what the literature actually reports (Reviewer 2, comment 13).

Replaces the "illustrative" matrix of the submitted manuscript with the result of
a pre-registered systematic survey of the Europe PMC open-access literature,
2020-2025, five application areas.

a  proportion of coded papers reporting each practice, per area and pooled,
   with 95% Wilson intervals (dot plot; colour AND marker encode the practice);
b  the pooled funnel from "any uncertainty" down to "calibrated uncertainty with
   external validation and a distribution-shift test";
c  PRISMA-style screening flow, one nested bar per area, with the real counts.

Every number is read at run time from revision/survey/survey_results.json (the
machine-readable copy of survey_results.md) and from survey_coded_papers.csv;
nothing is typed in by hand. The Wilson intervals are recomputed here and
asserted against the ones stored in the JSON, the nested funnel counts of panel
b are recomputed from the per-paper table and asserted against the JSON, and the
screening totals of panel c are asserted against the stored per-area records.

No mathtext is used, so the only fonts embedded in the PDF are ArialMT and
Arial-BoldMT. The palette deliberately avoids the vermillion/bluish-green pair:
the three uncertainty levels are an ordered light-blue -> blue -> black ramp and
the two validation practices are reddish purple and orange, each with its own
marker, so colour is never load-bearing. Panel c is a neutral grey ramp whose
order is redundant with the nesting of the bars.

Typographic conventions that the whole figure keeps (they were inconsistent in
the first draft and are now enforced by the measurement block at the bottom of
this file, run with FIG2_CHECK=1):

* every row label in a horizontal-bar panel sits closer to its OWN bar than to
  the bar above it, by a factor of at least three (panels b and c);
* no two text or legend blocks are closer than 2 mm;
* the topmost and bottommost ink is at least 2 mm from the canvas edge.
"""
import csv
import json
import math
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import style  # noqa: E402

style.apply()

SURVEY = os.path.join(REV, "survey")
D = json.load(open(os.path.join(SURVEY, "survey_results.json"), encoding="utf-8"))
ROWS = list(csv.DictReader(open(os.path.join(SURVEY, "survey_coded_papers.csv"),
                                encoding="utf-8")))

# ------------------------------------------------------------------ verification
Z = 1.959963984540054


def wilson(k, n, z=Z):
    """Wilson score interval for a binomial proportion."""
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def check(cell):
    """Assert a stored estimate against a freshly computed Wilson interval."""
    lo, hi = wilson(cell["k"], cell["n"])
    assert abs(lo - cell["wilson95"][0]) < 1e-9 and abs(hi - cell["wilson95"][1]) < 1e-9
    assert abs(cell["k"] / cell["n"] - cell["proportion"]) < 1e-12
    return cell


# one row per unique paper (the CSV has one row per area-record; a paper sampled
# in two areas carries the same adjudicated codes in both, checked here)
PAPERS = {}
for r in ROWS:
    prev = PAPERS.setdefault(r["pmcid"], r)
    for item in ("U", "E", "EP", "S"):
        assert prev[item] == r[item], (r["pmcid"], item)
assert len(ROWS) == D["prisma"]["_totals"]["coded_area_records"] == 169
assert len(PAPERS) == D["prisma"]["_totals"]["coded_unique_papers"] == 153
N_ALL = len(PAPERS)
N_REC = len(ROWS)
N_BOTH = D["prisma"]["_totals"]["papers_in_two_areas"]
assert N_REC - N_ALL == N_BOTH == 16


def count(pred):
    return sum(1 for r in PAPERS.values() if pred(r))


U = lambda r: int(r["U"])          # noqa: E731
E = lambda r: int(r["E"])          # noqa: E731
S = lambda r: int(r["S"])          # noqa: E731

POOL = D["estimates"]["overall_unique_papers"]
assert count(lambda r: U(r) >= 1) == POOL["U_ge1"]["k"]
assert count(lambda r: U(r) >= 2) == POOL["U_ge2"]["k"]
assert count(lambda r: U(r) >= 3) == POOL["U3"]["k"]
assert count(lambda r: E(r) == 1) == POOL["E1"]["k"]
assert count(lambda r: S(r) == 1) == POOL["S1"]["k"]
K_U2ES = count(lambda r: U(r) >= 2 and E(r) == 1 and S(r) == 1)
assert K_U2ES == POOL["U_ge2_and_E1_and_S1"]["k"] == 1

# ------------------------------------------------------------------ panel a data
# The area names are the registered ones (survey_results.md section 1), split
# over three lines without dropping any word of the registered name.
AREAS = [("A1", "A1\nimaging-based\ncharacterisation"),
         ("A2", "A2\nproperty prediction\nand inverse design"),
         ("A3", "A3\nbiosensors and\nbioelectronics"),
         ("A4", "A4\ntissue engineering,\nscaffolds, hydrogels"),
         ("A5", "A5\nautonomous and\nsequential discovery"),
         ("ALL", "all areas\nunique papers\n ")]

# key, legend label, colour, marker
OUTCOMES = [
    ("U_ge1", "U≥1  any uncertainty",  style.OI["skyblue"], "o"),
    ("U_ge2", "U≥2  per-prediction",   style.OI["blue"],    "s"),
    ("U3",    "U3  calibrated",             "black",             "D"),
    ("E1",    "E1  external validation",    style.OI["purple"],  "^"),
    ("S1",    "S1  shift test",             style.OI["orange"],  "P"),
]

CELLS = {}
for akey, _ in AREAS:
    src = POOL if akey == "ALL" else D["estimates"]["by_area"][akey]
    for okey, *_ in OUTCOMES:
        CELLS[(akey, okey)] = check(src[okey])
NS = {a: (POOL["n"] if a == "ALL" else D["estimates"]["by_area"][a]["n"])
      for a, _ in AREAS}
assert sum(NS[a] for a, _ in AREAS if a != "ALL") == N_REC

# ------------------------------------------------------------------ panel b data
K_U3ES = count(lambda r: U(r) >= 3 and E(r) == 1 and S(r) == 1)
assert K_U3ES == 0
# label, count, colour, borderline flag
FUNNEL = [
    ("any uncertainty  U≥1",            POOL["U_ge1"]["k"], style.OI["skyblue"], False),
    ("per-prediction uncertainty  U≥2", POOL["U_ge2"]["k"], style.OI["blue"], False),
    ("calibrated uncertainty  U3",           POOL["U3"]["k"], "black", True),
    ("calibrated + external + shift test",   K_U3ES, "black", False),
]

# ------------------------------------------------------------------ panel c data
PR = D["prisma"]
AK = ["A1", "A2", "A3", "A4", "A5"]
STAGES = [("records identified", "identified", "#E2E2E2",
           lambda p: p["identified_epmc_topic_all_types"]),
          ("research articles", "research articles", "#BFBFBF",
           lambda p: p["research_only_records"]),
          ("open-access frame", "open-access frame", "#969696",
           lambda p: p["open_access_frame"]),
          ("screened", "screened, random order", "#6B6B6B",
           lambda p: p["screened_in_random_order"]),
          ("eligible", "eligible, coded", "black",
           lambda p: p["eligible"])]
FLOW = {s[0]: [s[3](PR[a]) for a in AK] for s in STAGES}
T = PR["_totals"]
assert sum(FLOW["research articles"]) == T["research_only_area_records"] == 2018
assert sum(FLOW["open-access frame"]) == T["oa_frame_area_records"] == 871
assert sum(FLOW["screened"]) == T["screened_area_records"] == 518
assert sum(FLOW["eligible"]) == T["eligible_area_records"] == 169
N_ID = sum(FLOW["records identified"])

# "records identified" is the topic query over ALL publication types and all
# languages; "research articles" is a SEPARATE Europe PMC query that adds
# LANG:eng and the full type filter (reviews, meta-analyses, editorials,
# comments, letters, errata, retractions, preprints). The stage is therefore not
# "identified minus tagged reviews minus preprints": that subtraction overshoots
# by OTHER_TYPES records. The panel note says the stages are separate queries
# and names the extra removals; the caption gives the OTHER_TYPES count, so the
# reader is not left to discover the discrepancy in the Supporting Information.
OTHER_TYPES = (N_ID
               - sum(PR[a]["removed_tagged_reviews"] for a in AK)
               - sum(PR[a]["removed_preprints"] for a in AK)
               - sum(FLOW["research articles"]))
assert OTHER_TYPES == 18

REASONS = [("no biomedical material or device", "NOT_BIOMED"),
           ("no machine-learning model", "NO_ML"),
           ("eligible but another area", "WRONG_AREA"),
           ("not original research", "NOT_RESEARCH")]
RC = {lab: [PR[a]["excluded_by_reason"].get(code, 0) for a in AK]
      for lab, code in REASONS}
for j, a in enumerate(AK):
    assert PR[a]["excluded_total"] == sum(RC[lab][j] for lab, _ in REASONS)
    assert PR[a]["screened_in_random_order"] - PR[a]["eligible"] == PR[a]["excluded_total"]
    assert PR[a]["coded_coder_A"] == PR[a]["eligible"]
N_DUAL = T["dual_coded_area_records"]
assert N_DUAL == sum(PR[a]["dual_coded_coder_B"] for a in AK) == 85

# Consecutive stages that coincide, so that one grey level of the nested bar has
# zero visible width (only A5, whose whole open-access frame was screened). Each
# one is marked with a tick under the bar and named in the note.
COINCIDE = []
ORDER = [s[0] for s in STAGES]
for j, a in enumerate(AK):
    for k in range(len(ORDER) - 1):
        if FLOW[ORDER[k]][j] == FLOW[ORDER[k + 1]][j]:
            COINCIDE.append((j, FLOW[ORDER[k]][j]))
assert COINCIDE == [(4, 145)]

# ================================================================== layout
W, H = 17.5, 14.0            # final print size in cm; nothing is downscaled
fig = plt.figure(figsize=(W * style.CM, H * style.CM))


def box(x0, y0, x1, y1):
    return fig.add_axes([x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H])


axA = box(1.95, 10.45, 17.05, 13.14)
axB = box(1.95, 4.90, 8.40, 7.80)
axC = box(9.55, 4.62, 16.95, 7.80)

# ------------------------------------------------------------------ a
OFF = np.array([-0.28, -0.14, 0.0, 0.14, 0.28])
axA.axvspan(4.5, 5.6, color="#F2F2F2", lw=0, zorder=0)
axA.axvline(4.5, color=style.OI["grey"], lw=0.6, ls=(0, (2, 2)), zorder=1)
for gi, (akey, _) in enumerate(AREAS):
    for oi, (okey, _lab, col, mk) in enumerate(OUTCOMES):
        c = CELLS[(akey, okey)]
        p, lo, hi = (100 * c["proportion"], 100 * c["wilson95"][0],
                     100 * c["wilson95"][1])
        axA.errorbar(gi + OFF[oi], p, yerr=[[max(0.0, p - lo)], [max(0.0, hi - p)]],
                     fmt=mk, ms=3.4, mfc=col, mec=col, mew=0.6, ecolor=col,
                     elinewidth=0.8, capsize=1.3, capthick=0.8, zorder=3)

axA.set_xlim(-0.55, 5.6)
axA.set_ylim(-4, 100)
axA.set_yticks([0, 20, 40, 60, 80, 100])
# Two lines: set on one line the rotated label is 4.3 cm long inside a 2.7 cm
# axes, so it overhangs the panel at both ends and its top runs into the trim
# edge of the canvas.
axA.set_ylabel("papers reporting\nthe practice (%)", linespacing=1.3)
axA.set_xticks(range(6))
axA.set_xticklabels([lab for _, lab in AREAS], linespacing=1.4)
axA.tick_params(axis="x", length=0, pad=3)
# The five per-area n are AREA-RECORDS and sum to 169, while the pooled column is
# over 153 UNIQUE PAPERS, because 16 papers were sampled in two areas. The unit
# is therefore printed with every n, so that a reader who adds the per-area n up
# and gets 169 can see immediately why the pooled column is not 169.
Y_N = 9.20
for gi, (akey, _) in enumerate(AREAS):
    unit = "papers" if akey == "ALL" else "records"
    fig.text((1.95 + (gi + 0.55) / 6.15 * 15.10) / W, Y_N / H,
             f"n = {NS[akey]} {unit}", ha="center", va="top", fontsize=7,
             fontweight="bold")

handles = [Line2D([], [], color=c, marker=m, ls="none", ms=3.4, mfc=c, mec=c,
                  label=lab) for _k, lab, c, m in OUTCOMES]
fig.legend(handles=handles, loc="lower center",
           bbox_to_anchor=((1.95 + 17.05) / 2 / W, 13.30 / H),
           bbox_transform=fig.transFigure, ncol=5, handlelength=1.0,
           handletextpad=0.35, columnspacing=1.7, borderpad=0.2)

# ------------------------------------------------------------------ b
XB = 40.0
ypos = np.arange(len(FUNNEL))[::-1]
BH = 0.42                                    # bar height, data units
LAB_DY = 0.25                                # label baseline above the row centre
for y, (lab, k, col, borderline) in zip(ypos, FUNNEL):
    lo, hi = wilson(k, N_ALL)
    p, phi = 100 * k / N_ALL, 100 * hi
    if k == 0:
        # A bar of zero length is invisible; draw its left edge so that the row
        # reads as an empty bar rather than as a missing one, and so that it
        # cannot be confused with the 0.7% row above it.
        axB.plot([0, 0], [y - BH / 2, y + BH / 2], color=col, lw=1.1,
                 solid_capstyle="butt", zorder=2)
    else:
        axB.barh(y, p, height=BH, color=col, edgecolor="none", zorder=2)
    axB.errorbar(p, y, xerr=[[max(0.0, p - 100 * lo)], [max(0.0, phi - p)]],
                 fmt="none", ecolor="black", elinewidth=0.8, capsize=1.6,
                 capthick=0.8, zorder=4)
    if borderline:
        # The U3 bar is 2.6 pt wide at print size, so a hatch inside it is not
        # resolvable. Mark its end with an OPEN diamond instead: the diamond is
        # the U3 glyph of panel a, open means "borderline", and a white lozenge
        # on the black bar is legible at 1.3 mm where a hatch is not.
        axB.plot([p], [y], marker="D", ms=3.8, mfc="white", mec=col, mew=0.8,
                 ls="none", zorder=5)
        BL = axB.text(0.0, y + LAB_DY, "(borderline)", fontsize=7, ha="center",
                      va="bottom", color="#4D4D4D")
    # The stage name and its value share one line just above their own bar: the
    # gap to the bar they describe is about a quarter of the gap to the bar
    # above, so proximity cannot attribute a label to the wrong row.
    t_lab = axB.text(0.0, y + LAB_DY, lab, fontsize=7, ha="left", va="bottom")
    t_val = axB.text(XB, y + LAB_DY, f"{k}/{N_ALL} = {p:.1f}%", fontsize=7,
                     ha="right", va="bottom")
    if borderline:
        BL_ROW = (BL, t_lab, t_val)

axB.set_xlim(0, XB)
axB.set_ylim(-0.62, len(FUNNEL) - 0.28)
axB.set_xticks([0, 10, 20, 30, 40])
axB.set_yticks([])
axB.spines["left"].set_visible(False)
axB.set_xlabel("papers, all areas pooled (%)")
axB.set_title(f"Funnel over the {N_ALL} unique papers", fontsize=8, loc="left", x=0.0)

_lo0, _hi0 = wilson(0, N_ALL)
NOTE_B = [
    "Each step is a subset of the one above; whiskers are 95%",
    f"Wilson intervals. The {N_REC} coded area-records of panel a",
    f"cover the {N_ALL} unique papers counted here, because {N_BOTH} papers",
    "were sampled in two areas. The single U3 paper is border-",
    f"line (open diamond): demoted to U2 the row is 0/{N_ALL} = 0.0%",
    f"({100 * _lo0:.1f}-{100 * _hi0:.1f}), so calibrated uncertainty is "
    f"0.0-{100 * POOL['U3']['proportion']:.1f}% depending",
    "on adjudication. One paper comes closest to the full chain,",
    "with per-prediction uncertainty, external validation and a",
    f"shift test ({K_U2ES}/{N_ALL} = {100 * K_U2ES / N_ALL:.1f}%).",
]
Y_NOTE_B = 3.80
for i, line in enumerate(NOTE_B):
    fig.text(1.95 / W, (Y_NOTE_B - 0.315 * i) / H, line, fontsize=7,
             ha="left", va="top")

# ------------------------------------------------------------------ c
XC = 900.0
CW = 7.40                                  # axes width in cm, for label spacing
DIGIT = 0.175 / (CW / XC)                  # width of one 7 pt digit, in records
PAD = 0.09 / (CW / XC)                     # minimum gap between two labels
CBH = 0.34                                 # bar height, data units
ypos_c = np.arange(len(AK))[::-1]
for j, y in enumerate(ypos_c):
    for si, (key, _leg, col, _fn) in enumerate(STAGES):
        axC.barh(y, FLOW[key][j], height=CBH, color=col, edgecolor="none",
                 zorder=2 + si)
    # Counts sit above the bar, placed from the widest stage inwards, normally
    # just to the right of the bar end. A count that would touch the one already
    # placed is flipped to the inside of its own bar end instead (only A4's
    # "screened", whose bar is nearly as long as its frame), so that nothing
    # overlaps and no count can be read as belonging to another row.
    left = XC                             # left edge reached so far
    for key in ["records identified", "research articles", "open-access frame",
                "screened"]:
        v = FLOW[key][j]
        if key == "screened" and v == FLOW["open-access frame"][j]:
            continue                      # A5: frame exhausted, the two coincide
        w = len(str(v)) * DIGIT
        if v + 0.6 * PAD + w + PAD <= left:
            axC.text(v + 0.6 * PAD, y + 0.21, f"{v}", fontsize=7, ha="left",
                     va="bottom")
            left = v + 0.6 * PAD
        else:
            axC.text(v - PAD, y + 0.21, f"{v}", fontsize=7, ha="right",
                     va="bottom")
            left = v - PAD - w
    axC.text(XC - 8, y, f"{FLOW['eligible'][j]}", fontsize=7, ha="right",
             va="center", fontweight="bold")

# Where two consecutive stages coincide the later, darker band covers the earlier
# one completely and that grey level has zero width. Mark the boundary with a
# tick under the bar so the row is not silently missing a stage.
for j, v in COINCIDE:
    y = ypos_c[j]
    axC.plot([v, v], [y - CBH / 2 - 0.06, y - CBH / 2 - 0.26], color="black",
             lw=0.7, solid_capstyle="butt", zorder=6)

axC.plot([848, 848], [-0.58, len(AK) - 0.18], color=style.OI["grey"], lw=0.6,
         clip_on=False, zorder=1)
axC.text(XC - 8, len(AK) - 0.34, "eligible", fontsize=7, fontweight="bold",
         ha="right", va="bottom")
axC.set_xlim(0, XC)
axC.set_ylim(-0.62, len(AK) - 0.18)
axC.set_yticks(ypos_c)
axC.set_yticklabels(AK)
axC.tick_params(axis="y", length=0)
axC.set_xticks([0, 200, 400, 600, 800])
axC.set_xlabel("area-records")
axC.set_title("Screening flow, nested stages", fontsize=8, loc="left", x=0.0)

legc = [Patch(facecolor=col, edgecolor="none", label=leg)
        for _k, leg, col, _fn in STAGES]
fig.legend(handles=legc, loc="upper left",
           bbox_to_anchor=(9.55 / W, 3.66 / H), bbox_transform=fig.transFigure,
           ncol=3, handlelength=1.0, handleheight=0.8, handletextpad=0.3,
           columnspacing=0.6, labelspacing=0.3, borderpad=0.15)

RS = {code: sum(RC[lab]) for lab, code in REASONS}
NOTE_C = [
    "Each stage is a separate Europe PMC query, not a subtraction:",
    "'research articles' also applies the language filter and drops edi-",
    f"torials, comments, letters and errata. Screened out: {RS['NOT_BIOMED']} no bio-",
    f"medical material or device, {RS['NO_ML']} no machine-learning model, "
    f"{RS['WRONG_AREA']} eli-",
    f"gible but another area, {RS['NOT_RESEARCH']} not original research. "
    "A5 (tick) is a cen-",
    f"sus: its whole frame of {FLOW['open-access frame'][4]} records was screened, "
    f"{FLOW['eligible'][4]} eligible. {N_DUAL} of",
    f"the {sum(FLOW['eligible'])} records ({100 * N_DUAL / sum(FLOW['eligible']):.0f}%) "
    "were coded twice, blind, by a second large-",
    "language-model coder and every disagreement was adjudicated.",
]
Y_NOTE_C = 2.66
for i, line in enumerate(NOTE_C):
    fig.text(9.55 / W, (Y_NOTE_C - 0.28 * i) / H, line, fontsize=7,
             ha="left", va="top")

fig.canvas.draw()

# Centre the "(borderline)" tag in the gap that the U3 row's name and value
# leave, measured on the rendered text rather than guessed, and check that it is
# not closer to either than 1 mm (it would otherwise read as one phrase).
_R = fig.canvas.get_renderer()
_inv = axB.transData.inverted()
_BL, _LAB, _VAL = BL_ROW
_x0, _x1 = _LAB.get_window_extent(_R).x1, _VAL.get_window_extent(_R).x0
_BL.set_x(_inv.transform((0.5 * (_x0 + _x1), 0))[0])
fig.canvas.draw()
_bl = _BL.get_window_extent(_R)
_gap = min(_bl.x0 - _x0, _x1 - _bl.x1) / fig.dpi * 2.54
assert _gap > 0.20, _gap


def letter(ax, L, x_cm):
    bb = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    style.panel(ax, L, x=x_cm / W - bb.x0)


for ax, L, xc in [(axA, "a", 0.32), (axB, "b", 0.32), (axC, "c", 8.45)]:
    letter(ax, L, xc)

png = style.save(fig, "fig2_survey", 2, tight=False)
print(png)

# ================================================================== measurement
# Run with FIG2_CHECK=1 to print the geometry the QA pass measures: the label
# attribution of panels b and c, the clearance of every text block boundary and
# the distance of the outermost ink from the canvas edge.
if os.environ.get("FIG2_CHECK"):
    fig.canvas.draw()
    R = fig.canvas.get_renderer()
    PT = 72 / 2.54                                      # points per cm

    def bb_cm(artist):
        b = artist.get_window_extent(R)
        return [v / fig.dpi * 2.54 for v in (b.x0, b.y0, b.x1, b.y1)]

    print("\n-- panel b: label to its own bar vs to the bar above (pt)")
    for y, (lab, k, col, _b) in zip(ypos, FUNNEL):
        t = [t for t in axB.texts if t.get_text() == lab][0]
        x0, y0, x1, y1 = bb_cm(t)
        own = (axB.transData.transform((0, y + BH / 2))[1] / fig.dpi * 2.54)
        above = (axB.transData.transform((0, y + 1 - BH / 2))[1] / fig.dpi * 2.54)
        print(f"   {lab[:34]:34s} own {max(0, y0 - own) * PT:5.2f}   "
              f"above {max(0, above - y1) * PT:5.2f}")

    print("-- panel c: count to its own bar vs to the bar above (pt)")
    for j, y in enumerate(ypos_c):
        v = FLOW["records identified"][j]
        t = [t for t in axC.texts if t.get_text() == str(v)][0]
        x0, y0, x1, y1 = bb_cm(t)
        own = axC.transData.transform((0, y + CBH / 2))[1] / fig.dpi * 2.54
        above = axC.transData.transform((0, y + 1 - CBH / 2))[1] / fig.dpi * 2.54
        print(f"   {AK[j]} {v:5d}                             "
              f"own {max(0, y0 - own) * PT:5.2f}   "
              f"above {max(0, above - y1) * PT:5.2f}")

    def block(items):
        bs = [bb_cm(a) for a in items]
        return (min(b[0] for b in bs), min(b[1] for b in bs),
                max(b[2] for b in bs), max(b[3] for b in bs))

    ftext = [t for t in fig.texts]
    nb = block([t for t in ftext if t.get_text() in NOTE_B])
    nc = block([t for t in ftext if t.get_text() in NOTE_C])
    nrow = block([t for t in ftext if t.get_text().startswith("n = ")])
    lets = block([t for t in ftext if t.get_text() in ("b", "c")])
    lega = block([fig.legends[0]])
    legc_bb = block([fig.legends[1]])
    xlb = block([axB.xaxis.get_label()])
    xlc = block([axC.xaxis.get_label()])
    print("-- vertical clearance between blocks (cm)")
    for name, lo, hi in [("legend a top -> canvas", lega[3], H),
                         ("axA top ink -> legend a", block([axA.yaxis.get_ticklabels()[-1]])[3], lega[1]),
                         ("axA xticklabels -> n row", nrow[3], block(list(axA.xaxis.get_ticklabels()))[1]),
                         ("n row -> panel letters b, c", lets[3], nrow[1]),
                         ("axB xlabel -> note b", nb[3], xlb[1]),
                         ("axC xlabel -> legend c", legc_bb[3], xlc[1]),
                         ("legend c -> note c", nc[3], legc_bb[1]),
                         ("note b bottom -> canvas", 0.0, nb[1]),
                         ("note c bottom -> canvas", 0.0, nc[1])]:
        print(f"   {name:34s} {hi - lo:6.3f}")
    print("-- horizontal gap between neighbouring n labels (cm)")
    nlab = sorted([bb_cm(t) for t in ftext if t.get_text().startswith("n = ")])
    for u, v in zip(nlab, nlab[1:]):
        print(f"   {v[0] - u[2]:6.3f}")
    print("-- horizontal: note/legend right edge vs canvas (cm)")
    for name, b in [("note b", nb), ("note c", nc), ("n row", nrow),
                    ("legend a", lega), ("legend c", legc_bb)]:
        print(f"   {name:34s} left {b[0]:6.3f}  right {b[2]:6.3f}  "
              f"margin {W - b[2]:6.3f}")
