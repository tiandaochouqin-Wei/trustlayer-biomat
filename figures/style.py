"""One figure style for every revised figure (Reviewer 1 comment 4; Editor comment 1).

* Arial throughout; sizes are set for the PRINTED size (figures are drawn at their
  final width, never scaled down): >= 7 pt for all text, 8 pt axis labels,
  10 pt bold panel letters.
* Okabe–Ito colour-blind-safe palette with fixed semantic roles, plus redundant
  marker/line-style encoding so that no comparison relies on colour alone.
* Panel labels: bold lowercase a, b, c (no parentheses) at the top-left corner;
  sub-panels i, ii, iii (bold) via ``sublabel``.
* Export: vector PDF (TrueType fonts embedded), 600-dpi LZW TIFF for submission
  (Figure_n.tif) and 300-dpi PNG for the Word manuscript.
"""
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

CM = 1 / 2.54
WIDTH_1COL = 8.5 * CM      # BMEMat single column
WIDTH_15COL = 12.0 * CM
WIDTH_2COL = 17.5 * CM     # BMEMat double column

# Okabe & Ito (2008) palette
OI = {"black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7",
      "grey": "#999999", "lightgrey": "#DDDDDD"}

# semantic roles used consistently across all figures
ROLE = {
    "source": OI["blue"],          # source-calibrated / in-distribution
    "recal": OI["green"],          # target-recalibrated
    "fail": OI["vermillion"],      # under-coverage, violation, naive
    "heuristic": OI["orange"],     # model heuristic uncertainty
    "other": OI["purple"],
    "neutral": OI["grey"],
    "target": OI["green"],
    "ideal": OI["black"],
}
MARK = {"source": "o", "recal": "s", "fail": "v", "heuristic": "D", "other": "^"}

# ---------------------------------------------------------------- task identity
# Colour/marker/line style that identify a DESIGN REGION (a dataset), used by any
# panel whose series are tasks rather than calibration states (Fig. 4d, Fig. 9a,c).
# The four reserved ROLE hues (blue, bluish green, vermillion, orange) are kept
# free here on purpose, so that a reader never carries the source-calibrated /
# target-recalibrated / failure / heuristic meaning of those colours into a panel
# where colour means "which design region".  Marker and line style repeat the
# same information, so colour is never load-bearing.  One definition, imported by
# every figure that draws the six design regions, so the identities never diverge.
_DARK, _MID = "#4D4D4D", "#6E6E6E"
TASKSTYLE = {  # key: (colour, marker, line style, marker face)
    "glass_Tg":    ("black",        "o", "-",                           "black"),
    "glass_E":     (OI["skyblue"],  "^", (0, (4, 1.6)),                 OI["skyblue"]),
    "glass_HV":    (OI["purple"],   "s", (0, (1, 1.4)),                 OI["purple"]),
    "glass_Tliq":  (_DARK,          "D", (0, (5, 1.4, 1, 1.4)),         _DARK),
    "steel_yield": ("black",        "v", (0, (3, 1.2, 1, 1.2, 1, 1.2)), "white"),
    "polymer_Tg":  (_MID,           "P", (0, (6, 1.8)),                 _MID),
}
TASKS = [("glass_Tg", "glass Tg"), ("glass_E", "glass Young's modulus"),
         ("glass_HV", "glass microhardness"), ("glass_Tliq", "glass liquidus"),
         ("steel_yield", "steel yield strength"), ("polymer_Tg", "polymer Tg")]

# Canonical glyphs for the four decisions the trust layer returns (Box 1).
# Deliberately DISJOINT from MARK above -- MARK encodes data series (source,
# target-recalibrated, ...), DECISION encodes the decision a candidate receives,
# so no shape ever carries two meanings across figures.  The single overlap,
# "reject" = vermillion down-triangle, is intentional: it is the fail role.
# Every decision also carries a word label, so colour is never load-bearing.
DECISION = {
    "accept":        {"color": OI["green"],      "marker": "*", "filled": True,  "ms": 6.6},
    "reject":        {"color": OI["vermillion"], "marker": "v", "filled": True,  "ms": 4.6},
    "measure":       {"color": OI["purple"],     "marker": "P", "filled": True,  "ms": 5.0},
    "not certified": {"color": OI["grey"],       "marker": "X", "filled": True,  "ms": 5.0},
}

FIGROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(FIGROOT, "out")
os.makedirs(OUT, exist_ok=True)


def apply():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        # Greek letters must not fall back to the DejaVu mathtext fonts: a
        # $\alpha$ set in DejaVu Sans Oblique next to Arial body text is visible
        # in print and leaves a third font embedded in the PDF.
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "font.size": 7.5,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.1,
        "lines.markersize": 3.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "normal",
        "axes.titlepad": 4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 600,
        "figure.dpi": 150,
    })


def panel(ax, letter, x=-0.02, y=1.0, fig=None):
    """Bold lowercase panel letter at the top-left corner of the axes' bounding box."""
    fig = fig or ax.figure
    bb = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    fig.text(bb.x0 + x, bb.y1 + 0.005, letter, fontsize=10, fontweight="bold",
             ha="left", va="bottom", family="sans-serif")


def sublabel(ax, roman, x=0.02, y=0.98):
    ax.text(x, y, roman, transform=ax.transAxes, fontsize=8, fontweight="bold", ha="left", va="top")


def nominal(ax, level=0.90, orient="h", label=True):
    f = ax.axhline if orient == "h" else ax.axvline
    f(level, color=ROLE["ideal"], lw=0.7, ls=(0, (4, 3)), zorder=0)


def save(fig, stem, number=None, tight=True):
    """Save PDF + 300-dpi PNG (for Word) + 600-dpi TIFF (submission file).

    ``tight=False`` writes the declared canvas size exactly (no ``bbox_inches``
    cropping), so the file is at the final print width to the micrometre and the
    7 pt minimum is really 7 pt on the page. Use it when the layout already
    reserves its own margins.
    """
    bb = "tight" if tight else None
    fig.savefig(os.path.join(OUT, f"{stem}.pdf"), bbox_inches=bb)
    fig.savefig(os.path.join(OUT, f"{stem}.png"), dpi=300, bbox_inches=bb)
    name = f"Figure_{number}" if number is not None else stem
    tif = os.path.join(OUT, f"{name}.tif")
    fig.savefig(tif, dpi=600, bbox_inches=bb,
                pil_kwargs={"compression": "tiff_lzw"})
    # matplotlib writes RGBA; a 4th (alpha) sample is a common pre-flight
    # rejection at journal production even when it is fully opaque. Flatten.
    try:
        from PIL import Image
        with Image.open(tif) as im:
            if im.mode != "RGB":
                im.convert("RGB").save(tif, compression="tiff_lzw", dpi=(600, 600))
    except ImportError:
        pass
    return os.path.join(OUT, f"{stem}.png")
