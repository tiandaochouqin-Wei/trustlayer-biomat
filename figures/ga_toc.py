# -*- coding: utf-8 -*-
"""Table-of-contents (graphical abstract) figure for the BMEMat revision.

Replaces the submitted graphical abstract, which used a red/amber/green
traffic-light triad and the claim words "deployment-grade", "guaranteed
coverage" and "not certify".  None of that survives here.

The argument in one picture:
    any model -> a prediction -> the trust layer (four numbered steps)
    -> one of four decisions (style.DECISION glyphs + word labels)
    -> underneath, the headline empirical result as three marks on one
       coverage scale against the nominal 0.90.

Pure vector: every element is a matplotlib patch, line or text object.  No
icons, no photographs, no embedded raster.  Drawn at the final printed size
(55 x 50 mm, plus a 110 x 20 mm banner variant); nothing is ever downscaled.
All text is Arial at >= 7 pt at that printed size.  Okabe-Ito colours only,
with redundant shape encoding and a word label on every coloured element, so
no comparison depends on colour.

The data strip is read from revision/results/R1_core.json and checked against
the values quoted in the manuscript before drawing.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import style  # noqa: E402

style.apply()
# The figures are drawn at their exact printed size; style.save() uses
# bbox_inches="tight", so the default 0.1 in padding would inflate a 20 mm
# banner by 25%.  Drop it and let a full-size background patch define the box.
plt.rcParams["savefig.pad_inches"] = 0.0

CM = style.CM

# ------------------------------------------------------------------ colours --
GREEN = style.ROLE["recal"]        # bluish green  - trust layer / recalibrated
BLUE = style.ROLE["source"]        # blue          - source / in distribution
VERM = style.ROLE["fail"]          # vermillion    - under-coverage
GREY = style.OI["grey"]
TRACK = "#DCDCDC"
INK = "#1A1A1A"
BODY = "#333333"
EDGE = "#4D4D4D"
BAND_FILL = "#E4F4EF"
DISC = "#00654A"

F = 7.0                            # body text  (floor for the whole figure)
FB = 7.5                           # small bold headings

STEPS = [("1", "calibrate"),
         ("2", "check the\nshift"),
         ("3", "recalibrate\nor refuse"),
         ("4", "stratify")]
DECISIONS = ["accept", "reject", "measure", "not certified"]

# ------------------------------------------------------------ the data strip --
R1 = json.load(open(os.path.join(REV, "results", "R1_core.json")))
_id = [v["id_conformal"]["coverage"]["mean"] for v in R1.values()]
_sh = [v["shift_source"]["coverage"]["mean"] for v in R1.values()]
_rc = [v["shift_recal"]["coverage"]["mean"] for v in R1.values()]
_m = [v["m"] for v in R1.values()]
ID_LO, ID_HI = min(_id), max(_id)
SH_LO, SH_HI = min(_sh), max(_sh)
RC_LO, RC_HI = min(_rc), max(_rc)
M_LO, M_HI = min(_m), max(_m)
NTASK = len(R1)

# what the manuscript claims, re-derived here so the figure cannot drift from it
assert NTASK == 6, NTASK
assert round(ID_LO, 2) == round(ID_HI, 2) == 0.90, (ID_LO, ID_HI)
assert (round(SH_LO, 2), round(SH_HI, 2)) == (0.15, 0.76), (SH_LO, SH_HI)
assert (round(RC_LO, 2), round(RC_HI, 2)) == (0.90, 0.95), (RC_LO, RC_HI)
assert (M_LO, M_HI) == (13, 30), (M_LO, M_HI)

TXT_SHIFT = "%.2f–%.2f" % (SH_LO, SH_HI)
TXT_RECAL = "%.2f–%.2f" % (RC_LO, RC_HI)
TXT_M = "%d–%d" % (M_LO, M_HI)
ID_AT = sum(_id) / len(_id)

# label, colour, marker, (lo, hi) coverage, printed value, where to put it
ROWS = [("in distribution", BLUE, style.MARK["source"], (ID_AT, ID_AT), None, None),
        ("design region", VERM, style.MARK["fail"], (SH_LO, SH_HI), TXT_SHIFT,
         "center"),
        ("after %s target\nmeasurements" % TXT_M, GREEN, style.MARK["recal"],
         (RC_LO, RC_HI), TXT_RECAL, "inline")]


# ------------------------------------------------------------- primitives ----
class Canvas(object):
    """A figure whose axes fill it exactly, with data units in centimetres and
    a top-down drawing coordinate (x, d) inset from the paper edge by M."""

    def __init__(self, w_cm, h_cm, margin):
        self.W, self.H, self.M = w_cm, h_cm, margin
        self.fig = plt.figure(figsize=(w_cm * CM, h_cm * CM))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, w_cm)
        self.ax.set_ylim(0, h_cm)
        self.ax.set_axis_off()
        # defines the tight bbox, so the saved file is exactly w_cm x h_cm
        self.ax.add_patch(Rectangle((0, 0), w_cm, h_cm, fc="white", ec="none",
                                    zorder=-100))
        self.fig.patch.set_facecolor("white")

    # coordinate helpers -------------------------------------------------
    def X(self, x):
        return self.M + x

    def Y(self, d):
        return self.H - self.M - d

    # drawing helpers ----------------------------------------------------
    def box(self, x, d, w, h, fc="white", ec=EDGE, lw=0.8, r=0.09, z=2):
        p = FancyBboxPatch((self.X(x), self.Y(d) - h), w, h,
                           boxstyle="round,pad=0,rounding_size=%g" % r,
                           fc=fc, ec=ec, lw=lw, zorder=z, mutation_aspect=1.0)
        self.ax.add_patch(p)
        return p

    def txt(self, x, d, s, size=F, weight="normal", color=INK, ha="center",
            va="top", z=6, lsp=1.06, clip=False):
        t = self.ax.text(self.X(x), self.Y(d), s, fontsize=size,
                         fontweight=weight, color=color, ha=ha, va=va,
                         zorder=z, linespacing=lsp, family="sans-serif")
        t.set_clip_on(clip)
        return t

    def line(self, x0, d0, x1, d1, color=EDGE, lw=0.8, ls="-", z=3, cap="butt"):
        ln = Line2D([self.X(x0), self.X(x1)], [self.Y(d0), self.Y(d1)],
                    color=color, lw=lw, ls=ls, zorder=z, solid_capstyle=cap)
        self.ax.add_line(ln)
        return ln

    def arrow(self, x0, d0, x1, d1, color=EDGE, lw=0.9, ms=4.6, z=5):
        a = FancyArrowPatch((self.X(x0), self.Y(d0)), (self.X(x1), self.Y(d1)),
                            arrowstyle="-|>", mutation_scale=ms, lw=lw,
                            color=color, shrinkA=0, shrinkB=0,
                            joinstyle="miter", zorder=z)
        self.ax.add_patch(a)
        return a

    def dot(self, x, d, marker="o", color=INK, ms=4.0, filled=True, mew=0.9,
            z=6):
        ln = Line2D([self.X(x)], [self.Y(d)], marker=marker, ms=ms,
                    markerfacecolor=color if filled else "white",
                    markeredgecolor=color, markeredgewidth=mew, ls="none",
                    zorder=z)
        ln.set_clip_on(False)
        self.ax.add_line(ln)
        return ln

    def disc(self, x, d, num, r=0.145):
        self.ax.add_patch(Circle((self.X(x), self.Y(d)), r, fc=DISC, ec="none",
                                 zorder=5))
        self.txt(x, d, num, size=F, weight="bold", color="white", ha="center",
                 va="center_baseline", z=7)

    # QA -----------------------------------------------------------------
    def check(self):
        """Report any artist that leaves the margin box, and any text < 7 pt."""
        self.fig.canvas.draw()
        r = self.fig.canvas.get_renderer()
        inv = self.ax.transData.inverted()
        bad = []
        for t in self.ax.texts:
            if t.get_fontsize() < 7.0 - 1e-9:
                bad.append("TEXT %r only %.2f pt" % (t.get_text()[:22],
                                                     t.get_fontsize()))
            bb = t.get_window_extent(renderer=r)
            (x0, y0), (x1, y1) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
            if (x0 < self.M - 0.01 or x1 > self.W - self.M + 0.01
                    or y0 < self.M - 0.01 or y1 > self.H - self.M + 0.01):
                bad.append("OUT  %-24r x %.2f..%.2f  y %.2f..%.2f"
                           % (t.get_text()[:22].replace("\n", " / "),
                              x0 - self.M, x1 - self.M,
                              y0 - self.M, y1 - self.M))
        return bad


# ---------------------------------------------------------- shared blocks ----
def model_column(c, x0, w, d_box, h_box, d_dot, d_lab):
    """'any model' box -> arrow -> the bare point prediction it returns."""
    cx = x0 + w / 2
    c.box(x0, d_box, w, h_box, fc="white", ec=EDGE, lw=0.9)
    c.txt(cx, d_box + h_box / 2, "any\nmodel", size=FB, weight="bold",
          va="center_baseline", lsp=1.14)
    c.arrow(cx, d_box + h_box + 0.02, cx, d_dot - 0.13, lw=0.8)
    c.dot(cx, d_dot, marker="o", color=INK, ms=4.2)
    c.txt(cx, d_lab, "prediction", size=F, color=BODY)
    return cx


def trust_band(c, x0, x1, d0, d1, chip_d, label_x, disc_x, head_d):
    """The trust layer: one narrow band, four tiny numbered steps."""
    c.box(x0, d0, x1 - x0, d1 - d0, fc=BAND_FILL, ec=GREEN, lw=0.9, r=0.10, z=2)
    c.txt((x0 + x1) / 2, head_d, "trust layer", size=FB, weight="bold",
          color=DISC)
    for (num, lab), d in zip(STEPS, chip_d):
        c.disc(disc_x, d, num)
        c.txt(label_x, d, lab, size=F, ha="left", va="center_baseline",
              color=INK, lsp=1.10)


def decision_column(c, rail_x, glyph_x, word_x, rows, head_d, wrap=False):
    """The four decisions: style.DECISION glyph, then the word."""
    c.txt(rail_x, head_d, "decision", size=FB, weight="bold", ha="left",
          color=EDGE)
    c.line(rail_x, rows[0], rail_x, rows[-1], color=GREY, lw=0.7, z=3)
    for name, d in zip(DECISIONS, rows):
        g = style.DECISION[name]
        c.line(rail_x, d, glyph_x - 0.09, d, color=GREY, lw=0.7, z=3)
        c.dot(glyph_x, d, marker=g["marker"], color=g["color"], ms=g["ms"],
              filled=g["filled"], mew=1.0)
        lab = name.replace(" ", "\n") if (wrap and " " in name) else name
        c.txt(word_x, d, lab, size=F, ha="left", va="center_baseline",
              color=INK, lsp=1.10)


def data_strip(c, x_lab, x0, x1, rows_d, head_d, val_dy, head_left,
               header="interval coverage", force_right=False, nominal=0.90):
    """Three marks on one coverage scale (0 to 1) against the nominal 0.90."""
    def cx(v):
        return x0 + v * (x1 - x0)

    c.txt(head_left, head_d, header, size=F, weight="bold",
          ha="left", color=EDGE)
    c.txt(x1, head_d, "nominal %.2f" % nominal, size=F, ha="right", color=BODY)
    c.line(cx(nominal), rows_d[0] - 0.26, cx(nominal), rows_d[-1] + 0.20,
           color=INK, lw=0.7, ls=(0, (3.0, 2.2)), z=4)
    for (lab, col, mk, (lo, hi), val, where), d in zip(ROWS, rows_d):
        c.line(x0, d, x1, d, color=TRACK, lw=1.0, z=1, cap="round")
        c.txt(x_lab, d, lab, size=F, ha="right", va="center_baseline",
              color=BODY, lsp=1.10)
        if cx(hi) - cx(lo) > 0.28:
            c.line(cx(lo), d, cx(hi), d, color=col, lw=1.3, z=5, cap="butt")
            c.dot(cx(lo), d, marker=mk, color=col, ms=3.4)
            c.dot(cx(hi), d, marker=mk, color=col, ms=3.4)
        else:
            c.line(cx(lo), d, cx(hi), d, color=col, lw=1.3, z=5, cap="butt")
            c.dot(cx((lo + hi) / 2), d, marker=mk, color=col, ms=3.8)
        if val:
            if force_right:                      # banner: above the bar, flush
                t = c.txt(x1, d - val_dy, val, size=F, ha="right",
                          va="baseline", color=col, z=7)
                t.set_bbox(dict(facecolor="white", edgecolor="none", pad=0.6))
            elif where == "inline":              # on the row, left of a short bar
                t = c.txt(cx(lo) - 0.10, d, val, size=F, ha="right",
                          va="center_baseline", color=col, z=7)
                t.set_bbox(dict(facecolor="white", edgecolor="none", pad=0.8))
            else:                                # centred above a long bar
                c.txt((cx(lo) + cx(hi)) / 2, d - val_dy, val, size=F,
                      ha="center", va="baseline", color=col)


# ============================================================ 55 x 50 mm =====
def build_square():
    c = Canvas(5.5, 5.0, 0.13)                 # drawing area 5.24 x 4.74 cm

    # left: any model -> a prediction
    cx = model_column(c, 0.04, 1.18, d_box=0.66, h_box=0.74, d_dot=1.84,
                      d_lab=1.96)
    c.arrow(1.28, 1.84, 1.46, 1.84, lw=0.9)

    # middle: the trust layer, one narrow band, four numbered steps
    chip_d = [0.8225, 1.3875, 1.9525, 2.5175]
    trust_band(c, 1.50, 3.40, 0.10, 2.86, chip_d, label_x=1.99, disc_x=1.785,
               head_d=0.20)

    # right: the four decisions
    c.arrow(3.42, 1.48, 3.56, 1.48, lw=0.9)
    decision_column(c, rail_x=3.58, glyph_x=3.74, word_x=3.90,
                    rows=[0.70, 1.22, 1.74, 2.26], head_d=0.20)

    # underneath: the headline result
    c.line(0.0, 2.97, 5.24, 2.97, color="#E4E4E4", lw=0.7, z=0)
    data_strip(c, x_lab=1.98, x0=2.10, x1=5.24, rows_d=[3.56, 4.00, 4.44],
               head_d=3.06, val_dy=0.12, head_left=0.0, force_right=True)
    return c


# =========================================================== 110 x 20 mm =====
def build_broad():
    c = Canvas(11.0, 2.0, 0.10)                # drawing area 10.80 x 1.80 cm

    cx = model_column(c, 0.00, 1.08, d_box=0.30, h_box=0.62, d_dot=1.22,
                      d_lab=1.34)
    c.arrow(1.10, 1.22, 1.22, 1.22, lw=0.9)

    # the band runs horizontally here: number above, words below
    c.box(1.24, 0.30, 5.01, 1.12, fc=BAND_FILL, ec=GREEN, lw=0.9, r=0.10, z=2)
    c.txt(3.745, 0.00, "trust layer", size=FB, weight="bold", color=DISC)
    for i, (num, lab) in enumerate(STEPS):
        chx = 1.32 + 1.2125 * (i + 0.5)
        c.disc(chx, 0.52, num)
        c.txt(chx, 0.72, lab, size=F, va="top", color=INK, lsp=1.12)

    c.arrow(6.27, 0.86, 6.41, 0.86, lw=0.9)
    decision_column(c, rail_x=6.43, glyph_x=6.59, word_x=6.75,
                    rows=[0.38, 0.70, 1.02, 1.46], head_d=0.00, wrap=True)

    c.line(7.66, 0.02, 7.66, 1.78, color="#E4E4E4", lw=0.7, z=0)
    data_strip(c, x_lab=9.68, x0=9.80, x1=10.80, rows_d=[0.58, 1.04, 1.50],
               head_d=0.00, val_dy=0.12, head_left=7.74, header="coverage",
               force_right=True)
    return c


# ------------------------------------------------------------------ output ---
if __name__ == "__main__":
    print("R1_core.json, %d tasks:" % NTASK)
    print("  in distribution      %.3f-%.3f  -> nominal 0.90" % (ID_LO, ID_HI))
    print("  design region        %.3f-%.3f  -> %s" % (SH_LO, SH_HI, TXT_SHIFT))
    print("  after recalibration  %.3f-%.3f  -> %s" % (RC_LO, RC_HI, TXT_RECAL))
    print("  budget m             %d-%d" % (M_LO, M_HI))

    for builder, stem, number in [(build_square, "ga_toc", "TOC"),
                                  (build_broad, "ga_toc_broad", "TOCbroad")]:
        c = builder()
        bad = c.check()
        png = style.save(c.fig, stem, number)
        print("\n%s  (%.1f x %.1f mm)" % (stem, c.W * 10, c.H * 10))
        for b in bad:
            print("   !! " + b)
        if not bad:
            print("   all text >= 7 pt and inside the %.2f cm margin" % c.M)
        print("   " + png)
        plt.close(c.fig)
