"""Figure 6 - model-agnosticism across eight model families (R2_models).

Panels (all three are 8 models x 6 datasets grids, so every one of the 48 cells
is individually readable; an earlier dot-plot version of panel c hid points
whose widths differed by < 0.03 target IQR).

  a  source-calibrated coverage on the design-region target-test half
  b  target-recalibrated coverage, same colour scale, one shared colour bar
  c  target-recalibrated interval width, normalised by the target IQR

Data: revision/results/R2_models.json (exp_R2_models.py; 20 splits, alpha = 0.10).
Drawn at the final print size (17.5 cm wide); never scale this figure.
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

import style

style.apply()

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, os.pardir, "results", "R2_models.json")

MODELS = ["RF", "ExtraTrees", "HistGB", "GP", "SVR", "kNN", "Ridge", "MLP"]
# (first row, last row, family label) - families in the order used in the text
GROUPS = [(0, 2, "trees"), (3, 4, "kernel"), (5, 5, "instance"), (6, 6, "linear"), (7, 7, "neural")]
DSETS = ["glass_Tg", "glass_E", "glass_HV", "glass_Tliq", "steel_yield", "polymer_Tg"]
DLAB = ["glass Tg", "glass E", "glass HV", "glass liquidus", "steel yield", "polymer Tg"]

CM = style.CM
W, H = 17.35, 8.65                    # canvas; style.save crops tight and pads 0.1 in ->
                                      # the saved PDF/PNG/TIFF is 17.5 cm wide (BMEMat 2 columns)

# ---------------------------------------------------------------- layout (cm)
AX_TOP = 1.55                         # from the top of the canvas (2-line titles above)
AX_H = 4.25
A_L, PW = 2.45, 4.25                  # left edge of panel a; common panel width
GAP_AB, GAP_BC = 0.34, 1.45           # b has no y labels; c carries its own
B_L = A_L + PW + GAP_AB
C_L = B_L + PW + GAP_BC
CBAR_Y, CBAR_H = 7.40, 0.26           # from the top
BR_X = 1.05                           # family bracket rule

TITLE_FS = 7.5

# coverage colour scale (panels a, b) and width colour scale (panel c)
COV_CMAP = plt.get_cmap("cividis")
COV_LO, COV_HI = 0.0, 1.0
# single-hue sequential ramp anchored on the target-recalibrated role colour
# (Okabe-Ito bluish green): monotone in luminance, so it is colour-blind safe
# and greyscale safe, and it carries the same semantic role as elsewhere.
WID_CMAP = LinearSegmentedColormap.from_list(
    "recal_seq", ["#EAF7F2", "#A5DFCC", "#43BE9B", style.ROLE["recal"], "#00543D"])
WID_LO, WID_HI = 1.5, 6.0


def box(x_cm, y_top_cm, w_cm, h_cm):
    """figure-fraction rectangle from cm measured from the top-left corner."""
    return [x_cm / W, 1.0 - (y_top_cm + h_cm) / H, w_cm / W, h_cm / H]


# ------------------------------------------------------------------ the data
with open(RES, "r", encoding="utf-8") as fh:
    R = json.load(fh)["results"]

src = np.array([[R[m][d]["src_to_tgt"]["coverage"]["mean"] for d in DSETS] for m in MODELS])
rec = np.array([[R[m][d]["recal"]["coverage"]["mean"] for d in DSETS] for m in MODELS])
wid = np.array([[R[m][d]["recal"]["norm_width"]["mean"] for d in DSETS] for m in MODELS])
MVAL = [R["RF"][d]["m"] for d in DSETS]

fig = plt.figure(figsize=(W * CM, H * CM))
axa = fig.add_axes(box(A_L, AX_TOP, PW, AX_H))
axb = fig.add_axes(box(B_L, AX_TOP, PW, AX_H))
axc = fig.add_axes(box(C_L, AX_TOP, PW, AX_H))
cax_ab = fig.add_axes(box(3.07, CBAR_Y, 7.60, CBAR_H))
cax_c = fig.add_axes(box(13.22, CBAR_Y, 3.30, CBAR_H))


def ink(v, cmap, lo, hi):
    """black or white annotation, whichever has the higher contrast on the cell."""
    r, g, b, _ = cmap((v - lo) / (hi - lo))
    lin = [(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4) for c in (r, g, b)]
    lum = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    return "white" if 1.05 / (lum + 0.05) > (lum + 0.05) / 0.05 else "#111111"


def heat(ax, M, cmap, lo, hi, title, ylab):
    im = ax.imshow(M, cmap=cmap, vmin=lo, vmax=hi, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(DSETS)))
    ax.set_xticklabels(DLAB, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(MODELS if ylab else [])
    ax.set_title(title, fontsize=TITLE_FS, linespacing=1.35, pad=3)
    ax.tick_params(length=0, pad=2)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.5)
        s.set_color("#AAAAAA")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color=ink(v, cmap, lo, hi))
    for (_, r1, _) in GROUPS[:-1]:                    # white rules between families
        ax.axhline(r1 + 0.5, color="white", lw=1.2)
    return im


im_cov = heat(axa, src, COV_CMAP, COV_LO, COV_HI,
              "source-calibrated coverage\ndesign region", True)
heat(axb, rec, COV_CMAP, COV_LO, COV_HI,
     "target-recalibrated coverage\ndesign region, m = 13–30", False)
im_wid = heat(axc, wid, WID_CMAP, WID_LO, WID_HI,
              "target-recalibrated width\ndesign region", True)

# ------------------------------------------------------- family row brackets
for (r0, r1, name) in GROUPS:
    y0 = 1.0 - (AX_TOP + AX_H * (r0 / len(MODELS))) / H
    y1 = 1.0 - (AX_TOP + AX_H * ((r1 + 1) / len(MODELS))) / H
    fig.add_artist(Line2D([BR_X / W, BR_X / W], [y0 - 0.008, y1 + 0.008],
                          color=style.OI["grey"], lw=0.7))
    fig.text((BR_X - 0.10) / W, 0.5 * (y0 + y1), name, fontsize=7, color="#444444",
             ha="right", va="center")

# --------------------------------------------------------------- colour bars
cb = fig.colorbar(im_cov, cax=cax_ab, orientation="horizontal")
cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
cb.set_label("coverage (nominal 0.90)", fontsize=TITLE_FS, labelpad=2)
cb.outline.set_linewidth(0.5)
cb.outline.set_edgecolor("#AAAAAA")
cax_ab.tick_params(length=2, width=0.6, pad=2)
cax_ab.plot([0.90, 0.90], [0, 1], transform=cax_ab.get_xaxis_transform(),
            color="black", lw=1.0, ls=(0, (2, 1.4)), clip_on=False)
cax_ab.annotate("nominal 0.90", xy=(0.90, 1.0), xycoords=cax_ab.get_xaxis_transform(),
                xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                fontsize=7)

cb2 = fig.colorbar(im_wid, cax=cax_c, orientation="horizontal")
cb2.set_ticks([2, 3, 4, 5, 6])
cb2.set_label("normalised width (× target IQR)", fontsize=TITLE_FS, labelpad=2)
cb2.outline.set_linewidth(0.5)
cb2.outline.set_edgecolor("#AAAAAA")
cax_c.tick_params(length=2, width=0.6, pad=2)

# ------------------------------------------------------------ panel letters
fig.canvas.draw()
rend = fig.canvas.get_renderer()
# a and c sit left of their model tick labels; b has none, so its tight bbox
# starts at the rotated "glass Tg" label - nudge b right so that the letter is
# not mistaken for part of panel a (its title ends 0.8 cm to the left).
for ax, letter, dx in ((axa, "a", -0.02), (axb, "b", 0.0075), (axc, "c", 0.0)):
    style.panel(ax, letter, x=dx)

# ------------------------------------------------------------------ QA check
fig.canvas.draw()
inv = fig.transFigure.inverted()


def extent_cm(artist):
    bb = artist.get_window_extent(rend).transformed(inv)
    return (bb.x0 * W, bb.x1 * W, (1 - bb.y1) * H, (1 - bb.y0) * H)


for ax, nm in ((axa, "a"), (axb, "b"), (axc, "c")):
    x0, x1, yt, yb = extent_cm(ax.title)
    print(f"title {nm}: x {x0:.2f}-{x1:.2f} cm, y {yt:.2f}-{yb:.2f} cm")
for t in fig.texts:
    if t.get_text() in ("a", "b", "c") and t.get_fontweight() == "bold":
        x0, x1, yt, yb = extent_cm(t)
        print(f"letter {t.get_text()}: x {x0:.2f}-{x1:.2f} cm, y {yt:.2f}-{yb:.2f} cm")

png = style.save(fig, "fig6_models", 6)

# Flatten the submission TIFF to 3-channel RGB (matplotlib writes RGBA; a
# constant alpha channel can trip publisher preflight and inflates the file).
tif = os.path.join(style.OUT, "Figure_6.tif")
try:
    from PIL import Image
    with Image.open(tif) as raw:
        mode, size = raw.mode, raw.size
        if mode == "RGBA":
            flat = Image.new("RGB", size, "white")
            flat.paste(raw, mask=raw.split()[3])
        else:
            flat = raw.convert("RGB")
    flat.save(tif, compression="tiff_lzw", dpi=(600, 600))
    with Image.open(tif) as chk:
        print("TIFF:", chk.mode, chk.size, chk.info.get("dpi"))
except Exception as exc:                                  # pragma: no cover
    print("TIFF flatten skipped:", exc)

print(png)
print("tight bbox (cm):", np.round(np.array(
    fig.get_tightbbox(rend).size) * 2.54, 2), "-> saved +0.51 cm pad")
print("source-calibrated range", round(src.min(), 2), round(src.max(), 2))
print("recalibrated range", round(rec.min(), 2), round(rec.max(), 2))
print("width range", round(wid.min(), 2), round(wid.max(), 2))
print("m per dataset", dict(zip(DLAB, MVAL)))
