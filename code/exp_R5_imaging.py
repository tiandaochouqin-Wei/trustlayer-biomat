"""R5 imaging: split-conformal classification on PathMNIST with an independently acquired cohort.

Answers Reviewer 1 comment 2 (the collapse to 0.77 in the submitted Figure 9 occurred only
after a synthetic colour perturbation) and Reviewer 2 comment 8 (the "real distribution
shift" of Figure 9 must be interpreted cautiously).

Wording: "genuine" is used here only in the sense of "not synthetic". T2 below is an
independently acquired, non-colour-normalised cohort (different acquisition period, different
tile geometry) from the same pathology archive that contributed to the source cohort; it is
NOT a proven different institution (see Caveats in the .md).

Data
  source   PathMNIST 28x28 (MedMNIST v2, Yang et al. 2023 Sci. Data), derived from
           NCT-CRC-HE-100K: train 89,996 / val 10,004 (random 9:1 split of one cohort,
           so train and val share patients).
  T1       PathMNIST test = CRC-VAL-HE-7K (7,180 tiles, patients disjoint from 100K).
  T2       Kather et al. 2016 (Sci. Rep. 6:27988) texture tiles: 5,000 tiles, 150x150 px
           (0.495 um/px, i.e. 74 um field of view vs 112 um for the 224 px NCT tiles),
           10 H&E slides (CRC-Prim-HE-01..10), different acquisition period, NOT colour
           normalised. Resized to 28x28 RGB uint8 with PIL bicubic (antialiased; MedMNIST v2
           states "cubic spline interpolation" without naming the library). A non-antialiased
           order-3 spline resize (scipy.ndimage.zoom) is run as a sensitivity analysis.
           Label map (verified against medmnist.INFO['pathmnist']['label']):
             01_TUMOR->TUM, 04_LYMPHO->LYM, 06_MUCOSA->NORM, 07_ADIPOSE->ADI, 08_EMPTY->BACK
             (main analysis, 5 unambiguous classes, 3,125 tiles);
             02_STROMA->{STR,MUS}, 05_DEBRIS->{DEB,MUC} (set-valued sensitivity: covered if the
             prediction set contains any admissible label, correct if argmax is admissible);
             03_COMPLEX excluded (no PathMNIST counterpart).
  STRESS   the submitted per-channel scale (1.02, 0.99, 1.00) applied to T1 images -- kept
           ONLY as a secondary, clearly flagged synthetic stress test.

Models (trained here on the GPU, full train split, last-epoch weights, no model selection on
any held-out data; settings fixed a priori and not iterated):
  SmallCNN      the submitted architecture (3 conv + 2 fc), flips only;
  ResNet18      torchvision ResNet-18 with a 3x3 stride-1 stem and no max-pool, flips only;
  SmallCNN-aug       }  the same architectures trained with stain-colour augmentation (HED
  ResNet18-aug       }  jitter + brightness/contrast jitter, Tellez et al. 2019 style), which is
  ResNet18-augstrong }  standard practice in computational pathology, at two strengths; the
                        flips-only arms are therefore a deliberate worst case, not a recommended
                        pipeline.
  3 training seeds each. The submitted cached SmallCNN (20k-image subsample, 6 CPU epochs,
  demo/exp2_model_cache.pt, read-only) is re-evaluated as an audit of the original analysis.

Conformal protocol (revision/code/PROTOCOL.md), per split seed s = 0..19 and per model:
  source: seeded permutation of val -> cal half (calibration) / source-test half.
  target: seeded permutation of the target -> recal pool (first floor(n/2)) / target-test;
          recalibration with the first k in {0,20,30,50,100,200,400} labelled pool images
          (k = 0: source-calibrated; k = 30 = protocol headline m = clip(floor(|T|/3),10,30);
          k = 200 = value used in the submitted manuscript); every k is evaluated on the same
          target-test half. Target-test labels are used only for evaluation.
  scores: LAC 1 - p_y (Sadinle et al. 2019); APS (Romano et al. 2020), randomised version
          (u ~ U(0,1) per image, empty sets allowed, which keeps exact finite-sample
          validity). Set-valued labels: calibration score = min over admissible labels.
  alpha = 0.10; finite-sample quantile common.conformal_q.
  metrics: coverage, mean set size, singleton rate, empty-set rate, abstention rate
          (= non-singleton = empty or |set|>1; the literal |set|>1 rate is also reported as
          abstention_gt1), class-conditional coverage (min / macro over classes with >= 10 test
          images, plus the min restricted to the 5 classes shared by all domains so that the
          worst-class numbers are comparable across domains), accuracy, ECE (15 equal-width
          bins, max softmax), selective accuracy on the 50 % most confident (smallest NON-EMPTY
          set, ties by larger max softmax; an empty set means no class reached the threshold and
          is ranked least confident).
  NOTE: split conformal is marginal. Neither source calibration nor target recalibration
  targets class-conditional coverage, so worst-class coverage is below nominal in every domain
  including the source itself; it is reported for all domains side by side, never as a
  T2-specific finding.
Sensitivities on T2: set-valued 7 classes; slide-grouped recal/test halves (no slide on both
  sides; recal pool shuffled) on the 5 and on the 4 tissue classes; class-balanced slide-grouped
  halves on the 4 tissue classes
  (EMPTY excluded: its tiles come from only 2 slides, so no slide-disjoint split of the
  5-class set can be class-balanced); EMPTY excluded (4 classes: Kather EMPTY is white glass
  whereas the PathMNIST BACK class is dark Macenko-normalised background); non-antialiased
  spline resize; physical-scale-matched resize (150 px -> 19 px at ~4 um/px, reflect-padded to
  28, removing the 1.5x field-of-view mismatch); approximate Macenko stain harmonisation
  towards a PathMNIST-derived reference (EMPTY tiles cannot be normalised); scale-matched AND
  Macenko-harmonised.
Controls: macro coverage over the 5 classes shared by all domains (label-shift control);
  paired per-split coverage change versus the source-test half; PathMNIST val magnified 1.5x
  (the reverse of the field-of-view mismatch) as an accuracy diagnostic.
Aggregation: mean and 2.5-97.5 percentile over the 20 splits x 3 training seeds (60 values)
per architecture; per-training-seed summaries over 20 splits are also stored. The splits
resample one finite dataset per cohort, so the intervals describe split-to-split (and
training-seed) variability, not population sampling error.

Usage:  python -W ignore exp_R5_imaging.py [train|analyze|all]   (default all)
Outputs: results/R5_imaging.json, results/R5_imaging.md, results/splits/R5_imaging_splits.json,
         results/models/R5_<arch>_s<t>.pt (+ _probs.npz, _probs_extra.npz, _train.json),
         results/R5_data/kather2016_28x28.npz (processed external cohort),
         results/R5_data/kather2016_28x28_macenko.npz (harmonised sensitivity copy) and
         results/R5_data/kather2016_scalematched.npz (field-of-view sensitivity).
Runtime (RTX 4000 Ada): training ~30 min for 15 models, analysis ~1 min. GPU training is
seeded and cuDNN-deterministic; an independent retrain of SmallCNN seed 0 and ResNet18 seed 0
on this machine reproduced bit-identical softmax arrays, and the analysis is deterministic
given the saved arrays.
"""
import json
import os
import re
import sys
import time

import numpy as np
from joblib import Parallel, delayed

from common import (ALPHA, SEEDS, RES, DATA, BMEAT, SPLITS, conformal_q, summarize, check_disjoint,
                    beta_coverage_quantiles, dump)

TAG = "R5_imaging"
MODELS = os.path.join(RES, "models")
RDATA = os.path.join(RES, "R5_data")
for _d in (MODELS, RDATA):
    os.makedirs(_d, exist_ok=True)
KATHER_DIR = os.path.join(DATA, "kather2016", "Kather_texture_2016_image_tiles_5000")
SUBMITTED_CKPT = os.path.join(BMEAT, "demo", "exp2_model_cache.pt")

ARCHS = ["SmallCNN", "ResNet18", "SmallCNN-aug", "ResNet18-aug", "ResNet18-augstrong"]
TRAIN_SEEDS = [0, 1, 2]
EPOCHS, BATCH, LR, WD = 15, 256, 1e-3, 1e-4
# Stain-colour augmentation, fixed a priori (Tellez et al. 2019, Med. Image Anal., HED-jitter
# family), per image: each optical-density stain channel is scaled by U(1 +- a) and shifted by
# U(+- b), then brightness U(+- c) and contrast U(1 +- d) on the RGB image. Two strengths are
# trained so that the conclusion does not depend on one arbitrary augmentation magnitude.
AUG_CFG = {"aug": (0.20, 0.10, 0.10, 0.25), "augstrong": (0.40, 0.20, 0.20, 0.40)}
RGB_FROM_HED = [[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]]   # Ruifrok & Johnston 1999


def aug_cfg(arch):
    """Colour-augmentation setting of an architecture tag ('ResNet18-aug' -> AUG_CFG['aug'])."""
    return AUG_CFG.get(arch.split("-")[-1]) if "-" in arch else None
KGRID = [0, 20, 30, 50, 100, 200, 400]
K_HEAD = 30                     # protocol headline m (all targets have |T| >= 90)
K_SUBMITTED = 200               # recalibration size used in the submitted manuscript
STRESS_SCALE = (1.02, 0.99, 1.00)
METHODS = ["LAC", "APS"]
N_JOBS = 6                      # shared machine: joblib n_jobs <= 6
ACC_KEYS = ["val", "test_T1", "T2_Kather2016_5class", "T2sens_Kather2016_4class_noEMPTY",
            "T2sens_Kather2016_5class_scalematched", "T2sens_Kather2016_5class_macenko",
            "T2sens_Kather2016_5class_scalematched_macenko", "T2sens_Kather2016_7class_setvalued",
            "T2sens_Kather2016_5class_spline3resize", "val_macenko_diagnostic",
            "val_zoom1.5x_diagnostic", "stress_T1"]
ACC_HEAD = ["val (source)", "T1 CRC-VAL-HE-7K", "T2 5-class", "T2 4-class (no EMPTY)", "T2 scale-matched",
            "T2 Macenko", "T2 scale+Macenko", "T2 7-class set-valued", "T2 spline resize",
            "val Macenko (diag.)", "val 1.5x zoom (diag.)", "STRESS T1 colour"]

CODES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
INFO_NAMES = {"adipose": "ADI", "background": "BACK", "debris": "DEB", "lymphocytes": "LYM",
              "mucus": "MUC", "smooth muscle": "MUS", "normal colon mucosa": "NORM",
              "cancer-associated stroma": "STR", "colorectal adenocarcinoma epithelium": "TUM"}
KATHER_MAP = {"01_TUMOR": ("TUM",), "02_STROMA": ("STR", "MUS"), "03_COMPLEX": None,
              "04_LYMPHO": ("LYM",), "05_DEBRIS": ("DEB", "MUC"), "06_MUCOSA": ("NORM",),
              "07_ADIPOSE": ("ADI",), "08_EMPTY": ("BACK",)}
SHARED5 = ["ADI", "BACK", "LYM", "NORM", "TUM"]
SHARED4 = ["ADI", "LYM", "NORM", "TUM"]          # the same without BACK (Kather EMPTY)


# ======================================================================= data
def verify_label_order():
    import medmnist
    lab = medmnist.INFO["pathmnist"]["label"]
    codes = [INFO_NAMES[lab[str(i)]] for i in range(len(lab))]
    assert codes == CODES, codes
    return {str(i): lab[str(i)] for i in range(len(lab))}


def load_pathmnist():
    from medmnist import PathMNIST
    out = {}
    for sp in ("train", "val", "test"):
        ds = PathMNIST(split=sp, size=28, download=False)
        out[sp] = (np.asarray(ds.imgs, np.uint8), np.asarray(ds.labels).reshape(-1).astype(np.int64))
    return out


def load_kather():
    """5,000 Kather-2016 tiles -> 28x28 RGB uint8 (PIL bicubic, antialiased) plus a
    non-antialiased order-3 spline variant; cached in results/R5_data."""
    path = os.path.join(RDATA, "kather2016_28x28.npz")
    if os.path.exists(path):
        d = np.load(path, allow_pickle=False)
        return {k: d[k] for k in d.files}
    from PIL import Image
    from scipy import ndimage
    files, folder, slide, bic, spl = [], [], [], [], []
    for fo in sorted(os.listdir(KATHER_DIR)):
        for f in sorted(os.listdir(os.path.join(KATHER_DIR, fo))):
            im = Image.open(os.path.join(KATHER_DIR, fo, f)).convert("RGB")
            assert im.size == (150, 150), (fo, f, im.size)
            bic.append(np.asarray(im.resize((28, 28), Image.Resampling.BICUBIC), dtype=np.uint8))
            z = ndimage.zoom(np.asarray(im, np.float64), (28 / 150, 28 / 150, 1), order=3,
                             mode="reflect", grid_mode=True)
            spl.append(np.clip(np.rint(z), 0, 255).astype(np.uint8))
            files.append(f); folder.append(fo)
            slide.append(re.search(r"(CRC-Prim-HE-\d+)", f).group(1))
    out = {"files": np.array(files), "folder": np.array(folder), "slide": np.array(slide),
           "imgs_bicubic": np.stack(bic), "imgs_spline3": np.stack(spl)}
    np.savez_compressed(path, **out)
    return out


MAC_IO, MAC_BETA, MAC_ALPHA, MAC_MIN_PIX = 240.0, 0.15, 1.0, 100


def _od(I):
    return -np.log((I.reshape(-1, 3).astype(np.float64) + 1) / MAC_IO)


def _macenko_fit(OD):
    """Macenko et al. (2009) stain matrix (H first) and 99th-percentile concentrations;
    None if fewer than MAC_MIN_PIX tissue pixels (OD >= beta in every channel)."""
    ODh = OD[~np.any(OD < MAC_BETA, axis=1)]
    if len(ODh) < MAC_MIN_PIX:
        return None
    _, V = np.linalg.eigh(np.cov(ODh.T))
    T = ODh @ V[:, 1:3]
    phi = np.arctan2(T[:, 1], T[:, 0])
    a, b = np.percentile(phi, MAC_ALPHA), np.percentile(phi, 100 - MAC_ALPHA)
    v1 = V[:, 1:3] @ np.array([np.cos(a), np.sin(a)]); v2 = V[:, 1:3] @ np.array([np.cos(b), np.sin(b)])
    HE = np.stack([v1, v2], 1) if v1[0] > v2[0] else np.stack([v2, v1], 1)
    C = np.linalg.lstsq(HE, OD.T, rcond=None)[0]
    return HE, np.percentile(C, 99, axis=1)


def _macenko_apply(I, ref):
    OD = _od(I); f = _macenko_fit(OD)
    if f is None:
        return I, False
    HE, maxC = f
    C = np.linalg.lstsq(HE, OD.T, rcond=None)[0] * (ref[1] / maxC)[:, None]
    return np.clip(MAC_IO * np.exp(-(ref[0] @ C)).T, 0, 255).reshape(I.shape).astype(np.uint8), True


def load_kather_macenko(pm):
    """SENSITIVITY (pipeline harmonisation): Macenko stain normalisation of every Kather tile at
    its native 150x150 resolution towards a reference stain matrix estimated from 3,000 random
    non-background PathMNIST train images (source data only; NCT-CRC-HE-100K itself was
    Macenko-normalised to an unpublished reference image, so this only approximates that
    pipeline), then PIL bicubic to 28x28. Tiles with < 100 tissue pixels (the EMPTY class)
    are left unnormalised. Also normalises PathMNIST val (28x28) as a diagnostic of the
    normaliser's effect on source accuracy."""
    path = os.path.join(RDATA, "kather2016_28x28_macenko.npz")
    if os.path.exists(path):
        d = np.load(path, allow_pickle=False)
        return {k: d[k] for k in d.files}
    from PIL import Image
    X, y = pm["train"]
    idx = np.random.default_rng(0).choice(np.where(y != CODES.index("BACK"))[0], 3000, replace=False)
    ref = _macenko_fit(_od(X[idx]))
    imgs, ok = [], []
    for fo in sorted(os.listdir(KATHER_DIR)):
        for f in sorted(os.listdir(os.path.join(KATHER_DIR, fo))):
            I = np.asarray(Image.open(os.path.join(KATHER_DIR, fo, f)).convert("RGB"))
            J, o = _macenko_apply(I, ref)
            imgs.append(np.asarray(Image.fromarray(J).resize((28, 28), Image.Resampling.BICUBIC), dtype=np.uint8))
            ok.append(o)
    vimg, vok = zip(*[_macenko_apply(im, ref) for im in pm["val"][0]])
    out = {"imgs_macenko": np.stack(imgs), "macenko_ok": np.array(ok), "val_macenko": np.stack(vimg),
           "val_macenko_ok": np.array(vok), "ref_HE": ref[0], "ref_maxC": ref[1]}
    np.savez_compressed(path, **out)
    return out


PM_UM_PER_PX = 224 * 0.5 / 28          # PathMNIST: 224 px at 0.5 um/px downsampled to 28 px
KATHER_UM_PER_PX = 0.495
SCALE_SIDE = int(round(150 * KATHER_UM_PER_PX / PM_UM_PER_PX))     # 19 px


def _pad28(a):
    p = 28 - a.shape[0]; l = p // 2
    return np.pad(a, ((l, p - l), (l, p - l), (0, 0)), mode="reflect")


def load_kather_scalematched(pm, mac):
    """SENSITIVITY (field of view): the Kather tiles cover 74 um in 150 px, the NCT tiles 112 um
    in 224 px, so the bicubic 28x28 version is magnified 1.5x relative to PathMNIST. Here each tile is
    resized to SCALE_SIDE = 19 px, i.e. the same ~4 um/px as PathMNIST, and reflect-padded to
    28x28; the Macenko-harmonised tiles are processed the same way (colour and scale corrected).
    Also stores PathMNIST val magnified 1.5x (centre 19x19 crop upsampled to 28) as the reverse
    diagnostic. Cached in results/R5_data."""
    path = os.path.join(RDATA, "kather2016_scalematched.npz")
    if os.path.exists(path):
        d = np.load(path, allow_pickle=False)
        return {k: d[k] for k in d.files}
    from PIL import Image
    ref = (mac["ref_HE"], mac["ref_maxC"])
    files = [(fo, f) for fo in sorted(os.listdir(KATHER_DIR)) for f in sorted(os.listdir(os.path.join(KATHER_DIR, fo)))]

    def one(fo, f):
        I = np.asarray(Image.open(os.path.join(KATHER_DIR, fo, f)).convert("RGB"))
        J = _macenko_apply(I, ref)[0]
        rs = lambda a: np.asarray(Image.fromarray(a).resize((SCALE_SIDE, SCALE_SIDE), Image.Resampling.BICUBIC),
                                  dtype=np.uint8)
        return _pad28(rs(I)), _pad28(rs(J))
    out = Parallel(n_jobs=N_JOBS)(delayed(one)(fo, f) for fo, f in files)
    c0 = (28 - SCALE_SIDE) // 2
    vz = np.stack([np.asarray(Image.fromarray(x[c0:c0 + SCALE_SIDE, c0:c0 + SCALE_SIDE]).resize(
        (28, 28), Image.Resampling.BICUBIC), dtype=np.uint8) for x in pm["val"][0]])
    d = {"imgs_scale": np.stack([a for a, _ in out]), "imgs_scale_macenko": np.stack([b for _, b in out]),
         "val_zoom": vz}
    np.savez_compressed(path, **d)
    return d


def image_stats(imgs):
    """Descriptive cohort statistics on 28x28 uint8 images: per-channel mean/sd and the
    mean absolute 4-neighbour Laplacian of the grey image (high-frequency content)."""
    x = imgs.astype(np.float64)
    grey = x.mean(-1)
    lap = (4 * grey[:, 1:-1, 1:-1] - grey[:, :-2, 1:-1] - grey[:, 2:, 1:-1]
           - grey[:, 1:-1, :-2] - grey[:, 1:-1, 2:])
    return {"n": int(len(imgs)), "mean_rgb": x.mean((0, 1, 2)).round(2).tolist(),
            "sd_rgb": x.std((0, 1, 2)).round(2).tolist(), "mean_abs_laplacian": float(np.abs(lap).mean())}


# ======================================================================= models
def build(arch):
    import torch.nn as nn
    import torch.nn.functional as F
    if arch.startswith("SmallCNN"):
        class SmallCNN(nn.Module):           # identical to demo/exp2_imaging_pillar.py
            def __init__(self, n_classes=9):
                super().__init__()
                self.c1 = nn.Conv2d(3, 32, 3, padding=1); self.c2 = nn.Conv2d(32, 64, 3, padding=1)
                self.c3 = nn.Conv2d(64, 64, 3, padding=1)
                self.fc1 = nn.Linear(64 * 3 * 3, 128); self.fc2 = nn.Linear(128, n_classes)

            def forward(self, x):
                x = F.max_pool2d(F.relu(self.c1(x)), 2)
                x = F.max_pool2d(F.relu(self.c2(x)), 2)
                x = F.max_pool2d(F.relu(self.c3(x)), 2)
                return self.fc2(F.relu(self.fc1(x.flatten(1))))
        return SmallCNN()
    if arch.startswith("ResNet18"):
        from torchvision.models import resnet18
        m = resnet18(weights=None, num_classes=9)
        m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        m.maxpool = nn.Identity()
        return m
    raise ValueError(arch)


def colour_jitter(x, g, cfg):
    """Stain-colour augmentation of a [0,1] RGB batch on the GPU (HED jitter, Tellez et al. 2019):
    optical density OD = -log(x) is projected on the Ruifrok-Johnston H/E/DAB basis, each stain
    channel is scaled by U(1 +- a) and shifted by U(+- b) per image, and the image is mapped
    back; then brightness U(+- c) and contrast U(1 +- d) jitter."""
    import torch
    a, b_, c_, d_ = cfg
    B = x.shape[0]
    M = torch.tensor(RGB_FROM_HED, device=x.device, dtype=x.dtype)          # rows = stain OD vectors
    Minv = torch.linalg.inv(M)
    rnd = lambda *sh: torch.rand(*sh, device=x.device, generator=g, dtype=x.dtype)
    od = -torch.log(x.clamp_min(1e-6))
    s = torch.einsum("bchw,cd->bdhw", od, Minv)                             # stain concentrations
    s = s * (1 + a * (2 * rnd(B, 3, 1, 1) - 1)) + b_ * (2 * rnd(B, 3, 1, 1) - 1)
    y = torch.exp(-torch.einsum("bdhw,dc->bchw", s, M))
    con = 1 + d_ * (2 * rnd(B, 1, 1, 1) - 1)
    bri = c_ * (2 * rnd(B, 1, 1, 1) - 1)
    mu = y.mean(dim=(1, 2, 3), keepdim=True)
    return ((y - mu) * con + mu + bri).clamp(0, 1)


def _torch_setup():
    import torch
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return torch, ("cuda" if torch.cuda.is_available() else "cpu")


def predict_probs(model, imgs_u8, device, scale=None, bs=2048):
    """Softmax probabilities in fp32 (no autocast at inference); optional per-channel scale
    applied to [0,1] images followed by clamping (the submitted stress transform)."""
    import torch
    import torch.nn.functional as F
    model.eval(); out = []
    s = None if scale is None else torch.tensor(scale, device=device).view(1, 3, 1, 1)
    with torch.no_grad():
        for i in range(0, len(imgs_u8), bs):
            x = torch.from_numpy(imgs_u8[i:i + bs]).to(device).permute(0, 3, 1, 2).float().div_(255)
            if s is not None:
                x = (x * s).clamp(0, 1)
            out.append(F.softmax(model(x).float(), 1).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def eval_sets(pm, kat):
    return {"val": pm["val"][0], "test": pm["test"][0], "kather_bicubic": kat["imgs_bicubic"],
            "kather_spline3": kat["imgs_spline3"]}


def save_probs(model, pm, kat, device, stem):
    P = {k: predict_probs(model, v, device) for k, v in eval_sets(pm, kat).items()}
    P["test_stress"] = predict_probs(model, pm["test"][0], device, scale=STRESS_SCALE)
    np.savez_compressed(os.path.join(MODELS, stem + "_probs.npz"), **P)
    return P


def train_one(arch, tseed, pm, kat):
    torch, device = _torch_setup()
    import torch.nn.functional as F
    stem = f"R5_{arch}_s{tseed}"
    if os.path.exists(os.path.join(MODELS, stem + "_probs.npz")) and os.path.exists(os.path.join(MODELS, stem + ".pt")):
        print(f"[skip] {stem} exists", flush=True)
        return
    seed = 1000 * (ARCHS.index(arch) + 1) + tseed
    cfg = aug_cfg(arch)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    g = torch.Generator(device=device); g.manual_seed(seed)
    Xtr, ytr = pm["train"]
    X = torch.from_numpy(Xtr).to(device).permute(0, 3, 1, 2).contiguous()      # uint8 on GPU
    y = torch.from_numpy(ytr).to(device)
    model = build(arch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    n = len(y); steps = EPOCHS * int(np.ceil(n / BATCH))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    hist, t0 = [], time.time()
    for ep in range(EPOCHS):
        model.train(); perm = torch.randperm(n, device=device, generator=g)
        tl, tc = 0.0, 0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb = X[idx].float().div_(255)
            fh = torch.rand(len(idx), device=device, generator=g) < 0.5     # horizontal flip
            fv = torch.rand(len(idx), device=device, generator=g) < 0.5     # vertical flip
            xb = torch.where(fh[:, None, None, None], xb.flip(3), xb)
            xb = torch.where(fv[:, None, None, None], xb.flip(2), xb)
            if cfg is not None:
                xb = colour_jitter(xb, g, cfg)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                logits = model(xb)
                loss = F.cross_entropy(logits.float(), y[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tl += float(loss) * len(idx); tc += int((logits.argmax(1) == y[idx]).sum())
        hist.append({"epoch": ep + 1, "loss": tl / n, "train_acc_aug": tc / n, "t_sec": round(time.time() - t0, 1)})
        print(f"  {stem} ep {ep + 1:2d}/{EPOCHS} loss {tl / n:.4f} acc {tc / n:.4f} t {time.time() - t0:.0f}s", flush=True)
    torch.save(model.state_dict(), os.path.join(MODELS, stem + ".pt"))
    P = save_probs(model, pm, kat, device, stem)
    ptr = predict_probs(model, Xtr, device)
    info = {"arch": arch, "train_seed": tseed, "torch_seed": seed, "epochs": EPOCHS, "batch": BATCH,
            "optimizer": f"AdamW(lr={LR}, weight_decay={WD}), cosine annealing per step, bf16 autocast",
            "augmentation": ("random horizontal + vertical flips + HED stain jitter "
                             f"(alpha +-{cfg[0]}, beta +-{cfg[1]}) + brightness +-{cfg[2]} / contrast +-{cfg[3]}"
                             if cfg is not None else
                             "random horizontal + vertical flips only (no colour/stain augmentation)"),
            "n_params": int(sum(p.numel() for p in model.parameters())),
            "history": hist, "train_time_sec": round(time.time() - t0, 1),
            "acc_train_noaug": float((ptr.argmax(1) == ytr).mean()),
            "acc_val": float((P["val"].argmax(1) == pm["val"][1]).mean()),
            "acc_test": float((P["test"].argmax(1) == pm["test"][1]).mean())}
    json.dump(info, open(os.path.join(MODELS, stem + "_train.json"), "w"), indent=1)
    print(f"[done] {stem} val {info['acc_val']:.4f} test {info['acc_test']:.4f} ({info['train_time_sec']:.0f}s)", flush=True)
    del X, y, model
    torch.cuda.empty_cache()


def load_model(stem, device):
    import torch
    model = build("SmallCNN" if stem.startswith("R5_SmallCNN") else "ResNet18").to(device)
    ck = SUBMITTED_CKPT if "submitted" in stem else os.path.join(MODELS, stem + ".pt")
    model.load_state_dict(torch.load(ck, map_location=device))
    return model


def extra_probs(stem, mac, sca):
    """Softmax on the sensitivity image sets (Macenko-harmonised tiles, scale-matched tiles,
    both corrections, and the Macenko / 1.5x-zoom val diagnostics), from the saved checkpoint.
    Missing arrays are added to the cache; existing ones are reused unchanged."""
    path = os.path.join(MODELS, stem + "_probs_extra.npz")
    P = dict(np.load(path)) if os.path.exists(path) else {}
    need = {"kather_macenko": mac["imgs_macenko"], "val_macenko": mac["val_macenko"],
            "kather_scalematched": sca["imgs_scale"], "kather_scalematched_macenko": sca["imgs_scale_macenko"],
            "val_zoom": sca["val_zoom"]}
    miss = {k: v for k, v in need.items() if k not in P}
    if miss:
        torch, device = _torch_setup()
        model = load_model(stem, device)
        for k, v in miss.items():
            P[k] = predict_probs(model, v, device)
        np.savez_compressed(path, **P)
    return P


def submitted_model_probs(pm, kat):
    """Re-evaluate the cached submitted SmallCNN (read-only) under the same pipeline."""
    stem = "R5_SmallCNN-submitted_s0"
    path = os.path.join(MODELS, stem + "_probs.npz")
    if os.path.exists(path):
        return
    torch, device = _torch_setup()
    model = build("SmallCNN").to(device)
    model.load_state_dict(torch.load(SUBMITTED_CKPT, map_location=device))
    save_probs(model, pm, kat, device, stem)
    info = {"arch": "SmallCNN-submitted", "source": "demo/exp2_model_cache.pt (read-only)",
            "training": "20,000-image random subsample of train, 6 epochs, Adam lr 1e-3, CPU (demo/exp2_imaging_pillar.py)"}
    json.dump(info, open(os.path.join(MODELS, stem + "_train.json"), "w"), indent=1)


# ======================================================================= splits
def grouped_half(slide_codes, rng):
    """Slide-grouped recal/test split: slides permuted, cut at the slide boundary closest to
    half of the tiles; the recal pool is then shuffled so the first k tiles are a random
    sample of the pool slides. No slide contributes to both halves."""
    us = rng.permutation(np.unique(slide_codes))
    order = np.concatenate([np.where(slide_codes == u)[0] for u in us])
    cum = np.cumsum([int((slide_codes == u).sum()) for u in us])
    h = int(cum[np.argmin(np.abs(cum - len(slide_codes) / 2))])
    return {"rpool": rng.permutation(order[:h]), "ttest": order[h:]}


def balanced_slide_partitions(slide_codes, cls, min_per_class=10, lo=0.40, hi=0.60, top=10):
    """Class-balanced slide-disjoint bipartitions: all bipartitions of the slides whose two halves
    hold 40-60 % of the tiles and at least `min_per_class` tiles of every class, ranked by the
    total-variation distance between the halves' class distributions; the `top` most balanced are
    kept (a split seed picks one, and a coin decides which half is the recalibration pool).
    Returns (candidates, n_feasible)."""
    S = np.unique(slide_codes); C = np.unique(cls); n = len(slide_codes)
    cand = []
    for m in range(1, 2 ** len(S) - 1):
        sel = np.array([(m >> i) & 1 for i in range(len(S))], bool)
        if not sel[0]:                                   # each partition once (drop complements)
            continue
        a = np.isin(slide_codes, S[sel])
        if not (lo <= a.mean() <= hi):
            continue
        ca = np.array([(cls[a] == c).sum() for c in C]); cb = np.array([(cls[~a] == c).sum() for c in C])
        if ca.min() < min_per_class or cb.min() < min_per_class:
            continue
        cand.append((float(0.5 * np.abs(ca / ca.sum() - cb / cb.sum()).sum()), S[sel].tolist()))
    cand.sort(key=lambda t: (t[0], t[1]))
    return [{"slides": s, "class_mix_tv": tv} for tv, s in cand[:top]], len(cand)


def grouped_from_slides(slide_codes, slides, rng):
    a = np.where(np.isin(slide_codes, slides))[0]; b = np.where(~np.isin(slide_codes, slides))[0]
    if rng.random() < 0.5:
        a, b = b, a
    return {"rpool": rng.permutation(a), "ttest": b}


def class_mix_tv(cls, sp):
    """Total-variation distance between the class distributions of the two halves of a split."""
    C = np.unique(cls)
    a = np.array([(cls[sp["rpool"]] == c).mean() for c in C]); b = np.array([(cls[sp["ttest"]] == c).mean() for c in C])
    return float(0.5 * np.abs(a - b).sum())


def make_splits(seed, sizes, slide5, slide4, parts4):
    sp = {}
    rng = np.random.default_rng([seed, 0]); perm = rng.permutation(sizes["val"]); h = sizes["val"] // 2
    sp["val"] = {"cal": perm[:h], "stest": perm[h:]}
    for code, key in [(1, "T1"), (2, "T2"), (3, "T2set"), (5, "T2noE")]:
        n = sizes[key]; rng = np.random.default_rng([seed, code]); perm = rng.permutation(n); h = n // 2
        sp[key] = {"rpool": perm[:h], "ttest": perm[h:]}
    sp["T2grp"] = grouped_half(slide5, np.random.default_rng([seed, 4]))
    sp["T2grpE"] = grouped_half(slide4, np.random.default_rng([seed, 7]))
    rng = np.random.default_rng([seed, 6])
    sp["T2grpB"] = grouped_from_slides(slide4, parts4[int(rng.integers(len(parts4)))]["slides"], rng)
    for k, v in sp.items():
        check_disjoint(v, {"T2grp": slide5, "T2grpE": slide4, "T2grpB": slide4}.get(k))
    return sp


# ======================================================================= conformal
def aps_scores(p, u):
    """Randomised APS score for every class: mass of strictly more probable classes
    (in sorted order) + u * p_c (Romano et al. 2020)."""
    o = np.argsort(-p, axis=1, kind="stable")
    ps = np.take_along_axis(p, o, 1)
    s_sorted = np.cumsum(ps, 1) - ps + u[:, None] * ps
    s = np.empty_like(p)
    np.put_along_axis(s, o, s_sorted, 1)
    return s


def all_scores(method, p, u):
    return 1.0 - p if method == "LAC" else aps_scores(p, u)


def cal_scores(S, A):
    """Calibration score = min over admissible labels (single label: the label's score)."""
    return np.where(A, S, np.inf).min(1)


def ece(conf, correct, n_bins=15):
    edges = np.linspace(0, 1, n_bins + 1)
    b = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    return float(sum((b == i).mean() * abs(correct[b == i].mean() - conf[b == i].mean())
                     for i in range(n_bins) if (b == i).any()))


def set_metrics(sets, A, p, g, gnames, min_group=10):
    size = sets.sum(1); covered = (sets & A).any(1)
    pred = p.argmax(1); correct = A[np.arange(len(p)), pred]; conf = p.max(1)
    per = {nm: float(covered[g == j].mean()) for j, nm in enumerate(gnames) if (g == j).sum() >= min_group}
    # most confident = smallest NON-EMPTY set; an empty LAC/APS set means no class reaches the
    # threshold, i.e. the least confident case, so empty sets are ranked last
    order = np.lexsort((-conf, np.where(size == 0, 99, size))); h = len(p) // 2
    acc = float(correct.mean())
    # "abstention" = the set is not a singleton, i.e. the decision rule "act only on a singleton
    # prediction set" refuses; an empty set (no class reaches the threshold) is a refusal too.
    # The literal |set| > 1 rate required by the task specification is abstention_gt1.
    out = {"coverage": float(covered.mean()), "set_size": float(size.mean()),
           "singleton": float((size == 1).mean()), "empty": float((size == 0).mean()),
           "abstention": float((size != 1).mean()), "abstention_gt1": float((size > 1).mean()),
           "nonsingleton": float((size != 1).mean()),
           "class_min_cov": float(min(per.values())), "class_macro_cov": float(np.mean(list(per.values()))),
           "n_classes_eval": len(per),
           "accuracy": acc, "ece": ece(conf, correct.astype(float)),
           "sel_acc50": float(correct[order[:h]].mean()), "n_eval": int(len(p)), "class_cov": per}
    out["sel_gain50"] = out["sel_acc50"] - acc
    sh = [per[c] for c in SHARED5 if c in per]
    out["shared5_macro_cov"] = float(np.mean(sh)) if len(sh) == len(SHARED5) else None
    # worst-class coverage restricted to the classes shared by all domains, so that the
    # worst-class numbers of source, T1 and T2 are comparable (the plain class_min_cov is a min
    # over 9 classes in the source domain but over 4-5 in the external cohort)
    out["shared5_min_cov"] = float(np.min(sh)) if len(sh) == len(SHARED5) else None
    s4 = [per[c] for c in SHARED4 if c in per]      # same, without BACK (absent from the noEMPTY target)
    out["shared4_min_cov"] = float(np.min(s4)) if len(s4) == len(SHARED4) else None
    return out


def build_targets(pm, kat):
    """Static description of every evaluated domain (labels as admissible-set matrices)."""
    onehot = lambda y: np.eye(9, dtype=bool)[y]
    fo = kat["folder"]
    idx5 = np.where(np.isin(fo, [f for f, v in KATHER_MAP.items() if v is not None and len(v) == 1]))[0]
    idx7 = np.where(np.isin(fo, [f for f, v in KATHER_MAP.items() if v is not None]))[0]
    idx4 = np.where(np.isin(fo, [f for f, v in KATHER_MAP.items()
                                 if v is not None and len(v) == 1 and v[0] != "BACK"]))[0]
    Akat = np.zeros((len(fo), 9), bool)
    for f, v in KATHER_MAP.items():
        if v is not None:
            for c in v:
                Akat[fo == f, CODES.index(c)] = True
    kfold = [f for f, v in KATHER_MAP.items() if v is not None]
    kname = {f: "|".join(KATHER_MAP[f]) for f in kfold}
    g7_names = [kname[f] for f in kfold]
    g_all = np.array([kfold.index(f) if f in kfold else -1 for f in fo])
    yv, yt = pm["val"][1], pm["test"][1]
    T = {
        "S_source-test": dict(p="val", sub=None, A=onehot(yv), g=yv, gn=CODES, split="val", code=0,
                              desc="PathMNIST val (NCT-CRC-HE-100K) other half; in-distribution reference"),
        "T1_CRC-VAL-HE-7K": dict(p="test", sub=None, A=onehot(yt), g=yt, gn=CODES, split="T1", code=1,
                                 desc="PathMNIST test = CRC-VAL-HE-7K, independent patients, same group/pipeline"),
        "T2_Kather2016_5class": dict(p="kather_bicubic", sub=idx5, A=Akat[idx5], g=g_all[idx5], gn=g7_names,
                                     split="T2", code=2,
                                     desc="Kather 2016 external cohort, 5 unambiguous classes, PIL bicubic 28x28 (MAIN)"),
        "T2sens_Kather2016_7class_setvalued": dict(p="kather_bicubic", sub=idx7, A=Akat[idx7], g=g_all[idx7], gn=g7_names,
                                                   split="T2set", code=3,
                                                   desc="+ STROMA->{STR,MUS}, DEBRIS->{DEB,MUC} set-valued labels"),
        "T2sens_Kather2016_5class_slidegrouped": dict(p="kather_bicubic", sub=idx5, A=Akat[idx5], g=g_all[idx5],
                                                      gn=g7_names, split="T2grp", code=4,
                                                      desc="recal pool and target-test from disjoint slides"),
        "T2sens_Kather2016_5class_spline3resize": dict(p="kather_spline3", sub=idx5, A=Akat[idx5], g=g_all[idx5],
                                                       gn=g7_names, split="T2", code=2,
                                                       desc="same tiles/splits, non-antialiased order-3 spline resize"),
        "T2sens_Kather2016_5class_macenko": dict(p="kather_macenko", sub=idx5, A=Akat[idx5], g=g_all[idx5],
                                                 gn=g7_names, split="T2", code=2,
                                                 desc="same tiles/splits, Macenko-normalised towards a PathMNIST-derived "
                                                      "reference (pipeline harmonisation); EMPTY tiles not normalisable"),
        "T2sens_Kather2016_5class_scalematched": dict(p="kather_scalematched", sub=idx5, A=Akat[idx5], g=g_all[idx5],
                                                      gn=g7_names, split="T2", code=2,
                                                      desc=f"same tiles/splits, resized to {SCALE_SIDE} px (~4 um/px as in "
                                                           "PathMNIST) and reflect-padded to 28: removes the 1.5x field-of-view "
                                                           "mismatch"),
        "T2sens_Kather2016_5class_scalematched_macenko": dict(p="kather_scalematched_macenko", sub=idx5, A=Akat[idx5],
                                                              g=g_all[idx5], gn=g7_names, split="T2", code=2,
                                                              desc="same tiles/splits, scale-matched AND Macenko-harmonised "
                                                                   "(both corrections together)"),
        "T2sens_Kather2016_4class_noEMPTY": dict(p="kather_bicubic", sub=idx4, A=Akat[idx4], g=g_all[idx4],
                                                 gn=g7_names, split="T2noE", code=5,
                                                 desc="main tiles without 08_EMPTY (white glass) -> BACK (dark Macenko-"
                                                      "normalised background in NCT-CRC-HE-100K): 4 tissue classes"),
        "T2sens_Kather2016_4class_slidegrouped": dict(p="kather_bicubic", sub=idx4, A=Akat[idx4], g=g_all[idx4],
                                                      gn=g7_names, split="T2grpE", code=7,
                                                      desc="4 tissue classes, recal pool and target-test from disjoint slides "
                                                           "chosen at random (control for the class-balanced variant)"),
        "T2sens_Kather2016_4class_slidegrouped_balanced": dict(p="kather_bicubic", sub=idx4, A=Akat[idx4], g=g_all[idx4],
                                                               gn=g7_names, split="T2grpB", code=6,
                                                               desc="4 tissue classes, recal pool and target-test from disjoint "
                                                                    "slides chosen to match the two halves' class mixes"),
        "STRESS_T1_synthetic_colour": dict(p="test_stress", sub=None, A=onehot(yt), g=yt, gn=CODES, split="T1", code=1,
                                           desc=f"SECONDARY synthetic stress test: T1 x per-channel scale {STRESS_SCALE}"),
    }
    return T, {"T2": idx5, "T2set": idx7, "T2noE": idx4, "T2grp": idx5, "T2grpE": idx4, "T2grpB": idx4}


def analyze_split(P, T, sp, seed):
    """All metrics for one trained model and one split seed."""
    rows = {}
    pv = P["val"]; Tv = T["S_source-test"]
    for method in METHODS:
        Sv = all_scores(method, pv, np.random.default_rng([seed, 0, 99]).random(len(pv)))
        cal, st = sp["val"]["cal"], sp["val"]["stest"]
        q_src = conformal_q(cal_scores(Sv[cal], Tv["A"][cal]), ALPHA)
        m = set_metrics(Sv[st] <= q_src, Tv["A"][st], pv[st], Tv["g"][st], Tv["gn"]); m["qhat"] = float(q_src)
        rows[("S_source-test", method, 0)] = m
        for name, t in T.items():
            if name == "S_source-test":
                continue
            p = P[t["p"]] if t["sub"] is None else P[t["p"]][t["sub"]]
            u = np.random.default_rng([seed, t["code"], 99]).random(len(p))
            S = all_scores(method, p, u)
            rp, tt = sp[t["split"]]["rpool"], sp[t["split"]]["ttest"]
            for k in KGRID:
                if k > len(rp):
                    continue
                q = q_src if k == 0 else conformal_q(cal_scores(S[rp[:k]], t["A"][rp[:k]]), ALPHA)
                m = set_metrics(S[tt] <= q, t["A"][tt], p[tt], t["g"][tt], t["gn"])
                m["qhat"] = float(q) if np.isfinite(q) else None
                m["infinite_q"] = bool(not np.isfinite(q))
                rows[(name, method, k)] = m
    return rows


SCALARS = ["coverage", "set_size", "singleton", "empty", "abstention", "abstention_gt1", "nonsingleton",
           "class_min_cov", "class_macro_cov", "n_classes_eval", "shared5_macro_cov", "shared5_min_cov",
           "shared4_min_cov", "accuracy", "ece", "sel_acc50", "sel_gain50", "qhat", "n_eval"]


def aggregate(rowsets):
    """rowsets: list of per-(model, split) dicts -> summaries over all of them."""
    keys = sorted({k for r in rowsets for k in r}, key=lambda x: (x[0], x[1], x[2]))
    out = {}
    for (name, method, k) in keys:
        rs = [r[(name, method, k)] for r in rowsets if (name, method, k) in r]
        d = {m: summarize([r.get(m) for r in rs]) for m in SCALARS}
        cls = sorted({c for r in rs for c in r["class_cov"]})
        d["class_cov"] = {c: summarize([r["class_cov"].get(c) for r in rs]) for c in cls}
        d["n_rows"] = len(rs)
        out.setdefault(name, {}).setdefault(method, {})[str(k)] = d
    return out


def paired_drop(rowsets, name, method, k=0, metric="coverage"):
    """Per-split paired difference target(k) - source-test (same model, same split seed)."""
    return summarize([r[(name, method, k)][metric] - r[("S_source-test", method, 0)][metric]
                      for r in rowsets if (name, method, k) in r])


def kather_confusion(P, kat, key="kather_bicubic"):
    """Fraction of tiles of each Kather folder predicted as each PathMNIST class."""
    pred = P[key].argmax(1)
    return {f: np.bincount(pred[kat["folder"] == f], minlength=9).astype(float).__truediv__(
        (kat["folder"] == f).sum()).round(4).tolist() for f in KATHER_MAP}


def accuracies(P, pm, T):
    out = {"val": float((P["val"].argmax(1) == pm["val"][1]).mean()),
           "test_T1": float((P["test"].argmax(1) == pm["test"][1]).mean()),
           "stress_T1": float((P["test_stress"].argmax(1) == pm["test"][1]).mean()),
           "val_macenko_diagnostic": float((P["val_macenko"].argmax(1) == pm["val"][1]).mean()),
           "val_zoom1.5x_diagnostic": float((P["val_zoom"].argmax(1) == pm["val"][1]).mean())}
    for name, t in T.items():
        if t["sub"] is None:
            continue
        p = P[t["p"]][t["sub"]]
        out[name] = float(t["A"][np.arange(len(p)), p.argmax(1)].mean())
    return out


# ======================================================================= main
def run_train():
    pm = load_pathmnist(); kat = load_kather(); mac = load_kather_macenko(pm)
    sca = load_kather_scalematched(pm, mac)
    submitted_model_probs(pm, kat)
    extra_probs("R5_SmallCNN-submitted_s0", mac, sca)
    for arch in ARCHS:
        for t in TRAIN_SEEDS:
            train_one(arch, t, pm, kat)
            extra_probs(f"R5_{arch}_s{t}", mac, sca)


def run_analyze():
    t0 = time.time()
    label_map = verify_label_order()
    pm = load_pathmnist(); kat = load_kather(); mac = load_kather_macenko(pm)
    sca = load_kather_scalematched(pm, mac)
    T, glob = build_targets(pm, kat)
    idx5, idx7, idx4 = glob["T2"], glob["T2set"], glob["T2noE"]
    slide5 = np.unique(kat["slide"][idx5], return_inverse=True)[1]
    slide4 = np.unique(kat["slide"][idx4], return_inverse=True)[1]
    cls5 = np.unique(kat["folder"][idx5], return_inverse=True)[1]
    cls4 = np.unique(kat["folder"][idx4], return_inverse=True)[1]
    parts4, n_feasible4 = balanced_slide_partitions(slide4, cls4)
    sizes = {"val": len(pm["val"][1]), "T1": len(pm["test"][1]), "T2": len(idx5), "T2set": len(idx7),
             "T2noE": len(idx4)}
    splits = {s: make_splits(s, sizes, slide5, slide4, parts4) for s in SEEDS}
    grp_mix = {"T2grp_random_slides_5class": summarize([class_mix_tv(cls5, splits[s]["T2grp"]) for s in SEEDS]),
               "T2grpE_random_slides_4class": summarize([class_mix_tv(cls4, splits[s]["T2grpE"]) for s in SEEDS]),
               "T2grpB_balanced_slides_4class": summarize([class_mix_tv(cls4, splits[s]["T2grpB"]) for s in SEEDS]),
               "T2_tile_level_5class": summarize([class_mix_tv(cls5, splits[s]["T2"]) for s in SEEDS]),
               "T2noE_tile_level_4class": summarize([class_mix_tv(cls4, splits[s]["T2noE"]) for s in SEEDS]),
               "n_distinct_partitions": {f: len({tuple(sorted(np.unique((slide5 if f == "T2grp" else slide4)
                                                                       [splits[s][f]["rpool"]]))) for s in SEEDS})
                                         for f in ["T2grp", "T2grpE", "T2grpB"]},
               "n_feasible_balanced_partitions": n_feasible4,
               "balanced_candidates": parts4}
    # persist split indices (Kather indices mapped back to positions in the 5,000-tile file list)
    json.dump({"protocol": "R5 imaging splits; val indices into PathMNIST val, T1 into PathMNIST test, "
                           "T2/T2set/T2grp into kather_files (sorted folder/file order)",
               "kather_files": [f"{a}/{b}" for a, b in zip(kat["folder"], kat["files"])],
               "splits": {str(s): {f"{fam}_{part}": (glob[fam][ix] if fam in glob else ix).tolist()
                                   for fam, d in sp.items() for part, ix in d.items()}
                          for s, sp in splits.items()}},
              open(os.path.join(SPLITS, f"{TAG}_splits.json"), "w"))

    models = {a: [f"R5_{a}_s{t}" for t in TRAIN_SEEDS] for a in ARCHS}
    models["SmallCNN-submitted"] = ["R5_SmallCNN-submitted_s0"]
    res = {"meta": {}, "architectures": {}}
    for arch, stems in models.items():
        allrows, per_seed, accs, conf, conf_mac, trinfo = [], {}, [], [], [], []
        for stem in stems:
            P = dict(np.load(os.path.join(MODELS, stem + "_probs.npz")))
            P.update(extra_probs(stem, mac, sca))
            rows = Parallel(n_jobs=N_JOBS)(delayed(analyze_split)(P, T, splits[s], s) for s in SEEDS)
            allrows += rows
            per_seed[stem] = {name: {meth: {k: {m: summarize([r[(name, meth, int(k))][m] for r in rows
                                                               if (name, meth, int(k)) in r])["mean"]
                                                for m in ["coverage", "set_size", "class_min_cov", "accuracy"]}
                                            for k in (["0"] if name == "S_source-test" else ["0", str(K_HEAD), str(K_SUBMITTED)])}
                                     for meth in METHODS} for name in T}
            accs.append(accuracies(P, pm, T)); conf.append(kather_confusion(P, kat))
            conf_mac.append(kather_confusion(P, kat, "kather_macenko"))
            ti = os.path.join(MODELS, stem + "_train.json")
            trinfo.append(json.load(open(ti)) if os.path.exists(ti) else None)
        A = {"models": stems, "n_rows": len(allrows), "results": aggregate(allrows), "per_model": per_seed,
             "accuracy_full_sets": {k: summarize([a[k] for a in accs]) for k in accs[0]},
             "accuracy_full_sets_per_model": accs,
             "kather_confusion_mean": {f: np.mean([c[f] for c in conf], 0).round(4).tolist() for f in KATHER_MAP},
             "kather_confusion_mean_macenko": {f: np.mean([c[f] for c in conf_mac], 0).round(4).tolist() for f in KATHER_MAP},
             "training": [{k: v for k, v in (ti or {}).items() if k != "history"} | (
                 {"final_epoch": ti["history"][-1]} if ti and "history" in ti else {}) for ti in trinfo],
             "paired_drop_vs_source": {name: {meth: {"coverage_k0": paired_drop(allrows, name, meth),
                                                     "class_min_cov_k0": paired_drop(allrows, name, meth, metric="class_min_cov")}
                                              for meth in METHODS} for name in T if name != "S_source-test"},
             "per_split_LAC": [{"model": stems[i // len(SEEDS)], "seed": SEEDS[i % len(SEEDS)],
                                **{f"{name}_k{k}": r.get((name, "LAC", k), {}).get("coverage")
                                   for name in T for k in ([0] if name == "S_source-test" else [0, K_HEAD, K_SUBMITTED])}}
                               for i, r in enumerate(allrows)]}
        res["architectures"][arch] = A
        R = A["results"]
        print(f"{arch:20s} acc val {A['accuracy_full_sets']['val']['mean']:.3f} T1 {A['accuracy_full_sets']['test_T1']['mean']:.3f} "
              f"T2 {A['accuracy_full_sets']['T2_Kather2016_5class']['mean']:.3f} | LAC cov S "
              f"{R['S_source-test']['LAC']['0']['coverage']['mean']:.3f} T1 {R['T1_CRC-VAL-HE-7K']['LAC']['0']['coverage']['mean']:.3f} "
              f"T2 {R['T2_Kather2016_5class']['LAC']['0']['coverage']['mean']:.3f} -> k30 "
              f"{R['T2_Kather2016_5class']['LAC'][str(K_HEAD)]['coverage']['mean']:.3f} | stress "
              f"{R['STRESS_T1_synthetic_colour']['LAC']['0']['coverage']['mean']:.3f}", flush=True)

    res["meta"] = {
        "alpha": ALPHA, "n_split_seeds": len(SEEDS), "train_seeds": TRAIN_SEEDS, "kgrid": KGRID,
        "k_headline_protocol": K_HEAD, "k_submitted": K_SUBMITTED, "methods": METHODS,
        "pathmnist_label_order": label_map, "codes": CODES, "kather_label_map": {k: v for k, v in KATHER_MAP.items()},
        "n": {"train": int(len(pm["train"][1])), "val": int(len(pm["val"][1])), "test_T1": int(len(pm["test"][1])),
              "kather_total": int(len(kat["folder"])), "kather_5class": int(len(idx5)), "kather_7class": int(len(idx7)),
              "kather_4class_noEMPTY": int(len(idx4))},
        "split_sizes_seed0": {f"{fam}_{part}": int(len(ix)) for fam, d in splits[0].items() for part, ix in d.items()},
        "T2grp_rpool_sizes": [int(len(splits[s]["T2grp"]["rpool"])) for s in SEEDS],
        "targets": {n: t["desc"] for n, t in T.items()},
        "preprocessing": {"pathmnist": "MedMNIST v2 28x28 npz (download=False, ~/.medmnist/pathmnist.npz); "
                                       "MedMNIST v2 reports 'cubic spline interpolation' from 224x224",
                          "kather_main": "PIL Image.resize((28,28), BICUBIC) from 150x150 RGB (antialiased cubic, a=-0.5)",
                          "kather_sensitivity": "scipy.ndimage.zoom(order=3, mode='reflect', grid_mode=True), no antialiasing",
                          "kather_scalematched": f"PIL bicubic to {SCALE_SIDE}x{SCALE_SIDE} px (150 px x 0.495 um/px at the "
                                                 f"PathMNIST scale of {PM_UM_PER_PX:.1f} um/px), reflect-padded to 28x28",
                          "val_zoom_diagnostic": f"PathMNIST val centre-cropped to {SCALE_SIDE}x{SCALE_SIDE} and bicubic-"
                                                 "resized to 28x28 (magnifies val 1.5x, the reverse of the Kather mismatch)",
                          "input_scaling": "uint8/255, no mean/std normalisation (as in the submitted script)"},
        "training": {"epochs": EPOCHS, "batch": BATCH, "optimizer": f"AdamW lr {LR} wd {WD}, cosine per step",
                     "augmentation": "SmallCNN / ResNet18: horizontal + vertical flips only, no colour augmentation "
                                     "(deliberate worst case); the '-aug' and '-augstrong' arms add HED stain jitter (per "
                                     "image, each stain OD channel scaled by U(1 +- a) and shifted by U(+- b); "
                                     "Ruifrok-Johnston basis, Tellez et al. 2019) and brightness U(+- c) / contrast "
                                     f"U(1 +- d) with (a, b, c, d) = {AUG_CFG}",
                     "augmentation_choice": "the colour-augmentation settings were fixed before the augmented arms were run "
                                            "and were not iterated; no setting was chosen using val, T1 or T2 results. Two "
                                            "strengths are reported so that the conclusion does not rest on one magnitude",
                     "selection": "last epoch; no early stopping or selection on val/test/external data",
                     "precision": "bf16 autocast for training, fp32 inference",
                     "determinism": "seeded; cuDNN deterministic. An independent retrain of SmallCNN seed 0 and ResNet18 "
                                    "seed 0 on this machine reproduced bit-identical softmax arrays; checkpoints and softmax "
                                    "arrays are saved either way and the analysis is deterministic given them"},
        "aps": "randomised APS (Romano et al. 2020), u ~ U(0,1) per image from rng([seed, target_code, 99]); empty sets allowed",
        "set_valued": "calibration score = min over admissible labels; covered if set intersects admissible labels; "
                      "correct if argmax admissible",
        "selective": "50 % of target-test with the smallest set, ties broken by larger max softmax",
        "abstention": "'abstention' = fraction of images whose set is NOT a singleton (= 'empty' + 'abstention_gt1'), i.e. the "
                      "decision rule 'act only on a singleton set' refuses; 'abstention_gt1' is the literal |set| > 1 rate of "
                      "the task specification. Figures and tables should use 'abstention'/'nonsingleton' (or show 'empty' and "
                      "'abstention_gt1' stacked): under LAC a confident model produces empty, not large, sets, so "
                      "'abstention_gt1' alone reads as 'never abstains' where in fact most sets are empty",
        "ece": "15 equal-width bins on max softmax (Guo et al. 2017)",
        "class_conditional": "coverage within each class (Kather folder for T2) with >= 10 test images; min and macro mean. "
                             "class_min_cov is a min over the classes PRESENT in that domain (9 for source/T1/STRESS, 5 or 4 "
                             "for the Kather targets), so it is not comparable across domains; shared5_min_cov / "
                             "shared4_min_cov restrict the min to {ADI, BACK, LYM, NORM, TUM} / {ADI, LYM, NORM, TUM}, which "
                             "exist in every domain. Split conformal is marginal: neither source calibration nor target "
                             "recalibration targets class-conditional coverage, so worst-class coverage is below nominal in "
                             "every domain including the source (class-conditional validity would need a Mondrian/"
                             "class-conditional variant, which is not claimed anywhere in the manuscript)",
        "slide_grouping": grp_mix | {"note": "T2grp / T2grpE permute the slides and cut at the slide boundary nearest to half "
                                             "the tiles, so the two halves also differ in class mix (LYMPHO is concentrated "
                                             "in slides 01/02/05/09, MUCOSA is absent from 01/02/04/10 and EMPTY comes from "
                                             "only slides 06 and 10): their coverage mixes slide-level covariate shift with "
                                             "label shift. T2grpB keeps the slide disjointness but picks the most "
                                             "class-balanced bipartitions, and is restricted to the 4 tissue classes because "
                                             "no slide-disjoint split of the 5-class set can balance EMPTY; T2grpE is the "
                                             "random-slide control on the same 4 classes, so T2grpE vs T2grpB isolates the "
                                             "class-mix effect and T2noE vs T2grpE the slide effect. Only 10 slides exist, so "
                                             "even distinct partitions overlap heavily and the percentile intervals describe "
                                             "few effectively independent configurations"},
        "beta_theory_5_50_95": {str(k): beta_coverage_quantiles(k) for k in KGRID if k > 0} | {"n_cal_source": beta_coverage_quantiles(len(pm['val'][1]) // 2)},
        "cohort_image_stats": {"PathMNIST_train_sub10k": image_stats(pm["train"][0][::9]), "PathMNIST_val": image_stats(pm["val"][0]),
                               "PathMNIST_test_T1": image_stats(pm["test"][0]),
                               "PathMNIST_train_BACK_class": image_stats(pm["train"][0][pm["train"][1] == CODES.index("BACK")]),
                               "PathMNIST_val_zoom1.5x": image_stats(sca["val_zoom"]),
                               "Kather2016_5class_bicubic": image_stats(kat["imgs_bicubic"][idx5]),
                               "Kather2016_EMPTY_bicubic": image_stats(kat["imgs_bicubic"][kat["folder"] == "08_EMPTY"]),
                               "Kather2016_5class_spline3": image_stats(kat["imgs_spline3"][idx5]),
                               "Kather2016_5class_scalematched": image_stats(sca["imgs_scale"][idx5]),
                               "Kather2016_5class_macenko": image_stats(mac["imgs_macenko"][idx5])},
        "macenko": {"reference": "3,000 random non-BACK PathMNIST train images (seed 0), Io=240, beta=0.15, alpha=1 %",
                    "ref_HE_columns_H_E": mac["ref_HE"].round(4).tolist(), "ref_maxC": mac["ref_maxC"].round(4).tolist(),
                    "kather_normalised_per_folder": {f: [int(mac["macenko_ok"][kat["folder"] == f].sum()),
                                                         int((kat["folder"] == f).sum())] for f in KATHER_MAP},
                    "val_normalised": [int(mac["val_macenko_ok"].sum()), int(len(mac["val_macenko_ok"]))]},
        "kather_slides_per_folder": {f: {s: int(((kat["folder"] == f) & (kat["slide"] == s)).sum())
                                         for s in np.unique(kat["slide"])} for f in KATHER_MAP},
        "intervals": "mean and 2.5-97.5 percentile over 20 split seeds x 3 training seeds (60 values; 20 for the submitted model); "
                     "splits resample one finite dataset per cohort",
        "runtime_analyze_sec": None,
    }
    res["meta"]["runtime_analyze_sec"] = round(time.time() - t0, 1)
    dump(res, f"{TAG}.json")
    write_md(res)


# ======================================================================= markdown
def _f(s, d=3):
    if s is None or s.get("mean") is None:
        return "n/a"
    return f"{s['mean']:.{d}f} [{s['lo']:.{d}f}, {s['hi']:.{d}f}]"


def write_md(res):
    M = res["meta"]; L = []
    L.append("# R5 imaging: PathMNIST with an independently acquired cohort (Kather et al. 2016)\n")
    L.append("Wording used throughout: T2 is an **independently acquired, non-colour-normalised cohort** (Kather 2016: "
             "different acquisition period, different tile geometry, same pathology archive as the source cohort). It is "
             "'genuine' only in the sense of 'not synthetic'; it is not a proven different institution (see Caveats).\n")
    L.append("Script `revision/code/exp_R5_imaging.py`; full numbers in `revision/results/R5_imaging.json`; split "
             "indices in `revision/results/splits/R5_imaging_splits.json`; checkpoints and softmax arrays in "
             "`revision/results/models/R5_*`; processed external cohort in `revision/results/R5_data/kather2016_28x28.npz`.\n")
    L.append(f"alpha = {M['alpha']}; 20 split seeds per trained model; values are mean [2.5, 97.5 percentile] over "
             "20 splits x 3 training seeds (60 values; 20 for the submitted model). The splits resample one finite "
             "dataset per cohort, so the intervals describe split-to-split and training-seed variability, not "
             "population sampling error. Source calibration = random half of PathMNIST val (5,002); every target is "
             "halved into a recalibration pool and a target-test half; all target numbers are on the target-test half.\n")
    n = M["n"]
    L.append("## Setup\n")
    L.append(f"* Source: PathMNIST 28x28 (NCT-CRC-HE-100K, Macenko-normalised at source): train {n['train']:,}, val {n['val']:,} "
             "(val halves: 5,002 calibration / 5,002 source-test per split).")
    L.append(f"* T1: PathMNIST test = CRC-VAL-HE-7K ({n['test_T1']:,} tiles; 50 patients disjoint from the 100K cohort, same "
             "tile size and pipeline). Recal pool 3,590 / target-test 3,590.")
    L.append(f"* T2 (independently acquired, non-colour-normalised cohort): Kather et al. 2016 texture tiles ({n['kather_total']:,} tiles, 10 slides "
             "CRC-Prim-HE-01..10, 150x150 px at 0.495 um/px, not colour-normalised), resized to 28x28 with PIL bicubic "
             "(antialiased). Label map verified against medmnist.INFO: 01_TUMOR->TUM, 04_LYMPHO->LYM, 06_MUCOSA->NORM, "
             f"07_ADIPOSE->ADI, 08_EMPTY->BACK (main, {n['kather_5class']:,} tiles; recal pool 1,562 / target-test 1,563); "
             f"02_STROMA->{{STR,MUS}}, 05_DEBRIS->{{DEB,MUC}} (set-valued sensitivity, {n['kather_7class']:,} tiles); "
             "03_COMPLEX excluded.")
    L.append(f"* Sensitivities on T2: set-valued 7 classes; EMPTY excluded (4 tissue classes, {n['kather_4class_noEMPTY']:,} "
             "tiles: Kather EMPTY is white glass while the PathMNIST BACK class is dark Macenko-normalised background); "
             "slide-grouped recal/test halves (no slide on both sides) and a class-balanced slide-grouped variant on the 4 "
             f"tissue classes; non-antialiased order-3 spline resize; physical-scale-matched resize ({SCALE_SIDE} px at "
             "~4 um/px, reflect-padded to 28, removing the 1.5x field-of-view mismatch); approximate Macenko stain "
             "harmonisation; scale-matched and Macenko-harmonised together.")
    L.append("* STRESS (secondary, synthetic): T1 images x per-channel scale (1.02, 0.99, 1.00), as in the submitted script.")
    L.append(f"* Models: submitted SmallCNN architecture and ResNet-18 (3x3 stride-1 stem, no max-pool), each trained 3x "
             f"(seeds 0-2) on the full train split, {M['training']['epochs']} epochs, {M['training']['optimizer']}, batch "
             f"{M['training']['batch']}, {M['training']['selection']}. Augmentation regimes: flips only (SmallCNN, "
             "ResNet18) and flips + stain-colour augmentation at two strengths (SmallCNN-aug, ResNet18-aug and "
             f"ResNet18-augstrong; HED jitter + brightness/contrast jitter with (a, b, c, d) = {AUG_CFG}, settings fixed a "
             "priori and not iterated). **The flips-only arms omit the stain augmentation and stain normalisation that are standard "
             "practice in computational pathology, so they are a deliberate worst case, not a recommended pipeline.** The "
             "cached submitted model (20k subsample, 6 CPU epochs, flips only) is re-evaluated as an audit.")
    L.append("* Conformal: LAC (1 - p_y) and randomised APS (Romano et al. 2020; empty sets allowed); finite-sample quantile; "
             "recalibration k in {0, 20, 30, 50, 100, 200, 400} from the recal pool, all evaluated on the same target-test half. "
             "k = 30 is the protocol headline m; k = 200 is the submitted value.")
    L.append("* Metrics: coverage, mean set size, empty-set rate, abstention (= non-singleton set = empty or size > 1; the "
             "literal 'size > 1' rate is reported separately as abstention_gt1 in the JSON), class-conditional coverage "
             "(min over classes with >= 10 test images, and the min restricted to the classes shared by all domains), "
             "accuracy, ECE (15 bins, max softmax), selective accuracy on the 50 % most confident (smallest non-empty set, "
             "ties by max softmax; empty sets ranked least confident).")
    L.append("* Split conformal is **marginal**: neither source calibration nor target recalibration targets "
             "class-conditional coverage, so worst-class coverage is below nominal in every domain, including the source "
             "itself. Worst-class numbers are therefore reported for source, T1 and T2 side by side and are never used as "
             "T2-specific evidence.\n")
    for a, Aa in res["architectures"].items():
        for t in Aa["training"]:
            if t and "final_epoch" in t:
                L.append(f"  - {a} seed {t['train_seed']}: final loss {t['final_epoch']['loss']:.3f}, train acc (no aug) "
                         f"{t['acc_train_noaug']:.3f}, val {t['acc_val']:.3f}, test {t['acc_test']:.3f}, {t['train_time_sec']:.0f} s on GPU")
    L.append("")
    L.append("## Accuracy of the trained models (full evaluation sets)\n")
    L.append("| model | " + " | ".join(ACC_HEAD) + " |")
    L.append("|---|" + "---|" * len(ACC_KEYS))
    for a, A in res["architectures"].items():
        acc = A["accuracy_full_sets"]
        L.append(f"| {a} | " + " | ".join(_f(acc[k]) for k in ACC_KEYS) + " |")
    L.append("\nChance level on the 5-class T2 subset is 0.20 (balanced 625 tiles per class) and 0.25 on the 4-class "
             "subset. A 1.5x magnification of PathMNIST val alone (last diagnostic column) costs a large part of the "
             "accuracy by itself, so the field-of-view mismatch matters even though matching it on T2 does not recover "
             "accuracy (see the scale-matched column).\n")
    order = ["S_source-test", "T1_CRC-VAL-HE-7K", "T2_Kather2016_5class", "T2sens_Kather2016_4class_noEMPTY",
             "T2sens_Kather2016_7class_setvalued", "T2sens_Kather2016_5class_slidegrouped",
             "T2sens_Kather2016_4class_slidegrouped", "T2sens_Kather2016_4class_slidegrouped_balanced",
             "T2sens_Kather2016_5class_spline3resize",
             "T2sens_Kather2016_5class_scalematched", "T2sens_Kather2016_5class_macenko",
             "T2sens_Kather2016_5class_scalematched_macenko", "STRESS_T1_synthetic_colour"]
    for meth in METHODS:
        L.append(f"## Source-calibrated ({meth}) and recalibrated coverage\n")
        L.append("k = 0: calibrated on the source half of val only. k = 30: protocol headline m. k = 200: value used in the "
                 "submitted manuscript. STRESS row = synthetic colour perturbation, secondary stress test only.\n")
        for a, A in res["architectures"].items():
            R = A["results"]
            L.append(f"### {a} ({meth})\n")
            L.append("| domain | coverage k=0 | set size k=0 | empty k=0 | abstention (non-singleton) k=0 | size>1 k=0 | ECE | "
                     "sel. acc 50% (vs acc) | coverage k=30 | set size k=30 | coverage k=200 | set size k=200 |")
            L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
            for name in order:
                r = R[name][meth]; r0 = r["0"]
                rk = r.get(str(K_HEAD)); r2 = r.get(str(K_SUBMITTED))
                L.append(f"| {name} | {_f(r0['coverage'])} | {_f(r0['set_size'], 2)} | {r0['empty']['mean']:.3f} | "
                         f"{r0['abstention']['mean']:.3f} | {r0['abstention_gt1']['mean']:.3f} | "
                         f"{r0['ece']['mean']:.3f} | {r0['sel_acc50']['mean']:.3f} (vs {r0['accuracy']['mean']:.3f}) | "
                         f"{_f(rk['coverage']) if rk else '-'} | {_f(rk['set_size'], 2) if rk else '-'} | "
                         f"{_f(r2['coverage']) if r2 else '-'} | {_f(r2['set_size'], 2) if r2 else '-'} |")
            L.append("")
    L.append("## Worst-class coverage, source vs T1 vs T2, side by side\n")
    L.append("Split conformal targets marginal coverage only, so worst-class coverage is below the nominal 0.90 in EVERY "
             "domain, including the in-distribution source half. The plain minimum is taken over the classes present in a "
             "domain (9 in source/T1, 5 or 4 in the Kather targets) and is therefore not comparable across domains; the "
             "shared-class columns restrict the minimum to classes that exist everywhere. These numbers say nothing "
             "specific about the external cohort and are not used as evidence about it.\n")
    L.append("| model | method | k | source: min(9) | source: min(shared 5) | T1: min(9) | T1: min(shared 5) | "
             "T2: min(5) | T2: min(shared 5) | T2 no-EMPTY: min(4) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    T1n, T2n, T2En = "T1_CRC-VAL-HE-7K", "T2_Kather2016_5class", "T2sens_Kather2016_4class_noEMPTY"
    for a, A in res["architectures"].items():
        for meth in METHODS:
            R = A["results"]
            for k in [0, K_SUBMITTED]:
                ks = str(k); k0 = "0"
                L.append(f"| {a} | {meth} | {k} | {_f(R['S_source-test'][meth][k0]['class_min_cov'])} | "
                         f"{_f(R['S_source-test'][meth][k0]['shared5_min_cov'])} | "
                         f"{_f(R[T1n][meth][ks]['class_min_cov'])} | {_f(R[T1n][meth][ks]['shared5_min_cov'])} | "
                         f"{_f(R[T2n][meth][ks]['class_min_cov'])} | {_f(R[T2n][meth][ks]['shared5_min_cov'])} | "
                         f"{_f(R[T2En][meth][ks]['class_min_cov'])} |")
    L.append("\n(The source column repeats the k = 0 value in both rows: the source calibration set is never recalibrated.)\n")
    L.append("## Slide-grouped recalibration and the class-mix confound\n")
    sg = M["slide_grouping"]
    L.append("Class-mix distance (total variation between the class distributions of the recalibration pool and the "
             f"target-test half, mean over the 20 seeds): tile-level 5-class {_f(sg['T2_tile_level_5class'])}, tile-level "
             f"4-class {_f(sg['T2noE_tile_level_4class'])}, random slide-grouped 5-class {_f(sg['T2grp_random_slides_5class'])}, "
             f"random slide-grouped 4-class {_f(sg['T2grpE_random_slides_4class'])}, class-balanced slide-grouped 4-class "
             f"{_f(sg['T2grpB_balanced_slides_4class'])}. Distinct slide partitions across the 20 seeds: "
             f"{sg['n_distinct_partitions']}, out of {sg['n_feasible_balanced_partitions']} feasible balanced bipartitions. "
             f"{sg['note']}\n")
    L.append("Coverage after k = 200 on the 4 tissue classes, tile-level vs random slide-grouped vs class-balanced "
             "slide-grouped (LAC / APS):\n")
    L.append("| model | tile-level | random slides | class-balanced slides |")
    L.append("|---|---|---|---|")
    for a, A in res["architectures"].items():
        R = A["results"]
        cells = []
        for n in ["T2sens_Kather2016_4class_noEMPTY", "T2sens_Kather2016_4class_slidegrouped",
                  "T2sens_Kather2016_4class_slidegrouped_balanced"]:
            cells.append(f"{_f(R[n]['LAC'][str(K_SUBMITTED)]['coverage'])} / {_f(R[n]['APS'][str(K_SUBMITTED)]['coverage'])}")
        L.append(f"| {a} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("## Paired coverage change versus the source-test half (same model, same split seed), k = 0\n")
    L.append("| model | domain | LAC d(coverage) | LAC d(class-min cov) | APS d(coverage) |")
    L.append("|---|---|---|---|---|")
    for a, A in res["architectures"].items():
        for name in order[1:]:
            d = A["paired_drop_vs_source"][name]
            L.append(f"| {a} | {name} | {_f(d['LAC']['coverage_k0'])} | {_f(d['LAC']['class_min_cov_k0'])} | {_f(d['APS']['coverage_k0'])} |")
    L.append("")
    L.append("## Label-shift control: macro coverage over the 5 classes shared by all domains (ADI, BACK, LYM, NORM, TUM), k = 0\n")
    L.append("| model | method | source-test | T1 | T2 | T2 Macenko |")
    L.append("|---|---|---|---|---|---|")
    for a, A in res["architectures"].items():
        for meth in METHODS:
            R = A["results"]
            L.append(f"| {a} | {meth} | " + " | ".join(_f(R[n][meth]["0"]["shared5_macro_cov"]) for n in
                     ["S_source-test", "T1_CRC-VAL-HE-7K", "T2_Kather2016_5class", "T2sens_Kather2016_5class_macenko"]) + " |")
    L.append("")
    L.append("## Recalibration k-sweep, LAC coverage on the target-test half (mean [2.5, 97.5])\n")
    L.append("| model | domain | " + " | ".join(f"k={k}" for k in KGRID) + " |")
    L.append("|---|---|" + "---|" * len(KGRID))
    for a, A in res["architectures"].items():
        for name in order[1:]:
            r = A["results"][name]["LAC"]
            L.append(f"| {a} | {name} | " + " | ".join(_f(r[str(k)]["coverage"]) if str(k) in r else "-" for k in KGRID) + " |")
    L.append("\nMean LAC set size (of 9 classes) on the target-test half:\n")
    L.append("| model | domain | " + " | ".join(f"k={k}" for k in KGRID) + " |")
    L.append("|---|---|" + "---|" * len(KGRID))
    for a, A in res["architectures"].items():
        for name in order[1:]:
            r = A["results"][name]["LAC"]
            L.append(f"| {a} | {name} | " + " | ".join(f"{r[str(k)]['set_size']['mean']:.2f}" if str(k) in r else "-" for k in KGRID) + " |")
    L.append("\nTheoretical 5/50/95 % quantiles of test coverage given k calibration points (Beta law): " +
             "; ".join(f"k={k}: " + "/".join(f"{v:.3f}" for v in M["beta_theory_5_50_95"][str(k)]) for k in KGRID if k > 0) + "\n")
    L.append("## Per-class (Kather folder) coverage on T2, LAC, k = 0 and k = 200\n")
    L.append("| model | class | k=0 | k=200 |")
    L.append("|---|---|---|---|")
    for a, A in res["architectures"].items():
        r = A["results"]["T2_Kather2016_5class"]["LAC"]
        for c in r["0"]["class_cov"]:
            L.append(f"| {a} | {c} | {_f(r['0']['class_cov'][c])} | {_f(r[str(K_SUBMITTED)]['class_cov'].get(c))} |")
    L.append("")
    L.append("## Where the Kather-2016 tiles go (fraction predicted as each PathMNIST class, mean over training seeds)\n")
    L.append("| model | Kather folder | " + " | ".join(CODES) + " |")
    L.append("|---|---|" + "---|" * len(CODES))
    for a, A in res["architectures"].items():
        for f, v in A["kather_confusion_mean"].items():
            L.append(f"| {a} | {f} | " + " | ".join(f"{x:.2f}" for x in v) + " |")
    L.append("\nSame, after Macenko harmonisation (sensitivity):\n")
    L.append("| model | Kather folder | " + " | ".join(CODES) + " |")
    L.append("|---|---|" + "---|" * len(CODES))
    for a, A in res["architectures"].items():
        for f, v in A["kather_confusion_mean_macenko"].items():
            L.append(f"| {a} | {f} | " + " | ".join(f"{x:.2f}" for x in v) + " |")
    L.append(f"\nMacenko normalisation succeeded (>= {MAC_MIN_PIX} tissue pixels) per folder [normalised, total]: "
             f"{res['meta']['macenko']['kather_normalised_per_folder']}; PathMNIST val (28x28): {res['meta']['macenko']['val_normalised']}.\n")
    L.append("## Cohort image statistics (28x28 uint8)\n")
    L.append("| cohort | n | mean RGB | sd RGB | mean abs Laplacian |")
    L.append("|---|---|---|---|---|")
    for k, v in M["cohort_image_stats"].items():
        L.append(f"| {k} | {v['n']} | {v['mean_rgb']} | {v['sd_rgb']} | {v['mean_abs_laplacian']:.2f} |")
    L.append("\nThe high-frequency content of PathMNIST (mean |Laplacian| ~31) is closer to the antialiased PIL-bicubic "
             "resize of the Kather tiles (~37) than to the non-antialiased spline (~60), consistent with MedMNIST having used an "
             "antialiased cubic filter (heuristic only: tile fields of view differ). Kather tiles are darker, more saturated "
             "and have ~2x the pixel SD of PathMNIST (no colour normalisation).\n")
    L.append(interpretation(res))
    L.append(VERIFIER_SECTION)
    open(os.path.join(RES, f"{TAG}.md"), "w", encoding="utf-8").write("\n".join(L))


VERIFIER_SECTION = """
## Changes made after independent verification

An independent verifier reproduced the analysis (identical JSON apart from the runtime field, bit-identical softmax arrays
on an independent retrain of two models) and confirmed the protocol, the conformal implementations and the audit of the
submitted analysis. All of its substantive objections concerned interpretation, and all of them were adopted:

1. *Worst-class coverage was presented as T2-specific evidence.* It is not: split conformal is marginal, worst-class
   coverage is below nominal in the source domain as well, and on T1 it is in fact lower than on T2; the minima were also
   taken over different numbers of classes. The T2 argument now rests on set size, worst-class coverage is reported for
   source / T1 / T2 side by side with shared-class columns, and the marginal nature of the guarantee is stated. No
   class-conditional (Mondrian) arm was added because class-conditional validity is nowhere claimed.
2. *"Genuine external cohort" contradicted our own caveat.* T2 is now described consistently as an independently acquired,
   non-colour-normalised cohort (different acquisition period and tile geometry, same archive); "genuine" is used only as
   the opposite of "synthetic".
3. *A worse-than-chance classifier was framed as general CNN behaviour.* The flips-only arms are now labelled a deliberate
   worst case, and three colour-augmented arms (SmallCNN-aug, ResNet18-aug, ResNet18-augstrong; HED + brightness/contrast
   jitter at two strengths, settings fixed a priori and not iterated) were trained and are reported with equal weight,
   alongside the Macenko-harmonised arm and the T1 APS numbers. Colour augmentation changes the size of the coverage loss
   a great deal (T2 accuracy 0.13 -> 0.72, LAC coverage 0.24 -> 0.42 and APS 0.37 -> 0.78 for the best arm) without
   restoring validity, so every statement about the drop now names the training pipeline.
4. *"Bounds rather than isolates the colour-pipeline contribution" was unsupported.* The imperfect normaliser only
   approximates that contribution; the direction of the error is unknown, and the text now says so.
5. *Missing sensitivities.* Two were added as full analysis arms: T2 without the EMPTY class (white glass vs the dark
   Macenko-normalised BACK class of NCT-CRC-HE-100K) and a physical-scale-matched resize, plus the combination of scale
   matching with Macenko harmonisation and a 1.5x-magnified PathMNIST val diagnostic.
6. *Figure readiness.* "abstention" now means "the set is not a singleton" (= empty + size > 1); the literal size > 1 rate
   is kept as "abstention_gt1". Under LAC a confident model produces empty rather than large sets, so the literal
   definition alone would show ResNet-18 as never abstaining while most of its external sets are empty.
7. *Slide-grouped split confound.* The class-mix distance between the two halves is now reported, the structural reason is
   stated (EMPTY tiles come from only two slides, so no slide-disjoint split of the 5-class set can be class-balanced), the
   number of distinct partitions across seeds is given, and two further arms were added on the 4 tissue classes: a random
   slide-grouped control and a class-balanced slide-grouped variant. The controlled comparison changes the conclusion:
   most of the mean undercoverage under slide grouping was label shift between the halves, and what remains once the class
   mix is matched is a much wider spread of coverage across slide partitions, not a systematic bias.
8. *"Once the model is adequately trained" contradicted the under-fitting caveat.* Changed to "better-trained", with the
   under-fitting of the SmallCNN arms restated.
9. *Reproducibility wording.* The verifier's independent retrain of SmallCNN seed 0 and ResNet18 seed 0 reproduced
   bit-identical softmax arrays on this machine; the text says so instead of only warning that GPU training may differ.

No verifier issue was judged invalid, so there is nothing under "Verifier issues not adopted".
"""


def interpretation(res):
    """Plain-language reading generated from the numbers (no hand-typed values)."""
    A = res["architectures"]
    g = lambda a, n, meth, k, m: A[a]["results"][n][meth][str(k)][m]["mean"]
    rng_ = lambda a, n, meth, k, m: A[a]["results"][n][meth][str(k)][m]
    acc = lambda a, key: A[a]["accuracy_full_sets"][key]["mean"]
    s, r, o = "SmallCNN", "ResNet18", "SmallCNN-submitted"
    sa, ra, rs = "SmallCNN-aug", "ResNet18-aug", "ResNet18-augstrong"
    T1, T2 = "T1_CRC-VAL-HE-7K", "T2_Kather2016_5class"
    T2g, T2m, ST = "T2sens_Kather2016_5class_slidegrouped", "T2sens_Kather2016_5class_macenko", "STRESS_T1_synthetic_colour"
    T2set = "T2sens_Kather2016_7class_setvalued"
    T2gB, T2E, T2sc = ("T2sens_Kather2016_4class_slidegrouped_balanced", "T2sens_Kather2016_4class_noEMPTY",
                       "T2sens_Kather2016_5class_scalematched")
    T2gE = "T2sens_Kather2016_4class_slidegrouped"
    T2scm = "T2sens_Kather2016_5class_scalematched_macenko"
    f3 = lambda x: f"{x:.3f}"
    pr = lambda d: f"{d['mean']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]"
    kh, ks = K_HEAD, K_SUBMITTED
    L = ["## Plain reading (alpha = 0.10; mean [2.5, 97.5 percentile] over splits x training seeds)", ""]
    L.append("**Does the independently acquired cohort degrade coverage? Yes, for every model, and far more than the "
             "synthetic perturbation used in the submitted Figure 9. How far below nominal coverage falls depends strongly "
             "on how the classifier was trained: from a total collapse for the models trained without colour augmentation "
             "(a deliberate worst case) to a large but partial loss for the colour-augmented ones. No training pipeline we "
             "tried keeps source-calibrated coverage near nominal on this cohort.**")
    L.append("")
    L.append(f"1. In-distribution (source-test half of val), source-calibrated LAC coverage is "
             f"{f3(g(s, 'S_source-test', 'LAC', 0, 'coverage'))} (SmallCNN) and "
             f"{f3(g(r, 'S_source-test', 'LAC', 0, 'coverage'))} (ResNet-18), as guaranteed.")
    L.append(f"2. T2, Kather et al. 2016 (5 unambiguous classes), flips-only models: source-calibrated LAC coverage {pr(rng_(s, T2, 'LAC', 0, 'coverage'))} "
             f"(SmallCNN), {pr(rng_(r, T2, 'LAC', 0, 'coverage'))} (ResNet-18) and {pr(rng_(o, T2, 'LAC', 0, 'coverage'))} "
             f"(submitted model); APS {f3(g(s, T2, 'APS', 0, 'coverage'))} / {f3(g(r, T2, 'APS', 0, 'coverage'))} / "
             f"{f3(g(o, T2, 'APS', 0, 'coverage'))}. Accuracy collapses to {f3(acc(s, T2))} / {f3(acc(r, T2))} / {f3(acc(o, T2))} "
             f"with ECE {f3(g(s, T2, 'LAC', 0, 'ece'))} / {f3(g(r, T2, 'LAC', 0, 'ece'))} / {f3(g(o, T2, 'LAC', 0, 'ece'))}: "
             "the classifiers assign the unnormalised external tiles to the wrong tissue classes with high confidence "
             "(e.g. TUMOR and LYMPHO tiles -> BACK/DEB, EMPTY -> MUS for the flips-only models; see the confusion table). "
             f"The set-valued 7-class sensitivity gives {f3(g(s, T2set, 'LAC', 0, 'coverage'))} / "
             f"{f3(g(r, T2set, 'LAC', 0, 'coverage'))}, and the non-antialiased resize gives the same picture. For ResNet-18 "
             f"the source threshold is extreme (val accuracy {f3(acc(r, 'val'))} > 1 - alpha), so "
             f"{f3(g(r, T2, 'LAC', 0, 'empty'))} of external LAC sets are empty; an empty set is itself a usable "
             "'do not trust' signal, but it is not coverage.")
    L.append(f"2b. **These two models are a deliberate worst case**: they were trained with flips only, i.e. without the "
             f"stain-colour augmentation and stain normalisation that are standard in computational pathology, and their T2 "
             f"accuracy ({f3(acc(s, T2))} / {f3(acc(r, T2))}) is below the 0.20 chance level of the balanced 5-class task. "
             f"Training the same architectures with HED stain jitter and brightness/contrast jitter (settings fixed a "
             f"priori, two strengths) gives T2 accuracy {f3(acc(sa, T2))} (SmallCNN-aug), {f3(acc(ra, T2))} (ResNet18-aug) "
             f"and {f3(acc(rs, T2))} (ResNet18-augstrong) at val accuracy {f3(acc(sa, 'val'))} / {f3(acc(ra, 'val'))} / "
             f"{f3(acc(rs, 'val'))} and T1 accuracy {f3(acc(sa, 'test_T1'))} / {f3(acc(ra, 'test_T1'))} / "
             f"{f3(acc(rs, 'test_T1'))}. Source-calibrated coverage on T2 for these models is LAC "
             f"{pr(rng_(sa, T2, 'LAC', 0, 'coverage'))} / {pr(rng_(ra, T2, 'LAC', 0, 'coverage'))} / "
             f"{pr(rng_(rs, T2, 'LAC', 0, 'coverage'))} and APS {f3(g(sa, T2, 'APS', 0, 'coverage'))} / "
             f"{f3(g(ra, T2, 'APS', 0, 'coverage'))} / {f3(g(rs, T2, 'APS', 0, 'coverage'))}, against "
             f"{f3(g(sa, 'S_source-test', 'LAC', 0, 'coverage'))} / {f3(g(ra, 'S_source-test', 'LAC', 0, 'coverage'))} / "
             f"{f3(g(rs, 'S_source-test', 'LAC', 0, 'coverage'))} in-distribution and {f3(g(sa, T1, 'LAC', 0, 'coverage'))} / "
             f"{f3(g(ra, T1, 'LAC', 0, 'coverage'))} / {f3(g(rs, T1, 'LAC', 0, 'coverage'))} on T1. "
             "Colour augmentation therefore changes how far coverage falls, but does not restore validity under this shift; "
             "any statement about the size of the drop has to name the training pipeline.")
    L.append(f"3. T1, CRC-VAL-HE-7K, is not coverage-neutral for the better-trained models (it is for the under-trained "
             f"submitted model): LAC "
             f"{f3(g(s, T1, 'LAC', 0, 'coverage'))} (SmallCNN) and {f3(g(r, T1, 'LAC', 0, 'coverage'))} (ResNet-18); APS "
             f"{f3(g(s, T1, 'APS', 0, 'coverage'))} / {f3(g(r, T1, 'APS', 0, 'coverage'))}; worst-class LAC coverage "
             f"{f3(g(s, T1, 'LAC', 0, 'class_min_cov'))} / {f3(g(r, T1, 'LAC', 0, 'class_min_cov'))}. The submitted "
             f"0.898 vs 0.895 is reproduced only by the submitted model ({f3(g(o, 'S_source-test', 'LAC', 0, 'coverage'))} vs "
             f"{f3(g(o, T1, 'LAC', 0, 'coverage'))}), whose accuracy is {f3(acc(o, 'val'))} on val and "
             f"{f3(acc(o, 'test_T1'))} on T1, i.e. a model that was equally poor in both domains. The LAC/APS gap on T1 for "
             "ResNet-18 shows that the size of a measured coverage drop depends on the score function, not only on the shift.")
    L.append(f"4. Target recalibration restores marginal coverage on T2 (k = {kh}: {f3(g(s, T2, 'LAC', kh, 'coverage'))} / "
             f"{f3(g(r, T2, 'LAC', kh, 'coverage'))}; k = {ks}: {f3(g(s, T2, 'LAC', ks, 'coverage'))} / "
             f"{f3(g(r, T2, 'LAC', ks, 'coverage'))}; colour-augmented models {f3(g(sa, T2, 'LAC', ks, 'coverage'))} / "
             f"{f3(g(ra, T2, 'LAC', ks, 'coverage'))} at k = {ks}), **but the sets it returns are the point**: mean size "
             f"{g(s, T2, 'LAC', ks, 'set_size'):.2f} / {g(r, T2, 'LAC', ks, 'set_size'):.2f} of 9 classes for the "
             f"flips-only models and {g(sa, T2, 'LAC', ks, 'set_size'):.2f} / {g(ra, T2, 'LAC', ks, 'set_size'):.2f} / "
             f"{g(rs, T2, 'LAC', ks, 'set_size'):.2f} for the colour-augmented ones, versus "
             f"{g(s, T1, 'LAC', ks, 'set_size'):.2f} / {g(r, T1, 'LAC', ks, 'set_size'):.2f} / "
             f"{g(sa, T1, 'LAC', ks, 'set_size'):.2f} / {g(ra, T1, 'LAC', ks, 'set_size'):.2f} / "
             f"{g(rs, T1, 'LAC', ks, 'set_size'):.2f} on T1 at the same k. Recalibration on T2 therefore buys validity at a "
             "price that scales with the shift: close to the whole label space for the models trained without colour "
             "augmentation, and 2-3 classes (about triple the T1 size) for the colour-augmented ones. It turns a silent "
             "failure into an explicit statement of ignorance rather than rescuing the classifier. Worst-class coverage is NOT "
             "part of this argument: split conformal is marginal, so worst-class coverage is below nominal in the source "
             "domain too, and on T1 it is in fact lower than on T2 (see the side-by-side worst-class table above).")
    L.append(f"5. Recalibration relies on exchangeability between the labelled target images and the deployment images. When "
             f"recalibration tiles and test tiles come from disjoint slides (slide-grouped T2), coverage after k = {ks} is "
             f"{pr(rng_(s, T2g, 'LAC', ks, 'coverage'))} (SmallCNN) and {pr(rng_(r, T2g, 'LAC', ks, 'coverage'))} "
             "(ResNet-18): on average below nominal and highly variable across splits. Most of that mean shortfall is a "
             "class-mix artefact, not a slide effect: a random slide cut also changes the class mix between the halves "
             "(LYMPHO is concentrated in slides 01/02/05/09, MUCOSA is absent from 01/02/04/10, EMPTY comes only from slides "
             "06 and 10; class-mix total variation about "
             f"{res['meta']['slide_grouping']['T2grp_random_slides_5class']['mean']:.2f} against "
             f"{res['meta']['slide_grouping']['T2_tile_level_5class']['mean']:.2f} for tile-level splits). On the 4 tissue "
             f"classes, k = {ks} coverage is {pr(rng_(s, T2E, 'LAC', ks, 'coverage'))} / {pr(rng_(r, T2E, 'LAC', ks, 'coverage'))} "
             f"tile-level, {pr(rng_(s, T2gE, 'LAC', ks, 'coverage'))} / {pr(rng_(r, T2gE, 'LAC', ks, 'coverage'))} with random "
             f"slide grouping and {pr(rng_(s, T2gB, 'LAC', ks, 'coverage'))} / {pr(rng_(r, T2gB, 'LAC', ks, 'coverage'))} with "
             "class-balanced slide grouping (SmallCNN / ResNet-18). So when the class mix is matched the MEAN returns close "
             "to nominal, while the spread across slide partitions stays much wider than for tile-level splits. The "
             "practical reading: recalibrating on other slides is unbiased only if the labelled slides have the same class "
             "mix as the deployment slides, and even then a single small slide sample gives coverage that can land far from "
             "nominal. With only 10 slides the partitions overlap heavily, so these intervals themselves rest on few "
             "effectively independent configurations.")
    L.append(f"6. What drives the T2 collapse (sensitivities, flips-only models). (a) Stain harmonisation: approximate "
             f"Macenko normalisation of the external tiles raises accuracy to {f3(acc(s, T2m))} / {f3(acc(r, T2m))} and "
             f"source-calibrated coverage to LAC {f3(g(s, T2m, 'LAC', 0, 'coverage'))} / {f3(g(r, T2m, 'LAC', 0, 'coverage'))} "
             f"and APS {f3(g(s, T2m, 'APS', 0, 'coverage'))} / {f3(g(r, T2m, 'APS', 0, 'coverage'))} -- better, still far "
             "below nominal. Our Macenko reimplementation does not reproduce the NCT pipeline exactly (applied to 28x28 "
             f"source val it lowers accuracy to {f3(acc(s, 'val_macenko_diagnostic'))} / "
             f"{f3(acc(r, 'val_macenko_diagnostic'))}), so it only approximates the colour-pipeline contribution and the "
             "direction of that error is unknown. (b) Field of view: matching the physical scale (150 px -> "
             f"{SCALE_SIDE} px at ~4 um/px, reflect-padded) leaves the collapse essentially unchanged (accuracy "
             f"{f3(acc(s, T2sc))} / {f3(acc(r, T2sc))}; LAC coverage {f3(g(s, T2sc, 'LAC', 0, 'coverage'))} / "
             f"{f3(g(r, T2sc, 'LAC', 0, 'coverage'))}), and combining both corrections gives accuracy "
             f"{f3(acc(s, T2scm))} / {f3(acc(r, T2scm))} and LAC coverage {f3(g(s, T2scm, 'LAC', 0, 'coverage'))} / "
             f"{f3(g(r, T2scm, 'LAC', 0, 'coverage'))}, so the 1.5x magnification is not the main driver even though "
             f"magnifying PathMNIST val itself by 1.5x costs accuracy ({f3(acc(s, 'val'))} -> "
             f"{f3(acc(s, 'val_zoom1.5x_diagnostic'))} and {f3(acc(r, 'val'))} -> {f3(acc(r, 'val_zoom1.5x_diagnostic'))}). "
             "(c) Class definition: Kather 08_EMPTY is white glass (mean RGB "
             f"{res['meta']['cohort_image_stats']['Kather2016_EMPTY_bicubic']['mean_rgb']}) while the PathMNIST BACK class "
             f"is dark Macenko-normalised background ({res['meta']['cohort_image_stats']['PathMNIST_train_BACK_class']['mean_rgb']}), "
             f"so EMPTY->BACK is arguably a different class. Dropping it (4 tissue classes) leaves the trained models at "
             f"accuracy {f3(acc(s, T2E))} / {f3(acc(r, T2E))} and LAC coverage {f3(g(s, T2E, 'LAC', 0, 'coverage'))} / "
             f"{f3(g(r, T2E, 'LAC', 0, 'coverage'))}, i.e. the collapse holds, but the submitted model loses most of its "
             f"apparent T2 advantage (accuracy {f3(acc(o, T2))} -> {f3(acc(o, T2E))}, LAC coverage "
             f"{f3(g(o, T2, 'LAC', 0, 'coverage'))} -> {f3(g(o, T2E, 'LAC', 0, 'coverage'))}), which came largely from "
             "mapping white tiles onto BACK. (d) The same sensitivities for the best colour-augmented model "
             f"(ResNet18-augstrong): T2 accuracy {f3(acc(rs, T2))} rises to {f3(acc(rs, T2m))} (Macenko), "
             f"{f3(acc(rs, T2sc))} (scale-matched) and {f3(acc(rs, T2scm))} (both), with source-calibrated LAC coverage "
             f"{f3(g(rs, T2, 'LAC', 0, 'coverage'))} -> {f3(g(rs, T2m, 'LAC', 0, 'coverage'))} / "
             f"{f3(g(rs, T2sc, 'LAC', 0, 'coverage'))} / {f3(g(rs, T2scm, 'LAC', 0, 'coverage'))} and APS "
             f"{f3(g(rs, T2, 'APS', 0, 'coverage'))} -> {f3(g(rs, T2scm, 'APS', 0, 'coverage'))}. Even with standard "
             "colour augmentation at training time AND stain harmonisation AND scale matching at test time, coverage stays "
             "clearly below the nominal 0.90, so the residual shift is not fully explained by colour, scale or the EMPTY "
             "class.")
    L.append(f"7. SECONDARY synthetic stress test (T1 x per-channel scale (1.02, 0.99, 1.00)): for the models trained here it "
             f"adds little beyond the T1 cohort shift (LAC {f3(g(s, T1, 'LAC', 0, 'coverage'))} -> "
             f"{f3(g(s, ST, 'LAC', 0, 'coverage'))} SmallCNN; {f3(g(r, T1, 'LAC', 0, 'coverage'))} -> "
             f"{f3(g(r, ST, 'LAC', 0, 'coverage'))} ResNet-18; accuracy {f3(acc(s, 'test_T1'))} -> {f3(acc(s, 'stress_T1'))} "
             f"and {f3(acc(r, 'test_T1'))} -> {f3(acc(r, 'stress_T1'))}), and for the colour-augmented models it is almost "
             f"invisible ({f3(g(sa, T1, 'LAC', 0, 'coverage'))} -> {f3(g(sa, ST, 'LAC', 0, 'coverage'))}, "
             f"{f3(g(ra, T1, 'LAC', 0, 'coverage'))} -> {f3(g(ra, ST, 'LAC', 0, 'coverage'))}, "
             f"{f3(g(rs, T1, 'LAC', 0, 'coverage'))} -> {f3(g(rs, ST, 'LAC', 0, 'coverage'))}). The submitted collapse to "
             f"{f3(g(o, ST, 'LAC', 0, 'coverage'))} (accuracy {f3(acc(o, 'test_T1'))} -> {f3(acc(o, 'stress_T1'))}) reflects "
             "the fragility of the under-trained submitted model to a 2 % colour change, not a severe stain shift, and should "
             "not be presented as the main imaging evidence.")
    L.append(f"8. Selective prediction: on the source-test half the 50 % most confident images (smallest non-empty set) reach "
             f"accuracy {f3(g(s, 'S_source-test', 'LAC', 0, 'sel_acc50'))} / {f3(g(r, 'S_source-test', 'LAC', 0, 'sel_acc50'))} "
             f"(flips-only) and {f3(g(rs, 'S_source-test', 'LAC', 0, 'sel_acc50'))} (ResNet18-augstrong), but on T2 without "
             f"recalibration only {f3(g(s, T2, 'LAC', 0, 'sel_acc50'))} / {f3(g(r, T2, 'LAC', 0, 'sel_acc50'))} for the "
             f"flips-only models. For the colour-augmented models triage still works on T2 "
             f"({f3(g(sa, T2, 'LAC', 0, 'sel_acc50'))} / {f3(g(ra, T2, 'LAC', 0, 'sel_acc50'))} / "
             f"{f3(g(rs, T2, 'LAC', 0, 'sel_acc50'))} against overall accuracy {f3(acc(sa, T2))} / {f3(acc(ra, T2))} / "
             f"{f3(acc(rs, T2))}), so confidence ranking degrades gracefully where the classifier retains signal and fails "
             "completely where it does not -- another reason to report the training pipeline with the number.")
    L.append("")
    tr0 = A[s]["training"][0].get("acc_train_noaug", float("nan"))
    L.append("Caveats. (i) Kather 2016 tiles come from the University Medical Center Mannheim pathology archive, which also "
             "contributed to NCT-CRC-HE-100K; public metadata do not exclude patient overlap, so T2 is an independently "
             "acquired cohort (different period, no colour normalisation, 150 px / 74 um tiles vs 224 px / 112 um), NOT a "
             "proven different institution. The word 'genuine' is used in this report only as the opposite of 'synthetic'. "
             "(ii) The shift bundles colour pipeline, acquisition period, magnification and the EMPTY/BACK class definition; "
             "item 6 separates them as far as the data allow, and the colour pipeline is the largest single factor. "
             "(iii) T1 has no patient identifiers, so its halves are tile-level; T2 has slide-grouped sensitivities, both "
             "confounded as described in item 5. (iv) PathMNIST val shares patients with train (random 9:1 split of "
             "NCT-CRC-HE-100K), so 'in-distribution' is patient-overlapping. (v) The SmallCNN arms (no batch norm, 15 epochs, "
             f"fixed a-priori settings) under-fit (train accuracy without augmentation {tr0:.3f} for seed 0); hyperparameters "
             "were not tuned because val is the calibration set. (vi) The flips-only arms omit stain augmentation and stain "
             "normalisation on purpose and are a worst case; the -aug arms are the realistic pipeline. (vii) All coverage "
             "statements are marginal; nothing here claims class-conditional validity, which marginal split conformal does "
             "not provide in any domain.")
    L.append("")
    L.append("Suggested framing for the manuscript: on an independently acquired, non-colour-normalised histology cohort "
             "(Kather 2016: different acquisition period and tile geometry, same archive), source-calibrated conformal "
             "coverage falls far below nominal for every model we trained. How far depends on the training pipeline: "
             f"without stain augmentation (a deliberate worst case) LAC coverage is {f3(g(s, T2, 'LAC', 0, 'coverage'))} / "
             f"{f3(g(r, T2, 'LAC', 0, 'coverage'))}, with stain augmentation {f3(g(sa, T2, 'LAC', 0, 'coverage'))} / "
             f"{f3(g(ra, T2, 'LAC', 0, 'coverage'))} / {f3(g(rs, T2, 'LAC', 0, 'coverage'))}, and after approximate stain "
             "harmonisation of the external tiles "
             f"{f3(g(s, T2m, 'LAC', 0, 'coverage'))} / {f3(g(r, T2m, 'LAC', 0, 'coverage'))} -- against "
             f"{f3(g(r, T1, 'APS', 0, 'coverage'))} (APS) for ResNet-18 on the patient-disjoint NCT cohort T1 at "
             f"{f3(acc(r, 'test_T1'))} accuracy. A few dozen labelled target images restore marginal coverage in all cases, "
             f"but at a price that scales with the shift: on T2 the recalibrated sets hold "
             f"{g(rs, T2, 'LAC', ks, 'set_size'):.1f} classes of 9 for the best colour-augmented model and "
             f"{g(r, T2, 'LAC', ks, 'set_size'):.1f} for the worst flips-only model, against "
             f"{g(rs, T1, 'LAC', ks, 'set_size'):.1f} on T1 -- the trust layer converts a silent failure into an explicit "
             "statement of ignorance rather than rescuing the model. Recalibration drawn from other slides of the same site "
             "is unbiased only when those slides have the same class mix, and is in any case highly variable. The synthetic 2 % colour perturbation of the "
             "submitted Figure 9 should be presented as what it is -- a small perturbation that only a badly under-trained "
             "model was sensitive to -- and not as the evidence for shift sensitivity.")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("train", "all"):
        run_train()
    if stage in ("analyze", "all"):
        run_analyze()
