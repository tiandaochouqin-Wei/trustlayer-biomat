"""Pilot verification labels for every regex candidate (single verifier).

The verifier (the assistant acting as the 'LLM verification' step, reading
each sentence with its section label; ambiguous cases were checked against the
surrounding full text) assigned each candidate one verdict:
  TP  the sentence reports that THIS study's ML model did the coded thing
  FP  otherwise; fp_category names the false-positive mechanism
`supports` is the code the sentence supports after verification (may differ
from the pattern family, e.g. a U3 regex hit that only supports U2).
Labels are keyed by row order of pilot/screen_hits.csv and checked against the
stored pmcid + pattern so that a changed screen cannot silently misalign them.
"""
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent

TL = ("FP", "", "TRANSFER_LEARNING_ADAPTATION")
L = {}
def put(idx, verdict, supports="", cat="", note=""):
    for i in (idx if isinstance(idx, (list, range)) else [idx]):
        L[i] = (verdict, supports, cat, note)

put(0, "TP", "S1", "", "trained with Poisson-noise simulations, tested under Gaussian noise (synthetic shift)")
put(1, "FP", "", "HOMONYM", "image sharpness")
put(2, "FP", "", "HOMONYM", "material fabrication, not validation of predictions")
put(3, "FP", "", "NOT_AN_EVALUATION", "possible future uses of dataset")
put([4, 5], "FP", "", "LIMITATION_OR_FUTURE_WORK", "explicitly states cross-lab benchmarking NOT done (informative negative for E/S)")
put([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22], *TL[:2], TL[2],
    "fine-tuning on target data; not an evaluation of the unadapted model under shift")
put(19, "FP", "", "HOMONYM", "generative adversarial network")
put([23, 24], "TP", "S1", "", "accuracy re-tested under particle-height / coverslip-thickness variation")
put([25, 27, 28, 29], "FP", "", "ANALYTICAL_CALIBRATION_CURVE", "assay calibration curve")
put(26, "FP", "", "MEASUREMENT_REPLICATE_SPREAD", "assay replicate SD")
put([30, 31, 32, 33, 34], "FP", "", "HOMONYM", "perturbation theory (IFPTML)")
put(35, "FP", "", "NOT_AN_EVALUATION", "compounds tested in source assays")
put(36, "FP", "", "HOMONYM", "genetic drift in GA")
put(37, "TP", "EP", "", "de novo sequences synthesised and measured")
put([38, 39, 40], "FP", "", "HOMONYM", "generative adversarial network")
put(41, "FP", "", "NOT_AN_EVALUATION", "generic 'robust against overfitting'")
put(42, "FP", "", "SAMPLE_WISE_LOOCV", "leave-one-out CV is not a group/shift split")
put(43, "FP", "", "OTHER_WORK_CITED", "")
put([44, 45, 46, 47], "FP", "", "ANALYTICAL_CALIBRATION_CURVE", "signal-to-angle calibration")
put(48, "FP", "", "HOMONYM", "vascular aging")
put(49, "FP", "", "HOMONYM", "GA mutation standard deviation (hyper-parameter)")
put(50, "FP", "", "ANALYTICAL_CALIBRATION_CURVE", "release-assay calibration")
put(51, "FP", "", "HOMONYM", "ageing population")
put([52, 53], "TP", "S1", "", "explicit test of spatial (input-size) extrapolation; borderline")
put(54, "FP", "", "GENERIC_DISCUSSION", "known issue of temporal extrapolation, not tested")
put(55, "FP", "", "HOMONYM", "SEM = scanning electron microscope")
put([56, 58, 59, 60, 61, 62], "TP", "EP", "", "Pareto-optimal conditions run and compared with model")
put(57, "FP", "", "MEASUREMENT_REPLICATE_SPREAD", "")
put(63, "FP", "", "BACKGROUND_GENERAL", "generic BO description with citation")
put([64, 65, 66, 67], "TP", "U2", "", "GP surrogate in BO (uncertainty used internally)")
put(68, "FP", "", "NOT_ML", "experimental caution about protein concentration")
put(69, "TP", "EP", "", "")
put(70, "FP", "", "MEASUREMENT_REPLICATE_SPREAD", "replicates fabricated and tested")
put([71, 72, 73, 74, 75], "TP", "U2", "", "EI/MPI/LCB acquisition (GPyOpt GP surrogate)")
put(76, "TP", "EP", "", "conclusion restating own experimental verification")
put([77, 78], "FP", "", "BACKGROUND_GENERAL", "")
put(79, "FP", "U2", "CLAIM_WITHOUT_ASSESSMENT", "'well-calibrated uncertainties' asserted for GP, calibration never assessed -> supports U2 only")
put([80, 81, 82, 84, 85, 90, 92], "TP", "U2", "", "acquisition balancing exploration/exploitation")
put(83, "TP", "U2", "", "posterior variance")
put([86, 87, 88, 89, 91, 93, 94, 95, 96, 97], "TP", "U1", "", "RMSE +/- SD from cross-validation")
put([98, 99, 101, 104, 105, 106, 107, 108, 109, 111, 113, 114, 116, 119, 120, 121, 122, 123,
     124, 125, 126, 127, 128, 129, 130, 132, 133, 134, 135], "TP", "U2", "", "GPR surrogate (per-prediction SD available/used)")
put([100, 102, 103, 112, 115, 117, 118], "TP", "U2", "", "prediction uncertainty used for active learning / reported")
put(110, "FP", "", "HYPERPARAMETER_BO", "BO used only to tune GPR hyper-parameters")
put(131, "TP", "U2", "", "table of GPR mean +/- SD vs experiment (also EP)")


# Candidates that only the v1.1 screen proposes, keyed by (pmcid, pattern, sentence start).
NEW_V11 = {
    ("PMC10161767", "E_external", "Phase I: simulated data"): ("TP", "E1", "", "simulation-trained model tested on real experimental data"),
    ("PMC8665348", "S_group", "The model also performed great"): ("TP", "S1", "", "own result summarised at end of Introduction"),
    ("PMC8665348", "U1_spread", "To thoroughly evaluate the practical"): ("FP", "", "PROCEDURE_ONLY", "describes 5-fold CV set-up, no spread reported here"),
    ("PMC8665348", "U1_spread", "(A) The dataset was randomly divided"): ("FP", "", "PROCEDURE_ONLY", ""),
    ("PMC8665348", "U1_spread", "(C) The accuracy and AUC of different folds"): ("TP", "U1", "", "per-fold metrics plotted"),
    ("PMC8665348", "S_group", "It was also noticed from the image"): ("TP", "S1", "", "failure on new conditions reported"),
    ("PMC8471570", "S_group", "The initial CAD library of 20"): ("FP", "", "ORDINARY_HOLDOUT", "new lattice designs = ordinary test samples"),
    ("PMC11106676", "U1_spread", "SDV | Model"): ("TP", "U1", "", "SD of Sn/Sp across three resampled datasets (borderline)"),
    ("PMC11468966", "E_prospective", "The properties of the optimal hydrogel"): ("TP", "EP", "", "optimal formulation made and compared with prediction"),
    ("PMC11422183", "S_group", "During the past few years"): ("FP", "", "BACKGROUND_GENERAL", ""),
    ("PMC12157168", "E_prospective", "This cycle of generating new"): ("FP", "", "BACKGROUND_GENERAL", ""),
    ("PMC12157168", "E_prospective", "The final model was validated using the holdout"): ("TP", "EP", "", "test data never used in model generation"),
    ("PMC12157168", "E_prospective", "The model was tested by using a test set"): ("TP", "EP", "", "22 new experiments"),
}


def main_v11():
    v10 = list(csv.DictReader(open(HERE / "hit_verification.csv", encoding="utf-8")))
    k10 = {(r["pmcid"], r["pattern"], r["sentence"][:200]): r for r in v10}
    rows = list(csv.DictReader(open(HERE / "screen_hits_v11.csv", encoding="utf-8")))
    out, missing = [], []
    for i, r in enumerate(rows):
        k = (r["pmcid"], r["pattern"], r["sentence"][:200])
        if k in k10:
            o = k10[k]
            lab = (o["verdict"], o["supports"], o["fp_category"], o["note"])
        else:
            lab = next((v for (pm, pat, start), v in NEW_V11.items()
                        if pm == r["pmcid"] and pat == r["pattern"] and r["sentence"].startswith(start)), None)
            if lab is None:
                missing.append((i, r["pmcid"], r["pattern"], r["sentence"][:80]))
                continue
        out.append({"hit": i, "area": r["area"], "pmcid": r["pmcid"], "section": r["section"],
                    "pattern": r["pattern"], "family": r["family"], "match": r["match"],
                    "verdict": lab[0], "supports": lab[1], "fp_category": lab[2], "note": lab[3],
                    "sentence": r["sentence"][:300]})
    assert not missing, missing
    with open(HERE / "hit_verification_v11.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(len(out), "v1.1 labels written")


def main():
    rows = list(csv.DictReader(open(HERE / "screen_hits.csv", encoding="utf-8")))
    assert len(rows) == len(L) == max(L) + 1, (len(rows), len(L))
    out = []
    for i, r in enumerate(rows):
        v, sup, cat, note = L[i]
        out.append({"hit": i, "area": r["area"], "pmcid": r["pmcid"], "section": r["section"],
                    "pattern": r["pattern"], "family": r["family"], "match": r["match"],
                    "verdict": v, "supports": sup, "fp_category": cat, "note": note,
                    "sentence": r["sentence"][:300]})
    with open(HERE / "hit_verification.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(len(out), "labels written")


if __name__ == "__main__":
    main()
    if (HERE / "screen_hits_v11.csv").exists():
        main_v11()
