"""Regex screen for the three trust items (frozen v1 of the screen).

Each pattern proposes a *candidate* sentence; no pattern is sufficient on its
own to assign a code.  Every candidate is verified (LLM or human) against the
operational definitions in A6_survey_protocol.md.

Families and the code level a verified hit can support:
  U1  any uncertainty on model performance (SD over folds/seeds, CI, error bars)
  U2  per-prediction uncertainty (intervals, GP/ensemble/Bayesian std, quantiles)
  U3  calibrated per-prediction uncertainty (calibration assessed, conformal, coverage)
  E1  external validation (independent dataset / cohort / lab / instrument)
  EP  prospective experimental validation of model predictions (new samples made
      after the model was fixed) - recorded separately, see protocol
  S1  explicit distribution-shift / OOD / extrapolation / group-held-out evaluation
"""
import re

I = re.IGNORECASE

# ------------------------------------------------------------------ helpers
METRIC = re.compile(r"(\bR\s?\^?\s?2\b|R²|\bRMSE\b|\bMAE\b|\bMSE\b|\bMAPE\b|\baccurac(y|ies)\b|\bAUC\b|"
                    r"\bAUROC\b|\bF1\b|\bprecision\b|\brecall\b|\berror\b|\bscore\b|correlation coefficient|"
                    r"\bPearson\b|coefficient of determination)", I)
CVCTX = re.compile(r"(\bfolds?\b|cross[- ]validat\w*|\bruns\b|\brepeat(s|ed|ition)\b|random (seeds?|splits?|initiali[sz]ations?)|"
                   r"independent (runs|trainings?)|\btrials\b|\bbootstrap\w*)", I)
MLCTX = re.compile(r"(\bmodel\w*\b|\bpredict\w*|\bnetwork\w*\b|\bregress\w*|\bclassif\w*|machine learning|"
                   r"deep learning|\btrain\w*|\bML\b|\bANN\b|\bCNN\b|random forest|\bGPR?\b|\bSVM\b|\bSVR\b|XGBoost)", I)
FUTURE_NEG = re.compile(r"(\bfuture\b|further (work|stud\w+|research)|\bshould\b|\bcould\b|would be|remains? to be|"
                        r"\bnot (yet )?(been )?(considered|evaluated|assessed|addressed|quantified|tested|validated)|"
                        r"\blimitation|\black(s|ing)?\b|absence of|\bwithout\b|beyond the scope)", I)

PATTERNS = {
    # ---------------- U1: variability of performance metrics (requires METRIC or CV context)
    "U1_spread": (1, "U", re.compile(r"(±|\+/-|\bstandard deviations?\b|\bs\.d\.|\bSD\b|\bstd\b|standard errors?\b|"
                                     r"\bSEM\b|error bars?|confidence intervals?|\b95\s?%\s?CI\b|\bCI\b)", I)),
    # ---------------- U2: per-prediction uncertainty
    "U2_interval": (2, "U", re.compile(r"(prediction intervals?|predictive intervals?|credible intervals?|"
                                       r"uncertainty (bands?|bounds?|intervals?|estimates?)|"
                                       r"confidence (intervals?|bands?) (of|for|on|around) (the )?(each |individual )?predict\w*)", I)),
    "U2_uq": (2, "U", re.compile(r"(uncertainty quantification|\bUQ\b|predictive uncertaint\w*|prediction uncertaint\w*|"
                                 r"model uncertaint\w*|epistemic|aleatoric|uncertainty[- ]aware|"
                                 r"quantif\w+ (the )?(predictive |prediction |model )?uncertaint\w*|"
                                 r"uncertainty (of|in) (the )?(model|prediction)s?)", I)),
    "U2_gp": (2, "U", re.compile(r"(Gaussian process\w*|\bGPR\b|kriging)", I)),
    "U2_probabilistic": (2, "U", re.compile(r"(Bayesian neural network|\bBNNs?\b|Monte[- ]Carlo dropout|\bMC[- ]dropout|"
                                            r"variational inference|posterior (distribution|variance|predictive|mean|standard deviation)|"
                                            r"deep ensembles?|ensemble (variance|spread|uncertainty|standard deviation|disagreement)|"
                                            r"quantile regression|mixture density|evidential (deep learning|regression|network)|"
                                            r"NGBoost|natural gradient boosting|probabilistic (model|prediction|forecast|output|regression)\w*)", I)),
    "U2_acquisition": (2, "U", re.compile(r"(acquisition functions?|expected improvement|upper confidence bound|\bUCB\b|"
                                          r"probability of improvement|Thompson sampling|exploration[- ]exploitation|"
                                          r"explor\w+ and exploit\w+)", I)),
    # ---------------- U3: calibration of per-prediction uncertainty
    "U3_conformal": (3, "U", re.compile(r"(conformal\w*|Mondrian|jackknife\+|\bCV\+)", I)),
    "U3_coverage": (3, "U", re.compile(r"(coverage (probability|rate|level)|empirical coverage|\bPICP\b|nominal (coverage|level)|"
                                       r"interval coverage)", I)),
    "U3_calibration": (3, "U", re.compile(r"(calibration error|\bECE\b|reliability (diagrams?|curves?|plots?)|"
                                          r"calibration (curves?|plots?)|well[- ]calibrated|mis-?calibrat\w*|"
                                          r"over-?confiden\w*|under-?confiden\w*|temperature scaling|Platt scaling|"
                                          r"isotonic (regression|calibration)|Brier( score)?|negative log[- ]likelihood|"
                                          r"\bNLL\b|\bCRPS\b|continuous ranked probability|interval score|\bsharpness\b|"
                                          r"calibrated (uncertaint\w*|probabilit\w*|model|predict\w*))", I)),
    # ---------------- E1: external validation
    "E_external": (1, "E", re.compile(r"(external(ly)? (validat\w*|test\w*|data\w*|data set|cohorts?|evaluation)|"
                                      r"independent (external )?(test|validation|testing|hold[- ]?out)? ?(sets?|datasets?|data sets?|cohorts?|batch\w*|samples)\b|"
                                      r"(another|different|second|other|separate) (laborator(y|ies)|labs?|institutions?|hospitals?|sites?|"
                                      r"centers?|centres?|instruments?|microscopes?|cohorts?|datasets?|data sets?)|"
                                      r"multi-?(center|centre|site|institution(al)?)|"
                                      r"(unseen|new|fresh) (datasets?|data sets?|cohorts?|batch(es)?)|"
                                      r"(public(ly available)?|literature|published) (datasets?|data sets?|data) (for|as) (validation|testing|test))", I)),
    "E_prospective": (1, "EP", re.compile(r"(experimental(ly)? (validat\w*|verif\w*|confirm\w*|tested|test)|"
                                          r"validated experimentally|verified experimentally|confirmed experimentally|"
                                          r"prospective(ly)? (validat\w*|test\w*|evaluat\w*)|blind (test|validation|prediction)s?|"
                                          r"(synthesi[sz]ed|fabricated|prepared|printed) and (tested|characteri[sz]ed|measured|evaluated)|"
                                          r"validation (experiments?|samples?|runs?)|out-of-sample experiment\w*)", I)),
    # ---------------- S1: distribution-shift / extrapolation testing
    "S_ood": (1, "S", re.compile(r"(out[- ]of[- ](distribution|domain|sample)|\bOOD\b|distribution(al)? shifts?|"
                                 r"domain (shifts?|adaptation|generali[sz]ation)|dataset shifts?|covariate shifts?|"
                                 r"concept drift|applicability domain|extrapolat\w*)", I)),
    "S_group": (1, "S", re.compile(r"(leave[- ]one[- ]\w+[- ]out|leave[- ]\w+[- ]\w*[- ]?out|\bLO[A-Z]{1,2}O\b|"
                                   r"cluster[- ](based )?(splits?|cross[- ]validation)|scaffold split\w*|"
                                   r"group(ed)?[- ](k[- ]fold|cross[- ]validation|splits?)|"
                                   r"cross[- ](subjects?|users?|devices?|batch(es)?|sites?|laborator(y|ies)|instruments?|datasets?|"
                                   r"domains?|materials?|compositions?)|"
                                   r"unseen (subjects?|users?|participants?|materials?|compositions?|polymers?|devices?|"
                                   r"batch(es)?|classes|chemistr\w+|scaffolds?|domains?|conditions?)|"
                                   r"(time|temporal)[- ](split|hold[- ]?out|based split)|chronological split)", I)),
    "S_robust": (1, "S", re.compile(r"(\bdrift\w*|\bageing\b|\baging\b|batch[- ]to[- ]batch|"
                                    r"robust(ness)? (to|against|under)|noise (robustness|injection)|"
                                    r"(added|additive|injected|synthetic|artificial) (Gaussian |white |random )?noise|"
                                    r"adversarial|perturb\w*|transfer learning)", I)),
}

PATTERNS_V10 = dict(PATTERNS)  # frozen: the version the pilot hits were labelled with

# ------------------------------------------------------------------ v1.1
# Revisions motivated by the pilot false-positive / false-negative audit
# (A6_survey_protocol.md, section 6).  NOTE: tuned on the 25 pilot papers, so
# its pilot performance is optimistic; re-estimate on a fresh subset.
EVALCTX = re.compile(r"(\baccurac\w*|\bperformance\b|\berror\w*|\bRMSE\b|\bMAE\b|\bR ?2\b|R²|\bAUC\b|\bF1\b|"
                     r"\btested\b|\btesting\b|\bevaluat\w*|\bvalidat\w*|\bgenerali[sz]\w*|\bpredict\w*|"
                     r"\brobust\w*|\bextrapolat\w*)", I)
MEASCTX = re.compile(r"(independent(ly)? (experiments?|repeated|samples)|\breplicates?\b|n\s?=\s?\d|"
                     r"repeated (tests|experiments|measurements)|biological (replicates|repeats))", I)
PATTERNS_V11 = dict(PATTERNS_V10)
PATTERNS_V11.update({
    # U1: add SDV / per-fold reporting; drop bare "SEM" (scanning electron microscope)
    "U1_spread": (1, "U", re.compile(r"(±|\+/-|\bstandard deviations?\b|\bs\.d\.|\bSDV?\b|\bstd\b|standard errors?\b|"
                                     r"mean\s?±\s?SEM|error bars?|confidence intervals?|\b95\s?%\s?CI\b|\bCI\b|"
                                     r"(each|every|different|individual|all) folds?|across (the )?(five|ten|\d+) folds|"
                                     r"per[- ]fold)", I)),
    # U3: analytical 'calibration curve' removed; kept only uncertainty-calibration vocabulary
    "U3_calibration": (3, "U", re.compile(r"(calibration error|\bECE\b|reliability (diagrams?|curves?|plots?)|"
                                          r"(uncertainty|probability|confidence) calibration|calibrat\w+ (of|the) (uncertaint\w*|"
                                          r"probabilit\w*|predictive distribution|prediction intervals?)|"
                                          r"mis-?calibrat\w*|over-?confiden\w*|under-?confiden\w*|temperature scaling|"
                                          r"Platt scaling|isotonic (regression|calibration)|Brier( score)?|"
                                          r"negative log[- ]likelihood|\bNLL\b|\bCRPS\b|continuous ranked probability|"
                                          r"interval score|calibrated (uncertaint\w*|probabilit\w*|predict\w*))", I)),
    # E1: add simulation-to-experiment testing
    "E_external": (1, "E", re.compile(r"(external(ly)? (validat\w*|test\w*|data\w*|data set|cohorts?|evaluation)|"
                                      r"independent (external )?(test|validation|testing|hold[- ]?out)? ?(sets?|datasets?|data sets?|cohorts?)\b|"
                                      r"(another|different|second|other|separate) (laborator(y|ies)|labs?|institutions?|hospitals?|sites?|"
                                      r"centers?|centres?|instruments?|microscopes?|cohorts?)|"
                                      r"multi-?(center|centre|site|institution(al)?)|"
                                      r"(public(ly available)?|literature|published) (datasets?|data sets?|data) (for|as) (validation|testing|test)|"
                                      r"(trained|training) (on|with) (the )?(simulated|synthetic|computational|DFT|FEM?|in[- ]silico)\b.{0,120}"
                                      r"(tested|testing|evaluated|validated|applied) (on|with|to) (the )?(real|experimental|measured)|"
                                      r"(real[- ]world|experimental) (experimental )?data (are|were|is|was) used for (testing|validation))", I)),
    # EP: add comparisons of predictions with new experiments
    "E_prospective": (1, "EP", re.compile(r"(experimental(ly)? (validat\w*|verif\w*|confirm\w*)|"
                                          r"validated experimentally|verified experimentally|confirmed experimentally|"
                                          r"prospective(ly)? (validat\w*|test\w*|evaluat\w*)|blind (test|validation|prediction)s?|"
                                          r"validation (experiments?|runs?)|out-of-sample experiment\w*|"
                                          r"never (been )?used (in|for|during) (any |the )?(steps? of )?(model|training)|"
                                          r"new experimental data( points)?|"
                                          r"(predict\w+|model\w*) (values |results )?(were |was )?(compared|agree\w*) (with|to|against) "
                                          r"(the )?(corresponding |subsequent |new )?(experimental|measured) (results|values|data|measurements)|"
                                          r"close to (those|the values?|what was) predicted|"
                                          r"(produced|prepared|made|synthesi[sz]ed|fabricated|printed) (according to|at|following|using) "
                                          r"the (predicted|optimal|optimi[sz]ed|suggested|recommended))", I)),
    # S1: group-wise leave-X-out only; new-domain application; sim-to-real
    "S_group": (1, "S", re.compile(r"(leave[- ]one[- ](subject|patient|participant|user|group|cluster|batch|site|lab|laboratory|"
                                   r"material|composition|class|family|element|polymer|device|day|session|study|dataset|"
                                   r"condition|donor|animal)s?[- ]out|leave[- ]\w+[- ](group|cluster|batch|site|class|family)s?[- ]out|"
                                   r"\bLO(S|G|C|B|P)O\b|"
                                   r"cluster[- ](based )?(splits?|cross[- ]validation)|scaffold split\w*|"
                                   r"group(ed)?[- ](k[- ]fold|cross[- ]validation|splits?)|"
                                   r"cross[- ](subjects?|users?|devices?|batch(es)?|sites?|laborator(y|ies)|instruments?|datasets?|"
                                   r"domains?|materials?|compositions?)|"
                                   r"(unseen|new|novel) (subjects?|users?|participants?|materials?|compositions?|polymers?|devices?|"
                                   r"batch(es)?|classes|chemistr\w+|scaffolds?|domains?|conditions|cell types?|substrates?)|"
                                   r"directly applied to .{0,40}(new|unseen|other|different)|without (further |additional )?(re-?)?training|"
                                   r"(time|temporal)[- ](split|hold[- ]?out|based split)|chronological split)", I)),
    # S_robust: homonym-prone terms dropped (transfer learning, adversarial, perturbation, aging)
    "S_robust": (1, "S", re.compile(r"(sensor drift|signal drift|drift (compensation|correction)|batch[- ]to[- ]batch|"
                                    r"robust(ness)? (to|against|under) (\w+ ){0,2}(noise|variations?|changes?|shifts?|drift|different|"
                                    r"modest|the range)|noise (robustness|injection)|"
                                    r"(added|additive|injected|synthetic|artificial) (Gaussian |white |random )?noise|"
                                    r"(tested|evaluated) (under|with|on) (different|varying|new) (conditions|noise|instruments?))", I)),
})


def screen_sentence_v11(sent: str):
    out = []
    for pid, (lvl, fam, rx) in PATTERNS_V11.items():
        m = rx.search(sent)
        if not m:
            continue
        if pid == "U1_spread":
            if not (METRIC.search(sent) or CVCTX.search(sent)):
                continue
            if not MLCTX.search(sent) and not CVCTX.search(sent):
                continue
            if MEASCTX.search(sent) and not (METRIC.search(sent) and MLCTX.search(sent)):
                continue  # replicate spread of a wet-lab measurement
        if fam == "S" and not (EVALCTX.search(sent) or MLCTX.search(sent)):
            continue  # shift vocabulary with neither evaluation nor model context
        out.append((pid, fam, lvl, m.group(0)))
    return out


# Section classes that describe the study's own work.  Hits outside them
# (INTRO, DISCUSSION-only, CONCLUSION, OTHER) are still recorded but flagged.
OWN_WORK = {"ABSTRACT", "METHODS", "RESULTS", "RESULTS_DISCUSSION", "FIG", "TABLE"}

SECTION_RULES = [
    ("RESULTS_DISCUSSION", re.compile(r"results? (and|&) discussion|validation, result", I)),
    ("INTRO", re.compile(r"^\W*\d*\.?\s*(introduction|background)", I)),
    ("METHODS", re.compile(r"method|experimental|materials|dataset|data (collection|and code)|model|machine learning|"
                           r"computational|simulation|fabrication|preparation|synthesis|statistic|training|algorithm", I)),
    ("RESULTS", re.compile(r"result|performance|evaluation|validation|benchmark", I)),
    ("DISCUSSION", re.compile(r"discussion|limitation|outlook|future|challenge|perspective", I)),
    ("CONCLUSION", re.compile(r"conclusion|summary|concluding", I)),
]
SKIP_SEC = re.compile(r"acknowledg|conflict|competing|author contribution|credit authorship|funding|declaration|"
                      r"^references$|nomenclature|abbreviation|supplementary|supporting information|ethics|"
                      r"data availability|associated data", I)
SKIP_SECTYPE = {"ack", "ref-list", "fn-group", "COI-statement", "supplementary-material", "contrib-info",
                "associated-data", "data-availability", "funding", "author-contributions"}
SECTYPE_MAP = {"intro": "INTRO", "introduction": "INTRO", "methods": "METHODS", "materials|methods": "METHODS",
               "results": "RESULTS", "discussion": "DISCUSSION", "conclusions": "CONCLUSION",
               "discussion|interpretation": "RESULTS_DISCUSSION", "results|discussion": "RESULTS_DISCUSSION"}


def classify_section(title: str, sec_type: str = "") -> str:
    st = (sec_type or "").strip()
    t = " ".join((title or "").split())
    # title wins over sec-type when it says "results and discussion"
    if SECTION_RULES[0][1].search(t):
        return "RESULTS_DISCUSSION"
    if st in SECTYPE_MAP and st != "conclusions":
        return SECTYPE_MAP[st]
    for name, rx in SECTION_RULES[1:]:
        if rx.search(t):
            return name
    if st in SECTYPE_MAP:
        return SECTYPE_MAP[st]
    return "BODY_OTHER"


def screen_sentence(sent: str):
    """Return list of (pattern_id, family, level, matched_text) for one sentence."""
    out = []
    for pid, (lvl, fam, rx) in PATTERNS.items():
        m = rx.search(sent)
        if not m:
            continue
        if pid == "U1_spread":
            # spread tokens only count next to a performance metric or resampling context
            if not (METRIC.search(sent) or CVCTX.search(sent)):
                continue
            if not MLCTX.search(sent) and not CVCTX.search(sent):
                continue
        out.append((pid, fam, lvl, m.group(0)))
    return out
