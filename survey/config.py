"""Frozen search configuration for the systematic survey behind Figure 2 (R2.13).

Every block is applied to Europe PMC's TITLE_ABS field (title + abstract), so the
search never matches on full-text body words.  Queries are frozen here; any
change must be logged in A6_survey_protocol.md with its date.
"""

SEED = 20260919           # fixed seed for random ordering of each sampling frame
YEARS = (2020, 2025)      # inclusive publication-year window

# ---------------------------------------------------------------- blocks
ML = ('("machine learning" OR "deep learning" OR "artificial intelligence" OR '
      '"neural network" OR "neural networks" OR convolutional OR "random forest" OR '
      '"gradient boosting" OR XGBoost OR "support vector" OR "Gaussian process" OR '
      '"Bayesian optimization" OR "Bayesian optimisation" OR "active learning" OR '
      '"graph neural" OR "generative model" OR "variational autoencoder" OR "U-Net")')

# Biomedical-material block.  Bare implant* / scaffold* / "medical device" were
# tested and rejected in piloting (dominated by clinical-outcome and molecular-
# scaffold/QSAR records); they are admitted only with material context words.
BIOMAT = ('(biomaterial OR biomaterials OR "biomedical material" OR "biomedical materials" OR '
          'hydrogel* OR bioink* OR "tissue engineering" OR "tissue-engineered" OR '
          '"regenerative medicine" OR '
          '(scaffold* AND (tissue OR bone OR regenerat* OR cartilage OR electrospun OR '
          'electrospinning OR bioprint* OR porous)) OR '
          '(implant* AND (alloy* OR titanium OR coating* OR osseointegration OR corrosion OR '
          '"surface roughness" OR "additive manufacturing" OR "3D printed" OR "3D-printed")) OR '
          '"bioactive glass" OR "bioactive glasses" OR bioceramic* OR hydroxyapatite OR '
          '"calcium phosphate" OR "drug delivery" OR "drug carrier" OR "drug carriers" OR '
          'nanocarrier* OR "lipid nanoparticle" OR "lipid nanoparticles" OR liposome* OR '
          '"controlled release" OR "drug release" OR biocompatib* OR "wound dressing" OR '
          '"wound dressings")')

AREA_TERMS = {
    "A1_imaging": ('(microscopy OR microscopic OR micrograph OR micrographs OR "electron microscopy" OR '
                   '"scanning electron" OR SEM OR TEM OR "micro-CT" OR microCT OR tomography OR '
                   '"image analysis" OR "image segmentation" OR "image processing" OR '
                   '"computer vision" OR images OR imaging OR histology OR histological)'),
    "A2_property": ('("property prediction" OR "properties prediction" OR "inverse design" OR '
                    '"materials design" OR "material design" OR "materials discovery" OR '
                    '"rational design" OR "de novo design" OR "generative design" OR '
                    '"structure-property" OR "composition-property" OR QSPR OR '
                    '"formulation design" OR "formulation optimization" OR "formulation optimisation" OR '
                    '"mechanical properties" OR "mechanical property" OR modulus OR stiffness OR '
                    '"compressive strength" OR "tensile strength" OR "degradation rate" OR '
                    '"drug release" OR "release kinetics" OR "encapsulation efficiency" OR '
                    '"particle size" OR swelling OR gelation OR cytotoxicity OR "transfection efficiency")'),
    "A3_sensors": ('(biosensor* OR bioelectronic* OR "electronic skin" OR "e-skin" OR '
                   '"wearable sensor" OR "wearable sensors" OR "flexible sensor" OR "flexible sensors" OR '
                   '"flexible electronics" OR "strain sensor" OR "strain sensors" OR '
                   '"pressure sensor" OR "pressure sensors" OR "brain-computer interface" OR '
                   '"neural interface" OR "neural interfaces" OR triboelectric OR piezoelectric OR '
                   '"electronic nose" OR "electronic tongue" OR "sensor array" OR "sensor arrays" OR '
                   '"electrochemical sensor" OR "electrochemical sensors")'),
    "A4_tissue": ('("tissue engineering" OR "tissue-engineered" OR "regenerative medicine" OR '
                  '(scaffold* AND (tissue OR bone OR regenerat* OR cartilage OR electrospun OR '
                  'electrospinning OR bioprint* OR porous)) OR hydrogel* OR bioink* OR bioprint* OR '
                  'electrospun OR electrospinning)'),
    "A5_autonomous": ('("Bayesian optimization" OR "Bayesian optimisation" OR "active learning" OR '
                      '"self-driving" OR "autonomous laboratory" OR "autonomous laboratories" OR '
                      '"autonomous experimentation" OR "autonomous experiments" OR '
                      '"robotic platform" OR "robotic chemist" OR "closed-loop optimization" OR '
                      '"closed-loop optimisation" OR "closed-loop discovery" OR '
                      '"closed-loop experimentation" OR "high-throughput experimentation" OR '
                      '"sequential learning" OR "adaptive experimentation" OR "sequential design")'),
}

# Material-context block for A3 (device-defined area): keeps materials/device
# papers and drops pure signal-processing / activity-recognition work.
DEVICE_MATERIAL = ('(material OR materials OR nanomaterial* OR electrode* OR hydrogel* OR '
                   'polymer* OR composite* OR graphene OR nanowire* OR nanotube* OR MXene OR '
                   'textile* OR fiber OR fibers OR fibre OR fibres OR nanoparticle* OR film OR films)')

# Wider material block for A5, where the sequential-design terms are themselves
# the ML block and material classes are named rather than "biomaterial".
BIOMAT_WIDE = ('(' + BIOMAT[1:-1] + ' OR polymer* OR copolymer* OR peptide* OR nanoparticle* OR '
               'formulation* OR coating*)')

# ------------------------------------------------------------------ v2
# Refined after the pilot title/abstract screen of the first 12 random records
# per v1 frame (see pilot/ta_screen_v1.csv): v1 precision was 4/12 (A1),
# 10/12 (A2), 2/12 (A3), 1/12 (A4), 4/12 (A5).  Changes:
#  * BIOMAT_V2 drops "regenerative medicine" (pulled stem-cell / organoid /
#    transcriptomics papers with no material).
#  * A1 drops the generic "images OR imaging" (pulled clinical imaging and
#    'for biomedical imaging' nanoparticle papers) and adds histolog* / X-ray
#    diffraction / "image-based".
#  * A3 adds a biomedical-context block and drops generic material words
#    (material, fiber, film, electrode), which admitted structural-health-
#    monitoring, environmental and commercial-wearable clinical studies.
#  * A4 requires a TE material (scaffold/hydrogel/bioink/electrospun/biomaterial)
#    AND a tissue/regeneration context word.
#  * A5 adds the biomedical-context block (v1 admitted catalysts, photocatalysts,
#    autonomous vehicles and 'active learning' pedagogy).
BIOMAT_V2 = BIOMAT.replace('"regenerative medicine" OR ', '')

BIOMED = ('(biomedical OR biomedicine OR health OR healthcare OR wearable OR wearables OR skin OR '
          'implantable OR physiological OR patient OR patients OR medical OR clinical OR sweat OR '
          'glucose OR disease OR diseases OR diagnosis OR diagnostic OR "human motion" OR '
          'biomarker OR biomarkers OR prosthesis OR prosthetic OR rehabilitation OR '
          'drug OR drugs OR therapeutic OR therapeutics OR pharmaceutical OR vaccine OR '
          'tissue OR implant OR implants OR bone OR wound OR biocompatib* OR biomaterial OR '
          'biomaterials OR "gene delivery" OR mRNA)')

AREA_TERMS_V2 = dict(AREA_TERMS)
AREA_TERMS_V2["A1_imaging"] = ('(microscopy OR micrograph OR micrographs OR "electron microscopy" OR '
                               '"scanning electron" OR SEM OR TEM OR "micro-CT" OR microCT OR '
                               'tomography OR "image analysis" OR "image segmentation" OR '
                               '"image processing" OR "image-based" OR "computer vision" OR '
                               'histology OR histological OR "X-ray diffraction")')
AREA_TERMS_V2["A4_tissue"] = ('((scaffold* OR hydrogel* OR bioink* OR bioprint* OR electrospun OR '
                              'electrospinning OR biomaterial OR biomaterials) AND (tissue OR regenerat* OR '
                              'bone OR cartilage OR wound OR osteogen* OR "cell culture" OR '
                              '"cell adhesion" OR "cell viability" OR injectable))')
DEVICE_MATERIAL_V2 = ('(nanomaterial* OR hydrogel* OR polymer* OR composite* OR graphene OR '
                      'nanowire* OR nanotube* OR MXene OR textile* OR nanoparticle* OR '
                      '"sensing material" OR "sensing materials" OR ionogel* OR elastomer* OR '
                      'fabric OR fabrics OR nanofiber* OR nanofibre*)')


def area_core_v2(area: str) -> str:
    t = AREA_TERMS_V2[area]
    if area == "A3_sensors":
        return f"TITLE_ABS:{ML} AND TITLE_ABS:{t} AND TITLE_ABS:{DEVICE_MATERIAL_V2} AND TITLE_ABS:{BIOMED}"
    if area == "A4_tissue":
        return f"TITLE_ABS:{ML} AND TITLE_ABS:{t}"
    if area == "A5_autonomous":
        wide = '(' + BIOMAT_V2[1:-1] + ' OR polymer* OR copolymer* OR peptide* OR nanoparticle* OR formulation* OR coating*)'
        return f"TITLE_ABS:{t} AND TITLE_ABS:{wide} AND TITLE_ABS:{BIOMED}"
    return f"TITLE_ABS:{ML} AND TITLE_ABS:{BIOMAT_V2} AND TITLE_ABS:{t}"


YEAR_F = f"PUB_YEAR:[{YEARS[0]} TO {YEARS[1]}]"
# Original research only: drop records typed as review / editorial etc. and preprints.
TYPE_F = ('NOT (PUB_TYPE:review OR PUB_TYPE:"review-article" OR PUB_TYPE:"systematic review" OR '
          'PUB_TYPE:"systematic-review" OR PUB_TYPE:"meta-analysis" OR PUB_TYPE:editorial OR '
          'PUB_TYPE:comment OR PUB_TYPE:letter OR PUB_TYPE:"published erratum" OR '
          'PUB_TYPE:"retraction of publication" OR PUB_TYPE:"retracted publication") AND NOT SRC:PPR')
LANG_F = "LANG:eng"
OA_F = "OPEN_ACCESS:y"


QUERY_VERSION = "v2"   # default for scripts; v1 retained for the pilot record

# Frozen per-area choice after the pilot (A6_survey_protocol.md, section 4):
# the version with the larger expected number of eligible OA papers
# (frame size x pilot title/abstract precision).  v2 cut recall for A1 and A5.
FINAL_VERSION = {"A1_imaging": "v1", "A2_property": "v1", "A3_sensors": "v2",
                 "A4_tissue": "v2", "A5_autonomous": "v1"}


def area_core(area: str, version: str | None = None) -> str:
    """Topic query (title/abstract) for one area, without filters."""
    if (version or QUERY_VERSION) == "v2":
        return area_core_v2(area)
    if area == "A3_sensors":
        return f"TITLE_ABS:{ML} AND TITLE_ABS:{AREA_TERMS[area]} AND TITLE_ABS:{DEVICE_MATERIAL}"
    if area == "A4_tissue":
        return f"TITLE_ABS:{ML} AND TITLE_ABS:{AREA_TERMS[area]}"
    if area == "A5_autonomous":
        return f"TITLE_ABS:{AREA_TERMS[area]} AND TITLE_ABS:{BIOMAT_WIDE}"
    return f"TITLE_ABS:{ML} AND TITLE_ABS:{BIOMAT} AND TITLE_ABS:{AREA_TERMS[area]}"


def area_query(area: str, oa: bool = False, research_only: bool = True,
               version: str | None = None) -> str:
    q = f"({area_core(area, version)}) AND {YEAR_F} AND {LANG_F}"
    if research_only:
        q += f" AND {TYPE_F}"
    if oa:
        q += f" AND {OA_F}"
    return q


AREAS = list(AREA_TERMS)
