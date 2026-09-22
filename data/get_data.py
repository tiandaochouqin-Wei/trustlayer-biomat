"""Download / rebuild every dataset used in the paper and record versions.

Three files (polymer_tg.npz, delaney.csv, glass_gfa_magpie.npz) are not
re-distributed in this package and are rebuilt here from public sources so
the package is independently reproducible:

  * delaney.csv        - downloaded verbatim from the canonical MoleculeNet
                          mirror (ESOL/Delaney, Wu et al. 2018).
  * glass_gfa_magpie.npz- rebuilt from matminer's matbench_glass compositions
                          + Magpie composition descriptors (deterministic,
                          exact reconstruction).
  * polymer_tg.npz      - rebuilt from the OsBaran/polymer_tg_dataset
                          community upload on HuggingFace (SMILES + Tg only)
                          by recomputing the ten RDKit descriptors used
                          throughout the paper. DISCLOSED GAP (see Methods /
                          Limitations in the manuscript): this is a community
                          upload whose original experimental source we could
                          not establish, so it is used only as a
                          descriptor-level benchmark, not as curated,
                          provenance-checked polymer data.

Also rebuilds data/bioglass_tg.npz (the legacy-reproduction cache used by
exp_R9_local.py / exp_R9_wcp.py) from live SciGlass via common.glass_dataset,
so no file needs to be shipped for it either.
"""
import json, os, sys, zipfile, urllib.request
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(os.path.dirname(HERE), "code")

DELANEY_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
POLYMER_TG_PARQUET_URL = ("https://huggingface.co/datasets/OsBaran/polymer_tg_dataset/"
                           "resolve/main/data/train-00000-of-00001.parquet")
POLYMER_FEATS = ["MolWt", "NumAromRings", "FracCSP3", "HBD", "HBA", "RotB", "TPSA",
                  "Rings", "HeavyAtoms", "FracAromAtoms"]


def get_delaney():
    path = os.path.join(HERE, "delaney.csv")
    if not os.path.exists(path):
        print("downloading ESOL/Delaney (MoleculeNet delaney-processed.csv)")
        urllib.request.urlretrieve(DELANEY_URL, path)
    return path


def get_polymer_tg():
    path = os.path.join(HERE, "polymer_tg.npz")
    if os.path.exists(path):
        return path
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    pq = os.path.join(HERE, "_polymer_tg_raw.parquet")
    if not os.path.exists(pq):
        print("downloading OsBaran/polymer_tg_dataset (HuggingFace, community upload)")
        urllib.request.urlretrieve(POLYMER_TG_PARQUET_URL, pq)
    raw = pd.read_parquet(pq)
    smiles_col = next(c for c in raw.columns if "smiles" in c.lower() and "original" not in c.lower())
    tg_col = next(c for c in raw.columns if "tg" in c.lower() or "glass" in c.lower())

    rows, ys = [], []
    n_skipped = 0
    for smi, tg in zip(raw[smiles_col], raw[tg_col]):
        mol = None if pd.isna(smi) else Chem.MolFromSmiles(str(smi))
        if mol is None or pd.isna(tg):
            n_skipped += 1
            continue
        n_heavy = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 0)  # exclude "*" attachment points
        n_arom = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
        rows.append([
            Descriptors.MolWt(mol),
            float(rdMolDescriptors.CalcNumAromaticRings(mol)),
            rdMolDescriptors.CalcFractionCSP3(mol),
            float(rdMolDescriptors.CalcNumHBD(mol)),
            float(rdMolDescriptors.CalcNumHBA(mol)),
            float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
            rdMolDescriptors.CalcTPSA(mol),
            float(rdMolDescriptors.CalcNumRings(mol)),
            float(n_heavy),
            (n_arom / n_heavy) if n_heavy else 0.0,
        ])
        ys.append(float(tg))
    X = np.array(rows, float)
    y = np.array(ys, float)
    np.savez(path, X=X, y=y, feat=np.array(POLYMER_FEATS))
    os.remove(pq)
    print(f"rebuilt polymer_tg.npz: {X.shape[0]} records, {n_skipped} skipped "
          f"(unparsable SMILES or missing Tg). Source: HuggingFace "
          f"OsBaran/polymer_tg_dataset; original experimental provenance of the "
          f"Tg measurements is not established beyond that upload (disclosed limitation).")
    return path


def get_glass_gfa_magpie():
    path = os.path.join(HERE, "glass_gfa_magpie.npz")
    if os.path.exists(path):
        return path
    from matminer.datasets import load_dataset
    from matminer.featurizers.composition import ElementProperty
    from pymatgen.core import Composition

    print("rebuilding glass_gfa_magpie.npz from matbench_glass + Magpie descriptors")
    df = load_dataset("matbench_glass")
    ep = ElementProperty.from_preset("magpie")
    X = np.array([ep.featurize(Composition(c)) for c in df["composition"]], float)
    y = df["gfa"].astype(int).to_numpy()
    np.savez(path, X=X, y=y)
    return path


def get_bioglass_tg():
    path = os.path.join(HERE, "bioglass_tg.npz")
    if os.path.exists(path):
        return path
    sys.path.insert(0, CODE)
    from common import glass_dataset
    print("rebuilding bioglass_tg.npz from live SciGlass (glass_dataset('Tg'))")
    ds = glass_dataset("Tg")
    np.savez(path, X=ds.X, y=ds.y, P=ds.shift, feat=np.array(ds.feat))
    return path


def main():
    versions = {}
    import glasspy, matminer, medmnist, rdkit
    versions["glasspy"] = glasspy.__version__ if hasattr(glasspy, "__version__") else "0.6.0"
    versions["matminer"] = matminer.__version__
    versions["medmnist"] = medmnist.__version__
    versions["rdkit"] = rdkit.__version__
    import glasspy.data as gd
    gd.SciGlass()                                   # caches SciGlass locally
    from matminer.datasets import load_dataset
    for ds in ["steel_strength", "expt_formation_enthalpy", "matbench_glass"]:
        load_dataset(ds)
    from medmnist import PathMNIST
    for split in ["train", "val", "test"]:
        PathMNIST(split=split, download=True, size=28)
    url = "https://zenodo.org/records/53169/files/Kather_texture_2016_image_tiles_5000.zip?download=1"
    zp = os.path.join(HERE, "Kather_texture_2016_image_tiles_5000.zip")
    if not os.path.exists(zp):
        print("downloading Kather 2016 tiles (258 MB)")
        urllib.request.urlretrieve(url, zp)
    with zipfile.ZipFile(zp) as z:
        z.extractall(os.path.join(HERE, "kather2016"))

    get_delaney()
    get_glass_gfa_magpie()
    get_polymer_tg()
    get_bioglass_tg()

    json.dump(versions, open(os.path.join(HERE, "versions.json"), "w"), indent=1)
    print("done:", versions)
if __name__ == "__main__":
    main()
