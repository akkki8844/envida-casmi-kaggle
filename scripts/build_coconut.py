"""Convert the COCONUT CSV (CC0) into a compact candidate table: parent SMILES, InChIKey14, exact mass.

    python scripts/build_coconut.py data/coconut/coconut_csv_lite-11-2026.zip data/coconut/coconut_pool.parquet

Salts/solvents are stripped (largest organic fragment), stereo is dropped (the metric ignores it),
and rows are deduplicated by InChIKey14.
"""
import io
import sys
import zipfile
from multiprocessing import Pool, cpu_count

import pandas as pd


def _init():
    global Chem, Descriptors, inchi, chooser
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, inchi
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog("rdApp.*")
    chooser = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)


def _one(args):
    smi, np_like = args
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        if "." in smi:
            mol = chooser.choose(mol)
        if mol.GetNumHeavyAtoms() < 3:
            return None
        Chem.RemoveStereochemistry(mol)
        key = inchi.MolToInchiKey(mol)
        if not key:
            return None
        return key[:14], Chem.MolToSmiles(mol), Descriptors.ExactMolWt(mol), np_like
    except Exception:
        return None


def main(src, dst):
    z = zipfile.ZipFile(src)
    name = next(i.filename for i in z.infolist() if i.filename.endswith(".csv"))
    with z.open(name) as f:
        df = pd.read_csv(io.TextIOWrapper(f, encoding="utf-8", errors="replace"),
                         usecols=["canonical_smiles", "np_likeness"], dtype={"canonical_smiles": str})
    df = df.dropna(subset=["canonical_smiles"])
    print(f"COCONUT rows: {len(df):,}")
    work = list(zip(df["canonical_smiles"], pd.to_numeric(df["np_likeness"], errors="coerce").fillna(0.0)))
    with Pool(max(cpu_count() - 1, 1), initializer=_init) as p:
        out = [r for r in p.imap(_one, work, chunksize=2000) if r is not None]
    res = pd.DataFrame(out, columns=["inchikey14", "smiles", "mass", "np_likeness"])
    res = res.drop_duplicates("inchikey14").sort_values("mass").reset_index(drop=True)
    res = res[(res["mass"] >= 50) & (res["mass"] <= 2000)]
    res.to_parquet(dst, index=False)
    print(f"unique structures: {len(res):,} -> {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
