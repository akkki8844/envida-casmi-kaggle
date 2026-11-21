"""Train the APRIOR classifiers on every enveda-180 molecule and export them for the Kaggle notebook.

    python scripts/aprior_export.py --trees 300 --labels 4 [--out build/aprior_model]

Writes <out>/aprior_<j>.txt (LightGBM text boosters, one per observed-adduct label), aprior_meta.json (labels, feature
layout, train size, sha256 of each booster) and a self-check table aprior_check.json (probabilities of a few fixed
SMILES) that the notebook cell recomputes before it trusts the boosters. Features are the notebook's own featuriser
(scripts/aprior_study.py feats, identical to its _ap_featurizer)."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import aprior_study as S  # noqa: E402

CHECK_SMILES = ["CC(=O)O", "CCOC(C)=O", "OCC(O)CO", "c1ccccc1O", "CN1CCC[C@H]1c1cccnc1",
                "OC(=O)c1ccccc1O", "CC(C)Cc1ccc(C(C)C(=O)O)cc1", "OCC1OC(O)C(O)C(O)C1O"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trees", type=int, default=300)
    ap.add_argument("--labels", type=int, choices=(4, 7), default=4)
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--out", default="build/aprior_model")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    t = pq.read_table(ROOT / "data/train.parquet", columns=["ingest_lib", "inchikey14", "normalized_smiles", "adduct"])
    t = t.filter(pc.equal(t["ingest_lib"], "enveda-180")).to_pandas()
    t = t[t.adduct.isin(S.TA)]
    smi = t.drop_duplicates("inchikey14").set_index("inchikey14").normalized_smiles
    keys = np.array(sorted(smi.index))
    X, ok = S.featurize(smi.loc[keys].tolist(), a.procs, S.OUT / "feats_e180.npz")
    obs = S.OBS4 if a.labels == 4 else S.OBS7
    _, lab = S.labels(t, obs)
    ix = np.flatnonzero(ok)
    Y = np.stack([lab[keys[i]] for i in ix])
    models = S.fit(X[ix], Y, a.trees)
    meta = dict(version="envida-aprior-2.0", labels=list(obs), trees=a.trees, n_train=int(len(ix)),
                label_rates=[round(float(v), 4) for v in Y.mean(0)], feature_dim=int(X.shape[1]),
                featuriser="goodpjw2008 _ap_featurizer: Morgan count r2 1024 + 22 SMARTS counts + 11 descriptors",
                boosters=[])
    for j, m in enumerate(models):
        p = out / f"aprior_{j}.txt"
        m.booster_.save_model(str(p))
        meta["boosters"].append(dict(file=p.name, label=obs[j], sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    Xc = np.stack([S.feats(s) for s in CHECK_SMILES])
    Pc = np.stack([m.predict_proba(Xc)[:, 1] for m in models], 1)
    (out / "aprior_check.json").write_text(json.dumps(dict(smiles=CHECK_SMILES, proba=Pc.round(6).tolist()), indent=1))
    (out / "aprior_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    sys.exit(main())
