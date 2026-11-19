import numpy as np
import pandas as pd

from casmi.baseline import BaselineRanker, Params
from casmi.chem import key14
from casmi.metric import mrr_at_25
from casmi.predict import predict
from tests.test_baseline import SMILES, _toy


def test_predict_end_to_end_on_toy_data():
    pool, lib, spec = _toy()
    ranker = BaselineRanker(lib, pool, Params())
    rows = []
    for i, name in enumerate(["caffeine", "paracetamol"]):
        mz, inten, prec = spec[name]
        for j in range(2):  # two spectra per molecule
            rows.append({"molecule_id": f"m{i}", "spectrum_id": f"s{i}{j}", "ms2_mzs": mz,
                         "ms2_normalized_intensities": inten / inten.max(), "adduct": "[M+H]+",
                         "ionization_mode": "positive", "precursor_mz": prec})
    rows.append({"molecule_id": "weird", "spectrum_id": "x", "ms2_mzs": np.array([]),
                 "ms2_normalized_intensities": np.array([]), "adduct": "[M+X]+",
                 "ionization_mode": "positive", "precursor_mz": 5000.0})
    sub = predict(pd.DataFrame(rows), ranker)
    assert list(sub["molecule_id"]) == ["m0", "m1", "weird"]
    assert sub["smiles"].notna().all() and (sub["smiles"].str.len() > 0).all()
    preds = {r.molecule_id: r.smiles.split(";") for r in sub.itertuples()}
    assert mrr_at_25(preds, {"m0": SMILES["caffeine"], "m1": SMILES["paracetamol"]}) == 1.0
    assert all(len(v) <= 25 for v in preds.values())
    assert key14(preds["m0"][0]) == key14(SMILES["caffeine"])
