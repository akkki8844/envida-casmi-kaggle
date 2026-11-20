import numpy as np
import pandas as pd

from casmi.baseline import BaselineRanker, Exclusion, Params, Query
from casmi.chem import ADDUCT_DELTA, exact_mass, key14, morgan_bits
from casmi.library import Library, Pool
from casmi.spectra import spectrum_vectors, stack_csr

SMILES = {
    "caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "theobromine": "Cn1cnc2c1c(=O)[nH]c(=O)n2C",
    "theophylline": "Cn1c(=O)c2[nH]cnc2n(C)c1=O",
    "paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
}


def _toy():
    rng = np.random.default_rng(1)
    rows = [(n, s, exact_mass(s)) for n, s in SMILES.items()]
    df = pd.DataFrame(rows, columns=["name", "smiles", "mass"]).sort_values("mass").reset_index(drop=True)
    df["inchikey14"] = df["smiles"].map(key14)
    fps = np.stack([morgan_bits(s) for s in df["smiles"]])
    pool = Pool(df, fps, {k: i for i, k in enumerate(df["inchikey14"])})
    spec, meta, pk, nl = {}, [], [], []
    for i, r in df.iterrows():
        prec = r["mass"] + ADDUCT_DELTA["[M+H]+"]
        mz = np.sort(rng.uniform(40, prec, 25))
        inten = rng.uniform(0.05, 1.0, 25)
        spec[r["name"]] = (mz, inten, prec)
        p, l = spectrum_vectors(mz, inten, prec)
        pk.append(p)
        nl.append(l)
        meta.append({"row": i, "struct_id": i, "ingest_lib": "gnps", "mode": 0, "adduct": "[M+H]+",
                     "precursor_mz": prec, "neutral_mass": r["mass"], "collision_energy_ev": "20"})
    meta = pd.DataFrame(meta).sort_values("neutral_mass").reset_index(drop=True)
    order = pd.DataFrame(meta)["row"].to_numpy()
    lib = Library(meta, stack_csr([pk[i] for i in order]), stack_csr([nl[i] for i in order]))
    return pool, lib, spec


def _query(spec, name):
    mz, inten, prec = spec[name]
    p, l = spectrum_vectors(mz, inten, prec)
    return Query(stack_csr([p]), stack_csr([l]), prec - ADDUCT_DELTA["[M+H]+"], 0)


def test_library_hit_ranks_first():
    pool, lib, spec = _toy()
    r = BaselineRanker(lib, pool, Params())
    guesses, diag = r.rank(_query(spec, "caffeine"))
    assert key14(guesses[0]) == key14(SMILES["caffeine"])
    assert diag["best_sim"] > 0.99


def test_hidden_truth_keeps_candidate_in_pool_but_not_first_by_anchor():
    pool, lib, spec = _toy()
    r = BaselineRanker(lib, pool, Params())
    truth = pool.key_to_id[key14(SMILES["caffeine"])]
    ex = Exclusion(lib_struct_ids=np.array([truth]))
    guesses, _ = r.rank(_query(spec, "caffeine"), ex)
    keys = [key14(g) for g in guesses]
    assert key14(SMILES["caffeine"]) in keys  # still reachable through the pool (class-2 style)


def test_features_align_with_candidates_and_heuristic():
    pool, lib, spec = _toy()
    r = BaselineRanker(lib, pool, Params())
    q = _query(spec, "caffeine")
    cand, X, _ = r.features(q, Exclusion())
    assert X.shape == (cand.size, len(BaselineRanker.FEATURES))
    assert np.isfinite(X).all()
    h = X[:, BaselineRanker.FEATURES.index("h_score")]
    assert key14(pool.df["smiles"].iat[int(cand[np.argmax(h)])]) == key14(SMILES["caffeine"])
    # the heuristic path and the feature path must agree on the order
    c2, final, _ = r.score(q, Exclusion())
    assert np.array_equal(cand, c2) and np.allclose(final, h, atol=1e-6)


def test_class3_removes_truth_from_pool():
    pool, lib, spec = _toy()
    r = BaselineRanker(lib, pool, Params())
    truth = pool.key_to_id[key14(SMILES["caffeine"])]
    ex = Exclusion(lib_struct_ids=np.array([truth]), pool_ids=np.array([truth]))
    guesses, _ = r.rank(_query(spec, "caffeine"), ex)
    assert key14(SMILES["caffeine"]) not in [key14(g) for g in guesses]
