"""Offline study of the APRIOR adduct-pattern prior (goodpjw2008's 0.498 notebook, cell "Ours: APRIOR").

APRIOR learns P(adduct observed | structure) from enveda-180 and re-orders ranks 2-25 of each answer list by
-GAMMA*ln(rank) + LAM*loglik(observed pattern | candidate). This script asks, without a board submission:
  1. how good the per-adduct classifiers are (held-out enveda-180 molecules and the 250 enveda NP examples),
  2. how often the truth beats its own same-mass candidates on loglik alone (pairwise AUC within the list),
  3. what a ranks 2-25 reorder does to MRR on simulated lists (truth at a base rank drawn from a geometric
     distribution, the other slots hard same-mass decoys), for LAM in a grid.
Variants: 'base' = the notebook's recipe (60k sampled molecules, 300 trees, 4 observed labels); 'all' = every training
molecule; 'all7' = every training molecule and 7 observed labels.

    python scripts/aprior_study.py [--eval-n 3000] [--procs 10]
Writes build/aprior_study/report.json and features cache build/aprior_study/feats_*.npy.
"""
import argparse
import json
import math
import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build" / "aprior_study"
TA = ("[M+H]+", "[M-H]-", "[M+CH2O2-H]-", "[M+Na]+", "[M+NH4]+", "[M+K]+", "[M+Cl]-")   # notebook _AP_TA
OBS4 = ("[M+H]+", "[M-H]-", "[M+CH2O2-H]-", "[M+Na]+")                                       # notebook _AP_OBS
OBS7 = TA
SMARTS = {   # notebook _AP_SMARTS, unchanged
    "cooh": "[CX3](=O)[OX2H1]", "phenol": "c[OX2H1]", "sulfo": "S(=O)(=O)[OH]", "phos": "P(=O)[OH]",
    "amine_basic": "[NX3;H2,H1,H0;!$(NC=O);!$(N-a);!$(N-S(=O)=O);!$(N-C=N)]", "aniline": "[NX3;H2,H1;$(N-a)]",
    "pyridine": "[nX2]", "amide": "C(=O)[NX3]", "imidazole": "c1ncc[nH]1", "guan": "NC(=N)N",
    "ester": "[CX3](=O)[OX2][#6]", "lactone": "[C;R](=O)[O;R]", "glyco": "[OX2H][CX4][CX4][OX2H]", "enol": "[OX2H]C=C",
    "oh_any": "[OX2H]", "nh_any": "[#7;!H0]", "quat": "[N+;!$(N=O);!$(N-[O-])]", "nitro": "[N+](=O)[O-]", "thiol": "[SX2H]",
    "carbonyl": "[CX3]=O", "halogen": "[Cl,Br,I]",
    "acid_any": "[$([CX3](=O)[OX2H1]),$(S(=O)(=O)[OH]),$(P(=O)[OH]),$(c[OX2H1])]",
}
DIM = 1024 + len(SMARTS) + 11
_F = {}


def _init():
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    _F["pats"] = [Chem.MolFromSmarts(v) for v in SMARTS.values()]
    _F["gen"] = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)


def feats(smi):
    """The notebook's _ap_featurizer: Morgan counts (r2, 1024), SMARTS counts, 11 descriptors; None if unparsable."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    if not _F:
        _init()
    try:
        m = Chem.MolFromSmiles(smi)
    except Exception:
        m = None
    if m is None:
        return None
    fp = np.asarray(_F["gen"].GetCountFingerprintAsNumPy(m), dtype=np.float32)
    sm = np.array([len(m.GetSubstructMatches(p)) for p in _F["pats"]], np.float32)
    d = np.array([Descriptors.MolWt(m), Descriptors.MolLogP(m), Descriptors.TPSA(m), rdMolDescriptors.CalcNumHBD(m),
                  rdMolDescriptors.CalcNumHBA(m), rdMolDescriptors.CalcNumAromaticRings(m), m.GetNumHeavyAtoms(),
                  sum(a.GetSymbol() == "N" for a in m.GetAtoms()), sum(a.GetSymbol() == "O" for a in m.GetAtoms()),
                  sum(a.GetSymbol() == "S" for a in m.GetAtoms()), Chem.GetFormalCharge(m)], np.float32)
    return np.concatenate([fp, sm, d])


def _batch(smis):
    out = np.zeros((len(smis), DIM), np.float32); ok = np.zeros(len(smis), bool)
    for i, s in enumerate(smis):
        f = feats(s)
        if f is not None:
            out[i] = f; ok[i] = True
    return out, ok


def featurize(smiles, procs, cache):
    if cache.is_file():
        z = np.load(cache)
        if len(z["ok"]) == len(smiles):
            return z["X"], z["ok"]
    chunks = [smiles[i:i + 2000] for i in range(0, len(smiles), 2000)]
    with Pool(procs, initializer=_init) as p:
        res = p.map(_batch, chunks)
    X = np.concatenate([r[0] for r in res]); ok = np.concatenate([r[1] for r in res])
    np.savez(cache, X=X, ok=ok)
    return X, ok


def labels(df, obs):
    has = df.groupby("inchikey14")["adduct"].agg(set)
    return has, {k: np.array([a in s for a in obs], np.float32) for k, s in has.items()}


def fit(X, Y, trees, seed=0):
    import lightgbm as lgb
    return [lgb.LGBMClassifier(n_estimators=trees, learning_rate=0.05, num_leaves=63, min_child_samples=40, subsample=0.8,
                               subsample_freq=1, colsample_bytree=0.5, random_state=seed, n_jobs=12, verbose=-1).fit(X, Y[:, j])
            for j in range(Y.shape[1])]


def proba(models, X):
    return np.clip(np.stack([m.predict_proba(X)[:, 1] for m in models], 1), 1e-3, 1 - 1e-3)


def auc(y, s):
    y = np.asarray(y, bool)
    if y.all() or not y.any():
        return float("nan")
    r = pd.Series(s).rank().to_numpy()
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def ap_rerank(n, ll, lam, gamma=1.5):
    score = [-gamma * math.log(i + 1) + lam * ll[i] for i in range(n)]
    order = sorted(range(n), key=lambda i: (-score[i], i))
    return [0] + [i for i in order if i != 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-n", type=int, default=3000)
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--pool", default="build/judge_data")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    t = pq.read_table(ROOT / "data/train.parquet", columns=["ingest_lib", "inchikey14", "normalized_smiles", "adduct"])
    t = t.filter(pc.is_in(t["ingest_lib"], value_set=pa.array(["enveda-180", "enveda-np-examples"]))).to_pandas()
    t = t[t.adduct.isin(TA)]
    e180 = t[t.ingest_lib == "enveda-180"]; np_ex = t[t.ingest_lib == "enveda-np-examples"]
    smi = e180.drop_duplicates("inchikey14").set_index("inchikey14").normalized_smiles
    keys = np.array(sorted(smi.index))
    hold = np.array([zlib.crc32(k.encode()) % 100 < 15 for k in keys])
    tr_keys, ev_keys = keys[~hold], keys[hold]
    print(f"enveda-180 molecules {len(keys):,}: train {len(tr_keys):,} / held out {len(ev_keys):,}; NP examples "
          f"{np_ex.inchikey14.nunique()} ({time.time() - t0:.0f}s)", flush=True)
    X, ok = featurize(smi.loc[keys].tolist(), a.procs, OUT / "feats_e180.npz")
    print(f"featurised {ok.sum():,}/{len(ok):,} ({time.time() - t0:.0f}s)", flush=True)
    kix = {k: i for i, k in enumerate(keys)}

    _, lab4 = labels(e180, OBS4); _, lab7 = labels(e180, OBS7)
    _, nlab4 = labels(np_ex, OBS4); _, nlab7 = labels(np_ex, OBS7)
    np_smi = np_ex.drop_duplicates("inchikey14").set_index("inchikey14").normalized_smiles
    np_keys = np.array(sorted(np_smi.index))
    XN, okN = featurize(np_smi.loc[np_keys].tolist(), 2, OUT / "feats_np.npz")

    tr_ix = np.array([kix[k] for k in tr_keys]); tr_ix = tr_ix[ok[tr_ix]]
    rng = np.random.default_rng(0)
    variants = {"base": (rng.permutation(tr_ix)[:60000], 300, OBS4), "all": (tr_ix, 300, OBS4),
                "all600": (tr_ix, 600, OBS4), "all7": (tr_ix, 300, OBS7)}
    models = {}
    rep = {"n_train_all": int(len(tr_ix)), "variants": {}}
    for name, (ix, trees, obs) in variants.items():
        lab = lab4 if obs is OBS4 else lab7
        Y = np.stack([lab[keys[i]] for i in ix])
        models[name] = fit(X[ix], Y, trees)
        ev_ix = np.array([kix[k] for k in ev_keys]); ev_ix = ev_ix[ok[ev_ix]]
        P = proba(models[name], X[ev_ix]); Ye = np.stack([lab[keys[i]] for i in ev_ix])
        nl = nlab4 if obs is OBS4 else nlab7
        nk = np_keys[okN]; PN = proba(models[name], XN[okN]); YN = np.stack([nl[k] for k in nk])
        rep["variants"][name] = dict(
            n_train=int(len(ix)), trees=trees, labels=list(obs),
            auc_e180={o: round(auc(Ye[:, j], P[:, j]), 4) for j, o in enumerate(obs)},
            auc_np={o: round(auc(YN[:, j], PN[:, j]), 4) for j, o in enumerate(obs)})
        print(name, json.dumps(rep["variants"][name]), f"({time.time() - t0:.0f}s)", flush=True)

    # ---- candidate lists: hard same-mass decoys from our pool (train + COCONUT), truth excluded
    pool = pd.read_parquet(ROOT / a.pool / "pool.parquet")
    pfp = np.load(ROOT / a.pool / "pool_fp.npy", mmap_mode="r")
    pmass = pool.mass.to_numpy(); pkey = pool.inchikey14.astype(str).to_numpy()
    k2p = {k: i for i, k in enumerate(pkey)}
    pop8 = np.array([bin(i).count("1") for i in range(256)], np.int32)
    ev_sel = rng.permutation(ev_keys)[:a.eval_n]
    panels = {"e180": [(k, lab4[k], lab7[k]) for k in ev_sel],
              "np": [(k, nlab4[k], nlab7[k]) for k in np_keys]}
    smiles_of = dict(smi); smiles_of.update(dict(np_smi))
    lists = {}
    need = {}
    for pname, rows in panels.items():
        L = []
        for k, y4, y7 in rows:
            pid = k2p.get(k)
            if pid is None:
                continue
            m = pmass[pid]; tol = m * 5e-6
            lo, hi = np.searchsorted(pmass, m - tol), np.searchsorted(pmass, m + tol, "right")
            others = np.array([j for j in range(lo, hi) if j != pid and pkey[j] != k])
            if len(others) < 4:
                continue
            A = np.asarray(pfp[pid]); B = np.asarray(pfp[others])
            tani = pop8[A & B].sum(1) / np.maximum(pop8[A | B].sum(1), 1)
            o = others[np.argsort(-tani, kind="stable")]
            dec = list(o[:16]) + list(rng.permutation(o[16:])[:8])
            L.append((k, y4, y7, dec))
            for j in dec:
                need[pool.smiles.iat[j]] = None
        lists[pname] = L
    cand_smi = list(need)
    XC, okC = featurize(cand_smi, a.procs, OUT / f"feats_cand_{a.eval_n}.npz")
    cix = {s: i for i, s in enumerate(cand_smi)}
    print(f"lists: e180 {len(lists['e180'])}, np {len(lists['np'])}; {len(cand_smi):,} candidate structures "
          f"({time.time() - t0:.0f}s)", flush=True)

    geo = np.array([0.5 ** (r - 2) for r in range(2, 26)]); geo /= geo.sum()   # truth's base rank 2..25
    rep["lists"] = {}
    for pname, L in lists.items():
        rep["lists"][pname] = {}
        for name, mods in models.items():
            obs7 = len(variants[name][2]) == 7
            pair, gains = [], {lam: [] for lam in (0.25, 0.5, 0.75, 1.0, 1.5)}
            ll_sd = []
            r2 = np.random.default_rng(1)
            for k, y4, y7, dec in L:
                y = y7 if obs7 else y4
                tf = feats(smiles_of[k])
                rowsX = [tf] + [XC[cix[pool.smiles.iat[j]]] if okC[cix[pool.smiles.iat[j]]] else None for j in dec]
                keep = [i for i, r in enumerate(rowsX) if r is not None]
                if 0 not in keep or len(keep) < 5:
                    continue
                P = proba(mods, np.stack([rowsX[i] for i in keep]))
                ll = (y * np.log(P) + (1 - y) * np.log(1 - P)).sum(1)
                ll_sd.append(float(ll.std()))
                pair.append(float((ll[0] > ll[1:]).mean() + 0.5 * (ll[0] == ll[1:]).mean()))
                # simulated list: slot 0 a decoy (rank 1 is kept), truth at base rank r (2..), decoys elsewhere
                n = min(25, len(ll))
                r = int(r2.choice(np.arange(2, 26), p=geo))
                r = min(r, n)
                dec_ll = list(r2.permutation(ll[1:])[:n - 1])
                seq = dec_ll[:r - 1] + [ll[0]] + dec_ll[r - 1:]
                for lam in gains:
                    order = ap_rerank(n, seq, lam)
                    new_r = order.index(r - 1) + 1
                    gains[lam].append(1.0 / new_r - 1.0 / r)
            rep["lists"][pname][name] = dict(n=len(pair), pair_auc=round(float(np.mean(pair)), 4),
                                              ll_sd=round(float(np.median(ll_sd)), 3),
                                              mrr_gain={str(l): round(float(np.mean(g)), 4) for l, g in gains.items()})
            print(pname, name, json.dumps(rep["lists"][pname][name]), flush=True)
    (OUT / "report.json").write_text(json.dumps(rep, indent=1))
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    sys.exit(main())
