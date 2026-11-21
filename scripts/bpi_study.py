"""Offline study: is a molecule's MS2 base-peak intensity (test.parquet base_peak_intensity) predictable from structure,
well enough to tell same-mass candidates apart (an "intensity prior" for ranks 2-25, in the style of APRIOR)?

Per enveda-180 molecule and adduct ([M+H]+, [M-H]-): y = mean over its spectra of log10(base_peak_intensity) minus the
campaign mean of that collision-energy setting. One LightGBM regressor per adduct on the APRIOR features
(scripts/aprior_study.py). Candidate term: ll(s) = -sum_a (y_a - pred_a(s))^2 / (2 sigma_a^2), sigma_a = held-out RMSE.
Lists, panels and the simulated ranks 2-25 reorder are those of aprior_study.py (same seeds, same cached features).

    python scripts/bpi_study.py
"""
import json
import math
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import aprior_study as S  # noqa: E402

ADS = ("[M+H]+", "[M-H]-")


def targets(df):
    df = df[df.adduct.isin(ADS)].copy()
    df["lb"] = np.log10(df.base_peak_intensity.clip(lower=1.0))
    df["lb"] -= df.groupby(["adduct", "collision_energy_orig"]).lb.transform("mean")
    return {a: g.groupby("inchikey14").lb.mean() for a, g in df.groupby("adduct")}


def main():
    t0 = time.time()
    t = pq.read_table(ROOT / "data/train.parquet", columns=["ingest_lib", "inchikey14", "normalized_smiles", "adduct",
                                                            "base_peak_intensity", "collision_energy_orig"])
    t = t.filter(pc.is_in(t["ingest_lib"], value_set=pa.array(["enveda-180", "enveda-np-examples"]))).to_pandas()
    e180 = t[(t.ingest_lib == "enveda-180") & t.adduct.isin(S.TA)]
    np_ex = t[(t.ingest_lib == "enveda-np-examples") & t.adduct.isin(S.TA)]
    smi = e180.drop_duplicates("inchikey14").set_index("inchikey14").normalized_smiles
    keys = np.array(sorted(smi.index))
    hold = np.array([zlib.crc32(k.encode()) % 100 < 15 for k in keys])
    X, ok = S.featurize(smi.loc[keys].tolist(), 10, S.OUT / "feats_e180.npz")
    kix = {k: i for i, k in enumerate(keys)}
    Te = targets(e180)
    # NP examples: CE settings differ, so centre on the enveda-180 means of the same (adduct, CE)
    ref = e180[e180.adduct.isin(ADS)].copy(); ref["lb"] = np.log10(ref.base_peak_intensity.clip(lower=1.0))
    mu = ref.groupby(["adduct", "collision_energy_orig"]).lb.mean()
    npx = np_ex[np_ex.adduct.isin(ADS)].copy(); npx["lb"] = np.log10(npx.base_peak_intensity.clip(lower=1.0))
    npx["mu"] = [mu.get((a, c), np.nan) for a, c in zip(npx.adduct, npx.collision_energy_orig)]
    npx = npx[npx.mu.notna()]; npx["lb"] -= npx.mu
    Tn = {a: g.groupby("inchikey14").lb.mean() for a, g in npx.groupby("adduct")}
    np_smi = np_ex.drop_duplicates("inchikey14").set_index("inchikey14").normalized_smiles
    np_keys = np.array(sorted(np_smi.index))
    XN, okN = S.featurize(np_smi.loc[np_keys].tolist(), 2, S.OUT / "feats_np.npz")
    nix = {k: i for i, k in enumerate(np_keys)}

    import lightgbm as lgb
    models, sig, rep = {}, {}, {"adducts": {}}
    for a in ADS:
        y = Te[a]
        ks = np.array([k for k in y.index if k in kix and ok[kix[k]]])
        tr = np.array([not hold[kix[k]] for k in ks])
        ix = np.array([kix[k] for k in ks])
        m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=40, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.5, random_state=0, n_jobs=12, verbose=-1)
        m.fit(X[ix[tr]], y.loc[ks[tr]].to_numpy())
        pe = m.predict(X[ix[~tr]]); ye = y.loc[ks[~tr]].to_numpy()
        rmse = float(np.sqrt(np.mean((pe - ye) ** 2))); r2 = 1 - rmse ** 2 / float(ye.var())
        yn = Tn.get(a, pd.Series(dtype=float)); kn = [k for k in yn.index if k in nix and okN[nix[k]]]
        pn = m.predict(XN[[nix[k] for k in kn]]) if kn else np.zeros(0)
        r2n = 1 - float(np.mean((pn - yn.loc[kn].to_numpy()) ** 2)) / float(yn.loc[kn].var()) if len(kn) > 5 else float("nan")
        models[a], sig[a] = m, rmse
        rep["adducts"][a] = dict(n_train=int(tr.sum()), n_eval=int((~tr).sum()), rmse=round(rmse, 3), r2_e180=round(r2, 3),
                                 n_np=len(kn), r2_np=round(r2n, 3))
        print(a, rep["adducts"][a], f"({time.time() - t0:.0f}s)", flush=True)

    # ---- lists exactly as aprior_study.py builds them
    pool = pd.read_parquet(ROOT / "build/judge_data/pool.parquet")
    pfp = np.load(ROOT / "build/judge_data/pool_fp.npy", mmap_mode="r")
    pmass = pool.mass.to_numpy(); pkey = pool.inchikey14.astype(str).to_numpy()
    k2p = {k: i for i, k in enumerate(pkey)}
    pop8 = np.array([bin(i).count("1") for i in range(256)], np.int32)
    rng = np.random.default_rng(0)
    rng.permutation(np.flatnonzero(~hold))   # keep the rng stream aligned with aprior_study (its 60k sample draw)
    ev_keys = keys[hold]
    ev_sel = rng.permutation(ev_keys)[:3000]
    panels = {"e180": (list(ev_sel), Te), "np": (list(np_keys), Tn)}
    cz = np.load(S.OUT / "feats_cand_3000.npz")
    XC, okC = cz["X"], cz["ok"]
    smiles_of = dict(smi); smiles_of.update(dict(np_smi))
    need = {}
    lists = {}
    for pname, (rows, T) in panels.items():
        L = []
        for k in rows:
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
            L.append((k, dec))
            for j in dec:
                need[pool.smiles.iat[j]] = None
        lists[pname] = L
    cand = list(need)
    assert len(cand) == len(okC), f"candidate order differs from aprior_study cache: {len(cand)} vs {len(okC)}"
    cix = {s: i for i, s in enumerate(cand)}
    # APRIOR exactly as the notebook trains it (60k sampled training molecules, seed 0, 300 trees, 4 labels)
    tr_all = np.flatnonzero(~hold & ok)
    ap_ix = np.random.default_rng(0).permutation(tr_all)[:60000]
    _, lab4 = S.labels(e180, S.OBS4); _, nlab4 = S.labels(np_ex, S.OBS4)
    ap_models = S.fit(X[ap_ix], np.stack([lab4[keys[i]] for i in ap_ix]), 300)
    print(f"APRIOR base models refit ({time.time() - t0:.0f}s)", flush=True)
    geo = np.array([0.5 ** (r - 2) for r in range(2, 26)]); geo /= geo.sum()
    rep["joint"] = {}
    for pname, L in lists.items():
        T = panels[pname][1]; lab = lab4 if pname == "e180" else nlab4
        W = (0.0, 0.25, 0.5, 0.75, 1.0)
        gains = {w: [] for w in W}; r2 = np.random.default_rng(1)
        for k, dec in L:
            tf = S.feats(smiles_of[k])
            rowsX = [tf] + [XC[cix[pool.smiles.iat[j]]] if okC[cix[pool.smiles.iat[j]]] else None for j in dec]
            keep = [i for i, r in enumerate(rowsX) if r is not None]
            if 0 not in keep or len(keep) < 5 or k not in lab:
                continue
            Xl = np.stack([rowsX[i] for i in keep])
            P = S.proba(ap_models, Xl); y = lab[k]
            ll_ap = (y * np.log(P) + (1 - y) * np.log(1 - P)).sum(1)
            ll_b = np.zeros(len(keep))
            for a in ADS:
                if a in T and k in T[a].index:
                    ll_b -= (float(T[a][k]) - models[a].predict(Xl)) ** 2 / (2 * sig[a] ** 2)
            n = min(25, len(keep)); r = min(int(r2.choice(np.arange(2, 26), p=geo)), n)
            perm = r2.permutation(np.arange(1, len(keep)))[:n - 1]
            idx = list(perm[:r - 1]) + [0] + list(perm[r - 1:])
            for w in W:
                comb = [0.5 * ll_ap[i] + w * ll_b[i] for i in idx]
                order = S.ap_rerank(n, comb, 1.0)
                gains[w].append(1.0 / (order.index(r - 1) + 1) - 1.0 / r)
        rep["joint"][pname] = dict(n=len(gains[0.0]), mrr_gain={str(w): round(float(np.mean(g)), 4) for w, g in gains.items()})
        print("joint APRIOR(0.5) + w*BPI", pname, rep["joint"][pname], flush=True)
    rep["lists"] = {}
    for pname, L in lists.items():
        T = panels[pname][1]
        pair, gains, used = [], {lam: [] for lam in (0.25, 0.5, 1.0, 2.0)}, 0
        r2 = np.random.default_rng(1)
        for k, dec in L:
            obs = {a: float(T[a][k]) for a in ADS if a in T and k in T[a].index}
            if not obs:
                continue
            tf = S.feats(smiles_of[k])
            rowsX = [tf] + [XC[cix[pool.smiles.iat[j]]] if okC[cix[pool.smiles.iat[j]]] else None for j in dec]
            keep = [i for i, r in enumerate(rowsX) if r is not None]
            if 0 not in keep or len(keep) < 5:
                continue
            Xl = np.stack([rowsX[i] for i in keep])
            ll = np.zeros(len(keep))
            for a, yv in obs.items():
                ll -= (yv - models[a].predict(Xl)) ** 2 / (2 * sig[a] ** 2)
            used += 1
            pair.append(float((ll[0] > ll[1:]).mean()))
            n = min(25, len(ll)); r = min(int(r2.choice(np.arange(2, 26), p=geo)), n)
            dl = list(r2.permutation(ll[1:])[:n - 1]); seq = dl[:r - 1] + [ll[0]] + dl[r - 1:]
            for lam in gains:
                order = S.ap_rerank(n, seq, lam)
                gains[lam].append(1.0 / (order.index(r - 1) + 1) - 1.0 / r)
        rep["lists"][pname] = dict(n=used, pair_auc=round(float(np.mean(pair)), 4),
                                   mrr_gain={str(l): round(float(np.mean(g)), 4) for l, g in gains.items()})
        print(pname, rep["lists"][pname], flush=True)
    (S.OUT / "bpi_report.json").write_text(json.dumps(rep, indent=1))
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    sys.exit(main())
