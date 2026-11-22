"""Patch goodpjw2008's APRIOR cell with an intensity term (BPI): ll += (W / LAM) * -(y - pred)^2 / (2 sigma^2), where y is
the molecule's mean log10(base_peak_intensity) of [M+H]+ / [M-H]- spectra minus the enveda-180 mean of the same
(adduct, collision-energy setting), and pred a LightGBM regressor trained in the cell on the same molecules as APRIOR.
Offline (scripts/bpi_study.py): APRIOR alone +0.0162 reciprocal rank per affected list, APRIOR + BPI at W 0.5 +0.0225.
Every BPI step is wrapped so that a failure leaves plain APRIOR running.

    python scripts/patch_aprior_bpi.py <in notebook.ipynb> <out notebook.ipynb> [--w 0.5]
    python scripts/patch_aprior_bpi.py --cell-only <in notebook.ipynb> <out .py>   (patched cell source, for tests)
"""
import argparse
import json
from pathlib import Path

MARK = "# ===== Ours: APRIOR - adduct-pattern prior"

PATCHES = [
    ('_AP_VERSION = "1.0.0"\n',
     '_AP_VERSION = "1.0.0"\n'
     '# ---- envida patch: base-peak-intensity term (scripts/patch_aprior_bpi.py in akkki8844/envida-casmi-kaggle) ----\n'
     '_AP_BPI_W = __W__        # weight of the intensity log-likelihood in the -GAMMA*ln(rank) + LAM*ll + W*ll_bpi score\n'
     '_AP_BPI_ADS = ("[M+H]+", "[M-H]-")\n'
     '_AP_BPI_SIG = {"[M+H]+": 0.78, "[M-H]-": 0.62}   # held-out RMSE of the regressors (log10 units)\n'
     '_AP_BPI = {}\n'),
    ('    smi = dict(zip(sm["inchikey14"], sm["normalized_smiles"]))\n',
     '    smi = dict(zip(sm["inchikey14"], sm["normalized_smiles"]))\n'
     '    bpi_t = {}\n'
     '    if _AP_BPI_W > 0:\n'
     '        try:\n'
     '            bt = pq.read_table(train_path, columns=["ingest_lib", "inchikey14", "adduct", "base_peak_intensity", "collision_energy_orig"])\n'
     '            bt = bt.filter(pc.equal(bt["ingest_lib"], "enveda-180"))\n'
     '            bt = bt.filter(pc.is_in(bt["adduct"], value_set=__import__("pyarrow").array(list(_AP_BPI_ADS)))).to_pandas()\n'
     '            bt = bt[bt["base_peak_intensity"].astype(float) > 0]\n'
     '            bt["lb"] = np.log10(bt["base_peak_intensity"].astype(float))\n'
     '            bt["ce"] = bt["collision_energy_orig"].astype(str)\n'
     '            _mu = bt.groupby(["adduct", "ce"])["lb"].mean()\n'
     '            bt["lb"] = bt["lb"] - bt.groupby(["adduct", "ce"])["lb"].transform("mean")\n'
     '            bt = bt[bt["inchikey14"].isin(chosen)]\n'
     '            bpi_t = {a: g.groupby("inchikey14")["lb"].mean().to_dict() for a, g in bt.groupby("adduct")}\n'
     '            _AP_BPI["mu"] = {(a, c): float(v) for (a, c), v in _mu.items()}\n'
     '        except Exception as _e:\n'
     '            _AP_BPI["error"] = "targets: " + repr(_e)[:200]\n'
     '            bpi_t = {}\n'),
    ('    X, Y = [], []\n', '    X, Y, KK = [], [], []\n'),
    ('        X.append(f)\n        Y.append([k in has[a] for a in _AP_OBS])\n',
     '        X.append(f)\n        Y.append([k in has[a] for a in _AP_OBS])\n        KK.append(k)\n'),
    ('    info = dict(train_molecules=int(len(X)),',
     '    if bpi_t:\n'
     '        try:\n'
     '            import lightgbm as lgb\n'
     '            regs = {}\n'
     '            for a in _AP_BPI_ADS:\n'
     '                ix = [i for i, k in enumerate(KK) if k in bpi_t.get(a, {})]\n'
     '                if len(ix) < 1000:\n'
     '                    continue\n'
     '                regs[a] = lgb.LGBMRegressor(n_estimators=600 if trees >= 300 else trees, learning_rate=0.05, num_leaves=63,\n'
     '                                            min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.5,\n'
     '                                            random_state=seed, n_jobs=4, verbose=-1).fit(\n'
     '                    X[ix], np.array([bpi_t[a][KK[i]] for i in ix], np.float32))\n'
     '            _AP_BPI["models"] = regs\n'
     '        except Exception as _e:\n'
     '            _AP_BPI["error"] = "fit: " + repr(_e)[:200]\n'
     '    info = dict(train_molecules=int(len(X)),'),
    ('def ap_loglik(models, feats, smiles_list, observed):\n',
     'def ap_loglik(models, feats, smiles_list, observed, bpi_y=None):\n'),
    ('    ll = (y * np.log(P) + (1 - y) * np.log(1 - P)).sum(1)\n',
     '    ll = (y * np.log(P) + (1 - y) * np.log(1 - P)).sum(1)\n'
     '    if bpi_y and _AP_BPI_W > 0 and _AP_BPI.get("models"):\n'
     '        try:\n'
     '            add = np.zeros(len(ll))\n'
     '            for a, yv in bpi_y.items():\n'
     '                m = _AP_BPI["models"].get(a)\n'
     '                if m is not None:\n'
     '                    add -= (float(yv) - m.predict(X)) ** 2 / (2.0 * _AP_BPI_SIG[a] ** 2)\n'
     '            if np.all(np.isfinite(add)):\n'
     '                ll = ll + (_AP_BPI_W / _AP_LAM) * add\n'
     '        except Exception as _e:\n'
     '            _AP_BPI["error"] = "predict: " + repr(_e)[:200]\n'),
    ('            test = pd.read_parquet(locate("test.parquet"), columns=["molecule_id", "adduct"])\n',
     '            test = pd.read_parquet(locate("test.parquet"), columns=["molecule_id", "adduct"])\n'
     '            try:\n'
     '                bpi_test = pd.read_parquet(locate("test.parquet"), columns=["molecule_id", "adduct", "base_peak_intensity",\n'
     '                                                                         "collision_energy_orig"])\n'
     '            except Exception as _e:\n'
     '                bpi_test = None\n'
     '                _AP_BPI["error"] = "test columns: " + repr(_e)[:200]\n'),
    ('            models, info = ap_train(locate("train.parquet"), deadline=deadline)\n            audit.update(info)\n',
     '            models, info = ap_train(locate("train.parquet"), deadline=deadline)\n            audit.update(info)\n'
     '            bpi_y = {}\n'
     '            try:\n'
     '                import numpy as np\n'
     '                if _AP_BPI_W > 0 and _AP_BPI.get("models") and _AP_BPI.get("mu") and bpi_test is not None:\n'
     '                    tt = bpi_test[bpi_test["adduct"].astype(str).isin(_AP_BPI_ADS)].copy()\n'
     '                    tt["mu"] = [_AP_BPI["mu"].get((a, c), np.nan) for a, c in\n'
     '                                zip(tt["adduct"].astype(str), tt["collision_energy_orig"].astype(str))]\n'
     '                    tt["bp"] = tt["base_peak_intensity"].astype(float)\n'
     '                    tt = tt[np.isfinite(tt["mu"].astype(float)) & (tt["bp"] > 0)]\n'
     '                    tt["y"] = np.log10(tt["bp"]) - tt["mu"].astype(float)\n'
     '                    for (m_, a_), v in tt.groupby([tt["molecule_id"].astype(str), tt["adduct"].astype(str)])["y"].mean().items():\n'
     '                        bpi_y.setdefault(m_, {})[a_] = float(v)\n'
     '            except Exception as _e:\n'
     '                bpi_y = {}\n'
     '                _AP_BPI["error"] = "test targets: " + repr(_e)[:200]\n'
     '            audit["bpi"] = dict(w=_AP_BPI_W, models=sorted(_AP_BPI.get("models", {})), molecules=len(bpi_y),\n'
     '                                error=_AP_BPI.get("error"))\n'),
    ('                ll = ap_loglik(models, cfeats, cands, observed)\n',
     '                ll = ap_loglik(models, cfeats, cands, observed, bpi_y=bpi_y.get(mid))\n'),
    ('            print("[aprior] commit dry run ok:", _ap_json.dumps(_ap_info),',
     '            print("[aprior] envida BPI dry run:", sorted(_AP_BPI.get("models", {})), "mu", len(_AP_BPI.get("mu", {})),\n'
     '                  "error", _AP_BPI.get("error"), flush=True)\n'
     '            print("[aprior] commit dry run ok:", _ap_json.dumps(_ap_info),'),
]


def patch_source(src: str, w: float) -> str:
    for old, new in PATCHES:
        n = src.count(old)
        assert n == 1, f"patch anchor found {n} times: {old[:70]!r}"
        src = src.replace(old, new.replace("__W__", repr(float(w))))
    return src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--w", type=float, default=0.5)
    ap.add_argument("--cell-only", action="store_true")
    a = ap.parse_args()
    nb = json.loads(Path(a.inp).read_text(encoding="utf8"))
    hits = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code" and "".join(c["source"]).startswith(MARK)]
    assert len(hits) == 1, f"APRIOR cell found {len(hits)} times"
    i = hits[0]
    src = patch_source("".join(nb["cells"][i]["source"]), a.w)
    if a.cell_only:
        Path(a.out).write_text(src, encoding="utf8")
    else:
        nb["cells"][i]["source"] = src.splitlines(keepends=True)
        # compact JSON: Kaggle refuses kernel sources of 1 MB or more
        Path(a.out).write_text(json.dumps(nb, separators=(",", ":")), encoding="utf8")
    print(f"patched cell {i} with W={a.w} -> {a.out}")


if __name__ == "__main__":
    main()
