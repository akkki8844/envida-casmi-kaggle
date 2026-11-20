"""How often is an external-library hit (the extlib overlay's rule) the right structure?

    python scripts/extlib_precision.py [--e180 300]

Queries: timsTOF spectra from train.parquet (all enveda-np-examples molecules + a sample of enveda-180), grouped per
structure like a test molecule. For each, the overlay's matcher runs against the external libraries in data/extlib
(NIST rows dropped) and we record whether it fires, the tie size, and whether the hit is the true structure.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle_layers"))
import extlib_overlay as X  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e180", type=int, default=300)
    ap.add_argument("--out", default="data/extlib_precision.csv")
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--mb-precursor", action="store_true")
    a = ap.parse_args()
    if a.tau is not None:
        X.TAU = a.tau
    X.MB_USE_PRECURSOR = a.mb_precursor
    pf = pq.ParquetFile(ROOT / "data" / "train.parquet")
    meta = pf.read(columns=["ingest_lib", "inchikey14"]).to_pandas()
    rng = np.random.default_rng(0)
    np_keys = meta.loc[meta.ingest_lib == "enveda-np-examples", "inchikey14"].unique()
    e_keys = meta.loc[meta.ingest_lib == "enveda-180", "inchikey14"].unique()
    e_keys = rng.choice(e_keys, size=min(a.e180, len(e_keys)), replace=False)
    want = {("enveda-np-examples", k) for k in np_keys} | {("enveda-180", k) for k in e_keys}
    rows = np.flatnonzero([(l, k) in want for l, k in zip(meta.ingest_lib, meta.inchikey14)])
    t = pf.read(columns=["ingest_lib", "inchikey14", "ms2_mzs", "ms2_normalized_intensities", "precursor_mz",
                         "ionization_mode", "normalized_smiles"]).take(rows).to_pandas()
    print(f"{len(t)} spectra, {t.groupby(['ingest_lib', 'inchikey14']).ngroups} query molecules", flush=True)
    refs, n_nist = X.load_refs(str(ROOT / "data" / "extlib"))
    order = np.argsort([r[1] for r in refs], kind="stable")
    refs = [refs[i] for i in order]
    prec_arr = np.array([r[1] for r in refs])
    out = []
    for (lib, key), g in t.groupby(["ingest_lib", "inchikey14"]):
        scored = []
        for r in g.itertuples(index=False):
            m = np.asarray(r.ms2_mzs, dtype=np.float64)
            it = np.asarray(r.ms2_normalized_intensities, dtype=np.float64)
            prec = float(r.precursor_mz)
            mode = 1 if "pos" in str(r.ionization_mode).lower() else -1
            ok = (m > 0) & (it > 0) & (m <= prec + 0.5)
            m, it = m[ok], it[ok]
            o = np.argsort(m)
            m, it = m[o], it[o]
            tol = prec * X.PPM + X.ABS_TOL
            for ri in range(np.searchsorted(prec_arr, prec - tol), np.searchsorted(prec_arr, prec + tol, "right")):
                ref = refs[ri]
                if ref[0] != mode:
                    continue
                cos, n = X.greedy_cosine(m, it, ref[2], ref[3])
                if n >= X.K_PEAKS and cos >= X.TAU:
                    scored.append((cos, n, ref[4], ref[5], r.Index if hasattr(r, "Index") else id(r)))
        rec = dict(lib=lib, key=key, n_spec=len(g), fired=bool(scored))
        if scored:
            best = max(s[0] for s in scored)
            tie = {}
            for cos, n, smi, src, _ in sorted(scored, key=lambda s: -s[0]):
                if cos < best - X.TIE_EPS:
                    break
                tie.setdefault(X._key14(smi) or smi, (smi, src))
            first_key = next(iter(tie))
            # vote rule: per structure, best cosine per query spectrum, summed over the query's spectra
            per = {}
            for cos, n, smi, src, sp in scored:
                k = X._key14(smi) or smi
                per.setdefault(k, {})
                per[k][sp] = max(per[k].get(sp, 0.0), cos)
            vote_key = max(per, key=lambda k: (sum(per[k].values()), max(per[k].values())))
            rec.update(best_cos=round(best, 4), tie=len(tie), src=next(iter(tie.values()))[1],
                       public_rule_correct=first_key == key, truth_in_tie=key in tie,
                       truth_anywhere=key in per, vote_correct=vote_key == key)
        out.append(rec)
    df = pd.DataFrame(out)
    df.to_csv(ROOT / a.out, index=False)
    for lib, g in df.groupby("lib"):
        f = g[g.fired]
        print(f"\n{lib}: {len(g)} molecules, overlay fires on {len(f)} ({len(f) / len(g):.1%})")
        if len(f):
            print(f"  public rule (first hit) correct: {f.public_rule_correct.mean():.1%}; vote rule correct: "
                  f"{f.vote_correct.mean():.1%}; truth in tie set: {f.truth_in_tie.mean():.1%}; "
                  f"truth among any hit: {f.truth_anywhere.mean():.1%}")
            for name, sel in (("unique hit", f.tie == 1), ("tie 2-5", (f.tie > 1) & (f.tie <= 5)), ("tie > 5", f.tie > 5)):
                s = f[sel]
                if len(s):
                    print(f"  {name:>10}: n={len(s):3d}  correct(first)={s.public_rule_correct.mean():.2f}  "
                          f"truth-in-tie={s.truth_in_tie.mean():.2f}  sources={s.src.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
