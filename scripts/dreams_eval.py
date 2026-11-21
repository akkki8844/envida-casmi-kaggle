"""Do DreaMS analogue anchors add signal on top of v4? (harness study; nothing here ships to Kaggle yet)

    python scripts/dreams_eval.py --extra data/coconut/coconut_pool.parquet [--threads 8]

Library: precomputed DreaMS embeddings of train's enveda-180 spectra (index carries train_idx) + MassSpecGym spectra
(data/dreams/hengck23-dreams-enveda-casmi26, from dmitriigluzdov/casmi26-attributed-v4g-reproduction-assets).
Queries: the harness / training items' spectra embedded on CPU with the same DreaMS model, mean-pooled per molecule.
Hiding follows the harness: class 2 and 3 hide every library row of the structures in Exclusion.lib_struct_ids (truth;
for class 3 also its Tanimoto >= 0.7 neighbours); class 1 hides the truth's own enveda-180 rows (timsTOF).
Features per candidate against the top-50 anchor structures: max sim*T, max sim^3*T, sum of the top-10 sim*T,
max T over the top-10, the candidate's own best sim, and the query-relative d3. Appended to the v4 features.
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch  # only for the chunked library matmul
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
DD = ROOT / "data" / "dreams" / "hengck23-dreams-enveda-casmi26"

from casmi import fragments as fr  # noqa: E402
from casmi.baseline import BaselineRanker, Params  # noqa: E402
from casmi.chem import morgan_bits  # noqa: E402
from casmi.library import build_library, build_pool, load_raw_peaks, load_train_meta  # noqa: E402
from casmi.validation import DEFAULT_MIX, item_query, make_items, make_train_items  # noqa: E402
from train_v4 import aligned, rr  # noqa: E402

FB = BaselineRanker.BASE_FEATURES
NF = len(FB) + len(fr.FEATURES)
DFEAT = ["dr_s1T", "dr_s3T", "dr_sum10", "dr_tmax10", "dr_own", "dr_rel"]
TOPROWS, NANCH = 500, 50


class Safe(pickle.Unpickler):
    OK = {("numpy._core.multiarray", "_reconstruct"), ("numpy", "ndarray"), ("numpy", "dtype"),
          ("numpy._core.numeric", "_frombuffer")}

    def find_class(self, module, name):
        if (module, name) not in self.OK:
            raise pickle.UnpicklingError(f"blocked {module}.{name}")
        return super().find_class(module, name)


def library():
    idx = pd.read_parquet(DD / "Enveda180_DreaMS_index.parquet", columns=["inchikey14", "normalized_smiles",
                                                                          "ionization_mode"])
    msg = Safe(open(DD / "MassSpecGym_DreaMS_meta.pkl", "rb")).load()
    keys = np.concatenate([idx.inchikey14.astype(str).to_numpy(), np.array(msg["INCHIKEY"], dtype=object)])
    smis = np.concatenate([idx.normalized_smiles.astype(str).to_numpy(), np.array(msg["smiles"], dtype=object)])
    mode = np.concatenate([(idx.ionization_mode == "negative").to_numpy().astype(np.int8),
                           np.zeros(len(msg["smiles"]), np.int8)])
    is_e180 = np.zeros(len(keys), bool)
    is_e180[: len(idx)] = True
    embs = [np.load(DD / "Enveda180_DreaMS_embeddings.npy", mmap_mode="r"),
            np.load(DD / "MassSpecGym_DreaMS_embeddings.npy", mmap_mode="r")]
    return keys, smis, mode, is_e180, embs


def search(Q, qmode, embs, lmode, chunk=40000):
    """Top-TOPROWS library rows per query (cosine, same ion mode)."""
    nq = Q.shape[0]
    best_s = np.full((nq, TOPROWS), -np.inf, np.float32)
    best_i = np.zeros((nq, TOPROWS), np.int64)
    off = 0
    Qt = torch.from_numpy(Q)
    for E in embs:
        for a in tqdm(range(0, E.shape[0], chunk), desc="library pass"):
            X = torch.from_numpy(np.ascontiguousarray(E[a:a + chunk], dtype=np.float32))
            X = X / (X.norm(dim=1, keepdim=True) + 1e-9)
            S = (Qt @ X.T).numpy()
            S[qmode[:, None] != lmode[None, off + a: off + a + X.shape[0]]] = -np.inf
            cat_s = np.concatenate([best_s, S], 1)
            cat_i = np.concatenate([best_i, np.arange(off + a, off + a + X.shape[0])[None, :].repeat(nq, 0)], 1)
            top = np.argpartition(-cat_s, TOPROWS - 1, axis=1)[:, :TOPROWS]
            best_s = np.take_along_axis(cat_s, top, 1)
            best_i = np.take_along_axis(cat_i, top, 1)
        off += E.shape[0]
    return best_s, best_i


def features(cand, pool, anchors, fp_cache):
    n = cand.size
    F = np.full((n, len(DFEAT)), np.nan, np.float32)
    if not anchors:
        return F
    a_keys = [k for k, _, _ in anchors]
    s = np.array([v for _, v, _ in anchors], np.float32)
    afp = []
    for k, _, smi in anchors:
        if k not in fp_cache:
            sid = pool.key_to_id.get(k)
            fp_cache[k] = pool.fps[sid] if sid is not None else morgan_bits(smi)
        afp.append(fp_cache[k])
    A = np.stack(afp).astype(np.float32)
    C = pool.fps[cand].astype(np.float32)
    inter = C @ A.T
    T = inter / np.maximum(C.sum(1)[:, None] + A.sum(1)[None, :] - inter, 1e-9)
    F[:, 0] = (T * s).max(1)
    F[:, 1] = (T * s ** 3).max(1)
    F[:, 2] = (T[:, :10] * s[:10]).sum(1)
    F[:, 3] = T[:, :10].max(1)
    own = dict(zip(a_keys, s))
    keys = pool.df["inchikey14"].to_numpy()[cand]
    F[:, 4] = [own.get(k, 0.0) for k in keys]
    F[:, 5] = F[:, 1] - F[:, 1].max()
    return F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--extra", default=None)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--models", default="data/models")
    ap.add_argument("--dump-queries", action="store_true", help="only write the query spectra for the Kaggle job")
    a = ap.parse_args()
    md = Path(a.models)
    train = Path(a.data) / "train.parquet"
    meta = load_train_meta(train)
    pool = build_pool(meta, Path(a.cache), extra=a.extra)
    lib = build_library(train, meta, pool, Path(a.cache))
    del meta
    ranker = BaselineRanker(lib, pool, Params())
    dt, de = np.load(md / "train_features_v4.npz"), np.load(md / "eval_features_v4.npz", allow_pickle=True)
    Xt, yt, gt = dt["X"][:, :NF], dt["y"], dt["groups"]
    EX, Ey, Eg, Emeta = de["X"][:, :NF], de["y"], de["groups"], de["meta"]
    counts = {tuple(k): v for k, v in de["counts"]}
    eval_all = make_items(lib, pool, 60, seed=1)
    np_structs = set(lib.meta.loc[lib.meta["ingest_lib"] == "enveda-np-examples", "struct_id"].tolist())
    exclude = np_structs | {it.struct_id for it in eval_all}
    kt = aligned(ranker, lib, make_train_items(lib, pool, a.n, exclude), gt, yt)
    ke = aligned(ranker, lib, [it for it in eval_all if it.cls != 3], Eg, Ey)
    kept = kt + ke
    rows = {int(r) for it, _ in kept for r in lib.meta["row"].to_numpy()[it.rows]}
    raw = load_raw_peaks(train, rows)
    qs = [item_query(lib, it, raw) for it, _ in kept]

    qcache = md / "dreams_query_emb.npy"
    if a.dump_queries or not qcache.exists():
        # DreaMS runs on Kaggle (kaggle_jobs/dreams_embed), not locally: write the query spectra for that job
        out = ROOT / "build" / "dreams_queries"
        out.mkdir(parents=True, exist_ok=True)
        recs = [dict(q=i, mz=list(map(float, mz)), it=list(map(float, it)), prec=float(p))
                for i, q in enumerate(qs) for mz, it, p, _ in (q.peaks or [])]
        pd.DataFrame(recs).to_parquet(out / "queries.parquet")
        print(f"wrote {len(recs)} spectra of {len(qs)} queries to {out}; embed them on Kaggle, then save the "
              f"result as {qcache}")
        return
    Q = np.load(qcache)
    assert Q.shape[0] == len(qs), (Q.shape, len(qs))
    keys, smis, lmode, is_e180, embs = library()
    qmode = np.array([q.mode for q in qs], np.int8)
    scache = md / "dreams_search.npz"
    if scache.exists():
        d = np.load(scache)
        S, I = d["S"], d["I"]
    else:
        t0 = time.time()
        S, I = search(Q, qmode, embs, lmode)
        np.savez(scache, S=S, I=I)
        print(f"searched in {time.time() - t0:.0f}s", flush=True)

    pkeys = pool.df["inchikey14"].to_numpy()
    fp_cache = {}
    cols = []
    for (it, cand), s_row, i_row in tqdm(list(zip(kept, S, I)), desc="anchor features"):
        hidden = {pkeys[x] for x in it.ex.lib_struct_ids} if it.ex.lib_struct_ids.size else set()
        truth_key = pkeys[it.struct_id]
        order = np.argsort(-s_row)
        anchors, seen = [], set()
        for j in order:
            if not np.isfinite(s_row[j]):
                break
            r = i_row[j]
            k = keys[r]
            if k in hidden:
                continue
            if it.cls == 1 and is_e180[r] and k == truth_key:  # class 1 hides the truth's timsTOF spectra
                continue
            if k in seen:
                continue
            seen.add(k)
            anchors.append((k, float(s_row[j]), smis[r]))
            if len(anchors) >= NANCH:
                break
        cols.append(features(cand, pool, anchors, fp_cache))
    D = np.concatenate(cols)
    nt = int(gt.sum())
    Xt2, EX2 = np.concatenate([Xt, D[:nt]], 1), np.concatenate([EX, D[nt:]], 1)
    np.savez(md / "dreams_features.npz", train=D[:nt], eval=D[nt:])

    def report(name, scores):
        r = rr(scores, Ey, Eg)
        cells = {}
        for (panel, cls), n in sorted(counts.items()):
            sel = np.array([(p == panel and c == cls) for p, c in Emeta])
            cells[(panel, cls)] = r[sel].sum() / n
        mix = sum(DEFAULT_MIX[c] * cells.get(("np", c), 0.0) for c in (1, 2, 3))
        print(f"{name:<34} " + "  ".join(f"{p}-c{c} {v:.3f}" for (p, c), v in cells.items()) +
              f"   np mixed {mix:.4f}", flush=True)

    names = FB + fr.FEATURES + DFEAT
    report("heuristic", EX[:, FB.index("h_score")])
    report("dr_s3T alone", np.nan_to_num(EX2[:, NF + 1], nan=-1))
    st = np.concatenate([[0], np.cumsum(gt)[:-1]])
    rng = np.random.default_rng(0)
    gidx = rng.permutation(len(gt))
    val, trn = np.sort(gidx[: len(gt) // 5]), np.sort(gidx[len(gt) // 5:])

    def take(X, gs):
        r_ = np.concatenate([np.arange(st[g], st[g] + gt[g]) for g in gs])
        return X[r_], yt[r_], gt[gs]

    params = dict(objective="lambdarank", metric="map", eval_at=[25], learning_rate=0.05, num_leaves=31,
                  min_data_in_leaf=40, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  lambdarank_truncation_level=25, verbose=-1, seed=0)
    for X_all, EXs, tag in ((Xt, EX, "v4 (base + fragments)"), (Xt2, EX2, "v6 (+ DreaMS anchors)")):
        a_t, a_y, a_g = take(X_all, trn)
        v_t, v_y, v_g = take(X_all, val)
        nm = names[: X_all.shape[1]]
        m = lgb.train(params, lgb.Dataset(a_t, a_y, group=a_g, feature_name=nm), num_boost_round=1000,
                      valid_sets=[lgb.Dataset(v_t, v_y, group=v_g, feature_name=nm)],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        hv = rr(m.predict(v_t, num_iteration=m.best_iteration), v_y, v_g).mean()
        report(f"{tag} (val {hv:.3f})", m.predict(EXs, num_iteration=m.best_iteration))
        if X_all is Xt2:
            imp = dict(zip(nm, m.feature_importance("gain").round(0).tolist()))
            print("   DreaMS gains:", {k: imp[k] for k in DFEAT})


if __name__ == "__main__":
    main()
