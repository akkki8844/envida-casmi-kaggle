"""v4: add fragment-explanation features to the cached v3 features, train and compare offline.

    python scripts/train_v4.py --extra data/coconut/coconut_pool.parquet [--jobs 11] [--top-k 30]

Reuses data/models/train_features.npz (train_ranker.py) and data/models/eval_features.npz (blend_eval.py): the
candidate list of each query is re-derived from the mass window (cheap) and must match the cached group sizes.
Fragment features are computed for the heuristic top-k candidates of each query, as BaselineRanker.features does.
Writes data/models/{train,eval}_features_v4.npz, ranker_v4.txt (class-1 x1) and ranker_v4b.txt (class-1 x3).
"""
import argparse
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from casmi import fragments as fr  # noqa: E402
from casmi.baseline import BaselineRanker, Params  # noqa: E402
from casmi.library import build_library, build_pool, load_raw_peaks, load_train_meta  # noqa: E402
from casmi.validation import DEFAULT_MIX, item_query, make_items, make_train_items  # noqa: E402

FB = BaselineRanker.BASE_FEATURES
F = BaselineRanker.FEATURES
H, HS, IT = FB.index("h_score"), FB.index("has_spec"), FB.index("in_train")


def aligned(ranker, lib, items, groups, y):
    """Items with the truth among candidates, in collect_features order, checked against cached groups."""
    kept = []
    for it in items:
        q = item_query(lib, it)
        cand = ranker.candidates(q.neutral_mass, it.ex)
        if (cand == it.struct_id).any():
            kept.append((it, cand))
    assert len(kept) == len(groups), (len(kept), len(groups))
    s = 0
    for (it, cand), g in zip(kept, groups):
        assert cand.size == g and y[s:s + g].sum() == 1 and cand[np.flatnonzero(y[s:s + g])[0]] == it.struct_id
        s += g
    return kept


def frag_columns(ranker, lib, kept, X, groups, raw, top_k):
    FX = np.full((X.shape[0], len(fr.FEATURES)), np.nan, dtype=np.float32)
    s = 0
    for (it, cand), g in tqdm(list(zip(kept, groups)), desc="fragment features"):
        q = item_query(lib, it, raw)
        top = np.argsort(-X[s:s + g, H], kind="stable")[:top_k]
        if q.peaks:
            FX[s + top] = ranker.frag.match(q.peaks, q.mode, cand[top])
        s += g
    return np.concatenate([X[:, : len(FB)], FX], axis=1)


def rr(scores, y, groups):
    out, s = [], 0
    for g in groups:
        order = np.argsort(-scores[s:s + g], kind="stable")
        pos = np.flatnonzero(y[s:s + g][order])
        out.append(1.0 / (pos[0] + 1) if pos.size and pos[0] < 25 else 0.0)
        s += g
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--extra", default=None)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--jobs", type=int, default=11)
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--models", default="data/models")
    a = ap.parse_args()
    md = Path(a.models)
    train = Path(a.data) / "train.parquet"
    meta = load_train_meta(train)
    pool = build_pool(meta, Path(a.cache), extra=a.extra)
    lib = build_library(train, meta, pool, Path(a.cache))
    del meta
    ranker = BaselineRanker(lib, pool, Params()).enable_fragments(top_k=a.top_k)

    eval_all = make_items(lib, pool, 60, seed=1)
    np_structs = set(lib.meta.loc[lib.meta["ingest_lib"] == "enveda-np-examples", "struct_id"].tolist())
    exclude = np_structs | {it.struct_id for it in eval_all}
    train_items = make_train_items(lib, pool, a.n, exclude)
    eval_items = [it for it in eval_all if it.cls != 3]

    out_t, out_e = md / "train_features_v4.npz", md / "eval_features_v4.npz"
    if out_t.exists() and out_e.exists():
        dt, de = np.load(out_t), np.load(out_e, allow_pickle=True)
        Xt_all, yt_all, gt_all = dt["X"], dt["y"], dt["groups"]
        EX, Ey, Eg, Emeta, counts = de["X"], de["y"], de["groups"], de["meta"], de["counts"]
    else:
        d = np.load(md / "train_features.npz")
        Xt_all, yt_all, gt_all = d["X"].copy(), d["y"], d["groups"]
        Xt_all[(yt_all == 1) & (Xt_all[:, HS] == 0) & (Xt_all[:, IT] == 1), IT] = 0  # in_train leak fix
        de = np.load(md / "eval_features.npz", allow_pickle=True)
        EX, Ey, Eg, Emeta, counts = de["X"], de["y"], de["groups"], de["meta"], de["counts"]
        kt = aligned(ranker, lib, train_items, gt_all, yt_all)
        ke = aligned(ranker, lib, eval_items, Eg, Ey)
        print(f"aligned {len(kt)} train and {len(ke)} eval groups", flush=True)
        rows = {int(r) for it, _ in kt + ke for r in lib.meta["row"].to_numpy()[it.rows]}
        raw = load_raw_peaks(train, rows)
        need = []
        for kept, X, groups in ((kt, Xt_all, gt_all), (ke, EX, Eg)):
            s = 0
            for (_, cand), g in zip(kept, groups):
                need.extend(cand[np.argsort(-X[s:s + g, H], kind="stable")[: a.top_k]].tolist())
                s += g
        t0 = time.time()
        ranker.frag.precompute(need, n_jobs=a.jobs, progress=lambda it, total: tqdm(it, total=total, desc="fragmenting"))
        print(f"fragmented {len(ranker.frag.cache):,} structures in {time.time() - t0:.0f}s", flush=True)
        Xt_all = frag_columns(ranker, lib, kt, Xt_all, gt_all, raw, a.top_k)
        EX = frag_columns(ranker, lib, ke, EX, Eg, raw, a.top_k)
        np.savez(out_t, X=Xt_all, y=yt_all, groups=gt_all)
        np.savez(out_e, X=EX, y=Ey, groups=Eg, meta=Emeta, counts=counts)
    counts = {tuple(k): v for k, v in counts}

    def report(name, scores):
        r = rr(scores, Ey, Eg)
        cells = {}
        for (panel, cls), n in sorted(counts.items()):
            sel = np.array([(p == panel and c == cls) for p, c in Emeta])
            cells[(panel, cls)] = r[sel].sum() / n
        mix = sum(DEFAULT_MIX[c] * cells.get(("np", c), 0.0) for c in (1, 2, 3))
        cols = "  ".join(f"{p}-c{c} {v:.3f}" for (p, c), v in cells.items() if p == "np")
        print(f"{name:<34} {cols}   np mixed {mix:.4f}", flush=True)

    fx_cov = ~np.isnan(EX[:, len(FB)])
    print(f"eval rows with fragment features: {fx_cov.mean():.1%}; truth rows: {fx_cov[Ey == 1].mean():.1%}")
    report("heuristic", EX[:, H])
    report("fx_score alone (top-k)", np.nan_to_num(EX[:, F.index("fx_score")], nan=-1.0))

    st = np.concatenate([[0], np.cumsum(gt_all)[:-1]])
    is_c2 = np.array([bool(((yt_all[s:s + g] == 1) & (Xt_all[s:s + g, HS] == 0)).any()) for s, g in zip(st, gt_all)])
    rng = np.random.default_rng(0)
    gidx = rng.permutation(len(gt_all))
    val, trn = np.sort(gidx[: len(gt_all) // 5]), np.sort(gidx[len(gt_all) // 5:])

    def take(gs, ncol):
        rows = np.concatenate([np.arange(st[g], st[g] + gt_all[g]) for g in gs])
        return Xt_all[rows, :ncol], yt_all[rows], gt_all[gs]

    params = dict(objective="lambdarank", metric="map", eval_at=[25], learning_rate=0.05, num_leaves=31,
                  min_data_in_leaf=40, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  lambdarank_truncation_level=25, verbose=-1, seed=0)
    for ncol, tag in ((len(FB), "base (v3)"), (len(F), "base + fragments (v4)")):
        for dup in (1, 3):
            tr = np.concatenate([trn] + [trn[~is_c2[trn]]] * (dup - 1))
            tr = tr[np.argsort(st[tr], kind="stable")]
            Xt, yt, gt = take(tr, ncol)
            Xv, yv, gv = take(val, ncol)
            names = F[:ncol]
            model = lgb.train(params, lgb.Dataset(Xt, yt, group=gt, feature_name=names), num_boost_round=1000,
                              valid_sets=[lgb.Dataset(Xv, yv, group=gv, feature_name=names)],
                              callbacks=[lgb.early_stopping(50, verbose=False)])
            hv = rr(model.predict(Xv, num_iteration=model.best_iteration), yv, gv).mean()
            report(f"{tag} c1x{dup} (val {hv:.3f})", model.predict(EX[:, :ncol], num_iteration=model.best_iteration))
            if ncol == len(F):
                out = md / ("ranker_v4.txt" if dup == 1 else "ranker_v4b.txt")
                model.save_model(str(out), num_iteration=model.best_iteration)
                imp = dict(zip(names, model.feature_importance("gain").round(0).tolist()))
                print("   fragment gains:", {k: imp[k] for k in fr.FEATURES}, "->", out, flush=True)


if __name__ == "__main__":
    main()
