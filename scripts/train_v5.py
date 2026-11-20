"""v5: add FPNet fingerprint-agreement features to the v4 features, train and compare offline.

    python scripts/train_v5.py --extra data/coconut/coconut_pool.parquet

Needs data/models/{train,eval}_features_v4.npz (train_v4.py) and fpnet_f0/f1.npz (train_fpnet.py).
Each query is scored with the FPNet model that did NOT train on its structure (out-of-fold), so the ranker learns
how much to trust predictions for unseen structures. Writes {train,eval}_features_v5.npz and ranker_v5.txt.
"""
import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from casmi import fpnet as fpm  # noqa: E402
from casmi import fragments as fr  # noqa: E402
from casmi.baseline import BaselineRanker, Params  # noqa: E402
from casmi.library import build_library, build_pool, load_train_meta  # noqa: E402
from casmi.validation import DEFAULT_MIX, item_query, make_items, make_train_items  # noqa: E402
from train_v4 import aligned, rr  # noqa: E402

F = BaselineRanker.FEATURES
NB, NF = len(BaselineRanker.BASE_FEATURES), len(BaselineRanker.BASE_FEATURES) + len(fr.FEATURES)
HS = F.index("has_spec")


def fp_columns(net, lib, pool, kept, groups):
    keys = pool.df["inchikey14"].to_numpy()
    out = []
    for (it, cand), g in tqdm(list(zip(kept, groups)), desc="fpnet features"):
        q = item_query(lib, it)
        other = 1 - fpm.fold_of(keys[it.struct_id])
        p = net.query_probs(q.pk, q.nl, q.mode, models=[other])
        out.append(fpm.candidate_features(p, pool.fps[cand]))
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--extra", default=None)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--models", default="data/models")
    a = ap.parse_args()
    md = Path(a.models)
    train = Path(a.data) / "train.parquet"
    meta = load_train_meta(train)
    pool = build_pool(meta, Path(a.cache), extra=a.extra)
    lib = build_library(train, meta, pool, Path(a.cache))
    del meta
    ranker = BaselineRanker(lib, pool, Params())
    net = fpm.FPNet([md / "fpnet_f0.npz", md / "fpnet_f1.npz"])

    dt, de = np.load(md / "train_features_v4.npz"), np.load(md / "eval_features_v4.npz", allow_pickle=True)
    Xt, yt, gt = dt["X"], dt["y"], dt["groups"]
    EX, Ey, Eg, Emeta = de["X"], de["y"], de["groups"], de["meta"]
    counts = {tuple(k): v for k, v in de["counts"]}
    eval_all = make_items(lib, pool, 60, seed=1)
    np_structs = set(lib.meta.loc[lib.meta["ingest_lib"] == "enveda-np-examples", "struct_id"].tolist())
    exclude = np_structs | {it.struct_id for it in eval_all}
    kt = aligned(ranker, lib, make_train_items(lib, pool, a.n, exclude), gt, yt)
    ke = aligned(ranker, lib, [it for it in eval_all if it.cls != 3], Eg, Ey)
    Xt = np.concatenate([Xt[:, :NF], fp_columns(net, lib, pool, kt, gt)], axis=1)
    EX = np.concatenate([EX[:, :NF], fp_columns(net, lib, pool, ke, Eg)], axis=1)
    np.savez(md / "train_features_v5.npz", X=Xt, y=yt, groups=gt)
    np.savez(md / "eval_features_v5.npz", X=EX, y=Ey, groups=Eg, meta=Emeta, counts=de["counts"])

    def report(name, scores):
        r = rr(scores, Ey, Eg)
        cells = {}
        for (panel, cls), n in sorted(counts.items()):
            sel = np.array([(p == panel and c == cls) for p, c in Emeta])
            cells[(panel, cls)] = r[sel].sum() / n
        mix = sum(DEFAULT_MIX[c] * cells.get(("np", c), 0.0) for c in (1, 2, 3))
        cols = "  ".join(f"{p}-c{c} {v:.3f}" for (p, c), v in cells.items() if p == "np")
        print(f"{name:<34} {cols}   np mixed {mix:.4f}", flush=True)

    report("heuristic", EX[:, F.index("h_score")])
    report("fp_ll alone", EX[:, F.index("fp_ll")])

    st = np.concatenate([[0], np.cumsum(gt)[:-1]])
    rng = np.random.default_rng(0)
    gidx = rng.permutation(len(gt))
    val, trn = np.sort(gidx[: len(gt) // 5]), np.sort(gidx[len(gt) // 5:])

    def take(gs, ncol):
        rows = np.concatenate([np.arange(st[g], st[g] + gt[g]) for g in gs])
        return Xt[rows, :ncol], yt[rows], gt[gs]

    params = dict(objective="lambdarank", metric="map", eval_at=[25], learning_rate=0.05, num_leaves=31,
                  min_data_in_leaf=40, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  lambdarank_truncation_level=25, verbose=-1, seed=0)
    for ncol, tag in ((NF, "v4 (base + fragments)"), (len(F), "v5 (+ fpnet)")):
        a_t, a_y, a_g = take(trn, ncol)
        v_t, v_y, v_g = take(val, ncol)
        names = F[:ncol]
        model = lgb.train(params, lgb.Dataset(a_t, a_y, group=a_g, feature_name=names), num_boost_round=1000,
                          valid_sets=[lgb.Dataset(v_t, v_y, group=v_g, feature_name=names)],
                          callbacks=[lgb.early_stopping(50, verbose=False)])
        hv = rr(model.predict(v_t, num_iteration=model.best_iteration), v_y, v_g).mean()
        report(f"{tag} (val {hv:.3f})", model.predict(EX[:, :ncol], num_iteration=model.best_iteration))
        if ncol == len(F):
            model.save_model(str(md / "ranker_v5.txt"), num_iteration=model.best_iteration)
            imp = dict(zip(names, model.feature_importance("gain").round(0).tolist()))
            print("   fpnet gains:", {k: imp[k] for k in fpm.FEATURES}, flush=True)
            print("   top gains:", dict(sorted(imp.items(), key=lambda kv: -kv[1])[:8]), flush=True)


if __name__ == "__main__":
    main()
