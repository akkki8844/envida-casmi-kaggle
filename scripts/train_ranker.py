"""Train the v3 LightGBM LambdaRank model on structure-disjoint simulated queries.

    python scripts/train_ranker.py --extra data/coconut/coconut_pool.parquet [--n 1500] [--dup-c1 3] [--out data/models/ranker_v3b.txt]

The evaluation panels (enveda-np-examples structures, and the e180 eval sample) are excluded from training.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from casmi.baseline import BaselineRanker, Params  # noqa: E402
from casmi.library import build_library, build_pool, load_train_meta  # noqa: E402
from casmi.validation import collect_features, make_items, make_train_items  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--cache", default="data/cache")
ap.add_argument("--extra", default=None)
ap.add_argument("--n", type=int, default=1500)
ap.add_argument("--eval-n-per-class", type=int, default=60)
ap.add_argument("--eval-seed", type=int, default=1)
ap.add_argument("--features-out", default="data/models/train_features.npz")
ap.add_argument("--dup-c1", type=int, default=3, help="repeat class-1 training groups (v3b: 3; v3: 1)")
ap.add_argument("--out", default="data/models/ranker_v3b.txt")
a = ap.parse_args()

train = Path(a.data) / "train.parquet"
cache = Path(a.cache)
meta = load_train_meta(train)
pool = build_pool(meta, cache, extra=a.extra)
lib = build_library(train, meta, pool, cache)
del meta

# keep every structure the harness evaluates on out of training
eval_items = make_items(lib, pool, a.eval_n_per_class, seed=a.eval_seed)
np_structs = set(lib.meta.loc[lib.meta["ingest_lib"] == "enveda-np-examples", "struct_id"].tolist())
exclude = np_structs | {it.struct_id for it in eval_items}
print(f"excluded from training: {len(exclude):,} structures", flush=True)

ranker = BaselineRanker(lib, pool, Params())
fpath = Path(a.features_out)
if fpath.exists():
    d = np.load(fpath)
    X, y, groups = d["X"].copy(), d["y"], d["groups"]
    # features cached before the in_train fix: class-2 positives (all spectra hidden) are the only
    # in_train=1 & has_spec=0 positives; give them in_train=0 as features() now does
    it_c, hs_c = BaselineRanker.FEATURES.index("in_train"), BaselineRanker.FEATURES.index("has_spec")
    fix = (y == 1) & (X[:, hs_c] == 0) & (X[:, it_c] == 1)
    X[fix, it_c] = 0
    print(f"in_train fix applied to {int(fix.sum())} cached positive rows", flush=True)
else:
    items = make_train_items(lib, pool, a.n, exclude)
    print(f"{len(items)} training queries (class 1: {sum(i.cls == 1 for i in items)}, "
          f"class 2: {sum(i.cls == 2 for i in items)})", flush=True)
    t0 = time.time()
    X, y, groups, _ = collect_features(ranker, lib, items)
    print(f"features: {X.shape} in {time.time() - t0:.0f}s, {len(groups)} groups with a positive", flush=True)
    fpath.parent.mkdir(parents=True, exist_ok=True)
    np.savez(fpath, X=X, y=y, groups=groups)

# group-wise 80/20 split for early stopping
rng = np.random.default_rng(0)
gidx = rng.permutation(len(groups))
n_val = len(groups) // 5
starts = np.concatenate([[0], np.cumsum(groups)[:-1]])
def take(gs):
    rows = np.concatenate([np.arange(starts[g], starts[g] + groups[g]) for g in gs])
    return X[rows], y[rows], groups[gs]
Xv, yv, gv = take(np.sort(gidx[:n_val]))
# class-2 groups are those whose positive has no visible spectra; repeat class-1 groups --dup-c1 times
hs_c = BaselineRanker.FEATURES.index("has_spec")
is_c2 = np.array([bool(((y[s:s + g] == 1) & (X[s:s + g, hs_c] == 0)).any()) for s, g in zip(starts, groups)])
trn = np.sort(gidx[n_val:])
trn = np.concatenate([trn] + [trn[~is_c2[trn]]] * (a.dup_c1 - 1))
Xt, yt, gt = take(trn[np.argsort(starts[trn], kind="stable")])
print(f"train groups {len(gt)} (class 1 x{a.dup_c1}), valid groups {len(gv)}", flush=True)

params = dict(objective="lambdarank", metric="map", eval_at=[1, 5, 25], learning_rate=0.05, num_leaves=31,
              min_data_in_leaf=40, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambdarank_truncation_level=25, verbose=-1, seed=0)
dt = lgb.Dataset(Xt, yt, group=gt, feature_name=BaselineRanker.FEATURES)
dv = lgb.Dataset(Xv, yv, group=gv, reference=dt)
model = lgb.train(params, dt, num_boost_round=1000, valid_sets=[dv],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)])
print(f"best iteration {model.best_iteration}, valid {dict(model.best_score['valid_0'])}")

# MRR@25 on the held-out training groups: model vs heuristic
def mrr(scores_all, Xs, ys, gs):
    out, s = [], 0
    for g in gs:
        sc, yy = scores_all[s:s + g], ys[s:s + g]
        order = np.argsort(-sc, kind="stable")
        pos = np.flatnonzero(yy[order])
        out.append(1.0 / (pos[0] + 1) if pos.size and pos[0] < 25 else 0.0)
        s += g
    return float(np.mean(out))
h_col = BaselineRanker.FEATURES.index("h_score")
print(f"held-out MRR@25: heuristic {mrr(Xv[:, h_col], Xv, yv, gv):.4f}  model {mrr(model.predict(Xv), Xv, yv, gv):.4f}")

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
model.save_model(a.out, num_iteration=model.best_iteration)
imp = dict(zip(BaselineRanker.FEATURES, model.feature_importance("gain").round(1).tolist()))
print("gain importance:", json.dumps(dict(sorted(imp.items(), key=lambda kv: -kv[1])[:12])))
print(f"saved {a.out}")
