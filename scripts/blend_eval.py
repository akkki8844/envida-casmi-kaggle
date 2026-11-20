"""Offline comparison of rankers on cached harness features (no re-scoring per config).

    python scripts/blend_eval.py --extra data/coconut/coconut_pool.parquet [--dup 1 2 3] [--alpha 0 0.25 0.5 0.75 1]

1. Collects features for the harness queries once (data/models/eval_features.npz).
2. Trains LambdaRank models on the cached training features with class-1 groups duplicated `dup` times.
3. Scores heuristic, model and rank blends  alpha*model_rank + (1-alpha)*heuristic_rank  per panel/class.
Class-3 queries have no positive, so they score 0 whatever the ranker; the mixed MRR uses the harness weights.
"""
import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from casmi.baseline import BaselineRanker, Params  # noqa: E402
from casmi.validation import DEFAULT_MIX  # noqa: E402

F = BaselineRanker.FEATURES
H, HS, IT = F.index("h_score"), F.index("has_spec"), F.index("in_train")

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--cache", default="data/cache")
ap.add_argument("--extra", default=None)
ap.add_argument("--n-per-class", type=int, default=60)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--train-features", default="data/models/train_features.npz")
ap.add_argument("--eval-features", default="data/models/eval_features.npz")
ap.add_argument("--dup", type=int, nargs="+", default=[1, 2, 3])
ap.add_argument("--alpha", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
ap.add_argument("--save-dup", type=int, default=0, help="save the model trained with this dup to --out")
ap.add_argument("--out", default="data/models/ranker_v3.txt")
a = ap.parse_args()


def starts_of(groups):
    return np.concatenate([[0], np.cumsum(groups)[:-1]])


# ---- eval features -------------------------------------------------------------------------------
ep = Path(a.eval_features)
if ep.exists():
    d = np.load(ep, allow_pickle=True)
    EX, Ey, Eg, Emeta = d["X"], d["y"], d["groups"], d["meta"]
else:
    from casmi.library import build_library, build_pool, load_train_meta
    from casmi.validation import collect_features, make_items
    train = Path(a.data) / "train.parquet"
    meta = load_train_meta(train)
    pool = build_pool(meta, Path(a.cache), extra=a.extra)
    lib = build_library(train, meta, pool, Path(a.cache))
    del meta
    items = [it for it in make_items(lib, pool, a.n_per_class, seed=a.seed) if it.cls != 3]
    EX, Ey, Eg, m = collect_features(BaselineRanker(lib, pool, Params()), lib, items, desc="eval features")
    Emeta = np.array([(p, c) for p, c, _ in m], dtype=object)
    counts = {}
    for it in items:
        counts[(it.panel, it.cls)] = counts.get((it.panel, it.cls), 0) + 1
    np.savez(ep, X=EX, y=Ey, groups=Eg, meta=Emeta, counts=np.array(list(counts.items()), dtype=object))
    d = np.load(ep, allow_pickle=True)
counts = {tuple(k): v for k, v in d["counts"]}


def rr(scores, y, groups):
    out, s = [], 0
    for g in groups:
        order = np.argsort(-scores[s:s + g], kind="stable")
        pos = np.flatnonzero(y[s:s + g][order])
        out.append(1.0 / (pos[0] + 1) if pos.size and pos[0] < 25 else 0.0)
        s += g
    return np.array(out)


def ranks01(scores, groups):
    """Per-group rank in [0, 1], 1 = best (so blends are scale-free)."""
    out, s = np.empty_like(scores, dtype=np.float64), 0
    for g in groups:
        r = np.empty(g)
        r[np.argsort(scores[s:s + g], kind="stable")] = np.arange(g)
        out[s:s + g] = r / max(g - 1, 1)
        s += g
    return out


def report(name, scores):
    r = rr(scores, Ey, Eg)
    cells = {}
    for (panel, cls), n in sorted(counts.items()):
        sel = np.array([(p == panel and c == cls) for p, c in Emeta])
        cells[(panel, cls)] = r[sel].sum() / n  # queries without the truth in the window count as 0
    mix = sum(DEFAULT_MIX[c] * cells.get(("np", c), 0.0) for c in (1, 2, 3))
    cols = "  ".join(f"{p}-c{c} {v:.3f}" for (p, c), v in cells.items())
    print(f"{name:<28} {cols}   np mixed {mix:.4f}", flush=True)
    return mix


# ---- training data (in_train fix as in train_ranker.py) --------------------------------------------
d = np.load(a.train_features)
X, y, groups = d["X"].copy(), d["y"], d["groups"]
X[(y == 1) & (X[:, HS] == 0) & (X[:, IT] == 1), IT] = 0
st = starts_of(groups)
is_c2 = np.array([bool(((y[s:s + g] == 1) & (X[s:s + g, HS] == 0)).any()) for s, g in zip(st, groups)])
print(f"training groups: class 1 {int((~is_c2).sum())}, class 2 {int(is_c2.sum())}")

rng = np.random.default_rng(0)
gidx = rng.permutation(len(groups))
val, trn = np.sort(gidx[: len(groups) // 5]), np.sort(gidx[len(groups) // 5:])


def take(gs):
    rows = np.concatenate([np.arange(st[g], st[g] + groups[g]) for g in gs])
    return X[rows], y[rows], groups[gs]


params = dict(objective="lambdarank", metric="map", eval_at=[25], learning_rate=0.05, num_leaves=31,
              min_data_in_leaf=40, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambdarank_truncation_level=25, verbose=-1, seed=0)

h = EX[:, H]
best = (-1.0, None)
report("heuristic", h)
for dup in a.dup:
    tr = np.concatenate([trn] + [trn[~is_c2[trn]]] * (dup - 1))
    tr = tr[np.argsort(st[tr], kind="stable")]
    Xt, yt, gt = take(tr)
    Xv, yv, gv = take(val)
    model = lgb.train(params, lgb.Dataset(Xt, yt, group=gt, feature_name=F),
                      num_boost_round=1000, valid_sets=[lgb.Dataset(Xv, yv, group=gv)],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
    p = model.predict(EX, num_iteration=model.best_iteration)
    hr, pr = ranks01(h, Eg), ranks01(p, Eg)
    for al in a.alpha:
        mix = report(f"dup {dup} alpha {al:.2f} (it {model.best_iteration})", al * pr + (1 - al) * hr + 1e-9 * p)
        if mix > best[0]:
            best = (mix, (dup, al))
    if dup == a.save_dup:
        model.save_model(a.out, num_iteration=model.best_iteration)
        print(f"saved {a.out}")
print(f"best: dup {best[1][0]}, alpha {best[1][1]} -> np mixed {best[0]:.4f}")
