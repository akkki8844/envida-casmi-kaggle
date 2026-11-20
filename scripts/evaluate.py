"""Run the class-simulating validation harness on the baseline ranker.

    python scripts/evaluate.py [--n-per-class 60] [--seed 1] [--extra data/coconut/coconut_pool.parquet]
                               [--power 3] [--spec-penalty 1 0.3 0.1] [--out data/eval.csv]

Every combination of --power and --spec-penalty is evaluated on the same query molecules.
"""
import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from casmi.baseline import BaselineRanker, LearnedRanker, Params  # noqa: E402
from casmi.library import build_library, build_pool, load_train_meta  # noqa: E402
from casmi.validation import evaluate, make_items, summarise  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--cache", default="data/cache")
ap.add_argument("--extra", default=None, help="external candidate table (e.g. COCONUT) to add to the pool")
ap.add_argument("--n-per-class", type=int, default=60)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--power", type=float, nargs="+", default=[3.0])
ap.add_argument("--spec-penalty", type=float, nargs="+", default=[1.0])
ap.add_argument("--ppm", type=float, default=10.0)
ap.add_argument("--out", default="data/eval.csv")
ap.add_argument("--model", default=None, help="LightGBM ranker; if given it is evaluated after the heuristic configs")
a = ap.parse_args()

train = Path(a.data) / "train.parquet"
cache = Path(a.cache)
meta = load_train_meta(train)
pool = build_pool(meta, cache, extra=a.extra)
lib = build_library(train, meta, pool, cache)
del meta
print(f"pool {len(pool.df):,} (external {int(pool.df['in_db'].sum()):,}), library {len(lib.meta):,}", flush=True)
items = make_items(lib, pool, a.n_per_class, seed=a.seed)
print(f"{len(items)} query molecules", flush=True)
frames = []
for p, pen in itertools.product(a.power, a.spec_penalty):
    ranker = BaselineRanker(lib, pool, Params(power=p, ppm=a.ppm, spec_penalty=pen))
    df = evaluate(ranker, lib, pool, items, desc=f"power={p} pen={pen}")
    df["power"], df["spec_penalty"] = p, pen
    frames.append(df)
    print(f"\n=== power {p}, spec_penalty {pen}, ppm {a.ppm}, extra {a.extra}\n{summarise(df)}", flush=True)
if a.model:
    ranker = LearnedRanker(lib, pool, a.model, Params(ppm=a.ppm))
    df = evaluate(ranker, lib, pool, items, desc="model")
    df["power"], df["spec_penalty"] = -1, -1
    frames.append(df)
    print(f"\n=== learned ranker {a.model}\n{summarise(df)}", flush=True)
pd.concat(frames).to_csv(a.out, index=False)
