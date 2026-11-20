"""Train the two structure-fold FPNet models on the library spectra and export them for numpy inference.

    python scripts/train_fpnet.py --extra data/coconut/coconut_pool.parquet [--epochs 4] [--hidden 1024]

Fold f's model trains on spectra of structures with fold_of(key) == f (minus 3% of them held out for early
checks) and is evaluated on the other fold. The harness panels (enveda-np-examples) are left out of both.
Writes data/models/fpnet_f0.npz and fpnet_f1.npz.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from casmi.fpnet import N_BITS, V, fold_of, tokens  # noqa: E402
from casmi.library import build_library, build_pool, load_train_meta  # noqa: E402


class Net(nn.Module):
    def __init__(self, hidden: int, drop: float):
        super().__init__()
        # sparse gradients: a batch touches a few thousand of the 20k token rows; dense Adam over the whole table
        # made each step ~1.6 s on CPU
        self.emb = nn.EmbeddingBag(V, hidden, mode="sum", sparse=True)
        self.b1 = nn.Parameter(torch.zeros(hidden))
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, N_BITS)
        self.drop = nn.Dropout(drop)

    def forward(self, idx, off, w):
        h = torch.relu(self.emb(idx, off, per_sample_weights=w) + self.b1)
        h = torch.relu(self.l2(self.drop(h)))
        return self.l3(self.drop(h))


def batch(X, rows):
    sub = X[rows]
    return (torch.from_numpy(sub.indices.astype(np.int64)), torch.from_numpy(sub.indptr[:-1].astype(np.int64)),
            torch.from_numpy(sub.data.astype(np.float32)))


def tanimoto_at(prob, bits, thr=0.5):
    pb = prob >= thr
    inter = (pb & bits).sum(1)
    union = (pb | bits).sum(1)
    return float(np.mean(inter / np.maximum(union, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--extra", default=None)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--drop", type=float, default=0.2)
    ap.add_argument("--out", default="data/models")
    a = ap.parse_args()
    torch.manual_seed(0)
    train = Path(a.data) / "train.parquet"
    meta = load_train_meta(train)
    pool = build_pool(meta, Path(a.cache), extra=a.extra)
    lib = build_library(train, meta, pool, Path(a.cache))
    del meta
    m = lib.meta
    X = tokens(lib.pk, lib.nl, m["mode"].to_numpy())
    lib.pk = lib.nl = None
    sid = m["struct_id"].to_numpy()
    keys = pool.df["inchikey14"].to_numpy()
    sfold = np.array([fold_of(k) for k in keys[np.unique(sid)]])
    fold_map = dict(zip(np.unique(sid).tolist(), sfold.tolist()))
    fold = np.array([fold_map[s] for s in sid])
    keep = (m["ingest_lib"] != "enveda-np-examples").to_numpy()
    print(f"spectra {X.shape[0]:,}, tokens/spectrum {X.nnz / X.shape[0]:.0f}, fold sizes "
          f"{np.bincount(fold[keep])}", flush=True)
    rng = np.random.default_rng(0)
    for f in (0, 1):
        rows = np.flatnonzero(keep & (fold == f))
        ustruct = np.unique(sid[rows])
        held = set(rng.choice(ustruct, size=max(1, len(ustruct) // 33), replace=False).tolist())
        is_held = np.array([s in held for s in sid[rows]])
        tr, va = rows[~is_held], rows[is_held]
        oof = rng.choice(np.flatnonzero(keep & (fold != f)), size=20000, replace=False)
        net = Net(a.hidden, a.drop)
        dense = [p for n, p in net.named_parameters() if not n.startswith("emb.")]
        opts = [torch.optim.SparseAdam(list(net.emb.parameters()), lr=a.lr), torch.optim.Adam(dense, lr=a.lr)]
        steps = a.epochs * (len(tr) // a.batch)
        warm = max(1, steps // 20)
        sched = [torch.optim.lr_scheduler.LambdaLR(o, lambda s: min(1.0, (s + 1) / warm) * max(0.02, 1 - s / steps))
                 for o in opts]
        lossf = nn.BCEWithLogitsLoss()
        t0, step = time.time(), 0
        for ep in range(a.epochs):
            net.train()
            perm = rng.permutation(tr)
            run = 0.0
            for b in range(len(tr) // a.batch):
                r = np.sort(perm[b * a.batch:(b + 1) * a.batch])
                idx, off, w = batch(X, r)
                y = torch.from_numpy(pool.fps[sid[r]].astype(np.float32))
                loss = lossf(net(idx, off, w), y)
                for o in opts:
                    o.zero_grad()
                loss.backward()
                for o in opts:
                    o.step()
                for s in sched:
                    s.step()
                step += 1
                run += loss.item()
                if step % 250 == 0:
                    print(f"fold {f} ep {ep} step {step}/{steps} loss {run / 250:.4f} {time.time() - t0:.0f}s", flush=True)
                    run = 0.0
            net.eval()
            with torch.no_grad():
                for name, rr in (("held-in-fold", va), ("other fold", oof)):
                    rr = np.sort(rr[:20000])
                    probs = []
                    for c in range(0, len(rr), 2048):
                        idx, off, w = batch(X, rr[c:c + 2048])
                        probs.append(torch.sigmoid(net(idx, off, w)).numpy())
                    pr = np.concatenate(probs)
                    print(f"fold {f} ep {ep}: {name} tanimoto@0.5 {tanimoto_at(pr, pool.fps[sid[rr]].astype(bool)):.3f}",
                          flush=True)
        out = Path(a.out) / f"fpnet_f{f}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, E=net.emb.weight.detach().numpy().astype(np.float16), b1=net.b1.detach().numpy(),
                 W2=net.l2.weight.detach().numpy().astype(np.float16), b2=net.l2.bias.detach().numpy(),
                 W3=net.l3.weight.detach().numpy().astype(np.float16), b3=net.l3.bias.detach().numpy())
        print(f"saved {out} ({out.stat().st_size / 1e6:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
