"""Kaggle job: DreaMS embeddings of our harness query spectra (runs on Kaggle, not locally).

Inputs: govindmulchandani/envida-dreams-queries (queries.parquet: q, mz, it, prec; n_queries.json) and the DreaMS model
in dmitriigluzdov/casmi26-attributed-v4g-reproduction-assets (hengck23-dreams-enveda-casmi26, MIT label).
Output: /kaggle/working/dreams_query_emb.npy, float32 [n_queries, 1024]: per spectrum L2-normalised, mean-pooled per
query, L2-normalised again (zeros for a query without usable spectra).
"""
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

hit = glob.glob("/kaggle/input/**/hengck23-dreams-enveda-casmi26/dreams_minimal_model.py", recursive=True)[0]
D = os.path.dirname(hit)
sys.path.insert(0, D)
import dreams_minimal_model as mm  # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "cpu"
model = mm.load_model(os.path.join(D, "embedding_model_state_dict.pt"), device=dev)
qp = glob.glob("/kaggle/input/**/queries.parquet", recursive=True)[0]
n_queries = json.load(open(os.path.join(os.path.dirname(qp), "n_queries.json")))["n_queries"]
q = pd.read_parquet(qp)
print("device", dev, "| spectra", len(q), "| queries", n_queries, flush=True)

t0 = time.time()
specs, owner = [], []
for r in q.itertuples(index=False):
    mz, it = np.asarray(r.mz, np.float32), np.asarray(r.it, np.float32)
    ok = (mz > 0) & (it > 0)
    if ok.sum() == 0:
        continue
    specs.append(mm.preprocess_spectrum(mz[ok], it[ok], float(r.prec)))
    owner.append(int(r.q))
X = np.stack(specs)
E = np.zeros((len(X), 1024), np.float32)
with torch.inference_mode():
    for a in range(0, len(X), 256):
        E[a:a + 256] = mm.embed_preprocessed(model, torch.from_numpy(X[a:a + 256]))
E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
Q = np.zeros((n_queries, 1024), np.float32)
cnt = np.zeros(n_queries)
np.add.at(Q, np.asarray(owner), E)
np.add.at(cnt, np.asarray(owner), 1)
Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9
np.save("/kaggle/working/dreams_query_emb.npy", Q)
print(f"embedded {len(X)} spectra of {int((cnt > 0).sum())}/{n_queries} queries in {time.time() - t0:.0f}s", flush=True)
