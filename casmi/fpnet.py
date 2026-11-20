"""Spectrum -> Morgan fingerprint model (FPNet) and per-candidate fingerprint-agreement features.

Input tokens: the 0.02 Da peak / neutral-loss vectors of casmi.spectra pooled to 0.1 Da bins (peaks < 1500 Da,
losses < 500 Da) plus an ion-mode token, weighted by the vector values. The network is
    EmbeddingBag(V, H, sum) -> ReLU -> Linear(H, H) -> ReLU -> Linear(H, 2048) -> sigmoid
trained with torch (scripts/train_fpnet.py) and run here in numpy, so the Kaggle notebook needs no torch.

Two models are trained on disjoint structure folds (fold = crc32(inchikey14) % 2). Ranker-training features use the
model that did not see the query's structure; the submission averages both.
"""
from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .spectra import N_BINS

POOL = 5                      # 0.02 Da -> 0.1 Da
N_PK = N_BINS // POOL + 1     # 15001 peak tokens
N_NL = 5000                   # losses below 500 Da
V = N_PK + N_NL + 2           # + 2 ion-mode tokens
N_BITS = 2048
FEATURES = ["fp_ll", "fp_tani", "fp_cos", "fp_ll_rel", "fp_rank"]


def fold_of(key14: str) -> int:
    return zlib.crc32(key14.encode()) % 2


def _pool_matrix(n_in: int, n_out: int, offset: int) -> sp.csr_matrix:
    src = np.arange(n_in)
    dst = src // POOL
    ok = dst < n_out
    return sp.csr_matrix((np.ones(ok.sum(), dtype=np.float32), (src[ok], dst[ok] + offset)), shape=(n_in, V))


_MPK = _MNL = None


def tokens(pk: sp.csr_matrix, nl: sp.csr_matrix, modes) -> sp.csr_matrix:
    """(n, V) sparse token-weight matrix for n spectra."""
    global _MPK, _MNL
    if _MPK is None:
        _MPK = _pool_matrix(pk.shape[1], N_PK, 0)
        _MNL = _pool_matrix(nl.shape[1], N_NL, N_PK)
    modes = np.broadcast_to(np.asarray(modes, dtype=np.int64), (pk.shape[0],))
    mode = sp.csr_matrix((np.ones(pk.shape[0], dtype=np.float32), (np.arange(pk.shape[0]), N_PK + N_NL + modes)),
                         shape=(pk.shape[0], V))
    X = (pk.astype(np.float32) @ _MPK + nl.astype(np.float32) @ _MNL + mode).tocsr()
    X.sort_indices()
    return X


class FPNet:
    """numpy inference for one or more exported models (their probabilities are averaged)."""

    def __init__(self, paths):
        self.models = []
        for p in paths:
            d = np.load(p)
            self.models.append({k: d[k].astype(np.float32) for k in ("E", "b1", "W2", "b2", "W3", "b3")})

    def predict(self, X: sp.csr_matrix, models=None) -> np.ndarray:
        out = np.zeros((X.shape[0], N_BITS), dtype=np.float32)
        use = self.models if models is None else [self.models[i] for i in models]
        for m in use:
            h = np.maximum(X @ m["E"] + m["b1"], 0.0)
            h = np.maximum(h @ m["W2"].T + m["b2"], 0.0)
            z = h @ m["W3"].T + m["b3"]
            out += 1.0 / (1.0 + np.exp(-z))
        return out / len(use)

    def query_probs(self, pk, nl, mode, models=None) -> np.ndarray:
        """Mean bit probabilities over a query's spectra."""
        return self.predict(tokens(pk, nl, mode), models).mean(axis=0)


def candidate_features(p: np.ndarray, fps: np.ndarray) -> np.ndarray:
    """(n_cand, len(FEATURES)) agreement between predicted probabilities p (2048,) and candidate bits fps."""
    c = fps.astype(np.float32)
    pc = np.clip(p, 1e-4, 1 - 1e-4)
    lp, l1p = np.log(pc), np.log(1 - pc)
    ll = (c @ (lp - l1p) + l1p.sum()) / N_BITS
    inter = c @ p
    tani = inter / np.maximum(p.sum() + c.sum(1) - inter, 1e-9)
    cos = inter / np.maximum(np.linalg.norm(p) * np.sqrt(c.sum(1)), 1e-9)
    order = np.argsort(-ll, kind="stable")
    rk = np.empty(ll.size)
    rk[order] = np.arange(ll.size)
    return np.stack([ll, tani, cos, ll - ll.max(), rk / max(ll.size - 1, 1)], axis=1).astype(np.float32)


def default_paths(root: Path) -> list[Path]:
    return sorted(Path(root).glob("fpnet_f*.npz"))
