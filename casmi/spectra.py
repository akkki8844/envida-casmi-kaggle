"""Spectrum cleaning and sparse binned vectors (peak channel + neutral-loss channel)."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

BIN = 0.02  # Da
MAX_MZ = 1500.0
N_BINS = int(MAX_MZ / BIN) + 1
MAX_PEAKS = 64
REL_FLOOR = 0.005
NL_MIN = 1.0  # ignore "losses" smaller than this (precursor / isotopes)


def clean_peaks(
    mz: np.ndarray, inten: np.ndarray, precursor_mz: float,
    max_peaks: int = MAX_PEAKS, rel_floor: float = REL_FLOOR,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop peaks above precursor+2 Da and below rel_floor, keep top-N, sqrt + L2 normalise."""
    mz = np.asarray(mz, dtype=np.float64)
    inten = np.asarray(inten, dtype=np.float64)
    if mz.size == 0:
        return mz, inten
    inten = inten / max(inten.max(), 1e-12)
    keep = (mz <= precursor_mz + 2.0) & (inten >= rel_floor) & (mz > 0)
    mz, inten = mz[keep], inten[keep]
    if mz.size > max_peaks:
        top = np.argpartition(-inten, max_peaks - 1)[:max_peaks]
        mz, inten = mz[top], inten[top]
    w = np.sqrt(inten)
    n = np.linalg.norm(w)
    return mz, (w / n if n > 0 else w)


def _to_vec(bins: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Merge duplicate bins (sum weights) and L2 renormalise."""
    if bins.size == 0:
        return bins.astype(np.int32), w.astype(np.float32)
    ub, inv = np.unique(bins, return_inverse=True)
    uw = np.zeros(ub.size, dtype=np.float64)
    np.add.at(uw, inv, w)
    n = np.linalg.norm(uw)
    if n > 0:
        uw /= n
    return ub.astype(np.int32), uw.astype(np.float32)


def peak_vec(mz: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bins = np.floor(mz / BIN).astype(np.int64)
    ok = (bins >= 0) & (bins < N_BINS)
    return _to_vec(bins[ok], w[ok])


def nl_vec(mz: np.ndarray, w: np.ndarray, precursor_mz: float) -> tuple[np.ndarray, np.ndarray]:
    nl = precursor_mz - mz
    ok = nl >= NL_MIN
    bins = np.floor(nl[ok] / BIN).astype(np.int64)
    ok2 = (bins >= 0) & (bins < N_BINS)
    return _to_vec(bins[ok2], w[ok][ok2])


def spectrum_vectors(mz, inten, precursor_mz):
    """Return ((peak_idx, peak_val), (nl_idx, nl_val)) for one spectrum."""
    cmz, w = clean_peaks(mz, inten, precursor_mz)
    return peak_vec(cmz, w), nl_vec(cmz, w, precursor_mz)


def stack_csr(vectors: list[tuple[np.ndarray, np.ndarray]]) -> sp.csr_matrix:
    """Stack per-spectrum (idx, val) pairs into one CSR matrix (n x N_BINS)."""
    lens = np.fromiter((len(v[0]) for v in vectors), dtype=np.int64, count=len(vectors))
    indptr = np.concatenate([[0], np.cumsum(lens)])
    if len(vectors):
        indices = np.concatenate([v[0] for v in vectors]).astype(np.int32)
        data = np.concatenate([v[1] for v in vectors]).astype(np.float32)
    else:
        indices = np.zeros(0, dtype=np.int32)
        data = np.zeros(0, dtype=np.float32)
    return sp.csr_matrix((data, indices, indptr), shape=(len(vectors), N_BINS))
