"""Baseline retrieval ranker: library similarity + analog propagation over a mass-filtered pool.

For each query molecule:
  1. Neutral mass = median over its spectra (precursor_mz minus the adduct delta).
  2. Compare every query spectrum with library spectra of the same ion mode whose neutral mass is within
     +-analog_window Da, using a peak-cosine and a neutral-loss-cosine; similarity = max of the two.
  3. Per library structure keep the best similarity per query spectrum, average over the query's spectra.
  4. Candidates = pool structures within `ppm` of the neutral mass. Score(c) = max_a sim_a**p * Tanimoto(c, a)
     over the top anchors a. A candidate that is itself an anchor has Tanimoto 1 (exact library match).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from .chem import key14
from .library import Library, Pool


@dataclass
class Query:
    pk: sp.csr_matrix
    nl: sp.csr_matrix
    neutral_mass: float
    mode: int


@dataclass
class Exclusion:
    """Things hidden from the ranker for one query (used by the validation harness only)."""
    lib_struct_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    truth_id: int | None = None
    truth_libs: tuple[str, ...] = ()
    pool_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))


@dataclass
class Params:
    analog_window: float = 200.0
    ppm: float = 10.0
    max_ppm: float = 50.0
    top_rows: int = 1000        # per query spectrum
    n_anchors: int = 100
    min_sim: float = 0.1
    power: float = 3.0
    # Candidates that have visible library spectra in this ion mode are known structures: if they were the
    # answer their own spectra should match. Their analog score is multiplied by this (1.0 = v1 behaviour);
    # their own-spectrum score sim**power is never penalised.
    spec_penalty: float = 1.0


class BaselineRanker:
    def __init__(self, lib: Library, pool: Pool, params: Params | None = None):
        self.lib, self.pool, self.p = lib, pool, params or Params()
        m = lib.meta
        self.struct = m["struct_id"].to_numpy()
        self.libname = m["ingest_lib"].to_numpy()
        self.nmass = m["neutral_mass"].to_numpy()
        mode = m["mode"].to_numpy()
        self.block = {}
        for md in (0, 1):
            idx = np.flatnonzero(mode == md)
            self.block[md] = (int(idx[0]), int(idx[-1]) + 1) if idx.size else (0, 0)
        self.pool_mass = pool.mass
        self.pool_n = pool.fps.sum(axis=1).astype(np.float32)

    # ---- spectral similarity ------------------------------------------------------------------
    def anchor_scores(self, q: Query, ex: Exclusion) -> tuple[dict[int, float], np.ndarray]:
        """Returns ({struct_id: similarity}, sorted struct ids that have any visible spectra in the window)."""
        p = self.p
        b0, b1 = self.block[q.mode]
        nm = self.nmass[b0:b1]
        a = b0 + int(np.searchsorted(nm, q.neutral_mass - p.analog_window))
        b = b0 + int(np.searchsorted(nm, q.neutral_mass + p.analog_window, side="right"))
        if b <= a:
            return {}, np.zeros(0, dtype=np.int64)
        st = self.struct[a:b]
        bad = np.isin(st, ex.lib_struct_ids) if ex.lib_struct_ids.size else np.zeros(st.size, dtype=bool)
        if ex.truth_id is not None and ex.truth_libs:
            bad |= (st == ex.truth_id) & np.isin(self.libname[a:b], ex.truth_libs)
        visible = np.unique(st[~bad])
        sim_pk = (self.lib.pk[a:b] @ q.pk.T).toarray()
        sim_nl = (self.lib.nl[a:b] @ q.nl.T).toarray()
        sim = np.maximum(sim_pk, sim_nl)
        sim[bad, :] = 0.0
        nq = sim.shape[1]
        acc: dict[int, float] = {}
        for j in range(nq):
            col = sim[:, j]
            k = min(p.top_rows, col.size)
            top = np.argpartition(-col, k - 1)[:k] if col.size > k else np.arange(col.size)
            top = top[col[top] >= p.min_sim]
            if top.size == 0:
                continue
            order = np.argsort(-col[top])
            top = top[order]
            uid, first = np.unique(st[top], return_index=True)
            for u, f in zip(uid, first):
                acc[int(u)] = acc.get(int(u), 0.0) + float(col[top[f]])
        return {u: s / nq for u, s in acc.items()}, visible

    # ---- candidates -----------------------------------------------------------------------------
    def candidates(self, mass: float, ex: Exclusion) -> np.ndarray:
        ppm = self.p.ppm
        while True:
            tol = mass * ppm * 1e-6
            lo = int(np.searchsorted(self.pool_mass, mass - tol))
            hi = int(np.searchsorted(self.pool_mass, mass + tol, side="right"))
            cand = np.arange(lo, hi)
            if ex.pool_ids.size:
                cand = cand[~np.isin(cand, ex.pool_ids)]
            if cand.size >= 25 or ppm >= self.p.max_ppm:
                return cand
            ppm *= 2.0

    def tanimoto(self, cand: np.ndarray, anchors: np.ndarray) -> np.ndarray:
        fc = self.pool.fps[cand].astype(np.float32)
        fa = self.pool.fps[anchors].astype(np.float32)
        inter = fc @ fa.T
        union = self.pool_n[cand][:, None] + self.pool_n[anchors][None, :] - inter
        return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)

    # ---- ranking ---------------------------------------------------------------------------------
    def rank(self, q: Query, ex: Exclusion | None = None, k: int = 25) -> tuple[list[str], dict]:
        ex = ex or Exclusion()
        scores, visible = self.anchor_scores(q, ex)
        cand = self.candidates(q.neutral_mass, ex)
        diag = {"n_cand": int(cand.size), "n_anchor": len(scores),
                "best_sim": max(scores.values()) if scores else 0.0}
        if cand.size == 0:
            return [], diag
        mdiff = np.abs(self.pool_mass[cand] - q.neutral_mass)
        final = -mdiff * 1e-6  # tie-break: closest mass first
        has_spec = np.isin(cand, visible)
        if scores:
            top = sorted(scores.items(), key=lambda kv: -kv[1])[: self.p.n_anchors]
            a_ids = np.array([u for u, _ in top])
            a_w = np.array([s for _, s in top], dtype=np.float32) ** self.p.power
            T = self.tanimoto(cand, a_ids)
            weighted = T * a_w[None, :]
            best = weighted.max(axis=1)
            bonus = 1e-3 * weighted.sum(axis=1)
            own = np.array([scores.get(int(c), 0.0) for c in cand], dtype=np.float32) ** self.p.power
            best = np.where(has_spec, np.maximum(own, self.p.spec_penalty * best), best)
            # exact own-spectrum matches win ties against fingerprint twins that have no spectra
            final = best + bonus + has_spec * (own >= best - 1e-9) * (own > 0) * 1e-5 + final
            no_evidence = best <= 0
        else:
            no_evidence = np.ones(cand.size, dtype=bool)
        if self.p.spec_penalty < 1.0:  # with no evidence at all, prefer structures without spectra
            final = final - (has_spec & no_evidence) * 1e-4
        order = np.argsort(-final)
        out, seen = [], set()
        for i in order[: k * 4]:
            smi = self.pool.df["smiles"].iat[int(cand[i])]
            kk = key14(smi)
            if kk is None or kk in seen:
                continue
            seen.add(kk)
            out.append(smi)
            if len(out) >= k:
                break
        return out, diag
