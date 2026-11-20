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
        cand, final, diag = self.score(q, ex)
        return self.emit(cand, final, k), diag

    def emit(self, cand: np.ndarray, final: np.ndarray, k: int = 25) -> list[str]:
        """Best-first SMILES, deduplicated on the official key."""
        if cand.size == 0:
            return []
        order = np.argsort(-final, kind="stable")
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
        return out

    def score(self, q: Query, ex: Exclusion) -> tuple[np.ndarray, np.ndarray, dict]:
        """Heuristic v2 score for every candidate in the mass window."""
        scores, visible = self.anchor_scores(q, ex)
        cand = self.candidates(q.neutral_mass, ex)
        diag = {"n_cand": int(cand.size), "n_anchor": len(scores),
                "best_sim": max(scores.values()) if scores else 0.0}
        if cand.size == 0:
            return cand, np.zeros(0), diag
        return cand, self._heuristic(q, scores, visible, cand), diag

    def _heuristic(self, q: Query, scores: dict, visible: np.ndarray, cand: np.ndarray) -> np.ndarray:
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
        return final

    # ---- features for a learned ranker -------------------------------------------------------------
    FEATURES = [
        "h_score", "h_rank_pct", "h_gap", "s1_T", "s2_T", "s3_T", "sum_s3_T", "sum_s1_T", "t_max", "t_top1",
        "s_at_tmax", "n_t07", "n_t05", "own", "has_spec", "in_db", "in_train", "ppm", "abs_ppm", "fp_bits",
        "q_best_sim", "q_second_sim", "q_n_cand", "q_n_anchor", "q_n_spec", "q_mode", "q_mass", "q_n_spec_cand",
    ]

    def features(self, q: Query, ex: Exclusion) -> tuple[np.ndarray, np.ndarray, dict]:
        """Per-candidate feature matrix (columns = FEATURES) for the mass window."""
        scores, visible = self.anchor_scores(q, ex)
        cand = self.candidates(q.neutral_mass, ex)
        diag = {"n_cand": int(cand.size), "n_anchor": len(scores),
                "best_sim": max(scores.values()) if scores else 0.0}
        n = cand.size
        if n == 0:
            return cand, np.zeros((0, len(self.FEATURES)), dtype=np.float32), diag
        h = self._heuristic(q, scores, visible, cand).astype(np.float32)
        F = {k: np.zeros(n, dtype=np.float32) for k in self.FEATURES}
        F["h_score"] = h
        order = np.argsort(-h, kind="stable")
        r = np.empty(n)
        r[order] = np.arange(n)
        F["h_rank_pct"] = r / max(n - 1, 1)
        F["h_gap"] = h.max() - h
        has_spec = np.isin(cand, visible)
        F["has_spec"] = has_spec
        F["own"] = np.array([scores.get(int(c), 0.0) for c in cand], dtype=np.float32)
        if scores:
            top = sorted(scores.items(), key=lambda kv: -kv[1])[: self.p.n_anchors]
            a_ids = np.array([u for u, _ in top])
            s = np.array([v for _, v in top], dtype=np.float32)
            T = self.tanimoto(cand, a_ids).astype(np.float32)
            F["s1_T"] = (T * s).max(1)
            F["s2_T"] = (T * s**2).max(1)
            F["s3_T"] = (T * s**3).max(1)
            F["sum_s3_T"] = (T * s**3).sum(1)
            F["sum_s1_T"] = (T * s).sum(1)
            F["t_max"] = T.max(1)
            F["t_top1"] = T[:, 0]
            F["s_at_tmax"] = s[T.argmax(1)]
            F["n_t07"] = (T >= 0.7).sum(1)
            F["n_t05"] = (T >= 0.5).sum(1)
            F["q_best_sim"][:] = s[0]
            F["q_second_sim"][:] = s[1] if s.size > 1 else 0.0
        df = self.pool.df
        F["in_db"] = df["in_db"].to_numpy()[cand] if "in_db" in df else np.zeros(n)
        # a structure whose spectra are all hidden must look like a database-only one, or the model learns
        # "in train but no visible spectra" = the truth (never true on the real test set)
        in_train = df["in_train"].to_numpy()[cand].astype(bool) if "in_train" in df else np.ones(n, dtype=bool)
        if ex.lib_struct_ids.size:
            in_train &= ~np.isin(cand, ex.lib_struct_ids)
        F["in_train"] = in_train
        ppm = (self.pool_mass[cand] - q.neutral_mass) / q.neutral_mass * 1e6
        F["ppm"], F["abs_ppm"] = ppm, np.abs(ppm)
        F["fp_bits"] = self.pool_n[cand]
        F["q_n_cand"][:] = np.log1p(n)
        F["q_n_anchor"][:] = np.log1p(len(scores))
        F["q_n_spec"][:] = q.pk.shape[0]
        F["q_mode"][:] = q.mode
        F["q_mass"][:] = q.neutral_mass
        F["q_n_spec_cand"][:] = has_spec.mean()
        X = np.stack([np.asarray(F[k], dtype=np.float32) for k in self.FEATURES], axis=1)
        return cand, X, diag


class LearnedRanker(BaselineRanker):
    """Same candidates and features; ordering from a LightGBM model (falls back to the heuristic if absent)."""

    def __init__(self, lib: Library, pool: Pool, model_path, params: Params | None = None):
        super().__init__(lib, pool, params)
        import lightgbm as lgb
        self.model = lgb.Booster(model_file=str(model_path))

    def rank(self, q: Query, ex: Exclusion | None = None, k: int = 25) -> tuple[list[str], dict]:
        ex = ex or Exclusion()
        cand, X, diag = self.features(q, ex)
        if cand.size == 0:
            return [], diag
        pred = self.model.predict(X)
        return self.emit(cand, pred, k), diag
