"""MetFrag-style in-silico fragmentation and peak explanation.

fragment_masses(smiles): neutral masses of the substructures reachable by breaking up to `max_depth` times, where a
break is one acyclic bond or two bonds of the same ring. Atoms keep the hydrogens they carry in the intact molecule;
hydrogen rearrangements are allowed at matching time (+-H shifts).

FragmentScorer.match(peaks, mode, struct_ids): per-candidate features describing how much of the query's MS/MS
intensity the candidate's fragment ions explain. Same-formula isomers get different values, which is what the
formula-level evidence cannot do.
"""
from __future__ import annotations

import numpy as np

H_MASS = 1.00782503207
PROTON = 1.007276467
NA_SHIFT = 22.98976928 - H_MASS  # sodiated fragments: [frag+Na]+ = [frag+H]+ + (Na - H)
H_SHIFTS = np.array([-2, -1, 0, 1, 2])
MIN_FRAG_MASS = 25.0
PRECURSOR_GAP = 1.6  # ignore peaks within this many Da below the precursor (every candidate explains them)
FEATURES = ["fx_int", "fx_cheap", "fx_score", "fx_n", "frag_log_n", "fx_rel", "fx_rank"]
N_FEAT = len(FEATURES)
EMPTY = (np.zeros(0), np.zeros(0, dtype=np.int8))


def fragment_masses(smiles: str, max_depth: int = 2, max_atoms: int = 70) -> tuple[np.ndarray, np.ndarray]:
    """(unique fragment masses, minimal number of bonds broken to reach each)."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return EMPTY
    n = mol.GetNumAtoms()
    if n > max_atoms:
        max_depth = 1
    pt = Chem.GetPeriodicTable()
    amass = np.array([pt.GetMostCommonIsotopeMass(a.GetAtomicNum()) + a.GetTotalNumHs() * H_MASS
                      for a in mol.GetAtoms()])
    nbr = [0] * n
    bonds = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        nbr[i] |= 1 << j
        nbr[j] |= 1 << i
        bonds.append((i, j))
    ring_bonds = [list(r) for r in mol.GetRingInfo().BondRings()]
    ring_pairs = [(bonds[r[x]], bonds[r[y]]) for r in ring_bonds for x in range(len(r)) for y in range(x + 1, len(r))]
    in_ring = {bonds[k] for r in ring_bonds for k in r}

    def components(mask: int) -> list[int]:
        out, rem = [], mask
        while rem:
            comp = frontier = rem & -rem
            while frontier:
                new, f = 0, frontier
                while f:
                    low = f & -f
                    new |= nbr[low.bit_length() - 1]
                    f ^= low
                frontier = new & mask & ~comp
                comp |= frontier
            out.append(comp)
            rem &= ~comp
        return out

    def cut(mask: int, pairs) -> list[int]:
        for i, j in pairs:
            nbr[i] &= ~(1 << j)
            nbr[j] &= ~(1 << i)
        try:
            comps = components(mask)
        finally:
            for i, j in pairs:
                nbr[i] |= 1 << j
                nbr[j] |= 1 << i
        return comps if len(comps) > 1 else []

    full = (1 << n) - 1
    seen = {full: 0}  # mask -> bonds broken
    level = [full]
    for _ in range(max_depth):
        nxt = []
        for mask in level:
            ops = [((i, j),) for i, j in bonds if (mask >> i) & 1 and (mask >> j) & 1 and (i, j) not in in_ring]
            ops += [((i, j),) for i, j in in_ring if (mask >> i) & 1 and (mask >> j) & 1]  # opened rings: bridges now
            if mask == full:  # ring openings only at the first level (they dominate the cost)
                ops += ring_pairs
            for op in ops:
                c_cost = seen[mask] + len(op)
                for c in cut(mask, op):
                    if c not in seen:
                        seen[c] = c_cost
                        nxt.append(c)
                    elif c_cost < seen[c]:
                        seen[c] = c_cost
        level = nxt
    masks = [m for m in seen if m != full]
    if not masks:
        return EMPTY
    masses = np.round(np.array([amass[[k for k in range(n) if (m >> k) & 1]].sum() for m in masks]), 5)
    costs = np.array([seen[m] for m in masks], dtype=np.int8)
    order = np.lexsort((costs, masses))
    masses, costs = masses[order], costs[order]
    first = np.concatenate([[True], np.diff(masses) > 0])  # cheapest route per mass
    masses, costs = masses[first], costs[first]
    keep = masses >= MIN_FRAG_MASS
    return masses[keep], costs[keep]


def _worker(args):
    sid, smi = args
    try:
        return sid, fragment_masses(smi)
    except Exception:
        return sid, EMPTY


class FragmentScorer:
    def __init__(self, smiles: np.ndarray, top_k: int = 30, ppm: float = 10.0, min_tol: float = 0.003):
        self.smiles, self.top_k, self.ppm, self.min_tol = smiles, top_k, ppm, min_tol
        self.cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def masses(self, sid: int) -> tuple[np.ndarray, np.ndarray]:
        m = self.cache.get(sid)
        if m is None:
            m = self.cache[sid] = _worker((sid, self.smiles[sid]))[1]
        return m

    def precompute(self, struct_ids, n_jobs: int = 1, chunksize: int = 16, progress=None) -> None:
        todo = [int(s) for s in dict.fromkeys(struct_ids) if int(s) not in self.cache]
        args = [(s, self.smiles[s]) for s in todo]
        if n_jobs <= 1 or len(args) < 64:
            it = map(_worker, args)
        else:
            import multiprocessing as mp
            ctx = mp.get_context("fork" if hasattr(__import__("os"), "fork") else "spawn")
            pool = ctx.Pool(n_jobs)
            it = pool.imap_unordered(_worker, args, chunksize=chunksize)
        for sid, m in (progress(it, total=len(args)) if progress else it):
            self.cache[sid] = m
        if n_jobs > 1 and len(args) >= 64:
            pool.close()
            pool.join()

    @staticmethod
    def merge_peaks(peaks) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """peaks: list of (mz, intensity, precursor_mz, adduct). Returns mz, relative weight, sodiated flag."""
        from .spectra import clean_peaks
        mzs, ws, na = [], [], []
        for mz, inten, prec, adduct in peaks:
            cmz, w = clean_peaks(np.asarray(mz), np.asarray(inten), float(prec))
            keep = cmz < float(prec) - PRECURSOR_GAP
            if not keep.any():
                continue
            w = w[keep] ** 2
            mzs.append(cmz[keep])
            ws.append(w / w.sum())
            na.append(np.full(keep.sum(), "Na" in str(adduct)))
        if not mzs:
            return np.zeros(0), np.zeros(0), np.zeros(0, dtype=bool)
        return np.concatenate(mzs), np.concatenate(ws) / len(mzs), np.concatenate(na)

    def match(self, peaks, mode: int, struct_ids: np.ndarray) -> np.ndarray:
        """(len(struct_ids), N_FEAT) features; NaN rows when the query has no usable peaks."""
        out = np.full((len(struct_ids), N_FEAT), np.nan, dtype=np.float32)
        mz, w, na = self.merge_peaks(peaks)
        if mz.size == 0:
            return out
        tol = np.maximum(self.min_tol, mz * self.ppm * 1e-6)
        charge = -PROTON if mode == 1 else PROTON
        lo, hi = mz - tol, mz + tol
        for r, sid in enumerate(struct_ids):
            fm, fc = self.masses(int(sid))
            if fm.size == 0:
                out[r, :5] = 0.0
                continue
            best = np.zeros(mz.size)  # plausibility per peak: 0.5**(bonds broken - 1) * 0.5**|h|
            cheap = np.zeros(mz.size, dtype=bool)
            for h in H_SHIFTS:
                for shift_na in (False, True):
                    if shift_na and not na.any():
                        continue
                    ions = fm + charge + h * H_MASS + (NA_SHIFT if shift_na else 0.0)
                    a, b = np.searchsorted(ions, lo), np.searchsorted(ions, hi, side="right")
                    hit = b > a
                    if shift_na:
                        hit &= na
                    if not hit.any():
                        continue
                    minc = np.full(mz.size, 99.0)
                    for k in np.flatnonzero(hit):
                        minc[k] = fc[a[k]:b[k]].min()
                    plaus = np.where(hit, 0.5 ** (minc - 1.0) * 0.5 ** abs(h), 0.0)
                    best = np.maximum(best, plaus)
                    cheap |= hit & (minc <= 1) & (abs(h) <= 1)
            explained = best > 0
            out[r, 0] = w[explained].sum()
            out[r, 1] = w[cheap].sum()
            out[r, 2] = (w * best).sum()
            out[r, 3] = explained.mean()
            out[r, 4] = np.log1p(fm.size)
        ok = ~np.isnan(out[:, 0])
        if ok.any():
            v = out[ok, 2]
            out[ok, 5] = v - v.max()
            order = np.argsort(-v, kind="stable")
            rk = np.empty(v.size)
            rk[order] = np.arange(v.size)
            out[ok, 6] = rk / max(v.size - 1, 1)
        return out
