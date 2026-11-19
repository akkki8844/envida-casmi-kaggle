"""Structure-disjoint validation that simulates the three novelty classes by removal.

Class 1: query spectra come from the timsTOF libraries; the truth's spectra from those libraries are hidden
         but its spectra from other (public) libraries stay visible (cross-instrument library hit).
Class 2: all of the truth's spectra are hidden; the structure stays in the candidate pool.
Class 3: all spectra AND pool entries of the truth and of anything with Tanimoto >= 0.7 to it are hidden.

Two panels, which should be read side by side:
  np    - enveda-np-examples structures (timsTOF natural products; many also exist in public libraries)
  e180  - enveda-180 structures (synthetic, no/low public spectra): class-2/3-like only
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm

from .baseline import BaselineRanker, Exclusion, Query
from .chem import key14
from .library import Library, Pool
from .metric import reciprocal_rank

TIMSTOF_LIBS = ("enveda-np-examples", "enveda-180")
DEFAULT_MIX = {1: 0.2, 2: 0.4, 3: 0.4}  # competitor estimates disagree wildly; treat as a rough guide


@dataclass
class Item:
    panel: str
    cls: int
    struct_id: int
    rows: np.ndarray          # library row indices of the query spectra
    ex: Exclusion


def _tanimoto_to_all(pool: Pool, pool_n: np.ndarray, sid: int, chunk: int = 30000) -> np.ndarray:
    fa = pool.fps[sid].astype(np.float32)
    out = np.zeros(len(pool.df), dtype=np.float32)
    for s in range(0, len(out), chunk):
        fc = pool.fps[s:s + chunk].astype(np.float32)
        inter = fc @ fa
        union = pool_n[s:s + chunk] + pool_n[sid] - inter
        out[s:s + chunk] = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
    return out


def make_items(lib: Library, pool: Pool, n_per_class: int, seed: int = 0, nn_thresh: float = 0.7) -> list[Item]:
    rng = np.random.default_rng(seed)
    m = lib.meta
    pool_n = pool.fps.sum(axis=1).astype(np.float32)
    libs_of = m.groupby("struct_id")["ingest_lib"].agg(lambda s: set(s))
    by_struct_lib = m.groupby(["struct_id", "ingest_lib"]).indices

    def pick_rows(sid: int, libname: str) -> np.ndarray:
        rows = np.asarray(by_struct_lib[(sid, libname)])
        n = min(len(rows), 1 + int(rng.poisson(2.5)))
        return rng.choice(rows, size=n, replace=False)

    items: list[Item] = []

    def add(panel, cls, sid, libname):
        rows = pick_rows(sid, libname)
        if cls == 1:
            ex = Exclusion(truth_id=sid, truth_libs=TIMSTOF_LIBS)
        elif cls == 2:
            ex = Exclusion(lib_struct_ids=np.array([sid]))
        else:
            nb = np.flatnonzero(_tanimoto_to_all(pool, pool_n, sid) >= nn_thresh)
            nb = np.union1d(nb, [sid])
            # tautomers of the truth can sit in the pool under another plain InChIKey; they share the
            # formula (so the mass), so check the official key only inside a tight mass window
            mass, tk = pool.mass[sid], key14(pool.df["smiles"].iat[sid])
            lo = int(np.searchsorted(pool.mass, mass - mass * 5e-6))
            hi = int(np.searchsorted(pool.mass, mass + mass * 5e-6, side="right"))
            same = [i for i in range(lo, hi) if key14(pool.df["smiles"].iat[i]) == tk]
            nb = np.union1d(nb, same)
            ex = Exclusion(lib_struct_ids=nb, pool_ids=nb)
        items.append(Item(panel, cls, sid, rows, ex))

    # panel np
    np_structs = np.array(sorted(s for s, l in libs_of.items() if "enveda-np-examples" in l))
    rng.shuffle(np_structs)
    with_public = [s for s in np_structs if libs_of[s] - set(TIMSTOF_LIBS)]
    c1 = with_public[:n_per_class]
    rest = [s for s in np_structs if s not in set(c1)]
    half = len(rest) // 2
    for s in c1:
        add("np", 1, s, "enveda-np-examples")
    for s in rest[:half][:n_per_class]:
        add("np", 2, s, "enveda-np-examples")
    for s in rest[half:][:n_per_class]:
        add("np", 3, s, "enveda-np-examples")
    # panel e180
    e_structs = np.array(sorted(s for s, l in libs_of.items() if "enveda-180" in l))
    rng.shuffle(e_structs)
    for s in e_structs[:n_per_class]:
        add("e180", 2, s, "enveda-180")
    for s in e_structs[n_per_class:2 * n_per_class]:
        add("e180", 3, s, "enveda-180")
    return items


def item_query(lib: Library, item: Item) -> Query:
    m = lib.meta.iloc[item.rows]
    return Query(
        pk=lib.pk[item.rows], nl=lib.nl[item.rows],
        neutral_mass=float(np.median(m["neutral_mass"])), mode=int(m["mode"].iloc[0]),
    )


def evaluate(ranker: BaselineRanker, lib: Library, pool: Pool, items: list[Item], desc: str = "eval") -> pd.DataFrame:
    rows = []
    for it in tqdm(items, desc=desc):
        smiles = pool.df["smiles"].iat[it.struct_id]
        tk = key14(smiles)
        q = item_query(lib, it)
        guesses, diag = ranker.rank(q, it.ex)
        window = ranker.candidates(q.neutral_mass, it.ex)
        in_window = bool(it.struct_id in set(window.tolist()))
        rows.append({
            "panel": it.panel, "cls": it.cls, "struct_id": it.struct_id,
            "n_spectra": len(it.rows), "rr": reciprocal_rank(guesses, tk),
            "truth_in_window": in_window, **diag,
        })
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, mix: dict[int, float] = DEFAULT_MIX, n_boot: int = 500, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    lines = []
    g = df.groupby(["panel", "cls"])
    tab = g.agg(n=("rr", "size"), mrr=("rr", "mean"), top1=("rr", lambda s: (s == 1.0).mean()),
                window=("truth_in_window", "mean"), cand=("n_cand", "median"), best_sim=("best_sim", "median"))
    lines.append(tab.round(3).to_string())
    for panel, d in df.groupby("panel"):
        per = {c: x["rr"].to_numpy() for c, x in d.groupby("cls")}
        ws = {c: w for c, w in mix.items() if c in per}
        z = sum(ws.values())

        def mixed(arrs):
            return sum(ws[c] / z * arrs[c].mean() for c in ws)

        boots = [mixed({c: rng.choice(per[c], size=len(per[c])) for c in ws}) for _ in range(n_boot)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        lines.append(f"panel {panel}: mixed MRR@25 = {mixed(per):.4f}  (95% CI {lo:.4f}-{hi:.4f}; weights "
                     f"{ {c: round(w / z, 2) for c, w in ws.items()} })")
    return "\n".join(lines)
