"""External measured-library rank-1 overlay (post-processing layer for a CASMI26 submission.csv).

Adapted from the public apex-extlib overlay (Apache-2.0; ozertuu / heliosli direct-hit rule, as run in
jtheyogi/taeyangg4's notebooks). Same scoring rule: a test spectrum matching an external reference spectrum
(same ion mode, precursor within 10 ppm + 0.002 Da, greedy sqrt-intensity cosine >= TAU with >= K_PEAKS matched
peaks at 0.015 Da) promotes that reference structure to rank 1, unless the pipeline's own library match for the
molecule is already >= LIB_TAU.

Changes against the public cell:
  * rows whose source library or licence mentions NIST (GNPS-NIST14-MATCHES: NIST-matched annotations) are dropped,
    the competition bans NIST-derived data;
  * a promoted structure already in the list is found by InChIKey first block, not by SMILES string, so it moves
    instead of appearing twice;
  * ties: the GNPS multiplex-synthesis library annotates one pooled spectrum with up to ~12 isomers, so the public
    rule promoted an arbitrary one. Here every structure within TIE_EPS of the best cosine forms a tie set; if the
    list holds any of them, its best-ranked member becomes rank 1 (the pipeline breaks the tie); otherwise one is
    inserted only when at most MAX_TIE_INSERT structures tie, and the list is left alone when the hit is ambiguous;
  * a session-time guard (skip when the session is past GUARD_H hours) and an audit file;
  * any failure leaves submission.csv byte-for-byte unchanged.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import time

import numpy as np
import pandas as pd

TAU = 0.80
K_PEAKS = 4
LIB_TAU = 0.90
MZ_TOL = 0.015
PPM = 10e-6
ABS_TOL = 0.002
GUARD_H = 8.80
PROTON = 1.007276
TIE_EPS = 0.002       # references within this cosine of the best are a tie (same pooled spectrum)
MAX_TIE_INSERT = 2    # insert a tied structure that is not in the list only when at most this many tie
MAX_TIE_MOVE = 5      # larger tie sets (pooled multiplex spectra: up to ~150 isomers) leave the list alone


def _key14(smi: str):
    try:
        from rdkit import Chem
        m = Chem.MolFromSmiles(smi)
        return Chem.MolToInchiKey(m).split("-")[0] if m is not None else None
    except Exception:
        return None


def greedy_cosine(qm, qi, rm, ri, tol=MZ_TOL):
    """Greedy one-to-one peak matching on sqrt intensities (identical to the public overlay)."""
    if len(qm) == 0 or len(rm) == 0:
        return 0.0, 0
    qa, ra = np.sqrt(qi), np.sqrt(ri)
    qa = qa / (np.linalg.norm(qa) + 1e-12)
    ra = ra / (np.linalg.norm(ra) + 1e-12)
    j0 = np.searchsorted(rm, qm - tol, "left")
    j1 = np.searchsorted(rm, qm + tol, "right")
    pairs = [(qa[i] * ra[j], i, j) for i in range(len(qm)) for j in range(j0[i], j1[i])]
    if not pairs:
        return 0.0, 0
    pairs.sort(reverse=True)
    ui, uj, s, n = set(), set(), 0.0, 0
    for p, i, j in pairs:
        if i in ui or j in uj:
            continue
        ui.add(i)
        uj.add(j)
        s += p
        n += 1
    return float(s), n


def _is_nist(*vals) -> bool:
    return any("nist" in str(v).lower() for v in vals)


def load_refs(inp: str, log=print):
    refs, dropped = [], 0
    meta_hits = sorted(glob.glob(inp + "/**/ext_meta.parquet", recursive=True), key=len)
    mz_hits = sorted(glob.glob(inp + "/**/ext_mz.npy", recursive=True), key=len)
    in_hits = sorted(glob.glob(inp + "/**/ext_in.npy", recursive=True), key=len)
    if meta_hits and mz_hits and in_hits:
        meta = pd.read_parquet(meta_hits[0])
        mz_all = np.load(mz_hits[0], mmap_mode="r")
        in_all = np.load(in_hits[0], mmap_mode="r")
        src = meta["source_library"] if "source_library" in meta else pd.Series("", index=meta.index)
        lic = meta["licence"] if "licence" in meta else pd.Series("", index=meta.index)
        for idx, row in enumerate(meta.itertuples(index=False)):
            if _is_nist(src.iat[idx], lic.iat[idx]):
                dropped += 1
                continue
            try:
                mode = 1 if "pos" in str(getattr(row, "ionization_mode", "pos")).lower() else -1
                prec = float(row.precursor_mz)
                mz_p = np.asarray(mz_all[idx], dtype=np.float64)
                it_p = np.asarray(in_all[idx], dtype=np.float64)
                ok = (mz_p > 0) & (it_p > 0)
                mz_p, it_p = mz_p[ok], it_p[ok]
                if len(mz_p) < 3:
                    continue
                o = np.argsort(mz_p)
                refs.append((mode, prec, mz_p[o], it_p[o], str(row.normalized_smiles), "gnps"))
            except Exception:
                continue
        log(f"[extlib] source A {meta_hits[0]}: {len(refs)} spectra kept, {dropped} NIST-labelled rows dropped")
    mb_hits = sorted(glob.glob(inp + "/**/spectra.parquet", recursive=True),
                     key=lambda p: ("massbank" not in p.lower(), len(p)))
    if mb_hits and "massbank" in mb_hits[0].lower():
        cols = ["smiles", "exact_mass", "ion_mode", "mzs", "intensities", "is_casmi_subset", "source", "license"]
        import pyarrow.parquet as pq
        have = set(pq.ParquetFile(mb_hits[0]).schema_arrow.names)
        mb = pd.read_parquet(mb_hits[0], columns=[c for c in cols if c in have])
        mb = mb[(~mb.is_casmi_subset.astype(bool)) & mb.smiles.notna()]
        n0, nd = len(refs), 0
        for row in mb.itertuples(index=False):
            if _is_nist(getattr(row, "source", ""), getattr(row, "license", "")):
                nd += 1
                continue
            try:
                mode = 1 if "pos" in str(row.ion_mode).lower() else -1
                prec = float(row.exact_mass) + PROTON * mode
                mz_p = np.asarray(row.mzs, dtype=np.float64)
                it_p = np.asarray(row.intensities, dtype=np.float64)
                ok = (mz_p > 0) & (it_p > 0)
                mz_p, it_p = mz_p[ok], it_p[ok]
                if len(mz_p) < 3:
                    continue
                o = np.argsort(mz_p)
                refs.append((mode, prec, mz_p[o], it_p[o], str(row.smiles), "massbank"))
            except Exception:
                continue
        log(f"[extlib] source B {mb_hits[0]}: {len(refs) - n0} spectra kept, {nd} NIST-labelled rows dropped")
    return refs, dropped


def overlay(work="/kaggle/working", inp="/kaggle/input", t0_session=None, log=print) -> dict:
    t0 = time.monotonic()
    audit = dict(stage="start")
    sub_path = os.path.join(work, "submission.csv")
    if t0_session is not None and time.time() - t0_session > GUARD_H * 3600:
        log(f"[extlib] session past {GUARD_H} h; skipped")
        return dict(stage="time-guard")
    sub = pd.read_csv(sub_path, dtype=str, keep_default_na=False)
    shutil.copy(sub_path, os.path.join(work, "submission_pre_extlib.csv"))
    test_paths = sorted(glob.glob(inp + "/**/test.parquet", recursive=True),
                        key=lambda p: ("mass-spectra" not in p.lower(), len(p)))
    te = pd.read_parquet(test_paths[0])
    queries = {}
    for r in te.itertuples(index=False):
        m = np.asarray(r.ms2_mzs, dtype=np.float64)
        it = np.asarray(r.ms2_normalized_intensities, dtype=np.float64)
        prec = float(r.precursor_mz)
        mode = 1 if "pos" in str(getattr(r, "ionization_mode", "positive")).lower() else -1
        ok = (m > 0) & (it > 0) & (m <= prec + 0.5)
        m, it = m[ok], it[ok]
        o = np.argsort(m)
        queries.setdefault(str(r.molecule_id), []).append(dict(mz=m[o], it=it[o], prec=prec, mode=mode))
    lib_max = {}
    ev_path = os.path.join(work, "frozen_candidate_evidence.parquet")
    if os.path.exists(ev_path):
        ev = pd.read_parquet(ev_path, columns=["molecule_id", "library_max"])
        lib_max = {str(k): float(v) for k, v in ev.groupby("molecule_id", sort=False).library_max.first().items()}
    refs, n_nist = load_refs(inp, log)
    audit.update(refs=len(refs), nist_dropped=n_nist, lib_max_known=len(lib_max))
    if not refs:
        log("[extlib] no reference spectra; submission unchanged")
        return dict(audit, stage="no-refs")
    order = np.argsort([r[1] for r in refs], kind="stable")
    refs = [refs[i] for i in order]
    prec_arr = np.array([r[1] for r in refs])
    new_rows, hits, promoted, moved_existing, protected, ambiguous = [], 0, 0, 0, 0, 0
    decisions = []
    for mid, answer in zip(sub["molecule_id"].astype(str), sub["smiles"].astype(str)):
        smis = [s for s in answer.split(";") if s]
        new = list(smis)
        try:
            if lib_max.get(mid, 0.0) >= LIB_TAU:
                protected += 1
                new_rows.append(answer)
                continue
            scored = []  # (cos, n_peaks, smiles) of every qualifying reference
            for q in queries.get(mid, []):
                tol = q["prec"] * PPM + ABS_TOL
                lo = np.searchsorted(prec_arr, q["prec"] - tol, "left")
                hi = np.searchsorted(prec_arr, q["prec"] + tol, "right")
                for ri in range(lo, hi):
                    ref = refs[ri]
                    if ref[0] != q["mode"]:
                        continue
                    cos, n = greedy_cosine(q["mz"], q["it"], ref[2], ref[3])
                    if n >= K_PEAKS and cos >= TAU:
                        scored.append((cos, n, ref[4]))
            if scored:
                hits += 1
                best_cos = max(s[0] for s in scored)
                # pooled (multiplex) libraries annotate one spectrum with several isomers: collect the tie set
                tie, seen = [], set()
                for cos, n, smi in sorted(scored, key=lambda s: -s[0]):
                    if cos < best_cos - TIE_EPS:
                        break
                    k = _key14(smi) or smi
                    if k not in seen:
                        seen.add(k)
                        tie.append((k, smi, cos, n))
                keys = [_key14(s) or s for s in smis]
                in_list = [j for j, k in enumerate(keys) if k in seen]
                action = "none"
                if len(tie) > MAX_TIE_MOVE:  # a large pooled tie says little about which isomer it is
                    ambiguous += 1
                    action = "ambiguous-skip"
                elif in_list:  # the pipeline's own order breaks the tie (or confirms a unique hit)
                    j = in_list[0]
                    if j > 0:
                        new = [smis[j]] + smis[:j] + smis[j + 1:]
                        promoted += 1
                        moved_existing += 1
                        action = "moved"
                    else:
                        action = "already-rank1"
                elif len(tie) <= MAX_TIE_INSERT:
                    new = ([tie[0][1]] + smis)[:25]
                    promoted += 1
                    action = "inserted"
                else:
                    ambiguous += 1
                    action = "ambiguous-skip"
                decisions.append(dict(molecule_id=mid, cos=round(best_cos, 4), peaks=tie[0][3], tie=len(tie),
                                      smiles=tie[0][1], old_rank1=smis[0] if smis else "", action=action,
                                      changed=new[:1] != smis[:1]))
        except Exception as exc:  # one molecule's failure keeps its list
            new = smis
            decisions.append(dict(molecule_id=mid, error=repr(exc)))
        new_rows.append(";".join(new) if new else answer)
    out = sub.copy()
    out["smiles"] = new_rows
    assert len(out) == len(sub) and out["molecule_id"].tolist() == sub["molecule_id"].tolist()
    assert out["smiles"].str.split(";").map(len).max() <= 25 and (out["smiles"].str.len() > 0).all()
    tmp = os.path.join(work, "submission.extlib.tmp.csv")
    out.to_csv(tmp, index=False, lineterminator=chr(10))
    os.replace(tmp, sub_path)
    audit.update(stage="done", molecules=len(sub), protected=protected, hits=hits, promoted=promoted,
                 moved_existing=moved_existing, ambiguous_skipped=ambiguous, elapsed_s=round(time.monotonic() - t0, 1))
    with open(os.path.join(work, "extlib_audit.json"), "w") as f:
        json.dump(dict(audit, decisions=decisions), f, indent=1)
    log(f"[extlib] {audit}")
    return audit


def run(work="/kaggle/working", inp="/kaggle/input", t0_session=None):
    sub_path = os.path.join(work, "submission.csv")
    backup = None
    try:
        backup = open(sub_path, "rb").read() if os.path.exists(sub_path) else None
        return overlay(work, inp, t0_session)
    except Exception:
        import traceback
        traceback.print_exc()
        if backup is not None:
            with open(sub_path, "wb") as f:
                f.write(backup)
        print("[extlib] overlay failed; submission restored unchanged")
        return dict(stage="failed")
