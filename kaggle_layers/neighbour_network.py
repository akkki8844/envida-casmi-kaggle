"""envida layer: neighbour network (N-1c lambda 0.35, arm D, rank-1 lock), verbatim from the public notebook
jtheyogi/enveda-casmi26-apex-0498-sovereign cell 26 (taeyangg4 / analyticaobscura, Apache-2.0), with auto-run off.
run_layer() adds: run only after a full pipeline run (frozen_run_info.json), and a time cap tied to the session.
"""
_NNET_NOTEBOOK = False  # envida: never auto-run on import
"""N-1c lambda 0.35 neighbour network with structure switches and a rank-1 lock (H-A-152 s2).
Purpose: rerank candidates with molecular-network neighbour evidence; optional switches
  lib  (A) fixed known-structure nodes from competition train.parquet,
  mass (B) mass-difference (edit) plausibility weight on edges,
  steps(C) two-step propagation (neighbours' step-1 reranked top-5 as evidence),
  lock1    keep the plain N-1c (all switches off) leader at rank 1; the arm only reorders ranks 2-25
           (an arm leader that differs goes to rank 2, the rest shift down).
With every switch off the arithmetic is the n1c035_cell_v1 cell's (base gate). Copy of n035net_cell_v1.py 1.0.0
plus lock_rank1 and the lock1 argument of run() (= n035netL1_cell_v1.py L1-1.0.0, evaluated in H-A-152 s2), and here
(kernel version) a batched load_library (two passes, groups reduced as soon as their rows are read; same library arrays as
the one-shot 1.0.0 loader, peak RAM about 46 GB -> a few GB), a memory guard (_NNET_MIN_FREE_GB available RAM) in the
library stage and a 1800 s budget. Ranking functions are unchanged.
Usage: import definitions for CPU replays; notebook calls run() after PIPELINE_DONE.
Inputs: submission.csv; competition test.parquet (+ train.parquet when lib is on).
Outputs: atomic submission.csv when changed; n035net_audit.json; pre-change copy. Any error, timeout or memory guard keeps
the input bytes.
Version: L1-1.1.0 (2026-11-17)
"""
_NNET_ON = True
_NNET_LAM = 0.35
_NNET_CAP_S = 1800.0
_NNET_ARM = "D"
_NNET_LOCK1 = True
_NNET_VERSION = "L1-1.1.0"
_NNET_LIB_BATCH = 65536  # parquet rows per peak batch of the library build
_NNET_MIN_FREE_GB = 2.0  # memory guard: the library stage stops (input bytes kept) below this much available RAM
_NNET_ARMS = {"base": (False, False, 1), "A": (True, False, 1), "B": (False, True, 1),
              "C": (False, False, 2), "D": (True, True, 2)}
_NNET_LIB_K = 5          # fixed nodes kept per query
_NNET_LIB_M = 100        # exact checks per query and adduct after the prefilter
_NNET_LIB_ROWS = 8       # train rows merged per (inchikey14, adduct)
_NNET_OFF_EDIT = 0.5     # edge weight factor when the precursor difference is no known edit
# H-A-135 s1 edit masses (v4b transforms + common natural-product edits), Da
_NNET_EDITS = (2.01565, 14.01565, 15.0109, 15.99491, 18.01056, 28.0313, 30.01056, 31.98983, 33.96103,
               42.01056, 43.98983, 68.0626, 79.95681, 86.00039, 104.02621, 132.04226, 146.03678,
               146.05791, 152.01096, 162.03169, 162.05282, 176.03209, 176.04734)
_NNET_ADDUCTS = ("[M+H]+", "[M-H]-")


def _nnet_timecheck(deadline):
    import time
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("NNET time budget exhausted")


def _nnet_memcheck(where):
    try:
        import psutil
        free = psutil.virtual_memory().available / 2 ** 30
    except Exception:
        return
    if free < _NNET_MIN_FREE_GB:
        raise MemoryError("NNET memory guard: %.2f GB available at %s" % (free, where))


# ---- unchanged N-1c pieces (same arithmetic as n1c035_cell_v1) ----
def build_spectra(test, allowed, deadline=None):
    import numpy as np
    import pandas as pd
    required = {"molecule_id", "adduct", "precursor_mz", "ms2_mzs", "ms2_normalized_intensities"}
    if not required.issubset(test.columns):
        raise ValueError("missing spectrum columns")
    table = test.copy()
    table["_n035_id"] = table["molecule_id"].astype(str)
    spectra = {}
    for (mid, adduct), group in table.groupby(["_n035_id", "adduct"]):
        _nnet_timecheck(deadline)
        if mid not in allowed or str(adduct) not in ("[M+H]+", "[M-H]-"):
            continue
        prec = float(np.median(pd.to_numeric(group["precursor_mz"], errors="coerce")))
        if not np.isfinite(prec):
            raise ValueError("invalid precursor")
        mass_rows, intensity_rows = [], []
        for masses, weights in zip(group["ms2_mzs"], group["ms2_normalized_intensities"]):
            masses, weights = np.asarray(masses, dtype=float), np.asarray(weights, dtype=float)
            if masses.ndim != 1 or weights.ndim != 1 or len(masses) != len(weights):
                raise ValueError("invalid peak dimensions")
            if not np.all(np.isfinite(masses)) or not np.all(np.isfinite(weights)) or np.any(weights < 0):
                raise ValueError("invalid peak values")
            mass_rows.append(masses)
            intensity_rows.append(weights / max(float(weights.max()) if len(weights) else 1.0, 1e-9))
        masses = np.concatenate(mass_rows)
        weights = np.concatenate(intensity_rows)
        keep = masses < prec - 1.5
        masses, weights = masses[keep], weights[keep]
        if len(masses) == 0:
            continue
        bins, inverse = np.unique(np.round(masses, 2), return_inverse=True)
        maxima = np.zeros(len(bins))
        np.maximum.at(maxima, inverse, weights)
        strongest = np.argsort(-maxima)[:120]
        binned = bins[strongest]
        amplitude = np.sqrt(maxima[strongest])
        amplitude /= max(np.linalg.norm(amplitude), 1e-12)
        order = np.argsort(binned)
        spectra.setdefault(mid, {})[str(adduct)] = (binned[order], amplitude[order], prec)
    return spectra


def modified_cosine(left, right, deadline=None):
    import numpy as np
    a_mass, a_weight, a_prec = left
    b_mass, b_weight, b_prec = right
    options = []
    for i, mass in enumerate(a_mass):
        _nnet_timecheck(deadline)
        for delta in (0.0, a_prec - b_prec):
            lower = np.searchsorted(b_mass, mass - delta - 0.0101)
            upper = np.searchsorted(b_mass, mass - delta + 0.0101, side="right")
            options.extend((a_weight[i] * b_weight[j], i, j) for j in range(lower, upper))
    options.sort(reverse=True)
    used_left, used_right = set(), set()
    total, matches = 0.0, 0
    for weight, i, j in options:
        if i not in used_left and j not in used_right:
            used_left.add(i)
            used_right.add(j)
            total += weight
            matches += 1
    return total, matches


def build_graph(spectra, deadline=None, tau=0.6):
    ids = sorted(spectra)
    graph = {}
    for index, left in enumerate(ids):
        _nnet_timecheck(deadline)
        for right in ids[index + 1:]:
            best = None
            for adduct in sorted(set(spectra[left]).intersection(spectra[right])):
                cosine, matches = modified_cosine(spectra[left][adduct], spectra[right][adduct], deadline)
                if matches >= 6 and cosine >= tau and (best is None or cosine > best):
                    best = cosine
            if best is not None:
                graph.setdefault(left, []).append((right, best))
                graph.setdefault(right, []).append((left, best))
    return graph


def rerank_lists(lists, scores, lam=0.35):
    import numpy as np
    if not np.isfinite(lam) or lam < 0:
        raise ValueError("invalid lambda")
    output = {}
    for mid, candidates in lists.items():
        current = list(candidates)
        if lam and mid in scores and len(current) > 1:
            neighbour = np.asarray(scores[mid], dtype=float)
            if neighbour.shape != (len(current),) or not np.all(np.isfinite(neighbour)):
                raise ValueError("invalid neighbour scores")
            rank = -np.log(np.arange(1, len(current) + 1))
            standardize = lambda values: (values - values.mean()) / (values.std() + 1e-9)
            order = np.argsort(-(standardize(rank) + lam * standardize(neighbour)), kind="stable")
            current = [current[int(i)] for i in order]
        output[mid] = current
    return output


def _nnet_parse_submission(raw):
    import csv, io
    rows = list(csv.reader(io.StringIO(raw.decode("utf-8"), newline="")))
    if not rows or rows[0] != ["molecule_id", "smiles"] or len(rows) < 2:
        raise ValueError("invalid submission header or no rows")
    lists = {}
    for row in rows[1:]:
        if len(row) != 2 or not row[0] or row[0] in lists:
            raise ValueError("invalid or repeated molecule id")
        candidates = row[1].split(";")
        if not 1 <= len(candidates) <= 25 or any(not x.strip() for x in candidates):
            raise ValueError("invalid candidate list")
        lists[row[0]] = candidates
    return lists


# ---- B: mass-difference plausibility ----
def edit_plausible(delta, prec_a, prec_b):
    tol = max(0.01, 10e-6 * max(prec_a, prec_b))
    delta = abs(delta)
    return delta <= tol or any(abs(delta - edit) <= tol for edit in _NNET_EDITS)


def edge_deltas(spectra, graph, deadline=None, tau=0.6):
    """Precursor difference of each test-test edge in the adduct that gave its best cosine (build_graph rule)."""
    deltas = {}
    for left, neighbours in graph.items():
        for right, best in neighbours:
            key = (min(left, right), max(left, right))
            if key in deltas:
                continue
            _nnet_timecheck(deadline)
            chosen, value = None, None
            for adduct in sorted(set(spectra[key[0]]).intersection(spectra[key[1]])):
                cosine, matches = modified_cosine(spectra[key[0]][adduct], spectra[key[1]][adduct], deadline)
                if matches >= 6 and cosine >= tau and (value is None or cosine > value):
                    chosen, value = adduct, cosine
            if chosen is None or abs(value - best) > 1e-9:
                raise ValueError("edge adduct not recovered")
            pa, pb = spectra[key[0]][chosen][2], spectra[key[1]][chosen][2]
            deltas[key] = (abs(pa - pb), pa, pb)
    return deltas


# ---- A: library of known structures ----
def _nnet_lib_finish(m, w, g, gprec, n_groups):
    """Complete groups only: per (group, 0.01 Da bin) maximum of the normalised intensities below precursor - 1.5 (first
    maximum in file order), top 120 bins by (-max, bin), sqrt, L2 per group. Same steps as the one-shot 1.0.0 loader."""
    import numpy as np
    sel = m < gprec[g] - 1.5
    m, w, g = m[sel], w[sel], g[sel]
    b = np.rint(np.round(m, 2) * 100).astype(np.int64)
    comb = g.astype(np.int64) * 10_000_000 + b
    order = np.lexsort((-w, comb))
    comb_s, w_s = comb[order], w[order]
    first = np.ones(len(comb_s), dtype=bool)
    first[1:] = comb_s[1:] != comb_s[:-1]
    ucomb, umax = comb_s[first], w_s[first]
    ug, ub = ucomb // 10_000_000, ucomb % 10_000_000
    order = np.lexsort((ub, -umax, ug))
    ug, ub, umax = ug[order], ub[order], umax[order]
    start = np.ones(len(ug), dtype=bool)
    start[1:] = ug[1:] != ug[:-1]
    pos = np.arange(len(ug)) - np.maximum.accumulate(np.where(start, np.arange(len(ug)), 0))
    top = pos < 120
    ug, ub, amp = ug[top], ub[top], np.sqrt(umax[top])
    norm = np.sqrt(np.bincount(ug, weights=amp * amp, minlength=n_groups))
    amp = amp / np.maximum(norm[ug], 1e-12)
    return ug, ub, amp


def load_library(path, deadline=None, rows_per_group=_NNET_LIB_ROWS, batch_rows=_NNET_LIB_BATCH):
    """Consensus spectra per (inchikey14, adduct) from train.parquet, vectorised build_spectra rules, in two passes:
    (1) light columns of every row -> group codes, ranks, candidate rows (first rows_per_group with a finite precursor) and the
    last candidate row of each group; (2) peak columns in batches -> per-row checks and normalisation, peaks of kept rows wait
    until their group's last candidate row has been read, then the group is reduced. Same arrays as the one-shot loader."""
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    adducts = pa.array(list(_NNET_ADDUCTS))
    _nnet_memcheck("library start")
    light = pq.read_table(path, columns=["inchikey14", "normalized_smiles", "adduct", "precursor_mz"], use_threads=False)
    _nnet_timecheck(deadline)
    light = light.filter(pc.is_in(light.column("adduct"), value_set=adducts))
    ik = np.asarray(light.column("inchikey14").to_pylist(), dtype=object)
    ad = np.asarray(light.column("adduct").to_pylist(), dtype=object)
    smi = light.column("normalized_smiles")
    prec_row = light.column("precursor_mz").to_numpy(zero_copy_only=False).astype(float)
    ok = np.array([isinstance(x, str) and len(x) > 0 for x in ik])
    key = pd.Series(ik).astype(str) + "|" + pd.Series(ad).astype(str)
    codes, _ = pd.factorize(key, sort=False)
    codes = np.where(ok, codes, -1)
    del key
    rank = pd.Series(codes).groupby(codes).cumcount().to_numpy()
    candidate = (codes >= 0) & (rank < rows_per_group) & np.isfinite(prec_row)
    n_groups = int(codes.max()) + 1 if len(codes) else 0
    last = np.full(n_groups, -1, dtype=np.int64)
    np.maximum.at(last, codes[candidate], np.flatnonzero(candidate))
    gprec = np.full(n_groups, np.nan)
    _nnet_timecheck(deadline)
    pend = dict(m=[], w=[], g=[], pr=[], pg=[])
    pieces = []
    read = 0
    source = pq.ParquetFile(path)
    for batch in source.iter_batches(batch_size=batch_rows, use_threads=False,
                                     columns=["adduct", "ms2_mzs", "ms2_normalized_intensities"]):
        _nnet_timecheck(deadline)
        _nnet_memcheck("library peaks")
        table = pa.Table.from_batches([batch])
        table = table.filter(pc.is_in(table.column("adduct"), value_set=adducts))
        lo, hi = read, read + table.num_rows
        read = hi
        if hi > lo:
            mz_col = table.column("ms2_mzs").combine_chunks()
            it_col = table.column("ms2_normalized_intensities").combine_chunks()
            n_mz = pc.fill_null(pc.list_value_length(mz_col), 0).to_numpy().astype(np.int64)
            n_it = pc.fill_null(pc.list_value_length(it_col), 0).to_numpy().astype(np.int64)
            flat_mz = pc.list_flatten(mz_col).to_numpy(zero_copy_only=False).astype(float)
            flat_it = pc.list_flatten(it_col).to_numpy(zero_copy_only=False).astype(float)
            row_of_peak = np.repeat(np.arange(len(n_mz)), n_mz)
            if len(flat_mz) != len(flat_it) or np.any(n_mz != n_it):
                raise ValueError("library peak arrays misaligned")
            bad = np.zeros(len(n_mz), dtype=bool)
            badpk = ~np.isfinite(flat_mz) | ~np.isfinite(flat_it) | (flat_it < 0)
            bad |= np.bincount(row_of_peak[badpk], minlength=len(n_mz)) > 0
            nonempty = n_mz > 0
            starts = (np.cumsum(n_mz) - n_mz)[nonempty]
            rowmax = np.zeros(len(n_mz))
            if len(starts):
                rowmax[nonempty] = np.maximum.reduceat(flat_it, starts)
            keep_row = candidate[lo:hi] & ~bad
            pk = keep_row[row_of_peak]
            m, w, r = flat_mz[pk], flat_it[pk], row_of_peak[pk]
            w = w / np.maximum(rowmax[r], 1e-9)
            pend["m"].append(m)
            pend["w"].append(w)
            pend["g"].append(codes[lo:hi][r])
            pend["pr"].append(prec_row[lo:hi][keep_row])
            pend["pg"].append(codes[lo:hi][keep_row])
            del mz_col, it_col, flat_mz, flat_it, row_of_peak, badpk, m, w, r
        del table, batch
        if not pend["pg"]:
            continue
        arrays = {k: np.concatenate(v) for k, v in pend.items()}
        done_rows = last[arrays["pg"]] < hi
        if done_rows.any():
            prec_g = pd.Series(arrays["pr"][done_rows]).groupby(arrays["pg"][done_rows]).median()
            gprec[prec_g.index.to_numpy()] = prec_g.to_numpy()
        done_peaks = last[arrays["g"]] < hi
        if done_peaks.any():
            pieces.append(_nnet_lib_finish(arrays["m"][done_peaks], arrays["w"][done_peaks], arrays["g"][done_peaks],
                                           gprec, n_groups))
        for k, mask in (("m", done_peaks), ("w", done_peaks), ("g", done_peaks), ("pr", done_rows), ("pg", done_rows)):
            pend[k] = [arrays[k][~mask]] if (~mask).any() else []
        del arrays
    if read != len(codes) or any(pend.values()):
        raise ValueError("library batches did not cover every row")
    _nnet_timecheck(deadline)
    if pieces:
        ug = np.concatenate([p[0] for p in pieces])
        ub = np.concatenate([p[1] for p in pieces])
        amp = np.concatenate([p[2] for p in pieces])
    else:
        ug, ub, amp = np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0)
    del pieces
    order = np.lexsort((ub, ug))
    ug, ub, amp = ug[order], ub[order], amp[order]
    counts = np.bincount(ug, minlength=n_groups)
    present = np.flatnonzero(counts > 0)
    first_row = pd.Series(np.arange(len(codes))[codes >= 0]).groupby(codes[codes >= 0]).min()
    frow = np.full(n_groups, -1, dtype=np.int64)
    frow[first_row.index.to_numpy()] = first_row.to_numpy()
    smiles_first = smi.take(pa.array(frow[present])).to_pylist()
    library = {}
    structures = {}
    for adduct in _NNET_ADDUCTS:
        mask = np.array([ad[frow[x]] == adduct for x in present], dtype=bool)
        groups = present[mask]
        sizes = counts[groups]
        ptr = np.concatenate([[0], np.cumsum(sizes)])
        take = np.isin(ug, groups)
        library[adduct] = dict(ptr=ptr, bins=ub[take], amp=amp[take], prec=gprec[groups],
                               ik=np.asarray([ik[frow[x]] for x in groups], dtype=object))
    for x, s in zip(present, smiles_first):
        name = ik[frow[x]]
        if name not in structures and isinstance(s, str) and s:
            structures[name] = s
    return library, structures


def _nnet_sparse(rows, cols, vals, shape):
    from scipy import sparse
    return sparse.csr_matrix((vals, (rows, cols)), shape=shape)


def library_raw_edges(spectra, library, deadline=None, tau=0.6, top_m=_NNET_LIB_M, min_matches=6):
    """Exact modified-cosine edges between query spectra and library entries (lossless prefilter, top_m cap)."""
    import numpy as np
    edges = []
    for adduct in _NNET_ADDUCTS:
        lib = library[adduct]
        n = len(lib["prec"])
        queries = sorted(mid for mid in spectra if adduct in spectra[mid])
        if not n or not queries:
            continue
        sizes = np.diff(lib["ptr"])
        lrow = np.repeat(np.arange(n), sizes)
        lfrag = lib["bins"]
        lnl = np.rint((np.repeat(lib["prec"], sizes) - lfrag / 100.0) * 100).astype(np.int64)
        qb = [np.rint(np.asarray(spectra[q][adduct][0]) * 100).astype(np.int64) for q in queries]
        qn = [np.rint((spectra[q][adduct][2] - np.asarray(spectra[q][adduct][0])) * 100).astype(np.int64) for q in queries]
        width = int(max(lfrag.max(initial=0), lnl.max(initial=0), max((x.max(initial=0) for x in qb), default=0),
                        max((x.max(initial=0) for x in qn), default=0))) + 4
        lnl = np.clip(lnl, 0, width - 1)
        LF = _nnet_sparse(lrow, lfrag, lib["amp"], (n, width)).T.tocsr()
        LN = _nnet_sparse(lrow, lnl, lib["amp"], (n, width)).T.tocsr()
        for start in range(0, len(queries), 64):
            _nnet_timecheck(deadline)
            block = queries[start:start + 64]
            rf, cf, vf, rn, cn, vn = [], [], [], [], [], []
            for k, q in enumerate(block):
                a = np.asarray(spectra[q][adduct][1])
                for off in (-1, 0, 1):
                    rf.append(np.full(len(a), k)); cf.append(qb[start + k] + off); vf.append(a)
                for off in (-2, -1, 0, 1, 2):
                    rn.append(np.full(len(a), k)); cn.append(np.clip(qn[start + k] + off, 0, width - 1)); vn.append(a)
            QF = _nnet_sparse(np.concatenate(rf), np.clip(np.concatenate(cf), 0, width - 1), np.concatenate(vf), (len(block), width))
            QN = _nnet_sparse(np.concatenate(rn), np.concatenate(cn), np.concatenate(vn), (len(block), width))
            S = (QF @ LF + QN @ LN).toarray()
            for k, q in enumerate(block):
                row = S[k]
                idx = np.flatnonzero(row >= tau - 1e-9)
                if len(idx) > top_m:
                    idx = idx[np.lexsort((idx, -row[idx]))[:top_m]]
                qspec = spectra[q][adduct]
                for j in sorted(int(x) for x in idx):
                    _nnet_timecheck(deadline)
                    lspec = (lib["bins"][lib["ptr"][j]:lib["ptr"][j + 1]] / 100.0, lib["amp"][lib["ptr"][j]:lib["ptr"][j + 1]], float(lib["prec"][j]))
                    cosine, matches = modified_cosine(qspec, lspec, deadline)
                    if matches >= min_matches and cosine >= tau:
                        edges.append((q, str(lib["ik"][j]), adduct, float(cosine), int(matches), float(qspec[2]), float(lib["prec"][j]),
                                      int(len(idx)), float(row[j])))
    return edges


def select_library_edges(raw, k=_NNET_LIB_K, exclude=frozenset()):
    """Best cosine per (query, structure) over adducts (first sorted adduct on ties), drop excluded, keep top k."""
    best = {}
    for q, ik, adduct, cosine, matches, qprec, lprec, *_ in sorted(raw, key=lambda e: (e[0], e[1], e[2])):
        if ik in exclude:
            continue
        old = best.get((q, ik))
        if old is None or cosine > old[0]:
            best[(q, ik)] = (cosine, abs(qprec - lprec), qprec, lprec)
    per = {}
    for (q, ik), value in best.items():
        per.setdefault(q, []).append((ik,) + value)
    return {q: sorted(v, key=lambda e: (-e[1], e[0]))[:k] for q, v in per.items()}


# ---- generalised network scores ----
def network_scores_net(lists, graph, deadline=None, evidence_lists=None, lib_edges=None, structures=None,
                       deltas=None, mass_on=False):
    import numpy as np
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = {}
    evidence_lists = lists if evidence_lists is None else evidence_lists
    lib_edges = lib_edges or {}

    def fingerprint(smiles):
        if smiles not in fingerprints:
            molecule = Chem.MolFromSmiles(smiles)
            fingerprints[smiles] = generator.GetFingerprint(molecule) if molecule is not None else None
        return fingerprints[smiles]
    result = {}
    for mid, candidates in lists.items():
        _nnet_timecheck(deadline)
        if (not graph.get(mid) and not lib_edges.get(mid)) or len(candidates) < 2:
            continue
        scores = np.zeros(len(candidates), dtype=float)
        terms = []
        for neighbour, cosine in graph.get(mid, []):
            top = evidence_lists.get(neighbour, [])[:5]
            reciprocal = 1.0 / np.arange(1, len(top) + 1)
            if len(top):
                reciprocal /= reciprocal.sum()
            weight = cosine ** 4
            if mass_on:
                delta, pa, pb = deltas[(min(mid, neighbour), max(mid, neighbour))]
                weight = weight * (1.0 if edit_plausible(delta, pa, pb) else _NNET_OFF_EDIT)
            terms.append((weight, [(fingerprint(smiles), w) for smiles, w in zip(top, reciprocal)]))
        for ik, cosine, delta, qprec, lprec in lib_edges.get(mid, []):
            weight = cosine ** 4
            if mass_on:
                weight = weight * (1.0 if edit_plausible(delta, qprec, lprec) else _NNET_OFF_EDIT)
            terms.append((weight, [(fingerprint(structures[ik]), 1.0)]))
        for weight, evidence in terms:
            for i, smiles in enumerate(candidates):
                _nnet_timecheck(deadline)
                candidate = fingerprint(smiles)
                if candidate is not None:
                    scores[i] += weight * sum(w * DataStructs.TanimotoSimilarity(candidate, other)
                                              for other, w in evidence if other is not None)
        result[mid] = scores
    return result


def rerank_net(lists, graph, lam=0.35, steps=1, deadline=None, lib_edges=None, structures=None, deltas=None, mass_on=False):
    kw = dict(lib_edges=lib_edges, structures=structures, deltas=deltas, mass_on=mass_on)
    scores = network_scores_net(lists, graph, deadline, None, **kw)
    output = rerank_lists(lists, scores, lam)
    for _ in range(int(steps) - 1):
        scores = network_scores_net(lists, graph, deadline, output, **kw)
        output = rerank_lists(lists, scores, lam)
    return output


def lock_rank1(base, arm_lists):
    """Keep base[mid][0] at rank 1; the arm order fills ranks 2-25 (its own leader drops to rank 2 when it differs)."""
    output = {}
    for mid, candidates in arm_lists.items():
        candidates = list(candidates)
        if sorted(base[mid]) != sorted(candidates):
            raise ValueError("lock: base and arm lists are not permutations")
        lead = base[mid][0]
        at = candidates.index(lead)
        output[mid] = [lead] + candidates[:at] + candidates[at + 1:]
    return output


def library_stage(spectra, inp, deadline=None):
    from pathlib import Path
    files = sorted(Path(inp).rglob("train.parquet"),
                   key=lambda path: ("mass-spectra" not in str(path).lower(), len(str(path))))
    if not files:
        raise FileNotFoundError("train.parquet absent")
    library, structures = load_library(files[0], deadline)
    _nnet_memcheck("library edges")
    raw = library_raw_edges(spectra, library, deadline)
    lib_edges = select_library_edges(raw)
    return lib_edges, structures, dict(raw_edges=len(raw), library_entries={a: int(len(library[a]["prec"])) for a in library})


def run(work="/kaggle/working", inp="/kaggle/input", pipeline_done=False, enabled=True,
        cap_s=600.0, lam=0.35, arm="base", lock1=True):
    import csv, io, json, os, time, uuid
    from pathlib import Path
    started = time.monotonic()
    audit = dict(version=_NNET_VERSION, arm=arm, lock1=bool(lock1), lambda_value=lam, cap_s=cap_s, status="started")
    target = Path(work) / "submission.csv"
    try:
        lib_on, mass_on, steps = _NNET_ARMS[arm]
        if not enabled or lam == 0:
            audit["status"] = "disabled"
        elif not pipeline_done:
            audit["status"] = "fast commit: original bytes kept"
        else:
            deadline = started + float(cap_s)
            _nnet_timecheck(deadline)
            before = target.read_bytes()
            lists = _nnet_parse_submission(before)
            files = sorted(Path(inp).rglob("test.parquet"),
                           key=lambda path: ("mass-spectra" not in str(path).lower(), len(str(path))))
            if not files:
                raise FileNotFoundError("test.parquet absent")
            import pandas as pd
            test = pd.read_parquet(files[0])
            if not set(lists).issubset(set(test["molecule_id"].astype(str))):
                raise ValueError("spectrum molecule ids missing")
            spectra = build_spectra(test, set(lists), deadline)
            graph = build_graph(spectra, deadline)
            lib_edges, structures = {}, {}
            if lib_on:
                lib_edges, structures, info = library_stage(spectra, inp, deadline)
                audit.update(library=info, with_library_neighbours=len(lib_edges),
                             library_edges=sum(map(len, lib_edges.values())))
            deltas = edge_deltas(spectra, graph, deadline) if mass_on else None
            output = rerank_net(lists, graph, lam, steps, deadline, lib_edges, structures, deltas, mass_on)
            if lock1:
                base_output = rerank_net(lists, graph, lam, 1, deadline)
                audit.update(lock_moved_to_rank2=sum(output[mid][0] != base_output[mid][0] for mid in lists),
                             base_top1_changed=sum(base_output[mid][0] != lists[mid][0] for mid in lists))
                output = lock_rank1(base_output, output)
            changed = sum(output[mid] != lists[mid] for mid in lists)
            audit.update(molecules=len(lists), with_spectra=len(spectra), with_neighbours=len(graph),
                         edges=sum(map(len, graph.values())) // 2, changed_lists=changed,
                         top1_changed=sum(output[mid][0] != lists[mid][0] for mid in lists))
            _nnet_timecheck(deadline)
            if changed:
                stream = io.StringIO(newline="")
                writer = csv.writer(stream, lineterminator="\n")
                writer.writerow(["molecule_id", "smiles"])
                writer.writerows((mid, ";".join(output[mid])) for mid in lists)
                after = stream.getvalue().encode("utf-8")
                verified = _nnet_parse_submission(after)
                if list(verified) != list(lists) or any(sorted(verified[m]) != sorted(lists[m]) for m in lists):
                    raise ValueError("candidate permutation check failed")
                _nnet_timecheck(deadline)
                if target.read_bytes() != before:
                    raise RuntimeError("submission changed concurrently")
                backup = Path(work) / "submission_pre_n035net.csv"
                with backup.open("xb") as handle:
                    handle.write(before)
                temporary = target.with_name("submission.csv.nnet_" + uuid.uuid4().hex)
                with temporary.open("xb") as handle:
                    handle.write(after)
                    handle.flush()
                    os.fsync(handle.fileno())
                _nnet_timecheck(deadline)
                if target.read_bytes() != before:
                    raise RuntimeError("submission changed before commit")
                _nnet_timecheck(deadline)
                os.replace(temporary, target)
            audit["status"] = "applied"
    except Exception as error:
        # No write to submission.csv happens before all checks succeed.
        audit.update(status="failed closed: original bytes kept", exception=repr(error)[:500])
    audit["elapsed_s"] = time.monotonic() - started
    try:
        with (Path(work) / "n035net_audit.json").open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(audit, handle, ensure_ascii=False, indent=2, default=str)
            handle.write("\n")
    except Exception:
        pass
    return audit


# ---- notebook entry: imports expose functions without running this block ----
if globals().get("_NNET_NOTEBOOK", False):
    _nnet_audit = run(work=str(globals().get("ROOT", "/kaggle/working")),
                      inp=str(globals().get("INPUT", "/kaggle/input")),
                      pipeline_done=bool(globals().get("PIPELINE_DONE")),
                      enabled=_NNET_ON, cap_s=_NNET_CAP_S, lam=_NNET_LAM, arm=_NNET_ARM, lock1=_NNET_LOCK1)
    print("[n035net]", _nnet_audit, flush=True)


# ---- envida wrapper -------------------------------------------------------------------------------
ENVIDA_GUARD_H = 8.80  # never let this layer run past this session time


def run_layer(work="/kaggle/working", inp="/kaggle/input", t0_session=None):
    import json as _json
    import os as _os
    import time as _time
    info_p = _os.path.join(work, "frozen_run_info.json")
    full = False
    try:
        info = _json.load(open(info_p))
        full = bool(info.get("is_full_run")) and info.get("processed_molecules") == info.get("output_molecules")
    except Exception:
        full = False
    cap = _NNET_CAP_S
    if t0_session is not None:
        cap = min(cap, ENVIDA_GUARD_H * 3600 - (_time.time() - t0_session) - 120)
    if cap < 300:
        print("[envida-nnet] skipped: %.0f s left before the guard" % cap)
        return dict(status="skipped: time")
    print("[envida-nnet] pipeline full run: %s, cap %.0f s" % (full, cap))
    return run(work=work, inp=inp, pipeline_done=full, enabled=_NNET_ON, cap_s=cap, lam=_NNET_LAM,
               arm=_NNET_ARM, lock1=_NNET_LOCK1)
