import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle_layers"))
import extlib_overlay as X  # noqa: E402

SPEC_A = (np.array([91.05, 119.05, 147.04, 165.05, 203.07]), np.array([0.3, 1.0, 0.5, 0.2, 0.4]))
SPEC_B = (np.array([77.04, 105.03, 133.06, 151.07, 189.09]), np.array([1.0, 0.4, 0.6, 0.3, 0.2]))
ETHYL_BENZOATE = "CCOC(=O)c1ccccc1"          # unique hit for molecule u
ISOMERS = ["Cc1ccc(C(=O)OC)cc1", "COC(=O)Cc1ccccc1", "Cc1ccccc1C(=O)OC"]   # one pooled spectrum, three isomers


def _lib(tmp, rows):
    """rows: (smiles, precursor_mz, spectrum, source_library)"""
    d = tmp / "inp" / "extlib"
    d.mkdir(parents=True)
    mz = np.zeros((len(rows), 128), np.float32)
    it = np.zeros((len(rows), 128), np.float32)
    meta = []
    for i, (smi, prec, (m, v), src) in enumerate(rows):
        mz[i, : len(m)] = m
        it[i, : len(m)] = np.sqrt(v / v.max())
        meta.append(dict(inchikey14=X._key14(smi), normalized_smiles=smi, adduct="[M+H]+", ionization_mode="positive",
                         instrument_type="x", precursor_mz=prec, source_library=src, source_id=str(i), licence="CC0"))
    pd.DataFrame(meta).to_parquet(d / "ext_meta.parquet")
    np.save(d / "ext_mz.npy", mz)
    np.save(d / "ext_in.npy", it)


def _case(tmp, queries, answers, lib_rows, lib_max=None):
    _lib(tmp, lib_rows)
    (tmp / "inp" / "comp").mkdir()
    pd.DataFrame([dict(molecule_id=mid, spectrum_id=mid, ms2_mzs=m, ms2_normalized_intensities=v, precursor_mz=p,
                       ionization_mode="positive", adduct="[M+H]+") for mid, p, (m, v) in queries]
                 ).to_parquet(tmp / "inp" / "comp" / "test.parquet")
    (tmp / "work").mkdir()
    pd.DataFrame(list(answers.items()), columns=["molecule_id", "smiles"]).to_csv(tmp / "work" / "submission.csv",
                                                                                index=False)
    if lib_max:
        pd.DataFrame([dict(molecule_id=k, library_max=v) for k, v in lib_max.items()]
                     ).to_parquet(tmp / "work" / "frozen_candidate_evidence.parquet")
    audit = X.run(str(tmp / "work"), str(tmp / "inp"), t0_session=time.time())
    out = pd.read_csv(tmp / "work" / "submission.csv", dtype=str, keep_default_na=False)
    return audit, dict(zip(out.molecule_id, out.smiles.str.split(";")))


def test_unique_hit_inserted_and_nist_rows_ignored(tmp_path):
    lib = [(ETHYL_BENZOATE, 220.0, SPEC_A, "GNPS-LIBRARY"),
           ("CC(C)Oc1ccccc1", 220.0, SPEC_B, "gnps2:GNPS-NIST14-MATCHES")]
    audit, out = _case(tmp_path, [("u", 220.0, SPEC_A), ("n", 220.0, SPEC_B)],
                       {"u": "CCO;CCCO", "n": "CCO;CCCO"}, lib)
    assert audit["nist_dropped"] == 1
    assert X._key14(out["u"][0]) == X._key14(ETHYL_BENZOATE) and out["u"][1:] == ["CCO", "CCCO"]
    assert out["n"] == ["CCO", "CCCO"]                       # only a NIST-labelled reference matched: no change


def test_tie_is_broken_by_the_list_and_ambiguous_ties_skipped(tmp_path):
    lib = [(s, 220.0, SPEC_A, "MULTIPLEX-SYNTHESIS-LIBRARY-ALL-PARTITION-1") for s in ISOMERS]
    ans = {"t": f"CCO;{ISOMERS[2]};{ISOMERS[1]};CCCO", "a": "CCO;CCCO"}
    audit, out = _case(tmp_path, [("t", 220.0, SPEC_A), ("a", 220.0, SPEC_A)], ans, lib)
    assert out["t"] == [ISOMERS[2], "CCO", ISOMERS[1], "CCCO"]   # best-ranked tied member promoted
    assert out["a"] == ["CCO", "CCCO"]                           # three-way tie, none in list: untouched
    assert audit["ambiguous_skipped"] == 1


def test_library_match_protects_the_list(tmp_path):
    lib = [(ETHYL_BENZOATE, 220.0, SPEC_A, "GNPS-LIBRARY")]
    _, out = _case(tmp_path, [("p", 220.0, SPEC_A)], {"p": "CCO;CCCO"}, lib, lib_max={"p": 0.95})
    assert out["p"] == ["CCO", "CCCO"]


def test_failure_restores_submission(tmp_path, monkeypatch):
    lib = [(ETHYL_BENZOATE, 220.0, SPEC_A, "GNPS-LIBRARY")]
    def boom(*a, **k):
        raise RuntimeError("x")
    monkeypatch.setattr(X, "load_refs", boom)       # a global failure, not a per-molecule one
    _lib(tmp_path, lib)
    (tmp_path / "inp" / "comp").mkdir()
    pd.DataFrame([dict(molecule_id="f", spectrum_id="f", ms2_mzs=SPEC_A[0], ms2_normalized_intensities=SPEC_A[1],
                       precursor_mz=220.0, ionization_mode="positive", adduct="[M+H]+")]
                 ).to_parquet(tmp_path / "inp" / "comp" / "test.parquet")
    (tmp_path / "work").mkdir()
    raw = b"molecule_id,smiles\nf,CCO;CCCO\n"
    (tmp_path / "work" / "submission.csv").write_bytes(raw)
    X.run(str(tmp_path / "work"), str(tmp_path / "inp"), t0_session=time.time())
    assert (tmp_path / "work" / "submission.csv").read_bytes() == raw
