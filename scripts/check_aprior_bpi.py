"""Local check of scripts/patch_aprior_bpi.py on the visible test molecules with real candidate lists (our v4 output).

Runs ap_run of (1) the original APRIOR cell, (2) the patched cell with W=0, (3) the patched cell with W=0.5, each in a
fresh namespace on a copy of the same submission.csv, and asserts:
  - W=0 output bytes == original output bytes (the patch is inert when switched off),
  - W=0.5 keeps rank 1 and the candidate set of every list, and BPI ran (models trained, molecules scored).
Uses small training (n_train 20k) to keep it quick; the logic is the same as the scoring run.

    python scripts/check_aprior_bpi.py <notebook.ipynb> [--sub data/kernel_out_v4/submission.csv]
"""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from patch_aprior_bpi import MARK, patch_source  # noqa: E402


def run_cell(src, sub, inp, n_train):
    ns = {"__name__": "aprior_check"}
    exec(compile(src, "aprior_cell", "exec"), ns)
    ns["_AP_N_TRAIN"] = n_train
    d = Path(tempfile.mkdtemp(prefix="aprior_"))
    shutil.copy(sub, d / "submission.csv")
    ap_train = ns["ap_train"]
    ns["ap_train"] = lambda p, deadline=None: ap_train(p, n_train=n_train, trees=300, deadline=deadline)
    audit = ns["ap_run"](str(d), str(inp), True, cap_s=3000)
    return (d / "submission.csv").read_bytes(), audit


def lists(b):
    rows = b.decode().splitlines()[1:]
    return {r.split(",", 1)[0]: r.split(",", 1)[1].strip('"').split(";") for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("--sub", default="data/kernel_out_v4/submission.csv")
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--inp", default="build/aprior_inp", help="folder holding only train.parquet and test.parquet")
    a = ap.parse_args()
    nb = json.loads(Path(a.notebook).read_text(encoding="utf8"))
    orig = "".join(next(c["source"] for c in nb["cells"] if "".join(c["source"]).startswith(MARK)))
    inp = Path(a.inp)   # the cell looks train.parquet / test.parquet up by name under this root (rglob)
    b0, a0 = run_cell(orig, ROOT / a.sub, inp, a.n_train)
    print("original:", json.dumps({k: v for k, v in a0.items() if k != "label_rates"}))
    b1, a1 = run_cell(patch_source(orig, 0.0), ROOT / a.sub, inp, a.n_train)
    print("patched W=0:", json.dumps({k: v for k, v in a1.items() if k != "label_rates"}))
    assert a0["status"] == "applied", a0
    assert b1 == b0, "W=0 output differs from the original cell"
    b2, a2 = run_cell(patch_source(orig, 0.5), ROOT / a.sub, inp, a.n_train)
    print("patched W=0.5:", json.dumps({k: v for k, v in a2.items() if k != "label_rates"}))
    assert a2["status"] == "applied", a2
    assert a2["bpi"]["models"] == ["[M+H]+", "[M-H]-"] and a2["bpi"]["molecules"] > 300 and not a2["bpi"]["error"], a2["bpi"]
    L0, L2 = lists(b0), lists(b2)
    assert list(L0) == list(L2)
    assert all(L0[m][0] == L2[m][0] and sorted(L0[m]) == sorted(L2[m]) for m in L0)
    diff = sum(L0[m] != L2[m] for m in L0)
    print(f"OK: W=0 identical to original; W=0.5 keeps rank 1 and candidate sets; {diff}/{len(L0)} lists ordered differently")


if __name__ == "__main__":
    main()
