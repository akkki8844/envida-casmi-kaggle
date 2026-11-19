"""Assemble the Kaggle dataset (code + RDKit wheels) and the submission notebook.

    python scripts/build_kaggle.py --user <kaggle_username> [--version v1]

Writes build/kaggle_dataset/ and build/kaggle_kernel/. Then:
    kaggle datasets create -p build/kaggle_dataset            (first time; private)
    kaggle datasets version -p build/kaggle_dataset -m "..."  (updates)
    kaggle kernels push -p build/kaggle_kernel
"""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMP = "enveda-CASMI26-molecule-id-mass-spectra"

ap = argparse.ArgumentParser()
ap.add_argument("--user", required=True)
ap.add_argument("--version", default="v1")
ap.add_argument("--wheels", default="data/kaggle_ds/wheels")
ap.add_argument("--dataset-version", default="v1", help="code dataset slug suffix to attach")
ap.add_argument("--power", type=float, default=3.0)
ap.add_argument("--ppm", type=float, default=10.0)
ap.add_argument("--spec-penalty", type=float, default=1.0)
ap.add_argument("--coconut", default="", help="local COCONUT pool parquet: shipped as a dataset, used as extra pool")
a = ap.parse_args()

ds_slug = f"envida-casmi-{a.dataset_version}-code"
wh_slug = "envida-rdkit-2026-3-3-wheels"
cn_slug = "envida-coconut-2026-11-pool"
kn_slug = f"envida-casmi-{a.version}"
build = ROOT / "build"
ds, wh, kn, cn = build / "kaggle_dataset", build / "kaggle_wheels", build / "kaggle_kernel", build / "kaggle_coconut"
for d in (ds, wh, kn, cn):
    shutil.rmtree(d, ignore_errors=True)
(ds / "casmi").mkdir(parents=True)
wh.mkdir(parents=True)
kn.mkdir(parents=True)
if a.coconut:
    cn.mkdir(parents=True)
    shutil.copy(ROOT / a.coconut, cn / "coconut_pool.parquet")
    (cn / "dataset-metadata.json").write_text(json.dumps({
        "title": "envida coconut 2026-11 pool",
        "id": f"{a.user}/{cn_slug}",
        "licenses": [{"name": "CC0-1.0"}],
    }, indent=2))

for f in (ROOT / "casmi").glob("*.py"):
    shutil.copy(f, ds / "casmi" / f.name)
(ds / "dataset-metadata.json").write_text(json.dumps({
    "title": f"envida casmi {a.dataset_version} code",
    "id": f"{a.user}/{ds_slug}",
    "licenses": [{"name": "CC0-1.0"}],
}, indent=2))
for w in (ROOT / a.wheels).glob("rdkit-*.whl"):
    shutil.copy(w, wh / w.name)
(wh / "dataset-metadata.json").write_text(json.dumps({
    "title": "envida rdkit 2026.3.3 wheels",
    "id": f"{a.user}/{wh_slug}",
    "licenses": [{"name": "other"}],
}, indent=2))

CELLS = [
    # 1. offline install of the pinned RDKit (the metric pins 2026.03.3)
    r'''import glob, os, subprocess, sys, time
from importlib import metadata
T0 = time.time()
tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
try:
    have = metadata.version("rdkit")
except metadata.PackageNotFoundError:
    have = None
print("python", sys.version.split()[0], "rdkit installed:", have)
if have != "2026.3.3":
    whl = glob.glob(f"/kaggle/input/**/rdkit-2026.3.3-{tag}-*.whl", recursive=True)
    if whl:
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--force-reinstall", "-q", whl[0]], check=True)
    else:
        print("WARNING: no matching RDKit wheel for", tag)
import rdkit
print("rdkit", rdkit.__version__)''',
    # 2. locate code and data
    r'''import shutil, zipfile
def find_code(root):
    hits = [os.path.dirname(p) for p in glob.glob(f"{root}/**/baseline.py", recursive=True)
            if os.path.isfile(os.path.join(os.path.dirname(p), "chem.py"))]
    return hits[0] if hits else None
src = find_code("/kaggle/input")
if src is None:  # the dataset may hold the package as casmi.zip
    zipfile.ZipFile(glob.glob("/kaggle/input/**/casmi.zip", recursive=True)[0]).extractall("/tmp/casmi_zip")
    src = find_code("/tmp/casmi_zip")
pkg = "/tmp/casmi_pkg/casmi"  # copy into a folder literally named casmi so `import casmi` works
shutil.rmtree("/tmp/casmi_pkg", ignore_errors=True)
shutil.copytree(src, pkg)
sys.path.insert(0, "/tmp/casmi_pkg")
code_dirs = [src]
train_path = glob.glob("/kaggle/input/**/train.parquet", recursive=True)[0]
test_path = glob.glob("/kaggle/input/**/test.parquet", recursive=True)[0]
print(code_dirs[0], train_path, test_path, sep="\n")''',
    # 3. build pool + library (cached in /tmp so it is not saved as notebook output)
    r'''from pathlib import Path
from casmi.library import build_library, build_pool, load_train_meta
cache = Path("/tmp/casmi_cache")
meta = load_train_meta(Path(train_path))
extra = glob.glob("/kaggle/input/**/coconut_pool.parquet", recursive=True) if USE_COCONUT else []
print("extra pool:", extra)
pool = build_pool(meta, cache, extra=Path(extra[0]) if extra else None)
lib = build_library(Path(train_path), meta, pool, cache)
del meta
print(f"pool {len(pool.df):,}  library {len(lib.meta):,}  elapsed {time.time()-T0:.0f}s")''',
    # 4. rank and write submission
    r'''import pandas as pd
from casmi.baseline import BaselineRanker, Params
from casmi.predict import predict
ranker = BaselineRanker(lib, pool, Params(power=POWER, ppm=PPM, spec_penalty=SPEC_PENALTY))
test = pd.read_parquet(test_path)
print(f"test: {len(test)} spectra, {test.molecule_id.nunique()} molecules")
sub = predict(test, ranker)
sub.to_csv("/kaggle/working/submission.csv", index=False)
print(sub.head(3))
print(f"total elapsed {time.time()-T0:.0f}s")''',
]
CELLS[0] = (f"POWER, PPM, SPEC_PENALTY = {a.power}, {a.ppm}, {a.spec_penalty}\n"
            f"USE_COCONUT = {bool(a.coconut)}\n") + CELLS[0]
nb = {
    "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
               "source": c.splitlines(keepends=True)} for c in CELLS],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 5,
}
(kn / "notebook.ipynb").write_text(json.dumps(nb, indent=1))
(kn / "kernel-metadata.json").write_text(json.dumps({
    "id": f"{a.user}/{kn_slug}",
    "title": f"envida casmi {a.version}",
    "code_file": "notebook.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": False,
    "enable_internet": False,
    "dataset_sources": [f"{a.user}/{ds_slug}", f"{a.user}/{wh_slug}"] + ([f"{a.user}/{cn_slug}"] if a.coconut else []),
    "competition_sources": [COMP],
    "kernel_sources": [],
}, indent=2))
print(f"built {ds} and {kn}")
