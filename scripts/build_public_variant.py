"""Build a Kaggle notebook = a public base notebook + our post-processing layers appended as final cells.

    python scripts/build_public_variant.py --base data/ext_nb/<dir>/<nb>.ipynb --meta data/ext_nb/<dir>/kernel-metadata.json \
        --slug envida-hx-extlib --layers extlib_overlay [--commit-mode smoke]

Each layer module from kaggle_layers/ is embedded as text, written to /tmp/envida_layers at run time, imported and
run as `run(t0_session=NB_T0)`; a layer must leave submission.csv unchanged on any failure. The layers' datasets
are added to the kernel metadata. Writes build/kaggle_<slug>/.
"""
import argparse
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USER = "govindmulchandani"
LAYER_DATASETS = {
    "extlib_overlay": ["taeyangg4/casmi26-extlib-gnps-full", "samartalwar/casmi-2026-spectral-library-massbankharmonized"],
}


def layer_cell(name: str) -> str:
    src = (ROOT / "kaggle_layers" / f"{name}.py").read_text(encoding="utf8")
    return (
        f"# ===== envida layer: {name} (kaggle_layers/{name}.py in akkki8844/envida-casmi-kaggle, embedded) =====\n"
        "import importlib, os, sys, time\n"
        "os.makedirs('/tmp/envida_layers', exist_ok=True)\n"
        f"open('/tmp/envida_layers/{name}.py', 'w').write({src!r})\n"
        "sys.path.insert(0, '/tmp/envida_layers')\n"
        f"_envida_{name} = importlib.import_module('{name}')\n"
        f"_envida_{name}_audit = _envida_{name}.run(t0_session=globals().get('NB_T0'))\n"
        f"print('[envida] {name}:', {{k: v for k, v in (_envida_{name}_audit or {{}}).items() if k != 'decisions'}})\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--layers", nargs="+", default=[])
    ap.add_argument("--commit-mode", default=None, help="override the base's COMMIT_MODE ('smoke' or 'fast')")
    a = ap.parse_args()
    nb = json.loads(Path(a.base).read_text(encoding="utf8"))
    if a.commit_mode:
        n = 0
        for c in nb["cells"]:
            if c["cell_type"] != "code":
                continue
            s = "".join(c["source"])
            s2, k = re.subn(r"^COMMIT_MODE = '[a-z]+'", f"COMMIT_MODE = '{a.commit_mode}'", s, flags=re.M)
            n += k
            c["source"] = s2.splitlines(keepends=True)
        assert n == 1, f"COMMIT_MODE assignments found: {n}"
    for c in nb["cells"]:  # outputs of the author's run are not ours to republish
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    for name in a.layers:
        nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                            "source": layer_cell(name).splitlines(keepends=True)})
    meta = json.loads(Path(a.meta).read_text(encoding="utf8"))
    meta.pop("id_no", None)
    meta.update(id=f"{USER}/{a.slug}", title=a.slug.replace("-", " "), code_file="notebook.ipynb", is_private=True,
                keywords=[])
    for name in a.layers:
        for d in LAYER_DATASETS.get(name, []):
            if d not in meta["dataset_sources"]:
                meta["dataset_sources"].append(d)
    # code submissions refuse "non-versioned" datasets: pin every source to its current version
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    pinned = []
    for src in meta["dataset_sources"]:
        parts = src.split("/")
        if len(parts) == 3:
            pinned.append(src)
            continue
        owner, slug = parts
        hits = [d for d in api.dataset_list(search=slug, user=owner) if str(d.ref) == f"{owner}/{slug}"]
        if not hits or not hits[0].current_version_number:
            raise SystemExit(f"cannot resolve the current version of {src}")
        pinned.append(f"{owner}/{slug}/{hits[0].current_version_number}")
    meta["dataset_sources"] = pinned
    print("pinned:", pinned)
    out = ROOT / "build" / f"kaggle_{a.slug}"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    (out / "notebook.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf8")
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf8")
    print(f"built {out}: {len(nb['cells'])} cells, layers {a.layers}, datasets {len(meta['dataset_sources'])}")


if __name__ == "__main__":
    main()
