"""Build envida-om500-finish: the verbatim 0.500 notebook (omerfarukmerey) + kozykappa's "lossless finishing" cells
(settings + snapshot helper, snapshots before P5 and before FR, finishing after UNIQ), with the ORIGINAL APRIOR cell
(kozykappa's unscored APRIOR v2 is left out). The figure cell (reads nothing, writes nothing; kozykappa drops it too) is
removed to stay under Kaggle's 1 MiB kernel source limit. Optionally the APRIOR cell gets our BPI patch.

    python scripts/build_om500_finish.py [--bpi 0.5] [--slug envida-om500-finish]
"""
import argparse
import glob
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from patch_aprior_bpi import MARK, patch_source  # noqa: E402

KOZY = "data/ext_nb/kozykappa_casmi26-0-500-stack-aprior-v2"


def src(c):
    return "".join(c["source"])


def code_cell(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.splitlines(keepends=True)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bpi", type=float, default=None)
    ap.add_argument("--slug", default="envida-om500-finish")
    a = ap.parse_args()
    base_dir = ROOT / "build/kaggle_envida-om500-verbatim"
    nb = json.loads((base_dir / "notebook.ipynb").read_text(encoding="utf8"))
    k = json.loads(Path(glob.glob(str(ROOT / KOZY / "*.ipynb"))[0]).read_text(encoding="utf8"))
    kc = [src(c) for c in k["cells"] if c["cell_type"] == "code"]
    one = lambda pred: [s for s in kc if pred(s)]  # noqa: E731
    settings = one(lambda s: s.startswith("# ===== ours: answer snapshots + settings"))
    pre_p5 = one(lambda s: s.startswith("snap('pre_p5')"))
    pre_fr = one(lambda s: s.startswith("snap('pre_fr')"))
    finish = one(lambda s: s.startswith("# ===== ours · lossless finishing"))
    assert all(len(x) == 1 for x in (settings, pre_p5, pre_fr, finish))

    cells = nb["cells"]
    find = lambda prefix: [i for i, c in enumerate(cells) if c["cell_type"] == "code" and src(c).startswith(prefix)]  # noqa: E731
    fig = find("# ── Figure: what this notebook adds")
    assert len(fig) == 1
    del cells[fig[0]]
    # insertion points (located after the deletion), applied from the bottom up
    i_uniq = find("_UNIQ_NOTEBOOK = True"); i_fr = find("# ===== Arsokan fp5"); i_p5 = find("# ── H-A-130 s5: P5 tail fix")
    i_sess = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and "Session: paths, run mode" in src(c)[:120]]
    assert all(len(x) == 1 for x in (i_uniq, i_fr, i_p5, i_sess)), (i_uniq, i_fr, i_p5, i_sess)
    assert i_sess[0] < i_p5[0] < i_fr[0] < i_uniq[0]
    cells.insert(i_uniq[0] + 1, code_cell(finish[0]))
    cells.insert(i_fr[0], code_cell(pre_fr[0]))
    cells.insert(i_p5[0], code_cell(pre_p5[0]))
    cells.insert(i_sess[0] + 1, code_cell(settings[0]))
    if a.bpi is not None:
        ap_i = find(MARK)
        assert len(ap_i) == 1
        cells[ap_i[0]]["source"] = patch_source(src(cells[ap_i[0]]), a.bpi).splitlines(keepends=True)
    cells.insert(0, {"cell_type": "markdown", "metadata": {}, "source": [
        "**envida variant.** omerfarukmerey's public 0.500 notebook (code cells byte for byte) plus kozykappa's lossless "
        "finishing cells (casmi26-0-500-stack-aprior-v2; its APRIOR v2 is not used)"
        + (f", with a base-peak-intensity term in APRIOR (W {a.bpi})" if a.bpi is not None else "")
        + ". The figure cell is removed (it reads no input and writes no file).\n"]})
    out = ROOT / "build" / f"kaggle_{a.slug}"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    meta = json.loads((base_dir / "kernel-metadata.json").read_text(encoding="utf8"))
    meta.update(id=f"govindmulchandani/{a.slug}", title=a.slug.replace("-", " "))
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf8")
    (out / "notebook.ipynb").write_text(json.dumps(nb, separators=(",", ":")), encoding="utf8")
    size = (out / "notebook.ipynb").stat().st_size
    assert size < 1048576, size
    print(f"built {out}: {len(cells)} cells, {size:,} bytes")


if __name__ == "__main__":
    main()
