"""Compare a local folder (or our Kaggle copy) with a Kaggle dataset's file list: names and sizes must match.

    python scripts/verify_dataset_copy.py heliosli/casmi26-rr-v3 data/hcopy/casmi26-rr-v3
    python scripts/verify_dataset_copy.py heliosli/casmi26-rr-v3 kaggle:govindmulchandani/casmi26-rr-v3
"""
import os
import sys
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi


def remote(api, ref):
    out, token = {}, None
    while True:
        r = api.dataset_list_files(ref, page_token=token, page_size=200)
        for f in r.files:
            out[str(f.name).replace("\\", "/")] = int(f.total_bytes)
        token = getattr(r, "next_page_token", None) or getattr(r, "nextPageToken", None)
        if not token:
            return out


def local(root):
    root = Path(root)
    return {p.relative_to(root).as_posix(): p.stat().st_size for p in root.rglob("*")
            if p.is_file() and p.name != "dataset-metadata.json"}


def main():
    src, dst = sys.argv[1], sys.argv[2]
    api = KaggleApi()
    api.authenticate()
    a = remote(api, src)
    b = remote(api, dst[len("kaggle:"):]) if dst.startswith("kaggle:") else local(dst)
    missing = sorted(set(a) - set(b))
    extra = sorted(set(b) - set(a))
    size = sorted(k for k in set(a) & set(b) if a[k] != b[k])
    print(f"{src}: {len(a)} files; {dst}: {len(b)} files; missing {len(missing)}, extra {len(extra)}, "
          f"size mismatch {len(size)}")
    for k in missing[:10]:
        print("  missing", k)
    for k in extra[:10]:
        print("  extra", k)
    for k in size[:10]:
        print("  size", k, a[k], b[k])
    sys.exit(0 if not (missing or extra or size) else 1)


if __name__ == "__main__":
    main()
