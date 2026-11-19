"""Parallel ranged download of one competition file (no secrets are printed).

    python scripts/fast_download.py train.parquet [--workers 16] [--out data]

Uses the Kaggle client for authentication to obtain the signed storage URL, then fetches byte ranges in parallel.
Requires that you already accepted the competition rules and have working Kaggle credentials.
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_api_service import ApiDownloadDataFileRequest

COMP = "enveda-CASMI26-molecule-id-mass-spectra"
CHUNK = 32 * 1024 * 1024

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("--workers", type=int, default=16)
ap.add_argument("--out", default="data")
a = ap.parse_args()

api = KaggleApi()
api.authenticate()
with api.build_kaggle_client() as kaggle:
    req = ApiDownloadDataFileRequest()
    req.competition_name = COMP
    req.file_name = a.name
    resp = kaggle.competitions.competition_api_client.download_data_file(req)
url = resp.request.url
resp.close()

probe = requests.get(url, headers={"Range": "bytes=0-0"}, timeout=60)
if probe.status_code not in (200, 206) or "Content-Range" not in probe.headers:
    sys.exit(f"server did not honour range requests (status {probe.status_code})")
total = int(probe.headers["Content-Range"].split("/")[1])
print(f"{a.name}: {total / 1e6:,.0f} MB, {a.workers} workers")

out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
dest = out / a.name
with open(dest, "wb") as f:
    f.truncate(total)

done = [0]
t0 = time.time()


def fetch(start: int) -> None:
    end = min(start + CHUNK, total) - 1
    for attempt in range(6):
        try:
            r = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
            r.raise_for_status()
            if len(r.content) != end - start + 1:
                raise IOError("short read")
            with open(dest, "r+b") as f:
                f.seek(start)
                f.write(r.content)
            done[0] += len(r.content)
            print(f"{done[0] / total * 100:5.1f}%  {done[0] / 1e6 / (time.time() - t0):5.2f} MB/s", flush=True)
            return
        except Exception:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"chunk at {start} failed")


with ThreadPoolExecutor(max_workers=a.workers) as ex:
    list(ex.map(fetch, range(0, total, CHUNK)))
print(f"saved {dest} ({os.path.getsize(dest):,} bytes)")
