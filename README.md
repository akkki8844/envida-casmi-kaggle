# envida-casmi-kaggle

Code and notes for the Kaggle competition
[Enveda CASMI 2026 - Molecule ID From Mass Spectra](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra).

- [CASMI26_research_report.md](CASMI26_research_report.md): rules, leaderboard picture, what top approaches do, validation advice, traps, plan.

No competition data (CC BY-NC 4.0), model weights or credentials are stored in this repo (`data/` and `build/` are git-ignored).

## v1: retrieval baseline

For each test molecule:

1. **Neutral mass** = median over its spectra of `precursor_mz - adduct_delta` (10 test adducts, checked against train labels: median error ~0 ppm).
2. **Spectral library** = train.parquet, quality-filtered (|precursor error| <= 20 ppm, test adducts only) and thinned to <= 4 spectra per (structure, library): 1.08M spectra.
   Each spectrum becomes two sparse vectors (0.02 Da bins, top-64 peaks, sqrt intensity, L2): fragment peaks and neutral losses.
3. **Anchors**: library spectra of the same ion mode within +-200 Da; similarity = max(peak cosine, neutral-loss cosine);
   per structure keep the best hit per query spectrum and average over the molecule's spectra.
4. **Candidates**: train structures (275k) within 10 ppm of the neutral mass (widened if fewer than 25).
5. **Score** = max over the top-100 anchors of `sim^3 x Tanimoto(candidate, anchor)` (Morgan r2, 2048 bits); an exact library match has Tanimoto 1.
   Output is deduplicated by the official key (RDKit tautomer canonicalisation, InChIKey14).

### Validation harness (`casmi/validation.py`)

Structure-disjoint, simulating the three novelty classes by removal:

- class 1: the truth's timsTOF spectra hidden, its public-library spectra visible
- class 2: all of the truth's spectra hidden, structure still in the pool
- class 3: the truth, its tautomers and every structure with Tanimoto >= 0.7 hidden from library and pool

Two panels: `np` (enveda-np-examples, timsTOF natural products) and `e180` (enveda-180).

v1 results (60 molecules per class, seed 1; MRR@25):

| power | np c1 | np c2 | np c3 | e180 c2 | e180 c3 | np mixed (0.2/0.4/0.4) | e180 mixed |
|---|---|---|---|---|---|---|---|
| 2 | 0.849 | 0.521 | 0 | 0.703 | 0 | 0.378 | 0.352 |
| **3** | 0.821 | 0.566 | 0 | 0.719 | 0 | 0.391 | 0.360 |
| 4 | 0.813 | 0.593 | 0 | 0.695 | 0 | 0.400 | 0.347 |

Differences between powers are inside the bootstrap CIs (about +-0.04). Forum reports say local panels overstate the leaderboard a lot, so treat these as relative numbers only.
Class 3 scores 0 by construction: v1 has no de novo channel. The pool is train structures only (no COCONUT/PubChem yet).

## Usage

```bash
uv venv --python 3.11 .venv && uv pip install -e .       # or install the deps in pyproject.toml
python scripts/fast_download.py train.parquet --out data  # needs Kaggle credentials + accepted rules
python scripts/prepare.py                                 # builds data/cache (~15 min)
python scripts/evaluate.py --n-per-class 60 --power 2 3 4 # validation harness
python scripts/make_submission.py --limit 20              # smoke test on the placeholder test set
python -m pytest -q
```

Kaggle: `python scripts/build_kaggle.py --user <you> --version v1` builds a private code dataset, an RDKit wheel dataset
and the submission notebook in `build/`; upload with `kaggle datasets create` and `kaggle kernels push`.

## Next ideas

- Add COCONUT (and a gated PubChem tier) to the candidate pool for class 2 coverage.
- A learned ranker (LightGBM LambdaRank) over library, analog, fragmentation and fingerprint features.
- A spectrum-to-fingerprint or contrastive model trained on enveda-180 + public libraries.
