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

**Public LB: v1 = 0.140.** The table above overstated it because the v1 harness kept class-2 truths in the pool;
a real class-2 molecule has no public spectra, so it is never a train structure.

## v2: COCONUT pool + realistic harness

- **Candidate pool** = train structures + COCONUT 2026-11 (CC0; `scripts/build_coconut.py`: largest fragment, stereo removed,
  deduplicated on InChIKey14) = 725,569 structures.
- **Harness fix**: a class-2 truth stays in the pool only if the external database has it. Under this rule v1 scores
  ~0.16 on the np panel, close to its real 0.140, so the harness now tracks the leaderboard much better.
- **Known-structure penalty** (`spec_penalty`): a candidate with visible library spectra that do not match the query is
  unlikely to be the answer; its analog score is multiplied by the penalty.

| config | np c1 | np c2 | np mixed | public LB |
|---|---|---|---|---|
| v1 (train pool) | ~0.82 | 0 | ~0.16 | 0.140 |
| v2 (+COCONUT, penalty 1.0) | 0.844 | 0.436 | 0.343 | **0.218** |
| v2b (+COCONUT, penalty 0.3) | 0.782 | 0.474 | 0.346 | 0.209 |

98% of np-panel truths are in COCONUT. The e180 panel scores 0 under the realistic rule: its synthetic compounds
would need PubChem. The penalty trades class 1 for class 2; on the LB the gap (0.009) is inside the noise (~0.016).

## v3: learned ranker (LightGBM LambdaRank)

- `BaselineRanker.features` gives 28 per-candidate features: the v2 heuristic score and rank, analog-similarity ×
  Tanimoto aggregates over the top 100 library anchors, the candidate's own spectral match, has_spec / in_train /
  in_db, ppm error, fingerprint density and query-level context.
- Training queries (`make_train_items`, `scripts/train_ranker.py`): 1,500 simulated class 1/2 queries on structures
  disjoint from the eval panels, preferring enveda-180 query spectra. Features are built in ~1 h.
- **Leak fixed**: a simulated class-2 truth kept `in_train=1` with all spectra hidden, which never happens on the
  test set (held-out MRR 0.90, fake). Hidden structures now look database-only (held-out MRR 0.70 vs heuristic 0.59).
- Class-1 groups are repeated ×3 in training (v3b), since the plain model under-trusted direct library hits.
  Rank blends with the heuristic did not help (`scripts/blend_eval.py`).

| config | np c1 | np c2 | np mixed | public LB |
|---|---|---|---|---|
| v2 heuristic | 0.844 | 0.423 | 0.338 | 0.218 |
| v3 LambdaRank | 0.602 | 0.618 | 0.368 | **0.235** |
| v3b LambdaRank, class 1 ×3 | 0.733 | 0.606 | 0.389 | pending |

Kaggle runtime ~21-26 min (library vectorisation ~9 min, ranking 400 molecules ~5-10 min).

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

- A gated PubChem tier for class-2 molecules that COCONUT does not cover.
- More ranker features: MetFrag-style fragment explanation, ICEBERG/CFM-ID simulated-spectrum similarity.
- More training queries (features take ~2.3 s/query) and class-3-style groups.
- A spectrum-to-fingerprint or contrastive model trained on enveda-180 + public libraries.
