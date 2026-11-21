# Rank-1 judge (same-formula isomer re-ranker) on the 0.495 public base

Goal: a genuinely new signal on top of the taeyangg4 0.495 pipeline (top-100 line 0.497 on 2026-11-21).
The public copies of heliosli's rr3 judge ship random demo weights, so we retrain the same architecture with real
data and deploy it with his FREEZE rule. Isomer re-rankers have lost badly on the public board (-0.058, 0.368), so
every step is validated offline before a submission, and deployment changes only non-training candidates on lists
without a strong library match.

## Architecture (heliosli rr3, Apache/CC BY-NC-SA code in heliosli/casmi26-rr-v3)
- Query: frozen fold-safe FPNet token states (`FPNet.encode`, d=512) of each spectrum.
- Candidate: GINE over the heavy-atom graph (7 atom codes, 3 bond codes; `rr_graph_cache.featurize`).
- Link: every (peak, fragment) mass match (<= 2 bond cuts, H shifts, `rr_frag.match_peaks`), attention-pooled per peak.
- Score: transformer over [molecule token, explained peak tokens]; listwise softmax CE over the query's candidates.
- Deploy: `f = z(-log rank) + alpha * z(score)`; FREEZE keeps training-set candidates in their slots;
  lists with `library_max >= 0.9` untouched.

## Jobs
| job | where | what | status |
|---|---|---|---|
| A | Kaggle CPU | graph cache of our pool (train + COCONUT, 725k) with `rr_graph_cache.py` | todo |
| B | local | training spectra table (FPNet items), pool meta, packed fingerprints, identity-disjoint split (`scripts/judge_data.py`: 300,374 spectra, at most 4 per structure/adduct/library, 53,946 fold-0 hold structures, 48,552 train / 5,394 val, pool 725,569) | done (not uploaded) |
| C | Kaggle GPU | training loop (ours) + `rr_model`, `rr_data.build_batch`, `rr_frag` + encoder `fpnet_fold0.pt` | todo |
| D | in C | held-out evaluation on structures that neither the encoder nor the judge saw | todo |
| E | Kaggle | deployment layer on the t495 base (FREEZE, lib_max gate), smoke, then submit | todo |

## Data rules
- Encoder `dmitriigluzdov/casmi26-fold-safe-fpnet` fold 0 (`split.parquet`): the judge trains and validates on
  structures of the encoder's held-out fold, so its token quality matches unseen test molecules.
- Training candidates: truth + 10 hardest same-window pool structures by Tanimoto + 5 random (`train_candidates`).
- Test-set adducts only; NIST-derived data never used.

## Priority note (2026-11-21)
heliosli's README says the real rr3 weights scored 0.494, no better than the public bases. The judge is on hold behind
cheaper, board-verified gains: goodpjw2008's public 0.498 notebook (taeyangg4 0.495 base + UNIQ + overlay checks +
APRIOR adduct-pattern prior) and refinements of APRIOR (`scripts/aprior_study.py`).

## Acceptance (before any submission)
- Held-out isomer top-1 clearly above the FPNet-only score (`f.z`) on the same candidate sets.
- FREEZE replay on held-out lists from our own pipeline: no loss on class 1 (truth in train).

## APRIOR study result (2026-11-21, scripts/aprior_study.py, docs/aprior_study_2026-11-21.json)
Held-out enveda-180 (2,953 lists of hard same-mass decoys, truth at a geometric base rank 2-25, rank 1 kept):
the notebook's lambda 0.5 is the best of {0.25, 0.5, 0.75, 1.0, 1.5} for every model variant (+0.016 reciprocal rank
per affected molecule). Training on all 153k molecules, 600 trees or 7 labels adds at most +0.003, under one molecule
on the public board. On the 250 NP examples the prior is weaker (pair AUC 0.63 vs 0.69) and lambda >= 0.75 loses.
Decision: keep APRIOR as published; no variant submission.
