# Enveda CASMI 2026 (Kaggle): research report

Compiled 2026-11-19 from the live Kaggle pages (read in Chrome, not signed in), the forum, ~20 GitHub repos, and the literature.
Tags: **[V]** read on an official Kaggle page. **[C]** claim from a competitor, unverified. **[U]** could not be verified.

## 1. Facts that matter

| Item | Value |
|---|---|
| Format | Code competition. Notebook submit, 9 h CPU or GPU, **no internet**, output `submission.csv` [V] |
| Deadlines | Entry and team merge **Dec 7**, final submission **Dec 14**, 23:59 UTC [V] |
| Limits | 5 submissions/day, 2 final picks, teams up to 5 [V] |
| Prizes | $50k: 16k / 12k / 9k / 7k / 6k [V] |
| Metric | MRR@25 per molecule. RDKit 2026.03.3 tautomer canonicalisation, then InChIKey14 match. Stereo and tautomers don't matter [V] |
| Submission | `molecule_id,smiles`, up to 25 SMILES joined by `;`, best first [V] |
| Test set | ~400 molecules, ~1,500 timsTOF spectra (press release says ~2,500), 1-16 spectra/molecule, 157-1,159 Da [V] |
| Test adducts | 10 types: [M+H]+, [M+NH4]+, [M-H2O+H]+, [M-2H2O+H]+, [M+Na]+, [M+K]+, [M-H]-, [M-H2O-H]-, [M+CH2O2-H]-, [M+Cl]- [V] |
| Test columns | No formula or structure. `precursor_mz` and `adduct` are given, so the neutral mass is derivable [V] |
| Train | ~2.5M spectra, ~275k structures. Only `enveda-180` (synthetic drug-like) and `enveda-np-examples` (250 natural products) are timsTOF [V] |
| Public LB | Only ~31% of test (~130 molecules). Private is the other 69%, so a shake-up is likely [V] |

**Classes (hidden mix).** 1: structure has public reference spectra. 2: structure is in PubChem or COCONUT, no public spectra. 3: not in PubChem, needs de novo [V].
Competitor estimates of the mix disagree: ~16-28% class 1, 20-55% class 2, 30-52% class 3 [C].

**Rules.**
- Public open-source external data is allowed, including pretrained models and non-commercial open data (host post, Oct 17) [V].
- **NIST is banned**, including derivatives and weights. Anything trained on NIST gets you removed (hosts) [V].
- Host-confirmed OK in forum threads: MassSpecGym and MIST weights, ICEBERG/GLACIER, DreaMS, COCONUT, PubChem, ChEBI, LIPID MAPS, CFM-ID spectra.
- Winners must deliver training and inference code under an OSI licence (MIT named) and be reproducible from open data [V].
- No private code sharing outside your team. Public sharing only through Kaggle [V].

## 2. Where scores are

- **Public LB:** #1 0.552, #5 0.524, #10 0.512, #25 0.503, #49 0.499. Rank ~684 of ~5,500 is about 0.492 ("top 10%") [C].
- **Public notebooks** reach 0.48-0.49 (`zz3t3rn41/casmi26-peaks-to-structures-v2`, `prvsiyan/analog-propagation-casmi-2026-baseline`, `alexchilton/casmi26-lb-0-488-...`). That is a heavily forked family. The 0.43+ lineage was never CV-validated [C].
- **From-scratch builds** land at 0.05-0.35. Plain library cosine scores 0.09.
- **Noise:** one public-LB molecule is about 0.008 MRR. Differences under ~0.015-0.02 are noise [C].

## 3. What the top approaches do

Everything near the top is **candidate retrieval plus ranking**. Nobody is winning with de novo generation.

1. **Candidate pool:** train structures plus COCONUT (~712k), filtered by neutral mass (1-10 ppm, per-adduct offsets), using the median mass across a molecule's spectra.
2. **Evidence channels:**
   - library entropy similarity
   - mass-shifted analog propagation (±CH2, ±OH, hexose; sim³ × Tanimoto)
   - spectrum-to-fingerprint network (FPNet)
   - in-silico fragmentation (MetFrag-lite, bond-break counts)
   - ICEBERG/GLACIER simulated spectra
   - DreaMS embeddings
3. **Ranker:** LightGBM LambdaRank or HistGBM, ~30 features, multiple seeds. "The ranker is the bottleneck", not coverage.
4. **Where MRR is lost:** ~98% of wrong winners share the true formula, so the lever is separating isomers. Fragmentation parsimony (fewer bond cleavages) is the useful signal.

**Things that hurt or did nothing:**
- Merging all of PubChem into the pool: LB 0.335 → 0.205 in one case.
- A natural-product filter on PubChem.
- Popularity prior: +0.61 in CV but only +0.075 on the LB.
- ChEBI/LIPID MAPS/NPAtlas: no new coverage.
- Rule-based biotransformation candidates: +0.004 to +0.010.
- LLM-based isomer rerankers.

**De novo (class 3):** honest benchmarks give top-10 exact match of ~4-15% on dissimilar molecules. It is a stretch goal for slots where retrieval has nothing.
Benchmark claims of 65-85% top-1 (GLMR, SpecBridge, GLACIER) look 4-6× above prior work and are unlikely to transfer here.

**Training objective** (Enveda-180 study, arXiv 2602.16507): contrastive or listwise loss beat fingerprint losses by ~10 points of top-1 retrieval on mass-matched candidates. Use a transformer encoder rather than an MLP.

## 4. Validation (the main risk)

Offline-to-LB gaps are huge. Reported: local 0.49-0.85 vs LB 0.19-0.34, and baseline CV 0.78 vs LB 0.09.

- Split by **structure**, never by spectrum. Dedupe candidates on tautomer-canonical InChIKey14.
- The visible `test.parquet` is a placeholder built from train. Use it only as a smoke test.
- Simulate classes by removal:
  - class 1: remove only the truth's own spectra from the library
  - class 2: remove all of the truth's spectra
  - class 3: also remove structures with Tanimoto ≥ 0.7 to the truth, from both library and candidates
- Always include a "truth-absent" control. A library bonus gave +0.09 offline and went to net zero on the LB.
- Use two panels that disagree: `enveda-np-examples` (library-reachable) and `enveda-180` with no public spectra (class-2-like). Accept a change only if it helps both.
- Evaluate MassSpecGym-trained models only on structures outside MassSpecGym.
- Calibrate local CV against one real submission early. Run any public fork unchanged on your own account as a control.

## 5. Traps

- **Don't hard-code the visible `molecule_id` to SMILES map** (thread 745715; ~two dozen forks at 0.416-0.418). Kaggle Staff say IDs don't overlap with the hidden test. It gives no private-set gain and could be judged non-compliant.
- **Forked-kernel gotcha** (thread 760172): some public kernels silently skip real inference unless they detect the scoring environment (`IS_RERUN`). Grep forks for it.
- **Unverified checkpoint provenance:** if public checkpoints turn out to be NIST-derived, every team using them is exposed.
- **Public-LB overfit and fork-family shared fate:** expect a private shake-up.
- **Silent environment bugs:** the RDKit wheel must match the Python version and the pinned RDKit (offline wheel dataset `aidensong123/casmi26-offline-rdkit-2026033`). One report says a Python 3.12 vs 3.13 mismatch zeroed ICEBERG scoring [C].
- A repo author claims winners must provide "original training code for every model weight used". The rules I read say winners deliver code and must be reproducible. I could not verify the stricter reading. Ask the host before relying on opaque public checkpoints.

## 6. Suggested plan (25 days to Dec 14)

Chrome and the browser tools can't join or submit for you. Accepting the rules and submitting need your Kaggle login.

**Days 1-3: baseline and harness**
1. Join (accept rules before Dec 7) and download the data.
2. Run a top public notebook unchanged as a control and note the LB score.
3. Build the structure-disjoint, class-simulating validation panels above, including the truth-absent control.

**Days 4-14: improve what moves class 2/3**
4. Tiered pool: train ∪ COCONUT first, PubChem only to fill empty slots, validated on the class-2 panel.
5. Isomer-aware LambdaRank ranker with fragmentation parsimony, analog propagation and an FPNet-style fingerprint model. Add ICEBERG/GLACIER features only on the top 50-100 candidates.
6. Train a contrastive spectrum-to-molecule model against equal-mass decoys, using `enveda-180` for instrument match and `enveda-np-examples` for domain adaptation.

**Days 15-25: harden**
7. Diversify away from the public fork family, ensemble only if the panels improve, and pick 2 finals with different risk profiles.
8. Document everything for reproducibility. Don't submit anything that depends on NIST or the visible-ID map.

## 7. Open questions to ask the hosts
- JESTR/GLMR checkpoint legality (unanswered in the threads).
- Whether the hidden-test `molecule_id` values will be remapped.
- The exact reproducibility bar for pretrained weights.

## 8. Not read
- ~50 of the ~159 forum threads, the host denovo tutorial notebook, and the Discord. The Kaggle notebook bodies were also not read (titles and scores only).
- `casmi-contest.org` and the Business Wire page (403 and TLS errors).
- No Enveda blog or hackathon found.

## Key links
- Competition: https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra
- Host external-data post: .../discussion/750330
- Host data/model table: .../discussion/748270
- Validation recipe: .../discussion/748524
- Where the MRR goes: .../discussion/743254
- Best repos: github.com/nghianguyen7171/casmi26, github.com/OmerFarukMerey/Enveda-CASMI-2026-Molecule-ID-From-Mass-Spectra-Competition-Solutions, github.com/Isaackjoshua/casmi-2026, github.com/shiferaxa/casmi26
