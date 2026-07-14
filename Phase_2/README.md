# Phase 2 — Submission and Changes

Phase 2 reran the pipeline across all challenge datasets. Five changes from airr_ml_submission_v9_parallel_task2.py
made purely for memory and runtime, so the pipeline could complete on the larger
Phase-2 datasets without running out of memory.

## Files

| File | Purpose |
|------|---------|
| `airr_ml_submission_v9_phase2_submission.py` | The submitted pipeline: the Phase-1 v9 pipeline with the five changes below. |
| `run_phase2.sh` | Driver. Runs each dataset with the Phase-2 environment (kmer5/6 off; `SKIP_XGB`/`REDUCE_TOPK` on for large datasets; Task 2 scored on test data). |
| `apply_phase2_changes.py` | Reproducibility. Applies the five changes to a clean Phase-1 file to regenerate the submission file (see below). |

## Reproduce

```bash
./run_phase2.sh /path/to/data /path/to/output          # datasets 1–95
```

Per dataset, the driver sets: `AIRR_MODELS` without `kmer56` (change 1);
`SKIP_XGB=1 REDUCE_TOPK=1` for large datasets (changes 2 and 4); and
`TASK2_USE_TEST_DATA=1`. **Large** = the 25 datasets with ≥1 GB training data:
`2, 10, 11, 14, 30–40, 49–53, 76–80`. Dataset 95 was run standalone with
`--n_jobs 1` and both toggles off.

To regenerate the submission file from the clean Phase-1 pipeline:

```bash
python3 apply_phase2_changes.py \
    ../airr_ml_docker/airr_ml_submission_v9_parallel_task2.py \
    airr_ml_submission_v9_phase2_submission.py
```

## Changes from Phase 1

| # | Change | Trigger | Location | Datasets |
|---|--------|---------|----------|----------|
| 1 | kmer5/6 SGD models disabled | `AIRR_MODELS` omits `kmer56` | `run_phase2.sh` | all |
| 2 | XGBoost disabled on large datasets | `SKIP_XGB=1` → `HAS_XGB=False` | after imports (`.py`) | large (25) |
| 3 | Fast-scorer swap for Task 2 | automatic | before Task 2 dispatch in `identify_associated_sequences` | any dataset where MALIDVJ/pos_aa wins Task 1 |
| 4 | `TOP_K_SEQUENCES` 50,000 → 10,000 on large datasets | `REDUCE_TOPK=1` | after `CFG = Cfg()` | large (25) |
| 5 | pos_aa empty-params fix (C → 1.0) | automatic | `C=bp.get('C', 1.0)` (all sites) | D95 |

**1. kmer5/6 disabled (all datasets).** The 5-/6-mer SGD models build very large
sparse feature matrices (~50–80 GB per dataset) and caused OOM kills when several
datasets ran in parallel. Impact is minimal: kmer5/6 was the best model on only
Dataset 3 in Phase 1 (AUC 0.58), and kmer4 / gapped k-mer / the ensemble matched
or beat it elsewhere.

**2. XGBoost disabled on large datasets (≥1 GB training data).** XGBoost was slow
and consistently weak on large datasets (typical AUC 0.50–0.56, never the best
model). Setting `SKIP_XGB=1` sets `HAS_XGB = False`, removing it from the model
set.

**3. Fast-scorer swap for Task 2.** MALIDVJ and pos_aa score sequences
sequentially (~5 min per repertoire). If a parallelisable model (VJ, kmer4,
gapped k-mer, …) reached an equal Task 1 AUC (within 0.001), it is used for Task
2 scoring instead; otherwise the best fast model within 0.02 AUC is used. This
cut Task 2 scoring from ~23 hours to ~40 minutes on 2 GB+ datasets. The swap
changes only *how* sequences are scored, not *which* sequences are eligible.

**4. `TOP_K_SEQUENCES` reduced on large datasets.** Reduced from 50,000 to 10,000
to prevent OOM during sequence deduplication; submissions are then padded back to
50,000 per dataset using real training sequences with importance score 0.
Controlled by `REDUCE_TOPK=1`.

**5. pos_aa empty-parameters bug fix.** When every candidate `C` failed to beat
AUC 0.5, the pos_aa best-parameters dict was empty and `bp['C']` raised a
`KeyError`. It now defaults to `C = 1.0`. Affected only Dataset 95.
