#!/bin/bash
# =============================================================================
# run_phase2.sh  -  Phase-2 driver for the AIRR-ML pipeline
# =============================================================================
# Reproduces the Phase-2 environment used at submission:
#
#   Change 1 (kmer5/6 disabled, ALL datasets) : AIRR_MODELS omits "kmer56".
#   Change 2 (SKIP_XGB) + Change 4 (REDUCE_TOPK) : exported =1 for LARGE
#           datasets, =0 otherwise.
#   Change 3 (fast-scorer swap) + Change 5 (pos_aa C=1.0) : in-code, no env
#           var needed - present in the patched .py.
#
# LARGE is the exact 25-dataset set from the original Phase-2 run_final.sh
# (verified against the run transcript): D2, D10, D11, D14, D30-40, D49-53,
# D76-80  ==  training directories >= 1 GB.
#
# Usage:
#   ./run_phase2.sh <data_dir> <output_dir> [dataset_numbers...]
#   ./run_phase2.sh ~/AIRR_data ~/output                # datasets 1-95
#   ./run_phase2.sh ~/AIRR_data ~/output 8 30 95        # specific datasets
# =============================================================================
set -u

if [ $# -lt 2 ]; then
    echo "Usage: $0 <data_dir> <output_dir> [dataset_numbers...]"
    exit 1
fi
DATA_DIR="$1"; OUTPUT_DIR="$2"; shift 2

# ---- paths / settings -------------------------------------------------------
SCRIPT="${SCRIPT:-$HOME/airr_ml_submission_v9_phase2.py}"   # the PATCHED file
N_JOBS="${N_JOBS:-16}"

# Change 1: full model set WITHOUT kmer56 (kmer5/6 disabled for all datasets)
AIRR_MODELS_BASE="vj,kmer4,gapped,pos_kmer,diversity,pos_aa,emerson,malidvj"

# Large datasets (>= 1 GB training dir) - drive SKIP_XGB / REDUCE_TOPK
LARGE="2 10 11 14 30 31 32 33 34 35 36 37 38 39 40 49 50 51 52 53 76 77 78 79 80"
is_large() { for d in $LARGE; do [ "$1" = "$d" ] && return 0; done; return 1; }

if [ $# -eq 0 ]; then DATASETS=($(seq 1 95)); else DATASETS=("$@"); fi
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Phase-2 run  |  script: $SCRIPT"
echo "LARGE (SKIP_XGB=1, REDUCE_TOPK=1): $LARGE"
echo "=============================================="

for N in "${DATASETS[@]}"; do
    TRAIN="$DATA_DIR/train_datasets/train_dataset_$N"
    OUT="$OUTPUT_DIR/dataset_$N"
    if [ ! -d "$TRAIN" ]; then echo "[D$N] SKIP - no train dir"; continue; fi

    # discover test dir(s): test_dataset_N and any test_dataset_N_*
    TESTS=""
    [ -d "$DATA_DIR/test_datasets/test_dataset_$N" ] && TESTS="$DATA_DIR/test_datasets/test_dataset_$N"
    for t in "$DATA_DIR"/test_datasets/test_dataset_${N}_*; do
        [ -d "$t" ] && TESTS="${TESTS:+$TESTS }$t"
    done
    [ -z "$TESTS" ] && TESTS="$TRAIN"   # fall back to train if no test dir

    # Changes 2 + 4: toggle for large datasets
    if is_large "$N"; then SKIP_XGB=1; REDUCE_TOPK=1; TAG="LARGE"; else SKIP_XGB=0; REDUCE_TOPK=0; TAG="small"; fi

    echo "[D$N] $TAG  SKIP_XGB=$SKIP_XGB REDUCE_TOPK=$REDUCE_TOPK"
    mkdir -p "$OUT"
    AIRR_MODELS="$AIRR_MODELS_BASE" \
    SKIP_XGB="$SKIP_XGB" REDUCE_TOPK="$REDUCE_TOPK" \
    TASK2_USE_TEST_DATA=1 \
    python3 "$SCRIPT" \
        --train_dir "$TRAIN" \
        --test_dirs $TESTS \
        --out_dir "$OUT" \
        --n_jobs "$N_JOBS"
done

echo "=============================================="
echo "Done. Per-dataset outputs in: $OUTPUT_DIR"
echo "Note: D95 was originally run standalone with --n_jobs 1 (SKIP_XGB=0 REDUCE_TOPK=0)."
echo "=============================================="
