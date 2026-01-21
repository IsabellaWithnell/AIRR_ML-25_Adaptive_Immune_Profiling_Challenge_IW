#!/bin/bash
# =============================================================================
# AIRR-ML Docker Runner - Run All Datasets
# =============================================================================
#
# Usage:
#   ./run_all.sh /path/to/data /path/to/output [datasets]
#
# Examples:
#   # Run all datasets 1-8
#   ./run_all.sh /data/airr /output/results
#
#   # Run specific datasets
#   ./run_all.sh /data/airr /output/results 1 2 3
#
#   # Run datasets 1-6
#   ./run_all.sh /data/airr /output/results 1 2 3 4 5 6
#
# Environment variables:
#   AIRR_MODELS         - Models to train (default: ALL models)
#   TASK2_MODEL_OVERRIDE - Task 2 model (default: BEST)
#   TASK2_USE_TEST_DATA  - Use test data for Task 2 (default: 1)
#   N_JOBS              - Number of parallel workers (default: 16)
# =============================================================================

set -e

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <data_dir> <output_dir> [dataset_numbers...]"
    echo ""
    echo "Examples:"
    echo "  $0 /path/to/data /path/to/output           # Run all datasets 1-8"
    echo "  $0 /path/to/data /path/to/output 1 2 3     # Run datasets 1, 2, 3"
    echo ""
    exit 1
fi

DATA_DIR="$1"
OUTPUT_DIR="$2"
shift 2

# Default datasets if none specified
if [ $# -eq 0 ]; then
    DATASETS=(1 2 3 4 5 6 7 8)
else
    DATASETS=("$@")
fi

# Default settings
AIRR_MODELS="${AIRR_MODELS:-vj,kmer4,kmer56,gapped,pos_kmer,diversity,pos_aa,emerson,malidvj}"
TASK2_MODEL_OVERRIDE="${TASK2_MODEL_OVERRIDE:-BEST}"
TASK2_USE_TEST_DATA="${TASK2_USE_TEST_DATA:-0}"
N_JOBS="${N_JOBS:-16}"

echo "=============================================="
echo "AIRR-ML Docker Runner"
echo "=============================================="
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Datasets to run: ${DATASETS[*]}"
echo "Models: $AIRR_MODELS"
echo "Task 2 model: $TASK2_MODEL_OVERRIDE"
echo "Task 2 use test data: $TASK2_USE_TEST_DATA"
echo "Parallel workers: $N_JOBS"
echo "=============================================="
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build Docker image if needed
echo "Building Docker image..."
docker build -t airr-ml:v9 .

# Run each dataset
for DATASET_NUM in "${DATASETS[@]}"; do
    echo ""
    echo "=============================================="
    echo "Running Dataset $DATASET_NUM"
    echo "=============================================="
    
    TRAIN_DIR="$DATA_DIR/train_datasets/train_dataset_${DATASET_NUM}"
    DATASET_OUTPUT="$OUTPUT_DIR/dataset_${DATASET_NUM}"
    
    # Check if training directory exists
    if [ ! -d "$TRAIN_DIR" ]; then
        echo "WARNING: Training directory not found: $TRAIN_DIR"
        echo "Skipping dataset $DATASET_NUM"
        continue
    fi
    
    # Find test directories
    TEST_DIRS=""
    
    # Check for single test directory
    if [ -d "$DATA_DIR/test_datasets/test_dataset_${DATASET_NUM}" ]; then
        TEST_DIRS="/data/test_datasets/test_dataset_${DATASET_NUM}"
    fi
    
    # Check for multiple test directories (test_dataset_N_*)
    for test_dir in "$DATA_DIR"/test_datasets/test_dataset_${DATASET_NUM}_*; do
        if [ -d "$test_dir" ]; then
            test_name=$(basename "$test_dir")
            if [ -z "$TEST_DIRS" ]; then
                TEST_DIRS="/data/test_datasets/$test_name"
            else
                TEST_DIRS="$TEST_DIRS /data/test_datasets/$test_name"
            fi
        fi
    done
    
    if [ -z "$TEST_DIRS" ]; then
        echo "WARNING: No test directories found for dataset $DATASET_NUM"
        echo "Using training directory as fallback"
        TEST_DIRS="/data/train_datasets/train_dataset_${DATASET_NUM}"
    fi
    
    echo "Train dir: $TRAIN_DIR"
    echo "Test dirs: $TEST_DIRS"
    echo "Output: $DATASET_OUTPUT"
    
    # Create output directory
    mkdir -p "$DATASET_OUTPUT"
    
    # Run Docker container
    docker run --rm \
        -v "$DATA_DIR:/data:ro" \
        -v "$OUTPUT_DIR:/output" \
        -e AIRR_MODELS="$AIRR_MODELS" \
        -e TASK2_MODEL_OVERRIDE="$TASK2_MODEL_OVERRIDE" \
        -e TASK2_USE_TEST_DATA="$TASK2_USE_TEST_DATA" \
        -e AIRR_ENSEMBLE_MODE="standard" \
        -e AIRR_DATASET_ID="$DATASET_NUM" \
        airr-ml:v9 \
        --train_dir "/data/train_datasets/train_dataset_${DATASET_NUM}" \
        --test_dirs $TEST_DIRS \
        --out_dir "/output/dataset_${DATASET_NUM}" \
        --n_jobs "$N_JOBS"
    
    echo "Dataset $DATASET_NUM completed!"
done

echo ""
echo "=============================================="
echo "All datasets completed!"
echo "=============================================="

# Concatenate all outputs
echo ""
echo "Concatenating all outputs..."

python3 << CONCAT_SCRIPT
import os
import glob
import pandas as pd

output_dir = "$OUTPUT_DIR"

print(f"  Output directory: {output_dir}")

# Find all prediction and sequence files
predictions_files = sorted(glob.glob(os.path.join(output_dir, '*/*_test_predictions.tsv')))
sequences_files = sorted(glob.glob(os.path.join(output_dir, '*/*_important_sequences.tsv')))

print(f"  Found {len(predictions_files)} prediction files")
print(f"  Found {len(sequences_files)} sequence files")

# Load and concatenate
df_list = []

for f in predictions_files:
    try:
        df = pd.read_csv(f, sep='\t')
        df_list.append(df)
        print(f"    Loaded: {os.path.basename(f)} ({len(df)} rows)")
    except Exception as e:
        print(f"    WARNING: Could not read {f}: {e}")

for f in sequences_files:
    try:
        df = pd.read_csv(f, sep='\t')
        df_list.append(df)
        print(f"    Loaded: {os.path.basename(f)} ({len(df)} rows)")
    except Exception as e:
        print(f"    WARNING: Could not read {f}: {e}")

if df_list:
    concatenated_df = pd.concat(df_list, ignore_index=True)
    
    # Ensure correct column order
    expected_cols = ['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']
    for col in expected_cols:
        if col not in concatenated_df.columns:
            concatenated_df[col] = -999.0
    concatenated_df = concatenated_df[expected_cols]
    
    # Save as CSV
    submissions_file = os.path.join(output_dir, 'submissions.csv')
    concatenated_df.to_csv(submissions_file, index=False)
    print(f"\n  SUCCESS: Combined submission saved to: {submissions_file}")
    
    # Save as Excel
    try:
        excel_file = os.path.join(output_dir, 'submissions.xlsx')
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            concatenated_df.to_excel(writer, sheet_name='All_Combined', index=False)
            for dataset in sorted(concatenated_df['dataset'].unique()):
                subset = concatenated_df[concatenated_df['dataset'] == dataset]
                sheet_name = dataset[:31]
                subset.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"  SUCCESS: Excel file saved to: {excel_file}")
    except Exception as e:
        print(f"  WARNING: Could not save Excel: {e}")
    
    print(f"\n  Total rows: {len(concatenated_df)}")
    print(f"\n  Summary by dataset:")
    for dataset in sorted(concatenated_df['dataset'].unique()):
        subset = concatenated_df[concatenated_df['dataset'] == dataset]
        n_preds = len(subset[subset['label_positive_probability'] != -999.0])
        n_seqs = len(subset[subset['label_positive_probability'] == -999.0])
        print(f"    {dataset}: {n_preds} predictions, {n_seqs} sequences")
else:
    print("  WARNING: No files found to concatenate")
CONCAT_SCRIPT

echo ""
echo "=============================================="
echo "Done!"
echo "=============================================="
echo "Output files:"
ls -la "$OUTPUT_DIR"/*.csv "$OUTPUT_DIR"/*.xlsx 2>/dev/null || echo "  (no combined files yet)"
