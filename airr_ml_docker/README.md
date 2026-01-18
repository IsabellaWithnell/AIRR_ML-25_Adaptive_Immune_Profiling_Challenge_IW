# AIRR-ML Docker Container

## Quick Start

### 1. Build the Docker image

```bash
docker build -t airr-ml:v9 .
```

### 2. Run a single dataset

```bash
docker run --rm \
    -v /path/to/your/data:/data:ro \
    -v /path/to/output:/output \
    airr-ml:v9 \
    --train_dir /data/train_datasets/train_dataset_1 \
    --test_dirs /data/test_datasets/test_dataset_1 \
    --out_dir /output/dataset_1 \
    --n_jobs 16
```

### 3. Run ANY dataset (1, 50, 100, etc.)

```bash
# Dataset 100
docker run --rm \
    -v /path/to/data:/data:ro \
    -v /path/to/output:/output \
    airr-ml:v9 \
    --train_dir /data/train_datasets/train_dataset_100 \
    --test_dirs /data/test_datasets/test_dataset_100 \
    --out_dir /output/dataset_100 \
    --n_jobs 16
```

### 4. Run all datasets with helper script

```bash
chmod +x run_all.sh

# Run datasets 1-8
./run_all.sh /path/to/your/data /path/to/output

# Run specific datasets (e.g., 1, 50, 100)
./run_all.sh /path/to/your/data /path/to/output 1 50 100
```

## Using Docker Compose

```bash
# Set environment variables
export DATA_DIR=/path/to/your/data
export OUTPUT_DIR=/path/to/output

# Run ANY dataset using DATASET_NUM
DATASET_NUM=1 docker-compose run --rm airr-ml
DATASET_NUM=50 docker-compose run --rm airr-ml
DATASET_NUM=100 docker-compose run --rm airr-ml

# Or use pre-defined shortcuts for datasets 1-8
docker-compose run --rm dataset1
docker-compose run --rm dataset6

# With custom models
DATASET_NUM=5 AIRR_MODELS="vj,kmer4" docker-compose run --rm airr-ml
```

## Directory Structure

Your data directory should look like:

```
/path/to/your/data/
├── train_datasets/
│   ├── train_dataset_1/
│   │   ├── sample1.tsv
│   │   ├── sample2.tsv
│   │   └── ...
│   ├── train_dataset_2/
│   └── ...
└── test_datasets/
    ├── test_dataset_1/
    ├── test_dataset_2/
    ├── test_dataset_7_1/
    ├── test_dataset_7_2/
    └── ...
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AIRR_MODELS` | `vj,kmer4,kmer56,gapped,pos_kmer,diversity,pos_aa,emerson,malidvj` | Models to train (ALL by default) |
| `TASK2_MODEL_OVERRIDE` | `BEST` | Which model to use for Task 2 |
| `TASK2_USE_TEST_DATA` | `1` | Use test data for important sequences |
| `AIRR_ENSEMBLE_MODE` | `standard` | Ensemble selection mode |
| `N_JOBS` | `16` | Number of parallel workers |

### All Models Included by Default

| Model Group | Individual Models | Description |
|-------------|-------------------|-------------|
| `vj` | vj_simple, vj_logfreq, vj_elasticnet, vj_interact | VJ gene usage |
| `kmer4` | kmer4_LR_freq, kmer4_LR_raw, kmer4_RF, kmer4_XGB | 4-mer features |
| `kmer56` | kmer5_sgd, kmer6_sgd | 5/6-mer SGD models |
| `gapped` | gapped_kmer | Gapped k-mer features |
| `pos_kmer` | pos_kmer4 | Positional k-mers |
| `diversity` | diversity metrics | Repertoire diversity |
| `pos_aa` | positional amino acid | T1D-style positional AA |
| `emerson` | Fisher's exact test | Public sequence enrichment |
| `malidvj` | VJ + positional + germline/nregion | MALIDVJ age-robust model |

### Example with custom settings (subset of models)

```bash
docker run --rm \
    -v /path/to/data:/data:ro \
    -v /path/to/output:/output \
    -e AIRR_MODELS="vj,kmer4,emerson" \
    -e TASK2_MODEL_OVERRIDE="kmer4_LR_freq" \
    -e TASK2_USE_TEST_DATA=1 \
    airr-ml:v9 \
    --train_dir /data/train_datasets/train_dataset_1 \
    --test_dirs /data/test_datasets/test_dataset_1 \
    --out_dir /output/dataset_1 \
    --n_jobs 16
```

## Output Files

After running, you'll find:

```
/path/to/output/
├── dataset_1/
│   ├── train_dataset_1_test_predictions.tsv
│   └── train_dataset_1_important_sequences.tsv
├── dataset_2/
│   └── ...
├── submissions.csv      # Combined all datasets
└── submissions.xlsx     # Excel with sheets per dataset
```

## Using Docker Compose

For easier management of multiple datasets:

```bash
# Set environment variables
export DATA_DIR=/path/to/your/data
export OUTPUT_DIR=/path/to/output

# Run a specific dataset
docker-compose run --rm dataset1

# Run all datasets
docker-compose up

# Run with custom models
AIRR_MODELS="vj,kmer4" docker-compose run --rm dataset1
```

## Troubleshooting

### Out of memory
Reduce `--n_jobs` or increase Docker memory limit:
```bash
docker run --memory=64g ...
```

### Permission denied on output
Make sure the output directory is writable:
```bash
chmod 777 /path/to/output
```

### Missing test directories
The script will fall back to training data if test directories are not found.

## KAGGLE Competition Hand-Tuned Configs (Datasets 1-8)

For the original KAGGLE competition, each dataset has optimized model configurations:

### Dataset 1: VJ + K-mer Ensemble
```bash
docker run --rm \
    -v /path/to/data:/data:ro \
    -v /path/to/output:/output \
    -e AIRR_MODELS="vj,kmer4" \
    -e TASK2_MODEL_OVERRIDE="AUTO" \
    -e AIRR_ENSEMBLE_MODE="vj_kmer" \
    airr-ml:v9 \
    --train_dir /data/train_datasets/train_dataset_1 \
    --test_dirs /data/test_datasets/test_dataset_1 \
    --out_dir /output/dataset_1 \
    --n_jobs 16
```

### Dataset 2: K-mer Dominant
```bash
docker run --rm \
    -v /path/to/data:/data:ro \
    -v /path/to/output:/output \
    -e AIRR_MODELS="kmer4,gapped,pos_kmer,emerson" \
    -e TASK2_MODEL_OVERRIDE="AUTO" \
    -e AIRR_ENSEMBLE_MODE="standard" \
    airr-ml:v9 \
    --train_dir /data/train_datasets/train_dataset_2 \
    --test_dirs /data/test_datasets/test_dataset_2 \
    --out_dir /output/dataset_2 \
    --n_jobs 16
```

### Dataset 3: 5/6-mer SGD + VJ
```bash
docker run --rm \
    -v /path/to/data:/data:ro \
    -v /path/to/output:/output \
    -e AIRR_MODELS="kmer56,vj" \
    -e TASK2_MODEL_OVERRIDE="AUTO" \
    -e AIRR_ENSEMBLE_MODE="none" \
    airr-ml:v9 \
    --train_dir /data/train_datasets/train_dataset_3 \
    --test_dirs /data/test_datasets/test_dataset_3 \
    --out_dir /output/dataset_3 \
    --n_jobs 16
```

### Dataset 4: Gapped K-mer Focus
```bash
docker run --rm \
    -v /path/to/data:/data:ro \
    -v /path/to/output:/output \
    -e AIRR_MODELS="gapped,kmer4,pos_kmer,emerson" \
    -e TASK2_MODEL_OVERRIDE="gapped_kmer" \
    -e AIRR_ENSEMBLE_MODE="standard" \
    airr-ml:v9 \
    --train_dir /data/train_datasets/train_dataset_4 \
    --test_dirs /data/test_datasets/test_dataset_4 \
    --out_dir /output/dataset_4 \
    --n_jobs 16
```

### Dataset 5: Full K-mer + VJ
```bash
docker run --rm \
    -v /path/to/data:/data:ro \
    -v /path/to/output:/output \
    -e AIRR_MODELS="kmer4,pos_kmer,gapped,vj,emerson" \
    -e TASK2_MODEL_OVERRIDE="AUTO" \
    -e AIRR_ENSEMBLE_MODE="standard" \
    airr-ml:v9 \
    --train_dir /data/train_datasets/train_dataset_5 \
    --test_dirs /data/test_datasets/test_dataset_5 \
    --out_dir /output/dataset_5 \
    --n_jobs 16
```

### Dataset 6: Full Suite + Diversity
```bash
docker run --rm \
    -v /path/to/data:/data:ro \
    -v /path/to/output:/output \
    -e AIRR_MODELS="kmer4,pos_kmer,gapped,vj,diversity,emerson" \
    -e TASK2_MODEL_OVERRIDE="emerson" \
    -e AIRR_ENSEMBLE_MODE="standard" \
    airr-ml:v9 \
    --train_dir /data/train_datasets/train_dataset_6 \
    --test_dirs /data/test_datasets/test_dataset_6 \
    --out_dir /output/dataset_6 \
    --n_jobs 16
```

### Dataset 7: Emerson Stacking (HSV-2)
```bash
docker run --rm \
    -v /path/to/data:/data:ro \
    -v /path/to/output:/output \
    -e AIRR_MODELS="emerson" \
    -e TASK2_MODEL_OVERRIDE="SKIP" \
    -e AIRR_ENSEMBLE_MODE="emerson_stacking" \
    airr-ml:v9 \
    --train_dir /data/train_datasets/train_dataset_7 \
    --test_dirs /data/test_datasets/test_dataset_7_1 /data/test_datasets/test_dataset_7_2 \
    --out_dir /output/dataset_7 \
    --n_jobs 16
```

### Dataset 8: MALIDVJ Age-Robust (T1D)
```bash
docker run --rm \
    -v /path/to/data:/data:ro \
    -v /path/to/output:/output \
    -e AIRR_MODELS="malidvj" \
    -e TASK2_MODEL_OVERRIDE="SKIP" \
    -e AIRR_ENSEMBLE_MODE="none" \
    airr-ml:v9 \
    --train_dir /data/train_datasets/train_dataset_8 \
    --test_dirs /data/test_datasets/test_dataset_8_1 /data/test_datasets/test_dataset_8_2 /data/test_datasets/test_dataset_8_3 \
    --out_dir /output/dataset_8 \
    --n_jobs 16
```

### Summary Table: KAGGLE Hand-Tuned Configurations

| Dataset | Models | Task 2 | Ensemble Mode | Description |
|---------|--------|--------|---------------|-------------|
| 1 | vj,kmer4 | AUTO | vj_kmer | VJ + K-mer ensemble |
| 2 | kmer4,gapped,pos_kmer,emerson | AUTO | standard | K-mer dominant |
| 3 | kmer56,vj | AUTO | none | 5/6-mer SGD |
| 4 | gapped,kmer4,pos_kmer,emerson | gapped_kmer | standard | Gapped k-mer |
| 5 | kmer4,pos_kmer,gapped,vj,emerson | AUTO | standard | Full k-mer + VJ |
| 6 | kmer4,pos_kmer,gapped,vj,diversity,emerson | emerson | standard | Full suite |
| 7 | emerson | SKIP | emerson_stacking | HSV-2 Emerson |
| 8 | malidvj | SKIP | none | T1D MALIDVJ |

