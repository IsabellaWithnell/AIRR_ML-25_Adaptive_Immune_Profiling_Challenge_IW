#!/usr/bin/env python3
"""
AIRR-ML Competition Submission - Unified Pipeline v9 + TASK 2 UNIFIED CONTROL + PARALLEL SCORING
==================================================================================================

This file adheres to the official competition template from:
https://github.com/uio-bmi/predict-airr

============================================================================
PARALLEL TASK 2 SCORING (NEW)
============================================================================
Task 2 sequence scoring is now parallelized across repertoires for 10-16x speedup.
- 400 repertoires: ~46 hours -> ~3-5 hours with 16 workers
- Uses multiprocessing.Pool for VJ, k-mer, vj_positional, and Emerson models
- MALIDVJ and pos_aa remain sequential (feature extractors not picklable)

============================================================================
UNIFIED CONTROL VIA BASH SCRIPT (.sh controls everything)
============================================================================
**DEFAULTS (when no environment variables set):**
  - TASK 1: Train ALL models, use ensemble, pick best
  - TASK 2: Use BEST model from Task 1, score TRAINING sequences

**CONTROL via .sh script (recommended):**
  - Set AIRR_MODELS to limit which models are trained
  - Set TASK2_MODEL_OVERRIDE to specify which model for Task 2
  - Set TASK2_USE_TEST_DATA=1 to score test sequences

**Examples in .sh:**
  Dataset 1: task1=kmer56, task2=kmer5_sgd → Only train 5/6-mer, use 5-mer for Task 2
  Dataset 4: task1=gapped,kmer4, task2=gapped_kmer → Train both, use gapped for Task 2
  Dataset 8: task1=malidvj, task2=malidvj → Train MALIDVJ, use it for Task 2

============================================================================
ENVIRONMENT VARIABLES (typically set by bash script):
============================================================================
**AIRR_MODELS** - Which models to train (Task 1)
  - Not set = train ALL models (Python default)
  - "kmer56" = only train kmer5_sgd, kmer6_sgd
  - "gapped,kmer4" = only train gapped_kmer + kmer4 variants
  - "ALL" = explicitly train all models

**TASK2_MODEL_OVERRIDE** - Which model for Task 2 feature extraction
  - Not set or "BEST" = use best-performing model from Task 1 (Python default)
  - "gapped_kmer" = force gapped k-mer model
  - "kmer5_sgd" = force 5-mer SGD model
  - "malidvj" = force MALIDVJ model
  - "AUTO" = use dataset-specific strategy (advanced)

**TASK2_USE_TEST_DATA** - Which sequences to score for Task 2
  - Not set or "0" = score TRAINING sequences (Python default)
  - "1" = score TEST sequences (recommended for competition)

============================================================================
USAGE EXAMPLES:
============================================================================

**1. Python alone (all defaults):**
  python3 script.py --train_dir train --test_dirs test --out_dir output
  → Trains ALL models, uses BEST for Task 2, scores TRAINING sequences

**2. From bash script (recommended):**
  # In .sh file:
  export AIRR_MODELS="kmer56"
  export TASK2_MODEL_OVERRIDE="kmer5_sgd"
  export TASK2_USE_TEST_DATA=1
  python3 script.py --train_dir train --test_dirs test --out_dir output
  → Trains only 5/6-mer, uses 5-mer for Task 2, scores TEST sequences

**3. Command line override:**
  export TASK2_MODEL_OVERRIDE="gapped_kmer"
  export TASK2_USE_TEST_DATA=1
  python3 script.py --train_dir train --test_dirs test --out_dir output
  → Trains ALL models, uses gapped_kmer for Task 2, scores TEST sequences

============================================================================
ORIGINAL v9 FEATURES (retained):
============================================================================

FIXES from v8:
- Improved ensemble model selection: now uses dynamic threshold
  - Only includes models within 0.02 AUC of the best model
  - Still enforces minimum AUC floor of 0.55
  - Prevents weak models from diluting ensemble performance

FIXES from v7 (retained):
- Added MALIDVJ model: VJ + Positional AA features (from T1D script)
  - Includes germline_frac and nregion_frac features
  - Uses L2 regularization with class_weight='balanced'
  - Age-robust feature selection when age data available

FIXES from v6 (retained):
- K-mer 5/6 SGD models now generate CV predictions for ensemble stacking
- Removed unused EmersonEnsemble class
- Removed unused ext_pub function

FIXES from v5 (retained):
- Age-robust feature selection NOW ACTUALLY USED when age data is available
- Age-robust sample weighting NOW ACTUALLY USED when age data is available  
- Feature selection done INSIDE CV loop (no data leakage)
- Fixed ensemble prediction path
- Fixed positional k-mer prediction path
- Added VJ interaction model to training loop

Usage (competition standard):
  python3 -m submission.main --train_dir /path/to/train --test_dirs /path/to/test1 /path/to/test2 --out_dir /path/to/output --n_jobs 4 --device cpu

Models included:
- MALIDVJ: VJ + Positional AA with germline/nregion features (T1D-style)
- VJ: vj_simple, vj_logfreq, vj_elasticnet, vj_interaction
- K-mer4: LR_freq, LR_raw, RF, XGB, positional, gapped
- K-mer5/6: SGD with calibration, early stopping, and CV predictions for ensemble
- Diversity: full metrics including evenness and log_richness
- Positional AA: T1D-style positional amino acid features
- Emerson: Fisher's exact test with grid search
- Age-robust: feature selection + sample weighting when age available
- Meta-learner ensemble (dynamic selection: models within 0.02 AUC of best)
"""

import os
import sys
import json
import glob
import argparse
import warnings
import gc
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple, Optional, Union, Iterator
from dataclasses import dataclass, field
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, fisher_exact
from scipy import sparse
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

warnings.filterwarnings('ignore')


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Cfg:
    CV_FOLDS: int = 5
    CV_FOLDS_T1D: int = 3  # T1D (Dataset 8) uses 3-fold CV to match original script
    SEED: int = 42
    SEED_KMER56: int = 123  # Dataset 3 kmer5/6 uses seed 123 to match original script
    SEED_DATASET1: int = 123  # Dataset 1 uses seed 123 to match 1_malid_tcr_multi_vj_new.py
    AA: Set[str] = field(default_factory=lambda: set('ACDEFGHIKLMNPQRSTVWY'))
    C_VALS: List[float] = field(default_factory=lambda: [5, 1, 0.2, 0.1, 0.05, 0.03])
    
    # T1D age-robust settings
    AGE_THRESHOLD: float = 25.0
    CORR_THRESHOLD: float = 0.05
    AGE_WEIGHT_BINS: int = 5
    
    # Mal-ID preprocessing
    DEDUPLICATE_BY_CDR3: bool = True
    MAX_SEQS_PER_SAMPLE: int = 10000000
    
    # Sequence scoring
    TOP_K_SEQUENCES: int = 50000
    
    # SGD settings for 5/6-mer
    SGD_ALPHA_VALUES: List[float] = field(default_factory=lambda: [0.00001, 0.0001, 0.001, 0.01])
    SGD_MAX_ITER: int = 2000
    SGD_TOL: float = 1e-4
    SGD_EARLY_STOPPING: bool = True
    SGD_N_ITER_NO_CHANGE: int = 10
    SGD_VALIDATION_FRACTION: float = 0.1
    
    # Sparse matrix settings
    SPARSE_THRESHOLD: int = 50000
    SPARSE_DENSITY_THRESHOLD: float = 0.1
    
    # Parallel loading settings
    N_WORKERS: int = 8
    USE_PARALLEL_LOADING: bool = True
    
    # =================================================================
    # PARAMETERS TO MATCH v10 EXACTLY
    # =================================================================
    # ElasticNet L1 ratios (for vj_elasticnet) - matches v10
    ELASTICNET_L1_RATIOS: List[float] = field(default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 0.9])
    
    # Emerson parameters - combined from v10 and dataset7
    EMERSON_P_THRESHOLDS: List[float] = field(default_factory=lambda: [0.05, 0.1])  # v10 uses [0.05, 0.1]
    EMERSON_MIN_COUNT: int = 2
    
    # RF parameters - matches v10
    RF_PARAMS: List[dict] = field(default_factory=lambda: [
        {'n_estimators': 100, 'max_depth': 10, 'min_samples_leaf': 5},
        {'n_estimators': 200, 'max_depth': 7, 'min_samples_leaf': 10}
    ])
    
    # XGB parameters - matches v10
    XGB_PARAMS: List[dict] = field(default_factory=lambda: [
        {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1},
        {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.05}
    ])
    
    # =================================================================
    # MODEL SELECTION FLAGS (read from AIRR_MODELS environment variable)
    # Set via: export AIRR_MODELS="vj,kmer4,emerson"
    # =================================================================
    RUN_VJ: bool = True           # vj_simple, vj_logfreq, vj_elasticnet, vj_interact
    RUN_KMER4: bool = True        # kmer4_LR_freq, kmer4_LR_raw, kmer4_RF, kmer4_XGB
    RUN_KMER56: bool = True       # kmer5_sgd, kmer6_sgd (high memory)
    RUN_GAPPED: bool = True       # gapped_kmer
    RUN_POS_KMER: bool = True     # pos_kmer4
    RUN_DIVERSITY: bool = True    # diversity metrics
    RUN_POS_AA: bool = True       # positional amino acid features
    RUN_EMERSON: bool = True      # Fisher's exact test classifier
    RUN_MALIDVJ: bool = True      # VJ + positional + germline/nregion
    
    def __post_init__(self):
        """Parse AIRR_MODELS environment variable to set model flags."""
        models_env = os.environ.get('AIRR_MODELS', '')
        if models_env:
            models_list = [m.strip().lower() for m in models_env.split(',')]
            print(f"\n*** MODEL SELECTION FROM ENV: {models_list} ***")
            
            # Reset all to False, then enable only specified models
            self.RUN_VJ = 'vj' in models_list
            self.RUN_KMER4 = 'kmer4' in models_list
            self.RUN_KMER56 = 'kmer56' in models_list
            self.RUN_GAPPED = 'gapped' in models_list
            self.RUN_POS_KMER = 'pos_kmer' in models_list
            self.RUN_DIVERSITY = 'diversity' in models_list
            self.RUN_POS_AA = 'pos_aa' in models_list
            self.RUN_EMERSON = 'emerson' in models_list
            self.RUN_MALIDVJ = 'malidvj' in models_list
            
            print(f"  VJ models: {self.RUN_VJ}")
            print(f"  K-mer4 models: {self.RUN_KMER4}")
            print(f"  K-mer5/6 SGD: {self.RUN_KMER56}")
            print(f"  Gapped k-mer: {self.RUN_GAPPED}")
            print(f"  Positional k-mer: {self.RUN_POS_KMER}")
            print(f"  Diversity: {self.RUN_DIVERSITY}")
            print(f"  Positional AA: {self.RUN_POS_AA}")
            print(f"  Emerson: {self.RUN_EMERSON}")
            print(f"  MALIDVJ: {self.RUN_MALIDVJ}")
            print()


CFG = Cfg()
AA = 'ACDEFGHIKLMNPQRSTVWY'
AA_TR = str.maketrans('', '', ''.join(c for c in ''.join(chr(i) for i in range(256)) if c not in AA))
VALID_AA = set(AA)


# =============================================================================
# PARALLEL TASK 2 SCORING FUNCTIONS
# =============================================================================
# These functions enable 10-16x faster Task 2 sequence scoring by parallelizing
# across repertoires using multiprocessing.
# =============================================================================

def _score_repertoire_kmer(args):
    """Score sequences using k-mer features (for multiprocessing)."""
    df, feature_scores, k, extractor_type = args
    
    results = []
    if 'junction_aa' not in df.columns:
        return results
    
    for _, row in df.iterrows():
        seq = row.get('junction_aa', '')
        if not isinstance(seq, str) or len(seq) < k:
            continue
        
        v_gene = str(row.get('v_call', '')).split('*')[0] if pd.notna(row.get('v_call')) else ''
        j_gene = str(row.get('j_call', '')).split('*')[0] if pd.notna(row.get('j_call')) else ''
        
        seq_clean = ''.join(c for c in seq.upper() if c in VALID_AA)
        if len(seq_clean) < k:
            continue
        
        score = 0.0
        if extractor_type == 'positional':
            for i in range(len(seq_clean) - k + 1):
                pos_km = f"p{i}_{seq_clean[i:i+k]}"
                score += feature_scores.get(pos_km, 0)
        elif extractor_type == 'gapped':
            for i in range(len(seq_clean) - 4):
                gapped = f"{seq_clean[i]}_{seq_clean[i+2]}_{seq_clean[i+4]}"
                score += feature_scores.get(gapped, 0)
        else:
            for i in range(len(seq_clean) - k + 1):
                score += feature_scores.get(seq_clean[i:i+k], 0)
        
        results.append((seq, v_gene or 'unknown', j_gene or 'unknown', score))
    return results


def _score_repertoire_vj(args):
    """Score sequences using VJ features (for multiprocessing)."""
    df, feature_scores = args
    
    results = []
    if 'junction_aa' not in df.columns:
        return results
    
    for _, row in df.iterrows():
        seq = row.get('junction_aa', '')
        if not isinstance(seq, str) or len(seq) < 8:
            continue
        
        v_gene = str(row.get('v_call', '')).split('*')[0] if pd.notna(row.get('v_call')) else ''
        j_gene = str(row.get('j_call', '')).split('*')[0] if pd.notna(row.get('j_call')) else ''
        
        score = feature_scores.get(f'v_{v_gene}', 0) + feature_scores.get(f'j_{j_gene}', 0)
        results.append((seq, v_gene or 'unknown', j_gene or 'unknown', score))
    return results


def _score_repertoire_vj_positional(args):
    """Score sequences using VJ + positional AA features (for multiprocessing)."""
    df, vj_scores, pos_scores = args
    
    results = []
    if 'junction_aa' not in df.columns:
        return results
    
    for _, row in df.iterrows():
        seq = row.get('junction_aa', '')
        if not isinstance(seq, str) or len(seq) < 8:
            continue
        
        v_gene = str(row.get('v_call', '')).split('*')[0] if pd.notna(row.get('v_call')) else ''
        j_gene = str(row.get('j_call', '')).split('*')[0] if pd.notna(row.get('j_call')) else ''
        
        seq_clean = ''.join(c for c in seq.upper() if c in VALID_AA)
        score = vj_scores.get(f'v_{v_gene}', 0.0) + vj_scores.get(f'j_{j_gene}', 0.0)
        
        if len(seq_clean) >= 10:
            for i, aa in enumerate(seq_clean[:3]):
                score += pos_scores.get(f'pos{i}_{aa}', 0.0)
            for i, aa in enumerate(seq_clean[-3:]):
                score += pos_scores.get(f'pos_end{i}_{aa}', 0.0)
        
        results.append((seq, v_gene or 'unknown', j_gene or 'unknown', score))
    return results


def _score_repertoire_emerson(args):
    """Score sequences using Emerson enrichment (for multiprocessing)."""
    df, enriched_seqs = args
    
    results = []
    if 'junction_aa' not in df.columns:
        return results
    
    for _, row in df.iterrows():
        seq = row.get('junction_aa', '')
        if not isinstance(seq, str) or len(seq) < 8:
            continue
        
        v_gene = str(row.get('v_call', '')).split('*')[0] if pd.notna(row.get('v_call')) else ''
        j_gene = str(row.get('j_call', '')).split('*')[0] if pd.notna(row.get('j_call')) else ''
        
        score = 1.0 if f"{seq}|{v_gene}" in enriched_seqs else 0.0
        results.append((seq, v_gene or 'unknown', j_gene or 'unknown', score))
    return results


def parallel_score_repertoires(data_to_score, score_fn, score_args, n_workers, desc="Scoring"):
    """
    Score repertoires in parallel and return deduplicated DataFrame.
    
    Args:
        data_to_score: List of repertoire data dicts (not used directly, just for counting)
        score_fn: Scoring function to apply in parallel
        score_args: List of argument tuples for score_fn
        n_workers: Number of parallel workers
        desc: Description for progress bar
    
    Returns:
        DataFrame with columns: junction_aa, v_call, j_call, importance_score
    """
    if n_workers <= 1:
        all_results = []
        for args in tqdm(score_args, desc=f"  {desc}"):
            all_results.extend(score_fn(args))
    else:
        print(f"  Using {n_workers} parallel workers for scoring...")
        with Pool(n_workers) as pool:
            results_list = list(tqdm(pool.imap(score_fn, score_args), total=len(score_args), desc=f"  {desc}"))
        all_results = [r for results in results_list for r in results]
    
    if not all_results:
        return pd.DataFrame()
    
    # Deduplicate and average scores for sequences appearing in multiple repertoires
    seq_scores = defaultdict(lambda: {'score': 0.0, 'count': 0})
    for seq, v, j, score in all_results:
        seq_scores[(seq, v, j)]['score'] += score
        seq_scores[(seq, v, j)]['count'] += 1
    
    return pd.DataFrame([
        {'junction_aa': seq, 'v_call': v, 'j_call': j, 'importance_score': d['score']/d['count']}
        for (seq, v, j), d in seq_scores.items()
    ])


# =============================================================================
# Task 2 Configuration: Model Selection
# =============================================================================
# Environment variables (typically set by bash script):
#   TASK2_MODEL_OVERRIDE - Force specific model (e.g., "kmer5_sgd", "gapped_kmer")
#                          If empty/not set, uses best-performing model from Task 1
#   TASK2_USE_TEST_DATA  - Set to "1" to score test sequences (default: "0" for training)
#
# DEFAULTS:
#   - Python trains ALL models (unless AIRR_MODELS limits them)
#   - Python uses BEST model for Task 2 (unless TASK2_MODEL_OVERRIDE is set)
#   - Python uses TRAINING data for Task 2 (unless TASK2_USE_TEST_DATA=1)
#
# Dataset-specific strategies (only used if TASK2_MODEL_OVERRIDE="AUTO"):
# Dataset-specific strategies for Task 2 (from airr_ml_v11_task2_extract_parallel_fixed_check.py)
# These MUST match the v11 script exactly!
TASK2_STRATEGIES = {
    # Dataset 1: vj_positional - VJ gene + positional AA (first 3 + last 3 positions)
    # Requires: VJ model + pos_aa model for positional scoring
    1: {'method': 'vj_positional', 'preferred': ['vj_interact', 'vj', 'pos_aa'], 'fallback': 'kmer4_LR_freq'},
    
    # Dataset 2: kmer4_lr - 4-mer L1 Logistic Regression
    2: {'method': 'kmer4_lr', 'preferred': ['kmer4_LR_freq', 'kmer4_LR_raw'], 'fallback': 'kmer5_sgd'},
    
    # Dataset 3: vj_ensemble - VJ-based ensemble (weak signal dataset)
    # Note: Must train VJ models even though Task 1 uses kmer56!
    3: {'method': 'vj_ensemble', 'preferred': ['vj_interact', 'vj', 'vj_elasticnet'], 'fallback': 'kmer5_sgd'},
    
    # Dataset 4: gapped_kmer - Gapped k-mer (2+1+2) L1 LR
    4: {'method': 'gapped_kmer', 'preferred': ['gapped_kmer'], 'fallback': 'kmer4_LR_freq'},
    
    # Dataset 5: kmer4_lr - 4-mer L1 Logistic Regression
    5: {'method': 'kmer4_lr', 'preferred': ['kmer4_LR_freq', 'kmer4_LR_raw'], 'fallback': 'kmer5_sgd'},
    
    # Dataset 6: emerson - Public sequence enrichment
    6: {'method': 'emerson', 'preferred': ['emerson'], 'fallback': 'kmer4_LR_freq'},
    
    # Dataset 7: emerson (HSV-2) - not in v11 but should use emerson
    7: {'method': 'emerson', 'preferred': ['emerson'], 'fallback': 'kmer5_sgd'},
    
    # Dataset 8: malidvj (T1D) - uses MALIDVJ age-robust model
    8: {'method': 'malidvj', 'preferred': ['malidvj', 'pos_aa'], 'fallback': 'kmer5_sgd'},
}

# Dataset-specific ensemble configurations (from source scripts)
# Format: {'mode': str, 'min_auc': float, 'auc_margin': float, 'task1_restrict': list or None, 'skip_kmer4_rf': bool, 'emerson_config': dict}
# Modes:
#   'dataset1_vj_kmer' - Combine best VJ + best kmer (L2 meta-learner, C=1.0 fixed)
#   'v10_floor_only'   - v10 style: pure floor threshold, no margin (L1 meta-learner)
#   'standard'         - Threshold-based selection with margin (L1 meta-learner)  
#   'none'             - No ensemble, pick single best model
#   'emerson_stacking' - Stacking over Emerson configs (L2 meta-learner)
#
# task1_restrict: If set, Task 1 will ONLY consider models matching these prefixes
#   None = consider all trained models
#   ['kmer5', 'kmer6'] = only consider kmer5_sgd, kmer6_sgd for Task 1 selection
#   ['emerson'] = only consider emerson models
#   ['malidvj'] = only consider malidvj model
#
# skip_kmer4_rf: If True, don't train kmer4_RF (v10 doesn't include RF for datasets 2,4)
#
# emerson_config: Dataset-specific Emerson parameters (p_thresholds, min_counts)
#   Dataset 7 uses [0.10, 0.12, 0.14] and min_counts [2, 3] (from airr_ml_dataset7_emerson_cv_0912.py)
#   Others use v10 defaults [0.05, 0.1] and min_count 2
#
# dataset1_config: Dataset 1 specific parameters (from 1_malid_tcr_multi_vj_new.py)
#   - c_values: [1.0, 0.2, 0.1, 0.05, 0.03] (NOT [5, 1, 0.2, 0.1, 0.05, 0.03])
#   - meta_c: 1.0 (fixed, not tuned)
#
# v10 thresholds (from 2456_airr_ml_v10_fixechd.py):
#   Dataset 2: threshold 0.55, models do NOT include kmer4_RF
#   Dataset 4: threshold 0.60, models do NOT include kmer4_RF
#   Dataset 5: threshold 0.55
#   Dataset 6: threshold 0.90
DATASET_ENSEMBLE_CONFIGS = {
    # Dataset 1: From 1_malid_tcr_multi_vj_new.py - always combine best VJ + best kmer
    # Uses C_VALUES = [1.0, 0.2, 0.1, 0.05, 0.03] and meta-learner L2 C=1.0 (fixed)
    # Task 1 uses ALL models (vj + kmer combined)
    1: {'mode': 'dataset1_vj_kmer', 'min_auc': 0.50, 'auc_margin': 1.0, 'task1_restrict': None, 'skip_kmer4_rf': False,
        'dataset1_config': {'c_values': [1.0, 0.2, 0.1, 0.05, 0.03], 'meta_c': 1.0}},
    # Dataset 2: From 2456_airr_ml_v10_fixechd.py - threshold 0.55, NO MARGIN (floor only), NO kmer4_RF
    2: {'mode': 'v10_floor_only', 'min_auc': 0.55, 'auc_margin': 0.0, 'task1_restrict': None, 'skip_kmer4_rf': True},
    # Dataset 3: From 3_kmer_5_6_only.py - NO ENSEMBLE, just pick best 5/6-mer
    # Task 1 ONLY considers kmer5_sgd and kmer6_sgd (VJ models trained for Task 2 only!)
    3: {'mode': 'none', 'min_auc': 0.50, 'auc_margin': 0.02, 'task1_restrict': ['kmer5', 'kmer6'], 'skip_kmer4_rf': False},
    # Dataset 4: From 2456_airr_ml_v10_fixechd.py - threshold 0.60, NO MARGIN (floor only), NO kmer4_RF
    4: {'mode': 'v10_floor_only', 'min_auc': 0.60, 'auc_margin': 0.0, 'task1_restrict': None, 'skip_kmer4_rf': True},
    # Dataset 5: From 2456_airr_ml_v10_fixechd.py - threshold 0.55, NO MARGIN (floor only)
    5: {'mode': 'v10_floor_only', 'min_auc': 0.55, 'auc_margin': 0.0, 'task1_restrict': None, 'skip_kmer4_rf': False},
    # Dataset 6: From 2456_airr_ml_v10_fixechd.py - threshold 0.90, NO MARGIN (floor only)
    6: {'mode': 'v10_floor_only', 'min_auc': 0.90, 'auc_margin': 0.0, 'task1_restrict': None, 'skip_kmer4_rf': False},
    # Dataset 7: From airr_ml_dataset7_emerson_cv_0912.py - stacking ensemble over Emerson configs
    # Uses DIFFERENT p_thresholds and min_counts than v10!
    # Task 1 ONLY considers emerson models
    # FIXED: p_thresholds must be decimals (0.10, 0.12, 0.14), not integers!
    7: {'mode': 'emerson_stacking', 'min_auc': 0.50, 'auc_margin': 0.05, 'task1_restrict': ['emerson'], 'skip_kmer4_rf': False,
        'emerson_config': {'p_thresholds': [0.10, 0.12, 0.14], 'min_counts': [2, 3]}},
    # Dataset 8: From t1d_model1_vj_positional_fixed.py - NO ENSEMBLE, single malidvj
    # Task 1 ONLY considers malidvj model
    8: {'mode': 'none', 'min_auc': 0.50, 'auc_margin': 0.02, 'task1_restrict': ['malidvj'], 'skip_kmer4_rf': False},
}


# =============================================================================
# Utility Functions (Competition Template)
# =============================================================================

def load_data_generator(data_dir: str, metadata_filename='metadata.csv') -> Iterator[
    Union[Tuple[str, pd.DataFrame, bool], Tuple[str, pd.DataFrame]]]:
    """Generator to load immune repertoire data (competition template)."""
    metadata_path = os.path.join(data_dir, metadata_filename)

    if os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)
        for row in metadata_df.itertuples(index=False):
            file_path = os.path.join(data_dir, row.filename)
            try:
                repertoire_df = pd.read_csv(file_path, sep='\t')
                yield row.repertoire_id, repertoire_df, row.label_positive
            except FileNotFoundError:
                continue
    else:
        search_pattern = os.path.join(data_dir, '*.tsv')
        tsv_files = glob.glob(search_pattern)
        for file_path in sorted(tsv_files):
            try:
                filename = os.path.basename(file_path)
                repertoire_df = pd.read_csv(file_path, sep='\t')
                yield filename.replace('.tsv', ''), repertoire_df
            except Exception:
                continue


def get_repertoire_ids(data_dir: str) -> list:
    """Retrieves repertoire IDs from metadata or filenames."""
    metadata_path = os.path.join(data_dir, 'metadata.csv')
    if os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)
        return metadata_df['repertoire_id'].tolist()
    else:
        search_pattern = os.path.join(data_dir, '*.tsv')
        tsv_files = glob.glob(search_pattern)
        return [os.path.basename(f).replace('.tsv', '') for f in sorted(tsv_files)]


def save_tsv(df: pd.DataFrame, path: str):
    """Save DataFrame to TSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep='\t', index=False)


def validate_dirs_and_files(train_dir: str, test_dirs: List[str], out_dir: str) -> None:
    """Validate input/output directories."""
    assert os.path.isdir(train_dir), f"Train directory `{train_dir}` does not exist."
    train_tsvs = glob.glob(os.path.join(train_dir, "*.tsv"))
    assert train_tsvs, f"No .tsv files found in train directory `{train_dir}`."
    metadata_path = os.path.join(train_dir, "metadata.csv")
    assert os.path.isfile(metadata_path), f"`metadata.csv` not found in train directory `{train_dir}`."

    for test_dir in test_dirs:
        assert os.path.isdir(test_dir), f"Test directory `{test_dir}` does not exist."
        test_tsvs = glob.glob(os.path.join(test_dir, "*.tsv"))
        assert test_tsvs, f"No .tsv files found in test directory `{test_dir}`."

    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception as e:
        print(f"Failed to create output directory `{out_dir}`: {e}")
        sys.exit(1)


def concatenate_output_files(out_dir: str) -> None:
    """Concatenates all output TSV files into submissions.csv."""
    predictions_pattern = os.path.join(out_dir, '*_test_predictions.tsv')
    sequences_pattern = os.path.join(out_dir, '*_important_sequences.tsv')

    predictions_files = sorted(glob.glob(predictions_pattern))
    sequences_files = sorted(glob.glob(sequences_pattern))

    df_list = []
    for pred_file in predictions_files:
        try:
            df = pd.read_csv(pred_file, sep='\t')
            df_list.append(df)
        except Exception:
            continue

    for seq_file in sequences_files:
        try:
            df = pd.read_csv(seq_file, sep='\t')
            df_list.append(df)
        except Exception:
            continue

    if df_list:
        concatenated_df = pd.concat(df_list, ignore_index=True)
    else:
        concatenated_df = pd.DataFrame(
            columns=['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call'])
    
    submissions_file = os.path.join(out_dir, 'submissions.csv')
    concatenated_df.to_csv(submissions_file, index=False)
    print(f"Concatenated output written to `{submissions_file}`.")


def get_dataset_pairs(train_dir: str, test_dir: str) -> List[Tuple[str, List[str]]]:
    """Returns list of (train_path, [test_paths]) tuples for dataset pairs."""
    test_groups = defaultdict(list)
    for test_name in sorted(os.listdir(test_dir)):
        if test_name.startswith("test_dataset_"):
            base_id = test_name.replace("test_dataset_", "").split("_")[0]
            test_groups[base_id].append(os.path.join(test_dir, test_name))

    pairs = []
    for train_name in sorted(os.listdir(train_dir)):
        if train_name.startswith("train_dataset_"):
            train_id = train_name.replace("train_dataset_", "")
            train_path = os.path.join(train_dir, train_name)
            pairs.append((train_path, test_groups.get(train_id, [])))

    return pairs


# =============================================================================
# Memory Helpers
# =============================================================================

def get_memory_mb() -> float:
    if HAS_PSUTIL:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    return 0.0


# =============================================================================
# Sparse Matrix Utilities
# =============================================================================

def should_use_sparse(n_features: int, n_samples: int = None, density: float = None) -> bool:
    if n_features > CFG.SPARSE_THRESHOLD:
        return True
    if density is not None and density < CFG.SPARSE_DENSITY_THRESHOLD:
        return True
    return False


def sparse_to_dense(X: Union[np.ndarray, sparse.spmatrix]) -> np.ndarray:
    return X.toarray() if sparse.issparse(X) else X


class SparseStandardScaler:
    """StandardScaler that works with both sparse and dense matrices."""
    
    def __init__(self, with_mean: bool = True, with_std: bool = True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_ = None
        self.std_ = None
        self.scale_ = None
        self._is_sparse = False
    
    def fit(self, X):
        self._is_sparse = sparse.issparse(X)
        if self._is_sparse:
            X_sample = X.toarray() if X.shape[0] < 1000 else X[:1000].toarray()
            self.mean_ = np.zeros(X.shape[1])
            self.std_ = np.std(X_sample, axis=0)
            self.std_[self.std_ == 0] = 1.0
            self.scale_ = self.std_
        else:
            self.mean_ = np.mean(X, axis=0)
            self.std_ = np.std(X, axis=0)
            self.std_[self.std_ == 0] = 1.0
            self.scale_ = self.std_
        return self
    
    def transform(self, X):
        if sparse.issparse(X):
            return X.multiply(1.0 / self.scale_).tocsr()
        else:
            X_out = X - self.mean_ if self.with_mean else X.copy()
            return X_out / self.scale_ if self.with_std else X_out
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


# =============================================================================
# Mal-ID Preprocessing
# =============================================================================

def preprocess_repertoire(df: pd.DataFrame, deduplicate: bool = True, max_seqs: int = None) -> pd.DataFrame:
    """Preprocess repertoire following Mal-ID official approach."""
    if deduplicate is None:
        deduplicate = CFG.DEDUPLICATE_BY_CDR3
    if max_seqs is None:
        max_seqs = CFG.MAX_SEQS_PER_SAMPLE
    df = df.copy()
    
    if 'junction_aa' not in df.columns:
        for col in ['cdr3_aa', 'CDR3_aa', 'aminoAcid']:
            if col in df.columns:
                df['junction_aa'] = df[col]
                break
    if 'junction_aa' not in df.columns:
        return df
    
    if 'duplicate_count' not in df.columns:
        for col in ['templates', 'count', 'clone_count']:
            if col in df.columns:
                df['duplicate_count'] = df[col]
                break
        else:
            df['duplicate_count'] = 1
    df['duplicate_count'] = pd.to_numeric(df['duplicate_count'], errors='coerce').fillna(1).astype(int)
    
    for orig, new in [('v_gene', 'v_call'), ('j_gene', 'j_call')]:
        if new not in df.columns and orig in df.columns:
            df[new] = df[orig]
    
    if deduplicate and 'junction_aa' in df.columns:
        group_cols = ['junction_aa']
        if 'v_call' in df.columns:
            group_cols.append('v_call')
        agg_dict = {'duplicate_count': 'sum'}
        if 'j_call' in df.columns:
            agg_dict['j_call'] = 'first'
        df = df.groupby(group_cols, as_index=False).agg(agg_dict)
    
    if max_seqs is not None and len(df) > max_seqs:
        df = df.nlargest(max_seqs, 'duplicate_count')
    return df


# =============================================================================
# Data Loading (Parallel Support)
# =============================================================================

def load_rep(path, preprocess=False):
    """Load a single repertoire file."""
    df = pd.read_csv(path, sep='\t', low_memory=False)
    for o, n in [('cdr3_aa', 'junction_aa'), ('v_gene', 'v_call'), ('j_gene', 'j_call')]:
        if o in df.columns and n not in df.columns:
            df = df.rename(columns={o: n})
    if preprocess:
        df = preprocess_repertoire(df)
    if 'junction_aa' in df.columns:
        df = df[df['junction_aa'].notna() & df['junction_aa'].apply(
            lambda x: isinstance(x, str) and len(x) >= 8)]
    return df


def _load_single_repertoire(task: Dict) -> Optional[Dict]:
    """Worker function for parallel loading."""
    filepath = task['filepath']
    preprocess = task.get('preprocess', False)
    try:
        df = load_rep(filepath, preprocess=preprocess)
        if len(df) == 0:
            return None
        return {
            'rep_id': task['rep_id'],
            'label': task.get('label', -1),
            'df': df,
            'filename': task.get('filename', '')
        }
    except Exception:
        return None


def load_train_data(train_dir: str, preprocess: bool = False, n_workers: int = None) -> Tuple[List[Dict], pd.DataFrame]:
    """Load training data with parallel support."""
    mp = os.path.join(train_dir, 'metadata.csv')
    if not os.path.exists(mp):
        return [], pd.DataFrame()
    
    meta = pd.read_csv(mp)
    if n_workers is None:
        n_workers = CFG.N_WORKERS
    
    tasks = []
    for _, r in meta.iterrows():
        fn = r.get('filename') or f"{r.get('repertoire_id')}.tsv"
        fp = os.path.join(train_dir, fn)
        if not os.path.exists(fp):
            fp = os.path.join(train_dir, 'data', fn)
        if not os.path.exists(fp):
            continue
        lbl = int(r.get('label_positive', r.get('disease_label', 0)) in [True, 1, 'True', 'true'])
        tasks.append({
            'filepath': fp,
            'rep_id': fn.replace('.tsv', ''),
            'label': lbl,
            'filename': fn,
            'preprocess': preprocess
        })
    
    if not tasks:
        return [], meta
    
    if CFG.USE_PARALLEL_LOADING and n_workers > 1 and len(tasks) > 1:
        print(f"  Loading {len(tasks)} repertoires with {n_workers} workers...")
        with Pool(n_workers) as pool:
            results = list(tqdm(pool.imap(_load_single_repertoire, tasks), total=len(tasks), desc="  Loading"))
    else:
        results = [_load_single_repertoire(t) for t in tqdm(tasks, desc="  Loading")]
    
    data = [r for r in results if r is not None]
    return data, meta


def load_test_data(test_dir: str, preprocess: bool = False, n_workers: int = None) -> List[Dict]:
    """Load test data with parallel support."""
    if not os.path.exists(test_dir):
        return []
    
    if n_workers is None:
        n_workers = CFG.N_WORKERS
    
    mp = os.path.join(test_dir, 'metadata.csv')
    if os.path.exists(mp):
        meta = pd.read_csv(mp)
        files = [(r.get('filename') or f"{r.get('repertoire_id')}.tsv") for _, r in meta.iterrows()]
    else:
        files = [f for f in os.listdir(test_dir) if f.endswith('.tsv')]
    
    tasks = []
    for fn in files:
        fp = os.path.join(test_dir, fn)
        if os.path.exists(fp):
            tasks.append({
                'filepath': fp,
                'rep_id': fn.replace('.tsv', ''),
                'filename': fn,
                'preprocess': preprocess
            })
    
    if not tasks:
        return []
    
    if CFG.USE_PARALLEL_LOADING and n_workers > 1 and len(tasks) > 1:
        with Pool(n_workers) as pool:
            results = list(tqdm(pool.imap(_load_single_repertoire, tasks), total=len(tasks), desc="  Loading test"))
    else:
        results = [_load_single_repertoire(t) for t in tasks]
    
    return [r for r in results if r is not None]


# =============================================================================
# Feature Extraction Functions
# =============================================================================

def ext_kmer(df, k=4):
    """Extract k-mer counts."""
    cnt = Counter()
    for seq in df['junction_aa'].dropna():
        if isinstance(seq, str):
            seq = ''.join(c for c in seq.upper() if c in CFG.AA)
            for i in range(len(seq) - k + 1):
                km = seq[i:i + k]
                if all(c in CFG.AA for c in km):
                    cnt[km] += 1
    return dict(cnt)


def ext_pos_kmer(df, k=4):
    """Extract position-tagged k-mers."""
    cnt = Counter()
    for seq in df['junction_aa'].dropna():
        if isinstance(seq, str):
            seq = ''.join(c for c in seq.upper() if c in CFG.AA)
            n = len(seq) - k + 1
            for i in range(n):
                km = seq[i:i + k]
                if all(c in CFG.AA for c in km):
                    pos = 'START' if i < n * 0.33 else ('END' if i > n * 0.67 else 'MID')
                    cnt[f"{pos}_{km}"] += 1
    return dict(cnt)


def ext_gap(df):
    """Extract gapped k-mer patterns."""
    cnt = Counter()
    for seq in df['junction_aa'].dropna():
        if isinstance(seq, str):
            seq = ''.join(c for c in seq.upper() if c in CFG.AA)
            for i in range(len(seq) - 5 + 1):
                cnt[f"{seq[i:i+2]}_X_{seq[i+3:i+5]}"] += 1
    return dict(cnt)


def ext_vj(df):
    """Extract V and J gene usage frequencies."""
    f = {}
    t = len(df)
    if t == 0:
        return f
    if 'v_call' in df.columns:
        for g, c in df['v_call'].dropna().apply(lambda x: str(x).split('*')[0]).value_counts().items():
            if g not in ('', 'nan'):
                f[f'v_{g}'] = c / t
    if 'j_call' in df.columns:
        for g, c in df['j_call'].dropna().apply(lambda x: str(x).split('*')[0]).value_counts().items():
            if g not in ('', 'nan'):
                f[f'j_{g}'] = c / t
    return f


def ext_vj_interact(df, top_v: List[str], top_j: List[str]):
    """Extract V×J interaction features."""
    features = {}
    total = len(df)
    if total == 0 or 'v_call' not in df.columns or 'j_call' not in df.columns:
        return features
    df_vj = df[['v_call', 'j_call']].dropna().copy()
    df_vj['v_gene'] = df_vj['v_call'].apply(lambda x: str(x).split('*')[0])
    df_vj['j_gene'] = df_vj['j_call'].apply(lambda x: str(x).split('*')[0])
    for v in top_v:
        for j in top_j:
            count = len(df_vj[(df_vj['v_gene'] == v) & (df_vj['j_gene'] == j)])
            features[f'vj_{v}_{j}'] = count / total
    return features


def get_top_vj_genes(all_data: List[Dict], top_k: int = 15) -> Tuple[List[str], List[str]]:
    """Get top V and J genes by prevalence."""
    v_counter, j_counter = Counter(), Counter()
    for d in all_data:
        df = d['df']
        if 'v_call' in df.columns:
            v_counter.update(df['v_call'].dropna().apply(lambda x: str(x).split('*')[0]).unique())
        if 'j_call' in df.columns:
            j_counter.update(df['j_call'].dropna().apply(lambda x: str(x).split('*')[0]).unique())
    return [g for g, _ in v_counter.most_common(top_k)], [g for g, _ in j_counter.most_common(top_k)]


def ext_div(df):
    """Extract diversity metrics."""
    if 'junction_aa' not in df.columns:
        return {}
    cnt = df['junction_aa'].value_counts()
    t = cnt.sum()
    if t == 0:
        return {}
    fr = cnt / t
    ent = -np.sum(fr * np.log(fr + 1e-10))
    sim = np.sum(fr ** 2)
    return {
        'richness': len(cnt),
        'log_richness': np.log1p(len(cnt)),
        'shannon': ent,
        'simpson': sim,
        'inv_simpson': 1 / (sim + 1e-10),
        'evenness': ent / np.log(len(cnt)) if len(cnt) > 1 else 0
    }


def ext_pos_aa(df):
    """Extract positional amino acid features."""
    f = {}
    col = 'junction_aa' if 'junction_aa' in df.columns else 'cdr3_aa'
    if col not in df.columns:
        return f
    cdr3s = [s.upper().translate(AA_TR) for s in df[col].dropna().astype(str) 
             if len(s.upper().translate(AA_TR)) >= 8]
    if not cdr3s:
        return f
    for rn, al in [('first3', [s[:3] for s in cdr3s]), 
                   ('last4', [s[-4:] for s in cdr3s]), 
                   ('mid', [s[3:-4] for s in cdr3s if len(s) > 7])]:
        aa = ''.join(al)
        if not aa:
            continue
        cnt = Counter(aa)
        t = sum(cnt.values())
        for a in AA:
            f[f'pos_{rn}_{a}'] = cnt.get(a, 0) / t
    return f


# =============================================================================
# MALIDVJ Model Feature Extraction (T1D-style)
# =============================================================================

def ext_malidvj_vj(df: pd.DataFrame) -> Dict[str, float]:
    """
    Extract V gene and J gene frequencies for MALIDVJ model.
    Exactly as in T1D script.
    """
    features = {}
    
    # V gene frequencies
    v_col = 'v_call' if 'v_call' in df.columns else 'v_gene'
    if v_col in df.columns:
        v_genes = df[v_col].dropna().astype(str)
        v_genes = v_genes.str.split('*').str[0]  # Remove allele
        v_counts = v_genes.value_counts(normalize=True)
        for v, freq in v_counts.items():
            if v not in ('', 'nan', 'unknown'):
                features[f'v_{v}'] = freq
    
    # J gene frequencies
    j_col = 'j_call' if 'j_call' in df.columns else 'j_gene'
    if j_col in df.columns:
        j_genes = df[j_col].dropna().astype(str)
        j_genes = j_genes.str.split('*').str[0]
        j_counts = j_genes.value_counts(normalize=True)
        for j, freq in j_counts.items():
            if j not in ('', 'nan', 'unknown'):
                features[f'j_{j}'] = freq
    
    return features


def ext_malidvj_positional(df: pd.DataFrame) -> Dict[str, float]:
    """
    Extract positional amino acid features for MALIDVJ model.
    Exactly as in T1D script - includes germline_frac and nregion_frac.
    
    Regions:
      - first3: First 3 AA of CDR3 (typically germline-encoded)
      - middle: Middle region (variable, contains key specificity determinants)
      - last4: Last 4 AA of CDR3 (typically germline-encoded)
    """
    features = {}
    
    junction_col = 'junction_aa' if 'junction_aa' in df.columns else 'cdr3_aa'
    if junction_col not in df.columns:
        return features
    
    seqs = df[junction_col].dropna().astype(str)
    cdr3s = []
    for seq in seqs:
        clean = seq.upper().translate(AA_TR)
        if len(clean) >= 8:
            cdr3s.append(clean)
    
    if not cdr3s:
        return features
    
    # Collect AAs from each region
    first3_aa = []
    last4_aa = []
    middle_aa = []
    
    for seq in cdr3s:
        first3_aa.extend(list(seq[:3]))
        last4_aa.extend(list(seq[-4:]))
        if len(seq) > 7:
            middle_aa.extend(list(seq[3:-4]))
    
    # Extract features for each region
    for region_name, aa_list in [('first3', first3_aa), ('last4', last4_aa), ('middle', middle_aa)]:
        if not aa_list:
            continue
        
        aa_counts = Counter(aa_list)
        total = sum(aa_counts.values())
        
        # Individual AA frequencies
        for aa in AA:
            features[f'pos_{region_name}_{aa}'] = aa_counts.get(aa, 0) / total
        
        # Summary features: germline vs N-region enrichment
        germline = sum(aa_counts.get(aa, 0) for aa in 'GALSTV')
        nregion = sum(aa_counts.get(aa, 0) for aa in 'RPHW')
        features[f'pos_{region_name}_germline_frac'] = germline / total
        features[f'pos_{region_name}_nregion_frac'] = nregion / total
    
    return features


def ext_malidvj(df: pd.DataFrame) -> Dict[str, float]:
    """
    Extract combined VJ + Positional features for MALIDVJ model.
    This is the complete T1D-style feature set.
    """
    vj = ext_malidvj_vj(df)
    pos = ext_malidvj_positional(df)
    return {**vj, **pos}


def ext_em_seq(df):
    """Extract unique CDR3+V sequences for Emerson classifier."""
    col = 'junction_aa' if 'junction_aa' in df.columns else 'cdr3_aa'
    if col not in df.columns:
        return set()
    j = df[col].dropna().astype(str)
    j = j[j != '']
    if 'v_call' in df.columns:
        v = df.loc[j.index, 'v_call'].fillna('').astype(str).str.split('*').str[0]
        return set((j + '|' + v).unique())
    return set(j.unique())


# =============================================================================
# Age-Robust Feature Selection (Inside CV - No Leakage)
# =============================================================================

def build_age_lookup(metadata: pd.DataFrame) -> Dict[str, float]:
    """Build a lookup table of ages by repertoire_id."""
    age_lookup = {}
    for _, row in metadata.iterrows():
        rep_id = str(row.get('repertoire_id', row.get('filename', '').replace('.tsv', '')))
        age = row.get('age_at_sampling_years') or row.get('age')
        if pd.notna(age):
            age_lookup[rep_id] = float(age)
    return age_lookup


def compute_age_bin_sample_weights(metadata: pd.DataFrame, all_data: List[Dict], n_bins: int = 5) -> np.ndarray:
    """
    Compute sample weights to balance age distribution across disease labels.
    This helps prevent the model from learning age as a proxy for disease.
    """
    ages, labels = [], []
    for d in all_data:
        rep_id = d['rep_id']
        row = metadata[metadata['repertoire_id'] == rep_id] if 'repertoire_id' in metadata.columns else pd.DataFrame()
        if len(row) > 0:
            age = row.iloc[0].get('age_at_sampling_years') or row.iloc[0].get('age')
            ages.append(float(age) if pd.notna(age) else np.nan)
        else:
            ages.append(np.nan)
        labels.append(d['label'])
    ages, labels = np.array(ages), np.array(labels)
    valid_mask = ~np.isnan(ages)
    if valid_mask.sum() < len(ages) * 0.5:
        print("    WARNING: Too many missing ages, using uniform weights")
        return np.ones(len(all_data))
    valid_ages = ages[valid_mask]
    bins = np.percentile(valid_ages, np.linspace(0, 100, n_bins + 1))
    bins[0], bins[-1] = -np.inf, np.inf
    age_bins = np.digitize(ages, bins) - 1
    age_bins[~valid_mask] = n_bins
    weights = np.ones(len(all_data))
    for bin_idx in range(n_bins + 1):
        for label in [0, 1]:
            mask = (age_bins == bin_idx) & (labels == label)
            if mask.sum() > 0:
                weights[mask] = 1.0 / mask.sum()
    return weights / weights.sum() * len(weights)


def analyze_age_confounding(metadata: pd.DataFrame, all_data: List[Dict]) -> Tuple[float, bool]:
    """
    Analyze confounding between disease status and age.
    Returns (correlation, is_confounded) tuple.
    """
    ages, labels = [], []
    for d in all_data:
        rep_id = d['rep_id']
        row = metadata[metadata['repertoire_id'] == rep_id] if 'repertoire_id' in metadata.columns else pd.DataFrame()
        if len(row) > 0:
            age = row.iloc[0].get('age_at_sampling_years') or row.iloc[0].get('age')
            if pd.notna(age):
                ages.append(float(age))
                labels.append(d['label'])
    
    if len(ages) < 20:
        return 0.0, False
    
    corr, p = spearmanr(ages, labels)
    is_confounded = abs(corr) > 0.2 and p < 0.05
    
    return corr, is_confounded


def select_age_robust_features(train_idx, all_data, feature_names, X, y, age_lookup,
                                age_threshold=None, corr_threshold=None):
    """
    Select features predictive in BOTH young AND old patients (INSIDE CV - no leakage).
    
    This ensures features are not just proxies for age, but genuinely predictive
    of disease status across age groups. This is critical for T1D and other
    age-confounded datasets.
    
    Args:
        train_idx: Indices of training samples (for this CV fold)
        all_data: List of all data dictionaries
        feature_names: List of feature names
        X: Full feature matrix
        y: Full label array  
        age_lookup: Dictionary mapping rep_id to age
        age_threshold: Age cutoff for young vs old (default: 25)
        corr_threshold: Minimum correlation required (default: 0.05)
    
    Returns:
        List of feature indices that are robust across age groups
    """
    if age_threshold is None:
        age_threshold = CFG.AGE_THRESHOLD
    if corr_threshold is None:
        corr_threshold = CFG.CORR_THRESHOLD
    train_data = [all_data[i] for i in train_idx]
    X_train = sparse_to_dense(X[train_idx]) if sparse.issparse(X) else X[train_idx]
    y_train = y[train_idx]
    young_mask = np.array([age_lookup.get(d['rep_id'], 999) <= age_threshold for d in train_data])
    old_mask = np.array([age_lookup.get(d['rep_id'], 0) > age_threshold for d in train_data])
    if young_mask.sum() < 20 or old_mask.sum() < 20:
        return list(range(len(feature_names)))
    robust_features = []
    for j in range(X_train.shape[1]):
        try:
            corr_young, _ = spearmanr(X_train[young_mask, j], y_train[young_mask])
            corr_old, _ = spearmanr(X_train[old_mask, j], y_train[old_mask])
            if np.isnan(corr_young) or np.isnan(corr_old):
                continue
            if np.sign(corr_young) == np.sign(corr_old) and abs(corr_young) > corr_threshold and abs(corr_old) > corr_threshold:
                robust_features.append(j)
        except:
            pass
    return robust_features if robust_features else list(range(len(feature_names)))


# =============================================================================
# Benjamini-Hochberg FDR Correction
# =============================================================================

def bh(pv):
    """Benjamini-Hochberg FDR correction."""
    n = len(pv)
    if n == 0:
        return pv
    idx = np.argsort(pv)
    adj = np.zeros(n)
    adj[-1] = pv[idx[-1]]
    for i in range(n - 2, -1, -1):
        adj[i] = min(adj[i + 1], pv[idx[i]] * n / (i + 1))
    res = np.zeros(n)
    res[idx] = adj
    return np.clip(res, 0, 1)


# =============================================================================
# Emerson Classifier
# =============================================================================

class EmClf:
    """Emerson-style classifier using Fisher's exact test."""
    def __init__(self, p=0.1, mc=2):
        self.p, self.mc = p, mc
        self.seqs, self.clf = set(), None
    
    def fit(self, ss, y):
        pos = [ss[i] for i in range(len(y)) if y[i] == 1]
        neg = [ss[i] for i in range(len(y)) if y[i] == 0]
        pc, nc = Counter(), Counter()
        for s in pos:
            for x in s: pc[x] += 1
        for s in neg:
            for x in s: nc[x] += 1
        cands = [x for x, c in pc.items() if c >= self.mc]
        if not cands:
            return self
        pv, tested = [], []
        np_, nn_ = len(pos), len(neg)
        for seq in cands:
            a, b = pc[seq], np_ - pc[seq]
            c, d = nc.get(seq, 0), nn_ - nc.get(seq, 0)
            _, p = fisher_exact([[a, b], [c, d]], alternative='greater')
            pv.append(p)
            tested.append(seq)
        adj = bh(np.array(pv))
        self.seqs = set(np.array(tested)[adj < self.p])
        if self.seqs:
            X = np.array([[len(s & self.seqs)] for s in ss])
            self.clf = LogisticRegression(random_state=CFG.SEED, max_iter=1000)
            self.clf.fit(X, y)
        return self
    
    def predict_proba(self, ss):
        if not self.seqs or self.clf is None:
            return np.full(len(ss), 0.5)
        X = np.array([[len(s & self.seqs)] for s in ss])
        return self.clf.predict_proba(X)[:, 1]


# =============================================================================
# K-mer 5/6 SGD Model
# =============================================================================

class KmSGD:
    """K-mer frequency classifier using SGDClassifier with sparse support.
    
    NOTE: Uses CFG.SEED_KMER56 (123) to match kmer_5_6_only.py exactly.
    """
    
    def __init__(self, k=5):
        self.k = k
        self.sc, self.m, self.cal = None, None, None
        self.fn, self.alpha, self.auc = None, None, None
        self._use_sparse = False
        # Use CFG.SEED_KMER56 to match kmer_5_6_only.py (which uses RANDOM_STATE=123)
        self._seed = CFG.SEED_KMER56
    
    def _enc(self, data, fit=False):
        feats = []
        for d in data:
            cnt, t = Counter(), 0
            for seq in d['df']['junction_aa'].dropna():
                if isinstance(seq, str) and len(seq) >= self.k:
                    for i in range(len(seq) - self.k + 1):
                        km = seq[i:i + self.k]
                        if all(c in CFG.AA for c in km):
                            cnt[km] += 1
                            t += 1
            feats.append({'id': d['rep_id'], **{k: v / t if t > 0 else 0 for k, v in cnt.items()}})
        df = pd.DataFrame(feats).fillna(0).set_index('id')
        if fit:
            self.fn = df.columns.tolist()
            self._use_sparse = should_use_sparse(len(self.fn))
        return df
    
    def _to_matrix(self, df):
        if self._use_sparse:
            feature_idx = {f: i for i, f in enumerate(self.fn)}
            X = sparse.lil_matrix((len(df), len(self.fn)), dtype=np.float32)
            for i, (idx, row) in enumerate(df.iterrows()):
                for feat in self.fn:
                    val = row.get(feat, 0)
                    if val > 0:
                        X[i, feature_idx[feat]] = val
            return X.tocsr()
        return df.reindex(columns=self.fn, fill_value=0).values
    
    def fit(self, data, y):
        df = self._enc(data, fit=True)
        X = self._to_matrix(df)
        self.sc = MaxAbsScaler()
        Xs = self.sc.fit_transform(X)
        cv = StratifiedKFold(n_splits=CFG.CV_FOLDS, shuffle=True, random_state=self._seed)
        best = -np.inf
        for a in CFG.SGD_ALPHA_VALUES:
            m = SGDClassifier(loss='log_loss', penalty='elasticnet', alpha=a, l1_ratio=0.15,
                              max_iter=CFG.SGD_MAX_ITER, tol=CFG.SGD_TOL,
                              early_stopping=CFG.SGD_EARLY_STOPPING,
                              n_iter_no_change=CFG.SGD_N_ITER_NO_CHANGE,
                              validation_fraction=CFG.SGD_VALIDATION_FRACTION,
                              random_state=self._seed, class_weight='balanced')
            try:
                sc = cross_val_score(m, Xs, y, cv=cv, scoring='roc_auc', n_jobs=-1)
                if sc.mean() > best:
                    best, self.alpha = sc.mean(), a
            except:
                pass
        self.auc = best
        self.m = SGDClassifier(loss='log_loss', penalty='elasticnet', alpha=self.alpha, l1_ratio=0.15,
                               max_iter=CFG.SGD_MAX_ITER, tol=CFG.SGD_TOL,
                               early_stopping=False,
                               random_state=self._seed, class_weight='balanced')
        self.m.fit(Xs, y)
        self.cal = CalibratedClassifierCV(self.m, method='sigmoid', cv=3)
        self.cal.fit(Xs, y)
        return best
    
    def predict(self, data):
        df = self._enc(data)
        X = self._to_matrix(df)
        return self.cal.predict_proba(self.sc.transform(X))[:, 1]
    
    def get_cv_predictions(self, data, y):
        """
        Get out-of-fold cross-validation predictions for ensemble stacking.
        This allows the SGD models to participate in the meta-learner ensemble.
        
        NOTE: Uses SEED=123 to match kmer_5_6_only.py exactly.
        """
        df = self._enc(data, fit=True)
        X = self._to_matrix(df)
        self.sc = MaxAbsScaler()
        Xs = self.sc.fit_transform(X)
        
        cv = StratifiedKFold(n_splits=CFG.CV_FOLDS, shuffle=True, random_state=self._seed)
        cv_preds = np.zeros(len(y))
        
        # First find best alpha (same as fit)
        best = -np.inf
        for a in CFG.SGD_ALPHA_VALUES:
            m = SGDClassifier(loss='log_loss', penalty='elasticnet', alpha=a, l1_ratio=0.15,
                              max_iter=CFG.SGD_MAX_ITER, tol=CFG.SGD_TOL,
                              early_stopping=CFG.SGD_EARLY_STOPPING,
                              n_iter_no_change=CFG.SGD_N_ITER_NO_CHANGE,
                              validation_fraction=CFG.SGD_VALIDATION_FRACTION,
                              random_state=self._seed, class_weight='balanced')
            try:
                sc = cross_val_score(m, Xs, y, cv=cv, scoring='roc_auc', n_jobs=-1)
                if sc.mean() > best:
                    best, self.alpha = sc.mean(), a
            except:
                pass
        
        self.auc = best
        
        # Now get OOF predictions with the best alpha
        for train_idx, val_idx in cv.split(Xs, y):
            # Train model on fold
            fold_model = SGDClassifier(loss='log_loss', penalty='elasticnet', alpha=self.alpha, l1_ratio=0.15,
                                       max_iter=CFG.SGD_MAX_ITER, tol=CFG.SGD_TOL,
                                       early_stopping=False,
                                       random_state=self._seed, class_weight='balanced')
            fold_model.fit(Xs[train_idx], y[train_idx])
            
            # Calibrate on training fold
            calibrated = CalibratedClassifierCV(fold_model, method='sigmoid', cv=3)
            calibrated.fit(Xs[train_idx], y[train_idx])
            
            # Predict on validation fold
            cv_preds[val_idx] = calibrated.predict_proba(Xs[val_idx])[:, 1]
        
        # Fit final model on all data (same as fit method)
        self.m = SGDClassifier(loss='log_loss', penalty='elasticnet', alpha=self.alpha, l1_ratio=0.15,
                               max_iter=CFG.SGD_MAX_ITER, tol=CFG.SGD_TOL,
                               early_stopping=False,
                               random_state=self._seed, class_weight='balanced')
        self.m.fit(Xs, y)
        self.cal = CalibratedClassifierCV(self.m, method='sigmoid', cv=3)
        self.cal.fit(Xs, y)
        
        return cv_preds, best
    
    def score_sequences(self, data):
        if self.m is None:
            return None
        all_seqs = []
        for d in data:
            df = d['df']
            if 'junction_aa' in df.columns:
                cols = ['junction_aa']
                if 'v_call' in df.columns:
                    cols.append('v_call')
                if 'j_call' in df.columns:
                    cols.append('j_call')
                all_seqs.append(df[cols].drop_duplicates())
        if not all_seqs:
            return None
        unique_seqs = pd.concat(all_seqs).drop_duplicates()
        coefficients = self.m.coef_[0]
        scaler_scale = self.sc.scale_
        adjusted_coefs = coefficients / np.where(scaler_scale == 0, 1, scaler_scale)
        kmer_to_idx = {f: i for i, f in enumerate(self.fn)}
        scores = []
        for seq in unique_seqs['junction_aa']:
            if not isinstance(seq, str) or len(seq) < self.k:
                scores.append(0.0)
                continue
            score = sum(adjusted_coefs[kmer_to_idx[seq[i:i+self.k]]] 
                       for i in range(len(seq) - self.k + 1) if seq[i:i+self.k] in kmer_to_idx)
            scores.append(score)
        unique_seqs = unique_seqs.copy()
        unique_seqs['importance_score'] = scores
        return unique_seqs.nlargest(CFG.TOP_K_SEQUENCES, 'importance_score')


# =============================================================================
# Meta-Learner Ensemble
# =============================================================================

class MetaL:
    """Stacking meta-learner for ensemble predictions."""
    def __init__(self, min_auc=0.55, auc_margin=0.02, mode='standard', task1_restrict=None, dataset1_config=None):
        self.min_auc = min_auc
        self.auc_margin = auc_margin  # Only include models within this margin of best
        self.mode = mode  # 'standard', 'dataset1_vj_kmer', 'none', 'emerson_stacking'
        self.task1_restrict = task1_restrict  # List of model prefixes to consider, or None for all
        self.dataset1_config = dataset1_config  # Dataset 1 specific: {'c_values': [...], 'meta_c': 1.0}
        self.models, self.meta, self.auc = [], None, 0.5
    
    def _filter_by_restriction(self, scores):
        """Filter scores to only include models matching task1_restrict prefixes."""
        if not self.task1_restrict:
            return scores
        
        filtered = {}
        for model_name, auc in scores.items():
            if any(model_name.startswith(prefix) or model_name == prefix for prefix in self.task1_restrict):
                filtered[model_name] = auc
        
        if not filtered:
            print(f"    WARNING: No models match task1_restrict {self.task1_restrict}, using all models")
            return scores
        
        print(f"    [Task 1 restriction active: {self.task1_restrict}]")
        print(f"    [Filtered to {len(filtered)}/{len(scores)} models: {list(filtered.keys())}]")
        return filtered
    
    def fit(self, scores, preds, y):
        # =================================================================
        # NONE MODE: No ensemble, just pick best single model
        # (Matches 3_kmer_5_6_only.py and t1d_model1_vj_positional_fixed.py)
        # =================================================================
        if self.mode == 'none':
            # Apply task1_restrict filter
            filtered_scores = self._filter_by_restriction(scores)
            best_model = max(filtered_scores.keys(), key=lambda x: filtered_scores[x])
            self.models = [best_model]
            self.auc = filtered_scores[best_model]
            print(f"    No ensemble mode: selecting best single model: {best_model} ({self.auc:.4f})")
            return
        
        # =================================================================
        # DATASET 1 MODE: Combine best VJ model + best kmer model
        # (Matches 1_malid_tcr_multi_vj_new.py behavior)
        # =================================================================
        if self.mode == 'dataset1_vj_kmer':
            # Find best VJ model
            vj_models = {n: a for n, a in scores.items() if n.startswith('vj')}
            best_vj = max(vj_models.keys(), key=lambda x: vj_models[x]) if vj_models else None
            
            # Find best kmer model (kmer4_LR variants)
            kmer_models = {n: a for n, a in scores.items() if n.startswith('kmer4_LR')}
            best_kmer = max(kmer_models.keys(), key=lambda x: kmer_models[x]) if kmer_models else None
            
            if best_vj and best_kmer:
                self.models = [best_vj, best_kmer]
                print(f"    Dataset 1 ensemble: {best_vj} ({vj_models[best_vj]:.4f}) + {best_kmer} ({kmer_models[best_kmer]:.4f})")
            elif best_vj:
                self.models = [best_vj]
                print(f"    Dataset 1 ensemble: {best_vj} only (no kmer models)")
                return
            elif best_kmer:
                self.models = [best_kmer]
                print(f"    Dataset 1 ensemble: {best_kmer} only (no VJ models)")
                return
            else:
                self.models = [max(scores.keys(), key=lambda x: scores[x])]
                return
        
        # =================================================================
        # EMERSON STACKING MODE: Stack all emerson models (different p-values)
        # (Matches airr_ml_dataset7_emerson_cv_0912.py behavior)
        # =================================================================
        elif self.mode == 'emerson_stacking':
            # Apply task1_restrict filter (should be ['emerson'] for dataset 7)
            filtered_scores = self._filter_by_restriction(scores)
            
            # For emerson stacking, include all available models (typically just emerson)
            # The original script stacks multiple emerson configs with different p-values
            # In unified script, we only have one emerson model, so just use all models above threshold
            best_auc = max(filtered_scores.values())
            threshold = max(best_auc - 0.1, self.min_auc)  # More permissive for emerson
            
            valid = {n: a for n, a in filtered_scores.items() if a >= threshold}
            print(f"    Emerson stacking: best={best_auc:.4f}, threshold={threshold:.4f}, selected={len(valid)}/{len(filtered_scores)} models")
            
            if not valid:
                self.models = [max(filtered_scores.keys(), key=lambda x: filtered_scores[x])]
                return
            self.models = [n for n, _ in sorted(valid.items(), key=lambda x: -x[1])]
        
        # =================================================================
        # V10_FLOOR_ONLY MODE: Pure floor threshold (matches v10 exactly)
        # No "within X of best" rule - just include all models >= min_auc
        # =================================================================
        elif self.mode == 'v10_floor_only':
            # Apply task1_restrict filter if set
            filtered_scores = self._filter_by_restriction(scores)
            
            # Pure floor threshold - no margin calculation
            # This matches v10's SmartMetaLearner exactly
            valid = {n: a for n, a in filtered_scores.items() if a >= self.min_auc}
            print(f"    v10 floor-only selection: min_auc={self.min_auc:.2f}, selected={len(valid)}/{len(filtered_scores)} models")
            
            if not valid:
                print(f"    WARNING: No models above {self.min_auc} threshold!")
                best_name = max(filtered_scores.keys(), key=lambda x: filtered_scores[x])
                self.models = [best_name]
                return
            self.models = [n for n, _ in sorted(valid.items(), key=lambda x: -x[1])]
        
        # =================================================================
        # STANDARD MODE: Use dynamic AUC threshold (margin from best)
        # =================================================================
        else:
            # Apply task1_restrict filter if set
            filtered_scores = self._filter_by_restriction(scores)
            
            # Dynamic threshold: only include models within auc_margin of the best
            best_auc = max(filtered_scores.values())
            dynamic_threshold = best_auc - self.auc_margin
            # Also enforce minimum AUC floor
            threshold = max(dynamic_threshold, self.min_auc)
            
            valid = {n: a for n, a in filtered_scores.items() if a >= threshold}
            print(f"    Ensemble selection: best={best_auc:.4f}, threshold={threshold:.4f}, selected={len(valid)}/{len(filtered_scores)} models")
            
            if not valid:
                self.models = [max(filtered_scores.keys(), key=lambda x: filtered_scores[x])]
                return
            self.models = [n for n, _ in sorted(valid.items(), key=lambda x: -x[1])]
        
        if len(self.models) < 2:
            return
        
        # Train meta-learner
        X = np.column_stack([preds[n] for n in self.models])
        
        # Dataset 1 and Dataset 7 use 5-fold CV (matching their standalone scripts)
        # Other datasets use 3-fold CV
        if self.mode in ['dataset1_vj_kmer', 'emerson_stacking']:
            n_folds = 5  # Match standalone scripts
        else:
            n_folds = 3
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=CFG.SEED)
        best = 0.5
        
        # Dataset 1 and Emerson stacking use L2 penalty
        # v10_floor_only and standard modes use L1 penalty (matches 2456_airr_ml_v10_fixechd.py)
        if self.mode in ['dataset1_vj_kmer', 'emerson_stacking']:
            penalty, solver = 'l2', 'lbfgs'
        else:
            penalty, solver = 'l1', 'liblinear'
        
        # Dataset 1: Use FIXED C=1.0 (from 1_malid_tcr_multi_vj_new.py)
        # Other modes: Search over C values
        if self.mode == 'dataset1_vj_kmer' and self.dataset1_config:
            meta_c = self.dataset1_config.get('meta_c', 1.0)
            c_values = [meta_c]  # Fixed, not tuned
            print(f"    [Dataset 1 meta-learner: Using FIXED C={meta_c}, 5-fold CV]")
        else:
            c_values = [0.01, 0.1, 1.0]
        
        for C in c_values:
            try:
                aucs = []
                for tr, val in cv.split(X, y):
                    m = LogisticRegression(penalty=penalty, C=C, solver=solver, max_iter=1000, random_state=CFG.SEED)
                    m.fit(X[tr], y[tr])
                    aucs.append(roc_auc_score(y[val], m.predict_proba(X[val])[:, 1]))
                if np.mean(aucs) > best:
                    best = np.mean(aucs)
                    self.meta = LogisticRegression(penalty=penalty, C=C, solver=solver, max_iter=1000, random_state=CFG.SEED)
            except:
                pass
        if self.meta:
            self.meta.fit(X, y)
            self.auc = best
    
    def predict(self, preds):
        av = [m for m in self.models if m in preds]
        if not av:
            return np.full(len(list(preds.values())[0]), 0.5)
        if self.meta and len(av) == len(self.models):
            return self.meta.predict_proba(np.column_stack([preds[n] for n in av]))[:, 1]
        return np.mean([preds[m] for m in av], axis=0)


# =============================================================================
# Training Functions
# =============================================================================

def train_lr(X, y, trans=None, sample_weights=None):
    """Train logistic regression with CV, optionally with sample weights."""
    if trans:
        X = trans(sparse_to_dense(X)) if sparse.issparse(X) else trans(X)
    X = np.nan_to_num(X) if not sparse.issparse(X) else X
    cv = StratifiedKFold(n_splits=CFG.CV_FOLDS, shuffle=True, random_state=CFG.SEED)
    best_auc, best_preds, best_p = 0.5, np.zeros(len(y)), {}
    for C in CFG.C_VALS:
        try:
            preds = np.zeros(len(y))
            for tr, val in cv.split(sparse_to_dense(X), y):
                sc = StandardScaler()
                X_tr_scaled = sparse_to_dense(sc.fit_transform(X[tr]))
                X_val_scaled = sparse_to_dense(sc.transform(X[val]))
                m = LogisticRegression(penalty='l1', C=C, solver='liblinear', random_state=CFG.SEED, max_iter=1000)
                if sample_weights is not None:
                    m.fit(X_tr_scaled, y[tr], sample_weight=sample_weights[tr])
                else:
                    m.fit(X_tr_scaled, y[tr])
                preds[val] = m.predict_proba(X_val_scaled)[:, 1]
            auc = roc_auc_score(y, preds)
            if auc > best_auc:
                best_auc, best_preds, best_p = auc, preds, {'C': C}
        except:
            pass
    return best_auc, best_preds, best_p


def train_vj_elasticnet(X, y, sample_weights=None):
    """
    Train VJ elasticnet model matching v10 EXACTLY.
    
    Uses:
    - penalty='elasticnet' with saga solver
    - L1_ratios from CFG.ELASTICNET_L1_RATIOS
    - max_iter=2000
    """
    X = np.nan_to_num(X) if not sparse.issparse(X) else X
    cv = StratifiedKFold(n_splits=CFG.CV_FOLDS, shuffle=True, random_state=CFG.SEED)
    best_auc, best_preds = 0.5, np.zeros(len(y))
    best_params = {'C': 0.1, 'l1_ratio': 0.5}
    
    for C in CFG.C_VALS:
        for l1_ratio in CFG.ELASTICNET_L1_RATIOS:
            try:
                preds = np.zeros(len(y))
                for tr, val in cv.split(X, y):
                    sc = StandardScaler()
                    X_tr_scaled = sc.fit_transform(X[tr])
                    X_val_scaled = sc.transform(X[val])
                    m = LogisticRegression(
                        penalty='elasticnet', C=C, l1_ratio=l1_ratio,
                        solver='saga', max_iter=2000, random_state=CFG.SEED
                    )
                    if sample_weights is not None:
                        m.fit(X_tr_scaled, y[tr], sample_weight=sample_weights[tr])
                    else:
                        m.fit(X_tr_scaled, y[tr])
                    preds[val] = m.predict_proba(X_val_scaled)[:, 1]
                auc = roc_auc_score(y, preds)
                if auc > best_auc:
                    best_auc, best_preds = auc, preds.copy()
                    best_params = {'C': C, 'l1_ratio': l1_ratio}
            except:
                pass
    
    return best_auc, best_preds, best_params


def train_lr_age_robust(X, y, all_data, feature_names, age_lookup, sample_weights=None):
    """
    Train logistic regression with age-robust feature selection INSIDE CV loop.
    This prevents data leakage from feature selection.
    
    Args:
        X: Feature matrix
        y: Labels
        all_data: List of data dictionaries
        feature_names: List of feature names
        age_lookup: Dictionary mapping rep_id to age
        sample_weights: Optional sample weights
        
    Returns:
        (best_auc, best_preds, best_params) tuple
    """
    X = np.nan_to_num(sparse_to_dense(X)) if sparse.issparse(X) else np.nan_to_num(X)
    cv = StratifiedKFold(n_splits=CFG.CV_FOLDS, shuffle=True, random_state=CFG.SEED)
    best_auc, best_preds, best_p = 0.5, np.zeros(len(y)), {}
    n_features_per_fold = []
    
    for C in CFG.C_VALS:
        try:
            preds = np.zeros(len(y))
            fold_n_features = []
            
            for tr, val in cv.split(X, y):
                # Age-robust feature selection using ONLY training fold
                robust_idx = select_age_robust_features(
                    tr, all_data, feature_names, X, y, age_lookup
                )
                fold_n_features.append(len(robust_idx))
                
                X_fold = X[:, robust_idx]
                
                sc = StandardScaler()
                X_tr_scaled = sc.fit_transform(X_fold[tr])
                X_val_scaled = sc.transform(X_fold[val])
                
                m = LogisticRegression(penalty='l1', C=C, solver='liblinear', 
                                       random_state=CFG.SEED, max_iter=1000)
                if sample_weights is not None:
                    m.fit(X_tr_scaled, y[tr], sample_weight=sample_weights[tr])
                else:
                    m.fit(X_tr_scaled, y[tr])
                preds[val] = m.predict_proba(X_val_scaled)[:, 1]
            
            auc = roc_auc_score(y, preds)
            if auc > best_auc:
                best_auc, best_preds, best_p = auc, preds, {'C': C}
                n_features_per_fold = fold_n_features
        except:
            pass
    
    if n_features_per_fold:
        best_p['n_features_mean'] = np.mean(n_features_per_fold)
        best_p['n_features_std'] = np.std(n_features_per_fold)
    
    return best_auc, best_preds, best_p


def train_malidvj(all_data: List[Dict], y: np.ndarray, 
                  has_age_data: bool = False, age_lookup: Dict[str, float] = None,
                  sample_weights: np.ndarray = None) -> Tuple[float, np.ndarray, Dict]:
    """
    Train MALIDVJ model: VJ + Positional AA features (T1D-style).
    
    This is the exact model from the T1D script with:
    - L2 regularization (not L1)
    - class_weight='balanced'
    - Age-robust feature selection when age data available
    - 3-fold CV (matching T1D script exactly via CFG.CV_FOLDS_T1D)
    
    Args:
        all_data: List of data dictionaries
        y: Labels
        has_age_data: Whether age data is available
        age_lookup: Dictionary mapping rep_id to age
        sample_weights: Optional sample weights
        
    Returns:
        (cv_auc, cv_preds, model_info) tuple
    """
    from sklearn.feature_extraction import DictVectorizer
    
    # Extract features for all samples
    features = []
    for d in all_data:
        f = ext_malidvj(d['df'])
        features.append(f)
    
    # Build feature matrix using DictVectorizer (exactly as T1D script)
    vec = DictVectorizer(sparse=False)
    X = vec.fit_transform(features)
    try:
        feature_names = list(vec.get_feature_names_out())
    except AttributeError:
        feature_names = list(vec.get_feature_names())
    
    # CV predictions with age-robust filtering INSIDE the loop (NO LEAKAGE)
    # Using CFG.CV_FOLDS_T1D (3-fold) to match T1D script exactly
    cv = StratifiedKFold(n_splits=CFG.CV_FOLDS_T1D, shuffle=True, random_state=CFG.SEED)
    cv_preds = np.zeros(len(y))
    fold_aucs = []
    n_features_per_fold = []
    
    for fold_idx, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
        # Age-robust filtering using ONLY training fold data
        if has_age_data and age_lookup:
            robust_idx = select_age_robust_features(
                tr_idx, all_data, feature_names, X, y, age_lookup
            )
            X_fold = X[:, robust_idx]
            n_features_per_fold.append(len(robust_idx))
        else:
            X_fold = X
            n_features_per_fold.append(len(feature_names))
        
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_fold[tr_idx])
        X_val = scaler.transform(X_fold[val_idx])
        
        # L2 regularization with class_weight='balanced' (exactly as T1D)
        model = LogisticRegression(
            penalty='l2', C=1.0, max_iter=1000,
            class_weight='balanced', random_state=CFG.SEED
        )
        
        if sample_weights is not None:
            model.fit(X_tr, y[tr_idx], sample_weight=sample_weights[tr_idx])
        else:
            model.fit(X_tr, y[tr_idx])
        
        preds = model.predict_proba(X_val)[:, 1]
        cv_preds[val_idx] = preds
        try:
            fold_aucs.append(roc_auc_score(y[val_idx], preds))
        except:
            pass
    
    try:
        cv_auc = roc_auc_score(y, cv_preds)
    except:
        cv_auc = 0.5
    
    # =================================================================
    # TRAIN FINAL MODEL ON ALL DATA (for prediction)
    # =================================================================
    # Use age-robust filtering on ALL training data
    if has_age_data and age_lookup:
        all_idx = np.arange(len(all_data))
        final_robust_idx = select_age_robust_features(
            all_idx, all_data, feature_names, X, y, age_lookup
        )
        X_final = X[:, final_robust_idx]
    else:
        X_final = X
        final_robust_idx = list(range(len(feature_names)))
    
    final_scaler = StandardScaler()
    X_final_scaled = final_scaler.fit_transform(X_final)
    
    final_model = LogisticRegression(
        penalty='l2', C=1.0, max_iter=1000,
        class_weight='balanced', random_state=CFG.SEED
    )
    if sample_weights is not None:
        final_model.fit(X_final_scaled, y, sample_weight=sample_weights)
    else:
        final_model.fit(X_final_scaled, y)
    
    # Store model info for prediction (INCLUDING trained model!)
    model_info = {
        'vectorizer': vec,
        'feature_names': feature_names,
        'has_age_data': has_age_data,
        'cv_auc': cv_auc,
        'n_features_mean': np.mean(n_features_per_fold) if n_features_per_fold else len(feature_names),
        'n_features_std': np.std(n_features_per_fold) if n_features_per_fold else 0,
        # NEW: Store trained model components for fast prediction
        'final_model': final_model,
        'final_scaler': final_scaler,
        'final_robust_idx': final_robust_idx
    }
    
    return cv_auc, cv_preds, model_info


def build_feature_matrix(data: List[Dict], cache: Dict[str, Dict], 
                         feature_list: List[str], normalize: bool = True) -> np.ndarray:
    """Build feature matrix from cached features."""
    n_samples = len(data)
    n_features = len(feature_list)
    X = np.zeros((n_samples, n_features), dtype=np.float32)
    feature_idx = {f: i for i, f in enumerate(feature_list)}
    
    for i, d in enumerate(data):
        f = cache[d['rep_id']]
        total = sum(f.values()) + 1 if normalize else 1
        for feat, count in f.items():
            if feat in feature_idx:
                X[i, feature_idx[feat]] = count / total if normalize else count
    return X


# =============================================================================
# ImmuneStatePredictor Class (Competition Template)
# =============================================================================

class ImmuneStatePredictor:
    """
    Competition-compliant predictor class that wraps all unified models.
    
    FIXED in v6:
    - Age-robust feature selection actually used when age data available
    - Age-robust sample weighting actually used when age data available
    - Feature selection done inside CV loop (no leakage)
    - Fixed ensemble prediction path
    - Fixed positional k-mer prediction path
    - Added VJ interaction model
    - Added positional AA features
    """

    def __init__(self, n_jobs: int = 1, device: str = 'cpu', **kwargs):
        """Initialize the predictor."""
        total_cores = os.cpu_count()
        if n_jobs == -1:
            self.n_jobs = total_cores
        else:
            self.n_jobs = min(n_jobs, total_cores)
        self.device = device
        
        # Set global config based on n_jobs
        CFG.N_WORKERS = self.n_jobs
        CFG.USE_PARALLEL_LOADING = self.n_jobs > 1
        
        # Model storage
        self.model = None
        self.best_model_name = None
        self.model_info = {}
        self.caches = {}
        self.train_data = None
        self.metadata = None
        self.important_sequences_ = None
        self.dataset_name = None
        
        # NEW: Cache test data for Task 2 scoring
        self.test_data_cache = None
        
        # Age-robust settings (stored after fitting)
        self.has_age_data = False
        self.age_lookup = {}
        self.sample_weights = None
        self.age_confounded = False
        
        # Dataset 8 (T1D/MALIDVJ) preprocessing flag
        self._use_preprocessing = False
        
        # Dataset-specific settings (set during fit)
        self._dataset_id = None
        self._skip_kmer4_rf = False
        self._dataset1_config = None

    def fit(self, train_dir_path: str):
        """Train the model on the provided training data."""
        print(f"\n{'='*70}")
        print(f"Training on: {train_dir_path}")
        print(f"{'='*70}")
        
        self.dataset_name = os.path.basename(train_dir_path)
        
        # =================================================================
        # DATASET 8 (MALIDVJ/T1D) PREPROCESSING
        # The T1D script (t1d_model1_vj_positional_fixed.py) uses Mal-ID
        # preprocessing: deduplication by CDR3+V and max_seqs capping.
        # This MUST be applied for Dataset 8 to match the original results.
        # =================================================================
        dataset_id = None
        try:
            dataset_id = int(''.join(c for c in self.dataset_name if c.isdigit()))
        except:
            pass
        
        # =================================================================
        # DATASET-SPECIFIC SEED CONFIGURATION
        # Dataset 1: Uses seed 123 to match 1_malid_tcr_multi_vj_new.py
        # Dataset 3: Uses seed 123 for kmer5/6 (already handled in Kmer56SGD class)
        # =================================================================
        if dataset_id == 1:
            CFG.SEED = CFG.SEED_DATASET1  # 123
            print(f"\n  [Dataset 1 detected: Using SEED={CFG.SEED} to match original script]")
        
        # Check if this is Dataset 8 (T1D/MALIDVJ) - needs preprocessing
        use_preprocessing = (dataset_id == 8) or CFG.RUN_MALIDVJ
        if use_preprocessing:
            print(f"\n  [Dataset 8 detected: Enabling Mal-ID preprocessing for MALIDVJ]")
        
        # Load data (with preprocessing for Dataset 8)
        self.train_data, self.metadata = load_train_data(train_dir_path, preprocess=use_preprocessing, n_workers=self.n_jobs)
        if not self.train_data:
            print("ERROR: No training data loaded")
            return self
        
        # Store preprocessing flag for use in predict_proba
        self._use_preprocessing = use_preprocessing
        
        # Store dataset_id for use in model training decisions
        self._dataset_id = dataset_id
        
        # Check if we should skip kmer4_RF for this dataset (v10 compatibility)
        self._skip_kmer4_rf = False
        self._dataset1_config = None
        if dataset_id and dataset_id in DATASET_ENSEMBLE_CONFIGS:
            self._skip_kmer4_rf = DATASET_ENSEMBLE_CONFIGS[dataset_id].get('skip_kmer4_rf', False)
            if self._skip_kmer4_rf:
                print(f"  [Dataset {dataset_id}: Skipping kmer4_RF to match v10]")
            
            # Dataset 1 specific config (different C values and fixed meta-learner)
            self._dataset1_config = DATASET_ENSEMBLE_CONFIGS[dataset_id].get('dataset1_config')
            if self._dataset1_config:
                # Override CFG.C_VALS for dataset 1 to match 1_malid_tcr_multi_vj_new.py
                CFG.C_VALS = self._dataset1_config['c_values']
                print(f"  [Dataset 1: Using C_VALUES={CFG.C_VALS} to match original script]")
                print(f"  [Dataset 1: Meta-learner will use fixed C={self._dataset1_config['meta_c']}]")
        
        y = np.array([d['label'] for d in self.train_data])
        print(f"Loaded {len(self.train_data)} samples (pos: {y.sum()}, neg: {len(y)-y.sum()})")
        
        # =================================================================
        # AGE-ROBUST SETUP (FIXED: Now actually used!)
        # =================================================================
        self.age_lookup = build_age_lookup(self.metadata)
        self.has_age_data = len(self.age_lookup) > len(self.train_data) * 0.5
        
        if self.has_age_data:
            print(f"\n  Age data available: {len(self.age_lookup)} samples with age")
            
            # Analyze confounding
            corr, self.age_confounded = analyze_age_confounding(self.metadata, self.train_data)
            print(f"  Age-disease correlation: {corr:.3f} (confounded: {self.age_confounded})")
            
            # Compute sample weights
            self.sample_weights = compute_age_bin_sample_weights(
                self.metadata, self.train_data, n_bins=CFG.AGE_WEIGHT_BINS
            )
            print(f"  Sample weights range: {self.sample_weights.min():.3f} - {self.sample_weights.max():.3f}")
            print(f"  Age-robust feature selection: ENABLED (inside CV)")
        else:
            print(f"\n  Age data: NOT AVAILABLE (standard training)")
            self.sample_weights = None
        
        top_v, top_j = get_top_vj_genes(self.train_data)
        
        # Extract features
        print("\nExtracting features...")
        self.caches = {
            'vj': {d['rep_id']: ext_vj(d['df']) for d in tqdm(self.train_data, desc="VJ", leave=False)},
            'vj_interact': {d['rep_id']: {**ext_vj(d['df']), **ext_vj_interact(d['df'], top_v, top_j)} 
                           for d in tqdm(self.train_data, desc="VJ+Int", leave=False)},
            'kmer4': {d['rep_id']: ext_kmer(d['df'], 4) for d in tqdm(self.train_data, desc="4mer", leave=False)},
            'pos4': {d['rep_id']: ext_pos_kmer(d['df'], 4) for d in tqdm(self.train_data, desc="Pos4", leave=False)},
            'gap': {d['rep_id']: ext_gap(d['df']) for d in tqdm(self.train_data, desc="Gap", leave=False)},
            'div': {d['rep_id']: ext_div(d['df']) for d in tqdm(self.train_data, desc="Div", leave=False)},
            'pos_aa': {d['rep_id']: ext_pos_aa(d['df']) for d in tqdm(self.train_data, desc="PosAA", leave=False)},
            'emerson': {d['rep_id']: ext_em_seq(d['df']) for d in tqdm(self.train_data, desc="Em", leave=False)},
        }
        
        ak4 = set().union(*[set(c.keys()) for c in self.caches['kmer4'].values()])
        ap4 = set().union(*[set(c.keys()) for c in self.caches['pos4'].values()])
        ag = set().union(*[set(c.keys()) for c in self.caches['gap'].values()])
        
        # Train models
        print("\nTraining models...")
        results = {}
        cv_preds = {}
        model_infos = {}
        
        # =================================================================
        # MALIDVJ Model (T1D-style: VJ + Positional with germline/nregion)
        # =================================================================
        if CFG.RUN_MALIDVJ:
            print("  malidvj...", end=" ", flush=True)
            try:
                auc, preds, info = train_malidvj(
                    self.train_data, y,
                    has_age_data=self.has_age_data,
                    age_lookup=self.age_lookup,
                    sample_weights=self.sample_weights
                )
                results['malidvj'] = auc
                cv_preds['malidvj'] = preds
                model_infos['malidvj'] = {'type': 'malidvj', **info}
                if self.has_age_data:
                    print(f"AUC: {auc:.4f} (robust feats: {info['n_features_mean']:.0f}±{info['n_features_std']:.0f})")
                else:
                    print(f"AUC: {auc:.4f}")
            except Exception as e:
                print(f"ERR: {e}")
        else:
            print("  malidvj... SKIPPED (disabled)")
        
        # =================================================================
        # VJ models (with age-robust option)
        # =================================================================
        if CFG.RUN_VJ:
            for name, cache in [('vj', self.caches['vj']), ('vj_logfreq', self.caches['vj']), 
                               ('vj_elasticnet', self.caches['vj']), ('vj_interact', self.caches['vj_interact'])]:
                print(f"  {name}...", end=" ", flush=True)
                try:
                    fn = sorted(set().union(*[set(cache[d['rep_id']].keys()) for d in self.train_data]))
                    if fn:
                        X = np.zeros((len(self.train_data), len(fn)))
                        for i, d in enumerate(self.train_data):
                            f = cache[d['rep_id']]
                            for j, n in enumerate(fn):
                                X[i, j] = f.get(n, 0)
                        trans = np.log1p if name == 'vj_logfreq' else None
                        if trans:
                            X = trans(X)
                        
                        # Special handling for vj_elasticnet to match v10 exactly
                        if name == 'vj_elasticnet':
                            auc, preds, bp = train_vj_elasticnet(X, y, sample_weights=self.sample_weights)
                            print(f"AUC: {auc:.4f} (C={bp['C']}, l1_ratio={bp['l1_ratio']})")
                        # Use age-robust training if age data available
                        elif self.has_age_data:
                            auc, preds, bp = train_lr_age_robust(
                                X, y, self.train_data, fn, self.age_lookup, self.sample_weights
                            )
                            if 'n_features_mean' in bp:
                                print(f"AUC: {auc:.4f} (robust feats: {bp['n_features_mean']:.0f}±{bp['n_features_std']:.0f})")
                            else:
                                print(f"AUC: {auc:.4f}")
                        else:
                            auc, preds, bp = train_lr(X, y, sample_weights=self.sample_weights)
                            print(f"AUC: {auc:.4f}")
                        results[name] = auc
                        cv_preds[name] = preds
                        model_infos[name] = {'type': name, 'fn': fn, 'bp': bp, 'cache_name': name.split('_')[0] if '_' in name else name}
                except Exception as e:
                    print(f"ERR: {e}")
        else:
            print("  vj models... SKIPPED (disabled)")
        
        # =================================================================
        # K-mer models (with age-robust option)
        # =================================================================
        kmer_models_to_run = []
        if CFG.RUN_KMER4:
            kmer_models_to_run.extend([
                ('kmer4_LR_freq', self.caches['kmer4'], ak4, True),
                ('kmer4_LR_raw', self.caches['kmer4'], ak4, False),
            ])
        if CFG.RUN_POS_KMER:
            kmer_models_to_run.append(('pos_kmer4', self.caches['pos4'], ap4, True))
        if CFG.RUN_GAPPED:
            kmer_models_to_run.append(('gapped_kmer', self.caches['gap'], ag, True))
        
        if not kmer_models_to_run:
            print("  kmer4/pos_kmer/gapped... SKIPPED (disabled)")
        
        for name, cache, af, norm in kmer_models_to_run:
            print(f"  {name}...", end=" ", flush=True)
            try:
                kl = sorted(af)
                X = build_feature_matrix(self.train_data, cache, kl, normalize=norm)
                
                if self.has_age_data:
                    auc, preds, bp = train_lr_age_robust(
                        X, y, self.train_data, kl, self.age_lookup, self.sample_weights
                    )
                else:
                    auc, preds, bp = train_lr(X, y, sample_weights=self.sample_weights)
                
                results[name] = auc
                cv_preds[name] = preds
                # FIXED: Store which extractor to use for predictions
                model_infos[name] = {'type': 'kmer', 'kl': kl, 'bp': bp, 'norm': norm, 
                                     'cache_name': name, 'extractor': name}
                print(f"AUC: {auc:.4f}")
            except Exception as e:
                print(f"ERR: {e}")
        
        # K-mer RF model - using CFG.RF_PARAMS to match v10
        # NOTE: Skip for datasets 2 and 4 to match v10 exactly (they don't include kmer4_RF)
        skip_rf = getattr(self, '_skip_kmer4_rf', False)
        if CFG.RUN_KMER4 and not skip_rf:
            print("  kmer4_RF...", end=" ", flush=True)
            try:
                kl = sorted(ak4)
                X = build_feature_matrix(self.train_data, self.caches['kmer4'], kl, normalize=True)
                cv = StratifiedKFold(n_splits=CFG.CV_FOLDS, shuffle=True, random_state=CFG.SEED)
                best_auc, best_preds, best_p = 0.5, np.zeros(len(y)), {}
                for params in CFG.RF_PARAMS:
                    preds = np.zeros(len(y))
                    for tr, val in cv.split(X, y):
                        sc = StandardScaler()
                        m = RandomForestClassifier(**params, random_state=CFG.SEED, n_jobs=-1)
                        if self.sample_weights is not None:
                            m.fit(sc.fit_transform(X[tr]), y[tr], sample_weight=self.sample_weights[tr])
                        else:
                            m.fit(sc.fit_transform(X[tr]), y[tr])
                        preds[val] = m.predict_proba(sc.transform(X[val]))[:, 1]
                    auc = roc_auc_score(y, preds)
                    if auc > best_auc:
                        best_auc, best_preds, best_p = auc, preds, params
                results['kmer4_RF'] = best_auc
                cv_preds['kmer4_RF'] = best_preds
                model_infos['kmer4_RF'] = {'type': 'kmer', 'kl': kl, 'bp': best_p, 'norm': True, 'mt': 'RF', 'extractor': 'kmer4'}
                print(f"AUC: {best_auc:.4f}")
            except Exception as e:
                print(f"ERR: {e}")
        
            # K-mer XGBoost model (if available) - using CFG.XGB_PARAMS to match v10
            if HAS_XGB:
                print("  kmer4_XGB...", end=" ", flush=True)
                try:
                    kl = sorted(ak4)
                    X = build_feature_matrix(self.train_data, self.caches['kmer4'], kl, normalize=True)
                    cv = StratifiedKFold(n_splits=CFG.CV_FOLDS, shuffle=True, random_state=CFG.SEED)
                    best_auc, best_preds, best_p = 0.5, np.zeros(len(y)), {}
                    for params in CFG.XGB_PARAMS:
                        preds = np.zeros(len(y))
                        for tr, val in cv.split(X, y):
                            sc = StandardScaler()
                            m = xgb.XGBClassifier(**params, random_state=CFG.SEED, use_label_encoder=False, 
                                                  eval_metric='logloss', verbosity=0)
                            if self.sample_weights is not None:
                                m.fit(sc.fit_transform(X[tr]), y[tr], sample_weight=self.sample_weights[tr])
                            else:
                                m.fit(sc.fit_transform(X[tr]), y[tr])
                            preds[val] = m.predict_proba(sc.transform(X[val]))[:, 1]
                        auc = roc_auc_score(y, preds)
                        if auc > best_auc:
                            best_auc, best_preds, best_p = auc, preds, params
                    results['kmer4_XGB'] = best_auc
                    cv_preds['kmer4_XGB'] = best_preds
                    model_infos['kmer4_XGB'] = {'type': 'kmer', 'kl': kl, 'bp': best_p, 'norm': True, 'mt': 'XGB', 'extractor': 'kmer4'}
                    print(f"AUC: {best_auc:.4f}")
                except Exception as e:
                    print(f"ERR: {e}")
        
        # Diversity
        if CFG.RUN_DIVERSITY:
            print("  diversity...", end=" ", flush=True)
            try:
                fn = sorted(set().union(*[set(self.caches['div'][d['rep_id']].keys()) for d in self.train_data]))
                X = np.zeros((len(self.train_data), len(fn)))
                for i, d in enumerate(self.train_data):
                    f = self.caches['div'][d['rep_id']]
                    for j, n in enumerate(fn):
                        X[i, j] = f.get(n, 0)
                auc, preds, bp = train_lr(X, y, sample_weights=self.sample_weights)
                results['diversity'] = auc
                cv_preds['diversity'] = preds
                model_infos['diversity'] = {'type': 'diversity', 'fn': fn, 'bp': bp}
                print(f"AUC: {auc:.4f}")
            except Exception as e:
                print(f"ERR: {e}")
        else:
            print("  diversity... SKIPPED (disabled)")
        
        # =================================================================
        # Positional AA features (T1D-style, with age-robust option)
        # =================================================================
        if CFG.RUN_POS_AA:
            print("  pos_aa...", end=" ", flush=True)
            try:
                fn = sorted(set().union(*[set(self.caches['pos_aa'][d['rep_id']].keys()) for d in self.train_data]))
                if fn:
                    X = np.zeros((len(self.train_data), len(fn)))
                    for i, d in enumerate(self.train_data):
                        f = self.caches['pos_aa'][d['rep_id']]
                        for j, n in enumerate(fn):
                            X[i, j] = f.get(n, 0)
                    
                    if self.has_age_data:
                        auc, preds, bp = train_lr_age_robust(
                            X, y, self.train_data, fn, self.age_lookup, self.sample_weights
                        )
                    else:
                        auc, preds, bp = train_lr(X, y, sample_weights=self.sample_weights)
                    
                    results['pos_aa'] = auc
                    cv_preds['pos_aa'] = preds
                    model_infos['pos_aa'] = {'type': 'pos_aa', 'fn': fn, 'bp': bp}
                    print(f"AUC: {auc:.4f}")
                else:
                    print("SKIP (no features)")
            except Exception as e:
                print(f"ERR: {e}")
        else:
            print("  pos_aa... SKIPPED (disabled)")
        
        # Emerson - using dataset-specific parameters
        # Dataset 7 uses different p_thresholds [0.10, 0.12, 0.14] and min_counts [2, 3]
        # AND stacks ALL configs together (not just the best one)
        # Others use v10 defaults [0.05, 0.1] and min_count 2
        if CFG.RUN_EMERSON:
            print("  emerson...", end=" ", flush=True)
            try:
                ss = [self.caches['emerson'][d['rep_id']] for d in self.train_data]
                # FIXED: Initialize best_p with default values to avoid KeyError during prediction
                best_auc, best_preds, best_p = 0.5, np.full(len(y), 0.5), {'p': 0.12, 'mc': 2}
                
                # Get dataset-specific emerson config
                dataset_id = getattr(self, '_dataset_id', None)
                emerson_config = None
                if dataset_id and dataset_id in DATASET_ENSEMBLE_CONFIGS:
                    emerson_config = DATASET_ENSEMBLE_CONFIGS[dataset_id].get('emerson_config')
                
                if emerson_config:
                    # Dataset 7: Use multiple p_thresholds AND min_counts
                    # ALSO store ALL configs for stacking (not just best)
                    # IMPORTANT: For Dataset 7, sequence discovery happens INSIDE each CV fold
                    # (matching airr_ml_dataset7_emerson_cv_0912.py behavior)
                    p_thresholds = emerson_config.get('p_thresholds', CFG.EMERSON_P_THRESHOLDS)
                    min_counts = emerson_config.get('min_counts', [CFG.EMERSON_MIN_COUNT])
                    print(f"(Dataset {dataset_id} config: p={p_thresholds}, mc={min_counts}, fold-aware)", end=" ")
                    
                    # For Dataset 7: Store ALL emerson configs for stacking
                    emerson_configs_results = []
                    
                    # Dataset 7: Proper CV with sequence discovery inside each fold
                    cv = StratifiedKFold(n_splits=CFG.CV_FOLDS, shuffle=True, random_state=CFG.SEED)
                    
                    for p in p_thresholds:
                        for mc in min_counts:
                            preds = np.zeros(len(y))
                            has_seqs = False
                            
                            for tr, val in cv.split(ss, y):
                                # Fit Emerson on TRAINING fold only (sequence discovery inside fold)
                                train_ss = [ss[i] for i in tr]
                                train_y = y[tr]
                                
                                clf = EmClf(p=p, mc=mc)
                                clf.fit(train_ss, train_y)
                                
                                if clf.seqs:
                                    has_seqs = True
                                    # Count matching sequences for train and val
                                    X_tr = np.array([[len(ss[i] & clf.seqs)] for i in tr])
                                    X_val = np.array([[len(ss[i] & clf.seqs)] for i in val])
                                    
                                    lr = LogisticRegression(random_state=CFG.SEED, max_iter=1000)
                                    lr.fit(X_tr, train_y)
                                    preds[val] = lr.predict_proba(X_val)[:, 1]
                            
                            if has_seqs:
                                auc = roc_auc_score(y, preds)
                                config_name = f'emerson_p{p}_mc{mc}'
                                emerson_configs_results.append({
                                    'name': config_name,
                                    'auc': auc,
                                    'preds': preds.copy(),
                                    'params': {'p': p, 'mc': mc}
                                })
                                
                                if auc > best_auc:
                                    best_auc, best_preds, best_p = auc, preds.copy(), {'p': p, 'mc': mc}
                else:
                    # Default: v10 parameters (sequence discovery BEFORE CV for efficiency)
                    p_thresholds = CFG.EMERSON_P_THRESHOLDS
                    min_counts = [CFG.EMERSON_MIN_COUNT]
                    emerson_configs_results = None  # Don't need stacking for non-dataset-7
                    
                    # Grid search over p_thresholds and min_counts
                    for p in p_thresholds:
                        for mc in min_counts:
                            clf = EmClf(p=p, mc=mc)
                            clf.fit(ss, y)
                            if clf.seqs:
                                cv = StratifiedKFold(n_splits=CFG.CV_FOLDS, shuffle=True, random_state=CFG.SEED)
                                preds = np.zeros(len(y))
                                for tr, val in cv.split(ss, y):
                                    X_tr = np.array([[len(ss[i] & clf.seqs)] for i in tr])
                                    X_val = np.array([[len(ss[i] & clf.seqs)] for i in val])
                                    lr = LogisticRegression(random_state=CFG.SEED, max_iter=1000)
                                    lr.fit(X_tr, y[tr])
                                    preds[val] = lr.predict_proba(X_val)[:, 1]
                                auc = roc_auc_score(y, preds)
                                
                                if auc > best_auc:
                                    best_auc, best_preds, best_p = auc, preds.copy(), {'p': p, 'mc': mc}
                
                # Store best emerson model
                results['emerson'] = best_auc
                cv_preds['emerson'] = best_preds
                model_infos['emerson'] = {'type': 'emerson', 'bp': best_p}
                
                # For Dataset 7: Also store individual configs for proper stacking
                if emerson_configs_results and len(emerson_configs_results) > 1:
                    print(f"(stacking {len(emerson_configs_results)} configs)", end=" ")
                    for cfg_result in emerson_configs_results:
                        name = cfg_result['name']
                        results[name] = cfg_result['auc']
                        cv_preds[name] = cfg_result['preds']
                        model_infos[name] = {'type': 'emerson', 'bp': cfg_result['params']}
                
                print(f"AUC: {best_auc:.4f}")
            except Exception as e:
                print(f"ERR: {e}")
        else:
            print("  emerson... SKIPPED (disabled)")
        
        # 5/6-mer SGD (now with CV predictions for ensemble stacking)
        if CFG.RUN_KMER56:
            print("  kmer5/6_sgd...", end=" ", flush=True)
            try:
                for k in [5, 6]:
                    model = KmSGD(k=k)
                    # Use get_cv_predictions to get OOF predictions for ensemble
                    preds, auc = model.get_cv_predictions(self.train_data, y)
                    nm = f'kmer{k}_sgd'
                    results[nm] = auc
                    cv_preds[nm] = preds  # Now SGD models can participate in ensemble!
                    model_infos[nm] = {'type': nm, 'model': model}
                    print(f"{k}mer:{auc:.3f}", end=" ")
                print()
            except Exception as e:
                print(f"ERR: {e}")
        else:
            print("  kmer5/6_sgd... SKIPPED (disabled)")
        
        # Ensemble (always run if we have at least 2 models)
        if len(results) >= 2:
            print("  ensemble...", end=" ", flush=True)
            
            # Check for ensemble mode from environment (set by shell script)
            env_ensemble_mode = os.environ.get('AIRR_ENSEMBLE_MODE', '').strip().lower()
            
            # Also detect dataset ID from dataset_name as fallback
            dataset_id = None
            try:
                dataset_id = int(''.join(c for c in self.dataset_name if c.isdigit()))
            except:
                pass
            
            # Determine ensemble configuration
            # Priority: 1) Environment variable, 2) Dataset-specific config, 3) Default
            if env_ensemble_mode == 'vj_kmer':
                # Explicit VJ+kmer mode from shell script (Dataset 1)
                mode = 'dataset1_vj_kmer'
                min_auc = 0.50
                auc_margin = 1.0
                print(f"\n    [Ensemble mode from ENV: vj_kmer]")
            elif env_ensemble_mode == 'none':
                # Explicit no-ensemble mode from shell script (Datasets 3, 8)
                mode = 'none'
                min_auc = 0.50
                auc_margin = 0.02
                print(f"\n    [Ensemble mode from ENV: none - will pick single best model]")
            elif env_ensemble_mode == 'emerson_stacking':
                # Explicit emerson stacking mode from shell script (Dataset 7)
                mode = 'emerson_stacking'
                min_auc = 0.50
                auc_margin = 0.10
                print(f"\n    [Ensemble mode from ENV: emerson_stacking]")
            elif env_ensemble_mode == 'standard':
                # Explicit standard mode from shell script - use dataset-specific params if available
                if dataset_id and dataset_id in DATASET_ENSEMBLE_CONFIGS:
                    config = DATASET_ENSEMBLE_CONFIGS[dataset_id]
                    mode = 'standard'
                    min_auc = config['min_auc']
                    auc_margin = config['auc_margin']
                else:
                    mode = 'standard'
                    min_auc = 0.55
                    auc_margin = 0.02
                print(f"\n    [Ensemble mode from ENV: standard, min_auc={min_auc}, margin={auc_margin}]")
            elif dataset_id and dataset_id in DATASET_ENSEMBLE_CONFIGS:
                # Fallback to dataset-specific config
                config = DATASET_ENSEMBLE_CONFIGS[dataset_id]
                mode = config['mode']
                min_auc = config['min_auc']
                auc_margin = config['auc_margin']
                print(f"\n    [Dataset {dataset_id} ensemble config: mode={mode}, min_auc={min_auc}, margin={auc_margin}]")
            else:
                # Default configuration
                mode = 'standard'
                min_auc = 0.55
                auc_margin = 0.02
            
            # Get task1_restrict from dataset config (if available)
            task1_restrict = None
            dataset1_config = None
            if dataset_id and dataset_id in DATASET_ENSEMBLE_CONFIGS:
                task1_restrict = DATASET_ENSEMBLE_CONFIGS[dataset_id].get('task1_restrict')
                dataset1_config = DATASET_ENSEMBLE_CONFIGS[dataset_id].get('dataset1_config')
            
            ml = MetaL(min_auc=min_auc, auc_margin=auc_margin, mode=mode, task1_restrict=task1_restrict, dataset1_config=dataset1_config)
            ml.fit(results, cv_preds, y)
            
            # For 'none' mode, we still want to record the best model but not as 'ensemble'
            if mode == 'none':
                # Don't add 'ensemble' to results - the best single model is already there
                print(f"    Selected: {ml.models[0]} (AUC: {ml.auc:.4f})")
            elif ml.meta and len(ml.models) > 1:
                results['ensemble'] = ml.auc
                cv_preds['ensemble'] = ml.predict(cv_preds)
                model_infos['ensemble'] = {'type': 'ensemble', 'ml': ml, 'models': ml.models}
                print(f"AUC: {ml.auc:.4f} (models: {ml.models})")
            else:
                print("SKIP (insufficient models)")
        else:
            print("  ensemble... SKIPPED (need >= 2 models)")
        
        # Store results
        self.model_info = model_infos
        
        # =================================================================
        # BEST MODEL SELECTION (with task1_restrict support)
        # =================================================================
        # Check if this dataset has model restrictions for Task 1
        task1_restrict = None
        if dataset_id and dataset_id in DATASET_ENSEMBLE_CONFIGS:
            task1_restrict = DATASET_ENSEMBLE_CONFIGS[dataset_id].get('task1_restrict')
        
        if task1_restrict and results:
            # Filter results to only include allowed models
            allowed_results = {}
            for model_name, auc in results.items():
                # Check if model name starts with any of the allowed prefixes
                if any(model_name.startswith(prefix) or model_name == prefix for prefix in task1_restrict):
                    allowed_results[model_name] = auc
            
            if allowed_results:
                self.best_model_name = max(allowed_results.keys(), key=lambda x: allowed_results[x])
                print(f"\n    [Task 1 model restriction: {task1_restrict}]")
                print(f"    [Eligible models: {list(allowed_results.keys())}]")
            else:
                # Fallback to best overall if no restricted models trained
                self.best_model_name = max(results.keys(), key=lambda x: results[x]) if results else None
                print(f"\n    [WARNING: No models matching task1_restrict {task1_restrict}, using best overall]")
        else:
            # No restriction - pick best from all models
            self.best_model_name = max(results.keys(), key=lambda x: results[x]) if results else None
        
        self.model = {'results': results, 'cv_preds': cv_preds, 'y': y, 'top_v': top_v, 'top_j': top_j}
        
        if self.best_model_name:
            print(f"\n*** BEST: {self.best_model_name} (AUC: {results[self.best_model_name]:.4f}) ***")
        else:
            print("\n*** WARNING: No models were trained! ***")
        if self.has_age_data:
            print(f"*** Age-robust training: ENABLED (sample weights + feature selection) ***")
        
        # Identify important sequences
        self.important_sequences_ = self.identify_associated_sequences(
            top_k=CFG.TOP_K_SEQUENCES, 
            dataset_name=self.dataset_name
        )
        
        print("Training complete.")
        return self

    def predict_proba(self, test_dir_path: str) -> pd.DataFrame:
        """Predict probabilities for test data."""
        print(f"\nPredicting on: {test_dir_path}")
        
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Load test data (with preprocessing if Dataset 8/MALIDVJ)
        use_preprocessing = getattr(self, '_use_preprocessing', False)
        if use_preprocessing:
            print(f"  [Using Mal-ID preprocessing for test data (Dataset 8)]")
        test_data = load_test_data(test_dir_path, preprocess=use_preprocessing, n_workers=self.n_jobs)
        if not test_data:
            print("WARNING: No test data loaded")
            return pd.DataFrame()
        
        repertoire_ids = [d['rep_id'] for d in test_data]
        y = self.model['y']
        top_v, top_j = self.model['top_v'], self.model['top_j']
        
        # Get predictions from best model
        info = self.model_info.get(self.best_model_name, {})
        mt = info.get('type', self.best_model_name)
        
        # =================================================================
        # MALIDVJ Model (T1D-style) - FAST: Use pre-trained model
        # =================================================================
        if mt == 'malidvj':
            vec = info['vectorizer']
            
            # Check if we have pre-trained model (new fast path)
            if 'final_model' in info:
                # FAST PATH: Use stored model, scaler, and feature indices
                final_model = info['final_model']
                final_scaler = info['final_scaler']
                final_robust_idx = info['final_robust_idx']
                
                # Only extract features for test data
                test_features = [ext_malidvj(d['df']) for d in test_data]
                X_test = vec.transform(test_features)
                X_test_selected = X_test[:, final_robust_idx]
                X_test_scaled = final_scaler.transform(X_test_selected)
                
                probabilities = final_model.predict_proba(X_test_scaled)[:, 1]
            else:
                # SLOW PATH (legacy): Re-train model (for backwards compatibility)
                from sklearn.feature_extraction import DictVectorizer
                
                feature_names = info['feature_names']
                
                # Extract features for training data
                train_features = [ext_malidvj(d['df']) for d in self.train_data]
                X_train = vec.transform(train_features)
                
                # Age-robust feature selection using ALL training data
                if self.has_age_data and self.age_lookup:
                    all_train_idx = np.arange(len(self.train_data))
                    robust_idx = select_age_robust_features(
                        all_train_idx, self.train_data, feature_names, X_train, y, self.age_lookup
                    )
                    X_train_selected = X_train[:, robust_idx]
                else:
                    X_train_selected = X_train
                    robust_idx = list(range(len(feature_names)))
                
                # Train final model on all training data
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_selected)
                
                model = LogisticRegression(
                    penalty='l2', C=1.0, max_iter=1000,
                    class_weight='balanced', random_state=CFG.SEED
                )
                if self.sample_weights is not None:
                    model.fit(X_train_scaled, y, sample_weight=self.sample_weights)
                else:
                    model.fit(X_train_scaled, y)
                
                # Extract features for test data and predict
                test_features = [ext_malidvj(d['df']) for d in test_data]
                X_test = vec.transform(test_features)
                X_test_selected = X_test[:, robust_idx]
                X_test_scaled = scaler.transform(X_test_selected)
                
                probabilities = model.predict_proba(X_test_scaled)[:, 1]
        
        # =================================================================
        # VJ models (including vj_interact) - FIXED
        # =================================================================
        elif mt in ['vj', 'vj_logfreq', 'vj_elasticnet', 'vj_interact']:
            fn, bp = info['fn'], info['bp']
            cache_name = info.get('cache_name', 'vj')
            
            X_tr = np.zeros((len(self.train_data), len(fn)))
            for i, d in enumerate(self.train_data):
                if cache_name == 'vj_interact' or mt == 'vj_interact':
                    f = {**ext_vj(d['df']), **ext_vj_interact(d['df'], top_v, top_j)}
                else:
                    f = ext_vj(d['df'])
                for j, n in enumerate(fn):
                    X_tr[i, j] = f.get(n, 0)
            if mt == 'vj_logfreq':
                X_tr = np.log1p(X_tr)
            
            # Age-robust feature selection if used
            if self.has_age_data and 'n_features_mean' in bp:
                all_idx = np.arange(len(self.train_data))
                robust_idx = select_age_robust_features(
                    all_idx, self.train_data, fn, X_tr, y, self.age_lookup
                )
                X_tr = X_tr[:, robust_idx]
            else:
                robust_idx = None
            
            sc = StandardScaler()
            sc.fit(X_tr)
            m = LogisticRegression(penalty='l1', C=bp['C'], solver='liblinear', random_state=CFG.SEED, max_iter=1000)
            if self.sample_weights is not None:
                m.fit(sc.transform(X_tr), y, sample_weight=self.sample_weights)
            else:
                m.fit(sc.transform(X_tr), y)
            
            X_te = np.zeros((len(test_data), len(fn)))
            for i, d in enumerate(test_data):
                if cache_name == 'vj_interact' or mt == 'vj_interact':
                    f = {**ext_vj(d['df']), **ext_vj_interact(d['df'], top_v, top_j)}
                else:
                    f = ext_vj(d['df'])
                for j, n in enumerate(fn):
                    X_te[i, j] = f.get(n, 0)
            if mt == 'vj_logfreq':
                X_te = np.log1p(X_te)
            if robust_idx is not None:
                X_te = X_te[:, robust_idx]
            probabilities = m.predict_proba(sc.transform(X_te))[:, 1]
        
        # =================================================================
        # Diversity model
        # =================================================================
        elif mt == 'diversity':
            fn, bp = info['fn'], info['bp']
            X_tr = np.zeros((len(self.train_data), len(fn)))
            for i, d in enumerate(self.train_data):
                f = ext_div(d['df'])
                for j, n in enumerate(fn):
                    X_tr[i, j] = f.get(n, 0)
            sc = StandardScaler()
            sc.fit(X_tr)
            m = LogisticRegression(penalty='l1', C=bp['C'], solver='liblinear', random_state=CFG.SEED, max_iter=1000)
            if self.sample_weights is not None:
                m.fit(sc.transform(X_tr), y, sample_weight=self.sample_weights)
            else:
                m.fit(sc.transform(X_tr), y)
            X_te = np.zeros((len(test_data), len(fn)))
            for i, d in enumerate(test_data):
                f = ext_div(d['df'])
                for j, n in enumerate(fn):
                    X_te[i, j] = f.get(n, 0)
            probabilities = m.predict_proba(sc.transform(X_te))[:, 1]
        
        # =================================================================
        # Positional AA model - NEW
        # =================================================================
        elif mt == 'pos_aa':
            fn, bp = info['fn'], info['bp']
            X_tr = np.zeros((len(self.train_data), len(fn)))
            for i, d in enumerate(self.train_data):
                f = ext_pos_aa(d['df'])
                for j, n in enumerate(fn):
                    X_tr[i, j] = f.get(n, 0)
            
            if self.has_age_data and 'n_features_mean' in bp:
                all_idx = np.arange(len(self.train_data))
                robust_idx = select_age_robust_features(
                    all_idx, self.train_data, fn, X_tr, y, self.age_lookup
                )
                X_tr = X_tr[:, robust_idx]
            else:
                robust_idx = None
            
            sc = StandardScaler()
            sc.fit(X_tr)
            m = LogisticRegression(penalty='l1', C=bp['C'], solver='liblinear', random_state=CFG.SEED, max_iter=1000)
            if self.sample_weights is not None:
                m.fit(sc.transform(X_tr), y, sample_weight=self.sample_weights)
            else:
                m.fit(sc.transform(X_tr), y)
            
            X_te = np.zeros((len(test_data), len(fn)))
            for i, d in enumerate(test_data):
                f = ext_pos_aa(d['df'])
                for j, n in enumerate(fn):
                    X_te[i, j] = f.get(n, 0)
            if robust_idx is not None:
                X_te = X_te[:, robust_idx]
            probabilities = m.predict_proba(sc.transform(X_te))[:, 1]
        
        # =================================================================
        # K-mer models - FIXED: handle pos_kmer4 and gapped correctly
        # =================================================================
        elif mt == 'kmer':
            kl, bp, norm = info['kl'], info['bp'], info['norm']
            model_type = info.get('mt', 'LR')
            extractor = info.get('extractor', 'kmer4_LR_freq')
            
            # FIXED: Use correct extractor
            if 'pos' in extractor or extractor == 'pos_kmer4':
                ext_fn = lambda df: ext_pos_kmer(df, 4)
            elif 'gap' in extractor or extractor == 'gapped_kmer':
                ext_fn = ext_gap
            else:
                ext_fn = lambda df: ext_kmer(df, 4)
            
            X_tr = np.zeros((len(self.train_data), len(kl)))
            for i, d in enumerate(self.train_data):
                f = ext_fn(d['df'])
                t = sum(f.values()) + 1 if norm else 1
                for j, km in enumerate(kl):
                    X_tr[i, j] = f.get(km, 0) / t if norm else f.get(km, 0)
            
            # Age-robust feature selection
            if self.has_age_data and 'n_features_mean' in bp:
                all_idx = np.arange(len(self.train_data))
                robust_idx = select_age_robust_features(
                    all_idx, self.train_data, kl, X_tr, y, self.age_lookup
                )
                X_tr = X_tr[:, robust_idx]
            else:
                robust_idx = None
            
            sc = StandardScaler()
            sc.fit(X_tr)
            
            if model_type == 'XGB' and HAS_XGB:
                m = xgb.XGBClassifier(**bp, random_state=CFG.SEED, use_label_encoder=False, 
                                      eval_metric='logloss', verbosity=0)
            elif model_type == 'RF':
                m = RandomForestClassifier(**bp, random_state=CFG.SEED, n_jobs=-1)
            else:
                m = LogisticRegression(penalty='l1', C=bp.get('C', 0.1), solver='liblinear', 
                                       random_state=CFG.SEED, max_iter=1000)
            
            if self.sample_weights is not None:
                m.fit(sc.transform(X_tr), y, sample_weight=self.sample_weights)
            else:
                m.fit(sc.transform(X_tr), y)
            
            X_te = np.zeros((len(test_data), len(kl)))
            for i, d in enumerate(test_data):
                f = ext_fn(d['df'])
                t = sum(f.values()) + 1 if norm else 1
                for j, km in enumerate(kl):
                    X_te[i, j] = f.get(km, 0) / t if norm else f.get(km, 0)
            if robust_idx is not None:
                X_te = X_te[:, robust_idx]
            probabilities = m.predict_proba(sc.transform(X_te))[:, 1]
        
        # =================================================================
        # Emerson model
        # =================================================================
        elif mt == 'emerson':
            bp = info['bp']
            clf = EmClf(p=bp['p'], mc=bp['mc'])
            clf.fit([self.caches['emerson'][d['rep_id']] for d in self.train_data], y)
            probabilities = clf.predict_proba([ext_em_seq(d['df']) for d in test_data])
        
        # =================================================================
        # K-mer 5/6 SGD models
        # =================================================================
        elif mt in ['kmer5_sgd', 'kmer6_sgd']:
            model = info['model']
            probabilities = model.predict(test_data)
        
        # =================================================================
        # Ensemble model - FIXED: actually get component predictions
        # =================================================================
        elif mt == 'ensemble':
            ml = info['ml']
            all_preds = {}
            for model_name in ml.models:
                if model_name in self.model_info:
                    try:
                        sub_preds = self._predict_single_model(test_data, model_name)
                        if sub_preds is not None:
                            all_preds[model_name] = sub_preds
                    except Exception as e:
                        print(f"    Warning: {model_name} failed: {e}")
            if all_preds:
                probabilities = ml.predict(all_preds)
            else:
                probabilities = np.full(len(test_data), 0.5)
        
        else:
            probabilities = np.full(len(test_data), 0.5)
        
        # Build output DataFrame
        predictions_df = pd.DataFrame({
            'ID': repertoire_ids,
            'dataset': [os.path.basename(test_dir_path)] * len(repertoire_ids),
            'label_positive_probability': probabilities,
            'junction_aa': -999.0,
            'v_call': -999.0,
            'j_call': -999.0
        })
        
        predictions_df = predictions_df[['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']]
        print(f"Prediction complete on {len(repertoire_ids)} examples.")
        return predictions_df
    
    def _predict_single_model(self, test_data: List[Dict], model_name: str) -> Optional[np.ndarray]:
        """Helper to get predictions from a single model (for ensemble)."""
        info = self.model_info.get(model_name, {})
        mt = info.get('type', model_name)
        y = self.model['y']
        top_v, top_j = self.model['top_v'], self.model['top_j']
        
        if mt == 'malidvj':
            from sklearn.feature_extraction import DictVectorizer
            
            vec = info['vectorizer']
            feature_names = info['feature_names']
            
            # Extract features for training data
            train_features = [ext_malidvj(d['df']) for d in self.train_data]
            X_train = vec.transform(train_features)
            
            # Age-robust feature selection using ALL training data
            if self.has_age_data and self.age_lookup:
                all_train_idx = np.arange(len(self.train_data))
                robust_idx = select_age_robust_features(
                    all_train_idx, self.train_data, feature_names, X_train, y, self.age_lookup
                )
                X_train_selected = X_train[:, robust_idx]
            else:
                X_train_selected = X_train
                robust_idx = list(range(len(feature_names)))
            
            # Train final model on all training data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_selected)
            
            model = LogisticRegression(
                penalty='l2', C=1.0, max_iter=1000,
                class_weight='balanced', random_state=CFG.SEED
            )
            if self.sample_weights is not None:
                model.fit(X_train_scaled, y, sample_weight=self.sample_weights)
            else:
                model.fit(X_train_scaled, y)
            
            # Extract features for test data and predict
            test_features = [ext_malidvj(d['df']) for d in test_data]
            X_test = vec.transform(test_features)
            X_test_selected = X_test[:, robust_idx]
            X_test_scaled = scaler.transform(X_test_selected)
            
            return model.predict_proba(X_test_scaled)[:, 1]
        
        elif mt in ['kmer5_sgd', 'kmer6_sgd']:
            return info['model'].predict(test_data)
        
        elif mt == 'emerson':
            bp = info['bp']
            clf = EmClf(p=bp['p'], mc=bp['mc'])
            clf.fit([self.caches['emerson'][d['rep_id']] for d in self.train_data], y)
            return clf.predict_proba([ext_em_seq(d['df']) for d in test_data])
        
        elif mt in ['vj', 'vj_logfreq', 'vj_elasticnet', 'vj_interact']:
            fn, bp = info['fn'], info['bp']
            cache_name = info.get('cache_name', 'vj')
            X_tr = np.zeros((len(self.train_data), len(fn)))
            for i, d in enumerate(self.train_data):
                if cache_name == 'vj_interact' or mt == 'vj_interact':
                    f = {**ext_vj(d['df']), **ext_vj_interact(d['df'], top_v, top_j)}
                else:
                    f = ext_vj(d['df'])
                for j, n in enumerate(fn):
                    X_tr[i, j] = f.get(n, 0)
            if mt == 'vj_logfreq':
                X_tr = np.log1p(X_tr)
            sc = StandardScaler()
            sc.fit(X_tr)
            m = LogisticRegression(penalty='l1', C=bp['C'], solver='liblinear', random_state=CFG.SEED, max_iter=1000)
            m.fit(sc.transform(X_tr), y)
            X_te = np.zeros((len(test_data), len(fn)))
            for i, d in enumerate(test_data):
                if cache_name == 'vj_interact' or mt == 'vj_interact':
                    f = {**ext_vj(d['df']), **ext_vj_interact(d['df'], top_v, top_j)}
                else:
                    f = ext_vj(d['df'])
                for j, n in enumerate(fn):
                    X_te[i, j] = f.get(n, 0)
            if mt == 'vj_logfreq':
                X_te = np.log1p(X_te)
            return m.predict_proba(sc.transform(X_te))[:, 1]
        
        elif mt == 'diversity':
            fn, bp = info['fn'], info['bp']
            X_tr = np.zeros((len(self.train_data), len(fn)))
            for i, d in enumerate(self.train_data):
                f = ext_div(d['df'])
                for j, n in enumerate(fn):
                    X_tr[i, j] = f.get(n, 0)
            sc = StandardScaler()
            sc.fit(X_tr)
            m = LogisticRegression(penalty='l1', C=bp['C'], solver='liblinear', random_state=CFG.SEED, max_iter=1000)
            m.fit(sc.transform(X_tr), y)
            X_te = np.zeros((len(test_data), len(fn)))
            for i, d in enumerate(test_data):
                f = ext_div(d['df'])
                for j, n in enumerate(fn):
                    X_te[i, j] = f.get(n, 0)
            return m.predict_proba(sc.transform(X_te))[:, 1]
        
        elif mt == 'kmer':
            kl, bp, norm = info['kl'], info['bp'], info['norm']
            model_type = info.get('mt', 'LR')
            extractor = info.get('extractor', 'kmer4')
            if 'pos' in extractor:
                ext_fn = lambda df: ext_pos_kmer(df, 4)
            elif 'gap' in extractor:
                ext_fn = ext_gap
            else:
                ext_fn = lambda df: ext_kmer(df, 4)
            X_tr = np.zeros((len(self.train_data), len(kl)))
            for i, d in enumerate(self.train_data):
                f = ext_fn(d['df'])
                t = sum(f.values()) + 1 if norm else 1
                for j, km in enumerate(kl):
                    X_tr[i, j] = f.get(km, 0) / t if norm else f.get(km, 0)
            sc = StandardScaler()
            sc.fit(X_tr)
            if model_type == 'XGB' and HAS_XGB:
                m = xgb.XGBClassifier(**bp, random_state=CFG.SEED, use_label_encoder=False, eval_metric='logloss', verbosity=0)
            elif model_type == 'RF':
                m = RandomForestClassifier(**bp, random_state=CFG.SEED, n_jobs=-1)
            else:
                m = LogisticRegression(penalty='l1', C=bp.get('C', 0.1), solver='liblinear', random_state=CFG.SEED, max_iter=1000)
            m.fit(sc.transform(X_tr), y)
            X_te = np.zeros((len(test_data), len(kl)))
            for i, d in enumerate(test_data):
                f = ext_fn(d['df'])
                t = sum(f.values()) + 1 if norm else 1
                for j, km in enumerate(kl):
                    X_te[i, j] = f.get(km, 0) / t if norm else f.get(km, 0)
            return m.predict_proba(sc.transform(X_te))[:, 1]
        
        return None

    def identify_associated_sequences(self, dataset_name: str, top_k: int = 50000) -> pd.DataFrame:
        """
        Identify top k important sequences using dataset-specific strategy.
        
        IMPROVED:
        - Dataset-specific model selection (matches task2_extract_parallel.py)
        - Can score TEST sequences if TASK2_USE_TEST_DATA=1 and test data cached
        - Can override model with TASK2_MODEL_OVERRIDE environment variable
        - Reuses cached features and trained models
        - Supports SKIP to skip Task 2 entirely (for datasets 7, 8)
        
        Args:
            dataset_name: Name of the dataset (e.g., "train_dataset_4")
            top_k: Number of sequences to return (default: 50000)
            
        Returns:
            DataFrame with top_k sequences formatted for competition
        """
        if self.train_data is None:
            return pd.DataFrame()
        
        # =================================================================
        # Check for SKIP - don't produce Task 2 output
        # =================================================================
        model_override = os.environ.get('TASK2_MODEL_OVERRIDE', '').strip()
        if model_override.upper() == "SKIP":
            print(f"\n{'='*70}")
            print(f"TASK 2: SKIPPED (TASK2_MODEL_OVERRIDE=SKIP)")
            print(f"{'='*70}\n")
            return pd.DataFrame()
        
        # Extract dataset ID from dataset_name
        try:
            dataset_id = int(''.join(c for c in dataset_name if c.isdigit()))
        except:
            dataset_id = None
        
        print(f"\n{'='*70}")
        print(f"TASK 2: Identifying top {top_k} associated sequences")
        print(f"Dataset: {dataset_name} (ID: {dataset_id})")
        
        # =================================================================
        # Model Selection - Now uses v11 Task 2 method names
        # =================================================================
        
        # Get dataset-specific method from v11 Task 2 strategies
        task2_method = None
        if dataset_id and dataset_id in TASK2_STRATEGIES:
            task2_method = TASK2_STRATEGIES[dataset_id].get('method')
        
        if model_override == "AUTO" or (not model_override and task2_method):
            # Use dataset-specific strategy from v11
            if dataset_id and dataset_id in TASK2_STRATEGIES:
                strategy = TASK2_STRATEGIES[dataset_id]
                task2_method = strategy.get('method', 'best')
                selected_model = None
                
                print(f"Task 2 Method: {task2_method} (from v11 strategies)")
                
                # For special methods, we may need to use a combination of models
                # or train on-the-fly
                if task2_method == 'vj_positional':
                    # Check if we have VJ model - use first available
                    for preferred in strategy['preferred']:
                        if preferred in self.model_info:
                            selected_model = preferred
                            break
                    if selected_model is None:
                        selected_model = self.best_model_name
                    print(f"  Using: {selected_model} for VJ+positional scoring")
                    
                elif task2_method == 'vj_ensemble':
                    # For Dataset 3 - need VJ model even if not trained in Task 1
                    for preferred in strategy['preferred']:
                        if preferred in self.model_info:
                            selected_model = preferred
                            break
                    if selected_model is None:
                        # Fallback - we may need to train VJ on-the-fly
                        print(f"  WARNING: VJ model not available, using fallback")
                        selected_model = strategy['fallback'] if strategy['fallback'] in self.model_info else self.best_model_name
                    print(f"  Using: {selected_model} for VJ ensemble scoring")
                    
                else:
                    # Standard model selection - try preferred models in order
                    for preferred in strategy['preferred']:
                        if preferred in self.model_info:
                            selected_model = preferred
                            break
                    
                    # Fallback within strategy
                    if selected_model is None:
                        if strategy['fallback'] in self.model_info:
                            selected_model = strategy['fallback']
                            print(f"  Using: {selected_model} (AUTO fallback)")
                        else:
                            selected_model = self.best_model_name
                            print(f"  Using: {selected_model} (BEST - AUTO fallback unavailable)")
                    else:
                        print(f"  Using: {selected_model}")
            else:
                # No strategy for this dataset, use best
                selected_model = self.best_model_name
                print(f"Model: {selected_model} (BEST - no AUTO strategy for dataset {dataset_id})")
        
        elif model_override and model_override != "BEST":
            # Specific model requested
            selected_model = model_override
            print(f"Model: {selected_model} (from TASK2_MODEL_OVERRIDE)")
        
        else:
            # Default: use best-performing model from Task 1
            selected_model = self.best_model_name
            print(f"Model: {selected_model} (BEST from Task 1)")
        
        if selected_model not in self.model_info:
            print(f"  ERROR: Model {selected_model} not found")
            print(f"  Available: {list(self.model_info.keys())}")
            return pd.DataFrame()
        
        # =================================================================
        # Data Selection (test vs training vs both)
        # =================================================================
        use_test_data = os.environ.get('TASK2_USE_TEST_DATA', '0').strip() == '1'
        also_score_train = os.environ.get('TASK2_ALSO_SCORE_TRAIN', '0').strip() == '1'

        if use_test_data and also_score_train and self.test_data_cache is not None:
            # BOTH mode: combine test and training data
            data_to_score = self.test_data_cache + self.train_data
            data_source = "TEST+TRAINING"
            print(f"Data source: TEST+TRAINING ({len(self.test_data_cache)} test + {len(self.train_data)} train = {len(data_to_score)} repertoires)")
        elif use_test_data and self.test_data_cache is not None:
            data_to_score = self.test_data_cache
            data_source = "TEST"
            print(f"Data source: TEST ({len(data_to_score)} repertoires)")
        else:
            data_to_score = self.train_data
            data_source = "TRAINING"
            if use_test_data:
                print(f"Data source: TRAINING ({len(data_to_score)} repertoires) [test data not cached]")
            else:
                print(f"Data source: TRAINING ({len(data_to_score)} repertoires)")
        
        print(f"{'='*70}\n")
        
        # =================================================================
        # Score sequences using selected model (or v11 Task 2 method)
        # =================================================================
        info = self.model_info[selected_model]
        mt = info.get('type', selected_model)
        y = self.model['y']
        top_v, top_j = self.model['top_v'], self.model['top_j']
        
        scored_seqs_df = None
        
        # =================================================================
        # VJ_POSITIONAL: Combined VJ + positional AA scoring (v11 Dataset 1)
        # =================================================================
        if task2_method == 'vj_positional':
            print(f"Scoring with VJ_POSITIONAL method (v11 Dataset 1)...")
            
            # Build combined VJ + positional features for training
            features_list = []
            for d in tqdm(self.train_data, desc="  Extracting features"):
                features = {}
                df = d['df']
                total = len(df)
                
                # VJ frequencies
                if 'v_call' in df.columns:
                    v_counts = df['v_call'].dropna().apply(lambda x: str(x).split('*')[0]).value_counts()
                    for gene, count in v_counts.items():
                        features[f'v_{gene}'] = count / total
                
                if 'j_call' in df.columns:
                    j_counts = df['j_call'].dropna().apply(lambda x: str(x).split('*')[0]).value_counts()
                    for gene, count in j_counts.items():
                        features[f'j_{gene}'] = count / total
                
                # Positional AA (first 3, last 3)
                cdr3s = df['junction_aa'].dropna().apply(str).tolist()
                cdr3s = [s.upper() for s in cdr3s if len(s) >= 10 and all(c in VALID_AA for c in s.upper())]
                
                if cdr3s:
                    first_aa = Counter()
                    last_aa = Counter()
                    
                    for seq in cdr3s:
                        for i, aa in enumerate(seq[:3]):
                            first_aa[f'pos{i}_{aa}'] += 1
                        for i, aa in enumerate(seq[-3:]):
                            last_aa[f'pos_end{i}_{aa}'] += 1
                    
                    total_seqs = len(cdr3s)
                    for k, v in first_aa.items():
                        features[k] = v / total_seqs
                    for k, v in last_aa.items():
                        features[k] = v / total_seqs
                
                features_list.append(features)
            
            # Build feature matrix
            all_feature_names = set()
            for f in features_list:
                all_feature_names.update(f.keys())
            feature_names = sorted(all_feature_names)
            
            X = np.zeros((len(self.train_data), len(feature_names)), dtype=np.float32)
            for i, features in enumerate(features_list):
                for name, val in features.items():
                    X[i, feature_names.index(name)] = val
            
            # Train L1 LR (matching v11)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model = LogisticRegression(penalty='l1', solver='saga', C=0.5, max_iter=1000, random_state=CFG.SEED)
            model.fit(X_scaled, y)
            
            # Get coefficients
            coefs = model.coef_[0]
            vj_scores = {}
            pos_scores = {}
            
            for idx, name in enumerate(feature_names):
                if name.startswith('v_') or name.startswith('j_'):
                    vj_scores[name] = coefs[idx]
                elif name.startswith('pos'):
                    pos_scores[name] = coefs[idx]
            
            print(f"  VJ features: {len(vj_scores)}, Positional features: {len(pos_scores)}")
            
            # PARALLEL scoring for vj_positional
            score_args = [(d['df'], vj_scores, pos_scores) for d in data_to_score]
            scored_seqs_df = parallel_score_repertoires(
                data_to_score, _score_repertoire_vj_positional, score_args,
                n_workers=self.n_jobs, desc="Scoring repertoires (vj_positional)"
            )
        
        # K-mer SGD models - use built-in scoring
        elif mt in ['kmer5_sgd', 'kmer6_sgd']:
            print(f"Scoring with {mt}...")
            model = info['model']
            scored_seqs_df = model.score_sequences(data_to_score)
        
        # K-mer LR/RF/XGB models
        elif mt == 'kmer':
            kl, bp = info['kl'], info['bp']
            extractor = info.get('extractor', 'kmer4')
            
            print(f"Scoring with {extractor} ({len(kl)} features)...")
            
            # Get extractor function
            if 'pos' in extractor:
                ext_fn = lambda df: ext_pos_kmer(df, 4)
            elif 'gap' in extractor:
                ext_fn = ext_gap
            else:
                ext_fn = lambda df: ext_kmer(df, 4)
            
            # Train model on training data
            X_tr = build_feature_matrix(self.train_data, 
                                        {d['rep_id']: ext_fn(d['df']) for d in self.train_data},
                                        kl, normalize=info.get('norm', True))
            
            sc = StandardScaler()
            X_tr_scaled = sc.fit_transform(X_tr)
            
            model = LogisticRegression(penalty='l1', C=bp.get('C', 0.1), solver='liblinear',
                                       random_state=CFG.SEED, max_iter=1000)
            model.fit(X_tr_scaled, y)
            
            # Get feature importance
            coefs = model.coef_[0] / sc.scale_
            feature_scores = {km: coefs[i] for i, km in enumerate(kl)}
            
            # Score all unique sequences
            # OPTIMIZED: Extract k-mers FROM sequence instead of checking all 158k+ k-mers
            # This is O(seq_length) instead of O(n_kmers) per sequence - ~10,000x faster!
            # FURTHER OPTIMIZED: Now parallelized across repertoires for 10-16x additional speedup
            
            # Determine k-mer size and extraction method based on extractor type
            if 'pos' in extractor:
                extractor_type = 'positional'
                k = 4
            elif 'gap' in extractor:
                extractor_type = 'gapped'
                k = 4
            else:
                extractor_type = 'standard'
                k = 4
            
            # PARALLEL scoring for k-mer models
            score_args = [(d['df'], feature_scores, k, extractor_type) for d in data_to_score]
            scored_seqs_df = parallel_score_repertoires(
                data_to_score, _score_repertoire_kmer, score_args,
                n_workers=self.n_jobs, desc=f"Scoring repertoires ({extractor})"
            )
        
        # VJ-based models
        elif mt in ['vj', 'vj_interact', 'vj_logfreq', 'vj_elasticnet']:
            fn, bp = info['fn'], info['bp']
            cache_name = info.get('cache_name', 'vj')
            
            print(f"Scoring with {mt} ({len(fn)} features)...")
            
            # Train model
            X_tr = np.zeros((len(self.train_data), len(fn)))
            for i, d in enumerate(self.train_data):
                if cache_name == 'vj_interact':
                    f = {**ext_vj(d['df']), **ext_vj_interact(d['df'], top_v, top_j)}
                else:
                    f = ext_vj(d['df'])
                for j, n in enumerate(fn):
                    X_tr[i, j] = f.get(n, 0)
            
            if mt == 'vj_logfreq':
                X_tr = np.log1p(X_tr)
            
            sc = StandardScaler()
            X_tr_scaled = sc.fit_transform(X_tr)
            
            model = LogisticRegression(penalty='l1', C=bp['C'], solver='liblinear',
                                       random_state=CFG.SEED, max_iter=1000)
            model.fit(X_tr_scaled, y)
            
            # Get feature importance
            coefs = model.coef_[0] / sc.scale_
            feature_scores = {fn[i]: coefs[i] for i in range(len(fn))}
            
            # PARALLEL scoring for VJ models
            score_args = [(d['df'], feature_scores) for d in data_to_score]
            scored_seqs_df = parallel_score_repertoires(
                data_to_score, _score_repertoire_vj, score_args,
                n_workers=self.n_jobs, desc=f"Scoring repertoires ({mt})"
            )
        
        # MALIDVJ model
        elif mt == 'malidvj':
            vec = info['vectorizer']
            feature_names = info['feature_names']
            
            print(f"Scoring with MALIDVJ ({len(feature_names)} features)...")
            
            # Train model
            train_features = [ext_malidvj(d['df']) for d in self.train_data]
            X_train = vec.transform(train_features)
            
            if self.has_age_data and self.age_lookup:
                all_train_idx = np.arange(len(self.train_data))
                robust_idx = select_age_robust_features(
                    all_train_idx, self.train_data, feature_names, X_train, y, self.age_lookup
                )
                X_train = X_train[:, robust_idx]
                feature_names_used = [feature_names[i] for i in robust_idx]
            else:
                feature_names_used = feature_names
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            model = LogisticRegression(penalty='l2', C=1.0, max_iter=1000,
                                       class_weight='balanced', random_state=CFG.SEED)
            model.fit(X_train_scaled, y)
            
            # Get feature importance
            coefs = model.coef_[0] / scaler.scale_
            feature_scores = {feature_names_used[i]: coefs[i] for i in range(len(feature_names_used))}
            
            # Score sequences
            all_seqs = []
            for d in tqdm(data_to_score, desc="  Scoring repertoires"):
                df = d['df']
                if 'junction_aa' not in df.columns:
                    continue
                
                for _, row in df.iterrows():
                    seq = row.get('junction_aa', '')
                    if not isinstance(seq, str) or len(seq) < 8:
                        continue
                    
                    v_gene = str(row.get('v_call', '')).split('*')[0] if pd.notna(row.get('v_call')) else ''
                    j_gene = str(row.get('j_call', '')).split('*')[0] if pd.notna(row.get('j_call')) else ''
                    
                    # Extract features
                    seq_feat = ext_malidvj(pd.DataFrame([row]))
                    score = sum(feature_scores.get(feat, 0) * val for feat, val in seq_feat.items())
                    
                    all_seqs.append({
                        'junction_aa': seq,
                        'v_call': v_gene if v_gene else 'unknown',
                        'j_call': j_gene if j_gene else 'unknown',
                        'importance_score': score
                    })
            
            if all_seqs:
                scored_seqs_df = pd.DataFrame(all_seqs).drop_duplicates(subset=['junction_aa', 'v_call', 'j_call'])
        
        # Positional AA model
        elif mt == 'pos_aa':
            fn, bp = info['fn'], info['bp']
            
            print(f"Scoring with positional AA ({len(fn)} features)...")
            
            # Train model
            X_tr = np.zeros((len(self.train_data), len(fn)))
            for i, d in enumerate(self.train_data):
                f = ext_pos_aa(d['df'])
                for j, n in enumerate(fn):
                    X_tr[i, j] = f.get(n, 0)
            
            sc = StandardScaler()
            X_tr_scaled = sc.fit_transform(X_tr)
            
            model = LogisticRegression(penalty='l1', C=bp['C'], solver='liblinear',
                                       random_state=CFG.SEED, max_iter=1000)
            model.fit(X_tr_scaled, y)
            
            # Get feature importance
            coefs = model.coef_[0] / sc.scale_
            feature_scores = {fn[i]: coefs[i] for i in range(len(fn))}
            
            # Score sequences
            all_seqs = []
            for d in tqdm(data_to_score, desc="  Scoring repertoires"):
                df = d['df']
                if 'junction_aa' not in df.columns:
                    continue
                
                for _, row in df.iterrows():
                    seq = row.get('junction_aa', '')
                    if not isinstance(seq, str) or len(seq) < 8:
                        continue
                    
                    v_gene = str(row.get('v_call', '')).split('*')[0] if pd.notna(row.get('v_call')) else ''
                    j_gene = str(row.get('j_call', '')).split('*')[0] if pd.notna(row.get('j_call')) else ''
                    
                    # Extract features
                    seq_feat = ext_pos_aa(pd.DataFrame([row]))
                    score = sum(feature_scores.get(feat, 0) * val for feat, val in seq_feat.items())
                    
                    all_seqs.append({
                        'junction_aa': seq,
                        'v_call': v_gene if v_gene else 'unknown',
                        'j_call': j_gene if j_gene else 'unknown',
                        'importance_score': score
                    })
            
            if all_seqs:
                scored_seqs_df = pd.DataFrame(all_seqs).drop_duplicates(subset=['junction_aa', 'v_call', 'j_call'])
        
        # Emerson model
        elif mt == 'emerson':
            bp = info['bp']
            
            print(f"Scoring with Emerson (p={bp['p']}, mc={bp['mc']})...")
            
            # Train Emerson classifier
            train_seqs = [self.caches['emerson'][d['rep_id']] for d in self.train_data]
            clf = EmClf(p=bp['p'], mc=bp['mc'])
            clf.fit(train_seqs, y)
            
            if not clf.seqs:
                print("  WARNING: No enriched sequences found")
                return pd.DataFrame()
            
            print(f"  Found {len(clf.seqs)} enriched sequences")
            
            # PARALLEL scoring for Emerson model
            score_args = [(d['df'], clf.seqs) for d in data_to_score]
            scored_seqs_df = parallel_score_repertoires(
                data_to_score, _score_repertoire_emerson, score_args,
                n_workers=self.n_jobs, desc="Scoring repertoires (emerson)"
            )
        
        # =================================================================
        # Format and return
        # =================================================================
        if scored_seqs_df is None or len(scored_seqs_df) == 0:
            print("  WARNING: No sequences scored, using fallback...")
            # Fallback to simple 4-mer scoring
            return self._fallback_scoring(data_to_score, dataset_name, top_k, y)
        
        print(f"  Total unique sequences: {len(scored_seqs_df)}")
        
        # Sort and take top k
        top_sequences_df = scored_seqs_df.nlargest(top_k, 'importance_score')
        
        # Format for competition
        top_sequences_df = top_sequences_df[['junction_aa', 'v_call', 'j_call']].copy()
        top_sequences_df['dataset'] = dataset_name
        top_sequences_df['ID'] = [f"{dataset_name}_seq_top_{i+1}" for i in range(len(top_sequences_df))]
        top_sequences_df['label_positive_probability'] = -999.0
        
        result = top_sequences_df[['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']]
        
        print(f"  Returning top {len(result)} sequences")
        print(f"  Score range: {scored_seqs_df['importance_score'].min():.4f} to {scored_seqs_df['importance_score'].max():.4f}")
        print(f"{'='*70}\n")
        
        return result
    
    def _fallback_scoring(self, data_to_score, dataset_name, top_k, y):
        """Fallback 4-mer scoring method."""
        print("  Using fallback 4-mer L1 LR scoring...")
        
        # Build 4-mer vocabulary
        kmer_vocab = Counter()
        for d in self.train_data:
            for seq in d['df']['junction_aa'].dropna():
                if isinstance(seq, str):
                    seq = ''.join(c for c in seq.upper() if c in VALID_AA)
                    for i in range(len(seq) - 3):
                        kmer_vocab[seq[i:i+4]] += 1
        
        kmer_list = [km for km, c in kmer_vocab.items() if c >= 5]
        kmer_to_idx = {km: i for i, km in enumerate(kmer_list)}
        
        # Build feature matrix
        X = np.zeros((len(self.train_data), len(kmer_list)), dtype=np.float32)
        for i, d in enumerate(self.train_data):
            cnt = Counter()
            total = 0
            for seq in d['df']['junction_aa'].dropna():
                if isinstance(seq, str):
                    seq = ''.join(c for c in seq.upper() if c in VALID_AA)
                    for j in range(len(seq) - 3):
                        km = seq[j:j+4]
                        if km in kmer_to_idx:
                            cnt[km] += 1
                            total += 1
            for km, c in cnt.items():
                X[i, kmer_to_idx[km]] = c / (total + 1)
        
        # Train model
        sc = StandardScaler()
        X_scaled = sc.fit_transform(X)
        model = LogisticRegression(penalty='l1', solver='saga', C=0.1, max_iter=1000, random_state=CFG.SEED)
        model.fit(X_scaled, y)
        
        # Get scores
        coefs = model.coef_[0] / sc.scale_
        kmer_scores = {km: coefs[idx] for km, idx in kmer_to_idx.items()}
        
        # Score all sequences
        all_seqs = []
        for d in data_to_score:
            df = d['df']
            if 'junction_aa' not in df.columns:
                continue
            
            for _, row in df.iterrows():
                seq = row.get('junction_aa', '')
                if not isinstance(seq, str) or len(seq) < 4:
                    continue
                
                v_gene = str(row.get('v_call', '')).split('*')[0] if pd.notna(row.get('v_call')) else 'unknown'
                j_gene = str(row.get('j_call', '')).split('*')[0] if pd.notna(row.get('j_call')) else 'unknown'
                
                seq_clean = ''.join(c for c in seq.upper() if c in VALID_AA)
                score = sum(kmer_scores.get(seq_clean[i:i+4], 0) for i in range(len(seq_clean) - 3))
                
                all_seqs.append({
                    'junction_aa': seq,
                    'v_call': v_gene,
                    'j_call': j_gene,
                    'importance_score': score
                })
        
        if not all_seqs:
            return pd.DataFrame()
        
        unique_seqs = pd.DataFrame(all_seqs).drop_duplicates(subset=['junction_aa', 'v_call', 'j_call'])
        top_sequences_df = unique_seqs.nlargest(top_k, 'importance_score')
        
        # Format
        top_sequences_df = top_sequences_df[['junction_aa', 'v_call', 'j_call']].copy()
        top_sequences_df['dataset'] = dataset_name
        top_sequences_df['ID'] = [f"{dataset_name}_seq_top_{i+1}" for i in range(len(top_sequences_df))]
        top_sequences_df['label_positive_probability'] = -999.0
        
        return top_sequences_df[['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']]



# =============================================================================
# Main Workflow Functions (Competition Template)
# =============================================================================

def _train_predictor(predictor: ImmuneStatePredictor, train_dir: str):
    """Trains the predictor on the training data."""
    print(f"Fitting model on examples in `{train_dir}`...")
    predictor.fit(train_dir)


def _generate_predictions(predictor: ImmuneStatePredictor, test_dirs: List[str], 
                          out_dir: str = None, train_dir: str = None) -> pd.DataFrame:
    """
    Generates predictions for all test directories and caches test data for Task 2.
    
    For datasets 7 and 8, saves each test set's predictions IMMEDIATELY after generating,
    so partial results are preserved if the job crashes.
    
    Args:
        predictor: The trained predictor
        test_dirs: List of test directory paths
        out_dir: Output directory (needed for immediate saving)
        train_dir: Training directory (needed for output filename)
    
    Returns:
        Combined predictions DataFrame
    """
    all_preds = []
    all_test_data = []  # Accumulate test data for Task 2
    
    train_basename = os.path.basename(train_dir) if train_dir else "predictions"
    
    # Sort test_dirs to ensure consistent ordering for numbered output files
    sorted_test_dirs = sorted(test_dirs)
    
    for i, test_dir in enumerate(sorted_test_dirs, start=1):
        print(f"Predicting on examples in `{test_dir}`...")
        
        # Load test data to cache for Task 2
        test_data = load_test_data(test_dir, n_workers=predictor.n_jobs)
        if test_data:
            all_test_data.extend(test_data)
        
        # Generate predictions
        try:
            preds = predictor.predict_proba(test_dir)
            if preds is not None and not preds.empty:
                all_preds.append(preds)
        except Exception as e:
            print(f"  ERROR predicting on {test_dir}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Cache test data for Task 2 sequence extraction
    if all_test_data:
        predictor.test_data_cache = all_test_data
        print(f"\n  Cached {len(all_test_data)} test repertoires for Task 2")
    
    if all_preds:
        combined = pd.concat(all_preds, ignore_index=True)
        # FIXED: Save predictions IMMEDIATELY after generating them (before Task 2)
        # This ensures predictions are saved even if Task 2 feature extraction fails/times out
        preds_path = os.path.join(out_dir, f"{train_basename}_test_predictions.tsv")
        save_tsv(combined, preds_path)
        print(f"Predictions written to `{preds_path}` (saved early to prevent data loss).")
        return combined
    return pd.DataFrame()


def _save_predictions(predictions: pd.DataFrame, out_dir: str, train_dir: str) -> None:
    """
    Saves combined predictions to TSV file.
    
    FIXED: Predictions are now saved in _generate_predictions() to prevent data loss
    if Task 2 fails or times out. This function is kept for backward compatibility.
    """
    if predictions.empty:
        print("Warning: No predictions to save")
        return
    # Already saved in _generate_predictions
    pass


def _save_important_sequences(predictor: ImmuneStatePredictor, out_dir: str, train_dir: str) -> None:
    """Saves important sequences to a TSV file."""
    seqs = predictor.important_sequences_
    if seqs is None or seqs.empty:
        print("Warning: No important sequences to save")
        return
    seqs_path = os.path.join(out_dir, f"{os.path.basename(train_dir)}_important_sequences.tsv")
    save_tsv(seqs, seqs_path)
    print(f"Important sequences written to `{seqs_path}`.")


def main(train_dir: str, test_dirs: List[str], out_dir: str, n_jobs: int, device: str) -> None:
    """Main entry point for competition."""
    validate_dirs_and_files(train_dir, test_dirs, out_dir)
    predictor = ImmuneStatePredictor(n_jobs=n_jobs, device=device)
    _train_predictor(predictor, train_dir)
    predictions = _generate_predictions(predictor, test_dirs, out_dir, train_dir)
    _save_predictions(predictions, out_dir, train_dir)
    _save_important_sequences(predictor, out_dir, train_dir)


def run():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="AIRR-ML Immune State Predictor (Unified v8)")
    parser.add_argument("--train_dir", required=True, help="Path to training data directory")
    parser.add_argument("--test_dirs", required=True, nargs="+", help="Path(s) to test data director(ies)")
    parser.add_argument("--out_dir", required=True, help="Path to output directory")
    parser.add_argument("--n_jobs", type=int, default=4, help="Number of CPU cores (-1 for all)")
    parser.add_argument("--device", type=str, default='cpu', choices=['cpu', 'cuda'], help="Device for computation")
    args = parser.parse_args()
    main(args.train_dir, args.test_dirs, args.out_dir, args.n_jobs, args.device)


if __name__ == "__main__":
    run()
