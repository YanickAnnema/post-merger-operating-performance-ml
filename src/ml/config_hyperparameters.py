"""
Hyperparameter Configuration — XGBoost for M&A Synergy Estimation
=================================================================

Defines parameter grids for cross-validation on the training set.
Imported by model_training.py.

Design rationale:
  - Regression task (continuous CFROA target), so objective = 'reg:squarederror'.
  - Small dataset (~3,100 train) → moderate depth, strong regularisation.
  - High feature sparsity (some features ~27% coverage) → XGBoost handles
    natively via missing=np.nan; no imputation needed.
  - Grid is intentionally compact to keep CV runtime manageable while
    covering the key axes: complexity (depth/estimators), regularisation
    (alpha/lambda), and learning rate.

Optimised for Spyder IDE (F5 execution).
"""

import numpy as np

# =============================================================================
# BASE PARAMETERS (fixed across all grid points)
# =============================================================================

BASE_PARAMS = {
    'objective':        'reg:squarederror',
    'booster':          'gbtree',
    'tree_method':      'hist',          # fast histogram-based; handles NaN natively
    'eval_metric':      'rmse',
    'verbosity':        0,
    'seed':             42,
    'n_jobs':           -1,
}


# =============================================================================
# HYPERPARAMETER GRID — searched via CV on training set only
# =============================================================================
#
# Axes:
#   learning_rate   : step size shrinkage — lower = more trees needed but better generalisation
#   max_depth       : tree depth — controls interaction order; 3–6 standard for tabular
#   n_estimators    : boosting rounds — upper bound; early stopping trims this
#   subsample       : row sampling per tree — regularisation against overfitting
#   colsample_bytree: feature sampling per tree — decorrelates trees
#   reg_alpha       : L1 penalty on leaf weights — sparsity-inducing
#   reg_lambda      : L2 penalty on leaf weights — shrinkage
#   min_child_weight: minimum sum of instance weight in a leaf — prevents fitting noise
#   gamma           : minimum loss reduction for a split — pruning control

PARAM_GRID = {
    'learning_rate':    [0.01, 0.05, 0.1],
    'max_depth':        [3, 4, 5, 6],
    'n_estimators':     [200, 500, 1000],
    'subsample':        [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'reg_alpha':        [0.0, 0.1, 1.0],
    'reg_lambda':       [1.0, 5.0, 10.0],
    'min_child_weight': [1, 3, 5],
    'gamma':            [0.0, 0.1, 0.5],
}

# Total grid is very large (~59k combos). We use RandomizedSearchCV with
# a fixed budget of iterations instead of exhaustive GridSearchCV.
N_RANDOM_ITER = 100      # number of random hyperparameter combos to evaluate
CV_FOLDS      = 5        # k-fold temporal block CV within training set
EARLY_STOP    = 50       # early stopping patience (rounds without improvement)


# =============================================================================
# COMPACT GRID — for quick iteration / debugging
# =============================================================================

PARAM_GRID_COMPACT = {
    'learning_rate':    [0.05, 0.1],
    'max_depth':        [3, 5],
    'n_estimators':     [300, 700],
    'subsample':        [0.8],
    'colsample_bytree': [0.8],
    'reg_alpha':        [0.0, 1.0],
    'reg_lambda':       [1.0, 5.0],
    'min_child_weight': [1, 5],
    'gamma':            [0.0],
}

N_RANDOM_ITER_COMPACT = 20


# =============================================================================
# CROSS-VALIDATION DESIGN
# =============================================================================
#
# Temporal block CV on training set (1995–2015):
#   - TimeSeriesSplit with 5 folds: each fold trains on earlier data,
#     validates on the next temporal block.
#   - Preserves chronological order within the training window.
#   - No shuffling at any stage.
#
# After CV selects best hyperparameters:
#   1. Retrain on full training set with best params
#   2. Evaluate on val (2016–2018) — model selection
#   3. Evaluate on test (2019–2022) — final report only (no tuning)


if __name__ == "__main__":
    # Quick sanity: print grid sizes
    from functools import reduce
    import operator

    full_size = reduce(operator.mul, [len(v) for v in PARAM_GRID.values()], 1)
    compact_size = reduce(operator.mul, [len(v) for v in PARAM_GRID_COMPACT.values()], 1)

    print(f"Full grid size:    {full_size:,} combos → sampling {N_RANDOM_ITER}")
    print(f"Compact grid size: {compact_size:,} combos → sampling {N_RANDOM_ITER_COMPACT}")
    print(f"CV folds:          {CV_FOLDS}")
    print(f"Early stopping:    {EARLY_STOP} rounds")
    print(f"\nBase params:")
    for k, v in BASE_PARAMS.items():
        print(f"  {k:<20}: {v}")
