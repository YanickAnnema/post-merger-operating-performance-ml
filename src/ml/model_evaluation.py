"""
Model Evaluation -- Advanced Diagnostics
=========================================

Runs AFTER model_training.py.  Loads the fitted model and prepared splits
from outputs/ and executes four post-training diagnostic routines:

  1. Frozen baseline spec
       Documents the locked split/target/feature definitions for thesis
       reproducibility.  Writes a frozen_spec.txt audit file.

  2. Diebold-Mariano significance test (XGBoost vs naive_mean on test errors)
       DM statistic with Harvey-Leybourne-Newbold (1997) small-sample correction.
       Squared-error loss function.  One-sided H1: XGBoost has lower MSE.
       Paired permutation test included as a distribution-free complement.

       Note on DM assumptions: the DM test was designed for time-series forecast
       errors.  In a cross-sectional M&A setting the iid assumption is more
       defensible than strict stationarity, but the test should be interpreted
       as a guide rather than a formal proof of significance.

  3. Time-slice stability (2019, 2020, 2021, 2022 separately)
       Subsets the test set by deal_year and reports RMSE, MAE, R2, Spearman.
       Detects regime sensitivity (e.g., COVID-19 disruption in 2020-2021).
       Produces timeslice_stability.csv + timeslice_stability.png.

  4. Channel ablation study
       Retrains XGBoost on train_cv using best hyperparameters but with one
       synergy channel's features removed at a time.  Early stopping on
       inner_val.  Evaluates on val and test.
       Incremental channel value = ablated test RMSE - full model test RMSE.
       Positive = removing that channel hurts performance.
       Produces channel_ablation.csv + channel_ablation.png.

Inputs (from ~/Desktop/outputs/):
  xgboost_model_final.pkl
  training_splits.pkl

Outputs (to ~/Desktop/outputs/):
  frozen_spec.txt
  significance_tests.txt
  timeslice_stability.csv
  channel_ablation.csv
  plots/
    timeslice_stability.png
    channel_ablation.png

Prerequisites:
  Run model_training.py first to produce xgboost_model_final.pkl and
  training_splits.pkl.

Optimised for Spyder IDE (F5 execution).
"""

import numpy as np
import pandas as pd
import pickle
import logging
import time
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Ensure code/ is on sys.path for Spyder / repo-root execution
_code_dir = str(Path(__file__).resolve().parent)
if _code_dir not in sys.path:
    sys.path.insert(0, _code_dir)

import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr, t as t_dist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Frozen spec / channel definitions -- imported from training module to ensure
# model_evaluation.py is always consistent with model_training.py
from model_training import (
    ALL_FEATURES,
    FEATURES_COST,
    FEATURES_REVENUE,
    FEATURES_OPERATIONAL,
    FEATURES_FINANCIAL,
    FEATURES_MACRO,
    CHANNEL_MAP,
    BINARY_FEATURES,
    POST_DEAL_COLS,
    BASE_PARAMS,
    CONFIG as TRAIN_CONFIG,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ML pipeline"

CONFIG = {
    'output_dir':      ML_OUTPUT_DIR,
    'plot_dir':        ML_OUTPUT_DIR / "plots",
    'model_pkl':       ML_OUTPUT_DIR / "xgboost_model_final.pkl",
    'splits_pkl':      ML_OUTPUT_DIR / "training_splits.pkl",

    # DM test
    'dm_loss':         'squared',   # 'squared' or 'absolute'
    'n_permutations':  10_000,

    # Bootstrap CI (for ablation)
    'n_bootstrap':     1_000,

    # Early stopping for ablation retraining
    'early_stop':      50,

    'seed': 42,
}

# Channels to ablate -- one removed at a time
CHANNEL_FEATURE_MAP: Dict[str, List[str]] = {
    'cost':        FEATURES_COST,
    'revenue':     FEATURES_REVENUE,
    'operational': FEATURES_OPERATIONAL,
    'financial':   FEATURES_FINANCIAL,
    'macro':       FEATURES_MACRO,
}


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# =============================================================================
# LOAD ARTIFACTS
# =============================================================================

def load_artifacts() -> Tuple[xgb.XGBRegressor, Dict, Dict]:
    """
    Load fitted model and prepared data splits from disk.

    Returns: (model, model_meta, splits)
    """
    model_pkl  = CONFIG['model_pkl']
    splits_pkl = CONFIG['splits_pkl']

    if not model_pkl.exists():
        raise FileNotFoundError(
            f"Model not found: {model_pkl}\n"
            "Run model_training.py first."
        )
    if not splits_pkl.exists():
        raise FileNotFoundError(
            f"Splits not found: {splits_pkl}\n"
            "Run model_training.py first."
        )

    with open(model_pkl, 'rb') as f:
        model_meta = pickle.load(f)
    with open(splits_pkl, 'rb') as f:
        splits = pickle.load(f)

    model = model_meta['model']
    logger.info(
        f"Loaded model: {model.n_estimators} estimators, "
        f"best_iteration={model_meta['best_iteration']}"
    )
    logger.info(
        f"Loaded splits: "
        f"train_cv={len(splits['X_train_cv']):,}  "
        f"inner_val={len(splits['X_inner_val']):,}  "
        f"val={len(splits['X_val']):,}  "
        f"test={len(splits['X_test']):,}"
    )

    return model, model_meta, splits


# =============================================================================
# SECTION 1 -- FROZEN BASELINE SPEC
# =============================================================================

def write_frozen_spec(
    model_meta: Dict,
    splits: Dict,
    output_dir: Path,
) -> None:
    """
    Write a thesis-reference audit file documenting all locked definitions:
    split boundaries, target variable, feature set, transformation parameters.
    Serves as the reproducibility anchor for the methodology chapter.
    """
    logger.info("=" * 60)
    logger.info("FROZEN BASELINE SPEC")
    logger.info("=" * 60)

    cfg = TRAIN_CONFIG
    features = splits['features_used']
    lines = [
        "=" * 72,
        "  FROZEN BASELINE SPECIFICATION",
        "  M&A Synergy Estimation -- XGBoost Pipeline",
        "=" * 72,
        f"Generated : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "PURPOSE",
        "  This file locks the canonical definitions used in the thesis.",
        "  Any change to split years, target, or feature list invalidates",
        "  prior results and requires full re-run from data_preparation.py.",
        "",
        "TARGET VARIABLE",
        f"  {cfg['target_col']}",
        "  Industry-adjusted CFROA difference, winsorised at 1%/99%.",
        "  Construction: Healy et al. (1992), industry benchmark at SIC-2 level.",
        "",
        "CHRONOLOGICAL SPLIT DESIGN  (no shuffling; temporal order preserved)",
        f"  Sample start : {cfg['sample_start_year']}",
        f"  Train window : 1995-{cfg['split_years']['train_end']}  "
        f"(inner holdout {cfg['early_stop_year']}-{cfg['split_years']['train_end']} "
        f"reserved for early stopping)",
        f"  Val window   : {cfg['split_years']['train_end']+1}-{cfg['split_years']['val_end']}  "
        f"(clean, never fitted on)",
        f"  Test window  : {cfg['split_years']['val_end']+1}-2022  "
        f"(clean, primary out-of-sample report)",
        "",
        "SPLIT SIZES",
        f"  train_cv  : {len(splits['X_train_cv']):,}",
        f"  inner_val : {len(splits['X_inner_val']):,}",
        f"  val       : {len(splits['X_val']):,}",
        f"  test      : {len(splits['X_test']):,}",
        "",
        f"FEATURE SET  ({len(features)} features; pre-deal only)",
    ]

    for ch_name, ch_feats in CHANNEL_FEATURE_MAP.items():
        present = [f for f in ch_feats if f in features]
        lines.append(f"  {ch_name:<14}: {len(present)} features")
        for f in present:
            lines.append(f"    - {f}")

    lines += [
        "",
        "TRANSFORMATIONS  (all bounds fitted on train_cv only)",
        f"  Winsorisation  : [{cfg['winsor_low']:.0%}, {cfg['winsor_high']:.0%}]",
        "  Imputation     : none (XGBoost handles NaN natively)",
        "  Scaling        : none (XGBoost is scale-invariant)",
        "",
        "MODEL",
        "  Algorithm      : XGBoost (xgb.XGBRegressor)",
        "  Objective      : reg:squarederror",
        f"  Tree method    : {BASE_PARAMS.get('tree_method', 'hist')}",
        f"  Best params    :",
    ]
    for k, v in sorted(model_meta['best_params'].items()):
        lines.append(f"    {k:<25}: {v}")
    lines += [
        f"    {'best_iteration':<25}: {model_meta['best_iteration']}",
        "",
        "LEAKAGE CONTROLS",
        "  1. Input file: ml_ready_nowinsor.csv (no full-sample clipping).",
        "  2. All transformation bounds computed on train_cv only.",
        "  3. Hard POST_DEAL_COLS guard prevents target-construction fields",
        "     from entering the feature matrix.",
        "  4. Inner holdout separates early stopping from val/test evaluation.",
        "",
        "STATUS: FROZEN -- any change requires explicit version bump",
        "=" * 72,
    ]

    spec_path = output_dir / "frozen_spec.txt"
    with open(spec_path, 'w') as f:
        f.write("\n".join(lines))
    logger.info(f"  Saved: {spec_path.name}")


# =============================================================================
# SECTION 2 -- SIGNIFICANCE TESTS
# =============================================================================

def diebold_mariano(
    e_naive: np.ndarray,
    e_model: np.ndarray,
    loss: str = 'squared',
    h: int = 1,
) -> Dict:
    """
    Diebold-Mariano test with Harvey-Leybourne-Newbold (1997) small-sample
    correction.

    H0: equal predictive accuracy (E[d] = 0)
    H1: model has lower loss than naive (E[d] > 0, one-sided)

    where d_i = L(e_naive_i) - L(e_model_i) and positive d means model is
    better on observation i.

    Args:
        e_naive: forecast errors from naive baseline (actual - predicted_naive)
        e_model: forecast errors from XGBoost (actual - predicted_xgb)
        loss:    'squared' (MSE-based) or 'absolute' (MAE-based)
        h:       forecast horizon (1 for one-step-ahead / cross-sectional)

    Returns dict with DM_raw, DM_hln, p_value_one_sided, d_bar, T.
    """
    if loss == 'squared':
        L_naive = e_naive ** 2
        L_model = e_model ** 2
    elif loss == 'absolute':
        L_naive = np.abs(e_naive)
        L_model = np.abs(e_model)
    else:
        raise ValueError(f"Unknown loss: {loss}. Use 'squared' or 'absolute'.")

    # Loss differential: positive when model has lower loss
    d = L_naive - L_model
    T = len(d)
    d_bar = d.mean()

    # Variance: plain sample variance / T for h=1.
    # For h > 1 a Newey-West estimator covering h-1 lags would be needed.
    # Since our setting is cross-sectional (h=1), no autocorrelation correction
    # is applied beyond the plain variance.
    var_d = np.var(d, ddof=1)
    se_d  = np.sqrt(var_d / T)

    if se_d < 1e-12:
        logger.warning("  DM: near-zero se(d) -- test may be degenerate")
        se_d = 1e-12

    DM_raw = d_bar / se_d

    # HLN small-sample correction: scales DM by sqrt((T+1-2h+h(h-1)/T)/T).
    # For h=1 this simplifies to sqrt((T-1)/T) -- minor correction for T~600.
    hln = np.sqrt(max((T + 1 - 2 * h + h * (h - 1) / T) / T, 0))
    DM_hln = DM_raw * hln

    # One-sided p-value: P(t_{T-1} > DM_hln) under H0
    p_one = 1.0 - float(t_dist.cdf(DM_hln, df=T - 1))

    return {
        'DM_raw':           float(DM_raw),
        'DM_hln':           float(DM_hln),
        'p_value_one_sided': p_one,
        'd_bar':            float(d_bar),
        'se_d':             float(se_d),
        'T':                T,
        'loss':             loss,
        'frac_d_positive':  float((d > 0).mean()),
    }


def paired_permutation_test(
    e_naive: np.ndarray,
    e_model: np.ndarray,
    loss: str = 'squared',
    n_perm: int = None,
    seed: int = None,
) -> Dict:
    """
    Paired permutation significance test -- no distributional assumptions.

    For each observation i, d_i = L(e_naive_i) - L(e_model_i).
    Under H0: E[d] = 0, the sign of d_i is exchangeable.
    We randomly flip the sign of each d_i and observe how often the
    resulting mean >= the observed mean(d).

    H0: equal predictive accuracy.
    H1: model has lower loss than naive (one-sided).

    p-value = P(mean(sign-flipped d) >= observed mean(d)) under H0.
    """
    if n_perm is None:
        n_perm = CONFIG['n_permutations']
    if seed is None:
        seed = CONFIG['seed']

    if loss == 'squared':
        d = e_naive ** 2 - e_model ** 2
    else:
        d = np.abs(e_naive) - np.abs(e_model)

    observed = d.mean()
    rng = np.random.default_rng(seed)
    n = len(d)

    count_geq = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=n)
        if (signs * d).mean() >= observed:
            count_geq += 1

    p_perm = count_geq / n_perm
    return {
        'observed_d_bar': float(observed),
        'p_value_permutation': float(p_perm),
        'n_permutations': n_perm,
        'frac_d_positive': float((d > 0).mean()),
        'loss': loss,
        'T': n,
    }


def run_significance_tests(
    model: xgb.XGBRegressor,
    splits: Dict,
    output_dir: Path,
) -> Dict:
    """
    Run DM + permutation tests: XGBoost vs naive_mean on the TEST set.

    Note: significance tests are run on test set only (primary unbiased
    evaluation set).  Val is not used here.
    """
    logger.info("=" * 60)
    logger.info("SIGNIFICANCE TESTS (test set)")
    logger.info("=" * 60)

    X_test       = splits['X_test']
    y_test       = splits['y_test']
    X_train_cv   = splits['X_train_cv']
    y_train_cv   = splits['y_train_cv']

    train_mean   = float(y_train_cv.mean())
    y_arr        = y_test.values

    pred_xgb     = model.predict(X_test)
    pred_naive   = np.full(len(y_arr), train_mean)

    e_naive = y_arr - pred_naive
    e_model = y_arr - pred_xgb

    # DM test
    dm = diebold_mariano(e_naive, e_model, loss=CONFIG['dm_loss'], h=1)

    # Permutation test
    perm = paired_permutation_test(
        e_naive, e_model, loss=CONFIG['dm_loss'],
        n_perm=CONFIG['n_permutations'], seed=CONFIG['seed']
    )

    # Descriptive
    rmse_xgb   = float(np.sqrt(mean_squared_error(y_arr, pred_xgb)))
    rmse_naive = float(np.sqrt(mean_squared_error(y_arr, pred_naive)))
    rel_imp    = (rmse_naive - rmse_xgb) / rmse_naive * 100

    logger.info(
        f"  Test RMSE: XGBoost={rmse_xgb:.6f}  naive={rmse_naive:.6f}  "
        f"rel. improvement={rel_imp:.2f}%"
    )
    logger.info(
        f"  DM (HLN) = {dm['DM_hln']:.4f}  "
        f"p (one-sided) = {dm['p_value_one_sided']:.4f}  "
        f"[loss={dm['loss']}, T={dm['T']}]"
    )
    logger.info(
        f"  Permutation p = {perm['p_value_permutation']:.4f}  "
        f"[{perm['n_permutations']:,} permutations]"
    )

    # Also run DM with absolute loss for robustness
    dm_abs   = diebold_mariano(e_naive, e_model, loss='absolute', h=1)
    perm_abs = paired_permutation_test(
        e_naive, e_model, loss='absolute',
        n_perm=CONFIG['n_permutations'], seed=CONFIG['seed']
    )
    logger.info(
        f"  DM absolute (HLN) = {dm_abs['DM_hln']:.4f}  "
        f"p = {dm_abs['p_value_one_sided']:.4f}"
    )

    # Interpret
    alpha = 0.05
    reject_dm   = dm['p_value_one_sided'] < alpha
    reject_perm = perm['p_value_permutation'] < alpha
    interpretation = (
        "XGBoost significantly outperforms naive_mean at the 5% level "
        "(both DM and permutation test)"
        if reject_dm and reject_perm else
        "XGBoost outperforms naive_mean but not at 5% significance "
        "(consistent with weak but positive signal)"
        if dm['d_bar'] > 0 else
        "XGBoost does NOT outperform naive_mean on this test set"
    )
    logger.info(f"  Interpretation: {interpretation}")

    # Write output
    lines = [
        "=" * 72,
        "  SIGNIFICANCE TESTS -- XGBoost vs naive_mean (test set)",
        "=" * 72,
        "",
        f"Test set: n={dm['T']}  train_mean={train_mean:.6f}",
        f"RMSE: XGBoost={rmse_xgb:.6f}  naive={rmse_naive:.6f}  "
        f"relative improvement={rel_imp:.2f}%",
        "",
        "DIEBOLD-MARIANO TEST (Harvey-Leybourne-Newbold small-sample correction)",
        f"  H0: equal predictive accuracy",
        f"  H1: XGBoost has lower {dm['loss']} error (one-sided)",
        f"  d_i = L(naive_i) - L(xgb_i); positive d = XGBoost better on obs i",
        f"  mean(d) = {dm['d_bar']:.6f}  se(d) = {dm['se_d']:.6f}",
        f"  DM_raw (t-stat)   = {dm['DM_raw']:.4f}",
        f"  DM_hln (corrected)= {dm['DM_hln']:.4f}",
        f"  p-value (one-sided, t_{dm['T']-1}) = {dm['p_value_one_sided']:.4f}",
        f"  Fraction of obs where XGBoost better = {dm['frac_d_positive']:.1%}",
        "",
        "  With absolute-error loss:",
        f"  DM_hln = {dm_abs['DM_hln']:.4f}  p = {dm_abs['p_value_one_sided']:.4f}",
        "",
        "  NOTE: DM was designed for time-series forecast errors. In this",
        "  cross-sectional setting the iid assumption is more plausible than",
        "  stationarity, but interpret as indicative, not strictly formal.",
        "",
        "PAIRED PERMUTATION TEST  (distribution-free)",
        f"  H0: equal predictive accuracy",
        f"  H1: XGBoost has lower {perm['loss']} error (one-sided)",
        f"  Observed mean(d) = {perm['observed_d_bar']:.6f}",
        f"  p-value = {perm['p_value_permutation']:.4f}  "
        f"[{perm['n_permutations']:,} sign-flip permutations]",
        "",
        "  With absolute-error loss:",
        f"  p = {perm_abs['p_value_permutation']:.4f}",
        "",
        "INTERPRETATION",
        f"  {interpretation}",
        "",
        "=" * 72,
    ]

    sig_path = output_dir / "significance_tests.txt"
    with open(sig_path, 'w') as f:
        f.write("\n".join(lines))
    logger.info(f"  Saved: {sig_path.name}")

    return {
        'dm_squared': dm, 'dm_absolute': dm_abs,
        'perm_squared': perm, 'perm_absolute': perm_abs,
        'rmse_xgb': rmse_xgb, 'rmse_naive': rmse_naive,
        'rel_improvement': rel_imp,
    }


# =============================================================================
# SECTION 3 -- TIME-SLICE STABILITY
# =============================================================================

def run_timeslice_stability(
    model: xgb.XGBRegressor,
    splits: Dict,
    output_dir: Path,
    plot_dir: Path,
) -> pd.DataFrame:
    """
    Compute per-year metrics on the test set (2019-2022).
    Detects regime sensitivity -- e.g. COVID-19 disruption (2020-2021)
    or the post-COVID M&A surge (2021-2022).

    Each year is evaluated independently.  Low n in individual years
    means metrics are noisy; error bars are added to the plot.
    """
    logger.info("=" * 60)
    logger.info("TIME-SLICE STABILITY (test set: 2019-2022)")
    logger.info("=" * 60)

    df_test_meta = splits['df_test_meta']
    X_test       = splits['X_test']
    y_test       = splits['y_test']
    features     = splits['features_used']
    y_train_mean = float(splits['y_train_cv'].mean())

    if 'deal_year' not in df_test_meta.columns:
        logger.warning("  'deal_year' not in df_test_meta -- time-slice skipped")
        return pd.DataFrame()

    pred_full = model.predict(X_test)
    years = sorted(
        pd.to_numeric(df_test_meta['deal_year'], errors='coerce').dropna().unique()
    )

    records = []
    for yr in years:
        yr = int(yr)
        mask = pd.to_numeric(df_test_meta['deal_year'], errors='coerce') == yr
        n = int(mask.sum())
        if n < 5:
            logger.info(f"  {yr}: n={n} -- too few observations, skipped")
            continue

        y_yr   = y_test.values[mask]
        p_yr   = pred_full[mask]
        p_naiv = np.full(n, y_train_mean)

        rmse_xgb  = float(np.sqrt(mean_squared_error(y_yr, p_yr)))
        rmse_naiv = float(np.sqrt(mean_squared_error(y_yr, p_naiv)))
        r2_yr     = float(r2_score(y_yr, p_yr))
        mae_yr    = float(mean_absolute_error(y_yr, p_yr))
        sp_yr, sp_p = spearmanr(y_yr, p_yr)

        records.append({
            'year': yr,
            'n': n,
            'rmse_xgb': rmse_xgb,
            'rmse_naive': rmse_naiv,
            'rmse_ratio': rmse_xgb / rmse_naiv,   # < 1 = XGBoost better
            'mae': mae_yr,
            'r2': r2_yr,
            'spearman': float(sp_yr),
            'sp_pval': float(sp_p),
            'y_mean': float(y_yr.mean()),
            'y_std': float(y_yr.std()),
        })
        logger.info(
            f"  {yr}: n={n:>4}  RMSE={rmse_xgb:.5f}  "
            f"naive={rmse_naiv:.5f}  R2={r2_yr:.4f}  "
            f"Spearman={float(sp_yr):.4f}"
        )

    df_ts = pd.DataFrame(records)
    if df_ts.empty:
        logger.warning("  No valid years -- timeslice DataFrame is empty")
        return df_ts

    df_ts.to_csv(output_dir / "timeslice_stability.csv", index=False)
    logger.info("  Saved: timeslice_stability.csv")

    # ---- Plot ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    years_plot = df_ts['year'].values

    # 1. RMSE: XGBoost vs naive
    ax = axes[0, 0]
    ax.plot(years_plot, df_ts['rmse_xgb'],   'o-', color='#2196F3', label='XGBoost')
    ax.plot(years_plot, df_ts['rmse_naive'],  's--', color='#9E9E9E', alpha=0.7, label='Naive mean')
    ax.set_title('RMSE by Year', fontsize=11)
    ax.set_ylabel('RMSE')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(years_plot)

    # 2. R2
    ax = axes[0, 1]
    colors = ['#4CAF50' if v >= 0 else '#F44336' for v in df_ts['r2']]
    ax.bar(years_plot, df_ts['r2'], color=colors, alpha=0.8, width=0.6)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_title('R2 by Year', fontsize=11)
    ax.set_ylabel('R2')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_xticks(years_plot)

    # 3. Spearman rank correlation
    ax = axes[1, 0]
    ax.bar(years_plot, df_ts['spearman'], color='#FF9800', alpha=0.8, width=0.6)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_title('Spearman Rank Correlation by Year', fontsize=11)
    ax.set_ylabel('Spearman rho')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_xticks(years_plot)

    # 4. n and y_std
    ax = axes[1, 1]
    ax2 = ax.twinx()
    ax.bar(years_plot, df_ts['n'], color='#90CAF9', alpha=0.6, width=0.6, label='n')
    ax2.plot(years_plot, df_ts['y_std'], 'D--', color='#7B1FA2', alpha=0.8,
             label='Target std')
    ax.set_title('Sample Size and Target Std by Year', fontsize=11)
    ax.set_ylabel('n deals', color='#1565C0')
    ax2.set_ylabel('Target std (CFROA)', color='#7B1FA2')
    ax.set_xticks(years_plot)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)

    fig.suptitle(
        'Time-Slice Stability -- Test Set (2019-2022)', fontsize=13
    )
    plt.tight_layout()
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / "timeslice_stability.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info("  Saved: timeslice_stability.png")

    return df_ts


# =============================================================================
# SECTION 4 -- CHANNEL ABLATION
# =============================================================================

def _retrain_ablated(
    model_meta: Dict,
    ablated_features: List[str],
    splits: Dict,
) -> xgb.XGBRegressor:
    """
    Retrain XGBoost with best_params from full model but a reduced feature set.
    Uses early stopping on inner_val to select n_estimators.

    Holding hyperparameters constant isolates the feature contribution from
    any tuning advantage.
    """
    best_params = {**BASE_PARAMS, **model_meta['best_params']}
    n_est = best_params.pop('n_estimators', 1000)

    model_abl = xgb.XGBRegressor(
        n_estimators=n_est,
        early_stopping_rounds=CONFIG['early_stop'],
        **best_params,
    )
    model_abl.fit(
        splits['X_train_cv'][ablated_features],
        splits['y_train_cv'],
        eval_set=[
            (splits['X_train_cv'][ablated_features], splits['y_train_cv']),
            (splits['X_inner_val'][ablated_features], splits['y_inner_val']),
        ],
        verbose=False,
    )
    return model_abl


def run_channel_ablation(
    model: xgb.XGBRegressor,
    model_meta: Dict,
    splits: Dict,
    output_dir: Path,
    plot_dir: Path,
) -> pd.DataFrame:
    """
    For each synergy channel, retrain XGBoost without that channel's features.
    Compare val/test RMSE against the full model.

    Incremental channel value (test) = ablated_test_RMSE - full_test_RMSE.
    Positive = removing that channel hurts -- channel is informative.
    Negative = removing that channel helps -- channel may add noise.

    Returns DataFrame with one row per channel + one row for full model.
    """
    logger.info("=" * 60)
    logger.info("CHANNEL ABLATION")
    logger.info("=" * 60)

    features_used = splits['features_used']
    y_val  = splits['y_val']
    y_test = splits['y_test']

    # Full model metrics
    pred_val_full  = model.predict(splits['X_val'])
    pred_test_full = model.predict(splits['X_test'])
    full_val_rmse  = float(np.sqrt(mean_squared_error(y_val,  pred_val_full)))
    full_test_rmse = float(np.sqrt(mean_squared_error(y_test, pred_test_full)))
    full_val_r2    = float(r2_score(y_val,  pred_val_full))
    full_test_r2   = float(r2_score(y_test, pred_test_full))

    records = [{
        'ablated_channel':  'none (full model)',
        'n_features':       len(features_used),
        'val_rmse':         full_val_rmse,
        'test_rmse':        full_test_rmse,
        'val_r2':           full_val_r2,
        'test_r2':          full_test_r2,
        'delta_test_rmse':  0.0,
        'delta_test_r2':    0.0,
    }]

    logger.info(
        f"  Full model: val RMSE={full_val_rmse:.6f}  "
        f"test RMSE={full_test_rmse:.6f}"
    )

    for channel, ch_feats in CHANNEL_FEATURE_MAP.items():
        # Features to REMOVE: only those present in features_used
        to_remove = set(ch_feats) & set(features_used)
        if not to_remove:
            logger.info(f"  {channel:<14}: no features present -- skipped")
            continue

        ablated = [f for f in features_used if f not in to_remove]
        if len(ablated) == 0:
            logger.warning(f"  {channel}: ablation would remove ALL features -- skipped")
            continue

        logger.info(
            f"  {channel:<14}: removing {len(to_remove)} features  "
            f"({len(ablated)} remaining)..."
        )
        t0 = time.time()
        model_abl = _retrain_ablated(model_meta, ablated, splits)

        pred_val_abl  = model_abl.predict(splits['X_val'][ablated])
        pred_test_abl = model_abl.predict(splits['X_test'][ablated])

        abl_val_rmse  = float(np.sqrt(mean_squared_error(y_val,  pred_val_abl)))
        abl_test_rmse = float(np.sqrt(mean_squared_error(y_test, pred_test_abl)))
        abl_val_r2    = float(r2_score(y_val,  pred_val_abl))
        abl_test_r2   = float(r2_score(y_test, pred_test_abl))

        delta_test_rmse = abl_test_rmse - full_test_rmse
        delta_test_r2   = abl_test_r2   - full_test_r2

        records.append({
            'ablated_channel':  channel,
            'n_features':       len(ablated),
            'val_rmse':         abl_val_rmse,
            'test_rmse':        abl_test_rmse,
            'val_r2':           abl_val_r2,
            'test_r2':          abl_test_r2,
            'delta_test_rmse':  delta_test_rmse,
            'delta_test_r2':    delta_test_r2,
        })
        logger.info(
            f"    test RMSE={abl_test_rmse:.6f}  "
            f"delta={delta_test_rmse:+.6f}  "
            f"({'hurts' if delta_test_rmse > 0 else 'helps or neutral'})  "
            f"({time.time()-t0:.1f}s)"
        )

    df_abl = pd.DataFrame(records)
    df_abl.to_csv(output_dir / "channel_ablation.csv", index=False)
    logger.info("  Saved: channel_ablation.csv")

    # ---- Plot ----
    ch_rows = df_abl[df_abl['ablated_channel'] != 'none (full model)'].copy()
    if ch_rows.empty:
        logger.warning("  No ablation rows to plot")
        return df_abl

    ch_rows = ch_rows.sort_values('delta_test_rmse', ascending=False)
    channels  = ch_rows['ablated_channel'].values
    deltas    = ch_rows['delta_test_rmse'].values
    colors    = ['#F44336' if d > 0 else '#4CAF50' for d in deltas]

    CH_COLORS = {
        "cost": "#1976D2", "revenue": "#388E3C", "operational": "#F57C00",
        "financial": "#7B1FA2", "macro": "#00838F",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: delta RMSE (main result)
    ax = axes[0]
    bars = ax.barh(range(len(channels)), deltas,
                   color=[CH_COLORS.get(c, '#9E9E9E') for c in channels], alpha=0.85)
    ax.axvline(0, color='black', lw=1.0)
    ax.set_yticks(range(len(channels)))
    ax.set_yticklabels(channels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Delta Test RMSE (ablated - full; positive = channel is informative)')
    ax.set_title('Channel Ablation -- Test RMSE Impact', fontsize=12)
    ax.grid(True, axis='x', alpha=0.3)
    for bar, d in zip(bars, deltas):
        ax.text(d + (0.0001 if d >= 0 else -0.0001),
                bar.get_y() + bar.get_height() / 2,
                f'{d:+.5f}', va='center', fontsize=8,
                ha='left' if d >= 0 else 'right')

    # Right: absolute test RMSE for each ablated model + full model reference
    ax = axes[1]
    all_channels = list(ch_rows['ablated_channel'].values) + ['full model']
    all_rmse     = list(ch_rows['test_rmse'].values) + [full_test_rmse]
    bar_colors   = [CH_COLORS.get(c, '#9E9E9E') for c in channels] + ['#212121']
    bars2 = ax.barh(range(len(all_channels)), all_rmse, color=bar_colors, alpha=0.85)
    ax.axvline(full_test_rmse, color='black', lw=1.2, ls='--',
               label=f'Full model ({full_test_rmse:.5f})')
    ax.set_yticks(range(len(all_channels)))
    ax.set_yticklabels(all_channels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Test RMSE')
    ax.set_title('Ablated Model RMSE vs Full Model', fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, axis='x', alpha=0.3)

    fig.suptitle('Channel Ablation Study -- Synergy Channel Contribution', fontsize=13)
    plt.tight_layout()
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / "channel_ablation.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info("  Saved: channel_ablation.png")

    return df_abl


# =============================================================================
# MAIN
# =============================================================================

def main():
    logger.info("=" * 72)
    logger.info("  MODEL EVALUATION -- Advanced Diagnostics")
    logger.info("=" * 72)
    t_start = time.time()

    output_dir = CONFIG['output_dir']
    plot_dir   = CONFIG['plot_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Load model + prepared splits
    logger.info("\n[LOAD] Loading model and prepared splits...")
    model, model_meta, splits = load_artifacts()

    # 1. Frozen spec
    logger.info("\n[1/4] Frozen baseline spec...")
    write_frozen_spec(model_meta, splits, output_dir)

    # 2. Significance tests
    logger.info("\n[2/4] Significance tests...")
    sig_results = run_significance_tests(model, splits, output_dir)

    # 3. Time-slice stability
    logger.info("\n[3/4] Time-slice stability...")
    df_ts = run_timeslice_stability(model, splits, output_dir, plot_dir)

    # 4. Channel ablation
    logger.info("\n[4/4] Channel ablation (retraining 5 models)...")
    df_abl = run_channel_ablation(model, model_meta, splits, output_dir, plot_dir)

    elapsed = time.time() - t_start
    logger.info(f"\n{'=' * 72}")
    logger.info(f"  EVALUATION COMPLETE -- {elapsed:.1f}s")
    logger.info(f"  Outputs: {output_dir}")
    logger.info(f"{'=' * 72}")

    return sig_results, df_ts, df_abl


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    sig_results, df_ts, df_abl = main()
