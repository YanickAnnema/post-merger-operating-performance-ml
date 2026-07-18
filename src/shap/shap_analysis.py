"""
SHAP Analysis Pipeline — M&A Synergy Estimation
================================================

Loads the frozen XGBoost model and test-set splits produced by model_training.py,
computes exact SHAP values via TreeExplainer, and produces thesis-quality outputs.

No model retraining.  SHAP is purely an attribution / interpretability tool.
All computation is performed on the test set (2019–2022) only.

Pipeline:
  1. load_artifacts()           -- model pkl + splits pkl from ~/Desktop/outputs/
  2. validate_features()        -- exact match: X_test cols vs features_used list
  3. validate_test_years()      -- assert only 2019-2022 in df_test_meta
  4. compute_shap()             -- shap.TreeExplainer (exact, no approximation)
  5. plot_summary()             -- beeswarm plot (shap_summary.png)
  6. plot_waterfall_examples()  -- 3 individual deals (shap_waterfall_examples.png)
  7. plot_dependence_plots()    -- top-6 features (shap_dependence_plots.png)
  8. plot_channel_bar()         -- mean |SHAP| per channel (shap_channel_mean.png)
  9. save_data_files()          -- shap_values_test.csv, shap_channel_mean.csv,
                                   shap_base_value.txt
 10. write_summary_report()     -- shap_summary_report.txt

Inputs (from ~/Desktop/outputs/):
  xgboost_model_final.pkl
  training_splits.pkl

Outputs (to ~/Desktop/outputs/):
  shap_summary.png
  shap_waterfall_examples.png
  shap_dependence_plots.png
  shap_channel_mean.png
  shap_values_test.csv
  shap_channel_mean.csv
  shap_base_value.txt
  shap_summary_report.txt

Optimised for Spyder IDE (F5 execution).
"""

import numpy as np
import pandas as pd
import pickle
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure code/ is on sys.path for Spyder / repo-root execution
_code_dir = Path(__file__).resolve().parent
_ml_dir = _code_dir.parent / "ML pipeline"
for _path in (str(_code_dir), str(_ml_dir)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

import shap

# Channel/feature definitions imported from training module (single source of truth)
from model_training import (
    ALL_FEATURES,
    FEATURES_COST,
    FEATURES_REVENUE,
    FEATURES_OPERATIONAL,
    FEATURES_FINANCIAL,
    FEATURES_MACRO,
    CHANNEL_MAP,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ML pipeline"
SHAP_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "SHAP pipeline"

CONFIG = {
    'output_dir':  SHAP_OUTPUT_DIR,
    'plot_dir':    SHAP_OUTPUT_DIR,
    'model_pkl':   ML_OUTPUT_DIR / "xgboost_model_final.pkl",
    'splits_pkl':  ML_OUTPUT_DIR / "training_splits.pkl",

    # Test-set year constraint
    'test_years':  {2019, 2020, 2021, 2022},

    # Plot quality
    'dpi':        300,

    # Number of top features for dependence plots
    'n_dependence': 6,

    # Number of waterfall examples
    'n_waterfall':  3,
}

# Channel colours (consistent with existing pipeline)
CHANNEL_COLORS: Dict[str, str] = {
    'cost':        '#1976D2',
    'revenue':     '#388E3C',
    'operational': '#F57C00',
    'financial':   '#7B1FA2',
    'macro':       '#00838F',
}

# Feature-level colours derived from channel map
FEATURE_COLORS: Dict[str, str] = {f: CHANNEL_COLORS[ch] for f, ch in CHANNEL_MAP.items()}

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
# STEP 1 -- LOAD ARTIFACTS
# =============================================================================

def load_artifacts():
    """
    Load frozen model and prepared data splits.

    Returns:
        model       -- fitted xgb.XGBRegressor
        model_meta  -- dict with 'features', 'best_iteration', etc.
        splits      -- dict with X_test, y_test, df_test_meta, features_used, ...
    """
    model_pkl  = CONFIG['model_pkl']
    splits_pkl = CONFIG['splits_pkl']

    if not model_pkl.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {model_pkl}\n"
            "Run model_training.py first."
        )
    if not splits_pkl.exists():
        raise FileNotFoundError(
            f"Splits artifact not found: {splits_pkl}\n"
            "Run model_training.py first."
        )

    with open(model_pkl, 'rb') as f:
        model_meta = pickle.load(f)
    with open(splits_pkl, 'rb') as f:
        splits = pickle.load(f)

    model = model_meta['model']
    logger.info(
        f"Model loaded: {model.n_estimators} estimators, "
        f"best_iteration={model_meta.get('best_iteration', 'N/A')}"
    )
    logger.info(
        f"Splits loaded: "
        f"train_cv={len(splits['X_train_cv']):,}  "
        f"inner_val={len(splits.get('X_inner_val', [])):,}  "
        f"val={len(splits['X_val']):,}  "
        f"test={len(splits['X_test']):,}"
    )

    return model, model_meta, splits


# =============================================================================
# STEP 2 -- FEATURE VALIDATION
# =============================================================================

def validate_features(
    model_meta: Dict,
    splits: Dict,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Critical validation:
      - features_used from training_splits.pkl must exactly match
        features_used stored in xgboost_model_final.pkl (same names, same order).
      - X_test column list must match features_used exactly.

    Raises ValueError with a descriptive message on any mismatch.

    Returns: (X_test, y_test, features_used)
    """
    features_model  = model_meta.get('features', [])
    features_splits = splits.get('features_used', [])

    # --- cross-artifact consistency ---
    if features_model != features_splits:
        extra_model  = set(features_model)  - set(features_splits)
        extra_splits = set(features_splits) - set(features_model)
        raise ValueError(
            "Feature list mismatch between model pkl and splits pkl.\n"
            f"  In model only : {sorted(extra_model)}\n"
            f"  In splits only: {sorted(extra_splits)}\n"
            "Model and splits must come from the same training run."
        )

    features_used = features_model   # both identical

    X_test = splits['X_test']
    y_test = splits['y_test']

    # --- column names ---
    missing_in_xtest = [f for f in features_used if f not in X_test.columns]
    extra_in_xtest   = [c for c in X_test.columns  if c not in set(features_used)]

    if missing_in_xtest or extra_in_xtest:
        raise ValueError(
            "X_test columns do not match features_used.\n"
            f"  Missing from X_test : {missing_in_xtest}\n"
            f"  Extra in X_test     : {extra_in_xtest}"
        )

    # --- column order ---
    if list(X_test.columns) != list(features_used):
        logger.warning(
            "X_test column order differs from features_used — reordering X_test."
        )
        X_test = X_test[features_used]

    logger.info(
        f"Feature validation passed: {len(features_used)} features, "
        "names and order verified."
    )
    return X_test, y_test, features_used


# =============================================================================
# STEP 3 -- TEMPORAL VALIDATION
# =============================================================================

def validate_test_years(splits: Dict) -> pd.DataFrame:
    """
    Assert that df_test_meta contains ONLY deals with deal_year in {2019-2022}.

    Returns df_test_meta (aligned with X_test index).
    """
    df_test_meta = splits.get('df_test_meta')
    if df_test_meta is None:
        raise KeyError("'df_test_meta' key not found in training_splits.pkl.")

    if 'deal_year' not in df_test_meta.columns:
        logger.warning(
            "'deal_year' column not found in df_test_meta — "
            "skipping year range assertion."
        )
        return df_test_meta

    years = pd.to_numeric(df_test_meta['deal_year'], errors='coerce').dropna()
    unexpected = set(years.unique()) - CONFIG['test_years']

    if unexpected:
        raise ValueError(
            f"Test set contains deal_year values outside {{2019-2022}}: "
            f"{sorted(unexpected)}"
        )

    yr_min, yr_max = int(years.min()), int(years.max())
    logger.info(
        f"Temporal validation passed: test set covers {yr_min}–{yr_max} "
        f"({len(df_test_meta):,} observations)."
    )
    return df_test_meta


# =============================================================================
# STEP 4 -- COMPUTE SHAP VALUES
# =============================================================================

def compute_shap(
    model,
    X_test: pd.DataFrame,
) -> Tuple[shap.TreeExplainer, np.ndarray]:
    """
    Compute exact SHAP values for X_test using TreeExplainer.

    TreeExplainer uses the model's internal tree structure to compute exact
    Shapley values (O(TLD^2) per sample, where T = trees, L = leaves,
    D = max depth).  No kernel approximation; no sampling.

    Returns:
        explainer    -- shap.TreeExplainer instance (base_value accessible via
                        explainer.expected_value)
        shap_values  -- np.ndarray of shape (n_test, n_features)
    """
    logger.info("Initialising TreeExplainer...")
    explainer = shap.TreeExplainer(model)

    logger.info(f"Computing SHAP values for {len(X_test):,} test observations...")
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        shap_values = explainer.shap_values(X_test)

    # shap_values may be a list (for multi-output); unwrap if needed
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    base_val = float(explainer.expected_value)
    if isinstance(explainer.expected_value, (list, np.ndarray)):
        base_val = float(explainer.expected_value[0])

    logger.info(
        f"SHAP computation complete.  "
        f"Shape: {shap_values.shape}  "
        f"Base value: {base_val:.6f}"
    )
    return explainer, shap_values


# =============================================================================
# HELPERS
# =============================================================================

def _mean_abs_shap_by_feature(
    shap_values: np.ndarray,
    features: List[str],
) -> pd.Series:
    """Mean absolute SHAP value per feature, sorted descending."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    return pd.Series(mean_abs, index=features).sort_values(ascending=False)


def _channel_shap_summary(
    mean_abs: pd.Series,
) -> pd.DataFrame:
    """
    Aggregate mean absolute SHAP to channel level.

    Returns DataFrame with columns:
      channel | n_features | mean_abs_shap | pct_of_total
    """
    rows = []
    total = mean_abs.sum()
    for ch, feats in CHANNEL_FEATURE_MAP.items():
        ch_feats = [f for f in feats if f in mean_abs.index]
        ch_mean  = mean_abs[ch_feats].mean() if ch_feats else 0.0
        rows.append({
            'channel':       ch,
            'n_features':    len(ch_feats),
            'mean_abs_shap': ch_mean,
            'pct_of_total':  (ch_mean * len(ch_feats) / total * 100)
                             if total > 0 else 0.0,
        })
    df = pd.DataFrame(rows)
    # Normalise pct_of_total to sum to 100 (avoids rounding issues)
    if df['pct_of_total'].sum() > 0:
        df['pct_of_total'] = df['pct_of_total'] / df['pct_of_total'].sum() * 100
    return df.sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)


def _get_waterfall_indices(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    df_test_meta: pd.DataFrame,
) -> Tuple[int, int, int,
           float, float, float,
           float, float, float,
           str, str, str]:
    """
    Identify test-set row indices for:
      - minimum predicted synergy
      - median predicted synergy
      - maximum predicted synergy

    Also retrieves y_true for each example so callers can compute residuals.

    Returns positional indices into X_test (0-based row positions),
    predicted values, actual values, and deal IDs.
    """
    preds   = model.predict(X_test)
    y_arr   = np.array(y_test)

    idx_min = int(np.argmin(preds))
    idx_max = int(np.argmax(preds))

    median_val = float(np.median(preds))
    idx_med    = int(np.argmin(np.abs(preds - median_val)))

    val_min = float(preds[idx_min])
    val_med = float(preds[idx_med])
    val_max = float(preds[idx_max])

    true_min = float(y_arr[idx_min])
    true_med = float(y_arr[idx_med])
    true_max = float(y_arr[idx_max])

    def _get_deal_id(pos: int) -> str:
        if df_test_meta is not None and 'deal_id' in df_test_meta.columns:
            try:
                return str(df_test_meta.iloc[pos]['deal_id'])
            except Exception:
                pass
        return f"obs_{pos}"

    id_min = _get_deal_id(idx_min)
    id_med = _get_deal_id(idx_med)
    id_max = _get_deal_id(idx_max)

    return (idx_min, idx_med, idx_max,
            val_min, val_med, val_max,
            true_min, true_med, true_max,
            id_min, id_med, id_max)


def _apply_thesis_style(fig, ax_or_axes=None):
    """Apply consistent thesis styling to a figure."""
    fig.patch.set_facecolor('white')
    axes = []
    if ax_or_axes is not None:
        if hasattr(ax_or_axes, '__iter__'):
            axes = list(ax_or_axes)
        else:
            axes = [ax_or_axes]
    for ax in axes:
        ax.set_facecolor('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#444444')
        ax.spines['bottom'].set_color('#444444')
        ax.tick_params(colors='#333333', labelsize=9)
        ax.xaxis.label.set_color('#333333')
        ax.yaxis.label.set_color('#333333')
        ax.title.set_color('#111111')


# =============================================================================
# STEP 5 -- BEESWARM SUMMARY PLOT
# =============================================================================

def plot_summary(
    shap_values: np.ndarray,
    X_test: pd.DataFrame,
    features_used: List[str],
    output_dir: Path,
) -> None:
    """
    SHAP beeswarm summary plot — all 29 features, test-set observations.

    Features sorted by mean |SHAP| descending.  Each point is one deal;
    colour encodes relative feature value (blue=low, red=high).
    """
    logger.info("Plotting: shap_summary.png")

    fig, ax = plt.subplots(figsize=(10, 9))

    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=features_used,
        show=False,
        plot_type='dot',
        max_display=len(features_used),
        plot_size=None,
        color_bar=True,
    )

    ax = plt.gca()
    ax.set_title(
        "SHAP Feature Attribution — Test Set (2019–2022)",
        fontsize=13, fontweight='bold', pad=12
    )
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    out_path = output_dir / "shap_summary.png"
    plt.savefig(out_path, dpi=CONFIG['dpi'], bbox_inches='tight',
                facecolor='white')
    plt.close()
    logger.info(f"  Saved: {out_path.name}")


# =============================================================================
# STEP 6 -- WATERFALL EXAMPLE PLOTS
# =============================================================================

def plot_waterfall_examples(
    explainer,
    shap_values: np.ndarray,
    X_test: pd.DataFrame,
    features_used: List[str],
    idx_min: int, idx_med: int, idx_max: int,
    val_min: float, val_med: float, val_max: float,
    true_min: float, true_med: float, true_max: float,
    id_min: str, id_med: str, id_max: str,
    output_dir: Path,
) -> None:
    """
    Three-panel waterfall plot:
      Left  — deal with lowest predicted synergy
      Centre — deal closest to median predicted synergy
      Right — deal with highest predicted synergy

    Uses shap.waterfall_plot (legacy) for compatibility across SHAP versions.
    Falls back to manual bar rendering if API is unavailable.
    """
    logger.info("Plotting: shap_waterfall_examples.png")

    base_val = float(explainer.expected_value)
    if isinstance(explainer.expected_value, (list, np.ndarray)):
        base_val = float(explainer.expected_value[0])

    cases = [
        (idx_min, val_min, true_min, id_min, "Lowest Predicted Synergy"),
        (idx_med, val_med, true_med, id_med, "Median Predicted Synergy"),
        (idx_max, val_max, true_max, id_max, "Highest Predicted Synergy"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.patch.set_facecolor('white')

    for ax, (idx, pred_val, true_val, deal_id, subtitle) in zip(axes, cases):
        sv   = shap_values[idx]          # (n_features,)
        fv   = X_test.iloc[idx].values   # feature values for this observation

        # Sort by absolute SHAP, show top 15
        order     = np.argsort(np.abs(sv))[::-1][:15]
        sv_top    = sv[order]
        fv_top    = fv[order]
        feat_top  = [features_used[i] for i in order]

        # Waterfall cumulative sum from base_val
        cumsum = np.concatenate([[base_val], base_val + np.cumsum(sv_top)])

        bar_colors = [
            '#D32F2F' if v >= 0 else '#1565C0'
            for v in sv_top
        ]

        y_positions = np.arange(len(sv_top))
        bar_widths  = sv_top

        # Draw horizontal bars (waterfall style)
        for i, (pos, w, c, left) in enumerate(
            zip(y_positions, bar_widths, bar_colors, cumsum[:-1])
        ):
            ax.barh(pos, w, left=left, color=c, height=0.55,
                    alpha=0.85, edgecolor='white', linewidth=0.5)

        # Feature labels with value annotation
        ax.set_yticks(y_positions)
        ax.set_yticklabels(
            [f"{fn} = {fv_top[i]:.3g}" for i, fn in enumerate(feat_top)],
            fontsize=7.5
        )
        ax.invert_yaxis()

        residual = pred_val - true_val

        ax.axvline(base_val, color='#555555', linewidth=1.0,
                   linestyle='--', label=f'Base={base_val:.4f}')
        ax.axvline(pred_val, color='#222222', linewidth=1.2,
                   linestyle='-', label=f'y_hat={pred_val:.4f}')
        ax.axvline(true_val, color='#2E7D32', linewidth=1.2,
                   linestyle=':', label=f'y_true={true_val:.4f}')

        ax.set_xlabel("Model output (CFROA synergy)", fontsize=8)
        sign = '+' if residual >= 0 else ''
        ax.set_title(
            f"{subtitle}\n"
            f"Deal ID: {deal_id}\n"
            f"y_hat={pred_val:.4f}  y_true={true_val:.4f}  resid={sign}{residual:.4f}",
            fontsize=8, fontweight='bold'
        )
        ax.legend(fontsize=7, loc='lower right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='x', labelsize=8)

    # Legend for bar colours
    legend_patches = [
        Patch(facecolor='#D32F2F', label='Positive contribution'),
        Patch(facecolor='#1565C0', label='Negative contribution'),
    ]
    fig.legend(
        handles=legend_patches, loc='lower center',
        ncol=2, fontsize=9, frameon=False,
        bbox_to_anchor=(0.5, -0.01)
    )

    fig.suptitle(
        "Individual Deal SHAP Explanations — Waterfall Analysis",
        fontsize=13, fontweight='bold', y=1.01
    )
    plt.tight_layout()

    out_path = output_dir / "shap_waterfall_examples.png"
    plt.savefig(out_path, dpi=CONFIG['dpi'], bbox_inches='tight',
                facecolor='white')
    plt.close()
    logger.info(f"  Saved: {out_path.name}")


# =============================================================================
# STEP 7 -- DEPENDENCE PLOTS
# =============================================================================

def plot_dependence_plots(
    shap_values: np.ndarray,
    X_test: pd.DataFrame,
    features_used: List[str],
    mean_abs: pd.Series,
    output_dir: Path,
) -> None:
    """
    SHAP dependence plots for the top-6 features by mean |SHAP|.

    Each subplot: SHAP value (y) vs. feature value (x).
    Colour encodes a candidate interaction variable chosen as a visualisation
    heuristic (highest absolute Pearson correlation between that feature and the
    focal feature's SHAP values).  This is NOT a formal SHAP interaction test;
    it selects a plausible co-variate to colour the scatter for visual inspection.

    Special override: for log_deal_value, colour by deal_cross_border
    (theory-motivated co-variate to highlight deal-size × border premium).
    """
    logger.info("Plotting: shap_dependence_plots.png")

    top6 = list(mean_abs.index[:CONFIG['n_dependence']])

    # Theory override for interaction colouring
    INTERACTION_OVERRIDE = {
        'log_deal_value': 'deal_cross_border',
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes_flat = axes.flatten()
    fig.patch.set_facecolor('white')

    for i, (feat, ax) in enumerate(zip(top6, axes_flat)):
        feat_idx  = features_used.index(feat)
        channel   = CHANNEL_MAP.get(feat, 'cost')
        base_color = CHANNEL_COLORS[channel]

        # Determine interaction feature
        interact_feat = INTERACTION_OVERRIDE.get(feat, 'auto')

        x_vals = X_test[feat].values
        y_vals = shap_values[:, feat_idx]

        if interact_feat == 'auto':
            # Auto-detect: feature with highest absolute Pearson correlation
            # with residual SHAP after removing main effect
            corrs = []
            for j, other in enumerate(features_used):
                if j == feat_idx:
                    corrs.append(0.0)
                    continue
                other_vals = X_test[other].values
                mask = ~(np.isnan(other_vals) | np.isnan(y_vals))
                if mask.sum() < 5:
                    corrs.append(0.0)
                    continue
                c = float(np.corrcoef(
                    other_vals[mask], y_vals[mask]
                )[0, 1])
                corrs.append(abs(c) if np.isfinite(c) else 0.0)
            best_j    = int(np.argmax(corrs))
            interact_feat = features_used[best_j]

        interact_idx  = features_used.index(interact_feat)
        interact_vals = X_test[interact_feat].values
        interact_ch   = CHANNEL_MAP.get(interact_feat, 'cost')

        # Scatter with interaction colour
        valid = ~(np.isnan(x_vals) | np.isnan(interact_vals) | np.isnan(y_vals))
        sc = ax.scatter(
            x_vals[valid], y_vals[valid],
            c=interact_vals[valid],
            cmap='coolwarm', alpha=0.55, s=14,
            edgecolors='none'
        )

        # Trend line (LOWESS-style via poly fit)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            try:
                order_x = np.argsort(x_vals[valid])
                z = np.polyfit(x_vals[valid][order_x], y_vals[valid][order_x], 1)
                p = np.poly1d(z)
                x_line = np.linspace(x_vals[valid].min(), x_vals[valid].max(), 100)
                ax.plot(x_line, p(x_line), color='#444444',
                        linewidth=1.2, linestyle='--', alpha=0.7)
            except Exception:
                pass

        ax.axhline(0, color='#888888', linewidth=0.8, linestyle=':')
        ax.set_xlabel(feat, fontsize=8)
        ax.set_ylabel(f"SHAP({feat})", fontsize=8)
        ax.set_title(
            f"{feat}\n(colour: {interact_feat})",
            fontsize=8, fontweight='bold'
        )
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=7)

        # Minimal colourbar
        cbar = plt.colorbar(sc, ax=ax, pad=0.02, shrink=0.75)
        cbar.ax.tick_params(labelsize=6)
        cbar.set_label(interact_feat, fontsize=6)

    # Channel legend
    legend_patches = [
        Patch(facecolor=CHANNEL_COLORS[ch], label=ch.capitalize())
        for ch in CHANNEL_COLORS
    ]
    fig.legend(
        handles=legend_patches, loc='lower center', ncol=5,
        fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02)
    )
    fig.suptitle(
        "SHAP Dependence Plots — Top 6 Features by Mean |SHAP|  (Test Set 2019–2022)",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()

    out_path = output_dir / "shap_dependence_plots.png"
    plt.savefig(out_path, dpi=CONFIG['dpi'], bbox_inches='tight',
                facecolor='white')
    plt.close()
    logger.info(f"  Saved: {out_path.name}")


# =============================================================================
# STEP 8 -- CHANNEL BAR CHART
# =============================================================================

def plot_channel_bar(
    channel_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Horizontal bar chart: mean absolute SHAP per synergy channel.
    Bars labelled with exact values.  Ordered by mean_abs_shap descending.
    """
    logger.info("Plotting: shap_channel_mean.png")

    df = channel_df.sort_values('mean_abs_shap', ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('white')

    colors = [CHANNEL_COLORS.get(ch, '#999999') for ch in df['channel']]
    bars = ax.barh(
        df['channel'].str.capitalize(), df['mean_abs_shap'],
        color=colors, height=0.55, alpha=0.88, edgecolor='white'
    )

    # Value labels
    for bar, val, pct in zip(bars, df['mean_abs_shap'], df['pct_of_total']):
        ax.text(
            bar.get_width() + bar.get_width() * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.5f}  ({pct:.1f}%)",
            va='center', ha='left', fontsize=9, color='#333333'
        )

    ax.set_xlabel("Mean absolute SHAP value", fontsize=10)
    ax.set_title(
        "Channel-Level Feature Attribution — Mean |SHAP|\n"
        "Test Set (2019–2022)",
        fontsize=11, fontweight='bold', pad=10
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=9)

    x_max = df['mean_abs_shap'].max()
    ax.set_xlim(0, x_max * 1.35)

    plt.tight_layout()

    out_path = output_dir / "shap_channel_mean.png"
    plt.savefig(out_path, dpi=CONFIG['dpi'], bbox_inches='tight',
                facecolor='white')
    plt.close()
    logger.info(f"  Saved: {out_path.name}")


# =============================================================================
# STEP 9 -- SAVE DATA FILES
# =============================================================================

def save_data_files(
    shap_values: np.ndarray,
    X_test: pd.DataFrame,
    features_used: List[str],
    mean_abs: pd.Series,
    channel_df: pd.DataFrame,
    explainer,
    output_dir: Path,
) -> None:
    """
    Save:
      shap_values_test.csv  -- raw SHAP matrix (observations × features)
      shap_channel_mean.csv -- channel-level summary
      shap_base_value.txt   -- single line: model base value
    """
    # SHAP value matrix
    shap_df = pd.DataFrame(
        shap_values, columns=features_used, index=X_test.index
    )
    shap_path = output_dir / "shap_values_test.csv"
    shap_df.to_csv(shap_path, index=True)
    logger.info(f"  Saved: {shap_path.name}  shape={shap_df.shape}")

    # Channel summary
    ch_path = output_dir / "shap_channel_mean.csv"
    channel_df.to_csv(ch_path, index=False)
    logger.info(f"  Saved: {ch_path.name}")

    # Base value
    base_val = float(explainer.expected_value)
    if isinstance(explainer.expected_value, (list, np.ndarray)):
        base_val = float(explainer.expected_value[0])
    bv_path = output_dir / "shap_base_value.txt"
    bv_path.write_text(f"{base_val:.10f}\n", encoding='ascii')
    logger.info(f"  Saved: {bv_path.name}  value={base_val:.10f}")


# =============================================================================
# STEP 10 -- SUMMARY REPORT
# =============================================================================

def write_summary_report(
    shap_values: np.ndarray,
    X_test: pd.DataFrame,
    features_used: List[str],
    mean_abs: pd.Series,
    channel_df: pd.DataFrame,
    explainer,
    df_test_meta: pd.DataFrame,
    idx_min: int, idx_med: int, idx_max: int,
    val_min: float, val_med: float, val_max: float,
    true_min: float, true_med: float, true_max: float,
    id_min: str, id_med: str, id_max: str,
    model_meta: Dict,
    output_dir: Path,
) -> None:
    """
    Write shap_summary_report.txt — complete audit trail for thesis.
    """
    base_val = float(explainer.expected_value)
    if isinstance(explainer.expected_value, (list, np.ndarray)):
        base_val = float(explainer.expected_value[0])

    # Date range from df_test_meta
    date_range_str = "N/A"
    if df_test_meta is not None and 'DateEffective' in df_test_meta.columns:
        dates = pd.to_datetime(df_test_meta['DateEffective'], errors='coerce').dropna()
        if len(dates):
            date_range_str = (
                f"{dates.min().date()} to {dates.max().date()}"
            )

    # Year range (ASCII only)
    year_range_str = "N/A"
    if df_test_meta is not None and 'deal_year' in df_test_meta.columns:
        years = pd.to_numeric(df_test_meta['deal_year'], errors='coerce').dropna()
        if len(years):
            year_range_str = f"{int(years.min())}-{int(years.max())}"

    # Top 10 features — distributional SHAP summary (no single global direction)
    # XGBoost effects are often non-monotonic; a single linear sign can mislead.
    # Instead report: mean SHAP (signed) and % of observations with positive SHAP.
    top10       = mean_abs.head(10)
    top10_stats = []
    for feat in top10.index:
        feat_idx = features_used.index(feat)
        sv_col   = shap_values[:, feat_idx]
        valid    = sv_col[~np.isnan(sv_col)]
        mean_sv  = float(np.mean(valid)) if len(valid) else float('nan')
        pct_pos  = float(np.mean(valid > 0) * 100) if len(valid) else float('nan')
        top10_stats.append((feat, float(top10[feat]), mean_sv, pct_pos))

    # SHAP version
    shap_ver = getattr(shap, '__version__', 'unknown')

    lines = [
        "=" * 72,
        "  SHAP ANALYSIS REPORT -- M&A Synergy Estimation",
        "=" * 72,
        "",
        f"Generated     : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"SHAP version  : {shap_ver}",
        f"SHAP explainer: shap.TreeExplainer (exact, no approximation)",
        f"Model type    : XGBoost XGBRegressor",
        f"Best iteration: {model_meta.get('best_iteration', 'N/A')}",
        f"Features used : {len(features_used)}",
        f"Target        : synergy_healy1992_w (CFROA-based, winsorised)",
        "",
        "MODEL BASE VALUE",
        "-" * 40,
        f"  base_value (E[f(x)]) = {base_val:.10f}",
        "  This is the model baseline value used by TreeExplainer (the value",
        "  all SHAP attributions sum to when no explicit background dataset is",
        "  provided; in practice equals the mean prediction over the training set).",
        "",
        "TEST SET SUMMARY",
        "-" * 40,
        f"  Observations : {len(X_test):,}",
        f"  Year range   : {year_range_str}",
        f"  Date range   : {date_range_str}",
        "",
        "TOP 10 FEATURES BY MEAN |SHAP|",
        "-" * 40,
    ]

    # Header for the top-10 table
    lines.append(
        f"  {'Rank':<4s}  {'Feature':<45s}  {'mean|SHAP|':>10s}  "
        f"{'mean_SHAP':>10s}  {'pct_pos':>7s}  [channel]"
    )
    lines.append("  " + "-" * 86)
    for rank, (feat, abs_val, mean_sv, pct_pos) in enumerate(top10_stats, 1):
        channel = CHANNEL_MAP.get(feat, '?')
        lines.append(
            f"  {rank:<4d}  {feat:<45s}  {abs_val:>10.6f}  "
            f"{mean_sv:>+10.6f}  {pct_pos:>6.1f}%  [{channel}]"
        )
    lines.append("")
    lines.append(
        "  Note: mean_SHAP is the signed average SHAP value (can be near zero if"
    )
    lines.append(
        "  effects are split across observations).  pct_pos = share of test deals"
    )
    lines.append(
        "  where this feature pushed the prediction above baseline.  XGBoost"
    )
    lines.append(
        "  effects are non-monotonic; neither column implies a universal direction."
    )

    lines += [
        "",
        "CHANNEL-LEVEL MEAN ABSOLUTE SHAP",
        "-" * 40,
    ]
    for _, row in channel_df.iterrows():
        lines.append(
            f"  {row['channel']:<15s}  n_features={row['n_features']:2d}  "
            f"mean_abs_shap={row['mean_abs_shap']:.6f}  "
            f"pct_of_total={row['pct_of_total']:5.1f}%"
        )

    lines += [
        "",
        "  These two columns measure different concepts:",
        "  - mean_abs_shap : average per-feature attribution within the channel.",
        "    Use this to compare how informative a typical feature is across",
        "    channels, independent of channel size.",
        "  - pct_of_total  : share of total absolute SHAP attribution from the",
        "    channel on the test set (weighted by n_features).  Use this as the",
        "    primary 'channel contribution' metric in thesis text -- it reflects",
        "    each channel's share of the model's total attribution mass, not",
        "    explained variance in the statistical sense.",
        "  Channels with many features (e.g. operational) may rank higher on",
        "  pct_of_total than mean_abs_shap relative to channels with few features",
        "  (e.g. macro).  Acknowledge both in discussion if rankings diverge.",
        "",
        "WATERFALL EXAMPLE DEALS",
        "-" * 40,
        f"  {'Case':<8s}  {'deal_id':<20s}  {'y_hat':>10s}  "
        f"{'y_true':>10s}  {'residual':>10s}  [row]",
        "  " + "-" * 66,
        f"  {'Lowest':<8s}  {id_min:<20s}  {val_min:>+10.6f}  "
        f"{true_min:>+10.6f}  {val_min - true_min:>+10.6f}  [{idx_min}]",
        f"  {'Median':<8s}  {id_med:<20s}  {val_med:>+10.6f}  "
        f"{true_med:>+10.6f}  {val_med - true_med:>+10.6f}  [{idx_med}]",
        f"  {'Highest':<8s}  {id_max:<20s}  {val_max:>+10.6f}  "
        f"{true_max:>+10.6f}  {val_max - true_max:>+10.6f}  [{idx_max}]",
        "",
        "  Note: residual = y_hat - y_true.  Positive residual = model",
        "  over-predicted; negative = model under-predicted.  Use these",
        "  cases in thesis narrative to show whether SHAP attribution",
        "  aligns directionally with realised synergy outcomes.",
        "",
        "OUTPUT FILES",
        "-" * 40,
        "  shap_summary.png           -- beeswarm plot, all 29 features",
        "  shap_waterfall_examples.png -- 3 individual deal explanations",
        "  shap_dependence_plots.png  -- top 6 features, interaction-coloured",
        "  shap_channel_mean.png      -- channel attribution bar chart",
        "  shap_values_test.csv       -- raw SHAP matrix (n_test x n_features)",
        "  shap_channel_mean.csv      -- channel-level summary table",
        "  shap_base_value.txt        -- base value (scalar)",
        "",
        "NOTES / ASSUMPTIONS",
        "-" * 40,
        "  - SHAP computed on test set only (2019-2022); no training data used.",
        "  - TreeExplainer uses the model's internal tree structure for exact",
        "    Shapley values.  No sampling or kernel approximation.",
        "  - Dependence plot colour variable is a visualisation heuristic: the",
        "    feature with the highest absolute Pearson correlation to the focal",
        "    feature's SHAP values is chosen as a candidate co-variate.  This is",
        "    NOT a formal SHAP interaction test and should not be described as one.",
        "    Exception: log_deal_value uses deal_cross_border as a theory-motivated",
        "    co-variate (deal size x cross-border premium).",
        "  - Channel pct_of_total computed as (channel mean x n_features) /",
        "    sum(all feature mean |SHAP|), normalised to 100%.",
        "  - SHAP values are model attributions, not causal estimates.",
        "    Thesis language must reflect this (e.g. 'the model attributes more",
        "    predictive weight to ...' not 'X causes synergy').",
        "  - Waterfall plots show top 15 features by |SHAP| per deal for",
        "    readability; all 29 features contribute to the SHAP decomposition.",
        "=" * 72,
    ]

    report_path = output_dir / "shap_summary_report.txt"
    # ASCII encoding: surfaces any remaining non-ASCII before the file is written,
    # preventing mojibake in thesis-facing outputs.
    report_path.write_text("\n".join(lines) + "\n", encoding='ascii')
    logger.info(f"  Saved: {report_path.name}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    logger.info("=" * 72)
    logger.info("  SHAP ANALYSIS PIPELINE — M&A Synergy Estimation")
    logger.info("=" * 72)

    output_dir = CONFIG['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    CONFIG['plot_dir'].mkdir(parents=True, exist_ok=True)

    # Step 1: Load artifacts
    logger.info("\n[STEP 1] Loading model and split artifacts...")
    model, model_meta, splits = load_artifacts()

    # Step 2: Feature validation
    logger.info("\n[STEP 2] Validating feature consistency...")
    X_test, y_test, features_used = validate_features(model_meta, splits)

    # Step 3: Temporal validation
    logger.info("\n[STEP 3] Validating test-set year range...")
    df_test_meta = validate_test_years(splits)

    # Step 4: SHAP computation
    logger.info("\n[STEP 4] Computing SHAP values (TreeExplainer, exact)...")
    explainer, shap_values = compute_shap(model, X_test)

    # Derived summaries (used by multiple steps)
    mean_abs   = _mean_abs_shap_by_feature(shap_values, features_used)
    channel_df = _channel_shap_summary(mean_abs)

    # Waterfall example indices
    (idx_min, idx_med, idx_max,
     val_min, val_med, val_max,
     true_min, true_med, true_max,
     id_min,  id_med,  id_max) = _get_waterfall_indices(
         model, X_test, y_test, df_test_meta
     )

    logger.info(
        f"  Waterfall examples -- "
        f"min: deal_id={id_min} (y_hat={val_min:.4f}, y_true={true_min:.4f}), "
        f"med: deal_id={id_med} (y_hat={val_med:.4f}, y_true={true_med:.4f}), "
        f"max: deal_id={id_max} (y_hat={val_max:.4f}, y_true={true_max:.4f})"
    )

    # Step 5: Summary beeswarm plot
    logger.info("\n[STEP 5] Beeswarm summary plot...")
    plot_summary(shap_values, X_test, features_used, output_dir)

    # Step 6: Waterfall examples
    logger.info("\n[STEP 6] Waterfall example plots...")
    plot_waterfall_examples(
        explainer, shap_values, X_test, features_used,
        idx_min, idx_med, idx_max,
        val_min, val_med, val_max,
        true_min, true_med, true_max,
        id_min,  id_med,  id_max,
        output_dir,
    )

    # Step 7: Dependence plots
    logger.info("\n[STEP 7] Dependence plots (top 6 features)...")
    plot_dependence_plots(
        shap_values, X_test, features_used, mean_abs, output_dir
    )

    # Step 8: Channel bar chart
    logger.info("\n[STEP 8] Channel attribution bar chart...")
    plot_channel_bar(channel_df, output_dir)

    # Step 9: Save data files
    logger.info("\n[STEP 9] Saving data files...")
    save_data_files(
        shap_values, X_test, features_used, mean_abs,
        channel_df, explainer, output_dir
    )

    # Step 10: Summary report
    logger.info("\n[STEP 10] Writing summary report...")
    write_summary_report(
        shap_values, X_test, features_used, mean_abs,
        channel_df, explainer, df_test_meta,
        idx_min, idx_med, idx_max,
        val_min, val_med, val_max,
        true_min, true_med, true_max,
        id_min,  id_med,  id_max,
        model_meta, output_dir,
    )

    logger.info("\n" + "=" * 72)
    logger.info("  SHAP ANALYSIS COMPLETE")
    logger.info("=" * 72)
    logger.info(f"  All outputs saved to: {output_dir}")
    logger.info("  Files produced:")
    for fname in [
        "shap_summary.png", "shap_waterfall_examples.png",
        "shap_dependence_plots.png", "shap_channel_mean.png",
        "shap_values_test.csv", "shap_channel_mean.csv",
        "shap_base_value.txt", "shap_summary_report.txt",
    ]:
        path = output_dir / fname
        status = "OK" if path.exists() else "MISSING"
        logger.info(f"    [{status}] {fname}")


if __name__ == "__main__":
    main()
