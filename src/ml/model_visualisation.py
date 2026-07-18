"""
Model Visualisation — M&A Synergy Estimation (ML Phase)
========================================================

Standalone script producing thesis-quality visualisations of ML pipeline
outputs.  Reads from artifacts written by model_training.py and model_evaluation.py;
no LSEG connection or re-training required.

Plots produced (all 300 DPI PNG):
  1.  sample_structure.png         — n per split + deals over time
  2.  model_comparison.png         — RMSE / MAE / R² across baselines + XGBoost
  3.  error_distribution_overlay.png — KDE of residuals, val vs test
  4.  correlation_heatmap.png      — Pearson correlation among all 29 features
  5.  feature_distributions.png    — Boxplots of continuous features by split
  6.  missingness_overview.png     — NaN% per feature at ML stage
  7.  channel_importance_summary.png — Channel-level gain share (bar + labels)

Run:  F5 in Spyder, or  python code/model_visualisation.py

Requires: training_splits.pkl, xgboost_model_final.pkl (from model_training.py)
          feature_importance.csv, channel_importance.csv (from model_training.py)
          ml_ready.csv or ml_ready_nowinsor.csv (from data_preparation.py)
"""

import sys
import pickle
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Resolve imports from code/ directory when run as standalone
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Suppress deprecation warnings from dependencies
warnings.filterwarnings('ignore')
matplotlib.use('Agg')    # headless rendering; remove if running interactively

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DAQ_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "DAQ pipeline"
ML_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ML pipeline"

CONFIG = {
    # Input artifacts (produced by model_training.py)
    'splits_pkl':    ML_OUTPUT_DIR / "training_splits.pkl",
    'model_pkl':     ML_OUTPUT_DIR / "xgboost_model_final.pkl",
    'imp_csv':       ML_OUTPUT_DIR / "feature_importance.csv",
    'channel_csv':   ML_OUTPUT_DIR / "channel_importance.csv",

    # Full dataset (pre-filtering) for coverage / distribution plots
    'ml_ready_nowinsor': DAQ_OUTPUT_DIR / "ml_ready_nowinsor.csv",
    'ml_ready_fallback': DAQ_OUTPUT_DIR / "ml_ready.csv",

    # Output directory (separate subfolder to avoid cluttering model outputs)
    'output_dir':  ML_OUTPUT_DIR / "thesis_vis",

    # Style
    'dpi':         300,
    'style':       'seaborn-v0_8-whitegrid',

    # Bootstrap resamples for comparison CIs
    'n_bootstrap': 1000,
    'seed':        42,

    # Ridge baseline
    'ridge_alpha': 1.0,

    # Missingness threshold below which a feature is flagged
    'missingness_warn_pct': 20.0,
}

# Canonical channel colors — match model_training.py
CH_COLORS = {
    "cost":        "#1976D2",
    "revenue":     "#388E3C",
    "operational": "#F57C00",
    "financial":   "#7B1FA2",
    "macro":       "#00838F",
    "other":       "#9E9E9E",
}

SPLIT_COLORS = {
    "train_cv":  "#90CAF9",
    "inner_val": "#FFE082",
    "val":       "#FF9800",
    "test":      "#4CAF50",
}

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# HELPERS
# =============================================================================

def _ensure_output_dir() -> Path:
    d = CONFIG['output_dir']
    d.mkdir(parents=True, exist_ok=True)
    return d


def _savefig(fig: plt.Figure, name: str, output_dir: Path) -> None:
    path = output_dir / name
    fig.savefig(path, dpi=CONFIG['dpi'], bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved: {name}")


def _load_artifacts():
    """
    Load all pkl/csv artifacts.  Returns (model, model_meta, splits, imp_df, ch_df, df_full).
    df_full is the full labeled dataset at ML stage (for distribution / coverage plots).
    """
    logger.info("Loading artifacts ...")

    # model_meta + model
    with open(CONFIG['model_pkl'], 'rb') as f:
        model_meta = pickle.load(f)
    model = model_meta['model']
    logger.info(f"  Model loaded. best_iteration={model_meta['best_iteration']}")

    # splits
    with open(CONFIG['splits_pkl'], 'rb') as f:
        splits = pickle.load(f)
    logger.info(
        f"  Splits loaded.  train_cv={len(splits['y_train_cv']):,}  "
        f"inner_val={len(splits['y_inner_val']):,}  "
        f"val={len(splits['y_val']):,}  test={len(splits['y_test']):,}"
    )

    # feature importance tables
    imp_df = pd.read_csv(CONFIG['imp_csv'])
    ch_df  = pd.read_csv(CONFIG['channel_csv'])

    # full dataset for coverage / distribution views
    ml_nowinsor = CONFIG['ml_ready_nowinsor']
    ml_fallback  = CONFIG['ml_ready_fallback']
    if ml_nowinsor.exists():
        df_full = pd.read_csv(ml_nowinsor, low_memory=False)
        logger.info(f"  Full dataset loaded from {ml_nowinsor.name}  n={len(df_full):,}")
    elif ml_fallback.exists():
        df_full = pd.read_csv(ml_fallback, low_memory=False)
        logger.warning(f"  ml_ready_nowinsor not found; falling back to {ml_fallback.name}")
    else:
        df_full = None
        logger.warning("  No full dataset CSV found — coverage plots will be skipped")

    return model, model_meta, splits, imp_df, ch_df, df_full


def _bootstrap_rmse_ci(y_true, y_pred, n=1000, seed=42):
    """Bootstrap 95% CI on RMSE."""
    rng = np.random.default_rng(seed)
    rmses = []
    idx = np.arange(len(y_true))
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        rmses.append(np.sqrt(mean_squared_error(y_true[s], y_pred[s])))
    return float(np.percentile(rmses, 2.5)), float(np.percentile(rmses, 97.5))


def _recompute_baselines(splits):
    """
    Recompute naive and Ridge baselines from splits for comparison plot.
    Fast: no cross-validation, just fit on train_cv and predict val/test.
    """
    X_tr = splits['X_train_cv']
    y_tr = splits['y_train_cv']
    X_va = splits['X_val']
    y_va = splits['y_val']
    X_te = splits['X_test']
    y_te = splits['y_test']

    train_mean   = float(y_tr.mean())
    train_median = float(y_tr.median())

    results = {}

    # Naive mean
    for name in ['val', 'test']:
        y = y_va if name == 'val' else y_te
        X = X_va if name == 'val' else X_te
        pred = np.full(len(y), train_mean)
        results.setdefault('naive_mean', {})[name] = {
            'rmse': float(np.sqrt(mean_squared_error(y, pred))),
            'mae':  float(mean_absolute_error(y, pred)),
            'r2':   float(r2_score(y, pred)),
        }

    # Naive median
    for name in ['val', 'test']:
        y = y_va if name == 'val' else y_te
        pred = np.full(len(y), train_median)
        results.setdefault('naive_median', {})[name] = {
            'rmse': float(np.sqrt(mean_squared_error(y, pred))),
            'mae':  float(mean_absolute_error(y, pred)),
            'r2':   float(r2_score(y, pred)),
        }

    # Ridge (with imputation + scaling on train)
    imp = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(imp.fit_transform(X_tr))
    X_va_s = scaler.transform(imp.transform(X_va))
    X_te_s = scaler.transform(imp.transform(X_te))
    ridge = Ridge(alpha=CONFIG['ridge_alpha']).fit(X_tr_s, y_tr)

    for name, X_s, y in [('val', X_va_s, y_va), ('test', X_te_s, y_te)]:
        pred = ridge.predict(X_s)
        results.setdefault('ridge', {})[name] = {
            'rmse': float(np.sqrt(mean_squared_error(y, pred))),
            'mae':  float(mean_absolute_error(y, pred)),
            'r2':   float(r2_score(y, pred)),
        }

    return results


# =============================================================================
# PLOT 1 — SAMPLE STRUCTURE
# =============================================================================

def fig_sample_structure(splits, df_full, output_dir):
    """
    Two panels:
      Left  — Observations per split (horizontal bar)
      Right — Deals per effective year, coloured by split assignment
    """
    logger.info("Plot 1: sample structure ...")

    split_ns = {
        'train_cv':  len(splits['y_train_cv']),
        'inner_val': len(splits['y_inner_val']),
        'val':       len(splits['y_val']),
        'test':      len(splits['y_test']),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # -- Left: horizontal bar of n per split
    ax = axes[0]
    labels = ['train_cv\n(1995–2012)', 'inner_val\n(2013–2015)',
              'val\n(2016–2018)', 'test\n(2019–2022)']
    keys   = ['train_cv', 'inner_val', 'val', 'test']
    vals   = [split_ns[k] for k in keys]
    colors = [SPLIT_COLORS[k] for k in keys]
    bars = ax.barh(labels, vals, color=colors, alpha=0.85, edgecolor='white')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 15, bar.get_y() + bar.get_height() / 2,
                f'{v:,}', va='center', fontsize=10, fontweight='bold')
    ax.set_xlabel('Number of deals', fontsize=11)
    ax.set_title('Observations per Split', fontsize=12, fontweight='bold')
    ax.set_xlim(0, max(vals) * 1.18)
    ax.grid(True, axis='x', alpha=0.3)
    ax.invert_yaxis()
    total = sum(vals)
    ax.text(0.97, 0.04, f'Total: {total:,}', transform=ax.transAxes,
            ha='right', fontsize=10, color='#444')

    # -- Right: time distribution from full dataset (if available)
    ax2 = axes[1]
    if df_full is not None and 'deal_year' in df_full.columns and 'split' in df_full.columns:
        df_labeled = df_full[df_full['split'].notna()].copy()
        yr_split = (
            df_labeled.groupby(['deal_year', 'split'])
            .size().reset_index(name='n')
        )
        split_order = ['train_cv', 'inner_val', 'val', 'test']
        yr_pivot = yr_split.pivot(index='deal_year', columns='split', values='n').fillna(0)
        # Reorder columns
        yr_pivot = yr_pivot.reindex(
            columns=[c for c in split_order if c in yr_pivot.columns]
        )
        bottom = np.zeros(len(yr_pivot))
        for col in yr_pivot.columns:
            ax2.bar(yr_pivot.index, yr_pivot[col], bottom=bottom,
                    color=SPLIT_COLORS.get(col, '#9E9E9E'), alpha=0.85,
                    label=col, width=0.7)
            bottom += yr_pivot[col].values
        ax2.set_xlabel('Deal effective year', fontsize=11)
        ax2.set_ylabel('Number of deals', fontsize=11)
        ax2.set_title('Deals per Year by Split Assignment', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9, loc='upper left')
        ax2.grid(True, axis='y', alpha=0.3)
        ax2.set_xticks(yr_pivot.index[::3])
        ax2.tick_params(axis='x', rotation=45)
    else:
        # Fallback: use df_*_meta from splits
        dfs = []
        for split_name in ['train_cv', 'inner_val', 'val', 'test']:
            meta_key = f'df_{split_name}_meta' if split_name != 'train_cv' else 'df_val_meta'
            # Approximate from deal_year in meta dataframes
        ax2.text(0.5, 0.5, 'Full dataset CSV not available.\nTime distribution skipped.',
                 ha='center', va='center', transform=ax2.transAxes, fontsize=11, color='#777')
        ax2.set_title('Deals per Year by Split Assignment', fontsize=12)

    fig.suptitle('Dataset Structure — M&A Synergy Estimation', fontsize=13, fontweight='bold')
    plt.tight_layout()
    _savefig(fig, 'sample_structure.png', output_dir)


# =============================================================================
# PLOT 2 — MODEL COMPARISON
# =============================================================================

def fig_model_comparison(model, splits, baselines, output_dir):
    """
    RMSE and R² side-by-side for naive_mean, naive_median, Ridge, XGBoost
    on val and test.  Bootstrap 95% CI bars on RMSE for val/test XGBoost.
    """
    logger.info("Plot 2: model comparison ...")

    y_va = splits['y_val'].values
    y_te = splits['y_test'].values
    xgb_val  = model.predict(splits['X_val'])
    xgb_test = model.predict(splits['X_test'])

    # Bootstrap CIs for XGBoost only
    ci_val_lo,  ci_val_hi  = _bootstrap_rmse_ci(y_va, xgb_val,  CONFIG['n_bootstrap'], CONFIG['seed'])
    ci_test_lo, ci_test_hi = _bootstrap_rmse_ci(y_te, xgb_test, CONFIG['n_bootstrap'], CONFIG['seed'])

    model_names   = ['Naive\nMean', 'Naive\nMedian', 'Ridge', 'XGBoost']
    bl_keys       = ['naive_mean', 'naive_median', 'ridge', '_xgb']
    bar_colors    = ['#BDBDBD', '#9E9E9E', '#5C6BC0', '#E53935']

    def _get(key, split, metric):
        if key == '_xgb':
            pred = xgb_val if split == 'val' else xgb_test
            y    = y_va    if split == 'val' else y_te
            if metric == 'rmse': return float(np.sqrt(mean_squared_error(y, pred)))
            if metric == 'r2':   return float(r2_score(y, pred))
            if metric == 'mae':  return float(mean_absolute_error(y, pred))
        return baselines[key][split][metric]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    metrics = [
        ('rmse', 'RMSE (lower is better)', False),
        ('mae',  'MAE  (lower is better)', False),
        ('r2',   'R²   (higher is better)', True),
    ]

    x = np.arange(len(model_names))
    width = 0.35

    for ax, (metric, ylabel, higher_better) in zip(axes, metrics):
        val_vals  = [_get(k, 'val',  metric) for k in bl_keys]
        test_vals = [_get(k, 'test', metric) for k in bl_keys]

        bars_v = ax.bar(x - width / 2, val_vals,  width, label='Val (2016–18)',
                        color=bar_colors, alpha=0.6, edgecolor='white')
        bars_t = ax.bar(x + width / 2, test_vals, width, label='Test (2019–22)',
                        color=bar_colors, alpha=0.95, edgecolor='white',
                        linewidth=0.5)

        # CI whiskers on XGBoost RMSE only
        if metric == 'rmse':
            xgb_idx = len(model_names) - 1
            rmse_val  = val_vals[xgb_idx]
            rmse_test = test_vals[xgb_idx]
            ax.errorbar(xgb_idx - width / 2, rmse_val,
                        yerr=[[rmse_val - ci_val_lo], [ci_val_hi - rmse_val]],
                        fmt='none', color='black', capsize=4, lw=1.5, zorder=5)
            ax.errorbar(xgb_idx + width / 2, rmse_test,
                        yerr=[[rmse_test - ci_test_lo], [ci_test_hi - rmse_test]],
                        fmt='none', color='black', capsize=4, lw=1.5, zorder=5)

        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=10)
        ax.set_title(ylabel, fontsize=11, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        ax.axhline(0, color='black', lw=0.5, alpha=0.5)

        # Shade best model column
        best_idx = int(np.argmax(test_vals) if higher_better else np.argmin(test_vals))
        ax.axvspan(best_idx - 0.5, best_idx + 0.5, alpha=0.06, color='#4CAF50')

    axes[0].legend(fontsize=9)
    fig.suptitle(
        'Model Benchmark Comparison — Val and Test Sets\n'
        '(XGBoost RMSE error bars = bootstrap 95% CI, 1,000 resamples)',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    _savefig(fig, 'model_comparison.png', output_dir)


# =============================================================================
# PLOT 3 — ERROR DISTRIBUTION OVERLAY
# =============================================================================

def fig_error_distribution_overlay(model, splits, output_dir):
    """
    KDE of residuals for val and test on the same axes using seaborn.
    Separate panel for squared errors.  Helps detect bias shift between splits.
    """
    logger.info("Plot 3: error distribution overlay ...")

    sets = [
        ('Val (2016–18)',  splits['X_val'],  splits['y_val'],  SPLIT_COLORS['val']),
        ('Test (2019–22)', splits['X_test'], splits['y_test'], SPLIT_COLORS['test']),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, (title, xlabel) in zip(
        axes,
        [('Residuals  (actual − predicted)',  'Residual'),
         ('Squared Errors',                   'Squared Error')]
    ):
        for (label, X, y, col) in sets:
            pred  = model.predict(X)
            resid = y.values - pred
            errors = resid if 'Residual' in xlabel else resid ** 2

            # Use seaborn kdeplot to avoid scipy.stats.gaussian_kde version issues
            sns.kdeplot(data=errors, ax=ax, label=label, color=col, lw=2.2, fill=True, alpha=0.12)

            mu = errors.mean()
            ax.axvline(mu, color=col, ls='--', alpha=0.7, lw=1.2)

        ax.axvline(0, color='black', lw=0.8, alpha=0.5)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        'Error Distribution — Val vs Test\n'
        '(dashed line = mean error per split)',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    _savefig(fig, 'error_distribution_overlay.png', output_dir)


# =============================================================================
# PLOT 4 — CORRELATION HEATMAP
# =============================================================================

def fig_correlation_heatmap(splits, output_dir):
    """
    Pearson correlation heatmap for all features used in training.
    Computed on the combined train_cv + inner_val set (the full training data).
    Rows/cols clustered by channel for interpretability.
    """
    logger.info("Plot 4: correlation heatmap ...")

    X_tr = splits['X_train_cv']
    X_iv = splits['X_inner_val']
    X_combined = pd.concat([X_tr, X_iv], axis=0, ignore_index=True)

    corr = X_combined.corr(method='pearson')
    n_feat = len(corr)

    fig, ax = plt.subplots(figsize=(max(12, n_feat * 0.52), max(10, n_feat * 0.48)))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        ax=ax,
        cmap='RdBu_r',
        center=0,
        vmin=-1, vmax=1,
        annot=(n_feat <= 20),
        fmt='.2f',
        annot_kws={'size': 6},
        linewidths=0.3,
        linecolor='#EEEEEE',
        cbar_kws={'label': 'Pearson r', 'shrink': 0.7},
        square=True,
    )
    ax.set_title(
        f'Feature Correlation Matrix (train_cv + inner_val, n={len(X_combined):,})',
        fontsize=12, fontweight='bold', pad=12
    )
    ax.tick_params(axis='x', rotation=45, labelsize=7.5)
    ax.tick_params(axis='y', rotation=0,  labelsize=7.5)

    plt.tight_layout()
    _savefig(fig, 'correlation_heatmap.png', output_dir)


# =============================================================================
# PLOT 5 — FEATURE DISTRIBUTIONS
# =============================================================================

def fig_feature_distributions(splits, imp_df, output_dir):
    """
    Box-and-whisker plots of the top numeric features (by gain importance)
    on train_cv vs test, side by side.  Binary features excluded.
    Outliers suppressed for readability (flierprops).
    """
    logger.info("Plot 5: feature distributions ...")

    BINARY = {
        "deal_tender_offer", "deal_friendly", "deal_cross_border",
        "deal_stock_payment", "deal_all_cash", "deal_industry_4dig",
        "deal_industry_2dig",
    }

    # Top continuous features by gain
    numeric_feats = [
        f for f in imp_df['feature']
        if f not in BINARY and f in splits['X_train_cv'].columns
    ][:12]

    if not numeric_feats:
        logger.warning("  No numeric features found — feature distribution plot skipped")
        return

    n_cols = 4
    n_rows = int(np.ceil(len(numeric_feats) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3.2))
    axes_flat = axes.flat if n_rows > 1 else [axes] if n_cols == 1 else axes.flat

    for ax, feat in zip(axes_flat, numeric_feats):
        data_tr = splits['X_train_cv'][feat].dropna().values
        data_te = splits['X_test'][feat].dropna().values

        # Clip to 1–99th percentile for box display
        p1, p99 = np.percentile(np.concatenate([data_tr, data_te]), [1, 99])
        data_tr_c = np.clip(data_tr, p1, p99)
        data_te_c = np.clip(data_te, p1, p99)

        bp = ax.boxplot(
            [data_tr_c, data_te_c],
            labels=['train_cv', 'test'],
            patch_artist=True,
            flierprops={'markersize': 2, 'alpha': 0.3},
            medianprops={'color': 'black', 'lw': 1.5},
            widths=0.5,
        )
        bp['boxes'][0].set_facecolor(SPLIT_COLORS['train_cv'])
        bp['boxes'][1].set_facecolor(SPLIT_COLORS['test'])
        for patch in bp['boxes']:
            patch.set_alpha(0.75)

        # Gain label
        gain_row = imp_df[imp_df['feature'] == feat]
        gain_str = f"  gain={gain_row['gain_pct'].values[0]:.1f}%" if len(gain_row) else ""
        ch = imp_df.loc[imp_df['feature'] == feat, 'channel'].values
        ch_col = CH_COLORS.get(ch[0], '#9E9E9E') if len(ch) else '#9E9E9E'

        # Shorten feature name for display
        short = feat.replace('financial_', 'fin_').replace('operational_', 'op_') \
                    .replace('revenue_', 'rev_').replace('cost_', 'c_') \
                    .replace('acquiror', 'acq').replace('target', 'tgt')
        ax.set_title(f"{short}{gain_str}", fontsize=8.5, color=ch_col, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        ax.tick_params(labelsize=8)

    # Hide unused axes
    for ax in list(axes_flat)[len(numeric_feats):]:
        ax.set_visible(False)

    fig.suptitle(
        'Feature Distributions — train_cv vs test  (top continuous features by gain importance)\n'
        'Values clipped to [1st, 99th] percentile; colour = synergy channel',
        fontsize=11, fontweight='bold'
    )
    legend_els = [
        mpatches.Patch(facecolor=CH_COLORS[c], label=c, alpha=0.8)
        for c in ["cost", "revenue", "operational", "financial", "macro"]
    ]
    fig.legend(handles=legend_els, fontsize=9, loc='lower right',
               ncol=5, bbox_to_anchor=(0.99, 0.01))
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    _savefig(fig, 'feature_distributions.png', output_dir)


# =============================================================================
# PLOT 6 — MISSINGNESS OVERVIEW
# =============================================================================

def fig_missingness_overview(splits, df_full, output_dir):
    """
    Horizontal bar chart showing NaN% per feature in the full ML-stage dataset
    (df_full) and in train_cv / test separately.
    """
    logger.info("Plot 6: missingness overview ...")

    features = list(splits['X_train_cv'].columns)

    miss_full  = None
    if df_full is not None:
        feats_in_full = [f for f in features if f in df_full.columns]
        miss_full = df_full[feats_in_full].isna().mean() * 100

    miss_train = splits['X_train_cv'].isna().mean() * 100
    miss_test  = splits['X_test'].isna().mean() * 100

    # Sort by overall missingness descending
    sorted_feats = miss_train.sort_values(ascending=False).index.tolist()

    fig, ax = plt.subplots(figsize=(10, max(7, len(sorted_feats) * 0.32)))

    y = np.arange(len(sorted_feats))
    width = 0.28

    ax.barh(y - width, [miss_train[f] for f in sorted_feats],
            height=width, color=SPLIT_COLORS['train_cv'], alpha=0.85, label='train_cv')
    ax.barh(y,         [miss_test[f]  for f in sorted_feats],
            height=width, color=SPLIT_COLORS['test'],     alpha=0.85, label='test')
    if miss_full is not None:
        ax.barh(y + width, [miss_full.get(f, np.nan) for f in sorted_feats],
                height=width, color='#CFD8DC', alpha=0.85, label='full dataset')

    ax.axvline(CONFIG['missingness_warn_pct'], color='#E53935', ls='--', lw=1.2,
               label=f'{CONFIG["missingness_warn_pct"]:.0f}% threshold')
    ax.set_yticks(y)
    ax.set_yticklabels(sorted_feats, fontsize=8.5)
    ax.set_xlabel('Missing (%)', fontsize=11)
    ax.set_title(
        'Feature Missingness — ML Stage\n'
        '(after label filter; before imputation)',
        fontsize=12, fontweight='bold'
    )
    ax.legend(fontsize=9)
    ax.grid(True, axis='x', alpha=0.3)
    ax.invert_yaxis()
    plt.tight_layout()
    _savefig(fig, 'missingness_overview.png', output_dir)


# =============================================================================
# PLOT 7 — CHANNEL IMPORTANCE SUMMARY
# =============================================================================

def fig_channel_importance_summary(ch_df, imp_df, output_dir):
    """
    Two panels:
      Left  — Horizontal bar of channel-level gain % with absolute feature counts
      Right — Top 3 features per channel with individual gain %
    """
    logger.info("Plot 7: channel importance summary ...")

    ch_df = ch_df.sort_values('gain_pct', ascending=True).reset_index(drop=True)
    n_ch  = len(ch_df)

    fig, axes = plt.subplots(1, 2, figsize=(15, max(5, n_ch * 0.9)))

    # -- Left: channel-level bar
    ax = axes[0]
    colors = [CH_COLORS.get(c, '#9E9E9E') for c in ch_df['channel']]
    bars = ax.barh(
        ch_df['channel'], ch_df['gain_pct'],
        color=colors, alpha=0.85, edgecolor='white'
    )
    for bar, pct in zip(bars, ch_df['gain_pct']):
        ax.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f'{pct:.1f}%', va='center', fontsize=10, fontweight='bold'
        )
    ax.set_xlabel('Gain Importance (%)', fontsize=11)
    ax.set_title('Channel-Level Feature Importance\n(sum of gain % by synergy channel)',
                 fontsize=11, fontweight='bold')
    ax.set_xlim(0, ch_df['gain_pct'].max() * 1.22)
    ax.grid(True, axis='x', alpha=0.3)

    # Feature count annotations on left side
    if 'n_features' in ch_df.columns:
        for i, row in ch_df.iterrows():
            ax.text(-0.5, i, f"n={int(row['n_features'])}",
                    va='center', ha='right', fontsize=8, color='#555')

    # -- Right: top features per channel
    ax2 = axes[1]
    yticks, ylabels, bar_vals, bar_cols = [], [], [], []
    pos = 0
    for _, ch_row in ch_df.sort_values('gain_pct', ascending=False).iterrows():
        ch = ch_row['channel']
        col = CH_COLORS.get(ch, '#9E9E9E')
        ch_feats = imp_df[imp_df['channel'] == ch].nlargest(3, 'gain_pct')
        for _, feat_row in ch_feats.iterrows():
            short = feat_row['feature'] \
                .replace('financial_', 'fin_').replace('operational_', 'op_') \
                .replace('revenue_', 'rev_').replace('cost_', 'c_') \
                .replace('acquiror', 'acq').replace('target', 'tgt')
            yticks.append(pos)
            ylabels.append(short)
            bar_vals.append(feat_row['gain_pct'])
            bar_cols.append(col)
            pos += 1
        pos += 0.6  # small gap between channels

    bars2 = ax2.barh(yticks, bar_vals, color=bar_cols, alpha=0.8, edgecolor='white')
    for bar, v in zip(bars2, bar_vals):
        ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                 f'{v:.1f}%', va='center', fontsize=8)
    ax2.set_yticks(yticks)
    ax2.set_yticklabels(ylabels, fontsize=8.5)
    ax2.invert_yaxis()
    ax2.set_xlabel('Gain Importance (%)', fontsize=11)
    ax2.set_title('Top 3 Features per Channel\n(by gain importance)',
                  fontsize=11, fontweight='bold')
    ax2.grid(True, axis='x', alpha=0.3)

    fig.suptitle('Feature Importance by Synergy Channel — XGBoost',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    _savefig(fig, 'channel_importance_summary.png', output_dir)


# =============================================================================
# MAIN
# =============================================================================

def run_visualisation():
    plt.style.use(CONFIG['style'])
    output_dir = _ensure_output_dir()

    logger.info("=" * 60)
    logger.info("MODEL VISUALISATION -- M&A Synergy Estimation")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")

    model, model_meta, splits, imp_df, ch_df, df_full = _load_artifacts()

    logger.info("Recomputing baselines for comparison plot ...")
    baselines = _recompute_baselines(splits)

    fig_sample_structure(splits, df_full, output_dir)
    fig_model_comparison(model, splits, baselines, output_dir)
    fig_error_distribution_overlay(model, splits, output_dir)
    fig_correlation_heatmap(splits, output_dir)
    fig_feature_distributions(splits, imp_df, output_dir)
    fig_missingness_overview(splits, df_full, output_dir)
    fig_channel_importance_summary(ch_df, imp_df, output_dir)

    logger.info("=" * 60)
    logger.info(f"All 7 figures saved to: {output_dir}")
    logger.info("=" * 60)

    print("\nSummary of outputs:")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  {f.name}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_visualisation()
