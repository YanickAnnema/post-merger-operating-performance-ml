"""
Data Preparation — M&A Synergy Estimation
==========================================

Reads the feature-engineered CSV produced by feature_engineering.py and
prepares a single clean, ML-ready dataset.  Train/val/test splitting is
intentionally NOT performed here — it is handled in model_training.py using
chronological (time-series) splitting, consistent with the M&A finance
literature (Ajayi et al. 2022; Amini et al. 2021).

Pipeline:
  1. load_data()                  — load CSV, parse dates
  2. leakage_check()              — assert no post-deal column in feature set
  3. filter_valid_target()        — drop rows with missing synergy label
  4. load_and_merge_macro()       — load pre-built macro_monthly.csv (from
                                    fetch_macro_data.py); merge on deal year-month
  5. winsorize_features()         — 1%/99% clipping on continuous features
  6. missing_value_report()       — per-channel coverage diagnostics
  7. impute_missing()             — median/mode imputation (linear model variant)
  8. scale_features()             — Z-score standardisation (linear model variant)
  9. correlation_check()          — flag |r| > threshold pairs (informational)
  10. save_outputs()              — three CSVs + prep_artifacts.pkl + prep_report.txt

Output files:
  ml_ready.csv
      Full prepared dataset, NaN intact — XGBoost-ready.
      Pass missing=np.nan when constructing xgb.DMatrix.
  ml_ready_imputed.csv
      Median/mode-imputed — Lasso/Ridge-ready before scaling.
  ml_ready_scaled.csv
      Imputed + Z-score scaled — Lasso/Ridge final input.
  prep_artifacts.pkl
      Serialised clip bounds, fill values, scale params, and feature list.
      Load in model_training.py to apply identical transformations to each fold.
  prep_report.txt
      Full diagnostic text log.

Macro proxy design:
  sp500_trailing_12m
      S&P 500 price return (month-end close, excludes dividends) in the 12 calendar months ending the
      month BEFORE the deal's effective month.  This avoids any same-month
      lookahead: if a deal closes in March 2005, the value is
      (price_Feb_2005 / price_Feb_2004) - 1.
      Source: Yahoo Finance ^GSPC (daily closes → month-end resampled).
      Rationale: captures merger wave and equity market valuation conditions,
      the primary driver of deal clustering (Andrade et al. 2001;
      Rhodes-Kropf et al. 2005).

  credit_spread_bbb_aaa
      Moody's Seasoned Baa yield minus Moody's Seasoned Aaa yield, monthly
      average, for the calendar month BEFORE the deal's effective month.
      Source: FRED series BAA and AAA (monthly, available from 1919).
      Rationale: captures financing conditions independently of equity market
      direction — financial synergies are more realizable when credit is cheap
      (Damodaran 2005; Harford 1999).  The spread is orthogonal to equity
      returns in several market cycles (e.g. 2009), providing complementary
      information.

Dependencies:
  No external downloads at run-time. Run fetch_macro_data.py once first:
    pip install yfinance pandas-datareader --break-system-packages
    python code/fetch_macro_data.py

Optimised for Spyder IDE (F5 execution).
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
import pickle
from typing import Dict, List, Optional, Tuple

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DAQ_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "DAQ pipeline"

CONFIG = {
    # Input: CSV produced by feature_engineering.py
    'input_path':  DAQ_OUTPUT_DIR / "full_deal_level_features.csv",

    # Output directory (same folder as input by default)
    'output_dir':  DAQ_OUTPUT_DIR,

    # File stem for all output files
    'output_stem': "ml_ready",

    # Primary target variable — winsorised Healy CFROA (produced by feature_engineering.py).
    # Switch to "synergy_healy1992" for sensitivity analysis on the unwinsorised version.
    'target_col':  "synergy_healy1992_w",

    # Date column (kept in all outputs for use by model_training.py)
    'date_col':    "DateEffective",

    # Chronological sample window and split boundaries.
    # Deals before sample_start_year are excluded: label yield is very low
    # (< 10%) in the pre-1995 period and Worldscope coverage is structurally
    # weaker, adding noise without adding information.
    # Split design (per project_plan.pdf, chronological / no-shuffle):
    #   train : sample_start_year … train_end   (inclusive)
    #   val   : train_end+1       … val_end      (inclusive)
    #   test  : val_end+1         … present      (all remaining labeled deals)
    'sample_start_year': 1995,
    'split_years': {
        'train_end': 2015,
        'val_end':   2018,
    },

    # Feature winsorisation percentiles.
    # Note: bounds are fitted on the full prepared sample here.
    # model_training.py re-fits on each training fold to prevent leakage.
    'winsor_low':  0.01,
    'winsor_high': 0.99,

    # Exclude feature from IMPUTED/SCALED outputs if coverage < this fraction.
    # Features are always retained in the raw (XGBoost) output.
    'min_obs_pct': 0.10,

    # Pearson |r| threshold for flagging correlated pairs (informational, no auto-drop).
    'corr_threshold': 0.85,

    # Macro data: pre-built by fetch_macro_data.py (run once).
    # Columns used: sp500_trailing_12m, credit_spread_bbb_aaa (already lag-adjusted).
    # If the file is absent, both macro columns are set to NaN with a warning.
    'macro_csv': DAQ_OUTPUT_DIR / "macro_monthly.csv",

    # Output filenames
    'artifacts_file': "prep_artifacts.pkl",
    'report_file':    "prep_report.txt",
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
# FEATURE DEFINITIONS
# =============================================================================
#
# All features listed here are pre-deal: built from acquiror/target financials
# at time t and deal screener metadata.  Post-deal data (t+3) is used only
# to construct the target variable and must never appear as a predictor.
#
# Source: feature_engineering.py — compute_ratio_features() and
#                                   compute_screener_features().
# Macro: loaded from macro_monthly.csv (built once by fetch_macro_data.py).

FEATURES_COST = [
    "cost_relative_asset_size",          # B_assets / A_assets
    "cost_ppe_intensity_diff",           # |PPE/assets_A − PPE/assets_B|
    "cost_inventory_turnover_gap",       # |revenue/inventory_A − revenue/inventory_B|
    # cost_sales_per_employee_gap excluded: TR.F.Employees unresolved in Worldscope
    # for this sample; the column is always NaN and adds no information.
    "cost_target_asset_utilization",     # B_revenue / B_assets
    "log_deal_value",                    # log(deal value USD) — scale proxy
    "deal_tender_offer",                 # 1 = tender offer
    "deal_friendly",                     # 1 = friendly acquisition
]

FEATURES_REVENUE = [
    "revenue_rd_intensity_diff",         # |R&D/assets_A − R&D/assets_B|
    "revenue_capex_intensity_diff",      # |capex/assets_A − capex/assets_B|
    "revenue_intangible_intensity_diff", # |intangibles/assets_A − intangibles/assets_B|
    "revenue_relative_size_sales",       # B_revenue / A_revenue
    "deal_cross_border",                 # 1 = cross-border transaction
]

FEATURES_OPERATIONAL = [
    "operational_asset_turnover_gap",    # |revenue/assets_A − revenue/assets_B|
    "operational_roa_gap",               # |EBIT/assets_A − EBIT/assets_B|
    "operational_acquiror_op_margin",    # EBIT / A_revenue
    "operational_target_cf_margin",      # B_CFO / B_revenue
    "deal_industry_4dig",                # 1 = same 4-digit SIC code
    "deal_industry_2dig",                # 1 = same 2-digit SIC code
]

FEATURES_FINANCIAL = [
    "financial_leverage_gap",            # |debt/assets_A − debt/assets_B|
    "financial_cash_ratio_diff",         # |cash/assets_A − cash/assets_B|
    "financial_acquiror_cash_to_sales",  # A_cash / A_revenue  (cash slack proxy)
    "financial_quick_ratio_acquiror",    # (current_assets − inventory) / current_liabilities
    "financial_quick_ratio_target",      # same for target firm
    "deal_stock_payment",                # 1 = stock-financed (% cash < 50)
    "deal_all_cash",                     # 1 = all-cash deal
    # Modified Altman (1968) Z-score — pre-deal financial health of each party.
    # X2 proxied by total_equity/TA; X4 denominator proxied by (TA − equity).
    # Captures financial distress risk; relevant to financial synergy channel
    # (tax shield realisation, financial slack transfer).
    "financial_altman_z_acquiror",       # acquiror Z-score at t
    "financial_altman_z_target",         # target Z-score at t
]

# Macroeconomic controls: control for merger waves and financing conditions.
# These absorb the time-dependency of deals through shared macro environment,
# which supports the use of chronological splitting in model_training.py.
FEATURES_MACRO = [
    "sp500_trailing_12m",       # S&P 500 12-month price return, excludes dividends (t-13 to t-1 relative to deal month)
    "credit_spread_bbb_aaa",    # Moody's Baa − Aaa yield spread, month t-1
]

ALL_FEATURES: List[str] = (
    FEATURES_COST
    + FEATURES_REVENUE
    + FEATURES_OPERATIONAL
    + FEATURES_FINANCIAL
    + FEATURES_MACRO
)

# Binary/dummy features — excluded from Z-score scaling; mode-imputed not median.
BINARY_FEATURES = {
    "deal_tender_offer", "deal_friendly", "deal_cross_border",
    "deal_stock_payment", "deal_all_cash", "deal_industry_4dig",
    "deal_industry_2dig",
}

# Columns that are strictly post-deal — hard leakage guard.
POST_DEAL_COLS = {
    "AB_t3_CFROA", "AB_t3_operating_cashflow", "AB_t3_total_assets",
    "Delta_CFROA_raw", "synergy_healy1992", "synergy_healy1992_w",
    "industry_CFROA_adjustment", "t3AB_fye",
}


# =============================================================================
# STEP 1 — LOAD DATA
# =============================================================================

def load_data(path: Path) -> pd.DataFrame:
    logger.info(f"Loading feature-engineered CSV: {path}")
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}\n"
            "Run feature_engineering.py first to produce this file."
        )
    df = pd.read_csv(path, low_memory=False)
    logger.info(f"  Shape: {df.shape}")
    if CONFIG['date_col'] in df.columns:
        df[CONFIG['date_col']] = pd.to_datetime(df[CONFIG['date_col']], errors='coerce')
        n_ok = df[CONFIG['date_col']].notna().sum()
        logger.info(f"  {CONFIG['date_col']}: {n_ok}/{len(df)} rows parsed")
    else:
        logger.warning(
            f"  Column '{CONFIG['date_col']}' not found — "
            "macro merge and chronological split will fail."
        )
    return df


# =============================================================================
# STEP 2 — LEAKAGE GUARD
# =============================================================================

def leakage_check(features: List[str]) -> None:
    """
    Hard assertion: no post-deal column may appear in the ML feature set.
    Raises ValueError immediately — this is never just a warning.
    """
    leaked = [f for f in features if f in POST_DEAL_COLS]
    if leaked:
        raise ValueError(
            f"DATA LEAKAGE DETECTED — post-deal columns in feature set: {leaked}\n"
            "Remove these from ALL_FEATURES immediately."
        )
    logger.info("  ✓ Leakage check passed — all defined features are pre-deal")


# =============================================================================
# STEP 3 — FILTER VALID TARGET
# =============================================================================

def filter_valid_target(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where the target variable is NaN."""
    target = CONFIG['target_col']
    if target not in df.columns:
        raise KeyError(
            f"Target column '{target}' not found. "
            "Run feature_engineering.py first, or adjust CONFIG['target_col']."
        )
    before = len(df)
    df = df[df[target].notna()].reset_index(drop=True)
    after  = len(df)
    pct    = 100 * after / before if before > 0 else 0.0
    logger.info(f"  Kept {after}/{before} deals ({pct:.1f}%) with non-NaN '{target}'")
    if after == 0:
        logger.warning(
            "  WARNING: 0 deals have a valid target. "
            "Check that feature_engineering.py produced synergy labels."
        )
    return df


# =============================================================================
# STEP 3b — SAMPLE WINDOW FILTER
# =============================================================================

def filter_sample_window(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop deals before CONFIG['sample_start_year'].

    Rationale: pre-1995 deals have structurally weaker Worldscope coverage and
    very low label yield (< 10%), adding noise rather than signal.  The training
    window starts at sample_start_year, so excluding earlier deals here also
    ensures the ML-ready sample is self-consistent with the split design.

    Uses the 'deal_year' column written by feature_engineering.py.
    Falls back to parsing DateEffective if 'deal_year' is absent.
    """
    start_year = CONFIG['sample_start_year']
    n_before   = len(df)

    if "deal_year" in df.columns:
        yr = pd.to_numeric(df["deal_year"], errors="coerce")
    elif CONFIG['date_col'] in df.columns:
        yr = df[CONFIG['date_col']].dt.year
        logger.warning("  'deal_year' column absent — parsed from DateEffective")
    else:
        logger.warning(
            f"  No year column found — sample_start_year={start_year} filter skipped"
        )
        return df

    n_pre  = int((yr < start_year).sum())
    n_null = int(yr.isna().sum())
    df     = df[yr >= start_year].reset_index(drop=True)
    n_after = len(df)

    logger.info(
        f"  Sample window: kept deals from {start_year} onward  "
        f"({n_after}/{n_before} rows; "
        f"dropped {n_pre} pre-{start_year}, {n_null} NaN-year)"
    )
    return df


# =============================================================================
# STEP 3c — CHRONOLOGICAL SPLIT ASSIGNMENT
# =============================================================================

def assign_split(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'split' column with values 'train' / 'val' / 'test' based on deal_year.

    Boundaries (inclusive, from CONFIG['split_years']):
      train : sample_start_year … train_end
      val   : train_end+1       … val_end
      test  : val_end+1         … present

    The split column is written to all three output CSVs so that model_training.py
    can filter directly without re-implementing the logic.

    IMPORTANT: imputation and scaling bounds computed in steps 7–8 use the FULL
    labeled sample (all splits).  model_training.py must re-fit these transforms
    on the training split only to prevent any lookahead leakage at fold time.
    """
    train_end = CONFIG['split_years']['train_end']
    val_end   = CONFIG['split_years']['val_end']
    start     = CONFIG['sample_start_year']

    if "deal_year" in df.columns:
        yr = pd.to_numeric(df["deal_year"], errors="coerce")
    elif CONFIG['date_col'] in df.columns:
        yr = df[CONFIG['date_col']].dt.year
    else:
        logger.warning("  No year column — split assignment skipped")
        df["split"] = np.nan
        return df

    conditions = [
        (yr >= start)      & (yr <= train_end),
        (yr > train_end)   & (yr <= val_end),
        (yr > val_end),
    ]
    choices = ["train", "val", "test"]

    df["split"] = np.select(conditions, choices, default=np.nan)

    counts = df["split"].value_counts()
    logger.info(
        f"  Split assignment (train≤{train_end} / val≤{val_end} / test>{val_end}):"
    )
    for s in ["train", "val", "test"]:
        n = int(counts.get(s, 0))
        logger.info(f"    {s:<6}: {n:>5} deals  ({100*n/max(len(df),1):.1f}%)")
    n_null = int((df["split"].isna() | (df["split"] == "nan")).sum())
    if n_null:
        logger.warning(f"    {n_null} rows have no split assignment (deal_year NaN)")

    return df


# =============================================================================
# STEP 4 — FETCH AND MERGE MACRO PROXIES
# =============================================================================

def load_and_merge_macro(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Load pre-built macro features from macro_monthly.csv (produced by
    fetch_macro_data.py) and merge into df on deal year-month.

    The CSV already has the 1-month lag applied, so no date arithmetic is
    performed here — the merge is a plain string key lookup on YYYY-MM.

    If the file is absent or cannot be read, both macro columns are set to NaN
    and the pipeline continues.  Run fetch_macro_data.py once to build the file.
    """
    date_col     = CONFIG['date_col']
    macro_csv    = CONFIG['macro_csv']
    macro_status: Dict[str, str] = {}

    # Ensure date column present
    if date_col not in df.columns or df[date_col].isna().all():
        logger.warning(f"  '{date_col}' absent — macro merge skipped")
        df['sp500_trailing_12m']    = np.nan
        df['credit_spread_bbb_aaa'] = np.nan
        macro_status = {k: 'SKIPPED (no date col)' for k in
                        ['sp500_trailing_12m', 'credit_spread_bbb_aaa']}
        return df, macro_status

    # Load macro CSV
    if not Path(macro_csv).exists():
        logger.warning(
            f"  macro_monthly.csv not found at {macro_csv}.\n"
            "  Run fetch_macro_data.py once to build it.\n"
            "  Both macro features set to NaN."
        )
        df['sp500_trailing_12m']    = np.nan
        df['credit_spread_bbb_aaa'] = np.nan
        macro_status = {k: f'MISSING — run fetch_macro_data.py' for k in
                        ['sp500_trailing_12m', 'credit_spread_bbb_aaa']}
        return df, macro_status

    try:
        macro = pd.read_csv(macro_csv, index_col='year_month', dtype=str)
        macro['sp500_trailing_12m']    = pd.to_numeric(macro['sp500_trailing_12m'],    errors='coerce')
        macro['credit_spread_bbb_aaa'] = pd.to_numeric(macro['credit_spread_bbb_aaa'], errors='coerce')
        logger.info(f"  Loaded macro_monthly.csv: {len(macro)} rows ({macro.index[0]} → {macro.index[-1]})")
    except Exception as exc:
        logger.warning(f"  Failed to load macro_monthly.csv: {exc}")
        df['sp500_trailing_12m']    = np.nan
        df['credit_spread_bbb_aaa'] = np.nan
        macro_status = {k: f'ERROR ({exc})' for k in
                        ['sp500_trailing_12m', 'credit_spread_bbb_aaa']}
        return df, macro_status

    # Build deal year-month key (YYYY-MM string) — matches macro CSV index
    df['_ym'] = df[date_col].dt.strftime('%Y-%m')

    for col in ['sp500_trailing_12m', 'credit_spread_bbb_aaa']:
        df[col] = df['_ym'].map(macro[col])
        n_filled = df[col].notna().sum()
        coverage = f"{n_filled}/{len(df)} deals ({100*n_filled/len(df):.1f}%)"
        macro_status[col] = f"OK — {coverage}"
        logger.info(f"  {col} merged: {coverage}")

    df = df.drop(columns=['_ym'])

    logger.info("  Macro merge complete:")
    for k, v in macro_status.items():
        logger.info(f"    {k:<28}: {v}")

    return df, macro_status


# =============================================================================
# STEP 5 — WINSORISE FEATURES
# =============================================================================

def winsorize_features(
    df: pd.DataFrame,
    features: List[str],
) -> Tuple[pd.DataFrame, Dict]:
    """
    Clip each continuous feature at [winsor_low, winsor_high] percentiles.

    Bounds are fitted on the full prepared sample here.
    model_training.py must re-fit these on each training fold only to
    prevent any form of lookahead leakage at split time.
    Binary features are excluded.
    """
    lo = CONFIG['winsor_low']
    hi = CONFIG['winsor_high']
    clip_bounds: Dict[str, Tuple[float, float]] = {}
    n_done = 0

    for feat in features:
        if feat in BINARY_FEATURES:
            continue
        if feat not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[feat]):
            continue
        s = df[feat].dropna()
        if s.empty:
            continue
        lo_val = float(s.quantile(lo))
        hi_val = float(s.quantile(hi))
        clip_bounds[feat] = (lo_val, hi_val)
        df[feat] = df[feat].clip(lower=lo_val, upper=hi_val)
        n_done += 1

    logger.info(
        f"  Winsorised {n_done} continuous features "
        f"at [{lo:.0%}, {hi:.0%}] percentiles"
    )
    return df, clip_bounds


# =============================================================================
# STEP 6 — MISSING VALUE REPORT
# =============================================================================

def missing_value_report(
    df: pd.DataFrame,
    features: List[str],
    report_lines: List[str],
) -> List[str]:
    """
    Print per-channel coverage statistics.
    Features with coverage below min_obs_pct are flagged LOW:
      - excluded from imputed/scaled outputs (coverage too low for median imputation)
      - retained in raw XGBoost output (XGBoost handles missing natively)
    """
    min_pct = CONFIG['min_obs_pct']
    n = len(df)

    channel_map: Dict[str, str] = {}
    for f in FEATURES_COST:        channel_map[f] = "COST"
    for f in FEATURES_REVENUE:     channel_map[f] = "REVENUE"
    for f in FEATURES_OPERATIONAL: channel_map[f] = "OPERATIONAL"
    for f in FEATURES_FINANCIAL:   channel_map[f] = "FINANCIAL"
    for f in FEATURES_MACRO:       channel_map[f] = "MACRO"

    header = (
        f"\n{'Feature':<46} {'Channel':<13} {'Non-NaN':>8} "
        f"{'Coverage':>9}  {'Status'}"
    )
    sep   = "-" * 90
    lines = ["\nMISSING VALUE REPORT (full prepared sample)", header, sep]
    low_coverage: List[str] = []

    for feat in features:
        channel = channel_map.get(feat, "OTHER")
        if feat not in df.columns:
            lines.append(f"  {feat:<46} {channel:<13} {'ABSENT':>8}")
            continue
        n_valid = int(df[feat].notna().sum())
        pct     = n_valid / n if n > 0 else 0.0
        status  = "OK" if pct >= min_pct else "LOW"
        if status == "LOW":
            low_coverage.append(feat)
        lines.append(
            f"  {feat:<46} {channel:<13} {n_valid:>8d} {pct:>8.1%}  {status}"
        )

    lines += [
        sep,
        f"  Features with coverage < {min_pct:.0%}: {len(low_coverage)} "
        f"({', '.join(low_coverage) if low_coverage else 'none'})",
        "  NOTE: LOW features are excluded from imputed/scaled outputs "
        "but kept in raw (XGBoost) output.",
    ]

    for line in lines:
        print(line)
    report_lines.extend(lines)
    return low_coverage


# =============================================================================
# STEP 7 — MEDIAN / MODE IMPUTATION  (Lasso/Ridge variant)
# =============================================================================

def impute_missing(
    df: pd.DataFrame,
    features: List[str],
    low_cov_feats: List[str],
) -> Tuple[pd.DataFrame, Dict]:
    """
    Impute missing values for the linear-model variant.

      Continuous features : median (robust to skew in financial ratios)
      Binary dummies       : mode (most frequent 0 or 1)
      LOW-coverage features: zeroed (coverage too sparse for meaningful imputation)

    Fill values are computed on the full sample here.
    model_training.py must re-compute fill values on each training fold only.
    XGBoost does NOT need this — use ml_ready.csv with missing=np.nan.
    """
    excluded    = set(low_cov_feats)
    fill_values: Dict[str, float] = {}
    n_imputed   = 0

    for feat in features:
        if feat not in df.columns:
            continue
        s = df[feat].dropna()
        if feat in excluded or s.empty:
            fill_values[feat] = 0.0
        elif feat in BINARY_FEATURES:
            fill_values[feat] = float(s.mode().iloc[0])
        else:
            fill_values[feat] = float(s.median())
        if df[feat].isna().any():
            n_imputed += 1

    df_imp = df.copy()
    for feat, fval in fill_values.items():
        if feat in df_imp.columns:
            df_imp[feat] = df_imp[feat].fillna(fval)

    logger.info(
        f"  Imputed {n_imputed} features "
        f"({len(excluded)} LOW-coverage features zeroed)"
    )
    return df_imp, fill_values


# =============================================================================
# STEP 8 — STANDARD SCALING  (Lasso/Ridge variant)
# =============================================================================

def scale_features(
    df: pd.DataFrame,
    features: List[str],
    low_cov_feats: List[str],
) -> Tuple[pd.DataFrame, Dict]:
    """
    Z-score standardise continuous features for linear models.

    Excluded from scaling: binary dummies (kept as {0,1}) and LOW-coverage features.
    Features with near-zero variance are centred only (std = 1) with a warning.

    Scale params are computed on the full sample here.
    model_training.py must re-compute on each training fold only.
    """
    excluded = set(low_cov_feats) | BINARY_FEATURES
    scale_params: Dict[str, Tuple[float, float]] = {}

    for feat in features:
        if feat in excluded or feat not in df.columns:
            continue
        s = df[feat].dropna()
        if s.empty:
            continue
        mu  = float(s.mean())
        std = float(s.std())
        if std < 1e-8:
            logger.warning(
                f"  '{feat}' near-zero std ({std:.2e}) — centred only"
            )
            std = 1.0
        scale_params[feat] = (mu, std)

    df_sc = df.copy()
    for feat, (mu, std) in scale_params.items():
        if feat in df_sc.columns:
            df_sc[feat] = (df_sc[feat] - mu) / std

    logger.info(
        f"  Z-score scaling applied to {len(scale_params)} continuous features"
    )
    return df_sc, scale_params


# =============================================================================
# STEP 9 — CORRELATION CHECK
# =============================================================================

def correlation_check(
    df: pd.DataFrame,
    features: List[str],
    report_lines: List[str],
) -> None:
    """
    Flag feature pairs with Pearson |r| > corr_threshold.
    Informational only — no features are removed here.
    Lasso handles multicollinearity through coefficient shrinkage.
    Review flagged pairs before applying Ridge or OLS.
    """
    threshold = CONFIG['corr_threshold']
    feat_cols = [
        f for f in features
        if f in df.columns and pd.api.types.is_numeric_dtype(df[f])
    ]
    if len(feat_cols) < 2:
        return

    corr_matrix = df[feat_cols].corr(method='pearson')
    high_pairs  = []
    for i in range(len(feat_cols)):
        for j in range(i + 1, len(feat_cols)):
            r = corr_matrix.iloc[i, j]
            if not np.isnan(r) and abs(r) > threshold:
                high_pairs.append((feat_cols[i], feat_cols[j], r))

    lines = [
        f"\nCORRELATION CHECK  (Pearson |r| > {threshold})",
        f"  {len(high_pairs)} highly correlated pair(s):",
    ]
    if high_pairs:
        for f1, f2, r in sorted(high_pairs, key=lambda x: -abs(x[2])):
            lines.append(f"    {f1:<46}  ↔  {f2:<46}  r = {r:+.3f}")
    else:
        lines.append("    None")
    lines.append(
        "  Lasso handles multicollinearity via shrinkage. "
        "Consider VIF in model_training.py before Ridge/OLS."
    )

    for line in lines:
        print(line)
    report_lines.extend(lines)


# =============================================================================
# DIAGNOSTIC SUMMARY
# =============================================================================

def print_summary(
    df: pd.DataFrame,
    features: List[str],
    macro_status: Dict,
    report_lines: List[str],
) -> None:
    target = CONFIG['target_col']
    s = df[target].dropna() if target in df.columns else pd.Series(dtype=float)

    lines = ["\n" + "=" * 70, "DATA PREPARATION SUMMARY", "=" * 70]
    lines.append(
        f"  Total deals  : {len(df)}\n"
        f"  Target       : {target}\n"
        f"  Target stats : mean={s.mean():+.4f}  std={s.std():.4f}  "
        f"[{s.min():+.4f}, {s.max():+.4f}]\n"
        f"  Features     : {len([f for f in features if f in df.columns])} / "
        f"{len(features)} present"
    )

    absent = [f for f in features if f not in df.columns]
    if absent:
        lines.append(f"  Absent features ({len(absent)}): {absent}")

    # Split counts
    if "split" in df.columns:
        lines.append("\n  Chronological split:")
        split_counts = df["split"].value_counts()
        for s_name in ["train", "val", "test"]:
            n = int(split_counts.get(s_name, 0))
            lines.append(
                f"    {s_name:<6}: {n:>5} deals  ({100*n/max(len(df),1):.1f}%)"
            )
        train_end = CONFIG['split_years']['train_end']
        val_end   = CONFIG['split_years']['val_end']
        start     = CONFIG['sample_start_year']
        lines.append(
            f"    Boundaries: train {start}–{train_end} | "
            f"val {train_end+1}–{val_end} | test {val_end+1}+"
        )

    lines.append("\n  Macro proxy status:")
    for k, v in macro_status.items():
        lines.append(f"    {k:<28}: {v}")

    lines.append(
        "\n  Outputs:\n"
        "    ml_ready_nowinsor.csv — raw, NaN intact, NO winsorisation (PRIMARY input for model_training.py)\n"
        "    ml_ready.csv          — winsorised at full-sample bounds    (reference / diagnostics only)\n"
        "    ml_ready_imputed.csv  — median/mode imputed                (Lasso/Ridge pre-scaling)\n"
        "    ml_ready_scaled.csv   — imputed + Z-scored                 (Lasso/Ridge final)\n"
        "    prep_artifacts.pkl    — clip/fill/scale params              (use in model_training.py)\n"
        "\n  NOTE: model_training.py reads ml_ready_nowinsor.csv so it can fit all\n"
        "        transformation bounds (winsorisation, imputation, scaling) on the\n"
        "        TRAIN split only, preventing full-sample lookahead leakage."
    )
    lines.append("=" * 70)

    for line in lines:
        print(line)
    report_lines.extend(lines)


# =============================================================================
# STEP 10 — SAVE OUTPUTS
# =============================================================================

def save_outputs(
    df_raw: pd.DataFrame,
    df_imp: pd.DataFrame,
    df_sc:  pd.DataFrame,
    artifacts: Dict,
    report_lines: List[str],
) -> None:
    out  = Path(CONFIG['output_dir'])
    stem = CONFIG['output_stem']
    out.mkdir(parents=True, exist_ok=True)

    def _save(df: pd.DataFrame, suffix: str) -> None:
        fname = f"{stem}.csv" if not suffix else f"{stem}{suffix}.csv"
        p = out / fname
        if df.empty:
            # Write headers-only CSV so stale data from previous runs is overwritten
            df.to_csv(p, index=False)
            logger.warning(f"  Saved {p.name:<38} (EMPTY — 0 rows, headers only)")
            return
        df.to_csv(p, index=False)
        logger.info(f"  Saved {p.name:<38} ({p.stat().st_size / 1024:.1f} KB, {len(df)} rows)")

    _save(df_raw, "")              # ml_ready.csv
    _save(df_imp, "_imputed")      # ml_ready_imputed.csv
    _save(df_sc,  "_scaled")       # ml_ready_scaled.csv

    art_path = out / CONFIG['artifacts_file']
    with open(art_path, "wb") as fh:
        pickle.dump(artifacts, fh)
    logger.info(f"  Saved {art_path.name:<38} (transformation params)")

    rep_path = out / CONFIG['report_file']
    with open(rep_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(report_lines))
    logger.info(f"  Saved {rep_path.name}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_data_preparation():
    train_end = CONFIG['split_years']['train_end']
    val_end   = CONFIG['split_years']['val_end']
    start     = CONFIG['sample_start_year']
    report_lines = [
        "DATA PREPARATION REPORT",
        "=" * 70,
        f"Input : {CONFIG['input_path']}",
        f"Target: {CONFIG['target_col']}",
        f"Sample window : {start} – present",
        f"Split design  : train {start}–{train_end} | "
        f"val {train_end+1}–{val_end} | test {val_end+1}+",
        "                (chronological / time-series, per project_plan.pdf;",
        "                 Ajayi et al. 2022; Amini et al. 2021)",
        "IMPORTANT       model_training.py must re-fit transforms on TRAIN only",
    ]

    logger.info("=" * 70)
    logger.info("DATA PREPARATION PIPELINE")
    logger.info("=" * 70)

    # Step 1 — load
    logger.info("\n[STEP 1] Load data")
    logger.info("-" * 70)
    df = load_data(Path(CONFIG['input_path']))

    # Step 2 — leakage guard
    logger.info("\n[STEP 2] Leakage check")
    logger.info("-" * 70)
    leakage_check(ALL_FEATURES)

    missing_feats = [f for f in ALL_FEATURES
                     if f not in df.columns and f not in FEATURES_MACRO]
    if missing_feats:
        logger.warning(
            f"  {len(missing_feats)} defined features absent from input "
            f"(will be skipped): {missing_feats}"
        )

    # Step 3 — filter valid target
    logger.info("\n[STEP 3] Filter valid target rows")
    logger.info("-" * 70)
    df = filter_valid_target(df)

    # Step 3b — sample window filter (drop pre-1995 deals)
    logger.info("\n[STEP 3b] Sample window filter")
    logger.info("-" * 70)
    df = filter_sample_window(df)

    # Step 3c — chronological split assignment
    logger.info("\n[STEP 3c] Chronological split assignment")
    logger.info("-" * 70)
    df = assign_split(df)

    # Step 4 — load and merge pre-built macro proxies
    # Run fetch_macro_data.py once to build macro_monthly.csv.
    logger.info("\n[STEP 4] Load and merge macro economic proxies")
    logger.info("-" * 70)
    df, macro_status = load_and_merge_macro(df)

    # Recompute present features after macro columns are added
    present_feats = [f for f in ALL_FEATURES if f in df.columns]
    logger.info(f"  {len(present_feats)}/{len(ALL_FEATURES)} features present in data")

    # Save unwinsorised snapshot BEFORE step 5 for use by model_training.py.
    # This lets model_training.py re-fit winsorisation bounds on the training
    # split only, preventing full-sample lookahead contamination of ml_ready.csv.
    nowinsor_path = Path(CONFIG['output_dir']) / f"{CONFIG['output_stem']}_nowinsor.csv"
    df.to_csv(nowinsor_path, index=False)
    logger.info(
        f"  Saved {nowinsor_path.name:<38} "
        f"({nowinsor_path.stat().st_size / 1024:.1f} KB, {len(df)} rows)  "
        f"[pre-winsorisation — primary input for model_training.py]"
    )

    # Step 5 — winsorise
    logger.info("\n[STEP 5] Winsorise features (full-sample bounds — reference only)")
    logger.info("-" * 70)
    df, clip_bounds = winsorize_features(df, present_feats)

    # Step 6 — missing value report
    logger.info("\n[STEP 6] Missing value analysis")
    logger.info("-" * 70)
    low_cov_feats = missing_value_report(df, present_feats, report_lines)

    # Step 7 — median imputation (Lasso/Ridge variant)
    logger.info("\n[STEP 7] Median/mode imputation  (Lasso/Ridge variant)")
    logger.info("-" * 70)
    df_imp, fill_values = impute_missing(df, present_feats, low_cov_feats)

    # Step 8 — standard scaling (Lasso/Ridge variant)
    logger.info("\n[STEP 8] Z-score scaling  (Lasso/Ridge variant)")
    logger.info("-" * 70)
    df_sc, scale_params = scale_features(df_imp, present_feats, low_cov_feats)

    # Step 9 — correlation check
    logger.info("\n[STEP 9] Correlation check")
    logger.info("-" * 70)
    correlation_check(df, present_feats, report_lines)

    # Summary
    logger.info("\n[SUMMARY]")
    print_summary(df, present_feats, macro_status, report_lines)

    # Step 10 — save
    logger.info("\n[STEP 10] Save outputs")
    logger.info("-" * 70)

    # Split counts for artifact record
    split_counts = (
        df["split"].value_counts().to_dict()
        if "split" in df.columns else {}
    )

    artifacts = {
        # Feature metadata
        'feature_list':   present_feats,
        'feature_groups': {
            'cost':        [f for f in FEATURES_COST        if f in present_feats],
            'revenue':     [f for f in FEATURES_REVENUE     if f in present_feats],
            'operational': [f for f in FEATURES_OPERATIONAL if f in present_feats],
            'financial':   [f for f in FEATURES_FINANCIAL   if f in present_feats],
            'macro':       [f for f in FEATURES_MACRO       if f in present_feats],
        },
        'binary_features':       list(BINARY_FEATURES & set(present_feats)),
        'low_coverage_features': low_cov_feats,
        'missing_from_input':    missing_feats,
        # Target / date config
        'target_col':      CONFIG['target_col'],
        'date_col':        CONFIG['date_col'],
        'macro_lag_months': 1,   # fixed — lag is pre-applied in macro_monthly.csv by fetch_macro_data.py
        'macro_status':    macro_status,
        # Chronological split metadata
        'sample_start_year': CONFIG['sample_start_year'],
        'split_years':       CONFIG['split_years'],
        'split_col':         'split',
        'split_counts':      split_counts,
        # Transformation params (full-sample — re-fit on train fold in model_training.py)
        'clip_bounds':   clip_bounds,   # {feat: (lo, hi)}
        'fill_values':   fill_values,   # {feat: fill_val}
        'scale_params':  scale_params,  # {feat: (mean, std)}
        # Sample size
        'n_total': len(df),
    }

    save_outputs(df, df_imp, df_sc, artifacts, report_lines)

    logger.info("\n" + "=" * 70)
    logger.info("✓ DATA PREPARATION COMPLETE")
    logger.info("=" * 70)
    logger.info(
        "\nNext steps:\n"
        "  1. If macro columns are NaN, run fetch_macro_data.py first to build\n"
        "     macro_monthly.csv, then re-run data_preparation.py.\n"
        "  2. Proceed to model_training.py:\n"
        "       - Read ml_ready.csv; use df['split'] to subset train/val/test\n"
        "       - Re-fit winsorisation/imputation/scaling on TRAIN split only\n"
        "       - XGBoost baseline + Lasso/Ridge baselines\n"
        "       - SHAP analysis for synergy attribution\n"
        "  3. Inspect financial_altman_z_acquiror / financial_altman_z_target coverage\n"
        "     in prep_report.txt before proceeding to modelling."
    )

    return df, artifacts


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    prepared_df, prep_artifacts = run_data_preparation()
