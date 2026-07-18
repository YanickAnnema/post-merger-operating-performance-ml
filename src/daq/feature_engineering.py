"""
Feature Engineering — M&A Synergy Estimation
=============================================

Reads the intermediate CSV produced by current_daq.py and computes:
  1. Healy et al. (1992) CFROA target variable
       AB_t_CFROA, AB_t3_CFROA, Delta_CFROA_raw
  2. Industry CFROA adjustment (requires AcquirorSIC2 in CSV and cached panel)
       industry_CFROA_adjustment, synergy_healy1992, synergy_healy1992_w
  3. Derived ratio features by synergy channel
       cost_*, revenue_*, operational_*, financial_*
  4. Deal-characteristic dummies from screener columns
       log_deal_value, deal_tender_offer, deal_friendly, deal_cross_border,
       deal_stock_payment, deal_all_cash, deal_industry_4dig, deal_industry_2dig
  5. Modified Altman (1968) Z-score for acquiror and target (financial distress channel)
       financial_altman_z_acquiror, financial_altman_z_target
       Note: X2 uses total_equity as proxy for retained earnings (TR.F.RetainEarn not
       fetched). X4 denominator uses (total_assets − total_equity) as proxy for total
       liabilities. Both substitutions are documented and conservative.
  6. Deal year metadata column (used for chronological split in data_preparation.py)
       deal_year

Run after current_daq.py has produced the intermediate CSV.

Optimized for Spyder IDE (F5 execution).
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
import pickle

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DAQ_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "DAQ pipeline"
CACHE_DIR = PROJECT_ROOT / "code" / "cache"

CONFIG = {
    # Path to the CSV produced by current_daq.py
    'input_path':    DAQ_OUTPUT_DIR / "full_deal_level.csv",
    # Directory where current_daq.py wrote its cache (for financial panel)
    'cache_dir':     CACHE_DIR,
    # Output file (written to the same directory as the input)
    'output_suffix': "_features",   # appended before .csv
    # Winsorisation percentiles for synergy_healy1992_w
    'winsor_low':  0.01,
    'winsor_high': 0.99,
    # CFROA outlier filter: rows with |CFROA| above this are excluded from benchmarks
    'cfroa_outlier_limit': 2.0,
    # Minimum deals with non-NaN synergy to apply winsorisation
    'winsor_min_obs': 20,
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
# HELPER FUNCTIONS
# =============================================================================

def _scol(df: pd.DataFrame, col: str) -> pd.Series:
    """Return numeric Series for col, or all-NaN if col is absent."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Divide num by den; zero denominator → NaN (avoids eager division warnings)."""
    den_safe = den.where(den != 0, other=np.nan)
    return num / den_safe


def _fc(df: pd.DataFrame, keyword: str):
    """Return first column whose name contains keyword (case-insensitive), or None."""
    kw = keyword.lower().replace(" ", "").replace(".", "")
    return next(
        (c for c in df.columns if kw in str(c).lower().replace(" ", "").replace(".", "")),
        None,
    )

# =============================================================================
# STEP 1 — LOAD DATA
# =============================================================================

def load_input(path: Path) -> pd.DataFrame:
    logger.info(f"Loading intermediate CSV: {path}")
    df = pd.read_csv(path, low_memory=False)
    logger.info(f"  Shape: {df.shape}")
    for col in ["tA_fye", "tB_fye", "t3AB_fye"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

# =============================================================================
# STEP 2 — CFROA TARGET VARIABLE (Healy et al. 1992)
# =============================================================================

def compute_cfroa(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pre-deal and post-deal CFROA from proxy columns already in the CSV.

    Pre-deal combined:   (A_CFO + B_CFO) / (A_assets + B_assets)
    Post-deal combined:  AB_t3_CFO / AB_t3_assets   (acquiror PermID carries entity)

    NaN propagates if either party has no data — deal excluded from target variable.
    """
    logger.info("Computing CFROA target variable...")

    a_cfo    = _scol(df, "A_t_operating_cashflow")
    b_cfo    = _scol(df, "B_t_operating_cashflow")
    a_assets = _scol(df, "A_t_total_assets")
    b_assets = _scol(df, "B_t_total_assets")
    ab3_cfo    = _scol(df, "AB_t3_operating_cashflow")
    ab3_assets = _scol(df, "AB_t3_total_assets")

    df["AB_t_CFROA"]      = _safe_div(a_cfo + b_cfo, a_assets + b_assets)
    df["AB_t3_CFROA"]     = _safe_div(ab3_cfo, ab3_assets)
    df["Delta_CFROA_raw"] = df["AB_t3_CFROA"] - df["AB_t_CFROA"]

    n = df["Delta_CFROA_raw"].notna().sum()
    logger.info(f"  Delta_CFROA_raw calculable: {n}/{len(df)} deals")
    return df

# =============================================================================
# STEP 3 — INDUSTRY CFROA BENCHMARK (Healy industry adjustment)
# =============================================================================

def load_financial_panel(cache_dir: Path) -> pd.DataFrame:
    """
    Load the most recently modified financials pickle from the cache directory.
    Returns empty DataFrame if no cache file is found.
    """
    files = sorted(cache_dir.glob("financials_*.pkl"), key=lambda f: f.stat().st_mtime)
    if not files:
        logger.warning(f"No financials cache found in {cache_dir} — industry adjustment skipped")
        return pd.DataFrame()
    path = files[-1]
    logger.info(f"Loading financial panel from cache: {path.name}")
    with open(path, "rb") as fh:
        panel = pickle.load(fh)
    logger.info(f"  Panel shape: {panel.shape}")
    return panel


def compute_industry_benchmarks(panel: pd.DataFrame,
                                permid_to_sic: dict) -> pd.DataFrame:
    """
    Compute median CFROA by (2-digit SIC, fiscal year) from the financial panel.
    Uses deal participants only (the panel already contains only their entities).
    Returns DataFrame with columns [sic2, year, industry_median_cfroa].
    """
    if panel.empty or not permid_to_sic:
        return pd.DataFrame(columns=["sic2", "year", "industry_median_cfroa"])

    # Locate CFO and assets columns (case-insensitive)
    cfo_col    = next((c for c in panel.columns
                       if str(c).strip().lower() == "tr.f.netcashflowop"), None)
    assets_col = next((c for c in panel.columns
                       if str(c).strip().lower() == "tr.f.totassets"), None)

    if cfo_col is None or assets_col is None:
        logger.warning("CFO or assets column not found in panel — benchmark skipped")
        return pd.DataFrame(columns=["sic2", "year", "industry_median_cfroa"])

    wp = panel[["PermID", "FYE", cfo_col, assets_col]].copy()
    wp["_cfo"]    = pd.to_numeric(wp[cfo_col],    errors="coerce")
    wp["_assets"] = pd.to_numeric(wp[assets_col], errors="coerce")
    wp.loc[wp["_assets"] <= 0, "_assets"] = np.nan
    wp["_cfroa"] = wp["_cfo"] / wp["_assets"]
    wp = wp[wp["_cfroa"].between(-CONFIG['cfroa_outlier_limit'],
                                  CONFIG['cfroa_outlier_limit'])].copy()

    # Normalise PermID to string (matching key type in permid_to_sic)
    wp["_pid_str"] = wp["PermID"].astype(str).str.strip()
    wp["_sic"]     = wp["_pid_str"].map(permid_to_sic)
    wp = wp.dropna(subset=["_sic"])
    wp["_sic2"] = wp["_sic"].astype(str).str.strip().str[:2]
    wp["_year"] = pd.to_datetime(wp["FYE"], errors="coerce").dt.year
    wp = wp.dropna(subset=["_sic2", "_year", "_cfroa"])

    if wp.empty:
        logger.warning("No valid firm-years with SIC mapping — benchmark is empty")
        return pd.DataFrame(columns=["sic2", "year", "industry_median_cfroa"])

    benchmarks = (
        wp.groupby(["_sic2", "_year"])["_cfroa"]
        .median()
        .reset_index()
        .rename(columns={"_sic2": "sic2", "_year": "year",
                          "_cfroa": "industry_median_cfroa"})
    )
    logger.info(
        f"  Industry benchmarks: {len(benchmarks)} (SIC2 × year) cells, "
        f"{benchmarks['sic2'].nunique()} unique SIC2 groups"
    )
    return benchmarks


def apply_industry_adjustment(df: pd.DataFrame,
                               benchmarks: pd.DataFrame) -> pd.DataFrame:
    """
    Merge industry median CFROA at t and t+3, compute Healy adjustment, and
    produce synergy_healy1992 and winsorised synergy_healy1992_w.

    If benchmarks is empty or AcquirorSIC2 is absent/invalid, synergy_healy1992 falls
    back to Delta_CFROA_raw with a warning. AcquirorSIC2 is derived here from numeric
    SIC codes (TR.SICIndustryCode) in the CSV; description-string SIC codes are not used.
    """
    # AcquirorSIC2 is derived upstream in run_feature_engineering() from permid_to_sic
    # (numeric 2-digit string, e.g. "35"). The column written by current_daq.py from
    # description strings is overwritten there before this function is called.

    if (benchmarks.empty
            or "AcquirorSIC2" not in df.columns
            or df["AcquirorSIC2"].isna().all()):
        logger.warning(
            "Industry adjustment not applied — synergy_healy1992 = Delta_CFROA_raw. "
            "Ensure TR.SICIndustryCode is in the financial panel (check current_daq.py)."
        )
        df["industry_CFROA_adjustment"] = np.nan
        df["synergy_healy1992"] = df["Delta_CFROA_raw"]
    else:
        df["_year_t"]  = pd.to_datetime(df["tA_fye"],   errors="coerce").dt.year
        df["_year_t3"] = pd.to_datetime(df["t3AB_fye"], errors="coerce").dt.year

        # ── Diagnostic: coverage at each join stage ──────────────────────────
        n_total       = len(df)
        n_sic2        = df["AcquirorSIC2"].notna().sum()
        n_year_t      = df["_year_t"].notna().sum()
        n_year_t3     = df["_year_t3"].notna().sum()
        n_both_years  = (df["_year_t"].notna() & df["_year_t3"].notna()).sum()
        n_sic_year    = (df["AcquirorSIC2"].notna() & df["_year_t"].notna()).sum()

        logger.info(f"  Benchmark coverage diagnostics ({n_total} total deals):")
        logger.info(f"    Non-null AcquirorSIC2 : {n_sic2}/{n_total} ({100*n_sic2/n_total:.1f}%)")
        logger.info(f"    Non-null _year_t      : {n_year_t}/{n_total}")
        logger.info(f"    Non-null _year_t3     : {n_year_t3}/{n_total}")
        logger.info(f"    Both years non-null   : {n_both_years}/{n_total}")
        logger.info(f"    SIC2 × year_t valid   : {n_sic_year}/{n_total}")
        logger.info(f"    Benchmark cells (SIC2×year): {len(benchmarks)}  "
                    f"unique SIC2={benchmarks['sic2'].nunique()}")

        # Benchmark sparsity: distribution of firm-years per cell
        cell_sizes = benchmarks.groupby(["sic2", "year"]).size() if "sic2" in benchmarks.columns \
                     else pd.Series(dtype=int)
        # benchmarks here already is aggregated (one row per SIC2×year), so log cell count
        obs_per_sic2 = benchmarks.groupby("sic2").size()
        logger.info(f"    Years covered per SIC2: min={obs_per_sic2.min()}, "
                    f"median={obs_per_sic2.median():.0f}, max={obs_per_sic2.max()}")

        # Type/format check
        logger.info(f"  Key dtypes — AcquirorSIC2: {df['AcquirorSIC2'].dtype}, "
                    f"sic2: {benchmarks['sic2'].dtype}, "
                    f"_year_t: {df['_year_t'].dtype}, year: {benchmarks['year'].dtype}")
        logger.info(f"  AcquirorSIC2 sample: {df['AcquirorSIC2'].dropna().head(5).tolist()}")
        logger.info(f"  benchmarks.sic2 sample: {benchmarks['sic2'].head(5).tolist()}")

        df = df.merge(
            benchmarks.rename(columns={"industry_median_cfroa": "_ind_cfroa_t"}),
            left_on=["AcquirorSIC2", "_year_t"],
            right_on=["sic2", "year"],
            how="left",
        ).drop(columns=["sic2", "year"], errors="ignore")

        n_t_matched = df["_ind_cfroa_t"].notna().sum()
        logger.info(f"  Benchmark merge (t):  {n_t_matched}/{len(df)} rows matched")

        df = df.merge(
            benchmarks.rename(columns={"industry_median_cfroa": "_ind_cfroa_t3"}),
            left_on=["AcquirorSIC2", "_year_t3"],
            right_on=["sic2", "year"],
            how="left",
        ).drop(columns=["sic2", "year"], errors="ignore")

        n_t3_matched = df["_ind_cfroa_t3"].notna().sum()
        logger.info(f"  Benchmark merge (t+3): {n_t3_matched}/{len(df)} rows matched")

        # Breakdown: t only / t+3 only / both / neither
        has_t   = df["_ind_cfroa_t"].notna()
        has_t3  = df["_ind_cfroa_t3"].notna()
        logger.info(f"  Benchmark match breakdown: "
                    f"both={( has_t &  has_t3).sum()}  "
                    f"t_only={( has_t & ~has_t3).sum()}  "
                    f"t3_only={(~has_t &  has_t3).sum()}  "
                    f"neither={(~has_t & ~has_t3).sum()}")

        df["industry_CFROA_adjustment"] = df["_ind_cfroa_t3"] - df["_ind_cfroa_t"]
        df["synergy_healy1992"] = df["Delta_CFROA_raw"] - df["industry_CFROA_adjustment"]
        df = df.drop(columns=["_year_t", "_year_t3", "_ind_cfroa_t", "_ind_cfroa_t3"],
                     errors="ignore")

        n_adj = df["industry_CFROA_adjustment"].notna().sum()
        logger.info(f"  Industry adjustment applied to {n_adj}/{len(df)} deals")

    # Winsorise
    syn_valid = df["synergy_healy1992"].dropna()
    if len(syn_valid) >= CONFIG['winsor_min_obs']:
        p_low  = syn_valid.quantile(CONFIG['winsor_low'])
        p_high = syn_valid.quantile(CONFIG['winsor_high'])
        df["synergy_healy1992_w"] = df["synergy_healy1992"].clip(p_low, p_high)
        logger.info(
            f"  synergy_healy1992_w winsorised at "
            f"[{p_low:.4f}, {p_high:.4f}] over {len(syn_valid)} deals"
        )
    else:
        df["synergy_healy1992_w"] = df["synergy_healy1992"]
        logger.warning(
            f"  Only {len(syn_valid)} non-NaN synergy values — "
            "winsorisation skipped, raw value copied"
        )

    return df

# =============================================================================
# STEP 4 — DERIVED RATIO FEATURES
# =============================================================================

def compute_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived ratio features by synergy channel."""
    logger.info("Computing derived ratio features...")

    a_assets = _scol(df, "A_t_total_assets")
    b_assets = _scol(df, "B_t_total_assets")

    # COST CHANNEL
    df["cost_relative_asset_size"] = _safe_div(b_assets, a_assets)

    df["cost_ppe_intensity_diff"] = (
        _safe_div(_scol(df, "A_t_ppe_net"), a_assets)
        - _safe_div(_scol(df, "B_t_ppe_net"), b_assets)
    ).abs()

    df["cost_inventory_turnover_gap"] = (
        _safe_div(_scol(df, "A_t_revenue"), _scol(df, "A_t_inventory"))
        - _safe_div(_scol(df, "B_t_revenue"), _scol(df, "B_t_inventory"))
    ).abs()

    # employees field unresolved in Worldscope (see current_daq.py note) — will be NaN
    df["cost_sales_per_employee_gap"] = (
        _safe_div(_scol(df, "A_t_revenue"), _scol(df, "A_t_employees"))
        - _safe_div(_scol(df, "B_t_revenue"), _scol(df, "B_t_employees"))
    ).abs()

    df["cost_target_asset_utilization"] = _safe_div(_scol(df, "B_t_revenue"), b_assets)

    # REVENUE CHANNEL
    df["revenue_rd_intensity_diff"] = (
        _safe_div(_scol(df, "A_t_rd_expense"), a_assets)
        - _safe_div(_scol(df, "B_t_rd_expense"), b_assets)
    ).abs()

    df["revenue_capex_intensity_diff"] = (
        _safe_div(_scol(df, "A_t_capex"), a_assets)
        - _safe_div(_scol(df, "B_t_capex"), b_assets)
    ).abs()

    df["revenue_intangible_intensity_diff"] = (
        _safe_div(_scol(df, "A_t_intangible_assets"), a_assets)
        - _safe_div(_scol(df, "B_t_intangible_assets"), b_assets)
    ).abs()

    df["revenue_relative_size_sales"] = _safe_div(
        _scol(df, "B_t_revenue"), _scol(df, "A_t_revenue")
    )

    # OPERATIONAL CHANNEL
    df["operational_asset_turnover_gap"] = (
        _safe_div(_scol(df, "A_t_revenue"), a_assets)
        - _safe_div(_scol(df, "B_t_revenue"), b_assets)
    ).abs()

    # ROA from EBIT / TotAssets
    df["operational_roa_gap"] = (
        _safe_div(_scol(df, "A_t_ebit"), a_assets)
        - _safe_div(_scol(df, "B_t_ebit"), b_assets)
    ).abs()

    df["operational_acquiror_op_margin"] = _safe_div(
        _scol(df, "A_t_ebit"), _scol(df, "A_t_revenue")
    )

    df["operational_target_cf_margin"] = _safe_div(
        _scol(df, "B_t_operating_cashflow"), _scol(df, "B_t_revenue")
    )

    # FINANCIAL CHANNEL
    df["financial_leverage_gap"] = (
        _safe_div(_scol(df, "A_t_total_debt"), a_assets)
        - _safe_div(_scol(df, "B_t_total_debt"), b_assets)
    ).abs()

    df["financial_cash_ratio_diff"] = (
        _safe_div(_scol(df, "A_t_cash"), a_assets)
        - _safe_div(_scol(df, "B_t_cash"), b_assets)
    ).abs()

    df["financial_acquiror_cash_to_sales"] = _safe_div(
        _scol(df, "A_t_cash"), _scol(df, "A_t_revenue")
    )

    # Quick ratio: (current_assets - inventory) / current_liabilities
    df["financial_quick_ratio_acquiror"] = _safe_div(
        _scol(df, "A_t_current_assets") - _scol(df, "A_t_inventory"),
        _scol(df, "A_t_current_liabilities"),
    )
    df["financial_quick_ratio_target"] = _safe_div(
        _scol(df, "B_t_current_assets") - _scol(df, "B_t_inventory"),
        _scol(df, "B_t_current_liabilities"),
    )

    # Winsorise continuous ratio features at 1%/99% to remove extreme outliers.
    # These features can reach extreme values due to small denominators (e.g. near-zero
    # revenue, very small targets) or data scale mismatches. Winsorisation is applied
    # here, before the screener dummies, so the saved CSV already has clean values.
    # Only non-NaN observations are used to compute percentiles (per-variable bounds).
    _RATIO_FEATURES_TO_WINSORISE = [
        "cost_relative_asset_size",
        "cost_ppe_intensity_diff",
        "cost_inventory_turnover_gap",
        # cost_sales_per_employee_gap intentionally excluded: TR.F.Employees is not
        # resolved in the Worldscope pull for this sample and the column is always NaN.
        "cost_target_asset_utilization",
        "revenue_rd_intensity_diff",
        "revenue_capex_intensity_diff",
        "revenue_intangible_intensity_diff",
        "revenue_relative_size_sales",
        "operational_asset_turnover_gap",
        "operational_roa_gap",
        "operational_acquiror_op_margin",
        "operational_target_cf_margin",
        "financial_leverage_gap",
        "financial_cash_ratio_diff",
        "financial_acquiror_cash_to_sales",
        "financial_quick_ratio_acquiror",
        "financial_quick_ratio_target",
        # Altman Z-score composites (added in compute_altman_features)
        "financial_altman_z_acquiror",
        "financial_altman_z_target",
    ]
    lo = CONFIG.get("winsor_low",  0.01)
    hi = CONFIG.get("winsor_high", 0.99)
    for feat in _RATIO_FEATURES_TO_WINSORISE:
        if feat not in df.columns:
            continue
        s = df[feat]
        lb = s.quantile(lo)
        ub = s.quantile(hi)
        df[feat] = s.clip(lower=lb, upper=ub)
    logger.info(
        f"  Ratio features winsorised at [{lo:.0%}, {hi:.0%}]"
        f" ({len([f for f in _RATIO_FEATURES_TO_WINSORISE if f in df.columns])} features)"
    )

    logger.info("  ✓ Derived ratio features computed")
    return df

# =============================================================================
# STEP 4b — MODIFIED ALTMAN Z-SCORE  (financial distress channel)
# =============================================================================

def compute_altman_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a modified Altman (1968) Z-score for acquiror and target separately,
    using pre-deal (t) Worldscope columns already extracted by current_daq.py.

    Classical form:  Z = 1.2·X1 + 1.4·X2 + 3.3·X3 + 0.6·X4 + 1.0·X5
      X1 = working capital / total assets
      X2 = retained earnings / total assets   [PROXY: total_equity / total_assets]
      X3 = EBIT / total assets
      X4 = market capitalisation / total liabilities  [PROXY: market_cap / (TA − equity)]
      X5 = revenue / total assets

    Approximations documented here and in the thesis methodology:
      X2: TR.F.RetainEarn was not fetched in current_daq.py. total_equity
          (= book value of equity) is used as a conservative proxy; it overstates
          retained earnings for firms with large paid-in capital, so the Z-score
          will be slightly upward-biased for capital-intensive acquirors.
      X4: Total liabilities derived as (total_assets − total_equity). This is
          exact when no minority-interest or hybrid instruments are present, and
          a good approximation otherwise.

    If any required column is absent or produces a zero/negative denominator,
    the component is set to NaN; the composite score is NaN only if ALL five
    components are NaN. Partial availability yields a partial score.

    Coverage note: X4 requires market_cap, which has ~70–80 % coverage in the
    labelled sample. Expect the composite to have slightly lower coverage than
    pure balance-sheet features.

    Output columns (both winsorised later in compute_ratio_features loop):
      financial_altman_z_acquiror  — acquiror Z-score at t
      financial_altman_z_target    — target Z-score at t
    """
    logger.info("Computing modified Altman Z-scores (acquiror and target) ...")

    results = {}

    for prefix, label in [("A_t", "acquiror"), ("B_t", "target")]:
        ta    = _scol(df, f"{prefix}_total_assets")
        cur_a = _scol(df, f"{prefix}_current_assets")
        cur_l = _scol(df, f"{prefix}_current_liabilities")
        ebit  = _scol(df, f"{prefix}_ebit")
        rev   = _scol(df, f"{prefix}_revenue")
        mcap  = _scol(df, f"{prefix}_market_cap")
        eq    = _scol(df, f"{prefix}_total_equity")

        # X1: working capital / total assets
        x1 = _safe_div(cur_a - cur_l, ta)

        # X2 (approximation): total_equity / total_assets
        # Overstates retained earnings when paid-in capital is large.
        x2 = _safe_div(eq, ta)

        # X3: EBIT / total assets
        x3 = _safe_div(ebit, ta)

        # X4 (approximation): market_cap / (total_assets − total_equity)
        # Denominator is a proxy for total liabilities.
        total_liab = ta - eq
        x4 = _safe_div(mcap, total_liab.where(total_liab > 0, other=np.nan))

        # X5: revenue / total assets
        x5 = _safe_div(rev, ta)

        # Composite: weighted sum. NaN propagates per-component (not globally).
        # Use np.nansum on stacked array so partial coverage gives partial score.
        stacked = pd.DataFrame({
            "w1": 1.2 * x1,
            "w2": 1.4 * x2,
            "w3": 3.3 * x3,
            "w4": 0.6 * x4,
            "w5": 1.0 * x5,
        })
        # Score is NaN only if ALL components are NaN (i.e. no data at all for this firm)
        all_nan_mask = stacked.isna().all(axis=1)
        z_score = stacked.sum(axis=1, skipna=True)
        z_score[all_nan_mask] = np.nan

        col_name = f"financial_altman_z_{label}"
        df[col_name] = z_score

        n_valid = z_score.notna().sum()
        n_total = len(df)
        logger.info(
            f"  {col_name}: {n_valid}/{n_total} ({100*n_valid/n_total:.1f}%) non-NaN  "
            f"mean={z_score.mean():+.2f}  median={z_score.median():+.2f}"
        )
        results[label] = n_valid

    logger.info("  ✓ Altman Z-scores computed")
    return df


# =============================================================================
# STEP 5 — DEAL-CHARACTERISTIC DUMMIES
# =============================================================================

def compute_screener_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute binary and log features from deal screener columns.
    Uses .fillna(False) on isin() results to avoid pandas NA ambiguity with np.where.
    """
    logger.info("Computing screener-derived features...")

    # Log transaction value (cost channel)
    dv_col = _fc(df, "rankvalue") or _fc(df, "dealvalue")
    if dv_col:
        dv = pd.to_numeric(df[dv_col], errors="coerce")
        df["log_deal_value"] = np.where(dv > 0, np.log(dv), np.nan)

    # Tender offer dummy (cost channel)
    # Field: TR.MnAHasTenderAndMerger — keyword "hastender" matches the response column
    tender_col = _fc(df, "hastender")
    if tender_col:
        vals = df[tender_col].astype(str).str.upper().str.strip()
        is_missing = vals.isin(["NAN", ""]).fillna(False)
        is_tender  = vals.isin(["Y", "YES", "1", "TRUE"]).fillna(False)
        df["deal_tender_offer"] = np.where(is_missing, np.nan, is_tender.astype(float))

    # Friendly deal dummy (cost channel)
    attitude_col = _fc(df, "attitude")
    if attitude_col:
        vals = df[attitude_col].astype(str).str.upper().str.strip()
        is_missing  = vals.isin(["NAN", ""]).fillna(False)
        is_friendly = vals.isin(["FRIENDLY", "F"]).fillna(False)
        df["deal_friendly"] = np.where(is_missing, np.nan, is_friendly.astype(float))

    # Cross-border dummy (revenue channel)
    # Field: TR.MnAIsCrossBorder — keyword "crossborder" matches the response column
    cb_col = _fc(df, "crossborder")
    if cb_col:
        vals = df[cb_col].astype(str).str.upper().str.strip()
        is_missing = vals.isin(["NAN", ""]).fillna(False)
        is_cb      = vals.isin(["Y", "YES", "1", "TRUE"]).fillna(False)
        df["deal_cross_border"] = np.where(is_missing, np.nan, is_cb.astype(float))

    # Stock payment dummy: 1 if cash % < 50 (financial channel)
    pct_cash_col = _fc(df, "percentcash")
    if pct_cash_col:
        pct = pd.to_numeric(df[pct_cash_col], errors="coerce")
        df["deal_stock_payment"] = np.where(pct.notna(), (pct < 50.0).astype(float), np.nan)

    # All-cash dummy (financial channel).
    # Primary source: TR.MnAPaymentMethod string. Fallback: TR.MnAPercentCash >= 90.
    # LSEG sometimes omits TR.MnAPaymentMethod entirely when data is sparse, so the
    # percent-cash fallback ensures the column is always produced when deal_stock_payment
    # is already being built from TR.MnAPercentCash.
    payment_col  = _fc(df, "paymentmethod")
    pct_cash_for_all = _fc(df, "percentcash")  # reuse same pct cash col if found
    if payment_col:
        vals = df[payment_col].astype(str).str.upper().str.strip()
        is_missing  = vals.isin(["NAN", ""]).fillna(False)
        is_all_cash = vals.isin(["CASH", "C", "ALL CASH", "PURE CASH"]).fillna(False)
        df["deal_all_cash"] = np.where(is_missing, np.nan, is_all_cash.astype(float))
    elif pct_cash_for_all:
        # Fallback: >=90% cash consideration is treated as effectively all-cash.
        pct = pd.to_numeric(df[pct_cash_for_all], errors="coerce")
        df["deal_all_cash"] = np.where(pct.notna(), (pct >= 90.0).astype(float), np.nan)
        logger.info("  deal_all_cash: built from TR.MnAPercentCash >= 90 (payment method field absent)")

    # SIC-based industry relatedness dummies (cost / revenue / operational channels)
    # AcquirorSIC4 / TargetSIC4 are derived upstream in run_feature_engineering()
    # from permid_to_sic (numeric 4-digit integer). Falls through silently if absent
    # (i.e. permid_to_sic was empty because TR.SICIndustryCode is not yet in the panel).
    if "AcquirorSIC4" in df.columns and "TargetSIC4" in df.columns:
        acq4 = pd.to_numeric(df["AcquirorSIC4"], errors="coerce")
        tgt4 = pd.to_numeric(df["TargetSIC4"],   errors="coerce")
        valid = acq4.notna() & tgt4.notna()

        # 4-digit: exact match (e.g., 3571 == 3571)
        df["deal_industry_4dig"] = np.where(
            valid, (acq4 == tgt4).astype(float), np.nan
        )

        # 2-digit: compare first 2 digits of numeric code (e.g., "35" == "35")
        df["deal_industry_2dig"] = np.where(
            valid, (acq4.astype(str).str[:2] == tgt4.astype(str).str[:2]).astype(float), np.nan
        )
        logger.info(f"  deal_industry_4dig: {valid.sum()} valid pairs; "
                    f"same 4-dig={df['deal_industry_4dig'].sum():.0f}, "
                    f"same 2-dig={df['deal_industry_2dig'].sum():.0f}")

    logger.info("  ✓ Screener features computed")
    return df

# =============================================================================
# OUTPUT
# =============================================================================

def save_results(df: pd.DataFrame, input_path: Path) -> None:
    stem   = input_path.stem + CONFIG['output_suffix']
    output = input_path.parent / f"{stem}.csv"
    df.to_csv(output, index=False)
    logger.info(f"Saved: {output}  ({output.stat().st_size / 1024:.1f} KB)")


def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 60)
    total = len(df)

    for col, label in [
        ("Delta_CFROA_raw",          "Delta_CFROA_raw"),
        ("synergy_healy1992",        "synergy_healy1992 (raw)"),
        ("synergy_healy1992_w",      "synergy_healy1992_w (winsorised)"),
        ("industry_CFROA_adjustment","industry_CFROA_adjustment"),
    ]:
        if col in df.columns:
            s = df[col].dropna()
            if s.empty:
                print(f"  {label}: 0/{total} non-NaN")
            else:
                print(f"  {label}: {len(s)}/{total} "
                      f"mean={s.mean():.4f}  median={s.median():.4f}  std={s.std():.4f}")

    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

# =============================================================================
# MAIN
# =============================================================================

def run_feature_engineering():
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING PIPELINE")
    logger.info("=" * 60)

    # Step 1 — load
    df = load_input(CONFIG['input_path'])

    # Deal year metadata — used by data_preparation.py for sample-window filter
    # and chronological split assignment.  Not a predictive feature.
    if "DateEffective" in df.columns:
        df["deal_year"] = pd.to_datetime(
            df["DateEffective"], errors="coerce"
        ).dt.year.astype("Int64")   # nullable int preserves NaN
        n_yr = df["deal_year"].notna().sum()
        logger.info(f"  deal_year: {n_yr}/{len(df)} non-NaN  "
                    f"range {df['deal_year'].min()}–{df['deal_year'].max()}")
    else:
        logger.warning("  DateEffective absent — deal_year not added")

    # Step 2 — CFROA target variable
    logger.info("\n[STEP 2] CFROA target variable")
    logger.info("-" * 60)
    df = compute_cfroa(df)

    # Step 3 — industry CFROA adjustment
    logger.info("\n[STEP 3] Industry CFROA adjustment (Healy methodology)")
    logger.info("-" * 60)
    panel = load_financial_panel(CONFIG['cache_dir'])

    # Build PermID → numeric SIC mapping directly from the financial panel.
    # TR.SICIndustryCode is fetched alongside the Worldscope fields in current_daq.py
    # and returns a 4-digit integer code (e.g. 7389) for each entity.
    # Using the panel — rather than the TR.MnAAcquirorPriSic / TR.MnATargetPriSic
    # columns in the CSV — avoids the description-string problem: those screener
    # fields return text ("Business consulting services, nec"), so str[:2] would
    # give "Bu" instead of the correct 2-digit code "73".
    permid_to_sic = {}
    if not panel.empty:
        sic_panel_col = next(
            (c for c in panel.columns
             if str(c).lower().replace(".", "").replace("_", "") == "trsicindustrycode"),
            None,
        )
        if sic_panel_col:
            tmp = (
                panel[["PermID", sic_panel_col]]
                .dropna(subset=["PermID", sic_panel_col])
                .copy()
            )
            # Normalise PermID to canonical integer string (e.g. "4295905573").
            # Panel may store them as float64 → astype(str) gives "4295905573.0";
            # strip the ".0" so keys match the format used in the deals CSV.
            def _norm_permid(s: pd.Series) -> pd.Series:
                return (s.astype(str).str.strip()
                         .str.replace(r"\.0+$", "", regex=True))

            tmp["PermID"] = _norm_permid(tmp["PermID"])

            # Clean SIC values aggressively: reject strings like '', ' ', 'nan', 'None'.
            # The field may arrive as mixed string/numeric with dirty values.
            n_before = len(tmp)

            # Strip whitespace and reject known null-like strings
            sic_raw = tmp[sic_panel_col].astype(str).str.strip()
            sic_invalid_mask = sic_raw.isin(['', 'nan', 'none', 'NaN', 'NAN', 'None', 'NONE'])
            sic_raw = sic_raw[~sic_invalid_mask]

            # Coerce to numeric (remaining non-numeric strings → NaN)
            sic_numeric = pd.to_numeric(sic_raw, errors="coerce")

            # Log cleaning stats
            n_removed_strings = sic_invalid_mask.sum()
            n_coerce_nan = (sic_raw.notna() & sic_numeric.isna()).sum()
            logger.info(
                f"    SIC cleaning: {n_removed_strings} null-like strings, "
                f"{n_coerce_nan} non-numeric values → NaN"
            )

            # Rebuild tmp with cleaned SIC
            tmp[sic_panel_col] = sic_numeric
            tmp = tmp[tmp[sic_panel_col].notna()].copy()

            # One SIC code per entity; keep the most recent value
            tmp = tmp.drop_duplicates(subset=["PermID"], keep="last")

            # Store as integer SIC codes in the dict
            permid_to_sic = tmp.set_index("PermID")[sic_panel_col].astype(int).to_dict()

            logger.info(
                f"  PermID → SIC mapping: {len(permid_to_sic)} entries "
                f"({n_before} → {len(tmp)} after cleaning; source: {sic_panel_col})"
            )
        else:
            logger.warning(
                "  TR.SICIndustryCode not found in financial panel — "
                "industry benchmark will be skipped. "
                "Re-run current_daq.py to refresh the cached panel with the new field."
            )
    else:
        logger.warning("  Financial panel empty — industry benchmark skipped.")

    # Derive numeric SIC columns from permid_to_sic BEFORE benchmarks and screener features.
    # current_daq.py writes AcquirorSIC2 from description-string slices ("Bu", "Re", …),
    # which never match the "35"-style codes produced by compute_industry_benchmarks().
    # Always overwrite with numeric codes here so all downstream joins are consistent.
    if permid_to_sic:
        def _norm_pid(s: pd.Series) -> pd.Series:
            """Canonical PermID string: strip whitespace and trailing '.0'."""
            return s.astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)

        def _sic2(permid_series: pd.Series) -> pd.Series:
            """Map PermIDs → 2-digit SIC string (e.g., '35'). NaN when unmapped or non-numeric.
            Returns a Series aligned to the original input (no index shortening)."""
            normalized = _norm_pid(permid_series)
            mapped = normalized.map(permid_to_sic)
            # Coerce mapped values to numeric (handles any remaining dirty values)
            numeric = pd.to_numeric(mapped, errors="coerce")
            # Convert 4-digit int to 2-digit string, preserving NaN
            return numeric.apply(
                lambda x: str(int(x))[:2] if pd.notna(x) else np.nan,
                convert_dtype=False
            )

        def _sic4(permid_series: pd.Series) -> pd.Series:
            """Map PermIDs → 4-digit SIC integer. NaN when unmapped or non-numeric.
            Returns a Series aligned to the original input (no index shortening)."""
            normalized = _norm_pid(permid_series)
            mapped = normalized.map(permid_to_sic)
            # Coerce mapped values to numeric (handles any remaining dirty values)
            return pd.to_numeric(mapped, errors="coerce")

        if "AcquirorPermID" in df.columns:
            acq_sic2 = _sic2(df["AcquirorPermID"])
            acq_sic4 = _sic4(df["AcquirorPermID"])
            df["AcquirorSIC2"] = acq_sic2
            df["AcquirorSIC4"] = acq_sic4
            n_mapped = acq_sic2.notna().sum()
            logger.info(
                f"  AcquirorSIC2: {n_mapped}/{len(df)} mapped ({100*n_mapped/len(df):.1f}%), "
                f"sample: {acq_sic2.dropna().head(3).tolist()}"
            )
        if "TargetPermID" in df.columns:
            tgt_sic2 = _sic2(df["TargetPermID"])
            tgt_sic4 = _sic4(df["TargetPermID"])
            df["TargetSIC2"] = tgt_sic2
            df["TargetSIC4"] = tgt_sic4
            n_mapped = tgt_sic2.notna().sum()
            logger.info(
                f"  TargetSIC2:  {n_mapped}/{len(df)} mapped ({100*n_mapped/len(df):.1f}%)"
            )
    else:
        logger.warning(
            "  permid_to_sic is empty — AcquirorSIC2/TargetSIC2 not derived from numeric codes. "
            "Re-run current_daq.py to refresh the panel with TR.SICIndustryCode."
        )

    benchmarks = compute_industry_benchmarks(panel, permid_to_sic)
    df = apply_industry_adjustment(df, benchmarks)

    # Step 4 — derived ratio features
    logger.info("\n[STEP 4] Derived ratio features")
    logger.info("-" * 60)
    df = compute_ratio_features(df)

    # Step 4b — modified Altman Z-score (financial distress, financial channel)
    logger.info("\n[STEP 4b] Modified Altman Z-scores")
    logger.info("-" * 60)
    df = compute_altman_features(df)

    # Step 5 — screener dummies
    logger.info("\n[STEP 5] Screener-derived features")
    logger.info("-" * 60)
    df = compute_screener_features(df)

    # Save
    logger.info("\nSaving output...")
    save_results(df, CONFIG['input_path'])

    logger.info("\n" + "=" * 60)
    logger.info("✓ FEATURE ENGINEERING COMPLETE")
    logger.info("=" * 60)
    return df


if __name__ == "__main__":
    result_df = run_feature_engineering()
    print_summary(result_df)
