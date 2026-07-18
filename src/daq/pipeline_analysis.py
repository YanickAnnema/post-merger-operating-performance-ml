"""
Pipeline Analysis — M&A Synergy Estimation
===========================================

Runs the full pipeline (feature_engineering.py → data_preparation.py) and
produces a structured diagnostic report showing how the dataset changes at
each stage, broken down by deal year.

Stages:
  Stage 0 — Raw DAQ output          (full_deal_level.csv)
  Stage 1 — Feature-engineered      (full_deal_level_features.csv)
  Stage 2 — ML-ready (full sample)  (ml_ready.csv)

For each stage the report contains:
  - Overall shape (rows × columns)
  - Deal count and % survival relative to Stage 0, by effective year
  - FYE resolution rates (tA_fye, tB_fye, t3AB_fye)   [Stage 0 only]
  - Target variable (synergy_healy1992_w) coverage and descriptive stats
  - Per-channel feature completeness (% non-NaN)       [Stage 1+]
  - Macro variable coverage                            [Stage 2 only]
  - Full missing-value summary for key columns

Output:
  pipeline_analysis_report.txt   (same folder as the CSVs)

Optimised for Spyder IDE (F5 execution).
"""

import sys
import importlib
import logging
import traceback
from io import StringIO
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DAQ_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "DAQ pipeline"

CONFIG = {
    # DAQ pipeline output directory
    'output_dir':       DAQ_OUTPUT_DIR,

    # Input CSV names (must match current_daq.py / feature_engineering.py output)
    'stage0_csv':      "full_deal_level.csv",
    'stage1_csv':      "full_deal_level_features.csv",
    'stage2_csv':      "ml_ready.csv",



    # Where this script's code siblings live
    'code_dir':        Path(__file__).parent,

    # Report output file
    'report_filename': "pipeline_analysis_report.txt",

    # Whether to (re-)run feature_engineering.py and data_preparation.py
    # Set False to analyse already-existing CSVs without re-running the pipeline
    'run_feature_engineering': True,
    'run_data_preparation':    True,

    # Target variable produced by feature_engineering.py
    'target_col':      "synergy_healy1992_w",
    'date_col':        "DateEffective",

    # Feature groups (mirrors data_preparation.py FEATURES_* lists).
    # cost_sales_per_employee_gap removed: TR.F.Employees unresolved, always NaN.
    # financial_altman_z_* added: modified Altman (1968) scores from feature_engineering.py.
    'feature_channels': {
        "cost":        ["cost_relative_asset_size", "cost_ppe_intensity_diff",
                        "cost_inventory_turnover_gap",
                        # cost_sales_per_employee_gap excluded (always NaN)
                        "cost_target_asset_utilization", "log_deal_value",
                        "deal_tender_offer", "deal_friendly"],
        "revenue":     ["revenue_rd_intensity_diff", "revenue_capex_intensity_diff",
                        "revenue_intangible_intensity_diff",
                        "revenue_relative_size_sales", "deal_cross_border"],
        "operational": ["operational_asset_turnover_gap", "operational_roa_gap",
                        "operational_acquiror_op_margin",
                        "operational_target_cf_margin",
                        "deal_industry_4dig", "deal_industry_2dig"],
        "financial":   ["financial_leverage_gap", "financial_cash_ratio_diff",
                        "financial_acquiror_cash_to_sales",
                        "financial_quick_ratio_acquiror",
                        "financial_quick_ratio_target",
                        "deal_stock_payment", "deal_all_cash",
                        "financial_altman_z_acquiror",
                        "financial_altman_z_target"],
        "macro":       ["sp500_trailing_12m", "credit_spread_bbb_aaa"],
    },
}

# =============================================================================
# LOGGING — console only; report is written to file via ReportWriter
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# =============================================================================
# REPORT WRITER
# =============================================================================

class ReportWriter:
    """Accumulates report lines and writes a final txt file."""

    def __init__(self):
        self._buf = StringIO()

    def line(self, text: str = "") -> None:
        self._buf.write(text + "\n")

    def sep(self, char: str = "=", width: int = 72) -> None:
        self._buf.write(char * width + "\n")

    def header(self, text: str) -> None:
        self.sep()
        self.line(text)
        self.sep()

    def subheader(self, text: str) -> None:
        self.sep("-", 60)
        self.line(text)
        self.sep("-", 60)

    def save(self, path: Path) -> None:
        path.write_text(self._buf.getvalue(), encoding="utf-8")
        logger.info(f"Report saved: {path}")


# =============================================================================
# PIPELINE RUNNERS
# =============================================================================

def _add_code_dir_to_path() -> None:
    code_dir = str(CONFIG['code_dir'])
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)


def run_feature_engineering() -> bool:
    """Import and run feature_engineering.run_feature_engineering()."""
    if not CONFIG['run_feature_engineering']:
        logger.info("Skipping feature_engineering (run_feature_engineering=False)")
        return True
    logger.info("Running feature_engineering.py ...")
    _add_code_dir_to_path()
    try:
        import feature_engineering as fe
        importlib.reload(fe)          # reload in case it was already imported
        fe.run_feature_engineering()
        logger.info("✓ feature_engineering.py complete")
        return True
    except Exception:
        logger.error("feature_engineering.py FAILED:\n" + traceback.format_exc())
        return False


def run_data_preparation() -> bool:
    """Import and run data_preparation.run_data_preparation()."""
    if not CONFIG['run_data_preparation']:
        logger.info("Skipping data_preparation (run_data_preparation=False)")
        return True
    logger.info("Running data_preparation.py ...")
    _add_code_dir_to_path()
    try:
        import data_preparation as dp
        importlib.reload(dp)
        dp.run_data_preparation()
        logger.info("✓ data_preparation.py complete")
        return True
    except Exception:
        logger.error("data_preparation.py FAILED:\n" + traceback.format_exc())
        return False


# =============================================================================
# DATA LOADERS
# =============================================================================

def _load(csv_name: str) -> pd.DataFrame:
    path = CONFIG['output_dir'] / csv_name
    if not path.exists():
        logger.warning(f"File not found: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    date_col = CONFIG['date_col']
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    for fye_col in ["tA_fye", "tB_fye", "t3AB_fye"]:
        if fye_col in df.columns:
            df[fye_col] = pd.to_datetime(df[fye_col], errors="coerce")
    logger.info(f"Loaded {csv_name}: {df.shape}")
    return df


# =============================================================================
# ANALYSIS HELPERS
# =============================================================================

def _year_series(df: pd.DataFrame) -> pd.Series:
    """Return integer year column from DateEffective, or all-NaN if absent."""
    dc = CONFIG['date_col']
    if dc in df.columns:
        return df[dc].dt.year
    return pd.Series(np.nan, index=df.index)


def _pct(n: int, total: int) -> str:
    if total == 0:
        return "  n/a"
    return f"{100 * n / total:5.1f}%"


def _coverage(series: pd.Series) -> str:
    n_valid = series.notna().sum()
    n_total = len(series)
    return f"{n_valid:>6}/{n_total}  ({_pct(n_valid, n_total)})"


def _describe(series: pd.Series) -> str:
    s = series.dropna()
    if s.empty:
        return "  no data"
    return (f"  n={len(s):>5}  mean={s.mean():+.4f}  "
            f"median={s.median():+.4f}  std={s.std():.4f}  "
            f"[{s.min():+.4f}, {s.max():+.4f}]")


# =============================================================================
# PER-STAGE REPORT SECTIONS
# =============================================================================

def report_stage0(rw: ReportWriter, df: pd.DataFrame, label: str) -> None:
    rw.header(f"STAGE 0 — {label}")
    if df.empty:
        rw.line("  FILE NOT FOUND OR EMPTY — skip")
        return

    total = len(df)
    n_cols = len(df.columns)
    rw.line(f"  Rows: {total}    Columns: {n_cols}")
    rw.line()

    # --- Date range ---
    dc = CONFIG['date_col']
    if dc in df.columns:
        dates = df[dc].dropna()
        rw.line(f"  DateEffective range: {dates.min().date()} → {dates.max().date()}"
                f"  ({dates.notna().sum()} non-NaN)")
    rw.line()

    # --- Deal count by year ---
    rw.subheader("Deal count by effective year")
    year_col = _year_series(df)
    year_counts = year_col.value_counts().sort_index()
    rw.line(f"  {'Year':<8}  {'Deals':>6}  {'Share':>7}")
    rw.line(f"  {'-'*8}  {'-'*6}  {'-'*7}")
    for yr, cnt in year_counts.items():
        rw.line(f"  {int(yr):<8}  {cnt:>6}  {_pct(cnt, total):>7}")
    rw.line(f"  {'TOTAL':<8}  {total:>6}  {'100.0%':>7}")
    rw.line()

    # --- FYE resolution rates ---
    rw.subheader("FYE resolution rates (Worldscope join quality)")
    fye_cols = {
        "tA_fye":   "Acquiror pre-deal FYE resolved",
        "tB_fye":   "Target pre-deal FYE resolved",
        "t3AB_fye": "Combined post-deal FYE resolved",
    }
    for col, desc in fye_cols.items():
        if col in df.columns:
            rw.line(f"  {desc:<42}: {_coverage(df[col])}")
        else:
            rw.line(f"  {desc:<42}: column absent")

    # All three resolved simultaneously (max Healy label yield before FE)
    if all(c in df.columns for c in fye_cols):
        all_fye = df[list(fye_cols.keys())].notna().all(axis=1).sum()
        rw.line(f"  {'All 3 FYEs resolved':<42}: {all_fye:>6}/{total}  "
                f"({_pct(all_fye, total)})")
    rw.line()

    # --- FYE resolution by year ---
    rw.subheader("FYE resolution by effective year")
    if "tA_fye" in df.columns and "tB_fye" in df.columns and "t3AB_fye" in df.columns:
        df_tmp = df.copy()
        df_tmp["_year"] = year_col
        df_tmp["_all_fye"] = df[["tA_fye","tB_fye","t3AB_fye"]].notna().all(axis=1)
        rw.line(f"  {'Year':<8}  {'Deals':>6}  {'tA_fye':>8}  "
                f"{'tB_fye':>8}  {'t3AB_fye':>10}  {'All 3':>7}")
        rw.line(f"  {'-'*8}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*7}")
        for yr, grp in df_tmp.groupby("_year"):
            n = len(grp)
            a = grp["tA_fye"].notna().sum()
            b = grp["tB_fye"].notna().sum()
            c = grp["t3AB_fye"].notna().sum()
            all3 = grp["_all_fye"].sum()
            rw.line(f"  {int(yr):<8}  {n:>6}  "
                    f"{a:>4}/{n:<3}  {b:>4}/{n:<3}  "
                    f"{c:>4}/{n:<6}  {all3:>4}/{n:<3}")
    rw.line()

    # --- Key proxy field coverage ---
    rw.subheader("Key proxy field coverage (pre-deal)")
    proxy_check = [
        ("A_t_total_assets",        "Acquiror total assets (A_t)"),
        ("B_t_total_assets",        "Target total assets (B_t)"),
        ("A_t_operating_cashflow",  "Acquiror CFO (A_t)"),
        ("B_t_operating_cashflow",  "Target CFO (B_t)"),
        ("AB_t3_total_assets",      "Combined assets (AB_t3)"),
        ("AB_t3_operating_cashflow","Combined CFO (AB_t3)"),
        ("A_w_wacc",                "Acquiror WACC (StarMine)"),
        ("B_w_wacc",                "Target WACC (StarMine)"),
    ]
    for col, desc in proxy_check:
        if col in df.columns:
            rw.line(f"  {desc:<42}: {_coverage(pd.to_numeric(df[col], errors='coerce'))}")
        else:
            rw.line(f"  {desc:<42}: column absent")
    rw.line()


def report_stage1(rw: ReportWriter, df0: pd.DataFrame,
                  df: pd.DataFrame, label: str) -> None:
    rw.header(f"STAGE 1 — {label}")
    if df.empty:
        rw.line("  FILE NOT FOUND OR EMPTY — skip")
        return

    total    = len(df)
    total_s0 = len(df0) if not df0.empty else total
    rw.line(f"  Rows: {total}    Columns: {len(df.columns)}")
    rw.line(f"  Rows surviving from Stage 0: {_pct(total, total_s0)}")
    rw.line()

    # --- Target variable ---
    rw.subheader("Target variable")
    for col, desc in [
        ("Delta_CFROA_raw",          "Delta_CFROA_raw (unadjusted)"),
        ("synergy_healy1992",        "synergy_healy1992 (industry-adj)"),
        ("synergy_healy1992_w",      "synergy_healy1992_w (winsorised)"),
        ("industry_CFROA_adjustment","Industry CFROA adjustment"),
    ]:
        if col in df.columns:
            rw.line(f"  {desc}")
            rw.line(f"    Coverage : {_coverage(pd.to_numeric(df[col], errors='coerce'))}")
            rw.line(f"    Stats    :{_describe(pd.to_numeric(df[col], errors='coerce'))}")
        else:
            rw.line(f"  {desc}: column absent")
    rw.line()

    # --- Label yield by year ---
    rw.subheader("synergy_healy1992_w label yield by effective year")
    target = CONFIG['target_col']
    if target in df.columns:
        df_tmp = df.copy()
        df_tmp["_year"] = _year_series(df)
        df_tmp["_has_label"] = pd.to_numeric(df[target], errors="coerce").notna()
        rw.line(f"  {'Year':<8}  {'Deals':>6}  {'With label':>12}  "
                f"{'Yield%':>8}  {'Mean synergy':>14}  {'Median':>10}")
        rw.line(f"  {'-'*8}  {'-'*6}  {'-'*12}  {'-'*8}  {'-'*14}  {'-'*10}")
        for yr, grp in df_tmp.groupby("_year"):
            n       = len(grp)
            n_lab   = int(grp["_has_label"].sum())
            vals    = pd.to_numeric(grp[target], errors="coerce").dropna()
            mean_s  = f"{vals.mean():+.4f}" if not vals.empty else "   n/a"
            med_s   = f"{vals.median():+.4f}" if not vals.empty else "   n/a"
            rw.line(f"  {int(yr):<8}  {n:>6}  {n_lab:>12}  "
                    f"{_pct(n_lab, n):>8}  {mean_s:>14}  {med_s:>10}")
    else:
        rw.line(f"  Column '{target}' not found")
    rw.line()

    # --- Feature channel completeness ---
    rw.subheader("Feature completeness by synergy channel")
    channels = CONFIG['feature_channels']
    for ch_name, cols in channels.items():
        if ch_name == "macro":
            continue  # macro not in Stage 1
        present = [c for c in cols if c in df.columns]
        absent  = [c for c in cols if c not in df.columns]
        rw.line(f"  Channel: {ch_name.upper()}")
        for col in present:
            s = pd.to_numeric(df[col], errors="coerce")
            rw.line(f"    {col:<46}: {_coverage(s)}")
        for col in absent:
            rw.line(f"    {col:<46}: column absent")
        rw.line()

    # --- Derived ratio features descriptive stats ---
    rw.subheader("Derived ratio features — descriptive statistics")
    ratio_cols = [c for c in df.columns if c.startswith(
        ("cost_", "revenue_", "operational_", "financial_")
    ) and c not in {"deal_tender_offer", "deal_friendly", "deal_cross_border",
                    "deal_stock_payment", "deal_all_cash", "deal_industry_4dig",
                    "deal_industry_2dig"}]
    for col in sorted(ratio_cols):
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            rw.line(f"  {col:<46}: all NaN")
        else:
            rw.line(f"  {col:<46}:{_describe(pd.to_numeric(df[col], errors='coerce'))}")
    rw.line()


def report_stage2(rw: ReportWriter, df1: pd.DataFrame,
                  df: pd.DataFrame, label: str) -> None:
    rw.header(f"STAGE 2 — {label}")
    if df.empty:
        rw.line("  FILE NOT FOUND OR EMPTY — skip")
        return

    total    = len(df)
    total_s1 = len(df1) if not df1.empty else total
    rw.line(f"  Rows: {total}    Columns: {len(df.columns)}")
    rw.line(f"  Rows surviving from Stage 1 (label filter): "
            f"{total} / {total_s1}  ({_pct(total, total_s1)})")
    rw.line()

    # --- Target variable ---
    rw.subheader("Target variable after ML filter")
    target = CONFIG['target_col']
    if target in df.columns:
        s = pd.to_numeric(df[target], errors="coerce")
        rw.line(f"  {target}")
        rw.line(f"    Coverage : {_coverage(s)}")
        rw.line(f"    Stats    :{_describe(s)}")
    else:
        rw.line(f"  '{target}' not found in ml_ready.csv")
    rw.line()

    # --- Chronological split counts ---
    rw.subheader("Chronological split assignment")
    if "split" in df.columns:
        split_counts = df["split"].value_counts()
        rw.line(f"  {'Split':<8}  {'Deals':>6}  {'Share':>7}")
        rw.line(f"  {'-'*8}  {'-'*6}  {'-'*7}")
        for s_name in ["train", "val", "test"]:
            n = int(split_counts.get(s_name, 0))
            rw.line(f"  {s_name:<8}  {n:>6}  {_pct(n, total):>7}")
        n_none = int(df["split"].isna().sum()) + int((df["split"] == "nan").sum())
        if n_none:
            rw.line(f"  {'(unassigned)':<8}  {n_none:>6}  {_pct(n_none, total):>7}")

        # Year breakdown per split for quick sanity check
        if CONFIG['date_col'] in df.columns:
            rw.line()
            rw.line(f"  {'Year':<8}  {'Split':<8}  {'Deals':>6}")
            rw.line(f"  {'-'*8}  {'-'*8}  {'-'*6}")
            df_tmp = df.copy()
            df_tmp["_year"] = _year_series(df)
            for yr, grp in df_tmp.groupby("_year"):
                split_val = grp["split"].mode()
                s_name = split_val.iloc[0] if not split_val.empty else "n/a"
                rw.line(f"  {int(yr):<8}  {s_name:<8}  {len(grp):>6}")
    else:
        rw.line("  'split' column not found — run data_preparation.py to assign splits")
    rw.line()

    # --- Label yield by year (ML-ready) ---
    rw.subheader("Deals by effective year (ML-ready sample)")
    if target in df.columns:
        df_tmp = df.copy()
        df_tmp["_year"] = _year_series(df)
        rw.line(f"  {'Year':<8}  {'ML deals':>9}  {'Mean synergy':>14}  "
                f"{'Median':>10}  {'Std':>8}")
        rw.line(f"  {'-'*8}  {'-'*9}  {'-'*14}  {'-'*10}  {'-'*8}")
        for yr, grp in df_tmp.groupby("_year"):
            vals   = pd.to_numeric(grp[target], errors="coerce").dropna()
            n_lab  = len(vals)
            mean_s = f"{vals.mean():+.4f}" if not vals.empty else "   n/a"
            med_s  = f"{vals.median():+.4f}" if not vals.empty else "   n/a"
            std_s  = f"{vals.std():.4f}"  if len(vals) > 1   else "   n/a"
            rw.line(f"  {int(yr):<8}  {n_lab:>9}  {mean_s:>14}  "
                    f"{med_s:>10}  {std_s:>8}")
    rw.line()

    # --- Full feature coverage in ML-ready dataset ---
    rw.subheader("Feature completeness by synergy channel (ML-ready)")
    channels = CONFIG['feature_channels']
    for ch_name, cols in channels.items():
        present = [c for c in cols if c in df.columns]
        absent  = [c for c in cols if c not in df.columns]
        rw.line(f"  Channel: {ch_name.upper()}")
        for col in present:
            s = pd.to_numeric(df[col], errors="coerce")
            rw.line(f"    {col:<46}: {_coverage(s)}")
        for col in absent:
            rw.line(f"    {col:<46}: column absent")
        rw.line()

    # --- Highly correlated pairs (flag only) ---
    rw.subheader("High-correlation feature pairs (|r| > 0.85)")
    feat_cols = [c for c in df.columns
                 if any(c in ch for ch in channels.values())]
    feat_cols_present = [c for c in feat_cols if c in df.columns]
    if len(feat_cols_present) >= 2:
        num_feats = df[feat_cols_present].apply(
            pd.to_numeric, errors="coerce"
        )
        corr = num_feats.corr().abs()
        pairs = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                r = corr.iloc[i, j]
                if r > 0.85 and not np.isnan(r):
                    pairs.append((corr.columns[i], corr.columns[j], r))
        if pairs:
            rw.line(f"  {'Feature A':<46}  {'Feature B':<46}  {'|r|':>6}")
            rw.line(f"  {'-'*46}  {'-'*46}  {'-'*6}")
            for a, b, r in sorted(pairs, key=lambda x: -x[2]):
                rw.line(f"  {a:<46}  {b:<46}  {r:.3f}")
        else:
            rw.line("  No pairs with |r| > 0.85 found")
    else:
        rw.line("  Insufficient numeric features for correlation check")
    rw.line()


def report_flow_summary(rw: ReportWriter,
                        df0: pd.DataFrame,
                        df1: pd.DataFrame,
                        df2: pd.DataFrame) -> None:
    """Cross-stage attrition summary table."""
    rw.header("CROSS-STAGE ATTRITION SUMMARY")
    target = CONFIG['target_col']

    rows_0   = len(df0) if not df0.empty else 0
    rows_1   = len(df1) if not df1.empty else 0
    rows_2   = len(df2) if not df2.empty else 0
    label_1  = (df1[target].notna().sum()
                if (not df1.empty and target in df1.columns) else 0)

    rw.line(f"  {'Stage':<45}  {'Rows':>8}  {'% of Stage 0':>14}")
    rw.line(f"  {'-'*45}  {'-'*8}  {'-'*14}")
    rw.line(f"  {'Stage 0 — DAQ output (full_deal_level)':<45}  "
            f"{rows_0:>8}  {'100.0%':>14}")
    rw.line(f"  {'Stage 1 — Feature-engineered':<45}  "
            f"{rows_1:>8}  {_pct(rows_1, rows_0):>14}")
    rw.line(f"  {'Stage 1 — Rows with non-NaN target':<45}  "
            f"{label_1:>8}  {_pct(label_1, rows_0):>14}")
    rw.line(f"  {'Stage 2 — ML-ready (label-filtered)':<45}  "
            f"{rows_2:>8}  {_pct(rows_2, rows_0):>14}")
    rw.line()

    # Year-level cross-stage table
    rw.subheader("Attrition by effective year: Stage 0 → Stage 1 → Stage 2")
    dc = CONFIG['date_col']
    if (not df0.empty and dc in df0.columns
            and not df1.empty and dc in df1.columns):
        all_years = sorted(set(
            df0[dc].dt.year.dropna().astype(int).unique().tolist() +
            (df1[dc].dt.year.dropna().astype(int).unique().tolist()
             if not df1.empty and dc in df1.columns else [])
        ))
        rw.line(f"  {'Year':<8}  {'S0':>6}  {'S1':>6}  {'S1%':>6}  "
                f"{'Labeled':>8}  {'Lbl%':>6}  "
                f"{'S2':>6}  {'S2%':>6}")
        rw.line(f"  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*6}  "
                f"{'-'*8}  {'-'*6}  {'-'*6}  {'-'*6}")
        for yr in all_years:
            n0  = int((df0[dc].dt.year == yr).sum()) if not df0.empty else 0
            n1  = int((df1[dc].dt.year == yr).sum()) if not df1.empty else 0
            n1l = (int(((df1[dc].dt.year == yr) &
                        df1[target].notna()).sum())
                   if (not df1.empty and target in df1.columns) else 0)
            n2  = (int((df2[dc].dt.year == yr).sum())
                   if (not df2.empty and dc in df2.columns) else 0)
            rw.line(
                f"  {yr:<8}  {n0:>6}  {n1:>6}  {_pct(n1, n0):>6}  "
                f"{n1l:>8}  {_pct(n1l, n1) if n1 else '  n/a':>6}  "
                f"{n2:>6}  {_pct(n2, n0):>6}"
            )
    else:
        rw.line("  Date column unavailable in one or more stages — skipped")
    rw.line()


# =============================================================================
# MAIN
# =============================================================================

def run_analysis():
    rw = ReportWriter()

    # Report header
    rw.header("M&A SYNERGY ESTIMATION — PIPELINE ANALYSIS REPORT")
    rw.line(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    rw.line(f"  Date range configured in current_daq.py: "
            f"check CONFIG['ann_start'] / CONFIG['ann_end']")
    rw.line()

    # ------------------------------------------------------------------
    # Step 0: Load Stage 0 data (must exist; produced by current_daq.py)
    # ------------------------------------------------------------------
    logger.info("Loading Stage 0 CSV ...")
    df0 = _load(CONFIG['stage0_csv'])

    # ------------------------------------------------------------------
    # Step 1: Run feature_engineering.py
    # ------------------------------------------------------------------
    fe_ok = run_feature_engineering()
    df1 = _load(CONFIG['stage1_csv'])

    # ------------------------------------------------------------------
    # Step 2: Run data_preparation.py
    # ------------------------------------------------------------------
    dp_ok = run_data_preparation()
    df2 = _load(CONFIG['stage2_csv'])

    # ------------------------------------------------------------------
    # Build report
    # ------------------------------------------------------------------
    rw.line("Pipeline execution status:")
    rw.line(f"  feature_engineering.py : {'OK' if fe_ok else 'FAILED (see console)'}")
    rw.line(f"  data_preparation.py    : {'OK' if dp_ok else 'FAILED (see console)'}")
    rw.line()

    # Stage 0 — raw DAQ
    report_stage0(rw, df0, "RAW DAQ OUTPUT  (full_deal_level.csv)")

    # Stage 1 — feature-engineered
    report_stage1(rw, df0, df1,
                  "FEATURE-ENGINEERED  (full_deal_level_features.csv)")

    # Stage 2 — ML-ready
    report_stage2(rw, df1, df2, "ML-READY  (ml_ready.csv)")

    # Cross-stage attrition summary
    report_flow_summary(rw, df0, df1, df2)

    # ------------------------------------------------------------------
    # Save report
    # ------------------------------------------------------------------
    report_path = CONFIG['output_dir'] / CONFIG['report_filename']
    rw.save(report_path)

    # Console summary
    print("\n" + "=" * 72)
    print("PIPELINE ANALYSIS COMPLETE")
    print("=" * 72)
    print(f"  Stage 0 rows : {len(df0)}")
    print(f"  Stage 1 rows : {len(df1)}")
    print(f"  Stage 2 rows : {len(df2)}")
    target = CONFIG['target_col']
    if not df1.empty and target in df1.columns:
        n_lab = df1[target].notna().sum()
        print(f"  Labeled deals: {n_lab}  ({100*n_lab/max(len(df1),1):.1f}%)")
    print(f"\n  Report written to: {report_path}")
    print("=" * 72)

    return df0, df1, df2


# =============================================================================
# SPYDER ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    df_stage0, df_stage1, df_stage2 = run_analysis()
