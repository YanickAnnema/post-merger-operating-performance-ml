"""
M&A Deal Analysis - DAQ Pipeline

Target variable: Healy et al. (1992) industry-adjusted CFROA change.
  synergy_healy1992 = Delta_CFROA_raw - industry_CFROA_adjustment
  Delta_CFROA_raw   = CFROA_AB_t3 - CFROA_AB_t
  CFROA             = TR.F.NetCashFlowOp / TR.F.TotAssets
  AB_t baseline     = combined A + B pre-deal (both parties required)
  Industry adj.     = (industry_median_CFROA_t3 - industry_median_CFROA_t)
                      grouped by acquiror 2-digit SIC × fiscal year

NOTE on CFO field choice:
  TR.F.NetCashFlowOp is used for both the CFROA target variable and the
  operating_cashflow proxy feature. TR.F.OpCF was removed — it does not
  exist in Worldscope (confirmed via LSEG CodeCreator, March 2026).

NOTE on industry benchmark:
  Benchmarks are computed from deal participants only (acquirors and
  targets), not the full Worldscope universe. This is a coverage
  limitation: if the full universe is available it should be used instead.

Optimized for Spyder IDE (F5 execution).
"""

import hashlib
import pandas as pd
import numpy as np
import refinitiv.data as rd
from dateutil.relativedelta import relativedelta
import logging
from pathlib import Path
import pickle
from typing import Optional, List, Dict, Tuple
import time

# tqdm is optional — script runs without it
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # noqa: F811  # fallback: no progress bar
        return iterable

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Full sample window: 1993–2018.
    # 1993 is the earliest year with reliable Worldscope coverage for the US.
    # 2018 allows at least 3 years of post-merger data by the API cutoff (~2021+).
    # t+3 post-deal financials need to be available; deals after 2018 risk right-censoring.
    # Diagnostic on 2000-2005 produced only 36 labeled obs — too few for ML.
    # Expanding to 25 years with $250 M+ filter should yield ~500-1000 usable labels.
    'ann_start': "1985-01-01",
    'ann_end':   "2022-12-31",
    # Filter to public-only deals — Worldscope (TR.F.*) covers listed firms only.
    # Without this, ~93 % of screened deals have private targets → B_t_* always NaN
    # → Healy target always NaN → 0 usable labels. This is the primary fix.
    'filter_public_only': True,
    # Minimum deal value (USD millions). Raised from 50 to 250 to focus on deals
    # large enough to have reliable Worldscope coverage on both sides.
    # Literature precedent: Devos et al. (2009) use $100 M; Healy et al. (1992)
    # focus on top-50 deals. $250 M balances coverage quality vs. sample size.
    'min_deal_value_usd': 50,
    # 0.5 = 6-month batches for the screener.  1-year batches hit the LSEG
    # Gateway Time-out with 25 fields and the full corrected field names returning
    # real data (larger response payloads).  Reduce further to 0.25 if needed.
    'date_batch_years':   0.5,
    'batch_size':         200,     # WACC fetch batch size
    'financial_batch_size': 50,   # Worldscope financial fetch batch; smaller than screener because
                                  # each entity returns ~32 years × many fields → large payloads
    'output_dir':      Path(__file__).resolve().parents[2] / "outputs" / "DAQ pipeline",
    'output_filename': "full_deal_level.csv",
    'use_cache':  False,
    'cache_dir':  Path(__file__).resolve().parents[1] / "cache",
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
# FIELD DEFINITIONS
# =============================================================================

# Deal screener fields
FIELDS_DEALS = [
    "TR.MnASDCDealNumber",
    "TR.MnAAnnDate",
    "TR.MnAEffectiveDate",
    "TR.MnATarget",
    "TR.MnATargetMacroInd",
    "TR.MnATargetMidInd",
    "TR.MnATargetNation",
    "TR.MnAAcquiror",
    "TR.MnAAcquirorMacroInd",
    "TR.MnAAcquirorMidInd",
    "TR.MnAAcquirorNation",
    "TR.MnATargetPermId",
    "TR.MnAAcquirorPermId",
    "TR.MnARankValueIncNetDebt(Curn=USD,Scale=6)",
    "TR.MnAPctOfSharesOwnedPostMerger",
    "TR.MnAAcquirorPublicStatus",
    "TR.MnATargetPublicStatus",
    "TR.MnADealValue(Scale=6)",
    # Deal-characteristic fields for derived proxy features
    "TR.MnAHasTenderAndMerger",   # cost: tender offer dummy (confirmed field name)
    "TR.MnAAttitude",             # cost: friendly/hostile
    "TR.MnAIsCrossBorder",        # revenue: cross-border dummy (confirmed field name)
    "TR.MnATargetPriSic",         # industry relatedness dummies + benchmark grouping
    "TR.MnAAcquirorPriSic",       # industry relatedness dummies + benchmark grouping
    "TR.MnAPaymentMethod",        # financial: payment type string (used for deal_all_cash)
    "TR.MnAPercentCash",          # financial: cash % of consideration
]

# Worldscope (TR.F.*) proxy fields.
# All field codes verified against LSEG Workspace Data Item Browser (DIB, March 2026).
PROXY_FIELDS = {
    # ===== COST SYNERGIES =====
    "TR.F.TotAssets":   "total_assets",       # DIB: Total Assets
    "TR.F.SGATot":     "sga_expense",         # DIB: SG&A Expense (incl. selling costs;
    #   no standalone TR.F.Sell* expense exists in Worldscope — SG&A is the correct proxy)
    "TR.F.COGS":        "cogs",                # DIB: Cost of Goods Sold
    # NOTE: TR.CompanyNumEmploy confirmed in DIB but has NO Financial Period parameter —
    # it returns the most recently reported headcount (point-in-time, not FY-aligned).
    # Fetched separately outside this panel if needed; TR.CompanyNumEmployDate gives as-of date.
    "TR.F.InvntTot":     "inventory",           # DIB: Inventories - Total
    "TR.F.RcvblTot":     "accounts_receivable", # DIB: Accounts Receivable - Trade, Net
    "TR.F.PPENetTot":      "ppe_net",             # DIB: PP&E - Net
    "TR.F.AcctPble":     "accounts_payable",    # DIB: Accounts Payable

    # ===== REVENUE SYNERGIES =====
    "TR.F.TotRevenue":  "revenue",             # DIB: Total Revenue
    "TR.F.RnD":     "rd_expense",          # DIB: R&D Expense
    # FIXED: TR.F.CapEx does not exist. Confirmed correct code is TR.F.CAPEXTot.
    # Definition: all capex incl. fixed assets, software, intangibles > 1yr useful life.
    "TR.F.CAPEXTot":    "capex",
    # FIXED: TR.F.IntangAst does not exist. Two confirmed options:
    #   TR.F.IntangExclGoodwNetTot — net intangibles excluding goodwill (recommended:
    #     captures IP/patents/brand; excludes past-M&A goodwill artefact)
    #   TR.F.IntangGrossTot        — gross intangibles including goodwill
    "TR.F.IntangExclGoodwNetTot": "intangible_assets",
    # FIXED: TR.MktCapOrShsOut does not exist. Confirmed correct code is TR.F.MktCap.
    "TR.F.MktCap":      "market_cap",
    # REMOVED: TR.F.TotRevenue1YrGrowth does not exist in Worldscope.
    # CodeCreator returns only forward-looking estimate fields for "revenue growth".
    # Compute this feature manually after the financial panel is built:
    #   revenue_growth_1y = TotRevenue_t / TotRevenue_t-1 - 1
    # (requires two consecutive FY observations per entity)

    # ===== OPERATIONAL SYNERGIES =====
    "TR.F.EBIT":            "ebit",            # DIB: EBIT
    # FIXED: TR.F.OpCF does not exist. Confirmed correct code is TR.F.NetCashFlowOp,
    # which is the same field used for CFROA_CFO_FIELD (target variable).
    # Both proxy and target now use the same operating CF definition — intentional.
    "TR.F.NetCashFlowOp":   "operating_cashflow",
    "TR.F.NetIncAfterTax":  "net_income",      # DIB: Net Income After Tax
    "TR.F.EBITDA":          "ebitda",          # DIB: EBITDA
    # TR.EBITActValue and TR.GPMActValue removed — these are I/B/E/S consensus
    # estimate fields, not Worldscope TR.F.* fundamentals. They do not respond
    # to FRQ=FY / Scale=6 parameters and returned NaN for all entities.

    # ===== FINANCIAL SYNERGIES =====
    # FIXED: TR.F.DebtSTLT does not exist. Confirmed correct code is TR.F.DebtTot.
    # Definition: total borrowings incl. short-term and long-term debt.
    "TR.F.DebtTot":          "total_debt",
    # FIXED: TR.F.CashNearCashItem does not exist. Confirmed correct code is
    # TR.F.CashSTInvst (Cash & Short-Term Investments).
    "TR.F.CashSTInvst":      "cash",
    "TR.F.IntrExpnFinTot":           "interest_expense",  # DIB: Interest Expense
    "TR.F.TotShHoldEq":            "total_equity",       # DIB: Total Equity
    # FIXED: TR.F.BookValPerSh — confirmed correct code is TR.F.BookValuePerShr.
    # Definition: Shareholders Equity - Common / Common Shares Outstanding.
    "TR.F.BookValuePerShr":  "book_value_per_share",
    # FIXED: TR.F.DivCash does not exist. Two confirmed options:
    #   TR.F.DivPaidCashTotCF — total cash dividends paid (common + preferred) from CF stmt
    #   TR.F.DivComCashPaid   — common dividends only
    # TR.F.DivPaidCashTotCF preferred: broader coverage, sourced from cash flow statement.
    "TR.F.DivPaidCashTotCF": "dividends_cash",
    "TR.F.TotCurrAssets":    "current_assets",     # DIB: Total Current Assets
    "TR.F.TotCurrLiab":      "current_liabilities",# DIB: Total Current Liabilities
    # DIB-confirmed additions (March 2026):
    # NOL carryforward proxy: Worldscope does not expose raw NOL balances.
    # TR.F.DefTaxAssetLT (Deferred Tax Asset - Long-Term) is the best available
    # proxy because unrecognised NOLs generate long-term deferred tax assets.
    "TR.F.DefTaxAssetLT":    "deferred_tax_asset",  # DIB: Deferred Tax Asset - Long-Term
    # Effective tax rate (ETR) = income_tax_expense / pretax_income.
    # TR.F.ActualEffTaxRate is Japan-only (confirmed DIB). Compute ETR manually
    # at feature-engineering stage using these two income statement fields.
    "TR.F.IncTax":           "income_tax_expense",  # DIB: Income Tax Expense
    "TR.F.IncBefTax":        "pretax_income",        # DIB: Pre-Tax Income
    # NOTE: Acquirer pre-deal TSR uses TR.TotalReturn(From=..., To=...) — a market-data
    # field (not TR.F.*) that requires deal-specific date parameters. It cannot be batched
    # in this Worldscope panel and must be fetched in a separate per-deal loop.
}

# TR.F.NetCashFlowOp: used for both the Healy CFROA target variable and as the
# operating_cashflow proxy feature. TR.F.OpCF was removed — it does not exist in
# Worldscope (confirmed via DIB). Both uses now share this single field.
CFROA_CFO_FIELD = "TR.F.NetCashFlowOp"

# Note: CFROA_CFO_FIELD (TR.F.NetCashFlowOp) also appears in PROXY_FIELDS;
# dict.fromkeys() preserves insertion order and removes the duplicate.
#
# TR.SICIndustryCode is a company reference field (not a TR.F.* time-series).
# It returns a 4-digit numeric SIC code (e.g. 7389) for each entity and repeats
# the same value across all fiscal-year rows in the panel.  It is used by
# feature_engineering.py to build the permid_to_sic lookup and to derive
# AcquirorSIC2 / TargetSIC2 as proper 2-digit numeric codes rather than the
# first-two-characters of the description string returned by TR.MnAAcquirorPriSic.
FIELDS_FY = list(dict.fromkeys([
    "TR.F.PeriodEndDate",
    "TR.SICIndustryCode",   # numeric 4-digit SIC — used for industry benchmark grouping
    CFROA_CFO_FIELD,
    *PROXY_FIELDS.keys(),
]))

PARAMS_FY = {
    "FRQ":   "FY",
    "Curn":  "USD",
    "Scale": "6",   # string — consistent with LSEG API parameter conventions
}

# =============================================================================
# WACC FIELD DEFINITIONS
# =============================================================================
# StarMine WACC fields for the financial synergy channel.
# These are pre-computed LSEG StarMine cost-of-capital fields, NOT Worldscope
# TR.F.* fields, so they require a separate fetch (PARAMS_WACC below) because:
#   - They use 'Frq' not 'FRQ' (StarMine update schedule, not fiscal year)
#   - No Scale parameter applies (values are percentages / unitless weights)
#   - Coverage typically extends to ~1999; adequate for 2000–2005 window.
#
# Source notebooks: LSEG WACC_Model.ipynb, Peers_WACC_Comparison.ipynb
WACC_FIELDS = {
    "TR.WACC":              "wacc",               # Weighted average cost of capital (%)
    "TR.WACCBeta":          "wacc_beta",          # Levered beta used in WACC
    "TR.WACCCostofEquity":  "wacc_cost_equity",   # Cost of equity (%)
    "TR.WACCCostofDebt":    "wacc_cost_debt",     # Pre-tax cost of debt (%)
    "TR.WACCDebtWeight":    "wacc_debt_weight",   # Debt / (Debt + Equity)
    "TR.WACCEquityWeight":  "wacc_equity_weight", # Equity / (Debt + Equity)
    "TR.WACCTaxRate":       "wacc_tax_rate",      # Effective tax rate used in WACC (%)
}

# Frq=Y gives calendar-year-end snapshots; adequate for pre-deal annual matching.
# SDate/EDate are injected at fetch time.
PARAMS_WACC = {
    "Frq":  "Y",
    "Curn": "USD",
}

# build_role_join() looks up dict keys as column names in the panel.
# The WACC panel uses short names (not TR field names) as column headers
# because get_wacc_data() renames them before returning.
# So WACC_JOIN_MAP maps short_name → short_name (identity), not TR → short.
WACC_JOIN_MAP: Dict[str, str] = {v: v for v in WACC_FIELDS.values()}

# Maximum distance (days) between a t+3 target date and the closest available FYE.
# FYEs beyond this window are rejected as clearly wrong matches.
# 548 days ≈ ±18 months around the 3-year anniversary.
T3_MAX_DELTA_DAYS = 548

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def yyyymmdd(d: str) -> str:
    return pd.Timestamp(d).strftime("%Y%m%d")


def chunked(lst: List, n: int):
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


def date_range_batches(start_date: str, end_date: str, batch_years: float):
    """
    Yield (batch_start, batch_end) tuples covering [start_date, end_date].
    batch_years may be fractional (e.g. 0.5 = 6 months).
    Fractional values are rounded to the nearest whole number of months.
    """
    months = max(1, round(batch_years * 12))
    current = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    while current < end:
        batch_end = min(current + relativedelta(months=months), end)
        yield (current.strftime("%Y-%m-%d"), batch_end.strftime("%Y-%m-%d"))
        current = batch_end


def find_col(panel: pd.DataFrame, candidates_lower: List[str]) -> str:
    """Find a column by case-insensitive exact match. Raises if none found."""
    lookup = {str(c).strip().lower(): c for c in panel.columns}
    for name in candidates_lower:
        if name.lower() in lookup:
            return lookup[name.lower()]
    raise KeyError(
        f"None of the candidate columns found: {candidates_lower}. "
        f"Available: {list(panel.columns)}"
    )


def find_col_soft(panel: pd.DataFrame, tr_field: str) -> Optional[str]:
    """Return actual column name for tr_field (case-insensitive), or None if absent."""
    key = tr_field.strip().lower()
    return next((c for c in panel.columns if str(c).strip().lower() == key), None)


def find_cols(panel: pd.DataFrame, fields: List[str]) -> Dict[str, str]:
    """Map TR field names -> actual column names (case-insensitive). Raises on missing."""
    lookup = {str(c).strip().lower(): c for c in panel.columns}
    out = {}
    for f in fields:
        key = f.strip().lower()
        if key not in lookup:
            raise KeyError(f"Missing field '{f}'. Available: {list(panel.columns)[:50]}")
        out[f] = lookup[key]
    return out


def choose_first_available(panel: pd.DataFrame, tr_fields: List[str]) -> str:
    """Return first available field from a list of candidates. Raises if none found."""
    lookup = {str(c).strip().lower(): c for c in panel.columns}
    for f in tr_fields:
        if f.strip().lower() in lookup:
            return lookup[f.strip().lower()]
    raise KeyError(
        f"None of these fields were returned: {tr_fields}. "
        f"Available: {list(panel.columns)[:50]}"
    )

# =============================================================================
# FIELD VERIFICATION UTILITY
# =============================================================================

def verify_proxy_fields(test_ric: str = "MSFT.O") -> None:
    """
    Diagnostic utility — enumerate every available Worldscope TR.F.* field for
    a test RIC and cross-reference against PROXY_FIELDS.

    Run this once when adding new fields, after upgrading the LSEG library, or
    when a field silently returns null.  Not called by run_analysis(); invoke
    manually in Spyder before a full pipeline run.

    Pattern sourced from LSEG Discounted_Cash_Flow_Analysis.ipynb:
        rd.get_data(ric, ['TR.F.IncomeStatement.fieldname',
                          'TR.F.IncomeStatement.fielddescription',
                          'TR.F.IncomeStatement'], parameters={...})

    Args:
        test_ric: Any large-cap RIC with full Worldscope coverage.
                  Default 'MSFT.O'; swap to an actual deal participant if preferred.
    """
    logger.info("=" * 60)
    logger.info(f"PROXY FIELD VERIFICATION — Worldscope enumeration ({test_ric})")
    logger.info("=" * 60)

    safe_open_session()
    params = {
        "Period": "FY0",
        "reportingState": "Rsdt",  # standardised (Rsdt) for cross-country comparability
        "Scale": "6",
        "SORTA": "LISeq",          # sort by line-item sequence
    }

    available: dict = {}
    for stmt, label in [
        ("TR.F.IncomeStatement",   "Income Statement"),
        ("TR.F.CashflowStatement", "Cash Flow Statement"),
        ("TR.F.BalanceSheet",      "Balance Sheet"),
    ]:
        try:
            df = rd.get_data(
                universe=[test_ric],
                fields=[f"{stmt}.fieldname", f"{stmt}.fielddescription"],
                parameters=params,
                use_field_names_in_headers=True,
            )
            if df is not None and not df.empty:
                fn_col = next((c for c in df.columns if "fieldname"    in str(c).lower()), None)
                fd_col = next((c for c in df.columns if "description"  in str(c).lower()), None)
                if fn_col and fd_col:
                    for _, row in df.iterrows():
                        name = str(row[fn_col]).strip().upper()
                        desc = str(row[fd_col]).strip()
                        if name and name != "NAN":
                            available[name] = desc
                logger.info(f"  {label}: {len(df)} fields enumerated")
            else:
                logger.warning(f"  {label}: no data returned for {test_ric}")
        except Exception as e:
            logger.warning(f"  {label}: enumeration failed — {str(e)[:120]}")

    if not available:
        logger.warning("No Worldscope fields enumerated — cannot cross-reference PROXY_FIELDS")
        return

    logger.info(f"\nTotal Worldscope fields available via enumeration: {len(available)}")
    logger.info("\n--- PROXY_FIELDS cross-reference ---")
    confirmed, missing = [], []
    for tr_field in PROXY_FIELDS:
        if tr_field.strip().upper() in available:
            confirmed.append(tr_field)
        else:
            missing.append(tr_field)

    logger.info(f"✓ Confirmed in Worldscope ({len(confirmed)}): {confirmed}")
    if missing:
        logger.warning(
            f"⚠ NOT found in enumeration ({len(missing)}): {missing}\n"
            "  Possible reasons: market-data field (TR.* not TR.F.*), "
            "non-standard naming, or requires different fetch parameters.\n"
            "  Cross-check these in LSEG Workspace Data Item Browser (DIB)."
        )
    logger.info("=" * 60)


# =============================================================================
# CACHING FUNCTIONS
# =============================================================================

def _fields_hash() -> str:
    """Short hash of current PROXY_FIELDS + CFROA_CFO_FIELD for cache invalidation."""
    key_str = ",".join(sorted(PROXY_FIELDS.keys()) + [CFROA_CFO_FIELD])
    return hashlib.md5(key_str.encode()).hexdigest()[:8]


def _deals_fields_hash() -> str:
    """Short hash of FIELDS_DEALS for deal-cache invalidation when screener fields change."""
    key_str = ",".join(sorted(FIELDS_DEALS))
    return hashlib.md5(key_str.encode()).hexdigest()[:8]


def get_cache_path(name: str) -> Path:
    cache_dir = CONFIG['cache_dir']
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / f"{name}.pkl"


def load_from_cache(name: str) -> Optional[pd.DataFrame]:
    if not CONFIG['use_cache']:
        return None
    cache_file = get_cache_path(name)
    if cache_file.exists():
        logger.info(f"Loading from cache: {cache_file.name}")
        return pd.read_pickle(cache_file)
    return None


def save_to_cache(df: pd.DataFrame, name: str) -> None:
    if CONFIG['use_cache']:
        cache_file = get_cache_path(name)
        df.to_pickle(cache_file)
        logger.info(f"Saved to cache: {cache_file.name}")

# =============================================================================
# SESSION MANAGEMENT
# =============================================================================

# Module-level flag prevents double-open errors when re-running in Spyder
# (same kernel, multiple F5 presses). Reset manually via rd.close_session()
# if the session becomes stale.
_SESSION_OPEN = False


def safe_open_session():
    global _SESSION_OPEN
    if _SESSION_OPEN:
        logger.info("Session already open — skipping rd.open_session()")
        return True
    try:
        logger.info("Opening Refinitiv session...")
        rd.open_session()
        _SESSION_OPEN = True
        logger.info("✓ Session opened")
        return True
    except Exception as e:
        logger.error("=" * 60)
        logger.error("FAILED TO OPEN REFINITIV SESSION")
        logger.error("=" * 60)
        logger.error(f"Error: {str(e)[:200]}")
        logger.error("Troubleshooting: Is Refinitiv Workspace running and logged in?")
        logger.error("If stuck after a previous run: restart the Spyder kernel and retry.")
        logger.error("=" * 60)
        raise


def safe_close_session():
    global _SESSION_OPEN
    try:
        rd.close_session()
        _SESSION_OPEN = False
        logger.info("✓ Session closed")
    except Exception as e:
        _SESSION_OPEN = False
        logger.warning(f"Error closing session: {str(e)[:100]}")

# =============================================================================
# DATA RETRIEVAL FUNCTIONS
# =============================================================================

def retry_api_call(func, max_retries=4, delay=15, allow_partial_failure=False):
    """
    Retry wrapper with exponential backoff.

    Retriable conditions (server-side / transient):
      - "Network Error"            — connection lost
      - HTTP 500 / 502 / 503       — server error / bad gateway / unavailable
      - "timeout" or "time-out"    — LSEG Gateway Time-out (note: hyphenated in LSEG error text)
      - "gateway"                  — any gateway-level failure
      - "Error code -1"            — LSEG UDF execution failure (often recoverable with a retry)

    Non-retriable (data quality, never recoverable by retrying):
      - "Unable to collect data"
      - "specific identifier"

    Backoff: delay * (attempt + 1) seconds (linear, not exponential) to give
    the LSEG UDF engine time to recover from overload.
    """
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            error_msg = str(e)
            em_lower  = error_msg.lower()
            retriable = any([
                "Network Error" in error_msg,
                "500" in error_msg,
                "502" in error_msg,
                "503" in error_msg,
                "timeout"  in em_lower,          # generic timeout
                "time-out" in em_lower,          # LSEG Gateway Time-out (hyphenated)
                "gateway"  in em_lower,          # any gateway error
                "error code -1" in em_lower,     # LSEG UDF execution failure
            ])
            data_error = any([
                "Unable to collect data" in error_msg,
                "specific identifier"   in error_msg,
            ])
            if data_error:
                logger.warning(f"Data error (not retryable): {error_msg[:200]}")
                if allow_partial_failure:
                    return None
                raise
            if retriable and attempt < max_retries - 1:
                wait = delay * (attempt + 1)
                logger.warning(
                    f"Retriable error (attempt {attempt + 1}/{max_retries}, "
                    f"retrying in {wait}s): {error_msg[:200]}"
                )
                time.sleep(wait)
                continue
            logger.error(f"Non-retriable error: {error_msg[:200]}")
            if allow_partial_failure:
                return None
            raise
    return None


def build_screener(ann_start: str, ann_end: str) -> str:
    # IMPORTANT — keep this screener as minimal as possible.
    #
    # The LSEG official M&A screener example (DataQuery.ipynb) uses only two
    # conditions: IN(status) and a date filter.  Adding unsupported functions
    # (CONTAINS, field comparisons with parameters) to the screener string
    # causes the API to return an empty result with no error message.
    #
    # Business-logic filters (public status, min deal value) are applied
    # AFTER the API call in process_deals(), where we can inspect actual
    # values and log exactly what was removed.
    parts = [
        'SCREEN(U(IN(DEALS)/*UNV:DEALSMNA*/)',
        f'BETWEEN(TR.MnAEffectiveDate,{yyyymmdd(ann_start)},{yyyymmdd(ann_end)})/*dt:Date*/',
        'IN(TR.MnAStatus,"C")',
        'CURN=USD)',
    ]
    screener_str = ', '.join(parts)
    logger.info(f"Screener query: {screener_str}")
    return screener_str


def get_deal_data() -> pd.DataFrame:
    """Fetch M&A deal data with automatic date batching for large ranges."""
    filter_suffix = ""
    if CONFIG.get('filter_public_only', False):
        filter_suffix += "_public"
    if CONFIG.get('min_deal_value_usd', 0) > 0:
        filter_suffix += f"_min{CONFIG['min_deal_value_usd']}"
    cache_name = (
        f"deals_{yyyymmdd(CONFIG['ann_start'])}_{yyyymmdd(CONFIG['ann_end'])}"
        f"{filter_suffix}_{_deals_fields_hash()}"
    )
    cached = load_from_cache(cache_name)
    if cached is not None:
        return cached

    ann_start  = CONFIG['ann_start']
    ann_end    = CONFIG['ann_end']
    time_span  = (pd.Timestamp(ann_end) - pd.Timestamp(ann_start)).days / 365.25
    batch_years = CONFIG.get('date_batch_years', 1)

    safe_open_session()
    try:
        if time_span > batch_years:
            batch_months = max(1, round(batch_years * 12))
            logger.info(
                f"Date range: {time_span:.1f} yrs — fetching in "
                f"{batch_months}-month batches ({len(list(date_range_batches(ann_start, ann_end, batch_years)))} total)"
            )
            all_deals = []
            batches = list(date_range_batches(ann_start, ann_end, batch_years))
            for i, (bstart, bend) in enumerate(batches, 1):
                logger.info(f"Date batch {i}/{len(batches)}: {bstart} → {bend}")
                screener = build_screener(bstart, bend)

                def _fetch(s=screener):
                    return rd.get_data(
                        universe=[s],
                        fields=FIELDS_DEALS,
                        use_field_names_in_headers=True,
                    )

                # allow_partial_failure=True: a single timeout must not abort
                # the entire 25-year run.  Failed batches are logged and skipped.
                batch = retry_api_call(
                    _fetch,
                    max_retries=4,
                    delay=20,
                    allow_partial_failure=True,
                )
                if batch is not None and not batch.empty:
                    all_deals.append(batch)
                    logger.info(f"✓ {len(batch)} deals retrieved")
                else:
                    logger.warning(
                        f"⚠ Batch {i} ({bstart} → {bend}) returned no data — skipped. "
                        f"Re-run with a narrower date range to recover this window."
                    )
                if i < len(batches):
                    time.sleep(3)   # brief pause between batches to reduce server load

            if not all_deals:
                raise ValueError(f"No deals found in {ann_start} – {ann_end}")
            # Deduplicate on deal number only; do not use all-column drop_duplicates
            # as float NaN variation across batches can prevent true duplicate removal.
            deals = pd.concat(all_deals, ignore_index=True)
            deal_num_col = next(
                (c for c in deals.columns if 'sdcdealnumber' in str(c).lower().replace(' ', '')),
                None,
            )
            if deal_num_col:
                deals = deals.drop_duplicates(subset=[deal_num_col])
            else:
                deals = deals.drop_duplicates()
            n_batches_ok   = len(all_deals)
            n_batches_fail = len(batches) - n_batches_ok
            logger.info(
                f"✓ Total deals (deduplicated): {len(deals)}"
                f"  |  batches: {n_batches_ok}/{len(batches)} succeeded"
                f"{', ' + str(n_batches_fail) + ' failed (⚠ data gap — re-run those windows)' if n_batches_fail else ''}"
            )
            if n_batches_fail > 0:
                logger.warning(
                    f"  {n_batches_fail} batch(es) failed after retries. "
                    "The output CSV covers only the successful windows. "
                    "Consider re-running with date_batch_years: 0.25 to reduce batch size."
                )

        else:
            logger.info(f"Fetching deals {ann_start} → {ann_end}")
            screener = build_screener(ann_start, ann_end)

            def _fetch():
                return rd.get_data(
                    universe=[screener],
                    fields=FIELDS_DEALS,
                    use_field_names_in_headers=True,
                )

            deals = retry_api_call(_fetch, max_retries=3, delay=5)
            logger.info(f"✓ {len(deals)} deals retrieved")

    finally:
        safe_close_session()

    save_to_cache(deals, cache_name)
    return deals


def get_financial_data(permids: List[str]) -> pd.DataFrame:
    """Fetch Worldscope annual financials for a list of PermIDs."""
    # Cache key includes a hash of the current field set so stale cache is
    # not served after PROXY_FIELDS or CFROA_CFO_FIELD changes.
    cache_name = (
        f"financials_{yyyymmdd(CONFIG['ann_start'])}_{yyyymmdd(CONFIG['ann_end'])}"
        f"_{_fields_hash()}"
    )
    cached = load_from_cache(cache_name)
    if cached is not None:
        return cached

    sdate = (pd.Timestamp(CONFIG['ann_start']) - relativedelta(years=2)).strftime("%Y-%m-%d")
    edate = (pd.Timestamp(CONFIG['ann_end'])   + relativedelta(years=5)).strftime("%Y-%m-%d")
    params = {**PARAMS_FY, "SDate": sdate, "EDate": edate}

    logger.info(f"Fetching financials for {len(permids)} entities ({sdate} → {edate})")
    safe_open_session()
    try:
        panels = []
        fin_batch_sz  = CONFIG.get('financial_batch_size', 50)
        batch_count   = (len(permids) - 1) // fin_batch_sz + 1
        n_ok = n_fail = 0

        for i, batch in enumerate(chunked(permids, fin_batch_sz), 1):
            logger.info(f"Batch {i}/{batch_count} ({len(batch)} entities)")

            def _fetch(b=batch):
                return rd.get_data(
                    universe=b,
                    fields=FIELDS_FY,
                    parameters=params,
                    use_field_names_in_headers=True,
                )

            try:
                raw = retry_api_call(_fetch, max_retries=4, delay=20, allow_partial_failure=True)
                if raw is not None:
                    norm = normalize_history_to_panel(raw)
                    if not norm.empty:
                        panels.append(norm)
                        n_ok += 1
                        logger.info(f"✓ {len(norm)} rows")
                    else:
                        logger.warning(f"Batch {i} returned empty data")
                        n_fail += 1
                else:
                    logger.warning(f"Batch {i} failed — skipping")
                    n_fail += 1
                if i < batch_count:
                    time.sleep(1)
            except Exception as e:
                logger.error(f"Batch {i} error: {str(e)[:200]}")
                n_fail += 1

        logger.info(f"Batches: {n_ok} ok / {n_fail} failed of {batch_count}")

        if not panels:
            logger.error("No financial data retrieved — returning empty panel")
            return pd.DataFrame(columns=['PermID', 'FYE'])

        panel = (
            pd.concat(panels, ignore_index=True)
            .drop_duplicates(["PermID", "FYE"])
        )
        logger.info(f"✓ Panel: {len(panel)} rows, {panel['PermID'].nunique()} entities")

        # --- Coverage diagnostics: show non-NaN rates for key fields ---
        key_diag_fields = {
            "NetCashFlowOp (CFO)":  [c for c in panel.columns if "netcashflowop"  in c.lower() or "cfo" in c.lower()],
            "TotalAssets":          [c for c in panel.columns if "totalasset"      in c.lower() or "tot_asset" in c.lower()],
            "Revenue/Sales":        [c for c in panel.columns if "revenue" in c.lower() or "netsales" in c.lower()],
        }
        logger.info("  Panel field coverage (non-NaN / total rows):")
        for label, cols in key_diag_fields.items():
            if cols:
                nn = panel[cols[0]].notna().sum()
                logger.info(f"    {label:30s}: {nn:6d}/{len(panel)} ({100*nn/max(len(panel),1):.1f}%)")
            else:
                logger.info(f"    {label:30s}: column not found in panel")

        # Always write the financial panel to disk, regardless of use_cache.
        # feature_engineering.py loads the most-recent financials_*.pkl from the
        # cache directory to compute industry benchmarks.  If use_cache=False,
        # save_to_cache() is a no-op, which means feature_engineering.py silently
        # falls back to an empty benchmark and skips the Healy industry adjustment.
        cache_dir = Path(CONFIG['cache_dir'])
        cache_dir.mkdir(exist_ok=True)
        panel_path = cache_dir / f"{cache_name}.pkl"
        panel.to_pickle(panel_path)
        logger.info(f"  Financial panel written → {panel_path} (required by feature_engineering.py)")

        save_to_cache(panel, cache_name)   # no-op when use_cache=False; harmless when True
        return panel

    finally:
        safe_close_session()


def get_wacc_data(permids: List[str]) -> pd.DataFrame:
    """
    Fetch StarMine WACC panel for a list of PermIDs.

    Returns a tidy panel: PermID | WACC_date | wacc | wacc_beta | ...
    Columns mirror WACC_FIELDS values.

    Source: LSEG WACC_Model.ipynb / Peers_WACC_Comparison.ipynb — specifically
    the time-series WACC fetch pattern:
        rd.get_data(universe, ['TR.WACC.calcdate', 'TR.WACC'],
                    parameters={'SDate': ..., 'EDate': ..., 'Frq': 'Y'})

    NOTE: TR.WACC is a StarMine field, not Worldscope (TR.F.*).
    If StarMine coverage does not extend to the configured date window,
    this function returns an empty DataFrame and the pipeline continues
    normally — WACC columns will simply be absent from the output CSV.
    """
    cache_name = f"wacc_{yyyymmdd(CONFIG['ann_start'])}_{yyyymmdd(CONFIG['ann_end'])}"
    cached = load_from_cache(cache_name)
    if cached is not None:
        return cached

    sdate = (pd.Timestamp(CONFIG['ann_start']) - relativedelta(years=2)).strftime("%Y-%m-%d")
    edate = pd.Timestamp(CONFIG['ann_end']).strftime("%Y-%m-%d")
    params = {**PARAMS_WACC, "SDate": sdate, "EDate": edate}
    # TR.WACC.calcdate gives the date of each WACC snapshot in the time series
    fields = ["TR.WACC.calcdate"] + list(WACC_FIELDS.keys())

    logger.info(f"Fetching WACC data for {len(permids)} entities ({sdate} → {edate})")
    safe_open_session()
    try:
        panels = []
        batch_count = (len(permids) - 1) // CONFIG['batch_size'] + 1
        n_ok = n_fail = 0

        for i, batch in enumerate(chunked(permids, CONFIG['batch_size']), 1):
            logger.info(f"WACC batch {i}/{batch_count} ({len(batch)} entities)")

            def _fetch(b=batch):
                return rd.get_data(
                    universe=b,
                    fields=fields,
                    parameters=params,
                    use_field_names_in_headers=True,
                )

            try:
                raw = retry_api_call(_fetch, max_retries=2, delay=3, allow_partial_failure=True)
                if raw is not None and not raw.empty:
                    norm = raw.copy()
                    if "Instrument" in norm.columns:
                        norm = norm.rename(columns={"Instrument": "PermID"})
                    date_col = next(
                        (c for c in norm.columns if "calcdate" in str(c).lower()), None
                    )
                    if date_col is None:
                        logger.warning(f"WACC batch {i}: calcdate column absent — skipping")
                        n_fail += 1
                        continue
                    norm["WACC_date"] = pd.to_datetime(norm[date_col], errors="coerce")
                    norm["PermID"] = norm["PermID"].astype(str).str.strip()
                    norm = norm.dropna(subset=["PermID", "WACC_date"])
                    # Rename TR.WACC* columns to short names via WACC_FIELDS map
                    rename_map = {}
                    for tr_field, short_name in WACC_FIELDS.items():
                        actual = next(
                            (c for c in norm.columns
                             if str(c).strip().lower() == tr_field.lower()),
                            None,
                        )
                        if actual:
                            rename_map[actual] = short_name
                    norm = norm.rename(columns=rename_map)
                    keep = ["PermID", "WACC_date"] + [
                        v for v in WACC_FIELDS.values() if v in norm.columns
                    ]
                    panels.append(norm[keep])
                    n_ok += 1
                    logger.info(f"✓ {len(norm)} WACC rows")
                else:
                    n_fail += 1
                if i < batch_count:
                    time.sleep(1)
            except Exception as e:
                logger.warning(f"WACC batch {i} error: {str(e)[:150]}")
                n_fail += 1

        logger.info(f"WACC batches: {n_ok} ok / {n_fail} failed of {batch_count}")

        if not panels:
            logger.warning(
                "No WACC data retrieved — StarMine coverage may not extend to "
                f"{sdate}–{edate}.  WACC columns will be absent from output CSV."
            )
            return pd.DataFrame(columns=["PermID", "WACC_date"])

        wacc_panel = (
            pd.concat(panels, ignore_index=True)
            .drop_duplicates(["PermID", "WACC_date"])
        )
        logger.info(
            f"✓ WACC panel: {len(wacc_panel)} rows, "
            f"{wacc_panel['PermID'].nunique()} entities"
        )
        save_to_cache(wacc_panel, cache_name)
        return wacc_panel

    finally:
        safe_close_session()

# =============================================================================
# DATA PROCESSING FUNCTIONS
# =============================================================================

def normalize_history_to_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize rd.get_data() history output to: PermID, FYE, <fields...>.
    Handles both tidy (Instrument column) and MultiIndex column formats.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    dfx = df.copy()

    # Case 1: tidy — 'Instrument' column present
    if "Instrument" in dfx.columns:
        dfx = dfx.rename(columns={"Instrument": "PermID"})
        fye_col = next(
            (c for c in dfx.columns if str(c).strip().lower() == "tr.f.periodenddate"),
            None,
        )
        if fye_col is None:
            raise ValueError(
                f"Cannot find TR.F.PeriodEndDate in columns: {list(dfx.columns)}"
            )
        dfx["FYE"] = pd.to_datetime(dfx[fye_col], errors="coerce")
        dfx["PermID"] = dfx["PermID"].astype(str).str.strip()
        dfx = dfx.dropna(subset=["PermID", "FYE"]).copy()
        dfx = dfx.sort_values(["PermID", "FYE"]).drop_duplicates(
            ["PermID", "FYE"], keep="last"
        )
        cols = ["PermID", "FYE"] + [c for c in dfx.columns if c not in {"PermID", "FYE"}]
        return dfx[cols].reset_index(drop=True)

    # Case 2: MultiIndex tuple columns
    if len(dfx.columns) > 0 and isinstance(dfx.columns[0], tuple):
        tmp = dfx.copy()
        tmp.index = pd.to_datetime(tmp.index, errors="coerce")
        tmp = tmp[~tmp.index.isna()].copy()
        # pandas ≥ 2.1 changed stack() defaults and emits a FutureWarning
        # without future_stack=True.  Use it where available; fall back
        # silently for pandas < 2.1 (TypeError) and pandas ≥ 3.0 (removed).
        try:
            stacked = tmp.stack(level=0, future_stack=True).reset_index()
        except TypeError:
            stacked = tmp.stack(level=0).reset_index()
        stacked = stacked.rename(columns={"level_0": "Date", "level_1": "PermID"})
        if "Period End Date" in stacked.columns:
            stacked["FYE"] = pd.to_datetime(stacked["Period End Date"], errors="coerce")
        elif "TR.F.PeriodEndDate" in stacked.columns:
            stacked["FYE"] = pd.to_datetime(stacked["TR.F.PeriodEndDate"], errors="coerce")
        else:
            stacked["FYE"] = pd.to_datetime(stacked["Date"], errors="coerce")
        stacked["PermID"] = stacked["PermID"].astype(str).str.strip()
        stacked = stacked.dropna(subset=["PermID", "FYE"]).copy()
        stacked = stacked.sort_values(["PermID", "FYE"]).drop_duplicates(
            ["PermID", "FYE"], keep="last"
        )
        stacked = stacked.drop(columns=["Date"], errors="ignore")
        cols = ["PermID", "FYE"] + [c for c in stacked.columns if c not in {"PermID", "FYE"}]
        return stacked[cols].reset_index(drop=True)

    raise ValueError(f"Unexpected history shape. Columns={list(df.columns)[:30]}")


def process_deals(deals: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize deal data and extract SIC codes for industry benchmarking."""
    logger.info("Processing deal data...")

    required_upper = [
        "TR.MNASDCDEALNUMBER",
        "TR.MNAEFFECTIVEDATE",
        "TR.MNAACQUIRORPERMID",
        "TR.MNATARGETPERMID",
    ]
    col_lookup = {str(c).strip().upper(): c for c in deals.columns}
    missing = [c for c in required_upper if c not in col_lookup]
    if missing:
        raise ValueError(
            f"'deals' missing required columns: {missing}. Found: {list(deals.columns)}"
        )

    df = deals.copy()
    df["deal_id"]        = df[col_lookup["TR.MNASDCDEALNUMBER"]].astype(str).str.strip()
    df["DateEffective"]  = pd.to_datetime(df[col_lookup["TR.MNAEFFECTIVEDATE"]], errors="coerce")
    df["AcquirorPermID"] = df[col_lookup["TR.MNAACQUIRORPERMID"]].astype(str).str.strip()
    df["TargetPermID"]   = df[col_lookup["TR.MNATARGETPERMID"]].astype(str).str.strip()
    df = df.dropna(subset=["deal_id", "DateEffective", "AcquirorPermID", "TargetPermID"]).copy()

    # .astype(str) above converts float NaN to the literal string "nan", which
    # pd.dropna() does not catch.  Filter these out explicitly so they cannot
    # contaminate the entity universe sent to get_financial_data().
    _INVALID_PERMID = {"nan", "none", ""}
    perm_ok = (
        ~df["AcquirorPermID"].str.lower().isin(_INVALID_PERMID)
        & ~df["TargetPermID"].str.lower().isin(_INVALID_PERMID)
    )
    n_invalid = (~perm_ok).sum()
    if n_invalid:
        logger.warning(
            f"  Dropped {n_invalid} deal(s) where PermID is a missing-value string "
            f"('nan', 'none', or '') — these would have been sent as literal identifiers."
        )
    df = df[perm_ok].copy()

    # Reset index immediately after dropna so df and keys always share the same
    # sequential 0…N index.  Without this, the public-status boolean mask built
    # from df carries df's (sparse) original labels, which misaligns with keys'
    # reset_index labels and causes KeyError → empty output.
    df = df.reset_index(drop=True)

    # CombinedPermID: acquiror PermID carries the post-merger entity in Worldscope.
    # Standard assumption — verify for reverse mergers if coverage seems low at t+3.
    df["CombinedPermID"] = df["AcquirorPermID"]

    keys = df[["deal_id", "DateEffective", "TargetPermID",
               "AcquirorPermID", "CombinedPermID"]].copy().reset_index(drop=True)

    # Extract SIC codes for industry benchmark and relatedness dummies.
    # Column names depend on Refinitiv response; search case-insensitively.
    def _sic_col(keyword):
        kw = keyword.lower().replace(" ", "").replace(".", "")
        return next(
            (c for c in df.columns if kw in str(c).lower().replace(" ", "").replace(".", "")),
            None,
        )

    acq_sic_col = _sic_col("acquirorprisic")
    tgt_sic_col = _sic_col("targetprisic")

    if acq_sic_col:
        raw = df[acq_sic_col].astype(str).str.strip()
        keys["AcquirorSIC"]  = raw.replace({"nan": np.nan, "": np.nan})
        keys["AcquirorSIC2"] = keys["AcquirorSIC"].str[:2]
    else:
        logger.warning(
            "TR.MnAAcquirorPriSic not found — industry adjustment and SIC "
            "dummies will be skipped"
        )

    if tgt_sic_col:
        raw = df[tgt_sic_col].astype(str).str.strip()
        keys["TargetSIC"]  = raw.replace({"nan": np.nan, "": np.nan})
        keys["TargetSIC2"] = keys["TargetSIC"].str[:2]

    # ── Public-only filter (post-fetch) ──────────────────────────────────────
    # Applied here rather than in build_screener() because SCREEN IN() requires
    # an exact string match and the stored value varies ("Public", "Public (Listed)",
    # "Listed", …). We first log all distinct values so you can verify the match.
    acq_pub_col = next(
        (c for c in df.columns
         if "acquirorpublicstatus" in str(c).lower().replace(" ", "").replace(".", "")),
        None,
    )
    tgt_pub_col = next(
        (c for c in df.columns
         if "targetpublicstatus" in str(c).lower().replace(" ", "").replace(".", "")),
        None,
    )

    if acq_pub_col:
        acq_vals = df[acq_pub_col].value_counts(dropna=False).to_dict()
        logger.info(f"Acquiror public status values: {acq_vals}")
    else:
        logger.warning("TR.MnAAcquirorPublicStatus column not found in deal data")

    if tgt_pub_col:
        tgt_vals = df[tgt_pub_col].value_counts(dropna=False).to_dict()
        logger.info(f"Target public status values:   {tgt_vals}")
    else:
        logger.warning("TR.MnATargetPublicStatus column not found in deal data")

    if CONFIG.get('filter_public_only', False):
        before = len(df)
        if acq_pub_col and tgt_pub_col:
            # Accept any value that contains "public" (case-insensitive) so that
            # "Public", "Public (Listed)", "PUBLIC", etc. all pass.
            acq_mask = df[acq_pub_col].astype(str).str.lower().str.contains("public", na=False)
            tgt_mask = df[tgt_pub_col].astype(str).str.lower().str.contains("public", na=False)
            df   = df[acq_mask & tgt_mask].copy()
            keys = keys.loc[df.index].copy()
            after = len(df)
            logger.info(
                f"Public-only filter applied: {before} → {after} deals "
                f"({before - after} removed)"
            )
            if after == 0:
                logger.error(
                    "Public filter removed ALL deals.\n"
                    "  Check the status values logged above — none contain 'public'.\n"
                    "  Possible fix: widen the match string or set filter_public_only=False\n"
                    "  and inspect the raw values manually before re-enabling."
                )
        else:
            logger.warning(
                "filter_public_only=True but status columns not found — filter skipped"
            )
    # ─────────────────────────────────────────────────────────────────────────

    # ── Min deal value filter (post-fetch) ───────────────────────────────────
    # Moved here from build_screener(): deal value comparisons with field
    # parameters are not part of the official LSEG M&A screener syntax and
    # caused the screener to return an empty result silently.
    min_val = CONFIG.get('min_deal_value_usd', 0)
    if min_val > 0:
        dv_col = next(
            (c for c in df.columns
             if "dealvalue" in str(c).lower().replace(" ", "").replace(".", "")),
            None,
        )
        if dv_col:
            dv = pd.to_numeric(df[dv_col], errors="coerce")
            before = len(df)
            mask = dv >= min_val
            df   = df[mask].reset_index(drop=True)
            keys = keys[mask.values].reset_index(drop=True)
            logger.info(
                f"Min deal value filter (>=${min_val}M): {before} → {len(df)} deals "
                f"({before - len(df)} removed)"
            )
        else:
            logger.warning(
                f"min_deal_value_usd={min_val} set but deal value column not found — "
                "filter skipped"
            )
    # ─────────────────────────────────────────────────────────────────────────

    validate_deal_data(df)
    logger.info(f"Processed {len(df)} valid deals")
    return df, keys


def validate_deal_data(df: pd.DataFrame) -> None:
    logger.info("\n=== Deal Data Quality ===")
    total = len(df)
    logger.info(f"Total deals: {total}")
    logger.info(
        f"Date range: {df['DateEffective'].min().date()} – {df['DateEffective'].max().date()}"
    )
    dupes = df["deal_id"].duplicated().sum()
    if dupes > 0:
        logger.warning(f"Duplicate deal IDs: {dupes}")
    dv_cols = [c for c in df.columns if "dealvalue" in str(c).lower().replace(" ", "")]
    if dv_cols:
        dv = pd.to_numeric(df[dv_cols[0]], errors="coerce")
        n_valid = dv.notna().sum()
        logger.info(f"Deals with deal value: {n_valid}/{total}")
        if n_valid > 0:
            logger.info(f"Median deal value: ${dv.median():.1f}M")
    nat_cols = [c for c in df.columns if "acquirornation" in str(c).lower().replace(" ", "")]
    if nat_cols:
        logger.info("Top 5 acquiror nations:")
        for nation, count in df[nat_cols[0]].value_counts().head().items():
            logger.info(f"  {nation}: {count}")

# =============================================================================
# FYE MATCHING FUNCTIONS
# =============================================================================

def compute_fyes_vectorized(panel: pd.DataFrame, keys: pd.DataFrame,
                            permid_col: str,
                            effective_col: str = "DateEffective") -> pd.Series:
    """
    Vectorized: most recent FYE strictly before the deal effective date.

    Implementation note — why _key_label is required
    --------------------------------------------------
    pd.DataFrame.merge() with left_on/right_on always resets the output index
    to a new sequential RangeIndex, regardless of the left frame's index.  Each
    deal row in `keys` may match dozens of panel rows (one per fiscal year), so
    the merge output is much longer than `keys`.  Grouping by merged.index would
    group by the merge output's arbitrary position integers — not by deal identity
    — making .max() a no-op and the subsequent .reindex(keys.index) near-random.

    The fix mirrors compute_t3_fyes_vectorized(): stamp the original keys.index
    as a plain column (_key_label) before the merge, then groupby that column.
    """
    work = keys[[permid_col, effective_col]].copy()
    work["_key_label"] = keys.index          # preserve deal identity across merge

    merged = work[["_key_label", permid_col, effective_col]].merge(
        panel[["PermID", "FYE"]],
        left_on=permid_col,
        right_on="PermID",
        how="left",
    )
    merged = merged[merged["FYE"].notna() & (merged["FYE"] < merged[effective_col])]

    if merged.empty:
        return pd.Series(pd.NaT, index=keys.index)

    result = merged.groupby("_key_label")["FYE"].max()
    return result.reindex(keys.index)


def compute_t3_fyes_vectorized(panel: pd.DataFrame, keys: pd.DataFrame,
                               permid_col: str,
                               effective_col: str = "DateEffective") -> pd.Series:
    """
    Vectorized: FYE closest to (effective_date + 3 years), within T3_MAX_DELTA_DAYS.
    FYEs outside that window are treated as no-match (returns NaT for that deal).
    Uses ~1095.75-day offset; exact enough for annual FYE proximity.
    """
    work = keys[[permid_col, effective_col]].copy()
    # Store original index as a plain column so groupby can recover it after merge
    work["_key_label"] = keys.index
    work["_target_date"] = (
        pd.to_datetime(work[effective_col]) + pd.to_timedelta(3 * 365.25, unit="D")
    )

    merged = work[["_key_label", permid_col, "_target_date"]].merge(
        panel[["PermID", "FYE"]],
        left_on=permid_col,
        right_on="PermID",
        how="left",
    ).dropna(subset=["FYE", "_target_date"])

    if merged.empty:
        return pd.Series(pd.NaT, index=keys.index)

    merged["_delta"] = (merged["FYE"] - merged["_target_date"]).abs()
    merged = merged.dropna(subset=["_delta"])

    # Reject FYEs that are too far from the 3-year target date.
    # Without this guard, the closest FYE could be years away for entities
    # with sparse Worldscope coverage, silently corrupting the post-deal CFROA.
    merged = merged[merged["_delta"] <= pd.Timedelta(days=T3_MAX_DELTA_DAYS)]

    if merged.empty:
        return pd.Series(pd.NaT, index=keys.index)

    best_pos = merged.groupby("_key_label")["_delta"].idxmin()
    best = merged.loc[best_pos.values, ["_key_label", "FYE"]].set_index("_key_label")["FYE"]
    return best.reindex(keys.index)

# =============================================================================
# JOIN & FEATURE CONSTRUCTION FUNCTIONS
# =============================================================================

def build_role_join(panel: pd.DataFrame, role_col: str, fye_col: str,
                    prefix: str, field_map: Dict[str, str]) -> pd.DataFrame:
    """
    Build a joinable table for one role (acquiror/target) at a given FYE key.
    Missing fields are skipped with a warning rather than raising, so partial
    Worldscope coverage does not abort the pipeline.
    """
    panel_lookup = {str(c).strip().lower(): c for c in panel.columns}
    select_cols = ["PermID", "FYE"]
    rename_out  = {"PermID": role_col, "FYE": fye_col}

    for tr_field, short_name in field_map.items():
        actual = panel_lookup.get(tr_field.strip().lower())
        if actual is None:
            logger.warning(
                f"build_role_join: '{tr_field}' absent from panel — "
                f"'{prefix}{short_name}' will be missing"
            )
            continue
        select_cols.append(actual)
        rename_out[actual] = f"{prefix}{short_name}"

    if len(select_cols) == 2:
        logger.error(f"build_role_join: no proxy fields found for prefix '{prefix}'")
        return pd.DataFrame(columns=[role_col, fye_col])

    return panel[select_cols].copy().rename(columns=rename_out)




def compute_deal_features(
    keys: pd.DataFrame,
    panel: pd.DataFrame,
    wacc_panel: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Join pre-deal and post-deal proxy fields to deal keys.

    Returns deal-level DataFrame with columns:
      tA_fye, tB_fye, t3AB_fye  — matched fiscal year-ends
      A_t_*                      — acquiror pre-deal financials
      B_t_*                      — target pre-deal financials
      AB_t3_*                    — combined post-deal financials
      tA_wacc_date, tB_wacc_date — closest WACC snapshot date before deal date
      A_w_*                      — acquiror StarMine WACC fields (optional)
      B_w_*                      — target StarMine WACC fields (optional)

    Feature engineering (CFROA target variable, derived ratios, dummies) is
    handled separately in feature_engineering.py.
    """
    logger.info("Joining proxy fields to deal keys...")

    if panel.empty:
        logger.warning("Financial panel is empty — returning keys without proxy data")
        return keys.copy()

    keys = keys.copy()

    # ------------------------------------------------------------------
    # FYE matching
    # ------------------------------------------------------------------
    logger.info("Computing FYE keys...")
    keys["tA_fye"]   = compute_fyes_vectorized(panel, keys, "AcquirorPermID")
    keys["tB_fye"]   = compute_fyes_vectorized(panel, keys, "TargetPermID")
    keys["t3AB_fye"] = compute_t3_fyes_vectorized(panel, keys, "AcquirorPermID")

    # ------------------------------------------------------------------
    # Stage 0 join diagnostics
    # Separates "no identifier in panel" from "identifier present but
    # no FYE before deal date" from "FYE matched successfully".
    # This distinguishes fetch failures (Worldscope coverage gaps) from
    # date-logic failures (FYE window too narrow).
    # ------------------------------------------------------------------
    n_deals = len(keys)
    panel_ids = set(panel["PermID"].astype(str).str.strip().unique())

    def _fye_diag(role_col: str, fye_col: str, label: str) -> None:
        ids = keys[role_col].astype(str).str.strip()
        n_in_panel   = ids.isin(panel_ids).sum()
        # Has any row in panel with a valid FYE before DateEffective
        valid_rows   = panel[["PermID", "FYE"]].copy()
        valid_rows["PermID"] = valid_rows["PermID"].astype(str).str.strip()
        valid_rows   = valid_rows[valid_rows["FYE"].notna()]
        covered_ids  = set(valid_rows["PermID"].unique())
        # Of deals where id is in panel, how many have at least one historical row
        sub = keys[[role_col, "DateEffective"]].copy()
        sub["_id"] = ids
        sub = sub.merge(valid_rows, left_on="_id", right_on="PermID", how="inner")
        sub = sub[sub["FYE"] < sub["DateEffective"]]
        n_has_history = sub["_id"].nunique()
        n_assigned    = keys[fye_col].notna().sum()
        logger.info(
            f"  {label:20s}  "
            f"in_panel={n_in_panel}/{n_deals} ({100*n_in_panel/max(n_deals,1):.1f}%)  "
            f"has_pre-deal_row={n_has_history}/{n_deals} ({100*n_has_history/max(n_deals,1):.1f}%)  "
            f"fye_assigned={n_assigned}/{n_deals} ({100*n_assigned/max(n_deals,1):.1f}%)"
        )

    logger.info(f"Stage 0 FYE diagnostics — {n_deals} deals, panel {panel['PermID'].nunique()} unique entities:")
    _fye_diag("AcquirorPermID", "tA_fye",   "tA_fye  (acquiror)")
    _fye_diag("TargetPermID",   "tB_fye",   "tB_fye  (target)")

    n_t3     = keys["t3AB_fye"].notna().sum()
    n_all3   = (keys["tA_fye"].notna() & keys["tB_fye"].notna() & keys["t3AB_fye"].notna()).sum()
    logger.info(
        f"  {'t3AB_fye (combined)':20s}  "
        f"fye_assigned={n_t3}/{n_deals} ({100*n_t3/max(n_deals,1):.1f}%)"
    )
    logger.info(
        f"  All 3 FYEs resolved: {n_all3}/{n_deals} ({100*n_all3/max(n_deals,1):.1f}%) "
        f"— these deals are eligible for a Healy label"
    )

    # ------------------------------------------------------------------
    # Proxy field joins (Worldscope)
    # ------------------------------------------------------------------
    logger.info("Joining proxy fields (A_t, B_t, AB_t3)...")
    a_join   = build_role_join(panel, "AcquirorPermID", "tA_fye",   "A_t_",   PROXY_FIELDS)
    b_join   = build_role_join(panel, "TargetPermID",   "tB_fye",   "B_t_",   PROXY_FIELDS)
    ab3_join = build_role_join(panel, "AcquirorPermID", "t3AB_fye", "AB_t3_", PROXY_FIELDS)

    deal_level = keys.merge(a_join,   on=["AcquirorPermID", "tA_fye"],   how="left")
    deal_level = deal_level.merge(b_join,   on=["TargetPermID",   "tB_fye"],   how="left")
    deal_level = deal_level.merge(ab3_join, on=["AcquirorPermID", "t3AB_fye"], how="left")

    logger.info(f"✓ Proxy fields joined: {len(deal_level)} deals")

    # ------------------------------------------------------------------
    # StarMine WACC joins (financial synergy channel — optional)
    #
    # WACC_date is treated identically to FYE: compute_fyes_vectorized
    # finds the closest snapshot strictly before DateEffective.
    # No post-deal (t3) WACC join — WACC is a pre-deal input only.
    # ------------------------------------------------------------------
    if wacc_panel is not None and not wacc_panel.empty:
        logger.info("Joining WACC fields (A_w, B_w)...")
        # Rename WACC_date → FYE so compute_fyes_vectorized can reuse the
        # same logic (it groups on PermID + FYE and picks max FYE < deal date)
        wacc_for_fye = wacc_panel.rename(columns={"WACC_date": "FYE"})

        deal_level["tA_wacc_date"] = compute_fyes_vectorized(
            wacc_for_fye, deal_level, "AcquirorPermID"
        )
        deal_level["tB_wacc_date"] = compute_fyes_vectorized(
            wacc_for_fye, deal_level, "TargetPermID"
        )

        # Use WACC_JOIN_MAP (short_name → short_name) because get_wacc_data()
        # already renamed TR field names to short names before returning.
        aw_join = build_role_join(
            wacc_for_fye, "AcquirorPermID", "tA_wacc_date", "A_w_", WACC_JOIN_MAP
        )
        bw_join = build_role_join(
            wacc_for_fye, "TargetPermID",   "tB_wacc_date", "B_w_", WACC_JOIN_MAP
        )

        deal_level = deal_level.merge(
            aw_join, on=["AcquirorPermID", "tA_wacc_date"], how="left"
        )
        deal_level = deal_level.merge(
            bw_join, on=["TargetPermID", "tB_wacc_date"], how="left"
        )

        n_wacc = deal_level["A_w_wacc"].notna().sum()
        logger.info(
            f"✓ WACC fields joined: {n_wacc}/{len(deal_level)} deals have acquiror WACC"
        )
    else:
        logger.info("WACC panel not provided or empty — skipping WACC joins")

    return deal_level



# =============================================================================
# VALIDATION AND OUTPUT FUNCTIONS
# =============================================================================

def validate_financial_joins(deal_level: pd.DataFrame) -> None:
    """
    Per-stage coverage funnel — reports how many deals survive each join step.

    Funnel stages
    -------------
    1. tA_fye resolved   : acquiror has a Worldscope FYE before the deal date
    2. tB_fye resolved   : target has a Worldscope FYE before the deal date
    3. t3AB_fye resolved : combined entity has a Worldscope FYE ~3 yrs post-deal
    4. A_t financials    : acquiror pre-deal proxy fields populated
    5. B_t financials    : target pre-deal proxy fields populated
    6. AB_t3 financials  : combined post-deal proxy fields populated
    7. All 3 present     : maximum possible Healy label yield (A_t ∩ B_t ∩ AB_t3)

    Interpretation guide
    --------------------
    Low stage 1/2  → Worldscope coverage gap for acquirors/targets; check public filter.
    Low stage 3    → t+3 data not yet available (deal too recent) or entity PermID changed.
    Stage 4–6 < stage 1–3 → PermID mismatch between screener and Worldscope.
    Stage 7 / Total → effective label rate; target ~30–50 % for public-only deals.
    """
    logger.info("\n" + "=" * 60)
    logger.info("COVERAGE FUNNEL — validate_financial_joins()")
    logger.info("=" * 60)
    total = len(deal_level)
    logger.info(f"Deals entering funnel: {total}")

    if total == 0:
        logger.warning("No deals to validate.")
        return

    def _pct(n):
        return f"{n / total * 100:.1f}%" if total > 0 else "—"

    # ── Stage 1–3: FYE resolution ─────────────────────────────────────────────
    for stage, col, label in [
        (1, "tA_fye",   "tA_fye resolved  (acquiror pre-deal FYE)"),
        (2, "tB_fye",   "tB_fye resolved  (target pre-deal FYE)"),
        (3, "t3AB_fye", "t3AB_fye resolved (combined t+3 FYE)"),
    ]:
        if col in deal_level.columns:
            n = deal_level[col].notna().sum()
            logger.info(f"  [{stage}] {label}: {n}/{total} ({_pct(n)})")
        else:
            logger.warning(f"  [{stage}] Column '{col}' absent from deal_level")

    # ── Stage 4–6: Financial field population ────────────────────────────────
    # Use total_assets as the sentinel — it is one of the most broadly reported
    # Worldscope fields and is required for the CFROA target variable.
    sentinel = {
        "A_t_total_assets":   (4, "A_t financials   (acquiror pre-deal)"),
        "B_t_total_assets":   (5, "B_t financials   (target pre-deal)"),
        "AB_t3_total_assets": (6, "AB_t3 financials (combined post-deal)"),
    }
    present_counts = {}
    for col, (stage, label) in sentinel.items():
        if col in deal_level.columns:
            n = deal_level[col].notna().sum()
            present_counts[col] = n
            logger.info(f"  [{stage}] {label}: {n}/{total} ({_pct(n)})")
        else:
            present_counts[col] = 0
            logger.warning(f"  [{stage}] Column '{col}' absent — stage skipped")

    # ── Stage 7: Maximum Healy label yield (all three non-NaN) ───────────────
    # For the full CFROA target we additionally need operating_cashflow.
    # Check both total_assets AND operating_cashflow to surface CF-specific gaps.
    cfo_cols = {
        "A_t_operating_cashflow":   "A_t  CFO",
        "B_t_operating_cashflow":   "B_t  CFO",
        "AB_t3_operating_cashflow": "AB_t3 CFO",
    }
    logger.info("  --- CFO field coverage (needed for Healy target) ---")
    for col, label in cfo_cols.items():
        if col in deal_level.columns:
            n = deal_level[col].notna().sum()
            logger.info(f"      {label}: {n}/{total} ({_pct(n)})")
        else:
            logger.warning(f"      {label}: column '{col}' absent")

    # Full-label mask: all six inputs (assets + CFO for A_t, B_t, AB_t3) non-NaN
    asset_cols = list(sentinel.keys())
    cfo_col_list = list(cfo_cols.keys())
    all_six = [c for c in asset_cols + cfo_col_list if c in deal_level.columns]
    if len(all_six) == 6:
        full_label_mask = deal_level[all_six].notna().all(axis=1)
        n_labels = full_label_mask.sum()
        logger.info(f"  [7] All 6 Healy inputs present → max label yield: "
                    f"{n_labels}/{total} ({_pct(n_labels)})")
        if n_labels == 0:
            logger.warning(
                "  ⚠  ZERO usable Healy labels detected.\n"
                "     Likely causes (check in order):\n"
                "     1. filter_public_only=False  → private targets have no Worldscope data\n"
                "     2. Date range too narrow / recent → t+3 data not yet in Worldscope\n"
                "     3. PermID mismatch between screener and Worldscope panel\n"
                "     4. min_deal_value_usd too low → noisy micro-deals with sparse coverage"
            )
    else:
        missing_cols = [c for c in asset_cols + cfo_col_list if c not in deal_level.columns]
        logger.warning(
            f"  [7] Cannot compute full label yield — missing columns: {missing_cols}"
        )

    logger.info("=" * 60)


def merge_final_output(df: pd.DataFrame, deal_level: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [
        c for c in ["DateEffective", "AcquirorPermID", "TargetPermID", "CombinedPermID"]
        if c in deal_level.columns
    ]
    full = df.merge(
        deal_level.drop(columns=cols_to_drop),
        on="deal_id",
        how="left",
    )
    logger.info(f"Final dataset: {len(full)} rows × {len(full.columns)} columns")
    return full


def save_results(full_deal_level: pd.DataFrame) -> None:
    output_path = CONFIG['output_dir'] / CONFIG['output_filename']
    output_path.parent.mkdir(parents=True, exist_ok=True)
    full_deal_level.to_csv(output_path, index=False)
    logger.info(f"Results saved to: {output_path}  ({output_path.stat().st_size / 1024:.1f} KB)")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_analysis():
    """
    DAQ pipeline — fetches deal and financial data, joins proxy fields, saves CSV.

    Steps:
      1. Fetch M&A deal screener data
      2. Normalize deals; extract SIC codes where available
      3. Fetch Worldscope annual financials
      3b. Fetch StarMine WACC panel (financial synergy proxies — optional)
      4. Match FYEs and join proxy fields (A_t_*, B_t_*, AB_t3_*, A_w_*, B_w_*)
      5. Merge and save

    Feature engineering (CFROA target variable, derived ratios, deal dummies)
    is handled in feature_engineering.py, which reads the output CSV of this script.
    """
    full_deal_level = None
    try:
        logger.info("=" * 60)
        logger.info("M&A DAQ PIPELINE")
        logger.info("=" * 60)

        # Step 1
        logger.info("\n[STEP 1/5] Fetching deal data")
        logger.info("-" * 60)
        deals = get_deal_data()

        # Step 2
        logger.info("\n[STEP 2/5] Processing deals")
        logger.info("-" * 60)
        df, keys = process_deals(deals)

        # Step 3
        logger.info("\n[STEP 3/5] Fetching Worldscope financial data")
        logger.info("-" * 60)
        permids = sorted(
            set(keys["AcquirorPermID"]).union(set(keys["TargetPermID"]))
        )
        panel = get_financial_data(permids)

        # Step 3b — StarMine WACC (financial synergy channel)
        # WACC fields are not in Worldscope; fetched separately from StarMine.
        # Returns empty DataFrame gracefully if StarMine coverage is absent.
        logger.info("\n[STEP 3b/5] Fetching StarMine WACC data")
        logger.info("-" * 60)
        wacc_panel = get_wacc_data(permids)

        # Step 4
        logger.info("\n[STEP 4/5] Joining proxy fields")
        logger.info("-" * 60)
        deal_level = compute_deal_features(keys, panel, wacc_panel=wacc_panel)
        validate_financial_joins(deal_level)

        # Step 5
        logger.info("\n[STEP 5/5] Merging and saving")
        logger.info("-" * 60)
        full_deal_level = merge_final_output(df, deal_level)
        save_results(full_deal_level)

        logger.info("\n" + "=" * 60)
        logger.info("✓ PIPELINE COMPLETE")
        logger.info("=" * 60)
        logger.info("Output columns:")
        logger.info("  A_t_* / B_t_*     — acquiror / target pre-deal proxy financials")
        logger.info("  AB_t3_*           — combined post-deal proxy financials")
        logger.info("  A_w_* / B_w_*     — acquiror / target StarMine WACC fields")
        logger.info("  tA_fye / tB_fye / t3AB_fye       — Worldscope FYE matches")
        logger.info("  tA_wacc_date / tB_wacc_date       — WACC snapshot dates")
        logger.info("Run feature_engineering.py on the output CSV for derived features.")
        logger.info("=" * 60)

        return full_deal_level

    except Exception as e:
        logger.error(f"\n{'=' * 60}")
        logger.error(f"PIPELINE ERROR: {e}")
        logger.error(f"{'=' * 60}")
        if full_deal_level is not None and not full_deal_level.empty:
            logger.warning("Attempting to save partial results...")
            try:
                save_results(full_deal_level)
                logger.info("✓ Partial results saved")
            except Exception as save_err:
                logger.error(f"Could not save partial results: {save_err}")
        return full_deal_level if full_deal_level is not None else pd.DataFrame()

# =============================================================================
# SPYDER EXECUTION
# =============================================================================

if __name__ == "__main__":
    result_df = run_analysis()

    print("\n" + "=" * 60)
    print("SAMPLE OUTPUT (first 5 rows):")
    print("=" * 60)
    print(result_df.head())

    print(f"\nShape: {result_df.shape}")
    print(f"Columns: {list(result_df.columns)}")
