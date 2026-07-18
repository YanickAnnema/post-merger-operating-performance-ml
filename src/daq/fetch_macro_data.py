"""
fetch_macro_data.py — One-time macro data fetch for the M&A synergy pipeline.
==============================================================================

Downloads two long-run time series and stores them locally so that
data_preparation.py never needs a live internet connection or optional
packages installed at pipeline run-time.

Series fetched:
  S&P 500 monthly end-of-month closes (^GSPC) via yfinance, 1984–2023
  Moody's Baa / Aaa corporate bond yields via FRED (pandas_datareader), 1984–2023

Features produced (already lag-adjusted):
  sp500_trailing_12m    — 12-month price return, pre-lagged by 1 month.
                          Row "YYYY-MM" contains the return from the 12 months
                          ending the calendar month BEFORE that year-month.
                          Example: row "2005-03" holds (price_Feb-05 / price_Feb-04) − 1.
  credit_spread_bbb_aaa — Baa − Aaa yield spread, pre-lagged by 1 month.
                          Row "2005-03" holds the Feb-2005 spread.

Rationale for pre-computing the lag here:
  data_preparation.py only needs a simple CSV key-lookup on YYYY-MM strings.
  All date arithmetic and lag logic stays in one place, reducing the risk of
  off-by-one errors in the pipeline merge.

Output:
  macro_monthly.csv   (path in OUTPUT_PATH below)
  Columns: year_month, sp500_price_raw, sp500_trailing_12m,
           credit_spread_raw, credit_spread_bbb_aaa

  Rows with NaN in either feature column are retained so the merge always
  finds a match key — data_preparation.py handles NaN gracefully.

Usage:
  Run ONCE (F5 in Spyder) after installing the dependencies below.
  Re-run only if you extend the date range beyond 2023.

Dependencies:
  pip install yfinance pandas-datareader --break-system-packages

Optimised for Spyder IDE (F5 execution).
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
import sys

# =============================================================================
# CONFIGURATION
# =============================================================================

# Date range: start 13 months before thesis sample start (1993) to allow the
# trailing-12m computation to be defined for the earliest deals, plus a buffer
# to 2023 so post-2018 deals (if ever added) are covered.
FETCH_START = "1984-01-01"   # yfinance / FRED start date (extra buffer for lag)
FETCH_END   = "2023-12-31"   # inclusive end

# Lag applied to both macro variables (months).  1 = deal in month M sees the
# value from month M-1, preventing same-month lookahead.
LAG_MONTHS = 1

# S&P 500 ticker (Yahoo Finance)
SP500_TICKER = "^GSPC"

# FRED series identifiers for Moody's corporate yields
FRED_BAA = "BAA"   # Moody's Seasoned Baa Corporate Bond Yield
FRED_AAA = "AAA"   # Moody's Seasoned Aaa Corporate Bond Yield

# Output path — write next to the other pipeline CSVs
OUTPUT_PATH = Path(__file__).resolve().parents[2] / "outputs" / "DAQ pipeline" / "macro_monthly.csv"

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
# STEP 1 — S&P 500
# =============================================================================

def fetch_sp500() -> pd.Series:
    """
    Download S&P 500 daily closes from Yahoo Finance and resample to month-end.
    Returns a Series indexed by pd.Period (monthly), values are month-end closes.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.error(
            "yfinance not installed.\n"
            "  Install with:  pip install yfinance --break-system-packages"
        )
        sys.exit(1)

    logger.info(f"Downloading {SP500_TICKER} from Yahoo Finance ({FETCH_START} → {FETCH_END})")
    raw = yf.download(
        SP500_TICKER,
        start=FETCH_START,
        end=FETCH_END,
        progress=False,
        auto_adjust=True,
    )

    if raw.empty:
        logger.error("yfinance returned empty DataFrame — check ticker and internet connection")
        sys.exit(1)

    # Flatten MultiIndex columns if present (yfinance >= 0.2.x)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)

    close = raw["Close"].copy()
    # Resample to month-end; use last available trading day in each month
    monthly = close.resample("ME").last()
    monthly.index = monthly.index.to_period("M")
    monthly.name = "sp500_price_raw"

    n_months = monthly.notna().sum()
    logger.info(f"  S&P 500: {n_months} monthly observations ({monthly.index[0]} → {monthly.index[-1]})")
    return monthly


def compute_sp500_trailing_12m(monthly_prices: pd.Series) -> pd.Series:
    """
    Compute 12-month trailing price return for each month, then lag by LAG_MONTHS.

    Without lag:  return[M] = price[M] / price[M-12] − 1
    After lag:    the lookup table row "M+1" contains return[M].
    A deal effective in month M+1 retrieves return[M] — no same-month lookahead.

    Returns a Series keyed by PeriodIndex (monthly), values are the lagged returns.
    """
    # shift(12) aligns price[M-12] with month M
    r = monthly_prices / monthly_prices.shift(12) - 1

    # Lag: shift the return series forward by LAG_MONTHS periods.
    # After the shift, the index value "M+LAG" contains return[M].
    r.index = r.index + LAG_MONTHS
    r.name = "sp500_trailing_12m"

    n_valid = r.notna().sum()
    logger.info(f"  sp500_trailing_12m: {n_valid} valid months after lag={LAG_MONTHS}")
    return r

# =============================================================================
# STEP 2 — CREDIT SPREAD (Baa − Aaa)
# =============================================================================

def fetch_credit_spread() -> pd.Series:
    """
    Download Moody's Baa and Aaa yield series from FRED via pandas_datareader,
    compute the Baa − Aaa spread, lag by LAG_MONTHS.
    Returns a Series keyed by PeriodIndex (monthly), values are lagged spreads.
    """
    try:
        import pandas_datareader.data as web
    except ImportError:
        logger.error(
            "pandas_datareader not installed.\n"
            "  Install with:  pip install pandas-datareader --break-system-packages"
        )
        sys.exit(1)

    logger.info(f"Downloading FRED {FRED_BAA} and {FRED_AAA} ({FETCH_START} → {FETCH_END})")
    try:
        baa = web.DataReader(FRED_BAA, "fred", FETCH_START, FETCH_END).squeeze()
        aaa = web.DataReader(FRED_AAA, "fred", FETCH_START, FETCH_END).squeeze()
    except Exception as exc:
        logger.error(f"FRED download failed: {exc}")
        sys.exit(1)

    # FRED monthly series: already one obs per month; harmonise to month-end
    baa = baa.resample("ME").last()
    aaa = aaa.resample("ME").last()
    spread = baa - aaa
    spread.name = "credit_spread_raw"
    spread.index = spread.index.to_period("M")

    # Lag: shift forward so row "M+LAG" contains the spread from month M
    spread_lagged = spread.copy()
    spread_lagged.index = spread_lagged.index + LAG_MONTHS
    spread_lagged.name = "credit_spread_bbb_aaa"

    n_valid = spread_lagged.notna().sum()
    logger.info(f"  credit_spread_bbb_aaa: {n_valid} valid months after lag={LAG_MONTHS}")
    return spread, spread_lagged

# =============================================================================
# STEP 3 — ASSEMBLE AND SAVE
# =============================================================================

def build_and_save(
    sp500_prices:  pd.Series,
    sp500_lagged:  pd.Series,
    spread_raw:    pd.Series,
    spread_lagged: pd.Series,
) -> pd.DataFrame:
    """
    Join all series onto a common monthly PeriodIndex and write to CSV.
    The index is converted to YYYY-MM strings so the CSV is human-readable
    and pandas Period parsing is not required when loading in data_preparation.py.
    """
    # Build a full monthly grid covering the fetch range
    full_index = pd.period_range(start=FETCH_START, end=FETCH_END, freq="M")

    macro = pd.DataFrame(index=full_index)
    macro.index.name = "year_month"

    macro["sp500_price_raw"]       = sp500_prices.reindex(full_index)
    macro["sp500_trailing_12m"]    = sp500_lagged.reindex(full_index)
    macro["credit_spread_raw"]     = spread_raw.reindex(full_index)
    macro["credit_spread_bbb_aaa"] = spread_lagged.reindex(full_index)

    # Convert PeriodIndex to YYYY-MM strings for CSV compatibility
    macro.index = macro.index.strftime("%Y-%m")
    macro.index.name = "year_month"

    # Diagnostics
    n_rows = len(macro)
    n_sp   = macro["sp500_trailing_12m"].notna().sum()
    n_cs   = macro["credit_spread_bbb_aaa"].notna().sum()
    logger.info(
        f"\nMacro table: {n_rows} rows (months)"
        f"\n  sp500_trailing_12m    non-NaN: {n_sp}/{n_rows}"
        f"\n  credit_spread_bbb_aaa non-NaN: {n_cs}/{n_rows}"
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    macro.to_csv(OUTPUT_PATH)
    logger.info(f"\nSaved → {OUTPUT_PATH}")

    # Quick sanity check: print a few rows near Jan 2005
    sample = macro.loc["2004-11":"2005-03"]
    if not sample.empty:
        logger.info("\nSample rows (2004-11 → 2005-03):")
        logger.info(sample[["sp500_trailing_12m", "credit_spread_bbb_aaa"]].to_string())

    return macro


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("fetch_macro_data.py — one-time macro series download")
    logger.info("=" * 60)

    sp500_prices  = fetch_sp500()
    sp500_lagged  = compute_sp500_trailing_12m(sp500_prices)
    spread_raw, spread_lagged = fetch_credit_spread()

    macro = build_and_save(sp500_prices, sp500_lagged, spread_raw, spread_lagged)

    logger.info("\nDone. Run data_preparation.py to merge these features into ml_ready.csv.")
