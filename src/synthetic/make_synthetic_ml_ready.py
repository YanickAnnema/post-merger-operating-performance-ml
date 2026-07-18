"""Create a small synthetic ML-ready panel for demo and smoke testing.

This file does not use or approximate the licensed thesis dataset. It only
creates columns with the names expected by the ML scripts so the repository can
demonstrate workflow shape without redistributing LSEG/Worldscope data.
"""

from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = [
    "cost_acquiror_asset_utilization",
    "cost_target_asset_utilization",
    "cost_relative_asset_utilization",
    "revenue_acquiror_growth",
    "revenue_target_growth",
    "operational_acquiror_cf_margin",
    "operational_target_cf_margin",
    "financial_leverage_acquiror",
    "financial_leverage_target",
    "financial_altman_z_acquiror",
    "financial_altman_z_target",
    "deal_cross_border",
    "deal_all_cash",
    "deal_stock_payment",
]


def make_synthetic_ml_ready(n: int = 600, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    years = rng.integers(1995, 2023, size=n)
    df = pd.DataFrame({"announcement_year": years})

    for col in FEATURES:
        if col.startswith("deal_"):
            df[col] = rng.binomial(1, 0.35, size=n)
        else:
            df[col] = rng.normal(0, 1, size=n)

    signal = (
        -0.18 * df["cost_target_asset_utilization"]
        + 0.12 * df["financial_altman_z_acquiror"]
        - 0.08 * df["deal_cross_border"]
    )
    df["synergy_healy1992_w"] = signal + rng.normal(0, 0.9, size=n)

    df["split"] = np.select(
        [df["announcement_year"] <= 2015, df["announcement_year"] <= 2018],
        ["train", "val"],
        default="test",
    )
    return df


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out = root / "data" / "synthetic_ml_ready_nowinsor.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    make_synthetic_ml_ready().to_csv(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

