"""Merchant-level target list for ``no_acceptance`` districts.

The district-level greedy targeting in ``match/merchant_clustering.py`` answers
*which district* to acquire. This module drops one level deeper and answers
*which named merchant* to onboard first, with a per-visit price band and the
district's recoverable value allocated across its merchants with the same
submodular diminishing-returns shape.

Merchant names come from ``transactions.csv`` (OpenStreetMap / public operators);
they are real venues. Spend amounts and the leakage they anchor are simulated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.config import (
    PER_VISIT_BASE,
    PRICE_LEVEL,
    REGION,
    SUBCATEGORY_KEYWORDS,
)
from data.config import OUTPUT_DIR
from match.merchant_clustering import LAMBDA, district_table

MIN_VISITS_FOR_DATA_BAND = 5
MIN_SPREAD = 5.0

# Amount-tier fallback sub-categories: (low tier, mid tier, high tier).
FALLBACK_SUBCAT: dict[str, tuple[str, str, str]] = {
    "dining": ("Street & Local", "Casual Dining", "Fine Dining"),
    "retail": ("Local Shops", "General Retail", "Premium Retail"),
    "transport": ("Local Transit", "City Transport", "Premium Transport"),
    "lodging": ("Budget & Hostel", "Mid-scale Hotel", "Upscale Hotel"),
}


def _round5(x: float) -> float:
    return float(max(5.0, round(x / 5.0) * 5.0))


def classify_subcategory(
    name: str, category: str, median_visit: float, base_mid: float
) -> str:
    low = str(name).lower()
    for sub, kws in SUBCATEGORY_KEYWORDS.get(category, []):
        if any(k in low for k in kws):
            return sub
    lo_tier, mid_tier, hi_tier = FALLBACK_SUBCAT.get(
        category, ("Standard", "Standard", "Premium")
    )
    if median_visit >= 1.8 * base_mid:
        return hi_tier
    if median_visit <= 0.7 * base_mid:
        return lo_tier
    return mid_tier


def _price_band(
    p25: float, p75: float, median: float, category: str, price_level: float, visits: int
) -> tuple[float, float]:
    base_low, base_high = PER_VISIT_BASE.get(category, (15.0, 40.0))
    b_low, b_high = base_low * price_level, base_high * price_level
    if visits >= MIN_VISITS_FOR_DATA_BAND and np.isfinite(p25) and np.isfinite(p75):
        low = 0.5 * p25 + 0.5 * b_low
        high = 0.5 * p75 + 0.5 * b_high
    else:
        low, high = b_low, b_high
    low, high = _round5(low), _round5(high)
    if high < low + MIN_SPREAD:
        high = low + MIN_SPREAD
    return low, high


def build(leakage: pd.DataFrame, txns: pd.DataFrame) -> pd.DataFrame:
    districts = district_table(leakage)  # no_acceptance only, with recoverable_value
    district_val = districts.set_index(["dest_country", "dest_city", "dest_district"])

    # district -> city map (transactions lack dest_city)
    city_map = (
        leakage[["dest_country", "dest_district", "dest_city"]]
        .drop_duplicates()
        .set_index(["dest_country", "dest_district"])["dest_city"]
    )

    t = txns.copy()
    t["dest_city"] = t.set_index(["dest_country", "dest_district"]).index.map(city_map)
    keep = t.set_index(["dest_country", "dest_city", "dest_district"]).index.isin(
        district_val.index
    )
    t = t[keep]
    if t.empty:
        return pd.DataFrame()

    grouped = (
        t.groupby(
            ["dest_country", "dest_city", "dest_district", "category", "merchant_name"],
            as_index=False,
        )
        .agg(
            visits=("amount_usd", "size"),
            median_visit=("amount_usd", "median"),
            p25=("amount_usd", lambda s: s.quantile(0.25)),
            p75=("amount_usd", lambda s: s.quantile(0.75)),
            mcc=("mcc", "first"),
        )
    )

    rows = []
    key_cols = ["dest_country", "dest_city", "dest_district"]
    for key, dgrp in grouped.groupby(key_cols):
        drow = district_val.loc[key]
        total = float(drow["recoverable_value"])
        accept = float(drow["acceptance_density"])
        cash = float(drow["cash_intensity"])
        country = key[0]
        price_level = float(PRICE_LEVEL.get(country, 1.0))

        # rank merchants by footfall, allocate value with submodular marginal
        # weights (merchant #5 worth less than #1); normalise so the district
        # total is preserved exactly.
        d = dgrp.sort_values("visits", ascending=False).reset_index(drop=True)
        ranks = np.arange(1, len(d) + 1)
        marginal = np.exp(-LAMBDA * (ranks - 1)) - np.exp(-LAMBDA * ranks)
        share = marginal / marginal.sum()
        d["est_recoverable_value"] = total * share

        for i, r in d.iterrows():
            cat = r["category"]
            base_low, base_high = PER_VISIT_BASE.get(cat, (15.0, 40.0))
            base_mid = 0.5 * (base_low + base_high) * price_level
            low, high = _price_band(
                r["p25"], r["p75"], r["median_visit"], cat, price_level, int(r["visits"])
            )
            rows.append(
                {
                    "region": REGION.get(country, "—"),
                    "dest_country": country,
                    "dest_city": key[1],
                    "dest_district": key[2],
                    "category": cat,
                    "sub_category": classify_subcategory(
                        r["merchant_name"], cat, float(r["median_visit"]), base_mid
                    ),
                    "merchant_name": r["merchant_name"],
                    "mcc": int(r["mcc"]) if pd.notna(r["mcc"]) else 0,
                    "visits": int(r["visits"]),
                    "avg_spend_per_visit": _round5(float(r["median_visit"])),
                    "price_low": low,
                    "price_high": high,
                    "price_range": f"${low:,.0f}–{high:,.0f} / visit",
                    "est_recoverable_value": float(r["est_recoverable_value"]),
                    "acceptance_density": accept,
                    "cash_intensity": cash,
                    "credit_share": 1.0 - cash,
                    "cash_share": cash,
                }
            )

    out = pd.DataFrame(rows).sort_values("est_recoverable_value", ascending=False)
    out["priority_rank"] = np.arange(1, len(out) + 1)
    return out.reset_index(drop=True)


def run() -> pd.DataFrame:
    leakage = pd.read_csv(OUTPUT_DIR / "leakage_estimates.csv")
    txns = pd.read_csv(OUTPUT_DIR / "transactions.csv")
    detail = build(leakage, txns)
    detail.to_csv(OUTPUT_DIR / "merchant_targets_detail.csv", index=False)

    print("=" * 72)
    print("NEMU Match — merchant-level target list")
    print("=" * 72)
    print(f"target districts (no_acceptance): {detail['dest_district'].nunique()}")
    print(f"named merchant targets: {len(detail):,}")
    print(f"allocated recoverable value: ${detail['est_recoverable_value'].sum():,.0f}")
    print()
    print("top 12 merchants to onboard")
    cols = [
        "priority_rank",
        "region",
        "dest_country",
        "dest_district",
        "category",
        "sub_category",
        "merchant_name",
        "price_range",
        "est_recoverable_value",
    ]
    print(detail[cols].head(12).to_string(index=False))
    print(f"\nwrote {OUTPUT_DIR / 'merchant_targets_detail.csv'}")
    return detail


if __name__ == "__main__":
    run()
