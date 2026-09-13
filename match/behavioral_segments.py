"""Behavioural-pattern classification + incentive-recommendation drill.

Five traveller patterns NEMU tracks — each maps to one concrete offer.
Category → sub-category → merchant → voucher.

Signals used:
- ``pred_te_2x_points`` / ``pred_te_statement_credit`` — from the causal-forest
  uplift model (``match/uplift_model.py``).
- category concentration (HHI) and merchant diversity — from ``transactions.csv``.
- destination cash friction — mean ``cash_intensity`` from ``leakage_estimates``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.config import OUTPUT_DIR, PER_VISIT_BASE
from match.merchant_targets_detail import classify_subcategory

# Category concentration above this counts as "loyal to one category" (4
# categories => even split is 0.25, single-category is 1.0).
HHI_LOYAL = 0.50
ROUND_THRESHOLDS = np.array([50, 100, 200, 500, 1000], dtype=float)

PATTERN_ORDER = [
    "Points Optimiser",
    "Immediate Value Seeker",
    "Threshold Chaser",
    "Category Loyalist",
    "Merchant Explorer",
]

TARGETED = set(PATTERN_ORDER)

INTERVENTION = {
    "Points Optimiser": "2x / 3x points",
    "Immediate Value Seeker": "Cashback",
    "Threshold Chaser": "Spend ¥10,000, get ¥1,000 back",
    "Category Loyalist": "Dining / retail / attraction-specific reward",
    "Merchant Explorer": "Merchant-specific voucher",
}

OBSERVES = {
    "Points Optimiser": "Historically increases spend when points multipliers appear",
    "Immediate Value Seeker": "Stronger response to direct monetary savings",
    "Threshold Chaser": "Spend jumps when close to a reward threshold",
    "Category Loyalist": "Consistently spends heavily in one category",
    "Merchant Explorer": "Frequently tries new merchants and local businesses",
}


def _threshold_score(amounts: np.ndarray) -> float:
    """Share of tickets sitting just below a round reward threshold."""
    if len(amounts) == 0:
        return 0.0
    a = amounts[:, None]
    lo = 0.8 * ROUND_THRESHOLDS[None, :]
    hi = ROUND_THRESHOLDS[None, :]
    near = ((a >= lo) & (a < hi)).any(axis=1)
    return float(near.mean())


def classify_vectorised(up: pd.DataFrame) -> np.ndarray:
    """Assign one of the five patterns by priority (data-relative cutoffs)."""
    explorer_cut = up["distinct_merchants"].quantile(0.80)
    thr_cut = up["threshold_score"].quantile(0.85)

    conditions = [
        up["distinct_merchants"] >= explorer_cut,
        up["category_hhi"] >= HHI_LOYAL,
        up["threshold_score"] >= thr_cut,
        up["te_points"] >= up["te_credit"],
    ]
    choices = [
        "Merchant Explorer",
        "Category Loyalist",
        "Threshold Chaser",
        "Points Optimiser",
    ]
    return np.select(conditions, choices, default="Immediate Value Seeker")


def _voucher(pattern: str, cat: str, sub: str, merchant: str) -> str:
    if pattern == "Merchant Explorer":
        return f"{merchant} voucher"
    if pattern == "Category Loyalist":
        return f"{cat.title()} / {sub} reward"
    if pattern == "Points Optimiser":
        return f"3x points on {cat}"
    if pattern == "Immediate Value Seeker":
        return f"Cashback at {merchant}"
    if pattern == "Threshold Chaser":
        return f"Spend ¥10,000, get ¥1,000 back · {merchant}"
    return "—"


def build(
    uplift: pd.DataFrame, txns: pd.DataFrame, leakage: pd.DataFrame
) -> pd.DataFrame:
    # ---- member-level signals from the uplift model -----------------------
    up = (
        uplift.groupby("member_id")
        .agg(
            segment=("segment", "first"),
            home_market=("home_market", "first"),
            te_points=("pred_te_2x_points", "mean"),
            te_credit=("pred_te_statement_credit", "mean"),
            predicted_uplift=("predicted_uplift", "mean"),
            expected_spend=("expected_spend", "sum"),
            leakage_estimate=("leakage_estimate", "sum"),
        )
        .reset_index()
    )

    # ---- destination cash friction ----------------------------------------
    cash = leakage.groupby("member_id")["cash_intensity"].mean().rename("mean_cash")
    up = up.merge(cash, on="member_id", how="left")
    up["mean_cash"] = up["mean_cash"].fillna(0.0)

    # ---- transaction signals ----------------------------------------------
    tx = txns.groupby("member_id").agg(
        visits=("amount_usd", "size"),
        distinct_merchants=("merchant_name", "nunique"),
        total_spend=("amount_usd", "sum"),
    )
    tx["merchant_diversity"] = tx["distinct_merchants"] / tx["visits"].clip(lower=1)
    up = up.merge(tx, on="member_id", how="left")

    # category HHI + top category
    cat_spend = (
        txns.groupby(["member_id", "category"])["amount_usd"].sum().reset_index()
    )
    tot = cat_spend.groupby("member_id")["amount_usd"].transform("sum")
    cat_spend["share"] = cat_spend["amount_usd"] / tot.clip(lower=1e-9)
    hhi = cat_spend.groupby("member_id")["share"].apply(lambda s: float((s**2).sum()))
    top_cat = cat_spend.loc[
        cat_spend.groupby("member_id")["amount_usd"].idxmax()
    ].set_index("member_id")["category"]
    up = up.merge(hhi.rename("category_hhi"), on="member_id", how="left")
    up = up.merge(top_cat.rename("top_category"), on="member_id", how="left")

    # threshold bunching
    thr = (
        txns.groupby("member_id")["amount_usd"]
        .apply(lambda s: _threshold_score(s.to_numpy()))
        .rename("threshold_score")
    )
    up = up.merge(thr, on="member_id", how="left")

    up["spend_top_quartile"] = up["total_spend"] >= up["total_spend"].quantile(0.75)
    for c in ["visits", "distinct_merchants", "merchant_diversity", "category_hhi",
              "threshold_score", "total_spend", "predicted_uplift", "te_points",
              "te_credit"]:
        up[c] = up[c].fillna(0.0)

    up["pattern"] = classify_vectorised(up)
    up["is_targeted"] = True
    up["recommended_intervention"] = up["pattern"].map(INTERVENTION)

    # ---- incentive drill: top sub-category + merchant in the top category --
    merch = (
        txns.groupby(["member_id", "category", "merchant_name"])
        .agg(spend=("amount_usd", "sum"), med=("amount_usd", "median"))
        .reset_index()
    )
    merch = merch.merge(
        up[["member_id", "top_category"]], on="member_id", how="left"
    )
    merch = merch[merch["category"] == merch["top_category"]].copy()
    base_mid = {c: 0.5 * (b[0] + b[1]) for c, b in PER_VISIT_BASE.items()}
    merch["sub_category"] = [
        classify_subcategory(n, c, m, base_mid.get(c, 25.0))
        for n, c, m in zip(merch["merchant_name"], merch["category"], merch["med"])
    ]
    top_merchant = (
        merch.loc[merch.groupby("member_id")["spend"].idxmax()]
        .set_index("member_id")[["merchant_name", "sub_category"]]
    )
    up = up.merge(
        top_merchant.rename(
            columns={"merchant_name": "top_merchant", "sub_category": "top_subcategory"}
        ),
        on="member_id",
        how="left",
    )
    up["top_merchant"] = up["top_merchant"].fillna("—")
    up["top_subcategory"] = up["top_subcategory"].fillna("—")
    up["top_category"] = up["top_category"].fillna("—")

    up["recommended_voucher"] = [
        _voucher(p, c, s, mch)
        for p, c, s, mch in zip(
            up["pattern"], up["top_category"], up["top_subcategory"],
            up["top_merchant"],
        )
    ]
    return up


def run() -> pd.DataFrame:
    uplift = pd.read_csv(OUTPUT_DIR / "uplift_recommendations.csv")
    txns = pd.read_csv(OUTPUT_DIR / "transactions.csv")
    leakage = pd.read_csv(OUTPUT_DIR / "leakage_estimates.csv")
    seg = build(uplift, txns, leakage)
    cols = [
        "member_id", "segment", "home_market", "pattern", "is_targeted",
        "recommended_intervention", "top_category", "top_subcategory",
        "top_merchant", "recommended_voucher", "predicted_uplift",
        "te_points", "te_credit", "expected_spend", "leakage_estimate",
        "total_spend", "category_hhi", "merchant_diversity", "threshold_score",
        "mean_cash",
    ]
    seg[cols].to_csv(OUTPUT_DIR / "behavioral_segments.csv", index=False)

    print("=" * 72)
    print("NEMU Match — behavioural patterns + incentive drill")
    print("=" * 72)
    counts = seg["pattern"].value_counts().reindex(PATTERN_ORDER).fillna(0).astype(int)
    n = len(seg)
    for p in PATTERN_ORDER:
        print(f"  {p:<26} {counts[p]:>5}  ({counts[p]/n:>5.1%})  → {INTERVENTION[p]}")
    print(f"\ntargeted members: {n:,} / {n:,}")
    print(f"leakage at stake: ${seg['leakage_estimate'].sum():,.0f}")
    print(f"\nwrote {OUTPUT_DIR / 'behavioral_segments.csv'}")
    return seg


if __name__ == "__main__":
    run()
