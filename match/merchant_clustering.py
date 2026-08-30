"""Greedy submodular merchant targeting for ``no_acceptance`` districts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.config import OUTPUT_DIR

MERCHANT_BUDGET = 25
LAMBDA = 0.45


def coverage(n: np.ndarray | float, lam: float = LAMBDA) -> np.ndarray | float:
    """Share of district recoverable value unlocked by n signed merchants."""
    return 1.0 - np.exp(-lam * np.asarray(n, dtype=float))


def district_table(leakage: pd.DataFrame) -> pd.DataFrame:
    subset = leakage.loc[leakage["cause"] == "no_acceptance"].copy()
    agg = (
        subset.groupby(["dest_country", "dest_city", "dest_district"], as_index=False)
        .agg(
            recoverable_value=("leakage_estimate", "sum"),
            n_leaky_rows=("trip_id", "size"),
            n_members=("member_id", "nunique"),
            acceptance_density=("acceptance_density", "mean"),
            cash_intensity=("cash_intensity", "mean"),
        )
        .sort_values("recoverable_value", ascending=False)
        .reset_index(drop=True)
    )
    agg["rank_by_value"] = np.arange(1, len(agg) + 1)
    return agg


def greedy_select(districts: pd.DataFrame, budget: int = MERCHANT_BUDGET) -> pd.DataFrame:
    """Pick one merchant at a time by highest marginal covered value."""
    d = districts.reset_index(drop=True)
    n = np.zeros(len(d), dtype=int)
    log = []
    remaining = d["recoverable_value"].to_numpy(dtype=float)

    for step in range(1, budget + 1):
        current = coverage(n)
        nxt = coverage(n + 1)
        gain = remaining * (nxt - current)
        pick = int(np.argmax(gain))
        n[pick] += 1
        log.append(
            {
                "step": step,
                "dest_country": d.loc[pick, "dest_country"],
                "dest_city": d.loc[pick, "dest_city"],
                "dest_district": d.loc[pick, "dest_district"],
                "merchants_in_district_after": int(n[pick]),
                "marginal_value": float(gain[pick]),
                "covered_value_after": float(remaining[pick] * coverage(n[pick])),
            }
        )

    d = d.copy()
    d["merchants_signed"] = n
    d["coverage"] = coverage(n)
    d["covered_value"] = d["recoverable_value"] * d["coverage"]
    d["uncovered_value"] = d["recoverable_value"] - d["covered_value"]
    assert d["covered_value"].min() >= -1e-9
    return d, pd.DataFrame(log)


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    leakage = pd.read_csv(OUTPUT_DIR / "leakage_estimates.csv")
    districts = district_table(leakage)
    ranked, path = greedy_select(districts)

    ranked.to_csv(OUTPUT_DIR / "merchant_targets.csv", index=False)
    path.to_csv(OUTPUT_DIR / "merchant_selection_path.csv", index=False)

    print("=" * 72)
    print("NEMU Match — greedy submodular merchant targeting")
    print("=" * 72)
    print(f"no_acceptance rows: {int((leakage.cause == 'no_acceptance').sum()):,}")
    print(f"districts with recoverable value: {len(ranked)}")
    print(f"budget: {MERCHANT_BUDGET} merchants  |  λ={LAMBDA}")
    print(f"value covered by greedy set: ${ranked['covered_value'].sum():,.0f}  "
          f"of ${ranked['recoverable_value'].sum():,.0f} "
          f"({ranked['covered_value'].sum() / ranked['recoverable_value'].sum():.1%})")
    print()
    print("target list (signed > 0), ranked by recoverable value")
    show = ranked.loc[ranked["merchants_signed"] > 0].sort_values(
        "recoverable_value", ascending=False
    )
    cols = [
        "rank_by_value",
        "dest_country",
        "dest_district",
        "acceptance_density",
        "recoverable_value",
        "merchants_signed",
        "coverage",
        "covered_value",
    ]
    print(show[cols].round(2).to_string(index=False))
    print(f"\nwrote {OUTPUT_DIR / 'merchant_targets.csv'}")
    return ranked, path


if __name__ == "__main__":
    run()
