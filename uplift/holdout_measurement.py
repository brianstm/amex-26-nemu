"""Randomized holdout to audit the uplift model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.config import OUTPUT_DIR, RNG_SEED
from match.uplift_model import NOISE_SIGMA, true_treatment_effect

N_BOOT = 400
HOLDOUT_FRAC = 0.50


def bootstrap_diff(treat: np.ndarray, control: np.ndarray, rng: np.random.Generator, n: int = N_BOOT):
    if len(treat) == 0 or len(control) == 0:
        return float("nan"), float("nan"), float("nan")
    diffs = np.empty(n)
    for i in range(n):
        t = rng.choice(treat, size=len(treat), replace=True).mean()
        c = rng.choice(control, size=len(control), replace=True).mean()
        diffs[i] = t - c
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return float(diffs.mean()), float(lo), float(hi)


def run() -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED + 7)
    rec = pd.read_csv(OUTPUT_DIR / "uplift_recommendations.csv")
    eligible = rec.loc[rec["recommended_arm"] != "no_offer"].copy()
    members = eligible["member_id"].unique()
    treated_members = set(rng.choice(members, size=int(len(members) * HOLDOUT_FRAC), replace=False))
    eligible["holdout_cell"] = np.where(
        eligible["member_id"].isin(treated_members), "treatment", "control"
    )

    realized_te = np.zeros(len(eligible))
    for arm in eligible["recommended_arm"].unique():
        mask = (eligible["recommended_arm"] == arm) & (eligible["holdout_cell"] == "treatment")
        realized_te[mask.to_numpy()] = true_treatment_effect(eligible.loc[mask], arm)
    noise = rng.normal(0.0, NOISE_SIGMA, size=len(eligible))
    eligible["realized_spend"] = np.clip(
        eligible["expected_spend"] * (1.0 + realized_te + noise), 0.0, None
    )
    eligible["predicted_incremental"] = np.where(
        eligible["holdout_cell"] == "treatment",
        eligible["predicted_uplift"] * eligible["expected_spend"],
        0.0,
    )
    eligible["realized_incremental"] = eligible["realized_spend"] - eligible["expected_spend"]

    t = eligible.loc[eligible["holdout_cell"] == "treatment", "realized_spend"].to_numpy()
    c = eligible.loc[eligible["holdout_cell"] == "control", "realized_spend"].to_numpy()
    mean_diff, lo, hi = bootstrap_diff(t, c, rng)

    pred_mean = float(
        eligible.loc[eligible["holdout_cell"] == "treatment", "predicted_incremental"].mean()
    )
    realized_t = float(
        eligible.loc[eligible["holdout_cell"] == "treatment", "realized_incremental"].mean()
    )
    realized_c = float(
        eligible.loc[eligible["holdout_cell"] == "control", "realized_incremental"].mean()
    )
    realized_lift = realized_t - realized_c

    calib = (
        eligible.loc[eligible["holdout_cell"] == "treatment"]
        .groupby(["recommended_arm", "category"], as_index=False)
        .agg(
            n=("member_id", "size"),
            predicted=("predicted_incremental", "mean"),
            realized=("realized_incremental", "mean"),
        )
    )
    calib["calibration_gap"] = calib["realized"] - calib["predicted"]

    eligible.to_csv(OUTPUT_DIR / "holdout_assignments.csv", index=False)
    calib.to_csv(OUTPUT_DIR / "holdout_calibration.csv", index=False)
    summary = {
        "n_eligible_rows": int(len(eligible)),
        "n_treated_rows": int((eligible.holdout_cell == "treatment").sum()),
        "n_control_rows": int((eligible.holdout_cell == "control").sum()),
        "mean_spend_diff_treatment_minus_control": mean_diff,
        "spend_diff_ci_low": lo,
        "spend_diff_ci_high": hi,
        "predicted_incremental_per_treated": pred_mean,
        "realized_incremental_per_treated_net_of_control": realized_lift,
        "calibration_ratio": (realized_lift / pred_mean) if pred_mean else float("nan"),
    }
    (OUTPUT_DIR / "holdout_metrics.json").write_text(json.dumps(summary, indent=2))

    print("=" * 72)
    print("NEMU Uplift — randomized holdout")
    print("=" * 72)
    print(f"eligible member×category rows: {summary['n_eligible_rows']:,}  "
          f"(treat {summary['n_treated_rows']:,} / control {summary['n_control_rows']:,})")
    print(f"mean spend difference (T − C): ${mean_diff:,.2f}  "
          f"95% bootstrap CI [${lo:,.2f}, ${hi:,.2f}]")
    print(f"model-predicted incremental $ per treated row: ${pred_mean:,.2f}")
    print(f"realized incremental $ per treated row:        ${realized_lift:,.2f}")
    print(f"calibration ratio realized/predicted:          {summary['calibration_ratio']:.2f}")
    print()
    print("calibration by arm × category")
    print(calib.round(2).to_string(index=False))
    print(f"\nwrote {OUTPUT_DIR / 'holdout_assignments.csv'}")
    return eligible


if __name__ == "__main__":
    run()
