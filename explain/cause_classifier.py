"""Explain: classify leakage cause (rule + Elkan–Noto PU stretch)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.config import OUTPUT_DIR

OBS_ZERO = 8.0
LOW_ACCEPT = 0.35
HIGH_CASH = 0.30
LOW_DEMAND_CF = 40.0
LOW_LEAKAGE = 10.0
PU_OVERRIDE = 0.65


def extensive_margin(observed_spend: pd.Series) -> pd.Series:
    return (observed_spend < OBS_ZERO).astype(int)


def rule_cause(df: pd.DataFrame) -> pd.Series:
    """Causal priority: no_demand → no_acceptance → cash → rail_substitution."""
    a = df["acceptance_density"].to_numpy()
    cash = df["cash_intensity"].to_numpy()
    obs = df["observed_spend"].to_numpy()
    cf = df["cf_spend"].to_numpy()
    leak = df["leakage_estimate"].to_numpy()

    cause = np.full(len(df), "rail_substitution", dtype=object)

    is_cash = (cash >= HIGH_CASH) & (a >= LOW_ACCEPT) & (leak >= LOW_LEAKAGE)
    cause[is_cash] = "cash"

    is_no_acc = (a < LOW_ACCEPT) & (obs < OBS_ZERO)
    cause[is_no_acc] = "no_acceptance"

    is_no_demand = (cf < LOW_DEMAND_CF) | ((leak < LOW_LEAKAGE) & (obs < OBS_ZERO))
    cause[is_no_demand] = "no_demand"
    return pd.Series(cause, index=df.index, name="cause")


def _design_matrix(df: pd.DataFrame) -> np.ndarray:
    numeric = np.column_stack(
        [
            df["acceptance_density"].to_numpy(),
            df["cash_intensity"].to_numpy(),
            np.log1p(df["observed_spend"].to_numpy()),
            np.log1p(df["leakage_estimate"].to_numpy()),
            np.log1p(df["nights"].to_numpy()),
            df["extensive_margin"].to_numpy(),
        ]
    )
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    cats = enc.fit_transform(df[["category", "trip_type", "segment"]])
    return np.hstack([numeric, cats])


def pu_no_acceptance_scores(
    df: pd.DataFrame, declines: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, float]:
    """Elkan–Noto PU: declined auths are labeled positives; silence is unlabeled."""
    keys = declines[["trip_id", "category"]].drop_duplicates()
    keys["labeled_positive"] = 1
    labeled = (
        df[["trip_id", "category"]]
        .merge(keys, on=["trip_id", "category"], how="left")["labeled_positive"]
        .fillna(0)
        .to_numpy()
        .astype(int)
    )
    X = _design_matrix(df)
    clf = LogisticRegression(max_iter=400, class_weight="balanced")
    clf.fit(X, labeled)
    p_labeled = clf.predict_proba(X)[:, 1]
    c = float(p_labeled[labeled == 1].mean()) if labeled.sum() else 1.0
    c = max(c, 1e-3)
    return np.clip(p_labeled / c, 0.0, 1.0), labeled, c


def classify(df: pd.DataFrame, declines: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["extensive_margin"] = extensive_margin(out["observed_spend"])
    out["cause_rule"] = rule_cause(out)
    scores, labeled, c_hat = pu_no_acceptance_scores(out, declines)
    out["pu_no_acceptance"] = scores
    out["pu_labeled_positive"] = labeled
    cause = out["cause_rule"].to_numpy().copy()
    promote = (
        (scores >= PU_OVERRIDE)
        & (out["extensive_margin"].to_numpy() == 1)
        & (out["acceptance_density"].to_numpy() < LOW_ACCEPT)
        & (out["cause_rule"].to_numpy() != "no_demand")
    )
    cause[promote] = "no_acceptance"
    out["cause"] = cause
    out.attrs["pu_c_hat"] = c_hat
    return out


def score_hidden(df: pd.DataFrame) -> dict:
    """Cause accuracy vs oracle labels. Fitting never used these labels."""
    truth = pd.read_csv(OUTPUT_DIR / "ground_truth.csv")
    merged = df.merge(
        truth[["trip_id", "category", "true_cause", "acceptance_leakage"]],
        on=["trip_id", "category"],
        how="left",
    )
    acc = float((merged["cause"] == merged["true_cause"]).mean())
    w = merged["leakage_estimate"].to_numpy()
    w_acc = float(
        np.average(
            (merged["cause"] == merged["true_cause"]).astype(float),
            weights=np.maximum(w, 1e-9),
        )
    )
    by = (
        merged.groupby(["true_cause", "cause"], as_index=False)
        .size()
        .pivot(index="true_cause", columns="cause", values="size")
        .fillna(0)
        .astype(int)
    )
    print("cause mix (model)")
    print(merged["cause"].value_counts(normalize=True).mul(100).round(1).to_string())
    print()
    print("confusion (rows)  rows=true, cols=predicted")
    print(by.to_string())
    print()
    print(f"row accuracy:            {acc:.1%}")
    print(f"leakage-weighted accuracy: {w_acc:.1%}   "
          "(share of estimated $ tagged with the oracle cause)")
    return {
        "cause_row_accuracy": acc,
        "cause_value_weighted_accuracy": w_acc,
        "pu_c_hat": float(df.attrs.get("pu_c_hat", float("nan"))),
        "cause_mix": merged["cause"].value_counts().to_dict(),
    }


def run() -> pd.DataFrame:
    estimates = pd.read_csv(OUTPUT_DIR / "leakage_estimates.csv")
    declines = pd.read_csv(OUTPUT_DIR / "declines.csv")
    out = classify(estimates, declines)
    out.to_csv(OUTPUT_DIR / "leakage_estimates.csv", index=False)

    print("=" * 72)
    print("NEMU Explain — leakage cause classifier")
    print("=" * 72)
    print(
        "rules: "
        f"LOW_ACCEPT={LOW_ACCEPT}, HIGH_CASH={HIGH_CASH}, "
        f"OBS_ZERO={OBS_ZERO}, LOW_DEMAND_CF={LOW_DEMAND_CF}"
    )
    print(f"PU labeled positives (declines matched to trip×category): "
          f"{int(out['pu_labeled_positive'].sum()):,}")
    print()
    metrics = score_hidden(out)
    notice = json.loads((OUTPUT_DIR / "notice_metrics.json").read_text())
    notice.update(metrics)
    (OUTPUT_DIR / "notice_metrics.json").write_text(json.dumps(notice, indent=2, default=str))
    print(f"\nwrote cause column → {OUTPUT_DIR / 'leakage_estimates.csv'}")
    return out


if __name__ == "__main__":
    run()
