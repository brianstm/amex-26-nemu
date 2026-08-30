"""Notice: reconstruct presence and estimate counterfactual spend with PPML."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.generalized_linear_model import GLMResults

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.config import CATEGORIES, OUTPUT_DIR, SATURATED_ACCEPTANCE


FORMULA = (
    "observed_spend ~ "
    "C(home_market) + C(dest_country) + C(category) + "
    "np.log(acceptance_density) + I(np.log(acceptance_density) ** 2) + "
    "np.log(domestic_intensity) + C(trip_type) + np.log(nights) + C(segment)"
)


def reconstruct_panel(
    trips: pd.DataFrame,
    transactions: pd.DataFrame,
    members: pd.DataFrame,
) -> pd.DataFrame:
    """Cartesian trip × category grid; zeros fill unobserved spend."""
    grid = trips.merge(members, on="member_id", how="left")
    cat_df = pd.DataFrame({"category": list(CATEGORIES)})
    panel = grid.merge(cat_df, how="cross")
    observed = (
        transactions.groupby(["trip_id", "category"], as_index=False)["amount"]
        .sum()
        .rename(columns={"amount": "observed_spend"})
    )
    panel = panel.merge(observed, on=["trip_id", "category"], how="left")
    panel["observed_spend"] = panel["observed_spend"].fillna(0.0)
    cat_idx = panel["category"].map(
        {cat: i for i, cat in enumerate(CATEGORIES)}
    ).to_numpy()
    dom = np.column_stack(
        [panel[f"domestic_{cat}"].to_numpy() for cat in CATEGORIES]
    )
    panel["domestic_intensity"] = np.maximum(dom[np.arange(len(panel)), cat_idx], 1.0)
    assert len(panel) == len(trips) * len(CATEGORIES)
    assert panel["observed_spend"].min() >= 0
    return panel


def fit_ppml(panel: pd.DataFrame) -> GLMResults:
    """Poisson GLM used as PPML. HC0 standard errors."""
    model = smf.glm(formula=FORMULA, data=panel, family=sm.families.Poisson())
    result = model.fit(cov_type="HC0", maxiter=100)
    return result


def predict_counterfactual(
    result: GLMResults,
    panel: pd.DataFrame,
    saturated: float = SATURATED_ACCEPTANCE,
) -> pd.DataFrame:
    """Leakage = E[spend | a=0.95, x] − E[spend | a=a0, x], clipped at 0."""
    cf = panel.copy()
    cf["acceptance_density"] = saturated
    pred_cf = result.get_prediction(cf)
    frame = pred_cf.summary_frame()
    out = panel.copy()
    out["fitted_spend"] = np.clip(result.fittedvalues.to_numpy(), 0.0, None)
    out["cf_spend"] = np.clip(frame["mean"].to_numpy(), 0.0, None)
    out["cf_se"] = frame["mean_se"].to_numpy()
    out["cf_ci_low"] = np.clip(frame["mean_ci_lower"].to_numpy(), 0.0, None)
    out["cf_ci_high"] = np.clip(frame["mean_ci_upper"].to_numpy(), 0.0, None)

    out["leakage_estimate"] = np.clip(out["cf_spend"] - out["fitted_spend"], 0.0, None)
    out["leakage_ci_low"] = np.clip(out["cf_ci_low"] - out["fitted_spend"], 0.0, None)
    out["leakage_ci_high"] = np.clip(out["cf_ci_high"] - out["fitted_spend"], 0.0, None)
    assert (out["leakage_estimate"] >= -1e-9).all()
    return out


def _mape(y: np.ndarray, yhat: np.ndarray, floor: float = 5.0) -> float:
    mask = y > floor
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(yhat[mask] - y[mask]) / y[mask]))


def validate_against_ground_truth(estimates: pd.DataFrame, truth_path: Path) -> dict:
    """Score the estimator on the hidden file. Called after fitting."""
    truth = pd.read_csv(truth_path)
    merged = estimates.merge(
        truth[
            [
                "trip_id",
                "category",
                "true_spend",
                "acceptance_leakage",
                "total_leakage",
                "true_cause",
            ]
        ],
        on=["trip_id", "category"],
        how="left",
        validate="one_to_one",
    )
    assert merged["acceptance_leakage"].notna().all()

    row_corr = float(
        np.corrcoef(merged["leakage_estimate"], merged["acceptance_leakage"])[0, 1]
    )
    total_corr = float(
        np.corrcoef(merged["leakage_estimate"], merged["total_leakage"])[0, 1]
    )
    row_mape = _mape(
        merged["acceptance_leakage"].to_numpy(), merged["leakage_estimate"].to_numpy()
    )

    corridor = merged.groupby(["dest_country", "category"], as_index=False).agg(
        est=("leakage_estimate", "sum"),
        truth_acc=("acceptance_leakage", "sum"),
        truth_total=("total_leakage", "sum"),
        observed=("observed_spend", "sum"),
    )
    corr_corr = float(np.corrcoef(corridor["est"], corridor["truth_acc"])[0, 1])
    corr_mape = _mape(corridor["truth_acc"].to_numpy(), corridor["est"].to_numpy(), floor=100.0)

    hid_acc = float(merged["acceptance_leakage"].sum())
    hid_total = float(merged["total_leakage"].sum())
    found = float(merged["leakage_estimate"].sum())

    metrics = {
        "n_rows": int(len(merged)),
        "hidden_acceptance_leakage": hid_acc,
        "hidden_total_leakage": hid_total,
        "estimated_leakage": found,
        "recovery_ratio_vs_acceptance": found / hid_acc if hid_acc else float("nan"),
        "row_corr_vs_acceptance_leakage": row_corr,
        "row_corr_vs_total_leakage": total_corr,
        "row_mape_vs_acceptance_leakage": row_mape,
        "corridor_corr_vs_acceptance_leakage": corr_corr,
        "corridor_mape_vs_acceptance_leakage": corr_mape,
    }
    corridor.to_csv(OUTPUT_DIR / "validation_corridor.csv", index=False)
    merged[["trip_id", "category", "leakage_estimate", "acceptance_leakage", "total_leakage"]].to_csv(
        OUTPUT_DIR / "validation_rows.csv", index=False
    )
    return metrics, merged, corridor


def print_validation(
    result: GLMResults,
    metrics: dict,
    corridor: pd.DataFrame,
) -> None:
    print("=" * 72)
    print("NEMU Notice — PPML gravity model")
    print("=" * 72)
    print(f"specification:\n  {FORMULA}")
    print()
    beta = result.params.get("np.log(acceptance_density)")
    se = result.bse.get("np.log(acceptance_density)")
    print("key coefficients (identified off district variation within country)")
    print(f"  beta_log_acceptance = {beta:.3f}  (HC0 se {se:.3f}, z={beta / se:.1f})")
    beta2 = result.params.get("I(np.log(acceptance_density) ** 2)")
    if beta2 is not None:
        se2 = result.bse.get("I(np.log(acceptance_density) ** 2)")
        print(f"  beta_log_acceptance_sq = {beta2:.3f}  (HC0 se {se2:.3f}, z={beta2 / se2:.1f})")
        local = beta + 2.0 * beta2 * np.log(0.50)
        print(
            f"  local elasticity at density=0.50: {local:.2f}  "
            f"(10% density lift → {np.exp(local * np.log(1.10)):.3f}x spend)"
        )
    print()
    print("HIDDEN-FILE VALIDATION (ground_truth.csv was not in the likelihood)")
    print(f"  we hid ${metrics['hidden_acceptance_leakage']:,.0f} of acceptance leakage")
    print(f"  model found ${metrics['estimated_leakage']:,.0f}  "
          f"(recovery {metrics['recovery_ratio_vs_acceptance']:.1%})")
    print(f"  total true−observed gap was ${metrics['hidden_total_leakage']:,.0f} "
          "(includes cash — we do not claim that)")
    print()
    print("  row-level   corr vs acceptance leakage "
          f"{metrics['row_corr_vs_acceptance_leakage']:.3f}   "
          f"MAPE {metrics['row_mape_vs_acceptance_leakage']:.1%}")
    print("  row-level   corr vs TOTAL leakage      "
          f"{metrics['row_corr_vs_total_leakage']:.3f}   "
          "(lower on purpose: cash is a confound)")
    print("  corridor    corr vs acceptance leakage "
          f"{metrics['corridor_corr_vs_acceptance_leakage']:.3f}   "
          f"MAPE {metrics['corridor_mape_vs_acceptance_leakage']:.1%}")
    print()
    print("top corridors by estimated recoverable value")
    top = corridor.sort_values("est", ascending=False).head(10)
    print(
        top.assign(
            est=lambda d: d["est"].round(0),
            truth_acc=lambda d: d["truth_acc"].round(0),
        )[["dest_country", "category", "est", "truth_acc", "observed"]]
        .to_string(index=False)
    )


def run() -> pd.DataFrame:
    trips = pd.read_csv(OUTPUT_DIR / "trips.csv")
    txns = pd.read_csv(OUTPUT_DIR / "transactions.csv")
    members = pd.read_csv(OUTPUT_DIR / "members.csv")

    panel = reconstruct_panel(trips, txns, members)
    print(f"reconstructed panel: {len(panel):,} trip × category rows "
          f"({(panel['observed_spend'] == 0).mean():.1%} zeros)")

    result = fit_ppml(panel)
    estimates = predict_counterfactual(result, panel)

    keep = [
        "trip_id",
        "member_id",
        "home_market",
        "segment",
        "dest_country",
        "dest_city",
        "dest_district",
        "acceptance_density",
        "cash_intensity",
        "nights",
        "trip_type",
        "category",
        "observed_spend",
        "fitted_spend",
        "domestic_intensity",
        "cf_spend",
        "cf_se",
        "leakage_estimate",
        "leakage_ci_low",
        "leakage_ci_high",
    ]
    out = estimates[keep].copy()
    out.to_csv(OUTPUT_DIR / "leakage_estimates.csv", index=False)

    metrics, _merged, corridor = validate_against_ground_truth(
        estimates, OUTPUT_DIR / "ground_truth.csv"
    )
    (OUTPUT_DIR / "notice_metrics.json").write_text(json.dumps(metrics, indent=2))
    (OUTPUT_DIR / "ppml_summary.txt").write_text(result.summary().as_text())
    print_validation(result, metrics, corridor)
    print(f"\nwrote {OUTPUT_DIR / 'leakage_estimates.csv'}")
    return out


if __name__ == "__main__":
    run()
