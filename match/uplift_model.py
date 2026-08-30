"""Incentive assignment for ``rail_substitution`` leakage."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.config import OUTPUT_DIR, RNG_SEED

ARMS = ("no_offer", "2x_points", "statement_credit")
TAKE_RATE = 0.12
POINTS_COST_RATE = 0.005
STATEMENT_CREDIT_USD = 12.0
NOISE_SIGMA = 0.08


def true_treatment_effect(df: pd.DataFrame, arm: str) -> np.ndarray:
    """Oracle CATE. The estimator does not receive this function."""
    dining = (df["category"] == "dining").to_numpy(dtype=float)
    retail = (df["category"] == "retail").to_numpy(dtype=float)
    high = df["segment"].isin(["platinum", "centurion"]).to_numpy(dtype=float)
    core = (df["segment"] == "core").to_numpy(dtype=float)
    leisure = (df["trip_type"] == "leisure").to_numpy(dtype=float)
    if arm == "no_offer":
        return np.zeros(len(df))
    if arm == "2x_points":
        return np.clip(0.05 + 0.12 * dining + 0.06 * high + 0.05 * leisure - 0.04 * retail, 0.0, 0.35)
    if arm == "statement_credit":
        return np.clip(0.06 + 0.13 * retail + 0.07 * core - 0.03 * high + 0.02 * dining, 0.0, 0.35)
    raise ValueError(arm)


def incentive_cost(arm: str, expected_spend: np.ndarray) -> np.ndarray:
    if arm == "no_offer":
        return np.zeros_like(expected_spend, dtype=float)
    if arm == "2x_points":
        return POINTS_COST_RATE * expected_spend
    if arm == "statement_credit":
        return np.full_like(expected_spend, STATEMENT_CREDIT_USD, dtype=float)
    raise ValueError(arm)


def aggregate_members(leakage: pd.DataFrame) -> pd.DataFrame:
    rail = leakage.loc[leakage["cause"] == "rail_substitution"].copy()
    grain = (
        rail.groupby(["member_id", "category"], as_index=False)
        .agg(
            observed_spend=("observed_spend", "sum"),
            leakage_estimate=("leakage_estimate", "sum"),
            fitted_spend=("fitted_spend", "sum"),
            acceptance_density=("acceptance_density", "mean"),
            cash_intensity=("cash_intensity", "mean"),
            nights=("nights", "sum"),
            n_trips=("trip_id", "nunique"),
            segment=("segment", "first"),
            home_market=("home_market", "first"),
            trip_type=("trip_type", lambda s: s.mode().iat[0] if len(s) else "leisure"),
            domestic_intensity=("domestic_intensity", "first"),
        )
    )
    grain["expected_spend"] = grain["observed_spend"].clip(lower=1.0)
    return grain


def _features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    numeric = np.column_stack(
        [
            np.log1p(df["observed_spend"].to_numpy()),
            np.log1p(df["leakage_estimate"].to_numpy()),
            df["acceptance_density"].to_numpy(),
            df["cash_intensity"].to_numpy(),
            np.log1p(df["nights"].to_numpy()),
            np.log1p(df["n_trips"].to_numpy()),
            np.log1p(df["domestic_intensity"].to_numpy()),
        ]
    )
    names = [
        "log_spend",
        "log_leak",
        "acceptance_density",
        "cash_intensity",
        "log_nights",
        "log_trips",
        "log_domestic",
    ]
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    cats = enc.fit_transform(df[["category", "segment", "trip_type", "home_market"]])
    cat_names = list(enc.get_feature_names_out(["category", "segment", "trip_type", "home_market"]))
    return np.hstack([numeric, cats]), names + cat_names


def simulate_training(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Randomized three-arm assignment with known potential outcomes."""
    out = df.copy()
    out["arm"] = rng.choice(ARMS, size=len(out))
    te = np.zeros(len(out))
    for arm in ARMS:
        mask = out["arm"].to_numpy() == arm
        te[mask] = true_treatment_effect(out.loc[mask], arm)
    out["true_te"] = te
    noise = rng.normal(0.0, NOISE_SIGMA, size=len(out))
    out["outcome_spend"] = np.clip(
        out["expected_spend"] * (1.0 + te + noise), 0.0, None
    )
    return out


class XLearner:
    """Two-arm X-learner vs ``no_offer`` (Künzel et al. 2019)."""

    def __init__(self) -> None:
        self.mu0 = GradientBoostingRegressor(random_state=RNG_SEED, max_depth=3, n_estimators=80)
        self.mu1 = GradientBoostingRegressor(random_state=RNG_SEED, max_depth=3, n_estimators=80)
        self.tau0 = GradientBoostingRegressor(random_state=RNG_SEED, max_depth=3, n_estimators=80)
        self.tau1 = GradientBoostingRegressor(random_state=RNG_SEED, max_depth=3, n_estimators=80)
        self.p_treated = 0.5

    def fit(self, X: np.ndarray, t: np.ndarray, y: np.ndarray) -> "XLearner":
        self.mu0.fit(X[t == 0], y[t == 0])
        self.mu1.fit(X[t == 1], y[t == 1])
        d1 = y[t == 1] - self.mu0.predict(X[t == 1])
        d0 = self.mu1.predict(X[t == 0]) - y[t == 0]
        self.tau1.fit(X[t == 1], d1)
        self.tau0.fit(X[t == 0], d0)
        self.p_treated = float(t.mean())
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        g = self.p_treated
        return (1.0 - g) * self.tau1.predict(X) + g * self.tau0.predict(X)


def _try_causal_forest(X: np.ndarray, t: np.ndarray, y: np.ndarray, X_all: np.ndarray) -> np.ndarray | None:
    try:
        from econml.dml import CausalForestDML
    except ImportError:
        return None
    try:
        est = CausalForestDML(
            discrete_treatment=True,
            n_estimators=60,
            min_samples_leaf=25,
            max_depth=8,
            random_state=RNG_SEED,
        )
        n = len(t)
        if n > 4_000:
            rng = np.random.default_rng(RNG_SEED)
            idx = rng.choice(n, size=4_000, replace=False)
            est.fit(Y=y[idx], T=t[idx], X=X[idx], W=X[idx])
        else:
            est.fit(Y=y, T=t, X=X, W=X)
        return np.asarray(est.effect(X_all), dtype=float)
    except Exception as exc:
        print(f"CausalForestDML failed ({exc}); falling back to X-learner")
        return None


def estimate_uplift(train: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    X, _ = _features(train)
    y = train["outcome_spend"].to_numpy()
    method = "xlearner"
    uplift_cols = {}

    for arm in ("2x_points", "statement_credit"):
        mask = train["arm"].isin(["no_offer", arm]).to_numpy()
        t = (train.loc[mask, "arm"].to_numpy() == arm).astype(int)
        te_cf = _try_causal_forest(X[mask], t, y[mask], X)
        if te_cf is not None:
            te = te_cf / np.maximum(train["expected_spend"].to_numpy(), 1.0)
            method = "causal_forest_dml"
        else:
            learner = XLearner().fit(X[mask], t, y[mask])
            te = learner.predict(X) / np.maximum(train["expected_spend"].to_numpy(), 1.0)
        uplift_cols[arm] = np.clip(te, -0.05, 0.40)

    out = train.copy()
    out["pred_te_2x_points"] = uplift_cols["2x_points"]
    out["pred_te_statement_credit"] = uplift_cols["statement_credit"]
    out["pred_te_no_offer"] = 0.0
    return out, method


def recommend(df: pd.DataFrame) -> pd.DataFrame:
    spend = df["expected_spend"].to_numpy()
    rows = []
    for arm, te_col in (
        ("no_offer", "pred_te_no_offer"),
        ("2x_points", "pred_te_2x_points"),
        ("statement_credit", "pred_te_statement_credit"),
    ):
        te = df[te_col].to_numpy()
        net = te * spend * TAKE_RATE - incentive_cost(arm, spend)
        rows.append(net)
    nets = np.vstack(rows)
    best_idx = np.argmax(nets, axis=0)
    best_arm = np.array(ARMS)[best_idx]
    best_net = nets[best_idx, np.arange(len(df))]
    best_arm = np.where(best_net > 0, best_arm, "no_offer")
    best_net = np.where(best_net > 0, best_net, 0.0)
    te_map = {
        "no_offer": df["pred_te_no_offer"].to_numpy(),
        "2x_points": df["pred_te_2x_points"].to_numpy(),
        "statement_credit": df["pred_te_statement_credit"].to_numpy(),
    }
    pred_te = np.choose(
        pd.Series(best_arm).map({a: i for i, a in enumerate(ARMS)}).to_numpy(),
        [te_map[a] for a in ARMS],
    )
    out = df.copy()
    out["recommended_arm"] = best_arm
    out["predicted_uplift"] = pred_te
    out["expected_net_value"] = best_net
    out["true_te_recommended"] = 0.0
    for arm in ARMS:
        m = out["recommended_arm"] == arm
        out.loc[m, "true_te_recommended"] = true_treatment_effect(out.loc[m], arm)
    return out


def run() -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    leakage = pd.read_csv(OUTPUT_DIR / "leakage_estimates.csv")
    grain = aggregate_members(leakage)
    sim = simulate_training(grain, rng)
    scored, method = estimate_uplift(sim)
    rec = recommend(scored)

    keep = [
        "member_id",
        "category",
        "segment",
        "home_market",
        "trip_type",
        "expected_spend",
        "leakage_estimate",
        "arm",
        "outcome_spend",
        "true_te",
        "pred_te_2x_points",
        "pred_te_statement_credit",
        "recommended_arm",
        "predicted_uplift",
        "expected_net_value",
        "true_te_recommended",
    ]
    rec[keep].to_csv(OUTPUT_DIR / "uplift_recommendations.csv", index=False)

    corr_2x = float(np.corrcoef(rec["pred_te_2x_points"], true_treatment_effect(rec, "2x_points"))[0, 1])
    corr_sc = float(
        np.corrcoef(rec["pred_te_statement_credit"], true_treatment_effect(rec, "statement_credit"))[0, 1]
    )
    match_best = float((rec["recommended_arm"] == _oracle_arm(rec)).mean())

    print("=" * 72)
    print("NEMU Match — incentive uplift model")
    print("=" * 72)
    print(f"estimator: {method}")
    print(f"member × category rows (rail_substitution): {len(rec):,}")
    print(f"CATE corr vs oracle  2x_points={corr_2x:.3f}  statement_credit={corr_sc:.3f}")
    print(f"recommended-arm match vs oracle argmax-net: {match_best:.1%}")
    print()
    print("recommendation mix")
    print(rec["recommended_arm"].value_counts().to_string())
    print()
    print(f"sum expected net value of recommended policy: ${rec['expected_net_value'].sum():,.0f}")
    print(f"\nwrote {OUTPUT_DIR / 'uplift_recommendations.csv'}")

    metrics_path = OUTPUT_DIR / "notice_metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    metrics.update(
        {
            "uplift_method": method,
            "uplift_corr_2x": corr_2x,
            "uplift_corr_statement_credit": corr_sc,
            "uplift_arm_match": match_best,
            "uplift_expected_net": float(rec["expected_net_value"].sum()),
        }
    )
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str))
    return rec


def _oracle_arm(df: pd.DataFrame) -> np.ndarray:
    spend = df["expected_spend"].to_numpy()
    nets = []
    for arm in ARMS:
        te = true_treatment_effect(df, arm)
        nets.append(te * spend * TAKE_RATE - incentive_cost(arm, spend))
    best = np.argmax(np.vstack(nets), axis=0)
    arms = np.array(ARMS)[best]
    best_net = np.vstack(nets)[best, np.arange(len(df))]
    return np.where(best_net > 0, arms, "no_offer")


if __name__ == "__main__":
    run()
