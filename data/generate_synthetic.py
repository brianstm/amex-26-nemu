"""Synthetic card-member, trip, and spend generator.

``ground_truth.csv`` is hidden from the estimator.
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
    ACCEPT_SIGMOID_K,
    ACCEPT_SIGMOID_MID,
    CASH_CATEGORY_WEIGHT,
    CASH_INTENSITY,
    CATEGORIES,
    CURRENCY,
    DISTRICT_TRIP_WEIGHTS,
    DISTRICTS,
    DOMESTIC_LOGNORMAL,
    FX_LCU_PER_USD,
    MCC,
    N_MEMBERS,
    N_TRIPS,
    ORIGIN_COUNTRIES,
    ORIGIN_WEIGHTS,
    OSM_MERCHANTS,
    OUTPUT_DIR,
    PRICE_LEVEL,
    RNG_SEED,
    SATURATED_ACCEPTANCE,
    SEGMENT_LOG_SHIFT,
    SEGMENT_WEIGHTS,
    SEGMENTS,
    TRANSPORT_OPERATORS,
    TRIP_TYPE_MULT,
)
from data.public_venues import FALLBACK_VENUES


def acceptance_capture_rate(
    density: np.ndarray | float,
    k: float = ACCEPT_SIGMOID_K,
    midpoint: float = ACCEPT_SIGMOID_MID,
) -> np.ndarray | float:
    """Share of willing demand captured at a given district density."""
    density = np.asarray(density, dtype=float)
    return 1.0 / (1.0 + np.exp(-k * (density - midpoint)))


def cash_capture_rate(cash_intensity: np.ndarray | float, category: str) -> np.ndarray | float:
    """Cash suppression, independent of acceptance density."""
    weight = CASH_CATEGORY_WEIGHT[category]
    return 1.0 - np.asarray(cash_intensity, dtype=float) * weight


def _merchant_pool(country: str, city: str, district: str, category: str) -> list[str]:
    """OSM / fallback venue names for a district and category."""
    key = f"{country}|{city}|{district}"
    osm = list((OSM_MERCHANTS.get(key) or {}).get(category) or [])
    fallback = list((FALLBACK_VENUES.get(key) or {}).get(category) or [])
    pool = osm + fallback
    if category == "transport":
        pool = pool + list(TRANSPORT_OPERATORS.get(country) or [])
    seen: set[str] = set()
    unique: list[str] = []
    for name in pool:
        if name not in seen:
            seen.add(name)
            unique.append(name)
    if unique:
        return unique
    return [f"{district} {category.title()}"]


def _pick_merchants(
    rng: np.random.Generator,
    countries: np.ndarray,
    cities: np.ndarray,
    districts: np.ndarray,
    categories: np.ndarray,
) -> np.ndarray:
    n = len(countries)
    out = np.empty(n, dtype=object)
    cache: dict[tuple[str, str, str, str], list[str]] = {}
    for i in range(n):
        key = (str(countries[i]), str(cities[i]), str(districts[i]), str(categories[i]))
        pool = cache.get(key)
        if pool is None:
            pool = _merchant_pool(*key)
            cache[key] = pool
        out[i] = pool[int(rng.integers(0, len(pool)))]
    return out


def _label_true_cause(
    acceptance_density: np.ndarray,
    cash_intensity: np.ndarray,
    true_spend: np.ndarray,
    observed_spend: np.ndarray,
    category: np.ndarray,
) -> np.ndarray:
    """Oracle cause labels for scoring Explain. Never used in fitting."""
    capture = np.divide(
        observed_spend,
        np.maximum(true_spend, 1e-9),
        out=np.zeros_like(observed_spend),
        where=true_spend > 1e-9,
    )
    no_demand = true_spend < np.quantile(true_spend, 0.12)
    no_acceptance = (~no_demand) & (acceptance_density < 0.35) & (observed_spend < 8.0)
    cash = (
        (~no_demand)
        & (~no_acceptance)
        & (cash_intensity >= 0.30)
        & (acceptance_density >= 0.30)
        & (capture < 0.75)
    )
    cause = np.full(len(true_spend), "rail_substitution", dtype=object)
    cause[no_demand] = "no_demand"
    cause[no_acceptance] = "no_acceptance"
    cause[cash] = "cash"
    return cause


def generate_members(rng: np.random.Generator, n: int = N_MEMBERS) -> pd.DataFrame:
    origins = rng.choice(ORIGIN_COUNTRIES, size=n, p=np.array(ORIGIN_WEIGHTS, dtype=float))
    segments = rng.choice(SEGMENTS, size=n, p=np.array(SEGMENT_WEIGHTS, dtype=float))
    log_shift = np.array([SEGMENT_LOG_SHIFT[s] for s in segments], dtype=float)

    members = pd.DataFrame(
        {
            "member_id": [f"M{i:05d}" for i in range(n)],
            "home_market": origins,
            "segment": segments,
        }
    )
    for cat, (mu, sigma) in DOMESTIC_LOGNORMAL.items():
        members[f"domestic_{cat}"] = rng.lognormal(mean=mu + log_shift, sigma=sigma)
    return members


def generate_trips(
    rng: np.random.Generator,
    members: pd.DataFrame,
    n_trips: int = N_TRIPS,
) -> pd.DataFrame:
    member_ids = members["member_id"].to_numpy()
    raw = rng.gamma(shape=1.4, scale=1.0, size=len(member_ids))
    probs = raw / raw.sum()
    chosen = rng.choice(member_ids, size=n_trips, p=probs)

    dist_idx = rng.choice(
        len(DISTRICTS),
        size=n_trips,
        p=DISTRICT_TRIP_WEIGHTS / DISTRICT_TRIP_WEIGHTS.sum(),
    )
    countries = [DISTRICTS[i][0] for i in dist_idx]
    cities = [DISTRICTS[i][1] for i in dist_idx]
    districts = [DISTRICTS[i][2] for i in dist_idx]
    densities = np.array([DISTRICTS[i][3] for i in dist_idx], dtype=float)

    start_offsets = rng.integers(0, 540, size=n_trips)
    start_dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(start_offsets, unit="D")
    nights = rng.integers(2, 15, size=n_trips)
    end_dates = start_dates + pd.to_timedelta(nights, unit="D")

    member_seg = members.set_index("member_id")["segment"].reindex(chosen).to_numpy()
    p_business = np.where(np.isin(member_seg, ["platinum", "centurion"]), 0.55, 0.28)
    trip_type = np.where(rng.random(n_trips) < p_business, "business", "leisure")

    trips = pd.DataFrame(
        {
            "trip_id": [f"T{i:06d}" for i in range(n_trips)],
            "member_id": chosen,
            "dest_country": countries,
            "dest_city": cities,
            "dest_district": districts,
            "acceptance_density": densities,
            "start_date": start_dates,
            "end_date": end_dates,
            "nights": nights,
            "trip_type": trip_type,
        }
    )
    trips["cash_intensity"] = trips["dest_country"].map(CASH_INTENSITY)
    trips["price_level"] = trips["dest_country"].map(PRICE_LEVEL)
    trips["dest_currency"] = trips["dest_country"].map(CURRENCY)
    trips["fx_lcu_per_usd"] = trips["dest_country"].map(FX_LCU_PER_USD)
    return trips


def generate_spend_panel(
    rng: np.random.Generator,
    members: pd.DataFrame,
    trips: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build trip × category spend. Returns (transactions, ground_truth, declines, full)."""
    panel = trips.merge(members, on="member_id", how="left")
    rows = []

    for cat in CATEGORIES:
        true = (
            panel[f"domestic_{cat}"].to_numpy()
            * (panel["nights"].to_numpy() / 7.0)
            * panel["price_level"].to_numpy()
            * np.array(
                [TRIP_TYPE_MULT[t][cat] for t in panel["trip_type"]],
                dtype=float,
            )
            * rng.lognormal(mean=0.0, sigma=0.25, size=len(panel))
        )
        f_acc = acceptance_capture_rate(panel["acceptance_density"].to_numpy())
        g_cash = cash_capture_rate(panel["cash_intensity"].to_numpy(), cat)

        p_zero = np.clip((1.0 - f_acc) ** 1.35 * 0.55, 0.0, 0.85)
        is_zero = rng.random(len(panel)) < p_zero

        noise = rng.lognormal(mean=0.0, sigma=0.12, size=len(panel))
        observed = true * f_acc * g_cash * noise
        observed = np.where(is_zero, 0.0, observed)
        observed = np.clip(observed, 0.0, true)

        f_sat = float(acceptance_capture_rate(SATURATED_ACCEPTANCE))
        cf_spend = true * f_sat * g_cash
        acceptance_leakage = np.clip(cf_spend - observed, 0.0, None)
        total_leakage = np.clip(true - observed, 0.0, None)
        cash_leakage = np.clip(total_leakage - acceptance_leakage, 0.0, None)

        cause = _label_true_cause(
            panel["acceptance_density"].to_numpy(),
            panel["cash_intensity"].to_numpy(),
            true,
            observed,
            np.array([cat] * len(panel)),
        )

        rows.append(
            pd.DataFrame(
                {
                    "trip_id": panel["trip_id"].to_numpy(),
                    "member_id": panel["member_id"].to_numpy(),
                    "home_market": panel["home_market"].to_numpy(),
                    "segment": panel["segment"].to_numpy(),
                    "dest_country": panel["dest_country"].to_numpy(),
                    "dest_city": panel["dest_city"].to_numpy(),
                    "dest_district": panel["dest_district"].to_numpy(),
                    "acceptance_density": panel["acceptance_density"].to_numpy(),
                    "cash_intensity": panel["cash_intensity"].to_numpy(),
                    "price_level": panel["price_level"].to_numpy(),
                    "nights": panel["nights"].to_numpy(),
                    "trip_type": panel["trip_type"].to_numpy(),
                    "category": cat,
                    "true_spend": true,
                    "observed_spend": observed,
                    "acceptance_leakage": acceptance_leakage,
                    "cash_leakage": cash_leakage,
                    "total_leakage": total_leakage,
                    "true_cause": cause,
                    "extensive_margin": is_zero.astype(int),
                }
            )
        )

    full = pd.concat(rows, ignore_index=True)
    assert (full["observed_spend"] >= -1e-9).all()
    assert (full["total_leakage"] >= -1e-9).all()
    assert (full["acceptance_leakage"] >= -1e-9).all()
    assert (full["observed_spend"] <= full["true_spend"] + 1e-6).all()

    captured = full.loc[full["observed_spend"] > 0.5].copy()
    merchants = _pick_merchants(
        rng,
        captured["dest_country"].to_numpy(),
        captured["dest_city"].to_numpy(),
        captured["dest_district"].to_numpy(),
        captured["category"].to_numpy(),
    )
    fx = captured["dest_country"].map(FX_LCU_PER_USD).astype(float)
    currency = captured["dest_country"].map(CURRENCY)
    amount_usd = captured["observed_spend"].to_numpy()
    amount_local = amount_usd * fx.to_numpy()
    int_ccy = currency.isin(["JPY", "KRW", "VND", "IDR"]).to_numpy()
    amount_local = np.where(int_ccy, np.round(amount_local), np.round(amount_local, 2))

    txns = pd.DataFrame(
        {
            "txn_id": [f"X{i:07d}" for i in range(len(captured))],
            "trip_id": captured["trip_id"].to_numpy(),
            "member_id": captured["member_id"].to_numpy(),
            "category": captured["category"].to_numpy(),
            "mcc": captured["category"].map(MCC).to_numpy(),
            "merchant_name": merchants,
            "dest_country": captured["dest_country"].to_numpy(),
            "dest_district": captured["dest_district"].to_numpy(),
            "currency": currency.to_numpy(),
            "amount_usd": np.round(amount_usd, 2),
            "amount_local": amount_local,
            "amount": np.round(amount_usd, 2),
        }
    )

    truth = full[
        [
            "trip_id",
            "member_id",
            "category",
            "dest_country",
            "dest_district",
            "true_spend",
            "observed_spend",
            "acceptance_leakage",
            "cash_leakage",
            "total_leakage",
            "true_cause",
            "extensive_margin",
        ]
    ].copy()

    decline_pool = full[
        (full["acceptance_density"] < 0.40)
        & (full["true_cause"] == "no_acceptance")
        & (full["true_spend"] > 20)
    ]
    n_declines = min(2_400, len(decline_pool))
    declines = decline_pool.sample(n=n_declines, random_state=int(rng.integers(0, 10_000)))
    declines = pd.DataFrame(
        {
            "decline_id": [f"D{i:05d}" for i in range(n_declines)],
            "trip_id": declines["trip_id"].to_numpy(),
            "member_id": declines["member_id"].to_numpy(),
            "category": declines["category"].to_numpy(),
            "dest_district": declines["dest_district"].to_numpy(),
            "reason": "card_not_accepted",
        }
    )
    return txns, truth, declines, full


def print_summary(
    members: pd.DataFrame,
    trips: pd.DataFrame,
    txns: pd.DataFrame,
    truth: pd.DataFrame,
    declines: pd.DataFrame,
) -> None:
    print("=" * 72)
    print("NEMU synthetic data — sanity check")
    print("=" * 72)
    print(f"members:           {len(members):>8,}")
    print(f"trips:             {len(trips):>8,}")
    print(f"observed txns:     {len(txns):>8,}   (positive Amex-captured only)")
    print(f"ground-truth rows: {len(truth):>8,}   (trip × category)")
    print(f"declined auths:    {len(declines):>8,}   (PU positives)")
    print()
    print("home market mix")
    print(members["home_market"].value_counts(normalize=True).mul(100).round(1).to_string())
    print()
    print("segment mix")
    print(members["segment"].value_counts(normalize=True).mul(100).round(1).to_string())
    print()
    print("domestic intensity (USD / month-equivalent)")
    print(
        members[[c for c in members.columns if c.startswith("domestic_")]]
        .describe(percentiles=[0.1, 0.5, 0.9])
        .round(1)
        .to_string()
    )
    print()
    print("trips by destination country")
    print(trips["dest_country"].value_counts().to_string())
    print()
    print("acceptance_density by district (should span sparse → saturated)")
    dens = (
        trips.groupby(["dest_country", "dest_district"], as_index=False)["acceptance_density"]
        .mean()
        .sort_values("acceptance_density")
    )
    print(dens.head(8).to_string(index=False))
    print("...")
    print(dens.tail(8).to_string(index=False))
    print()
    print("cash_intensity by country")
    print(
        trips.groupby("dest_country")["cash_intensity"]
        .mean()
        .sort_values(ascending=False)
        .round(2)
        .to_string()
    )
    print()
    print("spend + leakage (USD)")
    agg = {
        "true_spend": truth["true_spend"].sum(),
        "observed_spend": truth["observed_spend"].sum(),
        "total_leakage": truth["total_leakage"].sum(),
        "acceptance_leakage": truth["acceptance_leakage"].sum(),
        "cash_leakage": truth["cash_leakage"].sum(),
    }
    for k, v in agg.items():
        print(f"  {k:<22} ${v:,.0f}")
    capture = agg["observed_spend"] / agg["true_spend"]
    print(f"  observed capture rate   {capture:.1%}")
    print()
    print("true cause mix (oracle, hidden from the model)")
    print(truth["true_cause"].value_counts(normalize=True).mul(100).round(1).to_string())
    print()
    print("mean observed vs true spend by acceptance bucket")
    buckets = pd.cut(
        truth["dest_district"].map(
            trips.drop_duplicates("dest_district").set_index("dest_district")["acceptance_density"]
        ),
        bins=[0, 0.3, 0.6, 0.85, 1.01],
        labels=["sparse <0.3", "mid 0.3-0.6", "high 0.6-0.85", "saturated >0.85"],
    )
    tmp = truth.copy()
    tmp["bucket"] = buckets
    print(
        tmp.groupby("bucket", observed=True)[["true_spend", "observed_spend", "acceptance_leakage"]]
        .mean()
        .round(1)
        .to_string()
    )
    print()
    print("Assertions passed: leakage >= 0, observed <= true, files written to outputs/")


def generate(seed: int = RNG_SEED) -> dict[str, Path]:
    rng = np.random.default_rng(seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    members = generate_members(rng)
    trips = generate_trips(rng, members)
    txns, truth, declines, _full = generate_spend_panel(rng, members, trips)

    trip_cols = [
        "trip_id",
        "member_id",
        "dest_country",
        "dest_city",
        "dest_district",
        "acceptance_density",
        "cash_intensity",
        "price_level",
        "dest_currency",
        "fx_lcu_per_usd",
        "start_date",
        "end_date",
        "nights",
        "trip_type",
    ]
    paths = {
        "members": OUTPUT_DIR / "members.csv",
        "trips": OUTPUT_DIR / "trips.csv",
        "transactions": OUTPUT_DIR / "transactions.csv",
        "ground_truth": OUTPUT_DIR / "ground_truth.csv",
        "declines": OUTPUT_DIR / "declines.csv",
    }
    members.to_csv(paths["members"], index=False)
    trips[trip_cols].to_csv(paths["trips"], index=False)
    txns.to_csv(paths["transactions"], index=False)
    truth.to_csv(paths["ground_truth"], index=False)
    declines.to_csv(paths["declines"], index=False)

    print_summary(members, trips, txns, truth, declines)
    for name, path in paths.items():
        print(f"  wrote {path} ({path.stat().st_size / 1_048_576:.2f} MB)")
    return paths


if __name__ == "__main__":
    generate()
