"""Market primitives for the synthetic DGP.

Country knobs load from ``data/external/calibration.json``.
Ticket fields (ISO currency, FX, MCC, OSM names) load from
``data/external/real_features.json``. No American Express data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
_CALIBRATION_PATH = Path(__file__).resolve().parent / "external" / "calibration.json"


def _load_calibration() -> dict:
    if not _CALIBRATION_PATH.exists():
        raise FileNotFoundError(
            f"Missing {_CALIBRATION_PATH}. Run: python -m data.calibrate_external"
        )
    return json.loads(_CALIBRATION_PATH.read_text())


_CAL = _load_calibration()

CATEGORIES = ("dining", "retail", "transport", "lodging")

ORIGIN_COUNTRIES = ("Singapore", "Hong Kong", "Australia", "Japan", "United States")
ORIGIN_WEIGHTS = (0.40, 0.20, 0.15, 0.15, 0.10)

PRICE_LEVEL: dict[str, float] = {
    k: float(v) for k, v in _CAL["price_level"].items()
}

CASH_INTENSITY: dict[str, float] = {
    k: float(v) for k, v in _CAL["cash_intensity"].items()
}

CASH_CATEGORY_WEIGHT: dict[str, float] = {
    "dining": 1.00,
    "retail": 0.85,
    "transport": 0.50,
    "lodging": 0.25,
}

DISTRICTS: list[tuple[str, str, str, float]] = [
    (
        str(row["country"]),
        str(row["city"]),
        str(row["district"]),
        float(row["acceptance_density"]),
    )
    for row in _CAL["districts"]
]

DISTRICT_TRIP_WEIGHTS = np.array(_CAL["district_trip_weights"], dtype=float)

assert len(DISTRICT_TRIP_WEIGHTS) == len(DISTRICTS)
assert set(PRICE_LEVEL) >= {d[0] for d in DISTRICTS}
assert set(CASH_INTENSITY) >= {d[0] for d in DISTRICTS}

SEGMENTS = ("core", "gold", "platinum", "centurion")
SEGMENT_WEIGHTS = (0.55, 0.28, 0.14, 0.03)
SEGMENT_LOG_SHIFT = {"core": 0.0, "gold": 0.35, "platinum": 0.75, "centurion": 1.20}

DOMESTIC_LOGNORMAL = {
    "dining": (np.log(420.0), 0.70),
    "retail": (np.log(310.0), 0.80),
    "transport": (np.log(170.0), 0.60),
    "lodging": (np.log(190.0), 0.90),
}

TRIP_TYPE_MULT = {
    "leisure": {"dining": 1.30, "retail": 1.40, "transport": 0.90, "lodging": 1.00},
    "business": {"dining": 0.85, "retail": 0.55, "transport": 1.25, "lodging": 1.45},
}

ACCEPT_SIGMOID_K = 8.0
ACCEPT_SIGMOID_MID = 0.40
SATURATED_ACCEPTANCE = 0.95

_FEATURES_PATH = Path(__file__).resolve().parent / "external" / "real_features.json"


def _load_real_features() -> dict:
    if not _FEATURES_PATH.exists():
        return {}
    return json.loads(_FEATURES_PATH.read_text())


_FEAT = _load_real_features()

CURRENCY: dict[str, str] = _FEAT.get(
    "currency",
    {
        "Singapore": "SGD",
        "Hong Kong": "HKD",
        "Japan": "JPY",
        "South Korea": "KRW",
        "Australia": "AUD",
        "Thailand": "THB",
        "Vietnam": "VND",
        "Indonesia": "IDR",
        "Malaysia": "MYR",
        "France": "EUR",
        "Italy": "EUR",
        "United States": "USD",
    },
)

_FX_FALLBACK = {
    "Singapore": 1.35,
    "Hong Kong": 7.80,
    "Japan": 150.0,
    "South Korea": 1350.0,
    "Australia": 1.55,
    "Thailand": 36.0,
    "Vietnam": 25000.0,
    "Indonesia": 16000.0,
    "Malaysia": 4.45,
    "France": 0.92,
    "Italy": 0.92,
    "United States": 1.0,
}
FX_LCU_PER_USD: dict[str, float] = {
    k: float(v) for k, v in (_FEAT.get("fx_lcu_per_usd") or _FX_FALLBACK).items()
}
if not FX_LCU_PER_USD:
    FX_LCU_PER_USD = dict(_FX_FALLBACK)

MCC: dict[str, int] = {
    str(k): int(v) for k, v in (_FEAT.get("mcc") or {
        "dining": 5812,
        "retail": 5311,
        "transport": 4111,
        "lodging": 7011,
    }).items()
}

OSM_MERCHANTS: dict[str, dict[str, list[str]]] = _FEAT.get("osm_merchants") or {}
TRANSPORT_OPERATORS: dict[str, list[str]] = _FEAT.get("transport_operators") or {}

N_MEMBERS = 5_000
N_TRIPS = 15_000
RNG_SEED = 26
