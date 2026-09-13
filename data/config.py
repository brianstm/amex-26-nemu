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

# Representative photos (public Wikimedia Commons) for the map hover cards.
# Built by data/fetch_place_images.py; falls back to {} if absent.
_IMAGES_PATH = Path(__file__).resolve().parent / "external" / "place_images.json"
_IMAGES = json.loads(_IMAGES_PATH.read_text()) if _IMAGES_PATH.exists() else {}
DISTRICT_IMAGES: dict[str, str] = _IMAGES.get("districts", {})
MERCHANT_IMAGES: dict[str, str] = _IMAGES.get("merchants", {})
CATEGORY_IMAGES: dict[str, str] = _IMAGES.get("categories", {})

DISTRICT_TRIP_WEIGHTS = np.array(_CAL["district_trip_weights"], dtype=float)

assert len(DISTRICT_TRIP_WEIGHTS) == len(DISTRICTS)
assert set(PRICE_LEVEL) >= {d[0] for d in DISTRICTS}
assert set(CASH_INTENSITY) >= {d[0] for d in DISTRICTS}

# --- Region grouping (destination continents/blocs) -------------------------
# Amex already knows the region; NEMU's job is the merchant/member layer under
# it. SEA is this competition's focus. LATAM/US are shown in the UI as blocs we
# do not yet have corridor data for.
REGION: dict[str, str] = {
    "Singapore": "SEA",
    "Malaysia": "SEA",
    "Thailand": "SEA",
    "Vietnam": "SEA",
    "Indonesia": "SEA",
    "Japan": "APAC",
    "South Korea": "APAC",
    "Hong Kong": "APAC",
    "Australia": "APAC",
    "France": "EMEA",
    "Italy": "EMEA",
    "United States": "US",
}
REGION_ORDER = ["SEA", "APAC", "EMEA", "LATAM", "US"]

# Approximate district-centre coordinates (lat, lon) for the map view. Public
# geographic reference points, not Amex data.
DISTRICT_COORDS: dict[str, tuple[float, float]] = {
    "Shibuya": (35.6595, 139.7005),
    "Ginza": (35.6717, 139.7650),
    "Gion": (35.0037, 135.7788),
    "Namba": (34.6659, 135.5010),
    "Gangnam": (37.4979, 127.0276),
    "Myeongdong": (37.5637, 126.9850),
    "Haeundae": (35.1587, 129.1604),
    "Sukhumvit": (13.7370, 100.5600),
    "Khao San": (13.7590, 100.4977),
    "Old City": (18.7883, 98.9853),
    "Patong": (7.8965, 98.2960),
    "Old Quarter": (21.0340, 105.8510),
    "Ba Dinh": (21.0350, 105.8140),
    "District 1": (10.7769, 106.7009),
    "Han River": (16.0710, 108.2240),
    "SCBD": (-6.2247, 106.8090),
    "Seminyak": (-8.6910, 115.1680),
    "Ubud": (-8.5069, 115.2625),
    "Malioboro": (-7.7930, 110.3660),
    "KLCC": (3.1580, 101.7120),
    "Bukit Bintang": (3.1466, 101.7110),
    "George Town": (5.4140, 100.3290),
    "Orchard": (1.3040, 103.8320),
    "Chinatown": (1.2830, 103.8440),
    "Central": (22.2810, 114.1580),
    "Mong Kok": (22.3190, 114.1690),
    "CBD": (-33.8688, 151.2093),
    "Laneways": (-37.8136, 144.9631),
    "Le Marais": (48.8590, 2.3620),
    "Montmartre": (48.8867, 2.3431),
    "Centro Storico": (41.8990, 12.4770),
    "Duomo": (43.7731, 11.2560),
}

# Per-visit spend band (USD) per category, before scaling by destination
# price_level. Used to blend/clamp the data-derived p25-p75 so thin-sample
# merchants still show a sensible "$low-high / visit" range.
PER_VISIT_BASE: dict[str, tuple[float, float]] = {
    "dining": (18.0, 34.0),
    "retail": (15.0, 40.0),
    "transport": (8.0, 22.0),
    "lodging": (60.0, 140.0),
}

# Sub-category keyword rules for the incentive drill
# (category -> ordered list of (sub_category, [name keywords])). First keyword
# hit wins; if none match, an amount-tier fallback is used (see
# match/merchant_targets_detail.py).
SUBCATEGORY_KEYWORDS: dict[str, list[tuple[str, list[str]]]] = {
    "dining": [
        ("Cafe & Coffee", ["cafe", "coffee", "starbucks", "kopi", "doutor",
                             "veloce", "espresso", "tea", "bakery"]),
        ("Fast Food", ["mcdonald", "burger", "kfc", "lotteria", "mos ",
                        "saizeriya", "gusto", "sukiya", "matsuya", "yoshinoya",
                        "pizza", "subway", "jollibee"]),
        ("Food Delivery", ["delivery", "grabfood", "foodpanda", "gojek",
                            "deliveroo", "uber eats"]),
    ],
    "retail": [
        ("Convenience", ["7-eleven", "familymart", "lawson", "ministop",
                          "circle k", "convenience", "mart"]),
        ("Electronics", ["bic camera", "yodobashi", "electronics", "camera",
                         "au shop", "apple", "samsung", "galaxy"]),
        ("Luxury", ["rolex", "ralph lauren", "gucci", "prada", "louis",
                    "chanel", "hermes", "kitamura"]),
        ("Apparel", ["uniqlo", "zara", "forever 21", "h&m", "skechers",
                     "cap", "wear", "aland", "toni & guy"]),
    ],
    "transport": [
        ("Ride-hail & Taxi", ["grab", "gojek", "taxi", "uber", "bluebird",
                               "cab"]),
        ("Rail & Metro", ["metro", "mrt", "rail", "line", "jr", "subway",
                          "station", "hachiman", "shibuya"]),
    ],
    "lodging": [
        ("Budget & Hostel", ["hostel", "guesthouse", "inn", "capsule",
                             "budget", "mets"]),
        ("Hotel", ["hotel", "resort", "ryokan", "washington", "tobu"]),
    ],
}

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
