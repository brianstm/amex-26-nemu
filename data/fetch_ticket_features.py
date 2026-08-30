"""Fetch public ticket fields: ISO FX rates + named OSM merchants.

Writes ``data/external/real_features.json``. Spend and acceptance stay simulated.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

from data.calibrate_external import (
    ISO2,
    OVERPASS_ENDPOINTS,
    WB_DISPLAY,
    _get_json,
    _wb_latest,
)

ROOT = Path(__file__).resolve().parents[1]
CAL_PATH = Path(__file__).resolve().parent / "external" / "calibration.json"
OUT_PATH = Path(__file__).resolve().parent / "external" / "real_features.json"

CURRENCY = {
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
}

MCC = {
    "dining": 5812,
    "retail": 5311,
    "transport": 4111,
    "lodging": 7011,
}

TRANSPORT_OPERATORS = {
    "Singapore": ["SMRT", "SBS Transit", "Grab", "ComfortDelGro", "Changi Airport"],
    "Hong Kong": ["MTR", "Citybus", "Star Ferry", "Hong Kong Tramways", "KMB"],
    "Japan": ["JR East", "Tokyo Metro", "Toei Subway", "Osaka Metro", "Keihan"],
    "South Korea": ["Seoul Metro", "Korail", "AREX", "Kakao T", "Busan Metro"],
    "Australia": ["Transport NSW", "Metro Trains Melbourne", "Uber", "Sydney Ferries"],
    "Thailand": ["BTS Skytrain", "MRT Blue Line", "Grab", "Airport Rail Link", "BMTA"],
    "Vietnam": ["Grab", "Hanoi Metro", "Saigon Metro", "Mai Linh Taxi", "Vietnam Airlines"],
    "Indonesia": ["TransJakarta", "MRT Jakarta", "Grab", "Blue Bird", "KAI Commuter"],
    "Malaysia": ["Rapid KL", "Grab", "KLIA Ekspres", "Prasarana", "ETS"],
    "France": ["RATP", "SNCF", "Vélib'", "Air France", "Optile"],
    "Italy": ["ATAC", "Trenitalia", "ATM Milano", "Italo", "ANM Napoli"],
    "United States": ["MTA", "Uber", "Lyft", "Amtrak", "United Airlines"],
}


def fetch_fx() -> tuple[dict[str, float], dict]:
    """Official exchange rate, LCU per USD (PA.NUS.FCRF)."""
    iso = list(ISO2.values())
    raw = _wb_latest("PA.NUS.FCRF", iso, mrv=6)
    fx: dict[str, float] = {}
    meta: dict = {}
    for internal, display in WB_DISPLAY.items():
        if display not in raw:
            continue
        year, val = raw[display]
        fx[internal] = round(float(val), 6)
        meta[internal] = {"year": year, "lcu_per_usd": fx[internal], "iso4217": CURRENCY[internal]}
    if "United States" not in fx:
        fx["United States"] = 1.0
        meta["United States"] = {"year": None, "lcu_per_usd": 1.0, "iso4217": "USD"}
    return fx, meta


def _classify_osm(tags: dict) -> str | None:
    amenity = (tags.get("amenity") or "").lower()
    tourism = (tags.get("tourism") or "").lower()
    shop = tags.get("shop")
    railway = (tags.get("railway") or "").lower()
    public_transport = (tags.get("public_transport") or "").lower()
    if amenity in {"restaurant", "cafe", "fast_food", "bar", "food_court"}:
        return "dining"
    if tourism in {"hotel", "guest_house", "hostel", "motel"}:
        return "lodging"
    if shop:
        return "retail"
    if railway in {"station", "halt", "subway_entrance"} or amenity in {
        "taxi",
        "bus_station",
        "ferry_terminal",
    } or public_transport in {"station", "stop_position"}:
        return "transport"
    return None


def _poi_name(tags: dict) -> str | None:
    for key in ("name:en", "name", "name:id", "name:th", "name:vi", "name:ko", "name:ja"):
        val = tags.get(key)
        if val and str(val).strip():
            return str(val).strip()
    return None


def fetch_osm_merchants(districts: list[dict], per_cat: int = 25) -> dict[str, dict[str, list[str]]]:
    out: dict[str, dict[str, list[str]]] = {}
    for row in districts:
        key = f"{row['country']}|{row['city']}|{row['district']}"
        osm = row.get("osm") or {}
        lat, lon = osm.get("lat"), osm.get("lon")
        buckets: dict[str, list[str]] = {c: [] for c in MCC}
        if lat is None or lon is None:
            out[key] = buckets
            print(f"  skip {key} (no coordinates)", flush=True)
            continue
        radius = int(osm.get("radius_m") or 1200)
        oql = (
            f"[out:json][timeout:50];\n"
            f"(\n"
            f'  node["amenity"~"^(restaurant|cafe|fast_food|bar)$"]["name"](around:{radius},{lat},{lon});\n'
            f'  node["tourism"~"^(hotel|guest_house|hostel)$"]["name"](around:{radius},{lat},{lon});\n'
            f'  node["shop"]["name"](around:{radius},{lat},{lon});\n'
            f'  node["railway"~"^(station|halt)$"]["name"](around:{radius},{lat},{lon});\n'
            f'  node["amenity"~"^(taxi|bus_station|ferry_terminal)$"]["name"](around:{radius},{lat},{lon});\n'
            f");\n"
            f"out tags 80;\n"
        )
        body = urllib.parse.urlencode({"data": oql}).encode()
        payload = None
        for endpoint in OVERPASS_ENDPOINTS:
            try:
                payload = _get_json(endpoint, data=body, timeout=55)
                break
            except Exception as exc:
                print(f"  overpass {row['district']}: {exc}", file=sys.stderr)
                time.sleep(2.0)
        seen: dict[str, set[str]] = {c: set() for c in MCC}
        for el in (payload or {}).get("elements") or []:
            tags = el.get("tags") or {}
            cat = _classify_osm(tags)
            name = _poi_name(tags)
            if not cat or not name:
                continue
            if name in seen[cat] or len(buckets[cat]) >= per_cat:
                continue
            seen[cat].add(name)
            buckets[cat].append(name)
        out[key] = buckets
        counts = {c: len(v) for c, v in buckets.items()}
        print(f"  {row['district']:16} {counts}", flush=True)
        time.sleep(0.4)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-osm", action="store_true")
    args = parser.parse_args()

    print("Fetching official FX (PA.NUS.FCRF, LCU per USD)…")
    fx, fx_meta = fetch_fx()

    merchants: dict[str, dict[str, list[str]]] = {}
    if not args.skip_osm:
        cal = json.loads(CAL_PATH.read_text())
        print("Fetching named OSM POIs per district…")
        try:
            merchants = fetch_osm_merchants(cal["districts"])
        except Exception as exc:
            print(f"OSM merchant fetch failed: {exc}", file=sys.stderr)
            merchants = {}

    payload = {
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "notes": (
            "Public ticket fields only. Spend, trips, and acceptance remain simulated. "
            "Merchant names are OSM POIs (ODbL); currency codes are ISO 4217; "
            "FX is World Bank PA.NUS.FCRF; MCC is ISO 18245."
        ),
        "currency": CURRENCY,
        "mcc": MCC,
        "fx_lcu_per_usd": fx,
        "fx_meta": fx_meta,
        "transport_operators": TRANSPORT_OPERATORS,
        "osm_merchants": merchants,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"\nWrote {OUT_PATH}")
    print("FX LCU per USD:")
    for k, v in sorted(fx.items()):
        print(f"  {k:16} {CURRENCY.get(k,'?'):4} {v:g}")


if __name__ == "__main__":
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    main()
