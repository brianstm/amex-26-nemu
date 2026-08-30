"""Refresh country knobs from World Bank + optional OSM amenity counts.

Writes ``data/external/calibration.json``. Acceptance density stays designed.
"""

from __future__ import annotations

import argparse
import json
import ssl
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import certifi

_SSL_CTX = ssl.create_default_context(cafile=certifi.where())

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_PATH = Path(__file__).resolve().parent / "external" / "calibration.json"

UA = "NEMU-hackathon-calibrate/1.0 (academic prototype)"

ISO2 = {
    "Singapore": "SG",
    "Hong Kong": "HK",
    "Japan": "JP",
    "South Korea": "KR",
    "Australia": "AU",
    "Thailand": "TH",
    "Vietnam": "VN",
    "Indonesia": "ID",
    "Malaysia": "MY",
    "France": "FR",
    "Italy": "IT",
    "United States": "US",
}

WB_DISPLAY = {
    "Singapore": "Singapore",
    "Hong Kong": "Hong Kong SAR, China",
    "Japan": "Japan",
    "South Korea": "Korea, Rep.",
    "Australia": "Australia",
    "Thailand": "Thailand",
    "Vietnam": "Viet Nam",
    "Indonesia": "Indonesia",
    "Malaysia": "Malaysia",
    "France": "France",
    "Italy": "Italy",
    "United States": "United States",
}

DESIGNED_DISTRICTS: list[tuple[str, str, str, float]] = [
    ("Japan", "Tokyo", "Shibuya", 0.94),
    ("Japan", "Tokyo", "Ginza", 0.91),
    ("Japan", "Kyoto", "Gion", 0.48),
    ("Japan", "Osaka", "Namba", 0.72),
    ("South Korea", "Seoul", "Gangnam", 0.93),
    ("South Korea", "Seoul", "Myeongdong", 0.81),
    ("South Korea", "Busan", "Haeundae", 0.55),
    ("Thailand", "Bangkok", "Sukhumvit", 0.68),
    ("Thailand", "Bangkok", "Khao San", 0.22),
    ("Thailand", "Chiang Mai", "Old City", 0.31),
    ("Thailand", "Phuket", "Patong", 0.40),
    ("Vietnam", "Hanoi", "Old Quarter", 0.16),
    ("Vietnam", "Hanoi", "Ba Dinh", 0.29),
    ("Vietnam", "Ho Chi Minh City", "District 1", 0.38),
    ("Vietnam", "Da Nang", "Han River", 0.21),
    ("Indonesia", "Jakarta", "SCBD", 0.44),
    ("Indonesia", "Bali", "Seminyak", 0.36),
    ("Indonesia", "Bali", "Ubud", 0.14),
    ("Indonesia", "Yogyakarta", "Malioboro", 0.19),
    ("Malaysia", "Kuala Lumpur", "KLCC", 0.77),
    ("Malaysia", "Kuala Lumpur", "Bukit Bintang", 0.61),
    ("Malaysia", "Penang", "George Town", 0.33),
    ("Singapore", "Singapore", "Orchard", 0.96),
    ("Singapore", "Singapore", "Chinatown", 0.78),
    ("Hong Kong", "Hong Kong", "Central", 0.95),
    ("Hong Kong", "Hong Kong", "Mong Kok", 0.70),
    ("Australia", "Sydney", "CBD", 0.92),
    ("Australia", "Melbourne", "Laneways", 0.88),
    ("France", "Paris", "Le Marais", 0.74),
    ("France", "Paris", "Montmartre", 0.41),
    ("Italy", "Rome", "Centro Storico", 0.52),
    ("Italy", "Florence", "Duomo", 0.38),
]

NOMINATIM_Q = {
    ("Japan", "Tokyo", "Shibuya"): "Shibuya, Tokyo, Japan",
    ("Japan", "Tokyo", "Ginza"): "Ginza, Chuo, Tokyo, Japan",
    ("Japan", "Kyoto", "Gion"): "Gion, Higashiyama, Kyoto, Japan",
    ("Japan", "Osaka", "Namba"): "Namba, Osaka, Japan",
    ("South Korea", "Seoul", "Gangnam"): "Gangnam-gu, Seoul, South Korea",
    ("South Korea", "Seoul", "Myeongdong"): "Myeongdong, Seoul, South Korea",
    ("South Korea", "Busan", "Haeundae"): "Haeundae Beach, Busan, South Korea",
    ("Thailand", "Bangkok", "Sukhumvit"): "Asok, Sukhumvit, Bangkok, Thailand",
    ("Thailand", "Bangkok", "Khao San"): "Khao San Road, Bangkok, Thailand",
    ("Thailand", "Chiang Mai", "Old City"): "Si Phum, Chiang Mai, Thailand",
    ("Thailand", "Phuket", "Patong"): "Patong Beach, Phuket, Thailand",
    ("Vietnam", "Hanoi", "Old Quarter"): "Hanoi Old Quarter, Vietnam",
    ("Vietnam", "Hanoi", "Ba Dinh"): "Ba Dinh Square, Hanoi, Vietnam",
    ("Vietnam", "Ho Chi Minh City", "District 1"): "Ben Thanh, District 1, Ho Chi Minh City, Vietnam",
    ("Vietnam", "Da Nang", "Han River"): "Han River Bridge, Da Nang, Vietnam",
    ("Indonesia", "Jakarta", "SCBD"): "SCBD, Jakarta, Indonesia",
    ("Indonesia", "Bali", "Seminyak"): "Seminyak, Bali, Indonesia",
    ("Indonesia", "Bali", "Ubud"): "Ubud, Bali, Indonesia",
    ("Indonesia", "Yogyakarta", "Malioboro"): "Jalan Malioboro, Yogyakarta",
    ("Malaysia", "Kuala Lumpur", "KLCC"): "KLCC, Kuala Lumpur, Malaysia",
    ("Malaysia", "Kuala Lumpur", "Bukit Bintang"): "Bukit Bintang, Kuala Lumpur, Malaysia",
    ("Malaysia", "Penang", "George Town"): "George Town, Penang, Malaysia",
    ("Singapore", "Singapore", "Orchard"): "Orchard Road, Singapore",
    ("Singapore", "Singapore", "Chinatown"): "Chinatown, Singapore",
    ("Hong Kong", "Hong Kong", "Central"): "Central, Hong Kong",
    ("Hong Kong", "Hong Kong", "Mong Kok"): "Mong Kok, Hong Kong",
    ("Australia", "Sydney", "CBD"): "Sydney CBD, New South Wales, Australia",
    ("Australia", "Melbourne", "Laneways"): "Degraves Street, Melbourne, Australia",
    ("France", "Paris", "Le Marais"): "Le Marais, Paris, France",
    ("France", "Paris", "Montmartre"): "Montmartre, Paris, France",
    ("Italy", "Rome", "Centro Storico"): "Pantheon, Rome, Italy",
    ("Italy", "Florence", "Duomo"): "Piazza del Duomo, Florence, Italy",
}

OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)

CASH_FLOOR = 0.04
CASH_CEIL = 0.85


def _get_json(url: str, data: bytes | None = None, timeout: int = 45) -> object:
    req = urllib.request.Request(url, data=data, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as resp:
        return json.loads(resp.read().decode())


def _wb_latest(
    indicator: str,
    iso2_list: list[str],
    *,
    source: int | None = None,
    date: str | None = None,
    mrv: int = 8,
) -> dict[str, tuple[int, float]]:
    """Map World Bank country display name → (year, value) for latest non-null."""
    countries = ";".join(iso2_list)
    params = {"format": "json", "per_page": "400"}
    if source is not None:
        params["source"] = str(source)
    if date is not None:
        params["date"] = date
    else:
        params["mrv"] = str(mrv)
    url = (
        f"https://api.worldbank.org/v2/country/{countries}/indicator/{indicator}"
        f"?{urllib.parse.urlencode(params)}"
    )
    payload = _get_json(url)
    if not isinstance(payload, list) or len(payload) < 2:
        raise RuntimeError(f"Unexpected World Bank payload for {indicator}: {payload!r}")
    if isinstance(payload[0], dict) and payload[0].get("message"):
        raise RuntimeError(f"World Bank error for {indicator}: {payload[0]['message']}")
    latest: dict[str, tuple[int, float]] = {}
    for row in payload[1] or []:
        if row.get("value") is None:
            continue
        name = row["country"]["value"]
        year = int(row["date"])
        if name not in latest or year > latest[name][0]:
            latest[name] = (year, float(row["value"]))
    return latest


def fetch_price_levels() -> tuple[dict[str, float], dict[str, dict]]:
    iso = list(ISO2.values())
    raw = _wb_latest("PA.NUS.GDP.PLI", iso, mrv=5)
    by_internal: dict[str, float] = {}
    meta: dict[str, dict] = {}
    sg_name = WB_DISPLAY["Singapore"]
    if sg_name not in raw:
        raise RuntimeError(f"Missing Singapore PLI. Got {raw!r}")
    sg_year, sg_pli = raw[sg_name]
    for internal, display in WB_DISPLAY.items():
        if display not in raw:
            continue
        year, pli = raw[display]
        by_internal[internal] = round(pli / sg_pli, 4)
        meta[internal] = {
            "year": year,
            "pli_us100": round(pli, 4),
            "relative_to_singapore": by_internal[internal],
        }
    meta["_singapore_pli"] = {"year": sg_year, "pli_us100": round(sg_pli, 4)}
    dest = {k: v for k, v in by_internal.items() if k != "United States"}
    return dest, meta


def fetch_cash_intensity() -> tuple[dict[str, float], dict[str, dict]]:
    iso = list(ISO2.values())
    used = _wb_latest("fin25e2", iso, source=28, mrv=5)
    unused = _wb_latest("fin25e2b", iso, source=28, mrv=5)
    digital = _wb_latest("g20.made", iso, source=28, mrv=8)

    cash: dict[str, float] = {}
    meta: dict[str, dict] = {}
    for internal, display in WB_DISPLAY.items():
        if internal == "United States":
            continue
        instore_yes = used.get(display)
        instore_no = unused.get(display)
        digi = digital.get(display)
        detail: dict = {}
        value: float | None = None
        source = ""
        if (
            instore_yes is not None
            and instore_no is not None
            and (instore_yes[1] + instore_no[1]) > 0
        ):
            value = instore_no[1] / (instore_yes[1] + instore_no[1])
            source = "findex_instore_noncard_share"
            detail = {
                "fin25e2_used_phone_or_card_instore": {
                    "year": instore_yes[0],
                    "pct": round(instore_yes[1], 4),
                },
                "fin25e2b_did_not_use_phone_or_card_instore": {
                    "year": instore_no[0],
                    "pct": round(instore_no[1], 4),
                },
            }
        elif digi is not None:
            value = 1.0 - digi[1] / 100.0
            source = "findex_one_minus_g20_made"
            detail = {
                "g20.made_digital_payment_pct": {
                    "year": digi[0],
                    "pct": round(digi[1], 4),
                }
            }
        else:
            raise RuntimeError(f"No Findex cash proxy for {internal} ({display})")
        clipped = min(CASH_CEIL, max(CASH_FLOOR, value))
        cash[internal] = round(clipped, 4)
        meta[internal] = {
            "source": source,
            "raw": round(value, 4),
            "clipped": cash[internal],
            **detail,
        }
    return cash, meta


def fetch_arrivals_2019() -> tuple[dict[str, float], dict[str, dict]]:
    iso = [v for k, v in ISO2.items() if k != "United States"]
    raw = _wb_latest("ST.INT.ARVL", iso, date="2015:2019")
    arrivals: dict[str, float] = {}
    meta: dict[str, dict] = {}
    for internal, display in WB_DISPLAY.items():
        if internal == "United States":
            continue
        if display not in raw:
            raise RuntimeError(f"Missing tourist arrivals for {internal}")
        year, val = raw[display]
        arrivals[internal] = val
        meta[internal] = {"year": year, "arrivals": val}
    return arrivals, meta


def geocode(query: str) -> tuple[float, float, str] | None:
    url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode(
        {"q": query, "format": "json", "limit": 1}
    )
    time.sleep(1.05)
    hits = _get_json(url, timeout=30)
    if not isinstance(hits, list) or not hits:
        return None
    hit = hits[0]
    return float(hit["lat"]), float(hit["lon"]), str(hit.get("display_name", query))


def _overpass_oql_one(lat: float, lon: float, radius_m: int) -> str:
    return (
        f'node["amenity"~"^(restaurant|cafe|fast_food|bar)$"]'
        f"(around:{radius_m},{lat},{lon})->.food;\n"
        f'node["tourism"~"^(hotel|guest_house|hostel)$"]'
        f"(around:{radius_m},{lat},{lon})->.stay;\n"
        f"(.food; .stay;);\n"
        f"out count;\n"
    )


def _parse_overpass_counts(payload: object) -> list[int]:
    if not isinstance(payload, dict):
        return []
    counts = []
    for el in payload.get("elements") or []:
        tags = el.get("tags") or {}
        if "total" in tags:
            counts.append(int(tags["total"]))
    return counts


def overpass_counts_batch(
    coords: list[tuple[float, float]],
    radius_m: int = 1200,
) -> list[int | None]:
    """One round-trip: emit `out count` after each district's node set."""
    parts = ["[out:json][timeout:180];"]
    for i, (lat, lon) in enumerate(coords):
        parts.append(
            f'node["amenity"~"^(restaurant|cafe|fast_food|bar)$"]'
            f"(around:{radius_m},{lat},{lon})->.f{i};\n"
            f'node["tourism"~"^(hotel|guest_house|hostel)$"]'
            f"(around:{radius_m},{lat},{lon})->.s{i};\n"
            f"(.f{i}; .s{i};);\n"
            f"out count;\n"
        )
    body = urllib.parse.urlencode({"data": "".join(parts)}).encode()
    last_err: Exception | None = None
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            payload = _get_json(endpoint, data=body, timeout=120)
            counts = _parse_overpass_counts(payload)
            if len(counts) == len(coords):
                return counts  # type: ignore[return-value]
            last_err = RuntimeError(f"expected {len(coords)} counts, got {len(counts)}")
        except Exception as exc:
            last_err = exc
            time.sleep(3.0)
            continue
    print(f"  batched overpass failed: {last_err}", file=sys.stderr)
    return [None] * len(coords)


def overpass_amenity_count(lat: float, lon: float, radius_m: int = 1200) -> int | None:
    oql = "[out:json][timeout:45];\n" + _overpass_oql_one(lat, lon, radius_m)
    body = urllib.parse.urlencode({"data": oql}).encode()
    last_err: Exception | None = None
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            payload = _get_json(endpoint, data=body, timeout=50)
            counts = _parse_overpass_counts(payload)
            if counts:
                return counts[0]
        except Exception as exc:
            last_err = exc
            time.sleep(2.0)
            continue
    print(f"  overpass failed at {lat:.4f},{lon:.4f}: {last_err}", file=sys.stderr)
    return None


def fetch_osm_counts() -> dict[tuple[str, str, str], dict]:
    geos: list[tuple[tuple[str, str, str], str, tuple[float, float, str] | None]] = []
    for country, city, district, _acc in DESIGNED_DISTRICTS:
        key = (country, city, district)
        query = NOMINATIM_Q[key]
        print(f"Nominatim {district}, {city} …", flush=True)
        try:
            geo = geocode(query)
        except Exception as exc:
            print(f"  nominatim error: {exc}", file=sys.stderr)
            geo = None
        geos.append((key, query, geo))

    coords = [(g[0], g[1]) for _, _, g in geos if g is not None]
    print(f"Overpass batch for {len(coords)} geocoded districts …", flush=True)
    batch = overpass_counts_batch(coords) if coords else []
    batch_iter = iter(batch)

    out: dict[tuple[str, str, str], dict] = {}
    for key, query, geo in geos:
        if geo is None:
            out[key] = {"query": query, "amenity_hotel_nodes": None, "error": "nominatim_miss"}
            continue
        lat, lon, display = geo
        count = next(batch_iter, None)
        out[key] = {
            "query": query,
            "display_name": display,
            "lat": lat,
            "lon": lon,
            "radius_m": 1200,
            "amenity_hotel_nodes": count,
        }
        print(f"  {key[2]:16} {count} nodes", flush=True)
    return out


def trip_weights(
    arrivals: dict[str, float],
    osm: dict[tuple[str, str, str], dict] | None,
) -> list[float]:
    """Within-country OSM ranks × a squashed country-arrivals tilt."""
    import math
    from collections import defaultdict

    amenity_logs: list[float] = []
    countries: list[str] = []
    for country, city, district, _acc in DESIGNED_DISTRICTS:
        count = None
        if osm is not None:
            count = (osm.get((country, city, district)) or {}).get("amenity_hotel_nodes")
        if count is not None and count >= 0:
            amenity_logs.append(math.log1p(float(count)))
        else:
            amenity_logs.append(math.log1p(50.0))
        countries.append(country)

    by_country: dict[str, list[float]] = defaultdict(list)
    for country, amenity in zip(countries, amenity_logs):
        by_country[country].append(amenity)
    mean_amenity = {c: sum(vals) / len(vals) for c, vals in by_country.items()}
    relative = [a / mean_amenity[c] for a, c in zip(amenity_logs, countries)]

    max_arr = max(arrivals[c] for c in set(countries))
    weights: list[float] = []
    for country, rel in zip(countries, relative):
        cf = (math.log1p(arrivals[country] / 1e6) / math.log1p(max_arr / 1e6)) ** 0.4
        weights.append(round((0.55 + 0.45 * cf) * rel, 4))
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-osm",
        action="store_true",
        help="Skip Overpass/Nominatim; trip weights use arrivals only.",
    )
    args = parser.parse_args()

    print("Fetching World Bank GDP price level index (PA.NUS.GDP.PLI)…")
    price_level, price_meta = fetch_price_levels()
    print("Fetching Global Findex cash / digital-payment shares…")
    cash, cash_meta = fetch_cash_intensity()
    print("Fetching WDI tourist arrivals (2015–2019)…")
    arrivals, arrivals_meta = fetch_arrivals_2019()

    osm: dict[tuple[str, str, str], dict] | None = None
    if not args.skip_osm:
        print("Geocoding districts + Overpass amenity counts…")
        try:
            osm = fetch_osm_counts()
        except Exception as exc:
            print(f"OSM skipped after error: {exc}", file=sys.stderr)
            osm = None

    weights = trip_weights(arrivals, osm)

    districts = []
    osm_serial = {}
    for (country, city, district, acc), w in zip(DESIGNED_DISTRICTS, weights, strict=True):
        rec = {
            "country": country,
            "city": city,
            "district": district,
            "acceptance_density": acc,
            "trip_weight": w,
        }
        if osm is not None:
            info = osm.get((country, city, district), {})
            rec["osm"] = info
            osm_serial[f"{country}|{city}|{district}"] = info
        districts.append(rec)

    payload = {
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "notes": (
            "Country knobs for the synthetic DGP. Rows in members/trips/"
            "transactions remain simulated. Acceptance density is designed "
            "(OSM does not observe card terminals)."
        ),
        "sources": {
            "price_level": {
                "publisher": "World Bank WDI / International Comparison Program",
                "indicator": "PA.NUS.GDP.PLI",
                "indicator_name": "Price level index (GDP), United States = 100",
                "transform": "divide by Singapore so PRICE_LEVEL['Singapore'] = 1.0",
                "replaces": "PA.NUS.PPPC.RF (archived; data.worldbank.org redirects here)",
            },
            "cash_intensity": {
                "publisher": "World Bank Global Findex Database 2025 (source 28)",
                "instore_indicators": ["fin25e2", "fin25e2b"],
                "fallback_indicator": "g20.made",
                "transform": (
                    "in-store: fin25e2b/(fin25e2+fin25e2b); "
                    "else 1 - g20.made/100; clipped to [0.04, 0.85]"
                ),
            },
            "trip_weights": {
                "arrivals_indicator": "ST.INT.ARVL",
                "arrivals_year_policy": "latest 2015–2019 (avoid COVID collapse)",
                "osm": None
                if args.skip_osm
                else "Nominatim centroid + Overpass restaurant/cafe/hotel nodes in 1.2 km",
                "formula": "(0.55 + 0.45 * (log1p(arrivals)/log1p(max))^0.4) * relative log OSM amenity within country",
            },
            "acceptance_density": {
                "source": "designed",
                "why_not_osm": (
                    "OSM counts mapped amenities, not issuer acceptance. "
                    "A dense cash strip (Khao San) would look saturated."
                ),
            },
        },
        "price_level": price_level,
        "price_level_meta": price_meta,
        "cash_intensity": cash,
        "cash_intensity_meta": cash_meta,
        "tourist_arrivals": arrivals_meta,
        "districts": districts,
        "district_trip_weights": weights,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nWrote {OUT_PATH}")
    print("\nPRICE_LEVEL (Singapore = 1)")
    for k, v in sorted(price_level.items(), key=lambda kv: -kv[1]):
        print(f"  {k:16} {v:.3f}")
    print("\nCASH_INTENSITY")
    for k, v in sorted(cash.items(), key=lambda kv: -kv[1]):
        src = cash_meta[k]["source"]
        print(f"  {k:16} {v:.3f}  ({src})")


if __name__ == "__main__":
    main()
