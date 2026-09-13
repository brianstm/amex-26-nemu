"""Fetch representative photos for districts and named merchants.

Uses the public Wikipedia REST summary endpoint (`page/summary/{title}`), which
returns a `thumbnail.source` — a real Wikimedia Commons photo of the place. The
result is cached to ``data/external/place_images.json`` so the dashboard reads it
offline. These are public, CC-licensed images, not Amex data.

Run:  ``python -m data.fetch_place_images``
"""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "external" / "place_images.json"

# district -> ordered list of Wikipedia titles to try (first hit with a photo wins)
DISTRICT_TITLES: dict[str, list[str]] = {
    "Shibuya": ["Shibuya"],
    "Ginza": ["Ginza"],
    "Gion": ["Gion"],
    "Namba": ["Namba", "Dōtonbori"],
    "Gangnam": ["Gangnam District"],
    "Myeongdong": ["Myeongdong"],
    "Haeundae": ["Haeundae District", "Haeundae Beach"],
    "Sukhumvit": ["Sukhumvit Road"],
    "Khao San": ["Khao San Road"],
    "Old City": ["Chiang Mai", "Chiang Mai Old City"],
    "Patong": ["Patong"],
    "Old Quarter": ["Old Quarter, Hanoi", "Hoàn Kiếm district", "Hanoi"],
    "Ba Dinh": ["Ba Đình district", "Ba Dinh District", "Hanoi"],
    "District 1": ["District 1, Ho Chi Minh City", "Ho Chi Minh City"],
    "Han River": ["Hàn River", "Da Nang"],
    "SCBD": ["Sudirman Central Business District", "Jakarta"],
    "Seminyak": ["Seminyak"],
    "Ubud": ["Ubud"],
    "Malioboro": ["Malioboro"],
    "KLCC": ["Kuala Lumpur City Centre", "Petronas Towers"],
    "Bukit Bintang": ["Bukit Bintang"],
    "George Town": ["George Town, Penang"],
    "Orchard": ["Orchard Road"],
    "Chinatown": ["Chinatown, Singapore"],
    "Central": ["Central, Hong Kong"],
    "Mong Kok": ["Mong Kok"],
    "CBD": ["Sydney central business district", "Sydney"],
    "Laneways": ["Laneways of Melbourne", "Melbourne"],
    "Le Marais": ["Le Marais"],
    "Montmartre": ["Montmartre"],
    "Centro Storico": ["Historic Centre of Rome", "Rome"],
    "Duomo": ["Florence Cathedral", "Florence"],
}

# named merchant -> Wikipedia titles (only merchants with a real page get a photo;
# everything else falls back to its district image in the app)
MERCHANT_TITLES: dict[str, list[str]] = {
    "KLIA Ekspres": ["KLIA Ekspres"],
    "ETS": ["Electric Train Service"],
    "Rapid Penang": ["Rapid Penang"],
    "Rapid KL": ["Rapid KL"],
    "Prasarana": ["Prasarana Malaysia"],
    "Swettenham Pier": ["Swettenham Pier", "Port of Penang"],
    "KOMTAR Terminal": ["Komtar"],
    "TransJakarta": ["TransJakarta"],
    "Grab": ["Grab (company)"],
    "UNIQLO": ["Uniqlo"],
    "7-Eleven": ["7-Eleven"],
    "Starbucks": ["Starbucks"],
    "McDonald's": ["McDonald's"],
    "Bic Camera": ["Bic Camera"],
    "Ladies' Market": ["Ladies' Market"],
    "Han Market": ["Hàn Market"],
}


# generic photo per category, used when a merchant has no brand photo of its own
CATEGORY_TITLES: dict[str, list[str]] = {
    "dining": ["Restaurant"],
    "retail": ["Shopping", "Retail"],
    "transport": ["Public transport", "Bus"],
    "lodging": ["Hotel"],
}


def _normalise(url: str) -> str:
    """Canonical upload.wikimedia.org thumb URL without tracking query params."""
    url = url.split("?", 1)[0]
    return url.replace("thumb.wikimedia.org", "upload.wikimedia.org")


def _thumb(title: str) -> str | None:
    slug = urllib.parse.quote(title.replace(" ", "_"))
    api = f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}"
    try:
        r = requests.get(api, headers={"User-Agent": "NEMU-demo/1.0"}, timeout=10)
        src = r.json().get("thumbnail", {}).get("source")
        return _normalise(src) if src else None
    except Exception:
        return None


def _resolve(titles: list[str]) -> str | None:
    for t in titles:
        url = _thumb(t)
        if url:
            return url
        time.sleep(0.15)
    return None


def run() -> dict:
    out = {"districts": {}, "merchants": {}, "categories": {}}
    for key, titles in CATEGORY_TITLES.items():
        url = _resolve(titles)
        if url:
            out["categories"][key] = url
        print(f"category {key:<16} -> {'ok' if url else 'MISSING'}")
    for key, titles in DISTRICT_TITLES.items():
        url = _resolve(titles)
        if url:
            out["districts"][key] = url
        print(f"district {key:<16} -> {'ok' if url else 'MISSING'}")
    for key, titles in MERCHANT_TITLES.items():
        url = _resolve(titles)
        if url:
            out["merchants"][key] = url
        print(f"merchant {key:<16} -> {'ok' if url else 'MISSING'}")
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nwrote {OUT} ({len(out['districts'])} districts, "
          f"{len(out['merchants'])} merchants, {len(out['categories'])} categories)")
    return out


if __name__ == "__main__":
    sys.exit(0 if run() else 0)
