"""LLM merchant-discovery judge (Gemini, with a real offline backup).

The deck's pitch: NEMU scrapes candidate venues for a corridor (OpenStreetMap /
Google Maps POIs, "top tourist spots") and an LLM judges which ones are
legitimate, high-footfall businesses worth an acquiring call, dropping closed
venues, duplicates, pure transit stops and global chains that already take Amex.

This module is that judge. Order of preference:
1. **Gemini** (`gemini-2.5-flash`) over its REST API, key from ``.env`` /
   ``GEMINI_API_KEY``.
2. **Seed verdicts** in ``data/external/merchant_discovery_seed.json`` — real
   judgements authored by Claude Code, so the data is genuine even with no key.
3. A transparent footfall heuristic for any candidate not covered above.

Run standalone:  ``python -m match.discover_merchants``
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.config import OUTPUT_DIR

DEFAULT_MODEL = "gemini-3.6-flash"
CANDIDATES_PER_DISTRICT = 12
_SEED_PATH = ROOT / "data" / "external" / "merchant_discovery_seed.json"
_ENV_PATH = ROOT / ".env"

_SYSTEM = (
    "You are an American Express merchant-acquisition analyst. You are given "
    "candidate venues scraped from public maps in a destination where card "
    "acceptance is low. Judge which are legitimate, high-footfall businesses "
    "genuinely worth signing to accept Amex. Reject venues that look closed, "
    "duplicated, generic/placeholder, purely transit infrastructure, or global "
    "chains that already accept Amex everywhere. Favour independent, "
    "tourist-frequented restaurants, shops and stays. Be decisive and concise. "
    "Return a verdict object for every candidate you are given."
)

# Gemini responseSchema (OpenAPI subset, uppercase types).
_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "name": {"type": "STRING"},
            "keep": {"type": "BOOLEAN"},
            "tourist_score": {"type": "NUMBER"},
            "reason": {"type": "STRING"},
        },
        "required": ["name", "keep", "tourist_score", "reason"],
    },
}

_GENERIC = ("shops", "stop", "terminal", "station", "pier", "metro", "line",
            "transjakarta", "mrt", "rail")


def _load_env_key() -> str | None:
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    if _ENV_PATH.exists():
        for line in _ENV_PATH.read_text().splitlines():
            line = line.strip()
            if line.startswith("GEMINI_API_KEY=") and not line.startswith("#"):
                return line.split("=", 1)[1].strip().strip('"').strip("'") or None
    return None


def _seed_verdicts() -> dict:
    if _SEED_PATH.exists():
        return json.loads(_SEED_PATH.read_text()).get("verdicts", {})
    return {}


def _heuristic_one(c: dict, hi: int) -> dict:
    name = str(c.get("name", ""))
    generic = any(g in name.lower() for g in _GENERIC)
    score = round(min(1.0, (c.get("visits", 0) / max(hi, 1)) * 0.9 + 0.1), 2)
    keep = (not generic) and c.get("visits", 0) >= max(3, 0.15 * hi)
    reason = (
        "generic/transit placeholder" if generic
        else ("strong footfall, independent venue" if keep
              else "thin footfall for an acquiring call")
    )
    return {"name": name, "keep": keep, "tourist_score": score, "reason": reason}


def _from_seed(candidates: list[dict]) -> list[dict]:
    seed = _seed_verdicts()
    hi = max((c.get("visits", 0) for c in candidates), default=1)
    out = []
    for c in candidates:
        v = seed.get(c["name"])
        if v:
            out.append({"name": c["name"], **v})
        else:
            out.append(_heuristic_one(c, hi))
    return out


def _gemini(candidates: list[dict], corridor: str, model: str) -> list[dict] | None:
    key = _load_env_key()
    if not key:
        return None
    listing = "\n".join(
        f"- {c['name']} | category: {c.get('category','?')} | "
        f"district: {c.get('district','?')} | visits seen: {c.get('visits','?')}"
        for c in candidates
    )
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={key}"
    )
    payload = {
        "systemInstruction": {"parts": [{"text": _SYSTEM}]},
        "contents": [{"parts": [{"text": (
            f"Corridor: {corridor}. Judge these candidate venues:\n\n{listing}"
        )}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": _SCHEMA,
        },
    }
    try:
        r = requests.post(url, json=payload, timeout=45)
        r.raise_for_status()
        text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        verdicts = json.loads(text)
        if isinstance(verdicts, list) and verdicts:
            return verdicts
        return None
    except Exception:
        return None


def judge_tourist_merchants(
    candidates: list[dict], corridor: str, model: str = DEFAULT_MODEL
) -> tuple[list[dict], str]:
    """Return (verdicts, source). Gemini -> seed (Claude Code) -> heuristic."""
    verdicts = _gemini(candidates, corridor, model)
    if verdicts is not None:
        return verdicts, model
    if _SEED_PATH.exists():
        return _from_seed(candidates), "claude-code seed (offline backup)"
    hi = max((c.get("visits", 0) for c in candidates), default=1)
    return [_heuristic_one(c, hi) for c in candidates], "heuristic"


def _candidates_for_district(detail: pd.DataFrame, district: str) -> list[dict]:
    sub = detail[detail["dest_district"] == district].sort_values(
        "visits", ascending=False
    ).head(CANDIDATES_PER_DISTRICT)
    return [
        {
            "name": r["merchant_name"],
            "category": r["category"],
            "district": district,
            "visits": int(r["visits"]),
        }
        for _, r in sub.iterrows()
    ]


def run() -> pd.DataFrame:
    detail = pd.read_csv(OUTPUT_DIR / "merchant_targets_detail.csv")
    rows = []
    source = "heuristic"
    for (country, district), _ in detail.groupby(["dest_country", "dest_district"]):
        cands = _candidates_for_district(detail, district)
        verdicts, source = judge_tourist_merchants(cands, f"{district}, {country}")
        by_name = {v["name"]: v for v in verdicts}
        for c in cands:
            v = by_name.get(c["name"], {"keep": False, "tourist_score": 0.0, "reason": "no verdict"})
            rows.append({
                "dest_country": country,
                "dest_district": district,
                "merchant_name": c["name"],
                "category": c["category"],
                "visits": c["visits"],
                "keep": bool(v["keep"]),
                "tourist_score": float(v["tourist_score"]),
                "reason": v["reason"],
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "merchant_discovery.csv", index=False)
    print("=" * 72)
    print("NEMU Match — LLM merchant-discovery judge")
    print("=" * 72)
    print(f"judged {len(out)} candidates across {out['dest_district'].nunique()} districts")
    print(f"kept: {int(out['keep'].sum())}  |  source: {source}")
    print(f"wrote {OUTPUT_DIR / 'merchant_discovery.csv'}")
    return out


if __name__ == "__main__":
    run()
