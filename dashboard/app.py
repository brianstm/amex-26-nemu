"""NEMU Streamlit dashboard. Reads files in ``outputs/``."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from urllib.parse import quote

import html as html_lib

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.config import (
    CATEGORY_IMAGES,
    CURRENCY,
    DISTRICT_COORDS,
    DISTRICT_IMAGES,
    FX_LCU_PER_USD,
    MERCHANT_IMAGES,
    OUTPUT_DIR,
    REGION,
    REGION_ORDER,
)

# Dashboard displays money in SGD. Pipeline stores USD; convert at World Bank FX.
SGD_PER_USD = float(FX_LCU_PER_USD.get("Singapore", 1.35))


def _env_value(name: str) -> str | None:
    """Read a secret from the environment, Streamlit secrets, or repo ``.env``."""
    val = os.environ.get(name)
    if val:
        return val.strip().strip('"').strip("'") or None
    try:
        secrets = st.secrets  # type: ignore[attr-defined]
        if name in secrets:
            return str(secrets[name]).strip() or None
    except Exception:
        pass
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith(f"{name}=") and not line.startswith("#"):
                return line.split("=", 1)[1].strip().strip('"').strip("'") or None
    return None


def _carto_tiles() -> tuple[str, str]:
    """Carto light basemap URL, with ``MAP_API`` key when available."""
    attr = (
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> '
        '&copy; <a href="https://carto.com/attributions">CARTO</a>'
    )
    base = "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
    key = _env_value("MAP_API")
    if key:
        return f"{base}?key={quote(key, safe='')}", attr
    return base, attr

_FALLBACK_IMG = next(iter(DISTRICT_IMAGES.values()), "")
_CAT_HEX = {
    "dining": "#A7FC04",
    "retail": "#2F5FC2",
    "transport": "#D97A59",
    "lodging": "#8532A8",
}
_CLUSTER_ICON_JS = """
function(cluster) {
  var n = cluster.getChildCount();
  var size = n < 10 ? 36 : (n < 40 ? 44 : 54);
  return L.divIcon({
    html: '<div style="background:rgba(167,252,4,0.55);border:2px solid #B0B0B0;border-radius:50%;'
      + 'width:' + size + 'px;height:' + size + 'px;display:flex;align-items:center;'
      + 'justify-content:center;font-family:Montserrat,sans-serif;font-weight:800;'
      + 'font-size:13px;color:#1E1E1E;box-shadow:0 1px 3px rgba(0,0,0,.12);">'
      + n + '</div>',
    className: '',
    iconSize: L.point(size, size)
  });
}
"""

CHARCOAL = "#1E1E1E"
INK = "#1A1A1A"
LIME = "#A7FC04"
WHITE = "#FFFFFF"
GRAY = "#D9D9D9"
BLUE_N = "#2F5FC2"
TERRACOTTA_M = "#D97A59"
PURPLE_U = "#8532A8"
FONT = "Montserrat, sans-serif"
PALETTE = [LIME, BLUE_N, TERRACOTTA_M, PURPLE_U, "#C4E64A", GRAY, WHITE]
CAUSE_COLORS = {
    "no_acceptance": BLUE_N,
    "rail_substitution": LIME,
    "cash": TERRACOTTA_M,
    "no_demand": PURPLE_U,
}

THEMES = {
    "Dark": {
        "plotly": "plotly_dark",
        "bg": CHARCOAL,
        "sidebar": INK,
        "card": "#2A2A2A",
        "text": "#F5F5F5",
        "muted": "#B4B4B4",
        "note": "#8A8A8A",
        "metric": LIME,
        "grid": "#333333",
        "axis": "#444444",
        "border": "#333333",
        "header_bg": CHARCOAL,
        "bar_observed": GRAY,
        "bar_recoverable": LIME,
        "claim_text": "#F5F5F5",
        "claim_strong": LIME,
    },
    "Light": {
        "plotly": "plotly_white",
        "bg": WHITE,
        "sidebar": "#F4F4F4",
        "card": "#F4F4F4",
        "text": CHARCOAL,
        "muted": "#555555",
        "note": "#6A6A6A",
        "metric": CHARCOAL,
        "grid": "#E6E6E6",
        "axis": "#CCCCCC",
        "border": "#E0E0E0",
        "header_bg": WHITE,
        "bar_observed": CHARCOAL,
        "bar_recoverable": LIME,
        "claim_text": CHARCOAL,
        "claim_strong": CHARCOAL,
    },
}

st.set_page_config(
    page_title="NEMU · Notice, Explain, Match, Uplift",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_all() -> dict:
    leakage = pd.read_csv(OUTPUT_DIR / "leakage_estimates.csv")
    truth = pd.read_csv(OUTPUT_DIR / "ground_truth.csv")
    merchants = pd.read_csv(OUTPUT_DIR / "merchant_targets.csv")
    txns_path = OUTPUT_DIR / "transactions.csv"
    txns = pd.read_csv(txns_path) if txns_path.exists() else pd.DataFrame()
    holdout = pd.read_csv(OUTPUT_DIR / "holdout_assignments.csv")
    calib = pd.read_csv(OUTPUT_DIR / "holdout_calibration.csv")
    corridor_val = pd.read_csv(OUTPUT_DIR / "validation_corridor.csv")
    md_path = OUTPUT_DIR / "merchant_targets_detail.csv"
    merch_detail = pd.read_csv(md_path) if md_path.exists() else pd.DataFrame()
    seg_path = OUTPUT_DIR / "behavioral_segments.csv"
    segments = pd.read_csv(seg_path) if seg_path.exists() else pd.DataFrame()
    disc_path = OUTPUT_DIR / "merchant_discovery.csv"
    discovery = pd.read_csv(disc_path) if disc_path.exists() else pd.DataFrame()
    metrics = json.loads((OUTPUT_DIR / "notice_metrics.json").read_text())
    holdout_m = json.loads((OUTPUT_DIR / "holdout_metrics.json").read_text())
    merged = leakage.merge(
        truth[["trip_id", "category", "acceptance_leakage", "total_leakage", "true_cause"]],
        on=["trip_id", "category"],
        how="left",
    )
    return {
        "leakage": leakage,
        "merged": merged,
        "merchants": merchants,
        "txns": txns,
        "holdout": holdout,
        "calib": calib,
        "corridor_val": corridor_val,
        "merch_detail": merch_detail,
        "segments": segments,
        "discovery": discovery,
        "metrics": metrics,
        "holdout_m": holdout_m,
    }


def to_sgd(x: float) -> float:
    """Convert a USD amount to SGD."""
    return float(x) * SGD_PER_USD


def money(x: float) -> str:
    """Format a USD amount as compact SGD (S$…)."""
    ax = abs(to_sgd(x))
    s = to_sgd(x)
    if ax >= 1_000_000:
        return f"S${s/1_000_000:,.2f}M"
    if ax >= 1_000:
        return f"S${s/1_000:,.0f}K"
    return f"S${s:,.0f}"


def format_sgd(x, decimals: int = 0) -> str:
    """Format one USD cell as SGD for Styler.format callables."""
    if pd.isna(x):
        return "—"
    return f"S${to_sgd(x):,.{decimals}f}"


def fx_rates_table() -> pd.DataFrame:
    """Country FX board: local units per USD, and SGD equivalent of 1 USD."""
    rows = []
    for country, lcu in sorted(FX_LCU_PER_USD.items()):
        rows.append(
            {
                "Country": country,
                "Currency": CURRENCY.get(country, "—"),
                "Local per 1 USD": float(lcu),
                "SGD per 1 USD": SGD_PER_USD,
            }
        )
    return pd.DataFrame(rows)


def style_fig(fig: go.Figure, theme: dict, height: int = 420) -> go.Figure:
    fig.update_layout(
        template=theme["plotly"],
        height=height,
        font=dict(family=FONT, color=theme["muted"], size=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=48, r=16, t=72, b=130),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.32,
            x=0,
            xanchor="left",
            title_text="",
            bgcolor="rgba(0,0,0,0)",
            font=dict(family=FONT, color=theme["muted"], size=12),
            tracegroupgap=12,
        ),
        title=dict(
            font=dict(family=FONT, color=theme["text"], size=16),
            x=0,
            xanchor="left",
            y=0.98,
            yanchor="top",
            pad=dict(b=18, l=0, r=0, t=4),
        ),
        coloraxis_colorbar=dict(outlinewidth=0),
        legend_title_text="",
        hoverlabel=dict(
            bgcolor=WHITE,
            bordercolor=GRAY,
            font=dict(family=FONT, color=CHARCOAL, size=12),
        ),
    )
    fig.update_xaxes(
        gridcolor=theme["grid"],
        zerolinecolor=theme["grid"],
        tickfont=dict(family=FONT, color=theme["muted"]),
        title_font=dict(family=FONT, color=theme["muted"]),
        linecolor=theme["axis"],
    )
    fig.update_yaxes(
        gridcolor=theme["grid"],
        zerolinecolor=theme["grid"],
        tickfont=dict(family=FONT, color=theme["muted"]),
        title_font=dict(family=FONT, color=theme["muted"]),
        linecolor=theme["axis"],
    )
    return fig


def inject_css(theme: dict) -> None:
    st.markdown(
        f"""
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
        <style>
        html, body, .stApp, .stMarkdown, p, label,
        [data-testid="stMarkdownContainer"],
        [data-testid="stMetricValue"],
        [data-testid="stMetricLabel"],
        [data-testid="stMetricDelta"],
        .stSelectbox, .stMultiSelect, .stRadio, .stTabs {{
            font-family: "Montserrat", sans-serif !important;
        }}
        span[data-testid="stIconMaterial"],
        [data-testid="stIconMaterial"] {{
            font-family: "Material Symbols Rounded", "Material Symbols Outlined",
                "Material Icons" !important;
            font-weight: 400 !important;
            font-style: normal !important;
            letter-spacing: normal !important;
            line-height: 1 !important;
        }}
        /* Hide Streamlit collapse control (Material icon ligature prints as text). */
        [data-testid="stSidebarCollapsedControl"],
        [data-testid="collapsedControl"],
        [data-testid="stHeader"] [data-testid="stBaseButton-headerNoPadding"],
        [data-testid="stHeader"] [data-testid="stBaseButton-header"],
        button[kind="headerNoPadding"],
        button[aria-label="keyboard_double_arrow_right"],
        button[aria-label="keyboard_double_arrow_left"] {{
            display: none !important;
        }}
        .stApp {{ background: {theme["bg"]}; }}
        [data-testid="stAppViewContainer"] {{ background: {theme["bg"]} !important; }}
        .block-container {{ padding-top: 1.4rem; max-width: 1400px; }}
        h1, h2, h3 {{
            color: {theme["text"]} !important;
            font-family: "Montserrat", sans-serif !important;
            font-weight: 800 !important;
            letter-spacing: -0.03em;
        }}
        h3 {{ font-weight: 700 !important; }}
        p, .stMarkdown p {{ color: {theme["muted"]}; font-weight: 400; }}
        [data-testid="stMetricValue"] {{
            font-family: "Montserrat", sans-serif !important;
            font-weight: 800 !important;
            color: {theme["metric"]} !important;
            letter-spacing: -0.03em;
        }}
        [data-testid="stMetricLabel"] {{
            color: {theme["muted"]} !important;
            font-weight: 500 !important;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-size: 0.68rem !important;
        }}
        [data-testid="stMetricDelta"] {{ color: {LIME} !important; }}
        section[data-testid="stSidebar"] {{
            background: {theme["sidebar"]};
            border-right: 1px solid {theme["border"]};
        }}
        section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2 {{
            color: {theme["text"]} !important;
            font-weight: 800 !important;
        }}
        .stTabs [data-baseweb="tab-highlight"] {{ background-color: {LIME}; }}
        .stTabs [aria-selected="true"] {{
            color: {theme["text"]} !important;
            font-weight: 700 !important;
        }}
        .stTabs [data-baseweb="tab"] {{
            color: {theme["muted"]};
            font-weight: 500;
            letter-spacing: 0.02em;
        }}
        div[data-testid="stExpander"], div[data-testid="stDataFrame"] {{
            border: 1px solid {theme["border"]};
        }}
        /* Streamlit 1.54+ chips use [data-tag], not data-baseweb="tag".
           Primary lime + white text fails contrast — force charcoal text/icons. */
        [data-testid="stMultiSelect"] [data-tag],
        [data-testid="stMultiSelect"] [data-tag] * {{
            background-color: {LIME} !important;
            color: {CHARCOAL} !important;
            -webkit-text-fill-color: {CHARCOAL} !important;
        }}
        [data-testid="stMultiSelect"] [data-tag] {{
            border: 1px solid {LIME} !important;
            border-radius: 4px !important;
        }}
        [data-testid="stMultiSelect"] [data-tag] button,
        [data-testid="stMultiSelect"] [data-tag] svg,
        [data-testid="stMultiSelect"] [data-tag] path {{
            background: transparent !important;
            color: {CHARCOAL} !important;
            fill: {CHARCOAL} !important;
            stroke: {CHARCOAL} !important;
            -webkit-text-fill-color: {CHARCOAL} !important;
        }}
        /* Keep the lime block size; allow "Nemu" glyphs to paint outside it. */
        .element-container:has(.nemu-wordmark),
        [data-testid="stMarkdownContainer"]:has(.nemu-wordmark),
        [data-testid="stMarkdownContainer"]:has(.nemu-wordmark) > * {{
            overflow: visible !important;
        }}
        .nemu-wordmark {{
            display: inline-block;
            background: {LIME};
            padding: 0.55rem 1.35rem 0.35rem 1.35rem;
            height: 4.1rem;
            box-sizing: border-box;
            line-height: 1;
            overflow: visible;
        }}
        .nemu-wordmark span {{
            font-family: "Montserrat", sans-serif;
            font-weight: 900;
            font-size: 3.2rem;
            line-height: 1.15;
            display: inline-block;
            color: {CHARCOAL};
            letter-spacing: -0.045em;
            overflow: visible;
        }}
        .nemu-pillars {{
            margin-top: 0.55rem;
            font-family: "Montserrat", sans-serif;
            font-weight: 300;
            font-size: 0.78rem;
            letter-spacing: 0.34em;
            color: {theme["muted"]};
            text-transform: none;
        }}
        .nemu-note {{
            margin-top: 0.35rem;
            font-weight: 400;
            font-size: 0.82rem;
            color: {theme["note"]};
            letter-spacing: 0.02em;
        }}
        .nemu-claim {{
            background: {theme["card"]};
            border-left: 4px solid {LIME};
            padding: 0.95rem 1.15rem;
            color: {theme["claim_text"]};
            font-weight: 400;
            font-size: 0.98rem;
            line-height: 1.45;
        }}
        .nemu-claim strong {{ color: {theme["claim_strong"]}; font-weight: 700; }}
        .nemu-explain {{
            background: {theme["card"]};
            border-left: 4px solid {LIME};
            padding: 0.75rem 1.05rem;
            margin: 0.2rem 0 1.1rem 0;
            color: {theme["claim_text"]};
            font-weight: 400;
            font-size: 0.94rem;
            line-height: 1.5;
        }}
        .nemu-what, .nemu-why {{
            display: inline-block;
            font-weight: 800;
            font-size: 0.62rem;
            letter-spacing: 0.12em;
            color: {CHARCOAL};
            background: {LIME};
            padding: 0.05rem 0.4rem;
            margin-right: 0.5rem;
            border-radius: 2px;
        }}
        .nemu-why {{ margin-top: 0.35rem; }}
        header[data-testid="stHeader"] {{ background: {theme["header_bg"]}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_filters(df: pd.DataFrame, country, category, segment, region=None) -> pd.DataFrame:
    out = df
    if region and "dest_country" in out.columns:
        out = out[out["dest_country"].map(REGION).isin(region)]
    if country:
        out = out[out["dest_country"].isin(country)]
    if category:
        out = out[out["category"].isin(category)]
    if segment and "segment" in out.columns:
        out = out[out["segment"].isin(segment)]
    return out


def _zoom_for(lats: pd.Series, lons: pd.Series) -> float:
    """Rough zoom level so the visible points fit the frame."""
    if len(lats) <= 1:
        return 12.0
    span = max(float(lats.max() - lats.min()), float(lons.max() - lons.min()))
    for limit, z in [(0.2, 12), (1, 9), (5, 6), (20, 4.2), (60, 3.2)]:
        if span < limit:
            return float(z)
    return 2.5


def merchant_map(md: pd.DataFrame, focus_districts: list, theme: dict) -> None:
    """Clustered merchant map: numbered blobs when zoomed out, pins when zoomed in.

    District selection only recentres / zooms the view — it does not hide other
    merchants. Pins are scattered around each district centre (no exact store
    coordinates).
    """
    md = md.copy()
    md["lat"] = md["dest_district"].map(lambda d: DISTRICT_COORDS.get(d, (None, None))[0])
    md["lon"] = md["dest_district"].map(lambda d: DISTRICT_COORDS.get(d, (None, None))[1])
    md = md.dropna(subset=["lat", "lon"])
    if md.empty:
        st.info("No mapped districts for the current filters.")
        return

    # Cap pins for performance; keep highest-value merchants.
    d = md.sort_values("est_recoverable_value", ascending=False).head(280).copy()
    rng = np.random.default_rng(26)
    ang = rng.uniform(0, 2 * np.pi, len(d))
    rad = 0.012 * np.sqrt(rng.uniform(0, 1, len(d)))
    d["pin_lat"] = d["lat"] + rad * np.cos(ang)
    d["pin_lon"] = d["lon"] + rad * np.sin(ang)

    if focus_districts:
        focus = d[d["dest_district"].isin(focus_districts)]
        if focus.empty:
            focus = d
        center_lat = float(focus["lat"].mean())
        center_lon = float(focus["lon"].mean())
        zoom = 13.0 if len(focus_districts) == 1 else _zoom_for(focus["lat"], focus["lon"]) + 2.5
        zoom = min(max(zoom, 11.0), 14.0)
    else:
        center_lat = float(d["lat"].mean())
        center_lon = float(d["lon"].mean())
        zoom = _zoom_for(d["lat"], d["lon"])

    tiles, attr = _carto_tiles()
    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles=tiles,
        attr=attr,
        control_scale=True,
    )
    cluster = MarkerCluster(
        name="Merchants",
        overlay=False,
        control=False,
        icon_create_function=_CLUSTER_ICON_JS,
        options={
            "showCoverageOnHover": False,
            "maxClusterRadius": 55,
            "spiderfyOnMaxZoom": True,
            "disableClusteringAtZoom": 15,
        },
    ).add_to(fmap)

    for row in d.itertuples(index=False):
        color = _CAT_HEX.get(row.category, "#777777")
        img = (
            MERCHANT_IMAGES.get(row.merchant_name)
            or CATEGORY_IMAGES.get(row.category)
            or DISTRICT_IMAGES.get(row.dest_district)
            or _FALLBACK_IMG
        )
        name = html_lib.escape(str(row.merchant_name))
        line1 = html_lib.escape(f"{row.dest_district} · {row.sub_category}")
        line2 = html_lib.escape(str(row.price_range))
        line3 = html_lib.escape(
            f"{int(row.visits)} visits · {money(row.est_recoverable_value)} recoverable"
        )
        popup_html = (
            f"<div style='max-width:220px;font-family:Montserrat,sans-serif;font-size:12px;'>"
            f"<img src='{html_lib.escape(img)}' style='width:210px;border-radius:6px;"
            f"margin-bottom:6px;display:block'/>"
            f"<b>{name}</b><br>{line1}<br>{line2}<br>{line3}</div>"
        )
        folium.CircleMarker(
            location=[float(row.pin_lat), float(row.pin_lon)],
            radius=7,
            color=color,
            weight=0,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=folium.Popup(popup_html, max_width=240),
            tooltip=name,
        ).add_to(cluster)

    if focus_districts:
        bounds = [
            [float(r.pin_lat), float(r.pin_lon)]
            for r in d[d["dest_district"].isin(focus_districts)].itertuples(index=False)
        ]
        if bounds:
            fmap.fit_bounds(bounds, padding=(40, 40))

    st_folium(
        fmap,
        height=520,
        use_container_width=True,
        returned_objects=[],
        key=f"merch-map-{'-'.join(sorted(focus_districts)) or 'all'}",
    )
    st.caption(
        "Numbered lime circles are clusters of merchants — zoom in to split them "
        "into smaller groups, then into coloured pins (dining / retail / "
        "transport / lodging). Pick a district above to fly to that area. Pin "
        "positions are illustrative (scattered around the district centre)."
    )


def explain(what: str, why: str) -> None:
    """Plain-English 'what am I looking at / why it matters' note per tab."""
    st.markdown(
        f'<div class="nemu-explain"><span class="nemu-what">WHAT</span> {what}'
        f'<br><span class="nemu-why">WHY IT MATTERS</span> {why}</div>',
        unsafe_allow_html=True,
    )


def graph_note(text: str) -> None:
    """Short description placed directly under a chart."""
    st.caption(text)


def main() -> None:
    data = load_all()
    m = data["metrics"]
    leakage = data["leakage"]

    theme = THEMES["Light"]
    inject_css(theme)
    st.markdown(
        """
        <div class="nemu-wordmark"><span>Nemu</span></div>
        <div class="nemu-note">Public FX, merchant names, and ISO codes — with simulated trips. Figures in <b>SGD</b>.</div>
        """,
        unsafe_allow_html=True,
    )
    rate_label = f"{SGD_PER_USD:.6f}"
    with st.expander(
        f"FX rates used · shown in SGD (Singapore rate {rate_label} per 1 USD)",
        expanded=False,
    ):
        st.caption(
            "Amounts start in USD and are converted to SGD using the World Bank "
            f"Singapore rate ({rate_label} SGD per 1 USD). "
            "The table below is the same public FX series attached to tickets."
        )
        fx = fx_rates_table()
        st.dataframe(
            fx.style.format(
                {
                    "Local per 1 USD": "{:,.6f}",
                    "SGD per 1 USD": "{:,.6f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Hidden acceptance leakage", money(m["hidden_acceptance_leakage"]))
    c2.metric(
        "Model recovered",
        money(m["estimated_leakage"]),
        delta=f"{m['recovery_ratio_vs_acceptance']:.0%} of hidden",
    )
    c3.metric("Corridor correlation", f"{m['corridor_corr_vs_acceptance_leakage']:.3f}")
    c4.metric(
        "Cause $ accuracy",
        f"{m.get('cause_value_weighted_accuracy', 0):.0%}",
    )

    st.markdown(
        f'<div class="nemu-claim"><strong>Blind test results</strong>'
        f'<ul style="margin:0.4rem 0 0 1.1rem;padding:0">'
        f'<li>Hidden: <strong>{money(m["hidden_acceptance_leakage"])}</strong> acceptance leakage</li>'
        f'<li>Model found: <strong>{money(m["estimated_leakage"])}</strong> '
        f'({m["recovery_ratio_vs_acceptance"]:.0%} recovered)</li>'
        f'<li>Cause accuracy: <strong>{m.get("cause_value_weighted_accuracy", 0):.0%}</strong> of dollars tagged correctly</li>'
        f'</ul></div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Filters")
        f_region = st.multiselect("Region", REGION_ORDER)
        # Region cascades into the destination list. LATAM / US have no corridor
        # data yet. Amex would light these up as NEMU expands.
        empty_regions = [r for r in f_region if r in {"LATAM", "US"}
                         and not (leakage["dest_country"].map(REGION) == r).any()]
        if empty_regions:
            st.caption(f"No corridor data yet for: {', '.join(empty_regions)}.")
        country_pool = leakage["dest_country"]
        if f_region:
            country_pool = country_pool[country_pool.map(REGION).isin(f_region)]
        countries = sorted(country_pool.unique())
        categories = sorted(leakage["category"].unique())
        segments = sorted(leakage["segment"].unique())
        f_country = st.multiselect("Destination", countries)
        f_cat = st.multiselect("Category", categories)
        f_seg = st.multiselect("Segment", segments)

    view = apply_filters(leakage, f_country, f_cat, f_seg, f_region)
    merged_view = apply_filters(data["merged"], f_country, f_cat, f_seg, f_region)

    tab_notice, tab_explain, tab_match, tab_uplift = st.tabs(
        ["Notice", "Explain", "Match", "Uplift"]
    )

    with tab_notice:
        st.subheader("Where to grow: what we see vs what we are missing")
        explain(
            "The same destinations, ranked two ways. On the left by the Amex "
            "spend we already see. On the right by the spend we are missing and "
            "could win back.",
            "A normal dashboard only shows where we already win. The opportunity "
            "is the places that jump up the right-hand list: smaller, cheaper "
            "markets where cards are not widely accepted yet.",
        )
        agg = (
            view.groupby("dest_country", as_index=False)
            .agg(
                observed=("observed_spend", "sum"),
                recoverable=("leakage_estimate", "sum"),
            )
        )
        left = agg.sort_values("observed", ascending=True).assign(
            observed_sgd=lambda d: d["observed"] * SGD_PER_USD,
            recoverable_sgd=lambda d: d["recoverable"] * SGD_PER_USD,
        )
        right = agg.sort_values("recoverable", ascending=True).assign(
            observed_sgd=lambda d: d["observed"] * SGD_PER_USD,
            recoverable_sgd=lambda d: d["recoverable"] * SGD_PER_USD,
        )
        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.bar(
                left,
                x="observed_sgd",
                y="dest_country",
                orientation="h",
                color_discrete_sequence=[theme["bar_observed"]],
                title="Ranked by observed Amex volume",
            )
            fig.update_xaxes(title="Observed spend (SGD)")
            fig.update_yaxes(title="")
            st.plotly_chart(style_fig(fig, theme), width="stretch")
        with col_b:
            fig = px.bar(
                right,
                x="recoverable_sgd",
                y="dest_country",
                orientation="h",
                color_discrete_sequence=[theme["bar_recoverable"]],
                title="Ranked by recoverable leakage",
            )
            fig.update_xaxes(title="Estimated recoverable (SGD)")
            fig.update_yaxes(title="")
            st.plotly_chart(style_fig(fig, theme), width="stretch")
        graph_note(
            "Left = where Amex already books volume. Right = where the model "
            "sees recoverable leakage. Countries that climb on the right "
            "(e.g. Vietnam, Indonesia) are the growth targets. Amounts in SGD."
        )

        rank = agg.copy()
        rank["rank_observed"] = rank["observed"].rank(ascending=False).astype(int)
        rank["rank_recoverable"] = rank["recoverable"].rank(ascending=False).astype(int)
        rank["rank_shift"] = rank["rank_observed"] - rank["rank_recoverable"]
        st.caption("Currency: SGD")
        st.dataframe(
            rank.sort_values("recoverable", ascending=False).style.format(
                {"observed": format_sgd, "recoverable": format_sgd}
            ),
            width="stretch",
            hide_index=True,
        )

        st.subheader("Break the opportunity down")
        explain(
            "The money we can win back, split by spending type first "
            "(food, shopping, transport, hotels). You can then go deeper by "
            "country and card tier.",
            "You send a food reward or a shopping reward, not a country reward. "
            "Seeing that most of the gap is food in Vietnam tells the team "
            "exactly what offer to send.",
        )
        grain = st.radio(
            "Group by",
            [
                "Category",
                "Category × Country",
                "Country",
                "Country × Category",
                "Country × Category × Segment",
            ],
            horizontal=True,
        )
        keys = {
            "Category": ["category"],
            "Category × Country": ["category", "dest_country"],
            "Country": ["dest_country"],
            "Country × Category": ["dest_country", "category"],
            "Country × Category × Segment": ["dest_country", "category", "segment"],
        }[grain]
        table = (
            view.groupby(keys, as_index=False)
            .agg(
                trips=("trip_id", "nunique"),
                members=("member_id", "nunique"),
                observed_spend=("observed_spend", "sum"),
                recoverable=("leakage_estimate", "sum"),
                mean_acceptance=("acceptance_density", "mean"),
                mean_cash=("cash_intensity", "mean"),
            )
            .sort_values("recoverable", ascending=False)
        )
        st.caption("Currency: SGD")
        st.dataframe(
            table.style.format(
                {
                    "observed_spend": format_sgd,
                    "recoverable": format_sgd,
                    "mean_acceptance": "{:.2f}",
                    "mean_cash": "{:.2f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

        txns = data.get("txns", pd.DataFrame())
        if not txns.empty and {"merchant_name", "currency", "amount_local"}.issubset(txns.columns):
            st.subheader("The raw card transactions underneath")
            st.write(
                "This is the ticket-level data every chart above is built from: "
                "one row per purchase. Each has a real currency code, a real "
                "merchant name from OpenStreetMap, and an `mcc` (the industry "
                "code, e.g. 5812 = restaurants). The amounts and which trip a "
                "ticket belongs to are simulated."
            )
            sample = txns
            if f_country:
                sample = sample[sample["dest_country"].isin(f_country)]
            if f_cat:
                sample = sample[sample["category"].isin(f_cat)]
            cols = [
                c
                for c in [
                    "txn_id",
                    "dest_country",
                    "dest_district",
                    "merchant_name",
                    "category",
                    "mcc",
                    "currency",
                    "amount_local",
                    "amount_usd",
                ]
                if c in sample.columns
            ]
            # The file is ordered by category, so a plain head() would show only
            # dining. Shuffle first so the preview mixes all four categories.
            sample = sample.sample(frac=1.0, random_state=26).copy()
            if "amount_usd" in sample.columns:
                sample["amount_sgd"] = sample["amount_usd"] * SGD_PER_USD
                cols = [("amount_sgd" if c == "amount_usd" else c) for c in cols]
            st.caption(
                f"Showing 25 of {len(sample):,} tickets, shuffled so you see a "
                f"mix of dining, retail, transport and lodging. "
                f"Currency: SGD (amount_sgd = amount_usd × {SGD_PER_USD:.4g})."
            )
            fmt = {}
            if "amount_local" in cols:
                fmt["amount_local"] = "{:,.2f}"
            if "amount_sgd" in cols:
                fmt["amount_sgd"] = "S${:,.2f}"
            head = sample[cols].head(25)
            st.dataframe(
                head.style.format(fmt) if fmt else head,
                width="stretch",
                hide_index=True,
            )

    with tab_explain:
        st.subheader("Why the gap, and which lever to pull")
        explain(
            "Every dollar we can win back is tagged with one of four reasons, "
            "so the fix is obvious.",
            "<b>no_acceptance</b>: sign up merchants (Match). "
            "<b>rail_substitution</b>: send an offer (Match). "
            "<b>cash</b>: a cash culture, not a coverage hole, so do not "
            "overspend. <b>no_demand</b>: do nothing.",
        )
        cause_cty = (
            view.groupby(["dest_country", "cause"], as_index=False)["leakage_estimate"]
            .sum()
            .rename(columns={"leakage_estimate": "usd"})
        )
        cause_cty["sgd"] = cause_cty["usd"] * SGD_PER_USD
        fig = px.bar(
            cause_cty,
            x="dest_country",
            y="sgd",
            color="cause",
            color_discrete_map=CAUSE_COLORS,
            title="Recoverable value by cause",
            labels={"cause": "", "sgd": "SGD", "dest_country": ""},
        )
        fig.update_xaxes(title="")
        fig.update_yaxes(title="SGD")
        st.plotly_chart(style_fig(fig, theme, height=560), width="stretch")
        graph_note(
            "Stacked bars show how much recoverable value sits in each destination, "
            "split by cause. Tall no_acceptance stacks mean acquiring; "
            "rail_substitution stacks mean offers. Amounts in SGD."
        )

        pie = view.groupby("cause", as_index=False)["leakage_estimate"].sum()
        pie["leakage_estimate"] = pie["leakage_estimate"] * SGD_PER_USD
        figp = px.pie(
            pie,
            names="cause",
            values="leakage_estimate",
            color="cause",
            color_discrete_map=CAUSE_COLORS,
            title="Portfolio mix (SGD)",
            hole=0.45,
            labels={"cause": ""},
        )
        st.plotly_chart(style_fig(figp, theme, height=380), width="stretch")
        graph_note(
            "Portfolio share of leakage by cause across the filtered corridor set. "
            "Use it to size acquiring vs incentive budget."
        )

    with tab_uplift:
        st.subheader("Proof the model works: hidden-file test")
        explain(
            "Each dot is a corridor. The bottom axis is the real leakage the "
            "data generator hid from the model. The side axis is what the model "
            "guessed without ever seeing it. On the dashed line means a perfect "
            "guess.",
            "It answers the obvious question for a synthetic ledger: can the "
            "method recover a known answer it was never shown? Yes — we hide "
            "the truth file from the likelihood and score against it later.",
        )
        st.write(
            "ground_truth.csv was written by the DGP and **never entered the likelihood**. "
            "Each point is a destination × category corridor. The dashed line is y = x."
        )
        cv = data["corridor_val"].copy()
        if f_country:
            cv = cv[cv["dest_country"].isin(f_country)]
        if f_cat:
            cv = cv[cv["category"].isin(f_cat)]
        lim = float(max(cv["est"].max(), cv["truth_acc"].max()) * 1.05) * SGD_PER_USD
        fig = px.scatter(
            cv.assign(
                truth_sgd=cv["truth_acc"] * SGD_PER_USD,
                est_sgd=cv["est"] * SGD_PER_USD,
            ),
            x="truth_sgd",
            y="est_sgd",
            color="dest_country",
            hover_data=["category"],
            color_discrete_sequence=PALETTE,
            title="Estimated leakage vs hidden acceptance leakage",
        )
        fig.add_trace(
            go.Scatter(
                x=[0, lim],
                y=[0, lim],
                mode="lines",
                name="y = x",
                line=dict(color=theme["muted"], dash="dash"),
            )
        )
        fig.update_xaxes(title="Hidden acceptance leakage (SGD)", range=[0, lim])
        fig.update_yaxes(title="PPML estimate (SGD)", range=[0, lim])
        st.plotly_chart(style_fig(fig, theme, height=520), width="stretch")
        graph_note(
            "Each point is a destination × category corridor. X = hidden truth "
            "the model never saw; Y = PPML estimate. Points near the dashed "
            "y = x line mean the model recovered the right answer. Amounts in SGD."
        )

        k1, k2, k3 = st.columns(3)
        k1.metric("Row correlation", f"{m['row_corr_vs_acceptance_leakage']:.3f}")
        k2.metric("Corridor correlation", f"{m['corridor_corr_vs_acceptance_leakage']:.3f}")
        k3.metric("Corridor MAPE", f"{m['corridor_mape_vs_acceptance_leakage']:.0%}")

        st.caption(
            "Cash leakage is *not* on this scatter. Destination-country fixed effects "
            "hold cash culture fixed; the x-axis is the acceptance-only residual the "
            "model was designed to recover."
        )

        row_sample = merged_view.sample(
            n=min(4_000, len(merged_view)), random_state=26
        ).assign(
            acceptance_leakage_sgd=lambda d: d["acceptance_leakage"] * SGD_PER_USD,
            leakage_estimate_sgd=lambda d: d["leakage_estimate"] * SGD_PER_USD,
        )
        fig2 = px.scatter(
            row_sample,
            x="acceptance_leakage_sgd",
            y="leakage_estimate_sgd",
            color="cause",
            opacity=0.35,
            color_discrete_map=CAUSE_COLORS,
            title="Trip by category (4k sample): noisier, still on the diagonal",
            labels={"cause": ""},
        )
        hi = float(
            max(
                row_sample["acceptance_leakage_sgd"].quantile(0.99),
                row_sample["leakage_estimate_sgd"].quantile(0.99),
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=[0, hi], y=[0, hi], mode="lines", name="y = x",
                line=dict(color=theme["muted"], dash="dash"),
            )
        )
        fig2.update_xaxes(title="Hidden acceptance leakage (SGD)")
        fig2.update_yaxes(title="PPML estimate (SGD)")
        st.plotly_chart(style_fig(fig2, theme, height=420), width="stretch")
        graph_note(
            "Same test at trip × category grain (4k sample). More noise than the "
            "corridor plot, but the cloud still tracks the diagonal — coloured by cause. "
            "Amounts in SGD."
        )


        st.subheader("Proof the offers pay off: randomised holdout")
        explain(
            "Half the eligible travellers were randomly held back and got no "
            "offer. We compare what the treated group actually spent with what "
            "the model said the offers would add.",
            "This is the honest ROI check. If the predicted extra dollars match "
            "the real ones (dots on the diagonal), the offer engine is "
            "calibrated and worth funding, not just optimistic.",
        )
        hm = data["holdout_m"]
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Holdout T − C spend", money(hm["mean_spend_diff_treatment_minus_control"]))
        h2.metric(
            "95% CI (SGD)",
            f"{to_sgd(hm['spend_diff_ci_low']):,.0f} to {to_sgd(hm['spend_diff_ci_high']):,.0f}",
        )
        h3.metric("Predicted incremental / treated", money(hm["predicted_incremental_per_treated"]))
        h4.metric("Calibration (realized / predicted)", f"{hm['calibration_ratio']:.2f}")
        calib = data["calib"].copy()
        calib_sgd = calib.assign(
            predicted=calib["predicted"] * SGD_PER_USD,
            realized=calib["realized"] * SGD_PER_USD,
            calibration_gap=calib["calibration_gap"] * SGD_PER_USD,
        )
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=calib_sgd["predicted"],
                y=calib_sgd["realized"],
                mode="markers+text",
                text=calib_sgd["recommended_arm"] + " / " + calib_sgd["category"],
                textposition="top center",
                marker=dict(size=12, color=LIME),
                name="arm × category",
            )
        )
        hi = float(max(calib_sgd["predicted"].max(), calib_sgd["realized"].max()) * 1.15)
        fig.add_trace(
            go.Scatter(
                x=[0, hi], y=[0, hi], mode="lines", name="y = x",
                line=dict(color=theme["muted"], dash="dash"),
            )
        )
        fig.update_xaxes(title="Predicted incremental SGD")
        fig.update_yaxes(title="Realized incremental SGD")
        fig.update_layout(title="Uplift calibration")
        st.plotly_chart(style_fig(fig, theme, height=480), width="stretch")
        graph_note(
            "Each marker is an offer arm × category cell. X = dollars the model "
            "predicted the offer would add; Y = dollars the holdout actually "
            "added. On the dashed line means the offer engine is calibrated. "
            "Amounts in SGD."
        )
        st.caption("Currency: SGD")
        st.dataframe(
            calib_sgd.style.format(
                {
                    "predicted": "S${:,.2f}",
                    "realized": "S${:,.2f}",
                    "calibration_gap": "S${:,.2f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    with tab_match:
        st.subheader("Which merchants to sign up first")
        explain(
            "Amex already knows the region. This goes one level deeper: the "
            "actual shops to sign up in the low-acceptance areas, each with a "
            "typical spend per visit and how much local spending is still cash "
            "versus card.",
            "You cannot sign up a whole country, you sign up merchants. Ranking "
            "them by how much spend they unlock, where later shops in the same "
            "area are worth less than the first, turns the plan into a call list.",
        )
        md = data["merch_detail"]
        if md.empty:
            st.warning(
                "merchant_targets_detail.csv not found. Run "
                "`python -m match.merchant_targets_detail`."
            )
        else:
            md = md.copy()
            if f_region:
                md = md[md["dest_country"].map(REGION).isin(f_region)]
            if f_country:
                md = md[md["dest_country"].isin(f_country)]
            if f_cat:
                md = md[md["category"].isin(f_cat)]
            districts = sorted(md["dest_district"].unique())
            f_dist = st.multiselect(
                "District",
                districts,
                help="Zooms the map to that area. Tables below still focus on the selection.",
            )

            st.markdown("**Map view**")
            st.caption(
                "Numbered lime circles are clusters — zoom in to split them into "
                "smaller groups, then into individual merchant pins. Pick a district "
                "above to fly to that area (the map keeps every merchant visible)."
            )
            merchant_map(md, f_dist, theme)

            md_show = md[md["dest_district"].isin(f_dist)] if f_dist else md

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Merchant targets", f"{len(md_show):,}")
            k2.metric("Recoverable value", money(md_show["est_recoverable_value"].sum()))
            if len(md_show):
                k3.metric(
                    "Typical spend / visit",
                    f"S${to_sgd(md_show['price_low'].median()):,.0f}-"
                    f"{to_sgd(md_show['price_high'].median()):,.0f}",
                )
                k4.metric("Avg card share", f"{md_show['credit_share'].mean():.0%}")

            top = md_show.sort_values("est_recoverable_value", ascending=False).head(15)
            top_plot = top.assign(
                est_recoverable_value_sgd=top["est_recoverable_value"] * SGD_PER_USD
            )
            fig = px.bar(
                top_plot.sort_values("est_recoverable_value_sgd"),
                x="est_recoverable_value_sgd",
                y="merchant_name",
                color="category",
                orientation="h",
                color_discrete_sequence=PALETTE,
                title="Top merchants to onboard, by recoverable value",
                custom_data=["dest_district", "sub_category", "price_range", "visits"],
            )
            fig.update_traces(
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "District: %{customdata[0]}<br>"
                    "Type: %{customdata[1]}<br>"
                    "Spend per visit: %{customdata[2]}<br>"
                    "Visits seen: %{customdata[3]}<br>"
                    "Recoverable value: S$%{x:,.0f}<extra></extra>"
                )
            )
            fig.update_xaxes(title="Estimated recoverable value (SGD)")
            fig.update_yaxes(title="")
            st.plotly_chart(style_fig(fig, theme, height=520), width="stretch")
            graph_note(
                "Top merchants ranked by estimated recoverable value. Longer bars "
                "are the first acquiring calls; colour is spend category. Amounts in SGD."
            )

            show = md_show.sort_values("est_recoverable_value", ascending=False).head(200)
            st.caption("Currency: SGD (est_recoverable_value converted from USD)")
            st.dataframe(
                show[
                    [
                        "priority_rank",
                        "region",
                        "dest_country",
                        "dest_city",
                        "dest_district",
                        "category",
                        "sub_category",
                        "merchant_name",
                        "price_range",
                        "visits",
                        "est_recoverable_value",
                        "credit_share",
                        "cash_share",
                        "acceptance_density",
                    ]
                ].style.format(
                    {
                        "est_recoverable_value": format_sgd,
                        "credit_share": "{:.0%}",
                        "cash_share": "{:.0%}",
                        "acceptance_density": "{:.2f}",
                    }
                ),
                width="stretch",
                hide_index=True,
            )

            st.markdown("**Cash vs card by district** (where signing merchants is the fix)")
            byd = (
                md.groupby("dest_district", as_index=False)
                .agg(card=("credit_share", "mean"), cash=("cash_share", "mean"))
                .sort_values("cash", ascending=False)
                .head(12)
            )
            figc = px.bar(
                byd.melt("dest_district", ["card", "cash"], "kind", "share"),
                x="share",
                y="dest_district",
                color="kind",
                orientation="h",
                barmode="stack",
                color_discrete_map={"card": LIME, "cash": TERRACOTTA_M},
                title="Share of local spend: card vs cash",
            )
            figc.update_xaxes(title="", tickformat=".0%")
            figc.update_yaxes(title="")
            st.plotly_chart(style_fig(figc, theme, height=420), width="stretch")
            graph_note(
                "Share of local spend that is already on card (lime) vs still cash "
                "(terracotta) by district. High-cash districts are where signing "
                "merchants unlocks the most volume."
            )

            with st.expander("How we discover merchants (and the LLM judge)"):
                st.write(
                    "Candidate venues come from **OpenStreetMap / Google Maps POIs** "
                    "in each target district. NEMU scrapes the top tourist spots for "
                    "a corridor, then an LLM decides which are legitimate, "
                    "high-footfall businesses worth an acquiring call and drops "
                    "closed venues, duplicates, pure transit stops and global chains "
                    "that already take Amex."
                )
                st.markdown(
                    "**The LLM:** Google **Gemini** (`gemini-3.6-flash`) in "
                    "`match/discover_merchants.py`. It sends the candidate list to "
                    "Gemini's API with a strict JSON schema, so every verdict "
                    "(`keep`, `tourist_score`, `reason`) comes back clean. If the key "
                    "is missing, it uses a real backup set of verdicts we judged "
                    "ahead of time, so the numbers are always genuine, not made up."
                )
                disc = data.get("discovery", pd.DataFrame())
                if not disc.empty:
                    dshow = disc
                    if f_dist:
                        dshow = dshow[dshow["dest_district"].isin(f_dist)]
                    st.dataframe(
                        dshow[["dest_district", "merchant_name", "category",
                               "visits", "keep", "tourist_score", "reason"]]
                        .head(60),
                        width="stretch",
                        hide_index=True,
                    )
                if f_dist:
                    if st.button("Ask Gemini to judge this district (live)"):
                        from match.discover_merchants import judge_tourist_merchants
                        cands = [
                            {"name": r.merchant_name, "category": r.category,
                             "district": r.dest_district, "visits": int(r.visits)}
                            for r in md[md["dest_district"].isin(f_dist)]
                            .sort_values("visits", ascending=False).head(12).itertuples()
                        ]
                        with st.spinner("Asking Gemini to judge these venues..."):
                            verdicts, src = judge_tourist_merchants(
                                cands, ", ".join(f_dist)
                            )
                        st.caption(f"Source: {src}")
                        st.dataframe(pd.DataFrame(verdicts), width="stretch", hide_index=True)
                else:
                    st.caption("Pick a district above to run the judge live on its venues.")
                st.caption(
                    "Merchant names are real OSM venues; the recoverable value "
                    "attached to them is allocated from district-level estimates on "
                    "simulated capture at the district level."
                )

            with st.expander("District-level summary (how the acquiring budget is spread)"):
                merch = data["merchants"].copy()
                if f_country:
                    merch = merch[merch["dest_country"].isin(f_country)]
                signed = merch[merch["merchants_signed"] > 0].sort_values(
                    "recoverable_value", ascending=False
                )
                st.caption("Currency: SGD")
                st.dataframe(
                    signed[
                        [
                            "rank_by_value",
                            "dest_country",
                            "dest_district",
                            "acceptance_density",
                            "recoverable_value",
                            "merchants_signed",
                            "coverage",
                            "covered_value",
                        ]
                    ].style.format(
                        {
                            "acceptance_density": "{:.2f}",
                            "recoverable_value": format_sgd,
                            "coverage": "{:.0%}",
                            "covered_value": format_sgd,
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                )

            st.divider()
            rc1, rc2 = st.columns([3, 1])
            rc1.caption(
                "Merchant list is re-scanned on a schedule so the target list stays "
                "fresh as venues open and close."
            )
            with rc2:
                cadence = st.selectbox("Re-scan cadence", ["Every 3 months", "Every 6 months"])
                st.button("Run scan now", disabled=True)
                st.caption(f"Next scan: {cadence.split()[-1]} from last refresh.")

        st.subheader("Who to reward, and with what")
        explain(
            "Each traveller is sorted into one of five behaviour types, then "
            "followed from their top spending category down to a specific "
            "merchant and the exact offer to send.",
            "A blanket points offer wastes money on people who respond to cash "
            "or a merchant voucher instead. Matching the offer to the behaviour "
            "type turns rewards into real return.",
        )
        seg = data["segments"]
        if seg.empty:
            st.warning(
                "behavioral_segments.csv not found. Run "
                "`python -m match.behavioral_segments`."
            )
        else:
            if f_seg:
                seg = seg[seg["segment"].isin(f_seg)]
            targeted = seg[seg["is_targeted"]] if "is_targeted" in seg.columns else seg
            o1, o2, o3 = st.columns(3)
            o1.metric("Targeted travellers", f"{len(targeted):,}")
            o2.metric("Behaviour types", "5")
            o3.metric("Leakage at stake", money(targeted["leakage_estimate"].sum()))

            pattern_info = [
                (
                    "Points Optimiser",
                    "Historically increases spend when points multipliers appear",
                    "2x / 3x points",
                ),
                (
                    "Immediate Value Seeker",
                    "Stronger response to direct monetary savings",
                    "Cashback",
                ),
                (
                    "Threshold Chaser",
                    "Spend jumps when close to a reward threshold",
                    "Spend ¥10,000, get ¥1,000 back",
                ),
                (
                    "Category Loyalist",
                    "Consistently spends heavily in one category",
                    "Dining / retail / attraction-specific reward",
                ),
                (
                    "Merchant Explorer",
                    "Frequently tries new merchants and local businesses",
                    "Merchant-specific voucher",
                ),
            ]
            counts = seg["pattern"].value_counts()
            ref = pd.DataFrame(
                [
                    {
                        "Behavioural Patterns": p,
                        "What NEMU observes": obs,
                        "Possible Intervention": act,
                        "Travellers": int(counts.get(p, 0)),
                    }
                    for p, obs, act in pattern_info
                ]
            )
            st.dataframe(ref, width="stretch", hide_index=True)

            dist = (
                seg.groupby("pattern", as_index=False)
                .size()
                .rename(columns={"size": "travellers"})
            )
            order = [p for p, _, _ in pattern_info]
            dist["pattern"] = pd.Categorical(dist["pattern"], categories=order, ordered=True)
            figp = px.bar(
                dist.sort_values("pattern", ascending=False),
                x="travellers",
                y="pattern",
                orientation="h",
                color_discrete_sequence=[LIME],
                title="How many travellers of each shopper type",
            )
            figp.update_xaxes(title="Travellers")
            figp.update_yaxes(title="")
            st.plotly_chart(style_fig(figp, theme, height=360), width="stretch")
            graph_note(
                "Headcount by behavioural pattern. Each bar is one of the five "
                "shopper types that map to a concrete offer."
            )

            st.markdown("**Who gets which offer**")
            st.write(
                "Every row is one traveller. It shows what kind of shopper they "
                "are, the category and the place they spend most on, and the exact "
                "voucher we would send them. Use the filter to look at one shopper "
                "type at a time."
            )
            with st.expander("What does 'Predicted lift' mean?"):
                st.write(
                    "It is how much more we expect a traveller to spend if they get "
                    "the offer next to them. So 17% means we expect their spend in "
                    "that category to go up by about 17%.\n\n"
                    "The number is not a guess. A machine-learning model (a causal "
                    "forest, see the last tab) learns from past offers how different "
                    "kinds of travellers responded, then predicts the lift for each "
                    "person and each offer. NEMU picks the offer with the highest "
                    "lift for that person. We then prove those predictions hold up "
                    "with a real randomised test in the last tab."
                )
            avail = [p for p, _, _ in pattern_info if (targeted["pattern"] == p).any()]
            f_pat = st.multiselect("Show only these shopper types", avail)
            drill = targeted.copy()
            if f_pat:
                drill = drill[drill["pattern"].isin(f_pat)]
            drill = drill.sort_values("predicted_uplift", ascending=False)

            tbl = drill.head(50).copy()
            tbl["top_category"] = tbl["top_category"].astype(str).str.title()
            tbl = tbl[
                [
                    "member_id",
                    "segment",
                    "home_market",
                    "pattern",
                    "top_category",
                    "top_subcategory",
                    "top_merchant",
                    "recommended_voucher",
                    "predicted_uplift",
                ]
            ].rename(
                columns={
                    "member_id": "Traveller",
                    "segment": "Card tier",
                    "home_market": "From",
                    "pattern": "Behaviour type",
                    "top_category": "Top category",
                    "top_subcategory": "Sub-type",
                    "top_merchant": "Top merchant",
                    "recommended_voucher": "Voucher to send",
                    "predicted_uplift": "Predicted lift",
                }
            )
            st.dataframe(
                tbl.style.format({"Predicted lift": "{:.0%}"}),
                width="stretch",
                hide_index=True,
            )


if __name__ == "__main__":
    main()
