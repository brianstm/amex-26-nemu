"""NEMU Streamlit dashboard. Reads files in ``outputs/``."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.config import OUTPUT_DIR

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
        "metrics": metrics,
        "holdout_m": holdout_m,
    }


def money(x: float) -> str:
    ax = abs(x)
    if ax >= 1_000_000:
        return f"${x/1_000_000:,.2f}M"
    if ax >= 1_000:
        return f"${x/1_000:,.0f}K"
    return f"${x:,.0f}"


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
        .nemu-wordmark {{
            display: inline-block;
            background: {LIME};
            padding: 0.12rem 1.35rem 0.02rem 1.35rem;
            line-height: 1;
        }}
        .nemu-wordmark span {{
            font-family: "Montserrat", sans-serif;
            font-weight: 900;
            font-size: 3.4rem;
            color: {CHARCOAL};
            letter-spacing: -0.045em;
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
        header[data-testid="stHeader"] {{ background: {theme["header_bg"]}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_filters(df: pd.DataFrame, country, category, segment) -> pd.DataFrame:
    out = df
    if country:
        out = out[out["dest_country"].isin(country)]
    if category:
        out = out[out["category"].isin(category)]
    if segment and "segment" in out.columns:
        out = out[out["segment"].isin(segment)]
    return out


def main() -> None:
    data = load_all()
    m = data["metrics"]
    leakage = data["leakage"]

    mode = st.session_state.get("theme_mode", "Dark")
    theme = THEMES[mode]
    inject_css(theme)
    st.markdown(
        """
        <div class="nemu-wordmark"><span>Nemu</span></div>
        <div class="nemu-pillars">Notice &nbsp; Explain &nbsp; Match &nbsp; Uplift</div>
        <div class="nemu-note">Hybrid ledger: ISO currencies, World Bank FX, OSM merchant names. Trips and acceptance are simulated. No American Express data.</div>
        """,
        unsafe_allow_html=True,
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
        f'<div class="nemu-claim"><strong>Demo claim.</strong> We hid {money(m["hidden_acceptance_leakage"])} of coverage leakage. '
        f'The gravity model found {money(m["estimated_leakage"])} '
        f'({m["recovery_ratio_vs_acceptance"]:.0%}) without seeing the labels, '
        f'and attributed {m.get("cause_value_weighted_accuracy", 0):.0%} of estimated '
        f'dollars to the correct cause.</div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Appearance")
        st.radio(
            "Mode",
            ["Dark", "Light"],
            horizontal=True,
            key="theme_mode",
            label_visibility="collapsed",
        )
        st.header("Filters")
        countries = sorted(leakage["dest_country"].unique())
        categories = sorted(leakage["category"].unique())
        segments = sorted(leakage["segment"].unique())
        f_country = st.multiselect("Destination", countries)
        f_cat = st.multiselect("Category", categories)
        f_seg = st.multiselect("Segment", segments)

    view = apply_filters(leakage, f_country, f_cat, f_seg)
    merged_view = apply_filters(data["merged"], f_country, f_cat, f_seg)

    tab_rank, tab_drill, tab_cause, tab_val, tab_merch, tab_hold = st.tabs(
        [
            "Corridor ranking",
            "Drill-down",
            "Causes",
            "Validation",
            "Merchant targets",
            "Holdout",
        ]
    )

    with tab_rank:
        st.subheader("Observed volume is not recoverable value")
        st.write(
            "Left: destinations ranked the way a dashboard of *captured* spend would "
            "rank them. Right: the same destinations ranked by Notice's recoverable "
            "coverage leakage. Sparse, cheaper markets jump up the list — that is the "
            "product."
        )
        agg = (
            view.groupby("dest_country", as_index=False)
            .agg(
                observed=("observed_spend", "sum"),
                recoverable=("leakage_estimate", "sum"),
            )
        )
        left = agg.sort_values("observed", ascending=True)
        right = agg.sort_values("recoverable", ascending=True)
        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.bar(
                left,
                x="observed",
                y="dest_country",
                orientation="h",
                color_discrete_sequence=[theme["bar_observed"]],
                title="Ranked by observed Amex volume",
            )
            fig.update_xaxes(title="Observed spend (USD)")
            fig.update_yaxes(title="")
            st.plotly_chart(style_fig(fig, theme), width="stretch")
        with col_b:
            fig = px.bar(
                right,
                x="recoverable",
                y="dest_country",
                orientation="h",
                color_discrete_sequence=[theme["bar_recoverable"]],
                title="Ranked by recoverable leakage",
            )
            fig.update_xaxes(title="Estimated recoverable (USD)")
            fig.update_yaxes(title="")
            st.plotly_chart(style_fig(fig, theme), width="stretch")

        rank = agg.copy()
        rank["rank_observed"] = rank["observed"].rank(ascending=False).astype(int)
        rank["rank_recoverable"] = rank["recoverable"].rank(ascending=False).astype(int)
        rank["rank_shift"] = rank["rank_observed"] - rank["rank_recoverable"]
        st.dataframe(
            rank.sort_values("recoverable", ascending=False).style.format(
                {"observed": "${:,.0f}", "recoverable": "${:,.0f}"}
            ),
            width="stretch",
            hide_index=True,
        )

    with tab_drill:
        st.subheader("Corridor → category → segment")
        grain = st.radio("Grain", ["country", "country × category", "country × category × segment"], horizontal=True)
        if grain == "country":
            keys = ["dest_country"]
        elif grain == "country × category":
            keys = ["dest_country", "category"]
        else:
            keys = ["dest_country", "category", "segment"]
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
        st.dataframe(
            table.style.format(
                {
                    "observed_spend": "${:,.0f}",
                    "recoverable": "${:,.0f}",
                    "mean_acceptance": "{:.2f}",
                    "mean_cash": "{:.2f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

        txns = data.get("txns", pd.DataFrame())
        if not txns.empty and {"merchant_name", "currency", "amount_local"}.issubset(txns.columns):
            st.subheader("Issuer tickets (real currency + OSM merchants)")
            st.write(
                "Spend and which trip the ticket belongs to are simulated. "
                "`currency` is ISO 4217, `amount_local` uses World Bank FX, "
                "`merchant_name` is an OpenStreetMap POI (or a public transport operator), "
                "`mcc` is ISO 18245. This is not an Amex merchant file."
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
            st.dataframe(
                sample[cols].head(25),
                width="stretch",
                hide_index=True,
            )

    with tab_cause:
        st.subheader("Which lever, per destination")
        st.write(
            "no_acceptance → send acquiring. rail_substitution → send an offer. "
            "cash → do not confuse a cash culture with a coverage hole. "
            "no_demand → do nothing."
        )
        cause_cty = (
            view.groupby(["dest_country", "cause"], as_index=False)["leakage_estimate"]
            .sum()
            .rename(columns={"leakage_estimate": "usd"})
        )
        fig = px.bar(
            cause_cty,
            x="dest_country",
            y="usd",
            color="cause",
            color_discrete_map=CAUSE_COLORS,
            title="Recoverable value by cause",
            labels={"cause": "", "usd": "USD", "dest_country": ""},
        )
        fig.update_xaxes(title="")
        fig.update_yaxes(title="USD")
        st.plotly_chart(style_fig(fig, theme, height=560), width="stretch")

        pie = view.groupby("cause", as_index=False)["leakage_estimate"].sum()
        figp = px.pie(
            pie,
            names="cause",
            values="leakage_estimate",
            color="cause",
            color_discrete_map=CAUSE_COLORS,
            title="Portfolio mix",
            hole=0.45,
            labels={"cause": ""},
        )
        st.plotly_chart(style_fig(figp, theme, height=380), width="stretch")

    with tab_val:
        st.subheader("Hidden-file test — the slide that answers “you don’t have real data”")
        st.write(
            "ground_truth.csv was written by the DGP and **never entered the likelihood**. "
            "Each point is a destination × category corridor. The dashed line is y = x."
        )
        cv = data["corridor_val"].copy()
        if f_country:
            cv = cv[cv["dest_country"].isin(f_country)]
        if f_cat:
            cv = cv[cv["category"].isin(f_cat)]
        lim = float(max(cv["est"].max(), cv["truth_acc"].max()) * 1.05)
        fig = px.scatter(
            cv,
            x="truth_acc",
            y="est",
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
        fig.update_xaxes(title="Hidden acceptance leakage (USD)", range=[0, lim])
        fig.update_yaxes(title="PPML estimate (USD)", range=[0, lim])
        st.plotly_chart(style_fig(fig, theme, height=520), width="stretch")

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
        )
        fig2 = px.scatter(
            row_sample,
            x="acceptance_leakage",
            y="leakage_estimate",
            color="cause",
            opacity=0.35,
            color_discrete_map=CAUSE_COLORS,
            title="Trip × category (4k sample) — noisier, still on the diagonal",
            labels={"cause": ""},
        )
        hi = float(
            max(row_sample["acceptance_leakage"].quantile(0.99), row_sample["leakage_estimate"].quantile(0.99))
        )
        fig2.add_trace(
            go.Scatter(
                x=[0, hi], y=[0, hi], mode="lines", name="y = x",
                line=dict(color=theme["muted"], dash="dash"),
            )
        )
        st.plotly_chart(style_fig(fig2, theme, height=420), width="stretch")

    with tab_merch:
        st.subheader("Where to sign the next merchant")
        st.write(
            "Greedy submodular selection on `no_acceptance` districts. "
            "Diminishing returns: the fifth restaurant in Ubud is worth less than "
            "the first one in the Old Quarter."
        )
        merch = data["merchants"].copy()
        if f_country:
            merch = merch[merch["dest_country"].isin(f_country)]
        signed = merch[merch["merchants_signed"] > 0].sort_values("recoverable_value", ascending=False)
        fig = px.bar(
            signed.sort_values("covered_value"),
            x="covered_value",
            y="dest_district",
            color="dest_country",
            orientation="h",
            color_discrete_sequence=PALETTE,
            title="Covered recoverable value after greedy assignment",
            hover_data=["merchants_signed", "acceptance_density"],
        )
        st.plotly_chart(style_fig(fig, theme, height=420), width="stretch")
        st.dataframe(
            signed[
                [
                    "rank_by_value",
                    "dest_country",
                    "dest_city",
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
                    "recoverable_value": "${:,.0f}",
                    "coverage": "{:.0%}",
                    "covered_value": "${:,.0f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    with tab_hold:
        st.subheader("Did the offer model pay for itself?")
        hm = data["holdout_m"]
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Holdout T − C spend", f"${hm['mean_spend_diff_treatment_minus_control']:,.0f}")
        h2.metric("95% CI", f"${hm['spend_diff_ci_low']:,.0f} – ${hm['spend_diff_ci_high']:,.0f}")
        h3.metric("Predicted incremental / treated", f"${hm['predicted_incremental_per_treated']:,.0f}")
        h4.metric("Calibration (realized / predicted)", f"{hm['calibration_ratio']:.2f}")
        calib = data["calib"].copy()
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=calib["predicted"],
                y=calib["realized"],
                mode="markers+text",
                text=calib["recommended_arm"] + " / " + calib["category"],
                textposition="top center",
                marker=dict(size=12, color=LIME),
                name="arm × category",
            )
        )
        hi = float(max(calib["predicted"].max(), calib["realized"].max()) * 1.15)
        fig.add_trace(
            go.Scatter(
                x=[0, hi], y=[0, hi], mode="lines", name="y = x",
                line=dict(color=theme["muted"], dash="dash"),
            )
        )
        fig.update_xaxes(title="Predicted incremental $")
        fig.update_yaxes(title="Realized incremental $")
        fig.update_layout(title="Uplift calibration")
        st.plotly_chart(style_fig(fig, theme, height=480), width="stretch")
        st.dataframe(
            calib.style.format(
                {"predicted": "${:,.2f}", "realized": "${:,.2f}", "calibration_gap": "${:,.2f}"}
            ),
            width="stretch",
            hide_index=True,
        )


if __name__ == "__main__":
    main()
