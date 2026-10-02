"""
frontend/app.py
────────────────
Streamlit map frontend for the Kenya Land Readiness Atlas.

Features:
- Leaflet map (via streamlit-folium) showing land-cover tile layer
- Click anywhere to get land-cover predictions and recommendations
- County search with bar-chart land-cover mix
- Built-up trend panel (static chart or EE time series if enabled)
- Hazard flag legend
- Exportable county-level table (dissertation join table)

Run locally:
    streamlit run frontend/app.py

Or from Docker:
    docker compose up frontend
"""

from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import folium
import pandas as pd
import requests
import streamlit as st
import yaml
from streamlit_folium import st_folium

# ── Config ────────────────────────────────────────────────────────────────────
_ROOT    = Path(__file__).resolve().parents[1]
_CFG     = yaml.safe_load((_ROOT / "configs" / "config.yaml").read_text())
_RGN     = yaml.safe_load((_ROOT / "configs" / "regions.yaml").read_text())

CLASSES  = _RGN["classes"]
COLOURS  = _RGN["class_colours"]
API_URL  = os.getenv("API_URL", "http://localhost:8000")

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Kenya Land Readiness Atlas",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.atlas-hero {
    background: linear-gradient(135deg, #0d1117 0%, #1a2744 50%, #0d2b1a 100%);
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    border: 1px solid #2d3748;
}
.atlas-hero h1 { color: #7ec8e3; font-size: 2.2rem; font-weight: 700; margin: 0; }
.atlas-hero p  { color: #a0aec0; font-size: 0.95rem; margin: 0.5rem 0 0 0; }

.result-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.result-card h3 { color: #58a6ff; font-size: 1.1rem; margin: 0 0 0.5rem 0; }

.action-badge {
    display: inline-block;
    padding: 0.3rem 0.9rem;
    border-radius: 99px;
    font-weight: 600;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.badge-settle  { background: #1f4e79; color: #90cdf4; }
.badge-farm    { background: #1e4620; color: #9ae6b4; }
.badge-protect { background: #4a1a00; color: #fbd38d; }

.conf-high   { color: #48bb78; font-weight: 600; }
.conf-medium { color: #ecc94b; font-weight: 600; }
.conf-low    { color: #f56565; font-weight: 600; }

.class-bar-label { font-size: 0.8rem; color: #a0aec0; }
.disclaimer { font-size: 0.75rem; color: #718096; font-style: italic; margin-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="atlas-hero">
  <h1>🛰️ Kenya Land Readiness Atlas</h1>
  <p>Sentinel-2 land-cover analysis · built-up trends · settle / farm / protect recommendations</p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🗺️ Navigate")
    mode = st.radio("Mode", ["Click on map", "County search"], label_visibility="collapsed")
    st.divider()
    st.markdown("### ⚙️ Settings")
    show_tiles   = st.toggle("Show land-cover tile layer", value=True)
    show_hazard  = st.toggle("Show flood-hazard overlay",  value=False)
    st.divider()
    st.markdown("### 🎨 Class legend")
    for cls, colour in COLOURS.items():
        st.markdown(
            f"<span style='display:inline-block;width:14px;height:14px;"
            f"background:{colour};border-radius:3px;margin-right:6px'></span>"
            f"<span style='font-size:0.85rem'>{cls.replace('_',' ').title()}</span>",
            unsafe_allow_html=True,
        )
    st.divider()
    st.caption("Model: EfficientNet-B0 fine-tuned on Sentinel-2 · Labels: WorldCover 2021")


# ── Map ───────────────────────────────────────────────────────────────────────
col_map, col_results = st.columns([3, 1.2])

with col_map:
    m = folium.Map(
        location=[0.5, 37.5],
        zoom_start=6,
        tiles="CartoDB dark_matter",
        prefer_canvas=True,
    )

    if show_tiles:
        folium.TileLayer(
            tiles=f"{API_URL}/tiles/{{z}}/{{x}}/{{y}}.png",
            attr="Kenya Land Readiness Atlas",
            name="Land Cover",
            opacity=0.65,
            overlay=True,
        ).add_to(m)

    # Study region boxes
    for region, info in _RGN["regions"].items():
        bb = info["bbox"]  # [min_lon, min_lat, max_lon, max_lat]
        folium.Rectangle(
            bounds=[[bb[1], bb[0]], [bb[3], bb[2]]],
            color="#7ec8e3", weight=1.5, fill=False,
            tooltip=f"{region.replace('_',' ').title()} study region",
        ).add_to(m)

    map_data = st_folium(m, height=600, width=None, returned_objects=["last_clicked"])


# ── Point prediction ──────────────────────────────────────────────────────────
with col_results:
    st.markdown("#### 📍 Point analysis")

    clicked = map_data.get("last_clicked")
    if mode == "Click on map" and clicked:
        lat, lon = clicked["lat"], clicked["lng"]
        st.markdown(f"**Lat:** `{lat:.5f}`  **Lon:** `{lon:.5f}`")

        with st.spinner("Running inference …"):
            try:
                resp = requests.post(
                    f"{API_URL}/predict/point",
                    json={"lat": lat, "lon": lon, "flood_risk": show_hazard},
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                st.error(f"API error: {exc}")
                data = None

        if data:
            probs = data["probabilities"]
            rec   = data["recommendation"]
            action= rec["action"]

            badge_cls = {"settle": "badge-settle", "farm": "badge-farm", "protect": "badge-protect"}.get(action, "")
            conf_cls  = {"high": "conf-high", "medium": "conf-medium", "low": "conf-low"}.get(rec["confidence"], "")

            st.markdown(f"""
            <div class="result-card">
              <h3>Recommendation</h3>
              <span class="action-badge {badge_cls}">{action}</span>
              &nbsp; <span class="{conf_cls}">{rec['confidence']} confidence</span>
              <p style="color:#a0aec0;font-size:0.85rem;margin-top:0.6rem">{rec['rationale']}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**Land-cover mix**")
            bar_data = pd.DataFrame({
                "Class":       [c.replace("_", " ").title() for c in probs],
                "Probability": list(probs.values()),
                "Colour":      [COLOURS[c] for c in probs],
            })
            import altair as alt
            chart = (
                alt.Chart(bar_data)
                .mark_bar(cornerRadius=4)
                .encode(
                    x=alt.X("Probability:Q", axis=alt.Axis(format=".0%"), scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y("Class:N", sort="-x"),
                    color=alt.Color("Colour:N", scale=None),
                    tooltip=["Class", alt.Tooltip("Probability:Q", format=".1%")],
                )
                .properties(height=200)
            )
            st.altair_chart(chart, use_container_width=True)

            st.markdown(
                '<p class="disclaimer">Scores are agreement with WorldCover 2021, '
                'not ground truth. Validate against hand-labelled points before citing.</p>',
                unsafe_allow_html=True,
            )

            # ── GenAI / RAG Spatial Underwriting Copilot UI ──
            st.markdown("---")
            if st.button("🤖 Generate GenAI Underwriting Dossier", use_container_width=True, type="primary"):
                with st.spinner("Generating RAG spatial underwriting report..."):
                    try:
                        copilot_resp = requests.post(
                            f"{API_URL}/underwrite/copilot",
                            json={
                                "lat": lat,
                                "lon": lon,
                                "probabilities": probs,
                                "prediction": data["prediction"],
                                "recommendation": rec,
                                "flood_risk": show_hazard,
                                "slope_deg": 4.5,
                                "county_name": "Kenya Spatial Corridor"
                            },
                            timeout=15,
                        )
                        copilot_resp.raise_for_status()
                        copilot_data = copilot_resp.json()
                        st.markdown(copilot_data["dossier"])
                        if copilot_data.get("llm_powered"):
                            st.caption("✨ Powered by Google Gemini API + RAG Statutory Knowledge Base")
                        else:
                            st.caption("⚡ Powered by Deterministic Regulatory RAG Engine (Local Fallback)")
                    except Exception as copilot_err:
                        st.error(f"Copilot service error: {copilot_err}")


    elif mode == "County search":
        county = st.text_input("County name (e.g. Kirinyaga)", placeholder="Enter county …")
        if county:
            with st.spinner("Loading county summary …"):
                try:
                    resp = requests.get(f"{API_URL}/predict/county/{county}", timeout=10)
                    if resp.status_code == 404:
                        st.warning("No pre-computed summary. Run the county aggregation pipeline first.")
                    else:
                        resp.raise_for_status()
                        d = resp.json()
                        st.json(d)
                except Exception as exc:
                    st.error(str(exc))
    else:
        st.info("Click on the map to analyse a location.")


# ── Built-up trend panel ──────────────────────────────────────────────────────
st.divider()
st.markdown("### 📈 Built-up trend (demonstration data)")
st.caption("Replace with real EE time series once NDBI sequences are exported.")

import numpy as np
import altair as alt

years  = list(range(2017, 2024))
_trend = {
    "highland":   [0.04, 0.048, 0.055, 0.062, 0.071, 0.079, 0.089],
    "coast":      [0.09, 0.097, 0.108, 0.117, 0.128, 0.135, 0.148],
    "arid":       [0.01, 0.011, 0.012, 0.013, 0.014, 0.016, 0.017],
    "lake_basin": [0.07, 0.078, 0.088, 0.097, 0.108, 0.118, 0.131],
}
rows = [
    {"Year": y, "Built-up share": v, "Region": r.replace("_", " ").title()}
    for r, vals in _trend.items() for y, v in zip(years, vals)
]
trend_df = pd.DataFrame(rows)
chart2 = (
    alt.Chart(trend_df)
    .mark_line(point=True)
    .encode(
        x="Year:O",
        y=alt.Y("Built-up share:Q", axis=alt.Axis(format=".0%")),
        color="Region:N",
        tooltip=["Region", "Year", alt.Tooltip("Built-up share:Q", format=".1%")],
    )
    .properties(height=250)
    .configure_view(strokeWidth=0)
    .configure_axis(grid=True, gridColor="#2d3748")
    .configure_legend(labelColor="#a0aec0", titleColor="#a0aec0")
)
st.altair_chart(chart2, use_container_width=True)

# ── County export table ───────────────────────────────────────────────────────
st.divider()
st.markdown("### 📊 County-level export table")
st.caption("Joins to dissertation HFVS and land allocation work as buildability input.")

county_path = _ROOT / "data" / "county_summaries" / "all_counties.csv"
if county_path.exists():
    county_df = pd.read_csv(county_path)
    st.dataframe(county_df, use_container_width=True)
    st.download_button("⬇ Download CSV", county_df.to_csv(index=False),
                       file_name="kenya_land_readiness.csv", mime="text/csv")
else:
    st.info("County table not generated yet. Run `python -m data.pipeline.county_aggregation` after training.")
