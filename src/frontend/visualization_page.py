"""
Visualization workspace for DataMind AI.

Supports:
- Automatic bivariate charts (scatter, line, bar, box)
- Geospatial mapping for datasets with geographic coordinates
- Correlation heatmaps
"""

from __future__ import annotations

import plotly.express as px
import streamlit as st
import uuid

from src.core.app_state import app_state
from src.visualisation.visualisation_engine import VisualizationEngine


def show_visualization_page() -> None:
    """Render the visualization workspace."""
    if not app_state.has_dataset():
        st.info("Upload a dataset to create visualizations.")
        return

    df = app_state.dataset
    st.markdown("## 📊 Visualization Workspace")
    st.caption("Generate bivariate comparisons, distribution plots, and interactive geographic maps.")

    tab_bivariate, tab_geo = st.tabs(["📈 Bivariate & Trends", "🗺 Geospatial Map"])

    with tab_bivariate:
        cols = df.columns.tolist()
        if len(cols) < 2:
            st.warning("Visualizations require at least two columns.")
            return

        c1, c2 = st.columns(2)
        with c1:
            col1 = st.selectbox("First Column (X-axis)", options=cols, index=0, key="vis_col1")
        with c2:
            col2 = st.selectbox("Second Column (Y-axis)", options=cols, index=min(1, len(cols) - 1), key="vis_col2")

        if st.button("Generate Visualization", type="primary"):
            try:
                engine = VisualizationEngine(df)
                fig = engine.generate(col1, col2)
                st.plotly_chart(fig, use_container_width=True, key=f"vis_chart_{uuid.uuid4()}")
            except Exception as exc:
                st.error(f"Visualization error: {exc}")

    with tab_geo:
        lat_candidates = [c for c in df.columns if c.lower() in ("latitude", "lat", "y")]
        lon_candidates = [c for c in df.columns if c.lower() in ("longitude", "lon", "lng", "x")]

        if lat_candidates and lon_candidates:
            st.success(f"Detected geographic columns: `{lat_candidates[0]}` and `{lon_candidates[0]}`")
            lat_col = st.selectbox("Latitude Column", options=lat_candidates, index=0)
            lon_col = st.selectbox("Longitude Column", options=lon_candidates, index=0)

            name_cols = [c for c in df.columns if c.lower() in ("name", "city", "location", "place", "country", "title")]
            hover_col = st.selectbox("Hover Label", options=["None"] + name_cols, index=0)

            if st.button("🗺 Plot Geospatial Map", type="primary"):
                df_geo = df.dropna(subset=[lat_col, lon_col])
                fig = px.scatter_geo(
                    df_geo,
                    lat=lat_col,
                    lon=lon_col,
                    hover_name=hover_col if hover_col != "None" else None,
                    title=f"Geographical Map ({len(df_geo)} points)",
                )
                fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True, key=f"geo_chart_{uuid.uuid4()}")
        else:
            st.info(
                "No latitude or longitude columns were detected in this dataset. "
                "Upload a dataset containing geographic coordinates to render maps."
            )
