"""app.py — Streamlit frontend for the House Price Prediction project.

Run with:
    streamlit run app.py
from inside the house_price_project/ directory.
"""

import os
import sys
import json

# Ensure src/ is importable regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from src.predict import predict, get_best_model_name, models_are_trained
from src.data_loader import load_data
from src.preprocessing import FEATURE_COLS

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "house_price_regression_dataset.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")

MODEL_NAMES = ["Linear Regression", "Random Forest", "XGBoost"]

# ── Guard — models must be trained first ─────────────────────────────────────
if not models_are_trained():
    st.error(
        "⚠️ **Models not found.** Please train the models first by running:\n\n"
        "```bash\npython src/train.py\n```"
    )
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏠 House Price Predictor")
    st.markdown("---")

    best_model = get_best_model_name()
    default_idx = MODEL_NAMES.index(best_model) if best_model in MODEL_NAMES else 0

    selected_model = st.selectbox(
        "🤖 Model",
        MODEL_NAMES,
        index=default_idx,
        help=f"Best performing model (lowest RMSE): **{best_model}**",
    )

    st.markdown(f"**Auto-selected best:** `{best_model}`")
    st.markdown("---")
    st.caption("Adjust the inputs on the Predict tab and click **Predict Price**.")

# ── Main Tabs ─────────────────────────────────────────────────────────────────
tab_predict, tab_compare, tab_explore = st.tabs(
    ["🏷️ Predict Price", "📊 Model Comparison", "🔍 Data Explorer"]
)

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.header("Predict House Price")
    st.markdown(f"Using model: **{selected_model}**")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        square_footage = st.slider(
            "📐 Square Footage", min_value=400, max_value=6000, value=2000, step=50
        )
        num_bedrooms = st.slider(
            "🛏️ Bedrooms", min_value=1, max_value=6, value=3, step=1
        )
        num_bathrooms = st.slider(
            "🚿 Bathrooms", min_value=1, max_value=5, value=2, step=1
        )
        year_built = st.slider(
            "🏗️ Year Built", min_value=1950, max_value=2024, value=2000, step=1
        )

    with col2:
        lot_size = st.number_input(
            "🌿 Lot Size (acres)", min_value=0.1, max_value=10.0, value=2.0, step=0.1,
            format="%.2f"
        )
        garage_size = st.selectbox(
            "🚗 Garage Size (cars)", options=[0, 1, 2], index=1
        )
        neighborhood_quality = st.slider(
            "⭐ Neighborhood Quality (1–10)", min_value=1, max_value=10, value=5, step=1
        )

    st.markdown("---")
    predict_btn = st.button("💰 Predict Price", type="primary", use_container_width=True)

    if predict_btn:
        features = {
            "Square_Footage": square_footage,
            "Num_Bedrooms": num_bedrooms,
            "Num_Bathrooms": num_bathrooms,
            "Year_Built": year_built,
            "Lot_Size": lot_size,
            "Garage_Size": garage_size,
            "Neighborhood_Quality": neighborhood_quality,
        }
        with st.spinner("Running prediction …"):
            price = predict(features, selected_model)

        st.markdown("### Estimated Price")
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.metric(
                label=f"Predicted by {selected_model}",
                value=f"${price:,.0f}",
            )

        # Feature summary table
        with st.expander("📋 Input summary"):
            summary_df = pd.DataFrame(
                {"Feature": list(features.keys()), "Value": list(features.values())}
            )
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL COMPARISON
# ════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.header("Model Comparison")
    st.markdown("Evaluation metrics on the **20 % held-out test set**.")
    st.markdown("---")

    with open(METRICS_PATH) as f:
        metrics = json.load(f)

    # Metrics table
    rows = []
    for name, m in metrics.items():
        rows.append(
            {
                "Model": name,
                "R²": m["r2"],
                "MAE ($)": f"{m['mae']:,.0f}",
                "RMSE ($)": f"{m['rmse']:,.0f}",
            }
        )
    metrics_df = pd.DataFrame(rows)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    st.markdown("---")

    names = list(metrics.keys())
    r2_vals = [metrics[n]["r2"] for n in names]
    mae_vals = [metrics[n]["mae"] for n in names]
    rmse_vals = [metrics[n]["rmse"] for n in names]

    COLORS = ["#3b82d4", "#7c5cd8", "#16a34a"]

    col_r2, col_mae, col_rmse = st.columns(3)

    with col_r2:
        fig = go.Figure(go.Bar(x=names, y=r2_vals, marker_color=COLORS, text=[f"{v:.4f}" for v in r2_vals], textposition="outside"))
        fig.update_layout(title="R² Score (higher = better)", yaxis_range=[0, 1.1], height=380, margin=dict(t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_mae:
        fig = go.Figure(go.Bar(x=names, y=mae_vals, marker_color=COLORS, text=[f"${v:,.0f}" for v in mae_vals], textposition="outside"))
        fig.update_layout(title="MAE — Mean Absolute Error (lower = better)", height=380, margin=dict(t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_rmse:
        fig = go.Figure(go.Bar(x=names, y=rmse_vals, marker_color=COLORS, text=[f"${v:,.0f}" for v in rmse_vals], textposition="outside"))
        fig.update_layout(title="RMSE — Root Mean Squared Error (lower = better)", height=380, margin=dict(t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.info(f"🏆 **Best model (lowest RMSE):** {best_model}", icon="ℹ️")

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — DATA EXPLORER
# ════════════════════════════════════════════════════════════════════════════
with tab_explore:
    st.header("Data Explorer")
    st.markdown("---")

    @st.cache_data
    def load_dataset():
        return load_data(DATA_PATH)

    df = load_dataset()
    st.markdown(f"**{len(df):,} rows** · **{len(df.columns)} columns**")

    # ── Raw data preview ──
    with st.expander("📄 Raw data preview (first 100 rows)"):
        st.dataframe(df.head(100), use_container_width=True, hide_index=True)

    st.markdown("---")
    col_s, col_h = st.columns(2)

    # ── Scatter: Square Footage vs Price ──
    with col_s:
        st.subheader("Square Footage vs Price")
        fig_scatter = px.scatter(
            df,
            x="Square_Footage",
            y="House_Price",
            color="Neighborhood_Quality",
            color_continuous_scale="Viridis",
            labels={
                "Square_Footage": "Square Footage (sq ft)",
                "House_Price": "House Price ($)",
                "Neighborhood_Quality": "Neighborhood Quality",
            },
            opacity=0.6,
            height=420,
        )
        fig_scatter.update_layout(margin=dict(t=30, b=20))
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Histogram: House Price distribution ──
    with col_h:
        st.subheader("House Price Distribution")
        fig_hist = px.histogram(
            df,
            x="House_Price",
            nbins=50,
            color_discrete_sequence=["#3b82d4"],
            labels={"House_Price": "House Price ($)"},
            height=420,
        )
        fig_hist.update_layout(margin=dict(t=30, b=20), bargap=0.05)
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")

    # ── Correlation heatmap ──
    st.subheader("Correlation Heatmap")
    fig_heat, ax = plt.subplots(figsize=(9, 6))
    corr = df[FEATURE_COLS + ["House_Price"]].corr()
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Feature Correlation Matrix", fontsize=13, pad=12)
    plt.tight_layout()
    st.pyplot(fig_heat)
    plt.close(fig_heat)

    st.markdown("---")

    # ── Box plots: Price by Bedrooms / Neighborhood Quality ──
    col_b1, col_b2 = st.columns(2)

    with col_b1:
        st.subheader("Price by Number of Bedrooms")
        fig_box = px.box(
            df,
            x="Num_Bedrooms",
            y="House_Price",
            color="Num_Bedrooms",
            labels={"Num_Bedrooms": "Bedrooms", "House_Price": "Price ($)"},
            color_discrete_sequence=px.colors.qualitative.Set2,
            height=380,
        )
        fig_box.update_layout(showlegend=False, margin=dict(t=30, b=20))
        st.plotly_chart(fig_box, use_container_width=True)

    with col_b2:
        st.subheader("Price by Neighborhood Quality")
        fig_box2 = px.box(
            df,
            x="Neighborhood_Quality",
            y="House_Price",
            color="Neighborhood_Quality",
            labels={"Neighborhood_Quality": "Neighborhood Quality", "House_Price": "Price ($)"},
            color_discrete_sequence=px.colors.qualitative.Pastel,
            height=380,
        )
        fig_box2.update_layout(showlegend=False, margin=dict(t=30, b=20))
        st.plotly_chart(fig_box2, use_container_width=True)
