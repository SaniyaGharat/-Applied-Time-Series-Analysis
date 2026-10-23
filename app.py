"""
Applied Time Series Analysis - Main Application Entrypoint.

Coordinates routing across development phases:
- Data: Ingestion, parsing, and regularization (Phase 1).
- EDA & Diagnostics: Stationarity, autocorrelation, decomposition, and transformations (Phase 2).
- Models: Model zoo, interactive fitting, per-model plots, residual diagnostics, grid search (Phases 3-4).
- Forecast: Model zoo comparison, holdout evaluation, and multi-step forecasting (Phase 5).
"""

import streamlit as st

from ui.compare_tab import render_compare_tab
from ui.dashboard_tab import render_dashboard_tab
from ui.data_tab import render_data_tab
from ui.eda_tab import render_eda_tab
from ui.forecast_tab import render_forecast_tab
from ui.models_tab import render_models_tab


def render_sidebar() -> None:
    """Render the sidebar branding."""
    st.sidebar.title("Applied Time Series Analysis")
    st.sidebar.caption("Econometric Modeling & Diagnostic Platform")


def init_session_state() -> None:
    """Initialize necessary session state keys if not already present."""
    defaults = {
        "raw_df": None,
        "source_name": None,
        "selected_date_col": None,
        "selected_target_col": None,
        "selected_exog_cols": [],
        "exog_df": None,
        "ts_series": None,
        "reg_report": None,
        "data_signature": None,
        "model_series": None,
        "model_base_series": None,
        "transform_info": {
            "steps": [],
            "params": {},
            "suggested_d": 0,
            "suggested_D": 0,
            "seasonal_period": None,
        },
        "model_results": {},
        "forecast_results": {},
        "model_mase_m_val": 1,
        "pending_spec": None,
        "selected_model_family": None,
        "uploader_key": 0,
        "cmp_bundles": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def main() -> None:
    """Main application execution pipeline and tab router."""
    st.set_page_config(
        page_title="Applied Time Series Analysis",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Subtle multi-tone pastel ambient background */
        .stApp {
            background: linear-gradient(135deg, #f5f3ff 0%, #fdf2f8 35%, #f0fdfa 70%, #f8fafc 100%) !important;
            background-attachment: fixed !important;
            color: #1e293b;
        }

        /* Sidebar: soft pastel surface */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #fbfbfe 0%, #f3f4fa 100%) !important;
            border-right: 1px solid rgba(226, 232, 240, 0.8) !important;
            box-shadow: 2px 0 12px rgba(99, 102, 241, 0.04);
        }

        /* Clean titles with subtle pastel emphasis */
        h1 {
            font-weight: 700 !important;
            letter-spacing: -0.02em;
            color: #1e1b4b !important;
        }
        h2, h3, h4, h5 {
            font-weight: 600 !important;
            letter-spacing: -0.01em;
            color: #312e81 !important;
        }

        /* Modern pill-style pastel tabs */
        div[data-testid="stTabs"] button[data-baseweb="tab"] {
            border-radius: 8px !important;
            padding: 8px 18px !important;
            font-weight: 500 !important;
            color: #64748b !important;
            border: none !important;
            background-color: transparent !important;
            transition: all 0.2s ease-in-out !important;
        }
        div[data-testid="stTabs"] button[data-baseweb="tab"]:hover {
            color: #4338ca !important;
            background-color: rgba(224, 231, 255, 0.5) !important;
        }
        div[data-testid="stTabs"] button[aria-selected="true"] {
            background-color: #e0e7ff !important;
            color: #3730a3 !important;
            font-weight: 600 !important;
            box-shadow: 0 1px 4px rgba(99, 102, 241, 0.12) !important;
        }
        div[data-testid="stTabs"] [data-testid="stTabsHeader"] {
            background: rgba(255, 255, 255, 0.75) !important;
            padding: 6px !important;
            border-radius: 12px !important;
            border: 1px solid rgba(226, 232, 240, 0.85) !important;
            backdrop-filter: blur(8px) !important;
            margin-bottom: 1.25rem !important;
        }

        /* Metric cards with soft pastel glow */
        [data-testid="stMetric"] {
            background: linear-gradient(135deg, #ffffff 0%, #f8faff 100%) !important;
            border: 1px solid rgba(199, 210, 254, 0.7) !important;
            border-radius: 12px !important;
            padding: 14px 18px !important;
            box-shadow: 0 2px 8px rgba(99, 102, 241, 0.04) !important;
        }
        [data-testid="stMetricLabel"] {
            color: #64748b !important;
            font-weight: 500 !important;
            font-size: 0.85rem !important;
        }
        [data-testid="stMetricValue"] {
            color: #1e1b4b !important;
            font-weight: 700 !important;
        }

        /* Modern pastel buttons */
        .stButton > button {
            border-radius: 10px !important;
            font-weight: 500 !important;
            border: 1px solid rgba(199, 210, 254, 0.85) !important;
            background: linear-gradient(135deg, #ffffff 0%, #eef2ff 100%) !important;
            color: #3730a3 !important;
            transition: all 0.15s ease-in-out !important;
            box-shadow: 0 1px 3px rgba(99, 102, 241, 0.08) !important;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%) !important;
            color: #312e81 !important;
            border-color: #a5b4fc !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.15) !important;
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #6366f1 0%, #818cf8 100%) !important;
            color: #ffffff !important;
            border: none !important;
            box-shadow: 0 2px 8px rgba(99, 102, 241, 0.25) !important;
        }
        .stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%) !important;
            box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35) !important;
        }

        /* Alert boxes in pastel tones */
        [data-testid="stAlert"] {
            border-radius: 10px !important;
        }

        /* Plotly charts styled as pastel cards */
        div.js-plotly-plot {
            background: #ffffff !important;
            border-radius: 12px !important;
            padding: 8px !important;
            border: 1px solid rgba(226, 232, 240, 0.85) !important;
            box-shadow: 0 2px 10px rgba(100, 116, 139, 0.04) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    render_sidebar()

    st.title("Applied Time Series Analysis")
    st.caption("End-to-End Time Series Modeling & Diagnostic Platform")

    # Primary Tab Router
    (
        tab_data,
        tab_eda,
        tab_models,
        tab_forecast,
        tab_compare,
        tab_dashboard,
    ) = st.tabs(
        [
            "Data",
            "EDA & Diagnostics",
            "Models",
            "Forecast & Compare",
            "Compare Datasets",
            "Dashboard",
        ]
    )

    with tab_data:
        render_data_tab()

    with tab_eda:
        render_eda_tab()

    with tab_models:
        render_models_tab()

    with tab_forecast:
        render_forecast_tab()

    with tab_compare:
        render_compare_tab()

    with tab_dashboard:
        render_dashboard_tab()


if __name__ == "__main__":
    main()

