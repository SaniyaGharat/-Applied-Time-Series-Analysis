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

        /* Clean neutral warm background */
        .stApp {
            background-color: #fdfbf9 !important;
            color: #1c1917;
        }

        /* Sidebar: crisp warm surface */
        [data-testid="stSidebar"] {
            background-color: #ffffff !important;
            border-right: 1px solid #f2ebe5 !important;
            box-shadow: 1px 0 4px rgba(0, 0, 0, 0.02);
        }

        /* Clean titles in warm charcoal */
        h1 {
            font-weight: 700 !important;
            letter-spacing: -0.02em;
            color: #1c1917 !important;
        }
        h2, h3, h4, h5 {
            font-weight: 600 !important;
            letter-spacing: -0.01em;
            color: #292524 !important;
        }

        /* Modern pill tabs with single pastel warm peach/sand accent */
        div[data-testid="stTabs"] button[data-baseweb="tab"] {
            border-radius: 8px !important;
            padding: 8px 18px !important;
            font-weight: 500 !important;
            color: #78716c !important;
            border: none !important;
            background-color: transparent !important;
            transition: all 0.2s ease-in-out !important;
        }
        div[data-testid="stTabs"] button[data-baseweb="tab"]:hover {
            color: #a35d41 !important;
            background-color: rgba(251, 240, 234, 0.7) !important;
        }
        div[data-testid="stTabs"] button[aria-selected="true"] {
            background-color: #fbf0ea !important;
            color: #8c462b !important;
            font-weight: 600 !important;
            border: 1px solid #ebd0c2 !important;
            box-shadow: 0 1px 2px rgba(194, 120, 92, 0.08) !important;
        }
        div[data-testid="stTabs"] [data-testid="stTabsHeader"] {
            background: #ffffff !important;
            padding: 6px !important;
            border-radius: 12px !important;
            border: 1px solid #f2ebe5 !important;
            margin-bottom: 1.25rem !important;
        }

        /* Metric cards: clean white with warm neutral border */
        [data-testid="stMetric"] {
            background: #ffffff !important;
            border: 1px solid #f2ebe5 !important;
            border-radius: 12px !important;
            padding: 14px 18px !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03) !important;
        }
        [data-testid="stMetricLabel"] {
            color: #78716c !important;
            font-weight: 500 !important;
            font-size: 0.85rem !important;
        }
        [data-testid="stMetricValue"] {
            color: #1c1917 !important;
            font-weight: 700 !important;
        }

        /* Buttons with single pastel warm sand/peach accent */
        .stButton > button {
            border-radius: 10px !important;
            font-weight: 500 !important;
            border: 1px solid #e7dfd8 !important;
            background: #ffffff !important;
            color: #44403c !important;
            transition: all 0.15s ease-in-out !important;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04) !important;
        }
        .stButton > button:hover {
            background: #fbf0ea !important;
            color: #8c462b !important;
            border-color: #ebd0c2 !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 3px 8px rgba(194, 120, 92, 0.12) !important;
        }
        .stButton > button[kind="primary"] {
            background: #c2785c !important;
            color: #ffffff !important;
            border: none !important;
            box-shadow: 0 2px 6px rgba(194, 120, 92, 0.22) !important;
        }
        .stButton > button[kind="primary"]:hover {
            background: #ab6247 !important;
            box-shadow: 0 4px 10px rgba(194, 120, 92, 0.3) !important;
        }

        /* Alert boxes */
        [data-testid="stAlert"] {
            border-radius: 10px !important;
        }

        /* Plotly charts styled as clean cards */
        div.js-plotly-plot {
            background: #ffffff !important;
            border-radius: 12px !important;
            padding: 8px !important;
            border: 1px solid #f2ebe5 !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03) !important;
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

