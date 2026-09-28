"""
Applied Time Series Analysis (ATSA) - Main Application Entrypoint.

Coordinates routing across development phases:
- 📥 Data: Ingestion, parsing, and regularization (Phase 1).
- 🔍 EDA & Diagnostics: Stationarity, autocorrelation, decomposition, and transformations (Phase 2).
- 🧮 Models: Model zoo, interactive fitting, per-model plots, residual diagnostics, grid search (Phases 3-4).
- 🔮 Forecast: Model zoo comparison, holdout evaluation, and multi-step forecasting (Phase 5).
"""

import streamlit as st

from ui.data_tab import render_data_tab
from ui.eda_tab import render_eda_tab
from ui.models_tab import render_models_tab


def render_sidebar() -> None:
    """Render the sidebar navigation and development roadmap."""
    st.sidebar.title("📈 ATSA Suite")
    st.sidebar.caption("Applied Time Series Analysis")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Project Roadmap")

    st.sidebar.markdown(
        """
        - **Phase 1 Data Ingestion** `[Active]`
        - **Phase 2 EDA & Diagnostics (stationarity, ACF/PACF, decomposition)** `[Active]`
        - **Phase 3 Model Zoo (AR, MA, ARMA, ARIMA, SARIMA, SARIMAX, Holt-Winters)** `[Active]`
        - **Phase 4 Model Selector UI with per-model plots** `[Active]`
        - **Phase 5 Forecast & Metrics Comparison** `[Upcoming]`
        - **Phase 6 Deployment** `[Upcoming]`
        - **Phase 7 Polish** `[Upcoming]`
        """
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        "💡 **Phases 1-4 Active**: Navigate between **📥 Data**, **🔍 EDA & Diagnostics**, and **🧮 Models** using the tabs above."
    )


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
        "selected_model_family": "arima",
        "uploader_key": 0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def main() -> None:
    """Main application execution pipeline and tab router."""
    st.set_page_config(
        page_title="Applied Time Series Analysis (ATSA)",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()
    render_sidebar()

    st.title("📈 Applied Time Series Analysis (ATSA)")
    st.caption("End-to-End Time Series Modeling & Diagnostic Platform")

    # Primary Tab Router
    tab_data, tab_eda, tab_models = st.tabs(["📥 Data", "🔍 EDA & Diagnostics", "🧮 Models"])

    with tab_data:
        render_data_tab()

    with tab_eda:
        render_eda_tab()

    with tab_models:
        render_models_tab()


if __name__ == "__main__":
    main()
