"""
Applied Time Series Analysis (ATSA) - Streamlit Web Application.

Phase 1: Data Ingestion & Preprocessing
- Upload CSV / Excel datasets or load reproducible demo data.
- Auto-detect candidate date and target numeric columns.
- Support day-first date parsing (dd/mm/yyyy) and integer Year parsing.
- Regularize series (duplicate aggregation, frequency alignment, missing data imputation).
- Persist state across reruns in st.session_state with clear-data capabilities.
- Interactive time series preview and Plotly visualization.
"""

from typing import Optional
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from modules.data_loader import (
    detect_date_columns,
    detect_numeric_columns,
    get_series_summary,
    load_uploaded_file,
    prepare_time_series,
    regularize_series,
)


def render_sidebar() -> None:
    """Render the sidebar navigation and development roadmap."""
    st.sidebar.title("📈 ATSA Suite")
    st.sidebar.caption("Applied Time Series Analysis")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Project Roadmap")

    st.sidebar.markdown(
        """
        - **Phase 1 Data Ingestion** `[Active]`
        - **Phase 2 EDA & Diagnostics (stationarity, ACF/PACF, decomposition)** `[Upcoming]`
        - **Phase 3 Model Zoo (AR, MA, ARMA, ARIMA, SARIMA, SARIMAX, Holt-Winters)** `[Upcoming]`
        - **Phase 4 Model Selector UI with per-model plots** `[Upcoming]`
        - **Phase 5 Forecast & Metrics Comparison** `[Upcoming]`
        - **Phase 6 Deployment** `[Upcoming]`
        - **Phase 7 Polish** `[Upcoming]`
        """
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        "💡 **Tip**: In Phase 1, load your data, set date/numeric columns, "
        "and regularize the time series before moving to diagnostic checks."
    )


def create_demo_data() -> pd.DataFrame:
    """
    Generate a demo monthly time series dataset using a fixed random seed.

    Returns
    -------
    pd.DataFrame
        Synthetic monthly dataset with date and numeric target columns.
    """
    rng = np.random.default_rng(42)
    dates = pd.date_range(start="2020-01-01", periods=48, freq="MS")
    trend = np.linspace(50, 120, len(dates))
    seasonality = 15 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
    noise = rng.normal(0, 3, len(dates))
    values = np.round(trend + seasonality + noise, 2)

    return pd.DataFrame({"Date": dates.strftime("%Y-%m-%d"), "Value": values})


def init_session_state() -> None:
    """Initialize necessary session state keys if not already present."""
    if "raw_df" not in st.session_state:
        st.session_state["raw_df"] = None
    if "source_name" not in st.session_state:
        st.session_state["source_name"] = None
    if "selected_date_col" not in st.session_state:
        st.session_state["selected_date_col"] = None
    if "selected_target_col" not in st.session_state:
        st.session_state["selected_target_col"] = None
    if "ts_series" not in st.session_state:
        st.session_state["ts_series"] = None
    if "reg_report" not in st.session_state:
        st.session_state["reg_report"] = None


def clear_session_state() -> None:
    """Reset dataset and column selection session state keys."""
    for key in [
        "raw_df",
        "source_name",
        "selected_date_col",
        "selected_target_col",
        "ts_series",
        "reg_report",
    ]:
        st.session_state[key] = None


def main() -> None:
    """Main application execution pipeline for Phase 1."""
    st.set_page_config(
        page_title="Applied Time Series Analysis (ATSA)",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()
    render_sidebar()

    st.title("📈 Applied Time Series Analysis (ATSA)")
    st.caption("Phase 1 Data Ingestion & Preprocessing")
    st.markdown(
        "Upload a dataset (**CSV** or **Excel**) or load reproducible demo data to begin "
        "ingestion, datetime parsing, frequency regularization, and exploratory visualization."
    )

    # Ingestion controls
    upload_col, demo_col, clear_col = st.columns([2.5, 1, 1])

    with upload_col:
        uploaded_file = st.file_uploader(
            "Upload Time Series Dataset (CSV, XLSX, XLS)",
            type=["csv", "xlsx", "xls"],
            help="Upload a file with a timestamp column and numeric values.",
            key="file_uploader",
        )

    with demo_col:
        st.write("")
        st.write("")
        if st.button("🧪 Load Demo Data", help="Load reproducible monthly demo data."):
            st.session_state["raw_df"] = create_demo_data()
            st.session_state["source_name"] = "Demo Data (Synthetic Monthly, Seed=42)"
            st.rerun()

    with clear_col:
        st.write("")
        st.write("")
        if st.button("🗑️ Clear Data", help="Reset all data and clear session state."):
            clear_session_state()
            st.rerun()

    # Handle file upload persistence
    if uploaded_file is not None:
        file_identifier = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get("uploaded_file_id") != file_identifier:
            try:
                st.session_state["raw_df"] = load_uploaded_file(uploaded_file)
                st.session_state["source_name"] = uploaded_file.name
                st.session_state["uploaded_file_id"] = file_identifier
                st.rerun()
            except Exception as err:
                st.error(f"Failed to load uploaded file: {err}")
                return

    raw_df = st.session_state.get("raw_df")
    source_name = st.session_state.get("source_name")

    if raw_df is None:
        st.info("👆 Please upload a CSV/XLSX file or click **Load Demo Data** to begin.")
        return

    st.success(f"Active Dataset: **{source_name}** ({len(raw_df)} rows, {len(raw_df.columns)} columns)")

    # 1. Column Selection & Date Parsing Configuration
    st.markdown("---")
    st.subheader("⚙️ 1. Column Selection & Date Configuration")

    candidate_dates = detect_date_columns(raw_df)
    candidate_numerics = detect_numeric_columns(raw_df)
    all_columns = list(raw_df.columns)

    cfg_col1, cfg_col2, cfg_col3 = st.columns([1.5, 1.5, 1.2])

    with cfg_col1:
        # Date column dropdown with auto-detected default
        default_date_idx = 0
        if candidate_dates and candidate_dates[0] in all_columns:
            default_date_idx = all_columns.index(candidate_dates[0])
        elif st.session_state["selected_date_col"] in all_columns:
            default_date_idx = all_columns.index(st.session_state["selected_date_col"])

        selected_date_col = st.selectbox(
            "Select Date / Timestamp Column:",
            options=all_columns,
            index=default_date_idx,
            help="Column containing timestamps, dates, or integer years.",
            key="date_col_select",
        )
        st.session_state["selected_date_col"] = selected_date_col

    with cfg_col2:
        # Exclude selected date column from target column options
        target_candidates = [
            c for c in (candidate_numerics if candidate_numerics else all_columns)
            if c != selected_date_col
        ]
        if not target_candidates:
            target_candidates = [c for c in all_columns if c != selected_date_col]

        default_target_idx = 0
        if (
            st.session_state["selected_target_col"] in target_candidates
        ):
            default_target_idx = target_candidates.index(st.session_state["selected_target_col"])

        selected_target_col = st.selectbox(
            "Select Target Numeric Column to Analyze:",
            options=target_candidates,
            index=default_target_idx,
            help="Numeric series to analyze and visualize. Excludes the selected date column.",
            key="target_col_select",
        )
        st.session_state["selected_target_col"] = selected_target_col

    with cfg_col3:
        st.write("")
        st.write("")
        dayfirst = st.checkbox(
            "Dates are day-first (dd/mm/yyyy)",
            value=False,
            help="Check this if date strings have the day preceding the month (e.g., 31/01/2023).",
            key="dayfirst_checkbox",
        )

    if not selected_date_col or not selected_target_col:
        st.warning("Please ensure both a date column and a target numeric column are selected.")
        return

    # Parse and prepare initial series
    try:
        raw_series = prepare_time_series(
            df=raw_df,
            date_col=selected_date_col,
            target_col=selected_target_col,
            sort_index=True,
            drop_na_dates=True,
            dayfirst=dayfirst,
        )
    except Exception as exc:
        st.error(f"Error parsing date column or target series: {exc}")
        return

    # 2. Preprocessing & Regularization Section
    st.markdown("---")
    st.subheader("🛠️ 2. Preprocessing & Regularization")
    st.markdown(
        "Enforce a uniform sampling frequency, merge any duplicate timestamps by mean, "
        "and impute missing dates or missing values."
    )

    prep_col1, prep_col2 = st.columns(2)

    with prep_col1:
        freq_options = ["Auto-detect", "D", "W", "MS", "M", "Q", "QS", "YS", "H"]
        selected_freq = st.selectbox(
            "Target Frequency:",
            options=freq_options,
            index=0,
            help="Choose 'Auto-detect' to automatically infer the frequency, or specify a fixed sampling frequency.",
            key="freq_select",
        )

    with prep_col2:
        fill_options = ["interpolate", "ffill", "bfill", "drop"]
        selected_fill = st.selectbox(
            "Missing Value Imputation Method:",
            options=fill_options,
            index=0,
            help="Method used to fill gaps introduced by frequency regularization or missing data.",
            key="fill_select",
        )

    # Perform regularization
    freq_arg = None if selected_freq == "Auto-detect" else selected_freq
    try:
        regularized_series, report = regularize_series(
            series=raw_series,
            freq=freq_arg,
            fill_method=selected_fill,
        )
    except Exception as reg_exc:
        st.error(f"Error during regularization: {reg_exc}")
        return

    # Store analysis series and report in session state
    st.session_state["ts_series"] = regularized_series
    st.session_state["reg_report"] = report

    # Display Regularization Report
    st.markdown("##### 📋 Regularization Report")
    r_c1, r_c2, r_c3, r_c4 = st.columns(4)
    r_c1.metric("Duplicates Merged", report["duplicates_merged"])
    r_c2.metric("Rows Added (Gaps)", report["rows_added"])
    r_c3.metric("NaNs Imputed", report["nans_filled"])
    r_c4.metric("Frequency Enforced", report["freq_used"])

    # 3. Series Summary & Visualizations
    st.markdown("---")
    st.subheader("📊 3. Analysis Series Summary")

    summary = get_series_summary(regularized_series)
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    m_col1.metric("Total Observations", summary["num_observations"])
    m_col2.metric("Start Date", str(summary["start_date"])[:10])
    m_col3.metric("End Date", str(summary["end_date"])[:10])
    m_col4.metric("Frequency", str(summary["inferred_frequency"]))
    m_col5.metric("Remaining NaNs", summary["missing_values"])

    # Tabs for interactive visualization, data table, and descriptive statistics
    tab_chart, tab_table, tab_stats = st.tabs(
        ["📈 Interactive Line Plot", "📄 Indexed Data Preview", "📋 Descriptive Statistics"]
    )

    with tab_chart:
        st.markdown(f"#### `{selected_target_col}` Time Series")
        chart_df = regularized_series.reset_index()
        fig = px.line(
            chart_df,
            x="date",
            y=selected_target_col,
            markers=len(chart_df) <= 120,
            title=f"{selected_target_col} ({report['freq_used']})",
            template="plotly_white",
        )
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title=selected_target_col,
            hovermode="x unified",
            xaxis=dict(rangeslider=dict(visible=True)),
        )
        st.plotly_chart(fig, width="stretch")

    with tab_table:
        st.markdown("#### Cleaned & Regularized Series (First 100 Observations)")
        preview_df = regularized_series.to_frame()
        st.dataframe(preview_df.head(100), width="stretch")

    with tab_stats:
        st.markdown("#### Statistical Distribution")
        stats_df = regularized_series.describe().to_frame(name=selected_target_col)
        st.dataframe(stats_df, width="stretch")


if __name__ == "__main__":
    main()
