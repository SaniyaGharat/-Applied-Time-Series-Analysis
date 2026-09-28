"""
Applied Time Series Analysis (ATSA) - Streamlit Web Application.

Phase 1: Project Skeleton & Data Ingestion
- Upload CSV / Excel files or load sample time series.
- Auto-detect date and target numeric columns with user confirmation.
- Set datetime index and inspect series summary.
- Interactive time series visualization and preview.
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
)


def render_sidebar() -> None:
    """Render the sidebar placeholders and application navigation for ATSA phases."""
    st.sidebar.title("📈 ATSA Suite")
    st.sidebar.caption("Applied Time Series Analysis")

    st.sidebar.markdown("---")
    st.sidebar.subheader("App Phases")

    # Roadmap navigation indicators
    st.sidebar.markdown(
        """
        - **Phase 1: Ingestion & Exploration** `[Active]`
        - **Phase 2: EDA & Decomposition** `[Coming Soon]`
        - **Phase 3: Stationarity & Tests** `[Coming Soon]`
        - **Phase 4: Modeling & Forecasting** `[Coming Soon]`
        """
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        "💡 **Tip**: Upload your dataset or generate a demo dataset to start "
        "inspecting your time series."
    )


def create_demo_data() -> pd.DataFrame:
    """
    Generate a demo monthly time series dataset for quick testing.

    Returns
    -------
    pd.DataFrame
        Synthetic monthly dataset with date and numeric target columns.
    """
    dates = pd.date_range(start="2020-01-01", periods=48, freq="MS")
    trend = np.linspace(50, 120, len(dates))
    seasonality = 15 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
    noise = np.random.normal(0, 3, len(dates))
    values = np.round(trend + seasonality + noise, 2)

    return pd.DataFrame({"Date": dates.strftime("%Y-%m-%d"), "Value": values})


def main() -> None:
    """Main application execution pipeline for Phase 1."""
    st.set_page_config(
        page_title="Applied Time Series Analysis (ATSA)",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    render_sidebar()

    st.title("📈 Applied Time Series Analysis (ATSA)")
    st.caption("Phase 1: Project Skeleton & Data Ingestion")
    st.markdown(
        "Upload a time series dataset in **CSV** or **Excel** format to begin ingestion, "
        "date parsing, and basic visualization."
    )

    # File upload section
    upload_col, demo_col = st.columns([3, 1])
    with upload_col:
        uploaded_file = st.file_uploader(
            "Upload Time Series Dataset (CSV or Excel)",
            type=["csv", "xlsx", "xls"],
            help="Upload a file containing a timestamp/date column and at least one numeric metric.",
        )

    with demo_col:
        st.write("")
        st.write("")
        load_demo = st.button("🧪 Load Demo Data", help="Load sample monthly dataset for testing.")

    raw_df: Optional[pd.DataFrame] = None

    if uploaded_file is not None:
        try:
            raw_df = load_uploaded_file(uploaded_file)
            st.success(f"Successfully loaded `{uploaded_file.name}` ({len(raw_df)} rows, {len(raw_df.columns)} columns)")
        except Exception as err:
            st.error(f"Failed to load file: {err}")
            return
    elif load_demo:
        raw_df = create_demo_data()
        st.info("Loaded synthetic demo dataset (48 monthly observations).")

    if raw_df is None:
        st.info("👆 Please upload a CSV/XLSX file or click **Load Demo Data** to proceed.")
        return

    st.markdown("---")
    st.subheader("⚙️ Column Configuration & Date Parsing")

    # Column auto-detection
    candidate_dates = detect_date_columns(raw_df)
    candidate_numerics = detect_numeric_columns(raw_df)

    col1, col2 = st.columns(2)

    all_columns = list(raw_df.columns)

    # Date Column Selection
    with col1:
        default_date_idx = 0
        if candidate_dates and candidate_dates[0] in all_columns:
            default_date_idx = all_columns.index(candidate_dates[0])

        selected_date_col = st.selectbox(
            "Select / Confirm Date Column:",
            options=all_columns,
            index=default_date_idx,
            help="Column containing timestamps or date strings to be parsed as the DatetimeIndex.",
        )

    # Target Numeric Column Selection
    with col2:
        numeric_options = candidate_numerics if candidate_numerics else all_columns
        default_num_idx = 0

        selected_target_col = st.selectbox(
            "Select Target Numeric Column to Analyze:",
            options=numeric_options,
            index=default_num_idx,
            help="Numeric series to analyze and visualize.",
        )

    if not selected_date_col or not selected_target_col:
        st.warning("Please ensure both a date column and a target numeric column are selected.")
        return

    # Process and build clean time series
    try:
        ts_series = prepare_time_series(
            df=raw_df,
            date_col=selected_date_col,
            target_col=selected_target_col,
            sort_index=True,
            drop_na_dates=True,
        )
    except Exception as exc:
        st.error(f"Error preparing time series: {exc}")
        return

    # Metadata & Summary metrics
    summary = get_series_summary(ts_series)

    st.markdown("---")
    st.subheader("📊 Time Series Summary")
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    m_col1.metric("Observations", summary["num_observations"])
    m_col2.metric("Start Date", str(summary["start_date"])[:10])
    m_col3.metric("End Date", str(summary["end_date"])[:10])
    m_col4.metric("Inferred Freq", str(summary["inferred_frequency"]))
    m_col5.metric("Missing Values", summary["missing_values"])

    # Tabs for visualization and table preview
    st.markdown("---")
    tab_chart, tab_table, tab_stats = st.tabs(["📈 Line Plot", "📄 Data Preview", "📋 Descriptive Stats"])

    with tab_chart:
        st.markdown(f"#### `{selected_target_col}` Over Time")

        chart_df = ts_series.reset_index()
        fig = px.line(
            chart_df,
            x="date",
            y=selected_target_col,
            markers=len(chart_df) <= 100,
            title=f"{selected_target_col} Time Series Plot",
            template="plotly_white",
        )
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title=selected_target_col,
            hovermode="x unified",
            xaxis=dict(rangeslider=dict(visible=True)),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_table:
        st.markdown("#### Cleaned Indexed Data (First 100 Rows)")
        preview_df = ts_series.to_frame()
        st.dataframe(preview_df.head(100), use_container_width=True)

    with tab_stats:
        st.markdown("#### Statistical Properties")
        stats_df = ts_series.describe().to_frame(name=selected_target_col)
        st.dataframe(stats_df, use_container_width=True)


if __name__ == "__main__":
    main()
