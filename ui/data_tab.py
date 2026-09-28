"""
Data Ingestion and Preprocessing UI Tab for ATSA.

Handles:
- Uploading CSV/Excel files and loading reproducible demo data.
- Column selection with date auto-detection and day-first toggle.
- Preprocessing and regularization (duplicate aggregation, frequency alignment, missing data filling).
- Session state caching and genuine widget clearing.
"""

from typing import Optional
import numpy as np
import pandas as pd
import streamlit as st

from modules.data_loader import (
    detect_date_columns,
    detect_numeric_columns,
    get_series_summary,
    load_uploaded_file,
    prepare_time_series,
    regularize_series,
)
from modules.plots import plot_time_series


def create_demo_data() -> pd.DataFrame:
    """
    Generate a reproducible demo monthly time series dataset using a fixed random seed.

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


def clear_data_action() -> None:
    """Reset session state and increment uploader key counter to genuinely empty widgets."""
    keys_to_clear = [
        "raw_df",
        "source_name",
        "uploaded_file_id",
        "selected_date_col",
        "selected_target_col",
        "ts_series",
        "reg_report",
        "model_series",
        "transform_info",
        "date_col_select",
        "target_col_select",
        "freq_select",
        "fill_select",
    ]
    for key in keys_to_clear:
        st.session_state.pop(key, None)

    # Increment uploader key so file_uploader widget is cleanly reset
    st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1


def render_data_tab() -> None:
    """Render the full Phase 1 Data Ingestion & Preprocessing tab."""
    st.markdown("### 📥 Dataset Ingestion & Configuration")
    st.markdown(
        "Upload a time series dataset (**CSV**, **XLSX**, or **XLS**) or load reproducible demo data to begin "
        "ingestion, datetime parsing, frequency regularization, and exploratory visualization."
    )

    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 0

    # Ingestion Controls
    upload_col, demo_col, clear_col = st.columns([2.5, 1, 1])

    with upload_col:
        uploader_widget_key = f"file_uploader_{st.session_state['uploader_key']}"
        uploaded_file = st.file_uploader(
            "Upload Time Series Dataset (CSV, XLSX, XLS)",
            type=["csv", "xlsx", "xls"],
            help="Upload a file with a timestamp column and numeric values.",
            key=uploader_widget_key,
        )

    with demo_col:
        st.write("")
        st.write("")
        if st.button("🧪 Load Demo Data", help="Load reproducible monthly demo data (seed=42)."):
            st.session_state["raw_df"] = create_demo_data()
            st.session_state["source_name"] = "Demo Data (Synthetic Monthly, Seed=42)"
            st.session_state.pop("uploaded_file_id", None)
            st.session_state.pop("date_col_select", None)
            st.session_state.pop("target_col_select", None)
            st.rerun()

    with clear_col:
        st.write("")
        st.write("")
        if st.button("🗑️ Clear Data", help="Reset all data and clear widget state."):
            clear_data_action()
            st.rerun()

    # Handle file upload persistence
    if uploaded_file is not None:
        file_identifier = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get("uploaded_file_id") != file_identifier:
            try:
                st.session_state["raw_df"] = load_uploaded_file(uploaded_file)
                st.session_state["source_name"] = uploaded_file.name
                st.session_state["uploaded_file_id"] = file_identifier
                # Reset column selection keys when a fresh file is loaded
                st.session_state.pop("date_col_select", None)
                st.session_state.pop("target_col_select", None)
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
    st.subheader("⚙️ 1. Column Selection & Date Parsing")

    candidate_dates = detect_date_columns(raw_df)
    candidate_numerics = detect_numeric_columns(raw_df)
    all_columns = list(raw_df.columns)

    cfg_col1, cfg_col2, cfg_col3 = st.columns([1.5, 1.5, 1.2])

    with cfg_col1:
        default_date_idx = 0
        if candidate_dates and candidate_dates[0] in all_columns:
            default_date_idx = all_columns.index(candidate_dates[0])
        elif st.session_state.get("selected_date_col") in all_columns:
            default_date_idx = all_columns.index(st.session_state["selected_date_col"])

        selected_date_col = st.selectbox(
            "Select Date / Timestamp Column:",
            options=all_columns,
            index=default_date_idx,
            help="Column containing timestamps, dates, or integer years (1800-2100).",
            key="date_col_select",
        )
        st.session_state["selected_date_col"] = selected_date_col

    with cfg_col2:
        # Exclude selected date column from target numeric options
        target_candidates = [
            c for c in (candidate_numerics if candidate_numerics else all_columns)
            if c != selected_date_col
        ]
        if not target_candidates:
            target_candidates = [c for c in all_columns if c != selected_date_col]

        default_target_idx = 0
        if st.session_state.get("selected_target_col") in target_candidates:
            default_target_idx = target_candidates.index(st.session_state["selected_target_col"])

        selected_target_col = st.selectbox(
            "Select Target Numeric Column:",
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
            help="Check this if date strings have day before month (e.g. 31/01/2023).",
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
        freq_display_options = [
            "Auto-detect",
            "MS - Month Start",
            "ME - Month End",
            "QS - Quarter Start",
            "QE - Quarter End",
            "D - Day",
            "W - Week",
            "YS - Year Start",
            "h - Hour",
        ]
        selected_freq_display = st.selectbox(
            "Target Frequency:",
            options=freq_display_options,
            index=0,
            help="Select 'Auto-detect' to infer frequency or specify an explicit frequency.",
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
    raw_freq_code = selected_freq_display.split()[0] if selected_freq_display != "Auto-detect" else None
    try:
        regularized_series, report = regularize_series(
            series=raw_series,
            freq=raw_freq_code,
            fill_method=selected_fill,
        )
    except Exception as reg_exc:
        st.error(f"Error during regularization: {reg_exc}")
        return

    # Store analysis series and report in session state
    st.session_state["ts_series"] = regularized_series
    st.session_state["reg_report"] = report

    # If model_series hasn't been set by transformations, default to the regularized series
    if "model_series" not in st.session_state or st.session_state["model_series"] is None:
        st.session_state["model_series"] = regularized_series
        st.session_state["transform_info"] = {"steps": [], "params": {}}

    # Display Warning if resample fallback occurred
    if report.get("method") == "resample" and report.get("warning"):
        st.warning(f"⚠️ **Frequency Alignment Notice**: {report['warning']}")

    # Display Regularization Report
    st.markdown("##### 📋 Regularization Report")
    r_c1, r_c2, r_c3, r_c4, r_c5 = st.columns(5)
    r_c1.metric("Duplicates Merged", report["duplicates_merged"])
    r_c2.metric("Rows Added (Gaps)", report["rows_added"])
    r_c3.metric("NaNs Imputed", report["nans_filled"])
    r_c4.metric("Frequency Enforced", report["freq_used"])
    r_c5.metric("Method Applied", report["method"].upper())

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
        fig = plot_time_series(
            regularized_series,
            title=f"{selected_target_col} ({report['freq_used']})",
            yaxis_title=selected_target_col,
            show_rangeslider=True,
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
