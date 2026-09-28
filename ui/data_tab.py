"""
Data Ingestion and Preprocessing UI Tab for ATSA.

Handles:
- Uploading CSV/Excel files and loading reproducible demo data.
- Column selection with date auto-detection and day-first toggle.
- Exogenous regressors multiselection and alignment.
- Preprocessing and regularization (duplicate aggregation, frequency alignment, missing data filling).
- Data signature tracking for robust state invalidation across changes.
- Session state caching and genuine widget clearing.
"""

from pathlib import Path
from typing import List, Optional, Tuple
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
from ui.model_common import reset_model_widgets

# Resolve repo root and data directory relative to this file
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

SAMPLE_DATASETS = {
    "AirPassengers (Monthly 1949-1960)": {
        "file": "AirPassengers.csv",
        "date_col": "Month",
        "target_col": "Passengers",
        "display_name": "AirPassengers (Monthly 1949-1960)",
    },
    "India Dataset (Placeholder)": {
        "file": "india_dataset_placeholder.csv",
        "date_col": "Date",
        "target_col": "Value",
        "display_name": "India Dataset (Placeholder)",
    },
}


def create_demo_data() -> pd.DataFrame:
    """
    Generate a reproducible demo monthly time series dataset using a fixed random seed.

    Returns
    -------
    pd.DataFrame
        Synthetic monthly dataset with date, numeric target, and exogenous feature columns.
    """
    rng = np.random.default_rng(42)
    dates = pd.date_range(start="2020-01-01", periods=48, freq="MS")
    trend = np.linspace(50, 120, len(dates))
    seasonality = 15 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
    noise = rng.normal(0, 3, len(dates))
    values = np.round(trend + seasonality + noise, 2)
    marketing_spend = np.round(20 + 0.3 * trend + rng.normal(0, 2, len(dates)), 2)

    return pd.DataFrame(
        {
            "Date": dates.strftime("%Y-%m-%d"),
            "Sales": values,
            "Marketing_Spend": marketing_spend,
        }
    )


def clear_data_action() -> None:
    """Reset session state and increment uploader key counter to genuinely empty widgets."""
    keys_to_clear = [
        "raw_df",
        "source_name",
        "uploaded_file_id",
        "current_sample_dataset",
        "sample_dataset_select",
        "selected_date_col",
        "selected_target_col",
        "selected_exog_cols",
        "exog_df",
        "ts_series",
        "reg_report",
        "data_signature",
        "model_series",
        "model_base_series",
        "transform_info",
        "model_results",
        "forecast_results",
        "pending_spec",
        "date_col_select",
        "target_col_select",
        "exog_col_select",
        "freq_select",
        "fill_select",
    ]
    for key in keys_to_clear:
        st.session_state.pop(key, None)

    reset_model_widgets()
    # Increment uploader key so file_uploader widget is cleanly reset
    st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1


def render_data_tab() -> None:
    """Render the full Phase 1 Data Ingestion & Preprocessing tab with exog support."""
    st.markdown("### 📥 Dataset Ingestion & Configuration")
    st.markdown(
        "Upload a time series dataset (**CSV**, **XLSX**, or **XLS**), choose a bundled sample dataset, or load "
        "reproducible demo data to begin ingestion, datetime parsing, frequency regularization, and exploratory visualization."
    )

    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 0

    # Ingestion Controls
    upload_col, sample_col, demo_col, clear_col = st.columns([2.0, 1.4, 0.9, 0.9])

    with upload_col:
        uploader_widget_key = f"file_uploader_{st.session_state['uploader_key']}"
        uploaded_file = st.file_uploader(
            "Upload Time Series Dataset (CSV, XLSX, XLS)",
            type=["csv", "xlsx", "xls"],
            help="Upload a file with a timestamp column and numeric values.",
            key=uploader_widget_key,
        )

    with sample_col:
        st.write("")
        sample_options = ["(Select a sample dataset)"] + list(SAMPLE_DATASETS.keys())
        current_sample = st.session_state.get("current_sample_dataset", "(Select a sample dataset)")
        s_idx = sample_options.index(current_sample) if current_sample in sample_options else 0
        chosen_sample = st.selectbox(
            "Sample datasets:",
            options=sample_options,
            index=s_idx,
            key="sample_dataset_select",
            help="Load bundled sample datasets from the repository.",
        )
        if chosen_sample != "(Select a sample dataset)" and chosen_sample != st.session_state.get("current_sample_dataset"):
            meta = SAMPLE_DATASETS[chosen_sample]
            sample_path = DATA_DIR / meta["file"]
            if sample_path.exists():
                sample_df = pd.read_csv(sample_path)
                st.session_state["raw_df"] = sample_df
                st.session_state["source_name"] = meta["display_name"]
                st.session_state["selected_date_col"] = meta["date_col"]
                st.session_state["selected_target_col"] = meta["target_col"]
                st.session_state["date_col_select"] = meta["date_col"]
                st.session_state["target_col_select"] = meta["target_col"]
                st.session_state["selected_exog_cols"] = []
                st.session_state["exog_col_select"] = []
                st.session_state["current_sample_dataset"] = chosen_sample
                st.session_state.pop("uploaded_file_id", None)
                st.rerun()
            else:
                st.error(f"Sample dataset file '{sample_path}' not found.")

    with demo_col:
        st.write("")
        st.write("")
        if st.button("🧪 Load Demo Data", help="Load reproducible monthly demo data (seed=42).", width="stretch"):
            st.session_state["raw_df"] = create_demo_data()
            st.session_state["source_name"] = "Demo Data (Synthetic Monthly, Seed=42)"
            st.session_state["selected_date_col"] = "Date"
            st.session_state["selected_target_col"] = "Sales"
            st.session_state["date_col_select"] = "Date"
            st.session_state["target_col_select"] = "Sales"
            st.session_state["selected_exog_cols"] = []
            st.session_state["exog_col_select"] = []
            st.session_state.pop("current_sample_dataset", None)
            st.session_state.pop("sample_dataset_select", None)
            st.session_state.pop("uploaded_file_id", None)
            st.rerun()

    with clear_col:
        st.write("")
        st.write("")
        if st.button("🗑️ Clear Data", help="Reset all data and clear widget state.", width="stretch"):
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
                st.session_state.pop("exog_col_select", None)
                st.session_state.pop("selected_date_col", None)
                st.session_state.pop("selected_target_col", None)
                st.session_state.pop("selected_exog_cols", None)
                st.session_state.pop("current_sample_dataset", None)
                st.session_state.pop("sample_dataset_select", None)
                st.rerun()
            except Exception as err:
                st.error(f"Failed to load uploaded file: {err}")
                return

    raw_df = st.session_state.get("raw_df")
    source_name = st.session_state.get("source_name")

    if raw_df is None:
        st.info("👆 Please upload a CSV/XLSX file, select a sample dataset, or click **Load Demo Data** to begin.")
        return

    # Upload validation: empty dataset
    if raw_df.empty or len(raw_df.columns) == 0:
        st.error("Uploaded dataset is empty (contains no data rows or columns).")
        return

    # Upload validation: single row or insufficient data
    if len(raw_df) < 2:
        st.error(f"Dataset has only {len(raw_df)} row(s). Time series analysis requires at least 2 observations.")
        return

    # Upload validation: only date column (single column)
    if len(raw_df.columns) < 2:
        st.error(
            "Dataset contains only a single column. Time series analysis requires at least a date column and a target numeric column."
        )
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
        if st.session_state.get("selected_date_col") in all_columns:
            default_date_idx = all_columns.index(st.session_state["selected_date_col"])
        elif candidate_dates and candidate_dates[0] in all_columns:
            default_date_idx = all_columns.index(candidate_dates[0])

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

        if not target_candidates:
            st.error("No valid target column found besides the date column. Please upload a dataset with a target series.")
            return

        default_target_idx = 0
        if st.session_state.get("selected_target_col") in target_candidates:
            default_target_idx = target_candidates.index(st.session_state["selected_target_col"])
        elif candidate_numerics and candidate_numerics[0] in target_candidates:
            default_target_idx = target_candidates.index(candidate_numerics[0])

        selected_target_col = st.selectbox(
            "Select Target Numeric Column:",
            options=target_candidates,
            index=default_target_idx,
            help="Numeric series to analyze and model. Excludes the selected date column.",
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

    # Exogenous Regressor Selection
    available_exog = [
        c for c in (candidate_numerics if candidate_numerics else all_columns)
        if c not in [selected_date_col, selected_target_col]
    ]

    selected_exog_cols = st.multiselect(
        "Optional Exogenous Regressor Columns (for SARIMAX):",
        options=available_exog,
        default=[c for c in st.session_state.get("selected_exog_cols", []) if c in available_exog],
        help="Select additional numeric driver columns to incorporate as external explanatory regressors.",
        key="exog_col_select",
    )
    st.session_state["selected_exog_cols"] = selected_exog_cols

    if not selected_date_col or not selected_target_col:
        st.warning("Please ensure both a date column and a target numeric column are selected.")
        return

    # Robustness checks on target column content
    raw_target_series = raw_df[selected_target_col]
    if raw_target_series.isna().all():
        st.error(f"The selected target column '{selected_target_col}' contains only missing (NaN) values.")
        return

    numeric_target_check = pd.to_numeric(raw_target_series, errors="coerce")
    if numeric_target_check.dropna().empty:
        st.error(
            f"The selected target column '{selected_target_col}' contains non-numeric data. "
            "Please select a numeric target column."
        )
        return

    if len(numeric_target_check.dropna()) < 2:
        st.error(
            f"The selected target column '{selected_target_col}' contains fewer than 2 valid numeric observations."
        )
        return

    # Parse and prepare initial target series
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

    if raw_series.dropna().empty or len(raw_series.dropna()) < 2:
        st.error(
            f"Target series '{selected_target_col}' contains fewer than 2 valid observations after date parsing."
        )
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

    # Perform regularization on target series
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

    # Process exogenous features with identical regularization
    if selected_exog_cols:
        try:
            exog_dict = {}
            for exog_c in selected_exog_cols:
                raw_exog_s = prepare_time_series(
                    df=raw_df,
                    date_col=selected_date_col,
                    target_col=exog_c,
                    sort_index=True,
                    drop_na_dates=True,
                    dayfirst=dayfirst,
                )
                reg_exog_s, _ = regularize_series(
                    series=raw_exog_s,
                    freq=raw_freq_code,
                    fill_method=selected_fill,
                )
                exog_dict[exog_c] = reg_exog_s.reindex(regularized_series.index).ffill().bfill()
            st.session_state["exog_df"] = pd.DataFrame(exog_dict, index=regularized_series.index)
        except Exception as exog_exc:
            st.warning(f"Could not regularize exogenous features: {exog_exc}")
            st.session_state["exog_df"] = None
    else:
        st.session_state["exog_df"] = None

    # Stale state & Data signature check
    current_signature = (
        source_name,
        selected_date_col,
        selected_target_col,
        dayfirst,
        selected_freq_display,
        selected_fill,
        tuple(sorted(selected_exog_cols)),
    )

    prev_signature = st.session_state.get("data_signature")
    if prev_signature != current_signature:
        st.session_state["data_signature"] = current_signature
        st.session_state["model_series"] = regularized_series
        st.session_state["model_base_series"] = regularized_series
        st.session_state["transform_info"] = {
            "steps": [],
            "params": {},
            "suggested_d": 0,
            "suggested_D": 0,
            "seasonal_period": None,
        }
        st.session_state["model_results"] = {}
        st.session_state["forecast_results"] = {}
        reset_model_widgets()

    # Store analysis series and report in session state
    st.session_state["ts_series"] = regularized_series
    st.session_state["reg_report"] = report

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
    m_col5.metric("Exogenous Features", len(selected_exog_cols))

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
        if st.session_state.get("exog_df") is not None:
            preview_df = preview_df.join(st.session_state["exog_df"])
        st.dataframe(preview_df.head(100), width="stretch")

    with tab_stats:
        st.markdown("#### Statistical Distribution")
        stats_df = regularized_series.describe().to_frame(name=selected_target_col)
        st.dataframe(stats_df, width="stretch")
