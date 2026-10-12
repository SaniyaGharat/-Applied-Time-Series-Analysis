"""
Dataset Comparison UI Tab for Applied Time Series Analysis (Phase 8).

Pure UI component coordinating side-by-side or stacked comparative evaluations of TWO distinct time series datasets:
- Slots A and B with independent namespaces ('cmpA_*', 'cmpB_*').
- Full econometric pipelines for both datasets via modules/pipeline.py.
- Stacked (default) or side-by-side comparative views across:
  1. Series & Summary Stats + Indexed Growth Overlay
  2. Stationarity (ADF/KPSS + Verdict Banner)
  3. ACF / PACF on differenced data
  4. Time Series Decomposition
  5. Model Evaluation Leaderboard (MASE ranking)
  6. Holdout Forecast Overlay (Top 3 candidate models)
  7. Future Projections with 95% Prediction Intervals
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import streamlit as st

from modules.data_loader import (
    detect_date_columns,
    detect_numeric_columns,
    load_uploaded_file,
    prepare_time_series,
    regularize_series,
)
from modules.diagnostics import decompose
from modules.pipeline import AnalysisBundle, build_arima_family_comparison, run_analysis
from modules.plots import (
    plot_acf_pacf_combined,
    plot_arima_dual_overlay,
    plot_arima_family_bars,
    plot_arima_multi_metric_bars,
    plot_arima_radar,
    plot_decomposition,
    plot_forecast_overlay,
    plot_future_forecast,
    plot_indexed_overlay,
    plot_time_series,
)
from ui.data_tab import SAMPLE_DATASETS, create_demo_data

# Resolve data directory
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


@st.cache_data(max_entries=10, ttl=3600)
def cached_run_analysis(
    series: pd.Series,
    name: str,
    m: Optional[int] = None,
    holdout: Optional[int] = None,
    horizon: Optional[int] = None,
) -> AnalysisBundle:
    """UI cache wrapper for deterministic pipeline execution."""
    return run_analysis(
        series=series,
        name=name,
        m=m,
        holdout=holdout,
        exog=None,
        horizon=horizon,
    )


@st.cache_data(max_entries=5, ttl=3600)
def _load_cached_compare_source(source_type: str) -> Optional[pd.DataFrame]:
    if source_type == "demo":
        return create_demo_data()
    elif source_type == "air":
        air_path = DATA_DIR / "AirPassengers.csv"
        if air_path.exists():
            return pd.read_csv(air_path)
    return None


@st.cache_data(max_entries=10, ttl=3600)
def _cached_prep_and_reg_slot(
    raw_df: pd.DataFrame,
    date_col: str,
    target_col: str,
    dayfirst: bool,
    freq_code: Optional[str],
    sel_fill: str,
) -> pd.Series:
    prep_s = prepare_time_series(raw_df, date_col=date_col, target_col=target_col, dayfirst=dayfirst)
    reg_s, _ = regularize_series(prep_s, freq=freq_code, fill_method=sel_fill)
    return reg_s


def _render_slot_config(
    slot: str,
    label: str,
) -> Tuple[Optional[pd.Series], Optional[str], Optional[int], Optional[int]]:
    """
    Render configuration widgets for a dataset slot (A or B).
    Uses isolated namespaces (cmpA_* or cmpB_*).

    Returns
    -------
    Tuple[Optional[pd.Series], Optional[str], Optional[int], Optional[int]]
        (prepared_series, source_name, m, holdout)
    """
    st.markdown(f"#### 📦 {label}")
    ns = f"cmp{slot}"

    source_options = [
        "Demo Monthly Sales (Synthetic)",
        "AirPassengers (Monthly 1949-1960)",
        "Upload CSV/XLSX",
    ]

    selected_source = st.selectbox(
        f"Select Data Source ({slot}):",
        options=source_options,
        index=0 if slot == "A" else 1,
        key=f"{ns}_source_select",
    )

    raw_df: Optional[pd.DataFrame] = None
    source_name = selected_source

    if selected_source == "Demo Monthly Sales (Synthetic)":
        raw_df = _load_cached_compare_source("demo")
        default_date = "Date"
        default_target = "Sales"
        source_name = f"Demo Data ({slot})"

    elif selected_source == "AirPassengers (Monthly 1949-1960)":
        raw_df = _load_cached_compare_source("air")
        if raw_df is not None:
            default_date = "Month"
            default_target = "Passengers"
            source_name = f"AirPassengers ({slot})"
        else:
            air_path = DATA_DIR / "AirPassengers.csv"
            st.error(f"AirPassengers dataset file not found at {air_path}.")
            return None, None, None, None

    else:
        uploaded_file = st.file_uploader(
            f"Upload CSV/Excel file for Dataset {slot}:",
            type=["csv", "xlsx", "xls"],
            key=f"{ns}_file_uploader",
        )
        if uploaded_file is not None:
            try:
                raw_df = load_uploaded_file(uploaded_file)
                source_name = uploaded_file.name
            except Exception as err:
                st.error(f"Failed to read file: {err}")
                return None, None, None, None
            default_date = None
            default_target = None
        else:
            st.info(f"Please upload a file for Dataset {slot}.")
            return None, None, None, None

    if raw_df is None or raw_df.empty:
        st.error(f"Dataset {slot} is empty.")
        return None, None, None, None

    if len(raw_df) < 2:
        st.error(f"Dataset {slot} has fewer than 2 rows. Time series analysis requires at least 2 rows.")
        return None, None, None, None

    if len(raw_df.columns) < 2:
        st.error(f"Dataset {slot} contains only 1 column. Requires both a date column and a target numeric column.")
        return None, None, None, None

    all_cols = list(raw_df.columns)
    detected_dates = detect_date_columns(raw_df)
    detected_numerics = detect_numeric_columns(raw_df)

    date_idx = 0
    if default_date and default_date in all_cols:
        date_idx = all_cols.index(default_date)
    elif detected_dates and detected_dates[0] in all_cols:
        date_idx = all_cols.index(detected_dates[0])

    c1, c2 = st.columns(2)
    with c1:
        date_col = st.selectbox(
            f"Date Column ({slot}):",
            options=all_cols,
            index=date_idx,
            key=f"{ns}_date_col",
        )
    with c2:
        target_candidates = [c for c in (detected_numerics if detected_numerics else all_cols) if c != date_col]
        if not target_candidates:
            target_candidates = [c for c in all_cols if c != date_col]

        if not target_candidates:
            st.error(f"No valid target column found for Dataset {slot}.")
            return None, None, None, None

        target_idx = 0
        if default_target and default_target in target_candidates:
            target_idx = target_candidates.index(default_target)
        elif detected_numerics and detected_numerics[0] in target_candidates:
            target_idx = target_candidates.index(detected_numerics[0])

        target_col = st.selectbox(
            f"Target Column ({slot}):",
            options=target_candidates,
            index=target_idx,
            key=f"{ns}_target_col",
        )

    # Preprocessing options
    p1, p2, p3 = st.columns(3)
    with p1:
        freq_opts = ["Auto-detect", "MS - Month Start", "D - Day", "W - Week", "h - Hour", "QS - Quarter Start", "YS - Year Start"]
        sel_freq = st.selectbox(f"Frequency ({slot}):", options=freq_opts, index=0, key=f"{ns}_freq")
    with p2:
        sel_fill = st.selectbox(f"Imputation ({slot}):", options=["interpolate", "ffill", "bfill", "drop"], index=0, key=f"{ns}_fill")
    with p3:
        dayfirst = st.checkbox("Day-first dates", value=False, key=f"{ns}_dayfirst")

    # Clean and regularize series
    try:
        freq_code = sel_freq.split()[0] if sel_freq != "Auto-detect" else None
        reg_s = _cached_prep_and_reg_slot(
            raw_df=raw_df,
            date_col=date_col,
            target_col=target_col,
            dayfirst=dayfirst,
            freq_code=freq_code,
            sel_fill=sel_fill,
        )
    except Exception as exc:
        st.error(f"Error preparing Dataset {slot}: {exc}")
        return None, None, None, None

    if len(reg_s.dropna()) < 2:
        st.error(f"Dataset {slot} has fewer than 2 valid observations after date processing.")
        return None, None, None, None

    return reg_s, source_name, None, None


def render_compare_tab() -> None:
    """Render the full Phase 8 Two-Dataset Comparison Workspace."""
    st.subheader("⚖️ Two-Dataset Comparative Analysis")
    st.markdown(
        "Compare statistical properties, stationarity, autocorrelation, component decomposition, "
        "model accuracy leaderboards, and out-of-sample projections across **TWO time series datasets** side-by-side."
    )

    # 1. Dataset Slot Configuration in Two Columns
    col_slot_a, col_slot_b = st.columns(2)

    with col_slot_a:
        series_a, name_a, m_a, holdout_a = _render_slot_config("A", "Dataset A")

    with col_slot_b:
        series_b, name_b, m_b, holdout_b = _render_slot_config("B", "Dataset B")

    st.markdown("---")

    # Run comparison action button
    btn_col, _, view_col = st.columns([1.5, 1.5, 2])
    with btn_col:
        run_clicked = st.button("🚀 Run Comparison", type="primary", use_container_width=True)

    with view_col:
        view_mode = st.radio(
            "Comparison Layout:",
            options=["Stacked (A above B)", "Side by side"],
            horizontal=True,
            key="cmp_view_mode",
        )

    if run_clicked:
        if series_a is None or len(series_a.dropna()) < 2:
            st.error("Dataset A is invalid or not loaded. Please select a valid dataset for Slot A.")
            return
        if series_b is None or len(series_b.dropna()) < 2:
            st.error("Dataset B is invalid or not loaded. Please select a valid dataset for Slot B.")
            return

        with st.spinner("Executing full econometric pipeline and model zoos for both datasets..."):
            try:
                bundle_a = cached_run_analysis(
                    series=series_a,
                    name=name_a or "Dataset A",
                    m=m_a,
                    holdout=holdout_a,
                    horizon=None,
                )
                bundle_b = cached_run_analysis(
                    series=series_b,
                    name=name_b or "Dataset B",
                    m=m_b,
                    holdout=holdout_b,
                    horizon=None,
                )
                st.session_state["cmp_bundles"] = {"A": bundle_a, "B": bundle_b}
                st.success("Comparison completed successfully! Explore the comparative tabs below.")
            except Exception as exc:
                st.error(f"An unexpected error occurred during comparison: {exc}")
                return

    # 2. Render Results if Bundles Available
    cmp_bundles = st.session_state.get("cmp_bundles")
    if not cmp_bundles or "A" not in cmp_bundles or "B" not in cmp_bundles:
        st.info("👆 Configure Dataset A and Dataset B above, then click **🚀 Run Comparison** to evaluate both series.")
        return

    b_a: AnalysisBundle = cmp_bundles["A"]
    b_b: AnalysisBundle = cmp_bundles["B"]

    st.caption(
        "📌 **Evaluation Notice**: Raw RMSE and MAE cannot be directly compared between datasets with differing scales or units. "
        "Use scale-free metrics (MASE, sMAPE, RelRMSE vs Naive) to assess relative predictability."
    )

    is_stacked = view_mode == "Stacked (A above B)"

    def render_dual_section(title: str, explanation: str, render_fn_a: Any, render_fn_b: Any) -> None:
        """Helper to render section either stacked (A above B) or side by side."""
        st.markdown(f"### {title}")
        with st.expander("ℹ️ What does this show?"):
            st.markdown(explanation)

        if is_stacked:
            st.markdown(f"##### 🔹 Dataset A: `{b_a.name}`")
            render_fn_a()
            st.markdown("---")
            st.markdown(f"##### 🔸 Dataset B: `{b_b.name}`")
            render_fn_b()
        else:
            c_left, c_right = st.columns(2)
            with c_left:
                st.markdown(f"##### 🔹 Dataset A: `{b_a.name}`")
                render_fn_a()
            with c_right:
                st.markdown(f"##### 🔸 Dataset B: `{b_b.name}`")
                render_fn_b()
        st.markdown("---")

    # =========================================================================
    # SECTION 1: Series & Summary Stats + Indexed Growth Overlay
    # =========================================================================
    def _sec1_a():
        fig = plot_time_series(b_a.series, title=f"{b_a.name} ({b_a.freq_label})", show_rangeslider=False)
        st.plotly_chart(fig, use_container_width=True, key="cmp_ts_plot_a")
        st.dataframe(b_a.series.describe().to_frame(name=b_a.name), use_container_width=True)

    def _sec1_b():
        fig = plot_time_series(b_b.series, title=f"{b_b.name} ({b_b.freq_label})", show_rangeslider=False)
        st.plotly_chart(fig, use_container_width=True, key="cmp_ts_plot_b")
        st.dataframe(b_b.series.describe().to_frame(name=b_b.name), use_container_width=True)

    render_dual_section(
        "1. Time Series & Descriptive Statistics",
        "This section displays the historical series and summary distribution metrics. "
        "The indexed growth chart below rebases both series to 100 at their start date to compare relative growth trajectory regardless of units.",
        _sec1_a,
        _sec1_b,
    )

    # Indexed Overlay
    st.markdown("#### 📈 Indexed Growth Comparison (Base = 100)")
    fig_idx = plot_indexed_overlay(b_a.series, b_b.series, names=(b_a.name, b_b.name))
    st.plotly_chart(fig_idx, use_container_width=True, key="cmp_indexed_growth_plot")
    st.markdown("---")

    # =========================================================================
    # SECTION 2: Stationarity (ADF & KPSS)
    # =========================================================================
    def _render_stationarity_box(bundle: AnalysisBundle):
        stat = bundle.stationarity
        verd = stat.get("verdict", "Unknown")
        if verd == "Stationary":
            st.success(f"**Verdict**: Stationary (Suggested differencing d={bundle.suggested_d})")
        else:
            st.warning(f"**Verdict**: {verd} (Suggested differencing d={bundle.suggested_d})")

        adf = stat.get("adf")
        kpss_res = stat.get("kpss")
        if adf and kpss_res:
            table_data = [
                {"Test": "ADF (H0: Unit Root)", "Statistic": f"{adf['statistic']:.4f}", "p-value": f"{adf['pvalue']:.4f}", "Used Lags": adf["used_lag"], "Result": "Reject H0 (Stationary)" if adf["is_stationary"] else "Fail to Reject (Non-Stat)"},
                {"Test": "KPSS (H0: Stationary)", "Statistic": f"{kpss_res['statistic']:.4f}", "p-value": f"{kpss_res['pvalue']:.4f}", "Used Lags": kpss_res["used_lag"], "Result": "Stationary" if kpss_res["is_stationary"] else "Reject (Non-Stat)"},
            ]
            st.dataframe(pd.DataFrame(table_data), use_container_width=True)

    render_dual_section(
        "2. Stationarity Diagnostics (ADF & KPSS)",
        "Assesses whether each series exhibits constant mean and variance using Augmented Dickey-Fuller and KPSS hypothesis tests. "
        "Non-stationary series are auto-assigned differencing orders (d=1, D=1) for ARIMA-class model estimation.",
        lambda: _render_stationarity_box(b_a),
        lambda: _render_stationarity_box(b_b),
    )

    # =========================================================================
    # SECTION 3: Autocorrelation (ACF / PACF)
    # =========================================================================
    def _render_acf(bundle: AnalysisBundle, slot: str):
        data = bundle.acf_pacf
        if data and "acf" in data and "pacf" in data:
            fig = plot_acf_pacf_combined(data["lags"], data["acf"], data["pacf"], data["conf_bound"])
            st.plotly_chart(fig, use_container_width=True, key=f"cmp_acf_plot_{slot}")
        else:
            st.info("ACF/PACF data not available.")

    render_dual_section(
        "3. Autocorrelation (ACF) & Partial Autocorrelation (PACF)",
        "Displays serial correlation patterns computed on the differenced series (d, D). "
        "The leading contiguous run of significant spikes guides autoregressive (AR) and moving average (MA) order selection.",
        lambda: _render_acf(b_a, "a"),
        lambda: _render_acf(b_b, "b"),
    )

    # =========================================================================
    # SECTION 4: Time Series Decomposition
    # =========================================================================
    def _render_decomp(bundle: AnalysisBundle, slot: str):
        d_info = bundle.decomposition
        if d_info.get("error"):
            st.info(f"Decomposition unavailable: {d_info['error']}")
        else:
            st.metric("Seasonal Strength (F_s)", f"{d_info.get('seasonal_strength', 0.0):.3f}")
            st.metric("Trend Strength (F_t)", f"{d_info.get('trend_strength', 0.0):.3f}")
            try:
                decomp_fig_data = decompose(bundle.series, period=bundle.m)
                if not decomp_fig_data.get("error"):
                    fig = plot_decomposition(decomp_fig_data)
                    st.plotly_chart(fig, use_container_width=True, key=f"cmp_decomp_plot_{slot}")
            except Exception:
                pass

    render_dual_section(
        "4. Time Series Decomposition",
        "Isolates trend, seasonal, and residual components using Hyndman variance decomposition. "
        "Quantifies relative seasonal and trend strengths on a scale from 0 (none) to 1 (dominant).",
        lambda: _render_decomp(b_a, "a"),
        lambda: _render_decomp(b_b, "b"),
    )

    # =========================================================================
    # SECTION 5: Model Leaderboards & Box-Jenkins Comparison
    # =========================================================================
    st.markdown("### 5. Out-of-Sample Holdout Leaderboard & Box-Jenkins Zoo")
    with st.expander("ℹ️ What does this show?"):
        st.markdown(
            "Ranks candidate models on out-of-sample holdout validation using MASE (Mean Absolute Scaled Error). "
            "A MASE < 1 indicates the model achieves higher accuracy than the naive baseline. "
            "Switch between the full model zoo and the dedicated **Box-Jenkins Comparison** to evaluate "
            "classical AR, MA, ARMA, ARIMA, and SARIMA architectures head-to-head."
        )

    sec5_view = st.radio(
        "Leaderboard Perspective:",
        ["🎯 Box-Jenkins Comparison (AR, MA, ARMA, ARIMA, SARIMA)", "📋 Full Model Zoo Leaderboard"],
        horizontal=True,
        key="cmp_sec5_view_mode",
    )

    if sec5_view == "🎯 Box-Jenkins Comparison (AR, MA, ARMA, ARIMA, SARIMA)":
        st.markdown("##### 🏛️ Classical Box-Jenkins Architecture Matrix (AR, MA, ARMA, ARIMA, SARIMA)")
        bj_df = build_arima_family_comparison(b_a, b_b)
        st.dataframe(bj_df, use_container_width=True)

        bj_plot_mode = st.radio(
            "Box-Jenkins Comparative Plot Type:",
            [
                "📊 3-Metric Overview Panel (MASE, sMAPE, AIC)",
                "📈 Stacked Holdout Forecast Trajectories",
                "🎯 Focus Metric Bars",
                "🕸️ Relative Accuracy Radar",
            ],
            horizontal=True,
            key="cmp_bj_plot_mode",
        )

        if bj_plot_mode == "📊 3-Metric Overview Panel (MASE, sMAPE, AIC)":
            fig_multi = plot_arima_multi_metric_bars(b_a, b_b, names=(b_a.name, b_b.name))
            st.plotly_chart(fig_multi, use_container_width=True, key="cmp_bj_multi_metric_plot")
        elif bj_plot_mode == "📈 Stacked Holdout Forecast Trajectories":
            fig_dual_ov = plot_arima_dual_overlay(b_a, b_b, names=(b_a.name, b_b.name))
            st.plotly_chart(fig_dual_ov, use_container_width=True, key="cmp_bj_dual_overlay_plot")
        elif bj_plot_mode == "🕸️ Relative Accuracy Radar":
            fig_radar = plot_arima_radar(b_a, b_b, names=(b_a.name, b_b.name))
            st.plotly_chart(fig_radar, use_container_width=True, key="cmp_bj_radar_plot")
        else:
            metric_choice = st.radio(
                "Comparative Metric for Focus Chart:",
                ["MASE (Holdout Error)", "sMAPE (%)", "AIC (Information Criterion)"],
                horizontal=True,
                key="cmp_bj_metric_choice",
            )
            metric_key = "MASE" if "MASE" in metric_choice else ("sMAPE" if "sMAPE" in metric_choice else "AIC")
            fig_bj_bars = plot_arima_family_bars(b_a, b_b, metric=metric_key, names=(b_a.name, b_b.name))
            st.plotly_chart(fig_bj_bars, use_container_width=True, key="cmp_bj_bars_plot")
        st.markdown("---")
    else:
        def _render_leaderboard(bundle: AnalysisBundle):
            st.success(f"🏆 **Top Performer**: **{bundle.best_label}** (RelRMSE: {bundle.rel_rmse_best:.3f})")
            if not bundle.comparison.empty:
                display_cols = [c for c in ["Rank", "Model", "Family", "Converged", "MASE", "sMAPE", "RelRMSE_vs_ref", "Coverage95", "AIC", "FitSec"] if c in bundle.comparison.columns]
                st.dataframe(bundle.comparison[display_cols], use_container_width=True)
            else:
                st.info("No models estimated.")

        if is_stacked:
            st.markdown(f"##### 🔹 Dataset A: `{b_a.name}`")
            _render_leaderboard(b_a)
            st.markdown("---")
            st.markdown(f"##### 🔸 Dataset B: `{b_b.name}`")
            _render_leaderboard(b_b)
        else:
            c_left, c_right = st.columns(2)
            with c_left:
                st.markdown(f"##### 🔹 Dataset A: `{b_a.name}`")
                _render_leaderboard(b_a)
            with c_right:
                st.markdown(f"##### 🔸 Dataset B: `{b_b.name}`")
                _render_leaderboard(b_b)
        st.markdown("---")

    # =========================================================================
    # SECTION 6: Holdout Forecast Overlay (Top Candidates & Box-Jenkins Zoo)
    # =========================================================================
    st.markdown("### 6. Holdout Forecast Overlay")
    with st.expander("ℹ️ What does this show?"):
        st.markdown(
            "Overlays model predictions directly against actual held-out observations. "
            "Highlights how accurately each architecture tracks local directional turning points and peaks. "
            "Select **Box-Jenkins Models** to visually compare AR, MA, ARMA, ARIMA, and SARIMA predictions."
        )

    overlay_view = st.radio(
        "Forecast Overlay Selection:",
        ["🎯 Box-Jenkins Models (AR, MA, ARMA, ARIMA, SARIMA)", "🏆 Top 3 Candidate Models Overall"],
        horizontal=True,
        key="cmp_sec6_overlay_view",
    )

    def _render_holdout_overlay(bundle: AnalysisBundle, slot: str, mode: str):
        selected_models: List[Any] = []
        if mode == "🎯 Box-Jenkins Models (AR, MA, ARMA, ARIMA, SARIMA)":
            bj_fams = {"ar", "ma", "arma", "arima", "sarima"}
            for m_res in bundle.model_results:
                if m_res.family in bj_fams and getattr(m_res, "test_forecast", None) is not None and not m_res.test_forecast.empty:
                    selected_models.append(m_res)
        else:
            if not bundle.comparison.empty:
                for _, r in bundle.comparison.iterrows():
                    if r.get("Converged"):
                        for m_res in bundle.model_results:
                            if m_res.label == r.get("Model"):
                                selected_models.append(m_res)
                                break
                    if len(selected_models) >= 3:
                        break

        if selected_models and len(bundle.series) > bundle.holdout:
            actual_holdout = bundle.series.iloc[-bundle.holdout:]
            fc_dict = {
                r.label: r.test_forecast
                for r in selected_models
                if getattr(r, "test_forecast", None) is not None and not r.test_forecast.empty
            }
            if fc_dict:
                fig = plot_forecast_overlay(actual=actual_holdout, forecasts=fc_dict)
                suffix = "bj" if "Box-Jenkins" in mode else "top"
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key=f"cmp_overlay_plot_{slot}_{suffix}",
                    config={"displayModeBar": "hover", "displaylogo": False},
                )
            else:
                st.info("Holdout forecasts not available.")
        else:
            st.info("Holdout overlay not available.")

    if is_stacked:
        st.markdown(f"##### 🔹 Dataset A: `{b_a.name}`")
        _render_holdout_overlay(b_a, "a", overlay_view)
        st.markdown("---")
        st.markdown(f"##### 🔸 Dataset B: `{b_b.name}`")
        _render_holdout_overlay(b_b, "b", overlay_view)
    else:
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown(f"##### 🔹 Dataset A: `{b_a.name}`")
            _render_holdout_overlay(b_a, "a", overlay_view)
        with c_right:
            st.markdown(f"##### 🔸 Dataset B: `{b_b.name}`")
            _render_holdout_overlay(b_b, "b", overlay_view)
    st.markdown("---")

    # =========================================================================
    # SECTION 7: Future Forward Projections
    # =========================================================================
    def _render_future(bundle: AnalysisBundle, slot: str):
        if bundle.future and bundle.future.df is not None and not bundle.future.df.empty:
            fig = plot_future_forecast(bundle.series, bundle.future.df, label=bundle.future.label)
            st.plotly_chart(fig, use_container_width=True, key=f"cmp_future_plot_{slot}")
        else:
            st.info("Future forecast projection not available.")

    render_dual_section(
        "7. Out-of-Sample Forward Projections (95% Interval)",
        "Refits the best-performing model on all historical data and projects forward with 95% confidence intervals. "
        "Interval divergence reflects accumulated multi-step variance uncertainty.",
        lambda: _render_future(b_a, "a"),
        lambda: _render_future(b_b, "b"),
    )
