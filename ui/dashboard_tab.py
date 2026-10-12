"""
Comparative Executive Dashboard UI Tab for Applied Time Series Analysis (Phase 8).

Pure UI component presenting the final executive synthesis across TWO time series datasets:
- Dataset names and date range badges.
- Standardized KPI cards in two parallel rows (Dataset A and Dataset B).
- Plain-English Key Findings callout box.
- Standardized cross-dataset KPI comparison table (KPI | A | B | note).
- Comparative charts:
  1. Indexed growth overlay (Base = 100)
  2. Grouped accuracy bars (MASE / sMAPE / RelRMSE)
  3. Component strength bars (Trend & Seasonality)
  4. Side-by-side out-of-sample forward projections
- Multi-format report downloads (CSV and 6-sheet openpyxl Excel workbook).
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import streamlit as st

from modules.export import build_dashboard_report
from modules.pipeline import AnalysisBundle, compare_datasets, compute_kpis
from modules.plots import (
    plot_future_forecast,
    plot_gauge_or_bar,
    plot_indexed_overlay,
    plot_kpi_grouped_bars,
)


def render_dashboard_tab() -> None:
    """Render the full Phase 8 Executive Comparison Dashboard."""
    st.subheader("📊 Two-Dataset Comparison & KPI Dashboard")

    cmp_bundles = st.session_state.get("cmp_bundles")
    if not cmp_bundles or "A" not in cmp_bundles or "B" not in cmp_bundles:
        st.info(
            "ℹ️ No comparative dataset bundles loaded yet. Please open the **⚖️ Compare Datasets** tab, "
            "configure Dataset A and Dataset B, and click **🚀 Run Comparison** to populate this dashboard."
        )
        return

    bundle_a: AnalysisBundle = cmp_bundles["A"]
    bundle_b: AnalysisBundle = cmp_bundles["B"]

    # Compute scale-free KPIs and comparative synthesis
    kpis_a = compute_kpis(bundle_a)
    kpis_b = compute_kpis(bundle_b)
    kpi_df, findings = compare_datasets(
        kpis_a=kpis_a,
        kpis_b=kpis_b,
        series_a=bundle_a.series,
        series_b=bundle_b.series,
    )

    # 1. Top Badges: Dataset Names & Date Ranges
    st.markdown(f"#### 🏷️ Active Datasets: **{bundle_a.name}** vs **{bundle_b.name}**")
    chip_col_a, chip_col_b = st.columns(2)
    with chip_col_a:
        st.info(
            f"🔹 **Dataset A (`{bundle_a.name}`)**\n\n"
            f"📅 Span: **{kpis_a['date range']}** | "
            f"⏱️ Freq: **{kpis_a['frequency']}** | "
            f"🔢 Obs: **{kpis_a['n_obs']}**"
        )
    with chip_col_b:
        st.success(
            f"🔸 **Dataset B (`{bundle_b.name}`)**\n\n"
            f"📅 Span: **{kpis_b['date range']}** | "
            f"⏱️ Freq: **{kpis_b['frequency']}** | "
            f"🔢 Obs: **{kpis_b['n_obs']}**"
        )

    st.markdown("---")

    # 2. Executive KPI Cards in Two Parallel Rows (A Row, B Row)
    def _render_kpi_row(prefix: str, name: str, kpis: Dict[str, Any], bundle: AnalysisBundle) -> None:
        st.markdown(f"##### {prefix} `{name}` Key Performance Indicators")
        c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11 = st.columns(11)

        c1.metric("Observations", str(kpis["n_obs"]))
        c2.metric("Frequency", str(kpis["frequency"]))
        c3.metric("Best Model", str(kpis["best model label"]))

        mase_val = kpis["best MASE"]
        c4.metric("Best MASE", f"{mase_val:.3f}" if (mase_val is not None and not np.isnan(mase_val)) else "N/A")

        rel_rmse_val = kpis["RelRMSE vs naive"]
        c5.metric("RelRMSE", f"{rel_rmse_val:.3f}" if (rel_rmse_val is not None and not np.isnan(rel_rmse_val)) else "N/A")

        c6.metric("Seasonal (F_s)", f"{kpis['seasonal_strength']:.2f}")
        c7.metric("Trend (F_t)", f"{kpis['trend_strength']:.2f}")
        c8.metric("Stationarity", str(kpis["stationarity verdict"])[:12])

        fc_chg = kpis["forecast_change_pct"]
        c9.metric("FC Change %", f"{fc_chg:+.1f}%" if (fc_chg is not None and not np.isnan(fc_chg)) else "N/A")

        width_pct = kpis["mean_interval_width_pct"]
        c10.metric("Interval Width", f"{width_pct:.1f}%" if (width_pct is not None and not np.isnan(width_pct)) else "N/A")

        c11.metric("Beat Naive", str(kpis["n_beating_naive/n_models_fit"]))

    _render_kpi_row("🔹 Dataset A:", bundle_a.name, kpis_a, bundle_a)
    st.write("")
    _render_kpi_row("🔸 Dataset B:", bundle_b.name, kpis_b, bundle_b)

    st.markdown("---")

    # 3. Key Findings Callout Box
    st.markdown("### 💡 Key Comparative Findings")
    if findings:
        findings_markdown = "\n\n".join([f"- {f}" for f in findings])
        st.info(f"**Automated Econometric Synthesis:**\n\n{findings_markdown}")
    else:
        st.info("No comparative findings generated.")

    st.markdown("---")

    # 4. Standardized Cross-Dataset KPI Matrix
    st.markdown("### 📋 Standardized Cross-Dataset KPI Matrix")
    st.caption(
        "📌 Note: Raw RMSE and MAE cannot be directly compared across datasets with different scales. "
        "Use MASE (< 1 denotes beating naive) and RelRMSE (< 1.0) for valid cross-dataset comparisons."
    )
    st.dataframe(kpi_df, use_container_width=True)

    st.markdown("---")

    # 5. Comparative Visualizations
    st.markdown("### 📈 Visual Comparative Analytics")

    # Chart 1: Indexed Growth Overlay
    st.markdown("#### 1. Indexed Growth Overlay (Base = 100)")
    fig_indexed = plot_indexed_overlay(bundle_a.series, bundle_b.series, names=(bundle_a.name, bundle_b.name))
    st.plotly_chart(fig_indexed, use_container_width=True, key="dash_chart_indexed")

    # Chart 2 & 3: Grouped Bars and Strength Bars Side-by-Side
    g_col1, g_col2 = st.columns(2)
    with g_col1:
        st.markdown("#### 2. Accuracy Comparison (Lower is Better)")
        fig_grouped = plot_kpi_grouped_bars(kpi_df, names=(bundle_a.name, bundle_b.name))
        st.plotly_chart(fig_grouped, use_container_width=True, key="dash_chart_grouped")

    with g_col2:
        st.markdown("#### 3. Component Strengths (Hyndman Metrics)")
        fig_strengths = plot_gauge_or_bar(
            trend_a=kpis_a["trend_strength"],
            trend_b=kpis_b["trend_strength"],
            seas_a=kpis_a["seasonal_strength"],
            seas_b=kpis_b["seasonal_strength"],
            names=(bundle_a.name, bundle_b.name),
        )
        st.plotly_chart(fig_strengths, use_container_width=True, key="dash_chart_strengths")

    # Chart 4: Side-by-Side Future Forecasts
    st.markdown("#### 4. Out-of-Sample Forward Projections")
    fc_col1, fc_col2 = st.columns(2)
    with fc_col1:
        st.markdown(f"##### 🔹 `{bundle_a.name}`: {bundle_a.future.label if bundle_a.future else 'Future'}")
        if bundle_a.future and bundle_a.future.df is not None and not bundle_a.future.df.empty:
            fig_fa = plot_future_forecast(bundle_a.series, bundle_a.future.df, label=bundle_a.future.label)
            st.plotly_chart(fig_fa, use_container_width=True, key="dash_chart_future_a")
        else:
            st.info("Dataset A forecast projection not available.")

    with fc_col2:
        st.markdown(f"##### 🔸 `{bundle_b.name}`: {bundle_b.future.label if bundle_b.future else 'Future'}")
        if bundle_b.future and bundle_b.future.df is not None and not bundle_b.future.df.empty:
            fig_fb = plot_future_forecast(bundle_b.series, bundle_b.future.df, label=bundle_b.future.label)
            st.plotly_chart(fig_fb, use_container_width=True, key="dash_chart_future_b")
        else:
            st.info("Dataset B forecast projection not available.")

    st.markdown("---")

    # 6. Report Downloads Section
    st.markdown("### 📥 Export Executive Comparison Reports")
    d_col1, d_col2 = st.columns(2)

    with d_col1:
        csv_bytes = kpi_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📄 Download KPI Table (CSV)",
            data=csv_bytes,
            file_name="atsa_two_dataset_comparison_kpis.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with d_col2:
        excel_report_bytes = build_dashboard_report(
            kpi_df=kpi_df,
            comparison_a=bundle_a.comparison,
            comparison_b=bundle_b.comparison,
            future_a=bundle_a.future.df if bundle_a.future else None,
            future_b=bundle_b.future.df if bundle_b.future else None,
            findings=findings,
        )
        st.download_button(
            label="📊 Download Multi-Sheet Workbook (Excel .xlsx)",
            data=excel_report_bytes,
            file_name="atsa_comparison_dashboard_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
