"""
Forecast & Model Comparison UI Tab for ATSA (Phase 5).

Provides:
- Section 1: Multi-model evaluation, styled comparison leaderboard, ranking by metric,
  Diebold-Mariano significance testing, Information Criteria groupings, and error by horizon.
- Section 2: Out-of-sample forward projections, future exogenous feature synthesis and editing,
  multi-model forward overlay, and CSV / multi-sheet Excel export.
"""

import hashlib
import json
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import streamlit as st

from modules.compare import (
    best_model,
    build_comparison_table,
    equal_weight_combination,
    make_future_exog,
    pairwise_dm,
    validate_uploaded_future_exog,
)
from modules.diagnostics import infer_period_from_frequency
from modules.export import build_excel_report
from modules.models import (
    ForecastResult,
    ModelResult,
    default_test_size,
    fit_full_and_forecast_detailed,
    make_model_key,
)
from modules.plots import (
    plot_error_by_horizon,
    plot_forecast_overlay,
    plot_future_forecast,
    plot_ic_bars,
    plot_metric_bars,
)
from ui.model_common import fit_all_default_models


@st.cache_data
def cached_future_forecast(
    y: pd.Series,
    spec: Dict[str, Any],
    steps: int,
    transform_info: Optional[Dict[str, Any]],
    exog: Optional[pd.DataFrame],
    future_exog: Optional[pd.DataFrame],
    m: Optional[int],
    exog_hash: str = "",
) -> ForecastResult:
    """UI cache wrapper for full-series forward forecast generation."""
    return fit_full_and_forecast_detailed(
        y=y,
        spec=spec,
        steps=steps,
        transform_info=transform_info,
        exog=exog,
        future_exog=future_exog,
        m=m,
    )


def render_forecast_tab() -> None:
    """Render the full Phase 5 Forecast & Model Comparison Workspace."""
    ts_series = st.session_state.get("ts_series")
    model_base_series = st.session_state.get("model_base_series")
    transform_info = st.session_state.get("transform_info", {})
    exog_df = st.session_state.get("exog_df")

    # 1. Precondition checks
    if ts_series is None or len(ts_series) == 0:
        st.info("ℹ️ No time series dataset is loaded yet. Please load or upload data in the 📥 Data tab.")
        return

    if model_base_series is None or len(model_base_series) == 0:
        st.info("ℹ️ Modeling base series is not set. Please review the dataset in the 📥 Data tab.")
        return

    n = len(model_base_series)
    if n < 30:
        st.warning(f"⚠️ The loaded time series has only {n} observations. At least 30 observations are required.")
        return

    inferred_m = transform_info.get("seasonal_period") or infer_period_from_frequency(ts_series) or 12
    active_m = int(st.session_state.get("model_m_input", inferred_m))
    holdout_size = int(st.session_state.get("model_holdout_size_input", default_test_size(n, active_m)))
    mase_choice = st.session_state.get("mase_scale_select", "m (seasonal naive)" if active_m >= 2 else "1 (non-seasonal naive)")
    mase_m = int(active_m) if "m" in mase_choice and active_m >= 2 else 1

    model_results = st.session_state.get("model_results", {})

    st.markdown("### 🔮 Forecast & Model Comparison Workspace")

    # Check if models have been estimated
    if not model_results:
        st.info(
            "ℹ️ No models have been estimated yet in the session. You can fit individual models in the "
            "**🧮 Models** tab or click the button below to automatically estimate all standard models now."
        )
        if st.button("🚀 Fit all default models", key="fc_fit_all_btn", width="stretch"):
            with st.spinner("Fitting all candidate model families..."):
                fit_all_default_models(
                    model_base_series=model_base_series,
                    holdout_size=holdout_size,
                    m=active_m,
                    mase_m=mase_m,
                    transform_info=transform_info,
                    exog_df=exog_df,
                )
            st.rerun()
        return

    # Filter eligible models: must match current holdout size
    current_holdout_idx = model_base_series.iloc[-holdout_size:].index
    eligible_models: List[ModelResult] = []
    for r in model_results.values():
        t_idx = getattr(r, "test_index", None)
        if t_idx is not None:
            if hasattr(t_idx, "equals") and t_idx.equals(current_holdout_idx):
                eligible_models.append(r)
            elif list(t_idx) == list(current_holdout_idx):
                eligible_models.append(r)

    if not eligible_models:
        st.info(
            f"ℹ️ Currently fitted models were estimated with a different holdout size than the active "
            f"setting ({holdout_size}). Please fit models in the **🧮 Models** tab or fit all defaults below."
        )
        if st.button("🚀 Refit all default models to current holdout", key="fc_refit_all_btn", width="stretch"):
            with st.spinner("Refitting all candidate models..."):
                fit_all_default_models(
                    model_base_series=model_base_series,
                    holdout_size=holdout_size,
                    m=active_m,
                    mase_m=mase_m,
                    transform_info=transform_info,
                    exog_df=exog_df,
                )
            st.rerun()
        return

    # -------------------------------------------------------------
    # SECTION 1: MODEL COMPARISON
    # -------------------------------------------------------------
    st.markdown("---")
    st.subheader("1. Holdout Model Comparison & Benchmarking")

    cmp_ctrl1, cmp_ctrl2, cmp_ctrl3 = st.columns([2.5, 1.2, 1.3])
    with cmp_ctrl1:
        eligible_labels = [r.label for r in eligible_models]
        default_selected = [r.label for r in eligible_models if r.converged]
        selected_model_labels = st.multiselect(
            "Select Models to Compare:",
            options=eligible_labels,
            default=default_selected if default_selected else eligible_labels,
            key="cmp_model_select",
        )

    with cmp_ctrl2:
        rank_metric = st.selectbox(
            "Rank By Metric:",
            options=["RMSE", "MAE", "MAPE", "sMAPE", "MASE"],
            index=0,
            key="cmp_rank_by",
        )

    with cmp_ctrl3:
        st.write("")
        st.write("")
        add_ensemble = st.checkbox("Add equal-weight combination", value=False, key="cmp_add_ensemble")

    selected_models = [r for r in eligible_models if r.label in selected_model_labels]
    if not selected_models:
        st.warning("Please select at least one model to display the comparison.")
        return

    # Add equal-weight combination if requested
    if add_ensemble:
        ensemble_model = equal_weight_combination(selected_models)
        if ensemble_model is not None:
            selected_models.append(ensemble_model)

    # Build comparison table
    comp_df = build_comparison_table(selected_models, rank_by=rank_metric.lower(), reference_family="naive")

    # Summary Banner
    best_key = best_model(comp_df, metric=rank_metric)
    best_row = comp_df[comp_df["Key"] == best_key].iloc[0] if best_key and not comp_df[comp_df["Key"] == best_key].empty else comp_df.iloc[0]
    best_label = best_row["Model"]
    best_val = best_row[rank_metric]
    rel_rmse = best_row["RelRMSE_vs_ref"]

    has_naive = any(getattr(r, "family", "") == "naive" and r.converged for r in eligible_models)
    if has_naive and not np.isnan(rel_rmse):
        if rel_rmse < 1.0:
            pct_improvement = (1.0 - rel_rmse) * 100.0
            st.success(
                f"🏆 **Top Performer**: **{best_label}** achieves the lowest {rank_metric} (**{best_val:.3f}**). "
                f"It outperforms the Naive baseline by **{pct_improvement:.1f}%** (RelRMSE = **{rel_rmse:.3f}**).",
                icon="✅",
            )
        else:
            st.warning(
                f"⚠️ **Top Performer**: **{best_label}** achieves the lowest {rank_metric} (**{best_val:.3f}**), "
                f"but does **not** beat the Naive benchmark (RelRMSE = **{rel_rmse:.3f}**).",
                icon="⚠️",
            )
    else:
        st.info(
            f"🏆 **Top Performer**: **{best_label}** achieves the lowest {rank_metric} (**{best_val:.3f}**). "
            f"*(Fit a Naive baseline to evaluate relative percentage improvement).* ",
            icon="ℹ️",
        )

    # Styled Leaderboard Table
    display_df = comp_df.drop(columns=["Key"], errors="ignore").copy()
    metric_cols = ["RMSE", "MAE", "MAPE", "sMAPE", "MASE"]

    styled_table = (
        display_df.style.format(
            {
                "RMSE": "{:.3f}",
                "MAE": "{:.3f}",
                "MAPE": "{:.2f}%",
                "sMAPE": "{:.2f}%",
                "MASE": "{:.3f}",
                "Coverage95": "{:.1f}%",
                "AvgWidth": "{:.3f}",
                "RelRMSE_vs_ref": "{:.3f}",
                "AIC": "{:.1f}",
                "BIC": "{:.1f}",
                "LB_min_p": "{:.4f}",
                "JB_p": "{:.4f}",
                "FitSec": "{:.3f}",
            },
            na_rep="N/A",
        )
        .highlight_min(subset=[c for c in metric_cols if c in display_df.columns], color="#dbeafe")
        .highlight_max(subset=[c for c in ["Coverage95"] if c in display_df.columns], color="#dcfce7")
    )

    st.dataframe(styled_table, width="stretch")
    st.caption(
        "📌 **Diagnostic & Comparison Notes**: "
        "AIC and BIC are strictly comparable **only within the same IC_group** and never across ARIMA-class vs ETS models. "
        "MAPE is unstable and unreliable if values are near zero. "
        f"A {holdout_size}-point holdout constitutes a small validation sample. "
        "Forecasts generated through a log or Box-Cox transformation represent **median** forecasts on the original scale."
    )

    # Comparison Visualizations Tabs
    tab_overlay, tab_metrics, tab_ic, tab_error, tab_sig = st.tabs(
        ["📈 Forecast Overlay", "📊 Metric Bars", "🏛️ Information Criteria", "📉 Error by Horizon", "🔬 Statistical Significance"]
    )

    with tab_overlay:
        actual_holdout = model_base_series.iloc[-holdout_size:]
        band_opts = ["None"] + [r.label for r in selected_models if getattr(r, "test_forecast", None) is not None and "lower" in r.test_forecast]
        selected_band_model = st.selectbox("Show 95% Prediction Interval for:", options=band_opts, index=0, key="cmp_band_select")

        fc_dict = {
            r.label: r.test_forecast
            for r in selected_models
            if getattr(r, "test_forecast", None) is not None and not r.test_forecast.empty
        }
        fig_overlay = plot_forecast_overlay(
            actual=actual_holdout,
            forecasts=fc_dict,
            show_bands_for=None if selected_band_model == "None" else selected_band_model,
        )
        st.plotly_chart(fig_overlay, width="stretch")

    with tab_metrics:
        fig_metric = plot_metric_bars(comp_df, metric=rank_metric)
        st.plotly_chart(fig_metric, width="stretch")

    with tab_ic:
        fig_ic = plot_ic_bars(comp_df)
        st.plotly_chart(fig_ic, width="stretch")

    with tab_error:
        err_dict = {}
        for r in selected_models:
            if getattr(r, "test_forecast", None) is not None and "mean" in r.test_forecast:
                act = getattr(r, "test_actual", None)
                if act is None:
                    act = actual_holdout
                err_dict[r.label] = pd.Series(act.values - r.test_forecast["mean"].values, index=act.index)
        fig_err = plot_error_by_horizon(err_dict)
        st.plotly_chart(fig_err, width="stretch")

    with tab_sig:
        st.markdown("##### Pairwise Diebold-Mariano Test")
        st.caption("Hypothesis test for equal predictive accuracy against a reference benchmark model.")
        c_sig1, c_sig2 = st.columns([2, 1])
        with c_sig1:
            converged_models = [r for r in selected_models if r.converged]
            label_to_key = {r.label: r.key for r in converged_models}
            ref_opts = list(label_to_key.keys())
            ref_choice = st.selectbox("Reference Benchmark Model:", options=ref_opts, index=0, key="dm_ref_select")
            ref_key = label_to_key.get(ref_choice)
        with c_sig2:
            dm_loss = st.selectbox("Loss Function:", options=["Squared Error (MSE)", "Absolute Error (MAE)"], index=0, key="dm_loss_select")
            dm_power = 2 if "Squared" in dm_loss else 1

        dm_df = pairwise_dm(selected_models, reference_key=ref_key, power=dm_power)
        if not dm_df.empty:
            st.dataframe(dm_df, width="stretch")
            st.caption("ℹ️ *Note: Indicative only: single forecast origin, small holdout, low power.*")
        else:
            st.info("No comparative Diebold-Mariano statistics could be computed.")

    # CSV Download for Comparison Table
    st.download_button(
        label="📥 Download Comparison Table (CSV)",
        data=comp_df.to_csv(index=False).encode("utf-8"),
        file_name="atsa_model_comparison_leaderboard.csv",
        mime="text/csv",
        key="btn_download_comparison_csv",
        width="stretch",
    )

    # -------------------------------------------------------------
    # SECTION 2: FUTURE OUT-OF-SAMPLE FORECAST
    # -------------------------------------------------------------
    st.markdown("---")
    st.subheader("2. Out-of-Sample Forward Forecast")

    converged_eligible = [r for r in eligible_models if r.converged and getattr(r, "family", "") != "combination"]
    if not converged_eligible:
        st.info("No converged statistical models available for future forecasting.")
        return

    f_col1, f_col2, f_col3 = st.columns([2, 1.2, 1.8])
    with f_col1:
        model_names = [r.label for r in converged_eligible]
        def_idx = 0
        if best_label in model_names:
            def_idx = model_names.index(best_label)
        chosen_fc_label = st.selectbox("Select Model for Future Projection:", options=model_names, index=def_idx, key="future_model_select")
        chosen_result = next((r for r in converged_eligible if r.label == chosen_fc_label), converged_eligible[0])

    with f_col2:
        max_h = max(1, min(60, n // 2))
        def_h = min(max_h, max(1, int(active_m or 12)))
        steps_ahead = st.number_input("Forecast Horizon (Steps Ahead):", min_value=1, max_value=max_h, value=def_h, step=1, key="future_horizon_input")
        if steps_ahead > 2 * holdout_size:
            st.warning("⚠️ Horizon > 2x holdout size: uncertainty grows rapidly and intervals widen.")

    with f_col3:
        # Multi-model overlay option
        other_models = [r.label for r in converged_eligible if r.label != chosen_fc_label]
        overlay_models = st.multiselect("Overlay additional models (means only, max 4):", options=other_models, max_selections=4, key="future_overlay_select")

    # Exogenous Regressor Handling for SARIMAX
    future_exog_df: Optional[pd.DataFrame] = None
    if chosen_result.family == "sarimax":
        st.markdown("##### 🌐 Future Exogenous Regressors Configuration")
        st.warning("⚠️ **Forecast depends on YOUR assumptions for future exogenous values.**")

        if exog_df is None or exog_df.empty:
            st.error("SARIMAX requires exogenous features, but none are selected in the Data tab.")
            return

        exog_mode = st.radio(
            "Exogenous Input Mode:",
            options=["Strategy", "Edit table", "Upload CSV"],
            horizontal=True,
            key="exog_mode_radio",
        )

        try:
            if exog_mode == "Strategy":
                s_c1, s_c2 = st.columns(2)
                with s_c1:
                    strat_choice = st.selectbox(
                        "Extrapolation Strategy:",
                        options=["repeat_last_season", "repeat_last", "mean_last_k", "linear_trend"],
                        index=0 if active_m >= 2 else 1,
                        key="exog_strat_choice",
                    )
                with s_c2:
                    k_val = st.number_input("Window k (for mean_last_k):", min_value=1, max_value=len(exog_df), value=min(12, len(exog_df)), step=1) if strat_choice == "mean_last_k" else None
                    m_val_ex = active_m if strat_choice == "repeat_last_season" else None

                future_exog_df = make_future_exog(
                    exog=exog_df,
                    steps=steps_ahead,
                    strategy=strat_choice,
                    m=m_val_ex,
                    k=k_val,
                )
                st.dataframe(future_exog_df.head(10), width="stretch")

            elif exog_mode == "Edit table":
                base_synth = make_future_exog(exog=exog_df, steps=steps_ahead, strategy="repeat_last_season" if active_m >= 2 else "repeat_last", m=active_m)
                edited_df = st.data_editor(base_synth, width="stretch", key="exog_data_editor")
                future_exog_df = validate_uploaded_future_exog(edited_df, required_columns=list(exog_df.columns), steps=steps_ahead, expected_index=base_synth.index)

            elif exog_mode == "Upload CSV":
                uploaded_exog_file = st.file_uploader("Upload Future Exogenous CSV (must have matching columns and steps rows):", type=["csv"], key="future_exog_uploader")
                if uploaded_exog_file is not None:
                    raw_up_df = pd.read_csv(uploaded_exog_file)
                    base_synth = make_future_exog(exog=exog_df, steps=steps_ahead, strategy="repeat_last")
                    future_exog_df = validate_uploaded_future_exog(raw_up_df, required_columns=list(exog_df.columns), steps=steps_ahead, expected_index=base_synth.index)
                    st.success("Uploaded future exogenous features verified.")
                else:
                    st.info("Please upload a CSV with future exogenous regressors.")
                    return

        except Exception as exog_err:
            st.error(f"❌ Exogenous validation error: {exog_err}")
            return

    # Generate Forecast Execution
    if "forecast_results" not in st.session_state:
        st.session_state["forecast_results"] = {}

    if st.button("🔮 Generate forecast", type="primary", key="btn_generate_forecast", width="stretch"):
        with st.spinner(f"Refitting {chosen_result.label} on full dataset and forecasting {steps_ahead} steps..."):
            try:
                exog_hash = ""
                if future_exog_df is not None:
                    exog_bytes = pd.util.hash_pandas_object(future_exog_df).values.tobytes()
                    exog_hash = hashlib.md5(exog_bytes).hexdigest()[:8]

                fc_key = make_model_key(
                    spec=chosen_result.spec,
                    test_size=steps_ahead,
                    transform_info=transform_info,
                    exog_cols=tuple(exog_df.columns) if exog_df is not None else None,
                    n_obs=len(model_base_series),
                    last_date=model_base_series.index[-1],
                    mase_m=mase_m,
                ) + f"_{steps_ahead}_{exog_hash}"

                fc_res = cached_future_forecast(
                    y=model_base_series,
                    spec=chosen_result.spec,
                    steps=steps_ahead,
                    transform_info=transform_info,
                    exog=exog_df if chosen_result.family == "sarimax" else None,
                    future_exog=future_exog_df if chosen_result.family == "sarimax" else None,
                    m=active_m,
                    exog_hash=exog_hash,
                )
                st.session_state["forecast_results"][fc_key] = fc_res
                st.session_state["active_forecast_key"] = fc_key

                # Also generate overlays if selected
                for ov_label in overlay_models:
                    ov_res = next((r for r in converged_eligible if r.label == ov_label), None)
                    if ov_res and ov_res.family != "sarimax":
                        ov_key = f"{ov_res.key}_{steps_ahead}"
                        ov_fc = cached_future_forecast(
                            y=model_base_series,
                            spec=ov_res.spec,
                            steps=steps_ahead,
                            transform_info=transform_info,
                            exog=None,
                            future_exog=None,
                            m=active_m,
                        )
                        st.session_state["forecast_results"][ov_key] = ov_fc

            except Exception as fc_err:
                st.error(f"❌ Forecast generation failed: {fc_err}")

    # Display Active Forecast
    active_fc_key = st.session_state.get("active_forecast_key")
    if active_fc_key and active_fc_key in st.session_state["forecast_results"]:
        active_fc: ForecastResult = st.session_state["forecast_results"][active_fc_key]

        fc_h1, fc_h2, fc_h3 = st.columns([3, 1, 1])
        with fc_h1:
            st.markdown(f"#### 📡 Forward Projection: {active_fc.label}")
        with fc_h2:
            if active_fc.converged:
                st.success("🟢 Converged", icon="✅")
            else:
                st.warning("🟡 Unconverged", icon="⚠️")
        with fc_h3:
            st.metric("Fit Time", f"{active_fc.fit_seconds:.3f} s")

        if active_fc.warnings:
            for w in active_fc.warnings:
                st.warning(f"⚠️ {w}")

        # Forecast Plot
        fig_future = plot_future_forecast(
            history=model_base_series,
            forecast_df=active_fc.df,
            label=active_fc.label,
            tail=min(len(model_base_series), 80),
            holdout_forecast=chosen_result.test_forecast if chosen_result else None,
        )

        # Multi-model overlay on the future plot
        palette = ["#059669", "#d97706", "#7c3aed", "#db2777"]
        for o_idx, ov_label in enumerate(overlay_models):
            ov_res = next((r for r in converged_eligible if r.label == ov_label), None)
            if ov_res:
                ov_key = f"{ov_res.key}_{steps_ahead}"
                if ov_key in st.session_state["forecast_results"]:
                    ov_fc_obj: ForecastResult = st.session_state["forecast_results"][ov_key]
                    if "mean" in ov_fc_obj.df.columns:
                        import plotly.graph_objects as go
                        fig_future.add_trace(
                            go.Scatter(
                                x=ov_fc_obj.df.index,
                                y=ov_fc_obj.df["mean"].values,
                                mode="lines",
                                line=dict(color=palette[o_idx % len(palette)], width=2, dash="dot"),
                                name=f"{ov_label} (Mean)",
                                hovertemplate=f"<b>{ov_label}</b>: %{{y:.3f}}<extra></extra>",
                            )
                        )

        st.plotly_chart(fig_future, width="stretch")

        # Future forecast values table
        st.markdown("##### 📋 Future Forecast Values (with 95% Confidence Intervals)")
        formatted_future = active_fc.df.copy()
        for c in formatted_future.columns:
            formatted_future[c] = formatted_future[c].round(4)
        st.dataframe(formatted_future, width="stretch")
        st.caption("📌 Note: Forecasts made through a log or Box-Cox transformation represent median forecasts on the original scale.")

        # Download Buttons: CSV & Multi-sheet Excel
        d_c1, d_c2 = st.columns(2)
        with d_c1:
            st.download_button(
                label="📥 Download Future Forecast (CSV)",
                data=active_fc.df.to_csv(index=True).encode("utf-8"),
                file_name=f"atsa_{active_fc.label.lower().replace(' ', '_')}_forecast.csv",
                mime="text/csv",
                key="btn_download_future_csv",
                width="stretch",
            )

        with d_c2:
            # Build Multi-sheet Excel report
            holdout_dict = {
                r.label: r.test_forecast
                for r in selected_models
                if getattr(r, "test_forecast", None) is not None
            }
            specs_list = [
                {
                    "label": r.label,
                    "family": r.family,
                    "spec": r.spec,
                    "transform_steps": transform_info.get("steps", []),
                    "holdout_size": holdout_size,
                    "mase_m": mase_m,
                }
                for r in selected_models
            ]
            excel_bytes = build_excel_report(
                comparison_df=comp_df,
                holdout_forecasts=holdout_dict,
                future_forecast_df=active_fc.df,
                specs=specs_list,
                metadata={
                    "holdout_size": holdout_size,
                    "mase_m": mase_m,
                    "run_date": str(pd.Timestamp.now()),
                    "transform_steps": transform_info.get("steps", []),
                },
            )
            st.download_button(
                label="📊 Download Comprehensive Report (Excel Workbook)",
                data=excel_bytes,
                file_name="atsa_complete_forecast_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="btn_download_excel_report",
                width="stretch",
            )
