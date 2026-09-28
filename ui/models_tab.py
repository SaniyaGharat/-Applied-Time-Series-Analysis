"""
Model Selector and Diagnostic UI Tab for ATSA (Phase 4).

Provides interactive model fitting, per-model visualizations, residual diagnostics,
parameter summaries, automated order grid search, and multi-model benchmarking across:
- Baselines: Naive, Seasonal Naive, Drift
- Classical: AR, MA, ARMA, ARIMA
- Seasonal & Regressors: SARIMA, SARIMAX
- Exponential Smoothing: Holt-Winters
"""

import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import streamlit as st

from modules.diagnostics import (
    compute_acf_pacf,
    infer_period_from_frequency,
    suggest_orders,
    theoretical_acf_pacf,
    transform_series,
)
from modules.models import (
    ModelResult,
    default_test_size,
    fit_model,
    grid_search_arima,
    list_model_families,
    make_model_key,
)
from modules.plots import (
    plot_acf_pacf_combined,
    plot_fit_vs_actual,
    plot_grid_results,
    plot_holdout_zoom,
    plot_hw_components,
    plot_inverse_roots,
    plot_residual_diagnostics,
    plot_theoretical_vs_empirical,
)
from ui.model_common import (
    cached_fit_model,
    cached_grid_search,
    compute_suggested_specs,
    fit_all_default_models,
    reset_model_widgets,
)


def render_models_tab() -> None:
    """Render the full Phase 4 Model Selector & Diagnostic Workspace."""
    # A1: Process pending spec before any widget is created
    if "pending_spec" in st.session_state and st.session_state["pending_spec"]:
        pending = st.session_state.pop("pending_spec")
        for k, v in pending.items():
            st.session_state[k] = v

    ts_series = st.session_state.get("ts_series")
    model_base_series = st.session_state.get("model_base_series")
    transform_info = st.session_state.get("transform_info", {})
    exog_df = st.session_state.get("exog_df")

    # 1. Precondition checks
    if ts_series is None or len(ts_series) == 0:
        st.info("ℹ️ No time series dataset is loaded yet. Please upload a dataset or click **Load Demo Data** in the 📥 Data tab.")
        return

    if model_base_series is None:
        st.info(
            "ℹ️ Modeling base series is not set. Please review the dataset in the 📥 Data tab "
            "or configure transformations in the 🔍 EDA & Diagnostics tab."
        )
        return

    n = len(model_base_series)
    if model_base_series.nunique() <= 1:
        st.warning("⚠️ The loaded time series is constant (all values are identical). Model estimation cannot proceed.")
        return

    if n < 30:
        st.warning(f"⚠️ The loaded time series has only {n} observations. Reliable time series modeling requires at least 30 observations.")
        return

    st.markdown("### 🧮 Model Zoo & Forecast Workspace")

    # 2. Active Pipeline Banner
    steps = transform_info.get("steps", [])
    steps_txt = " -> ".join(steps) if steps else "None (Regularized Series)"
    suggested_d = transform_info.get("suggested_d", 0)
    suggested_D = transform_info.get("suggested_D", 0)
    inferred_m = transform_info.get("seasonal_period") or infer_period_from_frequency(ts_series) or 12

    # A2: Stale widget reset when transform suggestions differ from last seen
    curr_suggestions = (suggested_d, suggested_D, inferred_m)
    if st.session_state.get("_last_transform_suggestions") != curr_suggestions:
        reset_model_widgets()
        st.session_state["_last_transform_suggestions"] = curr_suggestions

    st.info(
        f"🎯 **Active Modeling Pipeline**: `{steps_txt}` | "
        f"Differencing orders for models: **d={suggested_d}, D={suggested_D}** (m={inferred_m})\n\n"
        f"📌 *Note: Forecasts made through a log transform are MEDIAN forecasts on the original scale (a known bias), "
        f"and MAPE is unreliable when the series crosses or is near zero.*"
    )

    # 3. Global Estimation Controls (Holdout & Seasonal Period & MASE scale)
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.5, 1.5, 3])
    with ctrl_col1:
        max_test = max(1, int(0.3 * n))
        def_test = min(max_test, max(1, default_test_size(n, inferred_m)))
        curr_holdout = st.session_state.get("model_holdout_size_input", def_test)
        curr_holdout = min(max_test, max(1, int(curr_holdout)))
        st.session_state["model_holdout_size_input"] = curr_holdout
        holdout_size = st.number_input(
            "Holdout Test Size (N obs):",
            min_value=1,
            max_value=max_test,
            step=1,
            help="Number of points held out at the end of the series to validate forecast accuracy.",
            key="model_holdout_size_input",
        )

    with ctrl_col2:
        max_m = max(2, n // 2)
        def_m = min(max_m, max(2, int(inferred_m)))
        curr_m = st.session_state.get("model_m_input", def_m)
        curr_m = min(max_m, max(2, int(curr_m)))
        st.session_state["model_m_input"] = curr_m
        active_m = st.number_input(
            "Seasonal Period (m):",
            min_value=2,
            max_value=max_m,
            step=1,
            help="Length of seasonal cycle (e.g. 12 for monthly, 4 for quarterly, 7 for daily).",
            key="model_m_input",
        )

    with ctrl_col3:
        mase_opts = ["1 (non-seasonal naive)", "m (seasonal naive)"]
        def_mase = "m (seasonal naive)" if active_m >= 2 else "1 (non-seasonal naive)"
        st.session_state.setdefault("mase_scale_select", def_mase)
        mase_choice = st.selectbox(
            "MASE scale:",
            options=mase_opts,
            help="Benchmark naive denominator for MASE calculation.",
            key="mase_scale_select",
        )
        mase_m = int(active_m) if "m" in mase_choice and active_m >= 2 else 1
        st.session_state["model_mase_m_val"] = mase_m
        st.caption(f"Series: **{n}** obs | Train: **{n - holdout_size}** obs | Test: **{holdout_size}** obs | MASE scale m=**{mase_m}**")

    st.markdown("---")

    # 4. Model Family Selector
    st.subheader("1. Select Model Family")
    all_families = list_model_families()
    family_labels = [
        "Naive",
        "Seasonal Naive",
        "Drift",
        "AR",
        "MA",
        "ARMA",
        "ARIMA",
        "SARIMA",
        "SARIMAX",
        "Holt-Winters",
    ]
    label_to_key = {
        "Naive": "naive",
        "Seasonal Naive": "seasonal_naive",
        "Drift": "drift",
        "AR": "ar",
        "MA": "ma",
        "ARMA": "arma",
        "ARIMA": "arima",
        "SARIMA": "sarima",
        "SARIMAX": "sarimax",
        "Holt-Winters": "holt_winters",
    }
    key_to_label = {v: k for k, v in label_to_key.items()}

    current_family_key = st.session_state.get("selected_model_family", None)
    current_label = key_to_label.get(current_family_key)

    # Support ATSA_TEST fallback for automated AppTest runner
    use_fallback = os.environ.get("ATSA_TEST") == "1" or not hasattr(st, "pills")
    if use_fallback:
        fallback_options = ["(Select a model family)"] + family_labels
        idx = fallback_options.index(current_label) if current_label in fallback_options else 0
        chosen_label = st.selectbox(
            "Model Family:",
            options=fallback_options,
            index=idx,
            key="model_family_fallback",
        )
        if chosen_label == "(Select a model family)":
            chosen_label = None
    else:
        chosen_pill = st.pills(
            "Choose a forecasting architecture:",
            options=family_labels,
            selection_mode="single",
            default=current_label,
            key="model_family_select",
        )
        chosen_label = chosen_pill

    selected_family = label_to_key.get(chosen_label)
    st.session_state["selected_model_family"] = selected_family

    if not selected_family:
        st.info("👆 Please select a forecasting model family above to begin configuration and estimation.")
        return

    fam_meta = next((f for f in all_families if f["key"] == selected_family), None)
    if fam_meta:
        st.caption(f"📖 **{fam_meta['label']}**: {fam_meta['description']}")

    # SARIMAX Exogenous check
    if selected_family == "sarimax":
        if exog_df is None or exog_df.empty:
            st.info("Select exogenous columns in the Data tab")
            return
        else:
            st.caption(f"Exogenous features: `{', '.join(list(exog_df.columns))}`")

    # 5. Model Parameter Configuration Form
    st.markdown("##### ⚙️ Hyperparameters & Specification")

    # Compute suggested p and q on differenced base series
    diff_base, _ = transform_series(
        model_base_series,
        diff_order=suggested_d,
        seasonal_diff_order=suggested_D,
        seasonal_period=active_m if suggested_D > 0 else None,
    )
    acf_pacf_data = compute_acf_pacf(diff_base, nlags=min(20, len(diff_base) // 2 - 1))
    sugg_orders = suggest_orders(acf_pacf_data["acf"], acf_pacf_data["pacf"], acf_pacf_data["conf_bound"], m=active_m)
    suggested_p = max(1, min(3, sugg_orders["suggested_p"]))
    suggested_q = max(1, min(3, sugg_orders["suggested_q"]))

    with st.form("model_parameter_form"):
        spec: Dict[str, Any] = {"family": selected_family}

        if selected_family == "ar":
            c1, c2, c3 = st.columns(3)
            with c1:
                st.session_state.setdefault("form_ar_p", int(suggested_p))
                spec["p"] = st.number_input("AR Order (p):", min_value=1, max_value=10, step=1, key="form_ar_p")
            with c2:
                st.session_state.setdefault("form_ar_d", int(suggested_d))
                spec["d"] = st.number_input("d (0 if series is stationary):", min_value=0, max_value=2, step=1, key="form_ar_d")
            with c3:
                st.session_state.setdefault("form_ar_trend", "c" if int(st.session_state.get("form_ar_d", suggested_d)) == 0 else "n")
                spec["trend"] = st.selectbox("Trend Component:", options=["c", "n", "t", "ct"], key="form_ar_trend")

        elif selected_family == "ma":
            c1, c2, c3 = st.columns(3)
            with c1:
                st.session_state.setdefault("form_ma_q", int(suggested_q))
                spec["q"] = st.number_input("MA Order (q):", min_value=1, max_value=10, step=1, key="form_ma_q")
            with c2:
                st.session_state.setdefault("form_ma_d", int(suggested_d))
                spec["d"] = st.number_input("d (0 if series is stationary):", min_value=0, max_value=2, step=1, key="form_ma_d")
            with c3:
                st.session_state.setdefault("form_ma_trend", "c" if int(st.session_state.get("form_ma_d", suggested_d)) == 0 else "n")
                spec["trend"] = st.selectbox("Trend Component:", options=["c", "n", "t", "ct"], key="form_ma_trend")

        elif selected_family == "arma":
            c1, c2, c3 = st.columns(3)
            with c1:
                st.session_state.setdefault("form_arma_p", int(suggested_p))
                spec["p"] = st.number_input("AR Order (p):", min_value=1, max_value=10, step=1, key="form_arma_p")
            with c2:
                st.session_state.setdefault("form_arma_q", int(suggested_q))
                spec["q"] = st.number_input("MA Order (q):", min_value=1, max_value=10, step=1, key="form_arma_q")
            with c3:
                st.session_state.setdefault("form_arma_d", 0)
                spec["d"] = st.number_input("d (0 if series is stationary):", min_value=0, max_value=2, step=1, key="form_arma_d")

        elif selected_family == "arima":
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.session_state.setdefault("form_arima_p", int(suggested_p))
                spec["p"] = st.number_input("AR Order (p):", min_value=0, max_value=10, step=1, key="form_arima_p")
            with c2:
                st.session_state.setdefault("form_arima_d", int(suggested_d))
                spec["d"] = st.number_input("Integration Order (d):", min_value=0, max_value=2, step=1, key="form_arima_d")
            with c3:
                st.session_state.setdefault("form_arima_q", int(suggested_q))
                spec["q"] = st.number_input("MA Order (q):", min_value=0, max_value=10, step=1, key="form_arima_q")
            with c4:
                st.session_state.setdefault("form_arima_trend", "c" if int(st.session_state.get("form_arima_d", suggested_d)) == 0 else "n")
                spec["trend"] = st.selectbox("Trend Component:", options=["c", "n", "t", "ct"], key="form_arima_trend")

        elif selected_family in ["sarima", "sarimax"]:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.session_state.setdefault("form_s_p", int(suggested_p))
                spec["p"] = st.number_input("AR Order (p):", min_value=0, max_value=10, step=1, key="form_s_p")
                st.session_state.setdefault("form_s_P", 1)
                spec["P"] = st.number_input("Seasonal AR (P):", min_value=0, max_value=3, step=1, key="form_s_P")
            with c2:
                st.session_state.setdefault("form_s_d", int(suggested_d))
                spec["d"] = st.number_input("Integration Order (d):", min_value=0, max_value=2, step=1, key="form_s_d")
                st.session_state.setdefault("form_s_D", int(suggested_D))
                spec["D"] = st.number_input("Seasonal Diff (D):", min_value=0, max_value=2, step=1, key="form_s_D")
            with c3:
                st.session_state.setdefault("form_s_q", int(suggested_q))
                spec["q"] = st.number_input("MA Order (q):", min_value=0, max_value=10, step=1, key="form_s_q")
                st.session_state.setdefault("form_s_Q", 1)
                spec["Q"] = st.number_input("Seasonal MA (Q):", min_value=0, max_value=3, step=1, key="form_s_Q")
            with c4:
                spec["m"] = int(active_m)
                st.session_state.setdefault("form_s_trend", "c" if (suggested_d == 0 and suggested_D == 0) else "n")
                spec["trend"] = st.selectbox("Trend Component:", options=["c", "n", "t", "ct"], key="form_s_trend")

        elif selected_family == "holt_winters":
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.session_state.setdefault("form_hw_trend", "add")
                trend_opt = st.selectbox("Trend Type:", options=["None", "add", "mul"], key="form_hw_trend")
                spec["trend"] = None if trend_opt == "None" else trend_opt
            with c2:
                st.session_state.setdefault("form_hw_damped", False)
                spec["damped"] = st.checkbox("Damped Trend", key="form_hw_damped")
            with c3:
                st.session_state.setdefault("form_hw_seas", "add")
                seas_opt = st.selectbox("Seasonal Type:", options=["None", "add", "mul"], key="form_hw_seas")
                spec["seasonal"] = None if seas_opt == "None" else seas_opt
            with c4:
                spec["m"] = int(active_m)

        elif selected_family == "seasonal_naive":
            spec["m"] = int(active_m)
            st.caption(f"Seasonal Naive will project past values with period m={active_m}.")

        elif selected_family in ["naive", "drift"]:
            st.caption(f"{fam_meta['label']} has no tunable hyperparameters.")

        submitted = st.form_submit_button("Update model")

    # 6. Fit Model Execution
    with st.spinner(f"Estimating {fam_meta['label'] if fam_meta else selected_family}..."):
        result = cached_fit_model(
            y=model_base_series,
            spec=spec,
            test_size=holdout_size,
            transform_info=transform_info,
            exog=exog_df if selected_family == "sarimax" else None,
            m=active_m,
            mase_m=st.session_state.get("model_mase_m_val", mase_m),
        )

    # Store in session state registry
    if "model_results" not in st.session_state:
        st.session_state["model_results"] = {}
    st.session_state["model_results"][result.key] = result

    # Check for failure
    if not result.converged and (result.test_forecast.empty or result.test_forecast["mean"].isna().all()):
        st.error(f"❌ **Model Estimation Failed**: {'; '.join(result.warnings) if result.warnings else result.summary_text}")
        return

    # 7. Result Header & Evaluation Cards
    st.markdown("---")
    res_h1, res_h2, res_h3 = st.columns([3, 1.2, 1.2])
    with res_h1:
        st.markdown(f"#### 📊 {result.label}")
    with res_h2:
        if result.converged:
            st.success("🟢 Converged", icon="✅")
        else:
            st.warning("🟡 Unconverged", icon="⚠️")
    with res_h3:
        st.metric("Fit Duration", f"{result.fit_seconds:.3f} s")

    # Warning notifications
    if result.warnings:
        for w in result.warnings:
            st.warning(f"⚠️ {w}")

    # Metric Cards
    m_row1, m_row2, m_row3, m_row4, m_row5, m_row6, m_row7 = st.columns(7)
    m_test = result.metrics_test
    m_row1.metric("Holdout RMSE", f"{m_test['rmse']:.3f}" if not np.isnan(m_test['rmse']) else "N/A")
    m_row2.metric("Holdout MAE", f"{m_test['mae']:.3f}" if not np.isnan(m_test['mae']) else "N/A")

    mape_val = m_test["mape"]
    m_row3.metric("Holdout MAPE", f"{mape_val:.2f}%" if not np.isnan(mape_val) else "N/A")
    m_row4.metric("Holdout sMAPE", f"{m_test['smape']:.2f}%" if not np.isnan(m_test['smape']) else "N/A")
    m_row5.metric("Holdout MASE", f"{m_test['mase']:.3f}" if not np.isnan(m_test['mase']) else "N/A")

    ic = result.information_criteria or {}
    aic_val = ic.get("aic")
    bic_val = ic.get("bic")
    m_row6.metric("AIC", f"{aic_val:.1f}" if aic_val is not None else "N/A")
    m_row7.metric("BIC", f"{bic_val:.1f}" if bic_val is not None else "N/A")

    if np.isnan(mape_val) or mape_val > 100.0:
        st.caption("⚠️ *Note on MAPE*: MAPE is distorted when actual values cross or approach zero. Inspect sMAPE and MASE instead.")

    # 8. Result Tabs
    t_fc, t_diag, t_plots, t_summary, t_grid = st.tabs(
        [
            "📈 Fit & Forecast",
            "🧪 Residual Diagnostics",
            "📐 Model Plots",
            "📋 Parameters & Summary",
            "🔎 Auto Order Search",
        ]
    )

    # -------------------------------------------------------------
    # TAB 1: FIT & FORECAST
    # -------------------------------------------------------------
    with t_fc:
        st.subheader("Model Trajectory & Holdout Projection")
        fig_fit = plot_fit_vs_actual(
            actual=ts_series,
            fitted=result.fitted_train,
            forecast_df=result.test_forecast,
            title=f"{result.label} (In-Sample Fit & Out-of-Sample Holdout)",
        )
        st.plotly_chart(fig_fit, width="stretch", key=f"plot_fit_vs_actual_{selected_family}_{result.key}")

        fig_zoom = plot_holdout_zoom(actual=ts_series, forecast_df=result.test_forecast)
        st.plotly_chart(fig_zoom, width="stretch", key=f"plot_holdout_zoom_{selected_family}_{result.key}")

        st.markdown("##### 📄 Holdout Comparison Table")
        test_actual = ts_series.reindex(result.test_forecast.index)
        comp_df = pd.DataFrame(
            {
                "Actual": test_actual,
                "Forecast Mean": result.test_forecast["mean"],
                "Error": test_actual - result.test_forecast["mean"],
                "95% Lower": result.test_forecast["lower"],
                "95% Upper": result.test_forecast["upper"],
            },
            index=result.test_forecast.index,
        )
        st.dataframe(comp_df, width="stretch")

    # -------------------------------------------------------------
    # TAB 2: RESIDUAL DIAGNOSTICS
    # -------------------------------------------------------------
    with t_diag:
        st.subheader("Statistical Residual Diagnostics")
        burn_in_n = spec.get("d", 0) + spec.get("D", 0) * (spec.get("m", 1) or 1)
        if burn_in_n > 0:
            st.caption(f"ℹ️ The first {burn_in_n} in-sample residuals were dropped as integration burn-in.")

        if not result.residuals.empty:
            fig_resid = plot_residual_diagnostics(result.residuals)
            st.plotly_chart(fig_resid, width="stretch", key=f"plot_residual_diag_{selected_family}_{result.key}")

            d_col1, d_col2 = st.columns(2)
            with d_col1:
                st.markdown("#### Ljung-Box Test (White Noise Check)")
                if result.ljung_box is not None and not result.ljung_box.empty:
                    st.dataframe(result.ljung_box, width="stretch")
                    p_vals = result.ljung_box["p-value"].dropna()
                    if (p_vals > 0.05).all():
                        st.success("✅ No significant autocorrelation detected in residuals (all p > 0.05). Residuals resemble white noise.")
                    else:
                        st.warning("⚠️ Significant autocorrelation detected at one or more lags (p <= 0.05). Model may have uncaptured dynamics.")
                else:
                    st.info("Ljung-Box test could not be computed for this specification.")

            with d_col2:
                st.markdown("#### Jarque-Bera Test (Residual Normality)")
                jb_stat = result.normality.get("jb_stat")
                jb_pval = result.normality.get("pvalue")
                if jb_stat is not None and not np.isnan(jb_stat):
                    st.metric("JB Statistic", f"{jb_stat:.4f}")
                    st.metric("JB p-value", f"{jb_pval:.4f}")
                    if jb_pval > 0.05:
                        st.success("✅ Residuals conform to Gaussian normality (fail to reject H0, p > 0.05).")
                    else:
                        st.info("ℹ️ Residuals exhibit non-normal kurtosis or skew (p <= 0.05). Consider variance stabilization.")
                else:
                    st.info("Normality test sample size is insufficient.")
        else:
            st.info("No residuals available for baseline model.")

    # -------------------------------------------------------------
    # TAB 3: MODEL-SPECIFIC PLOTS
    # -------------------------------------------------------------
    with t_plots:
        st.subheader("Model-Specific Architectural Diagnostics")

        if selected_family in ["ar", "ma", "arma", "arima", "sarima", "sarimax"]:
            # 1. Identification: ACF/PACF of differenced series
            st.markdown("##### 1. Identification: Differenced Series Autocorrelation")
            d_val = spec.get("d", 0)
            D_val = spec.get("D", 0)
            m_val = spec.get("m", 12)
            diff_ser, _ = transform_series(
                model_base_series,
                diff_order=d_val,
                seasonal_diff_order=D_val,
                seasonal_period=m_val if D_val > 0 else None,
            )
            c_res = compute_acf_pacf(diff_ser, nlags=min(25, len(diff_ser) // 2 - 1))
            fig_comb = plot_acf_pacf_combined(
                c_res["lags"], c_res["acf"], c_res["pacf"], c_res["conf_bound"]
            )
            st.plotly_chart(fig_comb, width="stretch", key=f"plot_acf_pacf_ident_{selected_family}_{result.key}")
            st.caption(
                f"Contiguous suggestions on d={d_val}, D={D_val}: Suggested p={sugg_orders['suggested_p']}, "
                f"Suggested q={sugg_orders['suggested_q']} | Active model fitted: p={spec.get('p')}, q={spec.get('q')}"
            )

            # 2. Inverse Roots
            st.markdown("---")
            st.markdown("##### 2. Inverse AR & MA Roots (Stability & Invertibility)")
            ar_r = result.extras.get("ar_roots")
            ma_r = result.extras.get("ma_roots")
            fig_roots = plot_inverse_roots(ar_r, ma_r)
            st.plotly_chart(fig_roots, width="stretch", key=f"plot_inverse_roots_{selected_family}_{result.key}")

            ar_stable = True
            if ar_r is not None and len(ar_r) > 0:
                inv_ar = 1.0 / ar_r[ar_r != 0]
                if np.any(np.abs(inv_ar) >= 1.0):
                    ar_stable = False
            ma_invertible = True
            if ma_r is not None and len(ma_r) > 0:
                inv_ma = 1.0 / ma_r[ma_r != 0]
                if np.any(np.abs(inv_ma) >= 1.0):
                    ma_invertible = False

            if ar_stable and ma_invertible:
                st.success("✅ All inverse AR roots and MA roots lie strictly inside the unit circle (|1/r| < 1). The process is stable and invertible.")
            else:
                st.warning("⚠️ One or more inverse roots lie on or outside the unit circle (|1/r| >= 1). Process may be non-stationary or non-invertible.")

            # 3. Theoretical vs Empirical ACF/PACF (only if d==0 and D==0)
            st.markdown("---")
            st.markdown("##### 3. Theoretical vs. Empirical Correlation")
            if d_val == 0 and D_val == 0 and spec.get("P", 0) == 0 and spec.get("Q", 0) == 0:
                red_ar = result.extras.get("reduced_ar", [1.0])
                red_ma = result.extras.get("reduced_ma", [1.0])
                theo = theoretical_acf_pacf(red_ar, red_ma, nlags=min(20, len(c_res["lags"]) - 1))
                if theo is not None:
                    t_acf, t_pacf = theo
                    fig_theo = plot_theoretical_vs_empirical(
                        c_res["lags"], c_res["acf"], c_res["pacf"], t_acf, t_pacf, c_res["conf_bound"]
                    )
                    st.plotly_chart(fig_theo, width="stretch", key=f"plot_theoretical_vs_empirical_{selected_family}_{result.key}")
                else:
                    st.info("Theoretical ACF/PACF cannot be calculated for non-stationary AR polynomial.")
            else:
                st.info("ℹ️ Theoretical ACF/PACF overlay is displayed only for stationary, non-differenced processes (d=0, D=0).")

            # 4. Exogenous Regressor Coefficients (SARIMAX)
            if selected_family == "sarimax" and "exog_coefs" in result.extras:
                st.markdown("---")
                st.markdown("##### 4. Exogenous Regressor Coefficients")
                ex_coefs = result.extras["exog_coefs"]
                if ex_coefs:
                    exog_df_plot = pd.DataFrame(
                        {"Feature": list(ex_coefs.keys()), "Coefficient": list(ex_coefs.values())}
                    )
                    st.bar_chart(exog_df_plot.set_index("Feature"))

        elif selected_family == "holt_winters":
            fig_hw = plot_hw_components(result.extras, result.train_index)
            st.plotly_chart(fig_hw, width="stretch", key=f"plot_hw_components_{selected_family}_{result.key}")

            st.markdown("##### Smoothing Hyperparameters")
            sm = result.extras.get("smoothing", {})
            sm1, sm2, sm3, sm4 = st.columns(4)
            sm1.metric("Alpha (Level)", f"{sm['alpha']:.4f}" if sm.get("alpha") is not None else "N/A")
            sm2.metric("Beta (Trend)", f"{sm['beta']:.4f}" if sm.get("beta") is not None else "N/A")
            sm3.metric("Gamma (Season)", f"{sm['gamma']:.4f}" if sm.get("gamma") is not None else "N/A")
            sm4.metric("Phi (Damping)", f"{sm['phi']:.4f}" if sm.get("phi") is not None else "N/A")
            st.caption(f"Prediction interval methodology: *{result.notes}*")

        elif selected_family in ["naive", "seasonal_naive", "drift"]:
            st.info(f"**Baseline Architecture**: {result.summary_text}\n\n**Interval calculation**: {result.notes}")
            if selected_family == "drift":
                st.metric("Estimated Trajectory Slope", f"{result.extras.get('slope', 0.0):.4f}")

    # -------------------------------------------------------------
    # TAB 4: PARAMETERS & SUMMARY
    # -------------------------------------------------------------
    with t_summary:
        st.subheader("Model Parameter Estimates & Statistics")
        if result.params_table is not None and not result.params_table.empty:
            st.dataframe(result.params_table, width="stretch")
        else:
            st.caption("No parametric table for this baseline specification.")

        st.markdown("##### Statsmodels Summary Output")
        st.code(result.summary_text, language="text")

    # -------------------------------------------------------------
    # TAB 5: AUTOMATED ORDER SEARCH (ARIMA & SARIMA ONLY)
    # -------------------------------------------------------------
    with t_grid:
        if selected_family not in ["arima", "sarima"]:
            st.info("ℹ️ Automated Order Grid Search is available for ARIMA and SARIMA models.")
        else:
            st.subheader(f"Automated Order Grid Search ({selected_family.upper()})")
            st.caption(
                "Evaluates candidate specs sorted by model complexity and finds the optimal AIC/BIC specification. "
                "Note: AIC/BIC values are strictly comparable only across models that share identical differencing orders (d and D)."
            )

            g_c1, g_c2, g_c3 = st.columns(3)
            with g_c1:
                p_max = st.slider("Max AR order (p_max):", min_value=0, max_value=5, value=3, key="grid_p_max")
                q_max = st.slider("Max MA order (q_max):", min_value=0, max_value=5, value=3, key="grid_q_max")
            with g_c2:
                grid_d = spec.get("d", suggested_d)
                st.caption(f"Differencing order: **d = {grid_d}** (matches active form)")
                if selected_family == "sarima":
                    P_max = st.slider("Max Seasonal AR (P_max):", min_value=0, max_value=2, value=1, key="grid_P_max")
                    Q_max = st.slider("Max Seasonal MA (Q_max):", min_value=0, max_value=2, value=1, key="grid_Q_max")
                    grid_D = spec.get("D", suggested_D)
                    st.caption(f"Seasonal differencing order: **D = {grid_D}**")
            with g_c3:
                grid_crit = st.selectbox("Optimizing Criterion:", options=["AIC", "BIC"], index=0, key="grid_crit_select")
                max_models = st.number_input("Max Models to Evaluate:", min_value=10, max_value=100, value=40, step=10, key="grid_max_models")

            if st.button("Run grid search", help="Fit all candidate specifications on training data.", width="stretch"):
                seasonal_dict = (
                    {"P_range": list(range(P_max + 1)), "D": grid_D, "Q_range": list(range(Q_max + 1)), "m": active_m}
                    if selected_family == "sarima"
                    else None
                )

                with st.spinner("Executing grid search across candidate orders..."):
                    grid_df, best_info = cached_grid_search(
                        y=model_base_series,
                        p_range=list(range(p_max + 1)),
                        d=grid_d,
                        q_range=list(range(q_max + 1)),
                        seasonal=seasonal_dict,
                        test_size=holdout_size,
                        max_models=max_models,
                        criterion=grid_crit.lower(),
                    )

                st.session_state["last_grid_df"] = grid_df
                st.session_state["last_grid_best"] = best_info

            # Display last search results if present
            grid_df = st.session_state.get("last_grid_df")
            best_info = st.session_state.get("last_grid_best")
            if grid_df is not None and best_info is not None:
                st.success(f"Best specification by {best_info['criterion'].upper()}: **{best_info['best_spec_str']}**")
                if best_info.get("truncated"):
                    st.info(f"Grid search was truncated to {best_info['total_evaluated']} models (out of {best_info['total_grid_size']} possible combinations).")

                fig_grid = plot_grid_results(grid_df, top_n=15)
                st.plotly_chart(fig_grid, width="stretch", key=f"plot_grid_results_{selected_family}")

                st.dataframe(grid_df.drop(columns=["spec_dict"]).head(20), width="stretch")

                if st.button("Use best order", help="Inject best hyperparameters into form widgets.", width="stretch"):
                    best_s = best_info["best_spec"]
                    pending = {}
                    if selected_family == "arima":
                        pending["form_arima_p"] = int(best_s.get("p", 1))
                        pending["form_arima_d"] = int(best_s.get("d", 0))
                        pending["form_arima_q"] = int(best_s.get("q", 1))
                    elif selected_family in ("sarima", "sarimax"):
                        pending["form_s_p"] = int(best_s.get("p", 1))
                        pending["form_s_d"] = int(best_s.get("d", 0))
                        pending["form_s_q"] = int(best_s.get("q", 1))
                        pending["form_s_P"] = int(best_s.get("P", 0))
                        pending["form_s_D"] = int(best_s.get("D", 0))
                        pending["form_s_Q"] = int(best_s.get("Q", 1))
                    st.session_state["pending_spec"] = pending
                    st.toast(f"Staged best spec {best_info['best_spec_str']} for update!", icon="✨")
                    st.rerun()

    # -------------------------------------------------------------
    # MULTI-MODEL BENCHMARK EXPANDER
    # -------------------------------------------------------------
    st.markdown("---")
    with st.expander("Fit all models", expanded=False):
        st.markdown(
            "Quickly estimate all supported model families using standard suggested configurations, "
            "store their results in session state, and compare their holdout performance."
        )

        if st.button("Fit all models", width="stretch"):
            progress_bar = st.progress(0.0)
            fitted_all = fit_all_default_models(
                model_base_series=model_base_series,
                holdout_size=holdout_size,
                m=int(active_m),
                mase_m=st.session_state.get("model_mase_m_val", mase_m),
                transform_info=transform_info,
                exog_df=exog_df,
                progress_callback=lambda p: progress_bar.progress(p),
            )
            if fitted_all:
                results_list = [
                    {
                        "Model": r.label,
                        "Converged": r.converged,
                        "Test RMSE": r.metrics_test["rmse"],
                        "MAPE": r.metrics_test["mape"],
                        "AIC": (r.information_criteria or {}).get("aic"),
                        "BIC": (r.information_criteria or {}).get("bic"),
                    }
                    for r in fitted_all
                ]
                bench_df = pd.DataFrame(results_list).sort_values(by="Test RMSE", ascending=True).reset_index(drop=True)
                st.session_state["benchmark_df"] = bench_df

        if "benchmark_df" in st.session_state:
            st.markdown("##### 🏆 Holdout Performance Leaderboard (Sorted by RMSE)")
            st.dataframe(st.session_state["benchmark_df"], width="stretch")
