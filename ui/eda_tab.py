"""
EDA and Diagnostics UI Tab for ATSA.

Provides comprehensive diagnostic inspections:
- Tab 1: Stationarity testing (ADF, KPSS, joint verdict, rolling statistics).
- Tab 2: Autocorrelation (ACF, PACF stem plots, contiguous order suggestions, seasonal hints).
- Tab 3: Time series decomposition (Classical, STL, Hyndman trend & seasonality strength).
- Tab 4: Transformations & Differencing (Log, Box-Cox, Diff, Seasonal Diff, model_base_series handoff).
- Tab 5: Other EDA (Seasonal distribution box plot, Lag plot, Histogram + Q-Q plot, Ljung-Box test).
"""

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import streamlit as st

from modules.diagnostics import (
    adf_test,
    compute_acf_pacf,
    decompose,
    infer_period_from_frequency,
    kpss_test,
    ljung_box_test,
    rolling_stats,
    seasonal_subseries,
    stationarity_verdict,
    suggest_orders,
    transform_series,
)
from modules.plots import (
    plot_acf_pacf_combined,
    plot_decomposition,
    plot_distribution_and_qq,
    plot_lag,
    plot_rolling_stats,
    plot_seasonal_box,
    plot_stem_correlation,
    plot_time_series,
)
from ui.model_common import reset_model_widgets


# UI Layer Cached Wrappers for Pure Functions
@st.cache_data
def cached_adf_test(series: pd.Series, regression: str, autolag: str) -> Dict[str, Any]:
    return adf_test(series, regression=regression, autolag=autolag)


@st.cache_data
def cached_kpss_test(series: pd.Series, regression: str, nlags: str) -> Dict[str, Any]:
    return kpss_test(series, regression=regression, nlags=nlags)


@st.cache_data
def cached_acf_pacf(series: pd.Series, nlags: int) -> Dict[str, Any]:
    return compute_acf_pacf(series, nlags=nlags)


@st.cache_data
def cached_decomposition(
    series: pd.Series, model: str, period: Optional[int], method: str
) -> Dict[str, Any]:
    return decompose(series, model=model, period=period, method=method)


def render_eda_tab() -> None:
    """Render the full Phase 2 Exploratory Data Analysis & Diagnostics view."""
    series = st.session_state.get("ts_series")

    if series is None or len(series) == 0:
        st.info("ℹ️ No time series dataset is loaded yet. Please upload a dataset or click **Load Demo Data** in the 📥 Data tab.")
        return

    # Check for constant series or invalid data
    if series.nunique() <= 1:
        st.warning("⚠️ The loaded time series is constant (all values are identical). Statistical diagnostic tests cannot be computed.")
        return

    st.markdown("### 🔍 Exploratory Data Analysis & Diagnostics")
    st.caption(f"Analyzing `{series.name or 'Target Series'}` ({len(series)} observations)")

    eda_t1, eda_t2, eda_t3, eda_t4, eda_t5 = st.tabs(
        [
            "⚡ Stationarity",
            "📊 ACF / PACF",
            "🧩 Decomposition",
            "🔄 Transformations & Differencing",
            "📈 Other EDA",
        ]
    )

    # -------------------------------------------------------------
    # TAB 1: STATIONARITY
    # -------------------------------------------------------------
    with eda_t1:
        st.subheader("Stationarity Tests (ADF & KPSS)")
        st.markdown(
            "Stationarity is assessed using both the **Augmented Dickey-Fuller (ADF)** test "
            "(Null: unit root non-stationary) and the **KPSS** test (Null: level/trend stationary)."
        )

        stat_col1, stat_col2 = st.columns(2)
        with stat_col1:
            adf_reg = st.selectbox(
                "ADF Regression Specification:",
                options=["c", "ct", "ctt", "n"],
                format_func=lambda x: {
                    "c": "c - Constant (drift)",
                    "ct": "ct - Constant & Linear Trend",
                    "ctt": "ctt - Constant, Linear & Quadratic Trend",
                    "n": "n - No Constant / No Trend",
                }.get(x, x),
                index=0,
                key="adf_reg_select",
            )
            adf_autolag = st.selectbox(
                "ADF Lag Criterion:",
                options=["AIC", "BIC", "t-stat"],
                index=0,
                key="adf_autolag_select",
            )

        with stat_col2:
            kpss_reg = st.selectbox(
                "KPSS Regression Specification:",
                options=["c", "ct"],
                format_func=lambda x: {"c": "c - Level Stationarity", "ct": "ct - Trend Stationarity"}.get(x, x),
                index=0,
                key="kpss_reg_select",
            )
            kpss_lags = st.selectbox(
                "KPSS Lag Selection:",
                options=["auto", "legacy"],
                index=0,
                key="kpss_lags_select",
            )

        # Run tests with error handling
        try:
            adf_res = cached_adf_test(series, regression=adf_reg, autolag=adf_autolag)
            kpss_res = cached_kpss_test(series, regression=kpss_reg, nlags=kpss_lags)
            verdict = stationarity_verdict(adf_res, kpss_res)

            # Joint Verdict Banner
            if verdict["status"] == "success":
                st.success(f"**Joint Verdict: {verdict['verdict']}** — {verdict['explanation']}")
            elif verdict["status"] == "warning":
                st.warning(f"**Joint Verdict: {verdict['verdict']}** — {verdict['explanation']}")
            else:
                st.info(f"**Joint Verdict: {verdict['verdict']}** — {verdict['explanation']}")

            # Comparison Results Table
            t_col1, t_col2 = st.columns(2)
            with t_col1:
                st.markdown("#### Augmented Dickey-Fuller (ADF)")
                adf_summary = {
                    "Metric": ["Test Statistic", "p-value", "Used Lags", "Observations", "Stationary at 5%?"],
                    "Value": [
                        f"{adf_res['statistic']:.4f}",
                        f"{adf_res['pvalue']:.4f}",
                        str(adf_res["used_lag"]),
                        str(adf_res["nobs"]),
                        "Yes (Reject H0)" if adf_res["is_stationary"] else "No (Fail to Reject H0)",
                    ],
                }
                st.table(pd.DataFrame(adf_summary))
                st.caption(
                    f"Critical Values: 1%: {adf_res['critical_values']['1%']:.3f} | "
                    f"5%: {adf_res['critical_values']['5%']:.3f} | "
                    f"10%: {adf_res['critical_values']['10%']:.3f}"
                )

            with t_col2:
                st.markdown("#### KPSS Test")
                kpss_summary = {
                    "Metric": ["Test Statistic", "p-value", "Used Lags", "Observations", "Stationary at 5%?"],
                    "Value": [
                        f"{kpss_res['statistic']:.4f}",
                        f"{kpss_res['pvalue']:.4f}",
                        str(kpss_res["used_lag"]),
                        str(kpss_res["nobs"]),
                        "Yes (Fail to Reject H0)" if kpss_res["is_stationary"] else "No (Reject H0)",
                    ],
                }
                st.table(pd.DataFrame(kpss_summary))
                st.caption(
                    f"Critical Values: 1%: {kpss_res['critical_values']['1%']:.3f} | "
                    f"5%: {kpss_res['critical_values']['5%']:.3f} | "
                    f"10%: {kpss_res['critical_values']['10%']:.3f}"
                )

        except Exception as exc:
            st.error(f"Error computing stationarity tests: {exc}")

        # Rolling Statistics Section
        st.markdown("---")
        st.subheader("Rolling Mean & Standard Deviation")
        max_window = max(3, min(120, len(series) // 2))
        default_window = min(12, max_window)

        roll_window = st.slider(
            "Rolling Window Size:",
            min_value=2,
            max_value=max_window,
            value=default_window,
            help="Window length for calculating rolling statistics.",
            key="rolling_window_slider",
        )

        r_df = rolling_stats(series, window=roll_window)
        roll_fig = plot_rolling_stats(series, r_df, window=roll_window)
        st.plotly_chart(roll_fig, width="stretch")

    # -------------------------------------------------------------
    # TAB 2: ACF / PACF
    # -------------------------------------------------------------
    with eda_t2:
        st.subheader("Autocorrelation (ACF) & Partial Autocorrelation (PACF)")
        st.markdown(
            "The ACF measures total correlation between $y_t$ and $y_{t-k}$, while the PACF "
            "measures direct correlation after controlling for intermediate lags."
        )

        max_allowed_lags = max(2, len(series) // 2 - 1)
        default_lags = min(40, max_allowed_lags)

        selected_nlags = st.slider(
            "Number of Lags to Compute:",
            min_value=2,
            max_value=min(100, max_allowed_lags),
            value=default_lags,
            key="acf_nlags_slider",
        )

        inferred_m = infer_period_from_frequency(series)

        try:
            acf_pacf_res = cached_acf_pacf(series, nlags=selected_nlags)
            lags = acf_pacf_res["lags"]
            acf_vals = acf_pacf_res["acf"]
            pacf_vals = acf_pacf_res["pacf"]
            conf_bound = acf_pacf_res["conf_bound"]

            # Side-by-side stem plots
            col_acf, col_pacf = st.columns(2)
            with col_acf:
                fig_acf = plot_stem_correlation(
                    lags,
                    acf_vals,
                    conf_bound,
                    title=f"Autocorrelation Function (ACF, N={acf_pacf_res['nobs']})",
                    yaxis_title="ACF",
                    bar_color="#2563eb",
                )
                st.plotly_chart(fig_acf, width="stretch")

            with col_pacf:
                fig_pacf = plot_stem_correlation(
                    lags,
                    pacf_vals,
                    conf_bound,
                    title=f"Partial Autocorrelation (PACF, Yule-Walker)",
                    yaxis_title="PACF",
                    bar_color="#7c3aed",
                )
                st.plotly_chart(fig_pacf, width="stretch")

            # Order suggestions hint box
            suggestions = suggest_orders(acf_vals, pacf_vals, conf_bound, m=inferred_m)
            seasonal_text = f"\n- **Seasonal Hint (m={inferred_m})**: {suggestions['seasonal_hint']}" if suggestions.get("seasonal_hint") else ""
            st.info(
                f"💡 **Suggested Model Order Hints (Leading Contiguous Lags)**:\n"
                f"- **Suggested AR order $p$**: `{suggestions['suggested_p']}` (Contiguous significant PACF lags starting at lag 1)\n"
                f"- **Suggested MA order $q$**: `{suggestions['suggested_q']}` (Contiguous significant ACF lags starting at lag 1)"
                f"{seasonal_text}\n\n"
                f"*{suggestions['caption']}*"
            )

        except Exception as exc:
            st.error(f"Error computing ACF / PACF: {exc}")

    # -------------------------------------------------------------
    # TAB 3: DECOMPOSITION
    # -------------------------------------------------------------
    with eda_t3:
        st.subheader("Time Series Decomposition & Strength Metrics")
        st.markdown(
            "Decomposes the series into **Observed = Trend + Seasonal + Residual** (Additive) "
            "or **Observed = Trend × Seasonal × Residual** (Multiplicative)."
        )

        d_col1, d_col2, d_col3 = st.columns(3)
        with d_col1:
            decomp_model = st.selectbox(
                "Decomposition Model:",
                options=["additive", "multiplicative"],
                format_func=lambda x: x.capitalize(),
                key="decomp_model_select",
            )

        with d_col2:
            decomp_method = st.selectbox(
                "Algorithm:",
                options=["classical", "stl"],
                format_func=lambda x: "Classical (Moving Average)" if x == "classical" else "STL (Loess)",
                key="decomp_method_select",
            )

        with d_col3:
            inferred_p = infer_period_from_frequency(series)
            use_manual_period = st.checkbox(
                "Specify Seasonal Period Manually",
                value=(inferred_p is None),
                key="decomp_manual_p_check",
            )
            if use_manual_period:
                decomp_period = st.number_input(
                    "Seasonal Period:",
                    min_value=2,
                    max_value=max(3, len(series) // 2),
                    value=int(inferred_p or 12),
                    step=1,
                    key="decomp_period_input",
                )
            else:
                decomp_period = int(inferred_p) if inferred_p else None
                st.caption(f"Auto-inferred period: **{decomp_period}**")

        try:
            decomp_res = cached_decomposition(
                series,
                model=decomp_model,
                period=decomp_period,
                method=decomp_method,
            )

            if decomp_res.get("error"):
                st.error(f"⚠️ {decomp_res['error']}")
            else:
                # Hyndman Strength Metrics
                s_col1, s_col2, s_col3 = st.columns(3)
                s_col1.metric("Trend Strength (F_T)", f"{decomp_res['trend_strength']:.3f}", help="Hyndman metric: 0 (no trend) to 1 (deterministic trend)")
                s_col2.metric("Seasonal Strength (F_S)", f"{decomp_res['seasonal_strength']:.3f}", help="Hyndman metric: 0 (no seasonality) to 1 (perfect seasonality)")
                s_col3.metric("Cycle Period", str(decomp_res["period"]))

                decomp_fig = plot_decomposition(decomp_res)
                st.plotly_chart(decomp_fig, width="stretch")

        except Exception as exc:
            st.error(f"Error executing decomposition: {exc}")

    # -------------------------------------------------------------
    # TAB 4: TRANSFORMATIONS & DIFFERENCING
    # -------------------------------------------------------------
    with eda_t4:
        st.subheader("Transformations & Differencing Pipeline")
        st.markdown(
            "Stabilize non-constant variance using **Log** or **Box-Cox**, and identify differencing orders "
            "($d, D$). In Phase 3, models receive the variance-stabilized **base series** and perform differencing internally."
        )

        active_model_series = st.session_state.get("model_series", series)
        active_transform_info = st.session_state.get("transform_info", {"steps": []})
        steps_display = active_transform_info.get("steps")
        steps_text = " -> ".join(steps_display) if steps_display else "None (Regularized Series)"

        s_d = active_transform_info.get("suggested_d", 0)
        s_D = active_transform_info.get("suggested_D", 0)
        s_m = active_transform_info.get("seasonal_period")

        st.info(
            f"🎯 **Active Modeling Setup**: Pipeline: `{steps_text}` | "
            f"**Models will difference internally using d={s_d} and D={s_D}**" + (f" (m={s_m})" if s_D > 0 else "")
        )

        t_ctrl1, t_ctrl2, t_ctrl3, t_ctrl4 = st.columns(4)

        with t_ctrl1:
            var_transform = st.selectbox(
                "Variance Transform:",
                options=["None", "Log (ln)", "Box-Cox"],
                index=0,
                key="tf_var_select",
            )

        with t_ctrl2:
            diff_d = st.selectbox(
                "Regular Differencing (d):",
                options=[0, 1, 2],
                index=0,
                key="tf_diff_select",
            )

        with t_ctrl3:
            seas_d = st.selectbox(
                "Seasonal Differencing (D):",
                options=[0, 1],
                index=0,
                key="tf_seas_diff_select",
            )

        with t_ctrl4:
            inferred_seas_m = infer_period_from_frequency(series) or 12
            seas_period_val = st.number_input(
                "Seasonal Period (m):",
                min_value=2,
                max_value=max(3, len(series) // 2),
                value=int(inferred_seas_m),
                key="tf_seas_period_input",
            )

        apply_log = var_transform == "Log (ln)"
        apply_boxcox = var_transform == "Box-Cox"

        try:
            # 1. Base series (variance transform ONLY)
            base_s, base_info = transform_series(
                series,
                log=apply_log,
                boxcox_tf=apply_boxcox,
                diff_order=0,
                seasonal_diff_order=0,
            )

            # 2. Fully transformed series (with differencing)
            transformed_s, tf_info = transform_series(
                series,
                log=apply_log,
                boxcox_tf=apply_boxcox,
                diff_order=diff_d,
                seasonal_diff_order=seas_d,
                seasonal_period=seas_period_val if seas_d > 0 else None,
            )

            # Compare Original vs Transformed
            st.markdown("#### Original vs. Transformed Series")
            c_comp1, c_comp2 = st.columns(2)
            with c_comp1:
                fig_orig = plot_time_series(series, title="Original Series", line_color="#64748b")
                st.plotly_chart(fig_orig, width="stretch")
            with c_comp2:
                fig_trans = plot_time_series(
                    transformed_s,
                    title=f"Fully Transformed Series ({len(transformed_s)} obs)",
                    line_color="#059669",
                )
                st.plotly_chart(fig_trans, width="stretch")

            # Stationarity re-check on Transformed Series
            st.markdown("#### Stationarity Verification on Transformed Series")
            t_adf = cached_adf_test(transformed_s, regression="c", autolag="AIC")
            t_kpss = cached_kpss_test(transformed_s, regression="c", nlags="auto")
            t_verdict = stationarity_verdict(t_adf, t_kpss)

            if t_verdict["status"] == "success":
                st.success(f"**Transformed Verdict: {t_verdict['verdict']}** — {t_verdict['explanation']}")
            else:
                st.warning(f"**Transformed Verdict: {t_verdict['verdict']}** — {t_verdict['explanation']}")

            # ACF / PACF on Transformed Series
            st.markdown("#### Transformed Series Autocorrelation")
            tf_max_lags = max(2, len(transformed_s) // 2 - 1)
            tf_acf_res = cached_acf_pacf(transformed_s, nlags=min(30, tf_max_lags))
            tf_combined_fig = plot_acf_pacf_combined(
                tf_acf_res["lags"],
                tf_acf_res["acf"],
                tf_acf_res["pacf"],
                tf_acf_res["conf_bound"],
            )
            st.plotly_chart(tf_combined_fig, width="stretch")

            # Action Buttons to set or reset model series
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("🚀 Use Transformed Series for Modeling", help="Save variance-stabilized series and suggested d, D for modeling.", width="stretch"):
                    tf_info_copy = dict(tf_info)
                    tf_info_copy["suggested_d"] = diff_d
                    tf_info_copy["suggested_D"] = seas_d
                    tf_info_copy["seasonal_period"] = seas_period_val if seas_d > 0 else None

                    st.session_state["model_series"] = transformed_s
                    st.session_state["model_base_series"] = base_s
                    st.session_state["transform_info"] = tf_info_copy
                    reset_model_widgets()
                    st.toast("Transformed series set as active modeling series!", icon="🚀")
                    st.rerun()

            with btn_col2:
                if st.button("↺ Reset to Original Regularized Series", help="Restore active modeling series to original regularized data.", width="stretch"):
                    st.session_state["model_series"] = series
                    st.session_state["model_base_series"] = series
                    st.session_state["transform_info"] = {
                        "steps": [],
                        "params": {},
                        "suggested_d": 0,
                        "suggested_D": 0,
                        "seasonal_period": None,
                    }
                    reset_model_widgets()
                    st.toast("Reset active modeling series to original data.", icon="↺")
                    st.rerun()

        except Exception as exc:
            st.error(f"Error applying transformations: {exc}")

    # -------------------------------------------------------------
    # TAB 5: OTHER EDA
    # -------------------------------------------------------------
    with eda_t5:
        st.subheader("Supplemental Exploratory Diagnostics")

        # 1. Seasonal Box Plot
        inferred_cycle_period = infer_period_from_frequency(series) or 12
        sub_period = st.number_input(
            "Cycle Period for Seasonal Box Plot:",
            min_value=2,
            max_value=max(3, len(series) // 2),
            value=int(inferred_cycle_period),
            key="sub_period_input",
        )
        try:
            seas_df = seasonal_subseries(series, period=int(sub_period))
            box_fig = plot_seasonal_box(seas_df, period_name=f"Period = {sub_period}")
            st.plotly_chart(box_fig, width="stretch")
        except Exception as exc:
            st.error(f"Error rendering seasonal box plot: {exc}")

        # 2. Lag Plot
        st.markdown("---")
        st.markdown("#### Lag Plot Analysis")
        selected_lag = st.slider("Lag Order (k):", min_value=1, max_value=min(24, len(series) // 2), value=1)
        lag_fig = plot_lag(series, lag=selected_lag)
        st.plotly_chart(lag_fig, width="stretch")

        # 3. Distribution & Q-Q Plot
        st.markdown("---")
        st.markdown("#### Distribution & Normality Inspection")
        dist_fig = plot_distribution_and_qq(series)
        st.plotly_chart(dist_fig, width="stretch")

        # 4. Ljung-Box Test
        st.markdown("---")
        st.markdown("#### Ljung-Box Test for Autocorrelation")
        st.caption("Null Hypothesis (H0): The data is independently distributed (no serial autocorrelation).")
        try:
            lb_df = ljung_box_test(series)
            st.dataframe(lb_df, width="stretch")
        except Exception as exc:
            st.error(f"Error calculating Ljung-Box test: {exc}")

        # 5. Descriptive Stats
        st.markdown("---")
        st.markdown("#### Full Summary Statistics")
        st.dataframe(series.describe().to_frame(name=series.name or "Value"), width="stretch")
