"""
Unit and integration tests for ATSA Phase 2 diagnostics and plot builders.
Runnable with:
    python tests/test_diagnostics.py
or
    pytest tests/test_diagnostics.py
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from modules.diagnostics import (
    adf_test,
    compute_acf_pacf,
    decompose,
    infer_period_from_frequency,
    kpss_test,
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
    plot_residual_diagnostics,
    plot_rolling_stats,
    plot_seasonal_box,
    plot_stem_correlation,
    plot_time_series,
)


def test_white_noise_stationarity():
    """White noise (seed 42, n=300): ADF says stationary, verdict is 'Stationary'."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=300, freq="D")
    wn = pd.Series(rng.normal(0, 1, 300), index=dates, name="white_noise")

    res_adf = adf_test(wn)
    res_kpss = kpss_test(wn)
    verdict = stationarity_verdict(res_adf, res_kpss)

    print("\n--- 1. White Noise Stationarity Test ---")
    print(f"ADF Statistic: {res_adf['statistic']:.4f}, p-value: {res_adf['pvalue']:.4f}")
    print(f"KPSS Statistic: {res_kpss['statistic']:.4f}, p-value: {res_kpss['pvalue']:.4f}")
    print(f"Verdict: {verdict['verdict']}")

    assert res_adf["is_stationary"] is True, "ADF should reject unit root on white noise."
    assert verdict["verdict"] == "Stationary", f"Expected verdict 'Stationary', got '{verdict['verdict']}'."
    print("[PASS] White noise stationarity assertion passed.")


def test_random_walk():
    """Random walk: ADF fails to reject; after diff_order=1 it becomes stationary."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=300, freq="D")
    rw = pd.Series(np.cumsum(rng.normal(0, 1, 300)), index=dates, name="random_walk")

    res_rw_adf = adf_test(rw)
    print("\n--- 2. Random Walk Test ---")
    print(f"Random Walk ADF p-value: {res_rw_adf['pvalue']:.4f} (is_stationary={res_rw_adf['is_stationary']})")
    assert res_rw_adf["is_stationary"] is False, "ADF should fail to reject unit root on a random walk."

    rw_diff, info = transform_series(rw, diff_order=1)
    res_diff_adf = adf_test(rw_diff)
    print(f"Differenced ADF p-value: {res_diff_adf['pvalue']:.4e} (is_stationary={res_diff_adf['is_stationary']})")
    assert res_diff_adf["is_stationary"] is True, "Differenced random walk should be stationary."
    print("[PASS] Random walk unit root and differencing assertions passed.")


def test_decomposition():
    """Synthetic monthly series with trend + 12-period seasonality."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2010-01-01", periods=48, freq="MS")
    trend = np.linspace(10, 50, 48)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(48) / 12)
    noise = rng.normal(0, 0.5, 48)
    synth = pd.Series(trend + seasonal + noise, index=dates, name="synthetic_monthly")

    print("\n--- 3. Time Series Decomposition Test ---")
    decomp = decompose(synth, model="additive", period=None, method="classical")

    print(f"Inferred Period: {decomp['period']}")
    print(f"Seasonal Strength (F_S): {decomp['seasonal_strength']:.4f}")
    print(f"Trend Strength (F_T): {decomp['trend_strength']:.4f}")

    assert decomp["period"] == 12, f"Expected auto period 12, got {decomp['period']}"
    assert decomp["seasonal_strength"] > 0.8, f"Expected seasonal strength > 0.8, got {decomp['seasonal_strength']}"

    # Additive components sum back to observed
    mask = decomp["trend"].notna() & decomp["resid"].notna()
    sum_components = decomp["trend"][mask] + decomp["seasonal"][mask] + decomp["resid"][mask]
    assert np.allclose(decomp["observed"][mask], sum_components), "Additive components should sum to observed."
    print("[PASS] Additive components match observed series exactly.")

    # Multiplicative on series with zero returns error message
    synth_with_zero = synth.copy()
    synth_with_zero.iloc[10] = 0.0
    decomp_zero = decompose(synth_with_zero, model="multiplicative", period=12)
    assert decomp_zero.get("error") is not None, "Expected error on multiplicative with zero value."
    assert "positive" in decomp_zero["error"].lower(), f"Unexpected error message: {decomp_zero['error']}"
    print(f"[PASS] Multiplicative zero guard triggered as expected: '{decomp_zero['error']}'")


def test_transform_series():
    """transform_series log + diff(1) yields len(n-1) and info dict lists both steps in order."""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    pos_data = pd.Series([10.0, 14.0, 18.0, 25.0, 31.0, 42.0, 50.0, 65.0, 80.0, 100.0], index=dates)

    tf_s, info = transform_series(pos_data, log=True, diff_order=1)
    print("\n--- 4. Transformation Pipeline Test ---")
    print(f"Original length: {len(pos_data)}, Transformed length: {len(tf_s)}")
    print(f"Transformation steps recorded: {info['steps']}")

    assert len(tf_s) == len(pos_data) - 1, f"Expected len {len(pos_data) - 1}, got {len(tf_s)}"
    assert info["steps"] == ["log", "diff_1"], f"Expected steps ['log', 'diff_1'], got {info['steps']}"
    print("[PASS] Transformation pipeline steps and length assertions passed.")


def test_acf_nlags_capping():
    """ACF nlags capping on a short series (n=20)."""
    rng = np.random.default_rng(42)
    short_s = pd.Series(rng.normal(0, 1, 20), index=pd.date_range("2020-01-01", periods=20, freq="D"))

    res = compute_acf_pacf(short_s, nlags=50)
    print("\n--- 5. ACF nlags Capping Test ---")
    print(f"Requested nlags=50 on n=20 series -> Effective nlags capped at: {res['nlags']}")

    assert res["nlags"] == 9, f"Expected capped nlags to be 9 (20//2 - 1), got {res['nlags']}"
    assert len(res["acf"]) == 10, f"Expected 10 ACF points (lags 0..9), got {len(res['acf'])}"
    print("[PASS] ACF lag capping assertion passed.")


def test_plots_smoke():
    """Smoke test that every plots.py builder returns a go.Figure."""
    print("\n--- 6. Plot Builders Smoke Test ---")
    dates = pd.date_range("2020-01-01", periods=36, freq="MS")
    s = pd.Series(np.linspace(10, 50, 36) + np.sin(np.arange(36)), index=dates, name="sales")

    fig1 = plot_time_series(s)
    assert isinstance(fig1, go.Figure), "plot_time_series failed to return go.Figure"

    r_df = rolling_stats(s, window=6)
    fig2 = plot_rolling_stats(s, r_df, window=6)
    assert isinstance(fig2, go.Figure), "plot_rolling_stats failed to return go.Figure"

    acf_pacf_res = compute_acf_pacf(s, nlags=12)
    fig3 = plot_stem_correlation(acf_pacf_res["lags"], acf_pacf_res["acf"], acf_pacf_res["conf_bound"])
    assert isinstance(fig3, go.Figure), "plot_stem_correlation failed to return go.Figure"

    fig4 = plot_acf_pacf_combined(acf_pacf_res["lags"], acf_pacf_res["acf"], acf_pacf_res["pacf"], acf_pacf_res["conf_bound"])
    assert isinstance(fig4, go.Figure), "plot_acf_pacf_combined failed to return go.Figure"

    decomp = decompose(s, model="additive", period=12)
    fig5 = plot_decomposition(decomp)
    assert isinstance(fig5, go.Figure), "plot_decomposition failed to return go.Figure"

    seas_df = seasonal_subseries(s, period=12)
    fig6 = plot_seasonal_box(seas_df, period_name="Monthly")
    assert isinstance(fig6, go.Figure), "plot_seasonal_box failed to return go.Figure"

    fig7 = plot_lag(s, lag=1)
    assert isinstance(fig7, go.Figure), "plot_lag failed to return go.Figure"

    fig8 = plot_distribution_and_qq(s)
    assert isinstance(fig8, go.Figure), "plot_distribution_and_qq failed to return go.Figure"

    fig9 = plot_residual_diagnostics(decomp["resid"])
    assert isinstance(fig9, go.Figure), "plot_residual_diagnostics failed to return go.Figure"

    print("[PASS] All 9 plot builders returned go.Figure successfully.")


def test_suggest_orders_ar2():
    """A3: Simulated AR(2) (phi = 0.6, 0.3, n=1000, seed 42) must give suggested_p == 2."""
    print("\n--- 7. AR(2) Order Suggestion Test (A3) ---")
    rng = np.random.default_rng(42)
    n = 1000
    eps = rng.normal(0, 1, n)
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = 0.6 * y[t - 1] + 0.3 * y[t - 2] + eps[t]

    dates = pd.date_range("2000-01-01", periods=n, freq="D")
    s = pd.Series(y, index=dates, name="ar2_sim")

    res = compute_acf_pacf(s, nlags=15)
    sugg = suggest_orders(res["acf"], res["pacf"], res["conf_bound"])

    print(f"Computed suggested_p: {sugg['suggested_p']}, suggested_q: {sugg['suggested_q']}")
    assert sugg["suggested_p"] == 2, f"Expected suggested_p == 2, got {sugg['suggested_p']}"
    print("[PASS] A3 unit test passed: AR(2) simulated process gives suggested_p == 2.")


def test_decompose_yearly_series():
    """A4: Yearly series returns None for infer_period_from_frequency and decompose returns error string."""
    print("\n--- 8. Yearly Series Decompose Test (A4) ---")
    dates = pd.date_range("1990-01-01", periods=30, freq="YS")
    s = pd.Series(np.linspace(10, 50, 30), index=dates, name="yearly_series")

    inferred_p = infer_period_from_frequency(s)
    print(f"infer_period_from_frequency for yearly series: {inferred_p}")
    assert inferred_p is None, f"Expected None for yearly frequency, got {inferred_p}"

    decomp = decompose(s)
    print(f"Decomposition result error: {decomp.get('error')}")
    assert decomp.get("error") is not None, "Expected error message when period cannot be inferred"
    assert "explicit seasonal period" in decomp["error"], f"Unexpected error message: {decomp['error']}"
    print("[PASS] A4 unit test passed: Yearly series returns None and decompose returns error string.")


def test_infer_period_from_frequency_exact_tokens():
    """0.1: infer_period_from_frequency exact token test: MS->12, min->None, YS->None."""
    print("\n--- 9. Infer Period Exact Tokens Test (0.1) ---")
    assert infer_period_from_frequency("MS") == 12
    assert infer_period_from_frequency("min") is None
    assert infer_period_from_frequency("YS") is None

    # Also test via pd.Series with DatetimeIndex
    s_ms = pd.Series(range(10), index=pd.date_range("2020-01-01", periods=10, freq="MS"))
    assert infer_period_from_frequency(s_ms) == 12

    s_ys = pd.Series(range(10), index=pd.date_range("2020-01-01", periods=10, freq="YS"))
    assert infer_period_from_frequency(s_ys) is None

    s_min = pd.Series(range(10), index=pd.date_range("2020-01-01", periods=10, freq="min"))
    assert infer_period_from_frequency(s_min) is None
    print("[PASS] 0.1 exact token assertions passed: MS->12, min->None, YS->None.")


def test_auto_differencing_suggestion_verdict():
    """Unit test: stationary series gives d=0; random walk gives d=1."""
    print("\n--- 10. Auto Differencing Suggestion Verdict Test ---")
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=300, freq="D")

    # 1. Stationary white noise
    s_stat = pd.Series(rng.normal(0, 1, 300), index=dates)
    adf_stat = adf_test(s_stat)
    kpss_stat = kpss_test(s_stat)
    v_stat = stationarity_verdict(adf_stat, kpss_stat)["verdict"]
    d_stat = 0 if v_stat == "Stationary" else 1
    assert d_stat == 0, f"Expected d=0 for stationary series, got d={d_stat} (verdict: {v_stat})"

    # 2. Non-stationary random walk
    s_rw = pd.Series(np.cumsum(rng.normal(0, 1, 300)), index=dates)
    adf_rw = adf_test(s_rw)
    kpss_rw = kpss_test(s_rw)
    v_rw = stationarity_verdict(adf_rw, kpss_rw)["verdict"]
    d_rw = 0 if v_rw == "Stationary" else 1
    assert d_rw == 1, f"Expected d=1 for random walk, got d={d_rw} (verdict: {v_rw})"

    print(f"[PASS] Auto differencing suggestion: stationary d={d_stat}, random walk d={d_rw}.")


if __name__ == "__main__":
    print("==================================================")
    print("Starting ATSA Diagnostics & Visualization Tests...")
    print("==================================================")
    test_white_noise_stationarity()
    test_random_walk()
    test_decomposition()
    test_transform_series()
    test_acf_nlags_capping()
    test_plots_smoke()
    test_suggest_orders_ar2()
    test_decompose_yearly_series()
    test_infer_period_from_frequency_exact_tokens()
    test_auto_differencing_suggestion_verdict()
    print("\n==================================================")
    print("ALL TESTS COMPLETED AND VERIFIED SUCCESSFULLY!")
    print("==================================================")


