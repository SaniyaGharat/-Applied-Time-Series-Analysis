"""
Comprehensive Unit & Integration Test Suite for ATSA Phase 3 Model Zoo.

Verifies pure module functionality in modules/models.py and modules/metrics.py:
1. AR(2) parameter recovery, residual Ljung-Box test, and picklability.
2. MA(1) parameter recovery.
3. ARMA(1, 1) fit and metrics dictionary keys.
4. Random walk with drift: ARIMA(0,1,0) and drift positive slope.
5. Synthetic seasonal series: SARIMA, Holt-Winters (MAPE < 15%), and Seasonal Naive.
6. SARIMAX with external driver: success and safe failure when missing exog.
7. Variance transform round trips: Log and Box-Cox forecast scale and inverse exactness.
8. Multiplicative Holt-Winters zero guard failure handling.
9. Integration d > 0 burn-in residual trimming.
10. grid_search_arima sorting by AIC and max_models truncation.
11. fit_full_and_forecast horizon frequency continuation and bound ordering.
12. default_test_size holdout sizing logic.
13. metrics.py edge cases: zero-safe MAPE and finite MASE.

Run with:
    python tests/test_models.py
"""

import os
import pickle
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from statsmodels.tsa.arima.model import ARIMA

from modules.diagnostics import (
    inverse_variance_transform,
    theoretical_acf_pacf,
    transform_series,
)
from modules.metrics import accuracy_table, mae, mape, mase, rmse, smape
from modules.models import (
    ModelResult,
    default_test_size,
    fit_full_and_forecast,
    fit_model,
    grid_search_arima,
    list_model_families,
    make_model_key,
)


def test_ar2_recovery_and_picklability():
    """Simulated AR(2) (phi 0.6, 0.3, n=600, seed 42): recovers coefficients within 0.15, LB p > 0.05, picklable."""
    print("\n--- 1. Testing AR(2) Parameter Recovery & Picklability ---")
    rng = np.random.default_rng(42)
    n = 600
    eps = rng.normal(0, 1, n)
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = 0.6 * y[t - 1] + 0.3 * y[t - 2] + eps[t]

    dates = pd.date_range("2000-01-01", periods=n, freq="MS")
    s = pd.Series(y, index=dates, name="ar2")

    res = fit_model(s, {"family": "ar", "p": 2}, test_size=60)

    assert res.converged is True, f"AR(2) model failed to converge. Warnings: {res.warnings}"
    assert res.params_table is not None, "Params table is None"

    params = dict(zip(res.params_table["parameter"], res.params_table["estimate"]))
    phi1_est = params.get("ar.L1")
    phi2_est = params.get("ar.L2")

    print(f"Recovered phi1: {phi1_est:.4f} (True: 0.6), phi2: {phi2_est:.4f} (True: 0.3)")
    assert abs(phi1_est - 0.6) < 0.15, f"AR(1) parameter error too large: {abs(phi1_est - 0.6)}"
    assert abs(phi2_est - 0.3) < 0.15, f"AR(2) parameter error too large: {abs(phi2_est - 0.3)}"

    # Ljung-Box test at lag 10 > 0.05
    assert res.ljung_box is not None, "Ljung-Box table is None"
    lb_p_lag10 = res.ljung_box.loc[10, "p-value"]
    print(f"Ljung-Box p-value at lag 10: {lb_p_lag10:.4f}")
    assert lb_p_lag10 > 0.05, f"Expected Ljung-Box p-value at lag 10 > 0.05, got {lb_p_lag10}"

    # Picklability check (no statsmodels result objects attached)
    pickled_bytes = pickle.dumps(res)
    unpickled_res = pickle.loads(pickled_bytes)
    assert isinstance(unpickled_res, ModelResult), "Pickle round trip failed"
    print("[PASS] AR(2) recovery within 0.15, Ljung-Box lag 10 > 0.05, picklability verified.")


def test_ma1_recovery():
    """Simulated MA(1) (theta 0.6, n=600): 'ma' q=1 recovers theta within 0.15."""
    print("\n--- 2. Testing MA(1) Parameter Recovery ---")
    rng = np.random.default_rng(42)
    n = 600
    eps = rng.normal(0, 1, n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = eps[t] + 0.6 * eps[t - 1]

    dates = pd.date_range("2000-01-01", periods=n, freq="MS")
    s = pd.Series(y, index=dates, name="ma1")

    res = fit_model(s, {"family": "ma", "q": 1}, test_size=60)
    assert res.converged is True, f"MA(1) model failed to converge. Warnings: {res.warnings}"

    params = dict(zip(res.params_table["parameter"], res.params_table["estimate"]))
    theta1_est = params.get("ma.L1")
    print(f"Recovered theta1: {theta1_est:.4f} (True: 0.6)")
    assert abs(theta1_est - 0.6) < 0.15, f"MA(1) parameter error too large: {abs(theta1_est - 0.6)}"
    print("[PASS] MA(1) recovery within 0.15 verified.")


def test_arma11_fit_and_metrics():
    """Simulated ARMA(1,1): fit works, metrics_test keys present."""
    print("\n--- 3. Testing ARMA(1,1) Fit & Metrics ---")
    rng = np.random.default_rng(42)
    n = 600
    eps = rng.normal(0, 1, n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.5 * y[t - 1] + eps[t] + 0.4 * eps[t - 1]

    dates = pd.date_range("2000-01-01", periods=n, freq="MS")
    s = pd.Series(y, index=dates, name="arma11")

    res = fit_model(s, {"family": "arma", "p": 1, "q": 1}, test_size=60)
    assert res.converged is True, f"ARMA(1,1) fit failed. Warnings: {res.warnings}"

    required_keys = ["rmse", "mae", "mape", "smape", "mase"]
    for k in required_keys:
        assert k in res.metrics_test, f"Missing metric key {k} in metrics_test"
        assert not np.isnan(res.metrics_test[k]), f"Metric {k} is NaN"

    print(f"metrics_test: {res.metrics_test}")
    print("[PASS] ARMA(1,1) fit succeeded and all metric keys are present and finite.")


def test_random_walk_with_drift():
    """Random walk with drift: ARIMA(0,1,0) fits with finite RMSE; drift forecast slope is positive."""
    print("\n--- 4. Testing Random Walk with Drift & Drift Baseline ---")
    rng = np.random.default_rng(42)
    n = 200
    y = np.cumsum(rng.normal(0.5, 1.0, n))
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    s = pd.Series(y, index=dates, name="rw_drift")

    res_arima = fit_model(s, {"family": "arima", "p": 0, "d": 1, "q": 0, "trend": "c"}, test_size=20)
    assert res_arima.converged is True, f"ARIMA(0,1,0) fit failed: {res_arima.warnings}"
    assert np.isfinite(res_arima.metrics_test["rmse"]), "ARIMA test RMSE is not finite"

    res_drift = fit_model(s, {"family": "drift"}, test_size=20)
    assert res_drift.converged is True, f"Drift model fit failed: {res_drift.warnings}"
    fc_mean = res_drift.test_forecast["mean"]
    drift_slope = (fc_mean.iloc[-1] - fc_mean.iloc[0]) / (len(fc_mean) - 1)
    print(f"ARIMA RMSE: {res_arima.metrics_test['rmse']:.4f}, Drift slope: {drift_slope:.4f}")
    assert drift_slope > 0, f"Expected positive drift slope, got {drift_slope}"
    print("[PASS] Random walk ARIMA finite RMSE and positive drift slope verified.")


def test_seasonal_monthly_series():
    """Synthetic seasonal monthly series (n=120): SARIMA and Holt-Winters MAPE < 15%, and seasonal_naive works."""
    print("\n--- 5. Testing Seasonal Monthly Series (SARIMA, Holt-Winters, Seasonal Naive) ---")
    rng = np.random.default_rng(42)
    n = 120
    dates = pd.date_range("2010-01-01", periods=n, freq="MS")
    trend = np.linspace(50, 150, n)
    seasonal = 20 * np.sin(2 * np.pi * np.arange(n) / 12)
    noise = rng.normal(0, 2, n)
    s = pd.Series(trend + seasonal + noise, index=dates, name="seasonal_monthly")

    sarima_res = fit_model(
        s,
        {"family": "sarima", "p": 1, "d": 1, "q": 1, "P": 0, "D": 1, "Q": 1, "m": 12},
        test_size=12,
        m=12,
    )
    hw_res = fit_model(
        s,
        {"family": "holt_winters", "trend": "add", "seasonal": "add", "m": 12},
        test_size=12,
        m=12,
    )
    snaive_res = fit_model(s, {"family": "seasonal_naive", "m": 12}, test_size=12, m=12)

    assert sarima_res.converged is True, f"SARIMA failed: {sarima_res.warnings}"
    assert hw_res.converged is True, f"Holt-Winters failed: {hw_res.warnings}"
    assert snaive_res.converged is True, f"Seasonal Naive failed: {snaive_res.warnings}"

    sarima_mape = sarima_res.metrics_test["mape"]
    hw_mape = hw_res.metrics_test["mape"]

    print(f"SARIMA MAPE: {sarima_mape:.2f}%, Holt-Winters MAPE: {hw_mape:.2f}%")
    assert sarima_mape < 15.0, f"SARIMA MAPE too high: {sarima_mape}%"
    assert hw_mape < 15.0, f"Holt-Winters MAPE too high: {hw_mape}%"
    assert np.isfinite(snaive_res.metrics_test["rmse"]), "Seasonal naive RMSE is not finite"
    print("[PASS] SARIMA, Holt-Winters, and Seasonal Naive seasonal tests passed.")


def test_sarimax_with_and_without_exog():
    """SARIMAX with exog driver succeeds; missing exog returns failed ModelResult rather than raising."""
    print("\n--- 6. Testing SARIMAX with and without Exogenous Regressors ---")
    rng = np.random.default_rng(42)
    n = 100
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    x = pd.DataFrame({"promo": rng.normal(10, 2, n)}, index=dates)
    y = pd.Series(5.0 + 2.5 * x["promo"] + rng.normal(0, 0.5, n), index=dates, name="sales")

    res_ok = fit_model(y, {"family": "sarimax", "p": 1, "d": 0, "q": 0}, test_size=10, exog=x)
    assert res_ok.converged is True, f"SARIMAX with exog failed: {res_ok.warnings}"
    promo_coef = res_ok.params_table[res_ok.params_table["parameter"] == "promo"]["estimate"].values[0]
    print(f"Recovered exogenous coefficient for promo: {promo_coef:.4f} (True: 2.5)")
    assert abs(promo_coef - 2.5) < 0.25, f"Promo coefficient error too large: {abs(promo_coef - 2.5)}"

    # Missing exog must return failed ModelResult with warning rather than raising
    res_fail = fit_model(y, {"family": "sarimax", "p": 1, "d": 0, "q": 0}, test_size=10, exog=None)
    assert res_fail.converged is False, "Expected converged=False when exog is None"
    assert len(res_fail.warnings) > 0, "Expected warning recorded for missing exog"
    assert "exogenous" in res_fail.warnings[0].lower(), f"Unexpected warning: {res_fail.warnings[0]}"
    print(f"Safe failure warning captured: '{res_fail.warnings[0]}'")
    print("[PASS] SARIMAX exogenous driver and missing-exog safety verified.")


def test_log_and_boxcox_transform_round_trip():
    """Positive series: Log & Box-Cox forecasts on original scale and inverse_variance_transform round-trip."""
    print("\n--- 7. Testing Log & Box-Cox Variance Transform Round Trips ---")
    rng = np.random.default_rng(42)
    n = 100
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    y = pd.Series(np.exp(np.linspace(2, 4, n) + rng.normal(0, 0.1, n)), index=dates, name="exp_growth")

    # 1. Log Transform
    _, info_log = transform_series(y, log=True)
    res_log = fit_model(
        np.log(y),
        {"family": "arima", "p": 1, "d": 0, "q": 0},
        test_size=10,
        transform_info=info_log,
    )
    assert res_log.converged is True, f"Log ARIMA fit failed: {res_log.warnings}"
    assert np.allclose(inverse_variance_transform(np.log(y), info_log), y), "Log inverse transform mismatch"
    fc_log = res_log.test_forecast["mean"]
    print(f"Log forecast range: [{fc_log.min():.2f}, {fc_log.max():.2f}], Series range: [{y.min():.2f}, {y.max():.2f}]")
    assert fc_log.min() >= y.min() * 0.5 and fc_log.max() <= y.max() * 2.0, "Log forecast not in original scale"

    # 2. Box-Cox Transform
    y_bc, info_bc = transform_series(y, boxcox_tf=True)
    res_bc = fit_model(
        y_bc,
        {"family": "arima", "p": 1, "d": 0, "q": 0},
        test_size=10,
        transform_info=info_bc,
    )
    assert res_bc.converged is True, f"Box-Cox ARIMA fit failed: {res_bc.warnings}"
    assert np.allclose(inverse_variance_transform(y_bc, info_bc), y), "Box-Cox inverse transform mismatch"
    fc_bc = res_bc.test_forecast["mean"]
    print(f"Box-Cox forecast range: [{fc_bc.min():.2f}, {fc_bc.max():.2f}], Series range: [{y.min():.2f}, {y.max():.2f}]")
    assert fc_bc.min() >= y.min() * 0.5 and fc_bc.max() <= y.max() * 2.0, "Box-Cox forecast not in original scale"
    print("[PASS] Log and Box-Cox forecast scaling and inverse round-trips verified.")


def test_multiplicative_holt_winters_zero_guard():
    """Multiplicative Holt-Winters on a series containing a zero returns a failed ModelResult with a message."""
    print("\n--- 8. Testing Multiplicative Holt-Winters Zero Guard ---")
    dates = pd.date_range("2020-01-01", periods=50, freq="D")
    y = pd.Series(np.linspace(10, 50, 50), index=dates)
    y.iloc[10] = 0.0

    res = fit_model(y, {"family": "holt_winters", "trend": "mul", "seasonal": None}, test_size=10)
    assert res.converged is False, "Expected converged=False for multiplicative HW with zero value"
    assert len(res.warnings) > 0, "Expected warning recorded for zero value"
    assert "positive" in res.warnings[0].lower(), f"Unexpected warning: {res.warnings[0]}"
    print(f"Zero guard warning captured: '{res.warnings[0]}'")
    print("[PASS] Multiplicative Holt-Winters zero guard verified.")


def test_residual_burn_in():
    """d>0 burn-in: ARIMA(1,1,1) residual length == train length - 1 and first residual is not an outlier."""
    print("\n--- 9. Testing Integration Burn-in Residual Trimming ---")
    rng = np.random.default_rng(42)
    n = 100
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    y = pd.Series(np.cumsum(rng.normal(0, 1, n)), index=dates)

    res = fit_model(y, {"family": "arima", "p": 1, "d": 1, "q": 1}, test_size=20)
    train_len = len(y) - 20
    assert len(res.residuals) == train_len - 1, f"Expected {train_len - 1} residuals, got {len(res.residuals)}"
    first_res = res.residuals.iloc[0]
    print(f"Train length: {train_len}, Residual length: {len(res.residuals)}, First residual: {first_res:.4f}")
    assert abs(first_res) < 100.0, f"First residual is unexpectedly large: {first_res}"
    print("[PASS] Residual burn-in trimming verified.")


def test_grid_search_arima():
    """grid_search_arima on AR(2) series sorted by AIC, honors max_models=5 and reports truncated flag."""
    print("\n--- 10. Testing ARIMA Grid Search & Truncation ---")
    rng = np.random.default_rng(42)
    n = 200
    eps = rng.normal(0, 1, n)
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = 0.6 * y[t - 1] + 0.3 * y[t - 2] + eps[t]
    dates = pd.date_range("2000-01-01", periods=n, freq="MS")
    s = pd.Series(y, index=dates)

    df_grid, best = grid_search_arima(s, p_range=[0, 1, 2, 3], d=0, q_range=[0, 1, 2], test_size=20, max_models=5)
    assert len(df_grid) == 5, f"Expected 5 models evaluated, got {len(df_grid)}"
    assert best.get("truncated") is True, "Expected best['truncated'] == True"
    assert df_grid["aic"].is_monotonic_increasing, "Grid results are not sorted ascending by AIC"
    print(f"Grid models evaluated: {len(df_grid)}, Truncated flag: {best['truncated']}")
    print(f"Top 3 specs by AIC:\n{df_grid[['spec', 'aic', 'bic', 'converged']].head(3)}")
    print("[PASS] Grid search sorting and truncation verified.")


def test_fit_full_and_forecast():
    """fit_full_and_forecast returns steps rows with DatetimeIndex continuing frequency and lower <= mean <= upper."""
    print("\n--- 11. Testing Full-Fit Forecasting & Confidence Bounds ---")
    dates = pd.date_range("2020-01-01", periods=60, freq="MS")
    y = pd.Series(np.linspace(10, 40, 60) + np.random.randn(60), index=dates)

    fc_df = fit_full_and_forecast(y, {"family": "ar", "p": 1}, steps=12)
    assert len(fc_df) == 12, f"Expected 12 forecast rows, got {len(fc_df)}"
    assert isinstance(fc_df.index, pd.DatetimeIndex), "Forecast index is not DatetimeIndex"
    assert fc_df.index[0] == pd.Timestamp("2025-01-01"), f"Expected start date 2025-01-01, got {fc_df.index[0]}"
    assert (fc_df["lower"] <= fc_df["mean"]).all(), "Violated condition: lower <= mean"
    assert (fc_df["mean"] <= fc_df["upper"]).all(), "Violated condition: mean <= upper"
    print(f"Forecast shape: {fc_df.shape}, First date: {fc_df.index[0]}, Bounds consistent: True")
    print("[PASS] fit_full_and_forecast frequency continuation and bounds verified.")


def test_default_test_size_behavior():
    """default_test_size behaviour for (n=48, m=12) -> 12, (n=100, m=None) -> 12, (n=20, m=12) -> 6."""
    print("\n--- 12. Testing default_test_size Behavior ---")
    t1 = default_test_size(48, 12)
    t2 = default_test_size(100, None)
    t3 = default_test_size(20, 12)

    print(f"default_test_size(48, 12) = {t1} (Expected 12)")
    print(f"default_test_size(100, None) = {t2} (Expected 12)")
    print(f"default_test_size(20, 12) = {t3} (Expected 6)")

    assert t1 == 12, f"Expected 12, got {t1}"
    assert t2 == 12, f"Expected 12, got {t2}"
    assert t3 == 6, f"Expected 6, got {t3}"
    print("[PASS] default_test_size edge-case behavior verified.")


def test_metrics_edge_cases():
    """metrics: mape returns nan for all-zero truth; mase with a naive train benchmark is finite."""
    print("\n--- 13. Testing Metrics Edge Cases ---")
    yt_zero = np.zeros(10)
    yp = np.ones(10)
    mape_zero = mape(yt_zero, yp)
    assert np.isnan(mape_zero), f"Expected NaN for all-zero truth in MAPE, got {mape_zero}"

    yt_real = np.array([10.0, 12.0, 11.0])
    yp_real = np.array([9.5, 11.8, 11.2])
    train = np.array([8.0, 9.0, 10.0, 11.0, 10.5])
    mase_val = mase(yt_real, yp_real, y_train=train, m=1)
    assert np.isfinite(mase_val), f"Expected finite MASE, got {mase_val}"

    acc = accuracy_table(yt_real, yp_real, y_train=train, m=1)
    assert len(acc) == 5, f"Expected 5 metrics in accuracy table, got {len(acc)}"
    print(f"Zero truth MAPE: {mape_zero}, Finite MASE: {mase_val:.4f}, Accuracy table: {acc}")
    print("[PASS] Metrics edge cases verified.")


def test_list_model_families_order():
    """Model families must be returned in the exact required order."""
    print("\n--- 14. Testing list_model_families Ordering ---")
    fams = list_model_families()
    keys = [f["key"] for f in fams]
    expected_order = [
        "naive",
        "seasonal_naive",
        "drift",
        "ar",
        "ma",
        "arma",
        "arima",
        "sarima",
        "sarimax",
        "holt_winters",
    ]
    assert keys == expected_order, f"Expected {expected_order}, got {keys}"
    print(f"Supported model families ({len(keys)}): {', '.join(keys)}")
    print("[PASS] list_model_families ordering verified.")


def test_a2_burn_in_log_transformed_arima111():
    """A2: ARIMA(1,1,1) on log-transformed positive series: 1 NaN, min > 0.5*min(y), train RMSE < 3*test RMSE."""
    print("\n--- 15. Testing A2 Burn-in on Log-Transformed Series ---")
    rng = np.random.default_rng(42)
    n = 120
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    raw_y = pd.Series(np.exp(np.cumsum(rng.normal(0.01, 0.05, n)) + 2.0), index=dates)
    log_y, info = transform_series(raw_y, log=True)

    res = fit_model(log_y, {"family": "arima", "p": 1, "d": 1, "q": 1}, test_size=20, transform_info=info)

    assert res.converged is True, f"ARIMA(1,1,1) failed to converge: {res.warnings}"
    nan_count = res.fitted_train.isna().sum()
    print(f"Burn-in NaNs in fitted_train: {nan_count} (Expected: 1)")
    assert nan_count == 1, f"Expected exactly 1 NaN, got {nan_count}"

    min_fitted = res.fitted_train.dropna().min()
    min_y_bound = 0.5 * raw_y.min()
    print(f"Min fitted value: {min_fitted:.4f}, 0.5 * min(y): {min_y_bound:.4f}")
    assert min_fitted >= min_y_bound, f"Fitted value {min_fitted} falls below threshold {min_y_bound}"

    train_rmse = res.metrics_train["rmse"]
    test_rmse = res.metrics_test["rmse"]
    print(f"Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
    assert train_rmse < 3.0 * test_rmse, f"Train RMSE ({train_rmse}) is >= 3x Test RMSE ({test_rmse})"
    print("[PASS] A2 burn-in log-transformed ARIMA(1,1,1) test verified.")


def test_a3_grid_search_intercept_and_complexity():
    """A3: Series with mean 50: best ARIMA(p,0,q) has p>=1, trend and n_params columns exist."""
    print("\n--- 16. Testing A3 Grid Search Trend and Order Selection ---")
    rng = np.random.default_rng(42)
    n = 200
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    eps = rng.normal(0, 1, n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 50.0 + 0.7 * (y[t - 1] - 50.0) + eps[t]
    s = pd.Series(y, index=dates)

    df_grid, best = grid_search_arima(s, p_range=[0, 1, 2], d=0, q_range=[0, 1], test_size=20)
    assert "trend" in df_grid.columns, "Column 'trend' missing from grid results"
    assert "n_params" in df_grid.columns, "Column 'n_params' missing from grid results"
    print(f"Best specification by AIC: {best['best_spec_str']}")
    print(df_grid[["spec", "trend", "n_params", "aic", "bic", "converged"]].head(3))
    assert best["best_spec"]["p"] >= 1, f"Expected best model p >= 1, got {best['best_spec']['p']}"
    print("[PASS] A3 grid search intercept handling and complexity ordering verified.")


def test_a6_ar2_cross_check_arima_vs_sarimax():
    """A6: Cross-check AR(2) fit_model vs statsmodels ARIMA(train, order=(2,0,0), trend='c') AR coefficients."""
    print("\n--- 17. Testing A6 AR(2) Cross-Check: fit_model vs. statsmodels ARIMA ---")
    rng = np.random.default_rng(42)
    n = 600
    eps = rng.normal(0, 1, n)
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = 0.6 * y[t - 1] + 0.3 * y[t - 2] + eps[t]
    dates = pd.date_range("2000-01-01", periods=n, freq="MS")
    s = pd.Series(y, index=dates)
    train_s = s.iloc[:-60]

    # Model 1: fit_model interface
    res_fit = fit_model(s, {"family": "ar", "p": 2, "d": 0, "trend": "c"}, test_size=60)
    params_fit = dict(zip(res_fit.params_table["parameter"], res_fit.params_table["estimate"]))

    # Model 2: direct statsmodels ARIMA
    direct_mod = ARIMA(train_s, order=(2, 0, 0), trend="c")
    direct_res = direct_mod.fit()

    phi1_fit = params_fit["ar.L1"]
    phi2_fit = params_fit["ar.L2"]
    phi1_direct = direct_res.params["ar.L1"]
    phi2_direct = direct_res.params["ar.L2"]

    print(f"fit_model AR params:      phi1={phi1_fit:.6f}, phi2={phi2_fit:.6f}")
    print(f"statsmodels ARIMA params: phi1={phi1_direct:.6f}, phi2={phi2_direct:.6f}")

    diff_phi1 = abs(phi1_fit - phi1_direct)
    diff_phi2 = abs(phi2_fit - phi2_direct)
    print(f"Parameter differences: |d_phi1|={diff_phi1:.6f}, |d_phi2|={diff_phi2:.6f}")

    assert diff_phi1 < 0.02, f"phi1 discrepancy ({diff_phi1}) exceeds 0.02 threshold"
    assert diff_phi2 < 0.02, f"phi2 discrepancy ({diff_phi2}) exceeds 0.02 threshold"
    print("[PASS] A6 AR(2) parameter cross-check within 0.02 verified.")


def test_extras_roots_and_stability():
    """Extras contains roots for AR(2) and max inverse root modulus < 1."""
    print("\n--- 18. Testing Extras AR Roots & Modulus Stability ---")
    rng = np.random.default_rng(42)
    n = 200
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = 0.6 * y[t - 1] + 0.3 * y[t - 2] + rng.normal(0, 1)
    s = pd.Series(y, index=pd.date_range("2020-01-01", periods=n, freq="D"))

    res = fit_model(s, {"family": "ar", "p": 2, "d": 0}, test_size=20)
    assert "ar_roots" in res.extras, "ar_roots missing from extras"
    ar_roots = res.extras["ar_roots"]
    assert len(ar_roots) == 2, f"Expected 2 AR roots, got {len(ar_roots)}"
    inv_roots = 1.0 / ar_roots[ar_roots != 0]
    max_mod = float(np.max(np.abs(inv_roots)))
    print(f"AR roots: {ar_roots}, Max inverse-root modulus: {max_mod:.4f}")
    assert max_mod < 1.0, f"Expected max inverse root modulus < 1.0, got {max_mod}"
    print("[PASS] Extras AR roots and stability modulus verified.")


def test_holt_winters_extras_level_length():
    """Holt-Winters extras['level'] length equals the train length."""
    print("\n--- 19. Testing Holt-Winters Extras Level Series Length ---")
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    y = pd.Series(np.linspace(10, 50, 100), index=dates)

    res = fit_model(y, {"family": "holt_winters", "trend": "add", "seasonal": None}, test_size=20)
    assert "level" in res.extras, "'level' missing from HW extras"
    lvl = res.extras["level"]
    train_len = len(res.train_index)
    print(f"HW level length: {len(lvl)}, Train length: {train_len}")
    assert len(lvl) == train_len, f"Expected level length {train_len}, got {len(lvl)}"
    print("[PASS] Holt-Winters extras level length verified.")


def test_pickle_model_result_with_extras():
    """Pickle of ModelResult with populated extras succeeds cleanly."""
    print("\n--- 20. Testing Picklability of ModelResult Container with Extras ---")
    dates = pd.date_range("2020-01-01", periods=80, freq="D")
    y = pd.Series(np.linspace(10, 30, 80), index=dates)

    res = fit_model(y, {"family": "arima", "p": 1, "d": 1, "q": 1}, test_size=15)
    pickled = pickle.dumps(res)
    unpickled = pickle.loads(pickled)
    assert isinstance(unpickled, ModelResult)
    assert "ar_roots" in unpickled.extras
    print("[PASS] ModelResult with extras pickled and unpickled successfully.")


def test_make_model_key_deterministic_and_sensitive():
    """make_model_key is deterministic for identical inputs and differs when p changes."""
    print("\n--- 21. Testing make_model_key Determinism & Sensitivity ---")
    k1 = make_model_key({"family": "ar", "p": 1}, 20, None, None, 100, "2020-01-01")
    k2 = make_model_key({"family": "ar", "p": 1}, 20, None, None, 100, "2020-01-01")
    k3 = make_model_key({"family": "ar", "p": 2}, 20, None, None, 100, "2020-01-01")

    print(f"Key 1 (p=1): {k1}, Key 2 (p=1): {k2}, Key 3 (p=2): {k3}")
    assert k1 == k2, "make_model_key is not deterministic for identical inputs"
    assert k1 != k3, "make_model_key did not differentiate change in order p"
    assert len(k1) == 12, f"Expected 12 characters, got {len(k1)}"
    print("[PASS] make_model_key determinism and hyperparameter sensitivity verified.")


def test_theoretical_acf_pacf():
    """theoretical_acf_pacf for AR(1) phi=0.7 gives acf[1] approx 0.7 and returns None for non-stationary AR."""
    print("\n--- 22. Testing theoretical_acf_pacf Functionality ---")
    theo = theoretical_acf_pacf([1.0, -0.7], [1.0], nlags=10)
    assert theo is not None, "theoretical_acf_pacf returned None for stationary AR(1)"
    theo_acf, theo_pacf = theo
    print(f"Theoretical ACF at lag 1: {theo_acf[1]:.4f} (Expected: ~0.70)")
    assert abs(theo_acf[1] - 0.7) < 0.01, f"ACF lag 1 mismatch: {abs(theo_acf[1] - 0.7)}"

    # Non-stationary AR polynomial [1, -1.2]
    non_stat = theoretical_acf_pacf([1.0, -1.2], [1.0], nlags=10)
    assert non_stat is None, f"Expected None for non-stationary AR polynomial, got {non_stat}"
    print("[PASS] theoretical_acf_pacf calculation and non-stationary guard verified.")


if __name__ == "__main__":
    print("==================================================================")
    print("Starting Comprehensive ATSA Phase 3 & 4 Model Test Suite...")
    print("==================================================================")
    test_ar2_recovery_and_picklability()
    test_ma1_recovery()
    test_arma11_fit_and_metrics()
    test_random_walk_with_drift()
    test_seasonal_monthly_series()
    test_sarimax_with_and_without_exog()
    test_log_and_boxcox_transform_round_trip()
    test_multiplicative_holt_winters_zero_guard()
    test_residual_burn_in()
    test_grid_search_arima()
    test_fit_full_and_forecast()
    test_default_test_size_behavior()
    test_metrics_edge_cases()
    test_list_model_families_order()
    test_a2_burn_in_log_transformed_arima111()
    test_a3_grid_search_intercept_and_complexity()
    test_a6_ar2_cross_check_arima_vs_sarimax()
    test_extras_roots_and_stability()
    test_holt_winters_extras_level_length()
    test_pickle_model_result_with_extras()
    test_make_model_key_deterministic_and_sensitive()
    test_theoretical_acf_pacf()
    print("\n==================================================================")
    print("ALL 22 MODEL ZOO & DIAGNOSTIC TESTS PASSED SUCCESSFULLY!")
    print("==================================================================")
