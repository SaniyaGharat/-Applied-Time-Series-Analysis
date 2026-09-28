"""
Smoke Tests for Phase 4 Plot Builders in modules/plots.py.

Verifies that every plot builder returns a valid plotly.graph_objects.Figure
under both standard configurations and edge cases:
- plot_fit_vs_actual (standard, windowed, empty forecast)
- plot_holdout_zoom (standard, empty forecast)
- plot_inverse_roots (active roots, empty AR/MA roots)
- plot_theoretical_vs_empirical (overlay active, theoretical=None)
- plot_hw_components (full, level only without trend/season, empty extras)
- plot_grid_results (standard top-N, empty DataFrame)
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from modules.plots import (
    plot_fit_vs_actual,
    plot_grid_results,
    plot_holdout_zoom,
    plot_hw_components,
    plot_inverse_roots,
    plot_theoretical_vs_empirical,
)


def test_plot_fit_vs_actual_smoke():
    """Verify plot_fit_vs_actual returns a go.Figure."""
    dates = pd.date_range("2020-01-01", periods=50, freq="D")
    actual = pd.Series(np.linspace(10, 50, 50), index=dates, name="sales")
    fitted = actual.iloc[:40] + 0.5
    test_idx = dates[40:]
    forecast_df = pd.DataFrame(
        {
            "mean": np.linspace(42, 52, 10),
            "lower": np.linspace(38, 48, 10),
            "upper": np.linspace(46, 56, 10),
        },
        index=test_idx,
    )

    fig = plot_fit_vs_actual(actual, fitted, forecast_df, title="Test Fit")
    assert isinstance(fig, go.Figure)

    # Windowed version
    fig_win = plot_fit_vs_actual(actual, fitted, forecast_df, show_train_window=20)
    assert isinstance(fig_win, go.Figure)
    print("[PASS] plot_fit_vs_actual smoke tests passed.")


def test_plot_holdout_zoom_smoke():
    """Verify plot_holdout_zoom returns a go.Figure."""
    dates = pd.date_range("2020-01-01", periods=50, freq="D")
    actual = pd.Series(np.linspace(10, 50, 50), index=dates)
    test_idx = dates[40:]
    forecast_df = pd.DataFrame(
        {
            "mean": np.linspace(42, 52, 10),
            "lower": np.linspace(38, 48, 10),
            "upper": np.linspace(46, 56, 10),
        },
        index=test_idx,
    )

    fig = plot_holdout_zoom(actual, forecast_df)
    assert isinstance(fig, go.Figure)

    # Empty forecast fallback
    fig_empty = plot_holdout_zoom(actual, pd.DataFrame())
    assert isinstance(fig_empty, go.Figure)
    print("[PASS] plot_holdout_zoom smoke tests passed.")


def test_plot_inverse_roots_smoke():
    """Verify plot_inverse_roots with complex roots and empty roots."""
    # 1. Normal roots
    ar_roots = np.array([1.5 + 0.5j, 1.5 - 0.5j])
    ma_roots = np.array([2.0 + 0.0j])
    fig = plot_inverse_roots(ar_roots, ma_roots, title="Root Analysis")
    assert isinstance(fig, go.Figure)

    # 2. Empty roots (e.g. ARIMA(0,1,0) or naive)
    fig_empty = plot_inverse_roots([], [])
    assert isinstance(fig_empty, go.Figure)
    # Check that annotations contain "No AR terms" and "No MA terms"
    ann_texts = [ann.text for ann in fig_empty.layout.annotations]
    assert any("No AR terms" in txt for txt in ann_texts)
    assert any("No MA terms" in txt for txt in ann_texts)
    print("[PASS] plot_inverse_roots smoke tests passed.")


def test_plot_theoretical_vs_empirical_smoke():
    """Verify plot_theoretical_vs_empirical with and without theoretical series."""
    lags = np.arange(1, 11)
    emp_acf = np.linspace(0.8, 0.1, 10)
    emp_pacf = np.array([0.8, 0.05, 0.02, 0.01, 0, 0, 0, 0, 0, 0])
    theo_acf = np.linspace(0.75, 0.08, 10)
    theo_pacf = np.array([0.75, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    conf_bound = 0.2

    # Normal with theoretical
    fig = plot_theoretical_vs_empirical(lags, emp_acf, emp_pacf, theo_acf, theo_pacf, conf_bound)
    assert isinstance(fig, go.Figure)

    # Without theoretical overlay (e.g. non-stationary AR)
    fig_none = plot_theoretical_vs_empirical(lags, emp_acf, emp_pacf, None, None, conf_bound)
    assert isinstance(fig_none, go.Figure)
    print("[PASS] plot_theoretical_vs_empirical smoke tests passed.")


def test_plot_hw_components_smoke():
    """Verify plot_hw_components with full components and level-only components."""
    dates = pd.date_range("2020-01-01", periods=40, freq="D")

    # 1. Full components (level, trend, season)
    extras_full = {
        "level": pd.Series(np.linspace(10, 30, 40), index=dates),
        "trend": pd.Series(np.linspace(0.5, 0.5, 40), index=dates),
        "season": pd.Series(np.sin(np.arange(40)), index=dates),
    }
    fig_full = plot_hw_components(extras_full, dates)
    assert isinstance(fig_full, go.Figure)

    # 2. Level only without trend or season
    extras_level = {
        "level": pd.Series(np.linspace(10, 30, 40), index=dates),
        "trend": None,
        "season": None,
    }
    fig_lvl = plot_hw_components(extras_level, dates)
    assert isinstance(fig_lvl, go.Figure)

    # 3. Empty extras fallback
    fig_empty = plot_hw_components({}, dates)
    assert isinstance(fig_empty, go.Figure)
    print("[PASS] plot_hw_components smoke tests passed.")


def test_plot_grid_results_smoke():
    """Verify plot_grid_results with populated grid and empty DataFrame."""
    grid_data = pd.DataFrame(
        {
            "spec": ["ARIMA(1,0,1)", "ARIMA(2,0,0)", "ARIMA(0,0,1)", "ARIMA(1,0,0)"],
            "aic": [120.5, 122.1, 125.0, 127.3],
            "bic": [126.2, 127.8, 128.9, 131.2],
        }
    )
    fig = plot_grid_results(grid_data, top_n=3)
    assert isinstance(fig, go.Figure)

    # Empty DataFrame
    fig_empty = plot_grid_results(pd.DataFrame())
    assert isinstance(fig_empty, go.Figure)
    print("[PASS] plot_grid_results smoke tests passed.")


if __name__ == "__main__":
    print("==================================================")
    print("Starting ATSA Plot Models Smoke Tests...")
    print("==================================================")
    test_plot_fit_vs_actual_smoke()
    test_plot_holdout_zoom_smoke()
    test_plot_inverse_roots_smoke()
    test_plot_theoretical_vs_empirical_smoke()
    test_plot_hw_components_smoke()
    test_plot_grid_results_smoke()
    print("\n==================================================")
    print("ALL TESTS IN TEST_PLOTS_MODELS COMPLETED SUCCESSFULLY!")
    print("==================================================")
