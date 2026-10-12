"""
Unit and Integration Tests for Analytical Pipeline and Two-Dataset Comparison (Phase 8).
Runnable with:
    pytest tests/test_pipeline.py -v
"""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import pytest

from modules.pipeline import (
    AnalysisBundle,
    build_arima_family_comparison,
    compare_datasets,
    compute_kpis,
    run_analysis,
)
from modules.plots import (
    plot_arima_dual_overlay,
    plot_arima_family_bars,
    plot_arima_multi_metric_bars,
    plot_arima_radar,
    plot_single_dataset_arima_bars,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


def test_run_analysis_airpassengers_and_daily_synthetic():
    """Verify run_analysis on monthly AirPassengers and daily synthetic series: converged, picklable."""
    air_csv = DATA_DIR / "AirPassengers.csv"
    assert air_csv.exists(), f"AirPassengers dataset missing at {air_csv}"

    df_air = pd.read_csv(air_csv)
    s_air = pd.Series(
        df_air["Passengers"].values,
        index=pd.to_datetime(df_air["Month"]),
        name="Passengers",
    )

    # 1. Run pipeline on AirPassengers (monthly)
    bundle_air = run_analysis(s_air, "AirPassengers", m=12, holdout=12)
    assert isinstance(bundle_air, AnalysisBundle)
    assert bundle_air.n_obs == 144
    assert bundle_air.best_key != ""
    assert bundle_air.best_label != "None"
    assert bundle_air.future is not None
    assert bundle_air.future.converged is True
    assert len(bundle_air.model_results) >= 5

    # Verify bundle is picklable
    pickled_air = pickle.dumps(bundle_air)
    unpickled_air = pickle.loads(pickled_air)
    assert unpickled_air.name == "AirPassengers"
    assert unpickled_air.n_obs == 144
    assert unpickled_air.best_key == bundle_air.best_key

    # 2. Run pipeline on Daily Synthetic Series
    rng = np.random.default_rng(42)
    dates_daily = pd.date_range("2021-01-01", periods=100, freq="D")
    vals_daily = 100.0 + np.cumsum(rng.normal(0, 1, 100)) + 5.0 * np.sin(2 * np.pi * np.arange(100) / 7)
    s_daily = pd.Series(vals_daily, index=dates_daily, name="DailySales")

    bundle_daily = run_analysis(s_daily, "Daily Synthetic", m=7, holdout=14)
    assert isinstance(bundle_daily, AnalysisBundle)
    assert bundle_daily.n_obs == 100
    assert bundle_daily.best_key != ""
    assert bundle_daily.future is not None
    assert bundle_daily.future.converged is True

    # Verify daily bundle is picklable
    pickled_daily = pickle.dumps(bundle_daily)
    unpickled_daily = pickle.loads(pickled_daily)
    assert unpickled_daily.name == "Daily Synthetic"
    assert unpickled_daily.best_key == bundle_daily.best_key


def test_run_analysis_short_series_returns_bundle_with_notes():
    """Verify short series (n < 30) returns a valid AnalysisBundle with notes, never an unhandled exception."""
    rng = np.random.default_rng(42)
    dates_short = pd.date_range("2023-01-01", periods=20, freq="D")
    s_short = pd.Series(rng.normal(10, 2, 20), index=dates_short, name="Short")

    bundle_short = run_analysis(s_short, "Short Series", m=4, holdout=3)
    assert isinstance(bundle_short, AnalysisBundle)
    assert bundle_short.n_obs == 20
    assert len(bundle_short.notes) > 0
    assert any("30" in n or "short" in n.lower() for n in bundle_short.notes)

    # Picklable
    p = pickle.dumps(bundle_short)
    assert len(p) > 0


def test_compute_kpis_keys_finite_and_no_raw_rmse_ranking():
    """Verify compute_kpis keys present, forecast_change_pct is finite, and raw RMSE is marked not comparable."""
    air_csv = DATA_DIR / "AirPassengers.csv"
    df_air = pd.read_csv(air_csv)
    s_air = pd.Series(df_air["Passengers"].values, index=pd.to_datetime(df_air["Month"]), name="Passengers")
    bundle_air = run_analysis(s_air, "AirPassengers", m=12, holdout=12)

    kpis = compute_kpis(bundle_air)
    required_keys = [
        "n_obs",
        "date range",
        "frequency",
        "cv",
        "trend_strength",
        "seasonal_strength",
        "stationarity verdict",
        "best model label",
        "best MASE",
        "best sMAPE",
        "RelRMSE vs naive",
        "n_beating_naive/n_models_fit",
        "forecast_change_pct",
        "mean_interval_width_pct",
        "coverage95",
    ]
    for rk in required_keys:
        assert rk in kpis, f"Key '{rk}' missing from compute_kpis output"

    assert np.isfinite(kpis["forecast_change_pct"]), "forecast_change_pct must be finite"
    assert np.isfinite(kpis["mean_interval_width_pct"]), "mean_interval_width_pct must be finite"

    # Compare two bundles and assert raw RMSE/MAE notes
    rng = np.random.default_rng(42)
    s_other = pd.Series(np.linspace(1000, 5000, 60), index=pd.date_range("2010-01-01", periods=60, freq="MS"))
    bundle_other = run_analysis(s_other, "Other Dataset", m=12, holdout=6)

    kpis_other = compute_kpis(bundle_other)
    cmp_df, findings = compare_datasets(kpis, kpis_other)

    assert not cmp_df.empty
    assert "KPI" in cmp_df.columns
    assert "A" in cmp_df.columns
    assert "B" in cmp_df.columns
    assert "note" in cmp_df.columns

    # Assert raw RMSE note text
    rmse_row = cmp_df[cmp_df["KPI"] == "Raw Best RMSE"]
    assert not rmse_row.empty, "Raw Best RMSE row must exist in compare_datasets output"
    note_text = rmse_row["note"].iloc[0]
    assert "different units, not comparable" in note_text.lower(), f"Unexpected note: {note_text}"

    mae_row = cmp_df[cmp_df["KPI"] == "Raw Best MAE"]
    assert not mae_row.empty, "Raw Best MAE row must exist in compare_datasets output"
    assert "different units, not comparable" in mae_row["note"].iloc[0].lower()

    assert len(findings) > 0, "Plain-English findings must not be empty"


def test_best_key_refers_to_rank_1_row():
    """Verify best_key in AnalysisBundle corresponds to the Rank==1 row of the comparison table."""
    air_csv = DATA_DIR / "AirPassengers.csv"
    df_air = pd.read_csv(air_csv)
    s_air = pd.Series(df_air["Passengers"].values, index=pd.to_datetime(df_air["Month"]), name="Passengers")
    bundle = run_analysis(s_air, "AirPassengers", m=12, holdout=12)

    comp = bundle.comparison
    assert not comp.empty
    converged_comp = comp[comp["Converged"] == True]
    assert not converged_comp.empty

    top_row = converged_comp.iloc[0]
    assert top_row["Rank"] == 1
    assert bundle.best_label == top_row["Model"]

    # Locate model result with best_key
    matching_res = [r for r in bundle.model_results if r.key == bundle.best_key]
    assert len(matching_res) == 1
    assert matching_res[0].label == top_row["Model"]


def test_overlapping_date_correlation_threshold():
    """Verify first-difference correlation is returned only when date overlap >= 24."""
    # Case 1: Overlap of 36 dates (>= 24)
    dates_shared = pd.date_range("2020-01-01", periods=36, freq="MS")
    s1 = pd.Series(np.linspace(10, 50, 36) + np.random.default_rng(1).normal(0, 1, 36), index=dates_shared, name="S1")
    s2 = pd.Series(np.linspace(100, 200, 36) + np.random.default_rng(2).normal(0, 2, 36), index=dates_shared, name="S2")

    b1 = run_analysis(s1, "S1", m=12, holdout=6)
    b2 = run_analysis(s2, "S2", m=12, holdout=6)

    k1 = compute_kpis(b1)
    k2 = compute_kpis(b2)

    df_36, findings_36 = compare_datasets(k1, k2, series_a=s1, series_b=s2)
    has_corr_36 = any("Correlation" in str(k) for k in df_36["KPI"].values)
    assert has_corr_36, "First-difference correlation row must be present when overlap >= 24"
    assert any("correlation" in f.lower() for f in findings_36)

    # Case 2: Overlap of 10 dates (< 24)
    dates_short_overlap = pd.date_range("2020-01-01", periods=10, freq="MS")
    s3 = pd.Series(range(10), index=dates_short_overlap, name="S3")
    s4 = pd.Series(range(10), index=dates_short_overlap, name="S4")

    k3 = {"dataset_name": "S3", "_series": s3}
    k4 = {"dataset_name": "S4", "_series": s4}

    df_10, findings_10 = compare_datasets(k3, k4, series_a=s3, series_b=s4)
    has_corr_10 = any("Correlation" in str(k) for k in df_10["KPI"].values)
    assert not has_corr_10, "First-difference correlation must NOT be present when overlap < 24"


def test_build_arima_family_comparison_and_plots():
    """Verify build_arima_family_comparison and plot_arima_family_bars structure, families, and chart traces."""
    air_csv = DATA_DIR / "AirPassengers.csv"
    df_air = pd.read_csv(air_csv)
    s_air = pd.Series(df_air["Passengers"].values, index=pd.to_datetime(df_air["Month"]), name="AirPassengers")

    rng = np.random.default_rng(123)
    dates_b = pd.date_range("1950-01-01", periods=100, freq="MS")
    vals_b = 200.0 + np.cumsum(rng.normal(0, 2, 100)) + 10.0 * np.sin(2 * np.pi * np.arange(100) / 12)
    s_b = pd.Series(vals_b, index=dates_b, name="SyntheticMonthly")

    bundle_a = run_analysis(s_air, "AirPassengers", m=12, holdout=12)
    bundle_b = run_analysis(s_b, "SyntheticMonthly", m=12, holdout=12)

    bj_df = build_arima_family_comparison(bundle_a, bundle_b)
    assert isinstance(bj_df, pd.DataFrame)
    assert len(bj_df) == 5

    expected_cols = [
        "Family",
        f"{bundle_a.name} Model",
        f"{bundle_a.name} Converged",
        f"{bundle_a.name} MASE",
        f"{bundle_a.name} sMAPE (%)",
        f"{bundle_a.name} AIC",
        f"{bundle_b.name} Model",
        f"{bundle_b.name} Converged",
        f"{bundle_b.name} MASE",
        f"{bundle_b.name} sMAPE (%)",
        f"{bundle_b.name} AIC",
        "Top Performer (Lower MASE)",
        "Insights",
    ]
    for col in expected_cols:
        assert col in bj_df.columns, f"Expected column '{col}' missing from Box-Jenkins comparison table"

    families_in_table = bj_df["Family"].tolist()
    assert any("AR (" in fam for fam in families_in_table)
    assert any("MA (" in fam for fam in families_in_table)
    assert any("ARMA (" in fam for fam in families_in_table)
    assert any("ARIMA (" in fam for fam in families_in_table)
    assert any("SARIMA (" in fam for fam in families_in_table)

    # Test plot generation for each metric
    for metric in ["MASE", "sMAPE", "AIC"]:
        fig = plot_arima_family_bars(bundle_a, bundle_b, metric=metric, names=(bundle_a.name, bundle_b.name))
        assert fig is not None
        assert len(fig.data) == 2  # Two series traces (A and B)
        assert len(fig.data[0].x) == 5  # 5 Box-Jenkins architectures
        assert list(fig.data[0].x) == ["AR", "MA", "ARMA", "ARIMA", "SARIMA"]

    # Test multi-metric 3-panel figure
    fig_multi = plot_arima_multi_metric_bars(bundle_a, bundle_b, names=(bundle_a.name, bundle_b.name))
    assert fig_multi is not None
    assert len(fig_multi.data) == 6  # 2 series traces * 3 subplots = 6 traces

    # Test dual overlay stacked figure
    fig_dual = plot_arima_dual_overlay(bundle_a, bundle_b, names=(bundle_a.name, bundle_b.name))
    assert fig_dual is not None
    assert len(fig_dual.data) >= 6  # Actuals + multiple Box-Jenkins models

    # Test radar plot
    fig_radar = plot_arima_radar(bundle_a, bundle_b, names=(bundle_a.name, bundle_b.name))
    assert fig_radar is not None
    assert len(fig_radar.data) == 2

    # Test single-dataset bar plot
    fig_single = plot_single_dataset_arima_bars(bundle_a.model_results, metric="MASE")
    assert fig_single is not None
    assert len(fig_single.data) == 1
    assert list(fig_single.data[0].x) == ["AR", "MA", "ARMA", "ARIMA", "SARIMA"]


