"""
Unit and integration tests for multi-column sheet comparison and best performer determination.
"""

import io
import numpy as np
import pandas as pd
import pytest

from modules.compare import compare_columns_and_find_best
from modules.data_loader import get_excel_sheet_names, load_uploaded_file
from modules.export import build_column_comparison_report
from modules.pipeline import run_analysis
from modules.plots import (
    plot_column_component_bars,
    plot_column_metric_comparison,
    plot_multi_column_future_projections,
    plot_multi_indexed_overlay,
)


@pytest.fixture
def synthetic_multi_column_df():
    """Create a synthetic DataFrame with distinct time series columns."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=48, freq="MS")

    # Column 1: Strong trend, low noise
    c1 = np.linspace(100, 200, 48) + rng.normal(0, 1, 48)

    # Column 2: Strong seasonality, moderate noise
    c2 = 150 + 30 * np.sin(2 * np.pi * np.arange(48) / 12) + rng.normal(0, 2, 48)

    # Column 3: High noise / volatile
    c3 = 100 + rng.normal(0, 25, 48)

    return pd.DataFrame({
        "Date": dates,
        "Trend_Series": np.round(c1, 2),
        "Seasonal_Series": np.round(c2, 2),
        "Volatile_Series": np.round(c3, 2),
    })


def test_compare_columns_and_find_best(synthetic_multi_column_df):
    """Test full multi-column comparison engine and winner ranking."""
    df = synthetic_multi_column_df.set_index("Date")

    bundles = {}
    for col in ["Trend_Series", "Seasonal_Series", "Volatile_Series"]:
        bundles[col] = run_analysis(df[col], name=col, m=12)

    # 1. Balanced evaluation
    res_balanced = compare_columns_and_find_best(bundles, criterion="balanced")
    assert res_balanced["winner_column"] in bundles
    assert res_balanced["winner_score"] > 0
    assert "🏆 Winner Announcement" in res_balanced["synthesis"]
    assert len(res_balanced["leaderboard_df"]) == 3
    assert set(res_balanced["leaderboard_df"]["Column"]) == {"Trend_Series", "Seasonal_Series", "Volatile_Series"}

    # 2. Trend evaluation
    res_trend = compare_columns_and_find_best(bundles, criterion="trend")
    assert res_trend["winner_column"] == "Trend_Series"
    assert res_trend["category_winners"]["strongest_trend"] == "Trend_Series"

    # 3. Seasonality evaluation
    res_seas = compare_columns_and_find_best(bundles, criterion="seasonality")
    assert res_seas["category_winners"]["strongest_seasonality"] == "Seasonal_Series"


def test_compare_columns_empty_bundles():
    """Verify graceful handling when no bundles are passed."""
    res = compare_columns_and_find_best({}, criterion="balanced")
    assert res["winner_column"] is None
    assert res["leaderboard_df"].empty
    assert res["winner_score"] == 0.0


def test_build_column_comparison_report(synthetic_multi_column_df):
    """Verify 3-sheet Excel report generation."""
    df = synthetic_multi_column_df.set_index("Date")
    bundles = {
        "Trend_Series": run_analysis(df["Trend_Series"], name="Trend_Series", m=12),
        "Seasonal_Series": run_analysis(df["Seasonal_Series"], name="Seasonal_Series", m=12),
    }

    res = compare_columns_and_find_best(bundles, criterion="balanced")
    excel_bytes = build_column_comparison_report(
        leaderboard_df=res["leaderboard_df"],
        synthesis=res["synthesis"],
        criterion="balanced",
        category_winners=res["category_winners"],
        bundles=bundles,
    )

    assert isinstance(excel_bytes, bytes)
    assert len(excel_bytes) > 1000

    xl = pd.ExcelFile(io.BytesIO(excel_bytes))
    assert "Leaderboard" in xl.sheet_names
    assert "Executive_Summary" in xl.sheet_names
    assert "Future_Projections" in xl.sheet_names


def test_multi_column_plots(synthetic_multi_column_df):
    """Verify Plotly figure generation across all multi-column visualizers."""
    df = synthetic_multi_column_df.set_index("Date")
    bundles = {
        "Trend_Series": run_analysis(df["Trend_Series"], name="Trend_Series", m=12),
        "Seasonal_Series": run_analysis(df["Seasonal_Series"], name="Seasonal_Series", m=12),
    }
    res = compare_columns_and_find_best(bundles, criterion="balanced")

    fig1 = plot_multi_indexed_overlay({"Trend_Series": df["Trend_Series"], "Seasonal_Series": df["Seasonal_Series"]})
    assert len(fig1.data) >= 2

    fig2 = plot_column_metric_comparison(res["leaderboard_df"])
    assert len(fig2.data) == 2

    fig3 = plot_column_component_bars(res["leaderboard_df"])
    assert len(fig3.data) == 2

    fig4 = plot_multi_column_future_projections(bundles)
    assert len(fig4.data) >= 1
