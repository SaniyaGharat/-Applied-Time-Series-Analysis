import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from modules.data_loader import regularize_series

def test_a1_resample_fallback():
    # Month-start series
    dates = pd.date_range("2023-01-01", periods=6, freq="MS")
    values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    series = pd.Series(values, index=dates, name="sales")

    print("=== Input Series (Month-Start 'MS') ===")
    print(series)

    # User chooses 'ME' (Month-End)
    reg_series, report = regularize_series(series, freq="ME", fill_method="interpolate")

    print("\n=== Regularized Series (Requested 'ME') ===")
    print(reg_series)

    print("\n=== Regularization Report ===")
    for k, v in report.items():
        print(f"{k}: {v}")

    # Confirmations
    assert report["method"] == "resample", f"Expected method 'resample', got {report['method']}"
    assert report["warning"] is not None, "Expected warning in report, got None"
    assert int(reg_series.isna().sum()) == 0, "Expected no NaNs in output series"
    assert len(reg_series) == 6, f"Expected 6 observations, got {len(reg_series)}"
    print("\n[SUCCESS] Test A1 passed: Resample fallback triggered, 0 NaNs, warning reported.")


def test_a5_upsampling_value_error():
    """A5: Monthly data with target freq='D' (finer than data interval) raises ValueError."""
    print("\n=== Test A5: Upsampling Guard (Monthly data with target freq='D') ===")
    dates = pd.date_range("2023-01-01", periods=12, freq="MS")
    series = pd.Series(range(12), index=dates, name="monthly_data")

    try:
        regularize_series(series, freq="D")
        assert False, "Expected ValueError when upsampling monthly series to daily frequency."
    except ValueError as err:
        print(f"Captured expected ValueError: {err}")
        assert "finer than the data's actual sampling interval" in str(err)
        assert "upsampling would fabricate data" in str(err)
        print("[SUCCESS] Test A5 passed: Upsampling ValueError raised with precise message.")


if __name__ == "__main__":
    test_a1_resample_fallback()
    test_a5_upsampling_value_error()

