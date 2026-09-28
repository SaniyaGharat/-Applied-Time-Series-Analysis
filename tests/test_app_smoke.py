"""
End-to-End AppTest Smoke Test for ATSA Phase 4 Models UI.

Uses streamlit.testing.v1.AppTest to verify:
1. Booting app.py and loading demo time series dataset into session_state.
2. Iterating through all ten supported model families:
   - Naive
   - Seasonal Naive
   - Drift
   - AR
   - MA
   - ARMA
   - ARIMA
   - SARIMA
   - SARIMAX
   - Holt-Winters
3. Asserting no unhandled exceptions (not at.exception).
4. Asserting that for AR, ARIMA, and Holt-Winters, at least one Plotly figure is rendered.

Note: Streamlit AppTest supports driving both st.pills and the test-mode
selectbox fallback (key='model_family_fallback' under ATSA_TEST=1).
"""

import os
import sys

# Ensure ATSA_TEST environment variable is set for reproducible AppTest driving
os.environ["ATSA_TEST"] = "1"

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from ui.data_tab import create_demo_data


def test_app_smoke_all_ten_models():
    """Verify that all ten model families can be selected and rendered without exceptions."""
    print("\n--- Starting AppTest Smoke Test for All 10 Model Families ---")
    df = create_demo_data()

    at = AppTest.from_file("app.py")

    # Pre-populate session state with demo dataset and regularized series
    dates = pd.to_datetime(df["Date"])
    ts = pd.Series(df["Sales"].values, index=dates, name="Sales")
    exog = pd.DataFrame({"Marketing_Spend": df["Marketing_Spend"].values}, index=dates)

    at.session_state["raw_df"] = df
    at.session_state["source_name"] = "Demo Monthly Sales"
    at.session_state["selected_date_col"] = "Date"
    at.session_state["selected_target_col"] = "Sales"
    at.session_state["selected_exog_cols"] = ["Marketing_Spend"]
    at.session_state["ts_series"] = ts
    at.session_state["model_series"] = ts
    at.session_state["model_base_series"] = ts
    at.session_state["exog_df"] = exog
    at.session_state["transform_info"] = {
        "steps": [],
        "params": {},
        "suggested_d": 0,
        "suggested_D": 0,
        "seasonal_period": 12,
    }

    # Initial boot
    at.run()
    assert not at.exception, f"App threw exception on initial run: {at.exception}"

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

    for fam_label in family_labels:
        print(f"Testing model family: '{fam_label}'...")

        # Drive selector using the test fallback selectbox exposed for ATSA_TEST=1
        if at.selectbox(key="model_family_fallback"):
            at.selectbox(key="model_family_fallback").select(fam_label).run()
        elif at.pills(key="model_family_select"):
            at.pills(key="model_family_select").select(fam_label).run()
        else:
            # Direct session state assignment fallback
            key_map = {
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
            at.session_state["selected_model_family"] = key_map[fam_label]
            at.run()

        # Assert no exceptions
        assert not at.exception, f"Exception encountered while rendering model '{fam_label}': {at.exception}"

        # Assert Plotly chart rendering
        charts = at.get("plotly_chart")
        num_charts = len(charts)
        print(f"  -> Model '{fam_label}' successfully rendered with {num_charts} Plotly chart(s).")

        if fam_label in ["AR", "ARIMA", "Holt-Winters"]:
            assert num_charts >= 1, f"Expected at least 1 Plotly chart rendered for '{fam_label}', found {num_charts}"

    print("[PASS] All 10 model families rendered cleanly with zero exceptions in AppTest.")


if __name__ == "__main__":
    test_app_smoke_all_ten_models()
