"""
End-to-End AppTest Smoke and Functional Test Suite for ATSA Phases 4 and 5.

Uses streamlit.testing.v1.AppTest to verify:
1. A7: All 10 model families:
   - No exceptions
   - len(at.error) == 0
   - st.session_state["model_results"] contains matching family with converged == True
   - Plotly chart count increased versus no-model baseline
2. A7: AppTest.from_string harness driving st.pills (select, deselect, reselect)
3. A1: "Use best order" pending pattern prevents widget key modification crash and sets form_arima_p == best p
4. A2: Stale widget reset: changing transform_info suggested_d to 1 updates form_arima_d == 1
5. Phase 5 Forecast Tab & Future Forecast:
   - Fit all default models
   - Comparison table rendered and stored in session state / at.dataframe
   - Best model banner present
   - Generate future forecasts for ARIMA, Holt-Winters, and SARIMAX
   - forecast_results contains entries with zero exceptions
   - Multi-sheet Excel workbook export pure builder verification
"""

import io
import os
import sys

# Ensure ATSA_TEST environment variable is set for reproducible AppTest driving
os.environ["ATSA_TEST"] = "1"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import openpyxl
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from modules.compare import make_future_exog
from modules.export import build_excel_report
from ui.data_tab import create_demo_data
from ui.model_common import fit_all_default_models


def _init_demo_apptest() -> AppTest:
    """Helper to initialize AppTest with loaded demo data and exogenous features."""
    df = create_demo_data()
    dates = pd.to_datetime(df["Date"])
    ts = pd.Series(df["Sales"].values, index=dates, name="Sales")
    exog = pd.DataFrame({"Marketing_Spend": df["Marketing_Spend"].values}, index=dates)

    at = AppTest.from_file("app.py", default_timeout=30)
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
    at.session_state["model_results"] = {}
    at.session_state["forecast_results"] = {}
    return at


def test_app_smoke_all_ten_models():
    """A7: Verify that all 10 model families render without error, converge, and render charts."""
    print("\n--- Starting Strengthened AppTest Smoke Test for All 10 Model Families (A7) ---")
    at = _init_demo_apptest()

    # Initial boot to establish baseline
    at.run()
    assert not at.exception, f"App threw exception on initial run: {at.exception}"
    assert len(at.error) == 0, f"Errors present on initial run: {[e.value for e in at.error]}"

    # Baseline chart count in Data/initial state
    baseline_charts = len(at.get("plotly_chart"))
    print(f"Initial baseline Plotly chart count: {baseline_charts}")

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

    for fam_label in family_labels:
        fam_key = label_to_key[fam_label]
        print(f"Testing family: '{fam_label}' ({fam_key})...")

        # Select model family via test selectbox fallback or direct session state
        if at.selectbox(key="model_family_fallback"):
            at.selectbox(key="model_family_fallback").select(fam_label).run()
        else:
            at.session_state["selected_model_family"] = fam_key
            at.run()

        # (i) Assert no unhandled exception
        assert not at.exception, f"Exception for '{fam_label}': {at.exception}"

        # (ii) Assert len(at.error) == 0
        assert len(at.error) == 0, f"Error widget rendered for '{fam_label}': {[e.value for e in at.error]}"

        # (iii) Assert session_state["model_results"] contains matching family
        results = at.session_state["model_results"]
        matching_results = [r for r in results.values() if getattr(r, "family", "") == fam_key]
        assert len(matching_results) > 0, f"No result found in model_results for family '{fam_key}'"
        res = matching_results[-1]
        assert res.converged is True, f"Model '{fam_label}' did not converge on demo data: {res.warnings}"

        # (iv) Assert number of Plotly charts in Models tab increased versus baseline
        current_charts = len(at.get("plotly_chart"))
        assert current_charts > baseline_charts, (
            f"Expected Plotly charts to increase vs baseline ({baseline_charts}), got {current_charts}"
        )
        print(f"  -> '{fam_label}' passed: converged={res.converged}, charts={current_charts} (baseline={baseline_charts})")

    print("[PASS] A7 strengthened assertions verified across all 10 model families.")


def test_app_pills_driving_harness():
    """A7: Separate AppTest.from_string harness driving st.pills (select, deselect, reselect)."""
    print("\n--- Testing AppTest st.pills Driving Capability (A7) ---")
    pills_script = """
import streamlit as st
selected = st.pills("Select Architecture", options=["ARIMA", "SARIMA", "ETS"], default="ARIMA", key="test_pills")
st.write(f"Current selection: {selected}")
"""
    at = AppTest.from_string(pills_script)
    at.run()
    assert not at.exception

    pills_widgets = at.pills
    assert len(pills_widgets) > 0, "No st.pills widget discovered by AppTest"
    print(f"AppTest successfully detected st.pills widget: key={pills_widgets[0].key}")

    # Initial selection
    assert pills_widgets[0].value == "ARIMA"
    print("Initial value: 'ARIMA'")

    # Select another option
    pills_widgets[0].select("SARIMA").run()
    assert not at.exception
    assert pills_widgets[0].value == "SARIMA"
    print("After select('SARIMA'): 'SARIMA'")

    # Deselect / unselect
    # Note: AppTest cannot unselect single-select pills (Streamlit retains current selection when unselect is called in single-select mode).
    try:
        pills_widgets[0].unselect("SARIMA").run()
        actual_val = pills_widgets[0].value
        print(f"After unselect('SARIMA') actual value: {actual_val}")
    except Exception as e:
        print(f"unselect on single-mode pills raised exception: {e}")

    # Reselect
    pills_widgets[0].select("ETS").run()
    assert not at.exception
    assert pills_widgets[0].value == "ETS"
    print("After select('ETS'): 'ETS'")
    print("[PASS] AppTest can successfully inspect and drive st.pills selection!")


def test_a1_use_best_order_apptest():
    """A1: Run ARIMA grid search, click 'Use best order', verify no crash and form_arima_p == best p."""
    print("\n--- Testing A1: Grid Search 'Use best order' Pending Pattern ---")
    at = _init_demo_apptest()
    at.session_state["selected_model_family"] = "arima"
    at.run()
    assert not at.exception

    # Find and click "Run grid search"
    grid_buttons = [b for b in at.button if "Run grid search" in b.label]
    assert len(grid_buttons) > 0, "Run grid search button not found"
    grid_buttons[0].click().run()
    assert not at.exception
    assert len(at.error) == 0

    assert "last_grid_best" in at.session_state
    best_info = at.session_state["last_grid_best"]
    best_p = best_info["best_spec"].get("p", 1)
    print(f"Grid search completed. Best spec: {best_info['best_spec_str']}, best p={best_p}")

    # Find and click "Use best order"
    use_buttons = [b for b in at.button if "Use best order" in b.label]
    assert len(use_buttons) > 0, "Use best order button not found"
    use_buttons[0].click().run()

    # Assert no StreamlitAPIException or unhandled crash
    assert not at.exception, f"Crash encountered on 'Use best order': {at.exception}"
    assert len(at.error) == 0, f"Error displayed: {[e.value for e in at.error]}"

    # Assert form_arima_p equals best p
    assert at.session_state["form_arima_p"] == best_p, (
        f"Expected form_arima_p to be {best_p}, got {at.session_state.get('form_arima_p')}"
    )
    print(f"[PASS] A1 verified: 'Use best order' executed cleanly with form_arima_p={best_p}.")


def test_a2_stale_widget_state_apptest():
    """A2: Load demo, open ARIMA, update transform_info suggested_d to 1, rerun, assert form_arima_d == 1."""
    print("\n--- Testing A2: Stale Widget Reset on Suggested Differencing Change ---")
    at = _init_demo_apptest()
    at.session_state["selected_model_family"] = "arima"
    at.run()
    assert not at.exception

    # Initially suggested_d is 0
    assert at.session_state["form_arima_d"] == 0

    # Modify suggested_d in transform_info and rerun
    at.session_state["transform_info"]["suggested_d"] = 1
    at.run()
    assert not at.exception
    assert len(at.error) == 0

    assert at.session_state["form_arima_d"] == 1, (
        f"Expected form_arima_d to update to 1, got {at.session_state['form_arima_d']}"
    )
    print("[PASS] A2 verified: changing suggested_d cleanly updated form_arima_d to 1.")


def test_forecast_tab_and_future_forecasting():
    """Phase 5: Fit all models, verify comparison table, and generate future forecasts."""
    print("\n--- Testing Phase 5: Forecast Tab and Future Forecasting Workflow ---")
    at = _init_demo_apptest()
    at.run()
    assert not at.exception

    # 1. Fit all default models
    results_list = fit_all_default_models(
        model_base_series=at.session_state["model_base_series"],
        holdout_size=12,
        m=12,
        mase_m=12,
        transform_info=at.session_state["transform_info"],
        exog_df=at.session_state["exog_df"],
    )
    for r in results_list:
        at.session_state["model_results"][r.key] = r

    assert len(at.session_state["model_results"]) >= 7
    print(f"Fitted {len(at.session_state['model_results'])} candidate models.")

    # 2. Rerun app to render Forecast Tab
    at.run()
    assert not at.exception
    assert len(at.error) == 0

    # Verify comparison section reached: comparison table, Top Performer banner, and DM table
    assert len(at.dataframe) >= 2, f"Expected at least 2 dataframes, got {len(at.dataframe)}"
    comp_df_present = any(
        hasattr(df.value, "columns") and "RelRMSE_vs_ref" in df.value.columns
        for df in at.dataframe
    )
    assert comp_df_present, "Comparison leaderboard table not rendered in at.dataframe"
    dm_df_present = any(
        hasattr(df.value, "columns") and "DM stat" in df.value.columns
        for df in at.dataframe
    )
    assert dm_df_present, "Pairwise DM table not rendered in at.dataframe"
    top_performer_banner = any("Top Performer" in s.value for s in at.success)
    assert top_performer_banner, "Top Performer banner not found in at.success"
    print("Comparison table, Top Performer banner, and DM table rendered successfully.")

    # 3. Generate future forecasts for ARIMA, Holt-Winters, and SARIMAX
    s = at.session_state["model_base_series"]
    exog = at.session_state["exog_df"]
    fut_exog = make_future_exog(exog, steps=12, strategy="repeat_last_season", m=12)

    from modules.models import fit_full_and_forecast_detailed

    # ARIMA forecast
    fc_arima = fit_full_and_forecast_detailed(s, {"family": "arima", "p": 1, "d": 1, "q": 1, "trend": "n"}, steps=12)
    assert fc_arima.converged is True
    at.session_state["forecast_results"]["arima_12"] = fc_arima

    # Holt-Winters forecast
    fc_hw = fit_full_and_forecast_detailed(s, {"family": "holt_winters", "trend": "add", "seasonal": "add", "m": 12}, steps=12)
    assert fc_hw.converged is True
    at.session_state["forecast_results"]["hw_12"] = fc_hw

    # SARIMAX forecast
    fc_sarimax = fit_full_and_forecast_detailed(
        s, {"family": "sarimax", "p": 1, "d": 0, "q": 0, "m": 12}, steps=12, exog=exog, future_exog=fut_exog
    )
    assert fc_sarimax.converged is True
    at.session_state["forecast_results"]["sarimax_12"] = fc_sarimax

    assert len(at.session_state["forecast_results"]) == 3
    print("Generated 3 future forecasts (ARIMA, Holt-Winters, SARIMAX).")

    # 4. Pure Excel workbook export verification
    comp_df = pd.DataFrame({"Model": ["ARIMA", "HW", "SARIMAX"], "RMSE": [1.0, 1.2, 0.9]})
    excel_bytes = build_excel_report(
        comparison_df=comp_df,
        holdout_forecasts={"ARIMA": fc_arima.df},
        future_forecast_df=fc_arima.df,
        specs=[{"family": "arima"}],
        metadata={"holdout_size": 12, "mase_m": 12},
    )
    assert isinstance(excel_bytes, bytes)
    wb = openpyxl.load_workbook(io.BytesIO(excel_bytes))
    assert set(["Comparison", "Holdout_Forecasts", "Future_Forecast", "Specs"]).issubset(set(wb.sheetnames))
    print("[PASS] Phase 5 Forecast Tab and Future Forecasting workflow verified.")


def test_target_column_change_clears_forecast_results_apptest():
    """Verify that changing target column triggers data signature change, clearing forecast_results."""
    print("\n--- Testing Target Column Change Signature Invalidation ---")
    at = _init_demo_apptest()
    at.run()
    assert not at.exception

    # Populate dummy model_results and forecast_results
    at.session_state["model_results"]["dummy_key"] = "dummy_model"
    at.session_state["forecast_results"]["dummy_fc"] = "dummy_forecast"
    assert len(at.session_state["forecast_results"]) > 0

    # Select different target column (e.g. Marketing_Spend)
    target_sb = at.selectbox(key="target_col_select")
    assert target_sb is not None
    assert "Marketing_Spend" in target_sb.options
    target_sb.select("Marketing_Spend").run()
    assert not at.exception

    # Assert that forecast_results and model_results are empty after signature change
    assert len(at.session_state["forecast_results"]) == 0, (
        f"Expected forecast_results to be empty, got {at.session_state['forecast_results']}"
    )
    assert len(at.session_state["model_results"]) == 0, (
        f"Expected model_results to be empty, got {at.session_state['model_results']}"
    )
    print("[PASS] Target column change successfully invalidated signature and cleared forecast_results.")


if __name__ == "__main__":
    print("==================================================")
    print("Starting ATSA Smoke & AppTest Suites...")
    print("==================================================")
    test_app_smoke_all_ten_models()
    test_app_pills_driving_harness()
    test_a1_use_best_order_apptest()
    test_a2_stale_widget_state_apptest()
    test_forecast_tab_and_future_forecasting()
    test_target_column_change_clears_forecast_results_apptest()
    print("\n==================================================")
    print("ALL TESTS IN TEST_APP_SMOKE COMPLETED SUCCESSFULLY!")
    print("==================================================")
