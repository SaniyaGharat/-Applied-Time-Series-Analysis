"""
Export and Reporting Module for Applied Time Series Analysis (ATSA).

Pure Python module (no Streamlit dependencies).
Provides:
- Multi-sheet Excel workbook generator using openpyxl:
  Sheets: 'Comparison', 'Holdout_Forecasts', 'Future_Forecast', 'Specs'
"""

import io
import json
from typing import Any, Dict, List, Optional, Union
import pandas as pd


def build_excel_report(
    comparison_df: pd.DataFrame,
    holdout_forecasts: Optional[Dict[str, pd.DataFrame]] = None,
    future_forecast_df: Optional[pd.DataFrame] = None,
    specs: Optional[List[Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> bytes:
    """
    Generate a comprehensive multi-sheet Excel workbook containing:
    1. Comparison: Model metrics and ranking table.
    2. Holdout_Forecasts: Holdout predictions across candidate models.
    3. Future_Forecast: Forward projection mean and prediction intervals.
    4. Specs: JSON configuration, hyperparameters, transform steps, holdout size, MASE scale, date.

    Returns
    -------
    bytes
        Raw Excel workbook binary stream.
    """
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Sheet 1: Comparison
        if comparison_df is not None and not comparison_df.empty:
            comp_clean = comparison_df.copy()
            comp_clean.to_excel(writer, sheet_name="Comparison", index=False)
        else:
            pd.DataFrame({"Info": ["No comparison data available"]}).to_excel(
                writer, sheet_name="Comparison", index=False
            )

        # Sheet 2: Holdout_Forecasts
        if holdout_forecasts:
            holdout_cols = {}
            for label, fc_df in holdout_forecasts.items():
                if fc_df is not None and not fc_df.empty:
                    if "mean" in fc_df.columns:
                        holdout_cols[f"{label}_mean"] = fc_df["mean"]
                    if "lower" in fc_df.columns:
                        holdout_cols[f"{label}_lower"] = fc_df["lower"]
                    if "upper" in fc_df.columns:
                        holdout_cols[f"{label}_upper"] = fc_df["upper"]
            if holdout_cols:
                combined_holdout = pd.DataFrame(holdout_cols)
                combined_holdout.to_excel(writer, sheet_name="Holdout_Forecasts", index=True)
            else:
                pd.DataFrame({"Info": ["No holdout forecasts available"]}).to_excel(
                    writer, sheet_name="Holdout_Forecasts", index=False
                )
        else:
            pd.DataFrame({"Info": ["No holdout forecasts available"]}).to_excel(
                writer, sheet_name="Holdout_Forecasts", index=False
            )

        # Sheet 3: Future_Forecast
        if future_forecast_df is not None and not future_forecast_df.empty:
            future_forecast_df.to_excel(writer, sheet_name="Future_Forecast", index=True)
        else:
            pd.DataFrame({"Info": ["No future forecast available"]}).to_excel(
                writer, sheet_name="Future_Forecast", index=False
            )

        # Sheet 4: Specs
        specs_records: List[Dict[str, Any]] = []
        if specs:
            for s in specs:
                specs_records.append(
                    {
                        "Model": s.get("label", s.get("family", "Unknown")),
                        "Family": s.get("family", ""),
                        "Spec_JSON": json.dumps(s.get("spec", s), default=str),
                        "Transform_Steps": json.dumps(s.get("transform_steps", []), default=str),
                        "Holdout_Size": s.get("holdout_size", metadata.get("holdout_size") if metadata else None),
                        "MASE_m": s.get("mase_m", metadata.get("mase_m") if metadata else None),
                        "Run_Date": metadata.get("run_date") if metadata else str(pd.Timestamp.now()),
                    }
                )
        elif metadata:
            specs_records.append({
                "Model": "Metadata",
                "Spec_JSON": json.dumps(metadata, default=str),
                "Transform_Steps": json.dumps(metadata.get("transform_steps", []), default=str),
                "Holdout_Size": metadata.get("holdout_size"),
                "MASE_m": metadata.get("mase_m"),
                "Run_Date": metadata.get("run_date", str(pd.Timestamp.now())),
            })

        if specs_records:
            pd.DataFrame(specs_records).to_excel(writer, sheet_name="Specs", index=False)
        else:
            pd.DataFrame({"Info": ["No specs recorded"]}).to_excel(writer, sheet_name="Specs", index=False)

    return output.getvalue()


def build_dashboard_report(
    kpi_df: Optional[pd.DataFrame] = None,
    comparison_a: Optional[pd.DataFrame] = None,
    comparison_b: Optional[pd.DataFrame] = None,
    future_a: Optional[Union[pd.DataFrame, Any]] = None,
    future_b: Optional[Union[pd.DataFrame, Any]] = None,
    findings: Optional[List[str]] = None,
) -> bytes:
    """
    Generate an Excel workbook report for two-dataset comparison with 6 sheets:
    - KPIs: Comparative scale-free metrics and notes
    - Findings: Plain-English synthesized findings
    - Leaderboard_A: Model evaluation table for Dataset A
    - Leaderboard_B: Model evaluation table for Dataset B
    - Future_A: Future projection mean and prediction intervals for Dataset A
    - Future_B: Future projection mean and prediction intervals for Dataset B

    Returns
    -------
    bytes
        Raw Excel workbook binary stream loadable by openpyxl.
    """
    output = io.BytesIO()

    # Extract DataFrames if ForecastResult objects passed
    df_fa = future_a.df if hasattr(future_a, "df") else future_a
    df_fb = future_b.df if hasattr(future_b, "df") else future_b

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Sheet 1: KPIs
        if kpi_df is not None and not kpi_df.empty:
            kpi_df.to_excel(writer, sheet_name="KPIs", index=False)
        else:
            pd.DataFrame({"Info": ["No KPI data available"]}).to_excel(writer, sheet_name="KPIs", index=False)

        # Sheet 2: Findings
        if findings:
            findings_clean = [f.replace("**", "") for f in findings]
            pd.DataFrame({"Key Findings": findings_clean}).to_excel(writer, sheet_name="Findings", index=False)
        else:
            pd.DataFrame({"Key Findings": ["No comparative findings recorded"]}).to_excel(
                writer, sheet_name="Findings", index=False
            )

        # Sheet 3: Leaderboard_A
        if comparison_a is not None and not comparison_a.empty:
            comparison_a.to_excel(writer, sheet_name="Leaderboard_A", index=False)
        else:
            pd.DataFrame({"Info": ["No leaderboard data available for Dataset A"]}).to_excel(
                writer, sheet_name="Leaderboard_A", index=False
            )

        # Sheet 4: Leaderboard_B
        if comparison_b is not None and not comparison_b.empty:
            comparison_b.to_excel(writer, sheet_name="Leaderboard_B", index=False)
        else:
            pd.DataFrame({"Info": ["No leaderboard data available for Dataset B"]}).to_excel(
                writer, sheet_name="Leaderboard_B", index=False
            )

        # Sheet 5: Future_A
        if df_fa is not None and isinstance(df_fa, pd.DataFrame) and not df_fa.empty:
            df_fa.to_excel(writer, sheet_name="Future_A", index=True)
        else:
            pd.DataFrame({"Info": ["No future forecast available for Dataset A"]}).to_excel(
                writer, sheet_name="Future_A", index=False
            )

        # Sheet 6: Future_B
        if df_fb is not None and isinstance(df_fb, pd.DataFrame) and not df_fb.empty:
            df_fb.to_excel(writer, sheet_name="Future_B", index=True)
        else:
            pd.DataFrame({"Info": ["No future forecast available for Dataset B"]}).to_excel(
                writer, sheet_name="Future_B", index=False
            )

    return output.getvalue()


def build_column_comparison_report(
    leaderboard_df: pd.DataFrame,
    synthesis: str,
    criterion: str,
    category_winners: Dict[str, str],
    bundles: Optional[Dict[str, Any]] = None,
) -> bytes:
    """
    Generate a styled multi-sheet Excel report comparing multiple columns within a dataset/sheet.

    Sheets:
    1. Leaderboard: Overall rankings and scale-free metrics across all evaluated columns.
    2. Executive_Summary: Evaluation criteria, winner announcement, plain-English synthesis, and category winners.
    3. Future_Projections: Multi-step forward predictions across candidate columns.

    Returns
    -------
    bytes
        Raw Excel workbook binary stream.
    """
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Sheet 1: Leaderboard
        if leaderboard_df is not None and not leaderboard_df.empty:
            leaderboard_df.to_excel(writer, sheet_name="Leaderboard", index=False)
        else:
            pd.DataFrame({"Info": ["No leaderboard data available"]}).to_excel(
                writer, sheet_name="Leaderboard", index=False
            )

        # Sheet 2: Executive_Summary
        summary_rows = [
            {"Property": "Evaluation Criterion", "Value": criterion},
            {"Property": "Winning Column", "Value": str(category_winners.get("most_predictable", "N/A"))},
            {"Property": "Most Predictable Column (Lowest MASE)", "Value": str(category_winners.get("most_predictable", "N/A"))},
            {"Property": "Strongest Trend Column", "Value": str(category_winners.get("strongest_trend", "N/A"))},
            {"Property": "Most Stable Column (Lowest CV)", "Value": str(category_winners.get("most_stable", "N/A"))},
            {"Property": "Strongest Seasonality Column", "Value": str(category_winners.get("strongest_seasonality", "N/A"))},
            {"Property": "Executive Synthesis", "Value": synthesis.replace("**", "").replace("#", "")},
        ]
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Executive_Summary", index=False)

        # Sheet 3: Future_Projections
        future_frames = []
        if bundles:
            for col_name, bundle in bundles.items():
                fut = getattr(bundle, "future", None)
                if fut is not None and hasattr(fut, "df") and fut.df is not None and not fut.df.empty:
                    f_df = pd.DataFrame({
                        "Column": col_name,
                        "Date": fut.df.index,
                        "Mean_Forecast": fut.df["mean"].values if "mean" in fut.df else fut.df.iloc[:, 0].values,
                    })
                    if "lower" in fut.df:
                        f_df["Lower_95"] = fut.df["lower"].values
                    if "upper" in fut.df:
                        f_df["Upper_95"] = fut.df["upper"].values
                    future_frames.append(f_df)

        if future_frames:
            combined_future = pd.concat(future_frames, ignore_index=True)
            combined_future.to_excel(writer, sheet_name="Future_Projections", index=False)
        else:
            pd.DataFrame({"Info": ["No future forecast projections available"]}).to_excel(
                writer, sheet_name="Future_Projections", index=False
            )

    return output.getvalue()


