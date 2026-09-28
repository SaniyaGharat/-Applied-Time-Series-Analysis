"""
Export and Reporting Module for Applied Time Series Analysis (ATSA).

Pure Python module (no Streamlit dependencies).
Provides:
- Multi-sheet Excel workbook generator using openpyxl:
  Sheets: 'Comparison', 'Holdout_Forecasts', 'Future_Forecast', 'Specs'
"""

import io
import json
from typing import Any, Dict, List, Optional
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
