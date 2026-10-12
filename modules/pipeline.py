"""
Pure Analytical Pipeline and Dataset Comparison Engine for ATSA.

Pure Python module (zero Streamlit dependencies).
Provides:
- AnalysisBundle dataclass: comprehensive picklable container for a single time series dataset.
- run_analysis: automated full-pipeline runner (summary, diagnostics, differencing, model zoo, comparison, future forecast).
- compute_kpis: scale-free and relative KPI extractor.
- compare_datasets: cross-dataset comparative KPI table, plain-English findings, and overlap correlation.
"""

from dataclasses import dataclass, field
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from modules.compare import build_comparison_table
from modules.diagnostics import (
    adf_test,
    compute_acf_pacf,
    decompose,
    infer_period_from_frequency,
    kpss_test,
    stationarity_verdict,
    transform_series,
)
from modules.models import (
    ForecastResult,
    ModelResult,
    default_test_size,
    fit_full_and_forecast_detailed,
    fit_model,
    list_model_families,
)


@dataclass
class AnalysisBundle:
    """
    Standardized, fully picklable analytical bundle representing a single time series dataset.
    Contains no statsmodels or unpicklable runtime objects.
    """
    name: str
    series: pd.Series
    freq_label: str
    m: int
    holdout: int
    n_obs: int
    start: Any
    end: Any
    summary: Dict[str, Any]
    stationarity: Dict[str, Any]
    decomposition: Dict[str, Any]
    acf_pacf: Dict[str, Any]
    suggested_d: int
    suggested_D: int
    model_results: List[ModelResult]
    comparison: pd.DataFrame
    best_key: str
    best_label: str
    naive_rmse: float
    rel_rmse_best: float
    n_beating_naive: int
    n_models_fit: int
    n_failed: int
    future: Optional[ForecastResult]
    notes: List[str] = field(default_factory=list)


def run_analysis(
    series: pd.Series,
    name: str,
    m: Optional[int] = None,
    holdout: Optional[int] = None,
    exog: Optional[pd.DataFrame] = None,
    horizon: Optional[int] = None,
) -> AnalysisBundle:
    """
    Run full end-to-end analytical pipeline on a single time series dataset.

    Never raises unhandled exceptions to the caller; failures are recorded gracefully in notes.
    Reuses existing econometric diagnostics, model estimation zoo, and forecasting routines.

    Parameters
    ----------
    series : pd.Series
        Time series on original scale with DatetimeIndex or RangeIndex.
    name : str
        Human-readable name or label for the dataset.
    m : Optional[int], default=None
        Seasonal period. If None or <=1, inferred from frequency or defaults to 12.
    holdout : Optional[int], default=None
        Holdout test size for out-of-sample evaluation.
    exog : Optional[pd.DataFrame], default=None
        Optional aligned historical exogenous regressors (for SARIMAX).
    horizon : Optional[int], default=None
        Future multi-step projection steps. Defaults to m (or 12), capped at n // 2.

    Returns
    -------
    AnalysisBundle
        Structured, picklable analytical bundle.
    """
    notes: List[str] = []

    # 1. Clean series and check validity
    clean_s = series.dropna() if series is not None else pd.Series(dtype=float)
    n_obs = len(clean_s)

    # Date range and frequency
    if n_obs > 0:
        start_date = clean_s.index[0]
        end_date = clean_s.index[-1]
        freq_str = getattr(clean_s.index, "freqstr", None) or pd.infer_freq(clean_s.index)
        freq_label = str(freq_str) if freq_str else "Unknown"
    else:
        start_date = "N/A"
        end_date = "N/A"
        freq_label = "Unknown"

    # Infer or validate seasonal period m
    if m is None or int(m) < 1:
        inferred = infer_period_from_frequency(clean_s) if n_obs > 0 else None
        if inferred and inferred > 1:
            m = int(inferred)
        elif freq_label.upper().startswith("M") or "MONTH" in freq_label.upper():
            m = 12
        elif freq_label.upper().startswith("D") or "DAY" in freq_label.upper():
            m = 7
        elif freq_label.upper().startswith("Q") or "QUARTER" in freq_label.upper():
            m = 4
        elif freq_label.upper().startswith("H") or "HOUR" in freq_label.upper():
            m = 24
        else:
            m = 12 if n_obs >= 36 else 1
    m = max(1, int(m))

    # Holdout size
    if holdout is None or holdout <= 0:
        holdout = default_test_size(n_obs)
    else:
        holdout = min(int(holdout), max(1, n_obs // 2))

    # 2. Descriptive summary statistics
    if n_obs > 0:
        mean_val = float(clean_s.mean())
        std_val = float(clean_s.std(ddof=1)) if n_obs > 1 else 0.0
        cv_val = float(std_val / mean_val) if mean_val != 0.0 else np.nan
        min_val = float(clean_s.min())
        max_val = float(clean_s.max())
        missing_count = int(series.isna().sum()) if series is not None else 0
    else:
        mean_val, std_val, cv_val, min_val, max_val = 0.0, 0.0, np.nan, 0.0, 0.0
        missing_count = 0

    summary = {
        "mean": mean_val,
        "std": std_val,
        "cv": cv_val,
        "min": min_val,
        "max": max_val,
        "missing": missing_count,
    }

    # 3. Stationarity tests
    stationarity: Dict[str, Any] = {"adf": None, "kpss": None, "verdict": "Unknown"}
    suggested_d = 1
    if n_obs >= 10:
        try:
            adf_res = adf_test(clean_s)
            kpss_res = kpss_test(clean_s)
            verd = stationarity_verdict(adf_res, kpss_res)
            stationarity = {
                "adf": adf_res,
                "kpss": kpss_res,
                "verdict": verd.get("verdict", "Unknown"),
            }
            if verd.get("verdict") == "Stationary":
                suggested_d = 0
            else:
                suggested_d = 1
        except Exception as exc:
            notes.append(f"Stationarity testing could not be completed: {exc}")
            suggested_d = 1
    else:
        notes.append(f"Series has {n_obs} observations (< 10 required for ADF/KPSS testing).")

    # 4. Decomposition
    decomposition: Dict[str, Any] = {
        "trend_strength": 0.0,
        "seasonal_strength": 0.0,
        "period": m,
        "error": None,
    }
    suggested_D = 0
    if n_obs >= 2 * m and m > 1:
        try:
            decomp_out = decompose(clean_s, period=m)
            if decomp_out.get("error"):
                decomposition["error"] = decomp_out["error"]
                notes.append(f"Decomposition notice: {decomp_out['error']}")
            else:
                s_str = float(decomp_out.get("seasonal_strength", 0.0))
                t_str = float(decomp_out.get("trend_strength", 0.0))
                decomposition["seasonal_strength"] = s_str
                decomposition["trend_strength"] = t_str
                decomposition["period"] = decomp_out.get("period", m)
                if s_str > 0.6 and n_obs >= 3 * m:
                    suggested_D = 1
        except Exception as exc:
            decomposition["error"] = str(exc)
            notes.append(f"Decomposition failed: {exc}")
    else:
        err_msg = f"Insufficient observations ({n_obs}) for seasonal period m={m} (needs at least 2*m)."
        decomposition["error"] = err_msg
        notes.append(err_msg)

    # 5. ACF / PACF on differenced series
    acf_pacf: Dict[str, Any] = {}
    if n_obs >= 10:
        try:
            diff_s, _ = transform_series(
                clean_s,
                diff_order=suggested_d,
                seasonal_diff_order=suggested_D,
                seasonal_period=m if suggested_D > 0 else None,
            )
            nlags = min(40, max(2, len(diff_s) // 2 - 1))
            acf_pacf = compute_acf_pacf(diff_s, nlags=nlags)
        except Exception as exc:
            notes.append(f"ACF/PACF computation failed: {exc}")
    else:
        notes.append("ACF/PACF skipped due to short series.")

    # 6. Model Fitting Zoo
    model_results: List[ModelResult] = []
    if n_obs < 30:
        notes.append(f"Series has only {n_obs} observations (< 30 required for reliable statistical estimation).")
        # For short series, still fit baseline models (naive, drift) if holdout valid
        families_to_fit = [f for f in list_model_families() if f["key"] in ("naive", "drift")]
    else:
        families_to_fit = list_model_families()

    # Model skip rules matching fit_all_default_models
    for fam in families_to_fit:
        fam_key = fam["key"]
        spec = dict(fam["default_spec"])

        # Skip SARIMAX if no exogenous features
        if fam_key == "sarimax" and (exog is None or exog.empty):
            continue

        # Skip seasonal families if m < 2 or series too short
        if fam_key in ["seasonal_naive", "sarima", "holt_winters"] and (m < 2 or n_obs < 2 * m):
            continue

        # Apply suggested differencing orders
        if fam_key in ["ar", "ma", "arma", "arima", "sarima", "sarimax"]:
            spec["d"] = suggested_d
        if fam_key in ["sarima", "sarimax"]:
            spec["D"] = suggested_D
            spec["m"] = int(m)
        if fam_key in ["seasonal_naive", "holt_winters"]:
            spec["m"] = int(m)

        try:
            res = fit_model(
                y=clean_s,
                spec=spec,
                test_size=holdout,
                transform_info=None,
                exog=exog if fam_key == "sarimax" else None,
                m=m,
                mase_m=m,  # Each bundle uses mase_m = m of its OWN dataset
            )
            model_results.append(res)
        except Exception as exc:
            notes.append(f"Model '{fam.get('label', fam_key)}' failed to estimate: {exc}")

    # 7. Comparison Table & Leaderboard (Ranked by MASE)
    comp_df = build_comparison_table(model_results, rank_by="mase")
    n_models_fit = len(model_results)
    n_failed = sum(1 for r in model_results if not r.converged)

    best_key = ""
    best_label = "None"
    naive_rmse = np.nan
    rel_rmse_best = np.nan
    n_beating_naive = 0

    if not comp_df.empty:
        # Locate Naive model RMSE
        naive_rows = comp_df[comp_df["Family"] == "naive"]
        if not naive_rows.empty:
            naive_rmse = float(naive_rows["RMSE"].iloc[0])

        # Find Rank 1 model
        converged_ranked = comp_df[comp_df["Converged"] == True]
        if not converged_ranked.empty:
            top_row = converged_ranked.iloc[0]
            best_label = str(top_row["Model"])
            # Match back to ModelResult for best_key
            for r in model_results:
                if r.label == best_label:
                    best_key = r.key
                    break
            if not best_key:
                best_key = model_results[0].key

            rel_rmse_best = float(top_row.get("RelRMSE_vs_ref", np.nan))
        elif model_results:
            best_key = model_results[0].key
            best_label = model_results[0].label

        # Count models beating naive
        if not np.isnan(naive_rmse) and naive_rmse > 0:
            for _, r_row in comp_df.iterrows():
                if r_row.get("Converged") and float(r_row.get("RelRMSE_vs_ref", 999.0)) < 1.0:
                    n_beating_naive += 1

    # 8. Future Forecast for Best Converged Non-SARIMAX Model
    future_result: Optional[ForecastResult] = None
    fc_horizon = horizon if (horizon is not None and horizon > 0) else m
    fc_horizon = min(int(fc_horizon), max(1, n_obs // 2))

    # Find best converged non-sarimax model
    best_candidate_res: Optional[ModelResult] = None
    if not comp_df.empty:
        for _, row in comp_df.iterrows():
            if row.get("Converged") and row.get("Family") != "sarimax":
                # Find matching result
                for r in model_results:
                    if r.label == row.get("Model"):
                        best_candidate_res = r
                        break
                if best_candidate_res:
                    break

    if best_candidate_res is None and model_results:
        # Fallback to any converged model or first model
        for r in model_results:
            if r.converged and r.family != "sarimax":
                best_candidate_res = r
                break
        if best_candidate_res is None and model_results:
            best_candidate_res = model_results[0]

    if best_candidate_res is not None and n_obs >= 5:
        try:
            future_result = fit_full_and_forecast_detailed(
                y=clean_s,
                spec=best_candidate_res.spec,
                steps=fc_horizon,
                transform_info=None,
                exog=None,
                future_exog=None,
                m=m,
            )
        except Exception as exc:
            notes.append(f"Future forecast projection failed for {best_candidate_res.label}: {exc}")
            future_result = ForecastResult(
                df=pd.DataFrame(),
                label=best_candidate_res.label,
                spec=best_candidate_res.spec,
                converged=False,
                warnings=[str(exc)],
                notes=f"Future forecasting failed: {exc}",
                fit_seconds=0.0,
                n_train=n_obs,
                last_train_date=end_date,
                transform_steps=[],
            )

    return AnalysisBundle(
        name=name,
        series=series,
        freq_label=freq_label,
        m=m,
        holdout=holdout,
        n_obs=n_obs,
        start=start_date,
        end=end_date,
        summary=summary,
        stationarity=stationarity,
        decomposition=decomposition,
        acf_pacf=acf_pacf,
        suggested_d=suggested_d,
        suggested_D=suggested_D,
        model_results=model_results,
        comparison=comp_df,
        best_key=best_key,
        best_label=best_label,
        naive_rmse=naive_rmse,
        rel_rmse_best=rel_rmse_best,
        n_beating_naive=n_beating_naive,
        n_models_fit=n_models_fit,
        n_failed=n_failed,
        future=future_result,
        notes=notes,
    )


def compute_kpis(bundle: AnalysisBundle) -> Dict[str, Any]:
    """
    Compute a dictionary of scale-free and relative KPIs for a single AnalysisBundle.

    Parameters
    ----------
    bundle : AnalysisBundle
        Evaluated dataset bundle.

    Returns
    -------
    Dict[str, Any]
        Dictionary of standardized scale-free KPIs.
    """
    comp_df = bundle.comparison
    best_row = comp_df[comp_df["Model"] == bundle.best_label] if not comp_df.empty else pd.DataFrame()

    best_mase = float(best_row["MASE"].iloc[0]) if not best_row.empty else np.nan
    best_smape = float(best_row["sMAPE"].iloc[0]) if not best_row.empty else np.nan
    cov95 = float(best_row["Coverage95"].iloc[0]) if (not best_row.empty and "Coverage95" in best_row.columns) else np.nan
    raw_rmse = float(best_row["RMSE"].iloc[0]) if not best_row.empty else np.nan
    raw_mae = float(best_row["MAE"].iloc[0]) if not best_row.empty else np.nan

    # Forecast change %: last forecast mean vs last observed
    forecast_change_pct = np.nan
    if bundle.future and bundle.future.df is not None and not bundle.future.df.empty:
        if "mean" in bundle.future.df.columns and len(bundle.series.dropna()) > 0:
            last_obs = float(bundle.series.dropna().iloc[-1])
            last_fc = float(bundle.future.df["mean"].iloc[-1])
            if last_obs != 0.0:
                calc_pct = ((last_fc - last_obs) / abs(last_obs)) * 100.0
                forecast_change_pct = float(calc_pct) if np.isfinite(calc_pct) else 0.0
            else:
                forecast_change_pct = 0.0

    # Mean interval width %: avg (upper - lower) / abs(mean forecast) * 100
    mean_interval_width_pct = np.nan
    if bundle.future and bundle.future.df is not None and not bundle.future.df.empty:
        fc_df = bundle.future.df
        if "upper" in fc_df.columns and "lower" in fc_df.columns and "mean" in fc_df.columns:
            denom = fc_df["mean"].abs().replace(0, np.nan)
            widths = (fc_df["upper"] - fc_df["lower"]) / denom * 100.0
            val_mean = widths.dropna().mean()
            mean_interval_width_pct = float(val_mean) if np.isfinite(val_mean) else 0.0

    date_range_str = f"{str(bundle.start)[:10]} to {str(bundle.end)[:10]}"

    return {
        "dataset_name": bundle.name,
        "n_obs": bundle.n_obs,
        "date range": date_range_str,
        "frequency": bundle.freq_label,
        "cv": bundle.summary.get("cv", np.nan),
        "trend_strength": bundle.decomposition.get("trend_strength", 0.0),
        "seasonal_strength": bundle.decomposition.get("seasonal_strength", 0.0),
        "stationarity verdict": bundle.stationarity.get("verdict", "Unknown"),
        "best model label": bundle.best_label,
        "best MASE": best_mase,
        "best sMAPE": best_smape,
        "RelRMSE vs naive": bundle.rel_rmse_best,
        "n_beating_naive/n_models_fit": f"{bundle.n_beating_naive}/{bundle.n_models_fit}",
        "forecast_change_pct": forecast_change_pct,
        "mean_interval_width_pct": mean_interval_width_pct,
        "coverage95": cov95,
        "raw_rmse": raw_rmse,
        "raw_mae": raw_mae,
        "_series": bundle.series,
    }


def compare_datasets(
    kpis_a: Union[Dict[str, Any], AnalysisBundle],
    kpis_b: Union[Dict[str, Any], AnalysisBundle],
    series_a: Optional[pd.Series] = None,
    series_b: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Synthesize comparative KPI table and plain-English findings between two datasets.

    Raw RMSE and MAE are explicitly flagged as non-comparable across datasets.
    If date overlap >= 24 observations, calculates first-difference correlation (descriptive, not causal).

    Parameters
    ----------
    kpis_a : Dict[str, Any] or AnalysisBundle
        KPI dictionary or bundle for Dataset A.
    kpis_b : Dict[str, Any] or AnalysisBundle
        KPI dictionary or bundle for Dataset B.
    series_a : Optional[pd.Series]
        Dataset A time series for date alignment check.
    series_b : Optional[pd.Series]
        Dataset B time series for date alignment check.

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        Comparative DataFrame with columns ['KPI', 'A', 'B', 'note'] and list of plain-English findings.
    """
    # Convert bundles to KPI dicts if passed directly
    dict_a = compute_kpis(kpis_a) if isinstance(kpis_a, AnalysisBundle) else dict(kpis_a)
    dict_b = compute_kpis(kpis_b) if isinstance(kpis_b, AnalysisBundle) else dict(kpis_b)

    s_a = series_a if series_a is not None else dict_a.get("_series")
    s_b = series_b if series_b is not None else dict_b.get("_series")

    name_a = dict_a.get("dataset_name", "Dataset A")
    name_b = dict_b.get("dataset_name", "Dataset B")

    findings: List[str] = []

    # Format helpers
    def fmt_num(v: Any, digits: int = 2, pct: bool = False) -> str:
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            return "N/A"
        try:
            val = float(v)
            return f"{val:.{digits}f}%" if pct else f"{val:.{digits}f}"
        except Exception:
            return str(v)

    # 1. Comparative rows
    rows = [
        {"KPI": "Observations", "A": str(dict_a.get("n_obs", "N/A")), "B": str(dict_b.get("n_obs", "N/A")), "note": "Sample size"},
        {"KPI": "Date Range", "A": str(dict_a.get("date range", "N/A")), "B": str(dict_b.get("date range", "N/A")), "note": "Historical time span"},
        {"KPI": "Frequency", "A": str(dict_a.get("frequency", "N/A")), "B": str(dict_b.get("frequency", "N/A")), "note": "Sampling interval"},
        {"KPI": "Coefficient of Variation (CV)", "A": fmt_num(dict_a.get("cv"), 3), "B": fmt_num(dict_b.get("cv"), 3), "note": "Relative dispersion (std / mean)"},
        {"KPI": "Stationarity Verdict", "A": str(dict_a.get("stationarity verdict", "N/A")), "B": str(dict_b.get("stationarity verdict", "N/A")), "note": "Joint ADF & KPSS synthesis"},
        {"KPI": "Trend Strength", "A": fmt_num(dict_a.get("trend_strength"), 3), "B": fmt_num(dict_b.get("trend_strength"), 3), "note": "Hyndman strength F_t in [0, 1]"},
        {"KPI": "Seasonal Strength", "A": fmt_num(dict_a.get("seasonal_strength"), 3), "B": fmt_num(dict_b.get("seasonal_strength"), 3), "note": "Hyndman strength F_s in [0, 1]"},
        {"KPI": "Top Performing Model", "A": str(dict_a.get("best model label", "N/A")), "B": str(dict_b.get("best model label", "N/A")), "note": "Lowest holdout MASE winner"},
        {"KPI": "Best Model MASE", "A": fmt_num(dict_a.get("best MASE"), 3), "B": fmt_num(dict_b.get("best MASE"), 3), "note": "Scale-free error (< 1 beats naive)"},
        {"KPI": "Best Model sMAPE", "A": fmt_num(dict_a.get("best sMAPE"), 2, pct=True), "B": fmt_num(dict_b.get("best sMAPE"), 2, pct=True), "note": "Symmetric percentage error"},
        {"KPI": "RelRMSE vs Naive", "A": fmt_num(dict_a.get("RelRMSE vs naive"), 3), "B": fmt_num(dict_b.get("RelRMSE vs naive"), 3), "note": "Relative RMSE benchmark (< 1.0 beats naive)"},
        {"KPI": "Models Beating Naive", "A": str(dict_a.get("n_beating_naive/n_models_fit", "N/A")), "B": str(dict_b.get("n_beating_naive/n_models_fit", "N/A")), "note": "Proportion of candidate zoo beating naive"},
        {"KPI": "Holdout 95% Coverage", "A": fmt_num(dict_a.get("coverage95"), 1, pct=True), "B": fmt_num(dict_b.get("coverage95"), 1, pct=True), "note": "Empirical coverage on holdout set"},
        {"KPI": "Forecast Change %", "A": fmt_num(dict_a.get("forecast_change_pct"), 2, pct=True), "B": fmt_num(dict_b.get("forecast_change_pct"), 2, pct=True), "note": "Projected end vs last observed (%)"},
        {"KPI": "Mean Interval Width %", "A": fmt_num(dict_a.get("mean_interval_width_pct"), 2, pct=True), "B": fmt_num(dict_b.get("mean_interval_width_pct"), 2, pct=True), "note": "Avg 95% interval relative width"},
        {"KPI": "Raw Best RMSE", "A": fmt_num(dict_a.get("raw_rmse"), 2), "B": fmt_num(dict_b.get("raw_rmse"), 2), "note": "different units, not comparable"},
        {"KPI": "Raw Best MAE", "A": fmt_num(dict_a.get("raw_mae"), 2), "B": fmt_num(dict_b.get("raw_mae"), 2), "note": "different units, not comparable"},
    ]

    # 2. Date Overlap & Difference Correlation Check
    overlap_count = 0
    diff_corr = np.nan
    if s_a is not None and s_b is not None:
        try:
            clean_sa = s_a.dropna()
            clean_sb = s_b.dropna()
            common_idx = clean_sa.index.intersection(clean_sb.index)
            overlap_count = len(common_idx)
            if overlap_count >= 24:
                diff_a = clean_sa.loc[common_idx].diff().dropna()
                diff_b = clean_sb.loc[common_idx].diff().dropna()
                diff_corr = float(diff_a.corr(diff_b))
                rows.append({
                    "KPI": "First-Difference Correlation",
                    "A": f"{diff_corr:.3f}",
                    "B": f"{diff_corr:.3f}",
                    "note": f"descriptive only, not causal ({overlap_count} overlapping dates)",
                })
        except Exception:
            pass

    kpi_df = pd.DataFrame(rows)

    # 3. Plain-English Findings Synthesis
    s_str_a = dict_a.get("seasonal_strength") or 0.0
    s_str_b = dict_b.get("seasonal_strength") or 0.0
    if abs(s_str_a - s_str_b) >= 0.15:
        more_seas = name_a if s_str_a > s_str_b else name_b
        less_seas = name_b if s_str_a > s_str_b else name_a
        findings.append(f"**Seasonality**: {more_seas} exhibits significantly stronger seasonality ({max(s_str_a, s_str_b):.2f}) than {less_seas} ({min(s_str_a, s_str_b):.2f}).")
    elif max(s_str_a, s_str_b) > 0.6:
        findings.append(f"**Seasonality**: Both datasets display strong seasonal patterns (F_s = {s_str_a:.2f} and {s_str_b:.2f}).")
    else:
        findings.append(f"**Seasonality**: Neither dataset exhibits dominant seasonal cycles.")

    t_str_a = dict_a.get("trend_strength") or 0.0
    t_str_b = dict_b.get("trend_strength") or 0.0
    if abs(t_str_a - t_str_b) >= 0.15:
        more_trend = name_a if t_str_a > t_str_b else name_b
        less_trend = name_b if t_str_a > t_str_b else name_a
        findings.append(f"**Trend**: {more_trend} has a more prominent deterministic/stochastic trend (F_t = {max(t_str_a, t_str_b):.2f}) compared to {less_trend} (F_t = {min(t_str_a, t_str_b):.2f}).")

    m_a = dict_a.get("best model label", "Unknown")
    m_b = dict_b.get("best model label", "Unknown")
    if m_a != m_b:
        findings.append(f"**Model Selection**: Heterogeneous dynamics favored **{m_a}** on {name_a}, whereas **{m_b}** won on {name_b}.")
    else:
        findings.append(f"**Model Selection**: **{m_a}** delivered optimal holdout performance across both datasets.")

    mase_a = dict_a.get("best MASE")
    mase_b = dict_b.get("best MASE")
    if mase_a is not None and mase_b is not None and not (np.isnan(mase_a) or np.isnan(mase_b)):
        better_mase_name = name_a if mase_a < mase_b else name_b
        findings.append(f"**Forecastability**: {better_mase_name} achieved lower relative scale-free forecast error (best MASE = {min(mase_a, mase_b):.3f} vs {max(mase_a, mase_b):.3f}).")

    verd_a = dict_a.get("stationarity verdict", "Unknown")
    verd_b = dict_b.get("stationarity verdict", "Unknown")
    if verd_a == verd_b:
        findings.append(f"**Stationarity**: Both series share identical stationarity properties ({verd_a}).")
    else:
        findings.append(f"**Stationarity**: Differing unit root behavior ({name_a}: {verd_a}; {name_b}: {verd_b}).")

    if overlap_count >= 24 and not np.isnan(diff_corr):
        findings.append(
            f"**Temporal Co-movement**: First-difference correlation over {overlap_count} overlapping periods is r = {diff_corr:.3f} (descriptive co-movement only; not causal)."
        )

    # 4. Box-Jenkins / ARIMA Zoo Comparison Finding
    if isinstance(kpis_a, AnalysisBundle) and isinstance(kpis_b, AnalysisBundle):
        bj_families = {"ar", "ma", "arma", "arima", "sarima"}
        bj_a = [r for r in kpis_a.model_results if r.family in bj_families and r.metrics_test.get("mase") is not None and not np.isnan(r.metrics_test.get("mase"))]
        bj_b = [r for r in kpis_b.model_results if r.family in bj_families and r.metrics_test.get("mase") is not None and not np.isnan(r.metrics_test.get("mase"))]
        if bj_a and bj_b:
            best_bj_a = min(bj_a, key=lambda x: x.metrics_test.get("mase", float("inf")))
            best_bj_b = min(bj_b, key=lambda x: x.metrics_test.get("mase", float("inf")))
            findings.append(
                f"**Box-Jenkins Zoo (AR, MA, ARMA, ARIMA, SARIMA)**: Among classical Box-Jenkins architectures, "
                f"**{best_bj_a.label}** performed best on {name_a} (MASE: {best_bj_a.metrics_test['mase']:.3f}), "
                f"while **{best_bj_b.label}** was optimal on {name_b} (MASE: {best_bj_b.metrics_test['mase']:.3f})."
            )

    return kpi_df, findings


def build_arima_family_comparison(
    bundle_a: AnalysisBundle,
    bundle_b: AnalysisBundle,
) -> pd.DataFrame:
    """
    Build structured side-by-side comparison specifically for Box-Jenkins / ARIMA models:
    AR, MA, ARMA, ARIMA, and SARIMA.

    Returns DataFrame with columns:
    [
        'Family',
        'Model A', 'Converged A', 'MASE A', 'sMAPE A (%)', 'AIC A',
        'Model B', 'Converged B', 'MASE B', 'sMAPE B (%)', 'AIC B',
        'Top Performer', 'Insights'
    ]
    """
    families = [
        ("ar", "AR (Autoregressive)", "Linear autoregression on p lags"),
        ("ma", "MA (Moving Average)", "Invertible moving average on q error shocks"),
        ("arma", "ARMA (Autoregressive Moving Average)", "Mixed stationary AR(p) + MA(q) process"),
        ("arima", "ARIMA (Integrated ARMA)", "ARMA on d-differenced non-stationary series"),
        ("sarima", "SARIMA (Seasonal ARIMA)", "Multiplicative seasonal (P, D, Q)_s structure"),
    ]

    map_a = {r.family: r for r in bundle_a.model_results}
    map_b = {r.family: r for r in bundle_b.model_results}

    rows = []
    for fam_key, fam_title, role_desc in families:
        r_a = map_a.get(fam_key)
        r_b = map_b.get(fam_key)

        mase_a = r_a.metrics_test.get("mase") if r_a and r_a.metrics_test else np.nan
        mase_b = r_b.metrics_test.get("mase") if r_b and r_b.metrics_test else np.nan
        smape_a = r_a.metrics_test.get("smape") if r_a and r_a.metrics_test else np.nan
        smape_b = r_b.metrics_test.get("smape") if r_b and r_b.metrics_test else np.nan

        ic_a = (r_a.information_criteria or {}).get("aic") if r_a else None
        ic_b = (r_b.information_criteria or {}).get("aic") if r_b else None

        if not np.isnan(mase_a) and not np.isnan(mase_b):
            winner = bundle_a.name if mase_a < mase_b else bundle_b.name
        elif not np.isnan(mase_a):
            winner = bundle_a.name
        elif not np.isnan(mase_b):
            winner = bundle_b.name
        else:
            winner = "N/A"

        rows.append({
            "Family": fam_title,
            f"{bundle_a.name} Model": r_a.label if r_a else "Not Fitted",
            f"{bundle_a.name} Converged": "✅ Yes" if (r_a and r_a.converged) else ("❌ No" if r_a else "N/A"),
            f"{bundle_a.name} MASE": round(mase_a, 3) if not np.isnan(mase_a) else "N/A",
            f"{bundle_a.name} sMAPE (%)": f"{smape_a:.2f}%" if not np.isnan(smape_a) else "N/A",
            f"{bundle_a.name} AIC": f"{ic_a:.1f}" if ic_a is not None and not np.isnan(ic_a) else "N/A",
            f"{bundle_b.name} Model": r_b.label if r_b else "Not Fitted",
            f"{bundle_b.name} Converged": "✅ Yes" if (r_b and r_b.converged) else ("❌ No" if r_b else "N/A"),
            f"{bundle_b.name} MASE": round(mase_b, 3) if not np.isnan(mase_b) else "N/A",
            f"{bundle_b.name} sMAPE (%)": f"{smape_b:.2f}%" if not np.isnan(smape_b) else "N/A",
            f"{bundle_b.name} AIC": f"{ic_b:.1f}" if ic_b is not None and not np.isnan(ic_b) else "N/A",
            "Top Performer (Lower MASE)": winner,
            "Insights": role_desc,
        })

    return pd.DataFrame(rows)

