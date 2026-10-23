"""
Model Comparison and Future Exogenous Synthesis Module for Applied Time Series Analysis (ATSA).

Pure Python module (no Streamlit dependencies).
Provides:
- Information Criteria Grouping (IC_group)
- Standardized Model Comparison Table with ranking and relative metrics
- Equal-Weight Ensemble Combination
- Pairwise Diebold-Mariano hypothesis testing
- Future Exogenous Feature generators (repeat last, seasonal repeat, rolling mean, linear extrapolation)
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from modules.metrics import (
    accuracy_table,
    diebold_mariano,
    interval_coverage,
    mean_interval_width,
    relative_metric,
)
from modules.models import ModelResult


def ic_group(result: Union[ModelResult, Dict[str, Any]]) -> str:
    """
    Determine the Information Criteria comparison group for a model.
    Information criteria (AIC/BIC) are comparable ONLY within the same IC_group
    and never across different differencing orders or model classes.

    Parameters
    ----------
    result : ModelResult or dict
        Fitted model result container or spec.

    Returns
    -------
    str
        "ARIMA-class d=<d>,D=<D>,trend=<t>", "ETS", or "n/a" for baselines.
    """
    family = getattr(result, "family", None)
    spec = getattr(result, "spec", {})
    if family is None and isinstance(result, dict):
        family = result.get("family", "")
        spec = result.get("spec", result)

    family_clean = str(family).lower()

    if family_clean in ("naive", "seasonal_naive", "drift", "combination"):
        return "n/a"

    if family_clean == "holt_winters":
        return "ETS"

    if family_clean in ("ar", "ma", "arma", "arima", "sarima", "sarimax"):
        d = spec.get("d", 0)
        D = spec.get("D", 0)
        trend = spec.get("trend")
        if trend is None:
            trend = "c" if (d == 0 and D == 0) else "n"
        return f"ARIMA-class d={d},D={D},trend={trend}"

    return "n/a"


METRIC_COLUMN_MAP = {
    "RMSE": "RMSE",
    "MAE": "MAE",
    "MAPE": "MAPE",
    "SMAPE": "sMAPE",
    "MASE": "MASE",
}


def map_metric_column(metric: str) -> str:
    """Standardize user metric name to comparison table column name."""
    return METRIC_COLUMN_MAP.get(str(metric).strip().upper(), "RMSE")


def build_comparison_table(
    results: List[ModelResult],
    rank_by: str = "rmse",
    reference_family: str = "naive",
) -> pd.DataFrame:
    """
    Build standardized multi-model forecast comparison table.

    Columns
    -------
    Model, Family, Converged, RMSE, MAE, MAPE, sMAPE, MASE,
    Coverage95, AvgWidth, RelRMSE_vs_ref, AIC, BIC, IC_group,
    LB_min_p, JB_p, FitSec, Rank

    Ranked ascending by rank_by (lower is better); failed models are unranked and placed last.
    """
    if not results:
        cols = [
            "Model", "Family", "Converged", "RMSE", "MAE", "MAPE", "sMAPE", "MASE",
            "Coverage95", "AvgWidth", "RelRMSE_vs_ref", "AIC", "BIC", "IC_group",
            "LB_min_p", "JB_p", "FitSec", "Rank",
        ]
        return pd.DataFrame(columns=cols)

    # Find reference model RMSE for relative metric
    ref_rmse = np.nan
    for r in results:
        if getattr(r, "family", "").lower() == reference_family.lower() and getattr(r, "converged", False):
            m_test = getattr(r, "metrics_test", {}) or {}
            val = m_test.get("rmse")
            if val is not None and not np.isnan(val) and val > 0:
                ref_rmse = float(val)
                break

    rows: List[Dict[str, Any]] = []
    for r in results:
        label = getattr(r, "label", getattr(r, "key", "Unknown"))
        family = getattr(r, "family", "")
        converged = bool(getattr(r, "converged", False))
        m_test = getattr(r, "metrics_test", {}) or {}

        rmse_val = float(m_test.get("rmse", np.nan))
        mae_val = float(m_test.get("mae", np.nan))
        mape_val = float(m_test.get("mape", np.nan))
        smape_val = float(m_test.get("smape", np.nan))
        mase_val = float(m_test.get("mase", np.nan))

        # Prediction intervals metrics
        cov_val = np.nan
        width_val = np.nan
        fc_df = getattr(r, "test_forecast", None)
        test_act = getattr(r, "test_actual", None)
        if fc_df is not None and isinstance(fc_df, pd.DataFrame):
            if "lower" in fc_df and "upper" in fc_df:
                width_val = mean_interval_width(fc_df["lower"], fc_df["upper"])
                if test_act is not None and len(test_act) > 0:
                    cov_val = interval_coverage(test_act, fc_df["lower"], fc_df["upper"])

        rel_rmse = relative_metric(rmse_val, ref_rmse)

        # Information criteria
        ic = getattr(r, "information_criteria", None) or {}
        aic_val = ic.get("aic")
        bic_val = ic.get("bic")
        group = ic_group(r)

        # Ljung-Box min p-value
        lb_df = getattr(r, "ljung_box", None)
        lb_min_p = np.nan
        if lb_df is not None and isinstance(lb_df, pd.DataFrame) and not lb_df.empty:
            p_cols = [c for c in lb_df.columns if "p-value" in c.lower() or "lb_pvalue" in c.lower()]
            if p_cols:
                lb_min_p = float(lb_df[p_cols[0]].dropna().min())

        # Jarque-Bera normality p-value
        norm_res = getattr(r, "normality", {}) or {}
        jb_p = float(norm_res.get("pvalue", np.nan))

        fit_sec = float(getattr(r, "fit_seconds", 0.0))

        rows.append(
            {
                "Key": getattr(r, "key", label),
                "Model": label,
                "Family": family,
                "Converged": converged,
                "RMSE": rmse_val,
                "MAE": mae_val,
                "MAPE": mape_val,
                "sMAPE": smape_val,
                "MASE": mase_val,
                "Coverage95": cov_val,
                "AvgWidth": width_val,
                "RelRMSE_vs_ref": rel_rmse,
                "AIC": float(aic_val) if aic_val is not None else np.nan,
                "BIC": float(bic_val) if bic_val is not None else np.nan,
                "IC_group": group,
                "LB_min_p": lb_min_p,
                "JB_p": jb_p,
                "FitSec": round(fit_sec, 3),
            }
        )

    df = pd.DataFrame(rows)

    # Sort & Rank by requested metric (case-insensitive) using shared column mapping
    target_col = map_metric_column(rank_by)

    # Split into converged and non-converged
    conv_mask = df["Converged"] == True
    df_conv = df[conv_mask].copy()
    df_failed = df[~conv_mask].copy()

    if not df_conv.empty:
        df_conv = df_conv.sort_values(by=target_col, ascending=True).reset_index(drop=True)
        df_conv["Rank"] = np.arange(1, len(df_conv) + 1, dtype=int)
    else:
        df_conv["Rank"] = pd.Series(dtype=int)

    if not df_failed.empty:
        df_failed["Rank"] = np.nan

    out_df = pd.concat([df_conv, df_failed], ignore_index=True)
    return out_df


def equal_weight_combination(results: List[ModelResult]) -> Optional[ModelResult]:
    """
    Construct equal-weight combination model result from converged models.
    Mean forecast = arithmetic average of component models' test forecasts.
    No prediction intervals (Coverage and AvgWidth are NaN).
    """
    converged = [
        r for r in results
        if getattr(r, "converged", False)
        and getattr(r, "test_forecast", None) is not None
        and "mean" in r.test_forecast
        and getattr(r, "family", "") != "combination"
    ]
    n_models = len(converged)
    if n_models == 0:
        return None

    # Compute mean forecast
    all_means = [r.test_forecast["mean"] for r in converged]
    mean_forecast = pd.concat(all_means, axis=1).mean(axis=1)

    test_idx = converged[0].test_index
    mean_df = pd.DataFrame(
        {
            "mean": mean_forecast.values,
            "lower": np.nan,
            "upper": np.nan,
        },
        index=test_idx,
    )

    # Compute metrics against test_actual if available
    test_actual = None
    for r in converged:
        act = getattr(r, "test_actual", None)
        if act is not None and len(act) > 0:
            test_actual = act
            break

    # Retrieve mase_scale from member models' extras
    mase_scale_val = None
    for r in converged:
        if isinstance(getattr(r, "extras", None), dict) and "mase_scale" in r.extras:
            mase_scale_val = r.extras["mase_scale"]
            break

    if test_actual is not None:
        metrics = accuracy_table(test_actual, mean_forecast, scale=mase_scale_val)
    else:
        metrics = {
            "rmse": np.nan,
            "mae": np.nan,
            "mape": np.nan,
            "smape": np.nan,
            "mase": np.nan,
        }

    label = f"Equal-weight combination ({n_models} models)"
    key = f"comb_ew_{n_models}"

    return ModelResult(
        key=key,
        label=label,
        family="combination",
        spec={"family": "combination", "n_models": n_models},
        train_index=converged[0].train_index,
        test_index=test_idx,
        fitted_train=pd.Series(dtype=float, index=converged[0].train_index),
        test_forecast=mean_df,
        residuals=pd.Series(dtype=float),
        metrics_test=metrics,
        metrics_train={"rmse": np.nan, "mae": np.nan},
        information_criteria=None,
        params_table=None,
        summary_text=f"Simple equal-weight forecast combination across {n_models} models.",
        ljung_box=None,
        normality={"jb_stat": np.nan, "pvalue": np.nan},
        warnings=[],
        converged=True,
        fit_seconds=0.0,
        notes=f"Arithmetic average of forecasts from: {', '.join([r.label for r in converged])}.",
        extras={"mase_scale": mase_scale_val} if mase_scale_val is not None else {},
        test_actual=test_actual,
    )


def pairwise_dm(
    results: List[ModelResult],
    reference_key: Optional[str] = None,
    power: int = 2,
    against_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute pairwise Diebold-Mariano test comparing each model against a reference model.

    Parameters
    ----------
    results : list of ModelResult
        Candidate models to evaluate.
    reference_key : Optional[str], default=None
        Key or label of benchmark model to compare against. If None, defaults to first converged model.
    power : int, default=2
        Loss function power (2 for squared error, 1 for absolute error).
    against_key : Optional[str], default=None
        Deprecated alias for reference_key.

    Returns
    -------
    pd.DataFrame
        Columns: Model, DM stat, p-value, verdict, note
    """
    if reference_key is None and against_key is not None:
        reference_key = against_key

    cols = ["Model", "DM stat", "p-value", "verdict", "note"]
    if not results:
        return pd.DataFrame(columns=cols)

    ref_model = None
    if reference_key is not None:
        for r in results:
            if getattr(r, "key", "") == reference_key or getattr(r, "label", "") == reference_key:
                ref_model = r
                break
        if ref_model is None:
            raise ValueError(f"reference_key '{reference_key}' not found in results.")
    else:
        for r in results:
            if getattr(r, "converged", False):
                ref_model = r
                break
        if ref_model is None:
            ref_model = results[0]

    if not getattr(ref_model, "converged", False):
        return pd.DataFrame(columns=cols)

    ref_act = getattr(ref_model, "test_actual", None)
    ref_fc = getattr(ref_model, "test_forecast", None)
    if ref_act is None or ref_fc is None or "mean" not in ref_fc:
        return pd.DataFrame(columns=cols)

    e_ref = ref_act.values - ref_fc["mean"].values

    records: List[Dict[str, Any]] = []
    for r in results:
        label = getattr(r, "label", getattr(r, "key", "Unknown"))
        if getattr(r, "key", "") == getattr(ref_model, "key", ""):
            records.append({
                "Model": label,
                "DM stat": 0.0,
                "p-value": 1.0,
                "verdict": "reference model",
                "note": "Reference baseline",
            })
            continue

        if not getattr(r, "converged", False):
            records.append({
                "Model": label,
                "DM stat": np.nan,
                "p-value": np.nan,
                "verdict": "not converged",
                "note": "Model did not converge",
            })
            continue

        fc = getattr(r, "test_forecast", None)
        act = getattr(r, "test_actual", None)
        if act is None:
            act = ref_act
        if fc is None or "mean" not in fc or act is None:
            records.append({
                "Model": label,
                "DM stat": np.nan,
                "p-value": np.nan,
                "verdict": "missing forecast",
                "note": "Forecast unavailable",
            })
            continue

        e_model = act.values - fc["mean"].values
        dm_res = diebold_mariano(e_model, e_ref, h=1, power=power)
        stat = dm_res["stat"]
        pval = dm_res["pvalue"]

        if np.isnan(pval):
            verdict = "inconclusive"
        elif pval < 0.05:
            verdict = "better" if stat < 0 else "worse"
        else:
            verdict = "no significant difference"

        records.append({
            "Model": label,
            "DM stat": round(stat, 4) if not np.isnan(stat) else np.nan,
            "p-value": round(pval, 4) if not np.isnan(pval) else np.nan,
            "verdict": verdict,
            "note": dm_res["note"],
        })

    return pd.DataFrame(records)


def best_model(table: pd.DataFrame, metric: str = "RMSE") -> Optional[str]:
    """
    Return the key of the top-ranked converged model from the comparison table.
    Uses the Rank==1 row's key, or falls back to ranking by the specified metric
    using the shared column mapping.
    """
    if table.empty:
        return None

    conv = table[table["Converged"] == True] if "Converged" in table.columns else table
    if conv.empty:
        return None

    # Return the Rank==1 row's key
    if "Rank" in conv.columns:
        rank_1 = conv[conv["Rank"] == 1]
        if not rank_1.empty:
            first_row = rank_1.iloc[0]
            return str(first_row.get("Key", first_row.get("Model", "")))

    # Fallback using shared column mapping
    target_col = map_metric_column(metric)
    if target_col in conv.columns:
        ranked = conv.sort_values(by=target_col, ascending=True)
        first_row = ranked.iloc[0]
        return str(first_row.get("Key", first_row.get("Model", "")))

    first_row = conv.iloc[0]
    return str(first_row.get("Key", first_row.get("Model", "")))


def make_future_exog(
    exog: pd.DataFrame,
    steps: int,
    strategy: str,
    m: Optional[int] = None,
    k: Optional[int] = None,
    freq: Optional[str] = None,
) -> pd.DataFrame:
    """
    Synthesize future exogenous values extending historical exogenous observations.

    Parameters
    ----------
    exog : pd.DataFrame
        Historical exogenous regressors with DatetimeIndex.
    steps : int
        Number of forward steps to project.
    strategy : str
        Extrapolation heuristic: 'repeat_last', 'repeat_last_season', 'mean_last_k', 'linear_trend'.
    m : Optional[int]
        Seasonal cycle length required for 'repeat_last_season'.
    k : Optional[int]
        Window length required for 'mean_last_k'.
    freq : Optional[str]
        Frequency override for generating future DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Extrapolated future exogenous features with continuing DatetimeIndex.
    """
    if not isinstance(exog, pd.DataFrame) or exog.empty:
        raise ValueError("Exogenous data must be a non-empty DataFrame.")

    if steps <= 0:
        raise ValueError(f"Steps must be a positive integer, got {steps}.")

    # Validate numeric and no NaNs
    for col in exog.columns:
        if not pd.api.types.is_numeric_dtype(exog[col]):
            raise ValueError(f"Exogenous column '{col}' contains non-numeric values.")

    if exog.isna().any().any():
        raise ValueError("Exogenous data contains NaN missing values.")

    # Determine future DatetimeIndex
    last_date = exog.index[-1]
    inferred_freq = freq or getattr(exog.index, "freqstr", None) or pd.infer_freq(exog.index)
    if inferred_freq:
        future_index = pd.date_range(start=last_date, periods=steps + 1, freq=inferred_freq)[1:]
    else:
        diff_td = (exog.index[-1] - exog.index[0]) / max(1, len(exog) - 1)
        future_index = pd.DatetimeIndex([last_date + (i + 1) * diff_td for i in range(steps)])

    strat = str(strategy).lower()

    if strat == "repeat_last":
        last_row = exog.iloc[-1].values
        future_vals = np.tile(last_row, (steps, 1))

    elif strat == "repeat_last_season":
        if m is None or m <= 1:
            raise ValueError("Strategy 'repeat_last_season' requires an integer seasonal period m >= 2.")
        if len(exog) < m:
            raise ValueError(f"Exogenous series length ({len(exog)}) is smaller than seasonal period m={m}.")
        seasonal_block = exog.iloc[-m:].values
        n_repeats = int(np.ceil(steps / m))
        repeated = np.tile(seasonal_block, (n_repeats, 1))
        future_vals = repeated[:steps]

    elif strat == "mean_last_k":
        if k is None or k <= 0:
            raise ValueError("Strategy 'mean_last_k' requires an integer window size k >= 1.")
        if len(exog) < k:
            raise ValueError(f"Exogenous series length ({len(exog)}) is smaller than window k={k}.")
        mean_vals = exog.iloc[-k:].mean(axis=0).values
        future_vals = np.tile(mean_vals, (steps, 1))

    elif strat == "linear_trend":
        n = len(exog)
        x_hist = np.arange(n)
        x_future = np.arange(n, n + steps)
        future_cols = []
        for col in exog.columns:
            y_col = exog[col].values
            # Ordinary least squares line fit
            slope, intercept = np.polyfit(x_hist, y_col, 1)
            future_cols.append(intercept + slope * x_future)
        future_vals = np.column_stack(future_cols)

    else:
        raise ValueError(
            f"Unknown exogenous projection strategy '{strategy}'. "
            f"Choose from: 'repeat_last', 'repeat_last_season', 'mean_last_k', 'linear_trend'."
        )

    return pd.DataFrame(future_vals, index=future_index, columns=exog.columns)


def validate_uploaded_future_exog(
    df: pd.DataFrame,
    required_columns: List[str],
    steps: int,
    expected_index: Optional[pd.DatetimeIndex] = None,
) -> pd.DataFrame:
    """
    Validate user-uploaded or user-edited future exogenous table.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Uploaded exogenous data is not a valid DataFrame.")

    if len(df) != steps:
        raise ValueError(f"Future exogenous rows ({len(df)}) does not match required forecast horizon ({steps}).")

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Uploaded exogenous features missing required column(s): {missing}.")

    subset = df[required_columns].copy()
    for col in subset.columns:
        if not pd.api.types.is_numeric_dtype(subset[col]):
            try:
                subset[col] = pd.to_numeric(subset[col])
            except Exception:
                raise ValueError(f"Column '{col}' must contain numeric values.")

    if subset.isna().any().any():
        raise ValueError("Future exogenous features cannot contain missing (NaN) values.")

    if expected_index is not None and len(expected_index) == steps:
        subset.index = expected_index

    return subset


def compare_columns_and_find_best(
    bundles: Dict[str, Any],
    criterion: str = "balanced",
) -> Dict[str, Any]:
    """
    Compare multiple time series columns evaluated as AnalysisBundles and determine the best performer.

    Parameters
    ----------
    bundles : Dict[str, AnalysisBundle]
        Dictionary mapping column names to their fitted AnalysisBundle objects.
    criterion : str, default='balanced'
        Selection criterion:
        - 'balanced': Multi-criteria composite score (Predictability 40%, Trend 25%, Stability 20%, Seasonality 15%).
        - 'mase': Lowest forecast error / best out-of-sample predictability.
        - 'trend': Strongest structural trend strength.
        - 'stability': Most stable / lowest volatility (coefficient of variation).
        - 'seasonality': Strongest recurring cyclical seasonality.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'leaderboard_df': pd.DataFrame ranking all columns with all metrics.
        - 'winner_column': str, name of the winning column.
        - 'winner_bundle': AnalysisBundle of the winner.
        - 'winner_score': float, score of the winning column.
        - 'synthesis': str, detailed plain-English verdict explaining why it won.
        - 'category_winners': Dict[str, str] mapping category to column name.
        - 'criterion': str, criterion used.
    """
    if not bundles:
        empty_df = pd.DataFrame(
            columns=[
                "Rank",
                "Column",
                "Overall Score",
                "Best Model",
                "MASE",
                "sMAPE (%)",
                "Rel. RMSE vs Naive",
                "Trend Strength",
                "Seasonal Strength",
                "Volatility (CV)",
                "Stationarity",
                "Verdict",
            ]
        )
        return {
            "leaderboard_df": empty_df,
            "winner_column": None,
            "winner_bundle": None,
            "winner_score": 0.0,
            "synthesis": "No columns were provided for comparison.",
            "category_winners": {},
            "criterion": criterion,
        }

    records = []
    for col_name, bundle in bundles.items():
        s = getattr(bundle, "series", pd.Series(dtype=float)).dropna()
        n_obs = len(s)
        mean_val = float(s.mean()) if n_obs > 0 else 0.0
        std_val = float(s.std()) if n_obs > 1 else 0.0
        cv_val = float(std_val / abs(mean_val)) if (abs(mean_val) > 1e-9) else 0.0

        decomp = getattr(bundle, "decomposition", {}) or {}
        trend_str = float(decomp.get("trend_strength", 0.0))
        seas_str = float(decomp.get("seasonal_strength", 0.0))
        stationarity_dict = getattr(bundle, "stationarity", {}) or {}
        verdict_st = stationarity_dict.get("verdict", "Unknown")

        best_label = getattr(bundle, "best_label", "None") or "None"
        comp_df = getattr(bundle, "comparison", pd.DataFrame())

        best_mase = np.nan
        best_smape = np.nan
        best_rmse = np.nan
        if comp_df is not None and not comp_df.empty:
            conv = comp_df[comp_df["Converged"] == True]
            row = conv.iloc[0] if not conv.empty else comp_df.iloc[0]
            if "MASE" in row:
                best_mase = float(row["MASE"])
            if "sMAPE" in row:
                best_smape = float(row["sMAPE"])
            if "RMSE" in row:
                best_rmse = float(row["RMSE"])

        rel_rmse = getattr(bundle, "rel_rmse_best", np.nan)
        if (np.isnan(rel_rmse) or rel_rmse is None) and not np.isnan(best_rmse) and getattr(bundle, "naive_rmse", np.nan) > 0:
            rel_rmse = float(best_rmse / bundle.naive_rmse)

        # Normalized sub-scores in [0, 100]
        # 1. Predictability: lower MASE is better. MASE=0 => 100, MASE=1 => 50, MASE=2 => 33.3
        if np.isfinite(best_mase) and best_mase >= 0:
            pred_score = 100.0 / (1.0 + best_mase)
        else:
            pred_score = 20.0

        # 2. Trend: higher is better [0, 100]
        trend_score = min(100.0, max(0.0, trend_str * 100.0))

        # 3. Stability: lower CV is better. CV=0 => 100, CV=0.5 => 66.7, CV=1.0 => 50
        if np.isfinite(cv_val) and cv_val >= 0:
            stab_score = 100.0 / (1.0 + min(10.0, cv_val))
        else:
            stab_score = 30.0

        # 4. Seasonality: [0, 100]
        seas_score = min(100.0, max(0.0, seas_str * 100.0))

        # Composite score
        if criterion == "mase":
            score = pred_score
        elif criterion == "trend":
            score = trend_score
        elif criterion == "stability":
            score = stab_score
        elif criterion == "seasonality":
            score = seas_score
        else:  # balanced
            score = 0.40 * pred_score + 0.25 * trend_score + 0.20 * stab_score + 0.15 * seas_score

        records.append({
            "col_name": col_name,
            "bundle": bundle,
            "score": round(float(score), 1),
            "best_model": best_label,
            "mase": best_mase,
            "smape": best_smape,
            "rmse": best_rmse,
            "rel_rmse": rel_rmse,
            "trend_strength": trend_str,
            "seasonal_strength": seas_str,
            "cv": cv_val,
            "mean": mean_val,
            "std": std_val,
            "n_obs": n_obs,
            "stationarity": verdict_st,
            "pred_score": pred_score,
            "trend_score": trend_score,
            "stab_score": stab_score,
            "seas_score": seas_score,
        })

    # Sort records based on criterion
    if criterion == "mase":
        # Sort by MASE ascending (NaNs placed at end)
        records.sort(key=lambda r: (1e9 if not np.isfinite(r["mase"]) else r["mase"], -r["score"]))
    elif criterion == "trend":
        records.sort(key=lambda r: (-r["trend_strength"], -r["score"]))
    elif criterion == "stability":
        records.sort(key=lambda r: (1e9 if not np.isfinite(r["cv"]) else r["cv"], -r["score"]))
    elif criterion == "seasonality":
        records.sort(key=lambda r: (-r["seasonal_strength"], -r["score"]))
    else:
        records.sort(key=lambda r: -r["score"])

    # Determine category winners
    # Most predictable (lowest finite MASE)
    valid_mase = [r for r in records if np.isfinite(r["mase"])]
    most_pred = min(valid_mase, key=lambda r: r["mase"])["col_name"] if valid_mase else records[0]["col_name"]

    # Strongest trend
    strongest_trend = max(records, key=lambda r: r["trend_strength"])["col_name"]

    # Most stable (lowest finite CV)
    valid_cv = [r for r in records if np.isfinite(r["cv"])]
    most_stable = min(valid_cv, key=lambda r: r["cv"])["col_name"] if valid_cv else records[0]["col_name"]

    # Strongest seasonality
    strongest_seas = max(records, key=lambda r: r["seasonal_strength"])["col_name"]

    category_winners = {
        "most_predictable": most_pred,
        "strongest_trend": strongest_trend,
        "most_stable": most_stable,
        "strongest_seasonality": strongest_seas,
    }

    winner = records[0]
    winner_col = winner["col_name"]
    winner_bundle = winner["bundle"]
    winner_score = winner["score"]

    # Build plain-English synthesis explaining why this column won
    crit_labels = {
        "balanced": "Balanced Multi-Criteria Evaluation (Accuracy, Trend & Stability)",
        "mase": "Lowest Forecast Error (Out-of-sample MASE)",
        "trend": "Strongest Structural Trend Growth",
        "stability": "Lowest Volatility & Highest Consistency",
        "seasonality": "Strongest Recurring Seasonality",
    }
    crit_desc = crit_labels.get(criterion, "Evaluation")

    mase_str = f"{winner['mase']:.3f}" if np.isfinite(winner['mase']) else "N/A"
    smape_str = f"{winner['smape']:.1f}%" if np.isfinite(winner['smape']) else "N/A"
    rel_str = f"{winner['rel_rmse']:.3f}" if np.isfinite(winner['rel_rmse']) else "N/A"
    rel_pct = f"{(1.0 - winner['rel_rmse']) * 100:.1f}%" if np.isfinite(winner['rel_rmse']) and winner['rel_rmse'] < 1.0 else "None"

    gain_note = f" (beating the Naive baseline by {rel_pct})" if rel_pct != "None" else ""
    synthesis = (
        f"**🏆 Winner Announcement**: Column **`{winner_col}`** emerged as the **Best Column** with an Overall Performance Score of **{winner_score:.1f}/100** under *{crit_desc}*.\n\n"
        f"- **Model Fit & Forecast Accuracy**: Best model architecture is **{winner['best_model']}** with an out-of-sample MASE of **{mase_str}** and sMAPE of **{smape_str}**{gain_note}.\n"
        f"- **Structural Dynamics**: Trend strength is **{winner['trend_strength']:.2f}** with seasonal strength of **{winner['seasonal_strength']:.2f}** and coefficient of variation CV = **{winner['cv']:.2f}**.\n"
        f"- **Stationarity Status**: Diagnosed as **{winner['stationarity']}** across joint ADF and KPSS tests."
    )

    if len(records) > 1:
        other_notes = []
        if strongest_trend != winner_col:
            st_r = next(r for r in records if r["col_name"] == strongest_trend)
            other_notes.append(f"Column **`{strongest_trend}`** exhibited the strongest directional trend ({st_r['trend_strength']:.2f})")
        if most_stable != winner_col:
            ms_r = next(r for r in records if r["col_name"] == most_stable)
            other_notes.append(f"Column **`{most_stable}`** showed the lowest volatility (CV: {ms_r['cv']:.2f})")
        if most_pred != winner_col:
            mp_r = next(r for r in records if r["col_name"] == most_pred)
            other_notes.append(f"Column **`{most_pred}`** achieved the lowest raw MASE ({mp_r['mase']:.3f})")

        if other_notes:
            synthesis += "\n\n**Key Comparison Highlights**:\n- " + "\n- ".join(other_notes) + "."

    # Construct Leaderboard DataFrame
    rows = []
    for rank_idx, r in enumerate(records, start=1):
        is_winner = rank_idx == 1
        verdict = "🏆 Best Performer" if is_winner else ("Top Predictability" if r["col_name"] == most_pred else ("Strong Trend" if r["col_name"] == strongest_trend else "Contender"))
        rows.append({
            "Rank": rank_idx,
            "Column": r["col_name"],
            "Overall Score": r["score"],
            "Best Model": r["best_model"],
            "MASE": round(r["mase"], 3) if np.isfinite(r["mase"]) else np.nan,
            "sMAPE (%)": round(r["smape"], 2) if np.isfinite(r["smape"]) else np.nan,
            "Rel. RMSE vs Naive": round(r["rel_rmse"], 3) if np.isfinite(r["rel_rmse"]) else np.nan,
            "Trend Strength": round(r["trend_strength"], 2),
            "Seasonal Strength": round(r["seasonal_strength"], 2),
            "Volatility (CV)": round(r["cv"], 2) if np.isfinite(r["cv"]) else np.nan,
            "Stationarity": r["stationarity"],
            "Verdict": verdict,
        })

    leaderboard_df = pd.DataFrame(rows)

    return {
        "leaderboard_df": leaderboard_df,
        "winner_column": winner_col,
        "winner_bundle": winner_bundle,
        "winner_score": winner_score,
        "synthesis": synthesis,
        "category_winners": category_winners,
        "criterion": criterion,
    }

