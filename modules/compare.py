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

    # Sort & Rank by requested metric (case-insensitive)
    sort_key = rank_by.upper()
    metric_cols_upper = {c.upper(): c for c in ["RMSE", "MAE", "MAPE", "sMAPE", "MASE"]}
    target_col = metric_cols_upper.get(sort_key, "RMSE")

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

    if test_actual is not None:
        metrics = accuracy_table(test_actual, mean_forecast, y_train=None, m=1)
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
        extras={},
        test_actual=test_actual,
    )


def pairwise_dm(
    results: List[ModelResult],
    against_key: str,
    power: int = 2,
) -> pd.DataFrame:
    """
    Compute pairwise Diebold-Mariano test comparing each model against a reference model.

    Parameters
    ----------
    results : list of ModelResult
        Candidate models to evaluate.
    against_key : str
        Key or label of benchmark model to compare against.
    power : int, default=2
        Loss function power (2 for squared error, 1 for absolute error).

    Returns
    -------
    pd.DataFrame
        Columns: Model, DM stat, p-value, verdict, note
    """
    ref_model = None
    for r in results:
        if getattr(r, "key", "") == against_key or getattr(r, "label", "") == against_key:
            ref_model = r
            break

    cols = ["Model", "DM stat", "p-value", "verdict", "note"]
    if ref_model is None or not getattr(ref_model, "converged", False):
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
    """
    if table.empty:
        return None

    conv = table[table["Converged"] == True]
    if conv.empty:
        return None

    if "Rank" in conv.columns and conv["Rank"].notna().any():
        ranked = conv.sort_values(by="Rank", ascending=True)
        first_row = ranked.iloc[0]
    else:
        target_col = metric.upper()
        cols = {c.upper(): c for c in conv.columns}
        actual_col = cols.get(target_col, "RMSE")
        ranked = conv.sort_values(by=actual_col, ascending=True)
        first_row = ranked.iloc[0]

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
