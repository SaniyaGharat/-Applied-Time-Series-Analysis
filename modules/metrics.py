"""
Evaluation Metrics Module for Applied Time Series Analysis (ATSA).

Pure Python functions (no Streamlit dependencies) providing:
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE, zero-safe)
- Symmetric Mean Absolute Percentage Error (sMAPE)
- Mean Absolute Scaled Error (MASE, with in-sample seasonal naive baseline)
- Unified accuracy summary table
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd


def _to_numpy_arrays(
    y_true: Union[pd.Series, np.ndarray, list],
    y_pred: Union[pd.Series, np.ndarray, list],
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert inputs to 1D float numpy arrays and drop NaNs pairwise."""
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()

    if len(yt) != len(yp):
        raise ValueError(f"Lengths of y_true ({len(yt)}) and y_pred ({len(yp)}) do not match.")

    valid_mask = ~np.isnan(yt) & ~np.isnan(yp)
    return yt[valid_mask], yp[valid_mask]


def rmse(y_true: Union[pd.Series, np.ndarray, list], y_pred: Union[pd.Series, np.ndarray, list]) -> float:
    """Calculate Root Mean Squared Error (RMSE)."""
    yt, yp = _to_numpy_arrays(y_true, y_pred)
    if len(yt) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mae(y_true: Union[pd.Series, np.ndarray, list], y_pred: Union[pd.Series, np.ndarray, list]) -> float:
    """Calculate Mean Absolute Error (MAE)."""
    yt, yp = _to_numpy_arrays(y_true, y_pred)
    if len(yt) == 0:
        return float("nan")
    return float(np.mean(np.abs(yt - yp)))


def mape(y_true: Union[pd.Series, np.ndarray, list], y_pred: Union[pd.Series, np.ndarray, list]) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).

    Zeros in y_true are ignored safely. Returns np.nan if all y_true values are zero.
    """
    yt, yp = _to_numpy_arrays(y_true, y_pred)
    if len(yt) == 0:
        return float("nan")

    non_zero_mask = yt != 0
    if not np.any(non_zero_mask):
        return float("nan")

    pct_errors = np.abs((yt[non_zero_mask] - yp[non_zero_mask]) / yt[non_zero_mask])
    return float(np.mean(pct_errors) * 100.0)


def smape(y_true: Union[pd.Series, np.ndarray, list], y_pred: Union[pd.Series, np.ndarray, list]) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error (sMAPE).

    Formula: 200 * mean(|y - y_hat| / (|y| + |y_hat|)) over non-zero denominators.
    """
    yt, yp = _to_numpy_arrays(y_true, y_pred)
    if len(yt) == 0:
        return float("nan")

    denom = np.abs(yt) + np.abs(yp)
    valid = denom != 0
    if not np.any(valid):
        return 0.0

    return float(np.mean(200.0 * np.abs(yt[valid] - yp[valid]) / denom[valid]))


def mase(
    y_true: Union[pd.Series, np.ndarray, list],
    y_pred: Union[pd.Series, np.ndarray, list],
    y_train: Optional[Union[pd.Series, np.ndarray, list]] = None,
    m: int = 1,
) -> float:
    """
    Calculate Mean Absolute Scaled Error (MASE).

    Scales MAE by the in-sample mean absolute difference of a seasonal naive benchmark on y_train.
    """
    yt, yp = _to_numpy_arrays(y_true, y_pred)
    current_mae = mae(yt, yp)

    if np.isnan(current_mae):
        return float("nan")

    if y_train is None:
        return current_mae

    ytrain = np.asarray(y_train, dtype=float).ravel()
    ytrain = ytrain[~np.isnan(ytrain)]

    period_m = max(1, int(m or 1))
    if len(ytrain) <= period_m:
        period_m = 1

    if len(ytrain) <= 1:
        return current_mae

    scale = np.mean(np.abs(ytrain[period_m:] - ytrain[:-period_m]))
    if scale == 0:
        scale = 1e-8

    return float(current_mae / scale)


def accuracy_table(
    y_true: Union[pd.Series, np.ndarray, list],
    y_pred: Union[pd.Series, np.ndarray, list],
    y_train: Optional[Union[pd.Series, np.ndarray, list]] = None,
    m: int = 1,
) -> Dict[str, float]:
    """
    Compute full dictionary of forecast accuracy metrics: RMSE, MAE, MAPE, sMAPE, and MASE.
    """
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "mase": mase(y_true, y_pred, y_train=y_train, m=m),
    }


def interval_coverage(
    y_true: Union[pd.Series, np.ndarray, list],
    lower: Union[pd.Series, np.ndarray, list],
    upper: Union[pd.Series, np.ndarray, list],
) -> float:
    """
    Calculate empirical prediction interval coverage percentage (0.0 to 100.0).
    """
    yt = np.asarray(y_true, dtype=float).ravel()
    lo = np.asarray(lower, dtype=float).ravel()
    up = np.asarray(upper, dtype=float).ravel()

    if len(yt) != len(lo) or len(yt) != len(up):
        raise ValueError("Lengths of y_true, lower, and upper must match.")

    valid = ~np.isnan(yt) & ~np.isnan(lo) & ~np.isnan(up)
    if not np.any(valid):
        return float("nan")

    inside = (yt[valid] >= lo[valid]) & (yt[valid] <= up[valid])
    return float(np.mean(inside) * 100.0)


def mean_interval_width(
    lower: Union[pd.Series, np.ndarray, list],
    upper: Union[pd.Series, np.ndarray, list],
) -> float:
    """
    Calculate the mean width of prediction intervals (upper - lower).
    """
    lo = np.asarray(lower, dtype=float).ravel()
    up = np.asarray(upper, dtype=float).ravel()

    if len(lo) != len(up):
        raise ValueError("Lengths of lower and upper must match.")

    valid = ~np.isnan(lo) & ~np.isnan(up)
    if not np.any(valid):
        return float("nan")

    return float(np.mean(up[valid] - lo[valid]))


def winkler_score(
    y_true: Union[pd.Series, np.ndarray, list],
    lower: Union[pd.Series, np.ndarray, list],
    upper: Union[pd.Series, np.ndarray, list],
    alpha: float = 0.05,
) -> float:
    """
    Compute the Winkler Score for (1 - alpha) prediction intervals.
    Score = (U - L) + (2/alpha)*(L - y)*I(y < L) + (2/alpha)*(y - U)*I(y > U).
    """
    yt = np.asarray(y_true, dtype=float).ravel()
    lo = np.asarray(lower, dtype=float).ravel()
    up = np.asarray(upper, dtype=float).ravel()

    if len(yt) != len(lo) or len(yt) != len(up):
        raise ValueError("Lengths of y_true, lower, and upper must match.")

    valid = ~np.isnan(yt) & ~np.isnan(lo) & ~np.isnan(up)
    if not np.any(valid):
        return float("nan")

    y_v = yt[valid]
    l_v = lo[valid]
    u_v = up[valid]

    width = u_v - l_v
    penalty_low = np.where(y_v < l_v, (2.0 / alpha) * (l_v - y_v), 0.0)
    penalty_high = np.where(y_v > u_v, (2.0 / alpha) * (y_v - u_v), 0.0)

    return float(np.mean(width + penalty_low + penalty_high))


def relative_metric(value: float, reference: float) -> float:
    """
    Compute relative metric ratio (value / reference) safely.
    Returns np.nan if reference is 0 or NaN, or if value is NaN.
    """
    try:
        val = float(value)
        ref = float(reference)
    except (TypeError, ValueError):
        return float("nan")

    if np.isnan(val) or np.isnan(ref) or ref == 0.0:
        return float("nan")
    return float(val / ref)


def diebold_mariano(
    e1: Union[pd.Series, np.ndarray, list],
    e2: Union[pd.Series, np.ndarray, list],
    h: int = 1,
    power: int = 2,
) -> Dict[str, Any]:
    """
    Diebold-Mariano test for predictive accuracy equality with Harvey-Leybourne-Newbold (HLN)
    small-sample correction.

    Parameters
    ----------
    e1 : array-like
        Forecast error series from model 1 (y_true - y_pred1).
    e2 : array-like
        Forecast error series from model 2 (y_true - y_pred2).
    h : int, default=1
        Forecast horizon step.
    power : int, default=2
        Loss power: 2 for squared error loss, 1 for absolute error loss.

    Returns
    -------
    dict
        {'stat': float, 'pvalue': float, 'n': int, 'note': str}
    """
    from scipy.stats import t

    note_text = "Indicative only: single forecast origin, small holdout, low power."

    arr1 = np.asarray(e1, dtype=float).ravel()
    arr2 = np.asarray(e2, dtype=float).ravel()

    if len(arr1) != len(arr2):
        raise ValueError("Lengths of error series e1 and e2 must match.")

    valid = ~np.isnan(arr1) & ~np.isnan(arr2)
    arr1 = arr1[valid]
    arr2 = arr2[valid]
    n = len(arr1)

    if n < 5:
        return {"stat": float("nan"), "pvalue": float("nan"), "n": int(n), "note": note_text}

    if power == 2:
        d = (arr1 ** 2) - (arr2 ** 2)
    elif power == 1:
        d = np.abs(arr1) - np.abs(arr2)
    else:
        raise ValueError(f"Unsupported loss power {power}. Must be 1 or 2.")

    if np.all(d == 0) or np.isclose(np.var(d), 0.0):
        return {"stat": float("nan"), "pvalue": float("nan"), "n": int(n), "note": note_text}

    d_bar = float(np.mean(d))
    gamma_0 = float(np.mean((d - d_bar) ** 2))
    v_d = gamma_0

    h_lag = max(1, int(h))
    for k in range(1, h_lag):
        weight = 1.0 - (k / h_lag)
        cov_k = float(np.sum((d[k:] - d_bar) * (d[:-k] - d_bar)) / n)
        v_d += 2.0 * weight * cov_k

    if v_d <= 0:
        v_d = gamma_0 if gamma_0 > 0 else 1e-8

    dm_stat = d_bar / np.sqrt(v_d / n)

    # Harvey-Leybourne-Newbold (HLN) small-sample correction
    hln_term = (n + 1 - 2 * h_lag + (h_lag * (h_lag - 1)) / n) / n
    hln_factor = np.sqrt(max(0.0, hln_term))
    stat_corrected = float(dm_stat * hln_factor)

    pvalue = float(2.0 * (1.0 - t.cdf(abs(stat_corrected), df=n - 1)))

    return {
        "stat": stat_corrected,
        "pvalue": pvalue,
        "n": int(n),
        "note": note_text,
    }

