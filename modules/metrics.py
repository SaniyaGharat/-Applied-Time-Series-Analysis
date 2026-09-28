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
