"""
Diagnostics and Statistical Tests Module for Applied Time Series Analysis (ATSA).

Pure Python functions (no Streamlit dependencies) providing:
- Augmented Dickey-Fuller (ADF) & Kwiatkowski-Phillips-Schmidt-Shin (KPSS) tests.
- Stationarity synthesis verdict.
- ACF and PACF calculations with confidence intervals and heuristic order suggestions.
- Classical and STL time series decomposition with Hyndman strength metrics.
- Series transformations (Log, Box-Cox, Regular & Seasonal Differencing) with metadata tracking.
- Rolling statistics and seasonal subseries extraction.
- Ljung-Box test for autocorrelation diagnostics.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import warnings
import numpy as np
import pandas as pd
from scipy.stats import boxcox
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf


def adf_test(
    series: pd.Series,
    regression: str = "c",
    autolag: str = "AIC",
) -> Dict[str, Any]:
    """
    Perform the Augmented Dickey-Fuller (ADF) test for unit root non-stationarity.

    Null Hypothesis (H0): The series has a unit root (non-stationary).
    Alternative Hypothesis (H1): The series is stationary.

    Parameters
    ----------
    series : pd.Series
        Time series data to test.
    regression : str, default='c'
        Constant and trend order: 'c' (constant), 'ct' (constant + trend),
        'ctt' (constant + linear + quadratic trend), 'n' (no constant/trend).
    autolag : str, default='AIC'
        Lag selection criterion: 'AIC', 'BIC', 't-stat', or None.

    Returns
    -------
    Dict[str, Any]
        Test results containing test statistic, p-value, used lag, number of observations,
        critical values dictionary, and is_stationary boolean (p < 0.05).
    """
    clean_s = series.dropna()
    if len(clean_s) < 10:
        raise ValueError("Series is too short for ADF test (minimum 10 observations required).")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        res = adfuller(clean_s, regression=regression, autolag=autolag)
    stat, pval, used_lag, nobs, crit, _ = res

    return {
        "statistic": float(stat),
        "pvalue": float(pval),
        "used_lag": int(used_lag),
        "nobs": int(nobs),
        "critical_values": {k: float(v) for k, v in crit.items()},
        "is_stationary": bool(pval < 0.05),
    }


def kpss_test(
    series: pd.Series,
    regression: str = "c",
    nlags: Union[str, int] = "auto",
) -> Dict[str, Any]:
    """
    Perform the KPSS (Kwiatkowski-Phillips-Schmidt-Shin) test for stationarity.

    Null Hypothesis (H0): The series is trend-stationary (or level-stationary).
    Alternative Hypothesis (H1): The series has a unit root (non-stationary).

    Parameters
    ----------
    series : pd.Series
        Time series data to test.
    regression : str, default='c'
        'c' (level stationary) or 'ct' (trend stationary).
    nlags : Union[str, int], default='auto'
        Number of lags or 'auto'.

    Returns
    -------
    Dict[str, Any]
        Test results containing statistic, p-value, used lag, critical values,
        and is_stationary boolean (p > 0.05).
    """
    clean_s = series.dropna()
    if len(clean_s) < 10:
        raise ValueError("Series is too short for KPSS test (minimum 10 observations required).")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=InterpolationWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", message=".*p-value is smaller than the indicated p-value.*")
        warnings.filterwarnings("ignore", message=".*p-value is greater than the indicated p-value.*")
        stat, pval, used_lag, crit = kpss(clean_s, regression=regression, nlags=nlags)

    return {
        "statistic": float(stat),
        "pvalue": float(pval),
        "used_lag": int(used_lag),
        "nobs": int(len(clean_s)),
        "critical_values": {k: float(v) for k, v in crit.items()},
        "is_stationary": bool(pval > 0.05),
    }


def stationarity_verdict(adf: Dict[str, Any], kpss_res: Dict[str, Any]) -> Dict[str, str]:
    """
    Synthesize ADF and KPSS test outcomes into a joint stationarity verdict.

    Parameters
    ----------
    adf : Dict[str, Any]
        Result dict from adf_test.
    kpss_res : Dict[str, Any]
        Result dict from kpss_test.

    Returns
    -------
    Dict[str, str]
        Dictionary with 'verdict', 'explanation', and 'status' ('success', 'warning', 'info').
    """
    adf_stationary = adf["is_stationary"]
    kpss_stationary = kpss_res["is_stationary"]

    if adf_stationary and kpss_stationary:
        verdict = "Stationary"
        explanation = (
            "Both ADF (rejects unit root, p < 0.05) and KPSS (fails to reject stationarity, p > 0.05) "
            "agree that the series is stationary."
        )
        status = "success"
    elif not adf_stationary and not kpss_stationary:
        verdict = "Non-stationary (differencing needed)"
        explanation = (
            "Both ADF (fails to reject unit root, p >= 0.05) and KPSS (rejects stationarity, p <= 0.05) "
            "agree that the series is non-stationary. Differencing is recommended."
        )
        status = "warning"
    elif adf_stationary and not kpss_stationary:
        verdict = "Difference-stationary / conflicting (try differencing)"
        explanation = (
            "ADF indicates stationarity while KPSS indicates non-stationarity. The series is likely "
            "difference-stationary; differencing should be evaluated."
        )
        status = "info"
    else:
        verdict = "Trend-stationary / conflicting (try detrending)"
        explanation = (
            "ADF fails to reject unit root but KPSS indicates stationarity around a deterministic trend. "
            "Detrending or adding a trend term in modeling is recommended."
        )
        status = "info"

    return {
        "verdict": verdict,
        "explanation": explanation,
        "status": status,
    }


def compute_acf_pacf(
    series: pd.Series,
    nlags: int = 40,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Compute Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF).

    Parameters
    ----------
    series : pd.Series
        Time series data.
    nlags : int, default=40
        Requested number of lags. Will be automatically capped at min(nlags, len(series)//2 - 1).
    alpha : float, default=0.05
        Significance level for confidence bounds.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing lags array, acf values, pacf values, confidence intervals,
        and standard normal critical bound (z / sqrt(N)).
    """
    clean_s = series.dropna()
    n = len(clean_s)
    max_lags = max(1, n // 2 - 1)
    effective_nlags = min(nlags, max_lags)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        acf_vals, acf_conf = acf(clean_s, nlags=effective_nlags, alpha=alpha, fft=True)
        pacf_vals, pacf_conf = pacf(clean_s, nlags=effective_nlags, method="ywm", alpha=alpha)

    conf_bound = 1.96 / np.sqrt(n)
    lags = np.arange(len(acf_vals))

    return {
        "lags": lags,
        "acf": acf_vals,
        "pacf": pacf_vals,
        "acf_conf": acf_conf,
        "pacf_conf": pacf_conf,
        "conf_bound": conf_bound,
        "nlags": effective_nlags,
        "nobs": n,
    }


def suggest_orders(
    acf_vals: np.ndarray,
    pacf_vals: np.ndarray,
    conf_bound: float,
) -> Dict[str, Any]:
    """
    Provide heuristic hints for AR order (p) and MA order (q) from significant ACF/PACF cutoffs.

    Parameters
    ----------
    acf_vals : np.ndarray
        Array of ACF values starting at lag 0.
    pacf_vals : np.ndarray
        Array of PACF values starting at lag 0.
    conf_bound : float
        Confidence bound threshold (+/- z / sqrt(N)).

    Returns
    -------
    Dict[str, Any]
        Dictionary with significant AR lags, significant MA lags, suggested p, suggested q,
        and advisory caption.
    """
    sig_pacf_lags = [int(i) for i in range(1, len(pacf_vals)) if abs(pacf_vals[i]) > conf_bound]
    sig_acf_lags = [int(i) for i in range(1, len(acf_vals)) if abs(acf_vals[i]) > conf_bound]

    # Suggest order as first non-significant cutoff or first significant lag
    suggested_p = sig_pacf_lags[0] if sig_pacf_lags else 0
    suggested_q = sig_acf_lags[0] if sig_acf_lags else 0

    caption = (
        "Note: ACF and PACF cutoffs provide rough heuristic hints. For rigorous model selection, "
        "always compare information criteria (AIC, BIC) across candidate ARIMA/SARIMA specifications."
    )

    return {
        "sig_pacf_lags": sig_pacf_lags[:5],
        "sig_acf_lags": sig_acf_lags[:5],
        "suggested_p": suggested_p,
        "suggested_q": suggested_q,
        "caption": caption,
    }


def infer_period_from_frequency(series: pd.Series) -> Optional[int]:
    """
    Infer canonical seasonal period integer from a series DatetimeIndex frequency.

    Returns
    -------
    Optional[int]
        Inferred seasonal period (e.g. 12 for monthly, 7 for daily) or None.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        return None

    freq = series.index.freqstr or pd.infer_freq(series.index)
    if not freq:
        return None

    freq_upper = freq.upper()
    if freq_upper.startswith("D"):
        return 7
    elif freq_upper.startswith("W"):
        return 52
    elif freq_upper.startswith(("M", "MS", "ME")):
        return 12
    elif freq_upper.startswith(("Q", "QS", "QE")):
        return 4
    elif freq_upper.startswith(("H", "h")):
        return 24
    elif freq_upper.startswith(("Y", "YS", "YE", "A")):
        return None

    return None


def decompose(
    series: pd.Series,
    model: str = "additive",
    period: Optional[int] = None,
    method: str = "classical",
) -> Dict[str, Any]:
    """
    Decompose time series into trend, seasonal, and residual components with Hyndman strength metrics.

    Parameters
    ----------
    series : pd.Series
        Time series data indexed by DatetimeIndex.
    model : str, default='additive'
        'additive' or 'multiplicative'.
    period : Optional[int], default=None
        Seasonal cycle period length. If None, auto-inferred from frequency.
    method : str, default='classical'
        'classical' (moving-average decomposition) or 'stl' (Loess decomposition).

    Returns
    -------
    Dict[str, Any]
        Dictionary containing observed, trend, seasonal, resid series, period,
        trend strength (F_T), seasonal strength (F_S), and optional error message.
    """
    clean_s = series.dropna()

    # Determine period
    chosen_period = period
    if chosen_period is None or chosen_period <= 1:
        chosen_period = infer_period_from_frequency(clean_s)
        if chosen_period is None or chosen_period <= 1:
            chosen_period = 12 if len(clean_s) >= 24 else None

    if chosen_period is None or chosen_period <= 1:
        return {
            "error": "Could not determine seasonal period. Please specify an explicit seasonal period > 1.",
        }

    # Guard: need at least 2 full periods
    if len(clean_s) < 2 * chosen_period:
        return {
            "error": f"Series length ({len(clean_s)}) is insufficient for seasonal period {chosen_period}. "
                     f"At least 2 full periods ({2 * chosen_period} observations) are required.",
        }

    # Guard: multiplicative requires strictly positive values
    if model.lower() == "multiplicative" and (clean_s <= 0).any():
        return {
            "error": "Multiplicative decomposition requires strictly positive values (all values > 0). "
                     "Found zero or negative values in the series. Please select additive model or apply a shift/log.",
        }

    try:
        if method.lower() == "classical":
            decomp = seasonal_decompose(clean_s, model=model.lower(), period=chosen_period)
            observed = decomp.observed
            trend = decomp.trend
            seasonal = decomp.seasonal
            resid = decomp.resid
        elif method.lower() == "stl":
            if model.lower() == "multiplicative":
                # STL on log-transformed data
                log_s = np.log(clean_s)
                stl_obj = STL(log_s, period=chosen_period, robust=True).fit()
                observed = clean_s
                trend = np.exp(stl_obj.trend)
                seasonal = np.exp(stl_obj.seasonal)
                resid = np.exp(stl_obj.resid)
            else:
                stl_obj = STL(clean_s, period=chosen_period, robust=True).fit()
                observed = clean_s
                trend = stl_obj.trend
                seasonal = stl_obj.seasonal
                resid = stl_obj.resid
        else:
            return {"error": f"Unknown decomposition method '{method}'. Choose 'classical' or 'stl'."}

        # Hyndman strength calculations:
        # F_T = max(0, 1 - Var(resid) / Var(trend + resid))
        # F_S = max(0, 1 - Var(resid) / Var(seasonal + resid))
        var_resid = np.nanvar(resid, ddof=1)
        var_trend_resid = np.nanvar(trend + resid, ddof=1)
        var_seas_resid = np.nanvar(seasonal + resid, ddof=1)

        f_t = max(0.0, 1.0 - (var_resid / var_trend_resid)) if var_trend_resid > 0 else 0.0
        f_s = max(0.0, 1.0 - (var_resid / var_seas_resid)) if var_seas_resid > 0 else 0.0

        return {
            "observed": observed,
            "trend": trend,
            "seasonal": seasonal,
            "resid": resid,
            "period": chosen_period,
            "model": model.lower(),
            "method": method.lower(),
            "trend_strength": float(np.clip(f_t, 0.0, 1.0)),
            "seasonal_strength": float(np.clip(f_s, 0.0, 1.0)),
            "error": None,
        }
    except Exception as exc:
        return {"error": f"Decomposition failed: {str(exc)}"}


def transform_series(
    series: pd.Series,
    log: bool = False,
    boxcox_tf: bool = False,
    diff_order: int = 0,
    seasonal_diff_order: int = 0,
    seasonal_period: Optional[int] = None,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Apply variance stabilizing (Log / Box-Cox) and differencing transformations in sequential order.

    Parameters
    ----------
    series : pd.Series
        Original time series.
    log : bool, default=False
        Apply natural logarithm. Mutually exclusive with boxcox_tf.
    boxcox_tf : bool, default=False
        Apply SciPy Box-Cox power transform. Mutually exclusive with log.
    diff_order : int, default=0
        Order of regular differencing (0, 1, 2).
    seasonal_diff_order : int, default=0
        Order of seasonal differencing (0, 1).
    seasonal_period : Optional[int], default=None
        Period length for seasonal differencing.

    Returns
    -------
    Tuple[pd.Series, Dict[str, Any]]
        Transformed pd.Series and info dictionary recording applied transformation pipeline.

    Raises
    ------
    ValueError
        If parameters are invalid or values violate transformation domains.
    """
    if log and boxcox_tf:
        raise ValueError("Log and Box-Cox transformations are mutually exclusive. Choose at most one.")

    s = series.copy().dropna()
    steps_applied: List[str] = []
    params_record: Dict[str, Any] = {}

    # 1. Variance Stabilization
    if log:
        if (s <= 0).any():
            raise ValueError("Log transformation requires strictly positive values (all values > 0).")
        s = np.log(s)
        steps_applied.append("log")

    elif boxcox_tf:
        if (s <= 0).any():
            raise ValueError("Box-Cox transformation requires strictly positive values (all values > 0).")
        transformed_vals, lmbda = boxcox(s.values)
        s = pd.Series(transformed_vals, index=s.index, name=s.name)
        steps_applied.append("boxcox")
        params_record["boxcox_lambda"] = float(lmbda)

    # 2. Regular Differencing
    if diff_order > 0:
        for d in range(1, diff_order + 1):
            s = s.diff()
            steps_applied.append(f"diff_{d}")
        s = s.dropna()

    # 3. Seasonal Differencing
    if seasonal_diff_order > 0:
        if seasonal_period is None or seasonal_period <= 1:
            raise ValueError("Seasonal differencing requires a seasonal period > 1.")
        for D in range(1, seasonal_diff_order + 1):
            s = s.diff(seasonal_period)
            steps_applied.append(f"seasonal_diff_{seasonal_period}")
        s = s.dropna()

    info_dict = {
        "steps": steps_applied,
        "params": params_record,
        "original_len": len(series),
        "final_len": len(s),
        "log": log,
        "boxcox": boxcox_tf,
        "diff_order": diff_order,
        "seasonal_diff_order": seasonal_diff_order,
        "seasonal_period": seasonal_period,
    }

    return s, info_dict


def rolling_stats(series: pd.Series, window: int = 12) -> pd.DataFrame:
    """
    Compute rolling mean and rolling standard deviation.

    Parameters
    ----------
    series : pd.Series
        Time series.
    window : int, default=12
        Rolling window length.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'rolling_mean' and 'rolling_std'.
    """
    clean_s = series.dropna()
    effective_window = max(2, min(window, len(clean_s) - 1))
    r_mean = clean_s.rolling(window=effective_window).mean()
    r_std = clean_s.rolling(window=effective_window).std()

    return pd.DataFrame({"rolling_mean": r_mean, "rolling_std": r_std}, index=clean_s.index)


def seasonal_subseries(series: pd.Series, period: int = 12) -> pd.DataFrame:
    """
    Organize time series observations by cycle/season for subseries inspection.

    Parameters
    ----------
    series : pd.Series
        Time series with DatetimeIndex.
    period : int, default=12
        Seasonal cycle period.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'cycle', 'cycle_label', and 'value'.
    """
    clean_s = series.dropna()
    n = len(clean_s)

    if isinstance(clean_s.index, pd.DatetimeIndex) and period == 12:
        cycle_labels = [d.strftime("%b") for d in clean_s.index]
        cycle_idx = [d.month for d in clean_s.index]
    elif isinstance(clean_s.index, pd.DatetimeIndex) and period == 7:
        cycle_labels = [d.strftime("%a") for d in clean_s.index]
        cycle_idx = [d.weekday() for d in clean_s.index]
    elif isinstance(clean_s.index, pd.DatetimeIndex) and period == 4:
        cycle_labels = [f"Q{d.quarter}" for d in clean_s.index]
        cycle_idx = [d.quarter for d in clean_s.index]
    else:
        cycle_idx = [(i % period) + 1 for i in range(n)]
        cycle_labels = [f"Period {c}" for c in cycle_idx]

    return pd.DataFrame(
        {
            "cycle": cycle_idx,
            "cycle_label": cycle_labels,
            "value": clean_s.values,
        },
        index=clean_s.index,
    )


def ljung_box_test(series: pd.Series, lags: Optional[Union[int, List[int]]] = None) -> pd.DataFrame:
    """
    Perform the Ljung-Box test for autocorrelation at multiple lag orders.

    Parameters
    ----------
    series : pd.Series
        Time series or residual series.
    lags : Optional[Union[int, List[int]]], default=None
        Lags to test. If None, uses min(10, len(series)//5).

    Returns
    -------
    pd.DataFrame
        DataFrame with test statistics and p-values indexed by lag.
    """
    clean_s = series.dropna()
    n = len(clean_s)
    if n < 10:
        raise ValueError("Series is too short for Ljung-Box test (minimum 10 observations required).")

    if lags is None:
        effective_lags = min(10, max(2, n // 5))
    else:
        effective_lags = lags

    res = acorr_ljungbox(clean_s, lags=effective_lags, return_df=True)
    res = res.rename(columns={"lb_stat": "Statistic", "lb_pvalue": "p-value"})
    return res
