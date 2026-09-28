"""
Model Zoo Module for Applied Time Series Analysis (ATSA).

Pure Python module (no Streamlit imports).
Provides model fitting, forecasting, diagnostics, and grid search across:
- Baseline models: Naive, Seasonal Naive, Drift
- Autoregressive & Moving Average: AR(p), MA(q), ARMA(p, q)
- Integrated & Seasonal Models: ARIMA(p, d, q), SARIMA(p, d, q)(P, D, Q)[m]
- Exogenous Feature Models: SARIMAX(p, d, q)(P, D, Q)[m] + Exog
- Exponential Smoothing: Holt-Winters (Trend, Damped, Seasonal)
"""

from dataclasses import dataclass, field
import hashlib
import itertools
import json
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
from scipy.stats import jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from modules.diagnostics import inverse_variance_transform
from modules.metrics import accuracy_table, mae, rmse


@dataclass
class ModelResult:
    """
    Standardized, lightweight result container for fitted time series models.
    Picklable and free of non-serializable statsmodels result objects.
    """
    key: str
    label: str
    family: str
    spec: Dict[str, Any]
    train_index: Any
    test_index: Any
    fitted_train: pd.Series
    test_forecast: pd.DataFrame
    residuals: pd.Series
    metrics_test: Dict[str, float]
    metrics_train: Dict[str, float]
    information_criteria: Optional[Dict[str, Optional[float]]]
    params_table: Optional[pd.DataFrame]
    summary_text: str
    ljung_box: Optional[pd.DataFrame]
    normality: Dict[str, float]
    warnings: List[str] = field(default_factory=list)
    converged: bool = True
    fit_seconds: float = 0.0
    notes: str = ""
    extras: Dict[str, Any] = field(default_factory=dict)
    test_actual: Optional[pd.Series] = None


@dataclass
class ForecastResult:
    """
    Standardized container for full-dataset forecasts and estimation metadata.
    """
    df: pd.DataFrame
    label: str
    spec: Dict[str, Any]
    converged: bool
    warnings: List[str]
    notes: str
    fit_seconds: float
    n_train: int
    last_train_date: Any
    transform_steps: List[str]


def make_model_key(
    spec: Dict[str, Any],
    test_size: int,
    transform_info: Optional[Dict[str, Any]],
    exog_cols: Optional[Union[Tuple[str, ...], List[str]]],
    n_obs: int,
    last_date: Any,
    mase_m: int = 1,
) -> str:
    """
    Generate a deterministic 12-character MD5 hash key for model cache indexing.

    Parameters
    ----------
    spec : Dict[str, Any]
        Model specification dictionary.
    test_size : int
        Holdout test size.
    transform_info : Optional[Dict[str, Any]]
        Variance transformation metadata.
    exog_cols : Optional[Union[Tuple[str, ...], List[str]]]
        Names of exogenous columns used.
    n_obs : int
        Total number of observations in series.
    last_date : Any
        Last timestamp in the series.
    mase_m : int, default=1
        Seasonal period used for MASE scale baseline.

    Returns
    -------
    str
        First 12 characters of MD5 hex digest.
    """
    key_dict = {
        "spec": spec,
        "test_size": int(test_size),
        "transform_info": transform_info or {},
        "exog_cols": sorted(list(exog_cols)) if exog_cols else [],
        "n_obs": int(n_obs),
        "last_date": str(last_date),
        "mase_m": int(mase_m or 1),
    }
    encoded = json.dumps(key_dict, sort_keys=True, default=str).encode("utf-8")
    return hashlib.md5(encoded).hexdigest()[:12]


def _resolve_trend(spec: Dict[str, Any], d: int = 0, D: int = 0) -> Optional[str]:
    """
    Resolve trend specification: default to 'c' when d=0 and D=0, else 'n'.
    """
    val = spec.get("trend")
    if val is not None:
        return str(val)
    return "c" if (d == 0 and D == 0) else "n"


def _build_sarimax(
    y: pd.Series,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
    trend: Optional[str] = None,
    exog: Optional[pd.DataFrame] = None,
) -> SARIMAX:
    """
    Shared internal SARIMAX constructor enforcing stationarity and invertibility parameters.
    """
    return SARIMAX(
        y,
        exog=exog,
        order=order,
        seasonal_order=seasonal_order,
        trend=trend,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )


def default_test_size(n: int, m: Optional[int] = None) -> int:
    """
    Calculate a sensible holdout test set size:
    At least one full season when m is given, never exceeding 30% of total observations.
    """
    period_m = m if m and m > 1 else 1
    suggested = max(period_m, min(int(0.2 * n), 3 * (m or 4)))
    max_limit = max(1, int(0.3 * n))
    return int(min(suggested, max_limit))


def list_model_families() -> List[Dict[str, Any]]:
    """
    Return ordered list of supported model family specifications for UI selection.
    """
    return [
        {
            "key": "naive",
            "label": "Naive Baseline",
            "description": "Forecasts the most recent observed value forward.",
            "default_spec": {"family": "naive"},
            "required_inputs": [],
        },
        {
            "key": "seasonal_naive",
            "label": "Seasonal Naive Baseline",
            "description": "Forecasts values from the identical seasonal period in the prior cycle.",
            "default_spec": {"family": "seasonal_naive", "m": 12},
            "required_inputs": ["m"],
        },
        {
            "key": "drift",
            "label": "Random Walk with Drift",
            "description": "Linearly projects historical average trajectory from start to end.",
            "default_spec": {"family": "drift"},
            "required_inputs": [],
        },
        {
            "key": "ar",
            "label": "Autoregressive AR(p)",
            "description": "Models series as a linear combination of its own past p lags.",
            "default_spec": {"family": "ar", "p": 1, "d": 0, "trend": "c"},
            "required_inputs": ["p"],
        },
        {
            "key": "ma",
            "label": "Moving Average MA(q)",
            "description": "Models series as a linear combination of past forecast error terms.",
            "default_spec": {"family": "ma", "q": 1, "d": 0, "trend": "c"},
            "required_inputs": ["q"],
        },
        {
            "key": "arma",
            "label": "Autoregressive Moving Average ARMA(p, q)",
            "description": "Jointly incorporates stationary past lags and shock terms.",
            "default_spec": {"family": "arma", "p": 1, "q": 1, "d": 0},
            "required_inputs": ["p", "q"],
        },
        {
            "key": "arima",
            "label": "Autoregressive Integrated Moving Average ARIMA(p, d, q)",
            "description": "Differences series d times before fitting ARMA structure.",
            "default_spec": {"family": "arima", "p": 1, "d": 1, "q": 1, "trend": "n"},
            "required_inputs": ["p", "d", "q"],
        },
        {
            "key": "sarima",
            "label": "Seasonal ARIMA (SARIMA)",
            "description": "Incorporates multiplicative seasonal autoregressive and moving average terms.",
            "default_spec": {"family": "sarima", "p": 1, "d": 1, "q": 1, "P": 1, "D": 1, "Q": 1, "m": 12, "trend": "n"},
            "required_inputs": ["p", "d", "q", "P", "D", "Q", "m"],
        },
        {
            "key": "sarimax",
            "label": "SARIMAX with Exogenous Regressors",
            "description": "Extends seasonal ARIMA with external explanatory regressor drivers.",
            "default_spec": {"family": "sarimax", "p": 1, "d": 0, "q": 0, "P": 0, "D": 0, "Q": 0, "m": 12, "trend": "c"},
            "required_inputs": ["p", "d", "q", "exog"],
        },
        {
            "key": "holt_winters",
            "label": "Holt-Winters Exponential Smoothing",
            "description": "Triple exponential smoothing supporting level, trend, and seasonality.",
            "default_spec": {"family": "holt_winters", "trend": "add", "damped": False, "seasonal": "add", "m": 12},
            "required_inputs": ["trend", "seasonal", "m"],
        },
    ]


def _build_failed_result(
    key: str,
    label: str,
    family: str,
    spec: Dict[str, Any],
    train_idx: Any,
    test_idx: Any,
    error_message: str,
    extras: Optional[Dict[str, Any]] = None,
    test_actual: Optional[pd.Series] = None,
) -> ModelResult:
    """Construct a clean, non-raising failed ModelResult container."""
    empty_series = pd.Series(dtype=float, index=train_idx)
    empty_forecast = pd.DataFrame(
        {"mean": np.nan, "lower": np.nan, "upper": np.nan}, index=test_idx
    )
    empty_metrics = {
        "rmse": float("nan"),
        "mae": float("nan"),
        "mape": float("nan"),
        "smape": float("nan"),
        "mase": float("nan"),
    }
    return ModelResult(
        key=key,
        label=label,
        family=family,
        spec=spec,
        train_index=train_idx,
        test_index=test_idx,
        fitted_train=empty_series,
        test_forecast=empty_forecast,
        residuals=pd.Series(dtype=float),
        metrics_test=empty_metrics,
        metrics_train={"rmse": float("nan"), "mae": float("nan")},
        information_criteria=None,
        params_table=None,
        summary_text=f"Model fitting failed: {error_message}",
        ljung_box=None,
        normality={"jb_stat": float("nan"), "pvalue": float("nan")},
        warnings=[error_message],
        converged=False,
        fit_seconds=0.0,
        notes="Failed before or during model estimation.",
        extras=extras or {},
        test_actual=test_actual,
    )


def fit_model(
    y: pd.Series,
    spec: Dict[str, Any],
    test_size: int,
    transform_info: Optional[Dict[str, Any]] = None,
    exog: Optional[pd.DataFrame] = None,
    m: Optional[int] = None,
    mase_m: int = 1,
) -> ModelResult:
    """
    Unified, fail-safe estimation interface for all supported ATSA model families.

    Parameters
    ----------
    y : pd.Series
        Model base series (variance-transformed, undifferenced).
    spec : Dict[str, Any]
        Model specification hyperparameters containing at least 'family'.
    test_size : int
        Number of holdout test points reserved at the end of the series.
    transform_info : Optional[Dict[str, Any]]
        Variance transformation metadata to invert predictions to original scale.
    exog : Optional[pd.DataFrame]
        Exogenous feature regressors aligned with y index.
    m : Optional[int]
        Seasonal period length override.
    mase_m : int, default=1
        Seasonal period used exclusively for computing the in-sample MASE scaling denominator.

    Returns
    -------
    ModelResult
        Complete evaluation, forecast, diagnostic, and parameter container.
    """
    start_time = time.perf_counter()
    family = spec.get("family", "arima").lower()
    clean_y = y.dropna()
    n = len(clean_y)

    exog_cols = tuple(exog.columns) if exog is not None else None
    key = make_model_key(
        spec=spec,
        test_size=test_size,
        transform_info=transform_info,
        exog_cols=exog_cols,
        n_obs=n,
        last_date=clean_y.index[-1] if len(clean_y) > 0 else 0,
        mase_m=mase_m,
    )
    default_label = family.upper()

    if test_size <= 0 or test_size >= n:
        test_size = default_test_size(n, m)

    train_y = clean_y.iloc[:-test_size]
    test_y = clean_y.iloc[-test_size:]

    train_idx = train_y.index
    test_idx = test_y.index

    # Check sufficient observations
    p_order = spec.get("p", 0)
    q_order = spec.get("q", 0)
    d_order = spec.get("d", 0)
    P_order = spec.get("P", 0)
    Q_order = spec.get("Q", 0)
    D_order = spec.get("D", 0)
    m_val = spec.get("m", m or 1)

    min_required_obs = max(30, 3 * (p_order + q_order + P_order + Q_order + d_order))
    if len(train_y) < min_required_obs:
        return _build_failed_result(
            key,
            default_label,
            family,
            spec,
            train_idx,
            test_idx,
            f"Training sample size ({len(train_y)}) is insufficient for model order complexity. "
            f"At least {min_required_obs} observations required.",
        )

    # Exogenous validation
    train_exog: Optional[pd.DataFrame] = None
    test_exog: Optional[pd.DataFrame] = None
    if family == "sarimax":
        if exog is None:
            return _build_failed_result(
                key, default_label, family, spec, train_idx, test_idx,
                "Exogenous dataset (exog) is required for SARIMAX but was None.",
            )
        if exog.isna().any().any():
            return _build_failed_result(
                key, default_label, family, spec, train_idx, test_idx,
                "Exogenous dataset (exog) contains missing values (NaNs). Please impute or drop them.",
            )
        aligned_exog = exog.reindex(clean_y.index)
        train_exog = aligned_exog.iloc[:-test_size]
        test_exog = aligned_exog.iloc[-test_size:]

    collected_warnings: List[str] = []
    converged = True
    burn_in = d_order + D_order * (m_val or 1)
    extras: Dict[str, Any] = {}
    notes: str = ""

    try:
        # ---------------------------------------------------------
        # 1. BASELINE MODELS
        # ---------------------------------------------------------
        if family == "naive":
            label = "Naive Baseline"
            last_val = train_y.iloc[-1]
            fitted_vals = train_y.shift(1).bfill()
            resid = train_y - fitted_vals
            fc_mean = pd.Series(last_val, index=test_idx)

            sd_res = float(resid.std(ddof=1)) if len(resid) > 1 else 1.0
            h = np.arange(1, test_size + 1)
            margin = 1.96 * sd_res * np.sqrt(h)
            fc_lower = pd.Series(fc_mean.values - margin, index=test_idx)
            fc_upper = pd.Series(fc_mean.values + margin, index=test_idx)

            ic = None
            params_tbl = None
            summary_txt = f"Naive Model: Constant point forecast of last observed value ({last_val:.4f})."
            notes = "Forecast interval calculated as +/- 1.96 * sd(naive resid) * sqrt(h)."
            extras = {}

        elif family == "seasonal_naive":
            season_m = int(spec.get("m", m or 12))
            label = f"Seasonal Naive (m={season_m})"
            if len(train_y) < season_m:
                return _build_failed_result(
                    key, label, family, spec, train_idx, test_idx,
                    f"Train size ({len(train_y)}) is smaller than seasonal period m={season_m}.",
                )

            fitted_vals = train_y.shift(season_m).bfill()
            resid = train_y - fitted_vals

            fc_mean_vals = [train_y.iloc[-season_m + (i % season_m)] for i in range(test_size)]
            fc_mean = pd.Series(fc_mean_vals, index=test_idx)

            sd_res = float(resid.std(ddof=1)) if len(resid) > 1 else 1.0
            k_cycles = np.ceil(np.arange(1, test_size + 1) / season_m)
            margin = 1.96 * sd_res * np.sqrt(k_cycles)
            fc_lower = pd.Series(fc_mean.values - margin, index=test_idx)
            fc_upper = pd.Series(fc_mean.values + margin, index=test_idx)

            ic = None
            params_tbl = None
            summary_txt = f"Seasonal Naive Model: Periodic projection using cycle length m={season_m}."
            notes = "Forecast interval calculated as +/- 1.96 * sd(seasonal naive resid) * sqrt(k)."
            extras = {"m": int(season_m)}

        elif family == "drift":
            label = "Random Walk with Drift"
            slope = (train_y.iloc[-1] - train_y.iloc[0]) / max(1, len(train_y) - 1)
            h_in = np.arange(len(train_y))
            fitted_vals = pd.Series(train_y.iloc[0] + h_in * slope, index=train_idx)
            resid = train_y - fitted_vals

            h_out = np.arange(1, test_size + 1)
            fc_mean = pd.Series(train_y.iloc[-1] + h_out * slope, index=test_idx)

            sd_res = float(resid.std(ddof=1)) if len(resid) > 1 else 1.0
            margin = 1.96 * sd_res * np.sqrt(h_out)
            fc_lower = pd.Series(fc_mean.values - margin, index=test_idx)
            fc_upper = pd.Series(fc_mean.values + margin, index=test_idx)

            ic = None
            params_tbl = None
            summary_txt = f"Drift Model: Linear trajectory from start to end (slope={slope:.4f})."
            notes = "Forecast interval calculated as +/- 1.96 * sd(drift resid) * sqrt(h)."
            extras = {"slope": float(slope)}

        # ---------------------------------------------------------
        # 2. STATSMODELS ARIMA / SARIMA / SARIMAX FAMILIES
        # ---------------------------------------------------------
        elif family in ["ar", "ma", "arma", "arima", "sarima", "sarimax"]:
            trend_val = _resolve_trend(spec, d_order, D_order)

            if family == "ar":
                order = (p_order, d_order, 0)
                seasonal_order = (0, 0, 0, 0)
                label = f"AR({p_order})" + (f" (d={d_order})" if d_order > 0 else "")
            elif family == "ma":
                order = (0, d_order, q_order)
                seasonal_order = (0, 0, 0, 0)
                label = f"MA({q_order})" + (f" (d={d_order})" if d_order > 0 else "")
            elif family == "arma":
                order = (p_order, d_order, q_order)
                seasonal_order = (0, 0, 0, 0)
                label = f"ARMA({p_order},{q_order})" + (f" (d={d_order})" if d_order > 0 else "")
            elif family == "arima":
                order = (p_order, d_order, q_order)
                seasonal_order = (0, 0, 0, 0)
                label = f"ARIMA({p_order},{d_order},{q_order})"
            elif family in ["sarima", "sarimax"]:
                order = (p_order, d_order, q_order)
                if P_order > 0 or D_order > 0 or Q_order > 0:
                    season_m = m_val if (m_val and m_val > 1) else 12
                    seasonal_order = (P_order, D_order, Q_order, season_m)
                    prefix = "SARIMAX" if family == "sarimax" else "SARIMA"
                    label = f"{prefix}({p_order},{d_order},{q_order})({P_order},{D_order},{Q_order})[{seasonal_order[3]}]"
                else:
                    seasonal_order = (0, 0, 0, 0)
                    prefix = "SARIMAX" if family == "sarimax" else "SARIMA"
                    label = f"{prefix}({p_order},{d_order},{q_order})"

            with warnings.catch_warnings(record=True) as recorded_w:
                warnings.filterwarnings("ignore", category=FutureWarning)
                warnings.filterwarnings("ignore", category=UserWarning)

                model_obj = _build_sarimax(
                    train_y,
                    order=order,
                    seasonal_order=seasonal_order,
                    trend=trend_val,
                    exog=train_exog if family == "sarimax" else None,
                )
                res = model_obj.fit(disp=False)

                for w in recorded_w:
                    if issubclass(w.category, ConvergenceWarning):
                        converged = False
                        collected_warnings.append(f"Convergence issue: {str(w.message)}")

            if hasattr(res, "mle_retvals") and not res.mle_retvals.get("converged", True):
                converged = False
                collected_warnings.append("Optimizer reported that MLE estimation did not converge.")

            fitted_vals = res.fittedvalues
            resid = res.resid

            forecast_res = res.get_forecast(steps=test_size, exog=test_exog)
            fc_mean = forecast_res.predicted_mean
            conf_int = forecast_res.conf_int(alpha=0.05)
            fc_lower = pd.Series(conf_int.iloc[:, 0].values, index=test_idx)
            fc_upper = pd.Series(conf_int.iloc[:, 1].values, index=test_idx)

            ic = {
                "aic": float(res.aic) if getattr(res, "aic", None) is not None else None,
                "bic": float(res.bic) if getattr(res, "bic", None) is not None else None,
                "hqic": float(res.hqic) if getattr(res, "hqic", None) is not None else None,
                "llf": float(res.llf) if getattr(res, "llf", None) is not None else None,
            }

            params_tbl = pd.DataFrame(
                {
                    "parameter": res.params.index,
                    "estimate": res.params.values,
                    "std_err": getattr(res, "bse", pd.Series(index=res.params.index)).values,
                    "p_value": getattr(res, "pvalues", pd.Series(index=res.params.index)).values,
                }
            )
            summary_txt = str(res.summary())

            # Populate ARIMA family extras
            ar_roots = getattr(res, "arroots", None)
            if ar_roots is None and hasattr(res, "polynomial_reduced_ar"):
                ar_roots = np.roots(res.polynomial_reduced_ar)
            ma_roots = getattr(res, "maroots", None)
            if ma_roots is None and hasattr(res, "polynomial_reduced_ma"):
                ma_roots = np.roots(res.polynomial_reduced_ma)

            extras = {
                "ar_roots": np.asarray(ar_roots) if ar_roots is not None else np.array([]),
                "ma_roots": np.asarray(ma_roots) if ma_roots is not None else np.array([]),
                "reduced_ar": np.asarray(getattr(res, "polynomial_reduced_ar", np.array([1.0]))),
                "reduced_ma": np.asarray(getattr(res, "polynomial_reduced_ma", np.array([1.0]))),
                "d": int(d_order),
                "D": int(D_order),
                "m": int(m_val or 1),
            }
            if family == "sarimax" and train_exog is not None:
                exog_coefs = {}
                for col in train_exog.columns:
                    if col in res.params:
                        exog_coefs[col] = float(res.params[col])
                extras["exog_coefs"] = exog_coefs
            notes = f"Estimated via statsmodels SARIMAX (order={order}, seasonal_order={seasonal_order}, trend='{trend_val}')."

        # ---------------------------------------------------------
        # 3. EXPONENTIAL SMOOTHING (HOLT-WINTERS)
        # ---------------------------------------------------------
        elif family == "holt_winters":
            trend_type = spec.get("trend")
            damped_flag = spec.get("damped", False)
            seasonal_type = spec.get("seasonal")
            season_m = spec.get("m", m or 12) if seasonal_type else None

            label = f"Holt-Winters (trend={trend_type}, seas={seasonal_type})"

            if (trend_type == "mul" or seasonal_type == "mul") and (train_y <= 0).any():
                return _build_failed_result(
                    key, label, family, spec, train_idx, test_idx,
                    "Multiplicative Holt-Winters requires strictly positive values.",
                )

            with warnings.catch_warnings(record=True) as recorded_w:
                warnings.filterwarnings("ignore")
                hw_model = ExponentialSmoothing(
                    train_y,
                    trend=trend_type,
                    damped_trend=damped_flag,
                    seasonal=seasonal_type,
                    seasonal_periods=season_m,
                    initialization_method="estimated",
                )
                res = hw_model.fit()

            fitted_vals = res.fittedvalues
            resid = res.resid
            fc_mean = res.forecast(steps=test_size)

            # Prediction intervals by simulation or residual std fallback
            try:
                simulations = res.simulate(nsimulations=test_size, repetitions=500, rng=np.random.default_rng(42))
                fc_lower = pd.Series(np.percentile(simulations, 2.5, axis=1), index=test_idx)
                fc_upper = pd.Series(np.percentile(simulations, 97.5, axis=1), index=test_idx)
                notes = "Prediction intervals derived via Monte Carlo residual simulation (repetitions=500, seed=42)."
            except Exception:
                sd_res = float(resid.std(ddof=1)) if len(resid) > 1 else 1.0
                fc_lower = fc_mean - 1.96 * sd_res
                fc_upper = fc_mean + 1.96 * sd_res
                notes = "Prediction intervals derived via analytical standard error fallback (+/- 1.96 * residual std)."

            ic = {
                "aic": float(res.aic) if getattr(res, "aic", None) is not None else None,
                "bic": float(res.bic) if getattr(res, "bic", None) is not None else None,
                "hqic": None,
                "llf": None,  # SSE goes to extras as requested in A5
            }
            hw_params_rows = []
            for k_p, v_p in res.params.items():
                if isinstance(v_p, (np.ndarray, list)):
                    for idx_s, val_s in enumerate(v_p):
                        hw_params_rows.append({"parameter": f"{k_p}_{idx_s}", "estimate": float(val_s)})
                else:
                    try:
                        hw_params_rows.append({"parameter": k_p, "estimate": float(v_p) if v_p is not None else np.nan})
                    except (ValueError, TypeError):
                        hw_params_rows.append({"parameter": k_p, "estimate": float("nan")})
            params_tbl = pd.DataFrame(hw_params_rows)
            summary_txt = str(res.summary())

            level_series = getattr(res, "level", None)
            trend_series = getattr(res, "trend", None) if trend_type else None
            season_series = getattr(res, "season", None) if seasonal_type else None

            smoothing_dict = {
                "alpha": float(res.params.get("smoothing_level")) if res.params.get("smoothing_level") is not None and not np.isnan(res.params.get("smoothing_level")) else None,
                "beta": float(res.params.get("smoothing_trend")) if res.params.get("smoothing_trend") is not None and not np.isnan(res.params.get("smoothing_trend")) else None,
                "gamma": float(res.params.get("smoothing_seasonal")) if res.params.get("smoothing_seasonal") is not None and not np.isnan(res.params.get("smoothing_seasonal")) else None,
                "phi": float(res.params.get("damping_trend")) if res.params.get("damping_trend") is not None and not np.isnan(res.params.get("damping_trend")) else None,
            }
            extras = {
                "level": level_series,
                "trend": trend_series,
                "season": season_series,
                "smoothing": smoothing_dict,
                "sse": float(res.sse) if getattr(res, "sse", None) is not None else None,
            }

        else:
            return _build_failed_result(key, default_label, family, spec, train_idx, test_idx, f"Unsupported model family: '{family}'.")

        # ---------------------------------------------------------
        # BURN-IN TRIMMING (A2)
        # Set first burn_in entries of fitted_vals to NaN before inversion and metrics
        # ---------------------------------------------------------
        if burn_in > 0 and len(fitted_vals) > 0:
            effective_burn_in = min(burn_in, len(fitted_vals))
            fitted_vals = fitted_vals.copy()
            fitted_vals.iloc[:effective_burn_in] = np.nan

        # Residuals drop the same burn_in entries
        if burn_in > 0 and len(resid) > burn_in:
            effective_resid = resid.iloc[burn_in:]
        else:
            effective_resid = resid

        # Diagnostics: Ljung-Box test with model_df adjustment
        model_df = p_order + q_order + P_order + Q_order
        n_res = len(effective_resid.dropna())
        lb_lags = [l for l in [5, 10, 15, 20] if l < n_res and l > model_df]
        if not lb_lags:
            lb_lags = [l for l in [5, 10, 15, 20] if l < n_res]

        try:
            if lb_lags:
                use_df = model_df if all(l > model_df for l in lb_lags) else 0
                lb_table = acorr_ljungbox(effective_resid.dropna(), lags=lb_lags, model_df=use_df, return_df=True)
                lb_table = lb_table.rename(columns={"lb_stat": "Statistic", "lb_pvalue": "p-value"})
            else:
                lb_table = None
        except Exception:
            lb_table = None

        # Diagnostics: Normality (Jarque-Bera)
        clean_res = effective_resid.dropna().values
        if len(clean_res) >= 8:
            jb_stat, jb_pval = jarque_bera(clean_res)
            normality_res = {"jb_stat": float(jb_stat), "pvalue": float(jb_pval)}
        else:
            normality_res = {"jb_stat": float("nan"), "pvalue": float("nan")}

        # ---------------------------------------------------------
        # INVERT VARIANCE TRANSFORMATIONS FOR REPORTING
        # ---------------------------------------------------------
        fitted_train_orig = inverse_variance_transform(fitted_vals, transform_info)
        fc_mean_orig = inverse_variance_transform(fc_mean, transform_info)
        fc_lower_orig = inverse_variance_transform(fc_lower, transform_info)
        fc_upper_orig = inverse_variance_transform(fc_upper, transform_info)

        train_y_orig = inverse_variance_transform(train_y, transform_info)
        test_y_orig = inverse_variance_transform(test_y, transform_info)
        if not isinstance(test_y_orig, pd.Series):
            test_y_orig = pd.Series(test_y_orig, index=test_idx)

        test_forecast_df = pd.DataFrame(
            {
                "mean": fc_mean_orig.values if hasattr(fc_mean_orig, "values") else fc_mean_orig,
                "lower": fc_lower_orig.values if hasattr(fc_lower_orig, "values") else fc_lower_orig,
                "upper": fc_upper_orig.values if hasattr(fc_upper_orig, "values") else fc_upper_orig,
            },
            index=test_idx,
        )

        metrics_test = accuracy_table(test_y_orig, fc_mean_orig, y_train=train_y_orig, m=mase_m)
        metrics_train = {
            "rmse": rmse(train_y_orig, fitted_train_orig),
            "mae": mae(train_y_orig, fitted_train_orig),
        }

        fit_dur = float(time.perf_counter() - start_time)

        return ModelResult(
            key=key,
            label=label,
            family=family,
            spec=spec,
            train_index=train_idx,
            test_index=test_idx,
            fitted_train=fitted_train_orig if isinstance(fitted_train_orig, pd.Series) else pd.Series(fitted_train_orig, index=train_idx),
            test_forecast=test_forecast_df,
            residuals=effective_resid,
            metrics_test=metrics_test,
            metrics_train=metrics_train,
            information_criteria=ic,
            params_table=params_tbl,
            summary_text=summary_txt,
            ljung_box=lb_table,
            normality=normality_res,
            warnings=collected_warnings,
            converged=converged,
            fit_seconds=fit_dur,
            notes=notes,
            extras=extras,
            test_actual=test_y_orig,
        )

    except Exception as exc:
        actual_series = test_y_orig if "test_y_orig" in locals() and isinstance(test_y_orig, pd.Series) else None
        return _build_failed_result(key, default_label, family, spec, train_idx, test_idx, str(exc), test_actual=actual_series)


def grid_search_arima(
    y: pd.Series,
    p_range: List[int],
    d: int,
    q_range: List[int],
    seasonal: Optional[Dict[str, Any]] = None,
    test_size: int = 12,
    max_models: int = 60,
    criterion: str = "aic",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Exhaustive grid search across candidate ARIMA/SARIMA specifications on training data.

    Note: Information criteria (AIC / BIC) are only mathematically comparable across models
    that share identical differencing orders (d and D), because differencing alters the
    underlying sample size and data space.

    Parameters
    ----------
    y : pd.Series
        Model base series.
    p_range : List[int]
        Range of AR orders.
    d : int
        Integration order.
    q_range : List[int]
        Range of MA orders.
    seasonal : Optional[Dict[str, Any]], default=None
        Optional seasonal configuration: P_range, D, Q_range, m.
    test_size : int, default=12
        Holdout test size (grid search fits strictly on train portion).
    max_models : int, default=60
        Maximum candidate models to evaluate before truncating.
    criterion : str, default="aic"
        Sorting criterion ("aic" or "bic").

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        Results DataFrame sorted by chosen criterion, and best specification dict.
    """
    clean_y = y.dropna()
    train_y = clean_y.iloc[:-test_size] if test_size > 0 else clean_y

    is_seasonal = seasonal is not None
    p_vals = list(p_range)
    q_vals = list(q_range)

    if is_seasonal:
        P_vals = list(seasonal.get("P_range", [0, 1]))
        D_val = seasonal.get("D", 0)
        Q_vals = list(seasonal.get("Q_range", [0, 1]))
        m_val = seasonal.get("m", 12)
        grid_combos = list(itertools.product(p_vals, [d], q_vals, P_vals, [D_val], Q_vals, [m_val]))
        # Sort by total complexity (p + q + P + Q) ascending BEFORE truncating (A3)
        grid_combos.sort(key=lambda c: (c[0] + c[2] + c[3] + c[5], c[0], c[2]))
    else:
        grid_combos = list(itertools.product(p_vals, [d], q_vals))
        # Sort by total complexity (p + q) ascending BEFORE truncating (A3)
        grid_combos.sort(key=lambda c: (c[0] + c[2], c[0], c[2]))

    total_specs = len(grid_combos)
    truncated = total_specs > max_models
    if truncated:
        grid_combos = grid_combos[:max_models]

    results_records: List[Dict[str, Any]] = []

    for combo in grid_combos:
        if is_seasonal:
            p, d_c, q, P, D, Q, m_c = combo
            spec = {"family": "sarima", "p": p, "d": d_c, "q": q, "P": P, "D": D, "Q": Q, "m": m_c}
            order = (p, d_c, q)
            seas_order = (P, D, Q, m_c) if (P > 0 or D > 0 or Q > 0) else (0, 0, 0, 0)
            spec_str = f"SARIMA({p},{d_c},{q})({P},{D},{Q})[{m_c}]"
            trend_val = _resolve_trend(spec, d_c, D)
        else:
            p, d_c, q = combo
            spec = {"family": "arima", "p": p, "d": d_c, "q": q}
            order = (p, d_c, q)
            seas_order = (0, 0, 0, 0)
            spec_str = f"ARIMA({p},{d_c},{q})"
            trend_val = _resolve_trend(spec, d_c, 0)

        spec["trend"] = trend_val

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                mod = _build_sarimax(train_y, order=order, seasonal_order=seas_order, trend=trend_val)
                res = mod.fit(disp=False)
                aic_val = float(res.aic)
                bic_val = float(res.bic)
                conv = res.mle_retvals.get("converged", True) if hasattr(res, "mle_retvals") else True
                n_params = len(res.params) if hasattr(res, "params") else (p + q + (P + Q if is_seasonal else 0) + (1 if trend_val else 0))
        except Exception:
            aic_val = np.nan
            bic_val = np.nan
            conv = False
            n_params = p + q + (P + Q if is_seasonal else 0) + (1 if trend_val else 0)

        results_records.append(
            {
                "spec": spec_str,
                "p": p,
                "d": d_c,
                "q": q,
                "trend": trend_val,
                "n_params": n_params,
                "aic": aic_val,
                "bic": bic_val,
                "converged": conv,
                "spec_dict": spec,
            }
        )

    res_df = pd.DataFrame(results_records)
    sort_col = "bic" if criterion.lower() == "bic" else "aic"
    res_df = res_df.sort_values(by=sort_col, ascending=True).reset_index(drop=True)

    best_spec = res_df.iloc[0]["spec_dict"] if not res_df.empty else {}
    best_info = {
        "best_spec": best_spec,
        "best_spec_str": res_df.iloc[0]["spec"] if not res_df.empty else "",
        "criterion": sort_col,
        "total_evaluated": len(res_df),
        "total_grid_size": total_specs,
        "truncated": truncated,
    }

    return res_df, best_info


def fit_full_and_forecast_detailed(
    y: pd.Series,
    spec: Dict[str, Any],
    steps: int,
    transform_info: Optional[Dict[str, Any]] = None,
    exog: Optional[pd.DataFrame] = None,
    future_exog: Optional[pd.DataFrame] = None,
    m: Optional[int] = None,
) -> ForecastResult:
    """
    Refit model specification on the full available dataset and return detailed ForecastResult.
    Raises ValueError for invalid inputs (steps <= 0; SARIMAX without or with wrong-length future_exog).
    Numerical failures return converged=False with diagnostic message rather than raising.
    """
    start_time = time.perf_counter()
    clean_y = y.dropna()
    family = spec.get("family", "arima").lower()
    label = spec.get("label", family.upper())
    transform_steps = list(transform_info.get("steps", [])) if transform_info else []
    n_train = len(clean_y)
    last_date = clean_y.index[-1] if n_train > 0 else pd.Timestamp.now()

    if steps <= 0:
        raise ValueError(f"Forecast horizon (steps) must be positive, got {steps}.")

    # Determine continuation future DatetimeIndex
    freq = clean_y.index.freqstr or pd.infer_freq(clean_y.index)
    if freq:
        future_index = pd.date_range(start=last_date, periods=steps + 1, freq=freq)[1:]
    else:
        diff_td = (clean_y.index[-1] - clean_y.index[0]) / max(1, len(clean_y) - 1)
        future_index = pd.DatetimeIndex([last_date + (i + 1) * diff_td for i in range(steps)])

    if family == "sarimax":
        if future_exog is None:
            raise ValueError("Future exogenous features (future_exog) are required for SARIMAX forecasting.")
        if len(future_exog) != steps:
            raise ValueError(
                f"Length of future_exog ({len(future_exog)}) does not match required forecast steps ({steps})."
            )

    try:
        # 1. Fit on Full Data
        warnings_list: List[str] = []
        notes = ""
        converged = True

        if family == "naive":
            last_val = clean_y.iloc[-1]
            fc_mean = pd.Series(last_val, index=future_index)
            sd_res = float((clean_y - clean_y.shift(1).bfill()).std(ddof=1))
            h = np.arange(1, steps + 1)
            margin = 1.96 * sd_res * np.sqrt(h)
            fc_lower = fc_mean - margin
            fc_upper = fc_mean + margin
            notes = "Naive forecast projecting last observed level with expanding analytical uncertainty."

        elif family == "seasonal_naive":
            season_m = spec.get("m", m or 12)
            fc_mean_vals = [clean_y.iloc[-season_m + (step % season_m)] for step in range(steps)]
            fc_mean = pd.Series(fc_mean_vals, index=future_index)
            sd_res = float((clean_y - clean_y.shift(season_m).bfill()).std(ddof=1))
            k_cycles = np.ceil(np.arange(1, steps + 1) / season_m)
            margin = 1.96 * sd_res * np.sqrt(k_cycles)
            fc_lower = fc_mean - margin
            fc_upper = fc_mean + margin
            notes = f"Seasonal naive forecast projecting lag {season_m} seasonal cycle."

        elif family == "drift":
            slope = (clean_y.iloc[-1] - clean_y.iloc[0]) / max(1, len(clean_y) - 1)
            h_out = np.arange(1, steps + 1)
            fc_mean = pd.Series(clean_y.iloc[-1] + h_out * slope, index=future_index)
            fitted = clean_y.iloc[0] + np.arange(len(clean_y)) * slope
            sd_res = float((clean_y - fitted).std(ddof=1))
            margin = 1.96 * sd_res * np.sqrt(h_out)
            fc_lower = fc_mean - margin
            fc_upper = fc_mean + margin
            notes = "Drift forecast with expanding standard errors."

        elif family in ["ar", "ma", "arma", "arima", "sarima", "sarimax"]:
            p = spec.get("p", 0)
            d = spec.get("d", 0)
            q = spec.get("q", 0)
            P = spec.get("P", 0)
            D = spec.get("D", 0)
            Q = spec.get("Q", 0)
            m_s = spec.get("m", m or 1)
            trend = _resolve_trend(spec, d, D)

            if P > 0 or D > 0 or Q > 0:
                season_m = m_s if (m_s and m_s > 1) else 12
                seas_order = (P, D, Q, season_m)
            else:
                seas_order = (0, 0, 0, 0)

            with warnings.catch_warnings(record=True) as recorded_w:
                warnings.filterwarnings("ignore")
                model_obj = _build_sarimax(
                    clean_y,
                    order=(p, d, q),
                    seasonal_order=seas_order,
                    trend=trend,
                    exog=exog if family == "sarimax" else None,
                )
                res = model_obj.fit(disp=False)
                for w in recorded_w:
                    if issubclass(w.category, ConvergenceWarning):
                        converged = False
                        warnings_list.append("Optimizer reported that MLE estimation did not converge.")

                if family == "sarimax":
                    fc_res = res.get_forecast(steps=steps, exog=future_exog)
                else:
                    fc_res = res.get_forecast(steps=steps)

            fc_mean = fc_res.predicted_mean
            ci = fc_res.conf_int(alpha=0.05)
            fc_lower = pd.Series(ci.iloc[:, 0].values, index=future_index)
            fc_upper = pd.Series(ci.iloc[:, 1].values, index=future_index)
            notes = f"Estimated via statsmodels SARIMAX (order={(p, d, q)}, seasonal_order={seas_order}, trend='{trend}')."

        elif family == "holt_winters":
            trend_type = spec.get("trend")
            damped_flag = spec.get("damped", False)
            seasonal_type = spec.get("seasonal")
            season_m = spec.get("m", m or 12) if seasonal_type else None

            if (trend_type == "mul" or seasonal_type == "mul") and (clean_y <= 0).any():
                return ForecastResult(
                    df=pd.DataFrame({"mean": np.nan, "lower": np.nan, "upper": np.nan}, index=future_index),
                    label=label,
                    spec=spec,
                    converged=False,
                    warnings=["Multiplicative Holt-Winters requires strictly positive values."],
                    notes="Failed: non-positive values encountered.",
                    fit_seconds=float(time.perf_counter() - start_time),
                    n_train=n_train,
                    last_train_date=last_date,
                    transform_steps=transform_steps,
                )

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                hw_model = ExponentialSmoothing(
                    clean_y,
                    trend=trend_type,
                    damped_trend=damped_flag,
                    seasonal=seasonal_type,
                    seasonal_periods=season_m,
                    initialization_method="estimated",
                )
                res = hw_model.fit()

            fc_mean = res.forecast(steps=steps)
            try:
                sims = res.simulate(nsimulations=steps, repetitions=500, rng=np.random.default_rng(42))
                fc_lower = pd.Series(np.percentile(sims, 2.5, axis=1), index=future_index)
                fc_upper = pd.Series(np.percentile(sims, 97.5, axis=1), index=future_index)
                notes = "Prediction intervals derived via Monte Carlo residual simulation (repetitions=500, seed=42)."
            except Exception:
                sd_res = float(res.resid.std(ddof=1)) if len(res.resid) > 1 else 1.0
                fc_lower = fc_mean - 1.96 * sd_res
                fc_upper = fc_mean + 1.96 * sd_res
                notes = "Prediction intervals derived via analytical standard error fallback (+/- 1.96 * residual std)."

        else:
            raise ValueError(f"Unsupported model family: '{family}'.")

        # Invert transforms
        mean_orig = inverse_variance_transform(fc_mean, transform_info)
        lower_orig = inverse_variance_transform(fc_lower, transform_info)
        upper_orig = inverse_variance_transform(fc_upper, transform_info)

        df = pd.DataFrame(
            {
                "mean": mean_orig.values if hasattr(mean_orig, "values") else mean_orig,
                "lower": lower_orig.values if hasattr(lower_orig, "values") else lower_orig,
                "upper": upper_orig.values if hasattr(upper_orig, "values") else upper_orig,
            },
            index=future_index,
        )

        return ForecastResult(
            df=df,
            label=label,
            spec=spec,
            converged=converged,
            warnings=warnings_list,
            notes=notes,
            fit_seconds=float(time.perf_counter() - start_time),
            n_train=n_train,
            last_train_date=last_date,
            transform_steps=transform_steps,
        )

    except Exception as exc:
        return ForecastResult(
            df=pd.DataFrame({"mean": np.nan, "lower": np.nan, "upper": np.nan}, index=future_index),
            label=label,
            spec=spec,
            converged=False,
            warnings=[str(exc)],
            notes=f"Estimation failed numerically: {str(exc)}",
            fit_seconds=float(time.perf_counter() - start_time),
            n_train=n_train,
            last_train_date=last_date,
            transform_steps=transform_steps,
        )


def fit_full_and_forecast(
    y: pd.Series,
    spec: Dict[str, Any],
    steps: int,
    transform_info: Optional[Dict[str, Any]] = None,
    exog: Optional[pd.DataFrame] = None,
    future_exog: Optional[pd.DataFrame] = None,
    m: Optional[int] = None,
) -> pd.DataFrame:
    """
    Thin wrapper around fit_full_and_forecast_detailed returning the forecast DataFrame.
    """
    return fit_full_and_forecast_detailed(
        y=y,
        spec=spec,
        steps=steps,
        transform_info=transform_info,
        exog=exog,
        future_exog=future_exog,
        m=m,
    ).df

