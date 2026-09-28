"""
Shared Utilities, Model Caching, and Widget State Lifecycle for ATSA Model and Forecast Workspaces.

Provides:
- reset_model_widgets() for purging stale widget states
- cached_fit_model() and cached_grid_search() for UI caching
- compute_suggested_specs() for heuristic order suggestion
- fit_all_default_models() loop for batch estimating baseline and statistical models
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import streamlit as st

from modules.diagnostics import (
    adf_test,
    compute_acf_pacf,
    decompose,
    kpss_test,
    stationarity_verdict,
    suggest_orders,
    transform_series,
)
from modules.models import (
    ModelResult,
    fit_model,
    grid_search_arima,
    list_model_families,
)


def reset_model_widgets() -> None:
    """
    Purge model configuration form states, grid search results, benchmark leaderboard,
    and pending specs from Streamlit session state to prevent stale widget state.
    """
    keys_to_delete = [
        k for k in list(st.session_state.keys())
        if k.startswith("form_")
    ]
    specific_keys = [
        "model_holdout_size_input",
        "model_m_input",
        "last_grid_df",
        "last_grid_best",
        "benchmark_df",
        "pending_spec",
    ]
    for k in specific_keys:
        if k in st.session_state:
            keys_to_delete.append(k)

    for k in set(keys_to_delete):
        st.session_state.pop(k, None)


@st.cache_data(max_entries=50, ttl=3600)
def cached_fit_model(
    y: pd.Series,
    spec: Dict[str, Any],
    test_size: int,
    transform_info: Optional[Dict[str, Any]],
    exog: Optional[pd.DataFrame],
    m: Optional[int],
    mase_m: int = 1,
) -> ModelResult:
    """UI cache wrapper for deterministic model fitting."""
    return fit_model(
        y=y,
        spec=spec,
        test_size=test_size,
        transform_info=transform_info,
        exog=exog,
        m=m,
        mase_m=mase_m,
    )


@st.cache_data(max_entries=50, ttl=3600)
def cached_grid_search(
    y: pd.Series,
    p_range: List[int],
    d: int,
    q_range: List[int],
    seasonal: Optional[Dict[str, Any]],
    test_size: int,
    max_models: int = 60,
    criterion: str = "aic",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """UI cache wrapper for ARIMA/SARIMA grid search."""
    return grid_search_arima(
        y=y,
        p_range=p_range,
        d=d,
        q_range=q_range,
        seasonal=seasonal,
        test_size=test_size,
        max_models=max_models,
        criterion=criterion,
    )


def compute_suggested_specs(
    model_base_series: pd.Series,
    suggested_d: int = 0,
    suggested_D: int = 0,
    m: int = 12,
) -> Dict[str, Any]:
    """
    Compute heuristic orders (p, q, P, Q, d, D, m) based on ACF/PACF of differenced series.
    """
    try:
        diff_base, _ = transform_series(
            model_base_series,
            diff_order=suggested_d,
            seasonal_diff_order=suggested_D,
            seasonal_period=m if suggested_D > 0 else None,
        )
        nlags = min(20, max(1, len(diff_base) // 2 - 1))
        acf_pacf_data = compute_acf_pacf(diff_base, nlags=nlags)
        sugg = suggest_orders(acf_pacf_data["acf"], acf_pacf_data["pacf"], acf_pacf_data["conf_bound"], m=m)
        suggested_p = max(1, min(3, sugg["suggested_p"]))
        suggested_q = max(1, min(3, sugg["suggested_q"]))
    except Exception:
        suggested_p = 1
        suggested_q = 1

    return {
        "p": suggested_p,
        "q": suggested_q,
        "d": suggested_d,
        "P": 1 if suggested_D > 0 else 0,
        "D": suggested_D,
        "Q": 1 if suggested_D > 0 else 0,
        "m": m,
    }


def fit_all_default_models(
    model_base_series: pd.Series,
    holdout_size: int,
    m: int,
    mase_m: int = 1,
    transform_info: Optional[Dict[str, Any]] = None,
    exog_df: Optional[pd.DataFrame] = None,
    progress_callback: Optional[Any] = None,
) -> List[ModelResult]:
    """
    Fit standard default specifications across all supported model families and
    save them to st.session_state["model_results"].
    """
    if "model_results" not in st.session_state:
        st.session_state["model_results"] = {}

    all_families = list_model_families()
    suggested_d = transform_info.get("suggested_d", 0) if transform_info else 0
    suggested_D = transform_info.get("suggested_D", 0) if transform_info else 0

    if suggested_d == 0 and suggested_D == 0:
        verdict = "Non-stationary"
        try:
            adf_res = adf_test(model_base_series)
            kpss_res = kpss_test(model_base_series)
            verdict_res = stationarity_verdict(adf_res, kpss_res)
            verdict = verdict_res.get("verdict", "")
            if verdict != "Stationary":
                suggested_d = 1
            else:
                suggested_d = 0
        except Exception:
            suggested_d = 1

        if int(m) >= 2 and len(model_base_series) >= 3 * int(m):
            try:
                decomp_res = decompose(model_base_series, period=int(m))
                if (decomp_res.get("seasonal_strength") or 0.0) > 0.6:
                    suggested_D = 1
            except Exception:
                pass

        st.info(f"Auto-suggested differencing for default models (stationarity verdict '{verdict}'): d={suggested_d}, D={suggested_D}.")

    fitted_results: List[ModelResult] = []
    total = len(all_families)

    for idx, fam in enumerate(all_families):
        fam_key = fam["key"]
        spec_to_fit = dict(fam["default_spec"])

        # Skip sarimax if no exogenous data
        if fam_key == "sarimax" and (exog_df is None or exog_df.empty):
            if progress_callback:
                progress_callback((idx + 1) / total)
            continue

        # Skip seasonal families if m < 2 or series too short
        if fam_key in ["seasonal_naive", "sarima", "holt_winters"] and (int(m) < 2 or len(model_base_series) < 2 * int(m)):
            if progress_callback:
                progress_callback((idx + 1) / total)
            continue

        # Adjust suggested differencing for arima/sarima
        if fam_key in ["ar", "ma", "arma", "arima", "sarima", "sarimax"]:
            spec_to_fit["d"] = suggested_d
        if fam_key in ["sarima", "sarimax"]:
            spec_to_fit["D"] = suggested_D
            spec_to_fit["m"] = int(m)
        if fam_key in ["seasonal_naive", "holt_winters"]:
            spec_to_fit["m"] = int(m)

        try:
            res = cached_fit_model(
                y=model_base_series,
                spec=spec_to_fit,
                test_size=holdout_size,
                transform_info=transform_info,
                exog=exog_df if fam_key == "sarimax" else None,
                m=m,
                mase_m=mase_m,
            )
            st.session_state["model_results"][res.key] = res
            fitted_results.append(res)
        except Exception:
            pass

        if progress_callback:
            progress_callback((idx + 1) / total)

    return fitted_results
