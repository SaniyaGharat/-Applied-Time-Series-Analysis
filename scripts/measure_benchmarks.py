"""
Script to measure cold-start, memory (peak RSS), and execution time for model fitting.
Runs benchmarks outside the Streamlit app using psutil.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import psutil

# Ensure project root is in path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from ui.model_common import fit_all_default_models


def measure_fit_benchmarks():
    proc = psutil.Process(os.getpid())
    rss_start = proc.memory_info().rss / (1024 * 1024)

    # 1. 144 rows - AirPassengers dataset
    air_csv = ROOT_DIR / "data" / "AirPassengers.csv"
    if not air_csv.exists():
        raise FileNotFoundError(f"Missing {air_csv}")

    df_air = pd.read_csv(air_csv)
    s_air = pd.Series(
        df_air["Passengers"].values,
        index=pd.to_datetime(df_air["Month"]),
        name="Passengers",
    )

    rss_pre_144 = proc.memory_info().rss / (1024 * 1024)
    t0 = time.perf_counter()
    results_144 = fit_all_default_models(
        s_air,
        holdout_size=12,
        m=12,
        mase_m=12,
        transform_info={
            "steps": [],
            "params": {},
            "suggested_d": 0,
            "suggested_D": 0,
            "seasonal_period": 12,
        },
        exog_df=None,
    )
    t_144 = time.perf_counter() - t0
    rss_post_144 = proc.memory_info().rss / (1024 * 1024)

    # 2. 1000 rows - Synthetic daily/weekly series
    rng = np.random.default_rng(42)
    dates_1000 = pd.date_range("2000-01-01", periods=1000, freq="D")
    vals_1000 = 100.0 + np.cumsum(rng.normal(0, 1, 1000))
    s_1000 = pd.Series(vals_1000, index=dates_1000, name="Target")

    rss_pre_1000 = proc.memory_info().rss / (1024 * 1024)
    t0 = time.perf_counter()
    results_1000 = fit_all_default_models(
        s_1000,
        holdout_size=30,
        m=7,
        mase_m=7,
        transform_info={
            "steps": [],
            "params": {},
            "suggested_d": 0,
            "suggested_D": 0,
            "seasonal_period": 7,
        },
        exog_df=None,
    )
    t_1000 = time.perf_counter() - t0
    rss_post_1000 = proc.memory_info().rss / (1024 * 1024)

    from modules.models import list_model_families
    all_fams = [f["key"] for f in list_model_families()]

    fitted_144_fams = [r.family for r in results_144]
    skipped_144_fams = [f for f in all_fams if f not in fitted_144_fams]

    fitted_1000_fams = [r.family for r in results_1000]
    skipped_1000_fams = [f for f in all_fams if f not in fitted_1000_fams]

    print("=== Model Fitting Benchmarks ===")
    print(f"Base Process RSS: {rss_start:.2f} MB")
    print("\n--- AirPassengers (144 rows, m=12, holdout=12) ---")
    print(f"Actually Fit Families ({len(fitted_144_fams)}/{len(all_fams)}): {fitted_144_fams}")
    print(f"Skipped Families ({len(skipped_144_fams)}/{len(all_fams)}): {skipped_144_fams} (sarimax skipped: no exogenous features)")
    print("Individual Model Times:")
    for r in results_144:
        print(f"  - {r.label} ({r.family}): {r.fit_seconds:.3f} s (converged={r.converged})")
    print(f"Total Fit Time (144 rows): {t_144:.3f} s")
    print(f"Peak RSS after AirPassengers fit: {rss_post_144:.2f} MB (Delta: {rss_post_144 - rss_pre_144:.2f} MB)")

    print("\n--- Synthetic Series (1000 rows, m=7, holdout=30) ---")
    print(f"Actually Fit Families ({len(fitted_1000_fams)}/{len(all_fams)}): {fitted_1000_fams}")
    print(f"Skipped Families ({len(skipped_1000_fams)}/{len(all_fams)}): {skipped_1000_fams} (sarimax skipped: no exogenous features)")
    print("Individual Model Times:")
    for r in results_1000:
        print(f"  - {r.label} ({r.family}): {r.fit_seconds:.3f} s (converged={r.converged})")
    print(f"Total Fit Time (1000 rows): {t_1000:.3f} s")
    print(f"Peak RSS after 1000 rows fit: {rss_post_1000:.2f} MB (Delta: {rss_post_1000 - rss_pre_1000:.2f} MB)")
    if t_1000 > 60.0:
        print("\nWARNING: 1000-row fit exceeded 60 s! Grid search capping required.")
    else:
        print("\nPASS: 1000-row fit is well below the 60 s threshold.")


def measure_cold_start():
    print("\n=== Cold-Start Benchmark ===")
    from streamlit.testing.v1 import AppTest

    t0 = time.perf_counter()
    at = AppTest.from_file(str(ROOT_DIR / "app.py"), default_timeout=30).run()
    t_boot = time.perf_counter() - t0

    # Trigger demo data load
    t0_demo = time.perf_counter()
    at.button[0].click().run()
    t_demo = time.perf_counter() - t0_demo

    print(f"Time from start to first render (initial boot): {t_boot:.3f} s")
    print(f"Time to render with Demo Data: {t_demo:.3f} s")
    print(f"Total time to interactive demo state: {t_boot + t_demo:.3f} s")


if __name__ == "__main__":
    measure_fit_benchmarks()
    measure_cold_start()
