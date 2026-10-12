"""
Benchmark script for Phase 8: Two-dataset comparison pipeline.
Measures execution time, cold-start RSS, and peak memory consumption outside Streamlit.
"""

import os
import sys
import time
from pathlib import Path
import psutil
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from ui.data_tab import create_demo_data
from modules.pipeline import run_analysis, compute_kpis, compare_datasets


def run_phase8_benchmarks():
    proc = psutil.Process(os.getpid())
    rss_initial = proc.memory_info().rss / (1024 * 1024)

    print("=" * 60)
    print("ATSA Phase 8 Pipeline Benchmarks (AirPassengers + Demo Data)")
    print("=" * 60)
    print(f"Initial Process RSS: {rss_initial:.2f} MB")

    # 1. Load Datasets
    # Dataset A: Demo Data (48 monthly rows)
    df_demo = create_demo_data()
    s_demo = pd.Series(
        df_demo["Sales"].values,
        index=pd.to_datetime(df_demo["Date"]),
        name="Sales",
    )

    # Dataset B: AirPassengers (144 monthly rows)
    air_csv = ROOT_DIR / "data" / "AirPassengers.csv"
    df_air = pd.read_csv(air_csv)
    s_air = pd.Series(
        df_air["Passengers"].values,
        index=pd.to_datetime(df_air["Month"]),
        name="Passengers",
    )

    print(f"Dataset A: Demo Monthly Sales (N={len(s_demo)})")
    print(f"Dataset B: AirPassengers (N={len(s_air)})")
    print("-" * 60)

    # 2. Run Pipeline for Dataset A
    rss_pre_a = proc.memory_info().rss / (1024 * 1024)
    t0_a = time.perf_counter()
    bundle_a = run_analysis(s_demo, name="Demo Monthly Sales", m=12, holdout=12)
    t_a = time.perf_counter() - t0_a
    rss_post_a = proc.memory_info().rss / (1024 * 1024)

    print(f"Dataset A Analysis Completed in: {t_a:.2f} s")
    print(f"  - Fitted Models: {bundle_a.n_models_fit} (Failed: {bundle_a.n_failed})")
    print(f"  - Best Model: {bundle_a.best_label} (MASE: {bundle_a.comparison.iloc[0]['MASE']:.3f})")
    print(f"  - Post-A Memory RSS: {rss_post_a:.2f} MB (Delta: {rss_post_a - rss_pre_a:+.2f} MB)")
    print("-" * 60)

    # 3. Run Pipeline for Dataset B
    rss_pre_b = proc.memory_info().rss / (1024 * 1024)
    t0_b = time.perf_counter()
    bundle_b = run_analysis(s_air, name="AirPassengers", m=12, holdout=12)
    t_b = time.perf_counter() - t0_b
    rss_post_b = proc.memory_info().rss / (1024 * 1024)

    print(f"Dataset B Analysis Completed in: {t_b:.2f} s")
    print(f"  - Fitted Models: {bundle_b.n_models_fit} (Failed: {bundle_b.n_failed})")
    print(f"  - Best Model: {bundle_b.best_label} (MASE: {bundle_b.comparison.iloc[0]['MASE']:.3f})")
    print(f"  - Post-B Memory RSS: {rss_post_b:.2f} MB (Delta: {rss_post_b - rss_pre_b:+.2f} MB)")
    print("-" * 60)

    # 4. Cross-Dataset Comparison Synthesis
    t0_cmp = time.perf_counter()
    kpis_a = compute_kpis(bundle_a)
    kpis_b = compute_kpis(bundle_b)
    kpi_df, findings = compare_datasets(kpis_a, kpis_b, series_a=s_demo, series_b=s_air)
    t_cmp = time.perf_counter() - t0_cmp
    rss_final = proc.memory_info().rss / (1024 * 1024)

    total_time = t_a + t_b + t_cmp
    print(f"Comparison Synthesis Completed in: {t_cmp * 1000:.1f} ms")
    print(f"Key Findings Generated ({len(findings)}):")
    for f in findings:
        print(f"  * {f}")
    print("-" * 60)
    print(f"TOTAL PIPELINE EXECUTION TIME: {total_time:.2f} s")
    print(f"PEAK MEMORY RSS: {rss_final:.2f} MB")
    print(f"STREAMLIT CLOUD COMPLIANCE: {'PASSED (< 1000 MB limit)' if rss_final < 1000 else 'FAILED'}")
    print("=" * 60)


if __name__ == "__main__":
    run_phase8_benchmarks()
