# Streamlit Community Cloud Deployment Guide

This guide provides exact step-by-step instructions for deploying the **Applied Time Series Analysis (ATSA)** web application to [Streamlit Community Cloud](https://share.streamlit.io), along with a post-deployment smoke verification checklist.

---

## 📋 Pre-Deployment Checklist

Before deploying, confirm that all local tests and configurations pass:
- [x] Python dependencies pinned with `==` in `requirements.txt`.
- [x] Test-only packages isolated in `requirements-dev.txt`.
- [x] `.streamlit/config.toml` configured with `headless = true`, `gatherUsageStats = false`, `maxUploadSize = 20`, and light theme primary color `#2563eb`.
- [x] Bundled datasets (`data/AirPassengers.csv` and `data/india_dataset_placeholder.csv`) committed.
- [x] All automated tests pass: `pytest -q --durations=5`.
- [x] No bytecode syntax errors: `python -m compileall app.py modules ui`.

---

## 🚀 Exact Deployment Steps (Click-by-Click)

1. **Push Changes to GitHub**:
   - Confirm with git status that all files are committed to the `main` branch.
   - Run `git push origin main`.

2. **Access Streamlit Community Cloud**:
   - Navigate to [https://share.streamlit.io](https://share.streamlit.io).
   - Sign in using your GitHub account that has access to `SaniyaGharat/-Applied-Time-Series-Analysis`.

3. **Initiate New App Creation**:
   - In the top right corner of the workspace, click the blue **"Create app"** (or **"New app"**) button.
   - If prompted, select **"I already have an app"**.

4. **Fill Deployment Details**:
   - **Repository**: Select or enter `SaniyaGharat/-Applied-Time-Series-Analysis`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL**: Enter a custom app subdomain (e.g., `applied-time-series-analysis` -> `https://applied-time-series-analysis.streamlit.app`).

5. **Configure Advanced Settings**:
   - Click the **"Advanced settings"** link before deploying.
   - In the **Python version** dropdown, select **3.12** *(recommended for the highest stability and binary wheel compatibility across Linux cloud containers)* or **3.13**.
   - Under **Secrets**, leave blank (ATSA runs entirely offline with no external API keys).
   - Click **Save**.

6. **Launch Deployment**:
   - Click the green **"Deploy!"** button.
   - The build console will display package installation logs from `requirements.txt`. Deployment typically completes within 60–120 seconds.

---

## 🔍 Post-Deployment Smoke Checklist

Once the deployed app loads at its live URL, execute the following verification steps in sequence:

### 1. Ingestion & Bundled Data Load
- [ ] In **1. Ingestion & Preprocessing**, locate the **"Sample datasets"** dropdown.
- [ ] Select **`AirPassengers (Monthly 1949-1960)`**.
- [ ] Verify that:
  - Active Dataset indicates **144 rows, 2 columns**.
  - Date Column is automatically set to `Month`.
  - Target Column is automatically set to `Passengers`.
  - The Interactive Line Plot renders without errors.

### 2. Exploratory Data Analysis & Diagnostics
- [ ] Click the **2. Exploratory Data Analysis & Diagnostics** tab.
- [ ] Verify:
  - ADF and KPSS statistical tests show p-values and a combined stationarity verdict banner.
  - Autocorrelation (ACF) and Partial Autocorrelation (PACF) plots render.
  - STL decomposition renders Trend, Seasonal, and Residual components.

### 3. Model Zoo & "Fit All Models"
- [ ] Click the **3. Model Selector & Evaluation** tab.
- [ ] Click **"⚡ Fit All Default Models"**.
- [ ] Verify progress bar completes across all 10 model architectures (Naive, Seasonal Naive, SES, Holt-Winters, AR, MA, ARMA, ARIMA, SARIMA, SARIMAX) without unhandled exceptions.

### 4. Comparison Leaderboard & DM Test
- [ ] Click the **4. Forecast & Evaluation** tab.
- [ ] Verify:
  - The **Out-of-Sample Performance Comparison** table renders with models ranked by RMSE.
  - A green **"🏆 Top Performer"** banner identifies Rank 1.
  - The **Pairwise Diebold-Mariano Test** matrix renders with statistical significance coloring.

### 5. Multi-Step Forward Forecasting
- [ ] Under **Forward Horizon Configuration**, set steps to `12`.
- [ ] Click **"🔮 Generate Forward Forecast"**.
- [ ] Verify:
  - The Interactive Forward Forecast plot displays historical observations, test predictions, and future forecasts with shaded confidence intervals (80% and 95%).
  - A table of point forecasts and bounds is displayed.

### 6. Excel Report Download
- [ ] Click the **"📥 Download Comprehensive Forecast Report (.xlsx)"** button.
- [ ] Verify the file downloads as `ATSA_Forecast_Report.xlsx`.
- [ ] Open the spreadsheet and verify the existence of:
  - `Summary`
  - `Comparison Leaderboard`
  - `Pairwise DM Tests`
  - `Forecast Results`
  - `Model Metadata`
