# Streamlit Community Cloud Deployment Guide

This guide provides exact step-by-step instructions for deploying the **Applied Time Series Analysis (ATSA)** web application to [Streamlit Community Cloud](https://share.streamlit.io), along with a post-deployment smoke verification checklist.

---

## 📋 Pre-Deployment Checklist

Before deploying, confirm that all local tests and configurations pass:
- [x] Python dependencies pinned with `==` in `requirements.txt`.
- [x] Test-only packages isolated in `requirements-dev.txt`.
- [x] `.streamlit/config.toml` configured with `headless = true`, `gatherUsageStats = false`, `maxUploadSize = 20`, and light theme primary color `#2563eb`.
- [x] Bundled dataset (`data/AirPassengers.csv`) committed.
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
- [ ] In tab **`📥 Data`**, locate the **"Sample datasets:"** selectbox.
- [ ] Select **`AirPassengers (Monthly 1949-1960)`**.
- [ ] Verify that:
  - Active Dataset indicates **AirPassengers (Monthly 1949-1960)** (144 rows, 2 columns).
  - Date Column is automatically set to `Month`.
  - Target Column is automatically set to `Passengers`.
  - The Interactive Line Plot renders without errors.

### 2. Exploratory Data Analysis & Diagnostics
- [ ] Click the **`🔍 EDA & Diagnostics`** tab.
- [ ] Verify:
  - ADF and KPSS statistical tests show p-values and a combined stationarity verdict banner.
  - Autocorrelation (ACF) and Partial Autocorrelation (PACF) plots render.
  - STL decomposition renders Trend, Seasonal, and Residual components.

### 3. Model Zoo & Model Estimation
- [ ] Click the **`🧮 Models`** tab.
- [ ] Choose a forecasting architecture pill (e.g. `AR` or `SARIMA`), adjust hyperparameters if desired, and click **"Update model"**.
- [ ] Verify model summary card, in-sample fit vs actual plot, and residual diagnostic sub-tabs (Standardized Residuals, Residual ACF/PACF, Q-Q Plot & Histogram, Ljung-Box Test Table) render cleanly.
- [ ] Alternatively, navigate to **`🔮 Forecast & Compare`** tab and click **"🚀 Fit all default models"** to estimate all candidate model families simultaneously across the holdout.

### 4. Comparison Leaderboard & DM Test
- [ ] Click the **`🔮 Forecast & Compare`** tab.
- [ ] Under Section 1 ("1. Holdout Model Comparison & Benchmarking"):
  - Verify the styled leaderboard comparison table renders with models ranked by chosen metric (default RMSE).
  - Verify green **"🏆 Top Performer"** banner identifies Rank 1.
  - Switch sub-tabs to inspect **"📈 Forecast Overlay"**, **"📊 Metric Bars"**, **"🏛️ Information Criteria"**, **"📉 Error by Horizon"**, and **"🔬 Statistical Significance"** (Pairwise Diebold-Mariano Test).
  - Check the **"📥 Download Comparison Table (CSV)"** button.

### 5. Multi-Step Forward Forecasting
- [ ] Under Section 2 ("2. Out-of-Sample Forward Forecast"):
  - In **"Select Model for Future Projection:"**, pick the top performer or desired model.
  - Set **"Forecast Horizon (Steps Ahead):"** (e.g. `12`).
  - Click **"🔮 Generate forecast"**.
  - Verify:
    - Forward projection line plot renders historical data, holdout predictions, and future forecasts with shaded 95% confidence intervals.
    - A table of point forecasts and 95% bounds is displayed.

### 6. Excel & CSV Report Downloads
- [ ] Click the **"📥 Download Future Forecast (CSV)"** button to export forecast numbers.
- [ ] Click the **"📊 Download Comprehensive Report (Excel Workbook)"** button.
- [ ] Verify the file downloads as `atsa_complete_forecast_report.xlsx`.
- [ ] Open the spreadsheet and verify the existence of the 4 sheets generated by `modules/export.py`:
  - `Comparison`: Leaderboard rankings and evaluation metrics across candidate models.
  - `Holdout_Forecasts`: Validation holdout predictions (mean and intervals) across candidate models.
  - `Future_Forecast`: Out-of-sample forward projections and 95% confidence bounds.
  - `Specs`: Model JSON specifications, transform steps, holdout size, MASE scale, and run timestamp.

### 7. Two-Dataset Comparative Analysis (`⚖️ Compare Datasets`)
- [ ] Click the **`⚖️ Compare Datasets`** tab.
- [ ] Configure Slot A: Select **"Demo Monthly Sales (Synthetic)"**.
- [ ] Configure Slot B: Select **"AirPassengers (Monthly 1949-1960)"**.
- [ ] Click **"🚀 Run Comparison"**.
- [ ] Verify:
  - Both datasets execute the automated analytical pipeline with zero unhandled exceptions.
  - View layout toggle between **"Stacked (A above B)"** and **"Side by side"** functions smoothly.
  - All 7 sections render for Dataset A and Dataset B:
    1. Series line plots, descriptive summary stats, and indexed growth overlay (Base=100).
    2. ADF and KPSS hypothesis tests with stationarity verdict banners and suggested $d$.
    3. Differenced ACF and PACF autocorrelation charts with confidence bands.
    4. Time series decomposition with Hyndman seasonal and trend strength metric badges.
    5. Candidate model out-of-sample leaderboards ranked by MASE with Top Performer callouts.
    6. Holdout validation forecast overlay comparing top 3 candidate models against ground truth.
    7. Out-of-sample forward projections with 95% confidence bands.
  - Plain-English informational expanders ("ℹ️ What does this show?") render in each section.
  - Cross-dataset evaluation notice is visible: raw RMSE and MAE cannot be ranked across different series units.

### 8. Executive Comparative Dashboard (`📊 Dashboard`)
- [ ] Click the **`📊 Dashboard`** tab.
- [ ] Verify:
  - Top header displays dataset names and date span chips.
  - Dual rows of KPI cards render 11 scale-free metrics for Dataset A and Dataset B (22 total metrics).
  - Econometric synthesis callout box summarizes plain-English comparative findings (relative seasonality, dominant model, forecastability, and differencing correlation).
  - Comparative KPI matrix table displays KPI | Dataset A | Dataset B | Comparability Note.
  - 4 interactive comparative visualizations render:
    1. Indexed Growth Overlay (Base = 100)
    2. Grouped Accuracy Bar Chart (MASE, sMAPE, RelRMSE)
    3. Component Strength Comparisons (Hyndman Trend and Seasonal metrics)
    4. Side-by-Side Forward Projections
  - Click **"📥 Download KPI Comparison (CSV)"** and confirm file download.
  - Click **"📊 Download Executive Comparative Report (Excel Workbook)"** and open file:
    - Verify 6 sheets exist: `KPIs`, `Findings`, `Leaderboard_A`, `Leaderboard_B`, `Future_A`, `Future_B`.
