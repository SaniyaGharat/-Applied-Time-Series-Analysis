# Applied Time Series Analysis (ATSA)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://applied-time-series-analysis.streamlit.app)

An end-to-end interactive Streamlit web application designed for applied time series analysis, diagnostic checking, classical/modern econometric modeling, statistical comparison, and multi-step forecasting.

---

## 🌐 Live Demo
🔗 **App URL**: [https://applied-time-series-analysis.streamlit.app](https://applied-time-series-analysis.streamlit.app) *(Streamlit Community Cloud)*

---

## 📸 Screenshots
*(Screenshots section will be updated after deployment)*

---

## ✨ Features
- **Data Ingestion & Preprocessing (`📥 Data`)**: File upload (CSV, XLSX, XLS), bundled sample dataset (`AirPassengers`), synthetic demo data (`🧪 Load Demo Data`), duplicate aggregation, frequency alignment with automatic resample fallback, and missing data imputation.
- **Exploratory Data Analysis (EDA) & Diagnostics (`🔍 EDA & Diagnostics`)**: Joint ADF/KPSS stationarity testing with automated verdict, classical & STL decomposition with Hyndman strength metrics, ACF/PACF with automatic order suggestions, seasonal distribution, and Box-Cox / differencing pipelines.
- **Comprehensive Model Zoo (10 Architectures from `list_model_families()`) (`🧮 Models`)**:
  1. `naive`: Naive Baseline
  2. `seasonal_naive`: Seasonal Naive Baseline
  3. `drift`: Random Walk with Drift
  4. `ar`: Autoregressive AR(p)
  5. `ma`: Moving Average MA(q)
  6. `arma`: Autoregressive Moving Average ARMA(p, q)
  7. `arima`: Autoregressive Integrated Moving Average ARIMA(p, d, q)
  8. `sarima`: Seasonal ARIMA (SARIMA)
  9. `sarimax`: SARIMAX with Exogenous Regressors
  10. `holt_winters`: Holt-Winters Exponential Smoothing (additive / multiplicative / damped)
- **Automated Leaderboard & Evaluation (`🔮 Forecast & Compare`)**: Out-of-sample metrics (RMSE, MAE, MAPE, sMAPE, MASE), relative RMSE vs benchmark, Diebold-Mariano pairwise predictive accuracy hypothesis testing, and equal-weight ensemble combination.
- **Forward Forecasting & Export (`🔮 Forecast & Compare`)**: Multi-step forward point projections with 95% confidence intervals, multi-model overlay, and multi-sheet styled Excel workbook generation with sheets:
  - `Comparison`: Leaderboard rankings and out-of-sample metrics
  - `Holdout_Forecasts`: Holdout validation predictions and bounds
  - `Future_Forecast`: Forward projection point forecasts and intervals
  - `Specs`: Model JSON specifications, transforms, holdout size, and execution metadata

---

## 📁 Project Structure

```text
atsa/
├── .streamlit/
│   └── config.toml             # Headless server, upload limit, and light theme configuration
├── data/
│   ├── AirPassengers.csv       # Canonical Box-Jenkins monthly airline passenger series (144 rows)
│   └── india_dataset_placeholder.csv # Placeholder Indian macroeconomic dataset
├── modules/
│   ├── __init__.py
│   ├── data_loader.py          # File ingestion, date detection, frequency regularization
│   ├── diagnostics.py          # Stationarity tests (ADF/KPSS), ACF/PACF, STL decomposition
│   ├── models.py               # 10 statistical & econometric model implementations
│   ├── compare.py              # Out-of-sample leaderboard, DM test, equal-weight ensemble
│   ├── export.py               # Multi-sheet styled Excel workbook builder (openpyxl)
│   ├── plots.py                # Pure Plotly interactive figure builders
│   └── utils.py                # Helper utilities and type definitions
├── ui/
│   ├── __init__.py
│   ├── data_tab.py             # 📥 Data: Dataset ingestion, sample selector (AirPassengers), regularization
│   ├── eda_tab.py              # 🔍 EDA & Diagnostics: Stationarity tests, decomposition, transforms
│   ├── models_tab.py           # 🧮 Models: Model zoo tuning, diagnostics, residual checks
│   ├── forecast_tab.py         # 🔮 Forecast & Compare: Leaderboard, future forecasts, Excel export
│   └── model_common.py         # Shared model fitting and session state coordination
├── scripts/
│   └── measure_benchmarks.py   # Benchmark script for cold-start and RSS memory profiling
├── tests/
│   ├── __init__.py
│   ├── test_a1.py              # Frequency alignment and resample tests
│   ├── test_diagnostics.py     # Statistical diagnostics and decomposition tests
│   ├── test_models.py          # Core model zoo estimation and metrics tests
│   └── test_app_smoke.py       # End-to-end Streamlit AppTest smoke & edge case suite
├── app.py                      # Main Streamlit application entry point
├── requirements.txt            # Production dependencies pinned with ==
├── requirements-dev.txt        # Development and testing dependencies (pytest)
├── DEPLOY.md                   # Step-by-step deployment guide & post-deploy smoke checklist
└── README.md                   # Project documentation
```

---

## 🛠️ Installation & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/SaniyaGharat/-Applied-Time-Series-Analysis.git
cd -Applied-Time-Series-Analysis
```

### 2. Create and Activate Virtual Environment
```bash
# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
# Production dependencies
pip install -r requirements.txt

# Development and testing tools
pip install -r requirements-dev.txt
```

### 4. Run Unit and Smoke Tests
```bash
pytest -q --durations=5
```

### 5. Launch the Streamlit App
```bash
streamlit run app.py
```

---

## ⚠️ Limitations & Methodological Caveats

1. **Single Holdout Evaluation Origin**:
   Model evaluation and ranking are performed on a single fixed chronological holdout split rather than rolling-origin (expanding/sliding window) cross-validation. While computationally efficient for real-time web exploration, ranking can exhibit variance depending on the specific holdout horizon chosen.
2. **Median Forecasts under Log / Non-Linear Transforms**:
   When transformations (e.g. natural logarithm or Box-Cox) are applied, point forecasts generated on the transformed scale and inverted via exponentiation $\exp(\hat{y})$ correspond mathematically to conditional **medians** of the original process, not conditional means. Mean forecasting would require Jensen's inequality corrections ($\exp(\hat{y} + \frac{1}{2}\sigma^2)$).
3. **Diebold-Mariano (DM) Test Caveats**:
   The pairwise Diebold-Mariano test evaluates equal predictive accuracy assuming covariance-stationary loss differentials. In small evaluation horizons ($h < 30$), asymptotic normality approximations may display minor size distortions and reduced power. The Harvey-Leybourne-Newbold (HLN) small-sample adjusted statistic is reported to mitigate this.

---

## 🚀 Deployment to Streamlit Community Cloud

Follow these steps to deploy on Streamlit Community Cloud:

1. **Push Changes to GitHub**:
   Ensure all changes are committed and pushed to `https://github.com/SaniyaGharat/-Applied-Time-Series-Analysis`.
2. **Access Streamlit Community Cloud**:
   Visit [share.streamlit.io](https://share.streamlit.io) and sign in with your GitHub account.
3. **Configure New App**:
   - Click **New app**.
   - **Repository**: `SaniyaGharat/-Applied-Time-Series-Analysis`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL**: Choose custom subdomain if available (e.g. `applied-time-series-analysis`).
4. **Advanced Settings**:
   - Select **Python 3.12** (recommended for maximum pre-built wheel stability on Debian Linux containers) or **Python 3.13**.
   - No external secret keys are required.
5. **Deploy**:
   Click **Deploy!**. Streamlit will install `requirements.txt` and launch the app within ~2 minutes.
