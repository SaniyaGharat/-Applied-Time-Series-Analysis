# Applied Time Series Analysis (ATSA)

An interactive Streamlit web application designed for end-to-end applied time series analysis, diagnostic checking, and forecasting.

---

## 🚀 Project Overview

The goal of this application is to provide a clean, modular platform for loading, exploring, modeling, and forecasting time series data.

### Development Roadmap
- **Phase 1: Project Skeleton & Data Ingestion** *(Current)*
  - Modular project structure.
  - CSV and Excel file upload.
  - Auto-detection and confirmation of date and target numeric columns.
  - Datetime index parsing, sorting, and summary metrics.
  - Interactive preview and time series line plot.
- **Phase 2: Exploratory Data Analysis & Decomposition** *(Upcoming)*
  - Trend and seasonality inspection.
  - Classical and STL decomposition (additive / multiplicative).
  - Autocorrelation (ACF) and Partial Autocorrelation (PACF) plots.
- **Phase 3: Stationarity & Preprocessing** *(Upcoming)*
  - Augmented Dickey-Fuller (ADF) & KPSS tests.
  - Differencing, log transformations, and missing data imputation.
- **Phase 4: Time Series Modeling & Forecasting** *(Upcoming)*
  - Classical statistical models (ARIMA, SARIMA, Exponential Smoothing).
  - Train/test splitting and forecast validation.
  - Performance metrics (RMSE, MAE, MAPE).

---

## 📁 Project Structure

```text
atsa/
├── app.py                   # Streamlit web application entry point
├── requirements.txt         # Project dependencies
├── README.md                # Project documentation
└── modules/
    ├── __init__.py          # Python package initializer
    ├── data_loader.py       # File ingestion, date detection, time-series preparation
    └── utils.py             # Shared helper functions (placeholder for future phases)
```

---

## 🛠️ Installation & Setup

1. **Clone or navigate to the repository:**
   ```bash
   cd atsa
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   # Windows PowerShell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Streamlit app:**
   ```bash
   streamlit run app.py
   ```
